from argparse import ArgumentParser
from typing import Optional, List, Dict, Any, Mapping

import numpy as np
import torch
import wandb
from shift_dev import SHIFTDataset
from shift_dev.dataloader.image_processors import MultitaskImageProcessor
from shift_lab.ontologies.semantic_segmentation.shift_labels import id2label as shift_id2label
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import v2

from transformers.data.data_collator import InputDataClass
from transformers.training_args import OptimizerNames
from transformers.utils import logging
from transformers import (
    PvtV2Config,
    PvtV2Model,
    DeformableDetrForObjectDetection,
    DeformableDetrConfig,
    AutoBackbone,
    ResNetConfig,
    Trainer,
    TrainingArguments, EvalPrediction,
)

id2label = {k: v.name for k, v in shift_id2label.items()}
label2id = {v: k for k, v in id2label.items()}

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_FULL_RES = True
EVAL_FULL_RES = True
DO_REDUCE_LABELS = True
PRETRAINED_MODEL_NAME = "nvidia/mit-b0"
TRAIN_IMAGE_SIZE = {"height": 800, "width": 1280}
CLASS_ID_REMAP = None
IMAGE_TRANSFORMS = [
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
]
FRAME_TRANSFORMS = []

class DeformableDetrMetric:

    def __init__(self):
        self.metric = MeanAveragePrecision()

    def convert_eval_pred(self, eval_pred):
        preds = []
        for i in range(eval_pred.predictions[1].shape[0]):
            prediction_scores, predicted_labels = eval_pred.predictions[1][i].max(1)
            preds.append(
                {
                    "boxes": eval_pred.predictions[2][i],
                    "scores": prediction_scores,
                    "labels": predicted_labels,
                    # "masks": None,
                }
            )

        target = [
            {
                "boxes": eval_pred.label_ids[i]["boxes"],
                "labels": eval_pred.label_ids[i]["class_labels"],
                # "masks": None,
                # "iscrowd": None,
                # "area": None,
            }
            for i in range(len(eval_pred.label_ids))
        ]

        return preds, target

    def update(self, eval_pred: EvalPrediction):
        preds, target = self.convert_eval_pred(eval_pred)
        self.metric.update(
            preds=preds,
            target=target,
        )

    def compute(self):
        output = self.metric.compute()
        output["classes"] = output.pop("classes").tolist()
        self.metric = MeanAveragePrecision()
        return output


mean_ap_metric = DeformableDetrMetric()
def compute_metrics(eval_pred: EvalPrediction, calculate_result: bool = True) -> Optional[dict]:

    with torch.no_grad():
        mean_ap_metric.update(eval_pred)
        return mean_ap_metric.compute() if calculate_result else None

def shift_detr_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k == "labels":
            batch[k] = [f[k] for f in features]
        elif k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

def main(args):

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=1,
        load_best_model_at_end=True,
        dataloader_num_workers=args.workers,
        seed=args.seed,
        max_steps=args.max_steps,
        tf32=args.use_tf32,
        optim=OptimizerNames.ADAMW_8BIT if args.use_adam8bit else OptimizerNames.ADAMW_TORCH,
        dataloader_pin_memory=False if args.workers > 0 else True,
        compute_metrics_interval="batch",
        metric_for_best_model="eval_map"
    )

    image_processor_train = MultitaskImageProcessor.from_pretrained(
        PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS, class_id_remap=CLASS_ID_REMAP,
    )
    image_processor_train.size = TRAIN_IMAGE_SIZE
    image_processor_val = MultitaskImageProcessor.from_pretrained(
        PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS, class_id_remap=CLASS_ID_REMAP,
    )
    image_processor_val.size = TRAIN_IMAGE_SIZE

    keys_to_load = [
        Keys.images,  # images, shape (1, 3, H, W), uint8 (RGB)
        Keys.intrinsics,  # camera intrinsics, shape (3, 3)
        Keys.boxes2d,
    ]

    train_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="train",
        keys_to_load=keys_to_load,
        views_to_load=["front"],  # SHIFTDataset.VIEWS.remove("center"),
        shift_type="discrete",          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=FileBackend(),           # also supports HDF5Backend(), FileBackend()
        verbose=True,
        image_transforms=IMAGE_TRANSFORMS,
        frame_transforms=FRAME_TRANSFORMS,
        image_processor=image_processor_train,
        load_full_res=TRAIN_FULL_RES,
        depth_mask_semantic_ids=[label2id["sky"]],
    )

    val_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="minival" if args.use_minival else "val",
        keys_to_load=keys_to_load,
        views_to_load=["front"],  # SHIFTDataset.VIEWS.remove("center"),
        shift_type="discrete",
        backend=FileBackend(),
        verbose=True,
        image_processor=image_processor_val,
        load_full_res=EVAL_FULL_RES,
        depth_mask_semantic_ids=[label2id["sky"]],
    )

    label2id_boxes2d = train_dataset.scalabel_datasets["front/det_2d"].cats_name2id["boxes2d_classes"]
    id2label_boxes2d = {v: k for k, v in label2id_boxes2d.items()}
    id2label_boxes2d[6] = "no_box"

    model = DeformableDetrForObjectDetection(
        DeformableDetrConfig(
            use_timm_backbone=False,
            backbone="pvt_v2",
            backbone_config=PvtV2Config(
                mlp_ratios=[4, 4, 4, 4],
                output_hidden_states=True,
                # num_labels=23,
            ),
            encoder_layers=3,
            encoder_ffn_dim=256,
            decoder_layers=3,
            decoder_ffn_dim=256,
            id2label=id2label_boxes2d,
            # num_labels=7,
            # num_queries=150,
            # backbone_config=ResNetConfig(out_indices=[1, 2, 3])),
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=shift_detr_collator,
    )

    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="./ddetr_test", help="Output dir to store results.")
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.00006, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("-es", "--eval-steps", type=int, default=10, help="Number of steps between validation runs.")
    parser.add_argument("-ss", "--save-steps", type=int, default=None, help="Number of steps between checkpoints. Defaults to eval steps.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("-tf32", "--use-tf32", action="store_true", default=False, help="Set to True if your setup supports TF32 dtype.")
    parser.add_argument("-mv", "--use-minival", action="store_true", default=False, help="Use the minival validation set.")
    parser.add_argument("-bnb", "--use-adam8bit", action="store_true", default=False, help="Use ADAMW_8BIT optimizer (linux only).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checpoint to resume training.")
    parser.add_argument("-rwb", "--resume-wandb", type=str, default=None, help="ID of run to resume")
    parser.add_argument("-eval", "--eval-only", action="store_true", default=False, help="Only run evaluation step.")
    parser.add_argument("-semseg", "--semseg", action="store_true", default=False, help="Train semesg head.")
    parser.add_argument("-depth", "--depth", action="store_true", default=False, help="Train depth head.")
    parser.add_argument("-stl", "--save-total-limit", type=int, default=None, help="Maximum number of checkpoints to store at once.")

    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.save_steps is None:
        args.save_steps = args.eval_steps

    if args.use_tf32:
        logger.info("Using TF32 dtype.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.resume_wandb:
        wandb.init(project="huggingface", resume="must", id=args.resume_wandb)

    main(args)
