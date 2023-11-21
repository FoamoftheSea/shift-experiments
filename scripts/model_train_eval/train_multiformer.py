import sys
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any, Mapping

import numpy as np
import torch
import wandb
from shift_dev import SHIFTDataset
from shift_dev.dataloader.image_processors import MultitaskImageProcessor
from shift_lab.trainer import MultitaskTrainer
from shift_lab.ontologies.semantic_segmentation.shift_labels import id2label as shift_id2label
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend, ZipBackend
from torchvision.transforms import v2

from transformers.data.data_collator import InputDataClass
from transformers.models.multiformer.configuration_multiformer import MultiformerConfig
from transformers.models.multiformer.metrics_multiformer import MultiformerMetric
from transformers.models.multiformer.modeling_multiformer import MultiformerTask, Multiformer
from transformers.training_args import OptimizerNames
from transformers.utils import logging
from transformers import (
    PvtV2Config,
    PvtV2Model,
    AutoBackbone,
    TrainingArguments, EvalPrediction,
)

DO_REDUCE_LABELS = True
EVAL_IGNORE_IDS = {k for k, v in shift_id2label.items() if v.ignoreInEval}
id2label = {k: v.name for k, v in shift_id2label.items()}
label2id = {v: k for k, v in id2label.items()}
CLASS_ID_REMAP = None
CLASS_LOSS_WEIGHTS = {
    "building": 0.04222825955577863,
    "pedestrian": 0.047445990516605495,
    "pole": 0.04716318294296953,
    "road line": 0.04701319399433329,
    "road": 0.03254657779903941,
    "sidewalk": 0.045215864574844555,
    "vegetation": 0.04268060265261902,
    "vehicle": 0.04524841096107193,
    "wall": 0.04683261072198212,
    "traffic sign": 0.047584108152864686,
    "sky": 0.03540394188413042,
    "traffic light": 0.0475655153334132,
    "static": 0.04745211954958151,
    "dynamic": 0.04756501593671283,
    "terrain": 0.045982952375233296,
    "other": 0.04748171409451219,
    "ground": 0.0474869899781356,
    "fence": 0.04731463233410513,
    "guard rail": 0.047304633054814596,
    "rail track": 0.047465059495701845,
    "water": 0.04760968530230659,
    "bridge": 0.04740893878924414,
    "unlabeled": 0.0,
}
CLASS_LOSS_WEIGHTS = [CLASS_LOSS_WEIGHTS[id2label[cid]] for cid in sorted(id2label.keys())]
if DO_REDUCE_LABELS and 0 in id2label.keys():
    CLASS_LOSS_WEIGHTS.append(CLASS_LOSS_WEIGHTS.pop(0))

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# AutoBackbone.register(PvtV2Config, PvtV2Model)

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

mean_ap_metric = MultiformerMetric(id2label=id2label)


def compute_metrics(
    tasks: List[MultiformerTask],
    eval_pred: EvalPrediction,
    calculate_result: bool = True
) -> Optional[dict]:

    with torch.no_grad():
        for task in tasks:
            mean_ap_metric.update(task, eval_pred)
        return mean_ap_metric.compute() if calculate_result else None


def shift_multiformer_collator(features: List[InputDataClass]) -> Dict[str, Any]:

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
        Keys.segmentation_masks,
        Keys.depth_maps,
        # Keys.masks,
    ]

    train_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="train",
        keys_to_load=keys_to_load,
        views_to_load=["front"],  # SHIFTDataset.VIEWS.remove("center"),
        shift_type="discrete",          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=ZipBackend() if args.load_zip else FileBackend(),           # also supports HDF5Backend(), FileBackend()
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
        backend=ZipBackend() if args.load_zip else FileBackend(),
        verbose=True,
        image_processor=image_processor_val,
        load_full_res=EVAL_FULL_RES,
        depth_mask_semantic_ids=[label2id["sky"]],
    )

    label2id_boxes2d = train_dataset.scalabel_datasets["front/det_2d"].cats_name2id["boxes2d_classes"]
    id2label_boxes2d = {v: k for k, v in label2id_boxes2d.items()}

    # if sys.platform.startswith("win"):
    #     prefix = "C:"
    # else:
    #     prefix = "/mnt/c"
    # model_name_or_path = f"{prefix}/Users/Nate/transformers/ddetr_test1/pytorch_model.bin"
    model_config = MultiformerConfig(
            use_timm_backbone=False,
            backbone="pvt_v2",
            backbone_config=PvtV2Config(
                mlp_ratios=[4, 4, 4, 4],
                output_hidden_states=True,
                id2label=id2label,
                label2id=label2id,
                num_labels=len(id2label),
                do_reduce_labels=DO_REDUCE_LABELS,
                out_indices=[0, 1, 2, 3],
            ),
            encoder_layers=3,
            encoder_ffn_dim=256,
            decoder_layers=3,
            decoder_ffn_dim=256,
            id2label=id2label_boxes2d,
            num_queries=300,
            det2d_input_feature_levels=[0, 1, 2, 3],
            det2d_input_proj_kernels=[2, 1, 1, 1],
            det2d_input_proj_strides=[2, 1, 1, 1],
            det2d_extra_feature_levels=1,
        )
    # model_config = MultiformerConfig.from_pretrained(
    #     args.checkpoint,
    #     id2label=id2label_boxes2d,
    #     label2id=label2id_boxes2d,
    #     num_labels=len(id2label_boxes2d),
    # )
    if args.checkpoint is not None:
        model = Multiformer.from_pretrained(
            args.checkpoint,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = Multiformer(config=model_config)
        # model.load_state_dict(torch.load("C:/Users/Nate/shift-experiments/multiformer_pretrained_weights.bin"))

    if args.use_adam8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            params=[
                {"params": model.depth_decoder.parameters(), "lr": args.learning_rate / 5},
                {"params": model.depth_head.parameters(), "lr": args.learning_rate / 5},
                {"params": model.semantic_head.parameters(), "lr": args.learning_rate / 5},
                {"params": model.bbox_embed.parameters()},
                {"params": model.class_embed.parameters()},
                {"params": model.model.encoder.parameters()},
                {"params": model.model.decoder.parameters()},
                {"params": model.model.input_proj.parameters()},
                {"params": model.model.level_embed},
                {"params": model.model.query_position_embeddings.parameters()},
                {"params": model.model.reference_points.parameters()},
                {"params": model.model.backbone.parameters(), "lr": args.learning_rate / 10},
            ],
            lr=args.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            params=[
                {"params": model.bbox_embed.parameters()},
                {"params": model.class_embed.parameters()},
                {"params": model.model.encoder.parameters()},
                {"params": model.model.decoder.parameters()},
                {"params": model.model.input_proj.parameters()},
                {"params": model.model.level_embed},
                {"params": model.model.query_position_embeddings.parameters()},
                {"params": model.model.reference_points.parameters()},
                {"params": model.depth_decoder.parameters(), "lr": args.learning_rate / 5},
                {"params": model.depth_head.parameters(), "lr": args.learning_rate / 5},
                {"params": model.semantic_head.parameters(), "lr": args.learning_rate / 5},
                {"params": model.model.backbone.parameters(), "lr": args.learning_rate / 10},
            ],
            lr=args.learning_rate,
        )
    lr_sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset))
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
        # optim=OptimizerNames.ADAMW_8BIT if args.use_adam8bit else OptimizerNames.ADAMW_TORCH,
        dataloader_pin_memory=False if args.workers > 0 else True,
        include_inputs_for_metrics=True,
        # metric_for_best_model="eval_map"
    )

    # Set loss weights to the device where loss is calculated
    if CLASS_LOSS_WEIGHTS is not None:
        model.class_loss_weights = torch.tensor(CLASS_LOSS_WEIGHTS).to(device)

    trainer = MultitaskTrainer(
        loss_lambdas={"det_2d": 1.0, "semseg": 5.0, "depth": 1.0},
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=shift_multiformer_collator,
        optimizers=(optimizer, lr_sceduler),
        compute_metrics_interval="batch",
    )

    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train(resume_from_checkpoint=args.checkpoint if args.trainer_resume else None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="./ddetr_test", help="Output dir to store results.")
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0002, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("-es", "--eval-steps", type=int, default=1000, help="Number of steps between validation runs.")
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
    parser.add_argument("-zip", "--load-zip", action="store_true", default=False, help="Train with zipped archives.")
    parser.add_argument("-tr", "--trainer_resume", action="store_true", default=False, help="Whether to resume trainer state with checkpoint load.")

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
