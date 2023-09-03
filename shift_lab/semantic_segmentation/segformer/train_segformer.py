from typing import Optional

import torch
from argparse import ArgumentParser

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from torchvision.transforms import v2
from transformers import (
    SegformerImageProcessor,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from transformers.utils import logging

from shift_lab.semantic_segmentation.shift_labels import id2label
from shift_lab.semantic_segmentation.segformer.metrics import SHIFTSegformerEvalMetrics
from shift_lab.semantic_segmentation.segformer.trainer import (
    SHIFTSegformerTrainer,
    SHIFTSegformerForSemanticSegmentation,
)

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Omitting these classes from eval in accordance with Cityscapes
EVAL_IGNORE_IDS = {k for k, v in id2label.items() if v.ignoreInEval}
id2label = {k: v.name for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

PRETRAINED_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
IMAGE_TRANSFORMS = [
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
]
FRAME_TRANSFORMS = []
EVAL_FULL_RES = True
DO_REDUCE_LABELS = True

image_size = {"height": 512, "width": 1024}
image_processor_train = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS)
image_processor_train.size = image_size
image_processor_val = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS)
image_processor_val.size = image_size

CLASS_LOSS_WEIGHTS = {
    'building': 0.04222825955577863,
    'pedestrian': 0.047445990516605495,
    'pole': 0.04716318294296953,
    'road line': 0.04701319399433329,
    'road': 0.03254657779903941,
    'sidewalk': 0.045215864574844555,
    'vegetation': 0.04268060265261902,
    'vehicle': 0.04524841096107193,
    'wall': 0.04683261072198212,
    'traffic sign': 0.047584108152864686,
    'sky': 0.03540394188413042,
    'traffic light': 0.0475655153334132,
    'static': 0.04745211954958151,
    'dynamic': 0.04756501593671283,
    'terrain': 0.045982952375233296,
    'other': 0.04748171409451219,
    'ground': 0.0474869899781356,
    'fence': 0.04731463233410513,
    'guard rail': 0.047304633054814596,
    'rail track': 0.047465059495701845,
    'water': 0.04760968530230659,
    'bridge': 0.04740893878924414,
    'unlabeled': 0.0,
}
CLASS_LOSS_WEIGHTS = [CLASS_LOSS_WEIGHTS[id2label[cid]] for cid in sorted(id2label.keys())]
if DO_REDUCE_LABELS and 0 in id2label.keys():
    CLASS_LOSS_WEIGHTS.append(CLASS_LOSS_WEIGHTS.pop(0))

KEYS_TO_LOAD = [
    Keys.images,                # images, shape (1, 3, H, W), uint8 (RGB)
    Keys.intrinsics,            # camera intrinsics, shape (3, 3)
    Keys.timestamp,
    Keys.axis_mode,
    Keys.extrinsics,
    Keys.boxes2d,               # 2D boxes in image coordinate, (x1, y1, x2, y2)
    Keys.boxes2d_classes,       # class indices, shape (num_boxes,)
    Keys.boxes2d_track_ids,     # object ids, shape (num_ins,)
    # Keys.boxes3d,               # 3D boxes in camera coordinate, (x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z)
    # Keys.boxes3d_classes,       # class indices, shape (num_boxes,), the same as 'boxes2d_classes'
    # Keys.boxes3d_track_ids,     # object ids, shape (num_ins,), the same as 'boxes2d_track_ids'
    Keys.segmentation_masks,    # semantic masks, shape (1, H, W), long
    # Keys.masks,                 # instance masks, shape (num_ins, H, W), binary
    # Keys.depth_maps,            # depth maps, shape (1, H, W), float (meters)
]

metric = SHIFTSegformerEvalMetrics(ignore_class_ids=EVAL_IGNORE_IDS, reduced_labels=DO_REDUCE_LABELS)


def compute_metrics(eval_pred, calculate_result=True) -> Optional[dict]:
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metric.update(pred_labels, labels)

        return metric.compute() if calculate_result else None


def main(args):
    model = SHIFTSegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # Set loss weights to the device where loss is calculated
    loss_weights_tensor = torch.tensor(CLASS_LOSS_WEIGHTS)
    model.class_loss_weights = loss_weights_tensor.to(device)

    train_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="train",
        keys_to_load=KEYS_TO_LOAD,
        views_to_load=["front"],  # SHIFTDataset.VIEWS.remove("center"),
        shift_type="discrete",          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=FileBackend(),           # also supports HDF5Backend(), FileBackend()
        verbose=True,
        image_transforms=IMAGE_TRANSFORMS,
        frame_transforms=FRAME_TRANSFORMS,
        image_processor=image_processor_train,
    )

    val_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="minival" if args.use_minival else "val",
        keys_to_load=KEYS_TO_LOAD,
        views_to_load=["front"],  # SHIFTDataset.VIEWS.remove("center"),
        shift_type="discrete",
        backend=FileBackend(),
        verbose=True,
        image_processor=image_processor_val,
        eval_full_res=EVAL_FULL_RES,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=1,
        load_best_model_at_end=True,
        dataloader_num_workers=args.workers,
        seed=args.seed,
        max_steps=args.max_steps,
        tf32=args.use_tf32,
        optim=OptimizerNames.ADAMW_8BIT if args.use_adam8bit else OptimizerNames.ADAMW_TORCH,
        dataloader_pin_memory=False if args.workers > 0 else True,
        # lr_scheduler_type=SchedulerType.COSINE,
        # push_to_hub=True,
        # hub_model_id=hub_model_id,
        # hub_strategy="end",
    )
    # training_args.set_optimizer(
    #     name=OptimizerNames.ADAMW_8BIT,
    #     learning_rate=5e-5,
    #     weight_decay=0.0,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-8,
    # )
    # training_args.set_lr_scheduler(
    #     name=SchedulerType.COSINE,
    #     num_epochs=args.epochs,
    #     warmup_ratio=0.05,
    # )

    trainer = SHIFTSegformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="./segformer_output", help="Output dir to store results.")
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.00006, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between eval/checkpoints.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("-tf32", "--use-tf32", action="store_true", default=False, help="Set to True if your setup supports TF32 dtype.")
    parser.add_argument("-mv", "--use-minival", action="store_true", default=False, help="Use the minival validation set.")
    parser.add_argument("-bnb", "--use-adam8bit", action="store_true", default=False, help="Use ADAMW_8BIT optimizer (linux only).")

    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.use_tf32:
        logger.info("Using TF32 dtype.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    main(args)
