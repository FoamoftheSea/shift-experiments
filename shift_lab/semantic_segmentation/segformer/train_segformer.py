from typing import Optional

import torch
from argparse import ArgumentParser

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from torchvision.transforms import v2
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from transformers.utils import logging

from shift_lab.semantic_segmentation.labels import id2label
from shift_lab.semantic_segmentation.segformer.metrics import SHIFTSegformerEvalMetrics
from shift_lab.semantic_segmentation.segformer.trainer import SHIFTSegformerTrainer

id2label = {k: v.name for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

logger = logging.get_logger(__name__)

PRETRAINED_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
IMAGE_TRANSFORMS = [
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
]
FRAME_TRANSFORMS = []
EVAL_FULL_RES = True
image_size = {"height": 512, "width": 1024}
image_processor_train = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME, do_reduce_labels=True)
image_processor_train.size = image_size
image_processor_val = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME, do_reduce_labels=True)
image_processor_val.size = image_size

KEYS_TO_LOAD = [
    Keys.images,                # note: images, shape (1, 3, H, W), uint8 (RGB)
    Keys.intrinsics,            # note: camera intrinsics, shape (3, 3)
    Keys.timestamp,
    Keys.axis_mode,
    Keys.extrinsics,
    Keys.boxes2d,               # note: 2D boxes in image coordinate, (x1, y1, x2, y2)
    Keys.boxes2d_classes,       # note: class indices, shape (num_boxes,)
    Keys.boxes2d_track_ids,     # note: object ids, shape (num_ins,)
    # Keys.boxes3d,               # note: 3D boxes in camera coordinate, (x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z)
    # Keys.boxes3d_classes,       # note: class indices, shape (num_boxes,), the same as 'boxes2d_classes'
    # Keys.boxes3d_track_ids,     # note: object ids, shape (num_ins,), the same as 'boxes2d_track_ids'
    Keys.segmentation_masks,    # note: semantic masks, shape (1, H, W), long
    # Keys.masks,                 # note: instance masks, shape (num_ins, H, W), binary
    # Keys.depth_maps,            # note: depth maps, shape (1, H, W), float (meters)
]

metric = SHIFTSegformerEvalMetrics(ignore_class_ids={255})


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
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

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
        split="val",
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
        # optim=OptimizerNames.ADAMW_8BIT,
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
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between eval/checkpoints.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")

    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    main(args)
