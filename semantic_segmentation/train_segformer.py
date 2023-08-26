from argparse import ArgumentParser

import evaluate
import torch
from torch import nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
    SchedulerType,
)
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend, FileBackend, DataBackend
from transformers.training_args import OptimizerNames

from labels import id2label
id2label = {k: v.name for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

metric = evaluate.load("mean_iou")
feature_extractor = SegformerImageProcessor()

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

# KEYS_TO_LOAD = []
# groups_to_load = ["img", "semseg"]
# for group in groups_to_load:
#     KEYS_TO_LOAD.extend(SHIFTDataset.DATA_GROUPS[group])


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics


def main(args):
    pretrained_model_name = "nvidia/mit-b0"
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )

    train_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="train",
        keys_to_load=KEYS_TO_LOAD,
        views_to_load=["front"],
        shift_type="discrete",          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=FileBackend(),           # also supports HDF5Backend(), FileBackend()
        verbose=True,
    )

    val_dataset = SHIFTDataset(
        data_root=args.data_root,
        split="val",
        keys_to_load=KEYS_TO_LOAD,
        views_to_load=["front"],
        shift_type="discrete",
        backend=FileBackend(),
        verbose=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
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

    trainer = Trainer(
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
    parser.add_argument("-d", "--data-root", type=str, default="F:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.00006, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Train and eval batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("-gc", "--gradient-checkpointing", action="store_true", help="Turn on gradient checkpointing")

    args = parser.parse_args()
    main(args)
