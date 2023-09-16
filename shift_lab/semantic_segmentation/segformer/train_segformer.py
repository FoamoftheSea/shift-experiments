from typing import Optional

import torch
import wandb
from argparse import ArgumentParser

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.dataloader.image_processors import SegformerMultitaskImageProcessor
from shift_dev.utils.backend import FileBackend
from torchvision.transforms import v2
from transformers import (
    SegformerImageProcessor,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from transformers.utils import logging

from shift_lab.semantic_segmentation.segformer.constants import SegformerTask
from shift_lab.semantic_segmentation.shift_labels import id2label as shift_id2label, shift2cityscapes
from shift_lab.semantic_segmentation.segformer.metrics import SegformerEvalMetrics
from shift_lab.semantic_segmentation.segformer.trainer import (
    MultitaskSegformerTrainer,
    MultitaskSegformer,
)

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET = "shift"
TRAIN_ONTOLOGY = "cityscapes"

if TRAIN_ONTOLOGY == "shift":
    # Omitting these classes from eval in accordance with Cityscapes
    DO_REDUCE_LABELS = True
    EVAL_IGNORE_IDS = {k for k, v in shift_id2label.items() if v.ignoreInEval}
    id2label = {k: v.name for k, v in shift_id2label.items()}
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
elif TRAIN_ONTOLOGY == "cityscapes":
    from shift_lab.semantic_segmentation.labels import id2label
    DO_REDUCE_LABELS = False
    EVAL_IGNORE_IDS = {v.trainId for k, v in id2label.items() if v.ignoreInEval}
    CLASS_LOSS_WEIGHTS = None

    if TRAIN_DATASET == "shift":
        CLASS_ID_REMAP = {k: id2label[v.cityscapesId].trainId for k, v in shift_id2label.items()}
    else:
        CLASS_ID_REMAP = None

    id2label = {v.trainId: v.name for k, v in id2label.items() if v.trainId != 255}
    id2label[255] = "unlabeled"

else:
    raise ValueError("TRAIN_ONTOLOGY must be 'shift' or 'cityscapes'")

label2id = {v: k for k, v in id2label.items()}

if DO_REDUCE_LABELS and 0 in id2label.keys():
    CLASS_LOSS_WEIGHTS.append(CLASS_LOSS_WEIGHTS.pop(0))

# PRETRAINED_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
# PRETRAINED_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
# PRETRAINED_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
PRETRAINED_MODEL_NAME = "nvidia/mit-b0"
IMAGE_TRANSFORMS = [
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
]
FRAME_TRANSFORMS = []
TRAIN_FULL_RES = True
EVAL_FULL_RES = True
# Size of image passed to train/val if not FULL_RES
TRAIN_IMAGE_SIZE = {"height": 800, "width": 1280}


def main(args):

    tasks = []
    if args.semseg:
        tasks.append(SegformerTask.SEMSEG)
    if args.depth:
        tasks.append(SegformerTask.DEPTH)
    segformer_metrics = SegformerEvalMetrics(
        id2label=id2label, tasks=tasks, ignore_class_ids=EVAL_IGNORE_IDS, reduced_labels=DO_REDUCE_LABELS
    )

    def compute_metrics(eval_pred, calculate_result=True) -> Optional[dict]:
        task_names = {"logits": "semseg", "depth_pred": "depth"}
        label_names = {"logits": "labels", "depth_pred": "depth_labels"}

        with torch.no_grad():
            predictions, labels = eval_pred
    
            for pred_name, prediction in predictions.items():
                label = labels[label_names[pred_name]]
                segformer_metrics.update(task_names[pred_name], prediction, label)
    
            return segformer_metrics.compute() if calculate_result else None

    model = MultitaskSegformer.from_pretrained(
        args.checkpoint if args.checkpoint is not None else PRETRAINED_MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        tasks=tasks,
        do_reduce_labels=DO_REDUCE_LABELS,
    )
    # Set loss weights to the device where loss is calculated
    if CLASS_LOSS_WEIGHTS is not None:
        model.class_loss_weights = torch.tensor(CLASS_LOSS_WEIGHTS).to(device)

    image_processor_train = SegformerMultitaskImageProcessor.from_pretrained(
        PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS, class_id_remap=CLASS_ID_REMAP,
    )
    image_processor_train.size = TRAIN_IMAGE_SIZE
    image_processor_val = SegformerMultitaskImageProcessor.from_pretrained(
        PRETRAINED_MODEL_NAME, do_reduce_labels=DO_REDUCE_LABELS, class_id_remap=CLASS_ID_REMAP,
    )
    image_processor_val.size = TRAIN_IMAGE_SIZE

    keys_to_load = [
        Keys.images,  # images, shape (1, 3, H, W), uint8 (RGB)
        Keys.intrinsics,  # camera intrinsics, shape (3, 3)
        # Keys.timestamp,
        # Keys.axis_mode,
        # Keys.extrinsics,
        # Keys.boxes2d,               # 2D boxes in image coordinate, (x1, y1, x2, y2)
        # Keys.boxes2d_classes,       # class indices, shape (num_boxes,)
        # Keys.boxes2d_track_ids,     # object ids, shape (num_ins,)
        # Keys.boxes3d,               # 3D boxes in camera coordinate, (x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z)
        # Keys.boxes3d_classes,       # class indices, shape (num_boxes,), the same as 'boxes2d_classes'
        # Keys.boxes3d_track_ids,     # object ids, shape (num_ins,), the same as 'boxes2d_track_ids'
        # Keys.segmentation_masks,  # semantic masks, shape (1, H, W), long
        # Keys.masks,                 # instance masks, shape (num_ins, H, W), binary
        # Keys.depth_maps,  # depth maps, shape (1, H, W), float (meters)
    ]
    training_tasks = set()
    if args.semseg:
        keys_to_load.append(Keys.segmentation_masks)
        training_tasks.add("semseg")
    if args.depth:
        keys_to_load.append(Keys.depth_maps)
        training_tasks.add("depth")

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

    trainer = MultitaskSegformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        training_tasks=training_tasks,
    )
    if args.eval_only:
        trainer.evaluate()
    else:
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
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between validation runs.")
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
