from argparse import ArgumentParser
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union
from typing import Set, Optional

import numpy as np
import torch
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import (
    FileBackend,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
# from torchmetrics import Metric
# from torchmetrics.classification import Accuracy
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from transformers.data.data_collator import DataCollator
from transformers.deepspeed import deepspeed_init
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_torch_tpu_available,
    logging,
)

from labels import id2label
id2label = {k: v.name for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

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


class SHIFTSegformerTrainer(Trainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction, bool], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_metrics_interval: str = "batch",
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.compute_metrics_interval = compute_metrics_interval

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        metrics = None
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.compute_metrics_interval == "batch":
                is_last_batch = step == len(dataloader) - 1
                if self.compute_metrics is not None and preds_host is not None and labels_host is not None:
                    if args.include_inputs_for_metrics:
                        metrics = self.compute_metrics(
                            EvalPrediction(
                                predictions=nested_numpify(preds_host),
                                label_ids=nested_numpify(labels_host),
                                inputs=nested_numpify(inputs_host),
                            ),
                            calculate_result=is_last_batch,
                        )
                    else:
                        metrics = self.compute_metrics(
                            EvalPrediction(
                                predictions=nested_numpify(preds_host),
                                label_ids=nested_numpify(labels_host),
                            ),
                            calculate_result=is_last_batch,
                        )
                losses = nested_numpify(losses_host)
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            elif (
                self.compute_metrics_interval == "full"
                and args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and self.accelerator.sync_gradients
            ):
                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class SHIFTSegformerEvalMetrics:
    def __init__(self, ignore_class_ids: Optional[Set[int]] = None):
        self.total_area_intersect = Counter()
        self.total_area_union = Counter()
        self.total_label_area = Counter()
        self.ignore_class_ids = ignore_class_ids or set()

    def update(self, pred_labels: np.ndarray, gt_labels: np.ndarray):

        for class_id in np.unique(gt_labels):
            if class_id in self.ignore_class_ids:
                continue
            class_label = id2label[class_id]
            pred_pixels = pred_labels == class_id
            gt_pixels = gt_labels == class_id
            self.total_area_intersect.update({class_label: np.sum(np.bitwise_and(pred_pixels, gt_pixels))})
            self.total_area_union.update({class_label: np.sum(np.bitwise_or(pred_pixels, gt_pixels))})
            self.total_label_area.update({class_label: np.sum(gt_pixels)})

    def compute(self):
        accuracies = {f"accuracy_{k}": self.total_area_intersect[k] / self.total_label_area[k] for k in self.total_area_union}
        ious = {f"iou_{k}": self.total_area_intersect[k] / self.total_area_union[k] for k in self.total_area_union}
        metrics = {
            "overall_accuracy": sum(self.total_area_intersect.values()) / sum(self.total_label_area.values()),
            "mean_accuracy": np.mean(list(accuracies.values())),
        }
        metrics.update(accuracies)
        metrics.update(ious)

        return metrics


metric = SHIFTSegformerEvalMetrics(ignore_class_ids={255})


def compute_metrics(eval_pred, calculate_result=True) -> Optional[dict]:
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
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
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
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Train and eval batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("-gc", "--gradient-checkpointing", action="store_true", default=False, help="Turn on gradient checkpointing")
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between eval/checkpoints.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")

    args = parser.parse_args()
    main(args)
