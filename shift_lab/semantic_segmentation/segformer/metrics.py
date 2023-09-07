import numpy as np
import torch
from collections import Counter
from typing import Set, Optional, List

from shift_lab.semantic_segmentation.shift_labels import id2label
from transformers.models.glpn.modeling_glpn import SiLogLoss


class SegformerEvalMetrics:

    def __init__(
        self,
        tasks: List[str],
        semseg_ignore_class_ids: Optional[Set[int]] = None,
        semseg_reduced_labels: bool = False,
    ):
        assert all(task in ["semseg", "depth"] for task in tasks), "Unrecognized task passed to metrics."
        self.tasks = tasks
        self.metrics = {}
        if "semseg" in tasks:
            self.metrics["semseg"] = SegformerSemanticSegEvalMetric(ignore_class_ids=semseg_ignore_class_ids, reduced_labels=semseg_reduced_labels)
        if "depth" in tasks:
            self.metrics["depth"] = SegformerDepthEvalMetric()

    def update(self, task, prediction, label):
        self.metrics[task].update(prediction, label)

    def compute(self):
        return {task: self.metrics[task].compute() for task in self.tasks}


class SegformerSemanticSegEvalMetric:
    def __init__(self, ignore_class_ids: Optional[Set[int]] = None, reduced_labels: bool = False):
        self.total_area_intersect = Counter()
        self.total_area_union = Counter()
        self.total_label_area = Counter()
        self.ignore_class_ids = ignore_class_ids or set()
        self.reduced_labels = reduced_labels

    def update(self, logits: np.ndarray, gt_labels: np.ndarray):

        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=gt_labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()

        for class_id in id2label.keys():
            if class_id in self.ignore_class_ids:
                continue
            if self.reduced_labels:
                label_id = class_id - 1 if class_id != 0 else 255
            else:
                label_id = class_id
            pred_pixels = pred_labels == label_id
            gt_pixels = gt_labels == label_id
            class_label = id2label[class_id]
            self.total_area_intersect.update({class_label: np.sum(np.bitwise_and(pred_pixels, gt_pixels))})
            self.total_area_union.update({class_label: np.sum(np.bitwise_or(pred_pixels, gt_pixels))})
            self.total_label_area.update({class_label: np.sum(gt_pixels)})

    def compute(self):
        accuracies = {f"accuracy_{k}": self.total_area_intersect[k] / self.total_label_area[k] for k in self.total_area_union}
        ious = {f"iou_{k}": self.total_area_intersect[k] / self.total_area_union[k] for k in self.total_area_union}
        metrics = {
            "overall_accuracy": sum(self.total_area_intersect.values()) / sum(self.total_label_area.values()),
            "mean_accuracy": np.mean(list(accuracies.values())),
            "mean_iou": np.mean(list(ious.values())),
        }
        metrics.update(accuracies)
        metrics.update(ious)

        return metrics


class SegformerDepthEvalMetric:
    def __init__(self, silog_lambda=0.5):
        self.batch_mae = []
        self.batch_mse = []
        self.batch_rmse = []
        self.batch_silog = []
        self.silog_loss_func = SiLogLoss(silog_lambda)

    def update(self, prediction, label):
        self.batch_mae.append(np.mean(np.abs(label - prediction)))
        batch_mse = np.mean(np.power(label - prediction, 2))
        self.batch_mse.append(batch_mse)
        self.batch_rmse.append(np.sqrt(batch_mse))
        self.batch_silog.append(self.silog_loss_func(torch.from_numpy(prediction), torch.from_numpy(label)))

    def compute(self):
        return {
            "mae": np.mean(self.batch_mae),
            "mse": np.mean(self.batch_mse),
            "rmse": np.mean(self.batch_rmse),
            "silog": np.mean(self.batch_silog),
        }
