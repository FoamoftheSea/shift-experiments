from collections import Counter
from typing import Set, Optional

import numpy as np
import torch
from torch import nn

from shift_lab.semantic_segmentation.labels import id2label


# from torchmetrics import Metric
# from torchmetrics.classification import Accuracy

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
