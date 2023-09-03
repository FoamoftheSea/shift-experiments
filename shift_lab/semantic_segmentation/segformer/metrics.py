import numpy as np
from collections import Counter
from typing import Set, Optional

from shift_lab.semantic_segmentation.shift_labels import id2label


class SHIFTSegformerEvalMetrics:
    def __init__(self, ignore_class_ids: Optional[Set[int]] = None, reduced_labels: bool = False):
        self.total_area_intersect = Counter()
        self.total_area_union = Counter()
        self.total_label_area = Counter()
        self.ignore_class_ids = ignore_class_ids or set()
        self.reduced_labels = reduced_labels

    def update(self, pred_labels: np.ndarray, gt_labels: np.ndarray):

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
        }
        metrics.update(accuracies)
        metrics.update(ious)

        return metrics
