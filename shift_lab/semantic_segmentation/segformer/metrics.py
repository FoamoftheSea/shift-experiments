import numpy as np
import torch
from collections import Counter
from typing import Set, Optional, List, Dict


class SiLogLoss(torch.nn.Module):
    r"""
    Implements the Scale-invariant log scale loss [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ where $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """

    def __init__(self, lambd=0.5, log_predictions=True, log_labels=False):
        super().__init__()
        self.lambd = lambd
        self.log_predictions = log_predictions
        self.log_labels = log_labels

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = target[valid_mask] if self.log_labels else torch.log(target[valid_mask])
        y_hat = pred[valid_mask] if self.log_predictions else torch.log(pred[valid_mask])
        diff = y_hat - y
        loss = torch.pow(diff, 2).mean() - self.lambd * torch.pow(diff.mean(), 2)

        return loss


class IRMSELoss(torch.nn.Module):
    r"""
    Implements the Root Mean Squared Error of Inverse Depth [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).
    """

    def __init__(self, log_predictions=True, log_labels=False):
        super().__init__()
        self.log_predictions = log_predictions
        self.log_labels = log_labels

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = torch.exp(target[valid_mask]) if self.log_labels else target[valid_mask]
        y_hat = torch.exp(pred[valid_mask]) if self.log_predictions else pred[valid_mask]
        irmse = torch.sqrt(torch.pow((1000 / y) - (1000 / y_hat), 2).mean())

        return irmse


class SegformerEvalMetrics:

    def __init__(
        self,
        tasks: List[str],
        id2label: Optional[Dict[int, str]] = None,
        ignore_class_ids: Optional[Set[int]] = None,
        reduced_labels: bool = False,
    ):
        assert all(task in ["semseg", "depth"] for task in tasks), "Unrecognized task passed to metrics."
        self.tasks = tasks
        self.metrics = {}
        if "semseg" in tasks:
            assert id2label is not None, "Must pass id2label dict to SegformerEvalMetrics when training semseg."
            self.metrics["semseg"] = SegformerSemanticSegEvalMetric(
                id2label=id2label,
                ignore_class_ids=ignore_class_ids,
                reduced_labels=reduced_labels
            )
        if "depth" in tasks:
            self.metrics["depth"] = SegformerDepthEvalMetric()

    def update(self, task, prediction, label):
        self.metrics[task].update(prediction, label)

    def compute(self):
        return {task: self.metrics[task].compute() for task in self.tasks}


class SegformerSemanticSegEvalMetric:
    def __init__(
            self,
            id2label: Dict[int, str],
            ignore_class_ids: Optional[Set[int]] = None,
            reduced_labels: bool = False
    ):
        self.total_area_intersect = Counter()
        self.total_area_union = Counter()
        self.total_label_area = Counter()
        self.ignore_class_ids = ignore_class_ids or set()
        self.reduced_labels = reduced_labels
        self.id2label = id2label

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

        for class_id in self.id2label.keys():
            if class_id in self.ignore_class_ids:
                continue
            if self.reduced_labels:
                label_id = class_id - 1 if class_id != 0 else 255
            else:
                label_id = class_id
            pred_pixels = pred_labels == label_id
            gt_pixels = gt_labels == label_id
            class_label = self.id2label[class_id]
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
    def __init__(self, silog_lambda=0.5, log_predictions=True, log_labels=False, mask_value=0.0):
        self.batch_mae = []
        self.batch_mse = []
        self.batch_rmse = []
        self.batch_irmse = []
        self.batch_silog = []
        self.irmse_loss = IRMSELoss(log_predictions=log_predictions, log_labels=log_labels)
        self.silog_loss = SiLogLoss(lambd=silog_lambda, log_predictions=log_predictions, log_labels=log_labels)
        self.log_predictions = log_predictions
        self.log_labels = log_labels
        self.mask_value = mask_value

    def update(self, prediction, label):
        valid_pixels = np.where(label != self.mask_value)
        y = np.exp(label[valid_pixels]) if self.log_labels else label[valid_pixels]
        y_hat = np.exp(prediction[valid_pixels]) if self.log_predictions else prediction[valid_pixels]
        self.batch_mae.append(np.mean(np.abs(y - y_hat)))
        batch_mse = np.mean(np.power(y - y_hat, 2))
        self.batch_mse.append(batch_mse)
        self.batch_rmse.append(np.sqrt(batch_mse))
        self.batch_irmse.append(self.irmse_loss(torch.from_numpy(prediction), torch.from_numpy(label)))
        self.batch_silog.append(self.silog_loss(torch.from_numpy(prediction), torch.from_numpy(label)))

    def compute(self):
        return {
            "mae": np.mean(self.batch_mae),
            "mse": np.mean(self.batch_mse),
            "rmse": np.mean(self.batch_rmse),
            "irmse": np.mean(self.batch_irmse),
            "silog": np.mean(self.batch_silog),
        }
