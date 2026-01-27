# validation.py
# Validation loop implementation.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
Validation loop implementation.

Responsibilities:
- Run validation on a dataset
- Compute validation losses
- Calculate accuracy metrics
- Generate confusion matrix
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from config import RunConfig
from training.metrics import (
    calculate_center_prediction,
    calculate_center_accuracy,
    update_per_class_metrics,
    update_confusion_matrix
)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    cfg: RunConfig,
) -> Dict[str, Any]:
    """
    Validation:
    - Compute val loss (same loss as training)
    - Decode center argmax and class argmax at that cell
    - Metrics: accuracy, center_acc, per-class accuracy, confusion matrix

    Args:
        model: The model to validate
        loss_fn: The loss function
        loader: DataLoader for validation data
        cfg: RunConfig with validation settings

    Returns:
        dict with validation metrics:
            {
                "accuracy": overall classification accuracy,
                "center_acc": center prediction accuracy,
                "per_class_acc": list of per-class accuracies,
                "confusion_matrix": numpy array of confusion matrix,
                "Loss_center", "Loss_box", "Loss_class", "Loss_total": average losses
            }
    """
    model.eval()
    device = cfg.train.device

    # Accuracy
    correct = 0
    total = 0

    # Center hit rate
    center_correct = 0

    # Per-class accuracy
    num_classes = int(getattr(cfg.model, "num_classes", 11))
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    # Confusion matrix counts (rows: true class, columns: predicted)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    # Val loss accumulation (same keys as train_one_epoch)
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0,
                 "Loss_class": 0.0, "Loss_total": 0.0}
    n_batches = 0

    for x, ij_gt, bbox_gt_norm, cls_gt in loader:
        x = x.to(device)
        ij_gt = ij_gt.to(device)
        bbox_gt_norm = bbox_gt_norm.to(device)
        cls_gt = cls_gt.to(device)

        center_pred, box_pred, cls_pred = model(x)

        # Val losses
        losses = loss_fn(
            center_preds=center_pred,
            box_preds=box_pred,
            class_preds=cls_pred,
            gridIndices_gt=ij_gt,
            bbox_gt_norm=bbox_gt_norm,
            cls_gt=cls_gt,
        )
        for k in loss_sums:
            loss_sums[k] += float(losses[k].item())
        n_batches += 1

        # Decode center predictions
        i_hat, j_hat = calculate_center_prediction(center_pred, cfg)

        # Center hit
        center_correct += calculate_center_accuracy(i_hat, j_hat, ij_gt)

        # Update confusion matrix
        update_confusion_matrix(cls_pred, i_hat, j_hat, cls_gt, confusion_matrix, device)

        # Classification accuracy at predicted center
        B = x.shape[0]
        cls_logits = cls_pred[torch.arange(B, device=device), :, i_hat, j_hat]
        cls_hat = torch.argmax(cls_logits, dim=1)

        correct += (cls_hat == cls_gt).sum().item()
        total += B

        # Per-class accuracy
        update_per_class_metrics(
            cls_pred, i_hat, j_hat, cls_gt,
            correct_per_class, total_per_class,
            device, num_classes
        )

    acc = correct / max(total, 1)
    center_acc = center_correct / max(total, 1)

    val_losses = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}

    per_class_acc = (correct_per_class.float() /
                     torch.clamp(total_per_class.float(), min=1.0)).cpu().tolist()

    # Convert confusion matrix to numpy for easier handling outside torch
    confusion_matrix_numpy = confusion_matrix.cpu().numpy()

    return {
        "accuracy": acc,
        "center_acc": center_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion_matrix_numpy,
        **val_losses,
    }
