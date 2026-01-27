# metrics.py
# Utilities for calculating training and validation metrics.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
Metrics calculation utilities.

Responsibilities:
- Calculate center prediction accuracy
- Calculate classification accuracy
- Compute per-class accuracies
- Build confusion matrices
"""

import torch
from config import RunConfig


def calculate_center_prediction(center_pred: torch.Tensor, cfg: RunConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate predicted center cell indices from center prediction heatmap.

    Args:
        center_pred: Center predictions from model [B, 1, H, W]
        cfg: RunConfig containing grid configuration

    Returns:
        tuple of (i_hat, j_hat) - predicted row and column indices
    """
    batch_size = center_pred.shape[0]
    center_flat = center_pred[:, 0].reshape(batch_size, -1)
    pred_center_cell_index = torch.argmax(center_flat, dim=1)

    W = cfg.grid.W
    i_hat = pred_center_cell_index // W
    j_hat = pred_center_cell_index % W

    return i_hat, j_hat


def calculate_center_accuracy(i_hat: torch.Tensor, j_hat: torch.Tensor, 
                              gridIndices_gt: torch.Tensor) -> int:
    """
    Calculate number of correct center predictions.

    Args:
        i_hat: Predicted row indices
        j_hat: Predicted column indices  
        gridIndices_gt: Ground truth grid indices [B, 2]

    Returns:
        Number of correct center predictions
    """
    center_correct = ((i_hat == gridIndices_gt[:, 0]) & 
                     (j_hat == gridIndices_gt[:, 1])).sum().item()
    return center_correct


def calculate_classification_accuracy(cls_pred: torch.Tensor, i_hat: torch.Tensor, 
                                      j_hat: torch.Tensor, cls_gt: torch.Tensor,
                                      device: torch.device) -> int:
    """
    Calculate number of correct class predictions at predicted center locations.

    Args:
        cls_pred: Class predictions from model [B, num_classes, H, W]
        i_hat: Predicted row indices
        j_hat: Predicted column indices
        cls_gt: Ground truth class labels
        device: Device tensors are on

    Returns:
        Number of correct class predictions
    """
    batch_size = cls_pred.shape[0]
    cls_logits = cls_pred[torch.arange(batch_size, device=device), :, i_hat, j_hat]
    cls_hat = torch.argmax(cls_logits, dim=1)
    correct = (cls_hat == cls_gt).sum().item()
    return correct


def update_per_class_metrics(cls_pred: torch.Tensor, i_hat: torch.Tensor,
                             j_hat: torch.Tensor, cls_gt: torch.Tensor,
                             correct_per_class: torch.Tensor, 
                             total_per_class: torch.Tensor,
                             device: torch.device, num_classes: int) -> None:
    """
    Update per-class accuracy counters.

    Args:
        cls_pred: Class predictions from model [B, num_classes, H, W]
        i_hat: Predicted row indices
        j_hat: Predicted column indices
        cls_gt: Ground truth class labels
        correct_per_class: Tensor to update with correct counts per class
        total_per_class: Tensor to update with total counts per class
        device: Device tensors are on
        num_classes: Number of classes
    """
    batch_size = cls_pred.shape[0]
    cls_logits = cls_pred[torch.arange(batch_size, device=device), :, i_hat, j_hat]
    cls_hat = torch.argmax(cls_logits, dim=1)

    for c in range(num_classes):
        mask = (cls_gt == c)
        total_per_class[c] += mask.sum().item()
        correct_per_class[c] += ((cls_hat == cls_gt) & mask).sum().item()


def update_confusion_matrix(cls_pred: torch.Tensor, i_hat: torch.Tensor,
                           j_hat: torch.Tensor, cls_gt: torch.Tensor,
                           confusion_matrix: torch.Tensor, device: torch.device) -> None:
    """
    Update confusion matrix with predictions.

    Args:
        cls_pred: Class predictions from model [B, num_classes, H, W]
        i_hat: Predicted row indices
        j_hat: Predicted column indices
        cls_gt: Ground truth class labels
        confusion_matrix: Confusion matrix to update [num_classes, num_classes]
        device: Device tensors are on
    """
    batch_size = cls_pred.shape[0]
    cls_logits = cls_pred[torch.arange(batch_size, device=device), :, i_hat, j_hat]
    cls_hat = torch.argmax(cls_logits, dim=1)

    for true_label, predicted_label in zip(cls_gt.view(-1), cls_hat.view(-1)):
        tL = int(true_label.item())
        pL = int(predicted_label.item())
        confusion_matrix[tL, pL] += 1
