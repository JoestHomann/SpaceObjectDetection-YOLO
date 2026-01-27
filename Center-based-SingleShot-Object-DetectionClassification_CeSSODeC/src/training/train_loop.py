# train_loop.py
# Training loop for a single epoch.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
Training loop implementation.

Responsibilities:
- Run a single training epoch
- Handle automatic mixed precision
- Compute and accumulate training losses and metrics
"""

from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from config import RunConfig
from training.metrics import calculate_center_prediction, calculate_center_accuracy, calculate_classification_accuracy


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: RunConfig,
) -> Dict[str, float]:
    """
    Run a single training epoch. Uses automatic mixed precision if enabled in config.
    Accumulates and returns average losses over the epoch.

    Args:
        model: The model to train
        loss_fn: The loss function
        optimizer: The optimizer
        loader: DataLoader for training data
        cfg: RunConfig with training settings

    Returns:
        dict with average losses and accuracies:
            {"Loss_center", "Loss_box", "Loss_class", "Loss_total", "accuracy", "center_acc"}
    """
    model.train()  # Model in training mode

    device = cfg.train.device
    amp_enabled = cfg.train.activateAMP
    scaler = GradScaler(enabled=amp_enabled)

    # Initialize loss sums and batch counter
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0,
                 "Loss_class": 0.0, "Loss_total": 0.0}

    # Initialize variables for accuracy calculation (used for logging)
    n_batches = 0       # Number of batches processed
    correct = 0         # Number of correct predictions
    total = 0           # Total number of samples
    center_correct = 0  # Number of correct center predictions

    # Iterate over batches
    for x, gridIndices_gt, bbox_gt_norm, cls_gt in loader:
        x = x.to(device)
        gridIndices_gt = gridIndices_gt.to(device)
        bbox_gt_norm = bbox_gt_norm.to(device)
        cls_gt = cls_gt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast(enabled=amp_enabled):
            center_pred, box_pred, cls_pred = model(x)
            losses = loss_fn(
                center_preds=center_pred,
                box_preds=box_pred,
                class_preds=cls_pred,
                gridIndices_gt=gridIndices_gt,
                bbox_gt_norm=bbox_gt_norm,
                cls_gt=cls_gt,
            )

        # Accuracy calculation (for logging)
        with torch.no_grad():
            batch_size = x.shape[0]

            # Calculate center predictions
            i_hat, j_hat = calculate_center_prediction(center_pred, cfg)

            # Update center accuracy
            center_correct += calculate_center_accuracy(i_hat, j_hat, gridIndices_gt)

            # Update classification accuracy
            correct += calculate_classification_accuracy(
                cls_pred, i_hat, j_hat, cls_gt, device
            )
            total += batch_size

        # Backward pass and optimization step
        scaler.scale(losses["Loss_total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        # Loss accumulation
        for k in loss_sums:
            loss_sums[k] += losses[k].item()

        n_batches += 1

    out = {k: v / n_batches for k, v in loss_sums.items()}
    out["accuracy"] = correct / max(total, 1)
    out["center_acc"] = center_correct / max(total, 1)
    return out
