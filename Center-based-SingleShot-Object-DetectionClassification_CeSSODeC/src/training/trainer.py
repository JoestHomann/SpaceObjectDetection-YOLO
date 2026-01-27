# trainer.py
# Main training orchestrator.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
Main training orchestrator.

Responsibilities:
- Initialize model, optimizer, loss function
- Run training epochs
- Save checkpoints (last and best)
- Coordinate TensorBoard logging
"""

from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from config import RunConfig
from model import CeSSODeCModel
from losses import SingleObjectLoss
from checkpointIO import save_checkpoint, load_checkpoint

from training.data_utils import build_loaders
from training.train_loop import train_one_epoch
from training.validation import validate
from training.tensorboard_logger import (
    log_training_metrics,
    log_validation_metrics,
    log_confusion_matrix,
    log_predictions_vs_gt
)


def fit(cfg: RunConfig) -> None:
    """
    Full training loop:
    - Initialize model / optimizer / loss
    - Run epochs
    - Save last & best checkpoints
    - Log metrics to TensorBoard

    Args:
        cfg: RunConfig containing all configuration parameters
    """
    # TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/tensorboard")  # TODO: Make dynamic with timestamp or so

    torch.manual_seed(cfg.train.seed)  # Set seed for reproducibility

    device = torch.device(cfg.train.device)  # Device from config

    loaders = build_loaders(cfg)  # Build data loaders

    # Print config summary
    print(
        f"device={cfg.train.device} | epochs={cfg.train.epochs} | batch_size={cfg.train.batch_size} | "
        f"lr={cfg.train.lr} | weight_decay={cfg.train.weight_decay} | amp={cfg.train.activateAMP} | "
        f"num_workers={cfg.train.num_workers} | imgsz={cfg.grid.imgsz} | stride_S={cfg.grid.stride_S}"
    )
    print(f"train_batches={len(loaders['train'])} | val_batches={len(loaders['val'])}")

    # Initialize model, loss function, optimizer
    model = CeSSODeCModel(cfg.model, cfg.grid).to(device)
    loss_fn = SingleObjectLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    start_epoch = 0
    best_acc = 0.0

    # Resume if last checkpoint exists
    if cfg.train.ckpt_last_path is not None and Path(cfg.train.ckpt_last_path).is_file():
        meta = load_checkpoint(cfg.train.ckpt_last_path, model, optimizer)
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_acc = float(meta.get("best_acc", 0.0))

    # Train epochs and validate
    for epoch in range(start_epoch, cfg.train.epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            loaders["train"],
            cfg,
        )

        # Validate
        val_metrics = validate(
            model,
            loss_fn,
            loaders["val"],
            cfg,
        )

        # Print epoch summary
        print(
            f"epoch {epoch+1}/{cfg.train.epochs} | "
            f"train: total={train_metrics['Loss_total']:.4f} "
            f"center={train_metrics['Loss_center']:.4f} "
            f"box={train_metrics['Loss_box']:.4f} "
            f"class={train_metrics['Loss_class']:.4f} | "
            f"val: acc={val_metrics['accuracy']:.4f} "
            f"center_acc={val_metrics['center_acc']:.4f} "
            f"val_total={val_metrics['Loss_total']:.4f} | "
            f"best_acc={best_acc:.4f}"
        )

        # TensorBoard logging
        log_training_metrics(writer, train_metrics, epoch)
        log_validation_metrics(writer, val_metrics, epoch)
        writer.flush()  # Ensure data is written to disk

        # Confusion Matrix
        class_names = getattr(cfg.model, "class_names", None)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(int(getattr(cfg.model, "num_classes", 11)))]

        log_confusion_matrix(writer, val_metrics["confusion_matrix"], class_names, epoch)
        writer.flush()

        # Predictions vs Ground Truth Visualization
        if epoch % 1 == 0:  # Log every n epochs to save space
            log_predictions_vs_gt(writer, model, loaders["val"], cfg, epoch, images_to_visualize=4)
            writer.flush()

        acc = val_metrics["accuracy"]

        # Save last checkpoint
        save_checkpoint(
            cfg.train.ckpt_last_path,
            model,
            optimizer,
            meta={
                "epoch": epoch,
                "best_acc": best_acc,
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        # Save best checkpoint
        if acc > best_acc and cfg.train.ckpt_best_path is not None:
            best_acc = acc
            print(f"  new best: epoch={epoch+1} | best_acc={best_acc:.4f} -> saving best checkpoint")
            save_checkpoint(
                cfg.train.ckpt_best_path,
                model,
                optimizer,
                meta={
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
            )

    # Close TensorBoard writer
    writer.close()

    return
