# train.py
# Training loops for Center-based Single Shot Object Detection and Classification (CeSSODeC).
#
# Details:
#   None
#
# Syntax:
#
# Inputs:
#   None
#
# Outputs:
#   None
#
# Examples:
#   None
#
# See also:
#   None
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Email:                    st171800@stud.uni-stuttgart.de
# Created:                  24-Jan-2026 15:00:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


# TODO: IMPORTANT: Better explain what this file does/what the functions do


"""
Training loop and related functions.

Responsibilities:
- Build dataloaders
- Run training & validation loops
- Handle AMP (optional)
- Save last / best checkpoints

NO model/dataset/loss definitions here - only wiring.
"""
from pathlib import Path

from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import RunConfig
from dataset import SingleObjectYoloDataset
from model import CeSSODeCModel
from losses import SingleObjectLoss
from checkpointIO import save_checkpoint, load_checkpoint

# ---------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------

from torch.utils.data import DataLoader
from dataset import SingleObjectYoloDataset
from config import RunConfig

# Was macht der Dataloader?
# Er lädt die Daten in Batches und bereitet sie für das Training vor.


def build_loaders(cfg: RunConfig) -> dict[str, DataLoader]:
    """
    Builds PyTorch DataLoaders for training and validation.

    Returns:
        dict with keys:
            "train": DataLoader for training split
            "val":   DataLoader for validation split
    """

    train_dataset = SingleObjectYoloDataset(
        data_cfg=cfg.data,
        grid_cfg=cfg.grid,
        data="train",
    )

    val_dataset = SingleObjectYoloDataset(
        data_cfg=cfg.data,
        grid_cfg=cfg.grid,
        data="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
    }


# ---------------------------------------------------------
# TRAIN ONE EPOCH
# ---------------------------------------------------------

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

    Inputs:
        model: The model to train
        loss_fn: The loss function
        optimizer: The optimizer
        loader: DataLoader for training data
        cfg: RunConfig with training settings

    Outputs:
        dict with average losses: {"Loss_center", "Loss_box", "Loss_class", "Loss_total"}
    """
    model.train()   # Model in training mode

    device = cfg.train.device  # Device from config
    amp_enabled = cfg.train.activateAMP
    scaler = GradScaler(enabled=amp_enabled)

    # Initialize loss sums and batch counter
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0,
                 "Loss_class": 0.0, "Loss_total": 0.0}
    n_batches = 0

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

    # Backward pass and optimization step
        scaler.scale(losses["Loss_total"]).backward()
        scaler.step(optimizer)
        scaler.update()
    # Loss accumulation
        for k in loss_sums:
            loss_sums[k] += losses[k].item()

        n_batches += 1

    return {k: v / n_batches for k, v in loss_sums.items()}

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------


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
    - Metrics: accuracy, center_acc, per-class accuracy
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

        # Decode center argmax
        B = x.shape[0]
        center_flat = center_pred[:, 0].reshape(B, -1)
        idx = torch.argmax(center_flat, dim=1)

        W = cfg.grid.W
        i_hat = idx // W
        j_hat = idx % W

        # Center hit
        center_correct += ((i_hat == ij_gt[:, 0])
                           & (j_hat == ij_gt[:, 1])).sum().item()

        # Class at predicted center
        cls_logits = cls_pred[torch.arange(B, device=device), :, i_hat, j_hat]
        cls_hat = torch.argmax(cls_logits, dim=1)

        correct += (cls_hat == cls_gt).sum().item()
        total += B

        # Per-class
        for c in range(num_classes):
            mask = (cls_gt == c)
            total_per_class[c] += mask.sum().item()
            correct_per_class[c] += ((cls_hat == cls_gt) & mask).sum().item()

    acc = correct / max(total, 1)
    center_acc = center_correct / max(total, 1)

    val_losses = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}

    per_class_acc = (correct_per_class.float(
    ) / torch.clamp(total_per_class.float(), min=1.0)).cpu().tolist()

    return {
        "accuracy": acc,
        "center_acc": center_acc,
        "per_class_acc": per_class_acc,
        **val_losses,
    }


# ---------------------------------------------------------
# FIT LOOP
# ---------------------------------------------------------

def fit(cfg: RunConfig) -> None:
    """
    Full training loop:
    - init model / optimizer / loss
    - run epochs
    - save last & best checkpoints
    """
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
        # train one epoch
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            loaders["train"],
            cfg,
        )

        val_metrics = validate(
            model,
            loss_fn,
            loaders["val"],
            cfg,
        )

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

        acc = val_metrics["accuracy"]

        # save last
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


# def build_model_loss_optim(cfg: RunConfig) -> Tuple[CenterSingleObjNet, CenterSingleObjLoss, torch.optim.Optimizer]:
#     """
#     Build model, loss function, and optimizer.

#     Inputs:
#       cfg: RunConfig

#     Outputs:
#       (model, loss_fn, optimizer)
#     """
#     device = torch.device(cfg.train.device)

#     model = CeSSODeCModel(cfg.model, cfg.grid).to(device)
#     loss_fn = SingleObjectLoss(eps=1e-6, box_weight=5.0).to(device)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=cfg.train.lr,
#         weight_decay=cfg.train.weight_decay,
#     )

#     return model, loss_fn, optimizer TODO: Braucht man die Funktion evtl?


# def _set_seed(seed: int) -> None:
#     # Make runs more reproducible.
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed) TODO: Funktion kann implementiert werden, um Seed besser zu setzen
