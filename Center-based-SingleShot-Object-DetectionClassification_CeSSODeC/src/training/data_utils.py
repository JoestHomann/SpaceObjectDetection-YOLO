# data_utils.py
# Utilities for building data loaders.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
Data loader building utilities.

Responsibilities:
- Build PyTorch DataLoaders for training and validation
"""

from torch.utils.data import DataLoader
from config import RunConfig
from dataset import SingleObjectYoloDataset


def build_loaders(cfg: RunConfig) -> dict[str, DataLoader]:
    """
    Builds PyTorch DataLoaders for training and validation.

    Args:
        cfg: RunConfig containing data and training configuration

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
