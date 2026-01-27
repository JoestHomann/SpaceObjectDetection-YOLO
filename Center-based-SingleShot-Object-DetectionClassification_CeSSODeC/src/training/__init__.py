# __init__.py
# Training module initialization.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026

"""
Training module for Center-based Single Shot Object Detection and Classification (CeSSODeC).

This module contains refactored components from train.py:
- data_utils: DataLoader building
- train_loop: Training epoch execution
- validation: Validation loop
- metrics: Accuracy and metrics calculation
- tensorboard_logger: TensorBoard logging utilities
- trainer: Main training orchestrator

Usage:
    from training.trainer import fit
    fit(cfg)
"""

from training.trainer import fit
from training.train_loop import train_one_epoch
from training.validation import validate
from training.data_utils import build_loaders

__all__ = [
    "fit",
    "train_one_epoch",
    "validate",
    "build_loaders",
]
