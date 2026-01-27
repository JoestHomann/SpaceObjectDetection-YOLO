# train.py
# Training loops for Center-based Single Shot Object Detection and Classification (CeSSODeC).
#
# DEPRECATED: This file is kept for backwards compatibility.
# Please use the modular training package instead:
#   from training import fit, train_one_epoch, validate, build_loaders
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Email:                    st171800@stud.uni-stuttgart.de
# Created:                  24-Jan-2026 15:00:00
# Refactored:               27-Jan-2026
#
# Revision history:
#   - Added TensorBoard logging (25-Jan-2026, J. Homann)
#   - Added more TB logging and confusion matrix (27-Jan-2026, J. Homann)
#   - Refactored into modular structure (27-Jan-2026, Refactoring)
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

"""
Training loops and related functions.

DEPRECATED: This module now re-exports functions from the refactored training package.

The training functionality has been reorganized into a modular structure:
- training.data_utils: DataLoader building
- training.train_loop: Training epoch execution
- training.validation: Validation loop
- training.metrics: Accuracy and metrics calculation
- training.tensorboard_logger: TensorBoard logging
- training.trainer: Main training orchestrator

For new code, import directly from the training package:
    from training import fit, train_one_epoch, validate, build_loaders
"""

# Re-export functions from the refactored training package for backwards compatibility
from training.trainer import fit
from training.train_loop import train_one_epoch
from training.validation import validate
from training.data_utils import build_loaders

# Expose the functions in the module namespace
__all__ = ["fit", "train_one_epoch", "validate", "build_loaders"]



