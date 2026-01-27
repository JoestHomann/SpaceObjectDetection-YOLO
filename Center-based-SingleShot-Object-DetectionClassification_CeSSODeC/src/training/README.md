# Training Module

This module contains the refactored training code for Center-based Single Shot Object Detection and Classification (CeSSODeC).

## Overview

The training functionality has been reorganized from a single monolithic `train.py` file into a modular structure for better maintainability and readability.

## Module Structure

```
training/
├── __init__.py              # Module initialization, exports main functions
├── data_utils.py            # DataLoader building utilities
├── train_loop.py            # Training epoch execution
├── validation.py            # Validation loop and metrics
├── metrics.py               # Accuracy and metrics calculation utilities
├── tensorboard_logger.py    # TensorBoard logging utilities
└── trainer.py               # Main training orchestrator
```

## Components

### data_utils.py
- `build_loaders(cfg)`: Builds PyTorch DataLoaders for training and validation splits

### train_loop.py
- `train_one_epoch(model, loss_fn, optimizer, loader, cfg)`: Executes a single training epoch with AMP support

### validation.py
- `validate(model, loss_fn, loader, cfg)`: Runs validation and computes metrics

### metrics.py
Utility functions for calculating training and validation metrics:
- `calculate_center_prediction(center_pred, cfg)`: Decode center predictions
- `calculate_center_accuracy(i_hat, j_hat, gridIndices_gt)`: Calculate center hit accuracy
- `calculate_classification_accuracy(...)`: Calculate classification accuracy
- `update_per_class_metrics(...)`: Update per-class accuracy counters
- `update_confusion_matrix(...)`: Update confusion matrix

### tensorboard_logger.py
Utilities for logging to TensorBoard:
- `log_training_metrics(writer, metrics, epoch)`: Log training losses and accuracies
- `log_validation_metrics(writer, metrics, epoch)`: Log validation losses and accuracies
- `log_confusion_matrix(writer, confusion_matrix, class_names, epoch)`: Log confusion matrix
- `log_predictions_vs_gt(writer, model, loader, cfg, epoch, images_to_visualize)`: Log prediction visualizations

### trainer.py
- `fit(cfg)`: Main training orchestrator that initializes components and runs the training loop

## Usage

### Recommended (New Code)
```python
from training import fit

# Your config setup
cfg = RunConfig(...)

# Run training
fit(cfg)
```

### Backwards Compatible (Existing Code)
```python
from train import fit  # Still works, imports from refactored modules

# Your config setup
cfg = RunConfig(...)

# Run training
fit(cfg)
```

### Direct Module Access
```python
from training.train_loop import train_one_epoch
from training.validation import validate
from training.data_utils import build_loaders

# Use individual components as needed
loaders = build_loaders(cfg)
metrics = train_one_epoch(model, loss_fn, optimizer, loaders["train"], cfg)
val_metrics = validate(model, loss_fn, loaders["val"], cfg)
```

## Benefits of Refactoring

1. **Modularity**: Each file has a single, clear responsibility
2. **Maintainability**: Easier to find and modify specific functionality
3. **Testability**: Individual components can be tested in isolation
4. **Readability**: Smaller files are easier to understand
5. **Reusability**: Components can be imported and used independently
6. **Backwards Compatibility**: Existing code continues to work without changes

## Migration Guide

No changes are required to existing code! The original `train.py` now re-exports all functions from the new modular structure, so all existing imports will continue to work.

If you want to update to use the new structure:

**Before:**
```python
from train import fit, train_one_epoch, validate, build_loaders
```

**After (optional):**
```python
from training import fit, train_one_epoch, validate, build_loaders
```

Both approaches work identically.

## Authors

- J. Homann
- C. Kern
- F. Kolb

## Revision History

- 27-Jan-2026: Refactored train.py into modular structure
- 25-Jan-2026: Added TensorBoard logging
- 24-Jan-2026: Initial implementation
