# tensorboard_logger.py
# TensorBoard logging utilities.
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Created:                  27-Jan-2026
# Refactored from train.py

"""
TensorBoard logging utilities.

Responsibilities:
- Log training and validation metrics to TensorBoard
- Log confusion matrices
- Log prediction vs ground truth visualizations
"""

from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import visualizationHelpers as vh
from config import RunConfig


def log_training_metrics(writer: SummaryWriter, metrics: Dict[str, float], epoch: int) -> None:
    """
    Log training metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary containing training metrics
        epoch: Current epoch number
    """
    # Losses
    writer.add_scalar("train/Loss_total", metrics["Loss_total"], epoch)
    writer.add_scalar("train/Loss_center", metrics["Loss_center"], epoch)
    writer.add_scalar("train/Loss_box", metrics["Loss_box"], epoch)
    writer.add_scalar("train/Loss_class", metrics["Loss_class"], epoch)
    # Accuracies
    writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
    writer.add_scalar("train/center_acc", metrics["center_acc"], epoch)


def log_validation_metrics(writer: SummaryWriter, metrics: Dict[str, Any], epoch: int) -> None:
    """
    Log validation metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary containing validation metrics
        epoch: Current epoch number
    """
    # Losses
    writer.add_scalar("val/Loss_total", metrics["Loss_total"], epoch)
    writer.add_scalar("val/Loss_center", metrics["Loss_center"], epoch)
    writer.add_scalar("val/Loss_box", metrics["Loss_box"], epoch)
    writer.add_scalar("val/Loss_class", metrics["Loss_class"], epoch)
    # Accuracies
    writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
    writer.add_scalar("val/center_acc", metrics["center_acc"], epoch)


def log_confusion_matrix(writer: SummaryWriter, confusion_matrix, class_names, epoch: int) -> None:
    """
    Log confusion matrix to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        confusion_matrix: Confusion matrix as numpy array
        class_names: List of class names
        epoch: Current epoch number
    """
    figure_confusionMatrix = vh.plotConfMatrix(confusion_matrix, class_names)
    writer.add_figure("val/confusion_matrix", figure_confusionMatrix, epoch)
    plt.close(figure_confusionMatrix)  # Close the figure to free memory


def log_predictions_vs_gt(writer: SummaryWriter, model, loader: DataLoader, 
                         cfg: RunConfig, epoch: int, images_to_visualize: int = 4) -> None:
    """
    Log predictions vs ground truth visualization to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        model: The model being trained
        loader: DataLoader for validation data
        cfg: RunConfig with configuration
        epoch: Current epoch number
        images_to_visualize: Number of images to visualize
    """
    pred_vs_gt_visualization = vh.visualize_pred_vs_gt(
        model, loader, cfg, images2visualize=images_to_visualize
    )
    writer.add_image("val/pred_vs_gt", pred_vs_gt_visualization, epoch)
