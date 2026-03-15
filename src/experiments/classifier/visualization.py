"""Classifier Visualization Helpers

Standalone functions for classifier-specific visualizations:
- Annotated prediction grids with true/predicted labels
- Confusion matrix heatmaps

These are pure functions that take data + output path, save to disk,
and optionally log to TensorBoard.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.utils.tensorboard import safe_log_figure

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


def save_annotated_predictions(
    images: torch.Tensor,
    labels: Optional[List[int]],
    predictions: Optional[List[int]],
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """Save images with label annotations.

    Args:
        images: Image tensor (B, C, H, W)
        labels: True labels for each image
        predictions: Predicted labels for each image
        save_path: Path to save annotated image
        class_names: Optional list of class name strings for label display
    """
    save_path = Path(save_path)
    class_names = class_names or []

    n_images = min(images.size(0), 16)  # Limit to 16 images
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Convert image to numpy and transpose to (H, W, C)
        img = images[idx].cpu().numpy()
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        elif img.shape[0] == 1:  # Grayscale
            img = img[0]
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.axis("off")

        # Add title with labels
        title_parts: List[str] = []
        if labels is not None:
            true_label = (
                class_names[labels[idx]]
                if labels[idx] < len(class_names)
                else str(labels[idx])
            )
            title_parts.append(f"True: {true_label}")
        if predictions is not None:
            pred_label = (
                class_names[predictions[idx]]
                if predictions[idx] < len(class_names)
                else str(predictions[idx])
            )
            title_parts.append(f"Pred: {pred_label}")
            # Color code correct/incorrect
            if labels is not None and labels[idx] == predictions[idx]:
                ax.set_title("\n".join(title_parts), color="green", fontsize=8)
            else:
                ax.set_title("\n".join(title_parts), color="red", fontsize=8)
        elif title_parts:
            ax.set_title("\n".join(title_parts), fontsize=8)

    # Hide extra subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    confusion_matrix: Union[np.ndarray, torch.Tensor],
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    step: int = 0,
    epoch: Optional[int] = None,
    normalize: bool = False,
    tb_writer: Optional[Any] = None,
    tb_log_images: bool = True,
) -> None:
    """Save confusion matrix visualization.

    Args:
        confusion_matrix: Confusion matrix array, shape (n_classes, n_classes)
        save_path: Path to save the visualization
        class_names: Names of classes for labels
        step: Current training step (for TensorBoard)
        epoch: Current training epoch (optional, for title)
        normalize: Whether to normalize confusion matrix rows
        tb_writer: Optional TensorBoard SummaryWriter
        tb_log_images: Whether to log images to TensorBoard
    """
    save_path = Path(save_path)

    # Convert to numpy if needed
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    # Use provided class names or generate indices
    labels = class_names or [str(i) for i in range(len(confusion_matrix))]

    # Normalize if requested
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        confusion_matrix = confusion_matrix / row_sums

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Normalized Count" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    title = "Confusion Matrix"
    if normalize:
        title += " (Normalized)"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    ax.set_title(title, fontsize=14)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Save to TensorBoard
    if tb_writer is not None and tb_log_images:
        safe_log_figure(tb_writer, "confusion_matrix/matrix", fig, step, close=False)

    plt.close(fig)
