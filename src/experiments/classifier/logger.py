"""Classifier Logger

This module implements a logger specifically for classification experiments.
It provides functionality for logging classification metrics, confusion matrices,
and sample predictions with visualizations.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision.utils import make_grid, save_image

from src.base.logger import BaseLogger

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


class ClassifierLogger(BaseLogger):
    """Logger for classification experiments.

    This logger handles logging of classification-specific metrics including:
    - Scalar metrics (loss, accuracy, precision, recall, F1, etc.)
    - Confusion matrices with visualizations
    - Sample predictions with images and labels
    - Training progress to CSV files

    The logger creates a structured directory for outputs:
    - metrics.csv: Training/validation metrics over time
    - confusion_matrices/: Confusion matrix visualizations
    - predictions/: Sample prediction visualizations

    Example:
        >>> logger = ClassifierLogger(log_dir="outputs/logs/experiment_001")
        >>> logger.log_metrics({'loss': 0.5, 'accuracy': 0.95}, step=100, epoch=1)
        >>> confusion_mat = compute_confusion_matrix(...)
        >>> logger.log_confusion_matrix(confusion_mat, class_names=['Real', 'Fake'], step=100)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the classifier logger.

        Args:
            log_dir: Directory to save logs and visualizations
            class_names: Names of classes for confusion matrix labels
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or []

        # Create subdirectories
        self.metrics_dir = self.log_dir / "metrics"
        self.confusion_dir = self.log_dir / "confusion_matrices"
        self.predictions_dir = self.log_dir / "predictions"

        self.metrics_dir.mkdir(exist_ok=True)
        self.confusion_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)

        # Initialize metrics CSV file
        self.metrics_file = self.log_dir / "metrics.csv"
        self.csv_initialized = self.metrics_file.exists()
        self.csv_fieldnames = None

        # If CSV exists, load existing fieldnames
        if self.csv_initialized:
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                self.csv_fieldnames = reader.fieldnames

        # Track logged data for testing
        self.logged_metrics_history = []
        self.logged_confusion_matrices = []

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log scalar metrics to CSV file.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step/iteration
            epoch: Current training epoch (optional)
        """
        # Convert tensor values to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value

        # Add step and epoch
        log_entry = {"step": step}
        if epoch is not None:
            log_entry["epoch"] = epoch
        log_entry.update(processed_metrics)

        # Store for testing
        self.logged_metrics_history.append(log_entry)

        # Write to CSV
        self._write_metrics_to_csv(log_entry)

    def _write_metrics_to_csv(self, log_entry: Dict[str, Any]) -> None:
        """Write a single metrics entry to CSV file.

        Args:
            log_entry: Dictionary of metrics to write
        """
        # Initialize CSV on first write
        if not self.csv_initialized:
            self.csv_fieldnames = list(log_entry.keys())
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()
            self.csv_initialized = True
        else:
            # Update fieldnames if new metrics are added
            new_fields = set(log_entry.keys()) - set(self.csv_fieldnames)
            if new_fields:
                self.csv_fieldnames.extend(sorted(new_fields))
                # Re-write entire CSV with new headers
                self._rewrite_csv_with_new_fields()

        # Append metrics
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            # Fill missing fields with None
            row = {field: log_entry.get(field) for field in self.csv_fieldnames}
            writer.writerow(row)

    def _rewrite_csv_with_new_fields(self) -> None:
        """Re-write CSV file with updated field names."""
        # Read existing data
        existing_data = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # Write with new fieldnames
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            for row in existing_data:
                # Fill missing fields with None
                updated_row = {field: row.get(field) for field in self.csv_fieldnames}
                writer.writerow(updated_row)

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Log images for visualization.

        Args:
            images: Image tensor(s) to log. Shape (B, C, H, W) or (C, H, W)
            tag: Identifier for the image set
            step: Current training step
            epoch: Current training epoch (optional)
            **kwargs: Additional arguments:
                - normalize: Whether to normalize images (default: True)
                - nrow: Number of images per row in grid (default: 8)
                - labels: List of labels for each image (optional)
                - predictions: List of predicted labels (optional)
        """
        if isinstance(images, list):
            images = torch.stack(images)

        # Ensure 4D tensor (B, C, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Extract kwargs
        normalize = kwargs.get("normalize", True)
        nrow = kwargs.get("nrow", 8)
        labels = kwargs.get("labels", None)
        predictions = kwargs.get("predictions", None)

        # Create filename
        filename_parts = [tag, f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        filename = "_".join(filename_parts) + ".png"

        # Save image grid
        image_path = self.predictions_dir / filename
        save_image(images, image_path, normalize=normalize, nrow=nrow)

        # If labels/predictions provided, create annotated visualization
        if labels is not None or predictions is not None:
            self._save_annotated_predictions(
                images, labels, predictions, image_path.with_suffix(".annotated.png")
            )

    def _save_annotated_predictions(
        self,
        images: torch.Tensor,
        labels: Optional[List[int]],
        predictions: Optional[List[int]],
        save_path: Path,
    ) -> None:
        """Save images with label annotations.

        Args:
            images: Image tensor (B, C, H, W)
            labels: True labels for each image
            predictions: Predicted labels for each image
            save_path: Path to save annotated image
        """
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

            ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            ax.axis("off")

            # Add title with labels
            title_parts = []
            if labels is not None:
                true_label = (
                    self.class_names[labels[idx]]
                    if labels[idx] < len(self.class_names)
                    else str(labels[idx])
                )
                title_parts.append(f"True: {true_label}")
            if predictions is not None:
                pred_label = (
                    self.class_names[predictions[idx]]
                    if predictions[idx] < len(self.class_names)
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

    def log_confusion_matrix(
        self,
        confusion_matrix: Union[np.ndarray, torch.Tensor],
        class_names: Optional[List[str]] = None,
        step: int = 0,
        epoch: Optional[int] = None,
        normalize: bool = False,
    ) -> None:
        """Log confusion matrix with visualization.

        Args:
            confusion_matrix: Confusion matrix array, shape (n_classes, n_classes)
            class_names: Names of classes for labels (overrides constructor value)
            step: Current training step
            epoch: Current training epoch (optional)
            normalize: Whether to normalize confusion matrix rows
        """
        # Convert to numpy if needed
        if isinstance(confusion_matrix, torch.Tensor):
            confusion_matrix = confusion_matrix.cpu().numpy()

        # Store for testing
        self.logged_confusion_matrices.append(
            {
                "matrix": confusion_matrix.copy(),
                "step": step,
                "epoch": epoch,
                "normalize": normalize,
            }
        )

        # Use provided class names or fall back to instance class names
        labels = class_names or self.class_names
        if not labels:
            labels = [str(i) for i in range(len(confusion_matrix))]

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
        filename_parts = ["confusion_matrix", f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        if normalize:
            filename_parts.append("normalized")
        filename = "_".join(filename_parts) + ".png"

        save_path = self.confusion_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to a JSON file.

        Args:
            hyperparams: Dictionary of hyperparameter names to values
        """
        import json

        hyperparams_file = self.log_dir / "hyperparams.json"
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams, f, indent=2)

    def close(self) -> None:
        """Cleanup and finalize logging."""
        # Flush any remaining matplotlib figures
        plt.close("all")
