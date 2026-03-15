"""Experiment Logger

Unified logger for all experiment types. Handles logging of scalar metrics,
image grids, and hyperparameters to CSV files and TensorBoard.

Experiment-specific visualizations (annotated predictions, confusion matrices,
denoising process) live in their respective experiment visualization modules.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from src.utils.metrics_writer import MetricsWriter
from src.utils.tensorboard import (
    safe_log_hparams,
    safe_log_images,
)

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


class ExperimentLogger:
    """Unified logger for all experiment types.

    Handles logging of:
    - Scalar metrics (loss, accuracy, etc.) to CSV and TensorBoard
    - Image grids to PNG files and TensorBoard
    - Hyperparameters to TensorBoard

    The logger creates a structured directory for outputs:
    - metrics/metrics.csv: Training/validation metrics over time
    - Configurable subdirectories for images and other outputs

    Example:
        >>> logger = ExperimentLogger(
        ...     log_dir="outputs/logs/experiment_001",
        ...     subdirs={"images": "samples", "denoising": "denoising"},
        ... )
        >>> logger.log_metrics({'loss': 0.5, 'accuracy': 0.95}, step=100, epoch=1)
        >>> logger.log_images(samples, 'samples', step=100, save_dir=logger.dirs["images"])
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        subdirs: Optional[Dict[str, str]] = None,
        tensorboard_config: Optional[Dict[str, Any]] = None,
        tb_log_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the experiment logger.

        Args:
            log_dir: Directory to save logs and visualizations
            subdirs: Mapping of logical name → subdirectory name to create.
                Created under log_dir and accessible via self.dirs[name].
            tensorboard_config: TensorBoard configuration dict with keys:
                - enabled (bool): Enable TensorBoard logging
                - flush_secs (int): Flush frequency in seconds
                - log_images (bool): Log images to TensorBoard
                - log_histograms (bool): Log weight/gradient histograms
                - log_graph (bool): Log model computational graph
            tb_log_dir: Directory for TensorBoard event logs. Resolved via
                output.subdirs.tensorboard. Defaults to {log_dir}/../tensorboard.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.metrics_dir = self.log_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        self.dirs: Dict[str, Path] = {}
        if subdirs:
            for name, dirname in subdirs.items():
                dir_path = self.log_dir / dirname
                dir_path.mkdir(exist_ok=True)
                self.dirs[name] = dir_path

        # Initialize metrics writer (CSV + TensorBoard)
        self.tensorboard_config = tensorboard_config or {}
        self.tb_log_images = self.tensorboard_config.get("log_images", True)
        self.tb_log_histograms = self.tensorboard_config.get("log_histograms", False)

        if tb_log_dir is None:
            tb_log_dir = self.log_dir.parent / "tensorboard"

        self.metrics_writer = MetricsWriter(
            metrics_file=self.metrics_dir / "metrics.csv",
            tensorboard_config=self.tensorboard_config,
            tb_log_dir=tb_log_dir,
        )

        # Track logged data for testing
        self.logged_metrics_history: List[Dict[str, Any]] = []
        self.logged_images: List[Dict[str, Any]] = []

    def log_metrics(
        self,
        metrics: Mapping[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log scalar metrics to CSV file and TensorBoard.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step/iteration
            epoch: Current training epoch (optional)
        """
        # Convert tensor values to scalars
        processed_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value

        # Add step and epoch
        log_entry: Dict[str, Any] = {"step": step}
        if epoch is not None:
            log_entry["epoch"] = epoch
        log_entry.update(processed_metrics)

        # Store for testing
        self.logged_metrics_history.append(log_entry)

        # Write to CSV and TensorBoard
        self.metrics_writer.write_metrics(log_entry)
        self.metrics_writer.log_scalars(processed_metrics, step)

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """Log images as a grid to PNG file and TensorBoard.

        Args:
            images: Image tensor(s) to log. Shape (B, C, H, W) or (C, H, W)
            tag: Identifier for the image set
            step: Current training step
            epoch: Current training epoch (optional)
            save_dir: Directory to save images. Defaults to the first subdir
                or log_dir if no subdirs configured.
            **kwargs: Additional arguments:
                - normalize: Whether to normalize images (default: True)
                - nrow: Number of images per row in grid (default: 8)
                - value_range: Tuple of (min, max) for normalization
                - class_labels: List[int] of per-image class labels to annotate
                    on the saved grid. When provided, each cell in the grid gets
                    a small text overlay showing its class index.
        """
        if isinstance(images, list):
            images = torch.stack(images)

        # Ensure 4D tensor (B, C, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Extract kwargs
        normalize = kwargs.get("normalize", True)
        nrow = kwargs.get("nrow", 8)
        value_range = kwargs.get("value_range", None)
        class_labels: Optional[List[int]] = kwargs.get("class_labels", None)

        # Store for testing
        self.logged_images.append(
            {
                "images": images.clone(),
                "tag": tag,
                "step": step,
                "epoch": epoch,
                "class_labels": class_labels,
            }
        )

        # Create filename
        filename_parts = [tag, f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        filename = "_".join(filename_parts) + ".png"

        # Determine save directory
        if save_dir is not None:
            target_dir = Path(save_dir)
        elif self.dirs:
            target_dir = next(iter(self.dirs.values()))
        else:
            target_dir = self.log_dir

        image_path = target_dir / filename

        if class_labels is not None:
            # Render an annotated grid with per-cell class labels using matplotlib
            self._save_annotated_grid(
                images, image_path, class_labels, nrow, normalize, value_range
            )
        else:
            # Save plain image grid
            save_kwargs: Dict[str, Any] = {"normalize": normalize, "nrow": nrow}
            if value_range is not None:
                save_kwargs["value_range"] = value_range
            save_image(images, image_path, **save_kwargs)

        # Save to TensorBoard
        # NOTE: safe_log_images is safe to call even when TensorBoard is not
        # installed — it guards on writer being None and wraps all calls in
        # try/except (see src/utils/tensorboard.py).
        if self.tb_writer is not None and self.tb_log_images:
            # If value_range indicates [-1, 1] data, normalize for TensorBoard
            if value_range == (-1, 1) or value_range == [-1, 1]:
                tb_images = (images + 1.0) / 2.0
                tb_images = torch.clamp(tb_images, 0, 1)
            else:
                tb_images = images
            safe_log_images(self.tb_writer, f"images/{tag}", tb_images, step)

    def _save_annotated_grid(
        self,
        images: torch.Tensor,
        save_path: Path,
        class_labels: List[int],
        nrow: int,
        normalize: bool,
        value_range: Any,
    ) -> None:
        """Save an image grid annotated with per-cell class labels.

        Uses matplotlib to overlay a small class index on each image cell.

        Args:
            images: Image tensor (B, C, H, W)
            save_path: Path to save the annotated image
            class_labels: Per-image class labels (length must match batch size)
            nrow: Number of images per row in the grid
            normalize: Whether to normalize images for display
            value_range: Tuple of (min, max) for normalization
        """
        # Build the grid tensor via torchvision
        grid_kwargs: Dict[str, Any] = {"normalize": normalize, "nrow": nrow}
        if value_range is not None:
            grid_kwargs["value_range"] = value_range
        grid = make_grid(images, **grid_kwargs)

        # Convert to numpy HWC for matplotlib
        grid_np = grid.cpu().numpy()
        if grid_np.shape[0] in (1, 3):
            grid_np = np.transpose(grid_np, (1, 2, 0))
        grid_np = np.clip(grid_np, 0, 1)

        n_images = images.size(0)
        _, _, h, w = images.shape
        # make_grid adds 2px padding by default
        pad = 2

        fig, ax = plt.subplots(
            figsize=(nrow * 1.5, ((n_images + nrow - 1) // nrow) * 1.5)
        )
        ax.imshow(grid_np)
        ax.axis("off")

        for idx, label in enumerate(class_labels[:n_images]):
            row = idx // nrow
            col = idx % nrow
            x = col * (w + pad) + pad + 2
            y = row * (h + pad) + pad + 10
            ax.text(
                x,
                y,
                str(label),
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard.

        Args:
            hyperparams: Dictionary of hyperparameter names to values
        """
        if self.tb_writer is not None:
            safe_log_hparams(self.tb_writer, hyperparams)

    @property
    def tb_writer(self):
        """Access the TensorBoard writer for logger-specific image/figure logging."""
        return self.metrics_writer.tb_writer

    def close(self) -> None:
        """Cleanup and finalize logging."""
        self.metrics_writer.close()
        # Flush any remaining matplotlib figures
        plt.close("all")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
        return False
