"""Diffusion Logger

This module implements a logger specifically for diffusion model experiments.
It provides functionality for logging diffusion-specific metrics, generated samples,
and denoising process visualizations.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from src.base.logger import BaseLogger
from src.utils.tensorboard import (
    close_tensorboard_writer,
    create_tensorboard_writer,
    safe_log_figure,
    safe_log_hparams,
    safe_log_images,
    safe_log_scalar,
)

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


class DiffusionLogger(BaseLogger):
    """Logger for diffusion model experiments.

    This logger handles logging of diffusion-specific information including:
    - Scalar metrics (loss, timestep statistics, etc.)
    - Generated samples at different stages of training
    - Denoising process visualization (progressive denoising)
    - Training progress to CSV files

    The logger creates a structured directory for outputs:
    - metrics.csv: Training/validation metrics over time
    - samples/: Generated sample images during training
    - denoising/: Denoising process visualizations
    - quality/: Sample quality comparisons over time

    Example:
        >>> logger = DiffusionLogger(log_dir="outputs/logs/diffusion_001")
        >>> logger.log_metrics({'loss': 0.05, 'avg_timestep': 500}, step=1000, epoch=10)
        >>> logger.log_images(generated_samples, 'samples', step=1000)
        >>> logger.log_denoising_process(denoising_sequence, step=1000)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        tensorboard_config: Optional[Dict[str, Any]] = None,
        tb_log_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the diffusion logger.

        Args:
            log_dir: Directory to save logs and visualizations
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
        self.samples_dir = self.log_dir / "samples"
        self.denoising_dir = self.log_dir / "denoising"
        self.quality_dir = self.log_dir / "quality"

        self.metrics_dir.mkdir(exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        self.denoising_dir.mkdir(exist_ok=True)
        self.quality_dir.mkdir(exist_ok=True)

        # Initialize metrics CSV file
        self.metrics_file = self.log_dir / "metrics.csv"
        self.csv_initialized = self.metrics_file.exists()
        self.csv_fieldnames = None

        # If CSV exists, load existing fieldnames
        if self.csv_initialized:
            with open(self.metrics_file, "r") as f:
                reader = csv.DictReader(f)
                self.csv_fieldnames = reader.fieldnames

        # Initialize TensorBoard logging
        self.tensorboard_config = tensorboard_config or {}
        self.tb_enabled = self.tensorboard_config.get("enabled", False)
        self.tb_log_images = self.tensorboard_config.get("log_images", True)
        self.tb_log_histograms = self.tensorboard_config.get("log_histograms", False)

        if tb_log_dir is None:
            tb_log_dir = self.log_dir.parent / "tensorboard"
        flush_secs = self.tensorboard_config.get("flush_secs", 30)

        self.tb_writer = create_tensorboard_writer(
            log_dir=tb_log_dir,
            flush_secs=flush_secs,
            enabled=self.tb_enabled,
        )

        # Track logged data for testing
        self.logged_metrics_history = []
        self.logged_images = []
        self.logged_denoising_sequences = []

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

        # Write to TensorBoard
        if self.tb_writer is not None:
            for key, value in processed_metrics.items():
                safe_log_scalar(self.tb_writer, f"metrics/{key}", value, step)

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
                - value_range: Tuple of (min, max) for normalization
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

        # Store for testing
        self.logged_images.append(
            {
                "images": images.clone(),
                "tag": tag,
                "step": step,
                "epoch": epoch,
            }
        )

        # Create filename
        filename_parts = [tag, f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        filename = "_".join(filename_parts) + ".png"

        # Save image grid
        image_path = self.samples_dir / filename
        save_image(
            images,
            image_path,
            normalize=normalize,
            nrow=nrow,
            value_range=value_range,
        )

        # Save to TensorBoard
        if self.tb_writer is not None and self.tb_log_images:
            safe_log_images(self.tb_writer, f"images/{tag}", images, step)

    def log_denoising_process(
        self,
        denoising_sequence: Union[torch.Tensor, List[torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
        num_steps_to_show: int = 8,
    ) -> None:
        """Log a visualization of the denoising process.

        Shows the progressive denoising of an image from pure noise to final sample.

        Args:
            denoising_sequence: Sequence of images showing denoising steps.
                Either a list of tensors or a single tensor with shape (T, C, H, W)
                where T is the number of timesteps.
            step: Current training step
            epoch: Current training epoch (optional)
            num_steps_to_show: Number of denoising steps to show in visualization
        """
        # Convert list to tensor if needed
        if isinstance(denoising_sequence, list):
            denoising_sequence = torch.stack(denoising_sequence)

        # Ensure 4D tensor (T, C, H, W)
        if denoising_sequence.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor (T, C, H, W), got {denoising_sequence.shape}"
            )

        # Store for testing
        self.logged_denoising_sequences.append(
            {
                "sequence": denoising_sequence.clone(),
                "step": step,
                "epoch": epoch,
            }
        )

        # Select evenly spaced steps to show
        total_timesteps = denoising_sequence.size(0)
        if total_timesteps <= num_steps_to_show:
            indices = list(range(total_timesteps))
        else:
            indices = np.linspace(0, total_timesteps - 1, num_steps_to_show, dtype=int)

        selected_images = denoising_sequence[indices]

        # Create visualization
        n_images = len(indices)
        fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
        if n_images == 1:
            axes = [axes]

        for idx, (ax, img_idx) in enumerate(zip(axes, indices)):
            img = selected_images[idx].cpu().numpy()

            # Convert to displayable format
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0, 1)
            elif img.shape[0] == 1:  # Grayscale
                img = img[0]
                img = np.clip(img, 0, 1)
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

            ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            ax.axis("off")
            ax.set_title(f"t={img_idx}", fontsize=8)

        plt.tight_layout()

        # Save figure
        filename_parts = ["denoising", f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        filename = "_".join(filename_parts) + ".png"

        save_path = self.denoising_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Save to TensorBoard
        if self.tb_writer is not None and self.tb_log_images:
            safe_log_images(self.tb_writer, "denoising/process", selected_images, step)
            safe_log_figure(self.tb_writer, "denoising/figure", fig, step, close=False)

        plt.close(fig)

    def log_sample_comparison(
        self,
        images: torch.Tensor,
        tag: str,
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log sample images for quality comparison over time.

        Useful for tracking how sample quality improves during training.

        Args:
            images: Image tensor to log, shape (B, C, H, W)
            tag: Identifier for the comparison set
            step: Current training step
            epoch: Current training epoch (optional)
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Create filename
        filename_parts = [tag, f"step{step}"]
        if epoch is not None:
            filename_parts.insert(1, f"epoch{epoch}")
        filename = "_".join(filename_parts) + ".png"

        # Save to quality directory
        image_path = self.quality_dir / filename
        save_image(images, image_path, normalize=True, nrow=min(images.size(0), 8))

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to a YAML file and TensorBoard.

        Args:
            hyperparams: Dictionary of hyperparameter names to values
        """
        import yaml

        hyperparams_file = self.log_dir / "hyperparams.yaml"
        with open(hyperparams_file, "w") as f:
            yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=False)

        # Log to TensorBoard
        if self.tb_writer is not None:
            safe_log_hparams(self.tb_writer, hyperparams)

    def close(self) -> None:
        """Cleanup and finalize logging."""
        close_tensorboard_writer(self.tb_writer)
        self.tb_writer = None
        # Flush any remaining matplotlib figures
        plt.close("all")
