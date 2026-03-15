"""Diffusion Visualization Helpers

Standalone function for diffusion-specific visualization:
- Denoising process visualization (progressive denoising from noise to sample)

This is a pure function that takes data + output path, saves to disk,
and optionally logs to TensorBoard.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.tensorboard import safe_log_figure, safe_log_images

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


def save_denoising_process(
    denoising_sequence: Union[torch.Tensor, List[torch.Tensor]],
    save_path: Union[str, Path],
    num_steps_to_show: int = 8,
    step: int = 0,
    tb_writer: Optional[Any] = None,
    tb_log_images: bool = True,
) -> None:
    """Save a visualization of the denoising process.

    Shows the progressive denoising of an image from pure noise to final sample.

    Args:
        denoising_sequence: Sequence of images showing denoising steps.
            Either a list of tensors or a single tensor with shape (T, C, H, W)
            where T is the number of timesteps.
        save_path: Path to save the visualization
        num_steps_to_show: Number of denoising steps to show in visualization
        step: Current training step (for TensorBoard)
        tb_writer: Optional TensorBoard SummaryWriter
        tb_log_images: Whether to log images to TensorBoard
    """
    save_path = Path(save_path)

    # Convert list to tensor if needed
    if isinstance(denoising_sequence, list):
        denoising_sequence = torch.stack(denoising_sequence)

    # Ensure 4D tensor (T, C, H, W)
    if denoising_sequence.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (T, C, H, W), got {denoising_sequence.shape}"
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
        # Diffusion models output in [-1, 1]; remap to [0, 1] for display
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            img = np.clip((img + 1.0) / 2.0, 0, 1)
        elif img.shape[0] == 1:  # Grayscale
            img = img[0]
            img = np.clip((img + 1.0) / 2.0, 0, 1)
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.axis("off")
        ax.set_title(f"t={img_idx}", fontsize=8)

    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Save to TensorBoard
    if tb_writer is not None and tb_log_images:
        # Normalize [-1, 1] → [0, 1] for TensorBoard
        tb_images = (selected_images + 1.0) / 2.0
        tb_images = torch.clamp(tb_images, 0, 1)
        safe_log_images(tb_writer, "denoising/process", tb_images, step)
        safe_log_figure(tb_writer, "denoising/figure", fig, step, close=False)

    plt.close(fig)
