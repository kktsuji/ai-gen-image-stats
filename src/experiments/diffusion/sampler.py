"""Diffusion Sampler

This module implements a standalone sampler for generating images from trained
diffusion models. It separates sampling/inference logic from training,
enabling efficient inference without requiring training dependencies.
"""

from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from src.base.model import BaseModel
from src.experiments.diffusion.model import EMA


class DiffusionSampler:
    """Sampler for generating images from trained diffusion models.

    This class provides a clean interface for generating samples from trained
    diffusion models. It can be used independently for inference without
    requiring optimizer, dataloader, or other training infrastructure.

    Key Features:
    - Unconditional and conditional generation
    - EMA weight support for better quality
    - Classifier-free guidance
    - Progress bar for long sampling runs
    - Batch and per-class generation utilities

    Args:
        model: Trained diffusion model (DDPM or similar)
        device: Device to run sampling on ('cpu' or 'cuda')
        ema: Optional EMA instance for enhanced sample quality

    Example:
        >>> # Standalone inference usage
        >>> from src.experiments.diffusion.model import create_ddpm
        >>> from src.experiments.diffusion.sampler import DiffusionSampler
        >>>
        >>> # Load model from checkpoint
        >>> model = create_ddpm(image_size=64, num_classes=2, device="cuda")
        >>> checkpoint = torch.load("checkpoint.pth")
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>>
        >>> # Create sampler
        >>> sampler = DiffusionSampler(model=model, device="cuda")
        >>>
        >>> # Generate unconditional samples
        >>> samples = sampler.sample(num_samples=16)
        >>>
        >>> # Generate conditional samples with guidance
        >>> labels = torch.tensor([0, 1] * 8, device="cuda")
        >>> samples = sampler.sample(
        ...     num_samples=16,
        ...     class_labels=labels,
        ...     guidance_scale=3.0
        ... )
        >>>
        >>> # Generate samples for all classes
        >>> samples, labels = sampler.sample_by_class(
        ...     samples_per_class=4,
        ...     num_classes=2,
        ...     guidance_scale=3.0
        ... )
    """

    def __init__(
        self,
        model: BaseModel,
        device: str = "cpu",
        ema: Optional[EMA] = None,
    ):
        """Initialize the diffusion sampler.

        Args:
            model: Trained diffusion model (must implement sample() method)
            device: Device to perform sampling on ('cpu' or 'cuda')
            ema: Optional EMA instance for better sample quality
        """
        self.model = model
        self.device = device
        self.ema = ema

        # Move model to device
        self.model.to(self.device)

    def sample(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_ema: bool = True,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Generate samples from the diffusion model.

        This is the main sampling interface. Supports both unconditional
        and conditional generation with optional classifier-free guidance.

        Args:
            num_samples: Number of samples to generate
            class_labels: Class labels for conditional generation
                Shape: (num_samples,), values in [0, num_classes-1]
                None for unconditional generation
            guidance_scale: Classifier-free guidance scale
                0.0 disables guidance (standard sampling)
                > 1.0 increases adherence to class conditioning
                Typical values: 1.0-5.0
            use_ema: Whether to use EMA weights for generation
                Recommended: True for better sample quality
            show_progress: Whether to display progress bar during sampling
                Useful for large batches or slow sampling

        Returns:
            Generated samples tensor
            Shape: (num_samples, channels, height, width)
            Values: Range [-1, 1] (typical normalization for diffusion models)

        Raises:
            ValueError: If class_labels shape doesn't match num_samples
            RuntimeError: If model's sample() method fails

        Example:
            >>> # Unconditional generation
            >>> samples = sampler.sample(num_samples=64)
            >>>
            >>> # Conditional generation with guidance
            >>> labels = torch.randint(0, 10, (128,), device="cuda")
            >>> samples = sampler.sample(
            ...     num_samples=128,
            ...     class_labels=labels,
            ...     guidance_scale=3.0,
            ...     show_progress=True
            ... )
        """
        # Validate inputs
        if class_labels is not None:
            if len(class_labels) != num_samples:
                raise ValueError(
                    f"class_labels length ({len(class_labels)}) must match "
                    f"num_samples ({num_samples})"
                )

        # Set model to evaluation mode
        self.model.eval()

        # Apply EMA weights if requested
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        try:
            with torch.no_grad():
                # Move class labels to device if provided
                if class_labels is not None:
                    class_labels = class_labels.to(self.device)

                # Generate samples
                # Note: Progress bar for sampling steps is handled by model's sample()
                # This show_progress flag is for potential batch-wise sampling
                samples = self.model.sample(
                    batch_size=num_samples,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                )

            return samples

        finally:
            # Always restore original weights if EMA was used
            if use_ema and self.ema is not None:
                self.ema.restore()

    def sample_by_class(
        self,
        samples_per_class: int,
        num_classes: int,
        guidance_scale: float = 0.0,
        use_ema: bool = True,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Generate samples for each class.

        Convenience method for generating balanced samples across all classes.
        Useful for visual quality assessment and dataset augmentation.

        Args:
            samples_per_class: Number of samples to generate per class
            num_classes: Total number of classes to generate for
            guidance_scale: Classifier-free guidance scale (0.0 to disable)
            use_ema: Whether to use EMA weights
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (samples, class_labels):
            - samples: Generated sample tensor
                Shape: (num_classes * samples_per_class, C, H, W)
            - class_labels: List of corresponding class indices
                Length: num_classes * samples_per_class
                Format: [0, 0, ..., 1, 1, ..., num_classes-1, num_classes-1, ...]

        Example:
            >>> # Generate 4 samples for each of 10 classes
            >>> samples, labels = sampler.sample_by_class(
            ...     samples_per_class=4,
            ...     num_classes=10,
            ...     guidance_scale=3.0,
            ...     use_ema=True
            ... )
            >>> print(samples.shape)  # (40, 3, 64, 64)
            >>> print(len(labels))     # 40
            >>> print(labels[:8])      # [0, 0, 0, 0, 1, 1, 1, 1]
        """
        samples_list = []
        class_labels_list = []

        # Create iterator with optional progress bar
        class_range = range(num_classes)
        if show_progress:
            class_range = tqdm(class_range, desc="Generating samples by class")

        for class_idx in class_range:
            # Create class labels for this batch
            class_labels = torch.full(
                (samples_per_class,),
                class_idx,
                dtype=torch.long,
                device=self.device,
            )

            # Generate samples for this class
            samples = self.sample(
                num_samples=samples_per_class,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                use_ema=use_ema,
                show_progress=False,  # Don't show nested progress bars
            )

            samples_list.append(samples)
            class_labels_list.extend([class_idx] * samples_per_class)

        # Concatenate all samples
        all_samples = torch.cat(samples_list, dim=0)

        return all_samples, class_labels_list
