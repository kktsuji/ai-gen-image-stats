"""Diffusion Sampler

This module implements a standalone sampler for generating images from trained
diffusion models. It separates sampling/inference logic from training,
enabling efficient inference without requiring training dependencies.
"""

import logging
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from src.base.model import BaseModel
from src.experiments.diffusion.model import EMA

# Module-level logger
logger = logging.getLogger(__name__)


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

        logger.debug(f"DiffusionSampler initialized on device: {device}")
        if ema is not None:
            logger.debug("EMA weights available for sampling")

    def sample(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_ema: bool = True,
        show_progress: bool = False,
        progress_desc: str = "Denoising",
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
            show_progress: Whether to show tqdm progress bar for denoising steps
            progress_desc: Description label for the progress bar

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
            ... )
        """
        # Validate inputs
        if class_labels is not None:
            if len(class_labels) != num_samples:
                logger.error(
                    f"class_labels length ({len(class_labels)}) doesn't match "
                    f"num_samples ({num_samples})"
                )
                raise ValueError(
                    f"class_labels length ({len(class_labels)}) must match "
                    f"num_samples ({num_samples})"
                )

        logger.debug(f"Starting sample generation: {num_samples} samples")
        if class_labels is not None:
            unique_classes = torch.unique(class_labels).tolist()
            logger.debug(f"Conditional generation for classes: {unique_classes}")
            if guidance_scale > 0:
                logger.debug(f"Using classifier-free guidance (scale={guidance_scale})")
        else:
            logger.debug("Unconditional generation")

        if use_ema and self.ema is not None:
            logger.debug("Using EMA weights for sampling")

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
                samples = self.model.sample(
                    batch_size=num_samples,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    show_progress=show_progress,
                    progress_desc=progress_desc,
                )

            logger.debug(f"Sample generation completed: {samples.shape}")
            return samples

        finally:
            # Always restore original weights if EMA was used
            if use_ema and self.ema is not None:
                self.ema.restore()

    def sample_with_intermediates(
        self,
        num_samples: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_ema: bool = True,
        num_steps_to_capture: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and return intermediate denoising steps.

        Uses the model's ``return_intermediates`` capability to capture the
        full denoising trajectory, then selects evenly-spaced frames from
        sample index 0 to keep memory bounded.

        Args:
            num_samples: Number of samples to generate
            class_labels: Class labels for conditional generation
            guidance_scale: Classifier-free guidance scale
            use_ema: Whether to use EMA weights for generation
            num_steps_to_capture: Number of evenly-spaced intermediate frames
                to keep from the denoising trajectory of sample 0

        Returns:
            Tuple of:
            - samples: Final generated images, shape (N, C, H, W)
            - denoising_sequence: Intermediate steps for one sample,
              shape (num_steps_to_capture, C, H, W)
        """
        logger.info(
            f"Starting sample generation with intermediates: {num_samples} samples"
        )

        # Set model to evaluation mode
        self.model.eval()

        # Apply EMA weights if requested
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        try:
            with torch.no_grad():
                if class_labels is not None:
                    class_labels = class_labels.to(self.device)

                # Generate samples with full intermediate trajectory
                all_steps = self.model.sample(
                    batch_size=num_samples,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    return_intermediates=True,
                )
                # all_steps shape: (T+1, N, C, H, W)
                # where T+1 includes the initial noise + each denoising step

                # Final samples are the last timestep
                samples = all_steps[-1]  # (N, C, H, W)

                # Extract denoising trajectory for sample index 0
                trajectory = all_steps[:, 0]  # (T+1, C, H, W)

                # Select evenly-spaced frames
                total_steps = trajectory.shape[0]
                if total_steps <= num_steps_to_capture:
                    denoising_sequence = trajectory
                else:
                    import numpy as np

                    indices = np.linspace(
                        0, total_steps - 1, num_steps_to_capture, dtype=int
                    )
                    denoising_sequence = trajectory[
                        indices
                    ]  # (num_steps_to_capture, C, H, W)

            logger.info(
                f"Sample generation with intermediates completed: "
                f"samples={samples.shape}, denoising_sequence={denoising_sequence.shape}"
            )
            return samples, denoising_sequence

        finally:
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

        logger.info(
            f"Generating {samples_per_class} samples per class for {num_classes} classes "
            f"(total: {samples_per_class * num_classes} samples)"
        )

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
            )

            samples_list.append(samples)
            class_labels_list.extend([class_idx] * samples_per_class)

        # Concatenate all samples
        all_samples = torch.cat(samples_list, dim=0)

        logger.info(
            f"Class-based sample generation completed: {all_samples.shape}, "
            f"{len(class_labels_list)} labels"
        )

        return all_samples, class_labels_list
