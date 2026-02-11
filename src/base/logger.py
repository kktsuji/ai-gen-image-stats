"""Base Logger Interface

This module defines the abstract base class for all loggers in the experiments.
All experiment-specific loggers should inherit from BaseLogger and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


class BaseLogger(ABC):
    """Abstract base class for all loggers.

    This class provides a common interface for loggers across different experiment types
    (GAN, Diffusion, Classifier). It ensures all subclasses implement the required methods
    for logging metrics, images, and other artifacts during training.

    All loggers should inherit from this class and implement:
    - log_metrics(): Log scalar metrics (loss, accuracy, etc.)
    - log_images(): Log images (generated samples, visualizations, etc.)

    The class provides default implementations for:
    - log_hyperparams(): Log hyperparameters/configuration (optional)
    - close(): Cleanup and finalize logging (optional)

    Example:
        >>> class MyClassifierLogger(BaseLogger):
        ...     def __init__(self, log_dir: str):
        ...         self.log_dir = Path(log_dir)
        ...         self.log_dir.mkdir(parents=True, exist_ok=True)
        ...
        ...     def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        ...         # Write metrics to file or tensorboard
        ...         with open(self.log_dir / 'metrics.txt', 'a') as f:
        ...             f.write(f"Step {step}: {metrics}\\n")
        ...
        ...     def log_images(self, images: torch.Tensor, tag: str, step: int) -> None:
        ...         # Save images to disk
        ...         save_image(images, self.log_dir / f"{tag}_{step}.png")
    """

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log scalar metrics.

        This method must be implemented by all subclasses. It should handle
        logging of scalar values like loss, accuracy, learning rate, etc.

        Args:
            metrics: Dictionary of metric names to values. Values can be:
                - float/int: Scalar metrics
                - torch.Tensor: Single-element tensors (will be converted to scalar)
            step: Current training step/iteration
            epoch: Current training epoch (optional)

        Example:
            >>> logger.log_metrics({'loss': 0.5, 'accuracy': 0.95}, step=100, epoch=1)
        """
        pass

    @abstractmethod
    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Log images for visualization.

        This method must be implemented by all subclasses. It should handle
        saving or displaying images for visual inspection during training.

        Args:
            images: Image tensor(s) to log. Can be:
                - Single tensor of shape (B, C, H, W) or (C, H, W)
                - List of tensors
            tag: Identifier for the image set (e.g., 'generated_samples', 'validation')
            step: Current training step/iteration
            epoch: Current training epoch (optional)
            **kwargs: Additional arguments for experiment-specific logging

        Example:
            >>> logger.log_images(generated_images, 'samples', step=100, epoch=1)
        """
        pass

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration.

        This method provides a default implementation that can be overridden
        by subclasses. It logs configuration settings for reproducibility.

        Args:
            hyperparams: Dictionary of hyperparameter names to values

        Example:
            >>> logger.log_hyperparams({'lr': 0.001, 'batch_size': 32})
        """
        # Default implementation does nothing
        # Subclasses can override to implement specific logging
        pass

    def log_text(
        self,
        text: str,
        tag: str,
        step: Optional[int] = None,
    ) -> None:
        """Log text information.

        This method provides a default implementation that can be overridden
        by subclasses. It logs arbitrary text data like model summaries,
        predictions, or status messages.

        Args:
            text: Text content to log
            tag: Identifier for the text entry
            step: Current training step/iteration (optional)

        Example:
            >>> logger.log_text('Model initialized successfully', 'status')
        """
        # Default implementation does nothing
        # Subclasses can override to implement specific logging
        pass

    def log_histogram(
        self,
        values: Union[torch.Tensor, List[float]],
        tag: str,
        step: int,
        bins: int = 100,
    ) -> None:
        """Log histogram of values.

        This method provides a default implementation that can be overridden
        by subclasses. Useful for visualizing weight distributions, gradients, etc.

        Args:
            values: Values to create histogram from
            tag: Identifier for the histogram
            step: Current training step/iteration
            bins: Number of histogram bins

        Example:
            >>> logger.log_histogram(model.fc.weight, 'fc_weights', step=100)
        """
        # Default implementation does nothing
        # Subclasses can override to implement specific logging
        pass

    def close(self) -> None:
        """Cleanup and finalize logging.

        This method provides a default implementation that can be overridden
        by subclasses. It should handle any necessary cleanup like closing
        file handles, flushing buffers, or finalizing remote logging sessions.

        Example:
            >>> logger.close()
        """
        # Default implementation does nothing
        # Subclasses can override to implement specific cleanup
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
        return False
