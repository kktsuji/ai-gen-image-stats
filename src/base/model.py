"""Base Model Interface

This module defines the abstract base class for all models in the experiments.
All experiment-specific models should inherit from BaseModel and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models.

    This class provides a common interface for models across different experiment types
    (GAN, Diffusion, Classifier). It inherits from both nn.Module and ABC to ensure
    all subclasses implement the required methods.

    All models should inherit from this class and implement:
    - forward(): The forward pass of the model
    - compute_loss(): Calculate the loss for training

    The class provides default implementations for:
    - save_checkpoint(): Save model state to disk
    - load_checkpoint(): Load model state from disk

    Example:
        >>> class MyClassifier(BaseModel):
        ...     def __init__(self, num_classes: int):
        ...         super().__init__()
        ...         self.fc = nn.Linear(512, num_classes)
        ...
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return self.fc(x)
        ...
        ...     def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...         return F.cross_entropy(predictions, targets)
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        This method must be implemented by all subclasses. The signature will vary
        depending on the experiment type.

        Args:
            *args: Variable positional arguments specific to the model
            **kwargs: Variable keyword arguments specific to the model

        Returns:
            Model output(s). Type depends on the specific model implementation.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def compute_loss(
        self, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss for training.

        This method must be implemented by all subclasses. The implementation
        will vary depending on the experiment type and loss function requirements.

        Args:
            *args: Variable positional arguments (e.g., predictions, targets)
            **kwargs: Variable keyword arguments (e.g., auxiliary outputs, weights)

        Returns:
            Either a single loss tensor or a dictionary of loss tensors.
            If returning a dictionary, it should contain:
            - 'total': The total loss to optimize
            - Other keys: Individual loss components for logging

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> # Simple loss
            >>> return F.cross_entropy(predictions, targets)
            >>>
            >>> # Multiple loss components
            >>> return {
            ...     'total': cls_loss + 0.1 * reg_loss,
            ...     'classification': cls_loss,
            ...     'regularization': reg_loss
            ... }
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: Optional[int] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Save model checkpoint to disk.

        Saves the model state_dict along with optional training metadata.
        This is the recommended way to save model checkpoints as it is more
        flexible than saving the entire model object.

        Args:
            path: Path to save the checkpoint file
            epoch: Current training epoch (optional)
            optimizer_state: Optimizer state_dict (optional)
            metrics: Dictionary of metrics to save (optional)
            **kwargs: Additional metadata to save in the checkpoint

        Example:
            >>> model.save_checkpoint(
            ...     'checkpoints/model_epoch10.pth',
            ...     epoch=10,
            ...     optimizer_state=optimizer.state_dict(),
            ...     metrics={'accuracy': 0.95, 'loss': 0.05}
            ... )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
        }

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if metrics is not None:
            checkpoint["metrics"] = metrics

        # Add any additional metadata
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint from disk.

        Loads the model state_dict and returns any additional metadata
        that was saved with the checkpoint (epoch, metrics, etc.).

        Args:
            path: Path to the checkpoint file
            device: Device to load the model to (CPU/GPU). If None, uses current device
            strict: Whether to strictly enforce that the keys in state_dict match

        Returns:
            Dictionary containing the checkpoint metadata (epoch, metrics, etc.)
            excluding the model_state_dict which is already loaded into the model.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If state_dict keys don't match and strict=True

        Example:
            >>> metadata = model.load_checkpoint('checkpoints/model_best.pth')
            >>> print(f"Loaded from epoch {metadata['epoch']}")
            >>> print(f"Validation accuracy: {metadata['metrics']['accuracy']}")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load checkpoint
        if device is not None:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)

        # Load model state
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Return metadata (everything except the state_dict)
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        return metadata

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters.
                          If False, count all parameters.

        Returns:
            Number of parameters

        Example:
            >>> model = MyModel()
            >>> print(f"Trainable parameters: {model.count_parameters():,}")
            >>> print(f"Total parameters: {model.count_parameters(trainable_only=False):,}")
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze all model parameters (disable gradient computation).

        Useful for transfer learning or feature extraction.

        Example:
            >>> model = PretrainedModel()
            >>> model.freeze()  # Freeze all layers
            >>> model.fc = nn.Linear(512, 10)  # Replace final layer
            >>> # Now only fc layer will be trained
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters (enable gradient computation).

        Example:
            >>> model.freeze()
            >>> # ... train with frozen weights
            >>> model.unfreeze()  # Fine-tune all layers
        """
        for param in self.parameters():
            param.requires_grad = True

    def get_device(self) -> torch.device:
        """Get the device where the model is located.

        Returns:
            Device (CPU or CUDA) where model parameters are located

        Example:
            >>> device = model.get_device()
            >>> print(f"Model is on: {device}")
        """
        return next(self.parameters()).device
