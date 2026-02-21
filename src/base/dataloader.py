"""Base DataLoader Interface

This module defines the abstract base class for all dataloaders in the experiments.
All experiment-specific dataloaders should inherit from BaseDataLoader and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader


class BaseDataLoader(ABC):
    """Abstract base class for all dataloaders.

    This class provides a common interface for dataloaders across different experiment types
    (GAN, Diffusion, Classifier). It ensures all subclasses implement the required methods
    for creating training and validation data loaders.

    All dataloaders should inherit from this class and implement:
    - get_train_loader(): Create and return the training dataloader
    - get_val_loader(): Create and return the validation dataloader

    The class provides a default implementation for:
    - get_config(): Return dataloader configuration

    Example:
        >>> class MyClassifierDataLoader(BaseDataLoader):
        ...     def __init__(self, train_path: str, val_path: str, batch_size: int = 32):
        ...         self.train_path = train_path
        ...         self.val_path = val_path
        ...         self.batch_size = batch_size
        ...
        ...     def get_train_loader(self) -> DataLoader:
        ...         dataset = ImageFolderDataset(self.train_path, transform=train_transforms)
        ...         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        ...
        ...     def get_val_loader(self) -> DataLoader:
        ...         dataset = ImageFolderDataset(self.val_path, transform=val_transforms)
        ...         return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    """

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Create and return the training data loader.

        This method must be implemented by all subclasses. The implementation
        will vary depending on the experiment type and dataset requirements.

        Returns:
            DataLoader: PyTorch DataLoader for training data

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> train_loader = dataloader.get_train_loader()
            >>> for batch in train_loader:
            ...     # Training loop
            ...     pass
        """
        raise NotImplementedError("Subclasses must implement get_train_loader()")

    @abstractmethod
    def get_val_loader(self) -> Optional[DataLoader]:
        """Create and return the validation data loader.

        This method must be implemented by all subclasses. The implementation
        will vary depending on the experiment type and dataset requirements.
        For experiments without validation data, this may return None.

        Returns:
            DataLoader or None: PyTorch DataLoader for validation data,
                              or None if validation is not applicable

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> val_loader = dataloader.get_val_loader()
            >>> if val_loader is not None:
            ...     for batch in val_loader:
            ...         # Validation loop
            ...         pass
        """
        raise NotImplementedError("Subclasses must implement get_val_loader()")

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the dataloader.

        Returns a dictionary containing the dataloader's configuration parameters.
        This is useful for logging, debugging, and reproducing experiments.

        This method provides a default implementation that can be overridden
        by subclasses to include additional configuration details.

        Returns:
            Dictionary containing dataloader configuration parameters

        Example:
            >>> config = dataloader.get_config()
            >>> print(f"Batch size: {config['batch_size']}")
            >>> print(f"Num workers: {config['num_workers']}")
        """
        # Default implementation - subclasses can override to add more details
        config = {}

        # Try to extract common attributes if they exist
        common_attrs = [
            "batch_size",
            "num_workers",
            "shuffle",
            "pin_memory",
            "drop_last",
            "train_path",
            "val_path",
            "split_file",
        ]

        for attr in common_attrs:
            if hasattr(self, attr):
                config[attr] = getattr(self, attr)

        return config

    def __repr__(self) -> str:
        """String representation of the dataloader.

        Returns:
            String representation showing the class name and key configuration
        """
        config = self.get_config()
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        return f"{self.__class__.__name__}({config_str})"
