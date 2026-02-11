"""Classifier DataLoader

This module implements the dataloader for classification experiments.
It inherits from BaseDataLoader and provides classification-specific
data loading and preprocessing functionality.
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from src.base.dataloader import BaseDataLoader
from src.data.datasets import ImageFolderDataset
from src.data.transforms import (
    get_base_transforms,
    get_normalization_transform,
    get_train_transforms,
)


class ClassifierDataLoader(BaseDataLoader):
    """DataLoader for classification experiments.

    This dataloader provides functionality for loading image classification datasets
    with appropriate preprocessing and augmentation. It inherits from BaseDataLoader
    and implements classification-specific data loading logic.

    Features:
    - Automatic train/validation split support
    - Configurable image preprocessing and augmentation
    - Support for different normalization schemes (ImageNet, CIFAR10, custom, none)
    - Flexible batch size and worker configuration

    Args:
        train_path: Path to training data directory (ImageFolder structure)
        val_path: Path to validation data directory (optional)
        batch_size: Number of samples per batch (default: 32)
        num_workers: Number of worker processes for data loading (default: 4)
        image_size: Size to resize images to before cropping (default: 256)
        crop_size: Size of the final crop (default: 224)
        horizontal_flip: Whether to apply random horizontal flip during training (default: True)
        color_jitter: Whether to apply color jittering during training (default: False)
        rotation_degrees: Maximum rotation degrees for augmentation (default: 0)
        normalize: Normalization scheme ('imagenet', 'cifar10', 'none', or None for no normalization)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        drop_last: Whether to drop the last incomplete batch (default: False)
        shuffle_train: Whether to shuffle training data (default: True)

    Example:
        >>> dataloader = ClassifierDataLoader(
        ...     train_path="data/train",
        ...     val_path="data/val",
        ...     batch_size=32,
        ...     image_size=256,
        ...     crop_size=224,
        ...     normalize="imagenet"
        ... )
        >>> train_loader = dataloader.get_train_loader()
        >>> val_loader = dataloader.get_val_loader()
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,
        crop_size: int = 224,
        horizontal_flip: bool = True,
        color_jitter: bool = False,
        rotation_degrees: int = 0,
        normalize: Optional[str] = "imagenet",
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle_train: bool = True,
    ):
        """Initialize the classifier dataloader.

        Args:
            train_path: Path to training data directory
            val_path: Path to validation data directory (optional)
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            image_size: Size to resize images to before cropping
            crop_size: Size of the final crop
            horizontal_flip: Whether to apply random horizontal flip during training
            color_jitter: Whether to apply color jittering during training
            rotation_degrees: Maximum rotation degrees for augmentation
            normalize: Normalization scheme ('imagenet', 'cifar10', 'none', or None)
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            shuffle_train: Whether to shuffle training data
        """
        self.train_path = str(train_path)
        self.val_path = str(val_path) if val_path is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.rotation_degrees = rotation_degrees
        self.normalize = normalize
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train

        # Validate paths
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Training data path not found: {train_path}")

        if val_path is not None and not Path(val_path).exists():
            raise FileNotFoundError(f"Validation data path not found: {val_path}")

    def _get_train_transform(self):
        """Get training transforms with augmentation.

        Returns:
            Composed transforms for training data
        """
        # Get base augmentation transforms
        transform = get_train_transforms(
            image_size=self.image_size,
            crop_size=self.crop_size,
            horizontal_flip=self.horizontal_flip,
            color_jitter=self.color_jitter,
            rotation_degrees=self.rotation_degrees,
            normalize=self.normalize if self.normalize not in ["none", None] else None,
        )
        return transform

    def _get_val_transform(self):
        """Get validation transforms without augmentation.

        Returns:
            Composed transforms for validation data
        """
        # Get base transforms without augmentation
        transform = get_base_transforms(
            image_size=self.image_size,
            crop_size=self.crop_size,
            resize_mode="resize_crop",
        )

        # Add normalization if specified
        if self.normalize is not None and self.normalize != "none":
            from torchvision import transforms

            normalize_transform = get_normalization_transform(dataset=self.normalize)
            transform = transforms.Compose(
                list(transform.transforms) + [normalize_transform]
            )

        return transform

    def get_train_loader(self) -> DataLoader:
        """Create and return the training data loader.

        Creates a DataLoader for training data with augmentation and shuffling.

        Returns:
            DataLoader for training data

        Raises:
            FileNotFoundError: If training data path doesn't exist
            ValueError: If no valid images found in training path

        Example:
            >>> train_loader = dataloader.get_train_loader()
            >>> for images, labels in train_loader:
            ...     # Training loop
            ...     pass
        """
        transform = self._get_train_transform()

        # Create dataset
        train_dataset = ImageFolderDataset(root=self.train_path, transform=transform)

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        return train_loader

    def get_val_loader(self) -> Optional[DataLoader]:
        """Create and return the validation data loader.

        Creates a DataLoader for validation data without augmentation.
        Returns None if no validation path was specified.

        Returns:
            DataLoader for validation data, or None if val_path is None

        Raises:
            FileNotFoundError: If validation data path doesn't exist
            ValueError: If no valid images found in validation path

        Example:
            >>> val_loader = dataloader.get_val_loader()
            >>> if val_loader is not None:
            ...     for images, labels in val_loader:
            ...         # Validation loop
            ...         pass
        """
        if self.val_path is None:
            return None

        transform = self._get_val_transform()

        # Create dataset
        val_dataset = ImageFolderDataset(root=self.val_path, transform=transform)

        # Create dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch in validation
        )

        return val_loader

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset.

        Returns:
            Number of classes

        Raises:
            FileNotFoundError: If training data path doesn't exist
            ValueError: If no valid images found in training path
        """
        transform = self._get_train_transform()
        train_dataset = ImageFolderDataset(root=self.train_path, transform=transform)
        return len(train_dataset.get_classes())

    def get_class_names(self):
        """Get the class names from the dataset.

        Returns:
            List of class names

        Raises:
            FileNotFoundError: If training data path doesn't exist
            ValueError: If no valid images found in training path
        """
        transform = self._get_train_transform()
        train_dataset = ImageFolderDataset(root=self.train_path, transform=transform)
        return train_dataset.get_classes()
