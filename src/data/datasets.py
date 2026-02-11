"""
Dataset implementations for image loading and preprocessing.

This module provides dataset classes for various experiment types,
including base classes and specific implementations like ImageFolder.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.

    Provides a common interface for datasets used across different experiments.
    All custom datasets should inherit from this class.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Sample data (format depends on specific implementation)
        """
        pass

    @abstractmethod
    def get_classes(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class names in the dataset
        """
        pass

    @abstractmethod
    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the number of samples per class.

        Returns:
            Dictionary mapping class names to sample counts
        """
        pass


class ImageFolderDataset(BaseDataset):
    """
    Dataset for loading images from a folder structure.

    Expected directory structure:
        root/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                img2.jpg
                ...

    This wraps torchvision.datasets.ImageFolder with additional functionality
    for compatibility with the base architecture.

    Args:
        root: Root directory path
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
        extensions: Tuple of valid image extensions (default: common formats)
        is_valid_file: Optional function to filter files
        return_labels: Whether to return labels with images (default: True)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        return_labels: bool = True,
    ):
        self.root = Path(root)
        self.return_labels = return_labels

        # Validate root directory exists
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root}")

        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {root}")

        # Use torchvision's ImageFolder for the heavy lifting
        # Default extensions if not provided
        if extensions is None and is_valid_file is None:
            extensions = (
                ".jpg",
                ".jpeg",
                ".png",
                ".ppm",
                ".bmp",
                ".pgm",
                ".tif",
                ".tiff",
                ".webp",
            )

        try:
            self._dataset = datasets.ImageFolder(
                root=str(root),
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file,
            )
        except FileNotFoundError as e:
            # Convert FileNotFoundError from torchvision to ValueError for consistency
            raise ValueError(f"No valid images found in {root}") from e

        # Verify we have some samples
        if len(self._dataset) == 0:
            raise ValueError(f"No valid images found in {root}")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, target) where target is the class index if return_labels=True,
            otherwise returns only the image tensor.
        """
        image, label = self._dataset[index]

        if self.return_labels:
            return image, label
        else:
            return image

    def get_classes(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class names (subfolder names) sorted alphabetically
        """
        return self._dataset.classes

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the number of samples per class.

        Returns:
            Dictionary mapping class names to sample counts
        """
        counts = {}
        for _, label_idx in self._dataset.samples:
            class_name = self._dataset.classes[label_idx]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    @property
    def classes(self) -> List[str]:
        """Get class names (property for compatibility)."""
        return self._dataset.classes

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Get mapping from class name to class index."""
        return self._dataset.class_to_idx

    @property
    def samples(self) -> List[Tuple[str, int]]:
        """Get list of (image_path, class_index) tuples."""
        return self._dataset.samples

    @property
    def targets(self) -> List[int]:
        """Get list of class indices for all samples."""
        return self._dataset.targets

    def get_sample_weights(
        self, class_weights: Optional[Dict[int, float]] = None
    ) -> List[float]:
        """
        Calculate sample weights for weighted sampling.

        Args:
            class_weights: Optional dictionary mapping class index to weight.
                          If None, uses inverse frequency weighting.

        Returns:
            List of weights for each sample
        """
        if class_weights is None:
            # Compute inverse frequency weights
            class_counts = {}
            for _, label in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1

            # Weight = 1 / count
            total = len(self.samples)
            class_weights = {
                label: total / count for label, count in class_counts.items()
            }

        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for _, label in self.samples]
        return sample_weights

    def print_summary(self):
        """Print a summary of the dataset."""
        print(f"Dataset: {self.root}")
        print(f"Total samples: {len(self)}")
        print(f"Classes: {self.classes}")
        print("Class distribution:")

        class_counts = self.get_class_counts()
        for class_name in self.classes:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(self)) * 100 if len(self) > 0 else 0
            print(f"  - {class_name}: {count} images ({percentage:.2f}%)")


class SimpleImageDataset(BaseDataset):
    """
    Simple dataset for loading images from a flat directory.

    All images are in a single directory without class subfolders.
    Useful for generation tasks or when labels are not needed.

    Args:
        root: Root directory containing images
        transform: Optional transform to apply to images
        extensions: Tuple of valid image extensions
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        ),
    ):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root}")

        self.transform = transform
        self.extensions = extensions

        # Find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.root.glob(f"*{ext}"))
            self.image_paths.extend(self.root.glob(f"*{ext.upper()}"))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {root}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Transformed image tensor
        """
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def get_classes(self) -> List[str]:
        """Get list of class names (returns empty list for unlabeled dataset)."""
        return []

    def get_class_counts(self) -> Dict[str, int]:
        """Get class counts (returns empty dict for unlabeled dataset)."""
        return {}


def get_dataset(
    dataset_type: str, root: str, transform: Optional[Callable] = None, **kwargs
) -> BaseDataset:
    """
    Factory function to create a dataset instance.

    Args:
        dataset_type: Type of dataset ('imagefolder', 'simple')
        root: Root directory path
        transform: Optional transform to apply
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_type is unknown
    """
    dataset_type = dataset_type.lower()

    if dataset_type == "imagefolder":
        return ImageFolderDataset(root=root, transform=transform, **kwargs)
    elif dataset_type == "simple":
        return SimpleImageDataset(root=root, transform=transform, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Choose from: 'imagefolder', 'simple'"
        )
