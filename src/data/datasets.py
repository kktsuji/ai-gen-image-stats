"""
Dataset implementations for image loading and preprocessing.

This module provides dataset classes for various experiment types,
including base classes and specific implementations like ImageFolder.
"""

import json
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


class SplitFileDataset(BaseDataset):
    """
    Dataset that loads images from a split JSON file.

    The split JSON file is generated by the data_preparation experiment and
    contains train/val split information with image paths and labels.

    Args:
        split_file: Path to the split JSON file
        split: Which split to use ('train' or 'val')
        transform: Optional transform to apply to images
        return_labels: Whether to return labels with images (default: True)

    Example:
        >>> dataset = SplitFileDataset(
        ...     split_file="outputs/splits/train_val_split.json",
        ...     split="train",
        ...     transform=my_transform,
        ... )
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        split_file: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_labels: bool = True,
    ):
        self.split_file = str(split_file)
        self.split = split
        self.transform = transform
        self.return_labels = return_labels

        # Validate split key
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: '{split}'. Must be 'train' or 'val'")

        # Load and parse JSON
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if split not in data:
            raise KeyError(
                f"Split '{split}' not found in split file. "
                f"Available keys: {list(data.keys())}"
            )

        # Extract metadata
        metadata = data.get("metadata", {})
        self._classes_dict = metadata.get("classes", {})

        # Build class lists from metadata
        # classes_dict maps class_name -> label_idx
        sorted_classes = sorted(self._classes_dict.items(), key=lambda x: x[1])
        self._classes = [name for name, _ in sorted_classes]
        self._class_to_idx = dict(self._classes_dict)

        # Extract samples for this split
        split_entries = data[split]
        self._samples = [(entry["path"], entry["label"]) for entry in split_entries]
        self._targets = [entry["label"] for entry in split_entries]

        if len(self._samples) == 0:
            raise ValueError(f"No samples found in '{split}' split of {split_file}")

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, label) if return_labels=True,
            otherwise returns only the image tensor.

        Raises:
            FileNotFoundError: If image file does not exist
        """
        path, label = self._samples[index]

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Image file not found: {path} "
                f"(referenced in split file: {self.split_file})"
            )

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_labels:
            return image, label
        else:
            return image

    def get_classes(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class names sorted by label index
        """
        return list(self._classes)

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the number of samples per class.

        Returns:
            Dictionary mapping class names to sample counts
        """
        counts = {}
        for _, label_idx in self._samples:
            if label_idx < len(self._classes):
                class_name = self._classes[label_idx]
            else:
                class_name = f"class_{label_idx}"
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    @property
    def classes(self) -> List[str]:
        """Get class names (property for compatibility)."""
        return list(self._classes)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Get mapping from class name to class index."""
        return dict(self._class_to_idx)

    @property
    def samples(self) -> List[Tuple[str, int]]:
        """Get list of (image_path, class_index) tuples."""
        return list(self._samples)

    @property
    def targets(self) -> List[int]:
        """Get list of class indices for all samples."""
        return list(self._targets)


def get_dataset(
    dataset_type: str, root: str = "", transform: Optional[Callable] = None, **kwargs
) -> BaseDataset:
    """
    Factory function to create a dataset instance.

    Args:
        dataset_type: Type of dataset ('imagefolder', 'simple', 'splitfile')
        root: Root directory path (not used for 'splitfile' type)
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
    elif dataset_type == "splitfile":
        return SplitFileDataset(transform=transform, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Choose from: 'imagefolder', 'simple', 'splitfile'"
        )
