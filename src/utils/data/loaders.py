"""Factory functions for creating DataLoader instances.

This module provides factory functions that replace the ClassifierDataLoader
and DiffusionDataLoader wrapper classes. They create torch DataLoader instances
directly, with support for balancing strategies (weighted sampling, downsampling,
upsampling).
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.utils.data.balancing import downsample_dataset, upsample_dataset
from src.utils.data.datasets import SplitFileDataset
from src.utils.data.samplers import compute_weights_from_config, create_weighted_sampler

_logger = logging.getLogger(__name__)


def create_train_loader(
    split_file: str,
    batch_size: int,
    transform: transforms.Compose,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle: bool = True,
    return_labels: bool = True,
    balancing_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a training DataLoader from a split JSON file.

    Args:
        split_file: Path to split JSON file
        batch_size: Number of samples per batch
        transform: Transform to apply to images
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle training data
        return_labels: Whether to return class labels
        balancing_config: Optional balancing configuration (weighted_sampler,
            downsampling, upsampling)
        seed: Random seed for balancing and DataLoader reproducibility.
            When provided, sets generator for shuffle and worker_init_fn
            for worker-process RNG seeding. When None, no seeding is applied.

    Returns:
        DataLoader for training data

    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If no valid images found in split
    """
    if not Path(split_file).exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    train_dataset = SplitFileDataset(
        split_file=str(split_file),
        split="train",
        transform=transform,
        return_labels=return_labels,
    )

    sampler = None
    dataset_to_use = train_dataset

    # Apply balancing strategy if configured
    if balancing_config:
        ws_config = balancing_config.get("weighted_sampler", {})
        ds_config = balancing_config.get("downsampling", {})
        us_config = balancing_config.get("upsampling", {})

        # Priority: weighted_sampler > downsampling > upsampling
        if ws_config.get("enabled"):
            _logger.info("Applying weighted_sampler balancing strategy")
            weights = compute_weights_from_config(
                targets=train_dataset.targets,
                method=ws_config["method"],
                beta=ws_config.get("beta", 0.999),
                manual_weights=ws_config.get("manual_weights"),
            )
            _logger.info(f"Computed class weights: {weights}")
            sampler = create_weighted_sampler(
                targets=train_dataset.targets,
                class_weights=weights,
                replacement=ws_config.get("replacement", True),
                num_samples=ws_config.get("num_samples"),
            )
            shuffle = False  # sampler and shuffle are mutually exclusive

        elif ds_config.get("enabled"):
            _logger.info("Applying downsampling balancing strategy")
            dataset_to_use = downsample_dataset(
                train_dataset,
                target_ratio=ds_config.get("target_ratio", 1.0),
                seed=seed if seed is not None else 0,
            )

        elif us_config.get("enabled"):
            _logger.info("Applying upsampling balancing strategy")
            dataset_to_use = upsample_dataset(
                train_dataset,
                target_ratio=us_config.get("target_ratio", 1.0),
                seed=seed if seed is not None else 0,
            )

    # Set up reproducible DataLoader when seed is provided
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def _worker_init_fn(worker_id: int) -> None:
            worker_seed = seed + worker_id  # type: ignore[operator]
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _worker_init_fn

    return DataLoader(
        dataset_to_use,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def create_synthetic_augmentation_dataset(
    split_file: str,
    transform: transforms.Compose,
    return_labels: bool = True,
    limit_mode: Optional[str] = None,
    max_ratio: Optional[float] = None,
    max_samples: Optional[int] = None,
    real_train_size: int = 0,
    seed: Optional[int] = None,
) -> Union[SplitFileDataset, Subset]:
    """Create a dataset of synthetic (generated) images for augmentation.

    Loads generated images from a sample-selection output JSON file and
    optionally limits the number of samples via ratio or absolute cap.

    Args:
        split_file: Path to sample selection output JSON (SplitFileDataset format)
        transform: Transform to apply to images
        return_labels: Whether to return class labels
        limit_mode: Limiting strategy: None (use all), "max_ratio", or "max_samples"
        max_ratio: Max generated images as ratio of real training samples
        max_samples: Max absolute number of generated images
        real_train_size: Number of real training samples (used with max_ratio)
        seed: Random seed for reproducible subsampling

    Returns:
        Full dataset or a Subset if limiting is applied
    """
    if not Path(split_file).exists():
        raise FileNotFoundError(
            f"Synthetic augmentation split file not found: {split_file}"
        )

    dataset = SplitFileDataset(
        split_file=str(split_file),
        split="train",
        transform=transform,
        return_labels=return_labels,
    )

    if limit_mode is None or len(dataset) == 0:
        return dataset

    if limit_mode == "max_ratio" and max_ratio is not None:
        if real_train_size <= 0:
            raise ValueError(
                "real_train_size must be positive when using max_ratio limit mode"
            )
        max_n = int(real_train_size * max_ratio)
    elif limit_mode == "max_samples" and max_samples is not None:
        max_n = max_samples
    else:
        return dataset

    if max_n <= 0:
        _logger.warning(
            "Synthetic augmentation limit resolved to 0 samples "
            "(limit_mode=%s, max_ratio=%s, max_samples=%s, real_train_size=%d). "
            "Using full synthetic dataset instead.",
            limit_mode,
            max_ratio,
            max_samples,
            real_train_size,
        )
        return dataset

    if max_n >= len(dataset):
        return dataset

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_n)
    return Subset(dataset, indices)


def create_val_loader(
    split_file: str,
    batch_size: int,
    transform: transforms.Compose,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_labels: bool = True,
) -> Optional[DataLoader]:
    """Create a validation DataLoader from a split JSON file.

    Returns None if the val split is empty.

    Args:
        split_file: Path to split JSON file
        batch_size: Number of samples per batch
        transform: Transform to apply to images
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        return_labels: Whether to return class labels

    Returns:
        DataLoader for validation data, or None if val split is empty

    Raises:
        FileNotFoundError: If split file doesn't exist
    """
    if not Path(split_file).exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    try:
        val_dataset = SplitFileDataset(
            split_file=str(split_file),
            split="val",
            transform=transform,
            return_labels=return_labels,
        )
    except ValueError:
        return None

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def get_num_classes(split_file: str) -> int:
    """Get the number of classes from a split JSON file.

    Args:
        split_file: Path to split JSON file

    Returns:
        Number of classes

    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If no class metadata found in split file
    """
    split_path = Path(split_file)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("metadata", {}).get("classes", {})
    if not classes:
        raise ValueError(f"No class metadata found in split file: {split_file}")
    return len(classes)


def get_class_names(split_file: str) -> List[str]:
    """Get class names from a split JSON file.

    Args:
        split_file: Path to split JSON file

    Returns:
        List of class names sorted by label index

    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If no class metadata found in split file
    """
    split_path = Path(split_file)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes_dict = data.get("metadata", {}).get("classes", {})
    if not classes_dict:
        raise ValueError(f"No class metadata found in split file: {split_file}")
    sorted_classes = sorted(classes_dict.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_classes]
