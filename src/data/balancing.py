"""Dataset balancing utilities for handling imbalanced datasets.

This module provides functions for downsampling (undersampling) and
upsampling (oversampling) datasets to address class imbalance.

Both functions operate on datasets with a `targets` attribute and return
a `torch.utils.data.Subset` with balanced indices. They use local
`torch.Generator` instances seeded from the provided seed to ensure
reproducibility without corrupting the global random state.

Usage Examples
--------------

Example 1: Downsampling
~~~~~~~~~~~~~~~~~~~~~~~~
>>> from src.data.datasets import SplitFileDataset
>>> from src.data.balancing import downsample_dataset
>>>
>>> dataset = SplitFileDataset("outputs/splits/train_val_split.json", split="train")
>>> balanced = downsample_dataset(dataset, target_ratio=1.0, seed=42)
>>> print(f"Original: {len(dataset)}, Balanced: {len(balanced)}")

Example 2: Upsampling
~~~~~~~~~~~~~~~~~~~~~
>>> from src.data.balancing import upsample_dataset
>>>
>>> balanced = upsample_dataset(dataset, target_ratio=1.0, seed=42)
>>> print(f"Original: {len(dataset)}, Balanced: {len(balanced)}")
"""

import logging
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import Subset

from src.data.datasets import BaseDataset

_logger = logging.getLogger(__name__)


def downsample_dataset(
    dataset: BaseDataset,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> Subset:
    """Downsample majority class to achieve target minority:majority ratio.

    Uses a local torch.Generator seeded from the provided seed to avoid
    corrupting the global random state.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of minority:majority (1.0 = equal counts)
        seed: Seed for local random generator (typically from compute.seed)

    Returns:
        torch.utils.data.Subset with balanced indices

    Raises:
        ValueError: If dataset has no targets or target_ratio is invalid
        AttributeError: If dataset doesn't have a targets attribute

    Example:
        >>> balanced = downsample_dataset(dataset, target_ratio=1.0, seed=42)
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            f"Dataset {type(dataset).__name__} does not have 'targets' attribute"
        )

    targets = dataset.targets
    if not targets:
        raise ValueError("Dataset has no samples")

    # Count samples per class
    class_counts: Dict[int, int] = Counter(targets)

    # Identify minority and majority classes
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]

    _logger.info(
        f"Downsampling: minority class {minority_class} ({minority_count} samples), "
        f"majority class {majority_class} ({majority_count} samples)"
    )

    # Compute target majority count
    target_majority_count = int(minority_count / target_ratio)
    target_majority_count = min(target_majority_count, majority_count)

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Build balanced indices
    balanced_indices: List[int] = []
    for cls, indices in class_indices.items():
        if cls == majority_class:
            # Randomly select target_majority_count indices
            perm = torch.randperm(len(indices), generator=generator)
            selected = perm[:target_majority_count].tolist()
            balanced_indices.extend([indices[i] for i in selected])
        else:
            # Keep all minority class samples
            balanced_indices.extend(indices)

    _logger.info(
        f"Downsampling result: {len(balanced_indices)} samples "
        f"(from {len(targets)} original)"
    )

    return Subset(dataset, balanced_indices)


def upsample_dataset(
    dataset: BaseDataset,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> Subset:
    """Upsample minority class by duplication to achieve target ratio.

    Uses a local torch.Generator for reproducible sampling without
    corrupting the global random state.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of minority:majority (1.0 = equal counts)
        seed: Seed for local random generator

    Returns:
        torch.utils.data.Subset with duplicated minority indices

    Raises:
        ValueError: If dataset has no targets or target_ratio is invalid
        AttributeError: If dataset doesn't have a targets attribute

    Example:
        >>> balanced = upsample_dataset(dataset, target_ratio=1.0, seed=42)
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            f"Dataset {type(dataset).__name__} does not have 'targets' attribute"
        )

    targets = dataset.targets
    if not targets:
        raise ValueError("Dataset has no samples")

    # Count samples per class
    class_counts: Dict[int, int] = Counter(targets)

    # Identify minority and majority classes
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]

    _logger.info(
        f"Upsampling: minority class {minority_class} ({minority_count} samples), "
        f"majority class {majority_class} ({majority_count} samples)"
    )

    # Compute target minority count
    target_minority_count = int(majority_count * target_ratio)
    extra = target_minority_count - minority_count

    if extra <= 0:
        _logger.info("No upsampling needed (minority already meets target ratio)")
        return Subset(dataset, list(range(len(targets))))

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Start with all original indices
    all_indices: List[int] = list(range(len(targets)))

    # Randomly pick extra indices (with replacement) from minority class
    minority_indices = class_indices[minority_class]
    extra_selection = torch.randint(
        0, len(minority_indices), (extra,), generator=generator
    )
    extra_indices = [minority_indices[i] for i in extra_selection.tolist()]
    all_indices.extend(extra_indices)

    _logger.info(
        f"Upsampling result: {len(all_indices)} samples "
        f"(added {extra} duplicated minority samples)"
    )

    return Subset(dataset, all_indices)
