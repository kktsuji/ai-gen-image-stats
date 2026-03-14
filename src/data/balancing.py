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

    targets: List[int] = getattr(dataset, "targets")
    if not targets:
        raise ValueError("Dataset has no samples")

    if target_ratio <= 0 or target_ratio > 1.0:
        raise ValueError(f"target_ratio must be in (0, 1.0], got {target_ratio}")

    # Count samples per class
    class_counts: Dict[int, int] = Counter(targets)

    # Identify the smallest class as the target size reference
    min_count = min(class_counts.values())

    _logger.info(
        f"Downsampling: {len(class_counts)} classes, "
        f"min class count={min_count}, target_ratio={target_ratio}"
    )

    # Compute target count: each class should have at most this many samples
    target_count = int(min_count / target_ratio)

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Build balanced indices — downsample every class that exceeds target_count
    balanced_indices: List[int] = []
    for cls, indices in class_indices.items():
        if len(indices) > target_count:
            perm = torch.randperm(len(indices), generator=generator)
            selected = perm[:target_count].tolist()
            balanced_indices.extend([indices[i] for i in selected])
        else:
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

    targets: List[int] = getattr(dataset, "targets")
    if not targets:
        raise ValueError("Dataset has no samples")

    if target_ratio <= 0:
        raise ValueError(f"target_ratio must be positive, got {target_ratio}")

    # Count samples per class
    class_counts: Dict[int, int] = Counter(targets)

    # Identify the largest class as the target size reference
    max_count = max(class_counts.values())

    _logger.info(
        f"Upsampling: {len(class_counts)} classes, "
        f"max class count={max_count}, target_ratio={target_ratio}"
    )

    # Compute target count: each class should have at least this many samples
    target_count = int(max_count * target_ratio)

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Start with all original indices
    all_indices: List[int] = list(range(len(targets)))

    # Upsample every class that is below the target count
    total_extra = 0
    for cls, indices in class_indices.items():
        extra = target_count - len(indices)
        if extra > 0:
            extra_selection = torch.randint(
                0, len(indices), (extra,), generator=generator
            )
            extra_indices = [indices[i] for i in extra_selection.tolist()]
            all_indices.extend(extra_indices)
            total_extra += extra

    _logger.info(
        f"Upsampling result: {len(all_indices)} samples "
        f"(added {total_extra} duplicated samples)"
    )

    return Subset(dataset, all_indices)
