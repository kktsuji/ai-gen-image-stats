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
>>> from src.utils.data.datasets import SplitFileDataset
>>> from src.utils.data.balancing import downsample_dataset
>>>
>>> dataset = SplitFileDataset("outputs/splits/train_val_split.json", split="train")
>>> balanced = downsample_dataset(dataset, target_ratio=1.0, seed=42)
>>> print(f"Original: {len(dataset)}, Balanced: {len(balanced)}")

Example 2: Upsampling
~~~~~~~~~~~~~~~~~~~~~
>>> from src.utils.data.balancing import upsample_dataset
>>>
>>> balanced = upsample_dataset(dataset, target_ratio=1.0, seed=42)
>>> print(f"Original: {len(dataset)}, Balanced: {len(balanced)}")
"""

import logging
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import Subset

from src.utils.data.datasets import BaseDataset

_logger = logging.getLogger(__name__)


def downsample_dataset(
    dataset: BaseDataset,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> Subset:
    """Downsample over-represented classes toward the smallest class count.

    Multi-class generalization: every class with more samples than the target
    count is randomly downsampled to that target; smaller classes are kept
    whole. The target count is derived from the smallest class so that, with
    ``target_ratio=1.0``, all classes end up at the minority count. For a
    two-class dataset this reduces exactly to "downsample the majority to
    ``minority_count / target_ratio``".

    Uses a local torch.Generator seeded from the provided seed to avoid
    corrupting the global random state. Classes are processed in ascending
    label order so the random draws are reproducible.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of each minority class to the smallest
            class (1.0 = all classes equal to the smallest). Must be in (0, 1].
        seed: Seed for local random generator (typically from compute.seed)

    Returns:
        torch.utils.data.Subset with balanced indices

    Raises:
        ValueError: If dataset has no targets or target_ratio is not positive
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

    if target_ratio <= 0:
        raise ValueError(f"target_ratio must be positive, got {target_ratio}")

    # Count samples per class
    class_counts: Dict[int, int] = Counter(targets)
    min_count = min(class_counts.values())

    # Every class is downsampled to (at most) this target count. Derived from
    # the smallest class so the smallest class is always kept whole.
    target_count = int(min_count / target_ratio)

    _logger.info(
        f"Downsampling {len(class_counts)} classes toward target_count="
        f"{target_count} (smallest class has {min_count} samples)"
    )

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Build balanced indices, processing classes in ascending label order so the
    # generator is consumed deterministically.
    balanced_indices: List[int] = []
    for cls in sorted(class_indices):
        indices = class_indices[cls]
        if len(indices) > target_count:
            # Randomly select target_count indices for over-represented classes
            perm = torch.randperm(len(indices), generator=generator)
            selected = perm[:target_count].tolist()
            balanced_indices.extend([indices[i] for i in selected])
        else:
            # Keep all samples for classes at/below the target count
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
    """Upsample under-represented classes by duplication toward the largest class.

    Multi-class generalization: every class with fewer samples than the target
    count is duplicated (sampling with replacement) up to that target; larger
    classes are left untouched. The target count is derived from the largest
    class so that, with ``target_ratio=1.0``, all classes end up at the
    majority count. For a two-class dataset this reduces exactly to "upsample
    the minority to ``majority_count * target_ratio``".

    Uses a local torch.Generator for reproducible sampling without corrupting
    the global random state. Classes are processed in ascending label order so
    the duplicated indices are reproducible.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of each minority class to the largest class
            (1.0 = all classes equal to the largest). Must be positive.
        seed: Seed for local random generator

    Returns:
        torch.utils.data.Subset with duplicated minority indices

    Raises:
        ValueError: If dataset has no targets or target_ratio is not positive
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
    max_count = max(class_counts.values())

    # Every class is duplicated up to this target count.
    target_count = int(max_count * target_ratio)

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_indices.setdefault(target, []).append(idx)

    # Use local generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Start with all original indices, then append duplicates per class. Classes
    # are processed in ascending label order so the generator is consumed
    # deterministically.
    all_indices: List[int] = list(range(len(targets)))
    total_added = 0
    for cls in sorted(class_indices):
        indices = class_indices[cls]
        extra = target_count - len(indices)
        if extra <= 0:
            continue
        # Randomly pick extra indices (with replacement) from this class
        extra_selection = torch.randint(0, len(indices), (extra,), generator=generator)
        all_indices.extend(indices[i] for i in extra_selection.tolist())
        total_added += extra

    if total_added == 0:
        _logger.info("No upsampling needed (all classes already meet target ratio)")
    else:
        _logger.info(
            f"Upsampling result: {len(all_indices)} samples "
            f"(added {total_added} duplicated samples across "
            f"{len(class_counts)} classes)"
        )

    return Subset(dataset, all_indices)
