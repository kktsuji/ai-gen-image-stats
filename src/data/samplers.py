"""
Custom samplers for handling imbalanced datasets and special sampling strategies.

This module provides sampler utilities and factory functions for creating
samplers that work with the dataset implementations in this package.

Usage Examples
--------------

Example 1: Basic Weighted Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> from src.data.datasets import ImageFolderDataset
>>> from src.data.samplers import get_sampler_from_dataset
>>> from torch.utils.data import DataLoader
>>>
>>> # Load dataset
>>> dataset = ImageFolderDataset("data/train")
>>>
>>> # Create weighted sampler (automatically handles class imbalance)
>>> sampler = get_sampler_from_dataset(dataset, sampler_type="weighted")
>>>
>>> # Use with DataLoader (don't use shuffle with sampler)
>>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)

Example 2: Balanced Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Create balanced sampler (all classes have equal probability)
>>> sampler = get_sampler_from_dataset(dataset, sampler_type="balanced")
>>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)

Example 3: Custom Class Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> from src.data.samplers import create_weighted_sampler, compute_class_weights
>>>
>>> # Compute custom weights (e.g., emphasize minority class more)
>>> weights = compute_class_weights(dataset.targets, weight_mode="effective_num")
>>>
>>> # Create sampler with custom weights
>>> sampler = create_weighted_sampler(dataset.targets, class_weights=weights)
>>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)

Example 4: No Sampler (Default Behavior)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> # Return None to use default random sampling
>>> sampler = get_sampler_from_dataset(dataset, sampler_type=None)
>>> # sampler is None, so use shuffle instead
>>> loader = DataLoader(dataset, batch_size=32, shuffle=True)

Notes
-----
- When using a sampler, do NOT set shuffle=True in DataLoader
- Weighted sampling uses replacement by default
- Use 'balanced' for equal class probabilities regardless of distribution
- Use 'weighted' for inverse frequency weighting (more common in practice)
"""

from typing import Any, Dict, List, Optional

from torch.utils.data import Sampler, WeightedRandomSampler


def create_weighted_sampler(
    targets: List[int],
    class_weights: Optional[Dict[int, float]] = None,
    replacement: bool = True,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling class imbalance.

    This sampler is useful when training on imbalanced datasets where some
    classes have significantly more samples than others. It ensures that
    each class is sampled proportionally to its weight.

    Args:
        targets: List of class indices for all samples in the dataset
        class_weights: Optional dictionary mapping class index to weight.
                      If None, uses inverse frequency weighting (1/count).
        replacement: If True, samples are drawn with replacement.
                    If False, samples are exhausted in one epoch.
        num_samples: Number of samples to draw. If None, defaults to
                    len(targets) for replacement=True, or len(targets)
                    for replacement=False.

    Returns:
        WeightedRandomSampler configured with appropriate weights

    Example:
        >>> from src.data.datasets import ImageFolderDataset
        >>> dataset = ImageFolderDataset("path/to/data")
        >>> sampler = create_weighted_sampler(dataset.targets)
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    Example with custom weights:
        >>> # Emphasize minority class more
        >>> custom_weights = {0: 1.0, 1: 3.0}  # Class 1 is 3x more important
        >>> sampler = create_weighted_sampler(dataset.targets, custom_weights)
    """
    if not targets:
        raise ValueError("targets list cannot be empty")

    # Calculate class weights if not provided
    if class_weights is None:
        # Count samples per class
        class_counts = {}
        for target in targets:
            class_counts[target] = class_counts.get(target, 0) + 1

        # Calculate inverse frequency weights
        total = len(targets)
        class_weights = {label: total / count for label, count in class_counts.items()}

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[target] for target in targets]

    # Determine number of samples to draw
    if num_samples is None:
        num_samples = len(targets)

    # Create and return sampler
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
    )


def create_balanced_sampler(
    targets: List[int],
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Create a sampler that balances all classes equally.

    This is a convenience function that creates a weighted sampler where
    all classes have equal probability of being sampled, regardless of
    their frequency in the dataset.

    Args:
        targets: List of class indices for all samples in the dataset
        num_samples: Number of samples to draw. If None, defaults to len(targets).

    Returns:
        WeightedRandomSampler with uniform class probabilities

    Example:
        >>> from src.data.datasets import ImageFolderDataset
        >>> dataset = ImageFolderDataset("path/to/data")
        >>> sampler = create_balanced_sampler(dataset.targets)
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    # Count samples per class
    class_counts = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    # Calculate weights to equalize class probabilities
    # Weight = 1 / (class_count * num_classes)
    num_classes = len(class_counts)
    class_weights = {
        label: 1.0 / (count * num_classes) for label, count in class_counts.items()
    }

    return create_weighted_sampler(
        targets=targets,
        class_weights=class_weights,
        replacement=True,
        num_samples=num_samples,
    )


def compute_class_weights(
    targets: List[int],
    weight_mode: str = "inverse_freq",
    beta: float = 0.9999,
) -> Dict[int, float]:
    """
    Compute class weights based on the distribution of targets.

    Args:
        targets: List of class indices for all samples
        weight_mode: Mode for computing weights. Options:
                    - "inverse_freq": 1 / class_frequency (default)
                    - "balanced": Same as inverse_freq but normalized
                    - "effective_num": Effective number of samples weighting
        beta: Beta parameter for effective_num mode (default: 0.9999)

    Returns:
        Dictionary mapping class index to weight

    Raises:
        ValueError: If weight_mode is not recognized

    Example:
        >>> weights = compute_class_weights([0, 0, 0, 1, 1, 2])
        >>> print(weights)
        {0: 2.0, 1: 3.0, 2: 6.0}
    """
    if not targets:
        raise ValueError("targets list cannot be empty")

    # Count samples per class
    class_counts = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    total = len(targets)
    num_classes = len(class_counts)

    if weight_mode == "inverse_freq":
        # Weight = total / class_count
        weights = {label: total / count for label, count in class_counts.items()}

    elif weight_mode == "balanced":
        # Weight = total / (num_classes * class_count)
        weights = {
            label: total / (num_classes * count)
            for label, count in class_counts.items()
        }

    elif weight_mode == "effective_num":
        # Effective number of samples weighting
        # More robust for very imbalanced datasets
        # See: "Class-Balanced Loss Based on Effective Number of Samples"
        weights = {}
        for label, count in class_counts.items():
            effective_num = (1.0 - beta**count) / (1.0 - beta)
            weights[label] = 1.0 / effective_num

    else:
        raise ValueError(
            f"Unknown weight_mode: {weight_mode}. "
            f"Choose from: 'inverse_freq', 'balanced', 'effective_num'"
        )

    return weights


def compute_effective_num_weights(
    targets: List[int],
    beta: float = 0.9999,
) -> Dict[int, float]:
    """Compute class weights using the effective number of samples method.

    Reference: 'Class-Balanced Loss Based on Effective Number of Samples' (CVPR 2019)

    Effective number: E_n = (1 - beta^n) / (1 - beta)
    Weight: w = 1 / E_n

    Args:
        targets: List of class indices for all samples
        beta: Beta parameter controlling the effective number (default: 0.9999)

    Returns:
        Dictionary mapping class index to weight

    Raises:
        ValueError: If targets is empty or beta is out of range

    Example:
        >>> weights = compute_effective_num_weights([0, 0, 0, 1, 1, 2])
        >>> # Class 2 (1 sample) gets highest weight
    """
    if not targets:
        raise ValueError("targets list cannot be empty")

    if beta <= 0 or beta >= 1:
        raise ValueError(f"beta must be between 0 and 1 (exclusive), got {beta}")

    # Count samples per class
    class_counts: Dict[int, int] = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    weights: Dict[int, float] = {}
    for label, count in class_counts.items():
        effective_num = (1.0 - beta**count) / (1.0 - beta)
        weights[label] = 1.0 / effective_num

    return weights


def compute_weights_from_config(
    targets: List[int],
    method: str,
    beta: float = 0.999,
    manual_weights: Optional[List[float]] = None,
    normalize: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[int, float]:
    """Compute class weights based on config parameters.

    Dispatcher function that computes weights using the specified method.

    Args:
        targets: List of class indices
        method: 'inverse_frequency', 'effective_num', or 'manual'
        beta: Beta for effective_num method
        manual_weights: Per-class weights for manual mode
        normalize: Normalize weights to sum to num_classes
        num_classes: Number of classes (for normalization, inferred if None)

    Returns:
        Dict mapping class index to weight

    Raises:
        ValueError: If method is unknown or manual_weights invalid

    Example:
        >>> weights = compute_weights_from_config(
        ...     targets=[0, 0, 0, 1],
        ...     method="inverse_frequency",
        ... )
        >>> print(weights)
        {0: 1.333..., 1: 4.0}
    """
    if not targets:
        raise ValueError("targets list cannot be empty")

    if method == "inverse_frequency":
        weights = compute_class_weights(targets, weight_mode="inverse_freq")

    elif method == "effective_num":
        weights = compute_effective_num_weights(targets, beta=beta)

    elif method == "manual":
        if manual_weights is None or len(manual_weights) == 0:
            raise ValueError(
                "manual_weights must be a non-empty list when method='manual'"
            )
        # Map index to weight
        weights = {i: w for i, w in enumerate(manual_weights)}

    else:
        raise ValueError(
            f"Unknown weight method: {method}. "
            f"Choose from: 'inverse_frequency', 'effective_num', 'manual'"
        )

    # Normalize weights to sum to num_classes
    if normalize:
        if num_classes is None:
            num_classes = len(weights)
        total_weight = sum(weights.values())
        if total_weight > 0:
            scale = num_classes / total_weight
            weights = {k: v * scale for k, v in weights.items()}

    return weights


def get_sampler_from_dataset(
    dataset: Any,
    sampler_type: str = "weighted",
    **kwargs: Any,
) -> Optional[Sampler]:
    """
    Factory function to create a sampler from a dataset.

    This function inspects the dataset and creates an appropriate sampler
    based on the specified type.

    Args:
        dataset: Dataset object with a 'targets' attribute
        sampler_type: Type of sampler to create. Options:
                     - "weighted": Weighted random sampler (default)
                     - "balanced": Balanced class sampler
                     - None: Return None (use default shuffle)
        **kwargs: Additional arguments passed to sampler creation

    Returns:
        Sampler instance or None if sampler_type is None

    Raises:
        AttributeError: If dataset doesn't have 'targets' attribute
        ValueError: If sampler_type is not recognized

    Example:
        >>> dataset = ImageFolderDataset("path/to/data")
        >>> sampler = get_sampler_from_dataset(dataset, "balanced")
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    Example with no sampler:
        >>> sampler = get_sampler_from_dataset(dataset, sampler_type=None)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    if sampler_type is None or sampler_type.lower() == "none":
        return None

    # Ensure dataset has targets attribute
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            f"Dataset {type(dataset).__name__} does not have 'targets' attribute. "
            f"Cannot create sampler."
        )

    targets = dataset.targets

    if sampler_type == "weighted":
        return create_weighted_sampler(targets, **kwargs)
    elif sampler_type == "balanced":
        return create_balanced_sampler(targets, **kwargs)
    else:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type}. "
            f"Choose from: 'weighted', 'balanced', None"
        )
