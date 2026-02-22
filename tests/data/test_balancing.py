"""Tests for dataset balancing utilities.

This module tests downsample_dataset() and upsample_dataset() functions
for handling imbalanced datasets.
"""

import pytest
import torch

from src.data.balancing import downsample_dataset, upsample_dataset
from src.data.datasets import BaseDataset

# =============================================================================
# Helper: Mock Dataset
# =============================================================================


class MockImbalancedDataset(BaseDataset):
    """Mock dataset with configurable class distribution for testing."""

    def __init__(self, class_counts, return_labels=True):
        """Create a mock dataset with given class counts.

        Args:
            class_counts: Dict mapping class index to number of samples
            return_labels: Whether to return labels
        """
        self._targets = []
        self._classes = [f"class_{i}" for i in sorted(class_counts.keys())]
        self.return_labels = return_labels

        for cls_idx in sorted(class_counts.keys()):
            self._targets.extend([cls_idx] * class_counts[cls_idx])

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, index):
        # Return dummy data
        image = torch.randn(3, 8, 8)
        label = self._targets[index]
        if self.return_labels:
            return image, label
        return image

    def get_classes(self):
        return list(self._classes)

    def get_class_counts(self):
        from collections import Counter

        counts = Counter(self._targets)
        return {self._classes[k]: v for k, v in counts.items()}

    @property
    def targets(self):
        return list(self._targets)

    @property
    def classes(self):
        return list(self._classes)


# =============================================================================
# Unit Tests: downsample_dataset
# =============================================================================


@pytest.mark.unit
class TestDownsampleDataset:
    """Test downsample_dataset function."""

    def test_equal_ratio_produces_equal_classes(self):
        """Test that target_ratio=1.0 produces equal class counts."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset = downsample_dataset(dataset, target_ratio=1.0, seed=42)

        # Count classes in the subset
        class_counts = {}
        for idx in subset.indices:
            label = dataset.targets[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        assert class_counts[0] == 20  # Majority downsampled to minority count
        assert class_counts[1] == 20  # Minority unchanged
        assert len(subset) == 40

    def test_half_ratio(self):
        """Test that target_ratio=0.5 produces 2:1 majority:minority ratio."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset = downsample_dataset(dataset, target_ratio=0.5, seed=42)

        class_counts = {}
        for idx in subset.indices:
            label = dataset.targets[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        # target_majority_count = minority_count / target_ratio = 20 / 0.5 = 40
        assert class_counts[0] == 40
        assert class_counts[1] == 20
        assert len(subset) == 60

    def test_reproducibility_same_seed(self):
        """Test that same seed produces same results."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset1 = downsample_dataset(dataset, target_ratio=1.0, seed=42)
        subset2 = downsample_dataset(dataset, target_ratio=1.0, seed=42)

        assert subset1.indices == subset2.indices

    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset1 = downsample_dataset(dataset, target_ratio=1.0, seed=42)
        subset2 = downsample_dataset(dataset, target_ratio=1.0, seed=123)

        # Same class distribution but different indices
        assert subset1.indices != subset2.indices

    def test_does_not_corrupt_global_random_state(self):
        """Test that local generator doesn't affect global random state."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})

        # Set global state
        torch.manual_seed(99)
        expected = torch.randn(5)

        # Reset and run downsample
        torch.manual_seed(99)
        downsample_dataset(dataset, target_ratio=1.0, seed=42)
        actual = torch.randn(5)

        assert torch.allclose(expected, actual)

    def test_empty_dataset_raises(self):
        """Test that empty dataset raises ValueError."""
        dataset = MockImbalancedDataset({})
        dataset._targets = []
        with pytest.raises(ValueError, match="no samples"):
            downsample_dataset(dataset, target_ratio=1.0, seed=42)

    def test_no_targets_attribute_raises(self):
        """Test that dataset without targets raises AttributeError."""

        class NoTargets:
            pass

        with pytest.raises(AttributeError, match="targets"):
            downsample_dataset(NoTargets(), target_ratio=1.0, seed=42)

    def test_already_balanced_dataset(self):
        """Test that already balanced dataset is unchanged in count."""
        dataset = MockImbalancedDataset({0: 50, 1: 50})
        subset = downsample_dataset(dataset, target_ratio=1.0, seed=42)

        class_counts = {}
        for idx in subset.indices:
            label = dataset.targets[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        assert class_counts[0] == 50
        assert class_counts[1] == 50


# =============================================================================
# Unit Tests: upsample_dataset
# =============================================================================


@pytest.mark.unit
class TestUpsampleDataset:
    """Test upsample_dataset function."""

    def test_equal_ratio_produces_equal_classes(self):
        """Test that target_ratio=1.0 produces equal class counts."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset = upsample_dataset(dataset, target_ratio=1.0, seed=42)

        class_counts = {}
        for idx in subset.indices:
            label = dataset.targets[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        assert class_counts[0] == 100  # Majority unchanged
        assert class_counts[1] == 100  # Minority upsampled
        assert len(subset) == 200

    def test_half_ratio(self):
        """Test that target_ratio=0.5 produces half the majority count for minority."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset = upsample_dataset(dataset, target_ratio=0.5, seed=42)

        class_counts = {}
        for idx in subset.indices:
            label = dataset.targets[idx]
            class_counts[label] = class_counts.get(label, 0) + 1

        # target_minority_count = majority_count * target_ratio = 100 * 0.5 = 50
        assert class_counts[0] == 100
        assert class_counts[1] == 50
        assert len(subset) == 150

    def test_reproducibility_same_seed(self):
        """Test that same seed produces same results."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset1 = upsample_dataset(dataset, target_ratio=1.0, seed=42)
        subset2 = upsample_dataset(dataset, target_ratio=1.0, seed=42)

        assert subset1.indices == subset2.indices

    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset1 = upsample_dataset(dataset, target_ratio=1.0, seed=42)
        subset2 = upsample_dataset(dataset, target_ratio=1.0, seed=123)

        assert subset1.indices != subset2.indices

    def test_does_not_corrupt_global_random_state(self):
        """Test that local generator doesn't affect global random state."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})

        torch.manual_seed(99)
        expected = torch.randn(5)

        torch.manual_seed(99)
        upsample_dataset(dataset, target_ratio=1.0, seed=42)
        actual = torch.randn(5)

        assert torch.allclose(expected, actual)

    def test_already_balanced_no_extra(self):
        """Test that already balanced dataset gets no extra samples."""
        dataset = MockImbalancedDataset({0: 50, 1: 50})
        subset = upsample_dataset(dataset, target_ratio=1.0, seed=42)

        assert len(subset) == 100  # No extra samples added

    def test_duplicated_indices_come_from_minority(self):
        """Test that extra indices are from the minority class."""
        dataset = MockImbalancedDataset({0: 100, 1: 20})
        subset = upsample_dataset(dataset, target_ratio=1.0, seed=42)

        # Extra indices (beyond original) should all be minority class
        original_count = len(dataset)
        extra_indices = subset.indices[original_count:]
        for idx in extra_indices:
            assert dataset.targets[idx] == 1  # Minority class

    def test_empty_dataset_raises(self):
        """Test that empty dataset raises ValueError."""
        dataset = MockImbalancedDataset({})
        dataset._targets = []
        with pytest.raises(ValueError, match="no samples"):
            upsample_dataset(dataset, target_ratio=1.0, seed=42)
