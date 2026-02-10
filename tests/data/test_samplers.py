"""
Tests for custom samplers.

This module tests the sampler utilities for handling imbalanced datasets.
"""

import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.samplers import (
    compute_class_weights,
    create_balanced_sampler,
    create_weighted_sampler,
    get_sampler_from_dataset,
)

# =============================================================================
# Unit Tests (Fast - CPU Only)
# =============================================================================


@pytest.mark.unit
class TestCreateWeightedSampler:
    """Test create_weighted_sampler function."""

    def test_basic_creation(self):
        """Test basic sampler creation with default parameters."""
        targets = [0, 0, 0, 1, 1, 2]
        sampler = create_weighted_sampler(targets)

        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(targets)
        assert sampler.replacement is True

    def test_inverse_frequency_weights(self):
        """Test that inverse frequency weighting is correct."""
        targets = [0, 0, 0, 1, 1, 2]  # Class 0: 3, Class 1: 2, Class 2: 1
        sampler = create_weighted_sampler(targets)

        # Expected weights: total / class_count
        # Class 0: 6/3 = 2.0, Class 1: 6/2 = 3.0, Class 2: 6/1 = 6.0
        expected_weights = [2.0, 2.0, 2.0, 3.0, 3.0, 6.0]

        # Convert tensor to list for comparison
        actual_weights = sampler.weights.tolist()

        assert actual_weights == pytest.approx(expected_weights)

    def test_custom_class_weights(self):
        """Test sampler creation with custom class weights."""
        targets = [0, 0, 1, 1]
        custom_weights = {0: 1.0, 1: 3.0}

        sampler = create_weighted_sampler(targets, class_weights=custom_weights)

        # Expected sample weights based on custom class weights
        expected_weights = [1.0, 1.0, 3.0, 3.0]
        actual_weights = sampler.weights.tolist()

        assert actual_weights == pytest.approx(expected_weights)

    def test_without_replacement(self):
        """Test sampler creation without replacement."""
        targets = [0, 0, 1, 1]
        sampler = create_weighted_sampler(targets, replacement=False)

        assert sampler.replacement is False

    def test_custom_num_samples(self):
        """Test sampler with custom number of samples."""
        targets = [0, 0, 1, 1]
        num_samples = 10

        sampler = create_weighted_sampler(targets, num_samples=num_samples)

        assert sampler.num_samples == num_samples

    def test_empty_targets_raises_error(self):
        """Test that empty targets list raises ValueError."""
        with pytest.raises(ValueError, match="targets list cannot be empty"):
            create_weighted_sampler([])

    def test_single_class(self):
        """Test sampler with single class dataset."""
        targets = [0, 0, 0, 0]
        sampler = create_weighted_sampler(targets)

        # All samples should have equal weight
        expected_weights = [1.0, 1.0, 1.0, 1.0]
        actual_weights = sampler.weights.tolist()

        assert actual_weights == pytest.approx(expected_weights)


@pytest.mark.unit
class TestCreateBalancedSampler:
    """Test create_balanced_sampler function."""

    def test_basic_creation(self):
        """Test basic balanced sampler creation."""
        targets = [0, 0, 0, 1, 1, 2]
        sampler = create_balanced_sampler(targets)

        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(targets)
        assert sampler.replacement is True

    def test_equal_probability_per_class(self):
        """Test that all classes have equal probability."""
        targets = [0, 0, 0, 1, 1, 2]  # Imbalanced: 3, 2, 1
        sampler = create_balanced_sampler(targets)

        # Get weights
        weights = sampler.weights.tolist()

        # Calculate probability of each class being sampled
        # P(class) = sum(weights for samples of that class) / sum(all weights)
        total_weight = sum(weights)

        # Class 0: 3 samples, Class 1: 2 samples, Class 2: 1 sample
        prob_class_0 = sum(weights[0:3]) / total_weight
        prob_class_1 = sum(weights[3:5]) / total_weight
        prob_class_2 = weights[5] / total_weight

        # All probabilities should be equal (1/3 for 3 classes)
        expected_prob = 1.0 / 3.0
        assert prob_class_0 == pytest.approx(expected_prob, rel=1e-5)
        assert prob_class_1 == pytest.approx(expected_prob, rel=1e-5)
        assert prob_class_2 == pytest.approx(expected_prob, rel=1e-5)

    def test_custom_num_samples(self):
        """Test balanced sampler with custom number of samples."""
        targets = [0, 0, 1]
        num_samples = 100

        sampler = create_balanced_sampler(targets, num_samples=num_samples)

        assert sampler.num_samples == num_samples


@pytest.mark.unit
class TestComputeClassWeights:
    """Test compute_class_weights function."""

    def test_inverse_freq_mode(self):
        """Test inverse frequency weight computation."""
        targets = [0, 0, 0, 1, 1, 2]
        weights = compute_class_weights(targets, weight_mode="inverse_freq")

        # Expected: total / class_count
        # Class 0: 6/3 = 2.0, Class 1: 6/2 = 3.0, Class 2: 6/1 = 6.0
        assert weights[0] == pytest.approx(2.0)
        assert weights[1] == pytest.approx(3.0)
        assert weights[2] == pytest.approx(6.0)

    def test_balanced_mode(self):
        """Test balanced weight computation."""
        targets = [0, 0, 0, 1, 1, 2]
        weights = compute_class_weights(targets, weight_mode="balanced")

        # Expected: total / (num_classes * class_count)
        # num_classes = 3
        # Class 0: 6/(3*3) = 2/3, Class 1: 6/(3*2) = 1, Class 2: 6/(3*1) = 2
        assert weights[0] == pytest.approx(2.0 / 3.0)
        assert weights[1] == pytest.approx(1.0)
        assert weights[2] == pytest.approx(2.0)

    def test_effective_num_mode(self):
        """Test effective number weighting computation."""
        targets = [0, 0, 0, 1, 1, 2]
        weights = compute_class_weights(targets, weight_mode="effective_num")

        # Should return weights based on effective number formula
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert all(w > 0 for w in weights.values())

        # Minority classes should have higher weights
        assert weights[2] > weights[1] > weights[0]

    def test_empty_targets_raises_error(self):
        """Test that empty targets raises ValueError."""
        with pytest.raises(ValueError, match="targets list cannot be empty"):
            compute_class_weights([])

    def test_unknown_mode_raises_error(self):
        """Test that unknown weight mode raises ValueError."""
        targets = [0, 1, 2]
        with pytest.raises(ValueError, match="Unknown weight_mode"):
            compute_class_weights(targets, weight_mode="unknown")


@pytest.mark.unit
class TestGetSamplerFromDataset:
    """Test get_sampler_from_dataset factory function."""

    def test_weighted_sampler_creation(self):
        """Test creating weighted sampler from dataset."""

        # Create a mock dataset with targets attribute
        class MockDataset:
            def __init__(self):
                self.targets = [0, 0, 0, 1, 1, 2]

        dataset = MockDataset()
        sampler = get_sampler_from_dataset(dataset, sampler_type="weighted")

        assert isinstance(sampler, WeightedRandomSampler)

    def test_balanced_sampler_creation(self):
        """Test creating balanced sampler from dataset."""

        class MockDataset:
            def __init__(self):
                self.targets = [0, 0, 0, 1, 1, 2]

        dataset = MockDataset()
        sampler = get_sampler_from_dataset(dataset, sampler_type="balanced")

        assert isinstance(sampler, WeightedRandomSampler)

    def test_none_sampler_returns_none(self):
        """Test that None sampler type returns None."""

        class MockDataset:
            def __init__(self):
                self.targets = [0, 1, 2]

        dataset = MockDataset()
        sampler = get_sampler_from_dataset(dataset, sampler_type=None)

        assert sampler is None

    def test_none_string_returns_none(self):
        """Test that 'none' string returns None."""

        class MockDataset:
            def __init__(self):
                self.targets = [0, 1, 2]

        dataset = MockDataset()
        sampler = get_sampler_from_dataset(dataset, sampler_type="none")

        assert sampler is None

    def test_missing_targets_raises_error(self):
        """Test that dataset without targets raises AttributeError."""

        class MockDataset:
            pass

        dataset = MockDataset()

        with pytest.raises(AttributeError, match="does not have 'targets'"):
            get_sampler_from_dataset(dataset, sampler_type="weighted")

    def test_unknown_sampler_type_raises_error(self):
        """Test that unknown sampler type raises ValueError."""

        class MockDataset:
            def __init__(self):
                self.targets = [0, 1, 2]

        dataset = MockDataset()

        with pytest.raises(ValueError, match="Unknown sampler_type"):
            get_sampler_from_dataset(dataset, sampler_type="unknown")

    def test_kwargs_passed_to_sampler(self):
        """Test that kwargs are passed to sampler creation."""

        class MockDataset:
            def __init__(self):
                self.targets = [0, 0, 1, 1]

        dataset = MockDataset()
        sampler = get_sampler_from_dataset(
            dataset, sampler_type="weighted", num_samples=100
        )

        assert sampler.num_samples == 100


# =============================================================================
# Component Tests (Medium - CPU with small data)
# =============================================================================


@pytest.mark.component
class TestSamplerWithDataLoader:
    """Test samplers work correctly with DataLoader."""

    def test_weighted_sampler_with_dataloader(self):
        """Test that weighted sampler works with DataLoader."""

        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(10, 3, 8, 8)
                self.targets = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        dataset = SimpleDataset()
        sampler = create_weighted_sampler(dataset.targets)

        # Create DataLoader (should not raise errors)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)

        # Test iteration
        batch_count = 0
        for batch_data, batch_targets in loader:
            batch_count += 1
            assert batch_data.shape[0] <= 2  # Batch size
            assert batch_targets.shape[0] <= 2

        assert batch_count > 0

    def test_sampler_draws_minority_class_more(self):
        """Test that balanced sampler draws minority class more frequently."""

        # Create imbalanced dataset
        class ImbalancedDataset(torch.utils.data.Dataset):
            def __init__(self):
                # 90 samples of class 0, 10 samples of class 1
                self.targets = [0] * 90 + [1] * 10

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return idx, self.targets[idx]

        dataset = ImbalancedDataset()
        sampler = create_balanced_sampler(dataset.targets, num_samples=1000)

        loader = DataLoader(dataset, batch_size=20, sampler=sampler)

        # Count how many times each class appears
        class_counts = {0: 0, 1: 0}
        for _, batch_targets in loader:
            for target in batch_targets:
                class_counts[target.item()] += 1

        # With balanced sampling, both classes should appear ~equal times
        # Allow some variance (40-60% range for each class)
        total = sum(class_counts.values())
        class_0_ratio = class_counts[0] / total
        class_1_ratio = class_counts[1] / total

        assert 0.4 <= class_0_ratio <= 0.6
        assert 0.4 <= class_1_ratio <= 0.6

    def test_without_sampler_respects_class_distribution(self):
        """Test that without sampler, natural distribution is preserved."""

        # Create imbalanced dataset
        class ImbalancedDataset(torch.utils.data.Dataset):
            def __init__(self):
                # 90 samples of class 0, 10 samples of class 1
                self.targets = [0] * 90 + [1] * 10

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return idx, self.targets[idx]

        dataset = ImbalancedDataset()

        # No sampler, just shuffle
        loader = DataLoader(dataset, batch_size=100, shuffle=True)

        # Get all samples
        _, all_targets = next(iter(loader))

        # Count classes
        class_counts = {0: 0, 1: 0}
        for target in all_targets:
            class_counts[target.item()] += 1

        # Should match original distribution (90/10)
        assert class_counts[0] == 90
        assert class_counts[1] == 10


# =============================================================================
# Integration Tests (Slow - with real dataset structure)
# =============================================================================


@pytest.mark.integration
class TestSamplerWithRealDataset:
    """Test samplers with ImageFolderDataset."""

    def test_sampler_with_image_folder_dataset(self, tmp_path):
        """Test creating sampler from ImageFolderDataset."""
        from src.data.datasets import ImageFolderDataset
        from src.data.transforms import get_base_transforms

        # Create a small mock dataset structure
        class_0_dir = tmp_path / "class0"
        class_1_dir = tmp_path / "class1"
        class_0_dir.mkdir()
        class_1_dir.mkdir()

        # Create dummy images (3 for class0, 1 for class1 - imbalanced)
        from PIL import Image

        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 50, i * 50, i * 50))
            img.save(class_0_dir / f"img_{i}.png")

        img = Image.new("RGB", (32, 32), color=(200, 200, 200))
        img.save(class_1_dir / "img_0.png")

        # Create dataset
        transforms = get_base_transforms(
            image_size=32, crop_size=32, resize_mode="resize"
        )
        dataset = ImageFolderDataset(root=str(tmp_path), transform=transforms)

        # Create sampler from dataset
        sampler = get_sampler_from_dataset(dataset, sampler_type="balanced")

        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(dataset)

        # Test with DataLoader
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)

        # Should be able to iterate
        batch_count = 0
        for batch_data, _ in loader:
            batch_count += 1
            assert batch_data.shape[0] <= 2

        assert batch_count == 2  # 4 samples, batch size 2

    def test_get_sample_weights_integration(self, tmp_path):
        """Test dataset.get_sample_weights() method with samplers."""
        from src.data.datasets import ImageFolderDataset
        from src.data.transforms import get_base_transforms

        # Create mock dataset
        class_0_dir = tmp_path / "class0"
        class_1_dir = tmp_path / "class1"
        class_0_dir.mkdir()
        class_1_dir.mkdir()

        from PIL import Image

        for i in range(5):
            img = Image.new("RGB", (32, 32), color=(100, 100, 100))
            img.save(class_0_dir / f"img_{i}.png")

        for i in range(2):
            img = Image.new("RGB", (32, 32), color=(200, 200, 200))
            img.save(class_1_dir / f"img_{i}.png")

        # Create dataset
        transforms = get_base_transforms(
            image_size=32, crop_size=32, resize_mode="resize"
        )
        dataset = ImageFolderDataset(root=str(tmp_path), transform=transforms)

        # Get sample weights from dataset
        sample_weights = dataset.get_sample_weights()

        # Create sampler using these weights
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(dataset), replacement=True
        )

        # Test with DataLoader
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        # Should work without errors
        for batch_data, _ in loader:
            assert batch_data.shape[0] <= 4
            break  # Just test one batch
