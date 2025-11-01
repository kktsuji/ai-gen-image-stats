"""Pytest tests for train.py components"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from train import UnderSampledImageFolder


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with sample images for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    class0_dir = os.path.join(temp_dir, "class0")
    class1_dir = os.path.join(temp_dir, "class1")
    class2_dir = os.path.join(temp_dir, "class2")

    os.makedirs(class0_dir)
    os.makedirs(class1_dir)
    os.makedirs(class2_dir)

    # Create dummy images for class0 (10 images)
    for i in range(10):
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(os.path.join(class0_dir, f"img_{i}.jpg"))

    # Create dummy images for class1 (20 images)
    for i in range(20):
        img = Image.new("RGB", (64, 64), color=(0, 255, 0))
        img.save(os.path.join(class1_dir, f"img_{i}.jpg"))

    # Create dummy images for class2 (15 images)
    for i in range(15):
        img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        img.save(os.path.join(class2_dir, f"img_{i}.jpg"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def balanced_image_dir():
    """Create a temporary directory with balanced classes"""
    temp_dir = tempfile.mkdtemp()

    # Create class directories with equal samples
    for class_idx in range(3):
        class_dir = os.path.join(temp_dir, f"class{class_idx}")
        os.makedirs(class_dir)

        # Create 10 images per class
        for i in range(10):
            img = Image.new(
                "RGB", (64, 64), color=(class_idx * 80, class_idx * 80, class_idx * 80)
            )
            img.save(os.path.join(class_dir, f"img_{i}.jpg"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestUnderSampledImageFolderInitialization:
    """Test UnderSampledImageFolder initialization"""

    def test_init_basic(self, temp_image_dir):
        """Test basic initialization"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        assert dataset is not None
        assert len(dataset.classes) == 3
        assert dataset.classes == ["class0", "class1", "class2"]

    def test_init_with_transform(self, temp_image_dir):
        """Test initialization with transform"""
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )

        dataset = UnderSampledImageFolder(temp_image_dir, transform=transform)

        assert dataset is not None
        assert dataset.transform == transform

    def test_init_stores_original_samples(self, temp_image_dir):
        """Test that original samples are stored before under-sampling"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Original should have 10 + 20 + 15 = 45 samples
        assert len(dataset.original_samples) == 45
        assert len(dataset.original_targets) == 45

    def test_init_with_custom_min_samples(self, temp_image_dir):
        """Test initialization with custom min_samples_per_class"""
        min_samples = 5
        dataset = UnderSampledImageFolder(
            temp_image_dir, min_samples_per_class=min_samples
        )

        # Should have 5 samples per class * 3 classes = 15 total
        assert len(dataset.samples) == 15
        assert len(dataset.targets) == 15

    def test_init_auto_min_samples(self, temp_image_dir):
        """Test automatic min_samples calculation (uses minimum class count)"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Should automatically use 10 (minimum of 10, 20, 15)
        # 10 samples per class * 3 classes = 30 total
        assert len(dataset.samples) == 30
        assert len(dataset.targets) == 30


class TestUnderSampledImageFolderUnderSampling:
    """Test under-sampling functionality"""

    def test_balanced_class_distribution(self, temp_image_dir):
        """Test that under-sampling creates balanced class distribution"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Count samples per class
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        # All classes should have equal samples (10 each)
        assert class_counts[0] == 10
        assert class_counts[1] == 10
        assert class_counts[2] == 10

    def test_custom_min_samples_distribution(self, temp_image_dir):
        """Test balanced distribution with custom min_samples"""
        min_samples = 7
        dataset = UnderSampledImageFolder(
            temp_image_dir, min_samples_per_class=min_samples
        )

        # Count samples per class
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        # All classes should have 7 samples
        assert all(count == min_samples for count in class_counts.values())

    def test_min_samples_exceeds_available(self, temp_image_dir):
        """Test that min_samples is capped at available samples"""
        # Request more samples than smallest class has
        min_samples = 15  # class0 only has 10
        dataset = UnderSampledImageFolder(
            temp_image_dir, min_samples_per_class=min_samples
        )

        # Should be capped at 10 (smallest class)
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        assert all(count == 10 for count in class_counts.values())

    def test_undersampling_reduces_total_samples(self, temp_image_dir):
        """Test that under-sampling reduces total number of samples"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Original: 10 + 20 + 15 = 45
        # After under-sampling: 10 + 10 + 10 = 30
        assert len(dataset.original_samples) == 45
        assert len(dataset.samples) == 30

    def test_samples_are_from_original(self, temp_image_dir):
        """Test that under-sampled samples come from original dataset"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # All sampled paths should be in original samples
        original_paths = set(path for path, _ in dataset.original_samples)
        sampled_paths = set(path for path, _ in dataset.samples)

        assert sampled_paths.issubset(original_paths)

    def test_targets_match_samples(self, temp_image_dir):
        """Test that targets list matches samples list"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Extract class indices from samples
        sample_classes = [class_idx for _, class_idx in dataset.samples]

        # Should match targets list
        assert sample_classes == dataset.targets

    def test_randomization_produces_different_samples(self, temp_image_dir):
        """Test that under-sampling is randomized"""
        dataset1 = UnderSampledImageFolder(temp_image_dir)
        dataset2 = UnderSampledImageFolder(temp_image_dir)

        # Get sample paths
        paths1 = [path for path, _ in dataset1.samples]
        paths2 = [path for path, _ in dataset2.samples]

        # Should be different due to random sampling
        # (statistically very unlikely to be identical)
        assert paths1 != paths2


class TestUnderSampledImageFolderDataLoading:
    """Test data loading functionality"""

    def test_getitem_returns_image_and_label(self, temp_image_dir):
        """Test that __getitem__ returns image and label"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        image, label = dataset[0]

        assert image is not None
        assert isinstance(label, int)
        assert 0 <= label < len(dataset.classes)

    def test_getitem_with_transform(self, temp_image_dir):
        """Test that transform is applied to images"""
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        dataset = UnderSampledImageFolder(temp_image_dir, transform=transform)

        image, label = dataset[0]

        # Should be a tensor after transform
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)

    def test_len_returns_undersampled_count(self, temp_image_dir):
        """Test that __len__ returns under-sampled count"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        # Should return 30 (10 per class * 3 classes)
        assert len(dataset) == 30

    def test_can_iterate_through_dataset(self, temp_image_dir):
        """Test that we can iterate through entire dataset"""
        transform = transforms.ToTensor()
        dataset = UnderSampledImageFolder(temp_image_dir, transform=transform)

        count = 0
        for image, label in dataset:
            count += 1
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)

        assert count == len(dataset)

    def test_all_classes_represented(self, temp_image_dir):
        """Test that all classes are represented in under-sampled dataset"""
        dataset = UnderSampledImageFolder(temp_image_dir)

        labels = set()
        for _, label in dataset:
            labels.add(label)

        # Should have all 3 classes
        assert len(labels) == 3
        assert labels == {0, 1, 2}


class TestUnderSampledImageFolderEdgeCases:
    """Test edge cases"""

    def test_single_class(self):
        """Test with single class directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create single class
            class_dir = os.path.join(temp_dir, "class0")
            os.makedirs(class_dir)

            for i in range(10):
                img = Image.new("RGB", (64, 64), color=(255, 0, 0))
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))

            dataset = UnderSampledImageFolder(temp_dir)

            assert len(dataset.classes) == 1
            assert len(dataset.samples) == 10

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_min_samples_equals_one(self, temp_image_dir):
        """Test with min_samples_per_class=1"""
        dataset = UnderSampledImageFolder(temp_image_dir, min_samples_per_class=1)

        # Should have 1 sample per class * 3 classes = 3 total
        assert len(dataset.samples) == 3

        # All 3 classes should be represented
        class_set = set(dataset.targets)
        assert len(class_set) == 3

    def test_balanced_classes_no_undersampling_needed(self, balanced_image_dir):
        """Test with already balanced classes"""
        dataset = UnderSampledImageFolder(balanced_image_dir)

        # All classes have 10 samples, so should remain 10 * 3 = 30
        assert len(dataset.samples) == 30

        # Count per class
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        assert all(count == 10 for count in class_counts.values())

    def test_empty_directory_raises_error(self):
        """Test that empty directory raises appropriate error"""
        temp_dir = tempfile.mkdtemp()

        try:
            with pytest.raises(RuntimeError):
                UnderSampledImageFolder(temp_dir)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_very_imbalanced_classes(self):
        """Test with very imbalanced class distribution"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create highly imbalanced classes
            # Class 0: 100 samples, Class 1: 5 samples
            class0_dir = os.path.join(temp_dir, "class0")
            class1_dir = os.path.join(temp_dir, "class1")
            os.makedirs(class0_dir)
            os.makedirs(class1_dir)

            for i in range(100):
                img = Image.new("RGB", (64, 64), color=(255, 0, 0))
                img.save(os.path.join(class0_dir, f"img_{i}.jpg"))

            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(0, 255, 0))
                img.save(os.path.join(class1_dir, f"img_{i}.jpg"))

            dataset = UnderSampledImageFolder(temp_dir)

            # Should under-sample to 5 per class (minimum)
            assert len(dataset.samples) == 10

            class_counts = {}
            for _, class_idx in dataset.samples:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

            assert class_counts[0] == 5
            assert class_counts[1] == 5

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestUnderSampledImageFolderIntegration:
    """Integration tests"""

    def test_compatible_with_dataloader(self, temp_image_dir):
        """Test that dataset works with PyTorch DataLoader"""
        from torch.utils.data import DataLoader

        transform = transforms.ToTensor()
        dataset = UnderSampledImageFolder(temp_image_dir, transform=transform)

        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Get first batch
        for images, labels in loader:
            assert images.shape[0] <= 4  # batch size
            assert images.shape[1:] == (3, 64, 64)  # C, H, W
            assert labels.shape[0] == images.shape[0]
            break

    def test_with_standard_transforms(self, temp_image_dir):
        """Test with common torchvision transforms"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = UnderSampledImageFolder(temp_image_dir, transform=transform)

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

    def test_reproducible_with_same_seed(self, temp_image_dir):
        """Test that results are reproducible with same random seed"""
        import random

        # Set seed and create first dataset
        random.seed(42)
        dataset1 = UnderSampledImageFolder(temp_image_dir)
        paths1 = [path for path, _ in dataset1.samples]

        # Set same seed and create second dataset
        random.seed(42)
        dataset2 = UnderSampledImageFolder(temp_image_dir)
        paths2 = [path for path, _ in dataset2.samples]

        # Should be identical with same seed
        assert paths1 == paths2

    def test_maintains_inheritance_from_image_folder(self, temp_image_dir):
        """Test that UnderSampledImageFolder maintains ImageFolder behavior"""
        from torchvision.datasets import ImageFolder

        dataset = UnderSampledImageFolder(temp_image_dir)

        assert isinstance(dataset, ImageFolder)
        assert hasattr(dataset, "classes")
        assert hasattr(dataset, "class_to_idx")
        assert hasattr(dataset, "imgs")  # alias for samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
