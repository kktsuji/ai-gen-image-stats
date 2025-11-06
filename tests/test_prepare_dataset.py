"""Pytest tests for prepare_dataset.py components"""

import os
import random
import shutil
import tempfile

import pytest
from PIL import Image

from prepare_dataset import (
    split_dataset_into_train_val_by_count,
    split_dataset_into_train_val_by_ratio,
)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with sample PNG images for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create 100 dummy PNG images
    for i in range(100):
        img = Image.new("RGB", (64, 64), color=(i * 2, i * 2, i * 2))
        img.save(os.path.join(temp_dir, f"image_{i:03d}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def small_dataset_dir():
    """Create a temporary directory with a small number of images"""
    temp_dir = tempfile.mkdtemp()

    # Create 10 dummy PNG images
    for i in range(10):
        img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
        img.save(os.path.join(temp_dir, f"image_{i}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestSplitDatasetByRatio:
    """Test split_dataset_into_train_val_by_ratio function"""

    def test_basic_split(self, temp_dataset_dir):
        """Test basic train/val split with 0.2 ratio"""
        val_ratio = 0.2
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        # Should have 20 val images and 80 train images
        assert len(val_images) == 20
        assert len(train_images) == 80
        assert len(train_images) + len(val_images) == 100

    def test_different_ratios(self, temp_dataset_dir):
        """Test with different validation ratios"""
        test_ratios = [0.1, 0.2, 0.3, 0.5]

        for val_ratio in test_ratios:
            train_images, val_images = split_dataset_into_train_val_by_ratio(
                temp_dataset_dir, val_ratio
            )

            expected_val_count = int(100 * val_ratio)
            assert len(val_images) == expected_val_count
            assert len(train_images) == 100 - expected_val_count

    def test_no_overlap(self, temp_dataset_dir):
        """Test that train and val sets don't overlap"""
        val_ratio = 0.2
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        train_set = set(train_images)
        val_set = set(val_images)

        # No overlap between train and val
        assert len(train_set.intersection(val_set)) == 0

    def test_all_images_included(self, temp_dataset_dir):
        """Test that all images are included in either train or val"""
        val_ratio = 0.2
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        all_split_images = set(train_images + val_images)
        all_original_images = set(
            [os.path.join(temp_dataset_dir, f"image_{i:03d}.png") for i in range(100)]
        )

        assert all_split_images == all_original_images

    def test_ratio_zero(self, temp_dataset_dir):
        """Test with validation ratio of 0"""
        val_ratio = 0.0
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        assert len(val_images) == 0
        assert len(train_images) == 100

    def test_ratio_one(self, temp_dataset_dir):
        """Test with validation ratio of 1"""
        val_ratio = 1.0
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        assert len(val_images) == 100
        assert len(train_images) == 0

    def test_small_dataset(self, small_dataset_dir):
        """Test with small dataset (10 images)"""
        val_ratio = 0.2
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            small_dataset_dir, val_ratio
        )

        # 10 * 0.2 = 2 val images
        assert len(val_images) == 2
        assert len(train_images) == 8

    def test_randomization(self, temp_dataset_dir):
        """Test that splitting is randomized"""
        val_ratio = 0.2

        # Get two different splits
        train1, val1 = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )
        train2, val2 = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        # Should be different due to random shuffling
        # (statistically very unlikely to be identical)
        assert train1 != train2 or val1 != val2

    def test_reproducible_with_seed(self, temp_dataset_dir):
        """Test that results are reproducible with same random seed"""
        val_ratio = 0.2

        # Set seed and get first split
        random.seed(42)
        train1, val1 = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        # Set same seed and get second split
        random.seed(42)
        train2, val2 = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        # Should be identical with same seed
        assert train1 == train2
        assert val1 == val2

    def test_returns_full_paths(self, temp_dataset_dir):
        """Test that returned paths are valid file paths"""
        val_ratio = 0.2
        train_images, val_images = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        # All paths should exist and be PNG files
        for img_path in train_images + val_images:
            assert os.path.exists(img_path)
            assert img_path.endswith(".png")

    def test_empty_directory(self):
        """Test with empty directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            val_ratio = 0.2
            train_images, val_images = split_dataset_into_train_val_by_ratio(
                temp_dir, val_ratio
            )

            # Should return empty lists
            assert len(train_images) == 0
            assert len(val_images) == 0

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_fractional_count_rounding(self):
        """Test proper rounding when ratio results in fractional count"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create 15 images (15 * 0.2 = 3.0, but 15 * 0.3 = 4.5)
            for i in range(15):
                img = Image.new("RGB", (64, 64), color=(i * 16, i * 16, i * 16))
                img.save(os.path.join(temp_dir, f"image_{i}.png"))

            val_ratio = 0.3
            train_images, val_images = split_dataset_into_train_val_by_ratio(
                temp_dir, val_ratio
            )

            # int(15 * 0.3) = int(4.5) = 4
            assert len(val_images) == 4
            assert len(train_images) == 11

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestSplitDatasetByCount:
    """Test split_dataset_into_train_val_by_count function"""

    def test_basic_split(self, temp_dataset_dir):
        """Test basic train/val split with specific count"""
        val_count = 20
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        assert len(val_images) == 20
        assert len(train_images) == 80
        assert len(train_images) + len(val_images) == 100

    def test_different_counts(self, temp_dataset_dir):
        """Test with different validation counts"""
        test_counts = [10, 25, 50, 75]

        for val_count in test_counts:
            train_images, val_images = split_dataset_into_train_val_by_count(
                temp_dataset_dir, val_count
            )

            assert len(val_images) == val_count
            assert len(train_images) == 100 - val_count

    def test_no_overlap(self, temp_dataset_dir):
        """Test that train and val sets don't overlap"""
        val_count = 30
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        train_set = set(train_images)
        val_set = set(val_images)

        # No overlap between train and val
        assert len(train_set.intersection(val_set)) == 0

    def test_all_images_included(self, temp_dataset_dir):
        """Test that all images are included in either train or val"""
        val_count = 40
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        all_split_images = set(train_images + val_images)
        all_original_images = set(
            [os.path.join(temp_dataset_dir, f"image_{i:03d}.png") for i in range(100)]
        )

        assert all_split_images == all_original_images

    def test_count_zero(self, temp_dataset_dir):
        """Test with validation count of 0"""
        val_count = 0
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        assert len(val_images) == 0
        assert len(train_images) == 100

    def test_count_equals_total(self, temp_dataset_dir):
        """Test with validation count equal to total images"""
        val_count = 100
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        assert len(val_images) == 100
        assert len(train_images) == 0

    def test_count_exceeds_total(self, temp_dataset_dir):
        """Test with validation count exceeding total images"""
        val_count = 150
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Should take all available images for val
        assert len(val_images) == 100
        assert len(train_images) == 0

    def test_small_dataset(self, small_dataset_dir):
        """Test with small dataset (10 images)"""
        val_count = 3
        train_images, val_images = split_dataset_into_train_val_by_count(
            small_dataset_dir, val_count
        )

        assert len(val_images) == 3
        assert len(train_images) == 7

    def test_randomization(self, temp_dataset_dir):
        """Test that splitting is randomized"""
        val_count = 20

        # Get two different splits
        train1, val1 = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )
        train2, val2 = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Should be different due to random shuffling
        # (statistically very unlikely to be identical)
        assert train1 != train2 or val1 != val2

    def test_reproducible_with_seed(self, temp_dataset_dir):
        """Test that results are reproducible with same random seed"""
        val_count = 25

        # Set seed and get first split
        random.seed(42)
        train1, val1 = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Set same seed and get second split
        random.seed(42)
        train2, val2 = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Should be identical with same seed
        assert train1 == train2
        assert val1 == val2

    def test_returns_full_paths(self, temp_dataset_dir):
        """Test that returned paths are valid file paths"""
        val_count = 15
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # All paths should exist and be PNG files
        for img_path in train_images + val_images:
            assert os.path.exists(img_path)
            assert img_path.endswith(".png")

    def test_empty_directory(self):
        """Test with empty directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            val_count = 5
            train_images, val_images = split_dataset_into_train_val_by_count(
                temp_dir, val_count
            )

            # Should return empty lists
            assert len(train_images) == 0
            assert len(val_images) == 0

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_count_one(self, temp_dataset_dir):
        """Test with validation count of 1"""
        val_count = 1
        train_images, val_images = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        assert len(val_images) == 1
        assert len(train_images) == 99


class TestComparisonBetweenMethods:
    """Compare behavior between ratio and count methods"""

    def test_equivalent_results_with_matching_ratio_and_count(self, temp_dataset_dir):
        """Test that ratio and count methods give similar results when equivalent"""
        # 20% of 100 = 20
        val_ratio = 0.2
        val_count = 20

        random.seed(42)
        train_by_ratio, val_by_ratio = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        random.seed(42)
        train_by_count, val_by_count = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Should have same counts
        assert len(val_by_ratio) == len(val_by_count)
        assert len(train_by_ratio) == len(train_by_count)

        # With same seed, should have identical splits
        assert val_by_ratio == val_by_count
        assert train_by_ratio == train_by_count

    def test_both_methods_shuffle(self, temp_dataset_dir):
        """Test that both methods apply shuffling"""
        val_ratio = 0.2
        val_count = 20

        # Get original order
        import glob

        all_images = sorted(glob.glob(os.path.join(temp_dataset_dir, "*.png")))
        first_20_original = all_images[:20]

        # Split with both methods
        random.seed(99)
        _, val_by_ratio = split_dataset_into_train_val_by_ratio(
            temp_dataset_dir, val_ratio
        )

        random.seed(99)
        _, val_by_count = split_dataset_into_train_val_by_count(
            temp_dataset_dir, val_count
        )

        # Both should be identical when using same seed
        assert val_by_ratio == val_by_count

        # And should be shuffled (not just first 20 in order)
        # Statistically very unlikely to be in original order
        assert val_by_ratio != first_20_original


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_image(self):
        """Test with single image in directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(os.path.join(temp_dir, "single.png"))

            # Test by ratio
            train, val = split_dataset_into_train_val_by_ratio(temp_dir, 0.5)
            assert len(train) + len(val) == 1

            # Test by count
            train, val = split_dataset_into_train_val_by_count(temp_dir, 1)
            assert len(val) == 1
            assert len(train) == 0

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_only_png_files_selected(self):
        """Test that only PNG files are selected"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create PNG and non-PNG files
            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(os.path.join(temp_dir, f"image_{i}.png"))
                img.save(os.path.join(temp_dir, f"image_{i}.jpg"))  # Should be ignored

            # Create text file
            with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
                f.write("test")

            train, val = split_dataset_into_train_val_by_ratio(temp_dir, 0.2)

            # Should only have 5 total images (PNG only)
            assert len(train) + len(val) == 5

            # All should be PNG
            for img_path in train + val:
                assert img_path.endswith(".png")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_nonexistent_directory(self):
        """Test with non-existent directory"""
        nonexistent_dir = "/tmp/this_directory_should_not_exist_12345"

        # Should not raise error, just return empty lists
        train, val = split_dataset_into_train_val_by_ratio(nonexistent_dir, 0.2)
        assert len(train) == 0
        assert len(val) == 0

        train, val = split_dataset_into_train_val_by_count(nonexistent_dir, 10)
        assert len(train) == 0
        assert len(val) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
