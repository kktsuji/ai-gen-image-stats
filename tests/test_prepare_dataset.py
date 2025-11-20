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


def split_dataset_wrapper(normal_dir, abnormal_dir, val_ratio, output_dir):
    """Wrapper function to match test expectations with actual function signature"""
    # Create output directories
    train_dir = os.path.join(output_dir, "data/train")
    val_dir = os.path.join(output_dir, "data/val")

    train_normal_dir = os.path.join(train_dir, "0.Normal")
    train_abnormal_dir = os.path.join(train_dir, "1.Abnormal")
    val_normal_dir = os.path.join(val_dir, "0.Normal")
    val_abnormal_dir = os.path.join(val_dir, "1.Abnormal")

    os.makedirs(train_normal_dir, exist_ok=True)
    os.makedirs(train_abnormal_dir, exist_ok=True)
    os.makedirs(val_normal_dir, exist_ok=True)
    os.makedirs(val_abnormal_dir, exist_ok=True)

    # Split normal images
    split_dataset_into_train_val_by_ratio(
        normal_dir, train_normal_dir, val_normal_dir, val_ratio
    )

    # Split abnormal images
    split_dataset_into_train_val_by_ratio(
        abnormal_dir, train_abnormal_dir, val_abnormal_dir, val_ratio
    )


@pytest.fixture
def temp_normal_dir():
    """Create a temporary directory with normal PNG images for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create 100 dummy PNG images
    for i in range(100):
        img = Image.new("RGB", (64, 64), color=(i * 2, i * 2, i * 2))
        img.save(os.path.join(temp_dir, f"normal_{i:03d}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_abnormal_dir():
    """Create a temporary directory with abnormal PNG images for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create 80 dummy PNG images
    for i in range(80):
        img = Image.new("RGB", (64, 64), color=(255 - i * 3, 0, i * 3))
        img.save(os.path.join(temp_dir, f"abnormal_{i:03d}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing"""
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with sample PNG images for testing (legacy)"""
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
def small_normal_dir():
    """Create a temporary directory with a small number of normal images"""
    temp_dir = tempfile.mkdtemp()

    # Create 10 dummy PNG images
    for i in range(10):
        img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
        img.save(os.path.join(temp_dir, f"normal_{i}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def small_abnormal_dir():
    """Create a temporary directory with a small number of abnormal images"""
    temp_dir = tempfile.mkdtemp()

    # Create 8 dummy PNG images
    for i in range(8):
        img = Image.new("RGB", (64, 64), color=(255 - i * 30, 0, i * 30))
        img.save(os.path.join(temp_dir, f"abnormal_{i}.png"))

    yield temp_dir

    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestSplitDatasetIntoTrainValByRatio:
    """Test split_dataset_into_train_val_by_ratio function with new signature"""

    def test_basic_split(self, temp_normal_dir, temp_abnormal_dir, temp_output_dir):
        """Test basic train/val split with 0.2 ratio"""
        val_ratio = 0.2

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        # Check that directories were created
        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        assert os.path.exists(train_dir)
        assert os.path.exists(val_dir)
        assert os.path.exists(os.path.join(train_dir, "0.Normal"))
        assert os.path.exists(os.path.join(train_dir, "1.Abnormal"))
        assert os.path.exists(os.path.join(val_dir, "0.Normal"))
        assert os.path.exists(os.path.join(val_dir, "1.Abnormal"))

        # Check counts
        normal_train = len(os.listdir(os.path.join(train_dir, "0.Normal")))
        normal_val = len(os.listdir(os.path.join(val_dir, "0.Normal")))
        abnormal_train = len(os.listdir(os.path.join(train_dir, "1.Abnormal")))
        abnormal_val = len(os.listdir(os.path.join(val_dir, "1.Abnormal")))

        # 100 normal images: 20 val, 80 train
        # 80 abnormal images: 16 val, 64 train
        assert normal_val == 20
        assert normal_train == 80
        assert abnormal_val == 16
        assert abnormal_train == 64

    def test_different_ratios(self, temp_normal_dir, temp_abnormal_dir):
        """Test with different validation ratios"""
        test_ratios = [0.1, 0.3, 0.5]

        for val_ratio in test_ratios:
            temp_out = tempfile.mkdtemp()
            try:
                split_dataset_wrapper(
                    temp_normal_dir, temp_abnormal_dir, val_ratio, temp_out
                )

                train_dir = os.path.join(temp_out, "data/train")
                val_dir = os.path.join(temp_out, "data/val")

                normal_val = len(os.listdir(os.path.join(val_dir, "0.Normal")))
                abnormal_val = len(os.listdir(os.path.join(val_dir, "1.Abnormal")))

                # Check expected counts
                assert normal_val == int(100 * val_ratio)
                assert abnormal_val == int(80 * val_ratio)
            finally:
                if os.path.exists(temp_out):
                    shutil.rmtree(temp_out)

    def test_images_copied_correctly(
        self, temp_normal_dir, temp_abnormal_dir, temp_output_dir
    ):
        """Test that images are actually copied, not moved"""
        val_ratio = 0.2

        # Count original images
        original_normal_count = len(os.listdir(temp_normal_dir))
        original_abnormal_count = len(os.listdir(temp_abnormal_dir))

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        # Original directories should still have all images
        assert len(os.listdir(temp_normal_dir)) == original_normal_count
        assert len(os.listdir(temp_abnormal_dir)) == original_abnormal_count

        # New directories should have copies
        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        normal_train = len(os.listdir(os.path.join(train_dir, "0.Normal")))
        normal_val = len(os.listdir(os.path.join(val_dir, "0.Normal")))

        assert normal_train + normal_val == original_normal_count

    def test_all_images_png(self, temp_normal_dir, temp_abnormal_dir, temp_output_dir):
        """Test that all copied files are PNG images"""
        val_ratio = 0.2

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        for subdir in ["0.Normal", "1.Abnormal"]:
            for img in os.listdir(os.path.join(train_dir, subdir)):
                assert img.endswith(".png")
            for img in os.listdir(os.path.join(val_dir, subdir)):
                assert img.endswith(".png")

    def test_ratio_zero(self, temp_normal_dir, temp_abnormal_dir, temp_output_dir):
        """Test with validation ratio of 0"""
        val_ratio = 0.0

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        # All images in train, none in val
        assert len(os.listdir(os.path.join(train_dir, "0.Normal"))) == 100
        assert len(os.listdir(os.path.join(val_dir, "0.Normal"))) == 0
        assert len(os.listdir(os.path.join(train_dir, "1.Abnormal"))) == 80
        assert len(os.listdir(os.path.join(val_dir, "1.Abnormal"))) == 0

    def test_ratio_one(self, temp_normal_dir, temp_abnormal_dir, temp_output_dir):
        """Test with validation ratio of 1"""
        val_ratio = 1.0

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        # All images in val, none in train
        assert len(os.listdir(os.path.join(train_dir, "0.Normal"))) == 0
        assert len(os.listdir(os.path.join(val_dir, "0.Normal"))) == 100
        assert len(os.listdir(os.path.join(train_dir, "1.Abnormal"))) == 0
        assert len(os.listdir(os.path.join(val_dir, "1.Abnormal"))) == 80

    def test_small_dataset(self, small_normal_dir, small_abnormal_dir, temp_output_dir):
        """Test with small dataset"""
        val_ratio = 0.2

        split_dataset_wrapper(
            small_normal_dir, small_abnormal_dir, val_ratio, temp_output_dir
        )

        train_dir = os.path.join(temp_output_dir, "data/train")
        val_dir = os.path.join(temp_output_dir, "data/val")

        # 10 normal: 2 val, 8 train
        # 8 abnormal: 1 val (int(8*0.2)=1), 7 train
        assert len(os.listdir(os.path.join(val_dir, "0.Normal"))) == 2
        assert len(os.listdir(os.path.join(train_dir, "0.Normal"))) == 8
        assert len(os.listdir(os.path.join(val_dir, "1.Abnormal"))) == 1
        assert len(os.listdir(os.path.join(train_dir, "1.Abnormal"))) == 7

    def test_reproducible_with_seed(self, temp_normal_dir, temp_abnormal_dir):
        """Test that results are reproducible with same random seed"""
        val_ratio = 0.2

        # First split with seed
        temp_out1 = tempfile.mkdtemp()
        random.seed(42)
        split_dataset_wrapper(temp_normal_dir, temp_abnormal_dir, val_ratio, temp_out1)

        train_dir1 = os.path.join(temp_out1, "data/train")
        val_dir1 = os.path.join(temp_out1, "data/val")
        train_normal1 = sorted(os.listdir(os.path.join(train_dir1, "0.Normal")))
        val_normal1 = sorted(os.listdir(os.path.join(val_dir1, "0.Normal")))

        # Second split with same seed
        temp_out2 = tempfile.mkdtemp()
        random.seed(42)
        split_dataset_wrapper(temp_normal_dir, temp_abnormal_dir, val_ratio, temp_out2)

        train_dir2 = os.path.join(temp_out2, "data/train")
        val_dir2 = os.path.join(temp_out2, "data/val")
        train_normal2 = sorted(os.listdir(os.path.join(train_dir2, "0.Normal")))
        val_normal2 = sorted(os.listdir(os.path.join(val_dir2, "0.Normal")))

        try:
            # Should be identical with same seed
            assert train_normal1 == train_normal2
            assert val_normal1 == val_normal2
        finally:
            shutil.rmtree(temp_out1)
            shutil.rmtree(temp_out2)

    def test_empty_directories(self, temp_output_dir):
        """Test with empty input directories"""
        temp_empty_normal = tempfile.mkdtemp()
        temp_empty_abnormal = tempfile.mkdtemp()

        try:
            split_dataset_wrapper(
                temp_empty_normal, temp_empty_abnormal, 0.2, temp_output_dir
            )

            train_dir = os.path.join(temp_output_dir, "data/train")
            val_dir = os.path.join(temp_output_dir, "data/val")

            # Directories should be created but empty
            assert len(os.listdir(os.path.join(train_dir, "0.Normal"))) == 0
            assert len(os.listdir(os.path.join(val_dir, "0.Normal"))) == 0
            assert len(os.listdir(os.path.join(train_dir, "1.Abnormal"))) == 0
            assert len(os.listdir(os.path.join(val_dir, "1.Abnormal"))) == 0
        finally:
            shutil.rmtree(temp_empty_normal)
            shutil.rmtree(temp_empty_abnormal)

    def test_only_png_files_copied(self, temp_output_dir):
        """Test that only PNG files are processed"""
        temp_normal = tempfile.mkdtemp()
        temp_abnormal = tempfile.mkdtemp()

        try:
            # Create PNG and non-PNG files
            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(os.path.join(temp_normal, f"normal_{i}.png"))
                img.save(
                    os.path.join(temp_normal, f"normal_{i}.jpg")
                )  # Should be ignored
                img.save(os.path.join(temp_abnormal, f"abnormal_{i}.png"))

            # Create text file
            with open(os.path.join(temp_normal, "readme.txt"), "w") as f:
                f.write("test")

            split_dataset_wrapper(temp_normal, temp_abnormal, 0.2, temp_output_dir)

            train_dir = os.path.join(temp_output_dir, "data/train")
            val_dir = os.path.join(temp_output_dir, "data/val")

            # Should only have PNG files
            normal_total = len(os.listdir(os.path.join(train_dir, "0.Normal"))) + len(
                os.listdir(os.path.join(val_dir, "0.Normal"))
            )
            abnormal_total = len(
                os.listdir(os.path.join(train_dir, "1.Abnormal"))
            ) + len(os.listdir(os.path.join(val_dir, "1.Abnormal")))

            assert normal_total == 5
            assert abnormal_total == 5
        finally:
            shutil.rmtree(temp_normal)
            shutil.rmtree(temp_abnormal)

    def test_output_directory_structure(
        self, temp_normal_dir, temp_abnormal_dir, temp_output_dir
    ):
        """Test that output directory structure is correct"""
        val_ratio = 0.2

        split_dataset_wrapper(
            temp_normal_dir, temp_abnormal_dir, val_ratio, temp_output_dir
        )

        # Check expected structure
        expected_paths = [
            os.path.join(temp_output_dir, "data"),
            os.path.join(temp_output_dir, "data/train"),
            os.path.join(temp_output_dir, "data/train/0.Normal"),
            os.path.join(temp_output_dir, "data/train/1.Abnormal"),
            os.path.join(temp_output_dir, "data/val"),
            os.path.join(temp_output_dir, "data/val/0.Normal"),
            os.path.join(temp_output_dir, "data/val/1.Abnormal"),
        ]

        for path in expected_paths:
            assert os.path.exists(path)
            assert os.path.isdir(path)


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

    def test_small_dataset(self, temp_dataset_dir):
        """Test with small dataset (10 images)"""
        # Create a small dataset temporarily
        temp_small = tempfile.mkdtemp()
        try:
            for i in range(10):
                img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
                img.save(os.path.join(temp_small, f"image_{i}.png"))

            val_count = 3
            train_images, val_images = split_dataset_into_train_val_by_count(
                temp_small, val_count
            )

            assert len(val_images) == 3
            assert len(train_images) == 7
        finally:
            if os.path.exists(temp_small):
                shutil.rmtree(temp_small)

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


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_image_by_count(self):
        """Test with single image in directory"""
        temp_dir = tempfile.mkdtemp()

        try:
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(os.path.join(temp_dir, "single.png"))

            # Test by count
            train, val = split_dataset_into_train_val_by_count(temp_dir, 1)
            assert len(val) == 1
            assert len(train) == 0

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_only_png_files_selected_by_count(self):
        """Test that only PNG files are selected"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create PNG and non-PNG files
            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(os.path.join(temp_dir, f"image_{i}.png"))
                img.save(os.path.join(temp_dir, f"image_{i}.jpg"))  # Should be ignored

            # Create text file
            with open(os.path.join(temp_dir, "readme.txt"), "w", encoding="utf-8") as f:
                f.write("test")

            train, val = split_dataset_into_train_val_by_count(temp_dir, 2)

            # Should only have 5 total images (PNG only)
            assert len(train) + len(val) == 5

            # All should be PNG
            for img_path in train + val:
                assert img_path.endswith(".png")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_nonexistent_directory_by_count(self):
        """Test with non-existent directory"""
        nonexistent_dir = "/tmp/this_directory_should_not_exist_12345"

        # Should not raise error, just return empty lists
        train, val = split_dataset_into_train_val_by_count(nonexistent_dir, 10)
        assert len(train) == 0
        assert len(val) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
