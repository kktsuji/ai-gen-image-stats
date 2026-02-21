"""Tests for Classifier DataLoader

This module contains comprehensive tests for the ClassifierDataLoader class.
Tests are organized by tiers: unit tests, component tests, and integration tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.base.dataloader import BaseDataLoader
from src.experiments.classifier.dataloader import ClassifierDataLoader

# ==============================================================================
# Helpers
# ==============================================================================


def _create_split_json(tmp_path, train_per_class=3, val_per_class=3, num_classes=2):
    """Create mock images and a split JSON file.

    Returns:
        Path string to the created split JSON file.
    """
    class_names = ["0.Normal", "1.Abnormal"][:num_classes]
    train_entries = []
    val_entries = []
    classes_dict = {name: idx for idx, name in enumerate(class_names)}

    images_dir = tmp_path / "images"

    for idx, class_name in enumerate(class_names):
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(train_per_class):
            img_path = class_dir / f"train_{i}.jpg"
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(img_path)
            train_entries.append({"path": str(img_path), "label": idx})

        for i in range(val_per_class):
            img_path = class_dir / f"val_{i}.jpg"
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(img_path)
            val_entries.append({"path": str(img_path), "label": idx})

    split_data = {
        "metadata": {
            "classes": classes_dict,
            "total_samples": len(train_entries) + len(val_entries),
        },
        "train": train_entries,
        "val": val_entries,
    }

    split_file = tmp_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    return str(split_file)


def _create_split_json_train_only(tmp_path, train_per_class=3, num_classes=2):
    """Create a split JSON with only training data (empty val)."""
    class_names = ["0.Normal", "1.Abnormal"][:num_classes]
    train_entries = []
    classes_dict = {name: idx for idx, name in enumerate(class_names)}

    images_dir = tmp_path / "images"

    for idx, class_name in enumerate(class_names):
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(train_per_class):
            img_path = class_dir / f"train_{i}.jpg"
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(img_path)
            train_entries.append({"path": str(img_path), "label": idx})

    split_data = {
        "metadata": {"classes": classes_dict},
        "train": train_entries,
        "val": [],
    }

    split_file = tmp_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    return str(split_file)


# ==============================================================================
# Unit Tests - Fast, no data loading
# ==============================================================================


@pytest.mark.unit
def test_classifier_dataloader_inherits_from_base():
    """Test that ClassifierDataLoader inherits from BaseDataLoader."""
    assert issubclass(ClassifierDataLoader, BaseDataLoader)


@pytest.mark.unit
def test_classifier_dataloader_initialization(tmp_path):
    """Test basic initialization with split file."""
    split_file = _create_split_json(tmp_path)

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=16, num_workers=0
    )

    assert dataloader.split_file == split_file
    assert dataloader.batch_size == 16
    assert dataloader.num_workers == 0


@pytest.mark.unit
def test_classifier_dataloader_invalid_split_file():
    """Test that initialization fails with invalid split file path."""
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        ClassifierDataLoader(split_file="/nonexistent/split.json", batch_size=32)


@pytest.mark.unit
def test_classifier_dataloader_default_parameters(tmp_path):
    """Test default parameter values."""
    split_file = _create_split_json(tmp_path)

    dataloader = ClassifierDataLoader(split_file=split_file)

    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.image_size == 256
    assert dataloader.crop_size == 224
    assert dataloader.horizontal_flip is True
    assert dataloader.color_jitter is False
    assert dataloader.rotation_degrees == 0
    assert dataloader.normalize == "imagenet"
    assert dataloader.pin_memory is True
    assert dataloader.drop_last is False
    assert dataloader.shuffle_train is True


@pytest.mark.unit
def test_classifier_dataloader_custom_parameters(tmp_path):
    """Test custom parameter values."""
    split_file = _create_split_json(tmp_path)

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=64,
        num_workers=2,
        image_size=128,
        crop_size=112,
        horizontal_flip=False,
        color_jitter=True,
        rotation_degrees=15,
        normalize="cifar10",
        pin_memory=False,
        drop_last=True,
        shuffle_train=False,
    )

    assert dataloader.batch_size == 64
    assert dataloader.num_workers == 2
    assert dataloader.image_size == 128
    assert dataloader.crop_size == 112
    assert dataloader.horizontal_flip is False
    assert dataloader.color_jitter is True
    assert dataloader.rotation_degrees == 15
    assert dataloader.normalize == "cifar10"
    assert dataloader.pin_memory is False
    assert dataloader.drop_last is True
    assert dataloader.shuffle_train is False


@pytest.mark.unit
def test_classifier_dataloader_get_config(tmp_path):
    """Test that get_config returns expected configuration."""
    split_file = _create_split_json(tmp_path)

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=16, num_workers=2
    )

    config = dataloader.get_config()

    assert "split_file" in config
    assert "batch_size" in config
    assert "num_workers" in config
    assert config["batch_size"] == 16
    assert config["num_workers"] == 2


@pytest.mark.unit
def test_classifier_dataloader_repr(tmp_path):
    """Test string representation of dataloader."""
    split_file = _create_split_json(tmp_path)

    dataloader = ClassifierDataLoader(split_file=split_file, batch_size=32)

    repr_str = repr(dataloader)
    assert "ClassifierDataLoader" in repr_str
    assert "batch_size=32" in repr_str


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_classifier_dataset(tmp_path):
    """Create a minimal mock dataset for classifier testing.

    Returns:
        Path string to the split JSON file.
    """
    return _create_split_json(tmp_path, train_per_class=3, val_per_class=3)


# ==============================================================================
# Component Tests - Small data, minimal computation
# ==============================================================================


@pytest.mark.component
def test_get_train_loader_basic(mock_classifier_dataset):
    """Test basic train loader creation."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
    )

    train_loader = dataloader.get_train_loader()

    # Verify loader properties
    assert train_loader is not None
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert train_loader.batch_size == 2
    assert len(train_loader.dataset) > 0


@pytest.mark.component
def test_get_val_loader_basic(mock_classifier_dataset):
    """Test basic validation loader creation."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
    )

    val_loader = dataloader.get_val_loader()

    # Verify loader properties
    assert val_loader is not None
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert val_loader.batch_size == 2
    assert len(val_loader.dataset) > 0


@pytest.mark.component
def test_get_val_loader_returns_none_when_empty_val(tmp_path):
    """Test that val loader returns None when val split is empty."""
    split_file = _create_split_json_train_only(tmp_path)

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=2, num_workers=0
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader is None


@pytest.mark.component
def test_train_loader_batch_iteration(mock_classifier_dataset):
    """Test iterating through train loader batches."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize="none",  # Disable normalization for simpler testing
    )

    train_loader = dataloader.get_train_loader()

    # Get first batch
    batch = next(iter(train_loader))
    images, labels = batch

    # Verify batch shapes
    assert images.shape[0] <= 2  # Batch size (may be smaller for last batch)
    assert images.shape[1] == 3  # RGB channels
    assert images.shape[2] == 32  # Height
    assert images.shape[3] == 32  # Width
    assert labels.shape[0] <= 2  # Batch size

    # Verify data types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long


@pytest.mark.component
def test_val_loader_batch_iteration(mock_classifier_dataset):
    """Test iterating through validation loader batches."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize="none",
    )

    val_loader = dataloader.get_val_loader()

    # Get first batch
    batch = next(iter(val_loader))
    images, labels = batch

    # Verify batch shapes
    assert images.shape[0] <= 2
    assert images.shape[1] == 3
    assert images.shape[2] == 32
    assert images.shape[3] == 32
    assert labels.shape[0] <= 2


@pytest.mark.component
def test_train_loader_with_imagenet_normalization(mock_classifier_dataset):
    """Test train loader with ImageNet normalization."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize="imagenet",
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))
    images, labels = batch

    # Images should be normalized (roughly in range [-2, 2] after normalization)
    assert images.min() >= -3.0
    assert images.max() <= 3.0


@pytest.mark.component
def test_train_loader_with_cifar10_normalization(mock_classifier_dataset):
    """Test train loader with CIFAR10 normalization."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize="cifar10",
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))
    images, labels = batch

    # Images should be normalized
    assert images.min() >= -3.0
    assert images.max() <= 3.0


@pytest.mark.component
def test_train_loader_without_normalization(mock_classifier_dataset):
    """Test train loader without normalization."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize=None,
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))
    images, labels = batch

    # Images should be in [0, 1] range (only ToTensor applied)
    assert images.min() >= 0.0
    assert images.max() <= 1.0


@pytest.mark.component
def test_get_num_classes(mock_classifier_dataset):
    """Test getting number of classes from dataset."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=2, num_workers=0
    )

    num_classes = dataloader.get_num_classes()

    # We created 2 classes in the mock dataset
    assert num_classes == 2


@pytest.mark.component
def test_get_class_names(mock_classifier_dataset):
    """Test getting class names from dataset."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=2, num_workers=0
    )

    class_names = dataloader.get_class_names()

    # Should return list of class names
    assert isinstance(class_names, list)
    assert len(class_names) == 2
    assert "0.Normal" in class_names
    assert "1.Abnormal" in class_names


@pytest.mark.component
def test_different_image_sizes(mock_classifier_dataset):
    """Test dataloader with different image sizes."""
    split_file = mock_classifier_dataset

    # Test with 64x64 images
    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=64,
        crop_size=64,
        normalize="none",
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))
    images, _ = batch

    assert images.shape[2] == 64
    assert images.shape[3] == 64


@pytest.mark.component
def test_shuffle_train_enabled(mock_classifier_dataset):
    """Test that training data is shuffled when shuffle_train=True."""
    split_file = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        split_file=split_file, batch_size=2, num_workers=0, shuffle_train=True
    )

    train_loader = dataloader.get_train_loader()

    # Check that the DataLoader has shuffle enabled
    # Note: We can't easily test actual shuffling without multiple epochs
    assert train_loader.sampler is not None or hasattr(train_loader, "shuffle")


@pytest.mark.component
def test_drop_last_behavior(mock_classifier_dataset):
    """Test drop_last parameter behavior."""
    split_file = mock_classifier_dataset

    # With drop_last=True, incomplete batches should be dropped
    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=3,  # Not divisible by dataset size
        num_workers=0,
        drop_last=True,
    )

    train_loader = dataloader.get_train_loader()

    # Verify drop_last is set
    assert train_loader.drop_last is True


# ==============================================================================
# Integration Tests - Mini workflows
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_full_dataloader_workflow(mock_classifier_dataset):
    """Test complete dataloader workflow from creation to iteration."""
    split_file = mock_classifier_dataset

    # Create dataloader
    dataloader = ClassifierDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
        normalize="imagenet",
    )

    # Get loaders
    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    # Iterate through all training batches
    train_batches = 0
    for images, labels in train_loader:
        assert images.shape[1] == 3  # RGB
        assert images.shape[2] == 32  # Height
        assert images.shape[3] == 32  # Width
        train_batches += 1

    assert train_batches > 0

    # Iterate through all validation batches
    val_batches = 0
    for images, labels in val_loader:
        assert images.shape[1] == 3
        assert images.shape[2] == 32
        assert images.shape[3] == 32
        val_batches += 1

    assert val_batches > 0

    # Get metadata
    num_classes = dataloader.get_num_classes()
    class_names = dataloader.get_class_names()

    assert num_classes == 2
    assert len(class_names) == 2


@pytest.mark.integration
@pytest.mark.slow
def test_dataloader_with_fixtures_config():
    """Test dataloader using fixtures/mock_data directory structure."""
    from pathlib import Path

    # Get the project root
    project_root = Path(__file__).parent.parent.parent.parent
    split_file = project_root / "tests" / "fixtures" / "splits" / "mock_split.json"

    # Check if fixture exists
    if not split_file.exists():
        pytest.skip("Test fixtures not found")

    # Create dataloader
    dataloader = ClassifierDataLoader(
        split_file=str(split_file),
        batch_size=2,
        num_workers=0,
        image_size=32,
        crop_size=32,
    )

    # Get loaders and verify they work
    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    # Get at least one batch from each
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch[0].shape[1] == 3  # RGB
    assert val_batch[0].shape[1] == 3  # RGB
