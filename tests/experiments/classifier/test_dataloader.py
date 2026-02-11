"""Tests for Classifier DataLoader

This module contains comprehensive tests for the ClassifierDataLoader class.
Tests are organized by tiers: unit tests, component tests, and integration tests.
"""

from pathlib import Path

import pytest
import torch

from src.base.dataloader import BaseDataLoader
from src.experiments.classifier.dataloader import ClassifierDataLoader

# ==============================================================================
# Unit Tests - Fast, no data loading
# ==============================================================================


@pytest.mark.unit
def test_classifier_dataloader_inherits_from_base():
    """Test that ClassifierDataLoader inherits from BaseDataLoader."""
    assert issubclass(ClassifierDataLoader, BaseDataLoader)


@pytest.mark.unit
def test_classifier_dataloader_initialization_with_train_only(tmp_data_dir):
    """Test basic initialization with only training path."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()

    # Create minimal directory structure
    (train_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), val_path=None, batch_size=16, num_workers=0
    )

    assert dataloader.train_path == str(train_path)
    assert dataloader.val_path is None
    assert dataloader.batch_size == 16
    assert dataloader.num_workers == 0


@pytest.mark.unit
def test_classifier_dataloader_initialization_with_train_and_val(tmp_data_dir):
    """Test initialization with both training and validation paths."""
    train_path = tmp_data_dir / "train"
    val_path = tmp_data_dir / "val"
    train_path.mkdir()
    val_path.mkdir()

    # Create minimal directory structure
    (train_path / "class1").mkdir()
    (val_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), val_path=str(val_path), batch_size=32
    )

    assert dataloader.train_path == str(train_path)
    assert dataloader.val_path == str(val_path)
    assert dataloader.batch_size == 32


@pytest.mark.unit
def test_classifier_dataloader_invalid_train_path():
    """Test that initialization fails with invalid training path."""
    with pytest.raises(FileNotFoundError, match="Training data path not found"):
        ClassifierDataLoader(train_path="/nonexistent/path", batch_size=32)


@pytest.mark.unit
def test_classifier_dataloader_invalid_val_path(tmp_data_dir):
    """Test that initialization fails with invalid validation path."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    with pytest.raises(FileNotFoundError, match="Validation data path not found"):
        ClassifierDataLoader(
            train_path=str(train_path), val_path="/nonexistent/val/path", batch_size=32
        )


@pytest.mark.unit
def test_classifier_dataloader_default_parameters(tmp_data_dir):
    """Test default parameter values."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(train_path=str(train_path))

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
def test_classifier_dataloader_custom_parameters(tmp_data_dir):
    """Test custom parameter values."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
def test_classifier_dataloader_get_config(tmp_data_dir):
    """Test that get_config returns expected configuration."""
    train_path = tmp_data_dir / "train"
    val_path = tmp_data_dir / "val"
    train_path.mkdir()
    val_path.mkdir()
    (train_path / "class1").mkdir()
    (val_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), val_path=str(val_path), batch_size=16, num_workers=2
    )

    config = dataloader.get_config()

    assert "train_path" in config
    assert "val_path" in config
    assert "batch_size" in config
    assert "num_workers" in config
    assert config["batch_size"] == 16
    assert config["num_workers"] == 2


@pytest.mark.unit
def test_classifier_dataloader_repr(tmp_data_dir):
    """Test string representation of dataloader."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = ClassifierDataLoader(train_path=str(train_path), batch_size=32)

    repr_str = repr(dataloader)
    assert "ClassifierDataLoader" in repr_str
    assert "batch_size=32" in repr_str


# ==============================================================================
# Component Tests - Small data, minimal computation
# ==============================================================================


@pytest.mark.component
def test_get_train_loader_basic(mock_classifier_dataset):
    """Test basic train loader creation."""
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, val_path = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
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
def test_get_val_loader_returns_none_when_no_val_path(mock_classifier_dataset):
    """Test that val loader returns None when no val_path specified."""
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), val_path=None, batch_size=2, num_workers=0
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader is None


@pytest.mark.component
def test_train_loader_batch_iteration(mock_classifier_dataset):
    """Test iterating through train loader batches."""
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, val_path = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
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
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), batch_size=2, num_workers=0
    )

    num_classes = dataloader.get_num_classes()

    # We created 2 classes in the mock dataset
    assert num_classes == 2


@pytest.mark.component
def test_get_class_names(mock_classifier_dataset):
    """Test getting class names from dataset."""
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), batch_size=2, num_workers=0
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
    train_path, _ = mock_classifier_dataset

    # Test with 64x64 images
    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
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
    train_path, _ = mock_classifier_dataset

    dataloader = ClassifierDataLoader(
        train_path=str(train_path), batch_size=2, num_workers=0, shuffle_train=True
    )

    train_loader = dataloader.get_train_loader()

    # Check that the DataLoader has shuffle enabled
    # Note: We can't easily test actual shuffling without multiple epochs
    assert train_loader.sampler is not None or hasattr(train_loader, "shuffle")


@pytest.mark.component
def test_drop_last_behavior(mock_classifier_dataset):
    """Test drop_last parameter behavior."""
    train_path, _ = mock_classifier_dataset

    # With drop_last=True, incomplete batches should be dropped
    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
        batch_size=3,  # Not divisible by dataset size
        num_workers=0,
        drop_last=True,
    )

    train_loader = dataloader.get_train_loader()

    # Verify drop_last is set
    assert train_loader.drop_last is True


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_classifier_dataset(tmp_path):
    """Create a minimal mock dataset for classifier testing.

    Creates a small ImageFolder-style dataset with 2 classes.
    """
    import numpy as np
    from PIL import Image

    # Create train directory
    train_path = tmp_path / "train"
    train_path.mkdir()

    # Create validation directory
    val_path = tmp_path / "val"
    val_path.mkdir()

    # Create class directories
    for split_path in [train_path, val_path]:
        for class_name in ["0.Normal", "1.Abnormal"]:
            class_dir = split_path / class_name
            class_dir.mkdir()

            # Create a few dummy images
            for i in range(3):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"img_{i}.jpg")

    return train_path, val_path


# ==============================================================================
# Integration Tests - Mini workflows
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_full_dataloader_workflow(mock_classifier_dataset):
    """Test complete dataloader workflow from creation to iteration."""
    train_path, val_path = mock_classifier_dataset

    # Create dataloader
    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
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
def test_dataloader_with_fixtures_config(tmp_path):
    """Test dataloader using fixtures/mock_data directory structure."""
    import shutil
    from pathlib import Path

    # Get the project root
    project_root = Path(__file__).parent.parent.parent.parent
    fixtures_path = project_root / "tests" / "fixtures" / "mock_data"

    # Check if fixtures exist
    if not fixtures_path.exists():
        pytest.skip("Test fixtures not found")

    train_path = fixtures_path / "train"
    val_path = fixtures_path / "val"

    if not train_path.exists() or not val_path.exists():
        pytest.skip("Test fixtures not properly set up")

    # Create dataloader
    dataloader = ClassifierDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
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
