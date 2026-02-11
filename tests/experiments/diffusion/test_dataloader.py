"""Tests for Diffusion DataLoader

This module contains comprehensive tests for the DiffusionDataLoader class.
Tests are organized by tiers: unit tests, component tests, and integration tests.
"""

from pathlib import Path

import pytest
import torch

from src.base.dataloader import BaseDataLoader
from src.experiments.diffusion.dataloader import DiffusionDataLoader

# ==============================================================================
# Unit Tests - Fast, no data loading
# ==============================================================================


@pytest.mark.unit
def test_diffusion_dataloader_inherits_from_base():
    """Test that DiffusionDataLoader inherits from BaseDataLoader."""
    assert issubclass(DiffusionDataLoader, BaseDataLoader)


@pytest.mark.unit
def test_diffusion_dataloader_initialization_with_train_only(tmp_data_dir):
    """Test basic initialization with only training path."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()

    # Create minimal directory structure
    (train_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(
        train_path=str(train_path), val_path=None, batch_size=16, num_workers=0
    )

    assert dataloader.train_path == str(train_path)
    assert dataloader.val_path is None
    assert dataloader.batch_size == 16
    assert dataloader.num_workers == 0


@pytest.mark.unit
def test_diffusion_dataloader_initialization_with_train_and_val(tmp_data_dir):
    """Test initialization with both training and validation paths."""
    train_path = tmp_data_dir / "train"
    val_path = tmp_data_dir / "val"
    train_path.mkdir()
    val_path.mkdir()

    # Create minimal directory structure
    (train_path / "class1").mkdir()
    (val_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(
        train_path=str(train_path), val_path=str(val_path), batch_size=32
    )

    assert dataloader.train_path == str(train_path)
    assert dataloader.val_path == str(val_path)
    assert dataloader.batch_size == 32


@pytest.mark.unit
def test_diffusion_dataloader_invalid_train_path():
    """Test that initialization fails with invalid training path."""
    with pytest.raises(FileNotFoundError, match="Training data path not found"):
        DiffusionDataLoader(train_path="/nonexistent/path", batch_size=32)


@pytest.mark.unit
def test_diffusion_dataloader_invalid_val_path(tmp_data_dir):
    """Test that initialization fails with invalid validation path."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    with pytest.raises(FileNotFoundError, match="Validation data path not found"):
        DiffusionDataLoader(
            train_path=str(train_path), val_path="/nonexistent/val/path", batch_size=32
        )


@pytest.mark.unit
def test_diffusion_dataloader_default_parameters(tmp_data_dir):
    """Test default parameter values."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(train_path=str(train_path))

    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.image_size == 64
    assert dataloader.horizontal_flip is True
    assert dataloader.rotation_degrees == 0
    assert dataloader.color_jitter is False
    assert dataloader.color_jitter_strength == 0.1
    assert dataloader.pin_memory is True
    assert dataloader.drop_last is False
    assert dataloader.shuffle_train is True
    assert dataloader.return_labels is True


@pytest.mark.unit
def test_diffusion_dataloader_custom_parameters(tmp_data_dir):
    """Test custom parameter values."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=128,
        num_workers=2,
        image_size=128,
        horizontal_flip=False,
        rotation_degrees=15,
        color_jitter=True,
        color_jitter_strength=0.2,
        pin_memory=False,
        drop_last=True,
        shuffle_train=False,
        return_labels=False,
    )

    assert dataloader.batch_size == 128
    assert dataloader.num_workers == 2
    assert dataloader.image_size == 128
    assert dataloader.horizontal_flip is False
    assert dataloader.rotation_degrees == 15
    assert dataloader.color_jitter is True
    assert dataloader.color_jitter_strength == 0.2
    assert dataloader.pin_memory is False
    assert dataloader.drop_last is True
    assert dataloader.shuffle_train is False
    assert dataloader.return_labels is False


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_diffusion_dataset(tmp_path):
    """Create a minimal mock dataset for diffusion testing.

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
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"img_{i}.jpg")

    return train_path, val_path


@pytest.mark.unit
def test_diffusion_dataloader_get_config(tmp_data_dir):
    """Test that get_config returns expected configuration."""
    train_path = tmp_data_dir / "train"
    val_path = tmp_data_dir / "val"
    train_path.mkdir()
    val_path.mkdir()
    (train_path / "class1").mkdir()
    (val_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(
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
def test_diffusion_dataloader_repr(tmp_data_dir):
    """Test string representation of dataloader."""
    train_path = tmp_data_dir / "train"
    train_path.mkdir()
    (train_path / "class1").mkdir()

    dataloader = DiffusionDataLoader(train_path=str(train_path), batch_size=32)

    repr_str = repr(dataloader)
    assert "DiffusionDataLoader" in repr_str
    assert "batch_size=32" in repr_str


# ==============================================================================
# Component Tests - Small data, minimal computation
# ==============================================================================


@pytest.mark.component
def test_get_train_loader_basic(mock_diffusion_dataset):
    """Test basic train loader creation."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
    )

    train_loader = dataloader.get_train_loader()

    assert train_loader is not None
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert train_loader.batch_size == 2


@pytest.mark.component
def test_get_train_loader_with_labels(mock_diffusion_dataset):
    """Test train loader returns images and labels when return_labels=True."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
        return_labels=True,
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))

    # Should return tuple (images, labels)
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    images, labels = batch

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape[0] == labels.shape[0]  # Same batch size


@pytest.mark.component
def test_get_train_loader_without_labels(mock_diffusion_dataset):
    """Test train loader returns only images when return_labels=False."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
        return_labels=False,
    )

    train_loader = dataloader.get_train_loader()
    batch = next(iter(train_loader))

    # Should return only images (single tensor)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2  # Batch size
    assert batch.shape[1] == 3  # RGB channels


@pytest.mark.component
def test_train_loader_image_shape(mock_diffusion_dataset):
    """Test that images have correct shape."""
    train_path, _ = mock_diffusion_dataset

    image_size = 32
    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=image_size,
    )

    train_loader = dataloader.get_train_loader()
    images, _ = next(iter(train_loader))

    assert images.shape == (2, 3, image_size, image_size)


@pytest.mark.component
def test_train_loader_image_normalization(mock_diffusion_dataset):
    """Test that images are normalized to [-1, 1] range."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=4,
        num_workers=0,
        image_size=32,
    )

    train_loader = dataloader.get_train_loader()
    images, _ = next(iter(train_loader))

    # Check if values are approximately in [-1, 1] range
    # Allowing some tolerance for edge cases
    assert images.min() >= -1.1
    assert images.max() <= 1.1

    # Most values should be in the [-1, 1] range
    assert (images >= -1.0).sum() / images.numel() > 0.95
    assert (images <= 1.0).sum() / images.numel() > 0.95


@pytest.mark.component
def test_get_val_loader_basic(mock_diffusion_dataset):
    """Test basic validation loader creation."""
    train_path, val_path = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
    )

    val_loader = dataloader.get_val_loader()

    assert val_loader is not None
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert val_loader.batch_size == 2


@pytest.mark.component
def test_get_val_loader_returns_none_when_no_val_path(mock_diffusion_dataset):
    """Test that get_val_loader returns None when no validation path specified."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=None,
        batch_size=2,
        num_workers=0,
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader is None


@pytest.mark.component
def test_val_loader_no_shuffle(mock_diffusion_dataset):
    """Test that validation loader does not shuffle."""
    train_path, val_path = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        num_workers=0,
    )

    val_loader = dataloader.get_val_loader()

    # Validation loader should not shuffle
    assert val_loader.sampler is not None
    # Check that sampler doesn't shuffle (SequentialSampler)
    # We can check by getting samples twice and comparing order
    samples1 = []
    for batch in val_loader:
        if isinstance(batch, tuple):
            _, labels = batch
            samples1.extend(labels.tolist())
        break

    # Reset iterator
    val_loader_2 = dataloader.get_val_loader()
    samples2 = []
    for batch in val_loader_2:
        if isinstance(batch, tuple):
            _, labels = batch
            samples2.extend(labels.tolist())
        break

    # Should be in same order
    assert samples1 == samples2


@pytest.mark.component
def test_val_loader_no_drop_last(mock_diffusion_dataset):
    """Test that validation loader does not drop last batch."""
    train_path, val_path = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=3,  # Use odd batch size to ensure incomplete last batch
        num_workers=0,
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader.drop_last is False


@pytest.mark.component
def test_get_num_classes_with_labels(mock_diffusion_dataset):
    """Test get_num_classes returns correct number when return_labels=True."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        return_labels=True,
        num_workers=0,
    )

    num_classes = dataloader.get_num_classes()
    assert num_classes == 2  # mock_diffusion_dataset has 2 classes


@pytest.mark.component
def test_get_num_classes_without_labels(mock_diffusion_dataset):
    """Test get_num_classes returns None when return_labels=False."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        return_labels=False,
        num_workers=0,
    )

    num_classes = dataloader.get_num_classes()
    assert num_classes is None


@pytest.mark.component
def test_train_loader_with_augmentation(mock_diffusion_dataset):
    """Test train loader with augmentation enabled."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
        horizontal_flip=True,
        rotation_degrees=15,
        color_jitter=True,
        color_jitter_strength=0.1,
    )

    train_loader = dataloader.get_train_loader()
    images, _ = next(iter(train_loader))

    # Just verify that augmented images are loaded successfully
    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == torch.float32


@pytest.mark.component
def test_train_loader_without_augmentation(mock_diffusion_dataset):
    """Test train loader without augmentation."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
        horizontal_flip=False,
        rotation_degrees=0,
        color_jitter=False,
    )

    train_loader = dataloader.get_train_loader()
    images, _ = next(iter(train_loader))

    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == torch.float32


@pytest.mark.component
def test_different_image_sizes(mock_diffusion_dataset):
    """Test dataloader with different image sizes."""
    train_path, _ = mock_diffusion_dataset

    for image_size in [32, 64, 128]:
        dataloader = DiffusionDataLoader(
            train_path=str(train_path),
            batch_size=2,
            num_workers=0,
            image_size=image_size,
        )

        train_loader = dataloader.get_train_loader()
        images, _ = next(iter(train_loader))

        assert images.shape == (2, 3, image_size, image_size)


@pytest.mark.component
def test_train_and_val_consistency(mock_diffusion_dataset):
    """Test that train and val loaders have consistent behavior."""
    train_path, val_path = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
    )

    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    # Same shape
    assert train_images.shape[1:] == val_images.shape[1:]
    # Same dtype
    assert train_images.dtype == val_images.dtype
    # Same value range
    assert train_images.min() >= -1.1
    assert train_images.max() <= 1.1
    assert val_images.min() >= -1.1
    assert val_images.max() <= 1.1


# ==============================================================================
# Integration Tests - Mini workflows
# ==============================================================================


@pytest.mark.integration
def test_full_training_loop_simulation(mock_diffusion_dataset):
    """Test simulating a mini training loop with the dataloader."""
    train_path, val_path = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=2,
        num_workers=0,
        image_size=32,
        horizontal_flip=True,
        rotation_degrees=15,
        color_jitter=True,
    )

    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    # Simulate training epoch
    train_batches = 0
    for images, labels in train_loader:
        assert images.shape[0] <= 2
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == images.shape[0]
        train_batches += 1

    # Simulate validation epoch
    val_batches = 0
    for images, labels in val_loader:
        assert images.shape[0] <= 2
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == images.shape[0]
        val_batches += 1

    assert train_batches > 0
    assert val_batches > 0


@pytest.mark.integration
def test_unconditional_diffusion_workflow(mock_diffusion_dataset):
    """Test unconditional diffusion training workflow (no labels)."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=4,
        num_workers=0,
        image_size=32,
        return_labels=False,  # Unconditional
    )

    train_loader = dataloader.get_train_loader()
    num_classes = dataloader.get_num_classes()

    # Should not return num_classes for unconditional
    assert num_classes is None

    # Iterate through data
    for images in train_loader:
        # Should receive only images, not labels
        assert isinstance(images, torch.Tensor)
        assert images.shape[1:] == (3, 32, 32)
        assert images.min() >= -1.1
        assert images.max() <= 1.1
        break  # Just test one batch


@pytest.mark.integration
def test_conditional_diffusion_workflow(mock_diffusion_dataset):
    """Test conditional diffusion training workflow (with labels)."""
    train_path, _ = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        train_path=str(train_path),
        batch_size=4,
        num_workers=0,
        image_size=32,
        return_labels=True,  # Conditional
    )

    train_loader = dataloader.get_train_loader()
    num_classes = dataloader.get_num_classes()

    # Should return num_classes for conditional
    assert num_classes == 2

    # Iterate through data
    for images, labels in train_loader:
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[0] == labels.shape[0]
        assert labels.min() >= 0
        assert labels.max() < num_classes
        break  # Just test one batch
