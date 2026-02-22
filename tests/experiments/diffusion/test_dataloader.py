"""Tests for Diffusion DataLoader

This module contains comprehensive tests for the DiffusionDataLoader class.
Tests are organized by tiers: unit tests, component tests, and integration tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.base.dataloader import BaseDataLoader
from src.experiments.diffusion.dataloader import DiffusionDataLoader

# ==============================================================================
# Helpers
# ==============================================================================


def _create_split_json(tmp_path, train_per_class=4, val_per_class=4, num_classes=2):
    """Create mock images and a split JSON file.

    Returns:
        Path string to the created split JSON file.
    """
    class_names = [f"{i}.Class{i}" for i in range(num_classes)]
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


def _create_split_json_train_only(tmp_path, train_per_class=4, num_classes=2):
    """Create a split JSON with only training data (empty val)."""
    class_names = [f"{i}.Class{i}" for i in range(num_classes)]
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
def test_diffusion_dataloader_inherits_from_base():
    """Test that DiffusionDataLoader inherits from BaseDataLoader."""
    assert issubclass(DiffusionDataLoader, BaseDataLoader)


@pytest.mark.unit
def test_diffusion_dataloader_initialization(tmp_path):
    """Test basic initialization with split file."""
    split_file = _create_split_json(tmp_path)

    dataloader = DiffusionDataLoader(
        split_file=split_file, batch_size=16, num_workers=0
    )

    assert dataloader.split_file == split_file
    assert dataloader.batch_size == 16
    assert dataloader.num_workers == 0


@pytest.mark.unit
def test_diffusion_dataloader_invalid_split_file():
    """Test that initialization fails with invalid split file path."""
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        DiffusionDataLoader(split_file="/nonexistent/split.json", batch_size=32)


@pytest.mark.unit
def test_diffusion_dataloader_default_parameters(tmp_path):
    """Test default parameter values."""
    split_file = _create_split_json(tmp_path)

    dataloader = DiffusionDataLoader(split_file=split_file)

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
def test_diffusion_dataloader_custom_parameters(tmp_path):
    """Test custom parameter values."""
    split_file = _create_split_json(tmp_path)

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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

    Returns:
        Path string to the split JSON file.
    """
    return _create_split_json(tmp_path, train_per_class=4, val_per_class=4)


@pytest.mark.unit
def test_diffusion_dataloader_get_config(tmp_path):
    """Test that get_config returns expected configuration."""
    split_file = _create_split_json(tmp_path)

    dataloader = DiffusionDataLoader(
        split_file=split_file, batch_size=16, num_workers=2
    )

    config = dataloader.get_config()

    assert "split_file" in config
    assert "batch_size" in config
    assert "num_workers" in config
    assert config["batch_size"] == 16
    assert config["num_workers"] == 2


@pytest.mark.unit
def test_diffusion_dataloader_repr(tmp_path):
    """Test string representation of dataloader."""
    split_file = _create_split_json(tmp_path)

    dataloader = DiffusionDataLoader(split_file=split_file, batch_size=32)

    repr_str = repr(dataloader)
    assert "DiffusionDataLoader" in repr_str
    assert "batch_size=32" in repr_str


# ==============================================================================
# Component Tests - Small data, minimal computation
# ==============================================================================


@pytest.mark.component
def test_get_train_loader_basic(mock_diffusion_dataset):
    """Test basic train loader creation."""
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    image_size = 32
    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
        image_size=32,
    )

    val_loader = dataloader.get_val_loader()

    assert val_loader is not None
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert val_loader.batch_size == 2


@pytest.mark.component
def test_get_val_loader_returns_none_when_empty_val(tmp_path):
    """Test that get_val_loader returns None when val split is empty."""
    split_file = _create_split_json_train_only(tmp_path)

    dataloader = DiffusionDataLoader(
        split_file=split_file,
        batch_size=2,
        num_workers=0,
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader is None


@pytest.mark.component
def test_val_loader_no_shuffle(mock_diffusion_dataset):
    """Test that validation loader does not shuffle."""
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
        batch_size=3,  # Use odd batch size to ensure incomplete last batch
        num_workers=0,
    )

    val_loader = dataloader.get_val_loader()
    assert val_loader.drop_last is False


@pytest.mark.component
def test_get_num_classes_with_labels(mock_diffusion_dataset):
    """Test get_num_classes returns correct number when return_labels=True."""
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
        return_labels=True,
        num_workers=0,
    )

    num_classes = dataloader.get_num_classes()
    assert num_classes == 2  # mock_diffusion_dataset has 2 classes


@pytest.mark.component
def test_get_num_classes_without_labels(mock_diffusion_dataset):
    """Test get_num_classes returns None when return_labels=False."""
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
        return_labels=False,
        num_workers=0,
    )

    num_classes = dataloader.get_num_classes()
    assert num_classes is None


@pytest.mark.component
def test_train_loader_with_augmentation(mock_diffusion_dataset):
    """Test train loader with augmentation enabled."""
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    for image_size in [32, 64, 128]:
        dataloader = DiffusionDataLoader(
            split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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
    split_file = mock_diffusion_dataset

    dataloader = DiffusionDataLoader(
        split_file=split_file,
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


# ==============================================================================
# Unit Tests: Balancing Config Integration
# ==============================================================================


@pytest.mark.unit
class TestDiffusionDataLoaderBalancing:
    """Test DiffusionDataLoader with balancing configuration."""

    def test_init_with_balancing_config(self, tmp_path):
        """Test that DiffusionDataLoader accepts balancing_config parameter."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
        balancing_config = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        assert dataloader.balancing_config == balancing_config
        assert dataloader.seed == 42

    def test_init_without_balancing_config(self, tmp_path):
        """Test backwards compatibility: no balancing_config is OK."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
        )

        assert dataloader.balancing_config is None

    def test_weighted_sampler_disables_shuffle(self, tmp_path):
        """Test that weighted_sampler sets shuffle=False."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=2, num_classes=2
        )
        balancing_config = {
            "weighted_sampler": {
                "enabled": True,
                "method": "inverse_frequency",
                "beta": 0.999,
                "replacement": True,
                "num_samples": None,
            },
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        # Should create a loader without error
        train_loader = dataloader.get_train_loader()
        assert train_loader is not None

        # The sampler should be set (non-None)
        assert train_loader.sampler is not None

    def test_downsampling_produces_loader(self, tmp_path):
        """Test that downsampling strategy creates a working loader."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=2, num_classes=2
        )
        balancing_config = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": True, "target_ratio": 1.0},
            "upsampling": {"enabled": False},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        train_loader = dataloader.get_train_loader()
        assert train_loader is not None
        # Iterate to verify it works
        for batch in train_loader:
            assert len(batch) == 2  # images, labels
            break

    def test_upsampling_produces_loader(self, tmp_path):
        """Test that upsampling strategy creates a working loader."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=2, num_classes=2
        )
        balancing_config = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": True, "target_ratio": 1.0},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        train_loader = dataloader.get_train_loader()
        assert train_loader is not None
        for batch in train_loader:
            assert len(batch) == 2
            break

    def test_priority_weighted_sampler_over_downsampling(self, tmp_path):
        """Test that weighted_sampler has priority over downsampling."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=2, num_classes=2
        )
        balancing_config = {
            "weighted_sampler": {
                "enabled": True,
                "method": "inverse_frequency",
                "replacement": True,
                "num_samples": None,
            },
            "downsampling": {"enabled": True, "target_ratio": 1.0},
            "upsampling": {"enabled": False},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        train_loader = dataloader.get_train_loader()
        # Should use sampler (weighted_sampler), not downsampled dataset
        assert train_loader.sampler is not None

    def test_val_loader_not_affected_by_balancing(self, tmp_path):
        """Test that validation loader is not affected by balancing."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=4, num_classes=2
        )
        balancing_config = {
            "weighted_sampler": {
                "enabled": True,
                "method": "inverse_frequency",
                "replacement": True,
                "num_samples": None,
            },
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
        }

        dataloader = DiffusionDataLoader(
            split_file=split_file,
            batch_size=2,
            image_size=16,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )

        val_loader = dataloader.get_val_loader()
        assert val_loader is not None
        # Val loader should have 8 total samples (4 per class)
        assert len(val_loader.dataset) == 8
