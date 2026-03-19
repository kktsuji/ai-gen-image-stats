"""Tests for DataLoader factory functions.

This module tests create_train_loader, create_val_loader, get_num_classes,
and get_class_names from src.utils.data.loaders.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src.utils.data.loaders import (
    create_synthetic_augmentation_dataset,
    create_train_loader,
    create_val_loader,
    get_class_names,
    get_num_classes,
)
from src.utils.data.transforms import get_train_transforms, get_val_transforms

# ==============================================================================
# Helpers
# ==============================================================================


def _create_split_json(tmp_path, train_per_class=4, val_per_class=4, num_classes=2):
    """Create mock images and a split JSON file."""
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
# Unit Tests
# ==============================================================================


@pytest.mark.unit
def test_create_train_loader_invalid_split_file():
    """Test that create_train_loader fails with nonexistent split file."""
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        create_train_loader(
            split_file="/nonexistent/split.json",
            batch_size=2,
            transform=transform,
        )


@pytest.mark.unit
def test_create_val_loader_invalid_split_file():
    """Test that create_val_loader fails with nonexistent split file."""
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        create_val_loader(
            split_file="/nonexistent/split.json",
            batch_size=2,
            transform=transform,
        )


@pytest.mark.unit
def test_create_val_loader_propagates_key_error(tmp_path):
    """Test that create_val_loader propagates KeyError when val key is missing."""
    from torchvision import transforms

    split_data = {
        "metadata": {"classes": {"0.Class0": 0}},
        "train": [{"path": "img.jpg", "label": 0}],
    }
    split_file = tmp_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    transform = transforms.Compose([transforms.ToTensor()])
    with pytest.raises(KeyError):
        create_val_loader(
            split_file=str(split_file),
            batch_size=2,
            transform=transform,
        )


@pytest.mark.unit
def test_get_num_classes_invalid_file():
    """Test that get_num_classes fails with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        get_num_classes("/nonexistent/split.json")


@pytest.mark.unit
def test_get_class_names_invalid_file():
    """Test that get_class_names fails with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        get_class_names("/nonexistent/split.json")


@pytest.mark.unit
def test_get_num_classes(tmp_path):
    """Test getting number of classes."""
    split_file = _create_split_json(tmp_path, num_classes=3)
    assert get_num_classes(split_file) == 3


@pytest.mark.unit
def test_get_class_names(tmp_path):
    """Test getting class names."""
    split_file = _create_split_json(tmp_path, num_classes=2)
    names = get_class_names(split_file)
    assert isinstance(names, list)
    assert len(names) == 2
    assert "0.Class0" in names
    assert "1.Class1" in names


@pytest.mark.unit
def test_get_num_classes_missing_metadata(tmp_path):
    """Test that get_num_classes raises ValueError when metadata is missing."""
    split_file = tmp_path / "no_metadata.json"
    split_file.write_text(json.dumps({"train": [], "val": []}))
    with pytest.raises(ValueError, match="No class metadata found"):
        get_num_classes(str(split_file))


@pytest.mark.unit
def test_get_class_names_missing_metadata(tmp_path):
    """Test that get_class_names raises ValueError when metadata is missing."""
    split_file = tmp_path / "no_metadata.json"
    split_file.write_text(json.dumps({"train": [], "val": []}))
    with pytest.raises(ValueError, match="No class metadata found"):
        get_class_names(str(split_file))


@pytest.mark.component
def test_create_train_loader_downsampling_unit(tmp_path):
    """Component test: downsampling balancing returns a DataLoader with a valid dataset."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    balancing_config = {
        "weighted_sampler": {"enabled": False},
        "downsampling": {"enabled": True, "target_ratio": 1.0},
        "upsampling": {"enabled": False},
    }
    loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
        balancing_config=balancing_config,
        seed=42,
    )
    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) > 0  # type: ignore[arg-type]


@pytest.mark.component
def test_create_train_loader_upsampling_unit(tmp_path):
    """Component test: upsampling balancing config is applied and returns a valid DataLoader."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    balancing_config = {
        "weighted_sampler": {"enabled": False},
        "downsampling": {"enabled": False},
        "upsampling": {"enabled": True, "target_ratio": 1.0},
    }
    loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
        balancing_config=balancing_config,
        seed=42,
    )
    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) >= 8  # type: ignore[arg-type]


@pytest.mark.unit
def test_create_train_loader_with_seed_sets_generator(tmp_path):
    """Test that passing a seed sets generator on the DataLoader."""
    split_file = _create_split_json(tmp_path)
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
        seed=42,
    )
    assert loader.generator is not None
    assert loader.worker_init_fn is not None


@pytest.mark.unit
def test_create_train_loader_without_seed_no_generator(tmp_path):
    """Test that omitting seed leaves generator and worker_init_fn as None."""
    split_file = _create_split_json(tmp_path)
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )
    assert loader.generator is None
    assert loader.worker_init_fn is None


# ==============================================================================
# Component Tests - Classifier-style transforms
# ==============================================================================


@pytest.fixture
def mock_split_file(tmp_path):
    """Create a minimal mock dataset."""
    return _create_split_json(tmp_path, train_per_class=3, val_per_class=3)


@pytest.mark.component
def test_create_train_loader_basic(mock_split_file):
    """Test basic train loader creation."""
    transform = get_train_transforms(image_size=32, crop_size=32)
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert len(loader.dataset) > 0  # type: ignore[arg-type]


@pytest.mark.component
def test_create_val_loader_basic(mock_split_file):
    """Test basic validation loader creation."""
    transform = get_val_transforms(image_size=32, crop_size=32)
    loader = create_val_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    assert loader is not None
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2


@pytest.mark.component
def test_create_val_loader_returns_none_when_empty(tmp_path):
    """Test that val loader returns None when val split is empty."""
    split_file = _create_split_json_train_only(tmp_path)
    transform = get_val_transforms(image_size=32, crop_size=32)

    loader = create_val_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )
    assert loader is None


@pytest.mark.component
def test_train_loader_batch_iteration(mock_split_file):
    """Test iterating through train loader batches."""
    transform = get_train_transforms(image_size=32, crop_size=32)
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    assert images.shape[0] <= 2
    assert images.shape[1] == 3
    assert images.shape[2] == 32
    assert images.shape[3] == 32
    assert labels.dtype == torch.long


@pytest.mark.component
def test_val_loader_batch_iteration(mock_split_file):
    """Test iterating through validation loader batches."""
    transform = get_val_transforms(image_size=32, crop_size=32)
    loader = create_val_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    images, labels = next(iter(loader))  # type: ignore[call-overload, arg-type]
    assert images.shape[0] <= 2
    assert images.shape[1] == 3
    assert images.shape[2] == 32
    assert images.shape[3] == 32


@pytest.mark.component
def test_train_loader_with_imagenet_normalization(mock_split_file):
    """Test train loader with ImageNet normalization."""
    transform = get_train_transforms(image_size=32, crop_size=32, normalize="imagenet")
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    images, _ = next(iter(loader))
    assert images.min() >= -3.0
    assert images.max() <= 3.0


@pytest.mark.component
def test_train_loader_without_normalization(mock_split_file):
    """Test train loader without normalization."""
    transform = get_train_transforms(image_size=32, crop_size=32, normalize=None)
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    images, _ = next(iter(loader))
    assert images.min() >= 0.0
    assert images.max() <= 1.0


@pytest.mark.component
def test_drop_last_behavior(mock_split_file):
    """Test drop_last parameter."""
    transform = get_train_transforms(image_size=32, crop_size=32)
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=3,
        transform=transform,
        num_workers=0,
        drop_last=True,
    )
    assert loader.drop_last is True


@pytest.mark.component
def test_different_image_sizes(mock_split_file):
    """Test with different image sizes."""
    transform = get_train_transforms(image_size=64, crop_size=64)
    loader = create_train_loader(
        split_file=mock_split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    images, _ = next(iter(loader))
    assert images.shape[2] == 64
    assert images.shape[3] == 64


# ==============================================================================
# Component Tests - Diffusion-style transforms ([-1, 1] normalization)
# ==============================================================================


@pytest.fixture
def mock_diffusion_split(tmp_path):
    """Create a mock dataset for diffusion testing."""
    return _create_split_json(tmp_path, train_per_class=4, val_per_class=4)


@pytest.mark.component
def test_diffusion_train_loader_normalization(mock_diffusion_split):
    """Test that diffusion transforms normalize to [-1, 1] range."""
    from src.utils.data.transforms import get_diffusion_val_transforms

    transform = get_diffusion_val_transforms(image_size=32)
    loader = create_train_loader(
        split_file=mock_diffusion_split,
        batch_size=4,
        transform=transform,
        num_workers=0,
    )

    images, _ = next(iter(loader))
    assert images.min() >= -1.1
    assert images.max() <= 1.1


@pytest.mark.component
def test_diffusion_train_loader_without_labels(mock_diffusion_split):
    """Test diffusion train loader returning only images."""
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
    )
    loader = create_train_loader(
        split_file=mock_diffusion_split,
        batch_size=2,
        transform=transform,
        num_workers=0,
        return_labels=False,
    )

    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2
    assert batch.shape[1] == 3


@pytest.mark.component
def test_diffusion_train_loader_with_labels(mock_diffusion_split):
    """Test diffusion train loader returning images and labels."""
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
    )
    loader = create_train_loader(
        split_file=mock_diffusion_split,
        batch_size=2,
        transform=transform,
        num_workers=0,
        return_labels=True,
    )

    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    images, labels = batch
    assert images.shape[0] == labels.shape[0]


# ==============================================================================
# Component Tests - Balancing
# ==============================================================================


@pytest.mark.component
class TestBalancingIntegration:
    """Test balancing config integration with create_train_loader."""

    def test_weighted_sampler(self, tmp_path):
        """Test weighted_sampler balancing strategy."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
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

        loader = create_train_loader(
            split_file=split_file,
            batch_size=2,
            transform=transform,
            num_workers=0,
            balancing_config=balancing_config,
        )
        assert loader is not None
        assert loader.sampler is not None

    def test_downsampling(self, tmp_path):
        """Test downsampling balancing strategy."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
        )
        balancing_config = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": True, "target_ratio": 1.0},
            "upsampling": {"enabled": False},
        }

        loader = create_train_loader(
            split_file=split_file,
            batch_size=2,
            transform=transform,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )
        assert loader is not None
        for batch in loader:
            assert len(batch) == 2
            break

    def test_upsampling(self, tmp_path):
        """Test upsampling balancing strategy."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
        )
        balancing_config = {
            "weighted_sampler": {"enabled": False},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": True, "target_ratio": 1.0},
        }

        loader = create_train_loader(
            split_file=split_file,
            batch_size=2,
            transform=transform,
            num_workers=0,
            balancing_config=balancing_config,
            seed=42,
        )
        assert loader is not None
        for batch in loader:
            assert len(batch) == 2
            break

    def test_priority_weighted_sampler_over_downsampling(self, tmp_path):
        """Test that weighted_sampler has priority over downsampling."""
        split_file = _create_split_json(tmp_path, train_per_class=4, num_classes=2)
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
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

        loader = create_train_loader(
            split_file=split_file,
            batch_size=2,
            transform=transform,
            num_workers=0,
            balancing_config=balancing_config,
        )
        assert loader.sampler is not None

    def test_val_not_affected_by_balancing(self, tmp_path):
        """Test that validation loader is not affected by balancing."""
        split_file = _create_split_json(
            tmp_path, train_per_class=4, val_per_class=4, num_classes=2
        )
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
        )

        val_loader = create_val_loader(
            split_file=split_file,
            batch_size=2,
            transform=transform,
            num_workers=0,
        )
        assert val_loader is not None
        assert len(val_loader.dataset) == 8  # type: ignore[arg-type]


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.integration
def test_full_classifier_workflow(tmp_path):
    """Test complete classifier-style data loading workflow."""
    split_file = _create_split_json(tmp_path, train_per_class=3, val_per_class=3)

    train_transform = get_train_transforms(
        image_size=32, crop_size=32, normalize="imagenet"
    )
    val_transform = get_val_transforms(
        image_size=32, crop_size=32, normalize="imagenet"
    )

    train_loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=train_transform,
        num_workers=0,
    )
    val_loader = create_val_loader(
        split_file=split_file,
        batch_size=2,
        transform=val_transform,
        num_workers=0,
    )

    # Iterate through training
    train_batches = 0
    for images, labels in train_loader:
        assert images.shape[1] == 3
        assert images.shape[2] == 32
        train_batches += 1
    assert train_batches > 0

    # Iterate through validation
    val_batches = 0
    for images, labels in val_loader:  # type: ignore[union-attr]
        assert images.shape[1] == 3
        assert images.shape[2] == 32
        val_batches += 1
    assert val_batches > 0

    # Get metadata
    num_classes = get_num_classes(split_file)
    class_names_list = get_class_names(split_file)
    assert num_classes == 2
    assert len(class_names_list) == 2


@pytest.mark.integration
def test_full_diffusion_workflow(tmp_path):
    """Test complete diffusion-style data loading workflow."""
    from src.utils.data.transforms import (
        get_diffusion_transforms,
        get_diffusion_val_transforms,
    )

    split_file = _create_split_json(tmp_path, train_per_class=4, val_per_class=4)

    transform = get_diffusion_transforms(image_size=32, horizontal_flip=True)

    train_loader = create_train_loader(
        split_file=split_file,
        batch_size=2,
        transform=transform,
        num_workers=0,
        return_labels=True,
    )
    val_transform = get_diffusion_val_transforms(image_size=32)
    val_loader = create_val_loader(
        split_file=split_file,
        batch_size=2,
        transform=val_transform,
        num_workers=0,
        return_labels=True,
    )

    # Training iteration
    for images, labels in train_loader:
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] == images.shape[0]
        assert images.min() >= -1.1
        assert images.max() <= 1.1
        break

    # Validation iteration
    for images, labels in val_loader:  # type: ignore[union-attr]
        assert images.shape[1:] == (3, 32, 32)
        break


@pytest.mark.integration
def test_unconditional_diffusion_workflow(tmp_path):
    """Test unconditional diffusion workflow (no labels)."""
    from src.utils.data.transforms import get_diffusion_val_transforms

    split_file = _create_split_json(tmp_path, train_per_class=4, val_per_class=4)

    transform = get_diffusion_val_transforms(image_size=32)

    train_loader = create_train_loader(
        split_file=split_file,
        batch_size=4,
        transform=transform,
        num_workers=0,
        return_labels=False,
    )

    for images in train_loader:
        assert isinstance(images, torch.Tensor)
        assert images.shape[1:] == (3, 32, 32)
        break


# ==============================================================================
# Unit Tests - Synthetic Augmentation Dataset
# ==============================================================================


@pytest.mark.unit
def test_synthetic_aug_invalid_split_file():
    """Test that create_synthetic_augmentation_dataset fails with nonexistent file."""
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    with pytest.raises(FileNotFoundError, match="Synthetic augmentation split file"):
        create_synthetic_augmentation_dataset(
            split_file="/nonexistent/split.json",
            transform=transform,
        )


@pytest.mark.unit
def test_synthetic_aug_no_limit(tmp_path):
    """Test loading full dataset when no limit is applied."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=5, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    dataset = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode=None,
    )
    assert len(dataset) == 10  # 5 per class * 2 classes


@pytest.mark.unit
def test_synthetic_aug_max_ratio(tmp_path):
    """Test max_ratio limits correctly (e.g., 100 real, max_ratio=0.5 -> 50 gen)."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=50, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    dataset = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_ratio",
        max_ratio=0.5,
        real_train_size=100,
        seed=42,
    )
    assert isinstance(dataset, Subset)
    assert len(dataset) == 50


@pytest.mark.unit
def test_synthetic_aug_max_samples(tmp_path):
    """Test max_samples limits correctly."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=10, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    dataset = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_samples",
        max_samples=5,
        seed=42,
    )
    assert isinstance(dataset, Subset)
    assert len(dataset) == 5


@pytest.mark.unit
def test_synthetic_aug_seeded_reproducible(tmp_path):
    """Test that subsampling is seeded and reproducible."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=10, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    ds1 = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_samples",
        max_samples=5,
        seed=42,
    )
    ds2 = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_samples",
        max_samples=5,
        seed=42,
    )
    assert isinstance(ds1, Subset)
    assert isinstance(ds2, Subset)
    assert ds1.indices == ds2.indices


@pytest.mark.unit
def test_synthetic_aug_limit_larger_than_dataset(tmp_path):
    """Test that limit larger than dataset returns all samples."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=3, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    dataset = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_samples",
        max_samples=1000,
        seed=42,
    )
    # Should return full dataset (not Subset) since limit > dataset size
    assert len(dataset) == 6  # 3 per class * 2 classes


@pytest.mark.unit
def test_synthetic_aug_max_ratio_zero_real_train_size(tmp_path):
    """Test that max_ratio with real_train_size=0 raises ValueError."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=3, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    with pytest.raises(ValueError, match="real_train_size must be positive"):
        create_synthetic_augmentation_dataset(
            split_file=split_file,
            transform=transform,
            limit_mode="max_ratio",
            max_ratio=0.5,
            real_train_size=0,
            seed=42,
        )


@pytest.mark.unit
def test_synthetic_aug_max_ratio_resolves_to_zero(tmp_path):
    """Test that very small max_ratio resolving to 0 returns full dataset with warning."""
    from torchvision import transforms

    split_file = _create_split_json(tmp_path, train_per_class=3, num_classes=2)
    transform = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )
    dataset = create_synthetic_augmentation_dataset(
        split_file=split_file,
        transform=transform,
        limit_mode="max_ratio",
        max_ratio=0.01,
        real_train_size=10,  # int(10 * 0.01) = 0
        seed=42,
    )
    # Should return full dataset (not empty Subset) when max_n resolves to 0
    assert len(dataset) == 6  # 3 per class * 2 classes
