"""
Tests for dataset implementations.

Tests cover base dataset interface, ImageFolder implementation,
and dataset factory functions. Includes both unit and component tests.
"""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.utils.data.datasets import (
    BaseDataset,
    ImageFolderDataset,
    SimpleImageDataset,
    SplitFileDataset,
    get_dataset,
)

# Path to mock data
MOCK_DATA_DIR = Path(__file__).parent.parent.parent / "fixtures" / "mock_data"
TRAIN_DIR = MOCK_DATA_DIR / "train"
VAL_DIR = MOCK_DATA_DIR / "val"


@pytest.fixture
def sample_transform():
    """Create a simple transform for testing."""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )


@pytest.mark.unit
class TestBaseDataset:
    """Unit tests for BaseDataset abstract class."""

    def test_base_dataset_is_abstract(self):
        """BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()  # type: ignore[abstract]

    def test_base_dataset_has_required_methods(self):
        """BaseDataset defines required abstract methods."""
        required_methods = ["__len__", "__getitem__", "get_classes", "get_class_counts"]

        for method in required_methods:
            assert hasattr(BaseDataset, method)
            assert callable(getattr(BaseDataset, method))


@pytest.mark.unit
class TestImageFolderDatasetValidation:
    """Unit tests for ImageFolderDataset validation."""

    def test_nonexistent_directory_raises_error(self):
        """Should raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            ImageFolderDataset("/nonexistent/path")

    def test_file_instead_of_directory_raises_error(self, tmp_path):
        """Should raise NotADirectoryError when path is a file."""
        # Create a file instead of directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(NotADirectoryError):
            ImageFolderDataset(str(test_file))

    def test_empty_directory_raises_error(self, tmp_path):
        """Should raise ValueError when no valid images found."""
        # Create empty class directories
        (tmp_path / "class1").mkdir()
        (tmp_path / "class2").mkdir()

        with pytest.raises(ValueError, match="No valid images found"):
            ImageFolderDataset(str(tmp_path))


@pytest.mark.component
class TestImageFolderDatasetLoading:
    """Component tests for ImageFolderDataset with actual data."""

    def test_load_train_dataset(self):
        """Should successfully load training dataset."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        assert len(dataset) > 0
        assert dataset.root == TRAIN_DIR

    def test_load_val_dataset(self):
        """Should successfully load validation dataset."""
        dataset = ImageFolderDataset(str(VAL_DIR))

        assert len(dataset) > 0
        assert dataset.root == VAL_DIR

    def test_dataset_length(self):
        """Dataset length should match number of images."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        # We created 5 Normal + 3 Abnormal = 8 images
        assert len(dataset) == 8

    def test_get_classes(self):
        """Should return correct class names."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))
        classes = dataset.get_classes()

        assert isinstance(classes, list)
        assert len(classes) == 2
        assert "0.Normal" in classes
        assert "1.Abnormal" in classes

    def test_classes_property(self):
        """Classes property should work."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        assert hasattr(dataset, "classes")
        assert len(dataset.classes) == 2

    def test_class_to_idx_mapping(self):
        """Should have correct class to index mapping."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        assert hasattr(dataset, "class_to_idx")
        assert isinstance(dataset.class_to_idx, dict)
        assert len(dataset.class_to_idx) == 2

        # Indices should be 0 and 1
        indices = set(dataset.class_to_idx.values())
        assert indices == {0, 1}

    def test_get_class_counts(self):
        """Should return correct sample counts per class."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))
        counts = dataset.get_class_counts()

        assert isinstance(counts, dict)
        assert counts["0.Normal"] == 5
        assert counts["1.Abnormal"] == 3

    def test_samples_property(self):
        """Samples property should contain all image paths and labels."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        assert hasattr(dataset, "samples")
        assert len(dataset.samples) == 8

        # Each sample is a tuple of (path, label)
        for sample in dataset.samples:
            assert isinstance(sample, tuple)
            assert len(sample) == 2
            path, label = sample
            assert isinstance(path, str)
            assert isinstance(label, int)
            assert os.path.exists(path)

    def test_targets_property(self):
        """Targets property should contain all labels."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        assert hasattr(dataset, "targets")
        assert len(dataset.targets) == 8

        # All labels should be 0 or 1
        for target in dataset.targets:
            assert target in [0, 1]


@pytest.mark.component
class TestImageFolderDatasetGetItem:
    """Component tests for ImageFolderDataset __getitem__."""

    def test_getitem_without_transform(self):
        """Should return PIL Image when no transform applied."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))
        image, label = dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_getitem_with_transform(self, sample_transform):
        """Should return transformed tensor when transform applied."""
        dataset = ImageFolderDataset(str(TRAIN_DIR), transform=sample_transform)
        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # C, H, W
        assert image.shape == (3, 32, 32)
        assert isinstance(label, int)

    def test_getitem_all_samples(self, sample_transform):
        """Should be able to load all samples."""
        dataset = ImageFolderDataset(str(TRAIN_DIR), transform=sample_transform)

        for i in range(len(dataset)):
            image, label = dataset[i]
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)

    def test_getitem_with_target_transform(self):
        """Should apply target transform to labels."""

        def target_transform(x):
            return x + 10

        dataset = ImageFolderDataset(str(TRAIN_DIR), target_transform=target_transform)

        _, label = dataset[0]
        assert label >= 10  # Should be 10 or 11


@pytest.mark.component
class TestImageFolderDatasetSampleWeights:
    """Component tests for sample weighting."""

    def test_get_sample_weights_inverse_frequency(self):
        """Should compute inverse frequency weights."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))
        weights = dataset.get_sample_weights()

        assert len(weights) == len(dataset)
        assert all(w > 0 for w in weights)

        # Samples from minority class should have higher weight
        # We have 5 Normal (class 0) and 3 Abnormal (class 1)
        # So Abnormal samples should have higher weight
        class_weights_map = {}
        for (_, label), weight in zip(dataset.samples, weights):
            if label not in class_weights_map:
                class_weights_map[label] = weight

        # Both classes should have weights
        assert len(class_weights_map) == 2

    def test_get_sample_weights_custom(self):
        """Should use custom class weights when provided."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))

        # Custom weights: class 0 gets weight 1.0, class 1 gets weight 2.0
        custom_weights = {0: 1.0, 1: 2.0}
        weights = dataset.get_sample_weights(class_weights=custom_weights)

        assert len(weights) == len(dataset)

        # Verify weights match custom values
        for (_, label), weight in zip(dataset.samples, weights):
            assert weight == custom_weights[label]


@pytest.mark.component
class TestImageFolderDatasetPrintSummary:
    """Test dataset summary printing."""

    def test_print_summary(self, capsys):
        """Should print dataset summary without errors."""
        dataset = ImageFolderDataset(str(TRAIN_DIR))
        dataset.print_summary()

        captured = capsys.readouterr()
        output = captured.out

        # Check that key information is in the output
        assert "Dataset:" in output
        assert "Total samples:" in output
        assert "Classes:" in output
        assert "0.Normal" in output
        assert "1.Abnormal" in output


@pytest.mark.unit
class TestSimpleImageDatasetValidation:
    """Unit tests for SimpleImageDataset validation."""

    def test_nonexistent_directory_raises_error(self):
        """Should raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            SimpleImageDataset("/nonexistent/path")

    def test_empty_directory_raises_error(self, tmp_path):
        """Should raise ValueError when no valid images found."""
        with pytest.raises(ValueError, match="No valid images found"):
            SimpleImageDataset(str(tmp_path))


@pytest.mark.component
class TestSimpleImageDatasetLoading:
    """Component tests for SimpleImageDataset."""

    def test_load_flat_directory(self, tmp_path, sample_transform):
        """Should load images from flat directory."""
        # Create some test images
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 50, i * 50, i * 50))
            img.save(tmp_path / f"image_{i}.png")

        dataset = SimpleImageDataset(str(tmp_path), transform=sample_transform)

        assert len(dataset) == 3

    def test_getitem_returns_tensor(self, tmp_path, sample_transform):
        """Should return only image tensor, no label."""
        # Create test image
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(tmp_path / "test.png")

        dataset = SimpleImageDataset(str(tmp_path), transform=sample_transform)

        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 32, 32)

    def test_get_classes_returns_empty(self, tmp_path):
        """Should return empty list for classes (unlabeled)."""
        img = Image.new("RGB", (32, 32))
        img.save(tmp_path / "test.png")

        dataset = SimpleImageDataset(str(tmp_path))

        assert dataset.get_classes() == []
        assert dataset.get_class_counts() == {}


@pytest.mark.component
class TestDatasetFactory:
    """Component tests for get_dataset factory function."""

    def test_factory_creates_imagefolder(self):
        """Should create ImageFolderDataset."""
        dataset = get_dataset("imagefolder", str(TRAIN_DIR))

        assert isinstance(dataset, ImageFolderDataset)

    def test_factory_case_insensitive(self):
        """Should work with different case."""
        dataset = get_dataset("ImageFolder", str(TRAIN_DIR))

        assert isinstance(dataset, ImageFolderDataset)

    def test_factory_unknown_type_raises_error(self):
        """Should raise ValueError for unknown dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            get_dataset("unknown_type", str(TRAIN_DIR))

    def test_factory_passes_kwargs(self, sample_transform):
        """Should pass additional kwargs to dataset constructor."""
        dataset = get_dataset("imagefolder", str(TRAIN_DIR), transform=sample_transform)

        assert dataset.transform is not None  # type: ignore[attr-defined]
        image, _ = dataset[0]
        assert isinstance(image, torch.Tensor)


@pytest.mark.component
class TestDatasetIntegration:
    """Integration tests for datasets with DataLoader."""

    def test_imagefolder_with_dataloader(self, sample_transform):
        """Should work correctly with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ImageFolderDataset(str(TRAIN_DIR), transform=sample_transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Load one batch
        batch = next(iter(dataloader))
        images, labels = batch

        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)

    def test_weighted_sampling(self, sample_transform):
        """Should work with WeightedRandomSampler."""
        from torch.utils.data import DataLoader, WeightedRandomSampler

        dataset = ImageFolderDataset(str(TRAIN_DIR), transform=sample_transform)
        weights = dataset.get_sample_weights()

        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(dataset), replacement=True
        )

        dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

        # Should be able to load a batch
        batch = next(iter(dataloader))
        images, labels = batch

        assert images.shape[0] == 2
        assert labels.shape[0] == 2


@pytest.mark.unit
class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_dataset_with_single_image(self, tmp_path, sample_transform):
        """Should handle dataset with single image."""
        # Create single image
        img = Image.new("RGB", (32, 32))
        img.save(tmp_path / "single.png")

        dataset = SimpleImageDataset(str(tmp_path), transform=sample_transform)

        assert len(dataset) == 1
        image = dataset[0]
        assert isinstance(image, torch.Tensor)

    def test_dataset_with_different_extensions(self, tmp_path):
        """Should load images with different extensions."""
        # Create images with different extensions
        img = Image.new("RGB", (32, 32))
        img.save(tmp_path / "test.png")
        img.save(tmp_path / "test.jpg")
        img.save(tmp_path / "test.jpeg")

        dataset = SimpleImageDataset(str(tmp_path))

        # Should find all 3 images
        assert len(dataset) >= 3


@pytest.mark.unit
class TestSplitFileDatasetUnit:
    """Unit tests for SplitFileDataset."""

    def _make_split_json(self, tmp_path, image_paths):
        """Create a split JSON file referencing the given image paths."""
        import json

        split_data = {
            "metadata": {"classes": {"cat": 0, "dog": 1}},
            "train": [{"path": str(p), "label": 0} for p in image_paths],
            "val": [],
        }
        split_file = tmp_path / "split.json"
        split_file.write_text(json.dumps(split_data))
        return str(split_file)

    def test_missing_image_raises_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError when image file does not exist."""
        nonexistent = tmp_path / "ghost.png"
        split_file = self._make_split_json(tmp_path, [nonexistent])

        dataset = SplitFileDataset(split_file, split="train")
        with pytest.raises(FileNotFoundError):
            dataset[0]

    def test_get_dataset_splitfile_factory(self, tmp_path):
        """Should create SplitFileDataset via get_dataset factory."""
        img = Image.new("RGB", (4, 4))
        img_path = tmp_path / "real.png"
        img.save(img_path)
        split_file = self._make_split_json(tmp_path, [img_path])

        dataset = get_dataset("splitfile", split_file=split_file, split="train")
        assert isinstance(dataset, SplitFileDataset)
        assert len(dataset) == 1

    def test_get_classes_returns_sorted(self, tmp_path):
        """SplitFileDataset.get_classes() returns classes sorted by label index."""
        img = Image.new("RGB", (4, 4))
        img_path = tmp_path / "real.png"
        img.save(img_path)
        split_file = self._make_split_json(tmp_path, [img_path])

        dataset = SplitFileDataset(split_file, split="train")
        classes = dataset.get_classes()
        assert classes == ["cat", "dog"]


@pytest.mark.unit
class TestImageFolderDatasetPrintSummaryUnit:
    """Unit test for ImageFolderDataset.print_summary()."""

    def test_print_summary_outputs_content(self, tmp_path, capsys):
        """print_summary() writes dataset info to stdout."""
        (tmp_path / "cat").mkdir()
        (tmp_path / "dog").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "cat" / "a.png")
        Image.new("RGB", (4, 4)).save(tmp_path / "dog" / "b.png")

        dataset = ImageFolderDataset(str(tmp_path))
        dataset.print_summary()

        captured = capsys.readouterr()
        assert "Total samples:" in captured.out
        assert "Classes:" in captured.out
        assert "cat" in captured.out
        assert "dog" in captured.out


@pytest.mark.unit
class TestImageFolderGetItemUnit:
    """Unit tests for ImageFolderDataset __getitem__ paths."""

    def test_getitem_return_labels_false(self, tmp_path, sample_transform):
        """return_labels=False returns only the image, not a tuple."""
        (tmp_path / "cat").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "cat" / "a.png")

        dataset = ImageFolderDataset(
            str(tmp_path), transform=sample_transform, return_labels=False
        )
        item = dataset[0]
        # Should be a tensor, not a tuple
        assert isinstance(item, torch.Tensor)

    def test_getitem_return_labels_true(self, tmp_path, sample_transform):
        """return_labels=True (default) returns (image, label) tuple."""
        (tmp_path / "cat").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "cat" / "a.png")

        dataset = ImageFolderDataset(str(tmp_path), transform=sample_transform)
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2


@pytest.mark.unit
class TestImageFolderSampleWeightsUnit:
    """Unit tests for ImageFolderDataset.get_sample_weights()."""

    def test_get_sample_weights_inverse_frequency(self, tmp_path):
        """Inverse frequency weights: minority class gets higher weight."""
        (tmp_path / "cat").mkdir()
        (tmp_path / "dog").mkdir()
        # 2 cats, 4 dogs → cat weight should be higher
        for i in range(2):
            Image.new("RGB", (4, 4)).save(tmp_path / "cat" / f"{i}.png")
        for i in range(4):
            Image.new("RGB", (4, 4)).save(tmp_path / "dog" / f"{i}.png")

        dataset = ImageFolderDataset(str(tmp_path))
        weights = dataset.get_sample_weights()

        assert len(weights) == 6
        # cat samples (minority) should have higher weight
        cat_idx = dataset.class_to_idx["cat"]
        dog_idx = dataset.class_to_idx["dog"]
        cat_weight = None
        dog_weight = None
        for (_, label), w in zip(dataset.samples, weights):
            if label == cat_idx and cat_weight is None:
                cat_weight = w
            if label == dog_idx and dog_weight is None:
                dog_weight = w
        assert cat_weight is not None and dog_weight is not None
        assert cat_weight > dog_weight


@pytest.mark.unit
class TestSplitFileDatasetAccessorsUnit:
    """Unit tests for SplitFileDataset property accessors and methods."""

    def _make_split_json(self, tmp_path, entries_per_class=2):
        """Create a split JSON with real images."""
        import json

        images = []
        for cls_idx in range(2):
            cls_name = f"cls{cls_idx}"
            cls_dir = tmp_path / cls_name
            cls_dir.mkdir(exist_ok=True)
            for i in range(entries_per_class):
                img_path = cls_dir / f"img_{i}.png"
                Image.new("RGB", (4, 4)).save(img_path)
                images.append({"path": str(img_path), "label": cls_idx})

        split_data = {
            "metadata": {"classes": {"cls0": 0, "cls1": 1}},
            "train": images,
            "val": images[:1],
        }
        split_file = tmp_path / "split.json"
        split_file.write_text(json.dumps(split_data))
        return str(split_file)

    def test_getitem_returns_image_and_label(self, tmp_path, sample_transform):
        """SplitFileDataset[0] returns (image_tensor, label)."""
        split_file = self._make_split_json(tmp_path)
        dataset = SplitFileDataset(
            split_file, split="train", transform=sample_transform
        )
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_getitem_no_labels(self, tmp_path, sample_transform):
        """return_labels=False returns only image."""
        split_file = self._make_split_json(tmp_path)
        dataset = SplitFileDataset(
            split_file, split="train", transform=sample_transform, return_labels=False
        )
        item = dataset[0]
        assert isinstance(item, torch.Tensor)

    def test_get_class_counts(self, tmp_path):
        """get_class_counts() returns per-class sample counts."""
        split_file = self._make_split_json(tmp_path, entries_per_class=3)
        dataset = SplitFileDataset(split_file, split="train")
        counts = dataset.get_class_counts()
        assert counts == {"cls0": 3, "cls1": 3}

    def test_classes_property(self, tmp_path):
        """classes property returns sorted class names."""
        split_file = self._make_split_json(tmp_path)
        dataset = SplitFileDataset(split_file, split="train")
        assert dataset.classes == ["cls0", "cls1"]

    def test_class_to_idx_property(self, tmp_path):
        """class_to_idx property returns correct mapping."""
        split_file = self._make_split_json(tmp_path)
        dataset = SplitFileDataset(split_file, split="train")
        assert dataset.class_to_idx == {"cls0": 0, "cls1": 1}

    def test_samples_property(self, tmp_path):
        """samples property returns list of (path, label) tuples."""
        split_file = self._make_split_json(tmp_path, entries_per_class=2)
        dataset = SplitFileDataset(split_file, split="train")
        samples = dataset.samples
        assert len(samples) == 4
        for path, label in samples:
            assert isinstance(path, str)
            assert isinstance(label, int)

    def test_targets_property(self, tmp_path):
        """targets property returns list of label ints."""
        split_file = self._make_split_json(tmp_path, entries_per_class=2)
        dataset = SplitFileDataset(split_file, split="train")
        targets = dataset.targets
        assert len(targets) == 4
        assert set(targets) == {0, 1}


@pytest.mark.unit
class TestDatasetFactoryUnit:
    """Unit tests for get_dataset factory with simple and splitfile types."""

    def test_factory_simple(self, tmp_path):
        """get_dataset('simple', ...) returns SimpleImageDataset."""
        Image.new("RGB", (4, 4)).save(tmp_path / "img.png")
        dataset = get_dataset("simple", str(tmp_path))
        assert isinstance(dataset, SimpleImageDataset)
        assert len(dataset) == 1

    def test_factory_splitfile(self, tmp_path):
        """get_dataset('splitfile', ...) returns SplitFileDataset."""
        import json

        img_path = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(img_path)
        split_data = {
            "metadata": {"classes": {"a": 0}},
            "train": [{"path": str(img_path), "label": 0}],
            "val": [],
        }
        split_file = tmp_path / "split.json"
        split_file.write_text(json.dumps(split_data))

        dataset = get_dataset("splitfile", split_file=str(split_file), split="train")
        assert isinstance(dataset, SplitFileDataset)
        assert len(dataset) == 1
