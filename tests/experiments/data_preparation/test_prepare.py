"""Tests for Data Preparation - Split Generation Logic

This module tests the prepare_split function and helper utilities.
Tests cover deterministic splitting, ratio correctness, JSON output, and error handling.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.experiments.data_preparation.prepare import (
    _scan_image_files,
    _split_list,
    prepare_split,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _create_mock_class_dirs(base_dir, class_config):
    """Create mock class directories with images.

    Args:
        base_dir: Base directory to create class dirs in
        class_config: Dict mapping class_name to num_images
    """
    paths = {}
    for class_name, num_images in class_config.items():
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            img = Image.new("RGB", (32, 32), color="red")
            img.save(class_dir / f"img_{i:03d}.png")
        paths[class_name] = str(class_dir)
    return paths


# ============================================================================
# Unit Tests - _scan_image_files
# ============================================================================


@pytest.mark.unit
class TestScanImageFiles:
    """Test image file scanning."""

    def test_scans_png_files(self, tmp_path):
        """Test scanning finds PNG files."""
        class_dir = tmp_path / "class0"
        class_dir.mkdir()
        Image.new("RGB", (32, 32)).save(class_dir / "img.png")
        Image.new("RGB", (32, 32)).save(class_dir / "img2.png")

        files = _scan_image_files(str(class_dir))
        assert len(files) == 2

    def test_scans_jpg_files(self, tmp_path):
        """Test scanning finds JPG files."""
        class_dir = tmp_path / "class0"
        class_dir.mkdir()
        Image.new("RGB", (32, 32)).save(class_dir / "img.jpg")

        files = _scan_image_files(str(class_dir))
        assert len(files) == 1

    def test_returns_sorted_paths(self, tmp_path):
        """Test that results are sorted for determinism."""
        class_dir = tmp_path / "class0"
        class_dir.mkdir()
        for name in ["c.png", "a.png", "b.png"]:
            Image.new("RGB", (32, 32)).save(class_dir / name)

        files = _scan_image_files(str(class_dir))
        assert files == sorted(files)

    def test_missing_directory_raises(self):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _scan_image_files("/nonexistent/dir")

    def test_empty_directory_raises(self, tmp_path):
        """Test that directory with no images raises ValueError."""
        class_dir = tmp_path / "empty_class"
        class_dir.mkdir()
        (class_dir / "readme.txt").write_text("not an image")

        with pytest.raises(ValueError, match="No image files found"):
            _scan_image_files(str(class_dir))


# ============================================================================
# Unit Tests - _split_list
# ============================================================================


@pytest.mark.unit
class TestSplitList:
    """Test list splitting logic."""

    def test_basic_split(self):
        """Test basic 80/20 split."""
        import random

        items = [f"item_{i}" for i in range(10)]
        rng = random.Random(42)
        train, val = _split_list(items, 0.8, rng)

        assert len(train) == 8
        assert len(val) == 2

    def test_all_items_present(self):
        """Test that all items appear in exactly one split."""
        import random

        items = [f"item_{i}" for i in range(20)]
        rng = random.Random(42)
        train, val = _split_list(items, 0.7, rng)

        combined = set(train) | set(val)
        assert combined == set(items)
        assert len(train) + len(val) == len(items)

    def test_no_overlap(self):
        """Test that train and val have no overlap."""
        import random

        items = [f"item_{i}" for i in range(15)]
        rng = random.Random(42)
        train, val = _split_list(items, 0.6, rng)

        assert set(train).isdisjoint(set(val))

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces same split."""
        import random

        items = [f"item_{i}" for i in range(10)]
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        train1, val1 = _split_list(items, 0.8, rng1)
        train2, val2 = _split_list(items, 0.8, rng2)

        assert train1 == train2
        assert val1 == val2

    def test_small_list_at_least_one_each(self):
        """Test that a 2-item list has at least 1 in each split."""
        import random

        items = ["a", "b"]
        rng = random.Random(42)
        train, val = _split_list(items, 0.9, rng)

        assert len(train) >= 1
        assert len(val) >= 1


# ============================================================================
# Component Tests - prepare_split (uses filesystem)
# ============================================================================


@pytest.mark.component
class TestPrepareSplit:
    """Test the main prepare_split function."""

    def _make_config(self, tmp_path, classes_config, **overrides):
        """Create a config dict for prepare_split.

        Args:
            tmp_path: Temporary directory
            classes_config: Dict mapping class_name to num_images
            **overrides: Override split config keys
        """
        class_paths = _create_mock_class_dirs(tmp_path / "data", classes_config)
        save_dir = str(tmp_path / "output")

        split_config = {
            "seed": 42,
            "train_ratio": 0.8,
            "save_dir": save_dir,
            "split_file": "split.json",
            "force": False,
        }
        split_config.update(overrides)

        return {
            "experiment": "data_preparation",
            "classes": class_paths,
            "split": split_config,
        }

    def test_creates_split_file(self, tmp_path):
        """Test that prepare_split creates the JSON output file."""
        config = self._make_config(tmp_path, {"normal": 5, "abnormal": 5})
        output_path = prepare_split(config)

        assert Path(output_path).exists()
        assert output_path.endswith(".json")

    def test_json_structure(self, tmp_path):
        """Test that generated JSON has correct structure."""
        config = self._make_config(tmp_path, {"normal": 10, "abnormal": 5})
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "train" in data
        assert "val" in data

        metadata = data["metadata"]
        assert "created_at" in metadata
        assert metadata["seed"] == 42
        assert metadata["train_ratio"] == 0.8
        assert metadata["total_samples"] == 15
        assert "classes" in metadata
        assert "class_samples" in metadata
        assert "source_paths" in metadata

    def test_deterministic_output(self, tmp_path):
        """Test that same seed produces identical splits."""
        config1 = self._make_config(
            tmp_path, {"normal": 10, "abnormal": 10}, force=True
        )

        output_path1 = prepare_split(config1)
        with open(output_path1) as f:
            data1 = json.load(f)

        # Run again with force=True to overwrite
        output_path2 = prepare_split(config1)
        with open(output_path2) as f:
            data2 = json.load(f)

        # Train/val lists should be identical (ignore created_at timestamp)
        assert data1["train"] == data2["train"]
        assert data1["val"] == data2["val"]

    def test_train_val_ratio(self, tmp_path):
        """Test that train/val ratio is approximately correct."""
        config = self._make_config(
            tmp_path, {"normal": 50, "abnormal": 50}, train_ratio=0.7
        )
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        total = len(data["train"]) + len(data["val"])
        assert total == 100
        # Each class: 50 * 0.7 = 35 train, 15 val => total train=70, val=30
        assert len(data["train"]) == 70
        assert len(data["val"]) == 30

    def test_all_images_appear_once(self, tmp_path):
        """Test that every image appears exactly once in train+val."""
        config = self._make_config(tmp_path, {"normal": 10, "abnormal": 10})
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        train_paths = {entry["path"] for entry in data["train"]}
        val_paths = {entry["path"] for entry in data["val"]}

        # No overlap
        assert train_paths.isdisjoint(val_paths)
        # All images present
        assert len(train_paths) + len(val_paths) == 20

    def test_labels_correct(self, tmp_path):
        """Test that labels correspond to the correct class."""
        config = self._make_config(tmp_path, {"normal": 5, "abnormal": 5})
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        classes = data["metadata"]["classes"]
        for entry in data["train"] + data["val"]:
            # The label should match the class that the path belongs to
            path = entry["path"]
            label = entry["label"]
            for class_name, class_label in classes.items():
                if class_label == label:
                    # The class path should be a prefix of the file path
                    source_path = data["metadata"]["source_paths"][class_name]
                    assert path.startswith(source_path)
                    break

    def test_skip_existing_file(self, tmp_path):
        """Test that existing file is skipped when force=false."""
        config = self._make_config(tmp_path, {"normal": 5, "abnormal": 5})

        # First run creates the file
        output_path = prepare_split(config)
        with open(output_path) as f:
            data1 = json.load(f)

        # Second run skips (force=false by default)
        output_path2 = prepare_split(config)
        assert output_path == output_path2

    def test_force_overwrites_existing(self, tmp_path):
        """Test that force=true overwrites existing file."""
        config = self._make_config(tmp_path, {"normal": 5, "abnormal": 5}, force=True)

        output_path = prepare_split(config)
        assert Path(output_path).exists()

        # Modify file content to verify overwrite
        with open(output_path, "w") as f:
            json.dump({"modified": True}, f)

        # Run again with force=True
        prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)
        assert "metadata" in data  # Original structure, not modified

    def test_missing_class_directory_raises(self, tmp_path):
        """Test that missing class directory raises error."""
        config = {
            "experiment": "data_preparation",
            "classes": {"normal": str(tmp_path / "nonexistent")},
            "split": {
                "seed": 42,
                "train_ratio": 0.8,
                "save_dir": str(tmp_path / "output"),
                "split_file": "split.json",
                "force": False,
            },
        }
        with pytest.raises(FileNotFoundError):
            prepare_split(config)

    def test_empty_class_directory_raises(self, tmp_path):
        """Test that empty class directory raises error."""
        empty_dir = tmp_path / "data" / "empty"
        empty_dir.mkdir(parents=True)

        config = {
            "experiment": "data_preparation",
            "classes": {"empty": str(empty_dir)},
            "split": {
                "seed": 42,
                "train_ratio": 0.8,
                "save_dir": str(tmp_path / "output"),
                "split_file": "split.json",
                "force": False,
            },
        }
        with pytest.raises(ValueError, match="No image files found"):
            prepare_split(config)

    def test_class_metadata_correct(self, tmp_path):
        """Test that per-class metadata is accurate."""
        config = self._make_config(tmp_path, {"normal": 8, "abnormal": 4})
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        cs = metadata["class_samples"]

        assert cs["normal"]["total"] == 8
        assert cs["abnormal"]["total"] == 4
        assert cs["normal"]["train"] + cs["normal"]["val"] == 8
        assert cs["abnormal"]["train"] + cs["abnormal"]["val"] == 4

    def test_null_seed(self, tmp_path):
        """Test that null seed produces valid output (non-deterministic)."""
        config = self._make_config(
            tmp_path, {"normal": 5, "abnormal": 5}, seed=None, force=True
        )
        output_path = prepare_split(config)

        with open(output_path) as f:
            data = json.load(f)

        assert data["metadata"]["seed"] is None
        assert len(data["train"]) + len(data["val"]) == 10

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        config = self._make_config(tmp_path, {"normal": 5, "abnormal": 5})
        config["split"]["save_dir"] = str(tmp_path / "new" / "nested" / "dir")

        output_path = prepare_split(config)
        assert Path(output_path).exists()
