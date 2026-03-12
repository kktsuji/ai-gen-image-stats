"""Integration tests for data preparation end-to-end pipeline.

This module tests the complete data preparation workflow from configuration
validation through split generation to downstream dataset compatibility.

Test Coverage:
- Config validation
- Split file generation with prepare_split()
- Output JSON structure and correctness
- Train/val ratio enforcement
- Downstream compatibility with SplitFileDataset and DataLoader
"""

import json
import math
from pathlib import Path

import pytest
import torch
from torchvision import transforms

from src.data.datasets import SplitFileDataset
from src.experiments.data_preparation.config import validate_config
from src.experiments.data_preparation.prepare import prepare_split


class TestDataPreparationPipeline:
    """Test complete data preparation pipeline."""

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_preparation_and_downstream_usage(self, tmp_path, mock_dataset_medium):
        """Test complete data preparation pipeline with downstream dataset usage.

        Tests:
        - Config validation
        - Split file generation via prepare_split()
        - Output JSON structure (metadata, train, val)
        - Class mappings correctness
        - Train/val counts match train_ratio
        - All image paths in JSON exist on disk
        - Downstream: SplitFileDataset loads the JSON
        - Downstream: DataLoader iterates one batch successfully
        """
        # Build config matching data_preparation schema
        train_ratio = 0.8
        config = {
            "experiment": "data_preparation",
            "classes": {
                "normal": str(mock_dataset_medium / "0.Normal"),
                "abnormal": str(mock_dataset_medium / "1.Abnormal"),
            },
            "split": {
                "seed": 42,
                "train_ratio": train_ratio,
                "save_dir": str(tmp_path / "splits"),
                "split_file": "train_val_split.json",
                "force": True,
            },
        }

        # Step 1: Validate config
        validate_config(config)

        # Step 2: Run prepare_split
        output_path = prepare_split(config)
        assert Path(output_path).exists(), "Split file was not created"

        # Step 3: Load and validate JSON structure
        with open(output_path, "r") as f:
            split_data = json.load(f)

        assert "metadata" in split_data, "Missing 'metadata' key"
        assert "train" in split_data, "Missing 'train' key"
        assert "val" in split_data, "Missing 'val' key"

        metadata = split_data["metadata"]
        assert "classes" in metadata, "Missing 'classes' in metadata"
        assert "train_ratio" in metadata
        assert metadata["train_ratio"] == train_ratio

        # Step 4: Validate class mappings
        classes = metadata["classes"]
        assert len(classes) == 2, f"Expected 2 classes, got {len(classes)}"
        assert "abnormal" in classes
        assert "normal" in classes
        # Labels should be 0-indexed integers
        assert set(classes.values()) == {0, 1}

        # Step 5: Validate train/val counts match train_ratio
        train_entries = split_data["train"]
        val_entries = split_data["val"]
        total = len(train_entries) + len(val_entries)

        # mock_dataset_medium has 10 images per class = 20 total
        assert total == 20, f"Expected 20 total samples, got {total}"

        # Each class has 10 images, split at 0.8 → ~8 train + ~2 val per class
        # prepare_split() uses floor-based splitting, so math.floor matches
        expected_train_per_class = math.floor(10 * train_ratio)
        expected_val_per_class = 10 - expected_train_per_class
        expected_train = expected_train_per_class * 2
        expected_val = expected_val_per_class * 2

        assert len(train_entries) == expected_train, (
            f"Expected {expected_train} train samples, got {len(train_entries)}"
        )
        assert len(val_entries) == expected_val, (
            f"Expected {expected_val} val samples, got {len(val_entries)}"
        )

        # Step 6: Verify all paths in JSON exist on disk
        for entry in train_entries + val_entries:
            assert "path" in entry, "Entry missing 'path' key"
            assert "label" in entry, "Entry missing 'label' key"
            assert Path(entry["path"]).exists(), (
                f"Image path does not exist: {entry['path']}"
            )

        # Step 7: Load with SplitFileDataset (downstream compatibility)
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = SplitFileDataset(
            split_file=output_path,
            split="train",
            transform=transform,
            return_labels=True,
        )
        assert len(train_dataset) == expected_train

        val_dataset = SplitFileDataset(
            split_file=output_path,
            split="val",
            transform=transform,
            return_labels=True,
        )
        assert len(val_dataset) == expected_val

        # Verify class metadata propagated correctly
        assert len(train_dataset.get_classes()) == 2

        # Step 8: Create DataLoader and iterate one batch
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=False, num_workers=0
        )

        batch = next(iter(dataloader))
        images, labels = batch
        assert images.shape == (4, 3, 32, 32), f"Unexpected batch shape: {images.shape}"
        assert labels.shape == (4,), f"Unexpected labels shape: {labels.shape}"
        assert labels.dtype == torch.long
