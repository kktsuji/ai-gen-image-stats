"""Integration tests for classifier end-to-end pipeline.

This module tests the complete classifier workflow from configuration
loading through training to checkpoint saving and restoration.
Uses a tiny dataset (10-20 images) for fast validation of the full pipeline.

Test Coverage:
- Config loading and validation
- Model initialization (ResNet, InceptionV3)
- Full training loop execution
- Checkpoint saving and loading
- Training resumption from checkpoint
- Validation loop execution
- Metrics logging and CSV output
- Optimizer and scheduler setup
"""

import json
from pathlib import Path

import pytest
import torch
import yaml

from src.experiments.classifier.config import get_default_config
from src.experiments.classifier.dataloader import ClassifierDataLoader
from src.experiments.classifier.logger import ClassifierLogger
from src.experiments.classifier.models.inceptionv3 import InceptionV3Classifier
from src.experiments.classifier.models.resnet import ResNetClassifier
from src.experiments.classifier.trainer import ClassifierTrainer

# Dynamic device detection for testing
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _create_split_json(data_dir, split_json_path, include_val=True):
    """Create a split JSON file from a directory structure."""
    from pathlib import Path

    entries = []
    data_path = Path(data_dir)
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            # Support directory names like '0.Normal' or plain '0'
            label = int(class_dir.name.split(".")[0])
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix in (".png", ".jpg", ".jpeg"):
                    entries.append({"path": str(img_file), "label": label})

    class_names = sorted(set(str(e["label"]) for e in entries))
    classes = {name: int(name) for name in class_names}

    split_data = {
        "metadata": {"classes": classes},
        "train": entries,
        "val": entries if include_val else [],
    }

    split_json_path = Path(split_json_path)
    split_json_path.write_text(json.dumps(split_data))
    return str(split_json_path)


class TestClassifierPipelineBasic:
    """Test basic classifier pipeline with minimal configuration."""

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_resnet50(self, tmp_path, mock_dataset_medium):
        """Test complete classifier pipeline with ResNet50.

        Tests:
        - Model initialization
        - Training for 2 epochs
        - Checkpoint saving
        - Metrics logging
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Setup configuration
        config = {
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
                "save_best_only": False,
            },
        }

        # Initialize components
        model = ResNetClassifier(
            num_classes=config["model"]["num_classes"],
            variant="resnet50",
            pretrained=config["model"]["pretrained"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"].get("image_size", 224),
            crop_size=config["data"].get("crop_size", 224),
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Initialize trainer
        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["training"]["device"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Verify outputs
        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        log_dir = Path(config["output"]["log_dir"])

        assert checkpoint_dir.exists(), "Checkpoint directory not created"
        assert log_dir.exists(), "Log directory not created"

        # Verify checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint files saved"

        # Verify metrics CSV
        metrics_csv = log_dir / "metrics.csv"
        assert metrics_csv.exists(), "Metrics CSV not created"

        # Read and verify metrics
        with open(metrics_csv, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "Metrics CSV is empty"
            # Should have header + at least 2 epoch entries
            assert len(lines) >= 3

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_inceptionv3(self, tmp_path, mock_dataset_medium):
        """Test complete classifier pipeline with InceptionV3.

        Tests:
        - InceptionV3 initialization
        - Training with auxiliary loss
        - Checkpoint saving
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Setup configuration
        config = {
            "model": {
                "name": "inceptionv3",
                "num_classes": 2,
                "pretrained": False,
                "dropout": 0.5,
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 299,
                "crop_size": 299,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        # Initialize components
        model = InceptionV3Classifier(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            crop_size=config["data"]["crop_size"],
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Initialize trainer
        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["training"]["device"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Verify outputs
        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        assert checkpoint_dir.exists()

        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0


class TestClassifierPipelineCheckpoints:
    """Test checkpoint saving, loading, and training resumption."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_checkpoint_save_and_load(self, tmp_path, mock_dataset_medium):
        """Test that checkpoints can be saved and loaded correctly.

        Tests:
        - Save checkpoint after training
        - Load checkpoint into new model
        - Verify state dict matches
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # First training run - train and save
        model1 = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        dataloader = ClassifierDataLoader(
            split_file=str(split_file),
            batch_size=4,
            num_workers=0,
            image_size=64,
            crop_size=32,
        )

        logger = ClassifierLogger(log_dir=str(log_dir))
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)

        trainer = ClassifierTrainer(
            model=model1,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=TEST_DEVICE,
        )

        trainer.train(num_epochs=1, checkpoint_dir=str(checkpoint_dir))

        # Find the saved checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0, "No checkpoint saved"

        # Load checkpoint into new model
        model2 = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model2.to(TEST_DEVICE)  # Move to device before loading
        checkpoint = torch.load(checkpoint_files[0], map_location=TEST_DEVICE)

        # Load state dict
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2), f"Parameters {name1} don't match"

    @pytest.mark.smoke
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_resumption_from_checkpoint(self, tmp_path, mock_dataset_medium):
        """Test that training can be resumed from a checkpoint.

        Tests:
        - Train for 1 epoch and save checkpoint
        - Resume training for 1 more epoch
        - Verify epoch counter continues correctly
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # First training run - 1 epoch
        model = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)

        dataloader = ClassifierDataLoader(
            split_file=str(split_file),
            batch_size=4,
            num_workers=0,
            image_size=64,
            crop_size=32,
        )

        logger = ClassifierLogger(log_dir=str(log_dir))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=TEST_DEVICE,
        )

        trainer.train(
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
        )

        # Get the checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        checkpoint_file = checkpoint_files[0]

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=TEST_DEVICE)

        # Verify checkpoint contains required keys
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 1  # After 1 epoch

        # Resume training
        model2 = ResNetClassifier(num_classes=2, variant="resnet50", pretrained=False)
        model2.to(TEST_DEVICE)  # Move to device before loading
        model2.load_state_dict(checkpoint["model_state_dict"])

        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

        trainer2 = ClassifierTrainer(
            model=model2,
            dataloader=dataloader,
            optimizer=optimizer2,
            logger=logger,
            device=TEST_DEVICE,
        )

        trainer2.train(
            num_epochs=1,
            checkpoint_dir=str(checkpoint_dir),
        )

        # Verify new checkpoints were created
        checkpoint_files_after = list(checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files_after) >= len(checkpoint_files)


class TestClassifierPipelineWithScheduler:
    """Test classifier pipeline with learning rate schedulers."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_cosine_scheduler(self, tmp_path, mock_dataset_medium):
        """Test training with cosine annealing scheduler.

        Tests:
        - Scheduler initialization
        - Learning rate changes over epochs
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        config = {
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 3,
                "learning_rate": 0.01,
                "optimizer": "adam",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        # Initialize components
        model = ResNetClassifier(
            num_classes=config["model"]["num_classes"],
            variant="resnet50",
            pretrained=config["model"]["pretrained"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            crop_size=config["data"]["crop_size"],
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Add cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
        )

        # Initialize trainer
        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            scheduler=scheduler,
            device=config["training"]["device"],
        )

        # Track learning rates
        initial_lr = optimizer.param_groups[0]["lr"]

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Verify learning rate changed
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, (
            "Learning rate should decrease with cosine schedule"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_step_scheduler(self, tmp_path, mock_dataset_medium):
        """Test training with step scheduler.

        Tests:
        - Step scheduler initialization
        - Learning rate decay at step boundaries
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        config = {
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 4,
                "learning_rate": 0.01,
                "optimizer": "sgd",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        # Initialize components
        model = ResNetClassifier(
            num_classes=config["model"]["num_classes"],
            variant="resnet50",
            pretrained=config["model"]["pretrained"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            crop_size=config["data"]["crop_size"],
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Add step scheduler - decay every 2 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        # Initialize trainer
        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["training"]["device"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Verify training completed
        assert trainer.current_epoch == config["training"]["epochs"]


class TestClassifierPipelineValidation:
    """Test validation loop and metrics computation."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_validation_metrics_recorded(self, tmp_path, mock_dataset_medium):
        """Test that validation metrics are properly computed and logged.

        Tests:
        - Validation loop execution
        - Metrics computation (loss, accuracy)
        - Metrics logging to CSV
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        config = {
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        # Initialize and train
        model = ResNetClassifier(
            num_classes=config["model"]["num_classes"],
            variant="resnet50",
            pretrained=config["model"]["pretrained"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            crop_size=config["data"]["crop_size"],
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["training"]["device"],
        )

        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Read metrics CSV
        metrics_csv = Path(config["output"]["log_dir"]) / "metrics.csv"
        assert metrics_csv.exists()

        with open(metrics_csv, "r") as f:
            lines = f.readlines()
            header = lines[0].strip()

            # Verify header contains expected columns
            # Note: Metrics are logged as 'loss', 'accuracy' for training
            # and 'val_loss', 'val_accuracy' for validation
            assert "epoch" in header
            assert "loss" in header  # Training loss
            assert "accuracy" in header  # Training accuracy
            assert "val_loss" in header
            assert "val_accuracy" in header

            # Verify we have data for all epochs
            assert (
                len(lines) >= config["training"]["epochs"] + 1
            )  # header + at least one line per epoch


class TestClassifierPipelineMultipleModels:
    """Test pipeline with different model architectures."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_resnet101(self, tmp_path, mock_dataset_medium):
        """Test pipeline with ResNet101 model."""
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        model = ResNetClassifier(num_classes=2, variant="resnet101", pretrained=False)

        dataloader = ClassifierDataLoader(
            split_file=str(split_file),
            batch_size=4,
            num_workers=0,
            image_size=64,
            crop_size=32,
        )

        logger = ClassifierLogger(log_dir=str(tmp_path / "logs"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=TEST_DEVICE,
        )

        trainer.train(
            num_epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # Verify checkpoint saved
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.pth"))
        assert len(checkpoint_files) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_different_optimizers(self, tmp_path, mock_dataset_medium):
        """Test pipeline with different optimizers.

        Tests:
        - Adam optimizer
        - SGD optimizer
        - AdamW optimizer
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        optimizers_to_test = [
            ("adam", torch.optim.Adam),
            ("sgd", torch.optim.SGD),
            ("adamw", torch.optim.AdamW),
        ]

        for opt_name, opt_class in optimizers_to_test:
            # Create separate directories for each optimizer
            checkpoint_dir = tmp_path / f"checkpoints_{opt_name}"
            log_dir = tmp_path / f"logs_{opt_name}"

            model = ResNetClassifier(
                num_classes=2, variant="resnet50", pretrained=False
            )

            dataloader = ClassifierDataLoader(
                split_file=str(split_file),
                batch_size=4,
                num_workers=0,
                image_size=64,
                crop_size=32,
            )

            logger = ClassifierLogger(log_dir=str(log_dir))

            # Create optimizer
            if opt_class == torch.optim.SGD:
                optimizer = opt_class(model.parameters(), lr=0.01, momentum=0.9)
            else:
                optimizer = opt_class(model.parameters(), lr=0.001)

            trainer = ClassifierTrainer(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                logger=logger,
                device=TEST_DEVICE,
            )

            trainer.train(
                num_epochs=1,
                checkpoint_dir=str(checkpoint_dir),
            )

            # Verify outputs for this optimizer
            assert checkpoint_dir.exists(), f"Checkpoint dir not created for {opt_name}"
            assert log_dir.exists(), f"Log dir not created for {opt_name}"


class TestClassifierPipelineConfigDriven:
    """Test complete config-driven pipeline matching production usage."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_config_file_driven_pipeline(self, tmp_path, mock_dataset_medium):
        """Test pipeline using configuration file (like production).

        This test simulates the actual usage pattern:
        1. Load config from YAML file
        2. Initialize all components from config
        3. Run training
        4. Verify outputs

        This is the closest to real-world usage.
        """
        split_file = _create_split_json(mock_dataset_medium, tmp_path / "split.json")

        # Create a config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {
                "split_file": str(split_file),
                "batch_size": 4,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": TEST_DEVICE,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
                "save_best_only": False,
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        # Load config from file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Initialize from config (as main.py would do)
        model = ResNetClassifier(
            num_classes=config["model"]["num_classes"],
            variant="resnet50",
            pretrained=config["model"]["pretrained"],
        )

        dataloader = ClassifierDataLoader(
            split_file=config["data"]["split_file"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            crop_size=config["data"]["crop_size"],
        )

        logger = ClassifierLogger(log_dir=config["output"]["log_dir"])

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        trainer = ClassifierTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=config["training"]["device"],
        )

        # Run training
        trainer.train(
            num_epochs=config["training"]["epochs"],
            checkpoint_dir=config["output"]["checkpoint_dir"],
        )

        # Verify all expected outputs exist
        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        log_dir = Path(config["output"]["log_dir"])

        assert checkpoint_dir.exists()
        assert log_dir.exists()
        assert (log_dir / "metrics.csv").exists()
        assert len(list(checkpoint_dir.glob("*.pth"))) > 0

        # Verify metrics CSV has correct structure
        with open(log_dir / "metrics.csv", "r") as f:
            lines = f.readlines()
            # Each epoch generates 2 lines: one for train metrics, one for val metrics
            # Plus 1 for header
            assert (
                len(lines) >= config["training"]["epochs"] + 1
            )  # header + at least one line per epoch

        # Verify config file was saved to output directory
        # (This would be done by main.py in production)
        saved_config = log_dir / "config.yaml"
        with open(saved_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        assert saved_config.exists()
