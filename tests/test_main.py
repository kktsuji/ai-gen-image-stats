"""Integration tests for main CLI entry point.

This module tests the main entry point and experiment dispatcher,
ensuring proper configuration handling, experiment routing, and
end-to-end execution flow.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.main import main, setup_experiment_classifier


class TestMainEntryPoint:
    """Test suite for main CLI entry point."""

    @pytest.mark.integration
    def test_main_help_message(self):
        """Test that --help flag works and displays help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        # --help should exit with code 0
        assert exc_info.value.code == 0

    @pytest.mark.integration
    def test_main_missing_required_experiment(self):
        """Test that missing --experiment argument causes error."""
        with pytest.raises(SystemExit):
            main([])

    @pytest.mark.integration
    def test_main_invalid_experiment_type(self):
        """Test that invalid experiment type raises ValueError."""
        with pytest.raises(SystemExit):
            main(["--experiment", "invalid_experiment"])

    @pytest.mark.integration
    def test_main_dispatcher_classifier(self):
        """Test that classifier experiment is properly dispatched."""
        with patch("src.main.setup_experiment_classifier") as mock_setup:
            main(
                [
                    "--experiment",
                    "classifier",
                    "--model",
                    "resnet50",
                    "--train-path",
                    "tests/fixtures/mock_data/train",
                    "--val-path",
                    "tests/fixtures/mock_data/val",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                ]
            )

            # Verify setup function was called
            mock_setup.assert_called_once()

            # Verify config was passed correctly
            config = mock_setup.call_args[0][0]
            assert config["experiment"] == "classifier"
            assert config["model"]["name"] == "resnet50"
            assert config["training"]["epochs"] == 1
            assert config["data"]["batch_size"] == 2

    @pytest.mark.integration
    def test_main_dispatcher_gan_not_implemented(self):
        """Test that GAN experiment raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="GAN experiment"):
            main(
                [
                    "--experiment",
                    "gan",
                    "--train-path",
                    "tests/fixtures/mock_data/train",
                ]
            )

    @pytest.mark.integration
    def test_main_config_file_loading(self, tmp_path):
        """Test that config file is properly loaded and merged."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "inceptionv3",
                "num_classes": 2,
                "pretrained": True,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 16,
            },
            "training": {"epochs": 5, "learning_rate": 0.001},
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch("src.main.setup_experiment_classifier") as mock_setup:
            main(["--experiment", "classifier", "--config", str(config_file)])

            # Verify setup was called
            mock_setup.assert_called_once()

            # Verify config was loaded correctly
            config = mock_setup.call_args[0][0]
            assert config["model"]["name"] == "inceptionv3"
            assert config["training"]["epochs"] == 5

    @pytest.mark.integration
    def test_main_cli_overrides_config_file(self, tmp_path):
        """Test that CLI arguments override config file values."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "experiment": "classifier",
            "model": {"name": "resnet50"},
            "training": {"epochs": 100, "learning_rate": 0.001},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 32,
            },
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch("src.main.setup_experiment_classifier") as mock_setup:
            main(
                [
                    "--experiment",
                    "classifier",
                    "--config",
                    str(config_file),
                    "--epochs",
                    "5",
                    "--batch-size",
                    "4",
                ]
            )

            # Verify setup was called
            mock_setup.assert_called_once()

            # Verify CLI arguments override config file
            config = mock_setup.call_args[0][0]
            assert config["training"]["epochs"] == 5  # Overridden by CLI
            assert config["data"]["batch_size"] == 4  # Overridden by CLI
            assert (
                config["training"]["learning_rate"] == 0.001
            )  # Not overridden, from config file


class TestClassifierExperimentSetup:
    """Test suite for classifier experiment setup."""

    @pytest.mark.integration
    def test_setup_classifier_basic(self, tmp_path):
        """Test basic classifier setup with minimal config."""
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {"epochs": 1, "learning_rate": 0.001, "device": "cpu"},
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        # Mock the trainer to avoid actual training
        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train"
        ) as mock_train:
            setup_experiment_classifier(config)

            # Verify training was called
            mock_train.assert_called_once()

            # Check that output directories were created
            assert (tmp_path / "checkpoints").exists()
            assert (tmp_path / "logs").exists()
            assert (tmp_path / "logs" / "config.json").exists()

    @pytest.mark.integration
    def test_setup_classifier_inceptionv3(self, tmp_path):
        """Test classifier setup with InceptionV3 model."""
        config = {
            "experiment": "classifier",
            "model": {
                "name": "inceptionv3",
                "num_classes": 2,
                "pretrained": False,
                "dropout": 0.5,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {"epochs": 1, "learning_rate": 0.001, "device": "cpu"},
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train"
        ) as mock_train:
            setup_experiment_classifier(config)

            # Verify training was called
            mock_train.assert_called_once()

    @pytest.mark.integration
    def test_setup_classifier_with_scheduler(self, tmp_path):
        """Test classifier setup with learning rate scheduler."""
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.001,
                "device": "cpu",
                "scheduler": "cosine",
                "scheduler_kwargs": {"T_max": 10, "eta_min": 1e-6},
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train"
        ) as mock_train:
            setup_experiment_classifier(config)

            # Verify training was called
            mock_train.assert_called_once()

    @pytest.mark.integration
    def test_setup_classifier_with_seed(self, tmp_path):
        """Test that random seed is properly set."""
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "device": "cpu",
                "seed": 42,
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with (
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
            patch("torch.manual_seed") as mock_manual_seed,
        ):
            setup_experiment_classifier(config)

            # Verify seed was set
            mock_manual_seed.assert_called_once_with(42)

            # Verify training was called
            mock_train.assert_called_once()

    @pytest.mark.integration
    def test_setup_classifier_invalid_model(self, tmp_path):
        """Test that invalid model name raises ValueError."""
        config = {
            "experiment": "classifier",
            "model": {
                "name": "invalid_model",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {"epochs": 1, "learning_rate": 0.001, "device": "cpu"},
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with pytest.raises(ValueError, match="Invalid model name"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_invalid_optimizer(self, tmp_path):
        """Test that invalid optimizer raises ValueError."""
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "invalid_optimizer",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with pytest.raises(ValueError, match="Invalid optimizer"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_invalid_scheduler(self, tmp_path):
        """Test that invalid scheduler raises ValueError."""
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "scheduler": "invalid_scheduler",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            },
        }

        with pytest.raises(ValueError, match="Invalid scheduler"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_setup_classifier_full_execution_mini(self, tmp_path):
        """Test full execution of classifier training with tiny dataset.

        This test actually runs the training loop for 1 epoch on CPU
        with a minimal configuration to verify end-to-end functionality.
        """
        config = {
            "experiment": "classifier",
            "model": {"name": "resnet50", "num_classes": 2, "pretrained": False},
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 64,
                "crop_size": 32,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
                "save_best_only": True,
            },
        }

        # Actually run the training (no mocking)
        setup_experiment_classifier(config)

        # Verify outputs were created
        assert (tmp_path / "checkpoints").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "logs" / "config.json").exists()
        assert (tmp_path / "logs" / "metrics.csv").exists()

        # Verify checkpoint was saved
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.pth"))
        assert len(checkpoint_files) > 0


class TestExperimentDispatcher:
    """Test suite for experiment dispatcher logic."""

    @pytest.mark.unit
    def test_dispatcher_routes_to_classifier(self):
        """Test that dispatcher correctly routes to classifier."""
        with patch("src.main.setup_experiment_classifier") as mock_classifier:
            main(
                [
                    "--experiment",
                    "classifier",
                    "--train-path",
                    "tests/fixtures/mock_data/train",
                    "--epochs",
                    "1",
                ]
            )

            mock_classifier.assert_called_once()

    @pytest.mark.unit
    def test_dispatcher_routes_to_diffusion(self):
        """Test that dispatcher correctly routes to diffusion."""
        with patch("src.main.setup_experiment_diffusion") as mock_diffusion:
            main(
                [
                    "--experiment",
                    "diffusion",
                    "--train-path",
                    "tests/fixtures/mock_data/train",
                ]
            )

            mock_diffusion.assert_called_once()

    @pytest.mark.unit
    def test_dispatcher_routes_to_gan(self):
        """Test that dispatcher correctly routes to GAN (not implemented)."""
        with pytest.raises(NotImplementedError):
            main(
                [
                    "--experiment",
                    "gan",
                    "--train-path",
                    "tests/fixtures/mock_data/train",
                ]
            )
