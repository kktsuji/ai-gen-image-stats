"""Integration tests for main CLI entry point.

This module tests the main entry point and experiment dispatcher,
ensuring proper configuration handling, experiment routing, and
end-to-end execution flow with config-only mode.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from src.main import main, setup_experiment_classifier


class TestMainEntryPoint:
    """Test suite for main CLI entry point with config-only mode."""

    @pytest.mark.integration
    def test_main_help_message(self):
        """Test that --help flag works and displays help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        # --help should exit with code 0
        assert exc_info.value.code == 0

    @pytest.mark.integration
    def test_main_missing_required_experiment(self):
        """Test that missing config file causes error."""
        with pytest.raises(SystemExit):
            main([])

    @pytest.mark.integration
    def test_main_invalid_experiment_type(self, tmp_path):
        """Test that invalid experiment type raises ValueError."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"experiment": "invalid_experiment"}, f, default_flow_style=False)

        with pytest.raises(ValueError):
            main([str(config_file)])

    @pytest.mark.integration
    def test_main_dispatcher_classifier(self, tmp_path):
        """Test that classifier experiment is properly dispatched."""
        config_file = tmp_path / "classifier_config.yaml"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_classifier") as mock_setup:
            main([str(config_file)])

            # Verify setup function was called
            mock_setup.assert_called_once()

            # Verify config was passed correctly
            config = mock_setup.call_args[0][0]
            assert config["experiment"] == "classifier"
            assert config["model"]["name"] == "resnet50"
            assert config["training"]["epochs"] == 1
            assert config["data"]["batch_size"] == 2

    @pytest.mark.integration
    def test_main_dispatcher_gan_not_implemented(self, tmp_path):
        """Test that GAN experiment raises NotImplementedError."""
        config_file = tmp_path / "gan_config.yaml"
        config_data = {
            "experiment": "gan",
            "data": {"train_path": "tests/fixtures/mock_data/train"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with pytest.raises(NotImplementedError, match="GAN experiment"):
            main([str(config_file)])

    @pytest.mark.integration
    def test_main_config_file_loading(self, tmp_path):
        """Test that config file is properly loaded."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "inceptionv3",
                "num_classes": 2,
                "pretrained": True,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 16,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 5,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_classifier") as mock_setup:
            main([str(config_file)])

            # Verify setup was called
            mock_setup.assert_called_once()

            # Verify config was loaded correctly
            config = mock_setup.call_args[0][0]
            assert config["model"]["name"] == "inceptionv3"
            assert config["training"]["epochs"] == 5

    @pytest.mark.integration
    def test_main_cli_overrides_config_file(self, tmp_path):
        """Test that CLI overrides are NOT supported in config-only mode."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 32,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Attempting to add CLI overrides should fail
        with pytest.raises(SystemExit):
            main([str(config_file), "--epochs", "5"])


class TestClassifierExperimentSetup:
    """Test suite for classifier experiment setup."""

    @pytest.mark.integration
    def test_setup_classifier_basic(self, tmp_path):
        """Test basic classifier setup with minimal config."""
        config = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
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
            assert (tmp_path / "logs" / "config.yaml").exists()

    @pytest.mark.integration
    def test_setup_classifier_inceptionv3(self, tmp_path):
        """Test classifier setup with InceptionV3 model."""
        config = {
            "experiment": "classifier",
            "model": {
                "name": "inceptionv3",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
                "dropout": 0.5,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
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
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.001,
                "optimizer": "adam",
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
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adam",
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
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
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
            },
        }

        with pytest.raises(ValueError, match="Invalid model name"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_invalid_optimizer(self, tmp_path):
        """Test that invalid optimizer raises ValueError."""
        config = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
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
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "val_path": "tests/fixtures/mock_data/val",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adam",
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
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
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
        assert (tmp_path / "logs" / "config.yaml").exists()
        assert (tmp_path / "logs" / "metrics.csv").exists()

        # Verify checkpoint was saved
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.pth"))
        assert len(checkpoint_files) > 0


class TestExperimentDispatcher:
    """Test suite for experiment dispatcher logic with config-only mode."""

    @pytest.mark.unit
    def test_dispatcher_routes_to_classifier(self, tmp_path):
        """Test that dispatcher correctly routes to classifier."""
        config_file = tmp_path / "classifier_config.yaml"
        config_data = {
            "experiment": "classifier",
            "model": {
                "name": "resnet50",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": 256,
                "crop_size": 224,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_classifier") as mock_classifier:
            main([str(config_file)])

            mock_classifier.assert_called_once()

    @pytest.mark.unit
    def test_dispatcher_routes_to_diffusion(self, tmp_path):
        """Test that dispatcher correctly routes to diffusion."""
        config_file = tmp_path / "diffusion_config.yaml"
        config_data = {
            "experiment": "diffusion",
            "model": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 64,
                "num_timesteps": 1000,
                "beta_schedule": "linear",
                "channel_multipliers": [1, 2, 4],
                "num_classes": 2,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "class_dropout_prob": 0.1,
                "use_attention": [False, True, False],
            },
            "data": {
                "train_path": "tests/fixtures/mock_data/train",
                "batch_size": 16,
                "num_workers": 0,
                "image_size": 32,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "device": "cpu",
            },
            "output": {
                "checkpoint_dir": "outputs/checkpoints",
                "log_dir": "outputs/logs",
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_diffusion") as mock_diffusion:
            main([str(config_file)])

            mock_diffusion.assert_called_once()

    @pytest.mark.unit
    def test_dispatcher_routes_to_gan(self, tmp_path):
        """Test that dispatcher correctly routes to GAN (not implemented)."""
        config_file = tmp_path / "gan_config.yaml"
        config_data = {
            "experiment": "gan",
            "data": {"train_path": "tests/fixtures/mock_data/train"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with pytest.raises(NotImplementedError):
            main([str(config_file)])
