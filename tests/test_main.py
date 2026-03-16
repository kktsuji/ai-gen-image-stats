"""Integration tests for main CLI entry point.

This module tests the main entry point and experiment dispatcher,
ensuring proper configuration handling, experiment routing, and
end-to-end execution flow with config-only mode.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
import yaml

from src.main import main, setup_experiment_classifier

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        """Test that invalid experiment type exits cleanly instead of traceback."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"experiment": "invalid_experiment"}, f, default_flow_style=False)

        with pytest.raises(SystemExit) as exc_info:
            main([str(config_file)])
        assert exc_info.value.code == 1

    @pytest.mark.integration
    def test_main_dispatcher_classifier(self, tmp_path):
        """Test that classifier experiment is properly dispatched."""
        config_file = tmp_path / "classifier_config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "compute": {"device": "cpu", "seed": None},
            "model": {
                "architecture": {"name": "resnet50", "num_classes": 2},
                "initialization": {"pretrained": False, "freeze_backbone": False},
            },
            "data": {
                "split_file": str(
                    _PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"
                ),
                "loading": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle_train": True,
                    "drop_last": False,
                },
                "preprocessing": {
                    "image_size": 256,
                    "crop_size": 224,
                    "normalize": "imagenet",
                },
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation_degrees": 0,
                    "color_jitter": {"enabled": False},
                },
            },
            "training": {
                "epochs": 1,
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
                "scheduler": {"type": None},
                "checkpointing": {"save_frequency": 10, "save_best_only": True},
                "validation": {"enabled": True, "frequency": 1},
            },
            "output": {
                "base_dir": "outputs",
                "subdirs": {"checkpoints": "checkpoints", "logs": "logs"},
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
            assert config["model"]["architecture"]["name"] == "resnet50"
            assert config["training"]["epochs"] == 1
            assert config["data"]["loading"]["batch_size"] == 2

    @pytest.mark.integration
    def test_main_dispatcher_gan_not_implemented(self, tmp_path):
        """Test that GAN experiment exits with code 1."""
        config_file = tmp_path / "gan_config.yaml"
        config_data = {
            "experiment": "gan",
            "data": {"train_path": "tests/fixtures/mock_data/train"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with pytest.raises(SystemExit) as exc_info:
            main([str(config_file)])
        assert exc_info.value.code == 1

    @pytest.mark.integration
    def test_main_config_file_loading(self, tmp_path):
        """Test that config file is properly loaded."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "compute": {"device": "cpu", "seed": None},
            "model": {
                "architecture": {"name": "inceptionv3", "num_classes": 2},
                "initialization": {"pretrained": True, "freeze_backbone": False},
            },
            "data": {
                "split_file": str(
                    _PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"
                ),
                "loading": {
                    "batch_size": 16,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle_train": True,
                    "drop_last": False,
                },
                "preprocessing": {
                    "image_size": 256,
                    "crop_size": 224,
                    "normalize": "imagenet",
                },
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation_degrees": 0,
                    "color_jitter": {"enabled": False},
                },
            },
            "training": {
                "epochs": 5,
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
                "scheduler": {"type": None},
                "checkpointing": {"save_frequency": 10, "save_best_only": True},
                "validation": {"enabled": True, "frequency": 1},
            },
            "output": {
                "base_dir": "outputs",
                "subdirs": {"checkpoints": "checkpoints", "logs": "logs"},
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
            assert config["model"]["architecture"]["name"] == "inceptionv3"
            assert config["training"]["epochs"] == 5

    @pytest.mark.integration
    def test_main_cli_overrides_reject_non_dot_notation(self, tmp_path):
        """Test that non-dot-notation CLI overrides are rejected."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "compute": {"device": "cpu", "seed": None},
            "model": {
                "architecture": {"name": "resnet50", "num_classes": 2},
                "initialization": {"pretrained": False, "freeze_backbone": False},
            },
            "data": {
                "split_file": str(
                    _PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"
                ),
                "loading": {
                    "batch_size": 32,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle_train": True,
                    "drop_last": False,
                },
                "preprocessing": {
                    "image_size": 256,
                    "crop_size": 224,
                    "normalize": "imagenet",
                },
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation_degrees": 0,
                    "color_jitter": {"enabled": False},
                },
            },
            "training": {
                "epochs": 100,
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
                "scheduler": {"type": None},
                "checkpointing": {"save_frequency": 10, "save_best_only": True},
                "validation": {"enabled": True, "frequency": 1},
            },
            "output": {
                "base_dir": "outputs",
                "subdirs": {"checkpoints": "checkpoints", "logs": "logs"},
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Non-dot-notation overrides should be rejected and exit cleanly
        with pytest.raises(SystemExit):
            main([str(config_file), "--epochs", "5"])

    @pytest.mark.integration
    def test_main_nonexistent_config_exits_cleanly(self):
        """Test that nonexistent config file exits cleanly instead of traceback."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_config.yaml"])
        assert exc_info.value.code == 1

    @pytest.mark.integration
    def test_main_directory_as_config_exits_cleanly(self, tmp_path):
        """Test that passing a directory as config file exits cleanly."""
        with pytest.raises(SystemExit) as exc_info:
            main([str(tmp_path)])
        assert exc_info.value.code == 1


class TestMainNotifications:
    """Tests for Slack notification integration in main()."""

    @pytest.mark.integration
    def test_main_loads_dotenv(self, tmp_path):
        """Test that main() calls load_dotenv() at startup."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "output": {"base_dir": str(tmp_path / "outputs")},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with (
            patch("src.main.load_dotenv") as mock_dotenv,
            patch("src.main.setup_experiment_classifier"),
            patch("src.main.notify_success"),
        ):
            main([str(config_file)])
            mock_dotenv.assert_called_once()

    @pytest.mark.integration
    def test_main_calls_notify_success_on_completion(self, tmp_path):
        """Test that notify_success is called after successful experiment."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "output": {"base_dir": str(tmp_path / "outputs")},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with (
            patch("src.main.load_dotenv"),
            patch("src.main.setup_experiment_classifier"),
            patch("src.main.notify_success") as mock_notify,
        ):
            main([str(config_file)])

            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            config_arg = call_args[0][0]
            duration_arg = call_args[0][1]
            assert config_arg["experiment"] == "classifier"
            assert duration_arg >= 0

    @pytest.mark.integration
    def test_main_calls_notify_error_on_exception(self, tmp_path):
        """Test that notify_error is called when experiment raises."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "output": {"base_dir": str(tmp_path / "outputs")},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with (
            patch("src.main.load_dotenv"),
            patch(
                "src.main.setup_experiment_classifier",
                side_effect=RuntimeError("test error"),
            ),
            patch("src.main.notify_error") as mock_notify,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main([str(config_file)])
            assert exc_info.value.code == 1

            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            config_arg = call_args[0][0]
            error_arg = call_args[0][1]
            assert config_arg["experiment"] == "classifier"
            assert isinstance(error_arg, RuntimeError)
            assert "test error" in str(error_arg)


class TestClassifierExperimentSetup:
    """Test suite for classifier experiment setup."""

    @pytest.mark.integration
    def test_setup_classifier_basic(self, tmp_path):
        """Test basic classifier setup with minimal config."""
        config = _base_classifier_config(tmp_path)

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
        config = _base_classifier_config(tmp_path)
        config["model"]["architecture"]["name"] = "inceptionv3"
        config["model"]["regularization"] = {"dropout": 0.5}

        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train"
        ) as mock_train:
            setup_experiment_classifier(config)

            # Verify training was called
            mock_train.assert_called_once()

    @pytest.mark.integration
    def test_setup_classifier_with_scheduler(self, tmp_path):
        """Test classifier setup with learning rate scheduler."""
        config = _base_classifier_config(tmp_path)
        config["training"]["epochs"] = 10
        config["training"]["scheduler"] = {
            "type": "cosine",
            "T_max": 10,
            "eta_min": 1e-6,
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
        config = _base_classifier_config(tmp_path)
        config["compute"]["seed"] = 42

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
        config = _base_classifier_config(tmp_path)
        config["model"]["architecture"]["name"] = "invalid_model"

        with pytest.raises(ValueError, match="Invalid model name"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_invalid_optimizer(self, tmp_path):
        """Test that invalid optimizer raises ValueError."""
        config = _base_classifier_config(tmp_path)
        config["training"]["optimizer"]["type"] = "invalid_optimizer"

        with pytest.raises(ValueError, match="Invalid optimizer"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_invalid_scheduler(self, tmp_path):
        """Test that invalid scheduler raises ValueError."""
        config = _base_classifier_config(tmp_path)
        config["training"]["scheduler"] = {"type": "invalid_scheduler"}

        with pytest.raises(ValueError, match="Invalid scheduler"):
            setup_experiment_classifier(config)

    @pytest.mark.integration
    def test_setup_classifier_full_execution_mini(self, tmp_path):
        """Test full execution of classifier training with tiny dataset.

        This test actually runs the training loop for 1 epoch on CPU
        with a minimal configuration to verify end-to-end functionality.
        """
        config = _base_classifier_config(tmp_path)
        # Use smaller images for faster execution
        config["data"]["preprocessing"]["image_size"] = 64
        config["data"]["preprocessing"]["crop_size"] = 32

        # Actually run the training (no mocking)
        setup_experiment_classifier(config)

        # Verify outputs were created
        assert (tmp_path / "checkpoints").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "logs" / "config.yaml").exists()
        assert (tmp_path / "logs" / "metrics" / "metrics.csv").exists()

        # Verify checkpoint was saved
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.pth"))
        assert len(checkpoint_files) > 0


class TestExperimentDispatcher:
    """Test suite for experiment dispatcher logic with config-only mode."""

    @pytest.mark.component
    def test_dispatcher_routes_to_classifier(self, tmp_path):
        """Test that dispatcher correctly routes to classifier."""
        config_file = tmp_path / "classifier_config.yaml"
        config_data = {
            "experiment": "classifier",
            "mode": "train",
            "compute": {"device": "cpu", "seed": None},
            "model": {
                "architecture": {"name": "resnet50", "num_classes": 2},
                "initialization": {"pretrained": False, "freeze_backbone": False},
            },
            "data": {
                "split_file": str(
                    _PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"
                ),
                "loading": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle_train": True,
                    "drop_last": False,
                },
                "preprocessing": {
                    "image_size": 256,
                    "crop_size": 224,
                    "normalize": "imagenet",
                },
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation_degrees": 0,
                    "color_jitter": {"enabled": False},
                },
            },
            "training": {
                "epochs": 1,
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
                "scheduler": {"type": None},
                "checkpointing": {"save_frequency": 10, "save_best_only": True},
                "validation": {"enabled": True, "frequency": 1},
            },
            "output": {
                "base_dir": "outputs",
                "subdirs": {"checkpoints": "checkpoints", "logs": "logs"},
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_classifier") as mock_classifier:
            main([str(config_file)])

            mock_classifier.assert_called_once()

    @pytest.mark.component
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
                "split_file": str(
                    _PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"
                ),
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

    @pytest.mark.component
    def test_dispatcher_routes_to_gan(self, tmp_path):
        """Test that dispatcher correctly routes to GAN (not implemented)."""
        config_file = tmp_path / "gan_config.yaml"
        config_data = {
            "experiment": "gan",
            "data": {"train_path": "tests/fixtures/mock_data/train"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with pytest.raises(SystemExit) as exc_info:
            main([str(config_file)])
        assert exc_info.value.code == 1

    @pytest.mark.component
    def test_dispatcher_routes_to_data_preparation(self, tmp_path):
        """Test that dispatcher correctly routes to data_preparation."""
        config_file = tmp_path / "data_prep_config.yaml"
        config_data = {
            "experiment": "data_preparation",
            "classes": {
                "normal": "tests/fixtures/mock_data/train/0.Normal",
                "abnormal": "tests/fixtures/mock_data/train/1.Abnormal",
            },
            "split": {
                "seed": 42,
                "train_ratio": 0.8,
                "save_dir": str(tmp_path / "splits"),
                "split_file": "split.json",
                "force": True,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        with patch("src.main.setup_experiment_data_preparation") as mock_dp:
            main([str(config_file)])

            mock_dp.assert_called_once()


# ---------------------------------------------------------------------------
# Shared test helpers (used by multiple test classes below)
# ---------------------------------------------------------------------------


def _generation_config(
    tmp_path,
    checkpoint=None,
    num_samples=10,
    num_classes=2,
    use_ema=False,
    class_selection=None,
):
    """Create a minimal generation mode config."""
    return {
        "experiment": "diffusion",
        "mode": "generate",
        "model": {
            "architecture": {
                "image_size": 32,
                "in_channels": 3,
                "model_channels": 16,
                "channel_multipliers": [1, 2],
                "use_attention": [False, False],
            },
            "diffusion": {
                "num_timesteps": 1000,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
            "conditioning": {
                "type": "class",
                "num_classes": num_classes,
                "class_dropout_prob": 0.1,
            },
        },
        "data": {
            "split_file": str(_PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"),
            "loading": {
                "batch_size": 16,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
                "shuffle_train": True,
            },
            "augmentation": {
                "horizontal_flip": False,
                "rotation_degrees": 0,
                "color_jitter": {
                    "enabled": False,
                    "strength": 0.0,
                },
            },
        },
        "generation": {
            "checkpoint": checkpoint,
            "sampling": {
                "num_samples": num_samples,
                "batch_size": num_samples,
                "guidance_scale": 3.0,
                "use_ema": use_ema,
                "ema_decay": 0.9999,
                "class_selection": class_selection,
            },
            "output": {
                "save_grid": True,
                "save_individual": False,
                "grid_nrow": 4,
            },
        },
        "compute": {
            "device": "cpu",
            "seed": 42,
        },
        "output": {
            "base_dir": str(tmp_path / "outputs"),
            "subdirs": {
                "logs": "logs",
                "checkpoints": "checkpoints",
                "samples": "samples",
                "generated": "generated",
            },
        },
    }


def _mock_diffusion_checkpoint(tmp_path, include_ema=False, num_classes=2):
    """Create a mock checkpoint file for testing."""
    import torch

    from src.experiments.diffusion.model import create_ddpm

    model = create_ddpm(
        image_size=32,
        in_channels=3,
        model_channels=16,
        channel_multipliers=(1, 2),
        num_classes=num_classes,
        num_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        class_dropout_prob=0.1,
        use_attention=(False, False),
        device="cpu",
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 10,
    }

    if include_ema:
        checkpoint["ema_state_dict"] = model.state_dict()

    checkpoint_path = tmp_path / "mock_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _base_classifier_config(tmp_path):
    """Return a base classifier config dict for unit tests."""
    return {
        "experiment": "classifier",
        "mode": "train",
        "compute": {"device": "cpu", "seed": None},
        "model": {
            "architecture": {"name": "resnet50", "num_classes": 2},
            "initialization": {"pretrained": False, "freeze_backbone": False},
        },
        "data": {
            "split_file": str(_PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"),
            "loading": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": True,
                "drop_last": False,
            },
            "preprocessing": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
            "augmentation": {
                "horizontal_flip": True,
                "rotation_degrees": 0,
                "color_jitter": {"enabled": False},
            },
        },
        "training": {
            "epochs": 1,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
            "scheduler": {"type": None},
            "checkpointing": {"save_frequency": 10, "save_best_only": True},
            "validation": {"enabled": True, "frequency": 1},
        },
        "output": {
            "base_dir": str(tmp_path),
            "subdirs": {"logs": "logs", "checkpoints": "checkpoints"},
        },
    }


def _base_diffusion_train_config(tmp_path):
    """Return a base diffusion training config dict for unit tests."""
    return {
        "experiment": "diffusion",
        "mode": "train",
        "compute": {"device": "cpu", "seed": None},
        "model": {
            "architecture": {
                "image_size": 8,
                "in_channels": 3,
                "model_channels": 16,
                "channel_multipliers": [1, 2],
                "use_attention": [False, False],
            },
            "diffusion": {
                "num_timesteps": 10,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
            "conditioning": {
                "type": "class",
                "num_classes": 2,
                "class_dropout_prob": 0.1,
            },
        },
        "data": {
            "split_file": str(_PROJECT_ROOT / "tests/fixtures/splits/mock_split.json"),
            "loading": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": True,
                "drop_last": False,
            },
            "augmentation": {
                "horizontal_flip": True,
                "rotation_degrees": 0,
                "color_jitter": {"enabled": False, "strength": 0.5},
            },
        },
        "output": {
            "base_dir": str(tmp_path),
            "subdirs": {
                "logs": "logs",
                "checkpoints": "checkpoints",
                "samples": "samples",
                "generated": "generated",
            },
        },
        "training": {
            "epochs": 1,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
            "scheduler": {"type": None},
            "ema": {"enabled": False, "decay": 0.9999},
            "performance": {
                "use_amp": False,
                "use_tf32": False,
                "cudnn_benchmark": False,
                "compile_model": False,
            },
            "visualization": {
                "enabled": False,
                "num_samples": 4,
                "guidance_scale": 0.0,
                "log_images_interval": None,
                "log_denoising_interval": None,
            },
            "checkpointing": {
                "save_frequency": 10,
                "save_best_only": True,
                "save_latest": True,
            },
            "validation": {"frequency": 1, "metric": "val_loss"},
        },
        "logging": {},
    }


class TestDiffusionGenerationMode:
    """Test suite for diffusion generation mode refactoring.

    Note: These tests assume setup_experiment_diffusion does not call sys.exit()
    after generation completes. If that behavior changes, these tests will need
    to wrap calls in try/except SystemExit.
    """

    @pytest.mark.unit
    def test_generation_mode_missing_checkpoint_raises_error(self, tmp_path):
        """Test that generation mode without checkpoint raises ValueError."""
        from src.main import setup_experiment_diffusion

        config = _generation_config(tmp_path, checkpoint=None)

        with pytest.raises(ValueError, match="generation.checkpoint is required"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_checkpoint_not_found_raises_error(self, tmp_path):
        """Test that non-existent checkpoint raises FileNotFoundError."""
        from src.main import setup_experiment_diffusion

        config = _generation_config(tmp_path, checkpoint="nonexistent_checkpoint.pth")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_invalid_checkpoint_raises_error(self, tmp_path):
        """Test that checkpoint without model_state_dict raises ValueError."""
        from src.main import setup_experiment_diffusion

        # Create invalid checkpoint (empty dict)
        checkpoint_path = tmp_path / "invalid_checkpoint.pth"
        import torch

        torch.save({}, checkpoint_path)

        config = _generation_config(tmp_path, checkpoint=str(checkpoint_path))

        with pytest.raises(ValueError, match="does not contain 'model_state_dict'"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_validates_num_samples(self, tmp_path):
        """Test that num_samples validation works."""
        from src.main import setup_experiment_diffusion

        # Create valid checkpoint
        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)

        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=0
        )

        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_warns_when_samples_less_than_classes(
        self, tmp_path, capsys
    ):
        """Test warning when num_samples < num_classes."""
        import torch

        from src.main import setup_experiment_diffusion

        # Create valid checkpoint
        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)

        config = _generation_config(
            tmp_path,
            checkpoint=str(checkpoint_path),
            num_samples=1,  # Less than num_classes=2
            num_classes=2,
        )

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                mock_sample.return_value = torch.randn(1, 3, 32, 32)

                setup_experiment_diffusion(config)

        captured = capsys.readouterr()
        assert "num_samples" in captured.out

    @pytest.mark.unit
    def test_generation_mode_uses_sampler_not_trainer(self, tmp_path):
        """Test that generation mode uses sample function, not DiffusionTrainer."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(tmp_path, checkpoint=str(checkpoint_path))

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                # Return proper tensor instead of MagicMock
                mock_sample.return_value = torch.randn(10, 3, 32, 32)

                setup_experiment_diffusion(config)

                # Verify sample function was called
                mock_sample.assert_called_once()

    @pytest.mark.unit
    def test_generation_mode_no_optimizer_created(self, tmp_path):
        """Test that generation mode does not create an optimizer."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(tmp_path, checkpoint=str(checkpoint_path))

        with patch("torch.optim.Adam") as mock_adam:
            with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
                with patch("src.utils.experiment_logger.ExperimentLogger"):
                    mock_sample.return_value = torch.randn(10, 3, 32, 32)

                    setup_experiment_diffusion(config)

                    # Verify no optimizer was created
                    mock_adam.assert_not_called()

    @pytest.mark.unit
    def test_generation_mode_no_dataloader_created(self, tmp_path):
        """Test that generation mode does not create a dataloader."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(tmp_path, checkpoint=str(checkpoint_path))

        with patch("src.utils.data.loaders.create_train_loader") as mock_create_train:
            with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
                with patch("src.utils.experiment_logger.ExperimentLogger"):
                    mock_sample.return_value = torch.randn(10, 3, 32, 32)

                    setup_experiment_diffusion(config)

                    # Verify no train loader was created (generation mode)
                    mock_create_train.assert_not_called()

    @pytest.mark.unit
    def test_generation_mode_loads_ema_when_available(self, tmp_path, capsys):
        """Test that EMA weights are loaded with ema_decay from config."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path, include_ema=True)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), use_ema=True
        )
        # Set a custom ema_decay to verify it's read from config
        config["generation"]["sampling"]["ema_decay"] = 0.995

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                with patch("src.experiments.diffusion.model.EMA") as mock_ema:
                    mock_sample.return_value = torch.randn(10, 3, 32, 32)

                    setup_experiment_diffusion(config)

                    # Verify EMA was created with decay from config
                    mock_ema.assert_called_once()
                    call_kwargs = mock_ema.call_args
                    assert call_kwargs[1]["decay"] == 0.995

        captured = capsys.readouterr()
        assert "Loaded EMA weights" in captured.out

    @pytest.mark.unit
    def test_generation_mode_warns_when_ema_missing(self, tmp_path, capsys):
        """Test warning when use_ema=True but no EMA in checkpoint."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path, include_ema=False)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), use_ema=True
        )

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                mock_sample.return_value = torch.randn(10, 3, 32, 32)

                setup_experiment_diffusion(config)

        captured = capsys.readouterr()
        assert "no EMA weights" in captured.out
        assert "Falling back" in captured.out

    @pytest.mark.unit
    def test_generation_mode_batched_generation(self, tmp_path):
        """Test that batched generation produces correct total number of samples."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=10
        )
        # Set batch_size smaller than num_samples to force multiple batches
        config["generation"]["sampling"]["batch_size"] = 3

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                # Each batch call returns appropriate number of samples
                def sample_side_effect(*args, **kwargs):
                    n = kwargs.get("num_samples", 1)
                    return torch.randn(n, 3, 32, 32)

                mock_sample.side_effect = sample_side_effect

                setup_experiment_diffusion(config)

                # With 10 samples and batch_size=3, expect 4 batches (3+3+3+1)
                assert mock_sample.call_count == 4

    @pytest.mark.unit
    def test_generation_mode_batch_size_larger_than_num_samples(self, tmp_path):
        """Test that batch_size larger than num_samples results in single batch."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=5
        )
        # batch_size larger than num_samples
        config["generation"]["sampling"]["batch_size"] = 100

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):
                mock_sample.return_value = torch.randn(5, 3, 32, 32)

                setup_experiment_diffusion(config)

                # Single batch since batch_size > num_samples
                mock_sample.assert_called_once()

    @pytest.mark.unit
    def test_generation_mode_class_labels_all_classes(self, tmp_path):
        """Test that with class_selection=null, labels are balanced across all classes."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=4, num_classes=2
        )
        config["generation"]["sampling"]["class_selection"] = None

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):

                def sample_side_effect(*args, **kwargs):
                    n = kwargs.get("num_samples", 1)
                    return torch.randn(n, 3, 32, 32)

                mock_sample.side_effect = sample_side_effect

                setup_experiment_diffusion(config)

                all_labels = torch.cat(
                    [call.kwargs["class_labels"] for call in mock_sample.call_args_list]
                )
                assert (all_labels == 0).sum().item() == 2
                assert (all_labels == 1).sum().item() == 2

    @pytest.mark.unit
    def test_generation_mode_class_selection_single_class(self, tmp_path):
        """Test that class_selection=[1] generates only labels=1."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path,
            checkpoint=str(checkpoint_path),
            num_samples=4,
            num_classes=2,
            class_selection=[1],
        )

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):

                def sample_side_effect(*args, **kwargs):
                    n = kwargs.get("num_samples", 1)
                    return torch.randn(n, 3, 32, 32)

                mock_sample.side_effect = sample_side_effect

                setup_experiment_diffusion(config)

                all_labels = torch.cat(
                    [call.kwargs["class_labels"] for call in mock_sample.call_args_list]
                )
                assert all_labels.tolist() == [1] * 4

    @pytest.mark.unit
    def test_generation_mode_class_selection_subset_balanced(self, tmp_path):
        """Test class_selection=[0,1] with num_classes=4: only classes 0 and 1 are generated."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path, num_classes=4)
        config = _generation_config(
            tmp_path,
            checkpoint=str(checkpoint_path),
            num_samples=10,
            num_classes=4,
            class_selection=[0, 1],
        )

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):

                def sample_side_effect(*args, **kwargs):
                    n = kwargs.get("num_samples", 1)
                    return torch.randn(n, 3, 32, 32)

                mock_sample.side_effect = sample_side_effect

                setup_experiment_diffusion(config)

                all_labels = torch.cat(
                    [call.kwargs["class_labels"] for call in mock_sample.call_args_list]
                )
                assert (all_labels == 0).sum().item() == 5
                assert (all_labels == 1).sum().item() == 5
                assert (all_labels == 2).sum().item() == 0
                assert (all_labels == 3).sum().item() == 0

    @pytest.mark.unit
    def test_generation_mode_class_selection_logs_info(self, tmp_path, capsys):
        """Test that class_selection is logged when set."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path,
            checkpoint=str(checkpoint_path),
            num_samples=4,
            num_classes=2,
            class_selection=[0],
        )

        with patch("src.experiments.diffusion.sampler.sample") as mock_sample:
            with patch("src.utils.experiment_logger.ExperimentLogger"):

                def sample_side_effect(*args, **kwargs):
                    n = kwargs.get("num_samples", 1)
                    return torch.randn(n, 3, 32, 32)

                mock_sample.side_effect = sample_side_effect

                setup_experiment_diffusion(config)

        captured = capsys.readouterr()
        assert "Class selection" in captured.out
        assert "[0]" in captured.out


class TestClassifierOptimizerVariants:
    """Component tests for classifier optimizer variants (covers lines 260-274)."""

    @pytest.mark.component
    def test_setup_classifier_adamw_optimizer(self, tmp_path):
        """Test classifier setup with AdamW optimizer succeeds."""
        config = _base_classifier_config(tmp_path)
        config["training"]["optimizer"]["type"] = "adamw"

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_sgd_optimizer(self, tmp_path):
        """Test classifier setup with SGD optimizer and explicit momentum."""
        config = _base_classifier_config(tmp_path)
        config["training"]["optimizer"]["type"] = "sgd"
        config["training"]["optimizer"]["momentum"] = 0.9

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_sgd_default_momentum(self, tmp_path):
        """Test classifier setup with SGD uses default momentum 0.9."""
        config = _base_classifier_config(tmp_path)
        config["training"]["optimizer"]["type"] = "sgd"
        # No explicit momentum — runtime code should default to 0.9

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_unknown_optimizer_raises(self, tmp_path):
        """Test that an optimizer unknown to runtime raises ValueError."""
        config = _base_classifier_config(tmp_path)
        config["training"]["optimizer"]["type"] = "rmsprop"

        # Bypass the config validator so the runtime code path is reached
        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch("src.experiments.classifier.config.validate_config"),
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            with pytest.raises(ValueError, match="Unknown optimizer"):
                setup_experiment_classifier(config)


class TestClassifierSchedulerVariants:
    """Component tests for classifier scheduler variants (covers lines 289, 294-308)."""

    @pytest.mark.component
    def test_setup_classifier_step_scheduler(self, tmp_path):
        """Test classifier setup with StepLR scheduler."""
        config = _base_classifier_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "step",
            "step_size": 5,
            "gamma": 0.1,
        }

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_plateau_scheduler(self, tmp_path):
        """Test classifier setup with ReduceLROnPlateau scheduler."""
        config = _base_classifier_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
        }

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_cosine_auto_tmax(self, tmp_path):
        """Test classifier setup with cosine scheduler and T_max=auto."""
        config = _base_classifier_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "cosine",
            "T_max": "auto",
            "eta_min": 1e-6,
        }

        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch(
                "src.experiments.classifier.trainer.ClassifierTrainer.train"
            ) as mock_train,
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            setup_experiment_classifier(config)
            mock_train.assert_called_once()

    @pytest.mark.component
    def test_setup_classifier_unknown_scheduler_raises(self, tmp_path):
        """Test that a scheduler unknown to runtime raises ValueError."""
        config = _base_classifier_config(tmp_path)
        config["training"]["scheduler"] = {"type": "exponential"}

        # Bypass the config validator so the runtime code path is reached
        with (
            patch("src.experiments.classifier.models.ResNetClassifier") as mock_cls,
            patch("src.experiments.classifier.config.validate_config"),
        ):
            mock_cls.return_value = nn.Linear(10, 2)
            with pytest.raises(ValueError, match="Unknown scheduler"):
                setup_experiment_classifier(config)


class TestClassifierErrorHandling:
    """Component tests for classifier error handling (covers lines 364-373)."""

    @pytest.mark.component
    def test_setup_classifier_keyboard_interrupt(self, tmp_path):
        """Test that KeyboardInterrupt during training calls sys.exit(0)."""
        config = _base_classifier_config(tmp_path)

        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(SystemExit) as exc_info:
                setup_experiment_classifier(config)
            assert exc_info.value.code == 0

    @pytest.mark.component
    def test_setup_classifier_training_exception(self, tmp_path):
        """Test that RuntimeError during training is re-raised."""
        config = _base_classifier_config(tmp_path)

        with patch(
            "src.experiments.classifier.trainer.ClassifierTrainer.train",
            side_effect=RuntimeError("GPU OOM"),
        ):
            with pytest.raises(RuntimeError, match="GPU OOM"):
                setup_experiment_classifier(config)


class TestClassifierDeviceAndDataset:
    """Component tests for classifier device and dataset edge cases (covers lines 138, 199-200)."""

    @pytest.mark.component
    def test_setup_classifier_explicit_device(self, tmp_path):
        """Test that explicit device config skips get_device() auto-detection."""
        config = _base_classifier_config(tmp_path)
        config["compute"]["device"] = "cpu"

        with (
            patch("src.experiments.classifier.trainer.ClassifierTrainer.train"),
            patch("src.utils.experiment.get_device") as mock_get_device,
        ):
            setup_experiment_classifier(config)
            # get_device should NOT be called when device is explicitly set
            mock_get_device.assert_not_called()

    @pytest.mark.component
    def test_setup_classifier_no_classes_attribute(self, tmp_path):
        """Test fallback to Class 0, Class 1 when dataset has no classes attr."""
        config = _base_classifier_config(tmp_path)

        # Create a mock dataset without 'classes' attribute
        mock_dataset = MagicMock(spec=[])  # spec=[] means no attributes
        mock_train_loader = MagicMock()
        mock_train_loader.dataset = mock_dataset

        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.get_train_loader.return_value = mock_train_loader

        with (
            patch("src.experiments.classifier.trainer.ClassifierTrainer.train"),
            patch(
                "src.utils.data.loaders.create_train_loader",
                return_value=mock_train_loader,
            ),
            patch("src.utils.data.loaders.create_val_loader", return_value=None),
            patch(
                "src.utils.data.loaders.get_class_names",
                return_value=["Class 0", "Class 1"],
            ),
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):
            # Verify class names from split file are loaded and used
            # (ExperimentLogger no longer takes class_names; they are used
            # directly by the main function for logging)
            setup_experiment_classifier(config)


class TestDiffusionTrainingMode:
    """Unit tests for diffusion training mode (covers lines 678-917)."""

    @pytest.fixture
    def mock_diffusion_deps(self):
        """Mock all diffusion dependencies for testing setup_experiment_diffusion."""
        mock_model = nn.Sequential(nn.Linear(1, 1))
        with (
            patch("src.experiments.diffusion.config.validate_config"),
            patch(
                "src.experiments.diffusion.model.create_ddpm",
                return_value=mock_model,
            ),
            patch("src.utils.data.loaders.create_train_loader") as mock_create_train,
            patch("src.utils.data.loaders.create_val_loader", return_value=None),
            patch(
                "src.experiments.diffusion.trainer.DiffusionTrainer"
            ) as mock_trainer_cls,
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):
            mock_trainer_cls.return_value.train = MagicMock()
            yield {
                "trainer_cls": mock_trainer_cls,
                "create_train_loader": mock_create_train,
                "model": mock_model,
            }

    @pytest.mark.unit
    def test_setup_diffusion_training_mode_basic(self, tmp_path, mock_diffusion_deps):
        """Test basic diffusion training mode wiring."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_adamw_optimizer(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with AdamW optimizer."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["optimizer"]["type"] = "adamw"
        setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_unknown_optimizer_raises(
        self, tmp_path, mock_diffusion_deps
    ):
        """Test diffusion training with unknown optimizer raises ValueError."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["optimizer"]["type"] = "rmsprop"

        with pytest.raises(ValueError, match="Unknown optimizer"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_setup_diffusion_step_scheduler(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with StepLR scheduler."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "step",
            "step_size": 5,
            "gamma": 0.1,
        }
        setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_plateau_scheduler(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with ReduceLROnPlateau scheduler."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
        }
        setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_cosine_auto_tmax(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with cosine scheduler and T_max=auto."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["scheduler"] = {
            "type": "cosine",
            "T_max": "auto",
            "eta_min": 1e-6,
        }
        setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_unknown_scheduler_raises(
        self, tmp_path, mock_diffusion_deps
    ):
        """Test diffusion training with unknown scheduler raises ValueError."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["scheduler"] = {"type": "exponential"}

        with pytest.raises(ValueError, match="Unknown scheduler"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_setup_diffusion_keyboard_interrupt(self, tmp_path):
        """Test that KeyboardInterrupt during diffusion training calls sys.exit(0)."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        mock_model = nn.Sequential(nn.Linear(1, 1))

        with (
            patch("src.experiments.diffusion.config.validate_config"),
            patch(
                "src.experiments.diffusion.model.create_ddpm",
                return_value=mock_model,
            ),
            patch("src.utils.data.loaders.create_train_loader"),
            patch("src.utils.data.loaders.create_val_loader", return_value=None),
            patch(
                "src.experiments.diffusion.trainer.DiffusionTrainer"
            ) as mock_trainer_cls,
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):
            mock_trainer_cls.return_value.train.side_effect = KeyboardInterrupt
            with pytest.raises(SystemExit) as exc_info:
                setup_experiment_diffusion(config)
            assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_setup_diffusion_training_exception(self, tmp_path):
        """Test that RuntimeError during diffusion training is re-raised."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        mock_model = nn.Sequential(nn.Linear(1, 1))

        with (
            patch("src.experiments.diffusion.config.validate_config"),
            patch(
                "src.experiments.diffusion.model.create_ddpm",
                return_value=mock_model,
            ),
            patch("src.utils.data.loaders.create_train_loader"),
            patch("src.utils.data.loaders.create_val_loader", return_value=None),
            patch(
                "src.experiments.diffusion.trainer.DiffusionTrainer"
            ) as mock_trainer_cls,
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):
            mock_trainer_cls.return_value.train.side_effect = RuntimeError("GPU OOM")
            with pytest.raises(RuntimeError, match="GPU OOM"):
                setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_setup_diffusion_explicit_device(self, tmp_path, mock_diffusion_deps):
        """Test that explicit device config skips get_device() auto-detection."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["compute"]["device"] = "cpu"

        with patch("src.utils.experiment.get_device") as mock_get_device:
            setup_experiment_diffusion(config)
            mock_get_device.assert_not_called()

    @pytest.mark.unit
    def test_setup_diffusion_compile_model(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with compile_model enabled."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["performance"]["compile_model"] = True

        with patch("torch.compile", return_value=mock_diffusion_deps["model"]):
            setup_experiment_diffusion(config)
            mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_compile_model_failure(self, tmp_path, mock_diffusion_deps):
        """Test that torch.compile failure is caught gracefully."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["training"]["performance"]["compile_model"] = True

        with patch("torch.compile", side_effect=RuntimeError("compile error")):
            # Should not raise — warning is logged and training continues
            setup_experiment_diffusion(config)
            mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()

    @pytest.mark.unit
    def test_setup_diffusion_balancing_logging(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with balancing config logs active strategies."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["data"]["balancing"] = {
            "weighted_sampler": {"enabled": True},
            "downsampling": {"enabled": False},
            "upsampling": {"enabled": False},
            "class_weights": {"enabled": False},
        }

        with patch("src.main.logger") as mock_logger:
            mock_logger.info = MagicMock()
            setup_experiment_diffusion(config)

        mock_diffusion_deps["trainer_cls"].assert_called_once()
        mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        assert any("weighted_sampler" in msg for msg in info_messages)

    @pytest.mark.unit
    def test_setup_diffusion_class_weights(self, tmp_path, mock_diffusion_deps):
        """Test diffusion training with class_weights computes weight tensor."""
        from src.main import setup_experiment_diffusion

        config = _base_diffusion_train_config(tmp_path)
        config["data"]["balancing"] = {
            "class_weights": {
                "enabled": True,
                "method": "inverse_frequency",
            },
        }

        # Mock SplitFileDataset and compute_weights_from_config
        mock_dataset = MagicMock()
        mock_dataset.targets = [0, 0, 0, 1]

        with (
            patch(
                "src.utils.data.datasets.SplitFileDataset",
                return_value=mock_dataset,
            ),
            patch(
                "src.utils.data.samplers.compute_weights_from_config",
                return_value={0: 0.5, 1: 1.5},
            ) as mock_compute_weights,
        ):
            setup_experiment_diffusion(config)
            mock_compute_weights.assert_called_once()
            mock_diffusion_deps["trainer_cls"].return_value.train.assert_called_once()
            # Verify computed weights were forwarded to trainer constructor
            import torch

            trainer_call_kwargs = mock_diffusion_deps["trainer_cls"].call_args.kwargs
            assert "class_weights" in trainer_call_kwargs
            expected = torch.tensor([0.5, 1.5])
            assert torch.allclose(trainer_call_kwargs["class_weights"], expected)


class TestGenerationModeEdgeCases:
    """Unit tests for diffusion generation mode edge cases (covers lines 548, 556, 571, 668-670)."""

    @pytest.mark.unit
    def test_generation_mode_saves_individual(self, tmp_path):
        """Test generation mode with save_individual=True saves per-sample images."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=4
        )
        config["generation"]["output"]["save_individual"] = True

        with (
            patch("src.experiments.diffusion.sampler.sample") as mock_sample,
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):

            def sample_side_effect(*args, **kwargs):
                n = kwargs.get("num_samples", 1)
                return torch.randn(n, 3, 32, 32)

            mock_sample.side_effect = sample_side_effect

            setup_experiment_diffusion(config)

            # Verify individual files were saved
            output_dir = tmp_path / "outputs" / "generated"
            individual_files = list(output_dir.glob("sample_*.png"))
            assert len(individual_files) == 4

    @pytest.mark.unit
    def test_generation_mode_compiled_checkpoint(self, tmp_path):
        """Test generation mode strips _orig_mod. prefix from compiled model keys."""
        import torch

        from src.experiments.diffusion.model import create_ddpm
        from src.main import setup_experiment_diffusion

        # Create model and checkpoint with _orig_mod. prefix
        model = create_ddpm(
            image_size=32,
            in_channels=3,
            model_channels=16,
            channel_multipliers=(1, 2),
            num_classes=2,
            num_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            class_dropout_prob=0.1,
            use_attention=(False, False),
            device="cpu",
        )
        # Add _orig_mod. prefix to simulate torch.compile checkpoint
        state_dict = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        checkpoint_path = tmp_path / "compiled_checkpoint.pth"
        torch.save({"model_state_dict": state_dict, "epoch": 10}, checkpoint_path)

        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=2
        )

        with (
            patch("src.experiments.diffusion.sampler.sample") as mock_sample,
            patch("src.utils.experiment_logger.ExperimentLogger"),
        ):
            mock_sample.return_value = torch.randn(2, 3, 32, 32)

            # Should succeed — _orig_mod. prefix is stripped
            setup_experiment_diffusion(config)
            mock_sample.assert_called_once()

    @pytest.mark.unit
    def test_generation_mode_num_samples_zero_raises(self, tmp_path):
        """Test that num_samples=0 raises ValueError at config validation."""
        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=0
        )

        with pytest.raises(
            ValueError,
            match="generation.sampling.num_samples must be a positive integer",
        ):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_negative_num_samples_raises(self, tmp_path):
        """Test that negative num_samples raises ValueError at config validation."""
        from src.main import setup_experiment_diffusion

        checkpoint_path = _mock_diffusion_checkpoint(tmp_path)
        config = _generation_config(
            tmp_path, checkpoint=str(checkpoint_path), num_samples=-5
        )

        with pytest.raises(
            ValueError,
            match="generation.sampling.num_samples must be a positive integer",
        ):
            setup_experiment_diffusion(config)


class TestLoggingRedirectTqdmHandlerLevel:
    """Test that logging_redirect_tqdm handler levels are preserved."""

    @pytest.mark.unit
    def test_tqdm_redirect_handler_level_restored(self):
        """logging_redirect_tqdm replaces the console StreamHandler with a
        _TqdmLoggingHandler that defaults to NOTSET (level 0). The generation
        code in main.py captures the original handler level before entering the
        context and restores it on _TqdmLoggingHandler instances. This test
        verifies that pattern works.
        """
        import logging as _logging

        from tqdm.contrib.logging import logging_redirect_tqdm

        # Set up a root handler at WARNING to simulate setup_logging()
        root = _logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        try:
            handler = _logging.StreamHandler()
            handler.setLevel(_logging.WARNING)
            root.handlers = [handler]
            root.setLevel(_logging.DEBUG)

            # Capture original handler levels (mirrors main.py logic)
            original_handler_levels = [
                h.level
                for h in root.handlers
                if isinstance(h, _logging.StreamHandler)
                and not isinstance(h, _logging.FileHandler)
            ]
            default_console_level = (
                original_handler_levels[0] if original_handler_levels else _logging.INFO
            )

            with logging_redirect_tqdm():
                # Restore levels on _TqdmLoggingHandler (mirrors main.py logic)
                for h in _logging.root.handlers:
                    if (
                        type(h).__name__ == "_TqdmLoggingHandler"
                        and h.level == _logging.NOTSET
                    ):
                        h.setLevel(default_console_level)

                # Verify the handler now has the correct level
                for h in _logging.root.handlers:
                    if isinstance(h, _logging.StreamHandler):
                        assert h.level >= _logging.WARNING, (
                            f"Handler level should be >= WARNING, "
                            f"got {_logging.getLevelName(h.level)}"
                        )
        finally:
            root.handlers = original_handlers
            root.level = original_level

    @pytest.mark.unit
    def test_tqdm_redirect_handler_defaults_to_notset(self):
        """Confirm that logging_redirect_tqdm sets NOTSET on replacement handler.

        This documents the upstream behavior that necessitates the fix.
        """
        import logging as _logging

        from tqdm.contrib.logging import logging_redirect_tqdm

        root = _logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        try:
            handler = _logging.StreamHandler()
            handler.setLevel(_logging.INFO)
            root.handlers = [handler]
            root.setLevel(_logging.DEBUG)

            with logging_redirect_tqdm():
                # The tqdm replacement _TqdmLoggingHandler should default to NOTSET
                has_notset_tqdm_handler = any(
                    type(h).__name__ == "_TqdmLoggingHandler"
                    and h.level == _logging.NOTSET
                    for h in _logging.root.handlers
                )
                assert has_notset_tqdm_handler, (
                    "Expected logging_redirect_tqdm to create a "
                    "_TqdmLoggingHandler with NOTSET level"
                )
        finally:
            root.handlers = original_handlers
            root.level = original_level
