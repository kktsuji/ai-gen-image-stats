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


class TestDiffusionGenerationMode:
    """Test suite for diffusion generation mode refactoring."""

    @pytest.mark.unit
    def test_generation_mode_missing_checkpoint_raises_error(self, tmp_path):
        """Test that generation mode without checkpoint raises ValueError."""
        from src.main import setup_experiment_diffusion

        config = self._create_generation_config(tmp_path, checkpoint=None)

        with pytest.raises(ValueError, match="generation.checkpoint is required"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_checkpoint_not_found_raises_error(self, tmp_path):
        """Test that non-existent checkpoint raises FileNotFoundError."""
        from src.main import setup_experiment_diffusion

        config = self._create_generation_config(
            tmp_path, checkpoint="nonexistent_checkpoint.pth"
        )

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

        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path)
        )

        with pytest.raises(ValueError, match="does not contain 'model_state_dict'"):
            setup_experiment_diffusion(config)

    @pytest.mark.unit
    def test_generation_mode_validates_num_samples(self, tmp_path):
        """Test that num_samples validation works."""
        from src.main import setup_experiment_diffusion

        # Create valid checkpoint
        checkpoint_path = self._create_mock_checkpoint(tmp_path)

        config = self._create_generation_config(
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
        checkpoint_path = self._create_mock_checkpoint(tmp_path)

        config = self._create_generation_config(
            tmp_path,
            checkpoint=str(checkpoint_path),
            num_samples=1,  # Less than num_classes=2
            num_classes=2,
        )

        with patch("src.experiments.diffusion.sampler.DiffusionSampler") as mock_sampler:
            with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                # Mock sampler to return proper tensor
                mock_sampler_instance = MagicMock()
                mock_sampler.return_value = mock_sampler_instance
                mock_sampler_instance.sample.return_value = torch.randn(1, 3, 32, 32)

                # Should not raise, but should warn
                try:
                    setup_experiment_diffusion(config)
                except SystemExit:
                    pass  # May exit after generation

                captured = capsys.readouterr()
                assert "Warning" in captured.out
                assert "num_samples" in captured.out

    @pytest.mark.unit
    def test_generation_mode_uses_sampler_not_trainer(self, tmp_path):
        """Test that generation mode uses DiffusionSampler, not DiffusionTrainer."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = self._create_mock_checkpoint(tmp_path)
        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path)
        )

        with patch(
            "src.experiments.diffusion.sampler.DiffusionSampler"
        ) as mock_sampler:
            with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                mock_sampler_instance = MagicMock()
                mock_sampler.return_value = mock_sampler_instance
                # Return proper tensor instead of MagicMock
                mock_sampler_instance.sample.return_value = torch.randn(10, 3, 32, 32)

                try:
                    setup_experiment_diffusion(config)
                except SystemExit:
                    pass  # May exit after generation

                # Verify DiffusionSampler was created and used
                mock_sampler.assert_called_once()
                mock_sampler_instance.sample.assert_called_once()

    @pytest.mark.unit
    def test_generation_mode_no_optimizer_created(self, tmp_path):
        """Test that generation mode does not create an optimizer."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = self._create_mock_checkpoint(tmp_path)
        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path)
        )

        with patch("torch.optim.Adam") as mock_adam:
            with patch("src.experiments.diffusion.sampler.DiffusionSampler") as mock_sampler:
                with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                    # Mock sampler to return proper tensor
                    mock_sampler_instance = MagicMock()
                    mock_sampler.return_value = mock_sampler_instance
                    mock_sampler_instance.sample.return_value = torch.randn(10, 3, 32, 32)

                    try:
                        setup_experiment_diffusion(config)
                    except SystemExit:
                        pass  # May exit after generation

                    # Verify no optimizer was created
                    mock_adam.assert_not_called()

    @pytest.mark.unit
    def test_generation_mode_no_dataloader_created(self, tmp_path):
        """Test that generation mode does not create a dataloader."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = self._create_mock_checkpoint(tmp_path)
        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path)
        )

        with patch(
            "src.experiments.diffusion.dataloader.DiffusionDataLoader"
        ) as mock_dataloader:
            with patch("src.experiments.diffusion.sampler.DiffusionSampler") as mock_sampler:
                with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                    # Mock sampler to return proper tensor
                    mock_sampler_instance = MagicMock()
                    mock_sampler.return_value = mock_sampler_instance
                    mock_sampler_instance.sample.return_value = torch.randn(10, 3, 32, 32)

                    try:
                        setup_experiment_diffusion(config)
                    except SystemExit:
                        pass  # May exit after generation

                    # Verify no dataloader was created
                    mock_dataloader.assert_not_called()

    @pytest.mark.unit
    def test_generation_mode_loads_ema_when_available(self, tmp_path, capsys):
        """Test that EMA weights are loaded when available."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = self._create_mock_checkpoint(tmp_path, include_ema=True)
        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path), use_ema=True
        )

        with patch("src.experiments.diffusion.sampler.DiffusionSampler") as mock_sampler:
            with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                with patch("src.experiments.diffusion.model.EMA") as mock_ema:
                    # Mock sampler to return proper tensor
                    mock_sampler_instance = MagicMock()
                    mock_sampler.return_value = mock_sampler_instance
                    mock_sampler_instance.sample.return_value = torch.randn(10, 3, 32, 32)

                    try:
                        setup_experiment_diffusion(config)
                    except SystemExit:
                        pass  # May exit after generation

                    # Verify EMA was created and loaded
                    mock_ema.assert_called_once()

                    captured = capsys.readouterr()
                    assert "Loaded EMA weights" in captured.out

    @pytest.mark.unit
    def test_generation_mode_warns_when_ema_missing(self, tmp_path, capsys):
        """Test warning when use_ema=True but no EMA in checkpoint."""
        import torch

        from src.main import setup_experiment_diffusion

        checkpoint_path = self._create_mock_checkpoint(tmp_path, include_ema=False)
        config = self._create_generation_config(
            tmp_path, checkpoint=str(checkpoint_path), use_ema=True
        )

        with patch("src.experiments.diffusion.sampler.DiffusionSampler") as mock_sampler:
            with patch("src.experiments.diffusion.logger.DiffusionLogger"):
                # Mock sampler to return proper tensor
                mock_sampler_instance = MagicMock()
                mock_sampler.return_value = mock_sampler_instance
                mock_sampler_instance.sample.return_value = torch.randn(10, 3, 32, 32)

                try:
                    setup_experiment_diffusion(config)
                except SystemExit:
                    pass  # May exit after generation

                captured = capsys.readouterr()
                assert "Warning" in captured.out
                assert "no EMA weights" in captured.out
                assert "Falling back" in captured.out

    # Helper methods
    def _create_generation_config(
        self,
        tmp_path,
        checkpoint=None,
        num_samples=10,
        num_classes=2,
        use_ema=False,
    ):
        """Create a minimal generation mode config."""
        return {
            "experiment": "diffusion",
            "mode": "generate",
            "model": {
                "architecture": {
                    "image_size": 32,
                    "in_channels": 3,
                    "model_channels": 64,
                    "channel_multipliers": [1, 2, 4],
                    "use_attention": [False, True, False],
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
                "paths": {
                    "train": "tests/fixtures/mock_data/train",  # Required by validator
                },
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
                    "guidance_scale": 3.0,
                    "use_ema": use_ema,
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

    def _create_mock_checkpoint(self, tmp_path, include_ema=False):
        """Create a mock checkpoint file for testing."""
        import torch

        from src.experiments.diffusion.model import create_ddpm

        # Create a minimal model to get state dict
        model = create_ddpm(
            image_size=32,
            in_channels=3,
            model_channels=64,
            channel_multipliers=(1, 2, 4),
            num_classes=2,
            num_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            class_dropout_prob=0.1,
            use_attention=(False, True, False),
            device="cpu",
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
        }

        if include_ema:
            # Add mock EMA state dict
            checkpoint["ema_state_dict"] = model.state_dict()

        checkpoint_path = tmp_path / "mock_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
        return checkpoint_path
            checkpoint["ema_state_dict"] = model.state_dict()

        checkpoint_path = tmp_path / "mock_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
        return checkpoint_path
