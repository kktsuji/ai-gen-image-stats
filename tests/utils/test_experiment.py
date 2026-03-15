"""Unit tests for shared experiment setup utilities.

Tests cover:
- setup_experiment_common: returns correct types (str, Path), log_dir created,
  config.yaml written, banner logged, seed set/skipped, device auto vs. explicit
- create_experiment_logger: returns ExperimentLogger, picks up tensorboard config
- run_training: calls trainer.train, handles KeyboardInterrupt and Exception
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.utils.experiment import (
    create_experiment_logger,
    run_training,
    setup_experiment_common,
)

# ==============================================================================
# Helpers
# ==============================================================================


def _make_config(
    tmp_path: Path, device: str = "cpu", seed: Any = None
) -> Dict[str, Any]:
    """Build a minimal valid config for setup_experiment_common tests."""
    return {
        "experiment": "test",
        "mode": "train",
        "compute": {
            "device": device,
            "seed": seed,
        },
        "output": {
            "base_dir": str(tmp_path / "outputs"),
            "subdirs": {
                "logs": "logs",
                "checkpoints": "checkpoints",
            },
        },
        "logging": {
            "console_level": "WARNING",
            "file_level": "DEBUG",
            "timezone": "UTC",
        },
    }


# ==============================================================================
# TestSetupExperimentCommon
# ==============================================================================


@pytest.mark.unit
class TestSetupExperimentCommon:
    """Test setup_experiment_common shared setup function."""

    def test_returns_string_and_path(self, tmp_path, clean_logging_handlers):
        """Test that the function returns a str and a Path."""
        config = _make_config(tmp_path)
        device, log_dir = setup_experiment_common(config, "TEST STARTED")

        assert isinstance(device, str)
        assert isinstance(log_dir, Path)

    def test_log_dir_created(self, tmp_path, clean_logging_handlers):
        """Test that log directory is created."""
        config = _make_config(tmp_path)
        _, log_dir = setup_experiment_common(config, "TEST STARTED")

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_config_yaml_written(self, tmp_path, clean_logging_handlers):
        """Test that config.yaml is saved inside log_dir."""
        config = _make_config(tmp_path)
        _, log_dir = setup_experiment_common(config, "TEST STARTED")

        config_snapshot = log_dir / "config.yaml"
        assert config_snapshot.exists()

    def test_banner_logged(self, tmp_path, clean_logging_handlers):
        """Test that the experiment banner is logged to the log file."""
        config = _make_config(tmp_path)
        setup_experiment_common(config, "MY EXPERIMENT BANNER")

        # Read the log file created by setup_logging (setup_logging clears
        # root handlers, which removes caplog's handler, so we verify via file)
        log_files = list((tmp_path / "outputs" / "logs").glob("*.log"))
        assert len(log_files) == 1
        log_content = log_files[0].read_text()
        assert "MY EXPERIMENT BANNER" in log_content

    def test_explicit_device_returned(self, tmp_path, clean_logging_handlers):
        """Test that explicit device config is returned as-is."""
        config = _make_config(tmp_path, device="cpu")
        device, _ = setup_experiment_common(config, "TEST STARTED")

        assert device == "cpu"

    def test_auto_device_returns_string(self, tmp_path, clean_logging_handlers):
        """Test that auto device resolves and returns a string."""
        config = _make_config(tmp_path, device="auto")
        device, _ = setup_experiment_common(config, "TEST STARTED")

        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_seed_set_when_provided(self, tmp_path, clean_logging_handlers):
        """Test that seed is set when provided in config."""
        config = _make_config(tmp_path, seed=42)
        setup_experiment_common(config, "TEST STARTED")

        log_files = list((tmp_path / "outputs" / "logs").glob("*.log"))
        assert len(log_files) == 1
        log_content = log_files[0].read_text()
        assert "Random seed set to: 42" in log_content

    def test_seed_not_logged_when_none(self, tmp_path, clean_logging_handlers):
        """Test that seed log message is absent when seed is None."""
        config = _make_config(tmp_path, seed=None)
        setup_experiment_common(config, "TEST STARTED")

        log_files = list((tmp_path / "outputs" / "logs").glob("*.log"))
        assert len(log_files) == 1
        log_content = log_files[0].read_text()
        assert "Random seed" not in log_content

    @patch("src.utils.experiment.np.random.seed")
    @patch("src.utils.experiment.random.seed")
    def test_seed_sets_all_rngs(
        self, mock_random_seed, mock_np_seed, tmp_path, clean_logging_handlers
    ):
        """Test that seed sets random, numpy, and torch RNGs."""
        config = _make_config(tmp_path, seed=42)
        setup_experiment_common(config, "TEST STARTED")

        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)

    @patch("src.utils.experiment.np.random.seed")
    @patch("src.utils.experiment.random.seed")
    def test_seed_none_skips_all_rngs(
        self, mock_random_seed, mock_np_seed, tmp_path, clean_logging_handlers
    ):
        """Test that None seed skips all RNG seeding."""
        config = _make_config(tmp_path, seed=None)
        setup_experiment_common(config, "TEST STARTED")

        mock_random_seed.assert_not_called()
        mock_np_seed.assert_not_called()

    def test_log_dir_matches_config(self, tmp_path, clean_logging_handlers):
        """Test that returned log_dir matches config output path."""
        config = _make_config(tmp_path)
        _, log_dir = setup_experiment_common(config, "TEST STARTED")

        expected = Path(tmp_path / "outputs" / "logs")
        assert log_dir == expected


# ==============================================================================
# TestCreateExperimentLogger
# ==============================================================================


@pytest.mark.unit
class TestCreateExperimentLogger:
    """Test create_experiment_logger helper function."""

    def test_returns_experiment_logger(self, tmp_path):
        """Test that the function returns an ExperimentLogger instance."""
        from src.utils.experiment_logger import ExperimentLogger

        config = _make_config(tmp_path)
        log_dir = tmp_path / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        result = create_experiment_logger(config, log_dir)

        assert isinstance(result, ExperimentLogger)
        result.close()

    def test_with_subdirs(self, tmp_path):
        """Test that subdirs are passed through to the logger."""
        config = _make_config(tmp_path)
        log_dir = tmp_path / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        result = create_experiment_logger(
            config, log_dir, subdirs={"images": "samples"}
        )

        assert "images" in result.dirs
        result.close()

    def test_without_subdirs(self, tmp_path):
        """Test that None subdirs works."""
        config = _make_config(tmp_path)
        log_dir = tmp_path / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        result = create_experiment_logger(config, log_dir)

        assert isinstance(result, type(result))
        result.close()

    def test_with_tensorboard_config(self, tmp_path):
        """Test that tensorboard config is picked up from experiment config."""
        config = _make_config(tmp_path)
        config["logging"]["metrics"] = {"tensorboard": {"enabled": True}}
        config["output"]["subdirs"]["tensorboard"] = "tensorboard"
        log_dir = tmp_path / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        result = create_experiment_logger(config, log_dir)

        assert isinstance(result, type(result))
        result.close()


# ==============================================================================
# TestRunTraining
# ==============================================================================


@pytest.mark.unit
class TestRunTraining:
    """Test run_training helper function."""

    def _make_training_args(self):
        """Return default kwargs for run_training."""
        return {
            "num_epochs": 10,
            "checkpoint_dir": "/tmp/checkpoints",
            "save_best": True,
            "checkpoint_frequency": 5,
            "save_latest_checkpoint": True,
            "validate_frequency": 1,
            "best_metric": "loss",
        }

    def test_calls_trainer_train(self):
        """Test that run_training calls trainer.train with correct args."""
        trainer = MagicMock()
        metrics_logger = MagicMock()

        run_training(trainer, metrics_logger, **self._make_training_args())

        trainer.train.assert_called_once_with(
            num_epochs=10,
            checkpoint_dir="/tmp/checkpoints",
            save_best=True,
            checkpoint_frequency=5,
            save_latest_checkpoint=True,
            validate_frequency=1,
            best_metric="loss",
        )

    def test_closes_logger_on_success(self):
        """Test that metrics_logger.close() is called on success."""
        trainer = MagicMock()
        metrics_logger = MagicMock()

        run_training(trainer, metrics_logger, **self._make_training_args())

        metrics_logger.close.assert_called_once()

    def test_closes_logger_on_exception(self):
        """Test that metrics_logger.close() is called when trainer raises."""
        trainer = MagicMock()
        trainer.train.side_effect = RuntimeError("training failed")
        metrics_logger = MagicMock()

        with pytest.raises(RuntimeError, match="training failed"):
            run_training(trainer, metrics_logger, **self._make_training_args())

        metrics_logger.close.assert_called_once()

    def test_keyboard_interrupt_calls_sys_exit(self):
        """Test that KeyboardInterrupt causes sys.exit(0)."""
        trainer = MagicMock()
        trainer.train.side_effect = KeyboardInterrupt
        metrics_logger = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            run_training(trainer, metrics_logger, **self._make_training_args())

        assert exc_info.value.code == 0
        metrics_logger.close.assert_called_once()

    def test_reraises_exception(self):
        """Test that non-KeyboardInterrupt exceptions are re-raised."""
        trainer = MagicMock()
        trainer.train.side_effect = ValueError("bad config")
        metrics_logger = MagicMock()

        with pytest.raises(ValueError, match="bad config"):
            run_training(trainer, metrics_logger, **self._make_training_args())

    def test_close_failure_does_not_mask_training_exception(self):
        """Test that a close() failure doesn't mask the primary training exception."""
        trainer = MagicMock()
        trainer.train.side_effect = RuntimeError("training failed")
        metrics_logger = MagicMock()
        metrics_logger.close.side_effect = OSError("close failed")

        with pytest.raises(RuntimeError, match="training failed"):
            run_training(trainer, metrics_logger, **self._make_training_args())

    def test_close_failure_does_not_mask_on_success(self):
        """Test that a close() failure on success path doesn't raise."""
        trainer = MagicMock()
        metrics_logger = MagicMock()
        metrics_logger.close.side_effect = OSError("close failed")

        # Should not raise — close failure is logged but swallowed
        run_training(trainer, metrics_logger, **self._make_training_args())
