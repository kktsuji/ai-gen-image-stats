"""Tests for ExperimentLogger

This module contains tests for the unified ExperimentLogger implementation.
Tests are organized into unit, component, and integration tiers.
"""

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from src.utils.experiment_logger import ExperimentLogger

# Test fixtures


@pytest.fixture
def temp_log_dir():
    """Provide temporary directory for logger outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger(temp_log_dir):
    """Provide a basic logger instance with sample subdirs."""
    return ExperimentLogger(
        log_dir=temp_log_dir,
        subdirs={"images": "samples", "denoising": "denoising"},
    )


@pytest.fixture
def classifier_logger(temp_log_dir):
    """Provide a logger configured like the classifier experiment."""
    return ExperimentLogger(
        log_dir=temp_log_dir,
        subdirs={
            "images": "predictions",
            "confusion_matrices": "confusion_matrices",
        },
    )


# Unit Tests


@pytest.mark.unit
class TestExperimentLoggerInstantiation:
    """Test logger instantiation and initialization."""

    def test_can_instantiate_logger(self, temp_log_dir):
        """ExperimentLogger can be instantiated."""
        logger = ExperimentLogger(log_dir=temp_log_dir)
        assert isinstance(logger, ExperimentLogger)

    def test_creates_log_directory(self, temp_log_dir):
        """Logger creates log directory on instantiation."""
        log_dir = Path(temp_log_dir) / "experiment_001"
        ExperimentLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_creates_metrics_subdirectory(self, temp_log_dir):
        """Logger creates metrics subdirectory."""
        ExperimentLogger(log_dir=temp_log_dir)
        assert (Path(temp_log_dir) / "metrics").exists()

    def test_creates_configured_subdirectories(self, logger, temp_log_dir):
        """Logger creates requested subdirectories."""
        log_dir = Path(temp_log_dir)
        assert (log_dir / "samples").exists()
        assert (log_dir / "denoising").exists()

    def test_dirs_maps_logical_names(self, logger, temp_log_dir):
        """Logger maps logical names to directory paths."""
        log_dir = Path(temp_log_dir)
        assert logger.dirs["images"] == log_dir / "samples"
        assert logger.dirs["denoising"] == log_dir / "denoising"

    def test_no_subdirs_by_default(self, temp_log_dir):
        """Logger creates no extra subdirs when none specified."""
        logger = ExperimentLogger(log_dir=temp_log_dir)
        assert logger.dirs == {}

    def test_log_dir_is_path_object(self, logger):
        """Logger converts log_dir to Path object."""
        assert isinstance(logger.log_dir, Path)

    def test_initializes_empty_histories(self, logger):
        """Logger initializes empty tracking lists."""
        assert logger.logged_metrics_history == []
        assert logger.logged_images == []


@pytest.mark.unit
class TestLogMetrics:
    """Test the log_metrics() method."""

    def test_log_metrics_with_float_values(self, logger):
        """log_metrics() handles float metric values."""
        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=100, epoch=1)

        assert len(logger.logged_metrics_history) == 1
        assert logger.logged_metrics_history[0]["loss"] == 0.5
        assert logger.logged_metrics_history[0]["accuracy"] == 0.95

    def test_log_metrics_with_int_values(self, logger):
        """log_metrics() handles integer metric values."""
        metrics = {"correct": 95, "total": 100}
        logger.log_metrics(metrics, step=1)

        assert logger.logged_metrics_history[0]["correct"] == 95
        assert logger.logged_metrics_history[0]["total"] == 100

    def test_log_metrics_with_tensor_values(self, logger):
        """log_metrics() handles torch.Tensor metric values."""
        metrics = {
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor([0.95]),
        }
        logger.log_metrics(metrics, step=1)

        # Should convert tensors to scalars
        assert isinstance(logger.logged_metrics_history[0]["loss"], float)
        assert isinstance(logger.logged_metrics_history[0]["accuracy"], float)

    def test_log_metrics_stores_step_and_epoch(self, logger):
        """log_metrics() stores step and epoch information."""
        logger.log_metrics({"loss": 0.5}, step=100, epoch=5)

        entry = logger.logged_metrics_history[0]
        assert entry["step"] == 100
        assert entry["epoch"] == 5

    def test_log_metrics_writes_to_csv(self, logger, temp_log_dir):
        """log_metrics() writes metrics to CSV file."""
        logger.log_metrics({"loss": 0.5, "acc": 0.95}, step=1, epoch=1)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        assert csv_file.exists()

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["loss"] == "0.5"
            assert rows[0]["acc"] == "0.95"

    def test_log_metrics_multiple_calls(self, logger):
        """Multiple log_metrics() calls accumulate in history."""
        logger.log_metrics({"loss": 1.0}, step=1)
        logger.log_metrics({"loss": 0.8}, step=2)
        logger.log_metrics({"loss": 0.6}, step=3)

        assert len(logger.logged_metrics_history) == 3
        assert logger.logged_metrics_history[0]["loss"] == 1.0
        assert logger.logged_metrics_history[1]["loss"] == 0.8
        assert logger.logged_metrics_history[2]["loss"] == 0.6

    def test_log_metrics_handles_new_fields(self, logger, temp_log_dir):
        """log_metrics() handles new fields being added."""
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.log_metrics({"loss": 0.4, "accuracy": 0.9}, step=2)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            # First row should have loss but empty accuracy
            assert rows[0]["loss"] == "0.5"
            assert rows[0]["accuracy"] == ""
            # Second row should have both
            assert rows[1]["loss"] == "0.4"
            assert rows[1]["accuracy"] == "0.9"

    def test_log_metrics_with_scientific_notation(self, logger):
        """log_metrics() handles very small/large numbers."""
        metrics = {
            "very_small": 1e-10,
            "very_large": 1e10,
            "learning_rate": 3e-4,
        }
        logger.log_metrics(metrics, step=1)

        assert logger.logged_metrics_history[0]["very_small"] == 1e-10
        assert logger.logged_metrics_history[0]["very_large"] == 1e10


@pytest.mark.unit
class TestLogImages:
    """Test the log_images() method."""

    def test_log_images_with_4d_tensor(self, logger, temp_log_dir):
        """log_images() handles 4D image tensor (B, C, H, W)."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1000)

        # Check that image was stored in history
        assert len(logger.logged_images) == 1
        assert logger.logged_images[0]["tag"] == "samples"
        assert logger.logged_images[0]["step"] == 1000

        # Check that file exists in first subdir (samples)
        image_file = Path(temp_log_dir) / "samples" / "samples_step1000.png"
        assert image_file.exists()

    def test_log_images_with_3d_tensor(self, logger, temp_log_dir):
        """log_images() handles 3D image tensor (C, H, W)."""
        image = torch.randn(3, 32, 32)
        logger.log_images(image, tag="sample", step=500)

        assert len(logger.logged_images) == 1
        image_file = Path(temp_log_dir) / "samples" / "sample_step500.png"
        assert image_file.exists()

    def test_log_images_with_epoch(self, logger, temp_log_dir):
        """log_images() includes epoch in filename."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="train", step=1000, epoch=10)

        assert logger.logged_images[0]["epoch"] == 10
        image_file = Path(temp_log_dir) / "samples" / "train_epoch10_step1000.png"
        assert image_file.exists()

    def test_log_images_with_list_of_tensors(self, logger, temp_log_dir):
        """log_images() handles list of image tensors."""
        images = [torch.randn(3, 32, 32) for _ in range(3)]
        logger.log_images(images, tag="samples", step=1000)

        image_file = Path(temp_log_dir) / "samples" / "samples_step1000.png"
        assert image_file.exists()

    def test_log_images_with_normalize_kwarg(self, logger, temp_log_dir):
        """log_images() accepts normalize keyword argument."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1000, normalize=False)

        image_file = Path(temp_log_dir) / "samples" / "samples_step1000.png"
        assert image_file.exists()

    def test_log_images_with_nrow_kwarg(self, logger, temp_log_dir):
        """log_images() accepts nrow keyword argument."""
        images = torch.randn(8, 3, 32, 32)
        logger.log_images(images, tag="grid", step=1000, nrow=4)

        image_file = Path(temp_log_dir) / "samples" / "grid_step1000.png"
        assert image_file.exists()

    def test_log_images_with_value_range_kwarg(self, logger, temp_log_dir):
        """log_images() accepts value_range keyword argument."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1000, value_range=(-1, 1))

        image_file = Path(temp_log_dir) / "samples" / "samples_step1000.png"
        assert image_file.exists()

    def test_log_images_with_custom_save_dir(self, logger, temp_log_dir):
        """log_images() saves to custom save_dir."""
        images = torch.randn(4, 3, 32, 32)
        custom_dir = Path(temp_log_dir) / "denoising"
        logger.log_images(images, tag="test", step=100, save_dir=custom_dir)

        image_file = custom_dir / "test_step100.png"
        assert image_file.exists()

    def test_log_images_defaults_to_log_dir_when_no_subdirs(self, temp_log_dir):
        """log_images() saves to log_dir when no subdirs configured."""
        logger = ExperimentLogger(log_dir=temp_log_dir)
        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="test", step=100)

        image_file = Path(temp_log_dir) / "test_step100.png"
        assert image_file.exists()

    def test_log_images_for_classifier_layout(self, classifier_logger, temp_log_dir):
        """log_images() saves to predictions dir for classifier config."""
        images = torch.randn(4, 3, 32, 32)
        classifier_logger.log_images(images, tag="samples", step=100)

        image_file = Path(temp_log_dir) / "predictions" / "samples_step100.png"
        assert image_file.exists()

    def test_log_images_with_single_channel(self, logger, temp_log_dir):
        """log_images() handles grayscale images."""
        images = torch.randn(4, 1, 28, 28)
        logger.log_images(images, tag="grayscale", step=100)

        image_file = Path(temp_log_dir) / "samples" / "grayscale_step100.png"
        assert image_file.exists()

    def test_log_images_with_class_labels(self, logger, temp_log_dir):
        """log_images() annotates grid when class_labels provided."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_images(
            images, tag="conditional", step=100, class_labels=[0, 0, 1, 1]
        )

        image_file = Path(temp_log_dir) / "samples" / "conditional_step100.png"
        assert image_file.exists()
        # Verify labels are stored in history
        assert logger.logged_images[0]["class_labels"] == [0, 0, 1, 1]

    def test_log_images_without_class_labels_stores_none(self, logger):
        """log_images() stores None for class_labels when not provided."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="plain", step=1)
        assert logger.logged_images[0]["class_labels"] is None

    def test_log_images_stored_tensors_are_on_cpu(self, logger):
        """log_images() stores tensors on CPU regardless of input device."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1)
        stored = logger.logged_images[0]["images"]
        assert stored.device.type == "cpu"

    def test_log_images_stored_tensors_have_no_grad(self, logger):
        """log_images() stored tensors are detached (no grad_fn)."""
        images = torch.randn(4, 3, 32, 32, requires_grad=True)
        images = images * 2  # creates a grad_fn
        logger.log_images(images, tag="samples", step=1)
        stored = logger.logged_images[0]["images"]
        assert stored.grad_fn is None
        assert not stored.requires_grad


@pytest.mark.unit
class TestLogHyperparams:
    """Test the log_hyperparams() method."""

    def test_log_hyperparams_does_not_raise(self, logger):
        """log_hyperparams() completes without raising."""
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam",
        }
        logger.log_hyperparams(hyperparams)  # should not raise

    def test_log_hyperparams_does_not_write_yaml(self, logger, temp_log_dir):
        """log_hyperparams() does not write hyperparams.yaml."""
        logger.log_hyperparams({"learning_rate": 0.001, "epochs": 10})

        yaml_file = Path(temp_log_dir) / "hyperparams.yaml"
        assert not yaml_file.exists()

    def test_log_hyperparams_handles_various_types(self, logger):
        """log_hyperparams() handles various data types without raising."""
        hyperparams = {
            "float_val": 0.001,
            "int_val": 32,
            "str_val": "adam",
            "bool_val": True,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
        }
        logger.log_hyperparams(hyperparams)  # should not raise


@pytest.mark.unit
class TestContextManager:
    """Test context manager functionality."""

    def test_logger_supports_context_manager(self, temp_log_dir):
        """Logger can be used as a context manager."""
        with ExperimentLogger(log_dir=temp_log_dir) as logger:
            assert isinstance(logger, ExperimentLogger)
            logger.log_metrics({"loss": 0.5}, step=1)

    def test_context_manager_calls_close(self, temp_log_dir):
        """Context manager calls close() on exit."""
        logger = ExperimentLogger(log_dir=temp_log_dir)

        with logger:
            logger.log_metrics({"loss": 0.5}, step=1)

        # Should have called close() - no exceptions raised


@pytest.mark.unit
class TestTensorBoardIntegration:
    """Test TensorBoard integration in ExperimentLogger."""

    def test_tb_writer_none_by_default(self, temp_log_dir):
        """tb_writer is None when TensorBoard is not configured."""
        logger = ExperimentLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

    def test_tb_writer_none_when_disabled(self, temp_log_dir):
        """tb_writer is None when TensorBoard is explicitly disabled."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": False},
        )
        assert logger.tb_writer is None

    def test_tb_writer_created_when_enabled(self, temp_log_dir):
        """tb_writer is created when TensorBoard is enabled."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()

    def test_tb_log_dir_defaults_to_sibling_tensorboard(self, temp_log_dir):
        """Default TensorBoard log_dir is a sibling 'tensorboard' directory."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        expected_dir = Path(temp_log_dir).parent / "tensorboard"
        assert expected_dir.exists()
        logger.close()

    def test_tb_log_dir_custom(self, tmp_path):
        """Custom tb_log_dir places TensorBoard files in the specified directory."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "tb_custom"

        logger = ExperimentLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=tb_dir,
        )
        assert tb_dir.exists()
        logger.close()

    def test_log_metrics_calls_add_scalar_when_tb_enabled(self, temp_log_dir):
        """log_metrics() calls writer.add_scalar when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ExperimentLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.metrics_writer.tb_writer = mock_writer

        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=10)

        assert mock_writer.add_scalar.call_count == 2
        call_tags = {call[0][0] for call in mock_writer.add_scalar.call_args_list}
        assert "metrics/loss" in call_tags
        assert "metrics/accuracy" in call_tags

    def test_log_metrics_does_not_call_add_scalar_when_tb_disabled(self, temp_log_dir):
        """log_metrics() does not call add_scalar when TensorBoard is disabled."""
        logger = ExperimentLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

        logger.log_metrics({"loss": 0.5}, step=1)
        assert len(logger.logged_metrics_history) == 1

    def test_log_images_calls_add_images_when_tb_enabled(self, temp_log_dir):
        """log_images() calls writer.add_images when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            subdirs={"images": "samples"},
            tensorboard_config={"enabled": True, "log_images": True},
        )
        logger.metrics_writer.tb_writer = MagicMock()

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=5)

        assert logger.metrics_writer.tb_writer is not None
        logger.metrics_writer.tb_writer.add_images.assert_called_once()

    def test_log_images_skipped_when_log_images_false(self, temp_log_dir):
        """log_images() skips TensorBoard when log_images config is False."""
        from unittest.mock import MagicMock

        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            subdirs={"images": "samples"},
            tensorboard_config={"enabled": True, "log_images": False},
        )
        mock_writer = MagicMock()
        logger.metrics_writer.tb_writer = mock_writer
        logger.tb_log_images = False

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=5)

        mock_writer.add_images.assert_not_called()

    def test_log_hyperparams_calls_add_hparams_when_tb_enabled(self, temp_log_dir):
        """log_hyperparams() calls writer.add_hparams when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ExperimentLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.metrics_writer.tb_writer = mock_writer

        logger.log_hyperparams({"lr": 0.001, "batch_size": 32})

        mock_writer.add_hparams.assert_called_once()

    def test_close_sets_tb_writer_to_none(self, temp_log_dir):
        """close() sets tb_writer to None."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()
        assert logger.tb_writer is None

    def test_csv_logging_unchanged_when_tb_enabled(self, temp_log_dir):
        """CSV logging works correctly regardless of TensorBoard state."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        logger.log_metrics({"loss": 0.5, "acc": 0.95}, step=1, epoch=1)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        assert csv_file.exists()
        import csv as csv_module

        with open(csv_file) as f:
            rows = list(csv_module.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["loss"] == "0.5"
        logger.close()

    def test_tb_event_files_created_when_enabled(self, tmp_path):
        """TensorBoard event files are created in the log directory when enabled."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "tb"

        logger = ExperimentLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=tb_dir,
        )
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.close()

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_graceful_when_tensorboard_not_installed(self, temp_log_dir):
        """Logger degrades gracefully when tensorboard package is missing."""
        from unittest.mock import patch

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            logger = ExperimentLogger(
                log_dir=temp_log_dir,
                tensorboard_config={"enabled": True},
            )
            assert logger.tb_writer is None

            logger.log_metrics({"loss": 0.5}, step=1)
            assert len(logger.logged_metrics_history) == 1


@pytest.mark.component
class TestLoggerIntegration:
    """Integration tests for typical logger usage patterns."""

    def test_typical_training_loop_logging(self, logger):
        """Test typical usage pattern in a training loop."""
        for epoch in range(3):
            for step in range(5):
                logger.log_metrics(
                    {
                        "loss": 1.0 / (epoch + step + 1),
                        "accuracy": 0.5 + 0.1 * (epoch * 5 + step),
                    },
                    step=epoch * 5 + step,
                    epoch=epoch,
                )

            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="epoch_samples", step=epoch * 5, epoch=epoch)

        assert len(logger.logged_metrics_history) == 15
        assert len(logger.logged_images) == 3

    def test_complete_diffusion_workflow(self, temp_log_dir):
        """Test complete diffusion experiment workflow."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            subdirs={"images": "samples", "denoising": "denoising"},
        )

        logger.log_hyperparams({"lr": 0.0001, "batch_size": 64})

        for epoch in range(3):
            logger.log_metrics(
                {"loss": 0.5 - epoch * 0.1},
                step=epoch,
                epoch=epoch,
            )
            samples = torch.randn(8, 3, 32, 32)
            logger.log_images(samples, tag="samples", step=epoch, epoch=epoch)

        logger.close()

        log_dir = Path(temp_log_dir)
        assert (log_dir / "metrics" / "metrics.csv").exists()
        assert len(list((log_dir / "samples").glob("*.png"))) == 3

    def test_complete_classifier_workflow(self, temp_log_dir):
        """Test complete classifier experiment workflow."""
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            subdirs={
                "images": "predictions",
                "confusion_matrices": "confusion_matrices",
            },
        )

        logger.log_hyperparams({"model": "ResNet50", "lr": 0.001})

        for epoch in range(3):
            logger.log_metrics(
                {"train_loss": 0.5 - epoch * 0.1, "train_acc": 0.8 + epoch * 0.05},
                step=epoch,
                epoch=epoch,
            )
            val_images = torch.randn(8, 3, 64, 64)
            logger.log_images(
                val_images, tag="val_predictions", step=epoch, epoch=epoch
            )

        logger.close()

        log_dir = Path(temp_log_dir)
        assert (log_dir / "metrics" / "metrics.csv").exists()
        assert len(list((log_dir / "predictions").glob("*.png"))) >= 3

    def test_csv_persistence_across_sessions(self, temp_log_dir):
        """Test that CSV metrics persist across logger sessions."""
        logger1 = ExperimentLogger(log_dir=temp_log_dir)
        logger1.log_metrics({"loss": 1.0, "acc": 0.5}, step=1)
        logger1.log_metrics({"loss": 0.8, "acc": 0.6}, step=2)
        logger1.close()

        logger2 = ExperimentLogger(log_dir=temp_log_dir)
        logger2.log_metrics({"loss": 0.6, "acc": 0.7}, step=3)
        logger2.close()

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            assert rows[0]["loss"] == "1.0"
            assert rows[1]["loss"] == "0.8"
            assert rows[2]["loss"] == "0.6"
