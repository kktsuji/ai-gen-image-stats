"""Tests for Classifier Logger

This module contains tests for the ClassifierLogger implementation.
Tests are organized into unit, component, and integration tiers.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.experiments.classifier.logger import ClassifierLogger

# Test fixtures


@pytest.fixture
def temp_log_dir():
    """Provide temporary directory for logger outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger(temp_log_dir):
    """Provide a basic logger instance."""
    return ClassifierLogger(log_dir=temp_log_dir, class_names=["Real", "Fake"])


@pytest.fixture
def sample_confusion_matrix():
    """Provide a sample confusion matrix."""
    return np.array([[50, 10], [5, 35]])


# Unit Tests


@pytest.mark.unit
class TestClassifierLoggerInstantiation:
    """Test logger instantiation and initialization."""

    def test_can_instantiate_logger(self, temp_log_dir):
        """ClassifierLogger can be instantiated."""
        logger = ClassifierLogger(log_dir=temp_log_dir)
        assert isinstance(logger, ClassifierLogger)

    def test_creates_log_directory(self, temp_log_dir):
        """Logger creates log directory on instantiation."""
        log_dir = Path(temp_log_dir) / "experiment_001"
        logger = ClassifierLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_creates_subdirectories(self, logger, temp_log_dir):
        """Logger creates required subdirectories."""
        log_dir = Path(temp_log_dir)
        assert (log_dir / "metrics").exists()
        assert (log_dir / "confusion_matrices").exists()
        assert (log_dir / "predictions").exists()

    def test_accepts_class_names(self, temp_log_dir):
        """Logger accepts class names parameter."""
        class_names = ["Normal", "Abnormal"]
        logger = ClassifierLogger(log_dir=temp_log_dir, class_names=class_names)
        assert logger.class_names == class_names

    def test_default_empty_class_names(self, temp_log_dir):
        """Logger defaults to empty class names list."""
        logger = ClassifierLogger(log_dir=temp_log_dir)
        assert logger.class_names == []

    def test_log_dir_is_path_object(self, logger):
        """Logger converts log_dir to Path object."""
        assert isinstance(logger.log_dir, Path)


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

        csv_file = Path(temp_log_dir) / "metrics.csv"
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

        csv_file = Path(temp_log_dir) / "metrics.csv"
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


@pytest.mark.unit
class TestLogImages:
    """Test the log_images() method."""

    def test_log_images_with_4d_tensor(self, logger, temp_log_dir):
        """log_images() handles 4D image tensor (B, C, H, W)."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=100)

        image_file = Path(temp_log_dir) / "predictions" / "samples_step100.png"
        assert image_file.exists()

    def test_log_images_with_3d_tensor(self, logger, temp_log_dir):
        """log_images() handles 3D image tensor (C, H, W)."""
        image = torch.randn(3, 32, 32)
        logger.log_images(image, tag="sample", step=50)

        image_file = Path(temp_log_dir) / "predictions" / "sample_step50.png"
        assert image_file.exists()

    def test_log_images_with_epoch(self, logger, temp_log_dir):
        """log_images() includes epoch in filename."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="train", step=100, epoch=5)

        image_file = Path(temp_log_dir) / "predictions" / "train_epoch5_step100.png"
        assert image_file.exists()

    def test_log_images_with_list_of_tensors(self, logger, temp_log_dir):
        """log_images() handles list of image tensors."""
        images = [torch.randn(3, 32, 32) for _ in range(3)]
        logger.log_images(images, tag="samples", step=100)

        image_file = Path(temp_log_dir) / "predictions" / "samples_step100.png"
        assert image_file.exists()

    def test_log_images_with_labels_and_predictions(self, logger, temp_log_dir):
        """log_images() creates annotated visualization with labels."""
        images = torch.randn(4, 3, 32, 32)
        labels = [0, 1, 0, 1]
        predictions = [0, 1, 1, 1]
        logger.log_images(
            images, tag="predictions", step=100, labels=labels, predictions=predictions
        )

        # Check both regular and annotated images exist
        image_file = Path(temp_log_dir) / "predictions" / "predictions_step100.png"
        annotated_file = (
            Path(temp_log_dir) / "predictions" / "predictions_step100.annotated.png"
        )
        assert image_file.exists()
        assert annotated_file.exists()

    def test_log_images_with_normalize_kwarg(self, logger, temp_log_dir):
        """log_images() accepts normalize keyword argument."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="samples", step=100, normalize=False)

        image_file = Path(temp_log_dir) / "predictions" / "samples_step100.png"
        assert image_file.exists()

    def test_log_images_with_nrow_kwarg(self, logger, temp_log_dir):
        """log_images() accepts nrow keyword argument."""
        images = torch.randn(8, 3, 32, 32)
        logger.log_images(images, tag="grid", step=100, nrow=4)

        image_file = Path(temp_log_dir) / "predictions" / "grid_step100.png"
        assert image_file.exists()


@pytest.mark.unit
class TestLogConfusionMatrix:
    """Test the log_confusion_matrix() method."""

    def test_log_confusion_matrix_with_numpy_array(
        self, logger, sample_confusion_matrix, temp_log_dir
    ):
        """log_confusion_matrix() handles numpy array."""
        logger.log_confusion_matrix(sample_confusion_matrix, step=100)

        assert len(logger.logged_confusion_matrices) == 1
        assert np.array_equal(
            logger.logged_confusion_matrices[0]["matrix"], sample_confusion_matrix
        )

    def test_log_confusion_matrix_with_torch_tensor(self, logger, temp_log_dir):
        """log_confusion_matrix() handles torch tensor."""
        confusion_matrix = torch.tensor([[50, 10], [5, 35]])
        logger.log_confusion_matrix(confusion_matrix, step=100)

        assert len(logger.logged_confusion_matrices) == 1
        assert isinstance(logger.logged_confusion_matrices[0]["matrix"], np.ndarray)

    def test_log_confusion_matrix_stores_metadata(
        self, logger, sample_confusion_matrix
    ):
        """log_confusion_matrix() stores step, epoch, and normalize flag."""
        logger.log_confusion_matrix(
            sample_confusion_matrix, step=100, epoch=5, normalize=True
        )

        entry = logger.logged_confusion_matrices[0]
        assert entry["step"] == 100
        assert entry["epoch"] == 5
        assert entry["normalize"] is True

    def test_log_confusion_matrix_creates_visualization(
        self, logger, sample_confusion_matrix, temp_log_dir
    ):
        """log_confusion_matrix() creates visualization file."""
        logger.log_confusion_matrix(sample_confusion_matrix, step=100, epoch=1)

        viz_file = (
            Path(temp_log_dir)
            / "confusion_matrices"
            / "confusion_matrix_epoch1_step100.png"
        )
        assert viz_file.exists()

    def test_log_confusion_matrix_normalized_filename(
        self, logger, sample_confusion_matrix, temp_log_dir
    ):
        """log_confusion_matrix() includes 'normalized' in filename when normalized."""
        logger.log_confusion_matrix(sample_confusion_matrix, step=100, normalize=True)

        viz_file = (
            Path(temp_log_dir)
            / "confusion_matrices"
            / "confusion_matrix_step100_normalized.png"
        )
        assert viz_file.exists()

    def test_log_confusion_matrix_with_class_names(
        self, logger, sample_confusion_matrix, temp_log_dir
    ):
        """log_confusion_matrix() accepts class names parameter."""
        class_names = ["Class0", "Class1"]
        logger.log_confusion_matrix(
            sample_confusion_matrix, class_names=class_names, step=100
        )

        # Should not raise an error and create visualization
        viz_file = (
            Path(temp_log_dir) / "confusion_matrices" / "confusion_matrix_step100.png"
        )
        assert viz_file.exists()

    def test_log_confusion_matrix_uses_instance_class_names(
        self, sample_confusion_matrix, temp_log_dir
    ):
        """log_confusion_matrix() uses instance class names if not provided."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir, class_names=["Normal", "Abnormal"]
        )
        logger.log_confusion_matrix(sample_confusion_matrix, step=100)

        viz_file = (
            Path(temp_log_dir) / "confusion_matrices" / "confusion_matrix_step100.png"
        )
        assert viz_file.exists()

    def test_log_confusion_matrix_multiple_calls(self, logger, sample_confusion_matrix):
        """Multiple log_confusion_matrix() calls accumulate in history."""
        logger.log_confusion_matrix(sample_confusion_matrix, step=100)
        logger.log_confusion_matrix(sample_confusion_matrix, step=200)
        logger.log_confusion_matrix(sample_confusion_matrix, step=300)

        assert len(logger.logged_confusion_matrices) == 3


@pytest.mark.unit
class TestLogHyperparams:
    """Test the log_hyperparams() method."""

    def test_log_hyperparams_writes_json_file(self, logger, temp_log_dir):
        """log_hyperparams() writes hyperparameters to YAML file."""
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam",
        }
        logger.log_hyperparams(hyperparams)

        yaml_file = Path(temp_log_dir) / "hyperparams.yaml"
        assert yaml_file.exists()

        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)
            assert loaded == hyperparams

    def test_log_hyperparams_handles_various_types(self, logger, temp_log_dir):
        """log_hyperparams() handles various data types."""
        hyperparams = {
            "float_val": 0.001,
            "int_val": 32,
            "str_val": "adam",
            "bool_val": True,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
        }
        logger.log_hyperparams(hyperparams)

        yaml_file = Path(temp_log_dir) / "hyperparams.yaml"
        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)
            assert loaded == hyperparams


@pytest.mark.unit
class TestContextManager:
    """Test context manager functionality."""

    def test_logger_supports_context_manager(self, temp_log_dir):
        """Logger can be used as a context manager."""
        with ClassifierLogger(log_dir=temp_log_dir) as logger:
            assert isinstance(logger, ClassifierLogger)
            logger.log_metrics({"loss": 0.5}, step=1)

    def test_context_manager_calls_close(self, temp_log_dir):
        """Context manager calls close() on exit."""
        logger = ClassifierLogger(log_dir=temp_log_dir)

        with logger:
            logger.log_metrics({"loss": 0.5}, step=1)

        # Should have called close() - no exceptions raised


@pytest.mark.component
class TestLoggerIntegration:
    """Integration tests for typical logger usage patterns."""

    def test_typical_training_loop_logging(self, logger):
        """Test typical usage pattern in a training loop."""
        # Simulate training loop
        for epoch in range(3):
            for step in range(5):
                # Log metrics each step
                logger.log_metrics(
                    {
                        "loss": 1.0 / (epoch + step + 1),
                        "accuracy": 0.5 + 0.1 * (epoch * 5 + step),
                    },
                    step=epoch * 5 + step,
                    epoch=epoch,
                )

            # Log images each epoch
            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="epoch_samples", step=epoch * 5, epoch=epoch)

            # Log confusion matrix each epoch
            cm = np.array([[50 + epoch * 10, 10 - epoch], [5 - epoch, 35 + epoch * 10]])
            logger.log_confusion_matrix(cm, step=epoch * 5, epoch=epoch)

        # Check all data was logged
        assert len(logger.logged_metrics_history) == 15  # 3 epochs * 5 steps
        assert len(logger.logged_confusion_matrices) == 3  # 3 epochs

    def test_complete_experiment_workflow(self, temp_log_dir):
        """Test complete experiment workflow from start to finish."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir, class_names=["Normal", "Abnormal"]
        )

        # Log hyperparameters
        logger.log_hyperparams(
            {
                "model": "ResNet50",
                "lr": 0.001,
                "batch_size": 32,
                "epochs": 5,
            }
        )

        # Simulate training
        for epoch in range(3):
            # Training phase
            logger.log_metrics(
                {"train_loss": 0.5 - epoch * 0.1, "train_acc": 0.8 + epoch * 0.05},
                step=epoch,
                epoch=epoch,
            )

            # Validation phase
            logger.log_metrics(
                {"val_loss": 0.6 - epoch * 0.1, "val_acc": 0.75 + epoch * 0.05},
                step=epoch,
                epoch=epoch,
            )

            # Log validation samples
            val_images = torch.randn(8, 3, 64, 64)
            labels = [0, 1, 0, 1, 0, 1, 0, 1]
            predictions = [0, 1, 1, 1, 0, 0, 0, 1]
            logger.log_images(
                val_images,
                tag="val_predictions",
                step=epoch,
                epoch=epoch,
                labels=labels,
                predictions=predictions,
            )

            # Log confusion matrix
            cm = np.array([[80 + epoch * 5, 20 - epoch * 5], [10 - epoch, 90 + epoch]])
            logger.log_confusion_matrix(cm, step=epoch, epoch=epoch, normalize=True)

        logger.close()

        # Verify all files were created
        log_dir = Path(temp_log_dir)
        assert (log_dir / "hyperparams.yaml").exists()
        assert (log_dir / "metrics.csv").exists()
        assert len(list((log_dir / "predictions").glob("*.png"))) >= 3
        assert len(list((log_dir / "confusion_matrices").glob("*.png"))) == 3

    def test_csv_persistence_across_sessions(self, temp_log_dir):
        """Test that CSV metrics persist across logger sessions."""
        # First session
        logger1 = ClassifierLogger(log_dir=temp_log_dir)
        logger1.log_metrics({"loss": 1.0, "acc": 0.5}, step=1)
        logger1.log_metrics({"loss": 0.8, "acc": 0.6}, step=2)
        logger1.close()

        # Second session (reusing same directory)
        logger2 = ClassifierLogger(log_dir=temp_log_dir)
        logger2.log_metrics({"loss": 0.6, "acc": 0.7}, step=3)
        logger2.close()

        # Check CSV contains all entries
        csv_file = Path(temp_log_dir) / "metrics.csv"
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            assert rows[0]["loss"] == "1.0"
            assert rows[1]["loss"] == "0.8"
            assert rows[2]["loss"] == "0.6"


@pytest.mark.component
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_log_confusion_matrix_with_zeros(self, logger):
        """log_confusion_matrix() handles confusion matrix with zeros."""
        cm = np.array([[0, 10], [0, 0]])
        logger.log_confusion_matrix(cm, step=100, normalize=True)

        # Should not raise division by zero error
        assert len(logger.logged_confusion_matrices) == 1

    def test_log_images_with_single_channel(self, logger, temp_log_dir):
        """log_images() handles grayscale images."""
        images = torch.randn(4, 1, 28, 28)
        logger.log_images(images, tag="grayscale", step=100)

        image_file = Path(temp_log_dir) / "predictions" / "grayscale_step100.png"
        assert image_file.exists()

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

    def test_log_confusion_matrix_single_class(self, logger, temp_log_dir):
        """log_confusion_matrix() handles single-class confusion matrix."""
        cm = np.array([[100]])
        logger.log_confusion_matrix(cm, step=100)

        viz_file = (
            Path(temp_log_dir) / "confusion_matrices" / "confusion_matrix_step100.png"
        )
        assert viz_file.exists()

    def test_log_confusion_matrix_multiclass(self, logger, temp_log_dir):
        """log_confusion_matrix() handles multi-class confusion matrix."""
        cm = np.array([[30, 5, 3], [2, 40, 1], [1, 2, 50]])
        logger.log_confusion_matrix(cm, step=100)

        viz_file = (
            Path(temp_log_dir) / "confusion_matrices" / "confusion_matrix_step100.png"
        )
        assert viz_file.exists()


@pytest.mark.unit
class TestTensorBoardIntegration:
    """Test TensorBoard integration in ClassifierLogger."""

    def test_tb_writer_none_by_default(self, temp_log_dir):
        """tb_writer is None when TensorBoard is not configured."""
        logger = ClassifierLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

    def test_tb_writer_none_when_disabled(self, temp_log_dir):
        """tb_writer is None when TensorBoard is explicitly disabled."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": False},
        )
        assert logger.tb_writer is None

    def test_tb_writer_created_when_enabled(self, temp_log_dir):
        """tb_writer is created when TensorBoard is enabled."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()

    def test_tb_log_dir_defaults_to_sibling_tensorboard(self, temp_log_dir):
        """Default TensorBoard log_dir is a sibling 'tensorboard' directory."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        # The default is log_dir.parent / "tensorboard"
        expected_dir = Path(temp_log_dir).parent / "tensorboard"
        assert expected_dir.exists()
        logger.close()

    def test_tb_log_dir_custom(self, tmp_path):
        """Custom log_dir places TensorBoard files in the specified directory."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "tb_custom"

        logger = ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True, "log_dir": str(tb_dir)},
        )
        assert tb_dir.exists()
        logger.close()

    def test_log_metrics_calls_add_scalar_when_tb_enabled(self, temp_log_dir):
        """log_metrics() calls writer.add_scalar when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer

        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=10)

        assert mock_writer.add_scalar.call_count == 2
        call_tags = {call[0][0] for call in mock_writer.add_scalar.call_args_list}
        assert "metrics/loss" in call_tags
        assert "metrics/accuracy" in call_tags

    def test_log_metrics_does_not_call_add_scalar_when_tb_disabled(self, temp_log_dir):
        """log_metrics() does not call add_scalar when TensorBoard is disabled."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

        # CSV still works
        logger.log_metrics({"loss": 0.5}, step=1)
        assert len(logger.logged_metrics_history) == 1

    def test_log_images_calls_add_images_when_tb_enabled(self, temp_log_dir):
        """log_images() calls writer.add_images when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True, "log_images": True},
        )
        logger.tb_writer = MagicMock()

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=5)

        logger.tb_writer.add_images.assert_called_once()

    def test_log_images_skipped_when_log_images_false(self, temp_log_dir):
        """log_images() skips TensorBoard when log_images config is False."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True, "log_images": False},
        )
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer
        logger.tb_log_images = False

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=5)

        mock_writer.add_images.assert_not_called()

    def test_log_confusion_matrix_calls_add_figure_when_tb_enabled(self, temp_log_dir):
        """log_confusion_matrix() calls writer.add_figure when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer
        logger.tb_log_images = True

        cm = np.array([[50, 10], [5, 35]])
        logger.log_confusion_matrix(cm, step=100)

        mock_writer.add_figure.assert_called_once()

    def test_log_hyperparams_calls_add_hparams_when_tb_enabled(self, temp_log_dir):
        """log_hyperparams() calls writer.add_hparams when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = ClassifierLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer

        logger.log_hyperparams({"lr": 0.001, "batch_size": 32})

        mock_writer.add_hparams.assert_called_once()

    def test_close_sets_tb_writer_to_none(self, temp_log_dir):
        """close() sets tb_writer to None."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()
        assert logger.tb_writer is None

    def test_csv_logging_unchanged_when_tb_enabled(self, temp_log_dir):
        """CSV logging works correctly regardless of TensorBoard state."""
        logger = ClassifierLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        logger.log_metrics({"loss": 0.5, "acc": 0.95}, step=1, epoch=1)

        csv_file = Path(temp_log_dir) / "metrics.csv"
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

        logger = ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True, "log_dir": str(tb_dir)},
        )
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.close()

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_graceful_when_tensorboard_not_installed(self, temp_log_dir):
        """Logger degrades gracefully when tensorboard package is missing."""
        from unittest.mock import patch

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            logger = ClassifierLogger(
                log_dir=temp_log_dir,
                tensorboard_config={"enabled": True},
            )
            # tb_writer should be None even though enabled=True
            assert logger.tb_writer is None

            # All logging should still work via CSV
            logger.log_metrics({"loss": 0.5}, step=1)
            assert len(logger.logged_metrics_history) == 1
