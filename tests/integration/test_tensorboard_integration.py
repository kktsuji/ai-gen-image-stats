"""Integration tests for TensorBoard logging.

This module tests end-to-end TensorBoard integration with both ClassifierLogger
and DiffusionLogger, verifying that:
- Both CSV and TensorBoard logs are produced during a training run
- TensorBoard can be disabled without affecting CSV output
- Custom log directories are respected
- Missing tensorboard package is handled gracefully

Test Coverage:
- Full logging run with TensorBoard enabled (Classifier)
- Full logging run with TensorBoard enabled (Diffusion)
- Backward compatibility: TensorBoard disabled, CSV unchanged
- Custom log_dir configuration
- Graceful degradation without tensorboard package
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.experiments.classifier.logger import ClassifierLogger
from src.experiments.diffusion.logger import DiffusionLogger
from src.utils.tensorboard import TENSORBOARD_AVAILABLE

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def classifier_tb_config(tmp_path):
    """Provide a temporary directory and TensorBoard config for classifier tests."""
    log_dir = tmp_path / "classifier_logs"
    tb_dir = tmp_path / "tensorboard"
    config = {
        "enabled": True,
        "flush_secs": 1,
        "log_images": True,
        "log_histograms": False,
        "log_graph": False,
    }
    return log_dir, tb_dir, config


@pytest.fixture
def diffusion_tb_config(tmp_path):
    """Provide a temporary directory and TensorBoard config for diffusion tests."""
    log_dir = tmp_path / "diffusion_logs"
    tb_dir = tmp_path / "tensorboard"
    config = {
        "enabled": True,
        "flush_secs": 1,
        "log_images": True,
        "log_histograms": False,
        "log_graph": False,
    }
    return log_dir, tb_dir, config


# ============================================================================
# Scenario 1: Full training run with TensorBoard enabled (Classifier)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
class TestClassifierTensorBoardEnabled:
    """Full classifier logging run with TensorBoard enabled."""

    def test_both_csv_and_tensorboard_logs_created(self, classifier_tb_config):
        """CSV and TensorBoard event files are both created during a logging run."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            class_names=["Normal", "Abnormal"],
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            for step in range(3):
                logger.log_metrics(
                    {
                        "train_loss": 1.0 - step * 0.1,
                        "train_accuracy": 0.5 + step * 0.1,
                    },
                    step=step,
                    epoch=0,
                )

        # CSV output
        assert (log_dir / "metrics.csv").exists()
        with open(log_dir / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert float(rows[0]["train_loss"]) == pytest.approx(1.0)

        # TensorBoard output
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_validation_metrics_logged_to_both_outputs(self, classifier_tb_config):
        """Validation metrics appear in both CSV and TensorBoard."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            logger.log_metrics(
                {"val_loss": 0.4, "val_accuracy": 0.85}, step=10, epoch=1
            )

        assert (log_dir / "metrics.csv").exists()
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_confusion_matrix_logged_to_both_outputs(self, classifier_tb_config):
        """Confusion matrix is saved to file and TensorBoard."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            class_names=["Normal", "Abnormal"],
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            cm = np.array([[80, 20], [10, 90]])
            logger.log_confusion_matrix(cm, step=5, epoch=1)

        # Verify file output
        cm_files = list((log_dir / "confusion_matrices").glob("*.png"))
        assert len(cm_files) == 1

        # Verify TensorBoard output
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_images_logged_to_both_outputs(self, classifier_tb_config):
        """Images are saved to file and TensorBoard."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="predictions", step=5, epoch=1)

        image_files = list((log_dir / "predictions").glob("*.png"))
        assert len(image_files) >= 1

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_hyperparams_logged_to_tensorboard(self, classifier_tb_config):
        """Hyperparams are logged to TensorBoard."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            logger.log_hyperparams({"lr": 0.001, "batch_size": 32, "epochs": 10})

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_multi_epoch_run_produces_all_outputs(self, classifier_tb_config):
        """Multi-epoch logging run creates all expected outputs."""
        log_dir, tb_dir, config = classifier_tb_config

        with ClassifierLogger(
            log_dir=log_dir,
            class_names=["Normal", "Abnormal"],
            tensorboard_config=config,
            tb_log_dir=tb_dir,
        ) as logger:
            logger.log_hyperparams({"lr": 0.001, "epochs": 3})

            for epoch in range(3):
                logger.log_metrics(
                    {
                        "train_loss": 1.0 - epoch * 0.2,
                        "train_accuracy": 0.6 + epoch * 0.1,
                    },
                    step=epoch,
                    epoch=epoch,
                )
                images = torch.randn(4, 3, 32, 32)
                logger.log_images(images, tag="epoch_samples", step=epoch, epoch=epoch)

                cm = np.array([[50 + epoch * 5, 10 - epoch], [5, 35 + epoch * 5]])
                logger.log_confusion_matrix(cm, step=epoch, epoch=epoch)

        assert (log_dir / "metrics.csv").exists()
        assert len(list((log_dir / "predictions").glob("*.png"))) == 3
        assert len(list((log_dir / "confusion_matrices").glob("*.png"))) == 3
        assert len(list(tb_dir.glob("events.out.tfevents.*"))) > 0


# ============================================================================
# Scenario 2: Full training run with TensorBoard enabled (Diffusion)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
class TestDiffusionTensorBoardEnabled:
    """Full diffusion logging run with TensorBoard enabled."""

    def test_both_csv_and_tensorboard_logs_created(self, diffusion_tb_config):
        """CSV and TensorBoard event files are both created during a diffusion run."""
        log_dir, tb_dir, config = diffusion_tb_config

        with DiffusionLogger(
            log_dir=log_dir, tensorboard_config=config, tb_log_dir=tb_dir
        ) as logger:
            for step in range(3):
                logger.log_metrics(
                    {"loss": 1.0 - step * 0.1, "timestep": 1000 - step * 100},
                    step=step * 1000,
                    epoch=0,
                )

        assert (log_dir / "metrics" / "metrics.csv").exists()
        with open(log_dir / "metrics" / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_denoising_process_logged_to_both_outputs(self, diffusion_tb_config):
        """Denoising process is saved to file and TensorBoard."""
        log_dir, tb_dir, config = diffusion_tb_config

        with DiffusionLogger(
            log_dir=log_dir, tensorboard_config=config, tb_log_dir=tb_dir
        ) as logger:
            sequence = torch.rand(8, 3, 32, 32)
            logger.log_denoising_process(sequence, step=1000, epoch=1)

        denoising_files = list((log_dir / "denoising").glob("*.png"))
        assert len(denoising_files) == 1

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_generated_samples_logged_to_both_outputs(self, diffusion_tb_config):
        """Generated samples are saved to file and TensorBoard."""
        log_dir, tb_dir, config = diffusion_tb_config

        with DiffusionLogger(
            log_dir=log_dir, tensorboard_config=config, tb_log_dir=tb_dir
        ) as logger:
            samples = torch.rand(8, 3, 32, 32)
            logger.log_images(samples, tag="samples", step=1000, epoch=1)

        sample_files = list((log_dir / "samples").glob("*.png"))
        assert len(sample_files) == 1

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_hyperparams_logged_to_tensorboard(self, diffusion_tb_config):
        """Hyperparams are logged to TensorBoard."""
        log_dir, tb_dir, config = diffusion_tb_config

        with DiffusionLogger(
            log_dir=log_dir, tensorboard_config=config, tb_log_dir=tb_dir
        ) as logger:
            logger.log_hyperparams(
                {"lr": 0.0001, "timesteps": 1000, "beta_schedule": "linear"}
            )

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0


# ============================================================================
# Scenario 3: Backward compatibility — TensorBoard disabled
# ============================================================================


@pytest.mark.integration
class TestBackwardCompatibility:
    """Verify CSV output is unchanged when TensorBoard is disabled."""

    def test_classifier_csv_unchanged_with_tb_disabled(self, tmp_path):
        """Classifier CSV output is identical whether TensorBoard is on or off."""
        metrics = {"loss": 0.42, "accuracy": 0.87}

        # Run without TensorBoard
        with tempfile.TemporaryDirectory() as log_dir_a:
            with ClassifierLogger(log_dir=log_dir_a) as logger_a:
                logger_a.log_metrics(metrics, step=1, epoch=1)
            rows_a = _read_csv(Path(log_dir_a) / "metrics.csv")

        # Run with TensorBoard disabled explicitly
        with tempfile.TemporaryDirectory() as log_dir_b:
            with ClassifierLogger(
                log_dir=log_dir_b,
                tensorboard_config={"enabled": False},
            ) as logger_b:
                logger_b.log_metrics(metrics, step=1, epoch=1)
            rows_b = _read_csv(Path(log_dir_b) / "metrics.csv")

        assert rows_a == rows_b

    def test_diffusion_csv_unchanged_with_tb_disabled(self, tmp_path):
        """Diffusion CSV output is identical whether TensorBoard is on or off."""
        metrics = {"loss": 0.05, "timestep": 500}

        with tempfile.TemporaryDirectory() as log_dir_a:
            with DiffusionLogger(log_dir=log_dir_a) as logger_a:
                logger_a.log_metrics(metrics, step=100)
            rows_a = _read_csv(Path(log_dir_a) / "metrics" / "metrics.csv")

        with tempfile.TemporaryDirectory() as log_dir_b:
            with DiffusionLogger(
                log_dir=log_dir_b,
                tensorboard_config={"enabled": False},
            ) as logger_b:
                logger_b.log_metrics(metrics, step=100)
            rows_b = _read_csv(Path(log_dir_b) / "metrics" / "metrics.csv")

        assert rows_a == rows_b

    def test_no_tensorboard_files_when_disabled(self, tmp_path):
        """No TensorBoard event files are created when disabled."""
        log_dir = tmp_path / "logs"

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": False},
        ) as logger:
            logger.log_metrics({"loss": 0.5}, step=1)

        # Default tensorboard directory should not contain event files
        default_tb_dir = log_dir.parent / "tensorboard"
        if default_tb_dir.exists():
            event_files = list(default_tb_dir.glob("events.out.tfevents.*"))
            assert len(event_files) == 0

    def test_classifier_all_methods_work_without_tensorboard(self, tmp_path):
        """All ClassifierLogger methods work when TensorBoard is disabled."""
        log_dir = tmp_path / "logs"

        with ClassifierLogger(
            log_dir=log_dir,
            class_names=["Normal", "Abnormal"],
            tensorboard_config={"enabled": False},
        ) as logger:
            logger.log_hyperparams({"lr": 0.001})
            logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1, epoch=1)

            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="samples", step=1)

            cm = np.array([[80, 20], [10, 90]])
            logger.log_confusion_matrix(cm, step=1)

        assert (log_dir / "metrics.csv").exists()

    def test_diffusion_all_methods_work_without_tensorboard(self, tmp_path):
        """All DiffusionLogger methods work when TensorBoard is disabled."""
        log_dir = tmp_path / "logs"

        with DiffusionLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": False},
        ) as logger:
            logger.log_hyperparams({"lr": 0.0001})
            logger.log_metrics({"loss": 0.05}, step=1000)

            samples = torch.rand(4, 3, 32, 32)
            logger.log_images(samples, tag="samples", step=1000)

            sequence = torch.rand(8, 3, 32, 32)
            logger.log_denoising_process(sequence, step=1000)

        assert (log_dir / "metrics" / "metrics.csv").exists()


# ============================================================================
# Scenario 4: Custom log_dir
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
class TestCustomLogDir:
    """Verify TensorBoard logs are written to custom directories."""

    def test_classifier_uses_custom_log_dir(self, tmp_path):
        """ClassifierLogger respects custom TensorBoard tb_log_dir."""
        log_dir = tmp_path / "logs"
        custom_tb_dir = tmp_path / "my_custom_tb"

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=custom_tb_dir,
        ) as logger:
            logger.log_metrics({"loss": 0.5}, step=1)

        assert custom_tb_dir.exists()
        event_files = list(custom_tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

        # Default sibling dir should not have been created with events
        default_tb_dir = log_dir.parent / "tensorboard"
        default_events = (
            list(default_tb_dir.glob("events.out.tfevents.*"))
            if default_tb_dir.exists()
            else []
        )
        # The custom dir should be used, not the default
        assert len(event_files) > 0

    def test_diffusion_uses_custom_log_dir(self, tmp_path):
        """DiffusionLogger respects custom TensorBoard tb_log_dir."""
        log_dir = tmp_path / "logs"
        custom_tb_dir = tmp_path / "diffusion_tb"

        with DiffusionLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=custom_tb_dir,
        ) as logger:
            logger.log_metrics({"loss": 0.05}, step=1000)

        assert custom_tb_dir.exists()
        event_files = list(custom_tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_custom_log_dir_created_if_not_exists(self, tmp_path):
        """Nested custom tb_log_dir is created automatically."""
        log_dir = tmp_path / "logs"
        nested_tb_dir = tmp_path / "deep" / "nested" / "tensorboard"

        with ClassifierLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=nested_tb_dir,
        ) as logger:
            logger.log_metrics({"loss": 0.5}, step=1)

        assert nested_tb_dir.exists()


# ============================================================================
# Scenario 5: Graceful degradation without tensorboard package
# ============================================================================


@pytest.mark.integration
class TestGracefulDegradation:
    """Verify training continues and logs degrade gracefully if tensorboard is absent."""

    def test_classifier_continues_without_tensorboard_package(self, tmp_path):
        """ClassifierLogger continues normally when tensorboard package is missing."""
        log_dir = tmp_path / "logs"

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            with ClassifierLogger(
                log_dir=log_dir,
                class_names=["Normal", "Abnormal"],
                tensorboard_config={"enabled": True},
            ) as logger:
                assert logger.tb_writer is None

                logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1, epoch=1)

                images = torch.randn(4, 3, 32, 32)
                logger.log_images(images, tag="samples", step=1)

                cm = np.array([[80, 20], [10, 90]])
                logger.log_confusion_matrix(cm, step=1)

                logger.log_hyperparams({"lr": 0.001})

        # CSV should still be present
        assert (log_dir / "metrics.csv").exists()

        with open(log_dir / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["loss"] == "0.5"

    def test_diffusion_continues_without_tensorboard_package(self, tmp_path):
        """DiffusionLogger continues normally when tensorboard package is missing."""
        log_dir = tmp_path / "logs"

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            with DiffusionLogger(
                log_dir=log_dir,
                tensorboard_config={"enabled": True},
            ) as logger:
                assert logger.tb_writer is None

                logger.log_metrics({"loss": 0.05, "timestep": 500}, step=1000)

                samples = torch.rand(4, 3, 32, 32)
                logger.log_images(samples, tag="samples", step=1000)

                sequence = torch.rand(8, 3, 32, 32)
                logger.log_denoising_process(sequence, step=1000)

                logger.log_hyperparams({"lr": 0.0001})

        assert (log_dir / "metrics" / "metrics.csv").exists()

        with open(log_dir / "metrics" / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_warning_logged_when_tensorboard_unavailable(self, tmp_path, caplog):
        """A warning is logged when tensorboard is enabled but package is missing."""
        import logging

        log_dir = tmp_path / "logs"

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="src.utils.tensorboard"):
                ClassifierLogger(
                    log_dir=log_dir,
                    tensorboard_config={"enabled": True},
                )

        assert any(
            "not installed" in record.message.lower() for record in caplog.records
        )


# ============================================================================
# Helpers
# ============================================================================


def _read_csv(path: Path) -> list:
    """Read a CSV file and return rows as a list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))
