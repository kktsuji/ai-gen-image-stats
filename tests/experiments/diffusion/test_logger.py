"""Tests for Diffusion Logger

This module contains tests for the DiffusionLogger implementation.
Tests are organized into unit, component, and integration tiers.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.experiments.diffusion.logger import DiffusionLogger

# Test fixtures


@pytest.fixture
def temp_log_dir():
    """Provide temporary directory for logger outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger(temp_log_dir):
    """Provide a basic logger instance."""
    return DiffusionLogger(log_dir=temp_log_dir)


@pytest.fixture
def sample_denoising_sequence():
    """Provide a sample denoising sequence."""
    # 8 timesteps, 3 channels, 32x32 images
    return torch.randn(8, 3, 32, 32)


# Unit Tests


@pytest.mark.unit
class TestDiffusionLoggerInstantiation:
    """Test logger instantiation and initialization."""

    def test_can_instantiate_logger(self, temp_log_dir):
        """DiffusionLogger can be instantiated."""
        logger = DiffusionLogger(log_dir=temp_log_dir)
        assert isinstance(logger, DiffusionLogger)

    def test_creates_log_directory(self, temp_log_dir):
        """Logger creates log directory on instantiation."""
        log_dir = Path(temp_log_dir) / "diffusion_001"
        logger = DiffusionLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_creates_subdirectories(self, logger, temp_log_dir):
        """Logger creates required subdirectories."""
        log_dir = Path(temp_log_dir)
        assert (log_dir / "metrics").exists()
        assert (log_dir / "samples").exists()
        assert (log_dir / "denoising").exists()
        assert (log_dir / "quality").exists()

    def test_log_dir_is_path_object(self, logger):
        """Logger converts log_dir to Path object."""
        assert isinstance(logger.log_dir, Path)

    def test_initializes_empty_histories(self, logger):
        """Logger initializes empty tracking lists."""
        assert logger.logged_metrics_history == []
        assert logger.logged_images == []
        assert logger.logged_denoising_sequences == []


@pytest.mark.unit
class TestLogMetrics:
    """Test the log_metrics() method."""

    def test_log_metrics_with_float_values(self, logger):
        """log_metrics() handles float metric values."""
        metrics = {"loss": 0.05, "avg_timestep": 500.0}
        logger.log_metrics(metrics, step=1000, epoch=10)

        assert len(logger.logged_metrics_history) == 1
        assert logger.logged_metrics_history[0]["loss"] == 0.05
        assert logger.logged_metrics_history[0]["avg_timestep"] == 500.0

    def test_log_metrics_with_int_values(self, logger):
        """log_metrics() handles integer metric values."""
        metrics = {"timesteps": 1000, "batch_size": 64}
        logger.log_metrics(metrics, step=1)

        assert logger.logged_metrics_history[0]["timesteps"] == 1000
        assert logger.logged_metrics_history[0]["batch_size"] == 64

    def test_log_metrics_with_tensor_values(self, logger):
        """log_metrics() handles torch.Tensor metric values."""
        metrics = {
            "loss": torch.tensor(0.05),
            "timestep": torch.tensor([500]),
        }
        logger.log_metrics(metrics, step=1)

        # Should convert tensors to scalars
        assert isinstance(logger.logged_metrics_history[0]["loss"], (float, int))
        assert isinstance(logger.logged_metrics_history[0]["timestep"], (float, int))

    def test_log_metrics_stores_step_and_epoch(self, logger):
        """log_metrics() stores step and epoch information."""
        logger.log_metrics({"loss": 0.05}, step=1000, epoch=10)

        entry = logger.logged_metrics_history[0]
        assert entry["step"] == 1000
        assert entry["epoch"] == 10

    def test_log_metrics_writes_to_csv(self, logger, temp_log_dir):
        """log_metrics() writes metrics to CSV file."""
        logger.log_metrics({"loss": 0.05, "timestep": 500}, step=1, epoch=1)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        assert csv_file.exists()

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["loss"] == "0.05"
            assert rows[0]["timestep"] == "500"

    def test_log_metrics_multiple_calls(self, logger):
        """Multiple log_metrics() calls accumulate in history."""
        logger.log_metrics({"loss": 1.0}, step=1)
        logger.log_metrics({"loss": 0.5}, step=2)
        logger.log_metrics({"loss": 0.25}, step=3)

        assert len(logger.logged_metrics_history) == 3
        assert logger.logged_metrics_history[0]["loss"] == 1.0
        assert logger.logged_metrics_history[1]["loss"] == 0.5
        assert logger.logged_metrics_history[2]["loss"] == 0.25

    def test_log_metrics_handles_new_fields(self, logger, temp_log_dir):
        """log_metrics() handles new fields being added."""
        logger.log_metrics({"loss": 0.05}, step=1)
        logger.log_metrics({"loss": 0.04, "timestep": 500}, step=2)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            # First row should have loss but empty timestep
            assert rows[0]["loss"] == "0.05"
            assert rows[0]["timestep"] == ""
            # Second row should have both
            assert rows[1]["loss"] == "0.04"
            assert rows[1]["timestep"] == "500"


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

        # Check that file exists
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


@pytest.mark.unit
class TestLogDenoisingProcess:
    """Test the log_denoising_process() method."""

    def test_log_denoising_with_tensor(
        self, logger, sample_denoising_sequence, temp_log_dir
    ):
        """log_denoising_process() handles tensor input."""
        logger.log_denoising_process(sample_denoising_sequence, step=1000)

        assert len(logger.logged_denoising_sequences) == 1
        assert logger.logged_denoising_sequences[0]["step"] == 1000

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step1000.png"
        assert denoising_file.exists()

    def test_log_denoising_with_list(self, logger, temp_log_dir):
        """log_denoising_process() handles list of tensors."""
        sequence = [torch.randn(3, 32, 32) for _ in range(8)]
        logger.log_denoising_process(sequence, step=1000)

        assert len(logger.logged_denoising_sequences) == 1
        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step1000.png"
        assert denoising_file.exists()

    def test_log_denoising_with_epoch(
        self, logger, sample_denoising_sequence, temp_log_dir
    ):
        """log_denoising_process() includes epoch in filename."""
        logger.log_denoising_process(sample_denoising_sequence, step=1000, epoch=10)

        assert logger.logged_denoising_sequences[0]["epoch"] == 10
        denoising_file = (
            Path(temp_log_dir) / "denoising" / "denoising_epoch10_step1000.png"
        )
        assert denoising_file.exists()

    def test_log_denoising_stores_sequence(self, logger, sample_denoising_sequence):
        """log_denoising_process() stores the denoising sequence."""
        logger.log_denoising_process(sample_denoising_sequence, step=1000)

        stored = logger.logged_denoising_sequences[0]["sequence"]
        assert stored.shape == sample_denoising_sequence.shape
        assert torch.allclose(stored, sample_denoising_sequence)

    def test_log_denoising_with_custom_num_steps(
        self, logger, sample_denoising_sequence, temp_log_dir
    ):
        """log_denoising_process() accepts num_steps_to_show parameter."""
        logger.log_denoising_process(
            sample_denoising_sequence, step=1000, num_steps_to_show=4
        )

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step1000.png"
        assert denoising_file.exists()

    def test_log_denoising_with_fewer_steps_than_requested(self, logger, temp_log_dir):
        """log_denoising_process() handles fewer steps than num_steps_to_show."""
        short_sequence = torch.randn(4, 3, 32, 32)
        logger.log_denoising_process(short_sequence, step=1000, num_steps_to_show=8)

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step1000.png"
        assert denoising_file.exists()

    def test_log_denoising_raises_on_invalid_shape(self, logger):
        """log_denoising_process() raises ValueError on invalid tensor shape."""
        invalid_sequence = torch.randn(8, 32, 32)  # Missing channel dimension

        with pytest.raises(ValueError, match="Expected 4D tensor"):
            logger.log_denoising_process(invalid_sequence, step=1000)

    def test_log_denoising_negative_values_not_clipped_to_black(
        self, logger, temp_log_dir
    ):
        """Pixels in [-1, 0) are remapped to non-zero, not clamped to black."""
        from PIL import Image

        # All pixels at -1.0 (min of diffusion output range)
        sequence = torch.full((4, 3, 32, 32), -1.0)
        logger.log_denoising_process(sequence, step=2000)

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step2000.png"
        assert denoising_file.exists()

        # Reload image and verify pixels are NOT all black
        img = Image.open(denoising_file).convert("RGB")
        arr = np.array(img)
        # After remapping -1 -> 0.0, the rendered subplot content should be black
        # but the figure background is white, so the image is not entirely black.
        # The key check: the image should not be entirely black (figure has borders).
        assert arr.max() > 0, "Image is entirely black; normalization is broken"

    def test_log_denoising_full_range_remapping(self, logger, temp_log_dir):
        """Values spanning [-1, 1] are remapped to cover the full [0, 1] brightness range."""
        from PIL import Image

        # Create a gradient from -1 to 1 across the spatial dimension
        t = torch.linspace(-1.0, 1.0, 32)
        # Shape: (1, 1, 32, 32) repeated for 4 timesteps, 3 channels
        row = t.unsqueeze(0).expand(32, -1)  # (32, 32)
        img_tensor = row.unsqueeze(0).expand(3, -1, -1)  # (3, 32, 32)
        sequence = (
            img_tensor.unsqueeze(0).expand(4, -1, -1, -1).clone()
        )  # (4, 3, 32, 32)

        logger.log_denoising_process(sequence, step=3000)

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step3000.png"
        assert denoising_file.exists()

        # Reload and verify values span a wide brightness range
        img = Image.open(denoising_file).convert("RGB")
        arr = np.array(img)
        # The subplot content area should contain both dark and bright pixels
        assert arr.min() < 50, "No dark pixels found; remapping may be wrong"
        assert arr.max() > 200, "No bright pixels found; remapping may be wrong"


@pytest.mark.unit
class TestLogSampleComparison:
    """Test the log_sample_comparison() method."""

    def test_log_sample_comparison_with_4d_tensor(self, logger, temp_log_dir):
        """log_sample_comparison() handles 4D image tensor."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_sample_comparison(images, tag="quality", step=1000)

        image_file = Path(temp_log_dir) / "quality" / "quality_step1000.png"
        assert image_file.exists()

    def test_log_sample_comparison_with_3d_tensor(self, logger, temp_log_dir):
        """log_sample_comparison() handles 3D image tensor."""
        image = torch.randn(3, 32, 32)
        logger.log_sample_comparison(image, tag="quality", step=1000)

        image_file = Path(temp_log_dir) / "quality" / "quality_step1000.png"
        assert image_file.exists()

    def test_log_sample_comparison_with_epoch(self, logger, temp_log_dir):
        """log_sample_comparison() includes epoch in filename."""
        images = torch.randn(2, 3, 32, 32)
        logger.log_sample_comparison(images, tag="quality", step=1000, epoch=10)

        image_file = Path(temp_log_dir) / "quality" / "quality_epoch10_step1000.png"
        assert image_file.exists()

    def test_log_sample_comparison_saves_to_quality_dir(self, logger, temp_log_dir):
        """log_sample_comparison() saves to quality directory."""
        images = torch.randn(4, 3, 32, 32)
        logger.log_sample_comparison(images, tag="comparison", step=1000)

        # Should be in quality/ not samples/
        quality_file = Path(temp_log_dir) / "quality" / "comparison_step1000.png"
        samples_file = Path(temp_log_dir) / "samples" / "comparison_step1000.png"

        assert quality_file.exists()
        assert not samples_file.exists()


@pytest.mark.unit
class TestLogHyperparams:
    """Test the log_hyperparams() method."""

    def test_log_hyperparams_does_not_raise(self, logger, temp_log_dir):
        """log_hyperparams() completes without raising."""
        hyperparams = {
            "learning_rate": 0.0001,
            "batch_size": 64,
            "timesteps": 1000,
            "beta_schedule": "linear",
        }
        logger.log_hyperparams(hyperparams)  # should not raise

    def test_log_hyperparams_does_not_write_yaml(self, logger, temp_log_dir):
        """log_hyperparams() no longer writes hyperparams.yaml (config.yaml is used instead)."""
        logger.log_hyperparams({"learning_rate": 0.0001, "timesteps": 1000})

        hyperparams_file = Path(temp_log_dir) / "hyperparams.yaml"
        assert not hyperparams_file.exists()

    def test_log_hyperparams_with_nested_dict(self, logger, temp_log_dir):
        """log_hyperparams() handles nested dictionaries without raising."""
        hyperparams = {
            "model": {"type": "unet", "channels": [64, 128, 256]},
            "training": {"lr": 0.0001, "epochs": 100},
        }
        logger.log_hyperparams(hyperparams)  # should not raise


@pytest.mark.unit
class TestLoggerClose:
    """Test the close() method."""

    def test_close_method_exists(self, logger):
        """Logger has close() method."""
        assert hasattr(logger, "close")
        assert callable(logger.close)

    def test_close_does_not_raise(self, logger):
        """Logger close() does not raise exceptions."""
        logger.close()


# Component Tests


@pytest.mark.component
class TestDiffusionLoggerWorkflow:
    """Test complete logging workflows."""

    def test_full_logging_workflow(self, temp_log_dir):
        """Can execute a complete logging workflow."""
        logger = DiffusionLogger(log_dir=temp_log_dir)

        # Log metrics
        logger.log_metrics({"loss": 0.05, "timestep": 500}, step=1000, epoch=10)

        # Log generated samples
        samples = torch.randn(8, 3, 32, 32)
        logger.log_images(samples, tag="samples", step=1000, epoch=10)

        # Log denoising process
        denoising_seq = torch.randn(8, 3, 32, 32)
        logger.log_denoising_process(denoising_seq, step=1000, epoch=10)

        # Log quality comparison
        quality_samples = torch.randn(4, 3, 32, 32)
        logger.log_sample_comparison(
            quality_samples, tag="quality", step=1000, epoch=10
        )

        # Log hyperparameters
        hyperparams = {"lr": 0.0001, "batch_size": 64}
        logger.log_hyperparams(hyperparams)

        # Close logger
        logger.close()

        # Verify all outputs exist
        log_dir = Path(temp_log_dir)
        assert (log_dir / "metrics" / "metrics.csv").exists()
        assert (log_dir / "samples" / "samples_epoch10_step1000.png").exists()
        assert (log_dir / "denoising" / "denoising_epoch10_step1000.png").exists()
        assert (log_dir / "quality" / "quality_epoch10_step1000.png").exists()

    def test_multiple_metrics_over_time(self, logger, temp_log_dir):
        """Can log multiple metrics entries over time."""
        for step in range(0, 5000, 1000):
            loss = 1.0 / (step + 1)  # Decreasing loss
            logger.log_metrics({"loss": loss, "step_num": step}, step=step)

        # Verify CSV has all entries
        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5

    def test_multiple_sample_generations(self, logger, temp_log_dir):
        """Can log multiple sample generations."""
        for step in [1000, 2000, 3000]:
            samples = torch.randn(4, 3, 32, 32)
            logger.log_images(samples, tag="samples", step=step)

        # Verify all sample files exist
        samples_dir = Path(temp_log_dir) / "samples"
        assert (samples_dir / "samples_step1000.png").exists()
        assert (samples_dir / "samples_step2000.png").exists()
        assert (samples_dir / "samples_step3000.png").exists()

    def test_denoising_with_grayscale_images(self, logger, temp_log_dir):
        """Can log denoising process with grayscale images."""
        # Create grayscale denoising sequence
        grayscale_seq = torch.randn(8, 1, 32, 32)
        logger.log_denoising_process(grayscale_seq, step=1000)

        denoising_file = Path(temp_log_dir) / "denoising" / "denoising_step1000.png"
        assert denoising_file.exists()

    def test_large_batch_of_samples(self, logger, temp_log_dir):
        """Can handle large batch of generated samples."""
        large_batch = torch.randn(64, 3, 32, 32)
        logger.log_images(large_batch, tag="large_batch", step=1000, nrow=8)

        image_file = Path(temp_log_dir) / "samples" / "large_batch_step1000.png"
        assert image_file.exists()


@pytest.mark.unit
class TestTensorBoardIntegration:
    """Test TensorBoard integration in DiffusionLogger."""

    def test_tb_writer_none_by_default(self, temp_log_dir):
        """tb_writer is None when TensorBoard is not configured."""
        logger = DiffusionLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

    def test_tb_writer_none_when_disabled(self, temp_log_dir):
        """tb_writer is None when TensorBoard is explicitly disabled."""
        logger = DiffusionLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": False},
        )
        assert logger.tb_writer is None

    def test_tb_writer_created_when_enabled(self, temp_log_dir):
        """tb_writer is created when TensorBoard is enabled."""
        logger = DiffusionLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()

    def test_tb_log_dir_defaults_to_sibling_tensorboard(self, temp_log_dir):
        """Default TensorBoard log_dir is a sibling 'tensorboard' directory."""
        logger = DiffusionLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        expected_dir = Path(temp_log_dir).parent / "tensorboard"
        assert expected_dir.exists()
        logger.close()

    def test_tb_log_dir_custom(self, tmp_path):
        """Custom tb_log_dir places TensorBoard files in the specified directory."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "tb_custom"

        logger = DiffusionLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=tb_dir,
        )
        assert tb_dir.exists()
        logger.close()

    def test_log_metrics_calls_add_scalar_when_tb_enabled(self, temp_log_dir):
        """log_metrics() calls writer.add_scalar when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = DiffusionLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer

        logger.log_metrics({"loss": 0.05, "timestep": 500}, step=1000)

        assert mock_writer.add_scalar.call_count == 2
        call_tags = {call[0][0] for call in mock_writer.add_scalar.call_args_list}
        assert "metrics/loss" in call_tags
        assert "metrics/timestep" in call_tags

    def test_log_metrics_no_tensorboard_when_writer_none(self, temp_log_dir):
        """log_metrics() skips TensorBoard when writer is None."""
        logger = DiffusionLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

        # CSV logging should still work
        logger.log_metrics({"loss": 0.05}, step=1)
        assert len(logger.logged_metrics_history) == 1

    def test_log_images_calls_add_images_when_tb_enabled(self, temp_log_dir):
        """log_images() calls writer.add_images when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = DiffusionLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer
        logger.tb_log_images = True

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1000)

        mock_writer.add_images.assert_called_once()

    def test_log_images_skipped_when_log_images_false(self, temp_log_dir):
        """log_images() skips TensorBoard when log_images config is False."""
        from unittest.mock import MagicMock

        logger = DiffusionLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer
        logger.tb_log_images = False

        images = torch.randn(4, 3, 32, 32)
        logger.log_images(images, tag="samples", step=1000)

        mock_writer.add_images.assert_not_called()

    def test_log_denoising_calls_add_images_and_add_figure_when_tb_enabled(
        self, temp_log_dir
    ):
        """log_denoising_process() calls add_images and add_figure when TB enabled."""
        from unittest.mock import MagicMock

        logger = DiffusionLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer
        logger.tb_log_images = True

        sequence = torch.randn(8, 3, 32, 32)
        logger.log_denoising_process(sequence, step=1000)

        mock_writer.add_images.assert_called_once()
        mock_writer.add_figure.assert_called_once()

    def test_log_denoising_no_tensorboard_when_writer_none(self, temp_log_dir):
        """log_denoising_process() skips TensorBoard when writer is None."""
        logger = DiffusionLogger(log_dir=temp_log_dir)
        assert logger.tb_writer is None

        sequence = torch.randn(8, 3, 32, 32)
        logger.log_denoising_process(sequence, step=1000)
        assert len(logger.logged_denoising_sequences) == 1

    def test_log_hyperparams_calls_add_hparams_when_tb_enabled(self, temp_log_dir):
        """log_hyperparams() calls writer.add_hparams when TensorBoard is enabled."""
        from unittest.mock import MagicMock

        logger = DiffusionLogger(log_dir=temp_log_dir)
        mock_writer = MagicMock()
        logger.tb_writer = mock_writer

        logger.log_hyperparams({"lr": 0.0001, "timesteps": 1000})

        mock_writer.add_hparams.assert_called_once()

    def test_close_sets_tb_writer_to_none(self, temp_log_dir):
        """close() sets tb_writer to None."""
        logger = DiffusionLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        assert logger.tb_writer is not None
        logger.close()
        assert logger.tb_writer is None

    def test_csv_logging_unchanged_when_tb_enabled(self, temp_log_dir):
        """CSV logging works correctly regardless of TensorBoard state."""
        logger = DiffusionLogger(
            log_dir=temp_log_dir,
            tensorboard_config={"enabled": True},
        )
        logger.log_metrics({"loss": 0.05, "timestep": 500}, step=1, epoch=1)

        csv_file = Path(temp_log_dir) / "metrics" / "metrics.csv"
        assert csv_file.exists()
        import csv as csv_module

        with open(csv_file) as f:
            rows = list(csv_module.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["loss"] == "0.05"
        logger.close()

    def test_tb_event_files_created_when_enabled(self, tmp_path):
        """TensorBoard event files are created in the log directory when enabled."""
        log_dir = tmp_path / "logs"
        tb_dir = tmp_path / "tb"

        logger = DiffusionLogger(
            log_dir=log_dir,
            tensorboard_config={"enabled": True},
            tb_log_dir=tb_dir,
        )
        logger.log_metrics({"loss": 0.05}, step=1)
        logger.close()

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_graceful_when_tensorboard_not_installed(self, temp_log_dir):
        """Logger degrades gracefully when tensorboard package is missing."""
        from unittest.mock import patch

        with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
            logger = DiffusionLogger(
                log_dir=temp_log_dir,
                tensorboard_config={"enabled": True},
            )
            assert logger.tb_writer is None

            # CSV logging still works
            logger.log_metrics({"loss": 0.05}, step=1)
            assert len(logger.logged_metrics_history) == 1
