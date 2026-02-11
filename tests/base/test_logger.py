"""Tests for Base Logger Interface

This module contains tests for the BaseLogger abstract class and its interface.
Tests are organized into unit tests to ensure fast execution on CPU.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import torch

from src.base.logger import BaseLogger

# Test fixtures and helper classes


class MinimalValidLogger(BaseLogger):
    """Minimal valid logger implementation for testing.

    This is the simplest possible implementation that satisfies
    the BaseLogger interface requirements.
    """

    def __init__(self):
        self.logged_metrics = []
        self.logged_images = []

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Store metrics in memory for testing."""
        self.logged_metrics.append({"metrics": metrics, "step": step, "epoch": epoch})

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Store image info in memory for testing."""
        self.logged_images.append(
            {
                "images": images,
                "tag": tag,
                "step": step,
                "epoch": epoch,
                "kwargs": kwargs,
            }
        )


class FileLogger(BaseLogger):
    """Logger that writes to files for testing."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.txt"
        self.closed = False

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Write metrics to file."""
        with open(self.metrics_file, "a") as f:
            f.write(f"Step {step}, Epoch {epoch}: {metrics}\n")

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Save image info to file."""
        image_file = self.log_dir / f"{tag}_{step}.txt"
        with open(image_file, "w") as f:
            if isinstance(images, torch.Tensor):
                f.write(f"Shape: {images.shape}\n")
            else:
                f.write(f"Image count: {len(images)}\n")

    def close(self) -> None:
        """Mark logger as closed."""
        self.closed = True


class RichLogger(BaseLogger):
    """Logger that overrides optional methods."""

    def __init__(self):
        self.logged_hyperparams = None
        self.logged_text = []
        self.logged_histograms = []

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Mock implementation."""
        pass

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Mock implementation."""
        pass

    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Override to store hyperparams."""
        self.logged_hyperparams = hyperparams

    def log_text(
        self,
        text: str,
        tag: str,
        step: Optional[int] = None,
    ) -> None:
        """Override to store text."""
        self.logged_text.append({"text": text, "tag": tag, "step": step})

    def log_histogram(
        self,
        values: Union[torch.Tensor, List[float]],
        tag: str,
        step: int,
        bins: int = 100,
    ) -> None:
        """Override to store histogram info."""
        self.logged_histograms.append(
            {"values": values, "tag": tag, "step": step, "bins": bins}
        )


class IncompleteLogger(BaseLogger):
    """Logger that doesn't implement required abstract methods.

    This should fail to instantiate.
    """

    def __init__(self):
        pass

    # Missing log_metrics() and log_images() implementations


# Unit Tests


@pytest.mark.unit
class TestBaseLoggerInterface:
    """Test that BaseLogger enforces its interface requirements."""

    def test_cannot_instantiate_base_logger_directly(self):
        """BaseLogger is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLogger()

    def test_cannot_instantiate_incomplete_implementation(self):
        """Loggers that don't implement all abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteLogger()

    def test_can_instantiate_complete_implementation(self):
        """Loggers that implement all abstract methods can be instantiated."""
        logger = MinimalValidLogger()
        assert isinstance(logger, BaseLogger)

    def test_log_metrics_is_abstract(self):
        """log_metrics() method must be implemented by subclasses."""
        # This is enforced by ABC - test that it's in the class
        assert hasattr(BaseLogger, "log_metrics")
        assert hasattr(BaseLogger.log_metrics, "__isabstractmethod__")

    def test_log_images_is_abstract(self):
        """log_images() method must be implemented by subclasses."""
        assert hasattr(BaseLogger, "log_images")
        assert hasattr(BaseLogger.log_images, "__isabstractmethod__")


@pytest.mark.unit
class TestLogMetrics:
    """Test the log_metrics() method interface."""

    def test_log_metrics_called_successfully(self):
        """log_metrics() can be called with required parameters."""
        logger = MinimalValidLogger()
        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=100)

        assert len(logger.logged_metrics) == 1
        assert logger.logged_metrics[0]["metrics"] == metrics
        assert logger.logged_metrics[0]["step"] == 100

    def test_log_metrics_with_epoch(self):
        """log_metrics() accepts optional epoch parameter."""
        logger = MinimalValidLogger()
        metrics = {"loss": 0.3}
        logger.log_metrics(metrics, step=50, epoch=2)

        assert len(logger.logged_metrics) == 1
        assert logger.logged_metrics[0]["epoch"] == 2

    def test_log_metrics_with_float_values(self):
        """log_metrics() handles float metric values."""
        logger = MinimalValidLogger()
        metrics = {"loss": 0.123456789}
        logger.log_metrics(metrics, step=1)

        assert logger.logged_metrics[0]["metrics"]["loss"] == 0.123456789

    def test_log_metrics_with_int_values(self):
        """log_metrics() handles integer metric values."""
        logger = MinimalValidLogger()
        metrics = {"num_samples": 1000, "batch_size": 32}
        logger.log_metrics(metrics, step=1)

        assert logger.logged_metrics[0]["metrics"]["num_samples"] == 1000
        assert logger.logged_metrics[0]["metrics"]["batch_size"] == 32

    def test_log_metrics_with_tensor_values(self):
        """log_metrics() handles torch.Tensor metric values."""
        logger = MinimalValidLogger()
        metrics = {
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor([0.95]),  # 1D tensor
        }
        logger.log_metrics(metrics, step=1)

        # Logger should receive the tensors as-is
        assert torch.is_tensor(logger.logged_metrics[0]["metrics"]["loss"])
        assert torch.is_tensor(logger.logged_metrics[0]["metrics"]["accuracy"])

    def test_log_metrics_multiple_calls(self):
        """Multiple log_metrics() calls are handled correctly."""
        logger = MinimalValidLogger()

        logger.log_metrics({"loss": 1.0}, step=1)
        logger.log_metrics({"loss": 0.8}, step=2)
        logger.log_metrics({"loss": 0.6}, step=3)

        assert len(logger.logged_metrics) == 3
        assert logger.logged_metrics[0]["metrics"]["loss"] == 1.0
        assert logger.logged_metrics[1]["metrics"]["loss"] == 0.8
        assert logger.logged_metrics[2]["metrics"]["loss"] == 0.6

    def test_log_metrics_empty_dict(self):
        """log_metrics() handles empty metrics dict."""
        logger = MinimalValidLogger()
        logger.log_metrics({}, step=1)

        assert len(logger.logged_metrics) == 1
        assert logger.logged_metrics[0]["metrics"] == {}


@pytest.mark.unit
class TestLogImages:
    """Test the log_images() method interface."""

    def test_log_images_with_single_tensor(self):
        """log_images() handles a single image tensor."""
        logger = MinimalValidLogger()
        images = torch.randn(4, 3, 32, 32)  # Batch of 4 images
        logger.log_images(images, tag="samples", step=100)

        assert len(logger.logged_images) == 1
        assert torch.equal(logger.logged_images[0]["images"], images)
        assert logger.logged_images[0]["tag"] == "samples"
        assert logger.logged_images[0]["step"] == 100

    def test_log_images_with_single_image(self):
        """log_images() handles a single image (C, H, W)."""
        logger = MinimalValidLogger()
        image = torch.randn(3, 64, 64)  # Single RGB image
        logger.log_images(image, tag="sample", step=50)

        assert len(logger.logged_images) == 1
        assert torch.equal(logger.logged_images[0]["images"], image)

    def test_log_images_with_list_of_tensors(self):
        """log_images() handles a list of image tensors."""
        logger = MinimalValidLogger()
        images = [
            torch.randn(3, 32, 32),
            torch.randn(3, 32, 32),
            torch.randn(3, 32, 32),
        ]
        logger.log_images(images, tag="samples", step=100)

        assert len(logger.logged_images) == 1
        assert len(logger.logged_images[0]["images"]) == 3

    def test_log_images_with_epoch(self):
        """log_images() accepts optional epoch parameter."""
        logger = MinimalValidLogger()
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(images, tag="validation", step=100, epoch=5)

        assert logger.logged_images[0]["epoch"] == 5

    def test_log_images_with_kwargs(self):
        """log_images() accepts additional keyword arguments."""
        logger = MinimalValidLogger()
        images = torch.randn(2, 3, 32, 32)
        logger.log_images(
            images,
            tag="samples",
            step=100,
            normalize=True,
            nrow=4,
        )

        assert logger.logged_images[0]["kwargs"]["normalize"] is True
        assert logger.logged_images[0]["kwargs"]["nrow"] == 4

    def test_log_images_multiple_calls(self):
        """Multiple log_images() calls are handled correctly."""
        logger = MinimalValidLogger()

        logger.log_images(torch.randn(2, 3, 32, 32), tag="train", step=1)
        logger.log_images(torch.randn(2, 3, 32, 32), tag="val", step=1)
        logger.log_images(torch.randn(2, 3, 32, 32), tag="test", step=1)

        assert len(logger.logged_images) == 3
        assert logger.logged_images[0]["tag"] == "train"
        assert logger.logged_images[1]["tag"] == "val"
        assert logger.logged_images[2]["tag"] == "test"


@pytest.mark.unit
class TestOptionalMethods:
    """Test optional methods with default implementations."""

    def test_log_hyperparams_has_default_implementation(self):
        """log_hyperparams() has a default no-op implementation."""
        logger = MinimalValidLogger()
        # Should not raise an error
        logger.log_hyperparams({"lr": 0.001, "batch_size": 32})

    def test_log_hyperparams_can_be_overridden(self):
        """log_hyperparams() can be overridden by subclasses."""
        logger = RichLogger()
        hyperparams = {"lr": 0.001, "epochs": 100}
        logger.log_hyperparams(hyperparams)

        assert logger.logged_hyperparams == hyperparams

    def test_log_text_has_default_implementation(self):
        """log_text() has a default no-op implementation."""
        logger = MinimalValidLogger()
        # Should not raise an error
        logger.log_text("Test message", tag="status", step=1)

    def test_log_text_can_be_overridden(self):
        """log_text() can be overridden by subclasses."""
        logger = RichLogger()
        logger.log_text("Model initialized", tag="status", step=0)

        assert len(logger.logged_text) == 1
        assert logger.logged_text[0]["text"] == "Model initialized"
        assert logger.logged_text[0]["tag"] == "status"
        assert logger.logged_text[0]["step"] == 0

    def test_log_histogram_has_default_implementation(self):
        """log_histogram() has a default no-op implementation."""
        logger = MinimalValidLogger()
        values = torch.randn(100)
        # Should not raise an error
        logger.log_histogram(values, tag="weights", step=1, bins=50)

    def test_log_histogram_can_be_overridden(self):
        """log_histogram() can be overridden by subclasses."""
        logger = RichLogger()
        values = torch.randn(100)
        logger.log_histogram(values, tag="gradients", step=10, bins=50)

        assert len(logger.logged_histograms) == 1
        assert logger.logged_histograms[0]["tag"] == "gradients"
        assert logger.logged_histograms[0]["step"] == 10
        assert logger.logged_histograms[0]["bins"] == 50

    def test_close_has_default_implementation(self):
        """close() has a default no-op implementation."""
        logger = MinimalValidLogger()
        # Should not raise an error
        logger.close()

    def test_close_can_be_overridden(self):
        """close() can be overridden by subclasses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(tmpdir)
            assert not logger.closed

            logger.close()
            assert logger.closed


@pytest.mark.unit
class TestContextManager:
    """Test context manager functionality."""

    def test_logger_supports_context_manager(self):
        """Logger can be used as a context manager."""
        with MinimalValidLogger() as logger:
            assert isinstance(logger, BaseLogger)
            logger.log_metrics({"loss": 0.5}, step=1)

    def test_context_manager_calls_close(self):
        """Context manager calls close() on exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(tmpdir)
            assert not logger.closed

            with logger:
                assert not logger.closed

            # After exiting context, close() should have been called
            assert logger.closed

    def test_context_manager_close_on_exception(self):
        """Context manager calls close() even on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(tmpdir)

            try:
                with logger:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # close() should still be called
            assert logger.closed


@pytest.mark.unit
class TestLoggerPersistence:
    """Test logger persistence and file operations."""

    def test_file_logger_writes_metrics(self):
        """FileLogger writes metrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(tmpdir)
            logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1, epoch=1)
            logger.log_metrics({"loss": 0.3, "acc": 0.95}, step=2, epoch=1)

            # Check file exists and contains metrics
            assert logger.metrics_file.exists()
            content = logger.metrics_file.read_text()
            assert "Step 1" in content
            assert "Step 2" in content
            assert "loss" in content
            assert "acc" in content

    def test_file_logger_saves_image_info(self):
        """FileLogger saves image information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = FileLogger(tmpdir)
            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="samples", step=100)

            # Check image file exists
            image_file = Path(tmpdir) / "samples_100.txt"
            assert image_file.exists()
            content = image_file.read_text()
            assert "Shape:" in content or "4" in content

    def test_multiple_loggers_same_directory(self):
        """Multiple loggers can write to the same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = FileLogger(tmpdir)
            logger2 = FileLogger(tmpdir)

            logger1.log_metrics({"loss": 0.5}, step=1)
            logger2.log_metrics({"loss": 0.3}, step=2)

            # Both should write to the same file
            content = logger1.metrics_file.read_text()
            assert "Step 1" in content
            assert "Step 2" in content


@pytest.mark.unit
class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_typical_training_loop_logging(self):
        """Test typical usage pattern in a training loop."""
        logger = MinimalValidLogger()

        # Simulate training loop
        for epoch in range(3):
            for step in range(5):
                # Log metrics each step
                logger.log_metrics(
                    {"loss": 1.0 / (epoch + step + 1), "lr": 0.001},
                    step=epoch * 5 + step,
                    epoch=epoch,
                )

            # Log images each epoch
            images = torch.randn(4, 3, 32, 32)
            logger.log_images(images, tag="epoch_samples", step=epoch * 5, epoch=epoch)

        # Check all metrics and images were logged
        assert len(logger.logged_metrics) == 15  # 3 epochs * 5 steps
        assert len(logger.logged_images) == 3  # 3 epochs

    def test_logger_with_multiple_metric_types(self):
        """Test logging various metric types together."""
        logger = MinimalValidLogger()

        metrics = {
            "loss": 0.5,  # float
            "epoch": 10,  # int
            "accuracy": torch.tensor(0.95),  # tensor
            "learning_rate": 1e-4,  # scientific notation
        }

        logger.log_metrics(metrics, step=100)

        logged = logger.logged_metrics[0]["metrics"]
        assert logged["loss"] == 0.5
        assert logged["epoch"] == 10
        assert torch.is_tensor(logged["accuracy"])
        assert logged["learning_rate"] == 1e-4
