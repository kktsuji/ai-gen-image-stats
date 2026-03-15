"""Tests for Classifier Visualization Helpers

This module contains tests for the classifier-specific visualization functions:
- save_annotated_predictions
- save_confusion_matrix
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.experiments.classifier.visualization import (
    save_annotated_predictions,
    save_confusion_matrix,
)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_confusion_matrix():
    """Provide a sample confusion matrix."""
    return np.array([[50, 10], [5, 35]])


@pytest.mark.unit
class TestSaveAnnotatedPredictions:
    """Test the save_annotated_predictions() function."""

    def test_saves_annotated_image(self, temp_dir):
        """save_annotated_predictions() saves an image file."""
        images = torch.randn(4, 3, 32, 32)
        labels = [0, 1, 0, 1]
        predictions = [0, 1, 1, 1]
        save_path = temp_dir / "annotated.png"

        save_annotated_predictions(images, labels, predictions, save_path)
        assert save_path.exists()

    def test_with_class_names(self, temp_dir):
        """save_annotated_predictions() uses class names for labels."""
        images = torch.randn(4, 3, 32, 32)
        labels = [0, 1, 0, 1]
        predictions = [0, 1, 1, 1]
        save_path = temp_dir / "annotated_named.png"

        save_annotated_predictions(
            images, labels, predictions, save_path, class_names=["Real", "Fake"]
        )
        assert save_path.exists()

    def test_with_labels_only(self, temp_dir):
        """save_annotated_predictions() works with labels only (no predictions)."""
        images = torch.randn(4, 3, 32, 32)
        labels = [0, 1, 0, 1]
        save_path = temp_dir / "labels_only.png"

        save_annotated_predictions(images, labels, None, save_path)
        assert save_path.exists()

    def test_with_predictions_only(self, temp_dir):
        """save_annotated_predictions() works with predictions only (no labels)."""
        images = torch.randn(4, 3, 32, 32)
        predictions = [0, 1, 1, 1]
        save_path = temp_dir / "preds_only.png"

        save_annotated_predictions(images, None, predictions, save_path)
        assert save_path.exists()

    def test_limits_to_16_images(self, temp_dir):
        """save_annotated_predictions() limits display to 16 images."""
        images = torch.randn(32, 3, 32, 32)
        labels = list(range(32))
        predictions = list(range(32))
        save_path = temp_dir / "many_images.png"

        save_annotated_predictions(images, labels, predictions, save_path)
        assert save_path.exists()

    def test_grayscale_images(self, temp_dir):
        """save_annotated_predictions() handles single-channel images."""
        images = torch.randn(4, 1, 28, 28)
        labels = [0, 1, 0, 1]
        predictions = [0, 0, 1, 1]
        save_path = temp_dir / "grayscale.png"

        save_annotated_predictions(images, labels, predictions, save_path)
        assert save_path.exists()

    def test_raises_on_unexpected_channels(self, temp_dir):
        """save_annotated_predictions() raises ValueError for unsupported channel count."""
        images = torch.randn(4, 5, 32, 32)  # 5 channels — not RGB or grayscale
        save_path = temp_dir / "bad_channels.png"

        with pytest.raises(ValueError, match="Unexpected number of channels"):
            save_annotated_predictions(images, [0, 1, 0, 1], [0, 1, 1, 1], save_path)

    def test_raises_on_empty_batch(self, temp_dir):
        """save_annotated_predictions() raises ValueError when images is empty."""
        images = torch.empty(0, 3, 32, 32)
        save_path = temp_dir / "empty.png"

        with pytest.raises(ValueError, match="images must contain at least one sample"):
            save_annotated_predictions(images, [], [], save_path)


@pytest.mark.unit
class TestSaveConfusionMatrix:
    """Test the save_confusion_matrix() function."""

    def test_saves_confusion_matrix(self, temp_dir, sample_confusion_matrix):
        """save_confusion_matrix() saves a visualization file."""
        save_path = temp_dir / "confusion_matrix.png"
        save_confusion_matrix(sample_confusion_matrix, save_path)
        assert save_path.exists()

    def test_with_torch_tensor(self, temp_dir):
        """save_confusion_matrix() handles torch tensor input."""
        cm = torch.tensor([[50, 10], [5, 35]])
        save_path = temp_dir / "confusion_matrix_tensor.png"
        save_confusion_matrix(cm, save_path)
        assert save_path.exists()

    def test_with_class_names(self, temp_dir, sample_confusion_matrix):
        """save_confusion_matrix() uses provided class names."""
        save_path = temp_dir / "confusion_matrix_named.png"
        save_confusion_matrix(
            sample_confusion_matrix, save_path, class_names=["Real", "Fake"]
        )
        assert save_path.exists()

    def test_normalized(self, temp_dir, sample_confusion_matrix):
        """save_confusion_matrix() supports normalization."""
        save_path = temp_dir / "confusion_matrix_normalized.png"
        save_confusion_matrix(sample_confusion_matrix, save_path, normalize=True)
        assert save_path.exists()

    def test_with_zeros(self, temp_dir):
        """save_confusion_matrix() handles confusion matrix with zeros."""
        cm = np.array([[0, 10], [0, 0]])
        save_path = temp_dir / "confusion_matrix_zeros.png"
        save_confusion_matrix(cm, save_path, normalize=True)
        assert save_path.exists()

    def test_single_class(self, temp_dir):
        """save_confusion_matrix() handles single-class confusion matrix."""
        cm = np.array([[100]])
        save_path = temp_dir / "confusion_matrix_single.png"
        save_confusion_matrix(cm, save_path)
        assert save_path.exists()

    def test_multiclass(self, temp_dir):
        """save_confusion_matrix() handles multi-class confusion matrix."""
        cm = np.array([[30, 5, 3], [2, 40, 1], [1, 2, 50]])
        save_path = temp_dir / "confusion_matrix_multi.png"
        save_confusion_matrix(cm, save_path)
        assert save_path.exists()

    def test_logs_to_tensorboard(self, temp_dir, sample_confusion_matrix):
        """save_confusion_matrix() logs to TensorBoard when writer provided."""
        mock_writer = MagicMock()
        save_path = temp_dir / "confusion_matrix_tb.png"
        save_confusion_matrix(
            sample_confusion_matrix,
            save_path,
            step=100,
            tb_writer=mock_writer,
            tb_log_images=True,
        )
        mock_writer.add_figure.assert_called_once()

    def test_skips_tensorboard_when_disabled(self, temp_dir, sample_confusion_matrix):
        """save_confusion_matrix() skips TensorBoard when tb_log_images is False."""
        mock_writer = MagicMock()
        save_path = temp_dir / "confusion_matrix_no_tb.png"
        save_confusion_matrix(
            sample_confusion_matrix,
            save_path,
            step=100,
            tb_writer=mock_writer,
            tb_log_images=False,
        )
        mock_writer.add_figure.assert_not_called()
