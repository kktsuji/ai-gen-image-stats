"""Tests for Diffusion Visualization Helpers

This module contains tests for the diffusion-specific visualization function:
- save_denoising_process
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.experiments.diffusion.visualization import save_denoising_process


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_denoising_sequence():
    """Provide a sample denoising sequence."""
    return torch.randn(8, 3, 32, 32)


@pytest.mark.unit
class TestSaveDenoisingProcess:
    """Test the save_denoising_process() function."""

    def test_saves_denoising_image(self, temp_dir, sample_denoising_sequence):
        """save_denoising_process() saves a visualization file."""
        save_path = temp_dir / "denoising.png"
        save_denoising_process(sample_denoising_sequence, save_path)
        assert save_path.exists()

    def test_with_list_input(self, temp_dir):
        """save_denoising_process() handles list of tensors."""
        sequence = [torch.randn(3, 32, 32) for _ in range(8)]
        save_path = temp_dir / "denoising_list.png"
        save_denoising_process(sequence, save_path)
        assert save_path.exists()

    def test_with_custom_num_steps(self, temp_dir, sample_denoising_sequence):
        """save_denoising_process() accepts num_steps_to_show parameter."""
        save_path = temp_dir / "denoising_4steps.png"
        save_denoising_process(
            sample_denoising_sequence, save_path, num_steps_to_show=4
        )
        assert save_path.exists()

    def test_with_fewer_steps_than_requested(self, temp_dir):
        """save_denoising_process() handles fewer steps than num_steps_to_show."""
        short_sequence = torch.randn(4, 3, 32, 32)
        save_path = temp_dir / "denoising_short.png"
        save_denoising_process(short_sequence, save_path, num_steps_to_show=8)
        assert save_path.exists()

    def test_raises_on_invalid_shape(self):
        """save_denoising_process() raises ValueError on invalid tensor shape."""
        invalid_sequence = torch.randn(8, 32, 32)  # Missing channel dimension
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            save_denoising_process(invalid_sequence, "dummy.png")

    def test_grayscale_images(self, temp_dir):
        """save_denoising_process() handles grayscale images."""
        grayscale_seq = torch.randn(8, 1, 32, 32)
        save_path = temp_dir / "denoising_gray.png"
        save_denoising_process(grayscale_seq, save_path)
        assert save_path.exists()

    def test_negative_values_not_clipped_to_black(self, temp_dir):
        """Pixels in [-1, 0) are remapped to non-zero, not clamped to black."""
        from PIL import Image

        sequence = torch.full((4, 3, 32, 32), -1.0)
        save_path = temp_dir / "denoising_neg.png"
        save_denoising_process(sequence, save_path)
        assert save_path.exists()

        img = Image.open(save_path).convert("RGB")
        arr = np.array(img)
        assert arr.max() > 0, "Image is entirely black; normalization is broken"

    def test_full_range_remapping(self, temp_dir):
        """Values spanning [-1, 1] are remapped to cover the full [0, 1] brightness range."""
        from PIL import Image

        t = torch.linspace(-1.0, 1.0, 32)
        row = t.unsqueeze(0).expand(32, -1)
        img_tensor = row.unsqueeze(0).expand(3, -1, -1)
        sequence = img_tensor.unsqueeze(0).expand(4, -1, -1, -1).clone()

        save_path = temp_dir / "denoising_range.png"
        save_denoising_process(sequence, save_path)
        assert save_path.exists()

        img = Image.open(save_path).convert("RGB")
        arr = np.array(img)
        assert arr.min() < 50, "No dark pixels found; remapping may be wrong"
        assert arr.max() > 200, "No bright pixels found; remapping may be wrong"

    def test_logs_to_tensorboard(self, temp_dir, sample_denoising_sequence):
        """save_denoising_process() logs to TensorBoard when writer provided."""
        mock_writer = MagicMock()
        save_path = temp_dir / "denoising_tb.png"
        save_denoising_process(
            sample_denoising_sequence,
            save_path,
            step=1000,
            tb_writer=mock_writer,
            tb_log_images=True,
        )
        mock_writer.add_images.assert_called_once()
        mock_writer.add_figure.assert_called_once()

    def test_no_tensorboard_when_writer_none(self, temp_dir, sample_denoising_sequence):
        """save_denoising_process() works without TensorBoard writer."""
        save_path = temp_dir / "denoising_no_tb.png"
        save_denoising_process(
            sample_denoising_sequence, save_path, step=1000, tb_writer=None
        )
        assert save_path.exists()

    def test_skips_tensorboard_when_disabled(self, temp_dir, sample_denoising_sequence):
        """save_denoising_process() skips TB when tb_log_images is False."""
        mock_writer = MagicMock()
        save_path = temp_dir / "denoising_tb_disabled.png"
        save_denoising_process(
            sample_denoising_sequence,
            save_path,
            step=1000,
            tb_writer=mock_writer,
            tb_log_images=False,
        )
        mock_writer.add_images.assert_not_called()
        mock_writer.add_figure.assert_not_called()
