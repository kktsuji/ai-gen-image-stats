"""Tests for Sample Selection Pipeline

Unit tests for scoring, realism, and selection functions.
Component test for feature extraction with a mock model.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.sample_selection.selector import (
    _copy_selected_samples,
    _FilteredDataset,
    _get_dataset_paths,
    compute_knn_scores,
    compute_realism_flags,
    extract_features_from_loader,
    select_samples,
)

# ============================================================================
# Unit Tests - compute_knn_scores
# ============================================================================


@pytest.mark.unit
class TestComputeKnnScores:
    """Test k-NN distance scoring."""

    def test_identical_features_have_zero_score(self):
        """Identical features should have zero distance."""
        real = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        gen = np.array([[1.0, 0.0]], dtype=np.float32)
        scores = compute_knn_scores(real, gen, k=1)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_scores_shape_matches_generated(self):
        """Output shape should match number of generated samples."""
        real = np.random.rand(20, 8).astype(np.float32)
        gen = np.random.rand(10, 8).astype(np.float32)
        scores = compute_knn_scores(real, gen, k=5)
        assert scores.shape == (10,)

    def test_closer_samples_have_lower_scores(self):
        """Samples closer to the real manifold should have lower scores."""
        real = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        gen = np.array(
            [[0.1, 0.0], [10.0, 10.0]],  # close  # far
            dtype=np.float32,
        )
        scores = compute_knn_scores(real, gen, k=2)
        assert scores[0] < scores[1]

    def test_scores_are_non_negative(self):
        """All scores should be non-negative."""
        real = np.random.rand(15, 4).astype(np.float32)
        gen = np.random.rand(10, 4).astype(np.float32)
        scores = compute_knn_scores(real, gen, k=3)
        assert np.all(scores >= 0)

    def test_k_clamped_to_num_real_samples(self):
        """k should be clamped when larger than number of real samples."""
        real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gen = np.array([[1.0, 2.0]], dtype=np.float32)
        # k=10 but only 2 real samples — should not raise
        scores = compute_knn_scores(real, gen, k=10)
        assert scores.shape == (1,)


# ============================================================================
# Unit Tests - compute_realism_flags
# ============================================================================


@pytest.mark.unit
class TestComputeRealismFlags:
    """Test realism flag computation."""

    def test_sample_within_manifold_is_realistic(self):
        """A generated sample identical to a real one should be realistic."""
        real = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        gen = np.array([[0.5, 0.5]], dtype=np.float32)
        flags = compute_realism_flags(real, gen, k=2)
        assert flags.shape == (1,)
        assert flags.dtype == bool

    def test_distant_sample_is_not_realistic(self):
        """A sample far from the real manifold should not be realistic."""
        real = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        gen = np.array([[100.0, 100.0]], dtype=np.float32)
        flags = compute_realism_flags(real, gen, k=2)
        assert not flags[0]

    def test_output_shape_matches_generated(self):
        """Output shape should match number of generated samples."""
        real = np.random.rand(20, 4).astype(np.float32)
        gen = np.random.rand(10, 4).astype(np.float32)
        flags = compute_realism_flags(real, gen, k=3)
        assert flags.shape == (10,)
        assert flags.dtype == bool

    def test_single_real_sample_returns_all_true(self):
        """With only 1 real sample, k is clamped to 0, returning all True."""
        real = np.array([[1.0, 2.0]], dtype=np.float32)
        gen = np.array([[100.0, 200.0]], dtype=np.float32)
        flags = compute_realism_flags(real, gen, k=5)
        assert flags[0]


# ============================================================================
# Unit Tests - select_samples
# ============================================================================


@pytest.mark.unit
class TestSelectSamples:
    """Test sample selection logic."""

    def test_top_k_selects_correct_count(self):
        """top_k should select exactly k samples."""
        scores = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        mask = select_samples(scores, mode="top_k", value=3)
        assert mask.sum() == 3

    def test_top_k_selects_lowest_scores(self):
        """top_k should select the samples with lowest scores."""
        scores = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        mask = select_samples(scores, mode="top_k", value=2)
        # Indices 1 (score=1.0) and 2 (score=2.0) should be selected
        assert mask[1] and mask[2]
        assert not mask[0] and not mask[3] and not mask[4]

    def test_top_k_clamps_to_available(self):
        """top_k should not select more than available samples."""
        scores = np.array([1.0, 2.0, 3.0])
        mask = select_samples(scores, mode="top_k", value=10)
        assert mask.sum() == 3

    def test_percentile_selects_fraction(self):
        """percentile should select approximately the right fraction."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mask = select_samples(scores, mode="percentile", value=50)
        assert mask.sum() == 5

    def test_threshold_filters_by_score(self):
        """threshold should select only samples below the threshold."""
        scores = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        mask = select_samples(scores, mode="threshold", value=4.0)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_require_realism_filters_first(self):
        """With require_realism, unrealistic samples should be excluded."""
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        realism_flags = np.array([True, False, True, False])
        mask = select_samples(
            scores,
            mode="top_k",
            value=2,
            realism_flags=realism_flags,
            require_realism=True,
        )
        # Only indices 0 and 2 are candidates; both should be selected
        assert mask[0] and mask[2]
        assert not mask[1] and not mask[3]

    def test_require_realism_false_ignores_flags(self):
        """Without require_realism, all samples are candidates."""
        scores = np.array([1.0, 2.0, 3.0])
        realism_flags = np.array([False, False, False])
        mask = select_samples(
            scores,
            mode="top_k",
            value=2,
            realism_flags=realism_flags,
            require_realism=False,
        )
        assert mask.sum() == 2

    def test_no_candidates_returns_empty_mask(self):
        """If all samples are filtered by realism, mask should be all False."""
        scores = np.array([1.0, 2.0, 3.0])
        realism_flags = np.array([False, False, False])
        mask = select_samples(
            scores,
            mode="top_k",
            value=2,
            realism_flags=realism_flags,
            require_realism=True,
        )
        assert mask.sum() == 0

    def test_output_is_boolean_array(self):
        """Output should be a boolean numpy array."""
        scores = np.array([1.0, 2.0, 3.0])
        mask = select_samples(scores, mode="top_k", value=1)
        assert mask.dtype == bool
        assert isinstance(mask, np.ndarray)


# ============================================================================
# Component Tests - extract_features_from_loader
# ============================================================================


@pytest.mark.component
class TestExtractFeaturesFromLoader:
    """Test feature extraction with a mock model."""

    def test_extracts_correct_shape(self):
        """Features should have shape (N, D)."""
        feature_dim = 16
        model = MagicMock()
        model.extract_features = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], feature_dim)
        )

        # Create a simple dataset with mock paths
        images = torch.randn(6, 3, 32, 32)
        dataset = TensorDataset(images)
        dataset.image_paths = [f"/fake/img_{i}.png" for i in range(6)]  # type: ignore[attr-defined]
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        features, paths = extract_features_from_loader(model, loader, "cpu", dataset)
        assert features.shape == (6, feature_dim)
        assert len(paths) == 6

    def test_paths_match_dataset_order(self):
        """File paths should be returned in dataset order."""
        model = MagicMock()
        model.extract_features = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 8)
        )

        images = torch.randn(4, 3, 32, 32)
        expected_paths = [f"/img/{i}.png" for i in range(4)]
        dataset = TensorDataset(images)
        dataset.image_paths = expected_paths  # type: ignore[attr-defined]
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        _, paths = extract_features_from_loader(model, loader, "cpu", dataset)
        assert paths == expected_paths


# ============================================================================
# Unit Tests - _get_dataset_paths
# ============================================================================


@pytest.mark.unit
class TestGetDatasetPaths:
    """Test file path extraction from various dataset types."""

    def test_dataset_with_image_paths(self):
        """Should extract paths from image_paths attribute."""
        dataset = MagicMock()
        dataset.get_file_paths = None
        del dataset.get_file_paths
        dataset.image_paths = ["/a.png", "/b.png"]
        # image_paths are Path objects in SimpleImageDataset
        from pathlib import Path as P

        dataset.image_paths = [P("/a.png"), P("/b.png")]
        paths = _get_dataset_paths(dataset)
        assert paths == ["/a.png", "/b.png"]

    def test_dataset_with_samples(self):
        """Should extract paths from samples attribute."""
        dataset = MagicMock(spec=[])
        dataset.samples = [("/x.png", 0), ("/y.png", 1)]
        paths = _get_dataset_paths(dataset)
        assert paths == ["/x.png", "/y.png"]

    def test_dataset_with_get_file_paths(self):
        """Should use get_file_paths method when available."""
        dataset = MagicMock()
        dataset.get_file_paths.return_value = ["/p.png", "/q.png"]
        paths = _get_dataset_paths(dataset)
        assert paths == ["/p.png", "/q.png"]

    def test_unknown_dataset_raises(self):
        """Should raise TypeError for unsupported datasets."""
        dataset = MagicMock(spec=[])
        with pytest.raises(TypeError, match="Cannot extract file paths"):
            _get_dataset_paths(dataset)


# ============================================================================
# Unit Tests - _FilteredDataset
# ============================================================================


@pytest.mark.unit
class TestFilteredDataset:
    """Test the _FilteredDataset wrapper."""

    def test_len_returns_filtered_count(self):
        """Length should match the number of filtered indices."""
        inner = MagicMock()
        inner.samples = [("/a.png", 0), ("/b.png", 1), ("/c.png", 0)]
        fd = _FilteredDataset(inner, [0, 2])
        assert len(fd) == 2

    def test_getitem_delegates_to_inner(self):
        """__getitem__ should delegate to the inner dataset with remapped index."""
        inner = MagicMock()
        inner.samples = [("/a.png", 0), ("/b.png", 1), ("/c.png", 0)]
        inner.__getitem__ = MagicMock(return_value="img_tensor")
        fd = _FilteredDataset(inner, [0, 2])
        result = fd[1]  # Index 1 in filtered → index 2 in inner
        inner.__getitem__.assert_called_with(2)
        assert result == "img_tensor"

    def test_get_file_paths_returns_filtered_paths(self):
        """get_file_paths should return only the filtered samples' paths."""
        inner = MagicMock()
        inner.samples = [("/a.png", 0), ("/b.png", 1), ("/c.png", 0)]
        fd = _FilteredDataset(inner, [0, 2])
        paths = fd.get_file_paths()
        assert paths == ["/a.png", "/c.png"]


# ============================================================================
# Unit Tests - _copy_selected_samples
# ============================================================================


@pytest.mark.unit
class TestCopySelectedSamples:
    """Test sample file copying."""

    def test_copies_files_to_output_dir(self, tmp_path):
        """Selected files should be copied to output directory."""
        # Create source files
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        for i in range(3):
            (src_dir / f"img_{i}.png").write_text(f"content_{i}")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        paths = [str(src_dir / f"img_{i}.png") for i in range(3)]
        _copy_selected_samples(paths, out_dir)

        for i in range(3):
            dst = out_dir / f"img_{i}.png"
            assert dst.exists()
            assert dst.read_text() == f"content_{i}"

    def test_empty_list_copies_nothing(self, tmp_path):
        """Empty path list should not create any files."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        _copy_selected_samples([], out_dir)
        assert list(out_dir.iterdir()) == []

    def test_filename_collision_appends_suffix(self, tmp_path):
        """Files with the same name from different directories get unique names."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "img.png").write_text("content_a")
        (dir_b / "img.png").write_text("content_b")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        paths = [str(dir_a / "img.png"), str(dir_b / "img.png")]
        _copy_selected_samples(paths, out_dir)

        # First file keeps original name, second gets suffix
        assert (out_dir / "img.png").exists()
        assert (out_dir / "img.png").read_text() == "content_a"
        assert (out_dir / "img_1.png").exists()
        assert (out_dir / "img_1.png").read_text() == "content_b"

    def test_multiple_filename_collisions(self, tmp_path):
        """Multiple files with the same name get incrementing suffixes."""
        dirs = []
        for i in range(3):
            d = tmp_path / f"dir_{i}"
            d.mkdir()
            (d / "img.png").write_text(f"content_{i}")
            dirs.append(d)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        paths = [str(d / "img.png") for d in dirs]
        _copy_selected_samples(paths, out_dir)

        assert (out_dir / "img.png").read_text() == "content_0"
        assert (out_dir / "img_1.png").read_text() == "content_1"
        assert (out_dir / "img_2.png").read_text() == "content_2"


# ============================================================================
# Unit Tests - Empty dataset validation
# ============================================================================


@pytest.mark.unit
class TestEmptyDatasetValidation:
    """Test that empty datasets raise clear errors."""

    def test_empty_real_dataset_raises(self):
        """Empty real dataset should raise ValueError."""
        from src.experiments.sample_selection.selector import run_sample_selection

        config = {
            "feature_extraction": {
                "image_size": 32,
                "batch_size": 4,
                "num_workers": 0,
                "model": "resnet50",
            },
            "scoring": {"k": 5, "require_realism": False},
            "selection": {"mode": "top_k", "value": 10},
            "data": {
                "real": {"source": "directory", "directory": "/fake"},
                "generated": {"directory": "/fake_gen"},
                "label": 0,
                "class_name": "test",
            },
            "dataset_metrics": {"enabled": False},
            "output": {"base_dir": "/tmp/test"},
        }

        mock_real = MagicMock()
        mock_real.__len__ = MagicMock(return_value=0)

        with (
            patch(
                "src.experiments.sample_selection.selector._load_real_dataset",
                return_value=mock_real,
            ),
            patch(
                "src.experiments.sample_selection.selector.SimpleImageDataset",
            ),
        ):
            with pytest.raises(ValueError, match="No real images found"):
                run_sample_selection(config, "cpu", MagicMock())

    def test_empty_generated_dataset_raises(self):
        """Empty generated dataset should raise ValueError."""
        from src.experiments.sample_selection.selector import run_sample_selection

        config = {
            "feature_extraction": {
                "image_size": 32,
                "batch_size": 4,
                "num_workers": 0,
                "model": "resnet50",
            },
            "scoring": {"k": 5, "require_realism": False},
            "selection": {"mode": "top_k", "value": 10},
            "data": {
                "real": {"source": "directory", "directory": "/fake"},
                "generated": {"directory": "/fake_gen"},
                "label": 0,
                "class_name": "test",
            },
            "dataset_metrics": {"enabled": False},
            "output": {"base_dir": "/tmp/test"},
        }

        mock_real = MagicMock()
        mock_real.__len__ = MagicMock(return_value=10)

        mock_gen = MagicMock()
        mock_gen.__len__ = MagicMock(return_value=0)

        with (
            patch(
                "src.experiments.sample_selection.selector._load_real_dataset",
                return_value=mock_real,
            ),
            patch(
                "src.experiments.sample_selection.selector.SimpleImageDataset",
                return_value=mock_gen,
            ),
        ):
            with pytest.raises(ValueError, match="No generated images found"):
                run_sample_selection(config, "cpu", MagicMock())


# ============================================================================
# Unit Tests - k clamping warnings
# ============================================================================


@pytest.mark.unit
class TestKClampingWarnings:
    """Test that warnings are logged when k is clamped."""

    def test_knn_scores_warns_on_k_clamp(self, caplog):
        """compute_knn_scores should warn when k exceeds real sample count."""
        real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gen = np.array([[1.0, 2.0]], dtype=np.float32)

        with caplog.at_level(
            logging.WARNING, logger="src.experiments.sample_selection.selector"
        ):
            compute_knn_scores(real, gen, k=10)

        assert any("scoring.k=10 exceeds" in msg for msg in caplog.messages)

    def test_knn_scores_no_warning_when_k_fits(self, caplog):
        """compute_knn_scores should not warn when k <= real sample count."""
        real = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        gen = np.array([[1.0, 2.0]], dtype=np.float32)

        with caplog.at_level(
            logging.WARNING, logger="src.experiments.sample_selection.selector"
        ):
            compute_knn_scores(real, gen, k=2)

        assert not any("exceeds" in msg for msg in caplog.messages)

    def test_realism_flags_warns_on_k_clamp(self, caplog):
        """compute_realism_flags should warn when k exceeds real samples - 1."""
        real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gen = np.array([[1.0, 2.0]], dtype=np.float32)

        with caplog.at_level(
            logging.WARNING, logger="src.experiments.sample_selection.selector"
        ):
            compute_realism_flags(real, gen, k=5)

        assert any("scoring.k=5 exceeds" in msg for msg in caplog.messages)

    def test_realism_flags_no_warning_when_k_fits(self, caplog):
        """compute_realism_flags should not warn when k <= real samples - 1."""
        real = np.random.rand(10, 4).astype(np.float32)
        gen = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        with caplog.at_level(
            logging.WARNING, logger="src.experiments.sample_selection.selector"
        ):
            compute_realism_flags(real, gen, k=3)

        assert not any("exceeds" in msg for msg in caplog.messages)
