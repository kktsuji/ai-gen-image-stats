"""Tests for one-class anomaly detectors."""

import numpy as np
import pytest
import torch

from src.experiments.anomaly_detection.methods import (
    KNNDetector,
    MahalanobisDetector,
    PatchCoreDetector,
    build_detector,
)
from src.experiments.anomaly_detection.methods.patchcore import (
    greedy_coreset,
    nearest_distances,
)


def _normal_and_queries(dim=4, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(200, dim)).astype(np.float32)
    inlier = np.zeros((1, dim), dtype=np.float32)
    outlier = np.full((1, dim), 6.0, dtype=np.float32)
    return normal, np.vstack([inlier, outlier])


@pytest.mark.unit
class TestPooledDetectors:
    @pytest.mark.parametrize("detector", [KNNDetector(k=3), MahalanobisDetector()])
    def test_outlier_scores_higher(self, detector):
        normal, queries = _normal_and_queries()
        scores = detector.fit(normal).score(queries)
        assert scores.shape == (2,)
        assert scores[1] > scores[0]

    @pytest.mark.parametrize("detector", [KNNDetector(k=3), MahalanobisDetector()])
    def test_score_before_fit_raises(self, detector):
        with pytest.raises(RuntimeError, match="before fit"):
            detector.score(np.zeros((1, 4)))

    def test_mahalanobis_uses_covariance(self):
        # Elongated normal cloud: a point along the long axis is less anomalous
        # than one equally far along the short axis.
        rng = np.random.default_rng(0)
        normal = rng.normal(size=(500, 2)) * np.array([5.0, 0.5])
        scores = MahalanobisDetector().fit(normal).score(np.array([[3.0, 0], [0, 3.0]]))
        assert scores[1] > scores[0]


@pytest.mark.unit
class TestPatchCore:
    def test_greedy_coreset_covers_clusters(self):
        # Two tight, far-apart clusters: a 2-point coreset must hit both.
        feats = torch.cat([torch.zeros(50, 3), torch.full((50, 3), 10.0)])
        feats = feats + 0.01 * torch.randn(
            100, 3, generator=torch.Generator().manual_seed(0)
        )
        idx = greedy_coreset(feats, 2, 8, torch.Generator().manual_seed(0))
        chosen = feats[idx]
        assert (chosen[:, 0] < 5).sum() == 1 and (chosen[:, 0] > 5).sum() == 1

    def test_greedy_coreset_all_when_n_select_large(self):
        feats = torch.randn(5, 3)
        idx = greedy_coreset(feats, 10, 2, torch.Generator().manual_seed(0))
        assert idx.tolist() == list(range(5))

    def test_greedy_coreset_with_projection(self):
        feats = torch.randn(40, 16, generator=torch.Generator().manual_seed(1))
        idx = greedy_coreset(feats, 5, 4, torch.Generator().manual_seed(0))
        assert len(set(idx.tolist())) == 5

    def test_nearest_distances_chunked_matches_full(self):
        gen = torch.Generator().manual_seed(0)
        queries = torch.randn(11, 3, generator=gen)
        bank = torch.randn(7, 3, generator=gen)
        full = torch.cdist(queries, bank).min(dim=1).values
        assert torch.allclose(nearest_distances(queries, bank, chunk=3), full)

    def test_outlier_patch_raises_image_score(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(size=(30, 4, 3)).astype(np.float32)
        clean = np.zeros((1, 4, 3), dtype=np.float32)
        defect = clean.copy()
        defect[0, 2] = 8.0  # one anomalous patch
        detector = PatchCoreDetector(
            coreset_ratio=0.5, projection_dim=2, seed=0, device="cpu"
        ).fit(normal)
        scores = detector.score(np.vstack([clean, defect]))
        assert scores[1] > scores[0]

    def test_seed_makes_bank_deterministic(self):
        normal = np.random.default_rng(0).normal(size=(20, 4, 3)).astype(np.float32)
        banks = [
            PatchCoreDetector(0.2, 2, seed=7, device="cpu").fit(normal).memory_bank
            for _ in range(2)
        ]
        assert banks[0] is not None and banks[1] is not None
        assert torch.equal(banks[0], banks[1])

    def test_rejects_pooled_features(self):
        with pytest.raises(ValueError, match=r"\(N, P, D\)"):
            PatchCoreDetector(0.1, 2, seed=0, device="cpu").fit(np.zeros((5, 3)))

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            PatchCoreDetector(0.1, 2, seed=None, device="cpu").score(
                np.zeros((1, 2, 3))
            )


@pytest.mark.unit
class TestBuildDetector:
    def test_builds_each_type(self):
        params = {
            "knn": {"k": 3},
            "mahalanobis": {"shrinkage": "ledoit_wolf"},
            "patchcore": {"coreset_ratio": 0.1, "projection_dim": 8},
        }
        expected = {
            "knn": KNNDetector,
            "mahalanobis": MahalanobisDetector,
            "patchcore": PatchCoreDetector,
        }
        for method, cls in expected.items():
            detector = build_detector({"type": method, **params}, seed=0, device="cpu")
            assert isinstance(detector, cls)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_detector({"type": "svdd", "svdd": {}}, seed=0, device="cpu")
