"""Tests for frozen feature extraction and caching."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.experiments.anomaly_detection.features import (
    PathListDataset,
    build_feature_spec,
    get_features,
    patch_embeddings,
    spec_cache_name,
)


def _config(method="knn"):
    return {
        "feature_extraction": {"model": "resnet50", "image_size": 32, "crop_size": 32},
        "method": {
            "type": method,
            "patchcore": {
                "layers": ["layer2", "layer3"],
                "grid_size": 4,
                "feature_dim": 5,
            },
        },
    }


@pytest.mark.unit
class TestFeatureSpec:
    def test_pooled_spec(self):
        spec = build_feature_spec(_config("knn"))
        assert spec["kind"] == "pooled"
        assert "layers" not in spec

    def test_patch_spec(self):
        spec = build_feature_spec(_config("patchcore"))
        assert spec["kind"] == "patch"
        assert spec["layers"] == ["layer2", "layer3"]

    def test_cache_name_depends_on_spec(self):
        a = build_feature_spec(_config("knn"))
        b = dict(a, crop_size=16)
        assert spec_cache_name(a) != spec_cache_name(b)
        assert spec_cache_name(a) == spec_cache_name(dict(a))
        assert spec_cache_name(a).startswith("resnet50_pooled_")


@pytest.mark.unit
class TestPatchEmbeddings:
    def test_shape_and_upsampling(self):
        maps = [torch.randn(2, 4, 8, 8), torch.randn(2, 6, 2, 2)]
        out = patch_embeddings(maps, grid_size=4, feature_dim=5)
        assert out.shape == (2, 16, 5)

    def test_constant_maps_give_constant_embeddings(self):
        maps = [torch.ones(1, 4, 8, 8)]
        out = patch_embeddings(maps, grid_size=4, feature_dim=2)
        # Interior patches are exactly 1 (zero padding only affects borders).
        assert torch.allclose(out.reshape(4, 4, 2)[1:3, 1:3], torch.ones(2, 2, 2))


@pytest.mark.unit
class TestGetFeaturesCache:
    def _spec(self):
        return build_feature_spec(_config("knn"))

    def test_aligned_with_duplicates_and_no_cache(self):
        fake = np.arange(6, dtype=np.float32).reshape(3, 2)
        with patch(
            "src.experiments.anomaly_detection.features._extract", return_value=fake
        ) as mock_extract:
            out = get_features(["a", "b", "a", "c"], self._spec(), "cpu", 2, 0, None)
        assert mock_extract.call_args[0][0] == ["a", "b", "c"]
        np.testing.assert_array_equal(out, fake[[0, 1, 0, 2]])

    def test_cache_reused_and_extended(self, tmp_path):
        spec = self._spec()
        with patch(
            "src.experiments.anomaly_detection.features._extract",
            return_value=np.ones((2, 2), dtype=np.float32),
        ):
            get_features(["a", "b"], spec, "cpu", 2, 0, str(tmp_path))
        with patch(
            "src.experiments.anomaly_detection.features._extract",
            return_value=np.full((1, 2), 7.0, dtype=np.float32),
        ) as mock_extract:
            out = get_features(["b", "c"], spec, "cpu", 2, 0, str(tmp_path))
        assert mock_extract.call_args[0][0] == ["c"]
        np.testing.assert_array_equal(out, [[1, 1], [7, 7]])

    def test_cache_spec_mismatch_raises(self, tmp_path):
        spec = self._spec()
        with patch(
            "src.experiments.anomaly_detection.features._extract",
            return_value=np.ones((1, 2), dtype=np.float32),
        ):
            get_features(["a"], spec, "cpu", 2, 0, str(tmp_path))
        # Same file name but a tampered stored spec must be refused.
        cache = tmp_path / spec_cache_name(spec)
        data = dict(np.load(cache))
        data["spec"] = np.array('{"model": "other"}')
        np.savez(cache, **data)
        with pytest.raises(ValueError, match="different spec"):
            get_features(["a"], spec, "cpu", 2, 0, str(tmp_path))


@pytest.mark.unit
class TestExtraction:
    def test_pooled_and_patch_extraction(self, tiny_backbone, ad_split_file):
        import json

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        pooled = get_features(
            paths, build_feature_spec(_config("knn")), "cpu", 3, 0, None
        )
        assert pooled.shape == (len(paths), 6)
        patches = get_features(
            paths, build_feature_spec(_config("patchcore")), "cpu", 3, 0, None
        )
        assert patches.shape == (len(paths), 16, 5)
        assert patches.dtype == np.float16

    def test_hooks_removed_after_extraction(self, tiny_backbone, ad_split_file):
        import json

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        get_features(paths, build_feature_spec(_config("patchcore")), "cpu", 3, 0, None)
        assert not tiny_backbone.layer2._forward_hooks
        assert not tiny_backbone.layer3._forward_hooks

    def test_path_list_dataset(self, ad_split_file):
        import json

        from src.utils.data.transforms import get_val_transforms

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        dataset = PathListDataset(paths, get_val_transforms(16, 16, None))
        assert len(dataset) == len(paths)
        assert dataset[0].shape == (3, 16, 16)
