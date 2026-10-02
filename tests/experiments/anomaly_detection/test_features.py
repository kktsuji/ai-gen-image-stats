"""Tests for frozen feature extraction and caching."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.experiments.anomaly_detection.features import (
    PathListDataset,
    build_feature_spec,
    file_sha256,
    get_features,
    load_backbone_weights,
    patch_embeddings,
    spec_cache_name,
)


def _config(method="knn", checkpoint=None):
    return {
        "feature_extraction": {
            "model": "resnet50",
            "image_size": 32,
            "crop_size": 32,
            "checkpoint": checkpoint,
        },
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

    def test_no_checkpoint_keeps_imagenet_spec(self):
        # ImageNet specs must stay unchanged so existing caches remain valid.
        assert "checkpoint_sha256" not in build_feature_spec(_config("knn"))


def _save_classifier_checkpoint(path, backbone, extra=None):
    """Save a ``save_checkpoint``-style file: backbone weights plus a head."""
    state = dict(backbone.state_dict())
    state["fc.weight"] = torch.zeros(2, 6)
    state["fc.bias"] = torch.zeros(2)
    state.update(extra or {})
    torch.save({"model_state_dict": state, "epoch": 1}, path)
    return str(path)


class _Unsafe:
    """Stand-in for an arbitrary pickled object (not allowed by weights_only)."""


def _shifted_backbone(offset):
    from tests.experiments.anomaly_detection.conftest import TinyBackbone

    model = TinyBackbone()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(offset)
    return model


@pytest.mark.unit
class TestCheckpointWeights:
    def test_spec_records_content_hash(self, tmp_path):
        ckpt = _save_classifier_checkpoint(tmp_path / "a.pth", _shifted_backbone(1))
        spec = build_feature_spec(_config("knn", checkpoint=ckpt))
        assert spec["checkpoint_sha256"] == file_sha256(ckpt)
        assert ckpt not in str(spec)  # the path is not part of the cache key
        assert spec_cache_name(spec).startswith("resnet50-ft_pooled_")
        other = _save_classifier_checkpoint(tmp_path / "b.pth", _shifted_backbone(2))
        other_spec = build_feature_spec(_config("knn", checkpoint=other))
        assert spec_cache_name(spec) != spec_cache_name(other_spec)

    def test_loads_backbone_and_skips_head(self, tmp_path):
        from tests.experiments.anomaly_detection.conftest import TinyBackbone

        source = _shifted_backbone(1)
        ckpt = _save_classifier_checkpoint(tmp_path / "a.pth", source)
        model = TinyBackbone()
        load_backbone_weights(model, ckpt)
        for name, param in model.state_dict().items():
            assert torch.equal(param, source.state_dict()[name])
        assert not model.training

    def test_missing_backbone_key_raises(self, tmp_path):
        from tests.experiments.anomaly_detection.conftest import TinyBackbone

        state = {
            k: v
            for k, v in TinyBackbone().state_dict().items()
            if not k.startswith("layer3")
        }
        torch.save({"model_state_dict": state}, tmp_path / "a.pth")
        with pytest.raises(ValueError, match="does not match"):
            load_backbone_weights(TinyBackbone(), str(tmp_path / "a.pth"))

    def test_save_checkpoint_format_loads_weights_only(self, tmp_path):
        from src.utils.checkpoint import save_checkpoint
        from tests.experiments.anomaly_detection.conftest import TinyBackbone

        source = _shifted_backbone(1)
        path = tmp_path / "best_model.pth"
        save_checkpoint(
            path,
            model=source,
            optimizer=torch.optim.SGD(source.parameters(), lr=0.1),
            epoch=3,
            global_step=30,
            is_best=True,
            metrics={"loss": 0.5, "f1_1": 0.8},
            best_metric=0.8,
            best_metric_name="f1_1",
            trainer_class="ClassifierTrainer",
            save_optimizer=False,
        )
        model = TinyBackbone()
        load_backbone_weights(model, str(path))
        assert torch.equal(model.layer2.weight, source.layer2.weight)

    def test_arbitrary_pickled_objects_refused(self, tmp_path):
        import pickle

        from tests.experiments.anomaly_detection.conftest import TinyBackbone

        ckpt = _save_classifier_checkpoint(
            tmp_path / "a.pth", TinyBackbone(), {"evil": _Unsafe()}
        )
        with pytest.raises(pickle.UnpicklingError):
            load_backbone_weights(TinyBackbone(), ckpt)

    def test_unexpected_key_raises(self, tmp_path):
        from tests.experiments.anomaly_detection.conftest import TinyBackbone

        ckpt = _save_classifier_checkpoint(
            tmp_path / "a.pth", TinyBackbone(), {"layer9.weight": torch.zeros(1)}
        )
        with pytest.raises(ValueError, match="does not match"):
            load_backbone_weights(TinyBackbone(), ckpt)


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


def _crash_mid_write(file, **arrays):
    """Stand-in for np.savez that writes partial bytes, then fails."""
    if isinstance(file, (str, bytes)) or hasattr(file, "__fspath__"):
        with open(file, "wb") as f:
            f.write(b"PK-partial")
    else:
        file.write(b"PK-partial")
    raise OSError("disk full")


@pytest.mark.unit
class TestCacheWriteSafety:
    def _spec(self):
        return build_feature_spec(_config("knn"))

    def test_lock_file_used_and_no_temp_left(self, tmp_path):
        spec = self._spec()
        with patch(
            "src.experiments.anomaly_detection.features._extract",
            return_value=np.ones((1, 2), dtype=np.float32),
        ):
            get_features(["a"], spec, "cpu", 2, 0, str(tmp_path))
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == [spec_cache_name(spec), spec_cache_name(spec) + ".lock"]

    def test_failed_write_keeps_previous_cache(self, tmp_path):
        spec = self._spec()
        with patch(
            "src.experiments.anomaly_detection.features._extract",
            return_value=np.ones((1, 2), dtype=np.float32),
        ):
            get_features(["a"], spec, "cpu", 2, 0, str(tmp_path))
        cache = tmp_path / spec_cache_name(spec)
        before = cache.read_bytes()

        with (
            patch(
                "src.experiments.anomaly_detection.features._extract",
                return_value=np.full((1, 2), 5.0, dtype=np.float32),
            ),
            patch(
                "src.experiments.anomaly_detection.features.np.savez",
                side_effect=_crash_mid_write,
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            get_features(["b"], spec, "cpu", 2, 0, str(tmp_path))

        assert cache.read_bytes() == before
        assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.component
class TestCacheConcurrency:
    def test_concurrent_writers_keep_all_rows(self, tmp_path):
        """Two runs extending the same cache at once must not lose rows."""
        import threading
        import time

        spec = build_feature_spec(_config("knn"))

        def slow_extract(paths, *args):
            time.sleep(0.05)  # widen the read-modify-write window
            return np.array([[float(p[1:])] * 2 for p in paths], dtype=np.float32)

        groups = [[f"x{i}" for i in range(k, k + 3)] for k in (0, 10, 20, 30)]
        errors = []

        def worker(paths):
            try:
                get_features(paths, spec, "cpu", 2, 0, str(tmp_path))
            except Exception as e:  # pragma: no cover - surfaced below
                errors.append(e)

        with patch(
            "src.experiments.anomaly_detection.features._extract",
            side_effect=slow_extract,
        ):
            threads = [threading.Thread(target=worker, args=(g,)) for g in groups]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        with np.load(tmp_path / spec_cache_name(spec)) as data:
            stored = {str(p): row for p, row in zip(data["paths"], data["features"])}
        all_paths = [p for g in groups for p in g]
        assert sorted(stored) == sorted(all_paths)
        for path in all_paths:
            np.testing.assert_array_equal(stored[path], [float(path[1:])] * 2)


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

    def test_checkpoint_weights_change_features(
        self, tiny_backbone, ad_split_file, tmp_path
    ):
        import json

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        plain = get_features(
            paths, build_feature_spec(_config("knn")), "cpu", 3, 0, None
        )
        ckpt = _save_classifier_checkpoint(tmp_path / "a.pth", _shifted_backbone(1))
        spec = build_feature_spec(_config("knn", checkpoint=ckpt))
        tuned = get_features(paths, spec, "cpu", 3, 0, None, checkpoint=ckpt)
        assert tuned.shape == plain.shape
        assert not np.allclose(tuned, plain)
        assert torch.equal(
            tiny_backbone.layer2.weight, _shifted_backbone(1).layer2.weight
        )

    def test_checkpoint_changed_after_spec_raises(
        self, tiny_backbone, ad_split_file, tmp_path
    ):
        import json

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        ckpt = _save_classifier_checkpoint(tmp_path / "a.pth", _shifted_backbone(1))
        spec = build_feature_spec(_config("knn", checkpoint=ckpt))
        _save_classifier_checkpoint(tmp_path / "a.pth", _shifted_backbone(2))
        with pytest.raises(ValueError, match="changed"):
            get_features(paths, spec, "cpu", 3, 0, None, checkpoint=ckpt)

    def test_path_list_dataset(self, ad_split_file):
        import json

        from src.utils.data.transforms import get_val_transforms

        paths = [e["path"] for e in json.loads(ad_split_file.read_text())["test"]]
        dataset = PathListDataset(paths, get_val_transforms(16, 16, None))
        assert len(dataset) == len(paths)
        assert dataset[0].shape == (3, 16, 16)
