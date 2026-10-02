"""Tests for scripts/campaign_config.py (campaign sweep validation/expansion)."""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.campaign_config import (
    expand_runs,
    load_campaign,
    resolve_paths,
    validate_sweep,
)


def _base():
    return {
        "experiment": "anomaly_detection",
        "compute": {"device": "cpu", "seed": 0},
        "data": {"split_file": "x.json", "normal_pool": "all"},
        "feature_extraction": {"model": "resnet50", "cache_dir": "../shared/cache"},
        "method": {"type": "knn", "patchcore": {"layers": ["layer2"]}},
        "output": {"base_dir": "runs/placeholder", "subdirs": {"logs": "logs"}},
    }


def _sweep():
    return {
        "base_config": "base.yaml",
        "family": "ad-frozen",
        "done_marker": "reports/evaluation.json",
        "path_keys": ["data.split_file", "feature_extraction.cache_dir"],
        "splits": {
            "key": "data.split_file",
            "template": "../shared/splits/s{split}.json",
            "indices": [0, 1],
        },
        "seeds": {"key": "compute.seed", "values": [0, 1]},
        "name_template": "ad-{method}-{backbone}",
        "axes": {
            "method": {
                "knn": {"method.type": "knn"},
                "maha": {"method.type": "mahalanobis"},
            },
            "backbone": {
                "rn50": {"feature_extraction.model": "resnet50"},
                "incv3": {
                    "feature_extraction.model": "inceptionv3",
                    "method.patchcore.layers": ["Mixed_6e"],
                },
            },
        },
    }


@pytest.mark.unit
class TestValidateSweep:
    def test_valid(self):
        validate_sweep(_sweep(), _base())

    @pytest.mark.parametrize(
        "key",
        [
            "base_config",
            "family",
            "done_marker",
            "path_keys",
            "splits",
            "seeds",
            "name_template",
            "axes",
        ],
    )
    def test_missing_top_level(self, key):
        sweep = _sweep()
        del sweep[key]
        with pytest.raises(KeyError, match=key):
            validate_sweep(sweep, _base())

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda s: s["splits"].update(template="no-placeholder.json"), "{split}"),
            (lambda s: s["splits"].update(indices=[]), "indices"),
            (lambda s: s["splits"].update(indices=[0, 0]), "duplicates"),
            (lambda s: s["seeds"].update(values=[-1]), "values"),
            (lambda s: s["seeds"].update(values=[True]), "values"),
            (lambda s: s.update(family=""), "family"),
            (lambda s: s.update(path_keys="data.split_file"), "path_keys"),
            (lambda s: s.update(axes={}), "axes"),
            (lambda s: s["axes"].update(method={}), "method"),
            (lambda s: s["axes"]["method"].update(knn="not-a-map"), "knn"),
            (lambda s: s.update(name_template="ad-{method}"), "name_template"),
            (
                lambda s: s["axes"]["method"]["knn"].update({"output.base_dir": "x"}),
                "set by the driver",
            ),
            (
                lambda s: s["axes"]["method"]["knn"].update({"compute.seed": 3}),
                "set by the driver",
            ),
        ],
    )
    def test_invalid_values(self, mutate, match):
        sweep = _sweep()
        mutate(sweep)
        with pytest.raises(ValueError, match=match):
            validate_sweep(sweep, _base())

    def test_unknown_override_key_rejected(self):
        sweep = _sweep()
        sweep["axes"]["method"]["knn"]["method.typo"] = "knn"
        with pytest.raises(ValueError):
            validate_sweep(sweep, _base())

    def test_unknown_path_key_rejected(self):
        sweep = _sweep()
        sweep["path_keys"] = ["data.nope"]
        with pytest.raises(KeyError, match="data.nope"):
            validate_sweep(sweep, _base())


@pytest.mark.unit
class TestRequire:
    def test_message_without_prefix(self):
        from scripts.campaign_config import _require

        with pytest.raises(KeyError, match="Missing required field: key"):
            _require({}, "key", "")

    def test_message_with_prefix(self):
        from scripts.campaign_config import _require

        with pytest.raises(KeyError, match="Missing required field: sweep.key"):
            _require({}, "key", "sweep")


@pytest.mark.unit
class TestExpandRuns:
    def test_cartesian_product_and_order(self):
        runs = expand_runs(_sweep(), _base())
        assert len(runs) == 2 * 2 * 2 * 2  # methods x backbones x splits x seeds
        assert [r.name for r in runs[:4]] == ["ad-knn-rn50"] * 4
        assert [(r.split, r.seed) for r in runs[:4]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert sorted({r.name for r in runs}) == [
            "ad-knn-incv3",
            "ad-knn-rn50",
            "ad-maha-incv3",
            "ad-maha-rn50",
        ]

    def test_driver_keys_and_overrides_applied(self):
        runs = {(r.name, r.split, r.seed): r for r in expand_runs(_sweep(), _base())}
        run = runs[("ad-maha-incv3", 1, 1)]
        cfg = run.config
        assert cfg["data"]["split_file"] == "../shared/splits/s1.json"
        assert cfg["compute"]["seed"] == 1
        assert cfg["method"]["type"] == "mahalanobis"
        assert cfg["feature_extraction"]["model"] == "inceptionv3"
        assert cfg["method"]["patchcore"]["layers"] == ["Mixed_6e"]
        assert cfg["output"]["base_dir"] == "runs/split1/ad-frozen/ad-maha-incv3/seed1"
        assert run.output_dir == "runs/split1/ad-frozen/ad-maha-incv3/seed1"
        assert run.config_path == Path("configs/runs/ad-maha-incv3/split1_seed1.yaml")

    def test_base_config_not_mutated(self):
        base = _base()
        before = copy.deepcopy(base)
        expand_runs(_sweep(), base)
        assert base == before

    def test_conditions_do_not_share_list_objects(self):
        runs = expand_runs(_sweep(), _base())
        layers = [r.config["method"]["patchcore"]["layers"] for r in runs]
        assert len({id(x) for x in layers}) == len(layers)

    def test_duplicate_names_rejected(self):
        sweep = _sweep()
        sweep["name_template"] = "ad-{method}{backbone}"
        sweep["axes"]["method"] = {"a": {}, "ab": {}}
        sweep["axes"]["backbone"] = {"bc": {}, "c": {}}
        with pytest.raises(ValueError, match="Duplicate"):
            expand_runs(sweep, _base())


@pytest.mark.unit
class TestResolvePaths:
    def test_relative_paths_resolved_against_campaign(self, tmp_path):
        cfg = expand_runs(_sweep(), _base())[0].config
        out = resolve_paths(
            cfg, ["data.split_file", "feature_extraction.cache_dir"], tmp_path
        )
        campaign = tmp_path.resolve()
        assert out["output"]["base_dir"] == str(
            campaign / "runs/split0/ad-frozen/ad-knn-rn50/seed0"
        )
        assert out["data"]["split_file"] == str(
            campaign.parent / "shared/splits/s0.json"
        )
        assert out["feature_extraction"]["cache_dir"] == str(
            campaign.parent / "shared/cache"
        )
        # The stored config keeps its campaign-relative paths.
        assert cfg["data"]["split_file"] == "../shared/splits/s0.json"

    def test_absolute_and_none_untouched(self, tmp_path):
        cfg = _base()
        cfg["data"]["split_file"] = "/abs/split.json"
        cfg["feature_extraction"]["cache_dir"] = None
        out = resolve_paths(
            cfg, ["data.split_file", "feature_extraction.cache_dir"], tmp_path
        )
        assert out["data"]["split_file"] == "/abs/split.json"
        assert out["feature_extraction"]["cache_dir"] is None


@pytest.mark.unit
class TestLoadCampaign:
    def _write(self, campaign, sweep, base):
        (campaign / "configs").mkdir(parents=True)
        (campaign / "configs" / "sweep.yaml").write_text(yaml.safe_dump(sweep))
        if base is not None:
            (campaign / "configs" / "base.yaml").write_text(yaml.safe_dump(base))

    def test_loads_and_expands(self, tmp_path):
        self._write(tmp_path, _sweep(), _base())
        sweep, runs = load_campaign(tmp_path)
        assert sweep["family"] == "ad-frozen"
        assert len(runs) == 16

    def test_missing_sweep(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Sweep"):
            load_campaign(tmp_path)

    def test_missing_base(self, tmp_path):
        self._write(tmp_path, _sweep(), None)
        with pytest.raises(FileNotFoundError, match="Base config"):
            load_campaign(tmp_path)
