"""Tests for anomaly-detection config validation."""

import copy

import pytest
import yaml

from src.experiments.anomaly_detection.config import validate_config


def _config():
    return {
        "experiment": "anomaly_detection",
        "mode": "run",
        "compute": {"device": "cpu", "seed": 0},
        "data": {"split_file": "split.json", "normal_pool": "all"},
        "feature_extraction": {
            "model": "resnet50",
            "image_size": 256,
            "crop_size": 224,
            "batch_size": 8,
            "num_workers": 0,
            "cache_dir": None,
            "checkpoint": None,
        },
        "method": {
            "type": "knn",
            "knn": {"k": 5},
            "mahalanobis": {"shrinkage": "ledoit_wolf"},
            "patchcore": {
                "layers": ["layer2", "layer3"],
                "grid_size": 14,
                "feature_dim": 256,
                "coreset_ratio": 0.01,
                "projection_dim": 128,
            },
        },
        "threshold": {"normal_percentile": 95},
        "output": {
            "base_dir": "out",
            "subdirs": {"logs": "logs", "reports": "reports"},
        },
        "logging": {"console_level": "INFO", "file_level": "INFO"},
    }


@pytest.mark.unit
class TestValidateConfig:
    @pytest.mark.parametrize("method", ["knn", "mahalanobis", "patchcore"])
    def test_valid_methods(self, method):
        config = _config()
        config["method"]["type"] = method
        validate_config(config)

    def test_example_config_is_valid(self):
        with open("configs/examples/anomaly-detection.yaml", encoding="utf-8") as f:
            validate_config(yaml.safe_load(f))

    def test_only_selected_method_section_required(self):
        config = _config()
        del config["method"]["patchcore"]
        del config["method"]["mahalanobis"]
        validate_config(config)

    @pytest.mark.parametrize(
        "section", ["data", "feature_extraction", "method", "threshold", "logging"]
    )
    def test_missing_section(self, section):
        config = _config()
        del config[section]
        with pytest.raises(KeyError, match=section):
            validate_config(config)

    @pytest.mark.parametrize(
        "path",
        [
            ("data", "split_file"),
            ("data", "normal_pool"),
            ("feature_extraction", "model"),
            ("feature_extraction", "cache_dir"),
            ("feature_extraction", "checkpoint"),
            ("feature_extraction", "num_workers"),
            ("method", "type"),
            ("method", "knn"),
            ("threshold", "normal_percentile"),
            ("logging", "file_level"),
        ],
    )
    def test_missing_field(self, path):
        config = _config()
        del config[path[0]][path[1]]
        with pytest.raises(KeyError, match=path[1]):
            validate_config(config)

    @pytest.mark.parametrize(
        "path,value",
        [
            (("experiment",), "classifier"),
            (("mode",), "train"),
            (("data", "split_file"), ""),
            (("data", "normal_pool"), "red"),
            (("feature_extraction", "model"), "vit"),
            (("feature_extraction", "batch_size"), 0),
            (("feature_extraction", "num_workers"), -1),
            (("feature_extraction", "image_size"), True),
            (("feature_extraction", "crop_size"), 300),
            (("feature_extraction", "cache_dir"), ""),
            (("feature_extraction", "checkpoint"), ""),
            (("feature_extraction", "checkpoint"), 3),
            (("method", "type"), "svdd"),
            (("method", "knn", "k"), 0),
            (("method", "mahalanobis", "shrinkage"), "oas"),
            (("threshold", "normal_percentile"), 100),
            (("threshold", "normal_percentile"), "95"),
            (("logging", "console_level"), "LOUD"),
            (("data",), "not-a-dict"),
        ],
    )
    def test_invalid_values(self, path, value):
        config = _config()
        if path[-1] == "knn" or path[-1] == "k":
            config["method"]["type"] = "knn"
        if path[:2] == ("method", "mahalanobis"):
            config["method"]["type"] = "mahalanobis"
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            validate_config(config)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("layers", []),
            ("layers", ["Mixed_6e"]),
            ("layers", ["layer2", "layer2"]),
            ("grid_size", 0),
            ("feature_dim", 1.5),
            ("projection_dim", None),
            ("coreset_ratio", 0),
            ("coreset_ratio", 1.5),
            ("coreset_ratio", True),
        ],
    )
    def test_invalid_patchcore(self, field, value):
        config = _config()
        config["method"]["type"] = "patchcore"
        config["method"]["patchcore"][field] = value
        with pytest.raises(ValueError):
            validate_config(config)

    def test_patchcore_layers_follow_model(self):
        config = copy.deepcopy(_config())
        config["feature_extraction"]["model"] = "inceptionv3"
        config["method"]["type"] = "patchcore"
        config["method"]["patchcore"]["layers"] = ["Mixed_5d", "Mixed_6e"]
        validate_config(config)
