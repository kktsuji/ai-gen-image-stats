"""Anomaly Detection Configuration

Strict validation for the one-class anomaly-detection experiment: all
parameters must be explicitly specified in the config file (no implicit
defaults). Detectors are fit on normal images only and scored on the binary
test fold of an extended split file (see ``splits.py``).
"""

from typing import Any, Dict, List

from src.utils.config import (
    validate_compute_section,
    validate_experiment_section,
    validate_output_section,
)

VALID_FEATURE_MODELS = ["inceptionv3", "resnet50"]
VALID_NORMAL_POOLS = ["all", "suspicious"]
VALID_METHODS = ["knn", "mahalanobis", "patchcore"]
VALID_MODES = ["run", "adapt"]
VALID_ADAPTATION_METHODS = ["msc"]
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Intermediate layers that PatchCore may tap, per backbone.
PATCH_LAYERS: Dict[str, List[str]] = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "inceptionv3": [
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
    ],
}


def _require(section: Dict[str, Any], key: str, prefix: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required field: {prefix}.{key}")
    return section[key]


def _require_section(config: Dict[str, Any], key: str, prefix: str = "") -> Dict:
    name = f"{prefix}.{key}" if prefix else key
    if key not in config:
        raise KeyError(f"Missing required config key: {name}")
    if not isinstance(config[key], dict):
        raise ValueError(f"{name} must be a mapping")
    return config[key]


def _check_positive_int(value: Any, name: str, allow_zero: bool = False) -> None:
    low = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        kind = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {kind} integer")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate an anomaly-detection configuration.

    Raises:
        KeyError: If required fields are missing.
        ValueError: If values are invalid.
    """
    validate_experiment_section(config, "anomaly_detection", VALID_MODES)
    validate_compute_section(config)
    _validate_logging_section(config)
    _validate_data_section(config)
    _validate_feature_extraction_section(config)
    if config["mode"] == "adapt":
        # Representation adaptation (adapt.py): no detector, no threshold.
        validate_output_section(
            config, required_subdirs=["logs", "checkpoints", "reports"]
        )
        if config["feature_extraction"]["checkpoint"] is not None:
            raise ValueError(
                "feature_extraction.checkpoint must be null in adapt mode "
                "(adaptation starts from the ImageNet weights)"
            )
        _validate_adaptation_section(config)
        return
    validate_output_section(config, required_subdirs=["logs", "reports"])
    _validate_method_section(config)
    _validate_threshold_section(config)


def _validate_logging_section(config: Dict[str, Any]) -> None:
    log = _require_section(config, "logging")
    for key in ("console_level", "file_level"):
        level = _require(log, key, "logging")
        if level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid logging.{key}: '{level}'. Must be one of {VALID_LOG_LEVELS}"
            )


def _validate_data_section(config: Dict[str, Any]) -> None:
    data = _require_section(config, "data")
    split_file = _require(data, "split_file", "data")
    if not isinstance(split_file, str) or not split_file:
        raise ValueError("data.split_file must be a non-empty string")
    pool = _require(data, "normal_pool", "data")
    if pool not in VALID_NORMAL_POOLS:
        raise ValueError(
            f"Invalid data.normal_pool: '{pool}'. Must be one of {VALID_NORMAL_POOLS}"
        )


def _validate_feature_extraction_section(config: Dict[str, Any]) -> None:
    fe = _require_section(config, "feature_extraction")
    model = _require(fe, "model", "feature_extraction")
    if model not in VALID_FEATURE_MODELS:
        raise ValueError(
            f"Invalid feature_extraction.model: '{model}'. "
            f"Must be one of {VALID_FEATURE_MODELS}"
        )
    for key in ("image_size", "crop_size", "batch_size"):
        _check_positive_int(
            _require(fe, key, "feature_extraction"), f"feature_extraction.{key}"
        )
    _check_positive_int(
        _require(fe, "num_workers", "feature_extraction"),
        "feature_extraction.num_workers",
        allow_zero=True,
    )
    if fe["crop_size"] > fe["image_size"]:
        raise ValueError("feature_extraction.crop_size must be <= image_size")
    cache_dir = _require(fe, "cache_dir", "feature_extraction")
    if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir):
        raise ValueError(
            "feature_extraction.cache_dir must be a non-empty string or null"
        )
    checkpoint = _require(fe, "checkpoint", "feature_extraction")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError(
            "feature_extraction.checkpoint must be a non-empty string or null"
        )


def _validate_method_section(config: Dict[str, Any]) -> None:
    method = _require_section(config, "method")
    method_type = _require(method, "type", "method")
    if method_type not in VALID_METHODS:
        raise ValueError(
            f"Invalid method.type: '{method_type}'. Must be one of {VALID_METHODS}"
        )
    params = _require_section(method, method_type, "method")
    prefix = f"method.{method_type}"

    if method_type == "knn":
        _check_positive_int(_require(params, "k", prefix), f"{prefix}.k")

    elif method_type == "mahalanobis":
        shrinkage = _require(params, "shrinkage", prefix)
        if shrinkage != "ledoit_wolf":
            raise ValueError(f"{prefix}.shrinkage must be 'ledoit_wolf'")

    elif method_type == "patchcore":
        model = config["feature_extraction"]["model"]
        layers = _require(params, "layers", prefix)
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"{prefix}.layers must be a non-empty list")
        for layer in layers:
            if layer not in PATCH_LAYERS[model]:
                raise ValueError(
                    f"Invalid {prefix}.layers entry '{layer}' for model '{model}'. "
                    f"Must be one of {PATCH_LAYERS[model]}"
                )
        if len(set(layers)) != len(layers):
            raise ValueError(f"{prefix}.layers must not contain duplicates")
        for key in ("grid_size", "feature_dim", "projection_dim"):
            _check_positive_int(_require(params, key, prefix), f"{prefix}.{key}")
        ratio = _require(params, "coreset_ratio", prefix)
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0 < ratio <= 1
        ):
            raise ValueError(f"{prefix}.coreset_ratio must be in (0, 1]")


def _validate_threshold_section(config: Dict[str, Any]) -> None:
    threshold = _require_section(config, "threshold")
    pct = _require(threshold, "normal_percentile", "threshold")
    if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not 0 < pct < 100:
        raise ValueError("threshold.normal_percentile must be in (0, 100)")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _validate_adaptation_section(config: Dict[str, Any]) -> None:
    """``adaptation`` (mode adapt): Mean-Shifted Contrastive training settings."""
    ad = _require_section(config, "adaptation")
    method = _require(ad, "method", "adaptation")
    if method not in VALID_ADAPTATION_METHODS:
        raise ValueError(
            f"Invalid adaptation.method: '{method}'. "
            f"Must be one of {VALID_ADAPTATION_METHODS}"
        )
    layers = _require(ad, "trainable_layers", "adaptation")
    if (
        not isinstance(layers, list)
        or not layers
        or not all(isinstance(x, str) and x for x in layers)
    ):
        raise ValueError(
            "adaptation.trainable_layers must be a non-empty list of name patterns"
        )
    _check_positive_int(_require(ad, "epochs", "adaptation"), "adaptation.epochs")
    batch_size = _require(ad, "batch_size", "adaptation")
    _check_positive_int(batch_size, "adaptation.batch_size")
    if batch_size < 2:
        raise ValueError("adaptation.batch_size must be at least 2 (contrastive)")
    for key in ("learning_rate", "temperature"):
        value = _require(ad, key, "adaptation")
        if not _is_number(value) or value <= 0:
            raise ValueError(f"adaptation.{key} must be a positive number")
    if not isinstance(_require(ad, "update_frozen_bn_stats", "adaptation"), bool):
        raise ValueError("adaptation.update_frozen_bn_stats must be a boolean")
    momentum = _require(ad, "momentum", "adaptation")
    if not _is_number(momentum) or not 0 <= momentum < 1:
        raise ValueError("adaptation.momentum must be in [0, 1)")
    weight_decay = _require(ad, "weight_decay", "adaptation")
    if not _is_number(weight_decay) or weight_decay < 0:
        raise ValueError("adaptation.weight_decay must be a non-negative number")

    aug = _require_section(ad, "augmentation", "adaptation")
    scale = _require(aug, "crop_scale", "adaptation.augmentation")
    if (
        not isinstance(scale, list)
        or len(scale) != 2
        or not all(_is_number(x) for x in scale)
        or not 0 < scale[0] <= scale[1] <= 1
    ):
        raise ValueError(
            "adaptation.augmentation.crop_scale must be [min, max] "
            "with 0 < min <= max <= 1"
        )
    for key in ("horizontal_flip", "vertical_flip", "rotate90"):
        if not isinstance(_require(aug, key, "adaptation.augmentation"), bool):
            raise ValueError(f"adaptation.augmentation.{key} must be a boolean")
    blur = _require(aug, "blur_probability", "adaptation.augmentation")
    if not _is_number(blur) or not 0 <= blur <= 1:
        raise ValueError("adaptation.augmentation.blur_probability must be in [0, 1]")
