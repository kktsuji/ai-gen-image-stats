"""Classifier Configuration

This module provides configuration validation for classifier experiments.
Strict validation: all parameters must be explicitly specified in the config file.
"""

from typing import Any, Dict, Optional

from src.utils.config import (
    validate_checkpointing_section,
    validate_compute_section,
    validate_data_loading_section,
    validate_experiment_section,
    validate_optimizer_section,
    validate_output_section,
    validate_scheduler_section,
    validate_split_file,
    validate_training_epochs,
)

# Note: validate_config() is defined later in this file.


def get_model_specific_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration overrides.

    Args:
        model_name: Name of the model (resnet50, resnet101, resnet152, inceptionv3)

    Returns:
        Dictionary containing model-specific configuration overrides

    Raises:
        ValueError: If model_name is not supported

    Example:
        >>> config = get_model_specific_config("inceptionv3")
        >>> print(config["data"]["image_size"])
        299
    """
    model_configs = {
        "resnet50": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "resnet101": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "resnet152": {
            "data": {
                "image_size": 256,
                "crop_size": 224,
                "normalize": "imagenet",
            },
        },
        "inceptionv3": {
            "data": {
                "image_size": 320,  # Resize to slightly larger
                "crop_size": 299,  # InceptionV3 expects 299x299
                "normalize": "imagenet",
            },
            "model": {
                "dropout": 0.5,
            },
        },
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Must be one of {list(model_configs.keys())}"
        )

    return model_configs[model_name]


def _is_non_negative_number(value: Any) -> bool:
    """True for a real, non-negative int/float. Rejects bool (a subclass of int)."""
    return (
        not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0
    )


def validate_loss_section(loss: Any, num_classes: int) -> None:
    """Validate the optional model.loss section (strict when present).

    Args:
        loss: The model.loss config mapping.
        num_classes: Number of classes (used to validate focal alpha length).

    Raises:
        KeyError: If a required type-specific field is missing.
        ValueError: If a value is invalid.
    """
    if not isinstance(loss, dict):
        raise ValueError("model.loss must be a mapping")

    if "type" not in loss:
        raise KeyError("Missing required field: model.loss.type")

    loss_type = loss["type"]
    valid_types = ["cross_entropy", "focal", "class_balanced"]
    if loss_type not in valid_types:
        raise ValueError(
            f"Invalid model.loss.type: {loss_type}. Must be one of {valid_types}"
        )

    # Strict: reject unexpected keys so typos (e.g. "alphaa") fail instead of being
    # silently ignored, consistent with the project's no-implicit-config convention.
    allowed_keys = {
        "cross_entropy": {"type"},
        "focal": {"type", "gamma", "alpha"},
        "class_balanced": {"type", "beta", "base", "gamma"},
    }
    unexpected = set(loss) - allowed_keys[loss_type]
    if unexpected:
        raise ValueError(
            f"Unexpected model.loss fields for {loss_type}: {sorted(unexpected)}"
        )

    if loss_type == "focal":
        if "gamma" not in loss:
            raise KeyError("Missing required field: model.loss.gamma (focal)")
        if not _is_non_negative_number(loss["gamma"]):
            raise ValueError("model.loss.gamma must be a non-negative number")
        alpha = loss.get("alpha")
        if alpha is not None:
            if not isinstance(alpha, list) or len(alpha) != num_classes:
                raise ValueError(
                    "model.loss.alpha must be null or a list of length num_classes"
                )
            # Reject bool (a subclass of int) so e.g. alpha=[true, false] fails.
            if not all(
                not isinstance(a, bool) and isinstance(a, (int, float)) for a in alpha
            ):
                raise ValueError("model.loss.alpha entries must be numbers")

    elif loss_type == "class_balanced":
        if "beta" not in loss:
            raise KeyError("Missing required field: model.loss.beta (class_balanced)")
        beta = loss["beta"]
        # Open interval (0, 1): the reused compute_effective_num_weights rejects
        # beta <= 0, so accepting 0 here would pass validation then crash at model
        # build time. Cui et al. use beta in {0.9, 0.99, 0.999, 0.9999}.
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not (0 < beta < 1)
        ):
            raise ValueError("model.loss.beta must be a number in (0, 1)")
        if "base" not in loss:
            raise KeyError("Missing required field: model.loss.base (class_balanced)")
        if loss["base"] not in ["cross_entropy", "focal"]:
            raise ValueError("model.loss.base must be 'cross_entropy' or 'focal'")
        if loss["base"] == "focal":
            if "gamma" not in loss:
                raise KeyError(
                    "Missing required field: model.loss.gamma (class_balanced base=focal)"
                )
            if not _is_non_negative_number(loss["gamma"]):
                raise ValueError("model.loss.gamma must be a non-negative number")


def _is_positive_number(value: Any) -> bool:
    """True for a real, positive int/float. Rejects bool (a subclass of int)."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def validate_balancing_section(
    balancing: Any, num_classes: Optional[int] = None
) -> None:
    """Validate the optional data.balancing section (strict when present).

    Mirrors the strict style of ``validate_loss_section``: only the strategies
    the classifier actually consumes are allowed (weighted_sampler,
    downsampling, upsampling), and unexpected keys are rejected so typos fail
    loudly instead of being silently ignored.

    The up/downsampling strategies are multi-class aware (see
    ``src.utils.data.balancing``); ``target_ratio`` is the ratio of each
    minority class to the largest (upsampling) or smallest (downsampling) class.

    Args:
        balancing: The data.balancing config mapping.
        num_classes: Number of classes. When provided, ``manual_weights`` must
            supply exactly one weight per class; otherwise the dataloader maps
            weights by position (``{i: w for i, w in enumerate(...)}``) and a
            too-short list would pass validation then KeyError at train time on
            the first sample of an unweighted class.

    Raises:
        KeyError: If a required field is missing.
        ValueError: If a value is invalid or an unexpected key is present.
    """
    if not isinstance(balancing, dict):
        raise ValueError("data.balancing must be a mapping")

    valid_strategies = {"weighted_sampler", "downsampling", "upsampling"}
    unexpected = set(balancing) - valid_strategies
    if unexpected:
        raise ValueError(
            f"Unexpected data.balancing keys: {sorted(unexpected)}. "
            f"Allowed strategies: {sorted(valid_strategies)}"
        )

    # --- weighted_sampler ---
    if "weighted_sampler" in balancing:
        ws = balancing["weighted_sampler"]
        if not isinstance(ws, dict):
            raise ValueError("data.balancing.weighted_sampler must be a mapping")
        allowed = {
            "enabled",
            "method",
            "beta",
            "manual_weights",
            "replacement",
            "num_samples",
        }
        unexpected = set(ws) - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected data.balancing.weighted_sampler keys: {sorted(unexpected)}"
            )
        if "enabled" in ws and not isinstance(ws["enabled"], bool):
            raise ValueError(
                "data.balancing.weighted_sampler.enabled must be a boolean"
            )

        # The dataloader hard-reads ws["method"] when the sampler is enabled, so an
        # enabled sampler without a method would pass validation then KeyError at
        # train time. Require it here so the failure is a clear config error.
        if ws.get("enabled") and ws.get("method") is None:
            raise KeyError(
                "Missing required field: data.balancing.weighted_sampler.method "
                "(required when weighted_sampler.enabled is true)"
            )

        valid_methods = ["inverse_frequency", "effective_num", "manual"]
        if ws.get("method") is not None and ws["method"] not in valid_methods:
            raise ValueError(
                f"data.balancing.weighted_sampler.method must be one of "
                f"{valid_methods}, got {ws['method']!r}"
            )
        if ws.get("method") == "manual":
            mw = ws.get("manual_weights")
            if not isinstance(mw, list) or len(mw) == 0:
                raise ValueError(
                    "data.balancing.weighted_sampler.manual_weights must be a "
                    "non-empty list when method='manual'"
                )
            if not all(_is_positive_number(w) for w in mw):
                raise ValueError(
                    "data.balancing.weighted_sampler.manual_weights must contain "
                    "only positive numbers"
                )
            # The dataloader maps weights by position (one per class index), so a
            # list that doesn't cover every class KeyErrors at train time. Require
            # an exact per-class match when num_classes is known.
            if num_classes is not None and len(mw) != num_classes:
                raise ValueError(
                    "data.balancing.weighted_sampler.manual_weights must have one "
                    f"weight per class (expected {num_classes}, got {len(mw)})"
                )
        if ws.get("beta") is not None:
            beta = ws["beta"]
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not (0 < beta < 1)
            ):
                raise ValueError(
                    "data.balancing.weighted_sampler.beta must be a number in (0, 1)"
                )
        if "replacement" in ws and not isinstance(ws["replacement"], bool):
            raise ValueError(
                "data.balancing.weighted_sampler.replacement must be a boolean"
            )
        if ws.get("num_samples") is not None:
            ns = ws["num_samples"]
            if isinstance(ns, bool) or not isinstance(ns, int) or ns < 1:
                raise ValueError(
                    "data.balancing.weighted_sampler.num_samples must be a "
                    "positive integer or null"
                )

    # --- downsampling / upsampling (same shape) ---
    for strategy in ("downsampling", "upsampling"):
        if strategy in balancing:
            section = balancing[strategy]
            if not isinstance(section, dict):
                raise ValueError(f"data.balancing.{strategy} must be a mapping")
            unexpected = set(section) - {"enabled", "target_ratio"}
            if unexpected:
                raise ValueError(
                    f"Unexpected data.balancing.{strategy} keys: {sorted(unexpected)}"
                )
            if "enabled" in section and not isinstance(section["enabled"], bool):
                raise ValueError(f"data.balancing.{strategy}.enabled must be a boolean")
            if section.get("target_ratio") is not None:
                tr = section["target_ratio"]
                if not _is_positive_number(tr) or tr > 1.0:
                    raise ValueError(
                        f"data.balancing.{strategy}.target_ratio must be a positive "
                        "number in (0, 1.0]"
                    )


def validate_positive_class(positive_class: Any, num_classes: int) -> None:
    """Validate the optional data.positive_class index.

    Args:
        positive_class: The configured positive/abnormal class index.
        num_classes: Number of classes (positive_class must be a valid index).

    Raises:
        ValueError: If positive_class is not an integer in [0, num_classes).
    """
    if (
        isinstance(positive_class, bool)
        or not isinstance(positive_class, int)
        or not (0 <= positive_class < num_classes)
    ):
        raise ValueError(
            f"data.positive_class must be an integer in [0, {num_classes}), "
            f"got {positive_class!r}"
        )


def validate_contrast_class(contrast_class: Any, num_classes: int) -> None:
    """Validate the optional data.contrast_class index.

    The contrast class is the single negative class the hard-core direct
    evaluation ranks the positive class against (e.g. "suspicious"). See
    :mod:`src.experiments.classifier.hard_core`.

    Args:
        contrast_class: The configured contrast (suspicious) class index.
        num_classes: Number of classes (contrast_class must be a valid index).

    Raises:
        ValueError: If contrast_class is not an integer in [0, num_classes).
    """
    if (
        isinstance(contrast_class, bool)
        or not isinstance(contrast_class, int)
        or not (0 <= contrast_class < num_classes)
    ):
        raise ValueError(
            f"data.contrast_class must be an integer in [0, {num_classes}), "
            f"got {contrast_class!r}"
        )


def validate_config(config: Dict[str, Any]) -> None:
    """Validate classifier configuration.

    Checks that all required fields are present and have valid values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    # Check required top-level keys
    required_keys = ["experiment", "mode", "compute", "model", "data", "output"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    # Validate experiment type and mode
    validate_experiment_section(config, "classifier", ["train", "evaluate"])

    # Validate compute configuration
    validate_compute_section(config)

    # Validate model configuration
    model = config["model"]
    if "architecture" not in model:
        raise KeyError("Missing required field: model.architecture")

    architecture = model["architecture"]
    required_arch_fields = ["name", "num_classes"]
    for field in required_arch_fields:
        if field not in architecture:
            raise KeyError(f"Missing required field: model.architecture.{field}")
        if architecture[field] is None:
            raise ValueError(f"model.architecture.{field} cannot be None")

    valid_models = ["resnet50", "resnet101", "resnet152", "inceptionv3"]
    if architecture["name"] not in valid_models:
        raise ValueError(
            f"Invalid model name: {architecture['name']}. Must be one of {valid_models}"
        )

    if (
        not isinstance(architecture["num_classes"], int)
        or architecture["num_classes"] < 1
    ):
        raise ValueError("num_classes must be a positive integer")

    # Validate initialization
    if "initialization" in model:
        init = model["initialization"]
        if "pretrained" in init and not isinstance(init["pretrained"], bool):
            raise ValueError("initialization.pretrained must be a boolean")
        if "freeze_backbone" in init and not isinstance(init["freeze_backbone"], bool):
            raise ValueError("initialization.freeze_backbone must be a boolean")

    # Validate loss configuration (optional; defaults to cross_entropy when absent
    # for backward compatibility). When present, it is validated strictly.
    if "loss" in model:
        validate_loss_section(model["loss"], architecture["num_classes"])

    # Validate data configuration
    data = config["data"]
    required_data_sections = ["split_file", "loading", "preprocessing", "augmentation"]
    for section in required_data_sections:
        if section not in data:
            raise KeyError(f"Missing required field: data.{section}")

    # Validate split_file
    validate_split_file(data)

    # Validate loading
    validate_data_loading_section(data)

    # Validate preprocessing
    preprocessing = data["preprocessing"]
    required_preprocessing_fields = ["image_size", "crop_size"]
    for field in required_preprocessing_fields:
        if field not in preprocessing:
            raise KeyError(f"Missing required field: data.preprocessing.{field}")

    if (
        not isinstance(preprocessing["image_size"], int)
        or preprocessing["image_size"] < 1
    ):
        raise ValueError("image_size must be a positive integer")

    if (
        not isinstance(preprocessing["crop_size"], int)
        or preprocessing["crop_size"] < 1
    ):
        raise ValueError("crop_size must be a positive integer")

    valid_normalize = ["imagenet", "cifar10", "custom", None]
    if preprocessing.get("normalize") not in valid_normalize:
        raise ValueError(
            f"Invalid normalize option: {preprocessing.get('normalize')}. "
            f"Must be one of {valid_normalize}"
        )

    # Validate optional balancing section (strict when present)
    if "balancing" in data:
        validate_balancing_section(data["balancing"], architecture["num_classes"])

    # Validate optional positive/abnormal class index (defaults to 1 when absent
    # for backward compatibility with the binary task).
    if "positive_class" in data:
        validate_positive_class(data["positive_class"], architecture["num_classes"])

    # Validate optional hard-core contrast (suspicious) class. The index, when
    # set, must be a valid class; the name is used to auto-detect the index from
    # the split's class metadata when the index is omitted.
    if "contrast_class" in data:
        validate_contrast_class(data["contrast_class"], architecture["num_classes"])
    if "contrast_class_name" in data and not isinstance(
        data["contrast_class_name"], str
    ):
        raise ValueError("data.contrast_class_name must be a string")

    # Validate output configuration
    validate_output_section(config)

    valid_modes = ["train", "evaluate"]
    if config["mode"] not in valid_modes:
        raise ValueError(
            f"Invalid mode: {config['mode']}. Must be one of {valid_modes}"
        )

    # Mode-specific validation
    if config["mode"] == "train":
        if "training" not in config:
            raise KeyError(
                "Missing required section: training (required for train mode)"
            )

        training = config["training"]
        required_training_fields = ["epochs", "optimizer", "scheduler"]
        for field in required_training_fields:
            if field not in training:
                raise KeyError(f"Missing required field: training.{field}")

        validate_training_epochs(training)

        # Validate optimizer
        optimizer = training["optimizer"]
        if "type" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.type")
        if "learning_rate" not in optimizer:
            raise KeyError("Missing required field: training.optimizer.learning_rate")

        validate_optimizer_section(optimizer, valid_types=["adam", "adamw", "sgd"])

        # Validate scheduler
        validate_scheduler_section(training["scheduler"])

        # Validate checkpointing
        if "checkpointing" in training:
            validate_checkpointing_section(training["checkpointing"])

        # Validate validation section (required in train mode; the training path
        # assumes it is fully specified — no implicit defaults).
        if "validation" not in training:
            raise KeyError("Missing required field: training.validation")
        validation = training["validation"]
        if not isinstance(validation, dict):
            raise ValueError("training.validation must be a mapping")

        for field in ("enabled", "metric", "frequency", "early_stopping_patience"):
            if field not in validation:
                raise KeyError(f"Missing required field: training.validation.{field}")

        if not isinstance(validation["enabled"], bool):
            raise ValueError("training.validation.enabled must be a boolean")

        metric = validation["metric"]
        valid_metrics = [
            "accuracy",
            "loss",
            "balanced_accuracy",
            "f1_macro",
            "f1_1",
        ]
        # The early-stopping F1 metric must track the configured positive class,
        # not the hardcoded class 1. For a multi-class run with an explicit
        # data.positive_class, allow selecting that class's F1 (e.g. "f1_2").
        num_classes = architecture["num_classes"]
        positive_class = data.get("positive_class", 1 if num_classes <= 2 else None)
        if positive_class is not None and f"f1_{positive_class}" not in valid_metrics:
            valid_metrics.append(f"f1_{positive_class}")
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid training.validation.metric: {metric}. "
                f"Must be one of {valid_metrics}"
            )

        frequency = validation["frequency"]
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or frequency < 1
        ):
            raise ValueError("training.validation.frequency must be a positive integer")

        patience = validation["early_stopping_patience"]
        if patience is not None and (
            isinstance(patience, bool) or not isinstance(patience, int) or patience < 1
        ):
            raise ValueError(
                "training.validation.early_stopping_patience must be null "
                "or a positive integer"
            )

        # Validate synthetic_augmentation (optional, train-mode only)
        syn_aug = data.get("synthetic_augmentation", {})
        if syn_aug.get("enabled"):
            if (
                not isinstance(syn_aug.get("split_file"), str)
                or not syn_aug["split_file"]
            ):
                raise ValueError(
                    "data.synthetic_augmentation.split_file must be a non-empty string "
                    "when synthetic_augmentation is enabled"
                )

            # Mutual exclusion: balancing and synthetic_augmentation
            balancing = data.get("balancing", {})
            if any(
                balancing.get(k, {}).get("enabled")
                for k in ("weighted_sampler", "downsampling", "upsampling")
            ):
                raise ValueError(
                    "Cannot use both data.balancing and "
                    "data.synthetic_augmentation. "
                    "Disable one of them."
                )

            # Augmentation mode: "augment" (default) concatenates synthetic data
            # onto the full real training set; "replace_minority" drops the real
            # samples of replace_class first (TSTR: train on synthetic minority,
            # keep the real majority, test on real).
            aug_mode = syn_aug.get("mode", "augment")
            valid_aug_modes = ["augment", "replace_minority"]
            if aug_mode not in valid_aug_modes:
                raise ValueError(
                    f"Invalid synthetic_augmentation.mode: {aug_mode}. "
                    f"Must be one of {valid_aug_modes}"
                )
            if aug_mode == "replace_minority":
                replace_class = syn_aug.get("replace_class")
                if (
                    isinstance(replace_class, bool)
                    or not isinstance(replace_class, int)
                    or replace_class < 0
                ):
                    raise ValueError(
                        "synthetic_augmentation.replace_class must be a "
                        "non-negative integer when mode is 'replace_minority'"
                    )

            limit = syn_aug.get("limit", {})
            limit_mode = limit.get("mode")
            valid_limit_modes = [None, "max_ratio", "max_samples"]
            if limit_mode not in valid_limit_modes:
                raise ValueError(
                    f"Invalid synthetic_augmentation limit.mode: {limit_mode}. "
                    f"Must be one of {valid_limit_modes}"
                )

            if limit_mode == "max_ratio":
                if "max_ratio" not in limit:
                    raise ValueError(
                        "synthetic_augmentation limit.max_ratio is required "
                        "when limit.mode is 'max_ratio'"
                    )
                max_ratio = limit["max_ratio"]
                if (
                    isinstance(max_ratio, bool)
                    or not isinstance(max_ratio, (int, float))
                    or max_ratio <= 0
                ):
                    raise ValueError(
                        "synthetic_augmentation limit.max_ratio must be a positive number"
                    )

            if limit_mode == "max_samples":
                if "max_samples" not in limit:
                    raise ValueError(
                        "synthetic_augmentation limit.max_samples is required "
                        "when limit.mode is 'max_samples'"
                    )
                max_samples = limit["max_samples"]
                if (
                    isinstance(max_samples, bool)
                    or not isinstance(max_samples, int)
                    or max_samples <= 0
                ):
                    raise ValueError(
                        "synthetic_augmentation limit.max_samples must be a positive integer"
                    )

    elif config["mode"] == "evaluate":
        if "evaluation" not in config:
            raise KeyError(
                "Missing required section: evaluation (required for evaluate mode)"
            )

        evaluation = config["evaluation"]
        checkpoint = evaluation.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("evaluation.checkpoint is required for evaluate mode")

        # Validate optional evaluation split (defaults to "val")
        if "split" in evaluation:
            eval_split = evaluation["split"]
            if eval_split not in ("train", "val", "test"):
                raise ValueError(
                    f"evaluation.split must be one of 'train', 'val', 'test'; "
                    f"got {eval_split!r}"
                )

        # Validate reports subdir for evaluate mode
        subdirs = config.get("output", {}).get("subdirs", {})
        if "reports" not in subdirs or subdirs["reports"] is None:
            raise ValueError("output.subdirs.reports is required for evaluate mode")

        # Validate bootstrap configuration (required in evaluate mode)
        if "bootstrap" not in evaluation:
            raise KeyError(
                "Missing required field: evaluation.bootstrap "
                "(must be explicitly specified in evaluate mode)"
            )
        bootstrap = evaluation["bootstrap"]
        if not isinstance(bootstrap, dict):
            raise ValueError("evaluation.bootstrap must be a mapping")

        for field in ("enabled", "n_bootstrap", "confidence_level", "save_predictions"):
            if field not in bootstrap:
                raise KeyError(f"Missing required field: evaluation.bootstrap.{field}")

        if not isinstance(bootstrap["enabled"], bool):
            raise ValueError("evaluation.bootstrap.enabled must be a boolean")

        if not isinstance(bootstrap["save_predictions"], bool):
            raise ValueError("evaluation.bootstrap.save_predictions must be a boolean")

        n_bootstrap = bootstrap["n_bootstrap"]
        if (
            isinstance(n_bootstrap, bool)
            or not isinstance(n_bootstrap, int)
            or n_bootstrap < 1
        ):
            raise ValueError(
                "evaluation.bootstrap.n_bootstrap must be a positive integer"
            )

        confidence_level = bootstrap["confidence_level"]
        if (
            isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float))
            or confidence_level <= 0
            or confidence_level >= 1
        ):
            raise ValueError(
                "evaluation.bootstrap.confidence_level must be a number in (0, 1)"
            )
