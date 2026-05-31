"""Pipeline Configuration

Loads and strictly validates the YAML config that drives ``scripts/run_pipeline.py``.

The pipeline is a script-level orchestrator (not an ``src/experiments/*`` experiment),
so its config schema and validator live here, alongside the driver. The schema moves
every value that used to be hardcoded in ``run_pipeline.py`` (phase flags, runner/infra
settings, seeds, the global classifier overrides, and the baseline/fine-tune variant
matrix) into ``configs/pipeline.yaml`` so changing *what runs* is a config edit, not a
code change.

Strict validation: all parameters must be explicitly specified, mirroring the
``src/experiments/*/config.py`` convention (raises ``ValueError`` / ``KeyError``).
"""

import logging
import math
import os
from typing import Any, Dict, List

from src.utils.cli import dot_notation_to_dict, infer_type, validate_override_keys
from src.utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)

# The five phase flags (former RUN_* module globals).
PHASE_KEYS = (
    "data_preparation",
    "baseline_classifier",
    "ft_classifier",
    "evaluation",
    "summarize",
)

# Override keys the driver injects per job; they must not appear in any variant's
# `overrides` map. The learning rate is set via `ft.depths[].learning_rate` instead.
DRIVER_OWNED_KEYS = frozenset(
    {
        "compute.seed",
        "output.base_dir",
        "mode",
        "evaluation.checkpoint",
        "training.optimizer.learning_rate",
    }
)


def load_pipeline_config(path: str) -> Dict[str, Any]:
    """Load, validate, and normalize the pipeline config.

    After this returns, ``config["seeds"]`` is always the canonical ``{"list": [...]}``
    form (a ``range`` spec is expanded to an explicit int list), so downstream code
    only ever reads ``config["seeds"]["list"]``.

    Args:
        path: Path to the pipeline YAML config.

    Returns:
        The validated, normalized config dictionary.

    Raises:
        ValueError: If the configuration is invalid.
        KeyError: If required fields are missing.
    """
    config = load_config(path)
    validate_pipeline_config(config)
    config["seeds"] = {"list": resolve_seeds(config["seeds"])}
    _preflight_override_keys(config)
    return config


def resolve_seeds(seeds: Dict[str, Any]) -> List[int]:
    """Expand a validated ``seeds`` spec into an explicit list of ints.

    Args:
        seeds: A ``{"range": [start, stop[, step]]}`` or ``{"list": [...]}`` mapping
            (assumed already validated by :func:`validate_pipeline_config`).

    Returns:
        The resolved list of seed integers.
    """
    if "range" in seeds:
        return list(range(*seeds["range"]))
    return list(seeds["list"])


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """Validate the pipeline configuration.

    Args:
        config: Configuration dictionary to validate.

    As a side effect, scientific-notation numbers that arrive as *strings* (e.g. the
    YAML 1.1 literal ``1e-4``, which PyYAML does not parse as a float) — both in override
    maps and in ``ft.depths[].learning_rate`` — are normalized in place to floats, so the
    value validated here is identical to the one ``serialize_override`` later sends to the
    container. Strings the user quoted to force a string type (``"0"``, ``"true"``) are
    left untouched.

    Raises:
        ValueError: If the configuration is invalid.
        KeyError: If required fields are missing.
    """
    if config.get("experiment") != "pipeline":
        raise ValueError(
            f"Invalid experiment type: {config.get('experiment')}. Must be 'pipeline'"
        )

    _validate_phases(config)
    _validate_runner(config)
    _validate_configs_section(config)
    _validate_docker(config)
    _validate_seeds(config)
    _validate_classifier_overrides(config)
    _validate_baselines(config)
    _validate_ft(config)
    _validate_summarize(config)


def _require(config: Dict[str, Any], key: str) -> Any:
    """Return ``config[key]`` or raise a KeyError naming the missing key."""
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    # `bool` is a subclass of `int`; exclude it so True/False is never a valid int here.
    return isinstance(value, int) and not isinstance(value, bool)


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_phases(config: Dict[str, Any]) -> None:
    phases = _require(config, "phases")
    if not isinstance(phases, dict):
        raise ValueError("'phases' must be a dictionary")
    for key in PHASE_KEYS:
        if key not in phases:
            raise KeyError(f"Missing required phase flag: phases.{key}")
        if not _is_bool(phases[key]):
            raise ValueError(f"phases.{key} must be a boolean")
    unexpected = set(phases) - set(PHASE_KEYS)
    if unexpected:
        raise ValueError(f"Unknown phase flags: {sorted(unexpected)}")


def _validate_runner(config: Dict[str, Any]) -> None:
    runner = _require(config, "runner")
    if not isinstance(runner, dict):
        raise ValueError("'runner' must be a dictionary")

    for key in ("skip_completed", "delete_checkpoints_after_eval"):
        if key not in runner:
            raise KeyError(f"Missing required field: runner.{key}")
        if not _is_bool(runner[key]):
            raise ValueError(f"runner.{key} must be a boolean")

    if "gpu_cooldown_seconds" not in runner:
        raise KeyError("Missing required field: runner.gpu_cooldown_seconds")
    if (
        not _is_int(runner["gpu_cooldown_seconds"])
        or runner["gpu_cooldown_seconds"] < 0
    ):
        raise ValueError("runner.gpu_cooldown_seconds must be an integer >= 0")

    for key in ("classifier_notify_every", "classifier_max_parallel"):
        if key not in runner:
            raise KeyError(f"Missing required field: runner.{key}")
        if not _is_int(runner[key]) or runner[key] < 1:
            raise ValueError(f"runner.{key} must be an integer >= 1")

    if not _non_empty_str(runner.get("classifier_output_root")):
        raise ValueError("runner.classifier_output_root must be a non-empty string")


def _validate_configs_section(config: Dict[str, Any]) -> None:
    configs = _require(config, "configs")
    if not isinstance(configs, dict):
        raise ValueError("'configs' must be a dictionary")
    # The classifier config is always consumed (classifier phase + override preflight);
    # the data_preparation config is only consumed when that phase runs. Require the file
    # to exist for any config that will actually be used, so a typo'd path fails fast here
    # instead of deep inside a Docker container after other phases may have run.
    phases = config.get("phases", {})
    file_required = {
        "data_preparation": bool(phases.get("data_preparation")),
        "classifier": True,
    }
    for key in ("data_preparation", "classifier"):
        if key not in configs:
            raise KeyError(f"Missing required field: configs.{key}")
        value = configs[key]
        if not _non_empty_str(value):
            raise ValueError(f"configs.{key} must be a non-empty string")
        if not value.endswith(".yaml"):
            raise ValueError(f"configs.{key} must end with '.yaml'")
        if file_required[key] and not os.path.isfile(value):
            raise ValueError(f"configs.{key} points to a non-existent file: {value}")


def _validate_docker(config: Dict[str, Any]) -> None:
    docker = _require(config, "docker")
    if not isinstance(docker, dict):
        raise ValueError("'docker' must be a dictionary")
    for key in ("image", "shm_size"):
        if key not in docker:
            raise KeyError(f"Missing required field: docker.{key}")
        if not _non_empty_str(docker[key]):
            raise ValueError(f"docker.{key} must be a non-empty string")


def _validate_seeds(config: Dict[str, Any]) -> None:
    seeds = _require(config, "seeds")
    if not isinstance(seeds, dict):
        raise ValueError("'seeds' must be a dictionary")
    present = set(seeds) & {"range", "list"}
    if len(present) != 1:
        raise ValueError(
            "'seeds' must specify exactly one of 'range' or 'list', "
            f"got {sorted(seeds) or 'neither'}"
        )

    if "range" in seeds:
        spec = seeds["range"]
        if not isinstance(spec, list) or len(spec) not in (2, 3):
            raise ValueError(
                "seeds.range must be a list of [start, stop] or [start, stop, step]"
            )
        if not all(_is_int(x) for x in spec):
            raise ValueError("seeds.range values must be integers")
        start, stop = spec[0], spec[1]
        step = spec[2] if len(spec) == 3 else 1
        if step == 0:
            raise ValueError("seeds.range step must be non-zero")
        if stop <= start:
            raise ValueError(
                f"seeds.range stop ({stop}) must be greater than start ({start})"
            )
        resolved = list(range(start, stop, step))
    else:
        spec = seeds["list"]
        if not isinstance(spec, list) or len(spec) == 0:
            raise ValueError("seeds.list must be a non-empty list")
        if not all(_is_int(x) for x in spec):
            raise ValueError("seeds.list values must be integers")
        resolved = list(spec)

    if not resolved:
        raise ValueError("'seeds' resolves to an empty set")
    if any(s < 0 for s in resolved):
        raise ValueError("seed values must be non-negative")


def _is_yaml_unparsed_number(value: str) -> bool:
    """True if ``value`` is a scientific-notation number YAML 1.1 leaves as a string.

    PyYAML's 1.1 resolver natively parses plain ints, decimal floats, bools, ``null``,
    and lists; the one numeric form it leaves unparsed is scientific notation without an
    explicit decimal point/sign (e.g. ``1e-4``). Those are the only string overrides we
    coerce — so a value the user intentionally quoted to *force* a string (e.g. ``"0"``
    or ``"true"`` for a string-typed config key) is preserved as-is, not silently re-typed.
    """
    if "e" not in value and "E" not in value:
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _validate_override_map(name: str, mapping: Any) -> None:
    """Validate a dot-notation override map; reject driver-owned keys.

    Scientific-notation numeric strings that PyYAML leaves unparsed (e.g. the YAML 1.1
    literal ``1e-4``) are normalized in place to floats via ``infer_type`` so they later
    serialize as numbers, not quoted strings. Other strings are left untouched — any value
    quoted to force a string type (``"0"``, ``"true"``) keeps that type. See
    :func:`_is_yaml_unparsed_number`.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a dictionary")
    for key, value in mapping.items():
        if not _non_empty_str(key):
            raise ValueError(f"{name} keys must be non-empty strings")
        if key in DRIVER_OWNED_KEYS:
            raise ValueError(
                f"{name}.{key} is driver-owned and must not be set as an override "
                "(the pipeline injects it per job)"
            )
        if isinstance(value, dict):
            raise ValueError(
                f"{name}.{key} must be a scalar, list, or null (use flat dot-notation "
                "keys, not nested dicts)"
            )
        if isinstance(value, str) and _is_yaml_unparsed_number(value):
            mapping[key] = infer_type(value)


def _validate_classifier_overrides(config: Dict[str, Any]) -> None:
    overrides = _require(config, "classifier_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("'classifier_overrides' must be a dictionary")
    for key in ("checkpoint", "runtime"):
        if key not in overrides:
            raise KeyError(f"Missing required field: classifier_overrides.{key}")
        _validate_override_map(f"classifier_overrides.{key}", overrides[key])


def _validate_variant_list(name: str, variants: Any) -> List[str]:
    """Validate a list of ``{name, overrides}`` variants; return their names."""
    if not isinstance(variants, list) or len(variants) == 0:
        raise ValueError(f"'{name}' must be a non-empty list")
    names: List[str] = []
    for i, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{name}[{i}] must be a dict with 'name' and 'overrides'")
        if not _non_empty_str(variant.get("name")):
            raise ValueError(f"{name}[{i}].name must be a non-empty string")
        if "overrides" not in variant:
            raise KeyError(
                f"{name}[{variant['name']}] missing required field: overrides"
            )
        _validate_override_map(
            f"{name}[{variant['name']}].overrides", variant["overrides"]
        )
        names.append(variant["name"])
    if len(set(names)) != len(names):
        raise ValueError(f"'{name}' names must be unique, got {names}")
    return names


def _validate_baselines(config: Dict[str, Any]) -> None:
    _validate_variant_list("baselines", _require(config, "baselines"))


def _coerce_positive_float(name: str, value: Any) -> float:
    """Validate and return ``value`` as a positive float.

    Numeric strings (e.g. the YAML 1.1 scientific-notation literal ``"1e-4"``) are parsed
    via ``infer_type``. Returning the parsed float lets the caller store the normalized
    value back so it serializes as a number, not a quoted string.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    if isinstance(value, (int, float)):
        parsed: Any = value
    elif isinstance(value, str):
        parsed = infer_type(value)
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise ValueError(f"{name} must be a positive number, got {value!r}")
    else:
        raise ValueError(f"{name} must be a positive number")
    if float(parsed) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return float(parsed)


def _validate_ft(config: Dict[str, Any]) -> None:
    ft = _require(config, "ft")
    if not isinstance(ft, dict):
        raise ValueError("'ft' must be a dictionary")

    if "common" not in ft:
        raise KeyError("Missing required field: ft.common")
    _validate_override_map("ft.common", ft["common"])

    depths = _require(ft, "depths")
    if not isinstance(depths, list) or len(depths) == 0:
        raise ValueError("'ft.depths' must be a non-empty list")
    depth_names: List[str] = []
    for i, depth in enumerate(depths):
        if not isinstance(depth, dict):
            raise ValueError(f"ft.depths[{i}] must be a dict")
        if not _non_empty_str(depth.get("name")):
            raise ValueError(f"ft.depths[{i}].name must be a non-empty string")
        if "learning_rate" not in depth:
            raise KeyError(
                f"ft.depths[{depth['name']}] missing required field: learning_rate"
            )
        depth["learning_rate"] = _coerce_positive_float(
            f"ft.depths[{depth['name']}].learning_rate", depth["learning_rate"]
        )
        if "overrides" not in depth:
            raise KeyError(
                f"ft.depths[{depth['name']}] missing required field: overrides"
            )
        _validate_override_map(
            f"ft.depths[{depth['name']}].overrides", depth["overrides"]
        )
        depth_names.append(depth["name"])
    if len(set(depth_names)) != len(depth_names):
        raise ValueError(f"ft.depths names must be unique, got {depth_names}")

    _validate_variant_list("ft.balancing", _require(ft, "balancing"))


def _validate_summarize(config: Dict[str, Any]) -> None:
    summarize = _require(config, "summarize")
    if not isinstance(summarize, dict):
        raise ValueError("'summarize' must be a dictionary")
    for key in ("base_dir", "output_dir", "baseline_name"):
        if key not in summarize:
            raise KeyError(f"Missing required field: summarize.{key}")
        if not _non_empty_str(summarize[key]):
            raise ValueError(f"summarize.{key} must be a non-empty string")

    # The summarize step reads classifier reports from summarize.base_dir, but the
    # classifier jobs write them under runner.classifier_output_root. If a classifier
    # phase runs in the same pipeline, the two must point at the same directory or
    # summarize would aggregate a stale/empty location after the GPU work completes.
    # (When no classifier phase runs, base_dir may legitimately be a pre-existing dir
    # distinct from classifier_output_root, so the check is scoped to that case.)
    phases = config.get("phases", {})
    classifier_phase = phases.get("baseline_classifier") or phases.get("ft_classifier")
    if phases.get("summarize") and classifier_phase:
        output_root = config["runner"]["classifier_output_root"]
        if summarize["base_dir"] != output_root:
            raise ValueError(
                f"summarize.base_dir '{summarize['base_dir']}' must equal "
                f"runner.classifier_output_root '{output_root}' when a classifier phase "
                "runs in the same pipeline (summarize reads the directory the classifier "
                "jobs write to)"
            )

    # The summarize step compares every variant against summarize.baseline_name, which
    # must therefore be one of the baseline experiment names produced by the pipeline.
    # Enforce this only when both phases run (otherwise the baseline may be pre-existing).
    baseline_names = {
        b["name"] for b in config.get("baselines", []) if isinstance(b, dict)
    }
    if phases.get("summarize") and phases.get("baseline_classifier"):
        if summarize["baseline_name"] not in baseline_names:
            raise ValueError(
                f"summarize.baseline_name '{summarize['baseline_name']}' must be one of "
                f"the baseline names {sorted(baseline_names)}"
            )
    elif phases.get("summarize") and summarize["baseline_name"] not in baseline_names:
        logger.warning(
            "summarize.baseline_name '%s' is not among the configured baselines %s; "
            "it must already exist under summarize.base_dir",
            summarize["baseline_name"],
            sorted(baseline_names),
        )


def _preflight_override_keys(config: Dict[str, Any]) -> None:
    """Best-effort typo check of override keys against the classifier base config.

    Warns (never raises) so a transient/missing base config does not block the run; the
    authoritative key validation still happens inside ``src.main`` in each container.
    """
    classifier_config_path = config["configs"]["classifier"]
    try:
        base = load_config(classifier_config_path)
    except Exception as e:  # noqa: BLE001 - preflight must never block the pipeline
        logger.warning(
            "Skipping override-key preflight: could not load %s (%s)",
            classifier_config_path,
            e,
        )
        return

    maps: List[Dict[str, Any]] = [
        config["classifier_overrides"]["checkpoint"],
        config["classifier_overrides"]["runtime"],
        config["ft"]["common"],
    ]
    maps += [b["overrides"] for b in config["baselines"]]
    maps += [d["overrides"] for d in config["ft"]["depths"]]
    maps += [b["overrides"] for b in config["ft"]["balancing"]]

    nested: Dict[str, Any] = {}
    for flat in maps:
        for key, value in flat.items():
            nested = merge_configs(nested, dot_notation_to_dict(key, value))

    try:
        validate_override_keys(base, nested)
    except Exception as e:  # noqa: BLE001 - warn-only preflight
        logger.warning(
            "Override-key preflight against %s: %s", classifier_config_path, e
        )
