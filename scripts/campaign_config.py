"""Campaign Configuration

Loads, strictly validates and expands the ``configs/sweep.yaml`` of a research
campaign (see README "Organizing Experiments: Series and Campaigns").

A sweep combines one base experiment config with named override bundles along
one or more *axes*. The Cartesian product of the axis values gives the
conditions, and each condition is run on every split and seed::

    base_config: base.yaml               # relative to the campaign's configs/
    family: ad-frozen                    # runs/split{N}/<family>/<name>/seed{S}
    done_marker: reports/evaluation.json # relative to a run's output dir
    path_keys: [data.split_file, feature_extraction.cache_dir]
    splits:
      key: data.split_file
      template: "../shared/splits/cv_binary_ad_split{split}.json"
      indices: [0, 1, 2]
    seeds:
      key: compute.seed
      values: [0]
    name_template: "ad-{method}-{backbone}__{pool}"
    axes:
      method:
        knn: {method.type: knn}
        mahalanobis: {method.type: mahalanobis}
      ...

Every path in the base config and the sweep is written relative to the campaign
folder. The values of ``path_keys`` (and ``output.base_dir``, which the driver
sets) are resolved against the campaign folder only at run time, so the stored
per-run configs stay portable when the series is moved. A ``path_keys`` value
may contain ``{split}``, which is replaced by the split index of each run (e.g.
a per-split checkpoint ``runs/split{split}/clf/x/seed0/checkpoints/best.pth``).
The ``splits.template`` must contain ``{split}`` and may also name axes, e.g.
``../shared/splits-kctc/cv_binary_ad_split{split}_k{k}_d{draw}.json``: each
axis field is replaced by the condition's value name on that axis, so one sweep
can read a different split file per condition.

A campaign may hold several sweep files named ``sweep*.yaml`` (e.g. a training
stage ``sweep-train.yaml`` and a stage that uses its outputs, ``sweep.yaml``);
``load_campaign`` takes the file name. Condition names must be unique across
the sweep files, because the stored per-run configs are keyed by condition
name only; ``load_campaign`` rejects a name that another sweep file also uses.

Strict validation, mirroring ``scripts/pipeline_config.py``: every field is
required and override keys must exist in the base config.
"""

import copy
import itertools
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.utils.cli import dot_notation_to_dict, validate_override_keys
from src.utils.config import load_config, merge_configs

SWEEP_FILE = Path("configs") / "sweep.yaml"
SWEEP_GLOB = "sweep*.yaml"
REQUIRED_KEYS = (
    "base_config",
    "family",
    "done_marker",
    "path_keys",
    "splits",
    "seeds",
    "name_template",
    "axes",
)


@dataclass(frozen=True)
class Run:
    """One expanded run of a campaign."""

    name: str
    split: int
    seed: int
    config: Dict[str, Any]  # campaign-relative paths

    @property
    def output_dir(self) -> str:
        """Run output directory, relative to the campaign folder."""
        return str(Path(self.config["output"]["base_dir"]))

    @property
    def config_path(self) -> Path:
        """Where the per-run config is stored, relative to the campaign folder."""
        return (
            Path("configs")
            / "runs"
            / self.name
            / f"split{self.split}_seed{self.seed}.yaml"
        )


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping")
    if key not in mapping:
        name = f"{where}.{key}" if where else key
        raise KeyError(f"Missing required field: {name}")
    return mapping[key]


def _non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int_list(value: Any, name: str) -> List[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value)
    ):
        raise ValueError(f"{name} must be a non-empty list of non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _get_dotted(config: Dict[str, Any], key: str) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Key '{key}' does not exist in the base config")
        node = node[part]
    return node


def _set_dotted(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    return merge_configs(config, dot_notation_to_dict(key, value))


def _format_fields(template: str, name: str) -> set:
    """Field names of a ``str.format`` template; only plain ``{name}`` fields."""
    fields = set()
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        raise ValueError(f"{name} is not a valid template: {e}") from e
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise ValueError(
                f"{name}: only plain named fields like '{{split}}' are allowed, "
                f"got '{{{field}{'!' + conversion if conversion else ''}"
                f"{':' + spec if spec else ''}}}'"
            )
        fields.add(field)
    return fields


def validate_sweep(sweep: Dict[str, Any], base: Dict[str, Any]) -> None:
    """Validate a sweep definition against its base config.

    Raises:
        KeyError: If a required field or a referenced config key is missing.
        ValueError: If a value is invalid.
    """
    for key in REQUIRED_KEYS:
        _require(sweep, key, "sweep")

    _non_empty_str(sweep["base_config"], "sweep.base_config")
    _non_empty_str(sweep["family"], "sweep.family")
    _non_empty_str(sweep["done_marker"], "sweep.done_marker")

    splits = sweep["splits"]
    split_key = _non_empty_str(
        _require(splits, "key", "sweep.splits"), "sweep.splits.key"
    )
    template = _non_empty_str(
        _require(splits, "template", "sweep.splits"), "sweep.splits.template"
    )
    template_fields = _format_fields(template, "sweep.splits.template")
    if "split" not in template_fields:
        raise ValueError("sweep.splits.template must contain '{split}'")
    _int_list(_require(splits, "indices", "sweep.splits"), "sweep.splits.indices")

    seeds = sweep["seeds"]
    seed_key = _non_empty_str(_require(seeds, "key", "sweep.seeds"), "sweep.seeds.key")
    _int_list(_require(seeds, "values", "sweep.seeds"), "sweep.seeds.values")

    path_keys = sweep["path_keys"]
    if not isinstance(path_keys, list) or not all(
        isinstance(k, str) and k for k in path_keys
    ):
        raise ValueError("sweep.path_keys must be a list of non-empty strings")
    for key in (split_key, seed_key, "output.base_dir", *path_keys):
        _get_dotted(base, key)

    axes = sweep["axes"]
    if not isinstance(axes, dict) or not axes:
        raise ValueError("sweep.axes must be a non-empty mapping")
    driver_keys = {split_key, seed_key, "output.base_dir"}
    if "split" in axes:
        # 'split' is the split-index field of splits.template.
        raise ValueError("sweep.axes: 'split' is reserved and cannot be an axis name")
    for axis, values in axes.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"sweep.axes.{axis} must be a non-empty mapping")
        for value_name, overrides in values.items():
            where = f"sweep.axes.{axis}.{value_name}"
            if not isinstance(value_name, str) or not value_name:
                raise ValueError(f"{where}: value names must be non-empty strings")
            if not isinstance(overrides, dict):
                raise ValueError(f"{where} must be a mapping of dotted keys")
            for key, val in overrides.items():
                if key in driver_keys:
                    raise ValueError(f"{where}: '{key}' is set by the driver")
                validate_override_keys(base, dot_notation_to_dict(key, val))

    unknown = template_fields - {"split"} - set(axes)
    if unknown:
        raise ValueError(
            f"sweep.splits.template fields {sorted(unknown)} are neither 'split' "
            f"nor an axis ({sorted(axes)})"
        )

    name_template = _non_empty_str(sweep["name_template"], "sweep.name_template")
    fields = {f for _, f, _, _ in string.Formatter().parse(name_template) if f}
    if fields != set(axes):
        raise ValueError(
            f"sweep.name_template fields {sorted(fields)} must match the axes "
            f"{sorted(axes)}"
        )


def expand_runs(sweep: Dict[str, Any], base: Dict[str, Any]) -> List[Run]:
    """Expand a validated sweep into runs (conditions × splits × seeds).

    Conditions follow the axis order of the sweep file; within a condition the
    runs are ordered by split, then seed.
    """
    axes = sweep["axes"]
    axis_names = list(axes)
    runs: List[Run] = []
    names = set()
    for combo in itertools.product(*(list(axes[a].items()) for a in axis_names)):
        name = sweep["name_template"].format(
            **{axis: value_name for axis, (value_name, _) in zip(axis_names, combo)}
        )
        if name in names:
            raise ValueError(f"Duplicate condition name '{name}'")
        names.add(name)
        value_names = {
            axis: value_name for axis, (value_name, _) in zip(axis_names, combo)
        }
        condition = copy.deepcopy(base)
        for _, overrides in combo:
            for key, val in overrides.items():
                condition = _set_dotted(condition, key, copy.deepcopy(val))
        for split in sweep["splits"]["indices"]:
            for seed in sweep["seeds"]["values"]:
                cfg = _set_dotted(
                    condition,
                    sweep["splits"]["key"],
                    sweep["splits"]["template"].format(split=split, **value_names),
                )
                cfg = _set_dotted(cfg, sweep["seeds"]["key"], seed)
                for key in sweep["path_keys"]:
                    value = _get_dotted(cfg, key)
                    if isinstance(value, str) and "{split}" in value:
                        cfg = _set_dotted(
                            cfg, key, value.replace("{split}", str(split))
                        )
                cfg = _set_dotted(
                    cfg,
                    "output.base_dir",
                    f"runs/split{split}/{sweep['family']}/{name}/seed{seed}",
                )
                runs.append(
                    Run(name=name, split=split, seed=seed, config=copy.deepcopy(cfg))
                )
    return runs


def condition_names(sweep: Dict[str, Any]) -> List[str]:
    """Condition names of a sweep (``name_template`` over the axis value names)."""
    axes = sweep["axes"]
    axis_names = list(axes)
    return [
        sweep["name_template"].format(**dict(zip(axis_names, combo)))
        for combo in itertools.product(*(list(axes[a]) for a in axis_names))
    ]


def _check_names_unique_across_sweeps(
    configs_dir: Path, sweep_file: str, names: List[str]
) -> None:
    """Reject condition names that another ``sweep*.yaml`` also defines.

    Two sweeps sharing a name would share ``configs/runs/<name>/`` (one stage
    overwrites the other's record of what ran) and, with the same family, the
    run folders too (one stage's done markers would skip the other's runs).
    """
    own = set(names)
    for other in sorted(configs_dir.glob(SWEEP_GLOB)):
        if other.name == sweep_file:
            continue
        sweep = load_config(other)
        try:
            other_names = condition_names(sweep)
        except (KeyError, TypeError, AttributeError, ValueError, IndexError) as e:
            raise ValueError(
                f"Cannot read the condition names of {other} to check them "
                f"against {sweep_file}: {e!r}"
            ) from e
        shared = sorted(own.intersection(other_names))
        if shared:
            raise ValueError(
                f"Condition names {shared} are defined in both {sweep_file} and "
                f"{other.name}; names must be unique across a campaign's sweep files"
            )


def load_campaign(
    campaign_dir: Path, sweep_file: str = SWEEP_FILE.name
) -> tuple[Dict[str, Any], List[Run]]:
    """Load ``<campaign>/configs/<sweep_file>`` and its base config, validate, expand.

    Returns:
        (sweep, runs)
    """
    if Path(sweep_file).name != sweep_file or not Path(sweep_file).match(SWEEP_GLOB):
        raise ValueError(
            f"Sweep file '{sweep_file}' must be a file name matching {SWEEP_GLOB} "
            "in the campaign's configs/"
        )
    sweep_path = campaign_dir / SWEEP_FILE.parent / sweep_file
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")
    sweep = load_config(sweep_path)
    base_path = sweep_path.parent / _require(sweep, "base_config", "sweep")
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    base = load_config(base_path)
    validate_sweep(sweep, base)
    runs = expand_runs(sweep, base)
    _check_names_unique_across_sweeps(
        sweep_path.parent, sweep_file, sorted({r.name for r in runs})
    )
    return sweep, runs


def resolve_paths(
    config: Dict[str, Any], path_keys: List[str], campaign_dir: Path
) -> Dict[str, Any]:
    """Return a copy with campaign-relative path values made absolute.

    ``output.base_dir`` is always resolved; ``None`` values and absolute paths
    are left unchanged.
    """
    resolved = copy.deepcopy(config)
    for key in ("output.base_dir", *path_keys):
        value = _get_dotted(resolved, key)
        if value is None or Path(value).is_absolute():
            continue
        absolute = str((campaign_dir / value).resolve())
        resolved = _set_dotted(resolved, key, absolute)
    return resolved
