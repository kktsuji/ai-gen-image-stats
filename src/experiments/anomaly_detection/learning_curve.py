"""Anomaly Detection - Learning Curve over k Labelled Abnormal Images

Analyses a campaign whose arms are trained on splits keeping only ``k``
abnormal training images (``abnormal_subsample_split.py``), several random
draws per ``k``: a classifier, and one-class detectors on that classifier's
features. Configured by ``<campaign>/configs/learning_curve.yaml``.

The unit is the split, as in ``compare``: per split, an arm's value at ``k``
is the mean over the draws, and a split enters only when every draw of that
``k`` has finished (a partial mean would mix draws unevenly). ``k = all`` is
one run per split on the unsubsampled split (``full``).

Pre-specified test families, each BH-corrected separately:

- ``vs_classifier``: each detector at each ``k`` (and ``all``) against the
  classifier arm at the same ``k``, paired by split.
- ``vs_chance``: each arm at each ``k`` against the per-split chance level of
  PR-AUC (the abnormal fraction of the test fold), as in ``compare``.

Descriptive criteria (fixed in the config before running):

- per detector, the smallest ``k`` from which the mean difference to the
  classifier stays >= ``-criteria.ad_margin`` for that and every larger ``k``;
- the smallest ``k`` from which the classifier's mean stays >=
  ``criteria.classifier_target``.

``zero_references`` add ``k = 0`` points per detector (runs of other trees, e.g.
frozen ImageNet features): shown in the table and the figure, not tested.

Outputs to ``<campaign>/reports/``: ``learning_curve.csv``,
``learning_curve_comparisons.csv``, ``learning_curve.md`` and
``learning_curve.png``.

Run:
    python -m src.experiments.anomaly_detection.learning_curve work/<series>/<campaign>
"""

import argparse
import logging
import math
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.anomaly_detection.compare import (
    SplitValues,
    chance_per_split,
    check_checkpoints,
    correct_within_families,
    load_split_values,
    paired_row,
)
from src.experiments.classifier.cross_split_report import _t_ci
from src.utils.config import load_config

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("configs") / "learning_curve.yaml"
FULL = "all"  # the k label of the unsubsampled arm
K = Union[int, str]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping")
    if key not in mapping:
        raise KeyError(f"Missing required field: {where}.{key}")
    return mapping[key]


def _non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int_list(value: Any, name: str, minimum: int) -> List[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= minimum
            for v in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{name} must be a non-empty list of distinct integers >= {minimum}"
        )
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low < value < high
    ):
        raise ValueError(f"{name} must be in ({low}, {high})")
    return float(value)


def validate_config(cfg: Dict[str, Any]) -> None:
    """Strictly validate a learning-curve config (all fields required)."""
    for key in (
        "metric",
        "alpha",
        "correction",
        "runs_base",
        "ks",
        "draws",
        "classifier_arm",
        "arms",
        "zero_references",
        "criteria",
    ):
        _require(cfg, key, "learning_curve")
    _non_empty_str(cfg["metric"], "learning_curve.metric")
    _number(cfg["alpha"], "learning_curve.alpha", 0, 1)
    if cfg["correction"] not in ("benjamini-hochberg", "bonferroni"):
        raise ValueError(
            "learning_curve.correction must be benjamini-hochberg or bonferroni"
        )
    _non_empty_str(cfg["runs_base"], "learning_curve.runs_base")
    _int_list(cfg["ks"], "learning_curve.ks", 1)
    _int_list(cfg["draws"], "learning_curve.draws", 0)

    arms = cfg["arms"]
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError(
            "learning_curve.arms must map at least two arm names "
            "(the classifier and one detector)"
        )
    for name, arm in arms.items():
        where = f"learning_curve.arms.{name}"
        for key in ("label", "family", "template", "full"):
            _require(arm, key, where)
        _non_empty_str(arm["label"], f"{where}.label")
        _non_empty_str(arm["family"], f"{where}.family")
        template = _non_empty_str(arm["template"], f"{where}.template")
        try:
            fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        except ValueError as e:
            raise ValueError(f"{where}.template is not a valid template: {e}") from e
        if fields != {"k", "draw"}:
            raise ValueError(
                f"{where}.template must use exactly the fields {{k}} and {{draw}}, "
                f"got {sorted(fields)}"
            )
        if arm["full"] is not None:
            _non_empty_str(arm["full"], f"{where}.full")
    if cfg["classifier_arm"] not in arms:
        raise ValueError(
            f"learning_curve.classifier_arm '{cfg['classifier_arm']}' is not an arm"
        )

    zero = cfg["zero_references"]
    if not isinstance(zero, dict):
        raise ValueError(
            "learning_curve.zero_references must be a mapping ({} for none)"
        )
    detectors = set(arms) - {cfg["classifier_arm"]}
    for name, refs in zero.items():
        if name not in detectors:
            raise ValueError(
                f"learning_curve.zero_references.{name} is not a detector arm "
                f"({sorted(detectors)})"
            )
        if not isinstance(refs, list) or not refs:
            raise ValueError(
                f"learning_curve.zero_references.{name} must be a non-empty list"
            )
        for i, ref in enumerate(refs):
            for key in ("label", "base_dir", "family", "experiment"):
                _non_empty_str(
                    _require(ref, key, f"learning_curve.zero_references.{name}[{i}]"),
                    f"learning_curve.zero_references.{name}[{i}].{key}",
                )

    criteria = cfg["criteria"]
    _number(
        _require(criteria, "ad_margin", "learning_curve.criteria"),
        "learning_curve.criteria.ad_margin",
        0,
        1,
    )
    _number(
        _require(criteria, "classifier_target", "learning_curve.criteria"),
        "learning_curve.criteria.classifier_target",
        0,
        1,
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def arm_values(
    values: Dict[str, SplitValues],
    arm: Dict[str, Any],
    ks: Sequence[int],
    draws: Sequence[int],
) -> Tuple[Dict[K, SplitValues], List[str]]:
    """Draw-averaged value per split for each ``k`` of one arm.

    Args:
        values: Seed-averaged metric per (experiment, split) of the arm's family.

    Returns:
        ``{k: {split: value}}`` (``k`` = ``"all"`` for the ``full`` run) and a
        list of notes on splits left out because a draw had not finished.
    """
    out: Dict[K, SplitValues] = {}
    notes: List[str] = []
    for k in ks:
        per_draw = [values.get(arm["template"].format(k=k, draw=d), {}) for d in draws]
        splits = set().union(*per_draw)
        complete = {s for s in splits if all(s in pd_ for pd_ in per_draw)}
        for s in sorted(splits - complete):
            missing = [d for d, pd_ in zip(draws, per_draw) if s not in pd_]
            notes.append(
                f"{arm['label']}, k={k}: split {s} left out (draws {missing} missing)"
            )
        if complete:
            out[k] = {
                s: float(np.mean([pd_[s] for pd_ in per_draw]))
                for s in sorted(complete)
            }
    if arm["full"] is not None and arm["full"] in values:
        out[FULL] = dict(values[arm["full"]])
    elif arm["full"] is not None:
        notes.append(f"{arm['label']}, k={FULL}: no '{arm['full']}' runs")
    return out, notes


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def k_order(ks: Sequence[int]) -> List[K]:
    return [*sorted(ks), FULL]


def curve_table(
    arms: Dict[str, Dict[str, Any]],
    curves: Dict[str, Dict[K, SplitValues]],
    zero: Dict[str, List[Tuple[str, SplitValues]]],
    ks: Sequence[int],
) -> pd.DataFrame:
    """Across-split mean, SD and 95% t-CI per (arm, k), plus the k=0 points."""
    rows = []
    for name, arm in arms.items():
        points: List[Tuple[K, str, SplitValues]] = [
            (0, label, vals) for label, vals in zero.get(name, [])
        ]
        points += [(k, "", curves[name][k]) for k in k_order(ks) if k in curves[name]]
        for k, ref_label, vals in points:
            arr = np.array([vals[s] for s in sorted(vals)], dtype=np.float64)
            mean, lo, hi = _t_ci(arr, 0.95)
            rows.append(
                {
                    "arm": name,
                    "label": arm["label"],
                    "k": str(k),
                    "reference": ref_label,
                    "n_splits": len(arr),
                    "mean": mean,
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def build_comparisons(
    cfg: Dict[str, Any],
    curves: Dict[str, Dict[K, SplitValues]],
    chance: SplitValues,
) -> pd.DataFrame:
    """The ``vs_classifier`` and ``vs_chance`` families, corrected separately."""
    clf = cfg["classifier_arm"]
    rows: List[Dict[str, Any]] = []
    for name in cfg["arms"]:
        for k in k_order(cfg["ks"]):
            if k not in curves[name]:
                continue
            treatment = f"{name}@k={k}"
            if name != clf and k in curves[clf]:
                row = paired_row(
                    "vs_classifier",
                    treatment,
                    f"{clf}@k={k}",
                    curves[name][k],
                    curves[clf][k],
                )
                if row:
                    rows.append(row)
            if chance:
                row = paired_row(
                    "vs_chance", treatment, "chance", curves[name][k], chance
                )
                if row:
                    rows.append(row)
    return correct_within_families(rows, cfg["alpha"], cfg["correction"])


def smallest_k_from(passes: Dict[K, bool], ks: Sequence[int]) -> Optional[K]:
    """Smallest k from which ``passes`` holds for it and every larger k.

    The check starts at the largest k that has a value, so an arm without a
    ``full`` run (``"all"`` absent) is judged on its k values. Below that, a
    k without a value (missing runs) ends the run of passing values, so the
    answer is never earlier than a gap. None if that largest k fails or no k
    has a value.
    """
    order = k_order(ks)
    while order and order[-1] not in passes:
        order.pop()
    answer: Optional[K] = None
    for k in reversed(order):
        if not passes.get(k, False):
            break
        answer = k
    return answer


def evaluate_criteria(
    cfg: Dict[str, Any], curves: Dict[str, Dict[K, SplitValues]]
) -> Dict[str, Optional[K]]:
    """The two pre-set descriptive criteria (see the module docstring)."""
    clf = cfg["classifier_arm"]
    margin = cfg["criteria"]["ad_margin"]
    out: Dict[str, Optional[K]] = {}
    for name in cfg["arms"]:
        if name == clf:
            continue
        passes = {}
        for k in k_order(cfg["ks"]):
            t, r = curves[name].get(k, {}), curves[clf].get(k, {})
            common = sorted(set(t) & set(r))
            if common:
                diff = float(np.mean([t[s] - r[s] for s in common]))
                passes[k] = diff >= -margin
        out[f"{name}_within_margin"] = smallest_k_from(passes, cfg["ks"])
    target = cfg["criteria"]["classifier_target"]
    out[f"{clf}_reaches_target"] = smallest_k_from(
        {k: float(np.mean(list(v.values()))) >= target for k, v in curves[clf].items()},
        cfg["ks"],
    )
    return out


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(x) else f"{x:.{digits}f}"


def write_markdown(
    path: Path,
    cfg: Dict[str, Any],
    table: pd.DataFrame,
    comparisons: pd.DataFrame,
    criteria: Dict[str, Optional[K]],
    chance: SplitValues,
    notes: List[str],
) -> None:
    metric = cfg["metric"]
    lines = [f"# Learning curve over k ({metric})", ""]
    if chance:
        lines += [
            f"Chance level (mean abnormal fraction of the test folds): {_fmt(float(np.mean(list(chance.values()))))}",
            "",
        ]
    lines += [
        "## Mean per arm and k",
        "",
        "| Arm | k | Reference | Splits | Mean | SD | 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table.to_dict(orient="records"):
        lines.append(
            f"| {r['label']} | {r['k']} | {r['reference'] or '—'} | {r['n_splits']} | "
            f"{_fmt(r['mean'])} | {_fmt(r['std'])} | {_fmt(r['ci_lower'])}–{_fmt(r['ci_upper'])} |"
        )
    lines += ["", "## Pre-set criteria", ""]
    margin = cfg["criteria"]["ad_margin"]
    target = cfg["criteria"]["classifier_target"]
    for key, k in criteria.items():
        if key.endswith("_within_margin"):
            arm = key.removesuffix("_within_margin")
            what = f"{cfg['arms'][arm]['label']}: smallest k from which the mean difference to the classifier stays >= -{margin}"
        else:
            arm = key.removesuffix("_reaches_target")
            what = f"{cfg['arms'][arm]['label']}: smallest k from which the mean stays >= {target}"
        lines.append(f"- {what}: **{'not reached' if k is None else k}**")
    titles = {
        "vs_classifier": "Detectors vs the classifier at the same k",
        "vs_chance": "Each arm vs chance",
    }
    for fam, title in titles.items():
        sub = (
            pd.DataFrame(comparisons[comparisons["family"] == fam])
            if not comparisons.empty
            else comparisons
        )
        if sub.empty:
            continue
        lines += [
            "",
            f"## {title} (`{fam}`, {cfg['correction']} within the family)",
            "",
            "| Treatment | Reference | Splits | Diff | Better | p (t, corr.) | p (Wilcoxon, corr.) | dz | Significant |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sub.to_dict(orient="records"):
            lines.append(
                f"| {r['treatment']} | {r['reference']} | {r['n_splits']} | {r['mean_diff']:+.3f} | "
                f"{r['n_treatment_better']}/{r['n_splits']} | {_fmt(r['p_ttest_corrected'], 4)} | "
                f"{_fmt(r['p_wilcoxon_corrected'], 4)} | {_fmt(r['cohens_dz'], 2)} | "
                f"{'yes' if r['significant'] else 'no'} |"
            )
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curve(
    path: Path, cfg: Dict[str, Any], table: pd.DataFrame, chance: SplitValues
) -> None:
    """Mean and 95% CI per arm over k (categorical axis; k = 0 at the left)."""
    labels = ["0", *[str(k) for k in k_order(cfg["ks"])]]
    pos = {k: i for i, k in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["s", "^", "v", "D", "P"]
    for name, arm in cfg["arms"].items():
        sub = pd.DataFrame(table[table["arm"] == name])
        curve = pd.DataFrame(sub[sub["reference"] == ""])
        x = [pos[k] for k in curve["k"]]
        line = ax.errorbar(
            x,
            curve["mean"],
            yerr=[curve["mean"] - curve["ci_lower"], curve["ci_upper"] - curve["mean"]],
            marker="o",
            capsize=3,
            label=arm["label"],
        )
        for i, r in enumerate(
            pd.DataFrame(sub[sub["reference"] != ""]).to_dict(orient="records")
        ):
            ax.scatter(
                pos["0"],
                r["mean"],
                marker=markers[i % len(markers)],
                color=line[0].get_color(),
                label=f"{arm['label']}, k=0: {r['reference']}",
            )
    if chance:
        ax.axhline(
            float(np.mean(list(chance.values()))),
            ls="--",
            color="k",
            lw=1,
            label="chance",
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("k (labelled abnormal training images)")
    ax.set_ylabel(f"{cfg['metric']} (mean over splits, 95% CI)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_learning_curve(campaign_dir: Path) -> Path:
    """Run the learning-curve analysis of a campaign; returns its reports dir."""
    cfg_path = campaign_dir / CONFIG_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"Learning-curve config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    validate_config(cfg)
    metric = cfg["metric"]
    runs_base = str(campaign_dir / cfg["runs_base"])

    curves: Dict[str, Dict[K, SplitValues]] = {}
    notes: List[str] = []
    for name, arm in cfg["arms"].items():
        check_checkpoints(runs_base, arm["family"])
        values = load_split_values(runs_base, arm["family"], metric)
        curves[name], arm_notes = arm_values(values, arm, cfg["ks"], cfg["draws"])
        notes += arm_notes
        if not curves[name]:
            raise ValueError(f"No finished runs for arm '{name}' under {runs_base}")

    zero: Dict[str, List[Tuple[str, SplitValues]]] = {}
    for name, refs in cfg["zero_references"].items():
        for ref in refs:
            base = str(campaign_dir / ref["base_dir"])
            # As compare does for references: refuse runs whose checkpoint changed.
            check_checkpoints(base, ref["family"], {ref["experiment"]})
            values = load_split_values(base, ref["family"], metric)
            if ref["experiment"] not in values:
                raise ValueError(
                    f"zero_references.{name}: no '{ref['experiment']}' results under {base}"
                )
            zero.setdefault(name, []).append((ref["label"], values[ref["experiment"]]))

    detector_family = next(
        a["family"] for n, a in cfg["arms"].items() if n != cfg["classifier_arm"]
    )
    chance = chance_per_split(runs_base, detector_family, metric)
    table = curve_table(cfg["arms"], curves, zero, cfg["ks"])
    comparisons = build_comparisons(cfg, curves, chance)
    criteria = evaluate_criteria(cfg, curves)
    for note in notes:
        logger.warning(note)

    reports = campaign_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "learning_curve.csv", index=False)
    comparisons.to_csv(reports / "learning_curve_comparisons.csv", index=False)
    write_markdown(
        reports / "learning_curve.md", cfg, table, comparisons, criteria, chance, notes
    )
    plot_curve(reports / "learning_curve.png", cfg, table, chance)
    logger.info(f"Learning-curve report written to {reports}")
    return reports


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Learning curve over k of a campaign")
    parser.add_argument("campaign_dir", help="Campaign folder, e.g. work/<series>/05-x")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_learning_curve(Path(args.campaign_dir))


if __name__ == "__main__":
    main()
