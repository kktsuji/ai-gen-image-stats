"""Anomaly Detection - Campaign Comparison Report

Paired-by-split analysis of an anomaly-detection campaign
(``<campaign>/configs/analysis.yaml``). Three pre-specified test families, each
BH-corrected separately:

- ``vs_chance``: each condition against the per-split chance level of PR-AUC
  (the fraction of abnormal images in the test fold).
- ``vs_classifier``: each condition against the external classifier reference
  arms of the same backbone (seed-averaged per split, as in cross_split_report).
- ``pool``: within each method x backbone, one normal pool against another.

Outputs to ``<campaign>/reports/``: ``summary.csv``, ``comparisons.csv``,
``subclass_auc.csv``, ``report.md`` and ``<metric>_by_condition.png``.

Run:
    python -m src.experiments.anomaly_detection.compare work/<series>/<campaign>
"""

import argparse
import logging
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.classifier.cross_split_report import (
    build_across_split_summary,
    compute_split_means,
    load_per_split_results,
)
from src.utils.config import load_config
from src.utils.statistical_testing import (
    apply_correction_with_nan,
    cohens_d_paired,
    paired_ttest,
    wilcoxon_signed_rank,
)

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

ANALYSIS_FILE = Path("configs") / "analysis.yaml"
SplitValues = Dict[int, float]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def validate_analysis_config(cfg: Dict[str, Any]) -> None:
    """Strictly validate an analysis config (all fields required)."""
    for key in (
        "metric",
        "alpha",
        "correction",
        "runs",
        "references",
        "reference_groups",
        "pool_contrast",
    ):
        if key not in cfg:
            raise KeyError(f"Missing required field: analysis.{key}")
    if not isinstance(cfg["metric"], str) or not cfg["metric"]:
        raise ValueError("analysis.metric must be a non-empty string")
    alpha = cfg["alpha"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < alpha < 1
    ):
        raise ValueError("analysis.alpha must be in (0, 1)")
    if cfg["correction"] not in ("benjamini-hochberg", "bonferroni"):
        raise ValueError("analysis.correction must be benjamini-hochberg or bonferroni")
    for key in ("base_dir", "family"):
        if key not in cfg["runs"]:
            raise KeyError(f"Missing required field: analysis.runs.{key}")
    refs = cfg["references"]
    if not isinstance(refs, dict):
        raise ValueError("analysis.references must be a mapping")
    for name, ref in refs.items():
        for key in ("base_dir", "family", "experiment"):
            if not isinstance(ref, dict) or key not in ref:
                raise KeyError(
                    f"Missing required field: analysis.references.{name}.{key}"
                )
    groups = cfg["reference_groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("analysis.reference_groups must be a non-empty mapping")
    for pattern, names in groups.items():
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) for n in names)
        ):
            raise ValueError(
                f"analysis.reference_groups['{pattern}'] must be a non-empty list "
                "of reference names"
            )
        unknown = [n for n in names if n not in refs]
        if unknown:
            raise ValueError(
                f"analysis.reference_groups['{pattern}'] references unknown {unknown}"
            )
    contrast = cfg["pool_contrast"]
    if not isinstance(contrast, dict):
        raise ValueError("analysis.pool_contrast must be a mapping")
    for key in ("treatment", "control"):
        if key not in contrast:
            raise KeyError(f"Missing required field: analysis.pool_contrast.{key}")
        if not isinstance(contrast[key], str) or not contrast[key]:
            raise ValueError(f"analysis.pool_contrast.{key} must be a non-empty string")
    if contrast["treatment"] == contrast["control"]:
        raise ValueError("analysis.pool_contrast treatment and control must differ")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_split_values(
    base_dir: str, family: str, metric: str
) -> Dict[str, SplitValues]:
    """Seed-averaged metric per (experiment, split) under ``base_dir``."""
    per_split = load_per_split_results(base_dir=base_dir, family=family)
    values: Dict[str, SplitValues] = {}
    for split, results in per_split.items():
        for exp, metrics in compute_split_means(results, [metric]).items():
            if metric in metrics:
                values.setdefault(exp, {})[split] = metrics[metric]
    return values


def _split_and_condition(report_file: str) -> Tuple[int, str]:
    """Split index and condition name of a run's report file.

    Run layout (see README "Series and Campaigns"):
    ``<runs_base>/split{N}/<family>/<condition>/seed{S}/reports/<file>``,
    so counting from the end: file (-1), reports (-2), seed (-3),
    condition (-4), family (-5), split{N} (-6).
    """
    parts = Path(report_file).parts
    return int(parts[-6].removeprefix("split")), parts[-4]


def _completed(paths: List[str]) -> List[str]:
    """Keep report files whose run finished (its ``evaluation.json`` exists).

    ``evaluation.json`` is written last, so this restricts every part of the
    report to the same run set as the main metric.
    """
    return [p for p in paths if (Path(p).parent / "evaluation.json").exists()]


def chance_per_split(
    runs_base: str, family: str, metric: str = "pr_auc"
) -> SplitValues:
    """Chance level of ``metric`` per split.

    - ``pr_auc``: the abnormal fraction of the binary test fold. All conditions
      share the test folds, so any finished run's ``predictions_test.npz``
      gives the targets for its split.
    - ``roc_auc``: 0.5 for every split with a finished run.
    - Any other metric has no defined chance level: returns ``{}``, so the
      vs-chance comparison is skipped.
    """
    paths = _completed(
        sorted(
            glob(f"{runs_base}/split*/{family}/*/seed*/reports/predictions_test.npz")
        )
    )
    chance: SplitValues = {}
    if metric not in ("pr_auc", "roc_auc"):
        logger.warning(
            f"No chance level defined for metric '{metric}'; skipping vs_chance"
        )
        return chance
    for path in paths:
        split, _ = _split_and_condition(path)
        if split in chance:
            continue
        if metric == "roc_auc":
            chance[split] = 0.5
            continue
        with np.load(path) as data:
            chance[split] = float(np.mean(data["targets"]))
    return chance


def subclass_auc(runs_base: str, family: str) -> pd.DataFrame:
    """ROC-AUC of abnormal vs each normal subclass, per condition (split-mean).

    Seeds are averaged within each split first, so ``n_splits`` counts splits
    and every split carries equal weight regardless of its number of seeds.
    0.5 means the detector cannot tell CTCs from that subclass; values below
    0.5 mean that subclass scores as *more* anomalous than the CTCs.
    """
    rows: List[Dict[str, Any]] = []
    for path in _completed(
        sorted(glob(f"{runs_base}/split*/{family}/*/seed*/reports/subclass_scores.csv"))
    ):
        split, condition = _split_and_condition(path)
        table = pd.read_csv(path)
        abnormal = table.loc[table["subclass"] == "abnormal", "score"].to_numpy()
        for subclass in sorted(set(table["subclass"]) - {"abnormal"}):
            normal = table.loc[table["subclass"] == subclass, "score"].to_numpy()
            if len(abnormal) == 0 or len(normal) == 0:
                continue
            y = np.r_[np.ones(len(abnormal)), np.zeros(len(normal))]
            auc = roc_auc_score(y, np.r_[abnormal, normal])
            rows.append(
                {
                    "condition": condition,
                    "split": split,
                    "subclass": subclass,
                    "auc": auc,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["condition", "subclass", "auc_mean", "n_splits"])
    df = pd.DataFrame(rows)
    per_split = df.groupby(["condition", "subclass", "split"], as_index=False).agg(
        auc=("auc", "mean")
    )
    return (
        per_split.groupby(["condition", "subclass"])["auc"]
        .agg(auc_mean="mean", n_splits="count")
        .reset_index()
    )


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def paired_row(
    family: str, treatment: str, reference: str, t: SplitValues, r: SplitValues
) -> Optional[Dict[str, Any]]:
    """Paired-by-split comparison of treatment vs reference (None if < 2 splits)."""
    common = sorted(set(t) & set(r))
    if len(common) < 2:
        return None
    tv = np.array([t[s] for s in common])
    rv = np.array([r[s] for s in common])
    t_stat, t_p = paired_ttest(rv, tv)
    _, w_p = wilcoxon_signed_rank(rv, tv)
    return {
        "family": family,
        "treatment": treatment,
        "reference": reference,
        "n_splits": len(common),
        "treatment_mean": float(tv.mean()),
        "reference_mean": float(rv.mean()),
        "mean_diff": float((tv - rv).mean()),
        "n_treatment_better": int(np.sum(tv > rv)),
        "t_statistic": t_stat,
        "p_value_ttest": t_p,
        "p_value_wilcoxon": w_p,
        "cohens_dz": cohens_d_paired(rv, tv),
    }


def correct_within_families(
    rows: List[Dict[str, Any]], alpha: float, method: str
) -> pd.DataFrame:
    """BH (or Bonferroni) correction within each family.

    As in cross_split_report, the paired t-test is primary; where it is
    undefined (zero-variance differences) the corrected Wilcoxon p decides.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_ttest_corrected"] = np.nan
    df["p_wilcoxon_corrected"] = np.nan
    for _, idx in df.groupby("family").groups.items():
        df.loc[idx, "p_ttest_corrected"] = apply_correction_with_nan(
            df.loc[idx, "p_value_ttest"].tolist(), method=method
        )
        df.loc[idx, "p_wilcoxon_corrected"] = apply_correction_with_nan(
            df.loc[idx, "p_value_wilcoxon"].tolist(), method=method
        )

    flags: List[bool] = []
    for row in df.to_dict(orient="records"):
        tc = float(row["p_ttest_corrected"])
        wc = float(row["p_wilcoxon_corrected"])
        if math.isfinite(tc):
            flags.append(tc < alpha)
        else:
            flags.append(math.isfinite(wc) and wc < alpha)
    df["significant"] = flags
    return df


def build_comparisons(
    cfg: Dict[str, Any],
    conditions: Dict[str, SplitValues],
    references: Dict[str, SplitValues],
    chance: SplitValues,
) -> pd.DataFrame:
    """All three comparison families, corrected within each family."""
    rows: List[Optional[Dict[str, Any]]] = []
    for name, values in sorted(conditions.items()):
        rows.append(paired_row("vs_chance", name, "chance", values, chance))
        for pattern, ref_names in cfg["reference_groups"].items():
            if pattern in name:
                for ref in ref_names:
                    rows.append(
                        paired_row("vs_classifier", name, ref, values, references[ref])
                    )

    treatment = cfg["pool_contrast"]["treatment"]
    control = cfg["pool_contrast"]["control"]
    for name in sorted(conditions):
        if name.endswith(f"__{treatment}"):
            other = name[: -len(treatment)] + control
            if other in conditions:
                rows.append(
                    paired_row("pool", name, other, conditions[name], conditions[other])
                )
    return correct_within_families(
        [r for r in rows if r is not None], cfg["alpha"], cfg["correction"]
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt(x: float, digits: int = 3) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    secondary: pd.DataFrame,
    comparisons: pd.DataFrame,
    references: Dict[str, SplitValues],
    chance: SplitValues,
    sub_auc: pd.DataFrame,
    metric: str,
) -> None:
    lines = ["# Campaign comparison report", ""]
    lines += [
        f"Metric: `{metric}`. Unit of analysis: the split.",
        (
            f"Chance level of `{metric}`: mean {_fmt(float(np.mean(list(chance.values()))))}"
            f" over {len(chance)} splits"
            + (" (abnormal fraction of the test fold)." if metric == "pr_auc" else ".")
            if chance
            else f"No chance level is defined for `{metric}`; the vs-chance comparison is skipped."
        ),
        "",
        "## Conditions",
        "",
        "| Condition | n | Mean | 95% CI | PR-AUC vs all normals |",
        "| --- | --- | --- | --- | --- |",
    ]
    sec: Dict[str, float] = {
        str(r["experiment"]): float(r["mean"])
        for r in secondary.to_dict(orient="records")
    }
    ordered = summary.sort_values(by="mean", ascending=False)
    for r in ordered.to_dict(orient="records"):
        lines.append(
            f"| `{r['experiment']}` | {r['n_splits']} | {_fmt(float(r['mean']))} | "
            f"[{_fmt(float(r['ci_lower']))}, {_fmt(float(r['ci_upper']))}] | "
            f"{_fmt(sec.get(str(r['experiment']), float('nan')))} |"
        )
    lines += [
        "",
        "## References",
        "",
        "| Reference | n | Mean |",
        "| --- | --- | --- |",
    ]
    for name, values in references.items():
        lines.append(
            f"| `{name}` | {len(values)} | {_fmt(float(np.mean(list(values.values()))))} |"
        )

    titles = {
        "vs_chance": "Against chance",
        "vs_classifier": "Against classifier baselines (same backbone)",
        "pool": "Normal pool: all vs suspicious",
    }
    for fam, title in titles.items():
        sub = (
            pd.DataFrame(comparisons.loc[comparisons["family"] == fam])
            if not comparisons.empty
            else comparisons
        )
        lines += [
            "",
            f"## {title}",
            "",
            "| Treatment | Reference | Δ mean | better / n | dz | p (t, corrected) | significant |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sub.to_dict(orient="records"):
            lines.append(
                f"| `{r['treatment']}` | `{r['reference']}` | {float(r['mean_diff']):+.3f} | "
                f"{r['n_treatment_better']}/{r['n_splits']} | "
                f"{_fmt(float(r['cohens_dz']), 2)} | "
                f"{_fmt(float(r['p_ttest_corrected']), 4)} | "
                f"{'yes' if bool(r['significant']) else 'no'} |"
            )

    if not sub_auc.empty:
        pivot = sub_auc.pivot(index="condition", columns="subclass", values="auc_mean")
        lines += [
            "",
            "## ROC-AUC of abnormal vs each normal subclass (split mean)",
            "",
            "Below 0.5: that subclass scores as more anomalous than the CTCs.",
            "",
            "| Condition | " + " | ".join(pivot.columns) + " |",
            "| --- | " + " | ".join("---" for _ in pivot.columns) + " |",
        ]
        for cond, row in pivot.iterrows():
            lines.append(f"| `{cond}` | " + " | ".join(_fmt(v) for v in row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_conditions(
    path: Path,
    summary: pd.DataFrame,
    references: Dict[str, SplitValues],
    reference_groups: Dict[str, List[str]],
    chance: float,
    metric: str = "pr_auc",
) -> None:
    """One panel per reference group: condition means with 95% CI and
    reference lines for chance and the classifier arms."""
    groups = list(reference_groups.items())
    if not groups:
        logger.warning("No reference groups; skipping the condition figure")
        return
    fig, axes = plt.subplots(
        1, len(groups), figsize=(6 * len(groups), 4.2), sharex=True, squeeze=False
    )
    for ax, (pattern, ref_names) in zip(axes[0], groups):
        mask = summary["experiment"].str.contains(pattern, regex=False)
        sub = pd.DataFrame(summary.loc[mask]).sort_values(by="experiment")
        y = np.arange(len(sub))
        ax.errorbar(
            sub["mean"],
            y,
            xerr=[sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]],
            fmt="o",
            color="#2f5d9c",
            ecolor="#2f5d9c",
            elinewidth=1.5,
            capsize=3,
            markersize=6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["experiment"], fontsize=9)
        ax.invert_yaxis()
        lines = [("chance", chance, ":")] if math.isfinite(chance) else []
        lines += [
            (n, float(np.mean(list(references[n].values()))), "--") for n in ref_names
        ]
        for label, x, style in lines:
            ax.axvline(x, color="#666666", linestyle=style, linewidth=1)
            ax.text(
                x,
                1.0,
                f" {label} {x:.3f}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#444444",
            )
        ax.set_title(f"Conditions matching '{pattern.strip('-_')}'", fontsize=10)
        ax.set_xlabel(f"{metric} (mean over splits, 95% CI)")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_compare(campaign_dir: Path) -> Path:
    """Run the comparison for a campaign; returns its reports directory."""
    cfg_path = campaign_dir / ANALYSIS_FILE
    if not cfg_path.exists():
        raise FileNotFoundError(f"Analysis config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    validate_analysis_config(cfg)
    metric = cfg["metric"]
    runs_base = str(campaign_dir / cfg["runs"]["base_dir"])
    family = cfg["runs"]["family"]

    conditions = load_split_values(runs_base, family, metric)
    if not conditions:
        raise ValueError(f"No runs found under {runs_base} (family {family})")
    references: Dict[str, SplitValues] = {}
    for name, ref in cfg["references"].items():
        # Campaign-relative like runs.base_dir; absolute paths are used as-is.
        ref_base = str(campaign_dir / ref["base_dir"])
        values = load_split_values(ref_base, ref["family"], metric)
        if ref["experiment"] not in values:
            raise ValueError(
                f"Reference {name}: no '{ref['experiment']}' results under {ref_base}"
            )
        references[name] = values[ref["experiment"]]
    chance = chance_per_split(runs_base, family, metric)

    split_means: Dict[int, Dict[str, Dict[str, float]]] = {}
    for exp, values in conditions.items():
        for split, v in values.items():
            split_means.setdefault(split, {})[exp] = {metric: v}
    summary = build_across_split_summary(split_means, [metric])

    secondary_values = load_split_values(runs_base, family, "pr_auc_vs_all_normals")
    sec_means: Dict[int, Dict[str, Dict[str, float]]] = {}
    for exp, values in secondary_values.items():
        for split, v in values.items():
            sec_means.setdefault(split, {})[exp] = {"pr_auc_vs_all_normals": v}
    secondary = build_across_split_summary(sec_means, ["pr_auc_vs_all_normals"])

    comparisons = build_comparisons(cfg, conditions, references, chance)
    sub_auc = subclass_auc(runs_base, family)

    reports = campaign_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports / "summary.csv", index=False)
    comparisons.to_csv(reports / "comparisons.csv", index=False)
    sub_auc.to_csv(reports / "subclass_auc.csv", index=False)
    write_markdown(
        reports / "report.md",
        summary,
        secondary,
        comparisons,
        references,
        chance,
        sub_auc,
        metric,
    )
    plot_conditions(
        reports / f"{metric}_by_condition.png",
        summary,
        references,
        cfg["reference_groups"],
        float(np.mean(list(chance.values()))) if chance else float("nan"),
        metric,
    )
    logger.info(f"Comparison report written to {reports}")
    return reports


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare an AD campaign's conditions")
    parser.add_argument("campaign_dir", help="Campaign folder, e.g. work/<series>/01-x")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_compare(Path(args.campaign_dir))


if __name__ == "__main__":
    main()
