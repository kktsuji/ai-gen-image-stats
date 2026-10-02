"""Tests for the anomaly-detection campaign comparison report."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.anomaly_detection.compare import (
    build_comparisons,
    chance_per_split,
    correct_within_families,
    load_split_values,
    main,
    paired_row,
    run_compare,
    subclass_auc,
    validate_analysis_config,
)

N_SPLITS = 4


def _write_eval(run_dir: Path, metrics: dict) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "evaluation.json").write_text(json.dumps({"split": "test", **metrics}))


def _make_campaign(tmp_path: Path) -> Path:
    """Campaign with two AD conditions and one external reference arm."""
    campaign = tmp_path / "series" / "01-c"
    rng = np.random.default_rng(0)
    for split in range(N_SPLITS):
        for cond, base in (
            ("ad-knn-rn50__all", 0.30),
            ("ad-knn-rn50__suspicious", 0.20),
        ):
            run = campaign / "runs" / f"split{split}" / "ad-frozen" / cond / "seed0"
            _write_eval(
                run,
                {"pr_auc": base + 0.01 * split, "pr_auc_vs_all_normals": base / 4},
            )
            targets = np.array([0] * 8 + [1] * 2)
            np.savez(run / "reports" / "predictions_test.npz", targets=targets)
            scores = np.r_[
                rng.normal(0, 1, 6), rng.normal(2, 1, 2), rng.normal(5, 1, 2)
            ]
            pd.DataFrame(
                {
                    "subclass": ["red"] * 6 + ["junk"] * 2 + ["abnormal"] * 2,
                    "score": scores,
                }
            ).to_csv(run / "reports" / "subclass_scores.csv", index=False)
        # Reference arm with 2 seeds per split (seed-averaged by the loader).
        for seed, offset in ((0, -0.01), (1, 0.01)):
            ref = (
                tmp_path
                / "ref"
                / f"split{split}"
                / "binary-depth"
                / "ft-head__us"
                / f"seed{seed}"
            )
            _write_eval(ref, {"pr_auc": 0.9 + offset})
    (campaign / "configs").mkdir(parents=True)
    (campaign / "configs" / "analysis.yaml").write_text(yaml.safe_dump(_cfg(tmp_path)))
    return campaign


def _cfg(tmp_path: Path) -> dict:
    return {
        "metric": "pr_auc",
        "alpha": 0.05,
        "correction": "benjamini-hochberg",
        "runs": {"base_dir": "runs", "family": "ad-frozen"},
        "references": {
            "rn50-head": {
                "base_dir": str(tmp_path / "ref"),
                "family": "binary-depth",
                "experiment": "ft-head__us",
            }
        },
        "reference_groups": {"-rn50_": ["rn50-head"]},
        "pool_contrast": {"treatment": "all", "control": "suspicious"},
    }


@pytest.mark.unit
class TestValidateAnalysisConfig:
    def test_valid(self, tmp_path):
        validate_analysis_config(_cfg(tmp_path))

    @pytest.mark.parametrize(
        "key",
        [
            "metric",
            "alpha",
            "correction",
            "runs",
            "references",
            "reference_groups",
            "pool_contrast",
        ],
    )
    def test_missing(self, tmp_path, key):
        cfg = _cfg(tmp_path)
        del cfg[key]
        with pytest.raises(KeyError, match=key):
            validate_analysis_config(cfg)

    @pytest.mark.parametrize(
        "mutate,exc",
        [
            (lambda c: c.update(alpha=1.5), ValueError),
            (lambda c: c.update(alpha=True), ValueError),
            (lambda c: c.update(metric=""), ValueError),
            (lambda c: c.update(correction="holm"), ValueError),
            (lambda c: c["runs"].pop("family"), KeyError),
            (lambda c: c["references"]["rn50-head"].pop("experiment"), KeyError),
            (lambda c: c.update(reference_groups={"-x_": ["nope"]}), ValueError),
            (lambda c: c.update(reference_groups={}), ValueError),
            (lambda c: c.update(reference_groups={"-rn50_": "rn50-head"}), ValueError),
            (lambda c: c.update(reference_groups={"-rn50_": []}), ValueError),
            (lambda c: c.update(reference_groups={"-rn50_": [1]}), ValueError),
            (lambda c: c["pool_contrast"].pop("control"), KeyError),
            (lambda c: c.update(pool_contrast="all"), ValueError),
            (lambda c: c["pool_contrast"].update(treatment=""), ValueError),
            (lambda c: c["pool_contrast"].update(control=1), ValueError),
            (lambda c: c["pool_contrast"].update(control="all"), ValueError),
        ],
    )
    def test_invalid(self, tmp_path, mutate, exc):
        cfg = copy.deepcopy(_cfg(tmp_path))
        mutate(cfg)
        with pytest.raises(exc):
            validate_analysis_config(cfg)


@pytest.mark.unit
class TestStatistics:
    def test_paired_row_direction_and_counts(self):
        t = {0: 0.3, 1: 0.32, 2: 0.31, 3: 0.35}
        r = {0: 0.2, 1: 0.21, 2: 0.25, 3: 0.2, 9: 0.9}  # split 9 unpaired
        row = paired_row("pool", "a", "b", t, r)
        assert row is not None
        assert row["n_splits"] == 4
        assert row["mean_diff"] == pytest.approx(np.mean([0.1, 0.11, 0.06, 0.15]))
        assert row["n_treatment_better"] == 4
        assert row["cohens_dz"] > 0 and row["t_statistic"] > 0

    def test_paired_row_needs_two_splits(self):
        assert paired_row("pool", "a", "b", {0: 1.0}, {0: 0.5}) is None

    def test_correction_within_family_only(self):
        rows = [
            {"family": "a", "p_value_ttest": 0.01, "p_value_wilcoxon": 0.02},
            {"family": "a", "p_value_ttest": 0.04, "p_value_wilcoxon": 0.05},
            {"family": "b", "p_value_ttest": 0.04, "p_value_wilcoxon": 0.05},
        ]
        df = correct_within_families(rows, 0.05, "benjamini-hochberg")
        # Family b has one test, so its p is unchanged; family a is BH over 2.
        assert df.loc[2, "p_ttest_corrected"] == pytest.approx(0.04)
        assert df.loc[1, "p_ttest_corrected"] == pytest.approx(0.04)
        assert df.loc[0, "p_ttest_corrected"] == pytest.approx(0.02)
        assert df["significant"].tolist() == [True, True, True]

    def test_wilcoxon_fallback_when_ttest_undefined(self):
        rows = [
            {"family": "a", "p_value_ttest": float("nan"), "p_value_wilcoxon": 0.01}
        ]
        df = correct_within_families(rows, 0.05, "benjamini-hochberg")
        assert bool(df.loc[0, "significant"])

    def test_empty_rows(self):
        assert correct_within_families([], 0.05, "bonferroni").empty


@pytest.mark.component
class TestLoadingAndReport:
    def test_load_split_values_seed_averages(self, tmp_path):
        _make_campaign(tmp_path)
        values = load_split_values(str(tmp_path / "ref"), "binary-depth", "pr_auc")
        assert values["ft-head__us"] == pytest.approx({s: 0.9 for s in range(N_SPLITS)})

    def test_chance_per_split(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        chance = chance_per_split(str(campaign / "runs"), "ad-frozen")
        assert chance == {s: pytest.approx(0.2) for s in range(N_SPLITS)}

    def test_subclass_auc(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        df = subclass_auc(str(campaign / "runs"), "ad-frozen")
        assert set(df["subclass"]) == {"red", "junk"}
        assert (df["n_splits"] == N_SPLITS).all()
        assert (df["auc_mean"] > 0.5).all()

    def test_subclass_auc_empty(self, tmp_path):
        assert subclass_auc(str(tmp_path), "none").empty

    def test_build_comparisons_families(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        cfg = _cfg(tmp_path)
        conditions = load_split_values(str(campaign / "runs"), "ad-frozen", "pr_auc")
        refs = {
            "rn50-head": load_split_values(
                str(tmp_path / "ref"), "binary-depth", "pr_auc"
            )["ft-head__us"]
        }
        chance = chance_per_split(str(campaign / "runs"), "ad-frozen")
        df = build_comparisons(cfg, conditions, refs, chance)
        assert df.groupby("family").size().to_dict() == {
            "pool": 1,
            "vs_chance": 2,
            "vs_classifier": 2,
        }
        pool = df[df["family"] == "pool"].iloc[0]
        assert pool["treatment"] == "ad-knn-rn50__all"
        assert pool["reference"] == "ad-knn-rn50__suspicious"
        assert pool["mean_diff"] == pytest.approx(0.1)

    def test_run_compare_writes_outputs(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.experiments.anomaly_detection.compare.plot_conditions",
                lambda *a, **k: None,
            )
            reports = run_compare(campaign)
        for name in ("summary.csv", "comparisons.csv", "subclass_auc.csv", "report.md"):
            assert (reports / name).exists()
        text = (reports / "report.md").read_text()
        assert "Against chance" in text and "`rn50-head`" in text

    def test_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Analysis config"):
            run_compare(tmp_path)

    def test_no_runs(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        cfg = _cfg(tmp_path)
        cfg["runs"]["family"] = "missing"
        (campaign / "configs" / "analysis.yaml").write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="No runs"):
            run_compare(campaign)

    def test_missing_reference_arm(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        cfg = _cfg(tmp_path)
        cfg["references"]["rn50-head"]["experiment"] = "ft-nope__us"
        (campaign / "configs" / "analysis.yaml").write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="ft-nope__us"):
            run_compare(campaign)


@pytest.mark.component
class TestPlotAndCli:
    def test_cli_writes_figure(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        main([str(campaign)])
        assert (campaign / "reports" / "pr_auc_by_condition.png").stat().st_size > 0


def _in_memory():
    conditions = {
        "ad-knn-rn50__all": {s: 0.30 + 0.01 * s for s in range(N_SPLITS)},
        "ad-knn-rn50__suspicious": {s: 0.20 + 0.012 * s for s in range(N_SPLITS)},
    }
    refs = {"rn50-head": {s: 0.9 + 0.001 * s for s in range(N_SPLITS)}}
    chance = {s: 0.2 for s in range(N_SPLITS)}
    return conditions, refs, chance


@pytest.mark.unit
class TestInMemory:
    def test_build_comparisons_from_dicts(self, tmp_path):
        conditions, refs, chance = _in_memory()
        df = build_comparisons(_cfg(tmp_path), conditions, refs, chance)
        assert sorted(df["family"]) == [
            "pool",
            "vs_chance",
            "vs_chance",
            "vs_classifier",
            "vs_classifier",
        ]
        vs_ref = df[df["family"] == "vs_classifier"]
        assert (vs_ref["mean_diff"] < 0).all()
        assert (vs_ref["n_treatment_better"] == 0).all()

    def test_write_markdown(self, tmp_path):
        from src.experiments.anomaly_detection.compare import write_markdown

        conditions, refs, chance = _in_memory()
        comparisons = build_comparisons(_cfg(tmp_path), conditions, refs, chance)
        summary = pd.DataFrame(
            [
                {
                    "experiment": n,
                    "n_splits": 4,
                    "mean": 0.3,
                    "ci_lower": 0.2,
                    "ci_upper": 0.4,
                }
                for n in conditions
            ]
        )
        secondary = pd.DataFrame([{"experiment": "ad-knn-rn50__all", "mean": 0.05}])
        sub_auc = pd.DataFrame(
            [
                {"condition": "ad-knn-rn50__all", "subclass": "junk", "auc_mean": 0.2},
                {
                    "condition": "ad-knn-rn50__all",
                    "subclass": "red",
                    "auc_mean": float("nan"),
                },
            ]
        )
        out = tmp_path / "report.md"
        write_markdown(
            out, summary, secondary, comparisons, refs, chance, sub_auc, "pr_auc"
        )
        text = out.read_text()
        for heading in (
            "## Conditions",
            "## Against chance",
            "## Against classifier baselines (same backbone)",
            "## Normal pool: all vs suspicious",
            "## ROC-AUC of abnormal vs each normal subclass",
        ):
            assert heading in text
        assert "| `ad-knn-rn50__all` | 0.200 | nan |" in text

    def test_run_compare_with_stubbed_loaders(self, tmp_path):
        from unittest.mock import patch

        conditions, refs, chance = _in_memory()
        campaign = tmp_path / "c"
        (campaign / "configs").mkdir(parents=True)
        (campaign / "configs" / "analysis.yaml").write_text(
            yaml.safe_dump(_cfg(tmp_path))
        )

        def fake_load(base_dir, family, metric):
            if family == "binary-depth":
                return {"ft-head__us": refs["rn50-head"]}
            if metric == "pr_auc_vs_all_normals":
                return {n: {s: 0.05 for s in v} for n, v in conditions.items()}
            return conditions

        mod = "src.experiments.anomaly_detection.compare"
        with (
            patch(f"{mod}.load_split_values", side_effect=fake_load),
            patch(f"{mod}.chance_per_split", return_value=chance),
            patch(f"{mod}.subclass_auc", return_value=pd.DataFrame()),
            patch(f"{mod}.plot_conditions") as mock_plot,
        ):
            reports = run_compare(campaign)
        summary = pd.read_csv(reports / "summary.csv")
        assert set(summary["experiment"]) == set(conditions)
        assert len(pd.read_csv(reports / "comparisons.csv")) == 5
        mock_plot.assert_called_once()
        assert mock_plot.call_args[0][4] == pytest.approx(0.2)


@pytest.mark.component
class TestReviewFixes:
    def test_relative_reference_resolved_from_campaign(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        cfg = _cfg(tmp_path)
        # tmp_path/ref seen from tmp_path/series/01-c is ../../ref
        cfg["references"]["rn50-head"]["base_dir"] = "../../ref"
        (campaign / "configs" / "analysis.yaml").write_text(yaml.safe_dump(cfg))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.experiments.anomaly_detection.compare.plot_conditions",
                lambda *a, **k: None,
            )
            reports = run_compare(campaign)
        comparisons = pd.read_csv(reports / "comparisons.csv")
        assert (comparisons["reference"] == "rn50-head").sum() == 2

    def test_subclass_auc_averages_seeds_within_split(self, tmp_path):
        runs = tmp_path / "runs"

        def write(split, seed, junk_score):
            reports = (
                runs / f"split{split}" / "fam" / "cond" / f"seed{seed}" / "reports"
            )
            reports.mkdir(parents=True)
            pd.DataFrame(
                {
                    "subclass": ["abnormal", "junk"],
                    "score": [1.0, junk_score],
                }
            ).to_csv(reports / "subclass_scores.csv", index=False)
            (reports / "evaluation.json").write_text("{}")  # finished run

        # split0: two seeds with AUC 1 and 0 -> split mean 0.5; split1: AUC 1.
        write(0, 0, 0.0)
        write(0, 1, 2.0)
        write(1, 0, 0.0)
        df = subclass_auc(str(runs), "fam")
        row = df.iloc[0]
        assert row["n_splits"] == 2
        assert row["auc_mean"] == pytest.approx((0.5 + 1.0) / 2)


@pytest.mark.component
class TestFinalReviewFixes:
    def test_unfinished_runs_ignored_by_chance_and_subclass_auc(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        runs = campaign / "runs"
        # Make one condition's split-0 run unfinished (no evaluation.json).
        unfinished = (
            runs / "split0" / "ad-frozen" / "ad-knn-rn50__all" / "seed0" / "reports"
        )
        (unfinished / "evaluation.json").unlink()
        # The other condition's split 0 is unfinished too, so split 0 has no
        # finished run at all and must vanish from both.
        other = (
            runs
            / "split0"
            / "ad-frozen"
            / "ad-knn-rn50__suspicious"
            / "seed0"
            / "reports"
        )
        (other / "evaluation.json").unlink()

        chance = chance_per_split(str(runs), "ad-frozen")
        assert 0 not in chance and set(chance) == {1, 2, 3}
        df = subclass_auc(str(runs), "ad-frozen")
        assert (df["n_splits"] == N_SPLITS - 1).all()

    def test_chance_level_depends_on_metric(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        runs = str(campaign / "runs")
        assert chance_per_split(runs, "ad-frozen", "roc_auc") == {
            s: 0.5 for s in range(N_SPLITS)
        }
        assert chance_per_split(runs, "ad-frozen", "f1_1") == {}

    def test_no_chance_level_skips_vs_chance(self, tmp_path):
        conditions, refs, _ = _in_memory()
        df = build_comparisons(_cfg(tmp_path), conditions, refs, {})
        assert "vs_chance" not in set(df["family"])
        assert {"vs_classifier", "pool"} <= set(df["family"])


@pytest.mark.unit
class TestPrReviewFixes:
    def test_split_and_condition(self):
        from src.experiments.anomaly_detection.compare import _split_and_condition

        path = "/c/runs/split7/ad-frozen/ad-knn-rn50__all/seed0/reports/x.csv"
        assert _split_and_condition(path) == (7, "ad-knn-rn50__all")

    def test_plot_skipped_without_groups(self, tmp_path):
        from src.experiments.anomaly_detection.compare import plot_conditions

        out = tmp_path / "fig.png"
        plot_conditions(out, pd.DataFrame(), {}, {}, 0.2)
        assert not out.exists()
