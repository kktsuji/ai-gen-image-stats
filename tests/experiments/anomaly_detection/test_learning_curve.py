"""Tests for the learning curve over k labelled abnormal images."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.anomaly_detection.learning_curve import (
    arm_values,
    build_comparisons,
    curve_table,
    evaluate_criteria,
    main,
    run_learning_curve,
    smallest_k_from,
    validate_config,
    write_markdown,
)

KS = [1, 5, 20]
DRAWS = [0, 1]
N_SPLITS = 4


def _cfg():
    return {
        "metric": "pr_auc",
        "alpha": 0.05,
        "correction": "benjamini-hochberg",
        "runs_base": "runs",
        "ks": list(KS),
        "draws": list(DRAWS),
        "classifier_arm": "clf",
        "arms": {
            "clf": {
                "label": "Classifier",
                "family": "clf",
                "template": "rn50-k{k}-d{draw}",
                "full": "rn50-kall",
            },
            "maha": {
                "label": "Mahalanobis",
                "family": "ad-k",
                "template": "ad-maha-k{k}-d{draw}",
                "full": "ad-maha-kall",
            },
        },
        "zero_references": {
            "maha": [
                {
                    "label": "frozen",
                    "base_dir": "../01/runs",
                    "family": "ad-frozen",
                    "experiment": "ad-maha",
                }
            ]
        },
        "criteria": {"ad_margin": 0.05, "classifier_target": 0.9},
    }


# Mean values per arm and k (draw d adds +/- 0.01, split s adds 0.002 * s).
CLF = {1: 0.5, 5: 0.8, 20: 0.92, "all": 0.95}
MAHA = {1: 0.2, 5: 0.6, 20: 0.9, "all": 0.93}


def _write_eval(run_dir: Path, value: float, chance_targets=None) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "evaluation.json").write_text(
        json.dumps({"split": "test", "pr_auc": value})
    )
    if chance_targets is not None:
        np.savez(reports / "predictions_test.npz", targets=chance_targets)


def _make_campaign(tmp_path: Path, skip=()) -> Path:
    campaign = tmp_path / "series" / "05-k"
    targets = np.array([0] * 8 + [1] * 2)
    for split in range(N_SPLITS):
        for fam, prefix, means, tgt in (
            ("clf", "rn50", CLF, None),
            ("ad-k", "ad-maha", MAHA, targets),
        ):
            base = campaign / "runs" / f"split{split}" / fam
            for k in KS:
                for d in DRAWS:
                    name = f"{prefix}-k{k}-d{d}"
                    if (name, split) in skip:
                        continue
                    value = means[k] + (0.01 if d else -0.01) + 0.002 * split
                    _write_eval(base / name / "seed0", value, tgt)
            if (f"{prefix}-kall", split) not in skip:
                _write_eval(
                    base / f"{prefix}-kall" / "seed0",
                    means["all"] + 0.002 * split,
                    tgt,
                )
        _write_eval(
            tmp_path
            / "series"
            / "01"
            / "runs"
            / f"split{split}"
            / "ad-frozen"
            / "ad-maha"
            / "seed0",
            0.12,
        )
    (campaign / "configs").mkdir(parents=True)
    (campaign / "configs" / "learning_curve.yaml").write_text(yaml.safe_dump(_cfg()))
    return campaign


@pytest.mark.unit
class TestValidateConfig:
    def test_valid(self):
        validate_config(_cfg())

    @pytest.mark.parametrize(
        "key",
        [
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
        ],
    )
    def test_missing_field(self, key):
        cfg = _cfg()
        del cfg[key]
        with pytest.raises(KeyError, match=key):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda c: c.update(alpha=1.5), "alpha"),
            (lambda c: c.update(correction="holm"), "correction"),
            (lambda c: c.update(ks=[0, 1]), "ks"),
            (lambda c: c.update(ks=[1, 1]), "ks"),
            (lambda c: c.update(draws=[]), "draws"),
            (lambda c: c.update(classifier_arm="x"), "classifier_arm"),
            (lambda c: c["arms"].pop("maha"), "at least two"),
            (lambda c: c["arms"]["clf"].update(template="rn50-k{k}"), "exactly"),
            (lambda c: c["arms"]["clf"].update(template="k{k}-d{draw}-{x}"), "exactly"),
            (lambda c: c["arms"]["clf"].update(template="k{k"), "not a valid"),
            (lambda c: c["arms"]["clf"].update(template="a-{}-k{k}-d{draw}"), "plain"),
            (lambda c: c["arms"]["clf"].update(template="a-{0}-k{k}-d{draw}"), "plain"),
            (lambda c: c["arms"]["clf"].update(template="k{k!r}-d{draw}"), "plain"),
            (lambda c: c["arms"]["clf"].update(template="k{k:02d}-d{draw}"), "plain"),
            (lambda c: c["arms"]["clf"].update(full=""), "full"),
            (lambda c: c.update(zero_references={"clf": []}), "not a detector"),
            (lambda c: c.update(zero_references={"maha": []}), "non-empty list"),
            (lambda c: c["zero_references"]["maha"][0].pop("family"), "family"),
            (lambda c: c["criteria"].update(ad_margin=0), "ad_margin"),
            (
                lambda c: c["criteria"].update(classifier_target=True),
                "classifier_target",
            ),
        ],
    )
    def test_invalid(self, mutate, match):
        cfg = _cfg()
        mutate(cfg)
        with pytest.raises((ValueError, KeyError), match=match):
            validate_config(cfg)

    def test_full_may_be_null(self):
        cfg = _cfg()
        cfg["arms"]["clf"]["full"] = None
        validate_config(cfg)


@pytest.mark.unit
class TestArmValues:
    ARM = {"label": "A", "template": "a-k{k}-d{draw}", "full": "a-kall"}

    def test_draws_are_averaged(self):
        values = {
            "a-k1-d0": {0: 0.2, 1: 0.4},
            "a-k1-d1": {0: 0.4, 1: 0.6},
            "a-kall": {0: 0.9},
        }
        out, notes = arm_values(values, self.ARM, [1], [0, 1], {0, 1})
        assert out[1] == pytest.approx({0: 0.3, 1: 0.5})
        assert out["all"] == {0: 0.9}
        assert notes == ["A, k=all: split 1 missing"]

    def test_incomplete_split_left_out_with_a_note(self):
        values = {"a-k1-d0": {0: 0.2, 1: 0.4}, "a-k1-d1": {0: 0.4}}
        out, notes = arm_values(values, self.ARM, [1], [0, 1], {0, 1})
        assert out[1] == pytest.approx({0: 0.3})
        assert any("split 1 left out (draws [1] missing)" in n for n in notes)
        assert any("no 'a-kall' runs" in n for n in notes)

    def test_no_full_arm(self):
        arm = {**self.ARM, "full": None}
        out, notes = arm_values({"a-k1-d0": {0: 0.2}}, arm, [1], [0], {0})
        assert "all" not in out and notes == []

    def test_split_without_any_draw_is_noted(self):
        values = {
            "a-k1-d0": {0: 0.2, 1: 0.4},
            "a-k1-d1": {0: 0.4, 1: 0.6},
            "a-k5-d0": {0: 0.5},
            "a-k5-d1": {0: 0.7},
            "a-kall": {0: 0.9, 1: 0.9},
        }
        out, notes = arm_values(values, self.ARM, [1, 5, 20], [0, 1], {0, 1})
        assert out[5] == pytest.approx({0: 0.6})
        assert "A, k=5: split 1 left out (draws [0, 1] missing)" in notes
        assert "A, k=20: no split has every draw finished" in notes
        assert 20 not in out

    def test_splits_outside_expected_are_ignored(self):
        values = {"a-k1-d0": {0: 0.2, 7: 0.9}, "a-kall": {0: 0.9, 7: 0.9}}
        out, _ = arm_values(values, self.ARM, [1], [0], {0})
        assert out[1] == {0: 0.2} and out["all"] == {0: 0.9}


@pytest.mark.unit
class TestSmallestK:
    def test_must_hold_for_every_larger_k(self):
        assert smallest_k_from({1: True, 5: False, 20: True, "all": True}, KS) == 20
        assert smallest_k_from({1: True, 5: True, 20: True, "all": True}, KS) == 1
        assert smallest_k_from({1: True, 5: True, 20: True, "all": False}, KS) is None

    def test_a_gap_ends_the_run(self):
        assert smallest_k_from({1: True, 20: True, "all": True}, KS) == 20

    def test_without_a_full_run(self):
        assert smallest_k_from({1: True, 5: True, 20: True}, KS) == 1
        assert smallest_k_from({1: True, 5: False, 20: True}, KS) == 20
        assert smallest_k_from({1: True, 5: True, 20: False}, KS) is None
        assert smallest_k_from({}, KS) is None

    def test_criteria_with_full_null(self):
        cfg = _cfg()
        cfg["arms"]["clf"]["full"] = None
        curves = TestAnalysis()._curves()
        del curves["clf"]["all"]
        out = evaluate_criteria(cfg, curves)
        assert out["maha_within_margin"] == 20
        assert out["clf_reaches_target"] == 20


@pytest.mark.unit
class TestAnalysis:
    def _curves(self):
        def curve(means):
            return {
                k: {s: means[k] + 0.002 * s for s in range(N_SPLITS)}
                for k in [*KS, "all"]
            }

        return {"clf": curve(CLF), "maha": curve(MAHA)}

    def test_criteria(self):
        out = evaluate_criteria(_cfg(), self._curves())
        # maha - clf: -0.3, -0.2, -0.02, -0.02 -> within 0.05 from k=20.
        assert out["maha_within_margin"] == 20
        # clf: 0.5, 0.8, 0.92, 0.95 -> >= 0.9 from k=20.
        assert out["clf_reaches_target"] == 20

    def test_comparisons_families(self):
        chance = {s: 0.2 for s in range(N_SPLITS)}
        df = build_comparisons(_cfg(), self._curves(), chance)
        assert set(df["family"]) == {"vs_classifier", "vs_chance"}
        vs_clf = df[df["family"] == "vs_classifier"]
        assert list(vs_clf["treatment"]) == [f"maha@k={k}" for k in [*KS, "all"]]
        assert (vs_clf["reference"] == [f"clf@k={k}" for k in [*KS, "all"]]).all()
        assert len(df[df["family"] == "vs_chance"]) == 2 * 4

    def test_table_includes_zero_points(self):
        zero = {"maha": [("frozen", {s: 0.12 for s in range(N_SPLITS)})]}
        table = curve_table(_cfg()["arms"], self._curves(), zero, KS)
        row = table[(table["arm"] == "maha") & (table["k"] == "0")].iloc[0]
        assert row["reference"] == "frozen" and row["mean"] == pytest.approx(0.12)
        assert list(table[table["arm"] == "clf"]["k"]) == ["1", "5", "20", "all"]


@pytest.mark.component
class TestRunLearningCurve:
    def test_end_to_end(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        main([str(campaign)])
        reports = campaign / "reports"
        for name in (
            "learning_curve.csv",
            "learning_curve_comparisons.csv",
            "learning_curve.md",
            "learning_curve.png",
        ):
            assert (reports / name).exists()
        table = pd.read_csv(reports / "learning_curve.csv")
        clf_k5 = table[(table["arm"] == "clf") & (table["k"] == "5")].iloc[0]
        assert clf_k5["mean"] == pytest.approx(0.8 + 0.002 * 1.5)
        md = (reports / "learning_curve.md").read_text()
        assert "Chance level" in md and "0.200" in md
        assert "**20**" in md
        assert "## Notes" not in md

    def test_missing_draw_is_reported(self, tmp_path):
        campaign = _make_campaign(tmp_path, skip={("ad-maha-k5-d1", 2)})
        run_learning_curve(campaign)
        md = (campaign / "reports" / "learning_curve.md").read_text()
        assert "split 2 left out (draws [1] missing)" in md
        table = pd.read_csv(campaign / "reports" / "learning_curve.csv")
        row = table[(table["arm"] == "maha") & (table["k"] == "5")].iloc[0]
        assert row["n_splits"] == N_SPLITS - 1

    def test_changed_zero_reference_checkpoint_refused(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        ckpt = tmp_path / "series" / "01" / "model.pth"
        ckpt.write_bytes(b"weights")
        for split in range(N_SPLITS):
            reports = (
                tmp_path
                / "series"
                / "01"
                / "runs"
                / f"split{split}"
                / "ad-frozen"
                / "ad-maha"
                / "seed0"
                / "reports"
            )
            ev = json.loads((reports / "evaluation.json").read_text())
            ev["checkpoint_sha256"] = hashlib.sha256(b"weights").hexdigest()
            ev["checkpoint"] = "../../../../../../model.pth"
            (reports / "evaluation.json").write_text(json.dumps(ev))
        run_learning_curve(campaign)  # unchanged: accepted
        ckpt.write_bytes(b"retrained")
        with pytest.raises(ValueError, match="has changed"):
            run_learning_curve(campaign)

    def test_split_missing_everywhere_in_one_arm_is_noted(self, tmp_path):
        skip = {(f"ad-maha-k20-d{d}", 3) for d in DRAWS} | {("ad-maha-kall", 3)}
        campaign = _make_campaign(tmp_path, skip=skip)
        run_learning_curve(campaign)
        md = (campaign / "reports" / "learning_curve.md").read_text()
        assert "Mahalanobis, k=20: split 3 left out (draws [0, 1] missing)" in md
        assert "Mahalanobis, k=all: split 3 missing" in md

    def test_chance_missing_split_is_noted(self, tmp_path):
        skip = {(f"ad-maha-k{k}-d{d}", 3) for k in KS for d in DRAWS}
        skip.add(("ad-maha-kall", 3))
        campaign = _make_campaign(tmp_path, skip=skip)
        run_learning_curve(campaign)
        md = (campaign / "reports" / "learning_curve.md").read_text()
        assert "chance: split 3 has no finished detector run" in md

    def _add_checkpoint(self, tmp_path, campaign, experiment, split=0):
        ckpt = tmp_path / f"{experiment}.pth"
        ckpt.write_bytes(b"weights")
        reports = (
            campaign / "runs" / f"split{split}" / "ad-k" / experiment / "seed0"
        ) / "reports"
        if not (reports / "evaluation.json").exists():
            _write_eval(reports.parent, 0.5)
        ev = json.loads((reports / "evaluation.json").read_text())
        ev["checkpoint_sha256"] = hashlib.sha256(b"weights").hexdigest()
        ev["checkpoint"] = str(ckpt)
        (reports / "evaluation.json").write_text(json.dumps(ev))
        return ckpt

    def test_checkpoint_check_covers_only_the_arm_runs(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        other = self._add_checkpoint(tmp_path, campaign, "ad-unrelated")
        used = self._add_checkpoint(tmp_path, campaign, "ad-maha-k5-d0")
        other.write_bytes(b"retrained")
        run_learning_curve(campaign)  # an unrelated run does not block
        used.write_bytes(b"retrained")
        with pytest.raises(ValueError, match="has changed"):
            run_learning_curve(campaign)

    def test_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_learning_curve(tmp_path)
        campaign = _make_campaign(tmp_path)
        cfg = _cfg()
        cfg["zero_references"]["maha"][0]["experiment"] = "nope"
        (campaign / "configs" / "learning_curve.yaml").write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="no 'nope' results"):
            run_learning_curve(campaign)
        cfg = copy.deepcopy(_cfg())
        cfg["arms"]["clf"]["family"] = "missing"
        (campaign / "configs" / "learning_curve.yaml").write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="No finished runs for arm 'clf'"):
            run_learning_curve(campaign)


@pytest.mark.unit
class TestChanceLabel:
    @pytest.mark.parametrize(
        "metric,label",
        [("pr_auc", "mean abnormal fraction"), ("roc_auc", "0.5 for ROC-AUC")],
    )
    def test_label_follows_the_metric(self, tmp_path, metric, label):
        cfg = {**_cfg(), "metric": metric}
        path = tmp_path / "lc.md"
        write_markdown(
            path,
            cfg,
            curve_table(cfg["arms"], {"clf": {}, "maha": {}}, {}, KS),
            pd.DataFrame(),
            {},
            {0: 0.5},
            [],
        )
        assert f"Chance level ({label}" in path.read_text()


@pytest.mark.unit
class TestTemplateErrorMessage:
    def test_shows_the_field_as_written(self):
        cfg = _cfg()
        cfg["arms"]["clf"]["template"] = "k{k!r:>3}-d{draw}"
        with pytest.raises(ValueError, match=r"got '\{k!r:>3\}'"):
            validate_config(cfg)


@pytest.mark.unit
class TestPlotZeroTick:
    def _ticks(self, tmp_path, zero, monkeypatch):
        from src.experiments.anomaly_detection import learning_curve as lc

        seen = {}
        original = lc.plt.Axes.set_xticks

        def spy(ax, ticks, labels=None, **kw):
            seen["labels"] = list(labels) if labels is not None else None
            return original(ax, ticks, labels, **kw)

        monkeypatch.setattr(lc.plt.Axes, "set_xticks", spy)
        curves = TestAnalysis()._curves()
        table = curve_table(_cfg()["arms"], curves, zero, KS)
        lc.plot_curve(tmp_path / "lc.png", _cfg(), table, {})
        return seen["labels"]

    def test_zero_tick_only_with_zero_points(self, tmp_path, monkeypatch):
        zero = {"maha": [("frozen", {s: 0.12 for s in range(N_SPLITS)})]}
        assert self._ticks(tmp_path, zero, monkeypatch)[0] == "0"
        assert self._ticks(tmp_path, {}, monkeypatch) == ["1", "5", "20", "all"]
