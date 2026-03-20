"""Tests for Selection Evaluation Report Generator

Tests for the evaluation report module that aggregates
selection-eval evaluation.json files into comparison tables.
"""

import json

import pytest

from src.experiments.sample_selection.evaluation_report import (
    LOWER_IS_BETTER,
    _flatten_metrics,
    _parse_selection_eval_path,
    build_comparison_dataframe,
    generate_best_per_metric,
    generate_report,
    generate_selection_eval_table,
    load_selection_eval_results,
)


@pytest.mark.unit
def test_parse_selection_eval_path_standard():
    """Test parsing standard selection-eval path."""
    path = "outputs/diffusion-ws/selection-eval/n100-gs3_topk/reports/evaluation.json"
    result = _parse_selection_eval_path(path)
    assert result["diffusion_variant"] == "ws"
    assert result["gen_config"] == "n100-gs3"
    assert result["selection"] == "topk"


@pytest.mark.unit
def test_parse_selection_eval_path_unknown():
    """Test parsing path that doesn't match expected structure."""
    path = "some/random/path/evaluation.json"
    result = _parse_selection_eval_path(path)
    # Should not crash; parses whatever structure exists from path components
    # Path: some/random/path/evaluation.json
    #   reports_dir=some/random/path, combo_dir=some/random, selection_eval_dir=some,
    #   diffusion_dir=. → diffusion_variant="-" (normalised from empty)
    assert result["diffusion_variant"] == "-"
    assert result["gen_config"] == "random"
    assert result["selection"] == "-"


@pytest.mark.unit
def test_build_comparison_dataframe_empty():
    """Test building DataFrame from empty results."""
    df = build_comparison_dataframe([])
    assert df.empty


@pytest.mark.unit
def test_build_comparison_dataframe():
    """Test building DataFrame from results list."""
    results = [
        {
            "experiment": "ws_n100-gs3_topk",
            "diffusion_variant": "ws",
            "gen_config": "n100-gs3",
            "selection": "topk",
            "rvs_fid": 12.5,
            "rvs_precision": 0.8,
            "rvs_recall": 0.6,
        },
        {
            "experiment": "ds_n100-gs5_percentile",
            "diffusion_variant": "ds",
            "gen_config": "n100-gs5",
            "selection": "percentile",
            "rvs_fid": 15.3,
            "rvs_precision": 0.7,
            "rvs_recall": 0.5,
        },
    ]
    df = build_comparison_dataframe(results)
    assert len(df) == 2
    assert "experiment" in df.columns
    assert "rvs_fid" in df.columns


@pytest.mark.unit
def test_generate_selection_eval_table_empty():
    """Test generating table from empty DataFrame."""
    import pandas as pd

    df = pd.DataFrame()
    result = generate_selection_eval_table(df)
    assert "No evaluation results" in result


@pytest.mark.unit
def test_generate_selection_eval_table():
    """Test generating markdown table sorted by rvs_fid ascending."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "ws_n100-gs3_topk",
                "diffusion_variant": "ws",
                "gen_config": "n100-gs3",
                "selection": "topk",
                "rvs_fid": 15.0,
                "rvs_precision": 0.7,
            },
            {
                "experiment": "ds_n100-gs5_percentile",
                "diffusion_variant": "ds",
                "gen_config": "n100-gs5",
                "selection": "percentile",
                "rvs_fid": 10.0,
                "rvs_precision": 0.9,
            },
        ]
    )
    result = generate_selection_eval_table(df)
    assert "ws_n100-gs3_topk" in result
    assert "ds_n100-gs5_percentile" in result
    # Lower FID should appear first
    pos_ds = result.index("ds_n100-gs5_percentile")
    pos_ws = result.index("ws_n100-gs3_topk")
    assert pos_ds < pos_ws


@pytest.mark.unit
def test_generate_best_per_metric():
    """Test best-per-metric table generation."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "exp-a",
                "diffusion_variant": "ws",
                "rvs_fid": 10.0,
                "rvs_precision": 0.9,
                "rvs_recall": 0.5,
            },
            {
                "experiment": "exp-b",
                "diffusion_variant": "ds",
                "rvs_fid": 20.0,
                "rvs_precision": 0.7,
                "rvs_recall": 0.8,
            },
        ]
    )
    result = generate_best_per_metric(df)
    assert "exp-a" in result  # best rvs_fid (lower) and best rvs_precision (higher)
    assert "exp-b" in result  # best rvs_recall (higher)


@pytest.mark.unit
def test_generate_best_per_metric_fid_uses_min():
    """Test that FID metrics use idxmin (lower is better)."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "low-fid",
                "diffusion_variant": "ws",
                "rvs_fid": 5.0,
                "rvg_fid": 8.0,
            },
            {
                "experiment": "high-fid",
                "diffusion_variant": "ds",
                "rvs_fid": 50.0,
                "rvg_fid": 80.0,
            },
        ]
    )
    result = generate_best_per_metric(df)
    # Both FID metrics should select "low-fid" as best
    assert "low-fid" in result
    # Verify LOWER_IS_BETTER contains FID metrics
    assert "rvs_fid" in LOWER_IS_BETTER
    assert "rvg_fid" in LOWER_IS_BETTER


@pytest.mark.unit
def test_generate_best_per_metric_skips_all_nan():
    """Test that all-NaN metric columns are skipped without error."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "exp-a",
                "diffusion_variant": "ws",
                "rvs_fid": float("nan"),
                "rvs_precision": 0.9,
            },
            {
                "experiment": "exp-b",
                "diffusion_variant": "ds",
                "rvs_fid": float("nan"),
                "rvs_precision": 0.7,
            },
        ]
    )
    result = generate_best_per_metric(df)
    # rvs_fid is all NaN, should be skipped; rvs_precision should still appear
    assert "rvs_precision" in result
    assert "rvs_fid" not in result


@pytest.mark.unit
def test_generate_best_per_metric_no_key_metrics():
    """Test that generate_best_per_metric returns fallback when no KEY_METRICS columns exist."""
    import pandas as pd

    df = pd.DataFrame(
        [{"experiment": "exp-a", "diffusion_variant": "ws", "ds_real": 100}]
    )
    result = generate_best_per_metric(df)
    assert "No best-per-metric data available" in result


@pytest.mark.component
def test_load_selection_eval_results_empty_dir(tmp_path):
    """Test that an empty directory returns no results."""
    results = load_selection_eval_results(str(tmp_path))
    assert results == []


@pytest.mark.component
def test_load_selection_eval_results(tmp_path):
    """Test loading selection-eval results from directory structure."""
    # Create mock directory structure:
    # tmp_path/diffusion-ws/selection-eval/n100-gs3_topk/reports/evaluation.json
    for train, gen_sel in [("ws", "n100-gs3_topk"), ("ds", "n100-gs5_percentile")]:
        reports_dir = (
            tmp_path / f"diffusion-{train}" / "selection-eval" / gen_sel / "reports"
        )
        reports_dir.mkdir(parents=True)
        metrics = {
            "comparisons": {
                "real_vs_selected": {"fid": 12.5, "precision": 0.8, "recall": 0.6},
                "real_vs_generated": {"fid": 15.0, "precision": 0.7, "recall": 0.5},
            },
            "dataset_sizes": {"real": 100, "generated": 200, "selected": 50},
        }
        with open(reports_dir / "evaluation.json", "w") as f:
            json.dump(metrics, f)

    results = load_selection_eval_results(str(tmp_path))
    assert len(results) == 2

    # Check flattened metrics
    ws_result = [r for r in results if r["diffusion_variant"] == "ws"]
    assert len(ws_result) == 1
    assert ws_result[0]["rvs_fid"] == 12.5
    assert ws_result[0]["rvg_precision"] == 0.7
    assert ws_result[0]["ds_real"] == 100


@pytest.mark.component
def test_load_selection_eval_results_skips_malformed(tmp_path):
    """Test that malformed evaluation.json files are skipped with a warning."""
    # Create one valid and one malformed
    valid_dir = (
        tmp_path / "diffusion-ws" / "selection-eval" / "n100-gs3_topk" / "reports"
    )
    valid_dir.mkdir(parents=True)
    with open(valid_dir / "evaluation.json", "w") as f:
        json.dump({"comparisons": {}, "dataset_sizes": {}}, f)

    bad_dir = (
        tmp_path / "diffusion-ds" / "selection-eval" / "n100-gs5_percentile" / "reports"
    )
    bad_dir.mkdir(parents=True)
    (bad_dir / "evaluation.json").write_text("not valid json {{{")

    results = load_selection_eval_results(str(tmp_path))
    assert len(results) == 1
    assert results[0]["diffusion_variant"] == "ws"


@pytest.mark.component
def test_load_selection_eval_results_skips_reserved_keys(tmp_path, monkeypatch):
    """Test that reserved metadata keys in flattened metrics are skipped."""
    reports_dir = (
        tmp_path / "diffusion-ws" / "selection-eval" / "n100-gs3_topk" / "reports"
    )
    reports_dir.mkdir(parents=True)
    metrics = {
        "comparisons": {"real_vs_selected": {"fid": 10.0}},
        "dataset_sizes": {"real": 100},
    }
    with open(reports_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f)

    # Patch _flatten_metrics to inject a key that collides with reserved metadata
    import src.experiments.sample_selection.evaluation_report as report_mod

    original_flatten = report_mod._flatten_metrics

    def _flatten_with_conflict(m):
        flat = original_flatten(m)
        flat["experiment"] = "should_be_dropped"
        return flat

    monkeypatch.setattr(report_mod, "_flatten_metrics", _flatten_with_conflict)

    results = load_selection_eval_results(str(tmp_path))
    assert len(results) == 1
    # Reserved "experiment" key from flattened metrics must NOT overwrite metadata
    assert results[0]["experiment"] == "ws_n100-gs3_topk"
    assert results[0]["diffusion_variant"] == "ws"
    assert results[0]["rvs_fid"] == 10.0


@pytest.mark.unit
def test_parse_selection_eval_path_underscore_in_gen_config():
    """Test that gen_config with underscores is parsed correctly via rsplit."""
    path = "outputs/diffusion-ws/selection-eval/n100_gs3_topk/reports/evaluation.json"
    result = _parse_selection_eval_path(path)
    assert result["diffusion_variant"] == "ws"
    assert result["gen_config"] == "n100_gs3"
    assert result["selection"] == "topk"


@pytest.mark.unit
def test_flatten_metrics_skips_nested_dicts():
    """Test that nested dict values in comparisons are skipped."""
    metrics = {
        "comparisons": {
            "real_vs_selected": {
                "fid": 12.5,
                "per_class": {"class_0": 10.0, "class_1": 15.0},
            },
        },
        "dataset_sizes": {"real": 100},
    }
    flat = _flatten_metrics(metrics)
    assert flat["rvs_fid"] == 12.5
    assert "rvs_per_class" not in flat
    assert flat["ds_real"] == 100


@pytest.mark.unit
def test_flatten_metrics_skips_nested_dicts_in_dataset_sizes():
    """Test that nested dict values in dataset_sizes are skipped."""
    metrics = {
        "comparisons": {},
        "dataset_sizes": {
            "real": 100,
            "per_class": {"class_0": 50, "class_1": 50},
        },
    }
    flat = _flatten_metrics(metrics)
    assert flat["ds_real"] == 100
    assert "ds_per_class" not in flat


@pytest.mark.component
def test_generate_report_writes_output_files(tmp_path):
    """Test that generate_report creates markdown and CSV output files."""
    # Create mock directory structure
    reports_dir = (
        tmp_path / "diffusion-ws" / "selection-eval" / "n100-gs3_topk" / "reports"
    )
    reports_dir.mkdir(parents=True)
    metrics = {
        "comparisons": {
            "real_vs_selected": {"fid": 12.5, "precision": 0.8, "recall": 0.6},
            "real_vs_generated": {"fid": 15.0, "precision": 0.7, "recall": 0.5},
        },
        "dataset_sizes": {"real": 100, "generated": 200, "selected": 50},
    }
    with open(reports_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f)

    output_dir = tmp_path / "report_output"
    generate_report(base_dir=str(tmp_path), output_dir=str(output_dir))

    # Check output files exist
    assert (output_dir / "selection_eval_report.md").exists()
    assert (output_dir / "selection_eval_results.csv").exists()

    # Check report content
    report_text = (output_dir / "selection_eval_report.md").read_text()
    assert "Selection Evaluation Report" in report_text
    assert "Total experiments: 1" in report_text
    assert "ws_n100-gs3_topk" in report_text
