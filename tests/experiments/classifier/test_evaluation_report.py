"""Tests for Evaluation Report Generator

Tests for the evaluation report module that aggregates
evaluation.json files into comparison tables.
"""

import json

import pytest

from src.experiments.classifier.evaluation_report import (
    _parse_experiment_name,
    build_comparison_dataframe,
    generate_best_per_metric,
    generate_classifier_table,
    load_evaluation_results,
)


@pytest.mark.unit
def test_parse_experiment_name_baseline():
    """Test parsing baseline experiment names."""
    result = _parse_experiment_name("baseline-vanilla")
    assert result["type"] == "baseline"
    assert result["baseline_strategy"] == "vanilla"
    assert result["diffusion_variant"] == "-"


@pytest.mark.unit
def test_parse_experiment_name_synthetic():
    """Test parsing synthetic augmentation experiment names."""
    result = _parse_experiment_name("ws-n100-gs3-topk-all")
    assert result["type"] == "synthetic"
    assert result["diffusion_variant"] == "ws"
    assert result["gen_config"] == "n100-gs3"
    assert result["selection"] == "topk"
    assert result["aug_limit"] == "all"


@pytest.mark.unit
def test_parse_experiment_name_unknown():
    """Test parsing unknown experiment name format."""
    result = _parse_experiment_name("something")
    assert result["type"] == "unknown"


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
            "experiment": "baseline-vanilla",
            "type": "baseline",
            "accuracy": 83.5,
            "balanced_accuracy": 0.65,
            "recall_1": 0.3,
        },
        {
            "experiment": "ws-n100-gs3-topk-all",
            "type": "synthetic",
            "accuracy": 85.0,
            "balanced_accuracy": 0.75,
            "recall_1": 0.6,
        },
    ]
    df = build_comparison_dataframe(results)
    assert len(df) == 2
    assert "experiment" in df.columns
    assert "accuracy" in df.columns


@pytest.mark.unit
def test_generate_classifier_table_empty():
    """Test generating table from empty DataFrame."""
    import pandas as pd

    df = pd.DataFrame()
    result = generate_classifier_table(df)
    assert "No evaluation results" in result


@pytest.mark.unit
def test_generate_classifier_table():
    """Test generating markdown table."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline-vanilla",
                "type": "baseline",
                "recall_1": 0.3,
                "balanced_accuracy": 0.65,
            },
        ]
    )
    result = generate_classifier_table(df)
    assert "baseline-vanilla" in result
    assert "recall_1" in result


@pytest.mark.unit
def test_generate_best_per_metric():
    """Test best-per-metric table generation."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "exp-a",
                "type": "synthetic",
                "recall_1": 0.8,
                "balanced_accuracy": 0.7,
            },
            {
                "experiment": "exp-b",
                "type": "baseline",
                "recall_1": 0.3,
                "balanced_accuracy": 0.9,
            },
        ]
    )
    result = generate_best_per_metric(df)
    assert "exp-a" in result  # best recall_1
    assert "exp-b" in result  # best balanced_accuracy


@pytest.mark.component
def test_load_evaluation_results(tmp_path):
    """Test loading evaluation results from directory structure."""
    # Create mock evaluation.json files
    for exp_name in ["baseline-vanilla", "ws-n100-gs3-topk-all"]:
        reports_dir = tmp_path / exp_name / "reports"
        reports_dir.mkdir(parents=True)
        metrics = {"accuracy": 80.0, "balanced_accuracy": 0.7, "recall_1": 0.5}
        with open(reports_dir / "evaluation.json", "w") as f:
            json.dump(metrics, f)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 2
    assert results[0]["experiment"] == "baseline-vanilla"
    assert results[0]["accuracy"] == 80.0


@pytest.mark.component
def test_load_evaluation_results_skips_reserved_keys(tmp_path):
    """Test that reserved metadata keys in evaluation.json are skipped."""
    reports_dir = tmp_path / "baseline-vanilla" / "reports"
    reports_dir.mkdir(parents=True)
    # Include a conflicting "type" key that should be skipped
    metrics = {"accuracy": 80.0, "type": "should_be_ignored", "experiment": "evil"}
    with open(reports_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 1
    # Reserved keys should NOT be overwritten by evaluation.json
    assert results[0]["type"] == "baseline"
    assert results[0]["experiment"] == "baseline-vanilla"
    assert results[0]["accuracy"] == 80.0


@pytest.mark.component
def test_load_evaluation_results_skips_malformed_json(tmp_path):
    """Test that malformed evaluation.json files are skipped with a warning."""
    # Create one valid and one malformed evaluation.json
    valid_dir = tmp_path / "baseline-vanilla" / "reports"
    valid_dir.mkdir(parents=True)
    with open(valid_dir / "evaluation.json", "w") as f:
        json.dump({"accuracy": 80.0}, f)

    bad_dir = tmp_path / "broken-exp" / "reports"
    bad_dir.mkdir(parents=True)
    (bad_dir / "evaluation.json").write_text("not valid json {{{")

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 1
    assert results[0]["experiment"] == "baseline-vanilla"
