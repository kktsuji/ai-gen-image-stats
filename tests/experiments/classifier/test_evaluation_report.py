"""Tests for Evaluation Report Generator

Tests for the evaluation report module that aggregates
evaluation.json files into comparison tables.
"""

import json

import pytest

from src.experiments.classifier.evaluation_report import (
    _parse_experiment_name,
    aggregate_multi_seed,
    build_comparison_dataframe,
    build_mean_std_dataframe,
    generate_best_per_metric,
    generate_classifier_table,
    generate_statistical_comparison_table,
    load_evaluation_results,
)


@pytest.mark.unit
def test_parse_experiment_name_baseline():
    """Test parsing baseline experiment names."""
    result = _parse_experiment_name("baseline__vanilla")
    assert result["type"] == "baseline"
    assert result["baseline_strategy"] == "vanilla"
    assert result["diffusion_variant"] == "-"


@pytest.mark.unit
def test_parse_experiment_name_synthetic():
    """Test parsing synthetic augmentation experiment names."""
    result = _parse_experiment_name("ws__n100-gs3__topk__all")
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
            "experiment": "baseline__vanilla",
            "type": "baseline",
            "accuracy": 83.5,
            "balanced_accuracy": 0.65,
            "recall_1": 0.3,
        },
        {
            "experiment": "ws__n100-gs3__topk__all",
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
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "recall_1": 0.3,
                "balanced_accuracy": 0.65,
            },
        ]
    )
    result = generate_classifier_table(df)
    assert "baseline__vanilla" in result
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


@pytest.mark.unit
def test_generate_classifier_table_with_ci():
    """Test generating markdown table with CI bounds."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "recall_1": 0.3,
                "recall_1_ci_lower": 0.25,
                "recall_1_ci_upper": 0.35,
                "balanced_accuracy": 0.65,
                "balanced_accuracy_ci_lower": 0.60,
                "balanced_accuracy_ci_upper": 0.70,
            },
        ]
    )
    result = generate_classifier_table(df)
    assert "baseline__vanilla" in result
    # Should contain CI format: value [lower, upper]
    assert "[0.2500, 0.3500]" in result
    assert "[0.6000, 0.7000]" in result


@pytest.mark.unit
def test_generate_best_per_metric_with_ci():
    """Test best-per-metric table includes CI when available."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "exp-a",
                "type": "synthetic",
                "recall_1": 0.8,
                "recall_1_ci_lower": 0.75,
                "recall_1_ci_upper": 0.85,
                "balanced_accuracy": 0.7,
            },
            {
                "experiment": "exp-b",
                "type": "baseline",
                "recall_1": 0.3,
                "recall_1_ci_lower": 0.25,
                "recall_1_ci_upper": 0.35,
                "balanced_accuracy": 0.9,
            },
        ]
    )
    result = generate_best_per_metric(df)
    # Best recall_1 is exp-a with CI
    assert "[0.7500, 0.8500]" in result
    # Best balanced_accuracy is exp-b, no CI available
    assert "exp-b" in result


@pytest.mark.unit
def test_generate_classifier_table_without_ci():
    """Test table works normally when no CI columns present."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "recall_1": 0.3000,
                "balanced_accuracy": 0.6500,
            },
        ]
    )
    result = generate_classifier_table(df)
    assert "baseline__vanilla" in result
    assert "0.3000" in result
    # No CI brackets
    assert "[" not in result


@pytest.mark.component
def test_load_evaluation_results(tmp_path):
    """Test loading evaluation results from directory structure."""
    # Create mock evaluation.json files
    for exp_name in ["baseline__vanilla", "ws__n100-gs3__topk__all"]:
        reports_dir = tmp_path / exp_name / "reports"
        reports_dir.mkdir(parents=True)
        metrics = {"accuracy": 80.0, "balanced_accuracy": 0.7, "recall_1": 0.5}
        with open(reports_dir / "evaluation.json", "w") as f:
            json.dump(metrics, f)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 2
    baseline = [r for r in results if r["experiment"] == "baseline__vanilla"]
    assert len(baseline) == 1
    assert baseline[0]["accuracy"] == 80.0


@pytest.mark.component
def test_load_evaluation_results_skips_reserved_keys(tmp_path):
    """Test that reserved metadata keys in evaluation.json are skipped."""
    reports_dir = tmp_path / "baseline__vanilla" / "reports"
    reports_dir.mkdir(parents=True)
    # Include a conflicting "type" key that should be skipped
    metrics = {"accuracy": 80.0, "type": "should_be_ignored", "experiment": "evil"}
    with open(reports_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 1
    # Reserved keys should NOT be overwritten by evaluation.json
    assert results[0]["type"] == "baseline"
    assert results[0]["experiment"] == "baseline__vanilla"
    assert results[0]["accuracy"] == 80.0


@pytest.mark.component
def test_load_evaluation_results_skips_malformed_json(tmp_path):
    """Test that malformed evaluation.json files are skipped with a warning."""
    # Create one valid and one malformed evaluation.json
    valid_dir = tmp_path / "baseline__vanilla" / "reports"
    valid_dir.mkdir(parents=True)
    with open(valid_dir / "evaluation.json", "w") as f:
        json.dump({"accuracy": 80.0}, f)

    bad_dir = tmp_path / "broken-exp" / "reports"
    bad_dir.mkdir(parents=True)
    (bad_dir / "evaluation.json").write_text("not valid json {{{")

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 1
    assert results[0]["experiment"] == "baseline__vanilla"


# --- Multi-seed tests ---


def _create_multi_seed_results(tmp_path, exp_name, seeds, metrics_per_seed):
    """Helper to create multi-seed evaluation.json files."""
    for seed, metrics in zip(seeds, metrics_per_seed):
        reports_dir = tmp_path / exp_name / f"seed{seed}" / "reports"
        reports_dir.mkdir(parents=True)
        with open(reports_dir / "evaluation.json", "w") as f:
            json.dump(metrics, f)


@pytest.mark.component
def test_load_evaluation_results_multi_seed(tmp_path):
    """Test loading multi-seed evaluation results."""
    seeds = [0, 1, 2]
    metrics_list = [
        {"recall_1": 0.70, "balanced_accuracy": 0.75},
        {"recall_1": 0.72, "balanced_accuracy": 0.77},
        {"recall_1": 0.68, "balanced_accuracy": 0.73},
    ]
    _create_multi_seed_results(tmp_path, "baseline__vanilla", seeds, metrics_list)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 3
    assert all(r["experiment"] == "baseline__vanilla" for r in results)
    assert sorted(r["seed"] for r in results) == [0, 1, 2]


@pytest.mark.component
def test_load_evaluation_results_backward_compatibility(tmp_path):
    """Test that single-seed results are loaded alongside multi-seed."""
    # Multi-seed experiment
    _create_multi_seed_results(
        tmp_path,
        "baseline__vanilla",
        [0, 1],
        [{"recall_1": 0.70}, {"recall_1": 0.72}],
    )
    # Single-seed experiment (legacy layout)
    reports_dir = tmp_path / "ws__n100-gs3__topk__all" / "reports"
    reports_dir.mkdir(parents=True)
    with open(reports_dir / "evaluation.json", "w") as f:
        json.dump({"recall_1": 0.80}, f)

    results = load_evaluation_results(str(tmp_path))
    assert len(results) == 3
    # Multi-seed results have seed field
    multi = [r for r in results if "seed" in r]
    assert len(multi) == 2
    # Single-seed result has no seed field
    single = [r for r in results if "seed" not in r]
    assert len(single) == 1
    assert single[0]["experiment"] == "ws__n100-gs3__topk__all"


@pytest.mark.unit
def test_aggregate_multi_seed():
    """Test aggregating multi-seed results by experiment name."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"experiment": "exp-a", "seed": 0, "recall_1": 0.70, "f1_1": 0.65},
            {"experiment": "exp-a", "seed": 1, "recall_1": 0.72, "f1_1": 0.67},
            {"experiment": "exp-a", "seed": 2, "recall_1": 0.68, "f1_1": 0.63},
            {"experiment": "exp-b", "seed": 0, "recall_1": 0.80, "f1_1": 0.75},
            {"experiment": "exp-b", "seed": 1, "recall_1": 0.82, "f1_1": 0.77},
        ]
    )

    aggregated = aggregate_multi_seed(df, ["recall_1", "f1_1"])
    assert "exp-a" in aggregated
    assert "exp-b" in aggregated
    assert len(aggregated["exp-a"]["recall_1"]) == 3
    assert len(aggregated["exp-b"]["recall_1"]) == 2
    # Verify _seeds key is present and sorted
    assert "_seeds" in aggregated["exp-a"]
    assert list(aggregated["exp-a"]["_seeds"]) == [0, 1, 2]
    assert list(aggregated["exp-b"]["_seeds"]) == [0, 1]


@pytest.mark.unit
def test_aggregate_multi_seed_skips_single_seed():
    """Test that experiments with only one seed are excluded."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"experiment": "exp-a", "seed": 0, "recall_1": 0.70},
        ]
    )

    aggregated = aggregate_multi_seed(df, ["recall_1"])
    assert "exp-a" not in aggregated


@pytest.mark.unit
def test_aggregate_multi_seed_no_seed_column():
    """Test that aggregation returns empty dict without seed column."""
    import pandas as pd

    df = pd.DataFrame([{"experiment": "exp-a", "recall_1": 0.70}])
    aggregated = aggregate_multi_seed(df, ["recall_1"])
    assert aggregated == {}


@pytest.mark.unit
def test_build_mean_std_dataframe():
    """Test building mean +/- std DataFrame from multi-seed results."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": 0,
                "recall_1": 0.70,
                "diffusion_variant": "-",
                "gen_config": "-",
                "selection": "-",
                "aug_limit": "-",
                "baseline_strategy": "vanilla",
            },
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": 1,
                "recall_1": 0.80,
                "diffusion_variant": "-",
                "gen_config": "-",
                "selection": "-",
                "aug_limit": "-",
                "baseline_strategy": "vanilla",
            },
        ]
    )

    result = build_mean_std_dataframe(df, ["recall_1"])
    assert len(result) == 1
    assert result.iloc[0]["recall_1"] == pytest.approx(0.75)
    assert "recall_1_std" in result.columns
    assert result.iloc[0]["n_seeds"] == 2


@pytest.mark.unit
def test_build_mean_std_dataframe_no_seed_column():
    """Test that build_mean_std_dataframe returns df unchanged without seed column."""
    import pandas as pd

    df = pd.DataFrame([{"experiment": "exp-a", "recall_1": 0.70}])
    result = build_mean_std_dataframe(df, ["recall_1"])
    assert len(result) == 1
    assert result.iloc[0]["recall_1"] == 0.70


@pytest.mark.unit
def test_generate_statistical_comparison_table():
    """Test statistical comparison table generation with multi-seed data."""
    import pandas as pd

    # Create multi-seed data with clear improvement
    rows = []
    baseline_vals = [0.72, 0.68, 0.75, 0.70, 0.66]
    synthetic_vals = [0.78, 0.75, 0.77, 0.76, 0.71]

    for seed, (bl, syn) in enumerate(zip(baseline_vals, synthetic_vals)):
        rows.append(
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": seed,
                "recall_1": bl,
                "diffusion_variant": "-",
                "gen_config": "-",
                "selection": "-",
                "aug_limit": "-",
                "baseline_strategy": "vanilla",
            }
        )
        rows.append(
            {
                "experiment": "ws__n100-gs3__topk__all",
                "type": "synthetic",
                "seed": seed,
                "recall_1": syn,
                "diffusion_variant": "ws",
                "gen_config": "n100-gs3",
                "selection": "topk",
                "aug_limit": "all",
                "baseline_strategy": "-",
            }
        )

    df = pd.DataFrame(rows)
    result = generate_statistical_comparison_table(df, alpha=0.05)

    assert "baseline__vanilla" in result
    assert "ws__n100-gs3__topk__all" in result
    assert "cohens_d" in result
    assert "p_corrected" in result


@pytest.mark.unit
def test_generate_statistical_comparison_table_no_seed():
    """Test that statistical table returns empty for single-seed data."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"experiment": "baseline__vanilla", "type": "baseline", "recall_1": 0.7},
            {
                "experiment": "ws__n100-gs3__topk__all",
                "type": "synthetic",
                "recall_1": 0.8,
            },
        ]
    )
    result = generate_statistical_comparison_table(df)
    assert result == ""


@pytest.mark.unit
def test_generate_classifier_table_with_mean_std():
    """Test classifier table with mean +/- std formatting."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "recall_1": 0.75,
                "recall_1_std": 0.03,
                "n_seeds": 5,
            },
        ]
    )
    result = generate_classifier_table(df)
    assert "baseline__vanilla" in result
    assert "+/-" in result


@pytest.mark.unit
def test_aggregate_multi_seed_sorts_by_seed():
    """Test that seeds are sorted numerically, not lexicographically."""
    import numpy as np
    import pandas as pd

    # Insert seeds out of order to verify sorting
    df = pd.DataFrame(
        [
            {"experiment": "exp-a", "seed": 10, "recall_1": 0.90},
            {"experiment": "exp-a", "seed": 2, "recall_1": 0.70},
            {"experiment": "exp-a", "seed": 1, "recall_1": 0.60},
        ]
    )

    aggregated = aggregate_multi_seed(df, ["recall_1"])
    assert list(aggregated["exp-a"]["_seeds"]) == [1, 2, 10]
    # Values should follow seed order, not insertion order
    np.testing.assert_array_equal(aggregated["exp-a"]["recall_1"], [0.60, 0.70, 0.90])


@pytest.mark.unit
def test_generate_statistical_comparison_table_mismatched_seeds():
    """Test that mismatched seeds between baseline and synthetic are skipped."""
    import pandas as pd

    rows = []
    # Baseline has seeds 0, 1, 2
    for seed, val in [(0, 0.70), (1, 0.72), (2, 0.74)]:
        rows.append(
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": seed,
                "recall_1": val,
            }
        )
    # Synthetic has seeds 1, 2, 3 (mismatched — seed 0 missing, seed 3 extra)
    for seed, val in [(1, 0.80), (2, 0.82), (3, 0.84)]:
        rows.append(
            {
                "experiment": "ws__n100-gs3__topk__all",
                "type": "synthetic",
                "seed": seed,
                "recall_1": val,
            }
        )

    df = pd.DataFrame(rows)
    result = generate_statistical_comparison_table(df, alpha=0.05)
    # Should return empty string because seed sets don't match
    assert result == ""


@pytest.mark.unit
def test_generate_statistical_comparison_table_matching_seeds():
    """Test that matching seeds produce a valid comparison table."""
    import pandas as pd

    rows = []
    for seed, bl, syn in [(0, 0.70, 0.80), (1, 0.72, 0.82), (2, 0.74, 0.84)]:
        rows.append(
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": seed,
                "recall_1": bl,
            }
        )
        rows.append(
            {
                "experiment": "ws__n100-gs3__topk__all",
                "type": "synthetic",
                "seed": seed,
                "recall_1": syn,
            }
        )

    df = pd.DataFrame(rows)
    result = generate_statistical_comparison_table(df, alpha=0.05)
    assert "ws__n100-gs3__topk__all" in result
    assert "p_corrected" in result


@pytest.mark.unit
def test_aggregate_multi_seed_skips_duplicate_seeds():
    """Test that experiments with duplicate seeds are excluded."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"experiment": "exp-a", "seed": 0, "recall_1": 0.70},
            {"experiment": "exp-a", "seed": 0, "recall_1": 0.72},  # duplicate seed
            {"experiment": "exp-a", "seed": 1, "recall_1": 0.68},
        ]
    )

    aggregated = aggregate_multi_seed(df, ["recall_1"])
    assert "exp-a" not in aggregated


@pytest.mark.unit
def test_aggregate_multi_seed_skips_metric_with_nan():
    """Test that a metric with NaN for any seed is excluded (not misaligned)."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        [
            {"experiment": "exp-a", "seed": 0, "recall_1": 0.70, "f1_1": 0.65},
            {"experiment": "exp-a", "seed": 1, "recall_1": float("nan"), "f1_1": 0.67},
            {"experiment": "exp-a", "seed": 2, "recall_1": 0.68, "f1_1": 0.63},
        ]
    )

    aggregated = aggregate_multi_seed(df, ["recall_1", "f1_1"])
    assert "exp-a" in aggregated
    # recall_1 has a NaN, so it should be excluded entirely
    assert "recall_1" not in aggregated["exp-a"]
    # f1_1 is complete, so it should be present with all 3 values
    assert "f1_1" in aggregated["exp-a"]
    np.testing.assert_array_equal(aggregated["exp-a"]["f1_1"], [0.65, 0.67, 0.63])


@pytest.mark.unit
def test_generate_statistical_comparison_table_invalid_alpha():
    """Test that invalid alpha values raise ValueError."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": 0,
                "recall_1": 0.7,
            },
            {
                "experiment": "baseline__vanilla",
                "type": "baseline",
                "seed": 1,
                "recall_1": 0.7,
            },
        ]
    )
    with pytest.raises(ValueError, match="alpha must be in"):
        generate_statistical_comparison_table(df, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        generate_statistical_comparison_table(df, alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        generate_statistical_comparison_table(df, alpha=-0.5)
