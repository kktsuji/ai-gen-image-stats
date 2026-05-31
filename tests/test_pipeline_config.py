"""Unit tests for scripts/pipeline_config.py.

These validate the loader/validator *mechanism* (schema rules, seed normalization),
never the content of configs/pipeline.yaml — so editing the pipeline matrix requires no
test changes.
"""

import copy

import pytest
import yaml

from scripts.pipeline_config import (
    load_pipeline_config,
    resolve_seeds,
    validate_pipeline_config,
)


def _valid_config():
    """A minimal, schema-valid pipeline config dict."""
    return {
        "experiment": "pipeline",
        "phases": {
            "data_preparation": False,
            "baseline_classifier": True,
            "ft_classifier": True,
            "evaluation": True,
            "summarize": True,
        },
        "runner": {
            "skip_completed": True,
            "delete_checkpoints_after_eval": True,
            "gpu_cooldown_seconds": 5,
            "classifier_notify_every": 10,
            "classifier_max_parallel": 3,
            "classifier_output_root": "outputs/classifier",
        },
        "configs": {
            "data_preparation": "configs/data-preparation.yaml",
            "classifier": "configs/classifier.yaml",
        },
        "docker": {
            "image": "example/image:tag",
            "shm_size": "4g",
        },
        "seeds": {"range": [0, 3]},
        "classifier_overrides": {
            "checkpoint": {"training.checkpointing.save_optimizer": False},
            "runtime": {"data.loading.num_workers": 2},
        },
        "baselines": [
            {
                "name": "baseline__ws",
                "overrides": {"data.balancing.weighted_sampler.enabled": True},
            },
        ],
        "ft": {
            "common": {"data.synthetic_augmentation.enabled": False},
            "depths": [
                {
                    "name": "ft-mixed7",
                    "learning_rate": 0.0001,
                    "overrides": {"model.initialization.freeze_backbone": True},
                },
            ],
            "balancing": [
                {
                    "name": "ws",
                    "overrides": {"data.balancing.weighted_sampler.enabled": True},
                },
            ],
        },
        "summarize": {
            "base_dir": "outputs/classifier",
            "output_dir": "outputs/evaluation_report",
            "baseline_name": "baseline__ws",
        },
    }


@pytest.mark.unit
class TestValidateAccepts:
    def test_minimal_valid_config(self):
        validate_pipeline_config(_valid_config())

    def test_seeds_list_form(self):
        cfg = _valid_config()
        cfg["seeds"] = {"list": [0, 2, 4]}
        validate_pipeline_config(cfg)

    def test_learning_rate_as_string(self):
        cfg = _valid_config()
        cfg["ft"]["depths"][0]["learning_rate"] = "1e-4"
        validate_pipeline_config(cfg)

    def test_baseline_name_not_required_when_baselines_phase_off(self):
        cfg = _valid_config()
        cfg["phases"]["baseline_classifier"] = False
        cfg["summarize"]["baseline_name"] = "preexisting__baseline"
        # Warns but does not raise.
        validate_pipeline_config(cfg)


@pytest.mark.unit
class TestValidateRejects:
    def test_wrong_experiment(self):
        cfg = _valid_config()
        cfg["experiment"] = "classifier"
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_missing_phase(self):
        cfg = _valid_config()
        del cfg["phases"]["evaluation"]
        with pytest.raises(KeyError):
            validate_pipeline_config(cfg)

    def test_non_bool_phase(self):
        cfg = _valid_config()
        cfg["phases"]["evaluation"] = "yes"
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_seeds_both_forms(self):
        cfg = _valid_config()
        cfg["seeds"] = {"range": [0, 3], "list": [0, 1]}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_seeds_neither_form(self):
        cfg = _valid_config()
        cfg["seeds"] = {}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_seeds_range_stop_not_greater_than_start(self):
        cfg = _valid_config()
        cfg["seeds"] = {"range": [5, 5]}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_seeds_negative(self):
        cfg = _valid_config()
        cfg["seeds"] = {"list": [-1, 0]}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_seeds_empty_list(self):
        cfg = _valid_config()
        cfg["seeds"] = {"list": []}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_runner_negative_cooldown(self):
        cfg = _valid_config()
        cfg["runner"]["gpu_cooldown_seconds"] = -1
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_runner_zero_parallel(self):
        cfg = _valid_config()
        cfg["runner"]["classifier_max_parallel"] = 0
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_config_path_not_yaml(self):
        cfg = _valid_config()
        cfg["configs"]["classifier"] = "configs/classifier.json"
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_duplicate_baseline_names(self):
        cfg = _valid_config()
        cfg["baselines"].append(
            {
                "name": "baseline__ws",
                "overrides": {"data.balancing.upsampling.enabled": True},
            }
        )
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_duplicate_depth_names(self):
        cfg = _valid_config()
        cfg["ft"]["depths"].append(
            {"name": "ft-mixed7", "learning_rate": 0.0001, "overrides": {}}
        )
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_non_positive_learning_rate(self):
        cfg = _valid_config()
        cfg["ft"]["depths"][0]["learning_rate"] = 0
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_summarize_baseline_not_in_baselines(self):
        cfg = _valid_config()
        cfg["summarize"]["baseline_name"] = "baseline__missing"
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_driver_owned_key_in_overrides(self):
        cfg = _valid_config()
        cfg["baselines"][0]["overrides"]["compute.seed"] = 1
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_learning_rate_as_override_rejected(self):
        cfg = _valid_config()
        cfg["ft"]["depths"][0]["overrides"]["training.optimizer.learning_rate"] = 0.1
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)

    def test_nested_dict_override_rejected(self):
        cfg = _valid_config()
        cfg["baselines"][0]["overrides"] = {"data": {"balancing": {"enabled": True}}}
        with pytest.raises(ValueError):
            validate_pipeline_config(cfg)


@pytest.mark.unit
class TestResolveSeeds:
    def test_range_expands(self):
        assert resolve_seeds({"range": [0, 3]}) == [0, 1, 2]

    def test_range_with_step(self):
        assert resolve_seeds({"range": [0, 6, 2]}) == [0, 2, 4]

    def test_list_passthrough(self):
        assert resolve_seeds({"list": [0, 5, 9]}) == [0, 5, 9]


@pytest.mark.unit
class TestLoadPipelineConfig:
    def test_normalizes_seeds_to_list(self, tmp_path):
        cfg = _valid_config()
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))
        loaded = load_pipeline_config(str(path))
        assert loaded["seeds"] == {"list": [0, 1, 2]}

    def test_list_form_preserved(self, tmp_path):
        cfg = _valid_config()
        cfg["seeds"] = {"list": [3, 7]}
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))
        loaded = load_pipeline_config(str(path))
        assert loaded["seeds"] == {"list": [3, 7]}

    def test_invalid_config_raises(self, tmp_path):
        cfg = _valid_config()
        cfg["experiment"] = "nope"
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError):
            load_pipeline_config(str(path))

    def test_does_not_mutate_other_keys(self, tmp_path):
        cfg = _valid_config()
        expected_baselines = copy.deepcopy(cfg["baselines"])
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))
        loaded = load_pipeline_config(str(path))
        assert loaded["baselines"] == expected_baselines
