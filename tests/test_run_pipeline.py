"""Unit tests for scripts/run_pipeline.py job-matrix generation.

These validate the *mechanism* of build_classifier_jobs (counts, labels, out_dirs,
serialized overrides) on tiny synthetic configs — so editing the real pipeline matrix
requires no test changes. build_classifier_jobs is pure (no GPU/Docker/filesystem).
"""

import pytest

from scripts.run_pipeline import build_classifier_jobs


def _cfg(*, baselines, depths, balancing, seeds, baseline_phase=True, ft_phase=True):
    """Build a minimal cfg slice that build_classifier_jobs consumes."""
    return {
        "runner": {"classifier_output_root": "outputs/classifier"},
        "seeds": {"list": list(seeds)},
        "phases": {
            "baseline_classifier": baseline_phase,
            "ft_classifier": ft_phase,
        },
        "baselines": baselines,
        "ft": {
            "common": {"data.synthetic_augmentation.enabled": False},
            "depths": depths,
            "balancing": balancing,
        },
    }


def _single():
    return _cfg(
        baselines=[
            {
                "name": "baseline__vanilla",
                "overrides": {"data.synthetic_augmentation.enabled": False},
            },
        ],
        depths=[
            {
                "name": "ft-mixed7",
                "learning_rate": 0.0001,
                "overrides": {"model.initialization.trainable_layers": ["Mixed_7*"]},
            },
        ],
        balancing=[
            {
                "name": "ws",
                "overrides": {"data.balancing.weighted_sampler.enabled": True},
            },
        ],
        seeds=[0, 1],
    )


@pytest.mark.unit
class TestBuildClassifierJobs:
    def test_count_single_matrix(self):
        jobs = build_classifier_jobs(_single())
        # 1 baseline * 2 seeds + 1 depth * 1 balancing * 2 seeds = 4
        assert len(jobs) == 4

    @pytest.mark.parametrize(
        "n_base,n_depth,n_bal,n_seed", [(2, 2, 2, 2), (3, 3, 2, 5)]
    )
    def test_count_general_matrix(self, n_base, n_depth, n_bal, n_seed):
        cfg = _cfg(
            baselines=[
                {"name": f"baseline__{i}", "overrides": {}} for i in range(n_base)
            ],
            depths=[
                {"name": f"ft-{i}", "learning_rate": 0.0001, "overrides": {}}
                for i in range(n_depth)
            ],
            balancing=[{"name": f"b{i}", "overrides": {}} for i in range(n_bal)],
            seeds=range(n_seed),
        )
        jobs = build_classifier_jobs(cfg)
        assert len(jobs) == n_base * n_seed + n_depth * n_bal * n_seed

    def test_baseline_label_outdir_and_overrides(self):
        jobs = build_classifier_jobs(_single())
        label, out_dir, overrides = next(
            j for j in jobs if j[0].startswith("baseline__vanilla")
        )
        assert label == "baseline__vanilla/seed0"
        assert out_dir == "outputs/classifier/baseline__vanilla/seed0"
        assert overrides[:2] == ["--compute.seed", "0"]
        assert "--data.synthetic_augmentation.enabled" in overrides
        assert "false" in overrides

    def test_ft_label_outdir_and_lr_override(self):
        jobs = build_classifier_jobs(_single())
        label, out_dir, overrides = next(
            j for j in jobs if j[0].startswith("ft-mixed7__ws")
        )
        assert label == "ft-mixed7__ws/seed0"
        assert out_dir == "outputs/classifier/ft-mixed7__ws/seed0"
        assert overrides[:2] == ["--compute.seed", "0"]
        # ft.common is injected.
        assert "--data.synthetic_augmentation.enabled" in overrides
        # The per-depth learning rate is serialized as a CLI override.
        assert "--training.optimizer.learning_rate" in overrides
        lr_idx = overrides.index("--training.optimizer.learning_rate")
        assert overrides[lr_idx + 1] == "0.0001"
        # The depth's list-valued override is serialized as a Python literal.
        assert "--model.initialization.trainable_layers" in overrides
        tl_idx = overrides.index("--model.initialization.trainable_layers")
        assert overrides[tl_idx + 1] == "['Mixed_7*']"

    def test_baseline_phase_off_excludes_baselines(self):
        cfg = _single()
        cfg["phases"]["baseline_classifier"] = False
        jobs = build_classifier_jobs(cfg)
        assert all(not label.startswith("baseline__") for label, _, _ in jobs)
        assert len(jobs) == 2  # only ft

    def test_ft_phase_off_excludes_ft(self):
        cfg = _single()
        cfg["phases"]["ft_classifier"] = False
        jobs = build_classifier_jobs(cfg)
        assert all(
            "__ws/" not in label or label.startswith("baseline") for label, _, _ in jobs
        )
        assert len(jobs) == 2  # only baselines

    def test_both_phases_off_yields_no_jobs(self):
        cfg = _single()
        cfg["phases"]["baseline_classifier"] = False
        cfg["phases"]["ft_classifier"] = False
        assert build_classifier_jobs(cfg) == []

    def test_global_overrides_not_in_train_overrides(self):
        """Checkpoint/runtime overrides are appended later, not by build_classifier_jobs."""
        jobs = build_classifier_jobs(_single())
        for _, _, overrides in jobs:
            assert "--training.checkpointing.save_optimizer" not in overrides
            assert "--data.loading.num_workers" not in overrides


@pytest.mark.unit
def test_build_classifier_jobs_runs_without_docker_or_gpu():
    """The module imports and build_classifier_jobs runs purely (no Docker/GPU calls).

    The top-of-file import already proves the module loads in the CPU test environment;
    this confirms the matrix expansion is side-effect free.
    """
    jobs = build_classifier_jobs(_single())
    assert isinstance(jobs, list) and jobs
