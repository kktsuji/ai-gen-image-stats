"""Unit tests for scripts/run_pipeline.py job-matrix generation.

These validate the *mechanism* of build_classifier_jobs (counts, labels, out_dirs,
serialized overrides) on tiny synthetic configs — so editing the real pipeline matrix
requires no test changes. build_classifier_jobs is pure (no GPU/Docker/filesystem).
"""

import pytest

import scripts.run_pipeline as rp
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


# ---------------------------------------------------------------------------
# Fix #2: orchestration-layer coverage (skip gate + train -> eval -> cleanup).
# All Docker/subprocess/filesystem I/O is mocked; no GPU or containers involved.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsDone:
    def test_false_when_skip_disabled(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", {"runner": {"skip_completed": False}})
        monkeypatch.setattr(rp.os.path, "exists", lambda p: True)
        # All markers present, but skip_completed=False means never "done".
        assert rp._is_done("a", "b") is False

    def test_true_when_all_markers_exist(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", {"runner": {"skip_completed": True}})
        monkeypatch.setattr(rp.os.path, "exists", lambda p: True)
        assert rp._is_done("a", "b") is True

    def test_false_when_any_marker_missing(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", {"runner": {"skip_completed": True}})
        monkeypatch.setattr(rp.os.path, "exists", lambda p: p != "b")
        assert rp._is_done("a", "b") is False


def _exp_cfg(evaluation_splits=None):
    """Minimal CFG slice consumed by _run_classifier_experiment."""
    return {
        "configs": {"classifier": "configs/classifier.yaml"},
        "phases": {"evaluation": True},
        "runner": {
            "skip_completed": False,
            "delete_checkpoints_after_eval": True,
            "evaluation_splits": evaluation_splits or ["test"],
        },
        "classifier_overrides": {"checkpoint": {}, "runtime": {}},
    }


@pytest.mark.unit
class TestRunClassifierExperiment:
    def test_training_failure_skips_eval(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", _exp_cfg())
        calls = []
        monkeypatch.setattr(
            rp,
            "_throttled_run_and_wait",
            lambda config, overrides, **kw: (calls.append(overrides), 1)[1],
        )
        monkeypatch.setattr(rp.os.path, "exists", lambda p: False)

        rp._run_classifier_experiment("lbl", "out", ["--compute.seed", "0"])
        # Training rc != 0 -> early return; evaluation never launched.
        assert len(calls) == 1

    def test_train_eval_cleanup(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", _exp_cfg())
        calls = []
        monkeypatch.setattr(
            rp,
            "_throttled_run_and_wait",
            lambda config, overrides, **kw: (calls.append(overrides), 0)[1],
        )
        # best_model.pth "exists" so eval discovers it; nothing else exists.
        monkeypatch.setattr(
            rp.os.path, "exists", lambda p: p.endswith("best_model.pth")
        )
        monkeypatch.setattr(rp.os.path, "isdir", lambda p: True)
        removed = []
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: removed.append(p))

        rp._run_classifier_experiment("lbl", "out", [])

        assert len(calls) == 2  # train + eval
        eval_call = calls[1]
        assert "--mode" in eval_call and "evaluate" in eval_call
        assert any("best_model.pth" in tok for tok in eval_call)
        # Successful eval -> checkpoints cleaned up.
        assert removed == ["out/checkpoints"]

    def test_eval_failure_keeps_checkpoints(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", _exp_cfg())

        def fake_run(config, overrides, **kw):
            # Training (no --mode) succeeds; evaluation (--mode) fails.
            return 1 if "--mode" in overrides else 0

        monkeypatch.setattr(rp, "_throttled_run_and_wait", fake_run)
        monkeypatch.setattr(
            rp.os.path, "exists", lambda p: p.endswith("best_model.pth")
        )
        monkeypatch.setattr(rp.os.path, "isdir", lambda p: True)
        removed = []
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: removed.append(p))

        rp._run_classifier_experiment("lbl", "out", [])
        # Eval failed -> checkpoints retained for retry.
        assert removed == []

    def test_skips_when_already_complete(self, monkeypatch):
        cfg = _exp_cfg()
        cfg["runner"]["skip_completed"] = True
        monkeypatch.setattr(rp, "CFG", cfg)
        calls = []
        monkeypatch.setattr(
            rp, "_throttled_run_and_wait", lambda *a, **k: calls.append(1) or 0
        )
        # eval marker present -> fully skip (no container launched).
        monkeypatch.setattr(rp.os.path, "exists", lambda p: True)

        rp._run_classifier_experiment("lbl", "out", [])
        assert calls == []

    def test_no_eval_phase_trains_only(self, monkeypatch):
        cfg = _exp_cfg()
        cfg["phases"]["evaluation"] = False
        monkeypatch.setattr(rp, "CFG", cfg)
        calls = []
        monkeypatch.setattr(
            rp,
            "_throttled_run_and_wait",
            lambda config, overrides, **kw: (calls.append(overrides), 0)[1],
        )
        monkeypatch.setattr(rp.os.path, "exists", lambda p: False)
        removed = []
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: removed.append(p))

        rp._run_classifier_experiment("lbl", "out", [])
        assert len(calls) == 1  # train only, no eval
        assert removed == []  # no cleanup without a successful eval

    def test_falls_back_to_final_model(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", _exp_cfg())
        calls = []
        monkeypatch.setattr(
            rp,
            "_throttled_run_and_wait",
            lambda config, overrides, **kw: (calls.append(overrides), 0)[1],
        )
        # No best_model.pth; only final_model.pth exists -> eval uses the fallback.
        monkeypatch.setattr(
            rp.os.path, "exists", lambda p: p.endswith("final_model.pth")
        )
        monkeypatch.setattr(rp.os.path, "isdir", lambda p: True)
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: None)

        rp._run_classifier_experiment("lbl", "out", [])
        eval_call = calls[1]
        assert any("final_model.pth" in tok for tok in eval_call)

    def test_two_pass_val_test_eval(self, monkeypatch):
        # evaluation_splits=[val, test] -> train + one eval per split, each with its
        # own --evaluation.split appended last, then cleanup once both succeed.
        monkeypatch.setattr(rp, "CFG", _exp_cfg(evaluation_splits=["val", "test"]))
        calls = []
        monkeypatch.setattr(
            rp,
            "_throttled_run_and_wait",
            lambda config, overrides, **kw: (calls.append(overrides), 0)[1],
        )
        monkeypatch.setattr(
            rp.os.path, "exists", lambda p: p.endswith("best_model.pth")
        )
        monkeypatch.setattr(rp.os.path, "isdir", lambda p: True)
        removed = []
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: removed.append(p))

        rp._run_classifier_experiment("lbl", "out", [])

        assert len(calls) == 3  # train + val eval + test eval
        # Each eval pass ends with its own --evaluation.split (appended last so it
        # wins over any runtime evaluation.split).
        assert calls[1][-2:] == ["--evaluation.split", "val"]
        assert calls[2][-2:] == ["--evaluation.split", "test"]
        # Cleanup happens only after BOTH splits succeed.
        assert removed == ["out/checkpoints"]

    def test_two_pass_first_split_failure_keeps_checkpoints(self, monkeypatch):
        # If the val pass fails, the test pass is not attempted and checkpoints are
        # retained for a retry.
        monkeypatch.setattr(rp, "CFG", _exp_cfg(evaluation_splits=["val", "test"]))
        calls = []

        def fake_run(config, overrides, **kw):
            calls.append(overrides)
            # Training (no --mode) succeeds; the first eval (val) fails.
            if "--mode" in overrides:
                return 1
            return 0

        monkeypatch.setattr(rp, "_throttled_run_and_wait", fake_run)
        monkeypatch.setattr(
            rp.os.path, "exists", lambda p: p.endswith("best_model.pth")
        )
        monkeypatch.setattr(rp.os.path, "isdir", lambda p: True)
        removed = []
        monkeypatch.setattr(rp.shutil, "rmtree", lambda p: removed.append(p))

        rp._run_classifier_experiment("lbl", "out", [])

        assert len(calls) == 2  # train + val eval (which failed); test never attempted
        assert removed == []


@pytest.mark.unit
class TestRunOutputStreams:
    """run() stream wiring: suppressed runs hide stdout but keep stderr for crashes."""

    def _cfg(self):
        return {"docker": {"shm_size": "4g", "image": "img"}}

    def _capture_popen(self, monkeypatch):
        captured = {}

        class _FakeProc:
            pass

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(rp, "CFG", self._cfg())
        monkeypatch.setattr(rp.subprocess, "Popen", fake_popen)
        return captured

    def test_suppressed_keeps_stderr_drops_stdout(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run("configs/classifier.yaml", [], suppress_output=True)
        # stdout silenced to avoid interleaving concurrent runs...
        assert captured["kwargs"]["stdout"] is rp.subprocess.DEVNULL
        # ...but stderr is kept so a container crash still surfaces.
        assert captured["kwargs"]["stderr"] is rp.sys.stderr

    def test_unsuppressed_inherits_both_streams(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run("configs/classifier.yaml", [], suppress_output=False)
        # Foreground run streams both stdout and stderr (no redirection kwargs).
        assert "stdout" not in captured["kwargs"]
        assert "stderr" not in captured["kwargs"]

    def test_docker_mode_disable_notifications_uses_env_flag_not_env_kwarg(
        self, monkeypatch
    ):
        """Regression: Docker suppression blanks the webhook via `-e`, not env kwarg."""
        captured = self._capture_popen(monkeypatch)
        monkeypatch.setattr(rp, "LOCAL_MODE", False)
        rp.run("configs/classifier.yaml", [], disable_notifications=True)
        cmd = captured["cmd"]
        # Blanked via a container -e flag, immediately before the image.
        assert cmd[:2] == ["docker", "run"]
        assert "-e" in cmd and "SLACK_WEBHOOK_URL=" in cmd
        # No child-env override in Docker mode.
        assert "env" not in captured["kwargs"]


@pytest.mark.unit
class TestRunLocalMode:
    """run() local (--local / --no-docker) branch: launch src.main via this venv."""

    def _capture_popen(self, monkeypatch):
        captured = {}

        class _FakeProc:
            pass

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _FakeProc()

        # docker.* is unused in local mode, but keep a valid CFG so an accidental read
        # would not KeyError silently.
        monkeypatch.setattr(rp, "CFG", {"docker": {"shm_size": "4g", "image": "img"}})
        monkeypatch.setattr(rp, "LOCAL_MODE", True)
        monkeypatch.setattr(rp.subprocess, "Popen", fake_popen)
        return captured

    def test_local_uses_sys_executable_no_docker(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run("configs/classifier.yaml", ["--compute.seed", "0"])
        cmd = captured["cmd"]
        assert cmd[0] == rp.sys.executable
        assert "docker" not in cmd
        # src.main is invoked directly with the config and overrides appended.
        assert cmd[1:] == [
            "-m",
            "src.main",
            "configs/classifier.yaml",
            "--compute.seed",
            "0",
        ]

    def test_local_blanks_webhook_when_disabled(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run("configs/classifier.yaml", [], disable_notifications=True)
        # Suppression is done via the child env, not a container -e flag.
        assert captured["kwargs"]["env"]["SLACK_WEBHOOK_URL"] == ""
        assert "SLACK_WEBHOOK_URL=" not in captured["cmd"]

    def test_local_no_env_when_notifications_enabled(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run("configs/classifier.yaml", [], disable_notifications=False)
        assert "env" not in captured["kwargs"]

    def test_local_suppress_output_coexists_with_env(self, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        rp.run(
            "configs/classifier.yaml",
            [],
            suppress_output=True,
            disable_notifications=True,
        )
        kwargs = captured["kwargs"]
        # Stream wiring preserved (stdout dropped, stderr kept) alongside the env blank.
        assert kwargs["stdout"] is rp.subprocess.DEVNULL
        assert kwargs["stderr"] is rp.sys.stderr
        assert kwargs["env"]["SLACK_WEBHOOK_URL"] == ""


@pytest.mark.unit
class TestParseShmSize:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("4g", 4 * 1024**3),
            ("512m", 512 * 1024**2),
            ("1024k", 1024 * 1024),
            ("2G", 2 * 1024**3),
            ("1048576", 1048576),  # bare bytes, no suffix
            ("2.5g", int(2.5 * 1024**3)),
        ],
    )
    def test_parses_units(self, value, expected):
        assert rp._parse_shm_size(value) == expected

    @pytest.mark.parametrize("value", ["", "  ", "abc", "g", "4x"])
    def test_unparseable_returns_zero(self, value):
        # 0 signals "no threshold" so the shm warning is skipped rather than crashing.
        assert rp._parse_shm_size(value) == 0


@pytest.mark.unit
class TestPreflightLocalChecks:
    """--local preflight: fail fast on a depless interpreter, warn on a small /dev/shm."""

    def _cfg(self):
        return {"docker": {"shm_size": "4g", "image": "img"}}

    def test_raises_when_torch_missing(self, monkeypatch):
        monkeypatch.setattr(rp, "CFG", self._cfg())
        # Interpreter cannot import torch -> jobs would ModuleNotFoundError; fail early.
        monkeypatch.setattr(rp.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(SystemExit) as exc:
            rp._preflight_local_checks()
        assert "torch" in str(exc.value)

    def test_warns_when_shm_smaller_than_configured(self, monkeypatch, capsys):
        monkeypatch.setattr(rp, "CFG", self._cfg())
        monkeypatch.setattr(rp.importlib.util, "find_spec", lambda name: object())
        # /dev/shm is 1 GiB but config requests 4g -> warn.
        monkeypatch.setattr(
            rp.shutil, "disk_usage", lambda p: type("U", (), {"total": 1024**3})()
        )
        rp._preflight_local_checks()
        out = capsys.readouterr().out
        assert "WARNING" in out and "Bus error" in out

    def test_no_warn_when_shm_sufficient(self, monkeypatch, capsys):
        monkeypatch.setattr(rp, "CFG", self._cfg())
        monkeypatch.setattr(rp.importlib.util, "find_spec", lambda name: object())
        # /dev/shm is 8 GiB >= 4g -> no warning.
        monkeypatch.setattr(
            rp.shutil, "disk_usage", lambda p: type("U", (), {"total": 8 * 1024**3})()
        )
        rp._preflight_local_checks()
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_no_warn_when_shm_probe_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(rp, "CFG", self._cfg())
        monkeypatch.setattr(rp.importlib.util, "find_spec", lambda name: object())

        def _raise(_):
            raise OSError("no /dev/shm")

        monkeypatch.setattr(rp.shutil, "disk_usage", _raise)
        # Probe failure is non-fatal: torch present, so no exit and no shm warning.
        rp._preflight_local_checks()
        out = capsys.readouterr().out
        assert "WARNING" not in out
