"""Tests for scripts/run_campaign.py (campaign driver)."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts import run_campaign as rc
from tests.test_campaign_config import _base, _sweep


def _fake_process(returncode, stdout="", stderr="", before=None):
    """subprocess.run stand-in that writes to the file objects it is given,
    like a real child process does."""

    def run(cmd, **kwargs):
        if before is not None:
            before()
        kwargs["stdout"].write(stdout.encode("utf-8"))
        kwargs["stderr"].write(stderr.encode("utf-8"))
        return subprocess.CompletedProcess(cmd, returncode)

    return run


def _make_campaign(root: Path, sweep=None) -> Path:
    """Series folder with shared/ and one campaign; returns the campaign dir."""
    campaign = root / "series" / "01-test"
    (campaign / "configs").mkdir(parents=True)
    (root / "series" / "shared" / "splits").mkdir(parents=True)
    (campaign / "configs" / "sweep.yaml").write_text(yaml.safe_dump(sweep or _sweep()))
    (campaign / "configs" / "base.yaml").write_text(yaml.safe_dump(_base()))
    return campaign


# A stand-in for the Python interpreter: invoked as `<fake> -m src.main <config>`,
# it records the config it received and writes the done marker (or fails when
# the run name contains "maha", to exercise failure handling).
FAKE_PYTHON = f"""#!{sys.executable}
import json, sys, yaml, pathlib
cfg_path = sys.argv[3]
cfg = yaml.safe_load(open(cfg_path))
out = pathlib.Path(cfg["output"]["base_dir"])
if "maha" in str(out):
    sys.stderr.write("boom\\n")
    sys.exit(3)
(out / "reports").mkdir(parents=True, exist_ok=True)
(out / "reports" / "evaluation.json").write_text(json.dumps(cfg))
"""


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / "fake_python"
    path.write_text(FAKE_PYTHON)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.mark.unit
class TestExpandOnly:
    def test_writes_portable_configs_and_runs_nothing(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        with patch.object(rc, "run_one") as mock_run:
            summary = rc.run_campaign(campaign, expand_only=True)
        mock_run.assert_not_called()
        assert summary["runs"] == 16 and summary["conditions"] == 4
        stored = yaml.safe_load(
            (campaign / "configs/runs/ad-knn-rn50/split1_seed0.yaml").read_text()
        )
        assert stored["data"]["split_file"] == "../shared/splits/s1.json"
        assert stored["output"]["base_dir"] == "runs/split1/ad-frozen/ad-knn-rn50/seed0"

    def test_only_filters_and_done_runs_skipped(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        done = campaign / "runs/split0/ad-frozen/ad-knn-rn50/seed0/reports"
        done.mkdir(parents=True)
        (done / "evaluation.json").write_text("{}")
        summary = rc.run_campaign(campaign, expand_only=True, only="knn-rn50")
        assert summary["selected"] == 4
        assert summary["skipped_done"] == 1

    def test_force_includes_done_runs(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        done = campaign / "runs/split0/ad-frozen/ad-knn-rn50/seed0/reports"
        done.mkdir(parents=True)
        (done / "evaluation.json").write_text("{}")
        with patch.object(rc, "run_one", return_value=True) as mock_run:
            summary = rc.run_campaign(campaign, only="knn-rn50", force=True)
        assert mock_run.call_count == 4
        assert summary["succeeded"] == 4 and summary["skipped_done"] == 0


@pytest.mark.unit
class TestRunOne:
    def test_resolved_config_passed_and_temp_removed(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        _, runs = rc.load_campaign(campaign)
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["cwd"] = kwargs["cwd"]
            seen["config"] = yaml.safe_load(Path(cmd[3]).read_text())

            class Result:
                returncode = 0
                stderr = ""

            return Result()

        with patch.object(rc.subprocess, "run", side_effect=fake_run):
            ok = rc.run_one(campaign.resolve(), runs[0], ["data.split_file"])

        assert ok
        assert seen["cmd"][1:3] == ["-m", "src.main"]
        assert seen["cwd"] == rc.REPO_ROOT
        assert seen["config"]["output"]["base_dir"] == str(
            campaign.resolve() / runs[0].output_dir
        )
        assert seen["config"]["data"]["split_file"] == str(
            (campaign / "../shared/splits/s0.json").resolve()
        )
        assert not Path(seen["cmd"][3]).exists()


@pytest.mark.unit
class TestFailureReporting:
    @pytest.mark.parametrize(
        "stdout,stderr,expected",
        [
            ("line1\nRuntimeError: CUDA out of memory\n", "", "CUDA out of memory"),
            # A noisy stderr (warnings) must not hide the error logged to stdout.
            (
                "INFO start\nERROR Experiment failed: CUDA out of memory\n",
                "UserWarning: deprecated\n100%|####| 10/10\n",
                "ERROR Experiment failed: CUDA out of memory",
            ),
        ],
    )
    def test_error_tail_and_log_dir_reported(
        self, tmp_path, caplog, stdout, stderr, expected
    ):
        campaign = _make_campaign(tmp_path)
        _, runs = rc.load_campaign(campaign)

        with (
            patch.object(
                rc.subprocess, "run", side_effect=_fake_process(1, stdout, stderr)
            ),
            caplog.at_level("ERROR"),
        ):
            ok = rc.run_one(campaign.resolve(), runs[0], [])
        assert not ok
        assert expected in caplog.text
        assert "run logs:" in caplog.text and runs[0].output_dir in caplog.text


@pytest.mark.unit
class TestTailText:
    def test_only_last_bytes_read(self):
        import tempfile

        with tempfile.TemporaryFile() as f:
            f.write(b"A" * 1000 + b"END")
            assert rc._tail_text(f, max_bytes=10) == "AAAAAAAEND"
            assert rc._tail_text(f).endswith("END") and len(rc._tail_text(f)) == 1003

    def test_invalid_utf8_replaced(self):
        import tempfile

        with tempfile.TemporaryFile() as f:
            f.write("é".encode("utf-8")[1:] + b"ok")
            assert rc._tail_text(f).endswith("ok")


@pytest.mark.unit
class TestFormatFailureOutput:
    def test_stdout_first_then_stderr(self):
        out = rc.format_failure_output("a\nb\nERROR boom\n", "warn1\nwarn2\n")
        assert out.index("ERROR boom") < out.index("warn2")
        assert out.splitlines()[0] == "--- stdout (last 3 lines) ---"

    def test_tails_are_bounded(self):
        out = rc.format_failure_output(
            "\n".join(f"o{i}" for i in range(20)),
            "\n".join(f"e{i}" for i in range(20)),
        )
        assert "o11" not in out and "o12" in out and "o19" in out
        assert "e15" not in out and "e16" in out

    def test_no_output(self):
        assert rc.format_failure_output(None, "") == "(no output)"


def _done(campaign: Path, output_dir: str) -> None:
    reports = campaign / output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "evaluation.json").write_text("{}")


@pytest.mark.unit
class TestStoredConfigsAndStaleRuns:
    def _edit_base(self, campaign: Path, normal_pool: str) -> None:
        base = _base()
        base["data"]["normal_pool"] = normal_pool
        (campaign / "configs" / "base.yaml").write_text(yaml.safe_dump(base))

    def test_completed_run_config_kept_and_flagged_stale(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        rc.run_campaign(campaign, expand_only=True)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        stored = campaign / "configs/runs/ad-knn-rn50/split0_seed0.yaml"
        pending = campaign / "configs/runs/ad-knn-rn50/split1_seed0.yaml"

        self._edit_base(campaign, "suspicious")
        summary = rc.run_campaign(campaign, expand_only=True)

        assert summary["stale"] == ["configs/runs/ad-knn-rn50/split0_seed0.yaml"]
        assert yaml.safe_load(stored.read_text())["data"]["normal_pool"] == "all"
        assert (
            yaml.safe_load(pending.read_text())["data"]["normal_pool"] == "suspicious"
        )

    def test_unchanged_sweep_has_no_stale_runs(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        rc.run_campaign(campaign, expand_only=True)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        assert rc.run_campaign(campaign, expand_only=True)["stale"] == []

    def test_completed_run_without_stored_config_gets_one(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        summary = rc.run_campaign(campaign, expand_only=True)
        assert summary["stale"] == []
        assert (campaign / "configs/runs/ad-knn-rn50/split0_seed0.yaml").exists()

    def test_force_overwrites_only_rerun_selection(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        rc.run_campaign(campaign, expand_only=True)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        _done(campaign, "runs/split0/ad-frozen/ad-maha-rn50/seed0")
        self._edit_base(campaign, "suspicious")
        with patch.object(rc, "run_one", return_value=True):
            summary = rc.run_campaign(campaign, only="knn-rn50", force=True)

        knn = campaign / "configs/runs/ad-knn-rn50/split0_seed0.yaml"
        maha = campaign / "configs/runs/ad-maha-rn50/split0_seed0.yaml"
        assert yaml.safe_load(knn.read_text())["data"]["normal_pool"] == "suspicious"
        assert yaml.safe_load(maha.read_text())["data"]["normal_pool"] == "all"
        assert summary["stale"] == ["configs/runs/ad-maha-rn50/split0_seed0.yaml"]

    def test_main_returns_nonzero_when_stale(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        rc.run_campaign(campaign, expand_only=True)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        self._edit_base(campaign, "suspicious")
        assert rc.main([str(campaign), "--expand-only"]) == 1


@pytest.mark.unit
class TestForceAndStaleEdgeCases:
    def _setup_stale(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        rc.run_campaign(campaign, expand_only=True)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        base = _base()
        base["data"]["normal_pool"] = "suspicious"
        (campaign / "configs" / "base.yaml").write_text(yaml.safe_dump(base))
        return campaign

    def test_expand_only_with_force_keeps_records(self, tmp_path):
        campaign = self._setup_stale(tmp_path)
        stored = campaign / "configs/runs/ad-knn-rn50/split0_seed0.yaml"
        marker = (
            campaign / "runs/split0/ad-frozen/ad-knn-rn50/seed0/reports/evaluation.json"
        )

        summary = rc.run_campaign(campaign, expand_only=True, force=True)

        assert yaml.safe_load(stored.read_text())["data"]["normal_pool"] == "all"
        assert summary["stale"] == ["configs/runs/ad-knn-rn50/split0_seed0.yaml"]
        assert marker.exists()

    def test_failed_forced_rerun_is_retried_later(self, tmp_path):
        campaign = self._setup_stale(tmp_path)
        marker = (
            campaign / "runs/split0/ad-frozen/ad-knn-rn50/seed0/reports/evaluation.json"
        )
        target = ("ad-knn-rn50", 0, 0)

        with patch.object(rc, "run_one", return_value=False):
            rc.run_campaign(campaign, only="knn-rn50", force=True)
        assert not marker.exists()  # the old result no longer counts as done

        seen = []
        with patch.object(
            rc,
            "run_one",
            side_effect=lambda c, r, k, **kw: (
                seen.append((r.name, r.split, r.seed)) or True
            ),
        ):
            summary = rc.run_campaign(campaign, only="knn-rn50")
        assert target in seen
        assert summary["stale"] == []

    def test_stale_outside_only_selection_does_not_fail(self, tmp_path):
        campaign = self._setup_stale(tmp_path)  # stale run is ad-knn-rn50
        with patch.object(rc, "run_one", return_value=True):
            assert rc.main([str(campaign), "--only", "maha"]) == 0
            summary = rc.run_campaign(campaign, only="maha")
        assert summary["stale"] == ["configs/runs/ad-knn-rn50/split0_seed0.yaml"]
        assert summary["stale_selected"] == []

    def test_stale_inside_only_selection_fails(self, tmp_path):
        campaign = self._setup_stale(tmp_path)
        with patch.object(rc, "run_one", return_value=True):
            assert rc.main([str(campaign), "--only", "knn-rn50"]) == 1


@pytest.mark.unit
class TestFinalReviewFixes:
    def test_failed_run_marker_removed(self, tmp_path):
        """A run that wrote its done marker and then failed must not count as done."""
        campaign = _make_campaign(tmp_path)
        _, runs = rc.load_campaign(campaign)
        run = runs[0]
        marker = campaign.resolve() / run.output_dir / "reports/evaluation.json"

        def write_marker():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}")  # written before the failure

        fake_run = _fake_process(1, "ERROR late failure", "", before=write_marker)

        with patch.object(rc.subprocess, "run", side_effect=fake_run):
            ok = rc.run_one(
                campaign.resolve(), run, [], done_marker="reports/evaluation.json"
            )
        assert not ok
        assert not marker.exists()

    def test_successful_run_keeps_marker(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        _, runs = rc.load_campaign(campaign)
        run = runs[0]
        marker = campaign.resolve() / run.output_dir / "reports/evaluation.json"

        def fake_run(cmd, **kwargs):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(rc.subprocess, "run", side_effect=fake_run):
            assert rc.run_one(
                campaign.resolve(), run, [], done_marker="reports/evaluation.json"
            )
        assert marker.exists()

    def test_expand_only_force_counts_done_runs_as_skipped(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        _done(campaign, "runs/split0/ad-frozen/ad-knn-rn50/seed0")
        summary = rc.run_campaign(campaign, expand_only=True, force=True)
        assert summary["skipped_done"] == 1
        assert summary["selected"] == 16


@pytest.mark.unit
class TestInterpreterNotStartable:
    def test_missing_python_is_a_run_failure(self, tmp_path, caplog):
        campaign = _make_campaign(tmp_path)
        _, runs = rc.load_campaign(campaign)
        run = runs[0]
        marker = campaign.resolve() / run.output_dir / "reports/evaluation.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
        with caplog.at_level("ERROR"):
            ok = rc.run_one(
                campaign.resolve(),
                run,
                [],
                python=str(tmp_path / "no-such-python"),
                done_marker="reports/evaluation.json",
            )
        assert not ok
        assert "could not start" in caplog.text
        assert not marker.exists()

    def test_campaign_continues_after_unstartable_run(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        orig = rc.run_one
        with patch.object(
            rc,
            "run_one",
            side_effect=lambda c, r, k, **kw: orig(
                c, r, k, python=str(tmp_path / "no-such-python"), **kw
            ),
        ):
            summary = rc.run_campaign(campaign, only="knn-rn50")
        assert len(summary["failed"]) == 4 and summary["succeeded"] == 0


@pytest.mark.unit
class TestSplitKeyAlwaysResolved:
    def test_split_key_resolved_without_path_keys(self, tmp_path):
        sweep = _sweep()
        sweep["path_keys"] = ["feature_extraction.cache_dir"]  # split key omitted
        campaign = _make_campaign(tmp_path, sweep=sweep)
        seen = []
        with patch.object(
            rc, "run_one", side_effect=lambda c, r, k, **kw: seen.append(k) or True
        ):
            rc.run_campaign(campaign, only="knn-rn50")
        assert seen and all(k[0] == "data.split_file" for k in seen)
        assert all(k.count("data.split_file") == 1 for k in seen)


@pytest.mark.component
class TestEndToEnd:
    def test_runs_resume_and_failures(self, tmp_path, fake_python):
        campaign = _make_campaign(tmp_path)
        # Run through the real subprocess path with the fake interpreter.
        orig = rc.run_one
        with patch.object(
            rc,
            "run_one",
            side_effect=lambda c, r, k, **kw: orig(c, r, k, python=fake_python, **kw),
        ):
            summary = rc.run_campaign(campaign, jobs=2)

        assert summary["succeeded"] == 8  # knn conditions
        assert len(summary["failed"]) == 8  # maha conditions fail
        written = json.loads(
            (
                campaign
                / "runs/split1/ad-frozen/ad-knn-incv3/seed1/reports/evaluation.json"
            ).read_text()
        )
        assert written["compute"]["seed"] == 1
        assert os.path.isabs(written["data"]["split_file"])

        # Second pass: completed runs are skipped, failed ones are retried.
        with patch.object(
            rc,
            "run_one",
            side_effect=lambda c, r, k, **kw: orig(c, r, k, python=fake_python, **kw),
        ) as again:
            summary2 = rc.run_campaign(campaign)
        assert summary2["skipped_done"] == 8
        assert again.call_count == 8

    def test_main_exit_codes(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        assert rc.main([str(campaign), "--expand-only"]) == 0
        with patch.object(rc, "run_one", return_value=False):
            assert rc.main([str(campaign), "--only", "knn-rn50"]) == 1

    def test_main_uses_other_sweep_file(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        sweep = _sweep()
        sweep["family"] = "clf"
        sweep["name_template"] = "train-{method}-{backbone}"
        (campaign / "configs" / "sweep-train.yaml").write_text(yaml.safe_dump(sweep))
        assert (
            rc.main([str(campaign), "--expand-only", "--sweep", "sweep-train.yaml"])
            == 0
        )
        stored = yaml.safe_load(
            (campaign / "configs/runs/train-knn-rn50/split0_seed0.yaml").read_text()
        )
        assert stored["output"]["base_dir"] == "runs/split0/clf/train-knn-rn50/seed0"
        assert not (campaign / "configs/runs/ad-knn-rn50").exists()

    def test_main_rejects_bad_jobs(self, tmp_path):
        with pytest.raises(SystemExit):
            rc.main([str(tmp_path), "--jobs", "0"])
