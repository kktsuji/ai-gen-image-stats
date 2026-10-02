"""Tests for scripts/run_campaign.py (campaign driver)."""

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts import run_campaign as rc
from tests.test_campaign_config import _base, _sweep


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


@pytest.mark.component
class TestEndToEnd:
    def test_runs_resume_and_failures(self, tmp_path, fake_python):
        campaign = _make_campaign(tmp_path)
        # Run through the real subprocess path with the fake interpreter.
        orig = rc.run_one
        with patch.object(
            rc, "run_one", side_effect=lambda c, r, k: orig(c, r, k, python=fake_python)
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
            rc, "run_one", side_effect=lambda c, r, k: orig(c, r, k, python=fake_python)
        ) as again:
            summary2 = rc.run_campaign(campaign)
        assert summary2["skipped_done"] == 8
        assert again.call_count == 8

    def test_main_exit_codes(self, tmp_path):
        campaign = _make_campaign(tmp_path)
        assert rc.main([str(campaign), "--expand-only"]) == 0
        with patch.object(rc, "run_one", return_value=False):
            assert rc.main([str(campaign), "--only", "knn-rn50"]) == 1

    def test_main_rejects_bad_jobs(self, tmp_path):
        with pytest.raises(SystemExit):
            rc.main([str(tmp_path), "--jobs", "0"])
