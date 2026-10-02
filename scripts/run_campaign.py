"""Run a research campaign defined by ``<campaign>/configs/sweep.yaml``.

Expands the sweep into one config per (condition, split, seed), stores them
under ``<campaign>/configs/runs/`` with campaign-relative paths, and runs each
with ``python -m src.main`` from the repository root. Paths are resolved
against the campaign folder only at run time, so the campaign stays portable.
Runs whose done marker (e.g. ``reports/evaluation.json``) exists are skipped,
so an interrupted campaign resumes where it stopped.

Usage:
    python -m scripts.run_campaign work/<series>/<NN>-<campaign> [--expand-only]
        [--only SUBSTRING] [--jobs N] [--force]
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from scripts.campaign_config import Run, load_campaign, resolve_paths

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_run_configs(campaign_dir: Path, runs: List[Run]) -> None:
    """Store every expanded per-run config (campaign-relative paths)."""
    for run in runs:
        path = campaign_dir / run.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run.config, f, sort_keys=False, allow_unicode=True)


def is_done(campaign_dir: Path, run: Run, done_marker: str) -> bool:
    return (campaign_dir / run.output_dir / done_marker).exists()


def run_one(
    campaign_dir: Path, run: Run, path_keys: List[str], python: str = sys.executable
) -> bool:
    """Run one config via ``python -m src.main``; returns True on success.

    The resolved (absolute-path) config is written to a temporary file that is
    removed afterwards; ``src.main`` itself snapshots it into the run's logs.
    """
    resolved = resolve_paths(run.config, path_keys, campaign_dir)
    fd, tmp_name = tempfile.mkstemp(prefix="campaign-run-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
        result = subprocess.run(
            [python, "-m", "src.main", tmp_name],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        logger.error(f"FAILED {run.config_path} (exit {result.returncode}):\n{tail}")
        return False
    return True


def run_campaign(
    campaign_dir: Path,
    expand_only: bool = False,
    only: Optional[str] = None,
    jobs: int = 1,
    force: bool = False,
) -> Dict[str, Any]:
    """Expand and run a campaign; returns a summary dict."""
    campaign_dir = campaign_dir.resolve()
    sweep, runs = load_campaign(campaign_dir)
    write_run_configs(campaign_dir, runs)
    conditions = sorted({r.name for r in runs})
    logger.info(
        f"Campaign {campaign_dir.name}: {len(conditions)} conditions, "
        f"{len(runs)} runs; configs written to {campaign_dir / 'configs' / 'runs'}"
    )

    selected = [r for r in runs if only is None or only in r.name]
    todo = [
        r
        for r in selected
        if force or not is_done(campaign_dir, r, sweep["done_marker"])
    ]
    summary: Dict[str, Any] = {
        "conditions": len(conditions),
        "runs": len(runs),
        "selected": len(selected),
        "skipped_done": len(selected) - len(todo),
        "succeeded": 0,
        "failed": [],
    }
    if expand_only:
        logger.info(
            f"Expand only: {len(todo)} of {len(selected)} selected runs pending"
        )
        return summary

    logger.info(
        f"Running {len(todo)} runs ({summary['skipped_done']} already done) with {jobs} job(s)"
    )
    start = time.time()

    def task(item: tuple[int, Run]) -> tuple[Run, bool]:
        index, run = item
        ok = run_one(campaign_dir, run, sweep["path_keys"])
        logger.info(
            f"[{index}/{len(todo)}] {'ok  ' if ok else 'FAIL'} {run.name} "
            f"split{run.split} seed{run.seed} ({time.time() - start:.0f}s elapsed)"
        )
        return run, ok

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for run, ok in pool.map(task, enumerate(todo, start=1)):
            if ok:
                summary["succeeded"] += 1
            else:
                summary["failed"].append(str(run.config_path))

    logger.info(
        f"Done in {time.time() - start:.0f}s: {summary['succeeded']} succeeded, "
        f"{len(summary['failed'])} failed, {summary['skipped_done']} skipped"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a research campaign sweep")
    parser.add_argument("campaign_dir", help="Campaign folder, e.g. work/<series>/01-x")
    parser.add_argument(
        "--expand-only", action="store_true", help="Write per-run configs, run nothing"
    )
    parser.add_argument("--only", help="Run only conditions whose name contains this")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Concurrent runs (default 1)"
    )
    parser.add_argument("--force", action="store_true", help="Re-run completed runs")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    summary = run_campaign(
        Path(args.campaign_dir),
        expand_only=args.expand_only,
        only=args.only,
        jobs=args.jobs,
        force=args.force,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
