"""Synthetic-augmentation pipeline driver.

Generic orchestrator: Docker invocation, GPU-launch staggering, the bounded thread
pool, skip-completed, checkpoint cleanup, and throttled progress notifications all live
here. Every piece of *experiment configuration* (phase flags, runner/infra settings,
seeds, global classifier overrides, and the baseline/fine-tune variant matrix) is read
from a YAML config (default configs/pipeline.yaml) via scripts/pipeline_config.py, so
changing what runs is a config edit, not a code change.

Usage:
    python -m scripts.run_pipeline [configs/pipeline.yaml]

Naming convention:
    Dimension separator: "__" (double underscore)
    Within-dimension separator: "-" (hyphen) or "_" (single underscore)
    Transfer (frozen-depth sweep): {depth}__{balancing}  e.g. ft-mixed7__ws
    Baseline:                      baseline__{strategy}   e.g. baseline__ws (head-only, D0)
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from scripts.pipeline_config import load_pipeline_config  # noqa: E402
from src.utils.cli import serialize_overrides  # noqa: E402
from src.utils.notification import (  # noqa: E402
    _get_webhook_url,
    _post_slack,
    notify_success,
)

DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"

# Set once in main() before any worker thread spawns, then read-only. Holds the
# validated pipeline config so the orchestration helpers can read runner/docker/phase
# settings without threading them through every call (same load-once/read-only
# thread-safety model the previous module-global constants had).
CFG: Dict[str, Any] = {}

# A classifier job: (label, out_dir, train_overrides).
Job = Tuple[str, str, List[str]]


def _is_done(*markers: str) -> bool:
    """True if skip_completed is enabled and every marker file exists."""
    return CFG["runner"]["skip_completed"] and all(os.path.exists(m) for m in markers)


def build_classifier_jobs(cfg: Dict[str, Any]) -> List[Job]:
    """Expand the variant matrix into a flat list of classifier jobs.

    Pure (no I/O): mirrors the former hardcoded nested loops. Each job carries only its
    per-variant train_overrides (seed + variant flags + per-depth LR); the global
    checkpoint/runtime overrides are appended later in _run_classifier_experiment so they
    are applied uniformly at a single point.

    Args:
        cfg: The validated pipeline config (seeds already normalized to a list).

    Returns:
        List of (label, out_dir, train_overrides) tuples.
    """
    root = cfg["runner"]["classifier_output_root"]
    seeds = cfg["seeds"]["list"]
    jobs: List[Job] = []

    if cfg["phases"]["baseline_classifier"]:
        # outputs/<root>/baseline__{strategy}/seed{N}/
        for baseline in cfg["baselines"]:
            name = baseline["name"]
            for seed in seeds:
                out_dir = f"{root}/{name}/seed{seed}"
                train_overrides = [
                    "--compute.seed",
                    str(seed),
                    *serialize_overrides(baseline["overrides"]),
                ]
                jobs.append((f"{name}/seed{seed}", out_dir, train_overrides))

    if cfg["phases"]["ft_classifier"]:
        # outputs/<root>/{depth}__{balancing}/seed{N}/
        ft = cfg["ft"]
        for depth in ft["depths"]:
            for bal in ft["balancing"]:
                exp_name = f"{depth['name']}__{bal['name']}"
                for seed in seeds:
                    out_dir = f"{root}/{exp_name}/seed{seed}"
                    train_overrides = [
                        "--compute.seed",
                        str(seed),
                        *serialize_overrides(ft["common"]),
                        *serialize_overrides(bal["overrides"]),
                        *serialize_overrides(depth["overrides"]),
                        *serialize_overrides(
                            {"training.optimizer.learning_rate": depth["learning_rate"]}
                        ),
                    ]
                    jobs.append((f"{exp_name}/seed{seed}", out_dir, train_overrides))

    return jobs


def run(
    config: str,
    overrides: List[str],
    *,
    suppress_output: bool = False,
    disable_notifications: bool = False,
) -> "subprocess.Popen[bytes]":
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--network=host",
        f"--shm-size={CFG['docker']['shm_size']}",
        "-v",
        f"{os.getcwd()}:/work",
        "-w",
        "/work",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    # Suppress in-container Slack notifications by blanking the webhook env var. main.py's
    # load_dotenv(override=False) won't overwrite an already-set var, so the empty value wins
    # and notify_success/notify_error short-circuit on the missing webhook. Must precede the image.
    if disable_notifications:
        cmd += ["-e", "SLACK_WEBHOOK_URL="]
    cmd += [
        CFG["docker"]["image"],
        "python3",
        "-m",
        "src.main",
        config,
        *overrides,
    ]
    if suppress_output:
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return subprocess.Popen(cmd)


def _gpu_cooldown() -> None:
    """Fixed pause giving the GPU driver time to release the CUDA context."""
    cooldown = CFG["runner"]["gpu_cooldown_seconds"]
    if cooldown > 0:
        time.sleep(cooldown)


def run_and_wait(
    config: str,
    overrides: List[str],
    *,
    suppress_output: bool = False,
    disable_notifications: bool = False,
) -> int:
    """Launch a container, wait for it, apply the GPU cooldown, return the returncode."""
    proc = run(
        config,
        overrides,
        suppress_output=suppress_output,
        disable_notifications=disable_notifications,
    )
    proc.wait()
    _gpu_cooldown()
    return proc.returncode


# Serializes only the GPU-context *initialization* window across parallel classifier
# workers: a worker waits until gpu_cooldown_seconds have elapsed since the previous
# launch, starts its container, then releases the gate so runs overlap. This staggers
# launches (preventing the back-to-back CUDA-init crashes) without serializing the runs.
_launch_lock = threading.Lock()
_last_launch_monotonic = [0.0]
# Guards the shared progress counter / Slack notifications across worker threads.
_progress_lock = threading.Lock()
# Single "foreground" slot: at most one running classifier job streams its container
# stdout/stderr to the console; the rest run suppressed (their detail still lands in
# each run's own logs/ dir). A job acquires this non-blocking at start and holds it for
# its whole train -> eval, so the console shows one coherent run at a time.
_foreground_lock = threading.Lock()


def _throttled_run_and_wait(
    config: str,
    overrides: List[str],
    *,
    suppress_output: bool = False,
    disable_notifications: bool = False,
) -> int:
    """run_and_wait variant for parallel use: stagger the launch, then wait.

    Unlike run_and_wait (which sleeps *after* the run to let the driver release the
    CUDA context — pointless under parallelism, where peers keep using the GPU), this
    gates only the launch instant so concurrent workers don't init CUDA simultaneously.
    """
    cooldown = CFG["runner"]["gpu_cooldown_seconds"]
    if cooldown > 0:
        with _launch_lock:
            wait = _last_launch_monotonic[0] + cooldown - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_launch_monotonic[0] = time.monotonic()
    proc = run(
        config,
        overrides,
        suppress_output=suppress_output,
        disable_notifications=disable_notifications,
    )
    proc.wait()
    return proc.returncode


def _notify_classifier_progress(current: int, total: int) -> None:
    """Send a throttled "Classifier: current/total" Slack message.

    Posts on the first experiment (a "phase started" signal), every
    runner.classifier_notify_every experiments, and at completion. Stays silent when the
    webhook is unset, and never lets a notification failure interrupt the pipeline.
    """
    notify_every = CFG["runner"]["classifier_notify_every"]
    if not total or (current != 1 and current % notify_every != 0 and current != total):
        return
    url = _get_webhook_url()
    if not url:
        return
    try:
        _post_slack(f"Classifier: {current}/{total}", url)
    except Exception as e:  # noqa: BLE001 - notifications must never break the pipeline
        print(f"[NOTIFY] failed: {e}")


def _run_classifier_experiment(
    exp_label: str,
    out_dir: str,
    train_overrides: List[str],
    *,
    suppress_output: bool = True,
) -> None:
    """Train -> evaluate -> delete checkpoints for a single classifier experiment.

    The .pth files are discarded once reports/evaluation.json is written, so the
    completion marker is evaluation.json (not final_model.pth) to keep re-runs
    idempotent after cleanup.

    Runs as one unit on a worker thread (see _run_classifier_jobs). Container stdout
    is suppressed by default so concurrent runs don't interleave; per-run detail is
    still written to each run's own logs/ dir.
    """
    classifier_config = CFG["configs"]["classifier"]
    run_evaluation = CFG["phases"]["evaluation"]
    final_ckpt = f"{out_dir}/checkpoints/final_model.pth"
    eval_marker = os.path.join(out_dir, "reports", "evaluation.json")

    # Already evaluated (checkpoints possibly already cleaned up): skip entirely.
    if run_evaluation and _is_done(eval_marker):
        print(f"[SKIP] {exp_label}: already complete")
        return

    # --- Train (skip if a checkpoint already exists from a prior attempt) ---
    if _is_done(final_ckpt):
        print(f"[SKIP-TRAIN] {exp_label}: checkpoint exists")
    else:
        overrides = [
            "--output.base_dir",
            out_dir,
            *train_overrides,
            *serialize_overrides(CFG["classifier_overrides"]["checkpoint"]),
            *serialize_overrides(CFG["classifier_overrides"]["runtime"]),
        ]
        print(f"[CLS] {exp_label}: {classifier_config} {' '.join(overrides)}")
        rc = _throttled_run_and_wait(
            classifier_config,
            overrides,
            disable_notifications=True,
            suppress_output=suppress_output,
        )
        if rc != 0:
            print(f"[CLS] FAIL {exp_label}: training failed (rc={rc})")
            return

    if not run_evaluation:
        return

    # --- Evaluate: prefer best_model.pth, fall back to final_model.pth ---
    checkpoint = None
    for ckpt_name in ["best_model.pth", "final_model.pth"]:
        candidate = os.path.join(out_dir, "checkpoints", ckpt_name)
        if os.path.exists(candidate):
            checkpoint = candidate
            break
    if checkpoint is None:
        print(f"[EVAL] SKIP {exp_label}: no checkpoint found")
        return

    eval_overrides = [
        "--mode",
        "evaluate",
        "--output.base_dir",
        out_dir,
        "--evaluation.checkpoint",
        checkpoint,
        "--data.synthetic_augmentation.enabled",
        "false",
        *serialize_overrides(CFG["classifier_overrides"]["runtime"]),
    ]
    print(f"[EVAL] {exp_label}: {classifier_config} {' '.join(eval_overrides)}")
    rc = _throttled_run_and_wait(
        classifier_config,
        eval_overrides,
        disable_notifications=True,
        suppress_output=suppress_output,
    )
    if rc != 0:
        # Keep checkpoints so the experiment can be retried.
        print(f"[EVAL] FAIL {exp_label}: evaluation failed (rc={rc})")
        return

    # --- Cleanup: drop the now-unneeded checkpoints after a successful eval ---
    if CFG["runner"]["delete_checkpoints_after_eval"]:
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            print(f"[CLEANUP] {exp_label}: removed {ckpt_dir}")


def _run_classifier_job(
    exp_label: str, out_dir: str, train_overrides: List[str]
) -> None:
    """Worker entrypoint: claim the single console (foreground) slot if free, run the job.

    Exactly one concurrent job streams its container output to the console; the others
    run suppressed. The slot is grabbed non-blocking and held for the whole train -> eval
    so the streamed log stays coherent, then released for the next job to pick up.
    """
    is_foreground = _foreground_lock.acquire(blocking=False)
    if is_foreground:
        print(f"[CLS][FG] streaming console output for: {exp_label}")
    try:
        _run_classifier_experiment(
            exp_label, out_dir, train_overrides, suppress_output=not is_foreground
        )
    finally:
        if is_foreground:
            _foreground_lock.release()


def _run_classifier_jobs(jobs: List[Job], total: int) -> None:
    """Run classifier experiments through a bounded thread pool.

    Each job is an independent (label, out_dir, train_overrides) train -> eval ->
    cleanup unit. Up to runner.classifier_max_parallel run concurrently, sharing the GPU;
    container launches are staggered by _throttled_run_and_wait. At most one job at a
    time streams its container output to the console (see _run_classifier_job). The
    throttled progress notification is emitted in completion order under _progress_lock.
    """
    done = 0
    max_parallel = CFG["runner"]["classifier_max_parallel"]
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_classifier_job, label, out_dir, overrides): label
            for label, out_dir, overrides in jobs
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:  # noqa: BLE001 - one failed run must not abort the rest
                print(f"[CLS] ERROR {futures[future]}: {e}")
            with _progress_lock:
                done += 1
                _notify_classifier_progress(done, total)


def main() -> None:
    load_dotenv()
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PIPELINE_CONFIG
    cfg = load_pipeline_config(config_path)

    global CFG
    CFG = cfg

    start = time.time()
    phases = cfg["phases"]

    # ------------------------------------------------------------------
    # Data Preparation: create train/val split
    # ------------------------------------------------------------------
    if phases["data_preparation"]:
        data_prep_config = cfg["configs"]["data_preparation"]
        print(f"[DATA-PREP] {data_prep_config}")
        run_and_wait(data_prep_config, [])

    # ------------------------------------------------------------------
    # Classifier (head-only baselines + frozen-depth sweep, multi-seed): expand the
    # variant matrix into one job list, then run them through a bounded thread pool
    # (runner.classifier_max_parallel concurrent GPU containers). Each job writes to its
    # own out_dir/seed, so they never contend. Evaluation is interleaved per experiment
    # inside _run_classifier_experiment (train -> eval -> cleanup), gated by phases.evaluation.
    # Per-run notifications are suppressed inside the containers; the pipeline emits a
    # throttled "Classifier: current/total" instead.
    # ------------------------------------------------------------------
    classifier_jobs = build_classifier_jobs(cfg)
    if classifier_jobs:
        _run_classifier_jobs(classifier_jobs, len(classifier_jobs))

    # ------------------------------------------------------------------
    # Summarize: aggregate classifier evaluation reports (CPU-only, no Docker).
    # ------------------------------------------------------------------
    if phases["summarize"]:
        summarize = cfg["summarize"]
        print("[SUMMARIZE] Generating classifier evaluation report")
        subprocess.run(
            [
                "python3",
                "-m",
                "src.experiments.classifier.evaluation_report",
                "--base-dir",
                summarize["base_dir"],
                "--output-dir",
                summarize["output_dir"],
                "--baseline-name",
                summarize["baseline_name"],
            ],
            check=True,
        )

    notify_success(
        {"experiment": "pipeline", "output": {"base_dir": "outputs"}},
        time.time() - start,
    )


if __name__ == "__main__":
    main()
