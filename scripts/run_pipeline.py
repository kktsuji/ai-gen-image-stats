"""Synthetic-augmentation pipeline driver.

Generic orchestrator: Docker invocation, GPU-launch staggering, the bounded thread
pool, skip-completed, checkpoint cleanup, and throttled progress notifications all live
here. Every piece of *experiment configuration* (phase flags, runner/infra settings,
seeds, global classifier overrides, and the baseline/fine-tune variant matrix) is read
from a YAML config (default configs/pipeline.yaml) via scripts/pipeline_config.py, so
changing what runs is a config edit, not a code change.

Usage:
    python -m scripts.run_pipeline [configs/pipeline.yaml]
        [--set KEY=VALUE ...] [--set-run KEY=VALUE ...] [--local]

``--set`` overrides nested pipeline-config keys (e.g.
``--set runner.classifier_output_root=outputs/multisplit/split0/binary-depth``);
``--set-run`` injects flat classifier-run flags into every launch (e.g.
``--set-run data.split_file=outputs/splits/cv/cv_split0.json``). Together they let
one base pipeline YAML serve every split of a cross-validation sweep.

``--local`` (alias ``--no-docker``) runs each ``src.main`` job directly with the
current interpreter (this venv) instead of inside a Docker container, bypassing the
WSL+Docker layer that destabilizes long training runs on Windows. It is sugar for the
``runner.execution: local`` config field (injected as ``--set runner.execution=local``),
so the mode is validated once and recorded in the run's config snapshot; the ``docker:``
section is then optional. Equivalently, set ``runner.execution: local`` in the YAML.

Naming convention:
    Dimension separator: "__" (double underscore)
    Within-dimension separator: "-" (hyphen) or "_" (single underscore)
    Transfer (frozen-depth sweep): {depth}__{balancing}  e.g. ft-mixed7__ws
    Baseline:                      baseline__{strategy}   e.g. baseline__ws (head-only, D0)
"""

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from scripts.pipeline_config import (  # noqa: E402
    load_pipeline_config,
    parse_shm_size,
)
from src.utils.cli import serialize_overrides  # noqa: E402
from src.utils.notification import (  # noqa: E402
    notify_progress,
    notify_success,
)

DEFAULT_PIPELINE_CONFIG = "configs/pipeline.yaml"

# Runtime deps the pinned Docker image supplied regardless of the launching interpreter.
# In --local mode the pipeline's own interpreter must carry them, so the preflight verifies
# each is importable (via find_spec, without importing) before any job launches.
_REQUIRED_LOCAL_MODULES = ("torch", "torchvision")

# The ==-pinned dependency manifest the Docker image was built from. In --local mode the
# preflight compares the interpreter's installed deps against these pins to warn when a
# local run would not reproduce the container-frozen results.
_REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"

# Set once in main() before any worker thread spawns, then read-only. Holds the
# validated pipeline config so the orchestration helpers can read runner/docker/phase
# settings without threading them through every call (same load-once/read-only
# thread-safety model the previous module-global constants had).
CFG: Dict[str, Any] = {}

# A classifier job: (label, out_dir, train_overrides).
Job = Tuple[str, str, List[str]]


def _is_local() -> bool:
    """True when jobs run via the host interpreter instead of a Docker container.

    Reads the validated ``runner.execution`` config field (single source of truth, set in
    the YAML or via the ``--local`` alias), so there is no second module global to keep in
    sync with CFG.
    """
    return CFG["runner"]["execution"] == "local"


def _is_done(*markers: str) -> bool:
    """True if skip_completed is enabled and every marker file exists."""
    return CFG["runner"]["skip_completed"] and all(os.path.exists(m) for m in markers)


def _experiment_eval_complete(out_dir: str) -> bool:
    """True if this classifier experiment is already fully evaluated (skip_completed).

    An experiment is "done" only when the evaluation phase is on and every
    evaluation-split report already exists under ``out_dir/reports/`` (the same
    condition that makes ``_run_classifier_experiment`` skip a run entirely). Shared
    with the ``--local`` preflight gate so it agrees with the runner on what will
    actually launch a GPU pass. Returns False when skip_completed is off.
    """
    if not CFG["phases"]["evaluation"]:
        return False
    markers = [
        os.path.join(out_dir, "reports", _eval_report_name(s))
        for s in CFG["runner"]["evaluation_splits"]
    ]
    return _is_done(*markers)


def _eval_report_name(split: str) -> str:
    """Report filename main.py writes for a split (canonical for test).

    Mirrors src/main.py: the held-out test report keeps the canonical
    "evaluation.json" (what evaluation_report aggregates); other splits are
    written split-tagged so a val pass doesn't clobber the test report.
    """
    return "evaluation.json" if split == "test" else f"evaluation_{split}.json"


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
    local = _is_local()
    # The job itself is identical in both modes; only the interpreter and its wrapper
    # differ (this venv directly vs python3 inside the container). Kept in one place so a
    # change to how src.main is invoked can't make local and docker modes diverge.
    main_invocation = ["-m", "src.main", config, *overrides]
    if local:
        # Run directly with the interpreter driving the pipeline (this venv), no
        # container. Notification suppression is handled below via the child env, since
        # there is no `-e SLACK_WEBHOOK_URL=` container flag to blank the webhook with.
        cmd = [sys.executable, *main_invocation]
    else:
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
        cmd += [CFG["docker"]["image"], "python3", *main_invocation]

    popen_kwargs: Dict[str, Any] = {}
    if suppress_output:
        # Drop normal stdout (the logger streams INFO to stdout; concurrent runs would
        # interleave) but keep stderr so a crash — OOM kill, import error, traceback —
        # still surfaces on the console instead of vanishing. stderr is quiet during
        # normal runs, so this doesn't reintroduce the interleaving.
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = sys.stderr
    # Local mode inherits the parent env by default, which would let a stale shell-exported
    # SLACK_WEBHOOK_URL shadow .env (main.py's load_dotenv(override=False) keeps an
    # already-set var). Docker never leaked host env, so .env was always authoritative. Pin
    # the webhook in the child env to match that: blank it to suppress notifications, else
    # drop it so main.py's load_dotenv populates it from .env.
    if local:
        env = os.environ.copy()
        if disable_notifications:
            env["SLACK_WEBHOOK_URL"] = ""
        else:
            env.pop("SLACK_WEBHOOK_URL", None)
        popen_kwargs["env"] = env
    return subprocess.Popen(cmd, **popen_kwargs)


def _gpu_cooldown() -> None:
    """Fixed pause giving the GPU driver time to release the CUDA context.

    Applied after every job regardless of execution mode: local host processes and
    Docker containers both leave a CUDA context to tear down.
    """
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
    """Launch a job (Docker container or, in local mode, a host process), wait for it,
    apply the GPU cooldown, and return its returncode."""
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
# launch, starts its job (container, or host process in local mode), then releases the
# gate so runs overlap. This staggers launches (preventing the back-to-back CUDA-init
# crashes) without serializing the runs.
_launch_lock = threading.Lock()
_last_launch_monotonic = [0.0]
# Guards the shared progress counter / Slack notifications across worker threads.
_progress_lock = threading.Lock()
# Single "foreground" slot: at most one running classifier job streams its
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
    notify_progress(f"Classifier: {current}/{total}")


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
    eval_splits = CFG["runner"]["evaluation_splits"]
    final_ckpt = f"{out_dir}/checkpoints/final_model.pth"

    # Already evaluated (checkpoints possibly already cleaned up): skip entirely. The
    # per-split report markers under out_dir/reports/ are the completion signal — the run
    # is "done" only when every split has been evaluated (so a half-finished two-pass run
    # re-runs the rest). Shared with the --local preflight gate via _experiment_eval_complete.
    if _experiment_eval_complete(out_dir):
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

    # Evaluate each configured split (e.g. val then test) reusing the same
    # checkpoint. The per-pass --evaluation.split is appended last so it wins over
    # any evaluation.split left in classifier_overrides.runtime. Predictions are
    # written split-tagged (predictions_{split}.npz) for post-hoc threshold
    # analysis; checkpoints are kept until ALL splits succeed.
    for split in eval_splits:
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
            "--evaluation.split",
            split,
        ]
        print(
            f"[EVAL] {exp_label} (split={split}): "
            f"{classifier_config} {' '.join(eval_overrides)}"
        )
        rc = _throttled_run_and_wait(
            classifier_config,
            eval_overrides,
            disable_notifications=True,
            suppress_output=suppress_output,
        )
        if rc != 0:
            # Keep checkpoints so the experiment can be retried from where it stopped.
            print(
                f"[EVAL] FAIL {exp_label} (split={split}): evaluation failed (rc={rc})"
            )
            return

    # --- Cleanup: drop the now-unneeded checkpoints after all splits evaluated ---
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic-augmentation pipeline driver"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_PIPELINE_CONFIG,
        help="Path to the pipeline YAML config",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override a nested pipeline-config key (repeatable)",
    )
    parser.add_argument(
        "--set-run",
        action="append",
        default=[],
        dest="set_run_overrides",
        metavar="KEY=VALUE",
        help="Inject a flat classifier-run override into every launch (repeatable)",
    )
    parser.add_argument(
        "--local",
        "--no-docker",
        action="store_true",
        dest="local",
        help="Run src.main jobs directly with the current interpreter (this venv) "
        "instead of inside a Docker container. Alias for --set runner.execution=local; "
        "bypasses the WSL+Docker layer.",
    )
    return parser.parse_args()


def _pinned_requirement_versions() -> Dict[str, str]:
    """Map every ``name==version`` pin in requirements.txt to its version.

    Only plain ``name==version`` pins are recognized; a dependency pinned another way
    (``>=``, unpinned) is omitted (no version expectation). Returns ``{}`` if the file
    cannot be read, so a missing manifest degrades to "no drift check" rather than erroring.
    """
    pins: Dict[str, str] = {}
    try:
        lines = _REQUIREMENTS_FILE.read_text().splitlines()
    except OSError:
        return pins
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        name, version = name.strip(), version.strip()
        if name and version:
            pins[name] = version
    return pins


def _pending_local_gpu_jobs() -> bool:
    """True when this run will actually launch at least one classifier GPU pass.

    Only the classifier phases (baseline/ft) do GPU compute; ``data_preparation`` is
    CPU-only, ``evaluation`` is interleaved inside the classifier phases, and ``summarize``
    is CPU-only. ``build_classifier_jobs`` already reflects which classifier phases are
    enabled (it is pure/no-I/O), and a job whose reports already exist under
    ``skip_completed`` will be skipped without touching the GPU — so a summarize-only or
    fully-skip_completed --local run needs no CUDA and the fatal GPU preflight is skipped.
    """
    return any(
        not _experiment_eval_complete(out_dir)
        for _, out_dir, _ in build_classifier_jobs(CFG)
    )


def _preflight_local_checks(*, require_cuda: bool) -> None:
    """Guard the assumptions Docker used to enforce, before any --local job launches.

    1. Jobs run as ``sys.executable -m src.main``, so the pipeline's own interpreter must
       carry every GPU dep (torch, torchvision, ...). The pinned Docker image supplied
       these regardless of the launching interpreter; local mode inherits whatever
       ``python`` invoked us, so a missing dep would only surface mid-run.
    2. Docker's CUDA base image + ``--gpus all`` guaranteed a usable GPU. A CPU-only torch
       wheel imports fine but would silently run every job on CPU (10-100x slower), so
       verify ``torch.cuda.is_available()`` too. Fatal, like a missing dep: there is no
       valid CPU use for a GPU campaign. Only checked when ``require_cuda`` — a run that
       launches only CPU work (e.g. data_preparation, or classifier jobs all skip_completed)
       must not be blocked for lacking a GPU it never uses.
    3. The Docker image froze deps at ``==`` pins; a local venv may hold different versions,
       so results would silently diverge from the container baselines. Warn (not fatal:
       divergence is legitimate outside a frozen-deps campaign).
    4. Docker passed ``--shm-size`` to give DataLoader workers enough shared memory; a
       host process is bounded by ``/dev/shm`` instead, which defaults small on WSL2 —
       the exact platform --local targets — and triggers a Bus error mid-run. This is a
       warning, not fatal: it only bites when ``data.loading.num_workers > 0``, and
       ``num_workers=0`` is a valid escape hatch.
    """
    # 1. Check every required dep, not just torch — report them all at once.
    missing = [
        name
        for name in _REQUIRED_LOCAL_MODULES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            f"[RUN] --local: the launching interpreter ({sys.executable}) cannot import "
            f"{', '.join(repr(m) for m in missing)}. Local mode runs src.main jobs with "
            "this interpreter, so it must have the GPU deps installed. Launch via "
            "'venv/bin/python -m scripts.run_pipeline ... --local' (see CLAUDE.md)."
        )

    # 2. When GPU compute will actually run, import torch here (kept out of module scope so
    # the driver still loads on a torch-less interpreter) and require a usable CUDA device.
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                f"[RUN] --local: torch is installed but reports no CUDA device "
                f"(torch.cuda.is_available() is False) for interpreter {sys.executable}. "
                "Docker guaranteed a GPU; local mode would silently run every job on CPU. "
                "Install a CUDA-enabled torch build in this venv, or run in Docker mode."
            )

    # 3. Warn (don't abort) when an installed dep diverges from the ==-pinned version the
    # Docker image was built from — a local run would then not reproduce the container
    # baselines. Every ==-pinned dep is checked (the whole manifest is frozen for the
    # campaign, not just torch). Advisory only: divergence is legitimate outside one.
    drifted = []
    for name, pinned in _pinned_requirement_versions().items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        # Compare the public version only: a CUDA wheel reports a local segment
        # (e.g. '2.10.0+cu128') that still satisfies the plain pin '2.10.0'.
        if installed.split("+", 1)[0] != pinned:
            drifted.append(f"{name} {installed} (pinned {pinned})")
    if drifted:
        print(
            "[RUN] WARNING --local: installed deps differ from requirements.txt pins "
            f"[{', '.join(drifted)}]. Local runs use this venv, not the pinned Docker "
            "image, so numeric results may diverge from container-produced baselines."
        )

    # 4. The shm expectation lives in runner.shm_size (populated by _validate_runner, which
    # defaults it from docker.shm_size), so the check fires even for a local config that
    # dropped the docker section. Compare against .free (not .total): capacity that is
    # already consumed still triggers Bus errors.
    shm_size = CFG["runner"]["shm_size"]
    # Validation guarantees runner.shm_size is a positive, parseable size; `or 0` is a
    # defensive fallback so the `configured and ...` guard degrades to "no threshold".
    configured = parse_shm_size(shm_size) or 0
    try:
        available = shutil.disk_usage("/dev/shm").free
    except OSError:
        available = 0
    if configured and available and available < configured:
        print(
            f"[RUN] WARNING --local: /dev/shm has {available / 1024**3:.1f} GiB free but "
            f"runner.shm_size expects {shm_size}. Local mode cannot raise "
            "the shared-memory limit (no container), so multi-worker DataLoaders may crash "
            "with 'Bus error'. Reduce data.loading.num_workers (e.g. --set-run "
            "data.loading.num_workers=0) or enlarge /dev/shm."
        )


def main() -> None:
    load_dotenv()
    args = _parse_args()
    # --local is sugar for the config field: route it through the same --set override path
    # so runner.execution is validated once, captured in the config snapshot, and read from
    # CFG everywhere (no second module global). Appended last so it wins over any YAML value.
    overrides = list(args.set_overrides)
    if args.local:
        overrides.append("runner.execution=local")
    cfg = load_pipeline_config(
        args.config_path,
        overrides=overrides,
        run_overrides=args.set_run_overrides,
    )

    global CFG
    CFG = cfg
    phases = cfg["phases"]
    if _is_local():
        print(f"[RUN] local mode: launching src.main via {sys.executable} (no Docker)")
        # Run the preflight only when a src.main job will actually launch (data_preparation
        # or a not-yet-completed classifier job); require CUDA only when a classifier GPU
        # pass will run — data_preparation is CPU-only and a fully skip_completed run
        # launches nothing, so neither should be blocked for lacking a GPU.
        needs_cuda = _pending_local_gpu_jobs()
        launches_job = phases["data_preparation"] or needs_cuda
        if launches_job:
            _preflight_local_checks(require_cuda=needs_cuda)
        else:
            print(
                "[RUN] local mode: no local jobs will launch; skipping preflight checks"
            )

    start = time.time()

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
        # Only force a positive-class index when one was explicitly configured;
        # otherwise the report/threshold tools auto-detect it from each
        # evaluation.json (an explicit flag always overrides that auto-detection).
        positive_class_args = (
            ["--positive-class-index", str(summarize["positive_class"])]
            if summarize.get("positive_class") is not None
            else []
        )
        print("[SUMMARIZE] Generating classifier evaluation report")
        subprocess.run(
            [
                # Host-side CPU step (no Docker): use the same interpreter running the
                # pipeline (e.g. the venv) so it sees numpy etc. Bare "python3" would
                # resolve to whatever is on PATH, which may lack the deps.
                sys.executable,
                "-m",
                "src.experiments.classifier.evaluation_report",
                "--base-dir",
                summarize["base_dir"],
                "--output-dir",
                summarize["output_dir"],
                "--baseline-name",
                summarize["baseline_name"],
                *positive_class_args,
            ],
            check=True,
            # Report aggregation is pure filesystem I/O; cap it so a hung read can't
            # stall the pipeline indefinitely after all GPU work has finished.
            timeout=1800,
        )

        # Post-hoc decision-threshold analysis (CPU-only). Only meaningful for the
        # leak-free val->test protocol, i.e. when both splits were evaluated.
        eval_splits = cfg["runner"]["evaluation_splits"]
        if "val" in eval_splits and "test" in eval_splits:
            # The analysis reads predictions_{split}.npz, which main.py only writes
            # when evaluation.bootstrap.save_predictions is enabled. If those artifacts
            # are absent (flag off, or all eval passes failed), skip with a clear
            # message rather than letting the subprocess die on missing files after all
            # GPU work has finished.
            base_dir = summarize["base_dir"]
            has_val = bool(glob(f"{base_dir}/**/predictions_val.npz", recursive=True))
            has_test = bool(glob(f"{base_dir}/**/predictions_test.npz", recursive=True))
            if has_val and has_test:
                print("[SUMMARIZE] Generating decision-threshold analysis")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.experiments.classifier.threshold_analysis",
                        "--base-dir",
                        base_dir,
                        "--output-dir",
                        summarize["threshold_output_dir"],
                        "--criterion",
                        summarize["threshold_criterion"],
                        "--target-recall",
                        str(summarize["threshold_target_recall"]),
                        *positive_class_args,
                    ],
                    check=True,
                    timeout=1800,
                )
            else:
                print(
                    "[SUMMARIZE] Skipping decision-threshold analysis: no saved "
                    f"predictions_val/test.npz under {base_dir} "
                    "(enable evaluation.bootstrap.save_predictions to produce them)"
                )

        # Hard-core direct evaluation (abnormal-vs-suspicious restricted PR-AUC).
        # CPU-only post-hoc step over the saved predictions for the configured
        # split (default 'test'); only the chosen split's npz is needed, so this
        # runs independently of the val->test threshold analysis above.
        hc_base_dir = summarize["base_dir"]
        hc_split = summarize["hardcore_split"]
        if hc_split not in eval_splits:
            # Gate on this run's evaluated splits so a stale predictions_{split}
            # .npz from a previous run can't trigger the step on data the current
            # run never produced.
            print(
                "[SUMMARIZE] Skipping hard-core analysis: hardcore_split "
                f"{hc_split!r} not in evaluation_splits {eval_splits}"
            )
        elif glob(f"{hc_base_dir}/**/predictions_{hc_split}.npz", recursive=True):
            print("[SUMMARIZE] Generating hard-core direct-evaluation report")
            contrast_args = (
                ["--contrast-class-index", str(summarize["hardcore_contrast_class"])]
                if summarize.get("hardcore_contrast_class") is not None
                else ["--contrast-class-name", summarize["hardcore_contrast_name"]]
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.experiments.classifier.hard_core_analysis",
                    "--base-dir",
                    hc_base_dir,
                    "--output-dir",
                    summarize["hardcore_output_dir"],
                    "--split",
                    hc_split,
                    "--baseline-name",
                    summarize["baseline_name"],
                    *positive_class_args,
                    *contrast_args,
                ],
                check=True,
                timeout=1800,
            )
        else:
            print(
                "[SUMMARIZE] Skipping hard-core analysis: no saved "
                f"predictions_{hc_split}.npz under {hc_base_dir} "
                "(enable evaluation.bootstrap.save_predictions to produce them)"
            )

    notify_success(
        {"experiment": "pipeline", "output": {"base_dir": "outputs"}},
        time.time() - start,
    )


if __name__ == "__main__":
    main()
