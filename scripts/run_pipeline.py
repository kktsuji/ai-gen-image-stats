import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.utils.notification import (
    _get_webhook_url,  # noqa: E402
    _post_slack,
    notify_success,
)

DATA_PREPARATION_CONFIG = "configs/data-preparation.yaml"
DIFFUSION_CONFIG = "configs/diffusion.yaml"
SELECTION_CONFIG = "configs/sample-selection.yaml"
SELECTION_EVALUATE_CONFIG = "configs/sample-selection-evaluate.yaml"
CLASSIFIER_CONFIG = "configs/classifier.yaml"

# Naming convention:
#   Dimension separator: "__" (double underscore)
#   Within-dimension separator: "-" (hyphen) or "_" (single underscore)
#
#   Synthetic:  {train}__{gen}__{sel}__{dose}  e.g. us__n1000-gs2__topk__d2
#   Baseline:   baseline__{strategy}           e.g. baseline__vanilla (acts as dose D0)

# ON/OFF switches for each experiment phase
RUN_DATA_PREPARATION = False
RUN_TRAINING = False
RUN_GENERATION = True
RUN_SELECTION = True
RUN_SELECTION_EVALUATE = True
RUN_CLASSIFIER = True
RUN_BASELINE_CLASSIFIER = True
RUN_EVALUATION = True
RUN_SUMMARIZE = True

# Skip completed runs (set to False to rerun everything as before)
SKIP_COMPLETED = True

# Delete each classifier experiment's checkpoints/ dir after a successful eval
# (the .pth files are not needed once reports/evaluation.json is written).
DELETE_CHECKPOINTS_AFTER_EVAL = True

# Cooldown (seconds) inserted between consecutive GPU container runs.
# Launching `docker run --rm --gpus all` back-to-back at high frequency can start a
# new CUDA init before the NVIDIA driver has released the previous CUDA context,
# leading to crashes like "CUDA error: unknown error". This fixed pause prevents that.
# It trades off total wall time (runs x N seconds), so tune it here. 0 disables it.
GPU_COOLDOWN_SECONDS = 5

# Per-run Slack notifications from the classifier containers are suppressed (the pipeline
# launches many classifier runs x train+eval, which would flood the channel). Instead, the
# pipeline emits a throttled "Classifier: current/total" progress message on the first
# experiment (a "phase started" signal), every N experiments, and at completion. Other phases
# keep their normal per-run notifications. Keep N small relative to the experiment matrix size
# so progress stays visible early instead of leaving a multi-hour dead zone at the start.
CLASSIFIER_NOTIFY_EVERY = 10

# Number of classifier experiments (train -> eval -> cleanup units) to run
# concurrently, each in its own GPU container. The classifier is head-only
# (InceptionV3 frozen backbone, batch_size 16, AMP), so a single run uses a small
# fraction of the GPU and several can time-share it. Each job writes to its own
# out_dir/seed, so they never contend on shared state. Lower this if you hit CUDA
# OOM; set to 1 to fall back to fully sequential behavior. Container launches are
# still staggered by GPU_COOLDOWN_SECONDS (see _throttled_run_and_wait) to avoid
# back-to-back CUDA-init crashes.
CLASSIFIER_MAX_PARALLEL = 3

# Random seeds for multi-seed classifier runs (for statistical testing).
# Bump to range(30) if the head-only f1_1 std stays large after the first batch.
SEEDS = list(range(20))

# Checkpoint-slimming overrides for classifier runs.
# Keep only final_model.pth (weights only) per seed: the evaluation phase reads
# best_model.pth -> final_model.pth, so the periodic and latest checkpoints are
# redundant, and optimizer state is not needed for evaluation-only reuse.
CLASSIFIER_CHECKPOINT_OVERRIDES = [
    "--training.checkpointing.save_optimizer",
    "false",
    "--training.checkpointing.save_latest",
    "false",
    "--training.checkpointing.save_frequency",
    "100",
]

# DataLoader runtime settings for the high-frequency classifier phase.
# The classifier dataset is small (head-only, batch_size 16 / epochs 40), so fewer workers has
# negligible performance impact while reducing worker fork/exit churn and
# shared-memory pressure (Bus error / shm exhaustion). Use "0" for the most conservative.
CLASSIFIER_RUNTIME_OVERRIDES = [
    "--data.loading.num_workers",
    "2",
]

# Each variant: (name, extra CLI overrides for training)
# Dose-response experiment: a single diffusion variant (upsampling) is the
# fixed generator, so quantity is the only thing that varies downstream.
TRAIN_VARIANTS = [
    (
        "us",
        [
            "--data.balancing.upsampling.enabled",
            "true",
            "--training.epochs",
            "1000",
        ],
    ),
]

# Generation parameter variants: (name, extra CLI overrides)
# Single best generation config (guidance 2.0); large pool (~1000) so the
# top_k=400 selection can feed the full dose ladder (D4 needs +352).
GEN_VARIANTS = [
    (
        "n1000-gs2",
        [
            "--generation.sampling.num_samples",
            "1000",
            "--generation.sampling.guidance_scale",
            "2.0",
        ],
    ),
]

# Selection algorithm variants: (name, extra CLI overrides)
# - topk: realism-greedy top-N (value=400 so the dose ladder is a nested top-N
#   of one ranking; quantity decoupled from the selection method).
# - random / stratified: selection ABLATION arms isolating whether the
#   realism-greedy ranking collapses diversity (and thereby harms pr_auc). Both
#   select exactly 352 (= dose d4, the 1:1-balance quantity) so the added count
#   matches topk@d4; they must NOT use the d1-d4 nested ladder (a smaller dose
#   would let the downstream max_samples truncation re-impose a realism ranking
#   by score). See SELECTION_DOSES below.
SELECTION_VARIANTS = [
    ("topk", ["--selection.mode", "top_k", "--selection.value", "400"]),
    ("random", ["--selection.mode", "random", "--selection.value", "352"]),
    ("stratified", ["--selection.mode", "stratified", "--selection.value", "352"]),
]

# Per selection method, which classifier dose levels to run. Only topk carries a
# meaningful nested top-N ladder; the ablation arms select exactly 352 up front,
# so they run d4 alone (max_samples=352 truncation is a no-op, preserving the
# random / stratified ordering instead of re-ranking by realism score).
SELECTION_DOSES = {
    "topk": ["d1", "d2", "d3", "d4"],
    "random": ["d4"],
    "stratified": ["d4"],
}


# Classifier dose ladder: (name, extra CLI overrides).
# Quality is fixed (same top_k=400 ranking); only the number of added abnormal
# images varies via max_samples (deterministic top-N truncation). Real abnormal
# train = 84, so d2 doubles it and d4 reaches 1:1 balance (84 + 352 = 436).
def _dose_overrides(n: int) -> list[str]:
    return [
        "--data.synthetic_augmentation.limit.mode",
        "max_samples",
        "--data.synthetic_augmentation.limit.max_samples",
        str(n),
    ]


CLASSIFIER_VARIANTS = [
    ("d1", _dose_overrides(42)),  # 0.5x minority
    ("d2", _dose_overrides(84)),  # 1x (double minority)
    ("d3", _dose_overrides(168)),  # 2x minority
    ("d4", _dose_overrides(352)),  # balance to 1:1 (84 + 352 = 436)
]

# Lookup for the dose overrides selected per (selection method) via SELECTION_DOSES.
_DOSE_OVERRIDES_BY_NAME = dict(CLASSIFIER_VARIANTS)


# Baseline classifier variants: real data only, with different balancing strategies
BASELINE_VARIANTS = [
    (
        "baseline__vanilla",
        [
            "--data.synthetic_augmentation.enabled",
            "false",
        ],
    ),
    (
        "baseline__ws",
        [
            "--data.synthetic_augmentation.enabled",
            "false",
            "--data.balancing.weighted_sampler.enabled",
            "true",
        ],
    ),
    (
        "baseline__us",
        [
            "--data.synthetic_augmentation.enabled",
            "false",
            "--data.balancing.upsampling.enabled",
            "true",
        ],
    ),
]


def _is_done(*markers: str) -> bool:
    """SKIP_COMPLETED 有効かつ全マーカーが存在すれば True。"""
    return SKIP_COMPLETED and all(os.path.exists(m) for m in markers)


def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def run(
    config: str,
    overrides: list[str],
    *,
    suppress_output: bool = False,
    disable_notifications: bool = False,
) -> subprocess.Popen[bytes]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--network=host",
        "--shm-size=4g",
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
        "kktsuji/nvidia-cuda12.8.1-cudnn-runtime-ubuntu24.04",
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
    if GPU_COOLDOWN_SECONDS > 0:
        time.sleep(GPU_COOLDOWN_SECONDS)


def run_and_wait(
    config: str,
    overrides: list[str],
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
# workers: a worker waits until GPU_COOLDOWN_SECONDS have elapsed since the previous
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
    overrides: list[str],
    *,
    suppress_output: bool = False,
    disable_notifications: bool = False,
) -> int:
    """run_and_wait variant for parallel use: stagger the launch, then wait.

    Unlike run_and_wait (which sleeps *after* the run to let the driver release the
    CUDA context — pointless under parallelism, where peers keep using the GPU), this
    gates only the launch instant so concurrent workers don't init CUDA simultaneously.
    """
    if GPU_COOLDOWN_SECONDS > 0:
        with _launch_lock:
            wait = _last_launch_monotonic[0] + GPU_COOLDOWN_SECONDS - time.monotonic()
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

    Posts on the first experiment (a "phase started" signal), every CLASSIFIER_NOTIFY_EVERY
    experiments, and at completion. Stays silent when the webhook is unset, and never lets a
    notification failure interrupt the pipeline.
    """
    if not total or (
        current != 1 and current % CLASSIFIER_NOTIFY_EVERY != 0 and current != total
    ):
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
    train_overrides: list[str],
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
    final_ckpt = f"{out_dir}/checkpoints/final_model.pth"
    eval_marker = os.path.join(out_dir, "reports", "evaluation.json")

    # Already evaluated (checkpoints possibly already cleaned up): skip entirely.
    if RUN_EVALUATION and _is_done(eval_marker):
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
            *CLASSIFIER_CHECKPOINT_OVERRIDES,
            *CLASSIFIER_RUNTIME_OVERRIDES,
        ]
        print(f"[CLS] {exp_label}: {CLASSIFIER_CONFIG} {' '.join(overrides)}")
        rc = _throttled_run_and_wait(
            CLASSIFIER_CONFIG,
            overrides,
            disable_notifications=True,
            suppress_output=suppress_output,
        )
        if rc != 0:
            print(f"[CLS] FAIL {exp_label}: training failed (rc={rc})")
            return

    if not RUN_EVALUATION:
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
        *CLASSIFIER_RUNTIME_OVERRIDES,
    ]
    print(f"[EVAL] {exp_label}: {CLASSIFIER_CONFIG} {' '.join(eval_overrides)}")
    rc = _throttled_run_and_wait(
        CLASSIFIER_CONFIG,
        eval_overrides,
        disable_notifications=True,
        suppress_output=suppress_output,
    )
    if rc != 0:
        # Keep checkpoints so the experiment can be retried.
        print(f"[EVAL] FAIL {exp_label}: evaluation failed (rc={rc})")
        return

    # --- Cleanup: drop the now-unneeded checkpoints after a successful eval ---
    if DELETE_CHECKPOINTS_AFTER_EVAL:
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            print(f"[CLEANUP] {exp_label}: removed {ckpt_dir}")


def _run_classifier_job(
    exp_label: str, out_dir: str, train_overrides: list[str]
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


def _run_classifier_jobs(jobs: list[tuple[str, str, list[str]]], total: int) -> None:
    """Run classifier experiments through a bounded thread pool.

    Each job is an independent (label, out_dir, train_overrides) train -> eval ->
    cleanup unit. Up to CLASSIFIER_MAX_PARALLEL run concurrently, sharing the GPU;
    container launches are staggered by _throttled_run_and_wait. At most one job at a
    time streams its container output to the console (see _run_classifier_job). The
    throttled progress notification is emitted in completion order under _progress_lock.
    """
    done = 0
    with ThreadPoolExecutor(max_workers=CLASSIFIER_MAX_PARALLEL) as executor:
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
    start = time.time()

    # ------------------------------------------------------------------
    # Data Preparation: create train/val split
    # ------------------------------------------------------------------
    if RUN_DATA_PREPARATION:
        # outputs/splits-seed{N}-tr{R}/train_val_split.json
        print(f"[DATA-PREP] {DATA_PREPARATION_CONFIG}")
        run_and_wait(DATA_PREPARATION_CONFIG, [])

    # ------------------------------------------------------------------
    # Training (max 2 parallel, sliding window)
    # ------------------------------------------------------------------
    # NOTE: the GPU cooldown (run_and_wait) is intentionally NOT used here. There are
    # only 2 TRAIN_VARIANTS, so restart frequency is low (rapid-restart crashes are low
    # risk), and a cooldown sleep would serialize the deliberate max_parallel=2 overlap.
    if RUN_TRAINING:
        # outputs/diffusion-{train}/train/
        max_parallel = 2
        active: list[subprocess.Popen[bytes]] = []

        for name, overrides in TRAIN_VARIANTS:
            marker = f"outputs/diffusion-{name}/train/checkpoints/final_model.pth"
            if _is_done(marker):
                print(f"[SKIP] {name}: training already complete")
                continue

            # Wait for a slot to free up
            while len(active) >= max_parallel:
                for proc in active:
                    if proc.poll() is not None:
                        active.remove(proc)
                        break
                else:
                    # No process finished yet; wait for any one
                    os.wait()
                    active = [p for p in active if p.poll() is None]

            train_overrides = [
                "--output.base_dir",
                f"outputs/diffusion-{name}/train",
                *overrides,
            ]
            suppress = len(active) > 0
            tag = "BG" if suppress else "FG"
            print(f"[{tag}] {name}: {DIFFUSION_CONFIG} {' '.join(train_overrides)}")
            active.append(
                run(DIFFUSION_CONFIG, train_overrides, suppress_output=suppress)
            )

        # Wait for all remaining training processes
        for proc in active:
            proc.wait()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    if RUN_GENERATION:
        # outputs/diffusion-{train}/gen/{gen}/
        for train_name, train_overrides in TRAIN_VARIANTS:
            for gen_name, gen_overrides in GEN_VARIANTS:
                if SKIP_COMPLETED and _dir_nonempty(
                    f"outputs/diffusion-{train_name}/gen/{gen_name}/generated"
                ):
                    print(
                        f"[SKIP] {train_name}/{gen_name}: generation already complete"
                    )
                    continue

                overrides = [
                    "--mode",
                    "generate",
                    "--output.base_dir",
                    f"outputs/diffusion-{train_name}/gen/{gen_name}",
                    "--generation.checkpoint",
                    f"outputs/diffusion-{train_name}/train/checkpoints/final_model.pth",
                    *train_overrides,
                    *gen_overrides,
                ]
                print(
                    f"[GEN] {train_name}/{gen_name}: {DIFFUSION_CONFIG} {' '.join(overrides)}"
                )
                run_and_wait(DIFFUSION_CONFIG, overrides)

    # ------------------------------------------------------------------
    # Sample Selection
    # ------------------------------------------------------------------
    if RUN_SELECTION:
        # outputs/diffusion-{train}/selection/{gen}_{algo}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, sel_overrides in SELECTION_VARIANTS:
                    sel_base = f"outputs/diffusion-{train_name}/selection/{gen_name}__{sel_name}"
                    if _is_done(f"{sel_base}/reports/accepted_samples.json"):
                        print(
                            f"[SKIP] {train_name}/{gen_name}/{sel_name}: selection already complete"
                        )
                        continue

                    overrides = [
                        "--output.base_dir",
                        sel_base,
                        "--data.generated.directory",
                        f"outputs/diffusion-{train_name}/gen/{gen_name}/generated",
                        *sel_overrides,
                    ]
                    print(
                        f"[SEL] {train_name}/{gen_name}/{sel_name}: {SELECTION_CONFIG} {' '.join(overrides)}"
                    )
                    run_and_wait(SELECTION_CONFIG, overrides)

    # ------------------------------------------------------------------
    # Selection Evaluation
    # ------------------------------------------------------------------
    if RUN_SELECTION_EVALUATE:
        # outputs/diffusion-{train}/selection-eval/{gen}_{sel}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, _ in SELECTION_VARIANTS:
                    eval_base = (
                        f"outputs/diffusion-{train_name}/selection-eval/"
                        f"{gen_name}__{sel_name}"
                    )
                    if _is_done(f"{eval_base}/reports/evaluation.json"):
                        print(
                            f"[SKIP] {train_name}/{gen_name}/{sel_name}: "
                            "selection-eval already complete"
                        )
                        continue

                    overrides = [
                        "--output.base_dir",
                        eval_base,
                        "--data.generated.directory",
                        f"outputs/diffusion-{train_name}/gen/{gen_name}/generated",
                        "--data.selected.split_file",
                        f"outputs/diffusion-{train_name}/selection/{gen_name}__{sel_name}/reports/accepted_samples.json",
                    ]
                    print(
                        f"[SEL-EVAL] {train_name}/{gen_name}/{sel_name}: "
                        f"{SELECTION_EVALUATE_CONFIG} {' '.join(overrides)}"
                    )
                    run_and_wait(SELECTION_EVALUATE_CONFIG, overrides)

    # ------------------------------------------------------------------
    # Classifier progress tracking: per-run notifications are suppressed inside the
    # containers, so the pipeline emits a throttled "Classifier: current/total" instead.
    # ------------------------------------------------------------------
    classifier_total = 0
    if RUN_CLASSIFIER:
        doses_per_selection = sum(
            len(SELECTION_DOSES[name]) for name, _ in SELECTION_VARIANTS
        )
        classifier_total += (
            len(TRAIN_VARIANTS) * len(GEN_VARIANTS) * doses_per_selection * len(SEEDS)
        )
    if RUN_BASELINE_CLASSIFIER:
        classifier_total += len(BASELINE_VARIANTS) * len(SEEDS)
    # ------------------------------------------------------------------
    # Classifier (synthetic-augmentation + baseline, multi-seed): collect every
    # train -> eval -> cleanup unit into one job list, then run them through a
    # bounded thread pool (CLASSIFIER_MAX_PARALLEL concurrent GPU containers).
    # Each job writes to its own out_dir/seed, so they never contend.
    # ------------------------------------------------------------------
    classifier_jobs: list[tuple[str, str, list[str]]] = []

    if RUN_CLASSIFIER:
        # outputs/classifier/{train}__{gen}__{sel}__{cls}/seed{N}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, _ in SELECTION_VARIANTS:
                    for cls_name in SELECTION_DOSES[sel_name]:
                        cls_overrides = _DOSE_OVERRIDES_BY_NAME[cls_name]
                        for seed in SEEDS:
                            sel_split = (
                                f"outputs/diffusion-{train_name}/selection"
                                f"/{gen_name}__{sel_name}/reports/accepted_samples.json"
                            )
                            exp_name = (
                                f"{train_name}__{gen_name}__{sel_name}__{cls_name}"
                            )
                            out_dir = f"outputs/classifier/{exp_name}/seed{seed}"
                            train_overrides = [
                                "--compute.seed",
                                str(seed),
                                "--data.synthetic_augmentation.enabled",
                                "true",
                                "--data.synthetic_augmentation.split_file",
                                sel_split,
                                *cls_overrides,
                            ]
                            classifier_jobs.append(
                                (f"{exp_name}/seed{seed}", out_dir, train_overrides)
                            )

    if RUN_BASELINE_CLASSIFIER:
        # outputs/classifier/baseline__{strategy}/seed{N}/
        for baseline_name, baseline_overrides in BASELINE_VARIANTS:
            for seed in SEEDS:
                out_dir = f"outputs/classifier/{baseline_name}/seed{seed}"
                train_overrides = [
                    "--compute.seed",
                    str(seed),
                    *baseline_overrides,
                ]
                classifier_jobs.append(
                    (f"{baseline_name}/seed{seed}", out_dir, train_overrides)
                )

    # Evaluation is interleaved per experiment inside _run_classifier_experiment
    # (train -> eval -> checkpoint cleanup), gated by RUN_EVALUATION.
    if classifier_jobs:
        _run_classifier_jobs(classifier_jobs, classifier_total)

    # ------------------------------------------------------------------
    # Summarize: aggregate evaluation reports (CPU-only, no Docker)
    # ------------------------------------------------------------------
    if RUN_SUMMARIZE:
        # Selection-eval aggregation
        # outputs/evaluation_report/selection_evaluation_summary.json
        print("[SUMMARIZE] Generating selection-eval report")
        subprocess.run(
            [
                "python3",
                "-m",
                "src.experiments.sample_selection.evaluation_report",
                "--base-dir",
                "outputs",
                "--output-dir",
                "outputs/evaluation_report",
            ],
            check=True,
        )

        # Classifier aggregation
        # outputs/evaluation_report/classifier_evaluation_summary.json
        print("[SUMMARIZE] Generating classifier evaluation report")
        subprocess.run(
            [
                "python3",
                "-m",
                "src.experiments.classifier.evaluation_report",
                "--base-dir",
                "outputs/classifier",
                "--output-dir",
                "outputs/evaluation_report",
                "--selection-summary-pattern",
                "outputs/diffusion-*/selection-eval/*/reports/evaluation.json",
                # Compare against the strongest class-balancing baseline (weighted
                # sampler). vanilla (no balancing) flatters synthetic augmentation
                # because it conflates the augmentation's effect with simply
                # addressing the class imbalance; ws isolates the former.
                "--baseline-name",
                "baseline__ws",
            ],
            check=True,
        )

    notify_success(
        {"experiment": "pipeline", "output": {"base_dir": "outputs"}},
        time.time() - start,
    )


if __name__ == "__main__":
    main()
