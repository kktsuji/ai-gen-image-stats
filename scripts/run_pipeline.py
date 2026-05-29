import os
import shutil
import subprocess
import sys
import time
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
# Fixed to top_k=400 so the dose ladder is a nested top-N of one ranking
# (quantity decoupled from the selection method).
SELECTION_VARIANTS = [
    ("topk", ["--selection.mode", "top_k", "--selection.value", "400"]),
]


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
    exp_label: str, out_dir: str, train_overrides: list[str]
) -> None:
    """Train -> evaluate -> delete checkpoints for a single classifier experiment.

    The .pth files are discarded once reports/evaluation.json is written, so the
    completion marker is evaluation.json (not final_model.pth) to keep re-runs
    idempotent after cleanup.
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
        rc = run_and_wait(CLASSIFIER_CONFIG, overrides, disable_notifications=True)
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
    rc = run_and_wait(CLASSIFIER_CONFIG, eval_overrides, disable_notifications=True)
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
        classifier_total += (
            len(TRAIN_VARIANTS)
            * len(GEN_VARIANTS)
            * len(SELECTION_VARIANTS)
            * len(CLASSIFIER_VARIANTS)
            * len(SEEDS)
        )
    if RUN_BASELINE_CLASSIFIER:
        classifier_total += len(BASELINE_VARIANTS) * len(SEEDS)
    classifier_done = 0

    # ------------------------------------------------------------------
    # Classifier with Synthetic Augmentation (multi-seed): train -> eval -> cleanup
    # ------------------------------------------------------------------
    if RUN_CLASSIFIER:
        # outputs/classifier/{train}__{gen}__{sel}__{cls}/seed{N}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, _ in SELECTION_VARIANTS:
                    for cls_name, cls_overrides in CLASSIFIER_VARIANTS:
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
                            _run_classifier_experiment(
                                f"{exp_name}/seed{seed}", out_dir, train_overrides
                            )
                            classifier_done += 1
                            _notify_classifier_progress(
                                classifier_done, classifier_total
                            )

    # ------------------------------------------------------------------
    # Baseline Classifier (real data only, multi-seed): train -> eval -> cleanup
    # ------------------------------------------------------------------
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
                _run_classifier_experiment(
                    f"{baseline_name}/seed{seed}", out_dir, train_overrides
                )
                classifier_done += 1
                _notify_classifier_progress(classifier_done, classifier_total)

    # Evaluation is interleaved per experiment in the classifier/baseline loops
    # above (train -> eval -> checkpoint cleanup), gated by RUN_EVALUATION.

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
                "--baseline-name",
                "baseline__vanilla",
            ],
            check=True,
        )

    notify_success(
        {"experiment": "pipeline", "output": {"base_dir": "outputs"}},
        time.time() - start,
    )


if __name__ == "__main__":
    main()
