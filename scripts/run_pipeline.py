import os
import subprocess

DATA_PREPARATION_CONFIG = "configs/data-preparation.yaml"
DIFFUSION_CONFIG = "configs/diffusion.yaml"
SELECTION_CONFIG = "configs/sample-selection.yaml"
SELECTION_EVALUATE_CONFIG = "configs/sample-selection-evaluate.yaml"
CLASSIFIER_CONFIG = "configs/classifier.yaml"

# Naming convention:
#   Dimension separator: "__" (double underscore)
#   Within-dimension separator: "-" (hyphen) or "_" (single underscore)
#
#   Synthetic:  {train}__{gen}__{sel}__{cls}   e.g. ws__n100-gs3__topk__all
#   Baseline:   baseline__{strategy}           e.g. baseline__vanilla

# ON/OFF switches for each experiment phase
RUN_DATA_PREPARATION = False
RUN_TRAINING = False
RUN_GENERATION = False
RUN_SELECTION = False
RUN_SELECTION_EVALUATE = False
RUN_CLASSIFIER = False
RUN_BASELINE_CLASSIFIER = False
RUN_EVALUATION = False
RUN_SUMMARIZE = False

# Random seeds for multi-seed classifier runs (for statistical testing)
SEEDS = [0, 1, 2, 3, 4]

# Each variant: (name, extra CLI overrides for training)
TRAIN_VARIANTS = [
    ("noaug", ["--training.epochs", "10"]),
    (
        "ws",
        [
            "--data.balancing.weighted_sampler.enabled",
            "true",
            "--training.epochs",
            "1000",
        ],
    ),
    (
        "ds",
        [
            "--data.balancing.downsampling.enabled",
            "true",
            "--training.epochs",
            "1000",
        ],
    ),
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
GEN_VARIANTS = [
    (
        "n100-gs3",
        [
            "--generation.sampling.num_samples",
            "100",
            "--generation.sampling.guidance_scale",
            "3.0",
        ],
    ),
    (
        "n100-gs5",
        [
            "--generation.sampling.num_samples",
            "100",
            "--generation.sampling.guidance_scale",
            "5.0",
        ],
    ),
]

# Selection algorithm variants: (name, extra CLI overrides)
SELECTION_VARIANTS = [
    ("topk", ["--selection.mode", "top_k", "--selection.value", "50"]),
    ("percentile", ["--selection.mode", "percentile", "--selection.value", "20"]),
    ("threshold", ["--selection.mode", "threshold", "--selection.value", "1.0"]),
]

# Classifier synthetic augmentation limit variants: (name, extra CLI overrides)
CLASSIFIER_VARIANTS = [
    ("all", []),
    (
        "ratio50",
        [
            "--data.synthetic_augmentation.limit.mode",
            "max_ratio",
            "--data.synthetic_augmentation.limit.max_ratio",
            "0.5",
        ],
    ),
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


def run(
    config: str,
    overrides: list[str],
    *,
    suppress_output: bool = False,
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


def main() -> None:
    # ------------------------------------------------------------------
    # Data Preparation: create train/val split
    # ------------------------------------------------------------------
    if RUN_DATA_PREPARATION:
        # outputs/splits-seed{N}-tr{R}/train_val_split.json
        print(f"[DATA-PREP] {DATA_PREPARATION_CONFIG}")
        proc = run(DATA_PREPARATION_CONFIG, [])
        proc.wait()

    # ------------------------------------------------------------------
    # Training (max 2 parallel, sliding window)
    # ------------------------------------------------------------------
    if RUN_TRAINING:
        # outputs/diffusion-{train}/train/
        max_parallel = 2
        active: list[subprocess.Popen[bytes]] = []

        for name, overrides in TRAIN_VARIANTS:
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
                proc = run(DIFFUSION_CONFIG, overrides)
                proc.wait()

    # ------------------------------------------------------------------
    # Sample Selection
    # ------------------------------------------------------------------
    if RUN_SELECTION:
        # outputs/diffusion-{train}/selection/{gen}_{algo}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, sel_overrides in SELECTION_VARIANTS:
                    overrides = [
                        "--output.base_dir",
                        f"outputs/diffusion-{train_name}/selection/{gen_name}__{sel_name}",
                        "--data.generated.directory",
                        f"outputs/diffusion-{train_name}/gen/{gen_name}/generated",
                        *sel_overrides,
                    ]
                    print(
                        f"[SEL] {train_name}/{gen_name}/{sel_name}: {SELECTION_CONFIG} {' '.join(overrides)}"
                    )
                    proc = run(SELECTION_CONFIG, overrides)
                    proc.wait()

    # ------------------------------------------------------------------
    # Selection Evaluation
    # ------------------------------------------------------------------
    if RUN_SELECTION_EVALUATE:
        # outputs/diffusion-{train}/selection-eval/{gen}_{sel}/
        for train_name, _ in TRAIN_VARIANTS:
            for gen_name, _ in GEN_VARIANTS:
                for sel_name, _ in SELECTION_VARIANTS:
                    overrides = [
                        "--output.base_dir",
                        f"outputs/diffusion-{train_name}/selection-eval/{gen_name}__{sel_name}",
                        "--data.generated.directory",
                        f"outputs/diffusion-{train_name}/gen/{gen_name}/generated",
                        "--data.selected.split_file",
                        f"outputs/diffusion-{train_name}/selection/{gen_name}__{sel_name}/reports/accepted_samples.json",
                    ]
                    print(
                        f"[SEL-EVAL] {train_name}/{gen_name}/{sel_name}: "
                        f"{SELECTION_EVALUATE_CONFIG} {' '.join(overrides)}"
                    )
                    proc = run(SELECTION_EVALUATE_CONFIG, overrides)
                    proc.wait()

    # ------------------------------------------------------------------
    # Classifier Training with Synthetic Augmentation (multi-seed)
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
                            overrides = [
                                "--output.base_dir",
                                out_dir,
                                "--compute.seed",
                                str(seed),
                                "--data.synthetic_augmentation.enabled",
                                "true",
                                "--data.synthetic_augmentation.split_file",
                                sel_split,
                                *cls_overrides,
                            ]
                            print(
                                f"[CLS] {exp_name}/seed{seed}:"
                                f" {CLASSIFIER_CONFIG} {' '.join(overrides)}"
                            )
                            proc = run(CLASSIFIER_CONFIG, overrides)
                            proc.wait()

    # ------------------------------------------------------------------
    # Baseline Classifier Training (real data only, multi-seed)
    # ------------------------------------------------------------------
    if RUN_BASELINE_CLASSIFIER:
        # outputs/classifier/baseline__{strategy}/seed{N}/
        for baseline_name, baseline_overrides in BASELINE_VARIANTS:
            for seed in SEEDS:
                out_dir = f"outputs/classifier/{baseline_name}/seed{seed}"
                overrides = [
                    "--output.base_dir",
                    out_dir,
                    "--compute.seed",
                    str(seed),
                    *baseline_overrides,
                ]
                print(
                    f"[BASELINE] {baseline_name}/seed{seed}:"
                    f" {CLASSIFIER_CONFIG} {' '.join(overrides)}"
                )
                proc = run(CLASSIFIER_CONFIG, overrides)
                proc.wait()

    # ------------------------------------------------------------------
    # Evaluation: re-evaluate all classifier experiments with enriched metrics
    # ------------------------------------------------------------------
    if RUN_EVALUATION:
        import glob

        # Find all seed directories (multi-seed layout) and legacy
        # experiment dirs that have no seed subdirectories.
        seed_dirs = sorted(glob.glob("outputs/classifier/*/seed*/"))
        legacy_dirs = [
            d
            for d in sorted(glob.glob("outputs/classifier/*/"))
            if not glob.glob(os.path.join(d, "seed*/"))
        ]

        for exp_dir in [*seed_dirs, *legacy_dirs]:
            exp_label = exp_dir.rstrip("/").replace("outputs/classifier/", "")

            # Find best checkpoint, fall back to final
            checkpoint = None
            for ckpt_name in ["best_model.pth", "final_model.pth"]:
                candidate = os.path.join(exp_dir, "checkpoints", ckpt_name)
                if os.path.exists(candidate):
                    checkpoint = candidate
                    break

            if checkpoint is None:
                print(f"[EVAL] SKIP {exp_label}: no checkpoint found")
                continue

            overrides = [
                "--mode",
                "evaluate",
                "--output.base_dir",
                exp_dir.rstrip("/"),
                "--evaluation.checkpoint",
                checkpoint,
                "--data.synthetic_augmentation.enabled",
                "false",
            ]
            print(f"[EVAL] {exp_label}: {CLASSIFIER_CONFIG} {' '.join(overrides)}")
            proc = run(CLASSIFIER_CONFIG, overrides)
            proc.wait()

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
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
