# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run experiments
python -m src.main configs/<experiment>.yaml

# Example workflows
python -m src.main configs/data_preparation.yaml     # prepare train/val split
python -m src.main configs/diffusion.yaml            # train diffusion model
python -m src.main configs/diffusion-ws-gen.yaml     # generate images (mode: generate)
python -m src.main configs/examples/sample-selection.yaml     # select high-quality generated samples
python -m src.main configs/classifier.yaml --mode evaluate --evaluation.checkpoint path/to/model.pth  # evaluate trained classifier
python -m scripts.run_pipeline [configs/pipeline.yaml]       # run full synthetic augmentation pipeline
python -m src.experiments.sample_selection.evaluation_report  # aggregate selection-eval reports
python -m src.experiments.anomaly_detection.splits --src-dir <cv-binary dir> --normal-dir data/in-house/normal --path-remap data/ data/in-house/ --out-dir work/anomaly-detection/shared/splits  # derive one-class AD splits
python -m src.experiments.anomaly_detection.normal_subclass_split --src-dir work/anomaly-detection/shared/splits --out-dir work/anomaly-detection/shared/splits-normal-subclass  # normal-only 6-class splits (no abnormal image)
python -m src.experiments.anomaly_detection.abnormal_subsample_split --src-dir work/anomaly-detection/shared/splits --out-dir work/anomaly-detection/shared/splits-kctc --k 1 2 5 10 20 40 --draws 3  # AD splits whose train keeps k abnormal images (nested draws)
python -m src.main configs/examples/anomaly-detection.yaml   # one-class anomaly detection (fit on normals only)
python -m src.main configs/examples/anomaly-detection-adapt.yaml   # adapt the backbone on normals only (mode: adapt, Mean-Shifted Contrastive)
python -m scripts.run_campaign work/<series>/<NN>-<campaign> [--expand-only] [--only S] [--jobs N] [--sweep FILE]  # run a campaign sweep (configs/sweep.yaml by default)
python -m src.experiments.anomaly_detection.compare work/anomaly-detection/<NN>-<campaign>  # paired-by-split AD comparison report (configs/analysis.yaml)
python -m src.experiments.anomaly_detection.learning_curve work/anomaly-detection/<NN>-<campaign>  # learning curve over k labelled abnormal images (configs/learning_curve.yaml)

# Override config values with dot-notation
python -m src.main configs/diffusion.yaml --model.architecture.image_size 60
python -m src.main configs/diffusion.yaml --training.epochs 50 --data.loading.batch_size 16

# Testing (four-tier strategy) — always use venv/bin/pytest, not bare pytest
venv/bin/pytest -m unit                        # fast, CPU-only (< 100ms each) — run on every commit
venv/bin/pytest -m "unit or component"         # pre-push validation (< 1 min)
venv/bin/pytest -m "not smoke"                 # CI pipeline (< 5 min)
venv/bin/pytest -m smoke                       # GPU smoke tests — manual/weekly
venv/bin/pytest tests/experiments/diffusion/   # run tests for a single module
venv/bin/pytest -k "test_model"                # run tests matching a pattern
venv/bin/pytest --cov=src --cov-report=html    # with coverage

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # includes pytest, pytest-cov, etc.

# Docker (GPU training)
docker build -t kktsuji/nvidia-cuda12.8.1-cudnn-runtime-ubuntu24.04 .
docker run --rm -it --gpus all --network=host --shm-size=4g \
  -v $PWD:/work -w /work --user $(id -u):$(id -g) \
  kktsuji/nvidia-cuda12.8.1-cudnn-runtime-ubuntu24.04 \
  python3 -m src.main configs/diffusion.yaml
```

## Architecture

The project uses a **Vertical Slice** pattern. Each experiment type is fully self-contained under `src/experiments/<type>/`. Shared logic lives in composable utilities under `src/utils/`.

### Data Flow

1. `src/main.py` parses the YAML config and dispatches to the appropriate `setup_experiment_*` function
2. Each experiment function wires together: Config → DataLoader → Model → Trainer → Logger
3. Each trainer's `train()` method orchestrates the epoch loop, checkpointing, and validation
4. Metrics are dual-logged: application logs (Python `logging`) + experiment metrics (CSV/TensorBoard)

### Shared Utilities

- **`src/utils/checkpoint.py`** — `save_checkpoint`, `load_checkpoint`, `is_best_metric`: standalone checkpoint functions (used by ClassifierTrainer)
- **`src/utils/metrics_writer.py`** — `MetricsWriter`: composable CSV + TensorBoard metrics writing (used by both loggers)

### Experiment Slices

Each experiment under `src/experiments/<type>/` contains:

- `config.py` — validates the config strictly (all parameters must be explicitly specified, no implicit defaults)
- `trainer.py` — standalone trainer class (owns its training loop, checkpoint logic)
- `dataloader.py` — standalone dataloader class (duck typing)
- `logger.py` — standalone logger class, uses `MetricsWriter` for CSV + optional TensorBoard

### Configuration System

All parameters must be in the YAML config file. Individual values can be overridden via CLI (e.g., `--mode generate` or `--model.architecture.image_size 60`). Override keys must match existing keys in the config (typos are rejected). Values are auto-inferred (bool/None/int/float/str); wrap in quotes to force string (e.g., `--data.label "'0'"`).

Config priority: `CLI overrides > config_file`

Example configs are in `configs/examples/*.yaml`. Merging is done by `src/utils/config.py::merge_configs()` (deep merge for dicts, replacement for scalars/lists). Each experiment's `config.py` validates the user config strictly — there are no implicit defaults.

Key config sections: `experiment`, `mode` (train/generate/select/evaluate/run), `compute`, `model`, `data`, `output`, `training`, `generation`, `evaluation`, `logging`.

### Dataset System (`src/utils/data/`)

- **`SplitFileDataset`** — primary dataset for all training experiments; loads from a JSON split file produced by `data_preparation`. Used by both classifier and diffusion trainers.
- **`ImageFolderDataset`** — wraps `torchvision.ImageFolder` for raw class-directory layouts
- **`SimpleImageDataset`** — flat directory, label-free (for generation/unlabeled use)
- Factory: `get_dataset(dataset_type, root, transform, **kwargs)`

Split JSON files are generated by `src/experiments/data_preparation/prepare.py` and contain `train`/`val` entries with `path` + `label`, plus a `metadata.classes` dict.

### Sample Selection Specifics

- Non-training experiment (like `data_preparation`) — no trainer/optimizer
- Compares generated images to real images in feature space, per class
- Feature extraction via `InceptionV3Classifier.extract_features()` or `ResNetClassifier.extract_features()` (pretrained, frozen backbone)
- Scoring: per-sample mean k-NN distance to real manifold (lower = more realistic)
- Optional realism filter: precision criterion from Kynkäänniemi et al. (2019)
- Selection modes: `top_k`, `percentile`, `threshold` (mutually exclusive, config-driven)
- Outputs accepted samples JSON in SplitFileDataset-compatible format (`"train"` key), so downstream experiments can load selections directly
- Config sections: `feature_extraction`, `data.real` (split_file or directory), `data.generated`, `scoring`, `selection`, `dataset_metrics`
- **Selection evaluation report** (`evaluation_report.py`): aggregates `evaluation.json` files from selection-eval experiments into comparison tables (CSV/Markdown) across diffusion variants, generation configs, and selection methods. Key metrics: FID, precision, recall for real-vs-selected and real-vs-generated comparisons. Run standalone via `python -m src.experiments.sample_selection.evaluation_report [--output-dir DIR]`
- Example config: `configs/examples/sample-selection.yaml`

### Anomaly Detection Specifics

- One-class detectors fit on **normal images only** (frozen backbone features); no abnormal image is used for fitting or for the decision threshold
- Backbone weights: ImageNet (`feature_extraction.checkpoint: null`), or the backbone of a classifier checkpoint of the same model (the `fc.*` head is skipped; every other key must match). A fine-tuned classifier saw abnormal images, so its features give an upper-bound diagnostic, not a one-class method
- Splits: `splits.py` derives `cv_binary_ad_split{N}.json` from the binary CV splits (read-only). Binary `train`/`val`/`test` are inherited unchanged (test folds pair with the classifier runs); the other normal subclasses are k-folded with the same `repeat_seed`/`fold` geometry into `normal_extra_{train,val,test}`. Every entry carries a `subclass` tag
- `data.normal_pool`: `all` (suspicious train + all extra normals) or `suspicious` (suspicious train only)
- Normal-subclass splits (`normal_subclass_split.py`): from each AD split, a `SplitFileDataset` split for classifying the 6 normal subclasses (labels in alphabetical order). `train`/`val`/`test` = the suspicious entries of the AD `train`/`val`/`test` + `normal_extra_{train,val,test}`; no abnormal image anywhere. A classifier fine-tuned on it gives a backbone learned without CTC labels, usable via `feature_extraction.checkpoint`
- k-abnormal splits (`abnormal_subsample_split.py`): copies of each AD split whose `train` keeps only `k` of its abnormal (label-1) images; every other entry and partition is unchanged, so the test folds stay paired. Draws are nested (draw `d` shuffles the sorted abnormal train paths with `random.Random("<repeat_seed>-split<N>-draw<d>")`, each `k` keeps the first `k`). Outputs `cv_binary_ad_split{N}_k{K}_d{D}.json`, recording `k`, the draw, the kept paths and the source SHA-256 in `metadata.abnormal_subsample`. For a learning curve over the number of labelled abnormal images. `val` keeps all its abnormal images, so a run using only `k` labels must not select its checkpoint by a val metric (use `final_model.pth`, not `best_model.pth`)
- Methods (`method.type`): `knn` (reuses `compute_knn_scores`), `mahalanobis` (Ledoit-Wolf), `patchcore` (hooked intermediate layers, greedy coreset). Higher score = more anomalous
- Features are cached per spec in `feature_extraction.cache_dir` (ImageNet features are split-independent; a checkpoint enters the spec by content hash, so per-split checkpoints get separate caches); the series keeps splits and caches in `work/anomaly-detection/shared/` (see "Series and Campaigns")
- Threshold = `threshold.normal_percentile` of the normal validation scores
- `mode: adapt` (`adapt.py`): adapts the backbone on the training normals of `data.normal_pool` with the Mean-Shifted Contrastive loss (Reiss & Hoshen 2023; center = normalized mean of the initial normalized features, NT-Xent on mean-shifted features). No abnormal image or label is used. Only parameters matching `adaptation.trainable_layers` are trained (never the `fc.*` head); `adaptation.update_frozen_bn_stats` (required) decides whether the BatchNorm running statistics of the other layers are re-estimated (`true`, as the classifier training does) or kept (`false`); augmentation is geometry and blur only (never color). `feature_extraction.checkpoint` must be null (start from ImageNet); `feature_extraction.batch_size` batches the center computation and `adaptation.batch_size` the training. Writes `reports/adaptation.json`, `reports/adaptation_history.csv`, then `checkpoints/final_model.pth` last (classifier checkpoint format, for `feature_extraction.checkpoint`)
- Output: classifier-compatible `reports/evaluation.json` (`pr_auc` = average precision, `positive_class: 1`, plus `pr_auc_vs_all_normals`, and `checkpoint_sha256` + `checkpoint` (relative to `reports/`) when a checkpoint is used), `predictions_test.npz` (`scores`, not `probs`), `subclass_scores.csv`, `subclass_summary.csv`, `subclass_scores.png`. Use `output.base_dir` = `<campaign>/runs/split{N}/<family>/<exp>/seed0` so `cross_split_report.load_per_split_results` reads it
- **Campaign comparison** (`compare.py`): reads `<campaign>/configs/analysis.yaml` and writes `<campaign>/reports/` (`summary.csv`, `comparisons.csv`, `subclass_auc.csv`, `report.md`, `<metric>_by_condition.png`).
  - Unit: the split. Reference arms are seed-averaged per split, as in `cross_split_report`. Only finished runs (with `evaluation.json`, which the runner writes last) are used in every part of the report.
  - Before reporting, every finished run that recorded a checkpoint is checked against that file's current hash; a missing or changed checkpoint (e.g. an earlier stage re-run with `--force`) stops the report and lists the runs to redo.
  - Three test families, BH-corrected separately: vs chance (the abnormal fraction of the test fold for `pr_auc`, 0.5 for `roc_auc`, skipped for other metrics), vs classifier references of the same backbone, and normal pool (`all` vs `suspicious`).
  - `condition_references` (required; `{}` for none) adds one family `vs_<name>` per entry: each listed condition against one named run of another tree (`base_dir` campaign-relative, `family`, `pairs: {condition: reference experiment}`), e.g. the same method and pool in an earlier campaign. Its section title is the entry's `label`.
- **Learning curve** (`learning_curve.py`): reads `<campaign>/configs/learning_curve.yaml` and writes `reports/learning_curve{.csv,_comparisons.csv,.md,.png}` for arms trained on k-abnormal splits.
  - Each arm (`family`, `template` with exactly `{k}` and `{draw}`, `full` run on the unsubsampled split or null). Per split, an arm's value at k is the mean over draws; a split enters a k only when every draw finished (left-out splits are listed in the report).
  - Families, BH-corrected separately: `vs_classifier` (each detector vs the `classifier_arm` at the same k, paired by split) and `vs_chance`.
  - Pre-set descriptive criteria: per detector, the smallest k from which the mean difference to the classifier stays >= `-criteria.ad_margin` for it and every larger k; the smallest k from which the classifier's mean stays >= `criteria.classifier_target`.
  - `zero_references` (per detector, `{}` for none) add untested k = 0 points from other trees. Checkpoints are checked as in `compare`.
- Example config: `configs/examples/anomaly-detection.yaml`

### Diffusion Specifics

- Model: DDPM with U-Net backbone (`src/experiments/diffusion/model.py::create_ddpm()`)
- EMA support built into training; checkpoint saves both `model_state_dict` and `ema_state_dict`
- **Generation mode** (`mode: generate`): uses lightweight sampler functions (no optimizer/dataloader needed); reads `generation.checkpoint` from config
- Conditional generation: set `model.conditioning.type: class` and `model.conditioning.num_classes`; `return_labels` is derived automatically from conditioning type
- Class balancing: `data.balancing` section supports `weighted_sampler`, `downsampling`, `upsampling`, `class_weights`

### Classifier Specifics

- **Evaluate mode** (`mode: evaluate`): inference-only, loads checkpoint via `evaluation.checkpoint`, computes per-class metrics (precision, recall, F1, AUC, confusion matrix)
- Factory constructor: `ClassifierTrainer.for_evaluation()` builds a trainer without optimizer/scheduler for evaluation-only workflows
- **Evaluation report** (`evaluation_report.py`): aggregates `evaluation.json` files across experiments into comparison tables for analyzing synthetic augmentation effectiveness
- Config: `evaluation.checkpoint` required in evaluate mode, `output.subdirs.reports` required; `evaluation.split` (default `val`) selects which split to score
- Output: the held-out `test` split writes the canonical `reports/evaluation.json` (what `evaluation_report` aggregates); other splits write `reports/evaluation_<split>.json` (e.g. `evaluation_val.json`) so a `val` pass doesn't clobber the `test` report. Per-split predictions are saved as `reports/predictions_<split>.npz` for the threshold analysis.

### Testing Conventions

- Tests mirror `src/` structure under `tests/`
- Markers: `@pytest.mark.unit`, `@pytest.mark.component`, `@pytest.mark.integration`, `@pytest.mark.smoke`
- Shared fixtures in `tests/conftest.py`: `device_cpu`, `device_gpu`, `mock_image_tensor`, `mock_batch_tensor`, `tmp_output_dir`, `mock_config_diffusion`, etc.
- All tests must pass on CPU (no GPU required for unit/component/integration)

### Output Structure

```bash
outputs/<experiment_name>/
├── logs/            # timestamped log files + config_<timestamp>.yaml snapshot
├── checkpoints/     # checkpoint_epoch_N.pth, best_model.pth, latest_checkpoint.pth, final_model.pth
├── samples/         # training-time visualization samples
├── generated/       # generation mode outputs
├── metrics/         # CSV metrics
├── reports/         # evaluation reports (evaluation.json)
└── tensorboard/     # TensorBoard event files (optional)
```

### Series and Campaigns (research output layout)

The canonical rules are in README.md, under "Organizing Experiments: Series and Campaigns". Follow them for every new research campaign (2026-10 onward; older trees are not migrated). Research series live under `work/`, which is gitignored and must never be committed (the repo is public). `outputs/` is only the default destination for ad-hoc runs:

- **Levels**:
  - **Series** `work/<series>/`: a research line; its campaigns share inputs.
  - **Campaign** `work/<series>/<NN>-<name>/`: the runs that answer one question and are analyzed together.
  - **Run**: one config execution.
- **Layout**:
  - Series: `README.md` (purpose + campaign ledger) and `shared/` (series-shared inputs and caches, never results).
  - Campaign: `README.md`, `configs/` (campaign definition + exact per-run configs), `runs/` (`split{N}/<family>/<experiment>/seed{S}/`, readable by `cross_split_report`), `reports/` (campaign-level analysis).
- **Paths**:
  - Inside a campaign, paths are relative to the campaign folder.
  - Series-shared inputs are reached via `../shared/`.
  - Inputs outside the series (raw data, binary CV splits, other series' results) may break when a folder moves. That is accepted, but they must be recorded in the campaign README.
- **Moving**: move a whole series as a unit.
- **Before starting a campaign**: create its README (question, design, external inputs with split-file hashes, commands, commit), then add it to the series ledger.
- **Running**: define `configs/base.yaml` + `configs/sweep.yaml` and run `python -m scripts.run_campaign <campaign>`.
  - `scripts/campaign_config.py` validates the sweep and expands it (axes → conditions × splits × seeds).
  - `scripts/run_campaign.py` stores portable per-run configs in `configs/runs/` and resolves the paths at run time.
  - Runs whose `done_marker` exists are skipped.
  - `{split}` in a `path_keys` value becomes the run's split index. `splits.template` must contain `{split}` and may also name axes (`..._k{k}_d{draw}.json`), replaced by the condition's value names. A multi-stage campaign uses several `sweep*.yaml` files (`--sweep`); condition names must be unique across them (checked by the driver).

### Pipeline (`scripts/run_pipeline.py`)

The pipeline script orchestrates the full synthetic augmentation workflow. All experiment configuration (phase flags, runner/infra settings, seeds, global classifier overrides, and the baseline/fine-tune variant matrix) lives in a YAML config — `configs/pipeline.yaml` by default (gitignored, like other `configs/*.yaml`; template at `configs/examples/pipeline.yaml`). Run with `python -m scripts.run_pipeline [configs/pipeline.yaml]`. The driver (`scripts/run_pipeline.py`) keeps only the generic orchestration; `scripts/pipeline_config.py` strictly validates the config. Phases are toggled via the `phases.*` flags in the config:

- **data_preparation** — prepare the train/val split (GPU/Docker)
- **baseline_classifier** — train the real-only reference classifiers (GPU/Docker). The baseline path applies only its own overrides (balancing etc.) and does not force a transfer depth, so the backbone freeze state is inherited from the classifier base config: it is a head-only "D0" reference only when that config sets `freeze_backbone: true`, otherwise a full fine-tune
- **ft_classifier** — train the fine-tuned frozen-depth sweep variants (GPU/Docker)
- **evaluation** — evaluate each trained classifier, interleaved per experiment (GPU/Docker)
- **summarize** — aggregate classifier evaluation reports (CPU-only, no Docker). Runs `src.experiments.classifier.evaluation_report` to produce cross-experiment comparison tables

### Notifications

Optional Slack notifications on experiment completion/failure via `SLACK_WEBHOOK_URL` in `.env`.

## Tooling

- **Python linting/formatting**: `ruff` (replaces flake8/isort/black)
- **Python type checking**: `pyright`
- **Markdown formatting**: `prettier`

## Post-Modification Checklist

When modifying source code (`src/`), always update or add related tests (`tests/`) to cover the changes.

After modifying any project files, always run the following checks in order and fix any errors before finishing:

1. `bash .husky/pre-commit` — **Pre-commit hooks**: Verify all Husky pre-commit hooks pass by running the same script Git uses. This runs: `ruff check` (lint), `ruff format --check` (format), `pyright` (type check), and `prettier --check "**/*.md"` (Markdown format).
2. `venv/bin/python tests/fixtures/mock_data/create_mock_images.py` — **Generate test fixtures**: Run the script to create mock images needed for tests.
3. `venv/bin/pytest --cov=src -m unit` — **Run unit tests & check coverage**: Run unit tests and confirm that unit-test coverage remains above 80%. If below, add tests to bring it above 80% before finishing.
4. `venv/bin/pytest -m "component or integration"` — **Run component & integration tests**: Run remaining non-smoke tests.
