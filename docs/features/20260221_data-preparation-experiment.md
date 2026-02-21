# Data Preparation Experiment

## Overview

Create a new `data_preparation` experiment that separates dataset splitting from training experiments. Currently, `DiffusionDataLoader` and `ClassifierDataLoader` each accept `train_path` and `val_path` directly, requiring pre-organized data directories. This refactoring introduces a dedicated experiment that:

1. Scans class directories for image files
2. Performs a stratified, reproducible train/val split (per-class shuffling with a seed)
3. Saves the split as a JSON file with metadata
4. Allows training experiments (diffusion, classifier) to load data from the split JSON

**Objective**: Decouple data splitting (with its own seed) from model training (with its own seed), enabling reproducible splits that are reusable across multiple experiment types.

### Design

```
[data_preparation experiment]          [diffusion / classifier experiment]
  reads class dirs                       reads split JSON
  shuffles with split seed=42            trains with compute seed=123
  saves train_val_split.json
         ↓ (outputs/splits/)        ←── referenced by data.split_file
```

### Key Decisions

- **JSON** for split files (machine-generated, rarely hand-edited; Python built-in `json` module, no extra dependency)
- **YAML** remains for config files (human-edited, consistent with existing project)
- **Separate seeds**: `data_preparation.seed` for splitting, `compute.seed` for training
- **Relative paths** in JSON for portability across machines
- **Stratified split**: each class is split independently to maintain class balance

### New File Structure

```
src/experiments/
  ├── data_preparation/
  │   ├── __init__.py
  │   ├── default.yaml          ← default config
  │   ├── config.py             ← config validation
  │   └── prepare.py            ← split generation logic
  └── diffusion/
  │   ├── default.yaml          ← modified: data.paths → data.split_file
  │   ├── dataloader.py         ← modified: reads split JSON
  │   └── ...
  └── classifier/
      ├── default.yaml          ← modified: data.paths → data.split_file
      ├── dataloader.py         ← modified: reads split JSON
      └── ...
```

### Generated JSON Format

```json
{
  "metadata": {
    "created_at": "2026-02-21T12:00:00",
    "seed": 42,
    "train_ratio": 0.8,
    "total_samples": 600,
    "train_samples": 480,
    "val_samples": 120,
    "classes": {
      "normal": 0,
      "abnormal": 1
    },
    "class_samples": {
      "normal": { "total": 495, "train": 396, "val": 99 },
      "abnormal": { "total": 105, "train": 84, "val": 21 }
    },
    "source_paths": {
      "normal": "data/0.Normal",
      "abnormal": "data/1.Abnormal"
    }
  },
  "train": [
    { "path": "data/0.Normal/img001.png", "label": 0 },
    { "path": "data/1.Abnormal/img042.png", "label": 1 }
  ],
  "val": [
    { "path": "data/0.Normal/img099.png", "label": 0 },
    { "path": "data/1.Abnormal/img201.png", "label": 1 }
  ]
}
```

### Config Changes Summary

| File                            | Change                                          |
| ------------------------------- | ----------------------------------------------- |
| `data_preparation/default.yaml` | New file                                        |
| `data_preparation/config.py`    | New file — validation                           |
| `data_preparation/prepare.py`   | New file — split logic                          |
| `diffusion/default.yaml`        | `data.paths` → `data.split_file`                |
| `diffusion/dataloader.py`       | Read from split JSON instead of directory paths |
| `diffusion/config.py`           | Update validation for new data section          |
| `classifier/default.yaml`       | `data.paths` → `data.split_file`                |
| `classifier/dataloader.py`      | Read from split JSON instead of directory paths |
| `classifier/config.py`          | Update validation for new data section          |
| `src/main.py`                   | Add `data_preparation` experiment dispatch      |
| `src/utils/cli.py`              | Add `data_preparation` to valid experiments     |
| `src/data/datasets.py`          | Add `SplitFileDataset` class                    |

### Time Estimate

| Phase                                 | Estimate       |
| ------------------------------------- | -------------- |
| Phase 1: Data preparation experiment  | 1–2 hours      |
| Phase 2: New dataset class            | 30 min         |
| Phase 3: Update diffusion experiment  | 1 hour         |
| Phase 4: Update classifier experiment | 1 hour         |
| Phase 5: Update main.py & CLI         | 30 min         |
| Phase 6: Tests                        | 2–3 hours      |
| Phase 7: Documentation                | 30 min         |
| **Total**                             | **~6–8 hours** |

## Implementation Checklist

- [x] Phase 1: Create `data_preparation` experiment
  - [x] Task 1.1: Create `src/experiments/data_preparation/__init__.py`
  - [x] Task 1.2: Create `src/experiments/data_preparation/default.yaml`
  - [x] Task 1.3: Create `src/experiments/data_preparation/config.py` (validation)
  - [x] Task 1.4: Create `src/experiments/data_preparation/prepare.py` (split logic)
- [x] Phase 2: Add `SplitFileDataset` to `src/data/datasets.py`
  - [x] Task 2.1: Implement `SplitFileDataset` that reads split JSON and loads images
  - [x] Task 2.2: Update `get_dataset()` factory to include new dataset type
- [x] Phase 3: Update diffusion experiment
  - [x] Task 3.1: Modify `src/experiments/diffusion/default.yaml` (`data.paths` → `data.split_file`)
  - [x] Task 3.2: Modify `src/experiments/diffusion/dataloader.py` to read split JSON
  - [x] Task 3.3: Modify `src/experiments/diffusion/config.py` to validate new data section
- [x] Phase 4: Update classifier experiment
  - [x] Task 4.1: Modify `src/experiments/classifier/default.yaml` (`data.paths` → `data.split_file`)
  - [x] Task 4.2: Modify `src/experiments/classifier/dataloader.py` to read split JSON
  - [x] Task 4.3: Modify `src/experiments/classifier/config.py` to validate new data section
- [x] Phase 5: Update entry points
  - [x] Task 5.1: Add `setup_experiment_data_preparation()` to `src/main.py`
  - [x] Task 5.2: Add `data_preparation` to valid experiments in `src/utils/cli.py`
- [x] Phase 6: Tests
  - [x] Task 6.1: Create `tests/experiments/data_preparation/` with unit tests for `prepare.py`
  - [x] Task 6.2: Create `tests/experiments/data_preparation/test_config.py`
  - [x] Task 6.3: Add `tests/data/test_split_file_dataset.py` for `SplitFileDataset`
  - [x] Task 6.4: Update `tests/experiments/diffusion/test_dataloader.py` for split JSON loading
  - [x] Task 6.5: Update `tests/experiments/classifier/test_dataloader.py` for split JSON loading
  - [x] Task 6.6: Add test fixtures: mock split JSON files in `tests/fixtures/`
  - [x] Task 6.7: Update `tests/test_main.py` for `data_preparation` dispatch
  - [x] Task 6.8: Run all tests to verify no regressions
- [x] Phase 7: Documentation
  - [x] Task 7.1: Update `docs/standards/architecture.md` with `data_preparation` experiment
  - [x] Task 7.2: Update `README.md` with new experiment usage

## Phase Details

### Phase 1: Create `data_preparation` experiment

#### Task 1.1: `src/experiments/data_preparation/__init__.py`

Empty init file for the package.

#### Task 1.2: `src/experiments/data_preparation/default.yaml`

```yaml
# ------------------------------------------------------------------------------
# DATA PREPARATION CONFIGURATION
# ------------------------------------------------------------------------------
experiment: data_preparation

# Class directories to scan for images
classes:
  normal: "data/0.Normal"
  abnormal: "data/1.Abnormal"

# Split configuration
split:
  seed: 42
  train_ratio: 0.8
  save_dir: "outputs/splits"
  split_file: "train_val_split.json"
```

#### Task 1.3: `src/experiments/data_preparation/config.py`

Validation function that checks:

- `experiment` is `data_preparation`
- `classes` is a non-empty dict with valid directory paths
- `split.seed` is int or null
- `split.train_ratio` is float in (0.0, 1.0)
- `split.save_dir` is a string
- `split.split_file` is a string ending with `.json`

#### Task 1.4: `src/experiments/data_preparation/prepare.py`

Core logic:

1. Scan each class directory for image files (same extensions as `ImageFolderDataset`)
2. Sort file list for determinism before shuffling
3. Shuffle each class independently with `random.Random(seed)`
4. Split each class list by `train_ratio`
5. Merge train lists and val lists
6. Build JSON structure with metadata
7. Write JSON to `save_dir/split_file`
8. Log summary statistics (total, per-class, train/val counts)

Key implementation details:

- Use `random.Random(seed)` (not global random) for isolation
- Store **full relative paths** from project root (e.g., `data/0.Normal/img001.png`)
- Skip regeneration if file exists (add `force: false` config option)
- Validate that all class directories exist and contain images

### Phase 2: Add `SplitFileDataset`

#### Task 2.1: `SplitFileDataset` in `src/data/datasets.py`

A new `BaseDataset` subclass that:

- Takes a split JSON path and a split key (`"train"` or `"val"`)
- Reads the JSON file
- Uses full relative paths directly (from project root, e.g., `data/0.Normal/img001.png`)
- Loads images from disk with given transform
- Returns `(image, label)` or just `image` based on `return_labels`

```python
class SplitFileDataset(BaseDataset):
    def __init__(self, split_file, split, transform=None, return_labels=True):
        ...
    def __len__(self): ...
    def __getitem__(self, index): ...
    def get_classes(self): ...
    def get_class_counts(self): ...
```

Properties: `classes`, `class_to_idx`, `samples`, `targets` — for compatibility with existing code that accesses these on the dataset (e.g., `train_loader.dataset.classes` in `main.py`).

#### Task 2.2: Update `get_dataset()` factory

Add `"splitfile"` as a new dataset type option.

### Phase 3: Update diffusion experiment

#### Task 3.1: Modify `diffusion/default.yaml`

Replace:

```yaml
data:
  paths:
    train: data/train
    val: null
```

With:

```yaml
data:
  split_file: "outputs/splits/train_val_split.json"
```

Keep `loading` and `augmentation` sections unchanged.

#### Task 3.2: Modify `diffusion/dataloader.py`

Replace `train_path`/`val_path` constructor parameters with `split_file` parameter.

```python
class DiffusionDataLoader(BaseDataLoader):
    def __init__(self, split_file, batch_size=32, ...):
        self.split_file = split_file
        # Load and validate split JSON
        ...

    def get_train_loader(self):
        dataset = SplitFileDataset(
            split_file=self.split_file,
            split="train",
            transform=self._get_train_transform(),
            return_labels=self.return_labels,
        )
        ...

    def get_val_loader(self):
        dataset = SplitFileDataset(
            split_file=self.split_file,
            split="val",
            transform=self._get_val_transform(),
            return_labels=self.return_labels,
        )
        ...
```

#### Task 3.3: Modify `diffusion/config.py`

Update `_validate_data_config()`:

- Replace validation of `data.paths.train` / `data.paths.val` with validation of `data.split_file`
- Validate that `data.split_file` is a string path
- Keep validation of `data.loading` and `data.augmentation` unchanged

### Phase 4: Update classifier experiment

Same pattern as Phase 3, applied to classifier files.

#### Task 4.1: Modify `classifier/default.yaml`

Replace `data.paths` with `data.split_file`.

#### Task 4.2: Modify `classifier/dataloader.py`

Same refactoring as diffusion dataloader — replace `train_path`/`val_path` with `split_file`.

#### Task 4.3: Modify `classifier/config.py`

Update data validation section.

### Phase 5: Update entry points

#### Task 5.1: Add `setup_experiment_data_preparation()` to `main.py`

```python
def setup_experiment_data_preparation(config):
    from src.experiments.data_preparation.config import validate_config
    from src.experiments.data_preparation.prepare import prepare_split

    validate_config(config)
    prepare_split(config)
```

This is much simpler than training experiments — no model, no optimizer, no device.

Add to the dispatch in `main()`:

```python
elif experiment == "data_preparation":
    setup_experiment_data_preparation(config)
```

#### Task 5.2: Update `src/utils/cli.py`

Add `"data_preparation"` to `valid_experiments` list in both `parse_args()` and `validate_config()`.

### Phase 6: Tests

#### Task 6.1: `tests/experiments/data_preparation/test_prepare.py`

Test cases:

- Split with known seed produces deterministic output
- Train/val ratio is respected per class
- All images from source dirs appear exactly once across train+val
- Relative paths are correct
- JSON metadata is complete and accurate
- Empty class directory raises error
- Invalid train_ratio raises error
- Force regeneration overwrites existing file
- Skip regeneration when file exists and force=false

#### Task 6.2: `tests/experiments/data_preparation/test_config.py`

Test cases:

- Valid config passes validation
- Missing `classes` raises error
- Invalid `train_ratio` (0, 1, negative, >1) raises error
- Missing `split.split_file` raises error

#### Task 6.3: `tests/data/test_split_file_dataset.py`

Test cases:

- Load dataset from mock split JSON
- `__getitem__` returns correct (image, label) pairs
- `return_labels=False` returns only images
- `classes`, `class_to_idx`, `get_class_counts()` work correctly
- Invalid split key raises error
- Missing image file raises informative error

#### Task 6.4–6.5: Update existing dataloader tests

Update diffusion and classifier dataloader tests to use mock split JSON files instead of mock directory paths.

#### Task 6.6: Test fixtures

Create `tests/fixtures/splits/mock_split.json` with a small mock split for testing.

#### Task 6.7: Update `tests/test_main.py`

Add test for `data_preparation` experiment dispatch.

#### Task 6.8: Run all tests

```bash
python -m pytest tests/ -v
```

### Phase 7: Documentation

#### Task 7.1: Update `docs/standards/architecture.md`

Add `data_preparation` experiment to the directory structure and component responsibilities sections.

#### Task 7.2: Update `README.md`

Add usage example:

```bash
# Step 1: Prepare dataset split
python -m src.main configs/data_preparation/default.yaml

# Step 2: Train model using the split
python -m src.main configs/diffusion/default.yaml
```
