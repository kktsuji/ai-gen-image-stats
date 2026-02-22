# Remove `log_sample_comparison` and Unify with `log_images`

## Overview

`DiffusionLogger.log_sample_comparison()` is functionally a subset of `log_images()`. Both methods build a filename from `tag/step/epoch` and call `save_image`, but `log_sample_comparison` saves to a dedicated `quality/` directory while `log_images` saves to `samples/`. The only behavioral differences are:

| Feature                | `log_images` | `log_sample_comparison` |
| ---------------------- | :----------: | :---------------------: |
| Output directory       |  `samples/`  |       `quality/`        |
| TensorBoard logging    |      ✅      |           ❌            |
| Internal tracking list |      ✅      |           ❌            |
| Flexible kwargs        |      ✅      |     ❌ (hardcoded)      |

**Goal:** Remove `log_sample_comparison` and the separate `quality/` directory. All sample images (including quality comparisons) will be saved to the `samples/` directory via `log_images`.

**Approach:** Replace every call to `log_sample_comparison` with a call to `log_images`, using the existing `tag` parameter (e.g., `"quality_comparison"`) to distinguish quality-comparison images by filename rather than by subdirectory.

### File changes summary

| File                                                              | Change                                                                                                                                                                                  |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/experiments/diffusion/logger.py`                             | Remove `log_sample_comparison()`, remove `quality_dir` creation, update docstring                                                                                                       |
| `src/experiments/diffusion/trainer.py`                            | Remove `log_sample_comparison_interval` param; replace call with `log_images`                                                                                                           |
| `src/experiments/diffusion/config.py`                             | Remove `log_sample_comparison_interval` from validation list                                                                                                                            |
| `src/experiments/diffusion/default.yaml`                          | Remove `log_sample_comparison_interval` config key                                                                                                                                      |
| `src/main.py`                                                     | Remove `log_sample_comparison_interval` kwarg passed to `DiffusionTrainer`                                                                                                              |
| `tests/experiments/diffusion/test_logger.py`                      | Remove `TestLogSampleComparison` class; update integration test that uses `log_sample_comparison`; remove `quality/` dir assertion                                                      |
| `tests/experiments/diffusion/test_trainer.py`                     | Remove `log_sample_comparison` from mock logger; remove `log_sample_comparison_interval` from all trainer instantiations; update/remove `test_log_sample_comparison_called_at_interval` |
| `tests/experiments/diffusion/test_trainer_sampler_integration.py` | Remove `log_sample_comparison_interval` param                                                                                                                                           |
| `tests/experiments/diffusion/test_config.py`                      | Remove `log_sample_comparison_interval` assertion                                                                                                                                       |
| `tests/integration/test_diffusion_pipeline.py`                    | Remove `log_sample_comparison_interval` from all config dicts and trainer calls                                                                                                         |
| `tests/conftest.py`                                               | Remove `log_sample_comparison_interval` from fixture config                                                                                                                             |

## Implementation Checklist

- [x] Phase 1: Remove `log_sample_comparison` from logger
  - [x] Task 1.1: Remove `quality_dir` creation and `log_sample_comparison()` method from `DiffusionLogger`
  - [x] Task 1.2: Update `DiffusionLogger` class docstring (remove `quality/` directory reference)
- [x] Phase 2: Remove `log_sample_comparison_interval` from trainer
  - [x] Task 2.1: Remove `log_sample_comparison_interval` param from `DiffusionTrainer.__init__` (param, docstring, assignment, `viz_enabled`)
  - [x] Task 2.2: Remove `log_sample_comparison_interval` from `_should_generate_samples()`
  - [x] Task 2.3: Replace `log_sample_comparison` call in `_generate_samples()` with `log_images` call using tag `"quality_comparison"`
- [x] Phase 3: Remove from config layer
  - [x] Task 3.1: Remove `log_sample_comparison_interval` from `config.py` validation
  - [x] Task 3.2: Remove `log_sample_comparison_interval` from `default.yaml`
  - [x] Task 3.3: Remove `log_sample_comparison_interval` kwarg from `src/main.py`
- [x] Phase 4: Update tests
  - [x] Task 4.1: Remove `TestLogSampleComparison` class from `test_logger.py`
  - [x] Task 4.2: Update integration test in `test_logger.py` (replace `log_sample_comparison` with `log_images`, update assertions)
  - [x] Task 4.3: Remove `quality/` directory assertion from `test_logger.py` init tests
  - [x] Task 4.4: Remove `log_sample_comparison` from mock logger in `test_trainer.py`
  - [x] Task 4.5: Remove `log_sample_comparison_interval` from all trainer instantiations in `test_trainer.py`
  - [x] Task 4.6: Update/remove `test_log_sample_comparison_called_at_interval` in `test_trainer.py`
  - [x] Task 4.7: Remove `log_sample_comparison_interval` from `test_trainer_sampler_integration.py`
  - [x] Task 4.8: Remove `log_sample_comparison_interval` from `test_diffusion_pipeline.py`
  - [x] Task 4.9: Remove `log_sample_comparison_interval` from `test_config.py`
  - [x] Task 4.10: Remove `log_sample_comparison_interval` from `tests/conftest.py`
- [x] Phase 5: Validate
  - [x] Task 5.1: Run all diffusion-related tests and confirm they pass
  - [x] Task 5.2: Run full test suite to check for regressions

## Phase Details

### Phase 1: Remove `log_sample_comparison` from logger

In `src/experiments/diffusion/logger.py`:

- Delete `self.quality_dir` creation (lines 81, 86).
- Delete `log_sample_comparison()` method (lines 361–389).
- Update class docstring to remove the `quality/` directory mention.

### Phase 2: Remove `log_sample_comparison_interval` from trainer

In `src/experiments/diffusion/trainer.py`:

- Remove the `log_sample_comparison_interval` constructor parameter, its docstring mention, and `self.log_sample_comparison_interval` assignment.
- Remove from `viz_enabled` check list.
- Remove from `_should_generate_samples()` interval list.
- In `_generate_samples()`, replace the `# 3. log_sample_comparison` block (lines 1077–1087) with a `log_images` call that uses tag `"quality_comparison"` and the same `log_images_interval` check that already exists in step 2.

### Phase 3: Remove from config layer

- `config.py`: Remove `"log_sample_comparison_interval"` from the validation loop list.
- `default.yaml`: Remove the `log_sample_comparison_interval: 50` line.
- `src/main.py`: Remove the `log_sample_comparison_interval=...` kwarg.

### Phase 4: Update tests

- `test_logger.py`: Delete `TestLogSampleComparison` class (4 tests). Update the full-lifecycle integration test to use `log_images` instead of `log_sample_comparison`, and assert the image lands in `samples/` instead of `quality/`. Remove the `quality` dir existence assertion.
- `test_trainer.py`: Remove `log_sample_comparison` from mock logger class. Remove all `log_sample_comparison_interval=...` kwargs. Convert `test_log_sample_comparison_called_at_interval` to verify that quality comparison images now appear in `logged_images` via `log_images`.
- Other test files: Simply remove the `log_sample_comparison_interval` config/param references.

### Phase 5: Validate

Run the test suite to confirm no regressions.
