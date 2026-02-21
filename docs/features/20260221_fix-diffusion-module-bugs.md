# Fix Diffusion Module Bugs

## Overview

This plan addresses four bugs discovered during code investigation of `src/experiments/diffusion/`:

| #   | Severity   | Type          | File           | Summary                                                                        |
| --- | ---------- | ------------- | -------------- | ------------------------------------------------------------------------------ |
| 1   | **HIGH**   | Runtime error | `trainer.py`   | `logger` undefined — should be `_logger` (3 occurrences in `load_checkpoint`)  |
| 2   | **HIGH**   | Dead code     | `trainer.py`   | `raise` before `_logger.critical()` makes logging and second raise unreachable |
| 3   | **MEDIUM** | Logic error   | `config.py`    | `guidance_scale` validation rejects `< 1.0` but model code supports `>= 0.0`   |
| 4   | **LOW**    | Documentation | `default.yaml` | Comment typos using `-` instead of `=` / `:`                                   |

**Objective:** Fix all four bugs, update existing tests that encode the incorrect behavior, add new tests for previously untested error paths, and verify no regressions.

**Files changed:**

- `src/experiments/diffusion/trainer.py` (bugs 1, 2)
- `src/experiments/diffusion/config.py` (bug 3)
- `src/experiments/diffusion/default.yaml` (bug 4)
- `tests/experiments/diffusion/test_trainer.py` (new tests for bugs 1, 2)
- `tests/experiments/diffusion/test_config.py` (update + new tests for bug 3)

**Time estimate:** ~1 hour

## Implementation Checklist

- [x] Phase 1: Fix `logger` → `_logger` in `load_checkpoint` (Bug 1)
  - [x] Task 1.1: Replace all 3 `logger.exception(...)` calls with `_logger.exception(...)` in `load_checkpoint`
  - [x] Task 1.2: Add test that `load_checkpoint` handles a missing file without `NameError`
  - [x] Task 1.3: Add test that `load_checkpoint` handles a corrupt checkpoint without `NameError`
- [x] Phase 2: Fix unreachable code in `train_epoch` (Bug 2)
  - [x] Task 2.1: Remove the premature `raise` on line 286, retaining the `_logger.critical()` call followed by the `raise`
  - [x] Task 2.2: Add test that verifies the critical log message is emitted when batch data has unexpected length
- [x] Phase 3: Fix `guidance_scale` validation logic (Bug 3)
  - [x] Task 3.1: Change validation in `_validate_training_config` from `< 1.0` to `< 0.0`
  - [x] Task 3.2: Change validation in `_validate_generation_config` from `< 1.0` to `< 0.0`
  - [x] Task 3.3: Update existing test `test_invalid_guidance_scale` to use a truly invalid value (e.g., `-1.0`)
  - [x] Task 3.4: Add tests that `guidance_scale: 0.0` and `guidance_scale: 0.5` are accepted
  - [x] Task 3.5: Add test for `generation.sampling.guidance_scale` validation
- [x] Phase 4: Fix YAML comment typos (Bug 4)
  - [x] Task 4.1: Fix 4 comment typos in `default.yaml`
- [x] Phase 5: Run all tests and verify no regressions
  - [x] Task 5.1: Run the full test suite and confirm all tests pass

## Phase Details

### Phase 1: Fix `logger` → `_logger` in `load_checkpoint` (Bug 1)

**Problem:** `load_checkpoint()` in `trainer.py` references a bare `logger` variable that does not exist in the module scope. The module defines `_logger = logging.getLogger(__name__)` at line 24. This causes a `NameError` at runtime — and only during error handling, which masks the original checkpoint-loading error.

**Locations (3 occurrences):**

| Line | Current                                   | Fix                                        |
| ---- | ----------------------------------------- | ------------------------------------------ |
| 740  | `logger.exception(f"Error details: {e}")` | `_logger.exception(f"Error details: {e}")` |
| 752  | `logger.exception(f"Error details: {e}")` | `_logger.exception(f"Error details: {e}")` |
| 768  | `logger.exception(f"Error details: {e}")` | `_logger.exception(f"Error details: {e}")` |

**Tests to add:**

- `test_load_checkpoint_missing_file`: Call `load_checkpoint` with a nonexistent path and assert `FileNotFoundError` is raised (not `NameError`).
- `test_load_checkpoint_corrupt_file`: Call `load_checkpoint` with a corrupted file and assert the original error propagates (not `NameError`).

---

### Phase 2: Fix unreachable code in `train_epoch` (Bug 2)

**Problem:** In `train_epoch()`, line 286 raises `ValueError` immediately, making lines 287-290 (the `_logger.critical()` call and the duplicate `raise`) unreachable.

**Current code (lines 285-290):**

```python
else:
    raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
    _logger.critical(
        f"Unexpected batch data format: length {len(batch_data)}"
    )
    raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
```

**Fixed code:**

```python
else:
    _logger.critical(
        f"Unexpected batch data format: length {len(batch_data)}"
    )
    raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
```

**Test to add:**

- `test_train_epoch_unexpected_batch_length`: Mock the train loader to yield a tuple of length 3. Assert `ValueError` is raised and verify the critical log message is emitted.

---

### Phase 3: Fix `guidance_scale` validation logic (Bug 3)

**Problem:** Both `_validate_training_config` and `_validate_generation_config` reject `guidance_scale < 1.0`. However, the model code in `model.py` treats `guidance_scale > 0.0` as "guidance enabled" and `0.0` as "guidance disabled." This means:

- `guidance_scale: 0.0` (disable guidance) — **valid in code, rejected by validation**
- `guidance_scale: 0.5` (weak guidance) — **valid in code, rejected by validation**

The correct constraint is `guidance_scale >= 0.0` (non-negative).

**Locations (2 occurrences):**

| File        | Function                      | Current                                    | Fix                                        |
| ----------- | ----------------------------- | ------------------------------------------ | ------------------------------------------ |
| `config.py` | `_validate_training_config`   | `vis["guidance_scale"] < 1.0` → error      | `vis["guidance_scale"] < 0.0` → error      |
| `config.py` | `_validate_generation_config` | `sampling["guidance_scale"] < 1.0` → error | `sampling["guidance_scale"] < 0.0` → error |

Error messages should also be updated from `"must be >= 1.0"` to `"must be >= 0.0"`.

**Tests to update/add:**

- **Update** `test_invalid_guidance_scale`: Change the invalid value from `0.5` to `-1.0` and update expected message from `"must be >= 1.0"` to `"must be >= 0.0"`.
- **Add** `test_valid_guidance_scale_zero`: Set `guidance_scale: 0.0` and assert validation passes.
- **Add** `test_valid_guidance_scale_less_than_one`: Set `guidance_scale: 0.5` and assert validation passes.
- **Add** `test_invalid_generation_guidance_scale`: Set `generation.sampling.guidance_scale: -1.0` and assert `ValueError`.

---

### Phase 4: Fix YAML comment typos (Bug 4)

**Problem:** Several comments in `default.yaml` use `-` where `=` or `:` was intended.

| Line | Current text                                     | Fixed text                                       |
| ---- | ------------------------------------------------ | ------------------------------------------------ |
| 81   | `# Number of classes (required if type-"class")` | `# Number of classes (required if type="class")` |
| 155  | `# Only used when mode-train`                    | `# Only used when mode: train`                   |
| 195  | `# Classifier-free guidance scale (>-1.0)`       | `# Classifier-free guidance scale (>= 0.0)`      |
| 197  | `# Only used when mode-generate`                 | `# Only used when mode: generate`                |

No test changes needed — these are comments only.

---

### Phase 5: Run all tests and verify no regressions

Run the full test suite to confirm:

1. All existing tests pass (with the updated `test_invalid_guidance_scale`).
2. All new tests pass.
3. No unexpected failures in other modules.

```bash
python -m pytest tests/ -v
```
