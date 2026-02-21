# Fix Diffusion Module Bugs (v4)

## Overview

This plan addresses six **new** bugs discovered during a deeper investigation of `src/experiments/diffusion/`, beyond those already fixed in v1–v3.

| #   | Severity   | Type            | File            | Summary                                                                                                          |
| --- | ---------- | --------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1   | **MEDIUM** | Visualization   | `logger.py`     | `log_images()` sends raw [-1, 1] images to TensorBoard without normalization — dark/black previews               |
| 2   | **MEDIUM** | Runtime error   | `trainer.py`    | `train()` and `resume_training()` crash with `UnboundLocalError` when `num_epochs=0`                             |
| 3   | **LOW**    | Type annotation | `trainer.py`    | `scheduler` type hint declares `Optional[_LRScheduler]` but `ReduceLROnPlateau` doesn't inherit from it          |
| 4   | **LOW**    | Dead parameter  | `sampler.py`    | `show_progress` accepted by `sample()` and `sample_with_intermediates()` but silently ignored                    |
| 5   | **LOW**    | Dead code       | `trainer.py`    | `self.samples_per_class` stored in constructor but never read — `_generate_samples()` recalculates independently |
| 6   | **LOW**    | Unused import   | `dataloader.py` | `get_normalization_transform` imported but never used                                                            |

**Objective:** Fix all six bugs, add new tests to cover each fix, and verify no regressions.

**Files changed:**

- `src/experiments/diffusion/logger.py` (bug 1)
- `src/experiments/diffusion/trainer.py` (bugs 2, 3, 5)
- `src/experiments/diffusion/sampler.py` (bug 4)
- `src/experiments/diffusion/dataloader.py` (bug 6)
- `tests/experiments/diffusion/test_logger.py` (new tests for bug 1)
- `tests/experiments/diffusion/test_trainer.py` (new tests for bugs 2, 5)
- `tests/experiments/diffusion/test_sampler.py` (new/updated tests for bug 4)

**Time estimate:** ~1.5 hours

## Implementation Checklist

- [ ] Phase 1: Fix `log_images()` TensorBoard normalization (Bug 1)
  - [ ] Task 1.1: Normalize [-1, 1] → [0, 1] before calling `safe_log_images()` in `log_images()`
  - [ ] Task 1.2: Add test that TensorBoard images from `log_images()` are in [0, 1] range
  - [ ] Task 1.3: Add test that negative pixel values are preserved (not clipped to black)
- [ ] Phase 2: Fix `UnboundLocalError` when `num_epochs=0` (Bug 2)
  - [ ] Task 2.1: Initialize `train_metrics` and `val_metrics` before the loop in `train()`
  - [ ] Task 2.2: Initialize `train_metrics` and `val_metrics` before the loop in `resume_training()`
  - [ ] Task 2.3: Add test that `train()` with `num_epochs=0` does not crash
  - [ ] Task 2.4: Add test that `resume_training()` with `num_epochs=0` does not crash
- [ ] Phase 3: Fix `scheduler` type hint (Bug 3)
  - [ ] Task 3.1: Widen type hint to accept both `_LRScheduler` and `ReduceLROnPlateau`
- [ ] Phase 4: Remove unused `show_progress` parameter from sampler (Bug 4)
  - [ ] Task 4.1: Remove `show_progress` from `sample()` signature and docstring
  - [ ] Task 4.2: Remove `show_progress` from `sample_with_intermediates()` signature and docstring
  - [ ] Task 4.3: Update any callers that pass `show_progress` to these methods
  - [ ] Task 4.4: Add test that `sample()` does not accept `show_progress` keyword
- [ ] Phase 5: Remove dead `samples_per_class` parameter (Bug 5)
  - [ ] Task 5.1: Remove `samples_per_class` from constructor parameter, docstrings, and `self.samples_per_class`
  - [ ] Task 5.2: Update `main.py` caller to stop passing `samples_per_class`
  - [ ] Task 5.3: Update any tests that pass `samples_per_class` to `DiffusionTrainer`
- [ ] Phase 6: Remove unused import (Bug 6)
  - [ ] Task 6.1: Remove `get_normalization_transform` import from `dataloader.py`
- [ ] Phase 7: Run all tests and verify no regressions
  - [ ] Task 7.1: Run the full test suite and confirm all tests pass

## Phase Details

### Phase 1: Fix `log_images()` TensorBoard normalization (Bug 1)

**Problem:** In `log_images()` (logger.py line ~263), the disk save correctly uses `save_image(..., normalize=True)` which handles [-1, 1] → [0, 1] remapping. However, the TensorBoard path passes **raw [-1, 1] tensors** directly to `safe_log_images()`:

```python
# Current code (logger.py line ~263)
if self.tb_writer is not None and self.tb_log_images:
    safe_log_images(self.tb_writer, f"images/{tag}", images, step)
```

`safe_log_images()` calls `writer.add_images()`, which expects float images in [0, 1]. Negative pixel values are clipped to 0, producing **dark/black** TensorBoard previews. This is the exact same class of bug that was fixed for `log_denoising_process()` in v2 Bug 1 — the denoising method was fixed to normalize, but `log_images()` was missed.

**Fix:** Normalize before sending to TensorBoard, matching what was done for `log_denoising_process()`:

```python
# Save to TensorBoard
if self.tb_writer is not None and self.tb_log_images:
    # Normalize [-1, 1] → [0, 1] for TensorBoard (same as denoising fix)
    tb_images = (images + 1.0) / 2.0
    tb_images = torch.clamp(tb_images, 0, 1)
    safe_log_images(self.tb_writer, f"images/{tag}", tb_images, step)
```

**Tests to add:**

- `test_log_images_tensorboard_normalization`: Create a `DiffusionLogger` with TensorBoard enabled, log images with values in [-1, 1]. Mock `safe_log_images` and verify the tensor passed has values in [0, 1].
- `test_log_images_negative_values_not_black`: Log images where all pixel values are -1.0 (which should map to 0.0, not be clipped). Verify the TensorBoard tensor is all 0.0, not some negative artifact. Log images where all pixel values are -0.5 (should map to 0.25). Verify the tensor is approximately 0.25.

---

### Phase 2: Fix `UnboundLocalError` when `num_epochs=0` (Bug 2)

**Problem:** In both `train()` (trainer.py lines ~505–637) and `resume_training()` (lines ~660–805), the final checkpoint save after the loop references `train_metrics` and `val_metrics`:

```python
# Current code in train() (lines ~505-637)
for epoch in range(num_epochs):  # never executes when num_epochs=0
    ...
    train_metrics = self.train_epoch()
    val_metrics = None
    ...

# After loop — train_metrics and val_metrics are unbound!
if checkpoint_dir is not None:
    final_path = checkpoint_dir / "final_model.pth"
    self.save_checkpoint(
        final_path,
        epoch=self._current_epoch,
        metrics={**train_metrics, **(val_metrics if val_metrics else {})},
    )
```

If `num_epochs=0`, the loop body never executes, and `train_metrics`/`val_metrics` are never assigned, causing `UnboundLocalError`. Config validation requires `epochs >= 1`, so this only triggers via direct API calls (e.g., `trainer.train(num_epochs=0)`) — but it's a latent crash in a public method.

**Fix:** Initialize default values before the loop in both methods:

```python
# In train()
train_metrics = {}
val_metrics = None

for epoch in range(num_epochs):
    ...
```

```python
# In resume_training()
train_metrics = {}
val_metrics = None

for epoch in range(num_epochs):
    ...
```

**Tests to add:**

- `test_train_zero_epochs_no_crash`: Create a trainer, call `train(num_epochs=0, checkpoint_dir=tmp_dir)`. Assert no exception is raised.
- `test_resume_training_zero_epochs_no_crash`: Create a trainer, train for 1 epoch, save checkpoint. Call `resume_training(checkpoint_path=..., num_epochs=0, checkpoint_dir=tmp_dir)`. Assert no exception is raised.

---

### Phase 3: Fix `scheduler` type hint (Bug 3)

**Problem:** `DiffusionTrainer.__init__` (trainer.py line ~112) declares:

```python
scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
```

After the v3 fix, `ReduceLROnPlateau` is correctly handled at runtime via `isinstance()` checks. However, `ReduceLROnPlateau` does **not** inherit from `_LRScheduler` — it inherits directly from `object`. The type hint is misleading and causes false positives in static analysis tools (mypy, pyright).

**Fix:** Widen the type hint to accept both scheduler types:

```python
scheduler: Optional[
    "torch.optim.lr_scheduler.LRScheduler"
] = None,
```

Note: In PyTorch >= 2.0, `LRScheduler` is the public alias that both `_LRScheduler` and `ReduceLROnPlateau` are accessible under. For broad compatibility, we can use `Optional[object]` with a docstring clarification, or a `Union`:

```python
scheduler: Optional[
    Union[
        torch.optim.lr_scheduler._LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]
] = None,
```

No tests needed — this is a type annotation fix only.

---

### Phase 4: Remove unused `show_progress` parameter from sampler (Bug 4)

**Problem:** In `DiffusionSampler.sample()` (sampler.py line ~102) and `sample_with_intermediates()` (line ~206), the `show_progress` parameter is accepted and documented as "Whether to display progress bar during sampling", but it is **never used** — neither method passes it to `model.sample()`, and no progress bar is created internally. Users calling `sampler.sample(num_samples=64, show_progress=True)` get **no progress bar** with zero indication it was ignored.

Note: `show_progress` IS correctly used in `sample_by_class()` (line ~335) for the class iteration loop — the problem is only in `sample()` and `sample_with_intermediates()`.

**Fix:** Remove `show_progress` from `sample()` and `sample_with_intermediates()` signatures and docstrings. Update any internal callers that pass it.

**Callers to update:**

- `trainer.py` `_generate_samples()` passes `show_progress=False` to `sample_with_intermediates()` (line ~1046 and ~1054) — remove this kwarg.

**Tests to add:**

- `test_sample_rejects_show_progress_kwarg`: Call `sampler.sample(num_samples=1, show_progress=True)` and assert `TypeError` for unexpected keyword argument. This confirms the dead parameter was actually removed.

---

### Phase 5: Remove dead `samples_per_class` parameter (Bug 5)

**Problem:** `DiffusionTrainer.__init__` (trainer.py line ~116) accepts `samples_per_class: int = 2` and stores it as `self.samples_per_class` (line ~162). However, `_generate_samples()` (line ~1039) computes its own local value:

```python
samples_per_class = max(1, self.num_samples // num_classes)  # ignores self.samples_per_class
```

The `main.py` caller (line ~860) passes `samples_per_class=2` with the comment `"Will calculate based on num_samples internally"`. This confirms the parameter is intentionally dead code — the constructor accepts it, stores it, but nothing reads it.

**Fix:**

1. Remove `samples_per_class` from the constructor parameter list, docstrings (both class-level and `__init__`-level), and `self.samples_per_class` assignment.
2. Remove `samples_per_class=2` from the `DiffusionTrainer(...)` call in `main.py`.
3. Update any tests that pass `samples_per_class` to `DiffusionTrainer`.

**Tests to update:**

- Verify no existing tests pass `samples_per_class` to `DiffusionTrainer` — if they do, remove that kwarg.

---

### Phase 6: Remove unused import (Bug 6)

**Problem:** `dataloader.py` (line ~16) imports `get_normalization_transform` from `src.data.transforms`:

```python
from src.data.transforms import get_normalization_transform
```

This function is never used in the module. Normalization is hardcoded inline via `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` in both `_get_train_transform()` and `_get_val_transform()`.

**Fix:** Remove the unused import line.

No tests needed — this is an import cleanup only.

---

### Phase 7: Run all tests and verify no regressions

Run the full test suite to confirm:

1. All existing tests pass (no regressions).
2. All new tests pass.
3. No unexpected failures in other modules.

```bash
python -m pytest tests/ -v
```
