# Fix Diffusion Module Bugs (v3)

## Overview

This plan addresses four **new** bugs discovered during a deeper investigation of `src/experiments/diffusion/`, beyond those already fixed in `20260221_fix-diffusion-module-bugs.md` and `20260221_fix-diffusion-module-bugs-v2.md`.

| #   | Severity   | Type             | File         | Summary                                                                                                               |
| --- | ---------- | ---------------- | ------------ | --------------------------------------------------------------------------------------------------------------------- |
| 1   | **HIGH**   | Runtime error    | `trainer.py` | `ReduceLROnPlateau` scheduler crashes — `step()` called without metrics arg, `get_last_lr()` doesn't exist            |
| 2   | **HIGH**   | Logic error      | `trainer.py` | Best-model checkpoint never saved when validation data exists (`"loss"` key not found in `{"val_loss": ...}`)         |
| 3   | **MEDIUM** | Logic error      | `model.py`   | Dynamic thresholding is always a no-op — `torch.clamp(x_start, -1, 1)` before `dynamic_threshold()` forces `s == 1.0` |
| 4   | **MEDIUM** | State corruption | `trainer.py` | `resume_training()` inherited from `BaseTrainer` skips LR scheduler stepping and visualization                        |

**Objective:** Fix all four bugs, add new tests to cover each fix, update any existing tests encoding incorrect behavior, and verify no regressions.

**Files changed:**

- `src/experiments/diffusion/trainer.py` (bugs 1, 2, 4)
- `src/experiments/diffusion/model.py` (bug 3)
- `tests/experiments/diffusion/test_trainer.py` (new tests for bugs 1, 2, 4)
- `tests/experiments/diffusion/test_model.py` (new/updated tests for bug 3)

**Time estimate:** ~1.5 hours

## Implementation Checklist

- [ ] Phase 1: Fix `ReduceLROnPlateau` scheduler crash (Bug 1)
  - [ ] Task 1.1: Add `ReduceLROnPlateau`-specific handling in `train()` scheduler step
  - [ ] Task 1.2: Add test that training with `ReduceLROnPlateau` runs without error
  - [ ] Task 1.3: Add test that `ReduceLROnPlateau` receives metric value and adjusts LR
- [ ] Phase 2: Fix best-model checkpoint not saved with validation data (Bug 2)
  - [ ] Task 2.1: Add fallback to `train_metrics` when `best_metric` key not found in `val_metrics`
  - [ ] Task 2.2: Add test that best-model checkpoint is saved when using `best_metric="loss"` with validation data
  - [ ] Task 2.3: Add test that best-model checkpoint is saved when `best_metric` key exists in `val_metrics` directly (happy path)
- [ ] Phase 3: Fix dynamic thresholding no-op (Bug 3)
  - [ ] Task 3.1: Make fixed clamp and dynamic thresholding mutually exclusive in `p_mean_variance`
  - [ ] Task 3.2: Update existing `test_dynamic_threshold` to verify values change from unclamped input
  - [ ] Task 3.3: Add test that `p_mean_variance` with `use_dynamic_threshold=True` differs from `use_dynamic_threshold=False` on extreme inputs
- [ ] Phase 4: Fix `resume_training()` missing scheduler and visualization (Bug 4)
  - [ ] Task 4.1: Override `resume_training()` in `DiffusionTrainer` with scheduler stepping and visualization
  - [ ] Task 4.2: Add test that LR scheduler advances during resumed training
  - [ ] Task 4.3: Add test that visualization is generated during resumed training when intervals trigger
- [ ] Phase 5: Run all tests and verify no regressions
  - [ ] Task 5.1: Run the full test suite and confirm all tests pass

## Phase Details

### Phase 1: Fix `ReduceLROnPlateau` scheduler crash (Bug 1)

**Problem:** Config validation in `config.py` (line ~361) and `main.py` (lines 297–302) both accept `scheduler.type: plateau` and create a `ReduceLROnPlateau`. However, `DiffusionTrainer.train()` calls the scheduler like a standard `_LRScheduler`:

```python
# Current code (trainer.py lines ~528-529)
self.scheduler.step()
current_lr = self.scheduler.get_last_lr()[0]
```

Two problems:

1. `ReduceLROnPlateau.step()` **requires** a `metrics` argument — calling it without one raises `TypeError: ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'`.
2. `ReduceLROnPlateau` does **not** inherit from `_LRScheduler` and has no `get_last_lr()` method — this would raise `AttributeError`.

**Fix:** Add type-specific handling for `ReduceLROnPlateau`:

```python
if self.scheduler is not None:
    old_lr = self.optimizer.param_groups[0]["lr"]
    if isinstance(
        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        # ReduceLROnPlateau needs a metric value
        metric_for_plateau = train_metrics.get("loss", 0.0)
        if val_metrics is not None:
            metric_for_plateau = val_metrics.get(
                "val_loss", metric_for_plateau
            )
        self.scheduler.step(metric_for_plateau)
        current_lr = self.optimizer.param_groups[0]["lr"]
    else:
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]

    # ... rest of LR logging unchanged ...
```

**Tests to add:**

- `test_diffusion_trainer_with_plateau_scheduler`: Create a trainer with `ReduceLROnPlateau`. Train for 2 epochs. Assert no `TypeError` or `AttributeError` is raised.
- `test_diffusion_trainer_plateau_scheduler_receives_metric`: Set patience=0 and a sensitive factor. Train for multiple epochs with high loss values. Verify that LR is reduced (demonstrating the metric was successfully passed to `step()`).

---

### Phase 2: Fix best-model checkpoint not saved with validation data (Bug 2)

**Problem:** The YAML default is `training.validation.metric: loss`, which is passed as `best_metric` from `main.py` (line 843). When validation runs, `validate_epoch()` returns `{"val_loss": avg_loss}`. The code does:

```python
# Current code (trainer.py lines ~557-560)
metrics_to_check = val_metrics if val_metrics else train_metrics
current_metric_value = metrics_to_check.get(best_metric)
```

Since `"loss"` is not a key in `val_metrics` (which contains `"val_loss"`), `current_metric_value` is always `None`, and the `best_model.pth` save path is never reached. The bug is dormant with the current default `save_best_only: false`, but activates the moment a user enables it with validation data.

**Fix:** Add a fallback to `train_metrics` when the key is not found in `val_metrics`:

```python
# Determine current metric value for best model tracking
current_metric_value = None
if save_best:
    # Try validation metrics first, then training metrics
    metrics_to_check = val_metrics if val_metrics else train_metrics
    current_metric_value = metrics_to_check.get(best_metric)

    # Fallback: if key not found in validation metrics, try training metrics
    if current_metric_value is None and val_metrics is not None:
        current_metric_value = train_metrics.get(best_metric)

    if current_metric_value is not None:
        ...
```

**Tests to add:**

- `test_best_model_saved_with_validation_data`: Create a trainer, train for 2 epochs with `save_best=True, best_metric="loss"` and validation data enabled. Assert that `best_model.pth` is created.
- `test_best_model_saved_with_matching_val_metric_key`: Train with `best_metric="val_loss"` (matching key). Assert `best_model.pth` is created (happy path confirmation).

---

### Phase 3: Fix dynamic thresholding no-op (Bug 3)

**Problem:** In `p_mean_variance()` (model.py lines ~978-983), `x_start` is hard-clamped to `[-1, 1]` **before** dynamic thresholding runs:

```python
# Current code
x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
x_start = torch.clamp(x_start, -1.0, 1.0)

# Apply dynamic thresholding if enabled
if use_dynamic_threshold:
    x_start = self.dynamic_threshold(x_start, dynamic_threshold_percentile)
```

Inside `dynamic_threshold()`, `s = torch.quantile(abs_flat, percentile, ...)` computes on already-clamped data (all values ≤ 1.0), then `s = torch.clamp(s, min=1.0)` forces `s == 1.0`, and the final `x_start / s` divides by 1 — a no-op. Dynamic thresholding (from the Imagen paper) is designed as an **alternative** to fixed clamping that preserves relative signal magnitudes by using per-sample percentile-based clipping and rescaling.

**Fix:** Make fixed clamping and dynamic thresholding mutually exclusive:

```python
# Predict x_0
x_start = self.predict_start_from_noise(x_t, t, predicted_noise)

# Apply dynamic thresholding or fixed clamp
if use_dynamic_threshold:
    x_start = self.dynamic_threshold(x_start, dynamic_threshold_percentile)
else:
    x_start = torch.clamp(x_start, -1.0, 1.0)
```

**Tests to update/add:**

- **Update** `test_dynamic_threshold`: The existing test already correctly tests `dynamic_threshold()` in isolation with large values. No changes needed.
- **Add** `test_p_mean_variance_dynamic_threshold_has_effect`: Call `p_mean_variance` with extreme noise inputs using `use_dynamic_threshold=True` and `False`. Verify the two results differ, confirming dynamic thresholding actually modifies the output.
- **Add** `test_p_mean_variance_without_dynamic_threshold_clamps`: With `use_dynamic_threshold=False`, verify the result is consistent with `[-1, 1]`-clamped predictions.

---

### Phase 4: Fix `resume_training()` missing scheduler and visualization (Bug 4)

**Problem:** `DiffusionTrainer` overrides `train()` to add:

- Learning rate scheduler stepping (lines 526–548)
- Visualization / sample generation (lines 551–553)
- Hyperparameter logging (lines 502–503)

However, it does **not** override `resume_training()`, which inherits `BaseTrainer`'s version (base/trainer.py lines 530–700). That base implementation has its own independent training loop that:

- Never calls `self.scheduler.step()` → LR freezes at checkpoint value
- Never calls `_generate_samples()` → no visualization during resumed training
- Never logs hyperparameters to TensorBoard

If a user resumes a cosine-scheduled training run, the LR stays constant instead of following the cosine curve, silently degrading results.

**Fix:** Override `resume_training()` in `DiffusionTrainer` to include scheduler stepping and visualization logic, mirroring the enhancements in `train()`. The override should:

1. Call `self.load_checkpoint(checkpoint_path)` to restore state
2. Log hyperparameters to TensorBoard
3. Run the per-epoch loop with:
   - `train_epoch()` and `validate_epoch()`
   - Scheduler stepping (with `ReduceLROnPlateau` support from Bug 1 fix)
   - Visualization via `_generate_samples()` when intervals trigger
   - Best-model tracking (with Bug 2 fallback fix)
   - Checkpoint saving

```python
def resume_training(
    self,
    checkpoint_path: Union[str, Path],
    num_epochs: int,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_frequency: int = 1,
    validate_frequency: int = 1,
    save_best: bool = True,
    best_metric: str = "loss",
    best_metric_mode: str = "min",
    save_latest_checkpoint: bool = True,
) -> None:
    """Resume training from a checkpoint.

    Overrides BaseTrainer.resume_training() to include:
    - Learning rate scheduler stepping
    - Sample generation / visualization
    - Hyperparameter logging
    """
    checkpoint_info = self.load_checkpoint(checkpoint_path)
    start_epoch = checkpoint_info["epoch"]
    _logger.info(f"Resuming training from epoch {start_epoch}")
    _logger.info(f"Will train for {num_epochs} additional epochs")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    self._best_metric_name = best_metric
    logger = self.get_logger()

    # Log hyperparameters to TensorBoard at start of resumed training
    if self.config:
        logger.log_hyperparams(self.config)

    for epoch in range(num_epochs):
        self._current_epoch = start_epoch + epoch + 1

        # Training epoch
        train_metrics = self.train_epoch()

        # Log training metrics
        logger.log_metrics(
            train_metrics, step=self._global_step, epoch=self._current_epoch
        )

        # Validation
        val_metrics = None
        if validate_frequency > 0 and (epoch + 1) % validate_frequency == 0:
            val_metrics = self.validate_epoch()
            if val_metrics is not None:
                logger.log_metrics(
                    val_metrics, step=self._global_step, epoch=self._current_epoch
                )

        # Learning rate scheduler step (same logic as train())
        if self.scheduler is not None:
            old_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_for_plateau = train_metrics.get("loss", 0.0)
                if val_metrics is not None:
                    metric_for_plateau = val_metrics.get(
                        "val_loss", metric_for_plateau
                    )
                self.scheduler.step(metric_for_plateau)
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            if current_lr < 1e-7:
                _logger.warning(
                    f"Very low learning rate: {current_lr:.2e} - "
                    f"training may be ineffective"
                )
            if old_lr != current_lr:
                _logger.info(
                    f"Learning rate changed: {old_lr:.6e} -> {current_lr:.6e} "
                    f"(epoch {self._current_epoch})"
                )
            else:
                _logger.debug(f"Learning rate: {current_lr:.6e}")
            logger.log_metrics(
                {"learning_rate": current_lr},
                step=self._global_step,
                epoch=self._current_epoch,
            )

        # Generate sample images
        if self.viz_enabled and self._should_visualize(self._current_epoch):
            _logger.info(
                f"Generating sample images at epoch {self._current_epoch}"
            )
            self._generate_samples(logger, self._global_step,
                                   epoch=self._current_epoch)

        # Best model tracking (with fallback from Bug 2 fix)
        current_metric_value = None
        if save_best:
            metrics_to_check = val_metrics if val_metrics else train_metrics
            current_metric_value = metrics_to_check.get(best_metric)
            if current_metric_value is None and val_metrics is not None:
                current_metric_value = train_metrics.get(best_metric)

            if current_metric_value is not None:
                is_best = self._is_best_metric(
                    current_metric_value, best_metric_mode
                )
                if is_best:
                    self._best_metric = current_metric_value
                    if checkpoint_dir is not None:
                        best_path = checkpoint_dir / "best_model.pth"
                        self.save_checkpoint(
                            best_path,
                            epoch=self._current_epoch,
                            is_best=True,
                            metrics={
                                **train_metrics,
                                **(val_metrics if val_metrics else {}),
                            },
                        )

        # Regular checkpoint saving
        if checkpoint_dir is not None:
            if (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path_new = (
                    checkpoint_dir
                    / f"checkpoint_epoch_{self._current_epoch}.pth"
                )
                self.save_checkpoint(
                    checkpoint_path_new,
                    epoch=self._current_epoch,
                    is_best=False,
                    metrics={
                        **train_metrics,
                        **(val_metrics if val_metrics else {}),
                    },
                )
            if save_latest_checkpoint:
                latest_path = checkpoint_dir / "latest_checkpoint.pth"
                self.save_checkpoint(
                    latest_path,
                    epoch=self._current_epoch,
                    is_best=False,
                    metrics={
                        **train_metrics,
                        **(val_metrics if val_metrics else {}),
                    },
                )

    # Save final checkpoint
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "final_model.pth"
        self.save_checkpoint(
            final_path,
            epoch=self._current_epoch,
            is_best=False,
            metrics={
                **train_metrics,
                **(val_metrics if val_metrics else {}),
            },
        )
        _logger.info(f"Final model checkpoint saved: {final_path}")
```

**Tests to add:**

- `test_resume_training_scheduler_advances`: Create a trainer with `StepLR` scheduler. Train for 2 epochs, save checkpoint. Resume for 2 more epochs. Verify LR continues to decrease from the checkpoint value (not frozen).
- `test_resume_training_generates_samples`: Create a trainer with `log_images_interval=1`. Train for 1 epoch, save checkpoint. Resume for 2 more epochs. Verify that `logged_images` count increases during resumed training.

---

### Phase 5: Run all tests and verify no regressions

Run the full test suite to confirm:

1. All existing tests pass (no regressions).
2. All new tests pass.
3. No unexpected failures in other modules.

```bash
python -m pytest tests/ -v
```
