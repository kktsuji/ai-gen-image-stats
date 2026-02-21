# Fix Diffusion Module Bugs (v2)

## Overview

This plan addresses three **new** bugs discovered during a deeper investigation of `src/experiments/diffusion/`, beyond the four already fixed in `20260221_fix-diffusion-module-bugs.md`.

| #   | Severity   | Type             | File         | Summary                                                                                            |
| --- | ---------- | ---------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| 1   | **MEDIUM** | Visualization    | `logger.py`  | `log_denoising_process` clips [-1, 1] images to [0, 1] instead of remapping, producing black plots |
| 2   | **MEDIUM** | State corruption | `trainer.py` | EMA shadow stays stale when loading a checkpoint that lacks `ema_state_dict`                       |
| 3   | **LOW**    | Latent crash     | `model.py`   | `UNet.conv_out` assumes `channel_multipliers[0] == 1` but config validation doesn't enforce it     |

**Objective:** Fix all three bugs, add new tests to cover each fix, update any existing tests encoding incorrect behavior, and verify no regressions.

**Files changed:**

- `src/experiments/diffusion/logger.py` (bug 1)
- `src/experiments/diffusion/trainer.py` (bug 2)
- `src/experiments/diffusion/config.py` (bug 3)
- `tests/experiments/diffusion/test_logger.py` (new tests for bug 1)
- `tests/experiments/diffusion/test_trainer.py` (new tests for bug 2)
- `tests/experiments/diffusion/test_config.py` (new tests for bug 3)

**Time estimate:** ~1 hour

## Implementation Checklist

- [ ] Phase 1: Fix denoising visualization normalization (Bug 1)
  - [ ] Task 1.1: Remap [-1, 1] → [0, 1] in matplotlib rendering path
  - [ ] Task 1.2: Normalize `selected_images` before passing to TensorBoard
  - [ ] Task 1.3: Add test that pixel values in [-1, 1] range are correctly remapped
  - [ ] Task 1.4: Add test that pure [-1, 0) values are not clamped to black
- [ ] Phase 2: Fix EMA shadow stale after checkpoint load (Bug 2)
  - [ ] Task 2.1: Re-initialize EMA shadow from loaded model weights when `ema_state_dict` is absent
  - [ ] Task 2.2: Add warning log when EMA shadow re-initialized from model weights
  - [ ] Task 2.3: Add test that loading a checkpoint without `ema_state_dict` re-initializes shadow
  - [ ] Task 2.4: Add test that loading a checkpoint with `ema_state_dict` uses saved EMA state
- [ ] Phase 3: Add config validation for `channel_multipliers[0] == 1` (Bug 3)
  - [ ] Task 3.1: Add validation rule in `_validate_model_config` requiring first multiplier to be 1
  - [ ] Task 3.2: Add test that `channel_multipliers` starting with value != 1 raises `ValueError`
  - [ ] Task 3.3: Add test that `channel_multipliers` starting with 1 passes validation
- [ ] Phase 4: Run all tests and verify no regressions
  - [ ] Task 4.1: Run the full test suite and confirm all tests pass

## Phase Details

### Phase 1: Fix denoising visualization normalization (Bug 1)

**Problem:** `log_denoising_process()` in `logger.py` renders intermediate denoising steps for visual inspection. Diffusion models output images in the **[-1, 1]** range (confirmed by `DDPMModel.p_mean_variance` which clamps `x_start` to `[-1, 1]`). However, the rendering code clips directly to [0, 1]:

```python
# Current (line ~327-330)
img = np.clip(img, 0, 1)
```

This clamps all negative pixel values (roughly half the data, especially in early noisy steps) to 0, producing artificially **dark/black** visualizations. In contrast, the `log_images()` method correctly uses `save_image(..., normalize=True)` from torchvision, which handles the full range.

The TensorBoard path is also affected: raw [-1, 1] tensors are passed to `safe_log_images()` at line ~351, which expects [0, 1] float images.

**Fix (matplotlib path):** Replace `np.clip(img, 0, 1)` with `np.clip((img + 1.0) / 2.0, 0, 1)` in both the RGB and grayscale branches to linearly remap [-1, 1] → [0, 1].

**Fix (TensorBoard path):** Normalize `selected_images` to [0, 1] before passing to `safe_log_images()`:

```python
tb_images = (selected_images + 1.0) / 2.0
tb_images = torch.clamp(tb_images, 0, 1)
safe_log_images(self.tb_writer, "denoising/process", tb_images, step)
```

**Locations:**

| Line | Current                                      | Fix                                                  |
| ---- | -------------------------------------------- | ---------------------------------------------------- |
| ~327 | `img = np.clip(img, 0, 1)`                   | `img = np.clip((img + 1.0) / 2.0, 0, 1)` (RGB)       |
| ~330 | `img = np.clip(img, 0, 1)`                   | `img = np.clip((img + 1.0) / 2.0, 0, 1)` (grayscale) |
| ~351 | `safe_log_images(..., selected_images, ...)` | Normalize before passing                             |

**Tests to add:**

- `test_log_denoising_negative_values_not_clipped_to_black`: Create a denoising sequence with known negative values (e.g., all -1.0). After logging, reload the saved image and verify pixel values are non-zero (i.e., properly remapped, not black).
- `test_log_denoising_full_range_remapping`: Create a known pattern with values spanning [-1, 1] and verify the output covers the full brightness range.

---

### Phase 2: Fix EMA shadow stale after checkpoint load (Bug 2)

**Problem:** When `load_checkpoint()` in `trainer.py` loads a checkpoint that **lacks** `ema_state_dict` (e.g., from a non-EMA training run, or an older checkpoint format), the EMA shadow parameters remain at their **initial random values** from the EMA constructor, while the model weights have been updated to the checkpoint values:

```python
# Current code (line ~775-777)
if self.use_ema and self.ema is not None and "ema_state_dict" in checkpoint:
    self.ema.load_state_dict(checkpoint["ema_state_dict"])
    _logger.debug("  EMA state restored")
# If "ema_state_dict" is absent → shadow stays stale!
```

Any subsequent call to `ema.apply_shadow()` (e.g., during sampling via `DiffusionSampler`) would overwrite the loaded model weights with **garbage** from initialization, producing nonsensical outputs.

**Fix:** Add an `else` branch to re-initialize the EMA shadow from the newly loaded model weights:

```python
if self.use_ema and self.ema is not None:
    if "ema_state_dict" in checkpoint:
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        _logger.debug("  EMA state restored")
    else:
        # Re-initialize shadow from loaded model weights to prevent
        # stale shadow from overwriting the model during sampling
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema.shadow[name] = param.data.clone().to(self.device)
        _logger.warning(
            "No EMA state in checkpoint; shadow re-initialized from model weights"
        )
```

**Tests to add:**

- `test_load_checkpoint_without_ema_reinitializes_shadow`: Create a trainer with `use_ema=True`. Save a checkpoint, remove `ema_state_dict` from the saved dict, reload. Assert that EMA shadow parameters match the model's loaded weights (not the initial random weights).
- `test_load_checkpoint_with_ema_restores_saved_state`: Create a trainer with `use_ema=True`. Train, save checkpoint (includes EMA state), load checkpoint. Assert EMA shadow matches the saved state (existing test coverage exists but this confirms the happy path still works).

---

### Phase 3: Add config validation for `channel_multipliers[0] == 1` (Bug 3)

**Problem:** `UNet.conv_out` is hardcoded to accept `base_channels` input:

```python
self.conv_out = nn.Sequential(
    nn.GroupNorm(8, base_channels),
    nn.SiLU(),
    nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
)
```

After the upsampling path, the actual output channels are `base_channels * channel_multipliers[0]`. This only works when `channel_multipliers[0] == 1`. All current presets in `config.py` and `default.yaml` satisfy this (40: `[1, 2, 4]`, 64: `[1, 2, 2, 2]`, 128: `[1, 1, 2, 2, 4]`, 256: `[1, 1, 2, 2, 4, 4]`), but config validation does **not** enforce it. A user setting `channel_multipliers: [2, 4, 8]` would pass validation but crash with a shape mismatch at runtime.

**Approach:** Add validation (rather than changing the model architecture) because all existing presets already use `1` as the first multiplier, and changing `conv_out` would alter the model architecture and parameter count for all existing configs.

**Fix:** Add a check in `_validate_model_config` in `config.py`:

```python
if arch["channel_multipliers"][0] != 1:
    raise ValueError(
        "model.architecture.channel_multipliers[0] must be 1 "
        "(required by UNet output layer)"
    )
```

**Location:** After the existing validation that all channel_multipliers are positive integers (around line ~153 in `config.py`).

**Tests to add:**

- `test_invalid_channel_multipliers_first_not_one`: Set `channel_multipliers: [2, 4, 8]` and assert `ValueError` with message about first multiplier.
- `test_valid_channel_multipliers_first_is_one`: Set `channel_multipliers: [1, 2, 4]` and assert validation passes (may already be covered by existing tests, but add an explicit one for clarity).

---

### Phase 4: Run all tests and verify no regressions

Run the full test suite to confirm:

1. All existing tests pass (no regressions).
2. All new tests pass.
3. No unexpected failures in other modules.

```bash
python -m pytest tests/ -v
```
