# Enable log_sample_comparison and log_denoising_process

## Overview

**Objective:** Enable two currently-unused `DiffusionLogger` methods —
`log_sample_comparison` (writes to `quality/`) and `log_denoising_process`
(writes to `denoising/`) — during diffusion model training.

**Current state:**

- `log_images` is called every `visualization.interval` epochs via `_generate_samples()`
- `log_sample_comparison` and `log_denoising_process` are implemented in `DiffusionLogger`
  but never called

**Design decisions:**

- All three methods share the same sample generation (generate once, reuse)
- `visualization.interval` (single interval) is replaced by three independent intervals,
  each nullable to disable individually
- `visualization.enabled` becomes the master switch: if `false`, no sampling occurs at all
- `log_denoising_process` requires a denoising sequence (T, C, H, W); a new
  `DiffusionSampler.sample_with_intermediates()` method provides this

**New `visualization` config block:**

```yaml
visualization:
  enabled: true # Master switch: disables all three if false
  num_samples: 8 # Samples to generate (shared by all three)
  guidance_scale: 3.0 # Sampling guidance scale (shared by all three)
  log_images_interval: 10 # Save sample grid every N epochs (null = disable)
  log_sample_comparison_interval: 10 # Save quality comparison every N epochs (null = disable)
  log_denoising_interval: 10 # Save denoising process every N epochs (null = disable)
```

**Files changed:**

| File                                           | Change                                                                       |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `src/experiments/diffusion/default.yaml`       | Replace `interval` with three interval keys                                  |
| `src/experiments/diffusion/sampler.py`         | Add `sample_with_intermediates()`                                            |
| `src/experiments/diffusion/trainer.py`         | Replace `sample_images`/`sample_interval` params; update `_generate_samples` |
| `src/main.py`                                  | Read new yaml keys and pass to trainer                                       |
| `tests/experiments/diffusion/test_trainer.py`  | Update/add tests for new params                                              |
| `tests/experiments/diffusion/test_logger.py`   | Add tests for newly-called methods                                           |
| `tests/integration/test_diffusion_pipeline.py` | Integration test coverage                                                    |

**Expected outcome:**

- All three logger methods called during training based on their individual intervals
- Samples generated at most once per epoch (when any interval triggers)
- Backward-compatible: null interval disables that specific method

---

## Implementation Checklist

- [x] Phase 1: Update `default.yaml`
  - [x] Task 1.1: Remove `interval` key
  - [x] Task 1.2: Add `log_images_interval`, `log_sample_comparison_interval`, `log_denoising_interval`

- [x] Phase 2: Add `sample_with_intermediates()` to `DiffusionSampler`
  - [x] Task 2.1: Implement method that returns `(samples, denoising_sequence)` where `denoising_sequence` is shape `(T, C, H, W)`
  - [x] Task 2.2: Add unit tests for `sample_with_intermediates()`

- [x] Phase 3: Update `DiffusionTrainer`
  - [x] Task 3.1: Replace `sample_images: bool` + `sample_interval: int` constructor params with `log_images_interval`, `log_sample_comparison_interval`, `log_denoising_interval` (all `Optional[int]`)
  - [x] Task 3.2: Update `_generate_samples()` to generate samples once and call all three methods conditionally
  - [x] Task 3.3: Update training loop trigger condition in `train()`
  - [x] Task 3.4: Update class docstring and constructor docstring

- [x] Phase 4: Update `main.py`
  - [x] Task 4.1: Read new interval keys from `visualization_config`
  - [x] Task 4.2: Pass new params to `DiffusionTrainer()`
  - [x] Task 4.3: Remove `sample_images` and `sample_interval` arguments

- [x] Phase 5: Update tests
  - [x] Task 5.1: Update `test_trainer.py` — replace `sample_images`/`sample_interval` with new params
  - [x] Task 5.2: Add trainer tests: verify each method is called at the right interval
  - [x] Task 5.3: Add trainer tests: verify samples generated only once when multiple intervals trigger simultaneously
  - [x] Task 5.4: Update `test_logger.py` — add tests for `log_sample_comparison` and `log_denoising_process` being called
  - [x] Task 5.5: Update integration tests if needed

- [x] Phase 6: Validate
  - [x] Task 6.1: Run full test suite (`pytest`)
  - [ ] Task 6.2: Smoke test: run a short training (3–5 epochs) and verify files appear in `quality/` and `denoising/`

---

## Phase Details

### Phase 1: Update `default.yaml`

In `src/experiments/diffusion/default.yaml`, replace the `visualization` block:

```yaml
# Before
visualization:
  enabled: true
  interval: 10
  num_samples: 8
  guidance_scale: 3.0

# After
visualization:
  enabled: true                        # Master switch: if false, no sampling occurs at all
  num_samples: 8                       # Number of samples to generate (shared by all below)
  guidance_scale: 3.0                  # Classifier-free guidance scale (shared by all below)
  log_images_interval: 10              # Save sample grid every N epochs (null to disable)
  log_sample_comparison_interval: 10   # Save quality comparison every N epochs (null to disable)
  log_denoising_interval: 10           # Save denoising process every N epochs (null to disable)
```

---

### Phase 2: Add `sample_with_intermediates()` to `DiffusionSampler`

The existing `sample()` returns only the final samples tensor. `log_denoising_process`
requires the full denoising trajectory of shape `(T, C, H, W)`.

Add to `src/experiments/diffusion/sampler.py`:

```python
def sample_with_intermediates(
    self,
    num_samples: int = 1,
    guidance_scale: float = 0.0,
    use_ema: bool = True,
    num_steps_to_capture: int = 8,
    show_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples and return intermediate denoising steps.

    Returns:
        Tuple of:
        - samples: Final generated images, shape (N, C, H, W)
        - denoising_sequence: Intermediate steps for one sample,
          shape (num_steps_to_capture, C, H, W)
    """
```

This method hooks into the reverse diffusion loop and captures `num_steps_to_capture`
evenly-spaced intermediate frames from a **single sample** (index 0) to keep memory
usage bounded.

---

### Phase 3: Update `DiffusionTrainer`

#### 3.1 Constructor signature

```python
# Before
def __init__(
    self,
    ...
    sample_images: bool = True,
    sample_interval: int = 10,
    samples_per_class: int = 2,
    guidance_scale: float = 3.0,
    ...
):

# After
def __init__(
    self,
    ...
    log_images_interval: Optional[int] = 10,
    log_sample_comparison_interval: Optional[int] = 10,
    log_denoising_interval: Optional[int] = 10,
    samples_per_class: int = 2,
    guidance_scale: float = 3.0,
    ...
):
```

#### 3.2 Training loop trigger

```python
# Before
if self.sample_images and self.sample_interval > 0 and (epoch + 1) % self.sample_interval == 0:
    self._generate_samples(logger, self._global_step)

# After
if self.viz_enabled and self._should_visualize(epoch + 1):
    self._generate_samples(logger, self._global_step, epoch=self._current_epoch)
```

Where `_should_visualize(epoch)` returns `True` if any of the three intervals trigger.

#### 3.3 `_generate_samples()` logic

```python
def _generate_samples(self, logger: BaseLogger, step: int, epoch: int) -> None:
    # 1. Generate samples ONCE
    samples, denoising_seq = self.sampler.sample_with_intermediates(
        num_samples=..., guidance_scale=..., use_ema=...
    )

    # 2. log_images
    if self.log_images_interval and epoch % self.log_images_interval == 0:
        logger.log_images(samples, tag=f"samples_epoch_{epoch}", step=step, epoch=epoch)

    # 3. log_sample_comparison
    if self.log_sample_comparison_interval and epoch % self.log_sample_comparison_interval == 0:
        logger.log_sample_comparison(samples, tag="quality_comparison", step=step, epoch=epoch)

    # 4. log_denoising_process
    if self.log_denoising_interval and epoch % self.log_denoising_interval == 0:
        logger.log_denoising_process(denoising_seq, step=step, epoch=epoch)
```

---

### Phase 4: Update `main.py`

```python
# Before
trainer = DiffusionTrainer(
    ...
    sample_images=visualization_config["enabled"],
    sample_interval=visualization_config["interval"],
    samples_per_class=2,
    guidance_scale=visualization_config["guidance_scale"],
    ...
)

# After
trainer = DiffusionTrainer(
    ...
    log_images_interval=(
        visualization_config.get("log_images_interval")
        if visualization_config["enabled"] else None
    ),
    log_sample_comparison_interval=(
        visualization_config.get("log_sample_comparison_interval")
        if visualization_config["enabled"] else None
    ),
    log_denoising_interval=(
        visualization_config.get("log_denoising_interval")
        if visualization_config["enabled"] else None
    ),
    samples_per_class=2,
    guidance_scale=visualization_config["guidance_scale"],
    ...
)
```

---

### Phase 5: Tests

#### test_trainer.py additions

- `test_log_images_called_at_interval` — mock logger, verify `log_images` called at epoch 10 not epoch 5
- `test_log_sample_comparison_called_at_interval` — same for `log_sample_comparison`
- `test_log_denoising_called_at_interval` — same for `log_denoising_process`
- `test_samples_generated_once_when_all_intervals_trigger` — verify sampler called once when all three intervals align
- `test_visualization_disabled_when_all_intervals_null` — verify no sampling when all are `None`

#### test_logger.py additions

- `test_log_sample_comparison_saves_to_quality_dir`
- `test_log_denoising_process_saves_to_denoising_dir`

---

### Phase 6: Validate

```bash
# Unit + integration tests
pytest tests/experiments/diffusion/ -v

# Full suite
pytest

# Smoke test (short run)
python -m src.main --config configs/smoke_test.yaml
# Verify: outputs/<run>/quality/ and outputs/<run>/denoising/ contain files
```
