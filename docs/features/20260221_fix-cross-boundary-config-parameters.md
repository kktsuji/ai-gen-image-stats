# Fix Cross-Boundary Config Parameters

## Overview

Three configuration parameters in the diffusion experiment have wiring issues where values are hardcoded in code instead of being read from config, or are missing entirely from config sections where they are needed.

**Problems:**

1. **Hardcoded EMA decay (#1):** In generation mode, `EMA(model, decay=0.9999, ...)` is hardcoded in `src/main.py` (line 573) instead of reading from config. While `load_state_dict` overwrites the decay afterward, the initial construction should use a configured value for correctness and clarity.
2. **Ignored `num_samples` (#4):** `training.visualization.num_samples` exists in the YAML config but is never read. `main.py` hardcodes `samples_per_class=2` when creating the trainer. In the trainer's `_generate_samples`, unconditional mode also hardcodes `num_samples=8` instead of using the configured value.
3. **No generation batch size (#5):** Generation mode produces all samples in a single batch, which can OOM for large `num_samples`. A `batch_size` parameter should be added to `generation.sampling` to allow batched generation.

**Design decisions:**

- For #1: Add `ema_decay` to `generation.sampling` in the YAML config. Read it in `main.py` generation mode. This intentionally duplicates `training.ema.decay` as accepted by the user.
- For #4: Wire `training.visualization.num_samples` through to the trainer constructor. Derive `samples_per_class` from `num_samples // num_classes` for conditional, use `num_samples` directly for unconditional.
- For #5: Add `batch_size` to `generation.sampling` in the YAML config. Implement batched generation loop in `main.py` generation mode. This intentionally duplicates `data.loading.batch_size` as accepted by the user.

**Files changed:**

| File                                           | Changes                                                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `src/experiments/diffusion/default.yaml`       | Add `ema_decay` and `batch_size` to `generation.sampling`                                                    |
| `src/experiments/diffusion/config.py`          | Add validation for new `generation.sampling.ema_decay` and `generation.sampling.batch_size`                  |
| `src/main.py`                                  | Read `ema_decay` from config in generation mode; pass `num_samples` to trainer; implement batched generation |
| `src/experiments/diffusion/trainer.py`         | Accept `num_samples` parameter; use it in `_generate_samples` for both conditional and unconditional         |
| `tests/experiments/diffusion/test_config.py`   | Add tests for new config fields; update existing generation config tests                                     |
| `tests/experiments/diffusion/test_trainer.py`  | Update tests for `num_samples` parameter wiring                                                              |
| `tests/test_main.py`                           | Update generation mode tests for `ema_decay` from config and batched generation                              |
| `tests/integration/test_diffusion_pipeline.py` | Update integration tests for new parameters                                                                  |

**Time estimate:** ~2 hours

## Implementation Checklist

- [ ] Phase 1: Config changes (YAML + validation)
  - [ ] Task 1.1: Add `ema_decay` to `generation.sampling` in `default.yaml`
  - [ ] Task 1.2: Add `batch_size` to `generation.sampling` in `default.yaml`
  - [ ] Task 1.3: Add validation for `generation.sampling.ema_decay` in `config.py`
  - [ ] Task 1.4: Add validation for `generation.sampling.batch_size` in `config.py`
- [ ] Phase 2: Fix hardcoded EMA decay in generation mode (#1)
  - [ ] Task 2.1: Read `generation.sampling.ema_decay` in `main.py` generation mode and pass to `EMA()` constructor
- [ ] Phase 3: Wire `training.visualization.num_samples` (#4)
  - [ ] Task 3.1: Replace hardcoded `samples_per_class=2` in `main.py` with `num_samples` from `visualization_config`
  - [ ] Task 3.2: Add `num_samples` parameter to `DiffusionTrainer.__init__` (replacing `samples_per_class`)
  - [ ] Task 3.3: Update `_generate_samples` to derive `samples_per_class` from `self.num_samples` and `num_classes` for conditional, use `self.num_samples` directly for unconditional
- [ ] Phase 4: Add batched generation (#5)
  - [ ] Task 4.1: Implement batched generation loop in `main.py` generation mode using `generation.sampling.batch_size`
- [ ] Phase 5: Tests
  - [ ] Task 5.1: Add config validation tests for `generation.sampling.ema_decay`
  - [ ] Task 5.2: Add config validation tests for `generation.sampling.batch_size`
  - [ ] Task 5.3: Update trainer tests for `num_samples` parameter
  - [ ] Task 5.4: Update `main.py` generation mode tests for `ema_decay` from config
  - [ ] Task 5.5: Add/update tests for batched generation
  - [ ] Task 5.6: Run full test suite to confirm no regressions
- [ ] Phase 6: Documentation
  - [ ] Task 6.1: Update YAML comments in `default.yaml` if needed for clarity

## Phase Details

### Phase 1: Config changes (YAML + validation)

**Task 1.1 & 1.2:** Update `src/experiments/diffusion/default.yaml` `generation.sampling` section:

```yaml
generation:
  checkpoint: null

  sampling:
    num_samples: 100
    batch_size: 50 # NEW: Max samples per forward pass (prevents OOM)
    guidance_scale: 3.0
    use_ema: true
    ema_decay: 0.9999 # NEW: EMA decay rate (should match training.ema.decay)
```

**Task 1.3:** Add validation in `_validate_generation_config()` in `config.py`:

- `ema_decay` must be a float between 0 and 1 (exclusive) when present
- Only validated when `use_ema` is `true`

**Task 1.4:** Add validation in `_validate_generation_config()`:

- `batch_size` must be a positive integer when present

### Phase 2: Fix hardcoded EMA decay in generation mode

**Task 2.1:** In `src/main.py` `setup_experiment_diffusion()`, change:

```python
# Before (hardcoded):
ema = EMA(model, decay=0.9999, device=device)

# After (from config):
ema_decay = sampling_config.get("ema_decay", 0.9999)
ema = EMA(model, decay=ema_decay, device=device)
```

### Phase 3: Wire `training.visualization.num_samples`

**Task 3.1:** In `src/main.py`, replace:

```python
# Before (hardcoded):
samples_per_class=2,  # Will calculate based on num_samples internally

# After (from config):
num_samples=visualization_config["num_samples"],
```

**Task 3.2:** In `DiffusionTrainer.__init__`:

- Replace `samples_per_class: int = 2` parameter with `num_samples: int = 8`
- Store as `self.num_samples`
- Remove `self.samples_per_class`

**Task 3.3:** In `DiffusionTrainer._generate_samples`:

```python
# For conditional: derive samples_per_class from num_samples
if num_classes is not None and num_classes > 0:
    samples_per_class = max(1, self.num_samples // num_classes)
    class_labels = torch.arange(num_classes, device=self.device).repeat_interleave(samples_per_class)
    # total = num_classes * samples_per_class
    ...
else:
    # For unconditional: use num_samples directly
    samples, denoising_seq = self.sampler.sample_with_intermediates(
        num_samples=self.num_samples,
        ...
    )
```

### Phase 4: Add batched generation

**Task 4.1:** In `src/main.py` generation mode, replace the single `sampler.sample()` call with a batched loop:

```python
batch_size = sampling_config.get("batch_size", num_samples)
all_samples = []

for start_idx in range(0, num_samples, batch_size):
    end_idx = min(start_idx + batch_size, num_samples)
    batch_labels = class_labels[start_idx:end_idx] if class_labels is not None else None

    batch_samples = sampler.sample(
        num_samples=end_idx - start_idx,
        class_labels=batch_labels,
        guidance_scale=sampling_config["guidance_scale"],
        use_ema=sampling_config["use_ema"],
        show_progress=True,
    )
    all_samples.append(batch_samples)

samples = torch.cat(all_samples, dim=0)
```

### Phase 5: Tests

**Task 5.1:** In `tests/experiments/diffusion/test_config.py`:

- `test_generation_ema_decay_valid`: Verify `ema_decay: 0.9999` passes validation
- `test_generation_ema_decay_invalid`: Verify `ema_decay: 1.5` fails validation
- `test_generation_ema_decay_default`: Verify default config has `ema_decay` in generation

**Task 5.2:** In `tests/experiments/diffusion/test_config.py`:

- `test_generation_batch_size_valid`: Verify `batch_size: 50` passes validation
- `test_generation_batch_size_invalid`: Verify `batch_size: -1` fails validation
- `test_generation_batch_size_default`: Verify default config has `batch_size` in generation

**Task 5.3:** In `tests/experiments/diffusion/test_trainer.py`:

- Update `test_diffusion_trainer_initialization` to verify `num_samples` parameter
- Update `test_diffusion_trainer_sample_generation_during_training` to verify correct number of samples generated
- Update `test_diffusion_trainer_unconditional_sample_generation` to verify `num_samples` used correctly

**Task 5.4:** In `tests/test_main.py`:

- Update `test_generation_mode_loads_ema_when_available` to verify `ema_decay` is read from config
- Add test for custom `ema_decay` value being passed through

**Task 5.5:** In `tests/test_main.py` or `tests/integration/test_diffusion_pipeline.py`:

- Add test for batched generation producing same total number of samples
- Add test for `batch_size` larger than `num_samples` (single batch, no change)

**Task 5.6:** Run `pytest` to confirm no regressions across all test files.

### Phase 6: Documentation

**Task 6.1:** Ensure `default.yaml` comments clearly indicate:

- `generation.sampling.ema_decay` should match `training.ema.decay` for the checkpoint being used
- `generation.sampling.batch_size` prevents OOM for large generation runs
