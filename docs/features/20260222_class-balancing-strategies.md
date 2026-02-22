# Class Balancing Strategies for Imbalanced Datasets

## Overview

Implement four class balancing strategies for handling imbalanced datasets during diffusion model training. These strategies address the common problem where one class (e.g., Abnormal) has significantly fewer samples than another (e.g., Normal), which can bias the model toward the majority class.

### Objective

Add configurable class balancing to the `data.balancing` section of the diffusion YAML config, with four strategies:

1. **Weighted Sampling** (`WeightedRandomSampler`) — adjusts sampling probability at the DataLoader level
2. **Downsampling** (Undersampling) — reduces majority class to match minority class
3. **Upsampling** (Oversampling) — duplicates minority class samples to match majority class
4. **Class Weights in Loss** — applies per-class weights to the loss function

### Design Decisions

- **No per-strategy seed parameters.** All randomness derives deterministically from `compute.seed` using **local `torch.Generator`** instances so that balancing logic never corrupts the global random state.
- **Mutual exclusivity for data strategies.** Only ONE of weighted_sampler / downsampling / upsampling should be active. Priority if multiple enabled: `weighted_sampler > downsampling > upsampling`.
- **Class weights are independent.** `class_weights` can be combined with any data strategy.
- **Three weight computation methods:** `inverse_frequency`, `effective_num`, and `manual` — shared between `weighted_sampler` and `class_weights`.
- **Balancing applies to training only.** Validation data is never resampled or reweighted.

### Config YAML Addition

```yaml
data:
  balancing:
    weighted_sampler:
      enabled: false
      method: inverse_frequency # Options: inverse_frequency, effective_num, manual
      beta: 0.999 # Beta for effective_num method
      manual_weights: null # e.g., [1.0, 5.0] (required if method=manual)
      replacement: true # Sample with replacement
      num_samples: null # Samples per epoch (null = len(dataset))

    downsampling:
      enabled: false
      target_ratio: 1.0 # minority:majority ratio (1.0 = equal)

    upsampling:
      enabled: false
      target_ratio: 1.0 # minority:majority ratio (1.0 = equal)

    class_weights:
      enabled: false
      method: inverse_frequency # Options: inverse_frequency, effective_num, manual
      beta: 0.999
      manual_weights: null # e.g., [1.0, 5.0] (required if method=manual)
      normalize: true # Normalize weights to sum to num_classes
```

### Architecture Changes

| File                                      | Change                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| `configs/diffusion.yaml`                  | Add `data.balancing` section                                                               |
| `src/experiments/diffusion/default.yaml`  | Add `data.balancing` defaults                                                              |
| `src/data/samplers.py`                    | Add `compute_effective_num_weights()`, enhance `compute_class_weights()` with `beta` param |
| `src/data/balancing.py` (NEW)             | New module for downsampling/upsampling dataset wrappers                                    |
| `src/experiments/diffusion/dataloader.py` | Accept balancing config, apply strategy in `get_train_loader()`                            |
| `src/experiments/diffusion/config.py`     | Add `_validate_balancing_config()`                                                         |
| `src/main.py`                             | Pass `balancing_config` and `seed` to `DiffusionDataLoader`                                |
| `src/experiments/diffusion/trainer.py`    | Accept optional `class_weights` tensor for weighted loss                                   |
| `src/experiments/diffusion/model.py`      | Extend `compute_loss()` to accept `sample_weights`                                         |

### Estimated Effort

| Phase                                  | Est. Time       |
| -------------------------------------- | --------------- |
| Phase 1: Config & validation           | 1 hour          |
| Phase 2: Sampler weight computation    | 1 hour          |
| Phase 3: Dataset balancing (down/up)   | 1.5 hours       |
| Phase 4: Weighted sampling integration | 1 hour          |
| Phase 5: Class weights in loss         | 1.5 hours       |
| Phase 6: Main.py integration           | 1 hour          |
| Phase 7: Tests                         | 3 hours         |
| Phase 8: Documentation update          | 0.5 hours       |
| **Total**                              | **~10.5 hours** |

## Implementation Checklist

- [ ] Phase 1: Configuration & Validation
  - [ ] Task 1.1: Add `data.balancing` section to `configs/diffusion.yaml`
  - [ ] Task 1.2: Add `data.balancing` defaults to `src/experiments/diffusion/default.yaml`
  - [ ] Task 1.3: Add `_validate_balancing_config()` in `src/experiments/diffusion/config.py`
  - [ ] Task 1.4: Add balancing validation call in `_validate_data_config()`
- [ ] Phase 2: Weight Computation Utilities
  - [ ] Task 2.1: Add `compute_effective_num_weights()` to `src/data/samplers.py`
  - [ ] Task 2.2: Enhance `compute_class_weights()` to accept `beta` parameter for `effective_num` mode
  - [ ] Task 2.3: Add `compute_weights_from_config()` helper that dispatches on `method` (inverse_frequency / effective_num / manual)
- [ ] Phase 3: Dataset Balancing Module
  - [ ] Task 3.1: Create `src/data/balancing.py` with `downsample_dataset()` function
  - [ ] Task 3.2: Add `upsample_dataset()` function to `src/data/balancing.py`
  - [ ] Task 3.3: Both functions use local `torch.Generator` seeded from `compute.seed` for reproducibility
  - [ ] Task 3.4: Both functions operate on `SplitFileDataset` by returning a `torch.utils.data.Subset` or new index list
- [ ] Phase 4: Weighted Sampling Integration
  - [ ] Task 4.1: Add `balancing_config` and `seed` parameters to `DiffusionDataLoader.__init__()`
  - [ ] Task 4.2: In `get_train_loader()`, create `WeightedRandomSampler` when `weighted_sampler.enabled=True`
  - [ ] Task 4.3: When weighted_sampler is active, set `shuffle=False` (sampler and shuffle are mutually exclusive)
  - [ ] Task 4.4: Apply downsampling/upsampling to dataset before creating DataLoader when those strategies are enabled
  - [ ] Task 4.5: Log which balancing strategy is active and class distribution before/after
- [ ] Phase 5: Class Weights in Loss Function
  - [ ] Task 5.1: Add `class_weights` parameter to `DiffusionTrainer.__init__()`
  - [ ] Task 5.2: Create `ClassWeightedMSELoss` in `src/experiments/diffusion/trainer.py` (or a utility module)
  - [ ] Task 5.3: Replace `nn.MSELoss()` with weighted variant when `class_weights` config is enabled
  - [ ] Task 5.4: `ClassWeightedMSELoss` computes per-sample MSE, multiplies by class weight, then averages
  - [ ] Task 5.5: Pass labels to weighted loss in `compute_loss()` (extend model interface to return per-sample loss)
- [ ] Phase 6: Main.py Integration
  - [ ] Task 6.1: Read `data.balancing` config and pass to `DiffusionDataLoader`
  - [ ] Task 6.2: Compute class weight tensor and pass to `DiffusionTrainer` when `class_weights.enabled=True`
  - [ ] Task 6.3: Pass `compute.seed` to `DiffusionDataLoader` for local generator seeding
  - [ ] Task 6.4: Log balancing configuration summary at startup
- [ ] Phase 7: Tests
  - [ ] Task 7.1: Add config validation tests in `tests/experiments/diffusion/test_config.py`
  - [ ] Task 7.2: Add weight computation tests in `tests/data/test_samplers.py`
  - [ ] Task 7.3: Create `tests/data/test_balancing.py` for downsample/upsample functions
  - [ ] Task 7.4: Add dataloader integration tests in `tests/experiments/diffusion/test_dataloader.py`
  - [ ] Task 7.5: Add weighted loss tests in `tests/experiments/diffusion/test_trainer.py`
  - [ ] Task 7.6: Verify all existing tests still pass
- [ ] Phase 8: Documentation
  - [ ] Task 8.1: Update `docs/standards/architecture.md` to document `src/data/balancing.py`
  - [ ] Task 8.2: Update `README.md` with class balancing usage examples

## Phase Details

### Phase 1: Configuration & Validation

Add the `data.balancing` section to both config files and validate it.

**Task 1.1: `configs/diffusion.yaml`**

Add the `balancing` subsection under `data`, between `loading` and `augmentation`:

```yaml
data:
  split_file: "outputs/splits/train_val_split_seed0.json"

  loading:
    # ...existing...

  # Class Balancing: Strategies for handling imbalanced datasets
  # Only ONE data strategy should be enabled at a time.
  # Priority (if multiple enabled): weighted_sampler > downsampling > upsampling
  # class_weights is independent and can be combined with any data strategy.
  balancing:
    weighted_sampler:
      enabled: false
      method: inverse_frequency
      beta: 0.999
      manual_weights: null
      replacement: true
      num_samples: null

    downsampling:
      enabled: false
      target_ratio: 1.0

    upsampling:
      enabled: false
      target_ratio: 1.0

    class_weights:
      enabled: false
      method: inverse_frequency
      beta: 0.999
      manual_weights: null
      normalize: true

  augmentation:
    # ...existing...
```

**Task 1.2: `src/experiments/diffusion/default.yaml`**

Same structure with the same default values.

**Task 1.3–1.4: Validation in `config.py`**

Add `_validate_balancing_config()`:

- If `data.balancing` is missing, it's optional (backwards-compatible) — skip validation.
- Validate: `method` must be one of `inverse_frequency`, `effective_num`, `manual`.
- If `method=manual`, `manual_weights` must be a non-empty list of positive floats.
- `target_ratio` must be a positive float (0 < ratio <= 1.0).
- `beta` must be between 0 and 1 (exclusive).
- `num_samples` must be null or a positive integer.
- `replacement` must be a boolean.
- `normalize` must be a boolean.
- Warn (log warning) if multiple data strategies are enabled simultaneously.

### Phase 2: Weight Computation Utilities

Enhance `src/data/samplers.py` to support configurable weight methods.

**Task 2.1: `compute_effective_num_weights()`**

```python
def compute_effective_num_weights(
    targets: List[int],
    beta: float = 0.9999,
) -> Dict[int, float]:
    """Compute class weights using the effective number of samples method.

    Reference: 'Class-Balanced Loss Based on Effective Number of Samples' (CVPR 2019)

    Effective number: E_n = (1 - beta^n) / (1 - beta)
    Weight: w = 1 / E_n
    """
```

**Task 2.2: Enhance `compute_class_weights()`**

Add `beta` parameter to the existing function for the `effective_num` mode instead of hardcoding `0.9999`.

**Task 2.3: `compute_weights_from_config()`**

A dispatcher function:

```python
def compute_weights_from_config(
    targets: List[int],
    method: str,
    beta: float = 0.999,
    manual_weights: Optional[List[float]] = None,
    normalize: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[int, float]:
    """Compute class weights based on config parameters.

    Args:
        targets: List of class indices
        method: 'inverse_frequency', 'effective_num', or 'manual'
        beta: Beta for effective_num
        manual_weights: Per-class weights for manual mode
        normalize: Normalize weights to sum to num_classes
        num_classes: Number of classes (for normalization)

    Returns:
        Dict mapping class index to weight
    """
```

### Phase 3: Dataset Balancing Module

Create `src/data/balancing.py` with functions to modify dataset indices.

**Task 3.1: `downsample_dataset()`**

```python
def downsample_dataset(
    dataset: BaseDataset,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> Subset:
    """Downsample majority class to achieve target minority:majority ratio.

    Uses a local torch.Generator seeded from the provided seed to avoid
    corrupting the global random state.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of minority:majority (1.0 = equal counts)
        seed: Seed for local random generator (typically from compute.seed)

    Returns:
        torch.utils.data.Subset with balanced indices
    """
```

Logic:

1. Count samples per class, identify minority/majority.
2. Compute target majority count = `minority_count / target_ratio`.
3. Use local `torch.Generator().manual_seed(seed)` to randomly select `target_majority_count` indices from the majority class.
4. Return `Subset(dataset, minority_indices + sampled_majority_indices)`.

**Task 3.2: `upsample_dataset()`**

```python
def upsample_dataset(
    dataset: BaseDataset,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> Subset:
    """Upsample minority class by duplication to achieve target ratio.

    Uses a local torch.Generator for reproducible sampling.

    Args:
        dataset: Dataset with `targets` attribute
        target_ratio: Desired ratio of minority:majority (1.0 = equal counts)
        seed: Seed for local random generator

    Returns:
        torch.utils.data.Subset with duplicated minority indices
    """
```

Logic:

1. Count samples per class, identify minority/majority.
2. Compute target minority count = `majority_count * target_ratio`.
3. Need `extra = target_minority_count - current_minority_count` additional samples.
4. Use local generator to randomly pick `extra` indices (with replacement) from minority class.
5. Return `Subset(dataset, all_original_indices + extra_minority_indices)`.

### Phase 4: Weighted Sampling Integration

Modify `DiffusionDataLoader` to accept and apply balancing config.

**Task 4.1: New `__init__` parameters**

```python
def __init__(
    self,
    # ...existing params...
    balancing_config: Optional[Dict[str, Any]] = None,  # NEW
    seed: int = 0,  # NEW: from compute.seed
):
```

**Task 4.2–4.4: `get_train_loader()` changes**

```python
def get_train_loader(self) -> DataLoader:
    transform = self._get_train_transform()
    train_dataset = SplitFileDataset(...)

    sampler = None
    shuffle = self.shuffle_train
    dataset_to_use = train_dataset

    if self.balancing_config:
        # Priority: weighted_sampler > downsampling > upsampling
        ws_config = self.balancing_config.get("weighted_sampler", {})
        ds_config = self.balancing_config.get("downsampling", {})
        us_config = self.balancing_config.get("upsampling", {})

        if ws_config.get("enabled"):
            weights = compute_weights_from_config(
                targets=train_dataset.targets,
                method=ws_config["method"],
                beta=ws_config.get("beta", 0.999),
                manual_weights=ws_config.get("manual_weights"),
            )
            sampler = create_weighted_sampler(
                targets=train_dataset.targets,
                class_weights=weights,
                replacement=ws_config.get("replacement", True),
                num_samples=ws_config.get("num_samples"),
            )
            shuffle = False  # sampler and shuffle are mutually exclusive

        elif ds_config.get("enabled"):
            dataset_to_use = downsample_dataset(
                train_dataset,
                target_ratio=ds_config.get("target_ratio", 1.0),
                seed=self.seed,
            )

        elif us_config.get("enabled"):
            dataset_to_use = upsample_dataset(
                train_dataset,
                target_ratio=us_config.get("target_ratio", 1.0),
                seed=self.seed,
            )

    return DataLoader(
        dataset_to_use,
        batch_size=self.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        drop_last=self.drop_last,
    )
```

**Task 4.5: Logging**

Log which strategy is active and the class distribution before/after using Python `logging`.

### Phase 5: Class Weights in Loss Function

This is the most nuanced phase because diffusion models use MSE loss on **noise prediction**, not classification loss. The class weight must be applied **per-sample** based on the class label of the image.

**Task 5.1–5.2: `ClassWeightedMSELoss`**

```python
class ClassWeightedMSELoss(nn.Module):
    """MSE loss weighted by class.

    For each sample in the batch, compute MSE loss and multiply by
    the weight of that sample's class. This makes the model pay
    more attention to noise prediction for minority class images.

    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        predicted: torch.Tensor,  # (B, C, H, W)
        target: torch.Tensor,     # (B, C, H, W)
        class_labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # Per-sample MSE: mean over C, H, W dimensions
        per_sample_mse = F.mse_loss(predicted, target, reduction='none')
        per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))  # (B,)

        # Look up weights
        weights = self.class_weights[class_labels]  # (B,)

        # Weighted average
        loss = (per_sample_mse * weights).mean()
        return loss
```

**Task 5.3–5.5: Integration**

- In `DiffusionTrainer.__init__()`, accept optional `class_weights: Optional[torch.Tensor] = None`.
- If provided, use `ClassWeightedMSELoss(class_weights)` instead of `nn.MSELoss()`.
- Modify `model.compute_loss()` to return `(predicted_noise, noise)` so the trainer can compute weighted loss with labels. Alternatively, pass labels into the criterion.
- **Preferred approach:** The trainer computes the loss directly rather than delegating to `model.compute_loss()` when class weights are active:

```python
# In train_epoch:
if self.class_weights is not None and labels is not None:
    predicted_noise, noise = self.model(images, class_labels=labels)
    loss = self.weighted_criterion(predicted_noise, noise, labels)
else:
    loss = self.model.compute_loss(images, class_labels=labels, criterion=self.criterion)
```

This approach is **minimally invasive** — `model.compute_loss()` remains unchanged for the default case, and the weighted path uses the model's forward pass directly.

### Phase 6: Main.py Integration

**Task 6.1–6.3: Pass configs to components**

```python
# In main.py, when creating DiffusionDataLoader:
balancing_config = data_config.get("balancing")
seed = config["compute"].get("seed", 0)

dataloader = DiffusionDataLoader(
    # ...existing params...
    balancing_config=balancing_config,
    seed=seed,
)

# When creating DiffusionTrainer:
class_weight_tensor = None
if balancing_config and balancing_config.get("class_weights", {}).get("enabled"):
    cw_config = balancing_config["class_weights"]
    # Need to get targets from dataset to compute weights
    temp_dataset = SplitFileDataset(split_file=data_config["split_file"], split="train")
    weights_dict = compute_weights_from_config(
        targets=temp_dataset.targets,
        method=cw_config["method"],
        beta=cw_config.get("beta", 0.999),
        manual_weights=cw_config.get("manual_weights"),
        normalize=cw_config.get("normalize", True),
        num_classes=config["model"]["conditioning"].get("num_classes"),
    )
    class_weight_tensor = torch.zeros(len(weights_dict))
    for cls_idx, weight in weights_dict.items():
        class_weight_tensor[cls_idx] = weight

trainer = DiffusionTrainer(
    # ...existing params...
    class_weights=class_weight_tensor,
)
```

**Task 6.4: Log summary**

```python
if balancing_config:
    active = []
    if balancing_config.get("weighted_sampler", {}).get("enabled"):
        active.append("weighted_sampler")
    if balancing_config.get("downsampling", {}).get("enabled"):
        active.append("downsampling")
    if balancing_config.get("upsampling", {}).get("enabled"):
        active.append("upsampling")
    if balancing_config.get("class_weights", {}).get("enabled"):
        active.append("class_weights")
    logger.info(f"Active balancing strategies: {active if active else 'none'}")
```

### Phase 7: Tests

**Task 7.1: Config validation tests**

In `tests/experiments/diffusion/test_config.py`:

- Test valid balancing config passes validation
- Test invalid `method` raises `ValueError`
- Test manual method without `manual_weights` raises error
- Test multiple strategies enabled logs warning
- Test backwards compatibility (missing `balancing` key is OK)

**Task 7.2: Weight computation tests**

In `tests/data/test_samplers.py`:

- Test `compute_effective_num_weights()` returns expected values
- Test `compute_weights_from_config()` with each method
- Test `manual` method with provided weights
- Test `normalize=True` normalizes correctly

**Task 7.3: Balancing module tests**

Create `tests/data/test_balancing.py`:

- Test `downsample_dataset()` produces correct class distribution
- Test `upsample_dataset()` produces correct class distribution
- Test reproducibility (same seed → same indices)
- Test different `target_ratio` values
- Test that local generator doesn't affect global random state

**Task 7.4: Dataloader integration tests**

In `tests/experiments/diffusion/test_dataloader.py`:

- Test `DiffusionDataLoader` with weighted_sampler enabled
- Test that `shuffle=False` when sampler is active
- Test with downsampling/upsampling
- Test priority logic (weighted_sampler overrides others)

**Task 7.5: Weighted loss tests**

In `tests/experiments/diffusion/test_trainer.py`:

- Test `ClassWeightedMSELoss` produces correct weighted loss
- Test that equal weights produce same result as `nn.MSELoss`
- Test that higher weight increases loss contribution

**Task 7.6: Run all tests**

```bash
pytest tests/ -v
```

### Phase 8: Documentation

**Task 8.1:** Add `src/data/balancing.py` to the directory structure in `docs/standards/architecture.md` under `src/data/`.

**Task 8.2:** Add a "Class Balancing" section to `README.md` with example config snippets.
