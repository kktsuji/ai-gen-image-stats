# Noise Schedule Implementation for DDPM

## Overview

Added parametrized noise schedules to the DDPM implementation, allowing users to choose between different beta scheduling strategies. This provides more control over the diffusion process and can improve training stability and sample quality.

## Available Noise Schedules

### 1. **Linear Schedule** (default)
- **Description**: Linear interpolation between beta_start and beta_end
- **Formula**: β_t = β_start + (β_end - β_start) * (t / T)
- **Use case**: Standard baseline, works well for most cases
- **Parameters**: `beta_start=0.0001`, `beta_end=0.02`

### 2. **Cosine Schedule**
- **Description**: Cosine-based schedule from "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
- **Formula**: Based on cosine function with offset parameter s=0.008
- **Use case**: Better for high-resolution images, more stable training
- **Parameters**: Automatically computed, ignores beta_start/beta_end
- **Paper**: https://arxiv.org/abs/2102.09672

### 3. **Quadratic Schedule**
- **Description**: Quadratic interpolation between beta_start and beta_end
- **Formula**: β_t = (sqrt(β_start) + (sqrt(β_end) - sqrt(β_start)) * (t / T))²
- **Use case**: Smoother transitions, good for complex datasets
- **Parameters**: `beta_start=0.0001`, `beta_end=0.02`

### 4. **Sigmoid Schedule**
- **Description**: Sigmoid-based schedule for smooth transitions
- **Formula**: β_t = sigmoid((t - T/2) / k) * (β_end - β_start) + β_start
- **Use case**: Very smooth transitions, experimental
- **Parameters**: `beta_start=0.0001`, `beta_end=0.02`

## Usage

### Basic Usage

```python
from ddpm import create_ddpm

# Linear schedule (default)
model = create_ddpm(
    image_size=40,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
)

# Cosine schedule
model = create_ddpm(
    image_size=40,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="cosine",
)

# Quadratic schedule
model = create_ddpm(
    image_size=40,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="quadratic",
    beta_start=0.0001,
    beta_end=0.02,
)

# Sigmoid schedule
model = create_ddpm(
    image_size=40,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="sigmoid",
    beta_start=0.0001,
    beta_end=0.02,
)
```

### Training with Different Schedules

Modify the `BETA_SCHEDULE` parameter in `train()`:

```python
# In ddpm.py train() function:
BETA_SCHEDULE = "cosine"  # or "linear", "quadratic", "sigmoid"
BETA_START = 0.0001       # For linear, quadratic, sigmoid
BETA_END = 0.02           # For linear, quadratic, sigmoid
```

### Generation with Matching Schedule

**Important**: When generating samples, use the same schedule that was used during training!

```python
from ddpm import generate

samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=16,
    beta_schedule="cosine",  # MUST match training schedule
    guidance_scale=3.0,
)
```

## API Changes

### DDPM Class

Added `beta_schedule` parameter:

```python
class DDPM(nn.Module):
    def __init__(
        self,
        image_size: int = 40,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        num_classes: Optional[int] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",  # NEW
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        class_dropout_prob: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
```

### create_ddpm() Function

Updated signature:

```python
def create_ddpm(
    image_size: int = 40,
    in_channels: int = 3,
    model_channels: int = 64,
    num_classes: Optional[int] = None,
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",  # NEW
    beta_start: float = 0.0001,     # NEW (explicit)
    beta_end: float = 0.02,         # NEW (explicit)
    class_dropout_prob: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DDPM:
```

### generate() Function

Updated signature:

```python
def generate(
    model_path: str,
    num_samples: int = 16,
    class_labels: Optional[list] = None,
    guidance_scale: float = 3.0,
    image_size: int = 40,
    num_classes: int = 2,
    model_channels: int = 64,
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",  # NEW
    beta_start: float = 0.0001,     # NEW
    beta_end: float = 0.02,         # NEW
    out_dir: str = "./out/ddpm/samples",
    save_images: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
```

## Implementation Details

### New Methods in DDPM Class

1. **`_get_beta_schedule(schedule, timesteps, beta_start, beta_end)`**
   - Main dispatcher for different schedules
   - Validates schedule name
   - Returns appropriate beta tensor

2. **`_linear_beta_schedule(timesteps, beta_start, beta_end)`**
   - Linear interpolation
   - Original implementation

3. **`_cosine_beta_schedule(timesteps, s=0.008)`**
   - Cosine-based schedule
   - Paper: Nichol & Dhariwal (2021)
   - Ignores beta_start/beta_end

4. **`_quadratic_beta_schedule(timesteps, beta_start, beta_end)`**
   - Quadratic interpolation
   - Smoother transitions

5. **`_sigmoid_beta_schedule(timesteps, beta_start, beta_end)`**
   - Sigmoid-based schedule
   - Very smooth transitions

## Visualization

Use the provided script to visualize different schedules:

```bash
python visualize_schedules.py
```

This generates:
- `noise_schedules_comparison.png`: Individual plots for each schedule
- `noise_schedules_overlay.png`: All schedules overlaid for comparison
- Console output with numerical statistics

## Best Practices

1. **Start with Linear**: Use linear schedule as baseline
2. **Try Cosine for Better Results**: Cosine often gives better sample quality
3. **Match Training and Generation**: Always use the same schedule for training and generation
4. **Experiment**: Try different schedules for your specific dataset
5. **Monitor Training**: Different schedules may require different hyperparameters

## Schedule Comparison

| Schedule  | Smoothness | Training Stability | Sample Quality | Speed |
| --------- | ---------- | ------------------ | -------------- | ----- |
| Linear    | ⭐⭐⭐        | ⭐⭐⭐                | ⭐⭐⭐            | ⭐⭐⭐⭐  |
| Cosine    | ⭐⭐⭐⭐⭐      | ⭐⭐⭐⭐               | ⭐⭐⭐⭐           | ⭐⭐⭐   |
| Quadratic | ⭐⭐⭐⭐       | ⭐⭐⭐⭐               | ⭐⭐⭐⭐           | ⭐⭐⭐⭐  |
| Sigmoid   | ⭐⭐⭐⭐⭐      | ⭐⭐⭐                | ⭐⭐⭐            | ⭐⭐⭐   |

## Examples

### Training with Cosine Schedule

```python
# In train() function or custom training script
model = create_ddpm(
    image_size=40,
    num_classes=2,
    num_timesteps=1000,
    beta_schedule="cosine",
    device="cuda",
)

# Train as usual
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# ... training loop ...
```

### Generating with Matching Schedule

```python
# Generate samples using the same schedule as training
samples = generate(
    model_path="./out/ddpm/ddpm_model_cosine.pth",
    num_samples=16,
    num_classes=2,
    beta_schedule="cosine",  # MUST match training!
    guidance_scale=3.0,
)
```

### Comparing Schedules

```python
# Train models with different schedules
for schedule in ["linear", "cosine", "quadratic", "sigmoid"]:
    print(f"\nTraining with {schedule} schedule...")
    
    model = create_ddpm(
        image_size=40,
        num_classes=2,
        beta_schedule=schedule,
        device="cuda",
    )
    
    # Train...
    
    # Save with schedule name
    torch.save(
        model.state_dict(),
        f"./out/ddpm/model_{schedule}.pth"
    )
```

## Notes

- **Backward Compatibility**: Default is "linear" schedule, so existing code works without changes
- **Cosine Schedule**: Ignores `beta_start` and `beta_end` parameters (uses cosine formula)
- **Model Saving**: The beta schedule is NOT saved in the model checkpoint, so you must specify it when loading
- **Timesteps**: All schedules work with any number of timesteps (10, 100, 1000, etc.)

## Testing

Test the implementation:

```bash
# Quick test
python -c "from ddpm import create_ddpm; \
    m = create_ddpm(image_size=40, num_timesteps=10, beta_schedule='cosine'); \
    print('Cosine OK')"

# Run visualization
python visualize_schedules.py

# Run all tests
pytest tests/test_ddpm.py -v
```

## References

1. **Linear Schedule**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
2. **Cosine Schedule**: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
3. **Quadratic/Sigmoid**: Common variations in diffusion model literature

## Summary

The noise schedule parametrization provides:
- ✅ More control over diffusion process
- ✅ Better training stability (especially cosine)
- ✅ Improved sample quality
- ✅ Easy experimentation
- ✅ Backward compatible (defaults to linear)
- ✅ Well-documented and tested

Choose the schedule that works best for your specific use case and dataset!
