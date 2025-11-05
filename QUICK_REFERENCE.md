# DDPM Generate Function - Quick Reference

## Quick Start

### 1. Train a Model
```bash
python ddpm.py
```
This will train a DDPM model and save it to `./out/ddpm/ddpm_model.pth`

### 2. Generate Samples

#### Option A: Using Command Line
```bash
# Generate 16 samples (balanced across classes)
python generate_samples.py

# Generate with custom settings
python generate_samples.py --num_samples 32 --guidance_scale 5.0

# Generate from specific class
python generate_samples.py --num_samples 8 --class_label 0
```

#### Option B: Using Python
```python
from ddpm import generate

# Simple usage
samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=16,
    guidance_scale=3.0,
)

# Class-specific generation
samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=8,
    class_labels=[0, 0, 0, 0, 1, 1, 1, 1],  # 4 per class
    guidance_scale=5.0,
)
```

## Key Parameters

| Parameter        | Default              | Description                                                    |
| ---------------- | -------------------- | -------------------------------------------------------------- |
| `model_path`     | Required             | Path to trained model (.pth file)                              |
| `num_samples`    | 16                   | Number of samples to generate                                  |
| `class_labels`   | None                 | List of class indices (None = auto-balance)                    |
| `guidance_scale` | 3.0                  | Guidance strength (0.0 = unconditional, 3.0-7.0 = conditional) |
| `image_size`     | 40                   | Image size in pixels (square)                                  |
| `num_classes`    | 2                    | Number of classes in model                                     |
| `out_dir`        | "./out/ddpm/samples" | Output directory for images                                    |
| `save_images`    | True                 | Whether to save images to disk                                 |

## Guidance Scale Guide

- **0.0**: Unconditional generation (ignores class labels)
- **1.0-2.0**: Weak conditioning
- **3.0-5.0**: Moderate conditioning (recommended)
- **5.0-7.0**: Strong conditioning
- **7.0+**: Very strong conditioning (may reduce diversity)

## Output Files

When `save_images=True`, generates:
1. `generated_grid_guidance{scale}.png` - All samples in a grid
2. `generated_class{idx}_guidance{scale}.png` - Per-class grids
3. `sample_{idx:03d}_class{label}_guidance{scale}.png` - Individual images

## Common Use Cases

### Generate Balanced Samples
```python
samples = generate("./out/ddpm/ddpm_model.pth", num_samples=20)
# Creates 10 samples per class automatically
```

### Generate Only Class 0 (Normal)
```python
samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=16,
    class_labels=[0] * 16,
    guidance_scale=3.0,
)
```

### Compare Different Guidance Scales
```python
for scale in [0.0, 3.0, 5.0, 7.0]:
    generate(
        model_path="./out/ddpm/ddpm_model.pth",
        num_samples=8,
        guidance_scale=scale,
        out_dir=f"./samples_guidance_{scale}",
    )
```

### Generate Without Saving
```python
samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=4,
    save_images=False,
)
# Returns tensor only, no files saved
```

## Return Value

The `generate()` function returns a PyTorch tensor:
- **Shape**: `(num_samples, 3, image_size, image_size)`
- **Range**: `[-1.0, 1.0]`
- **Device**: Same as specified in `device` parameter

To convert to [0, 1] range for visualization:
```python
samples_normalized = (samples + 1.0) / 2.0
```

## Command Line Options

```bash
python generate_samples.py \
    --model_path ./out/ddpm/ddpm_model.pth \
    --num_samples 16 \
    --guidance_scale 3.0 \
    --num_classes 2 \
    --class_label 0 \
    --image_size 40 \
    --model_channels 64 \
    --num_timesteps 1000 \
    --out_dir ./my_samples \
    --device cuda
```

## Troubleshooting

**Model not found**: Make sure you've trained the model first with `python ddpm.py`

**CUDA out of memory**: Reduce `num_samples` or use `--device cpu`

**Wrong class_labels length**: Ensure `len(class_labels) == num_samples`

**Black/white images**: This is expected if model is untrained. Train first!

## Testing

Run tests to verify everything works:
```bash
# All tests
pytest tests/test_ddpm.py -v

# Only generation tests
pytest tests/test_ddpm.py::TestGenerateFunction -v
```

## Files

- `ddpm.py` - Main implementation (includes `train()` and `generate()`)
- `generate_samples.py` - Command-line interface
- `example_generate.py` - Usage examples
- `DDPM_USAGE.md` - Full documentation
- `tests/test_ddpm.py` - Test suite (67 tests)
