# DDPM Usage Guide

This guide explains how to use the DDPM (Denoising Diffusion Probabilistic Models) implementation for image generation.

## Training

To train a DDPM model, you can either:

1. **Run the training script directly:**
```bash
python ddpm.py
```

2. **Import and use the `train()` function in your code:**
```python
from ddpm import train

train()
```

The training function will:
- Load data from `./data/stats-train` (training) and `./data/stats` (validation)
- Train for 10 epochs with batch size 16
- Save the trained model to `./out/ddpm/ddpm_model.pth`
- Generate sample images during training

### Training Configuration

You can modify the training parameters in the `train()` function:

- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Batch size (default: 16)
- `LEARNING_RATE`: Learning rate (default: 0.0001)
- `NUM_CLASSES`: Number of classes (default: 2)
- `IMG_SIZE`: Image size (default: 40x40)
- `NUM_TIMESTEPS`: Number of diffusion timesteps (default: 1000)
- `MODEL_CHANNELS`: Base U-Net channels (default: 64)
- `CLASS_DROPOUT_PROB`: Probability of dropping class labels for classifier-free guidance (default: 0.1)
- `GUIDANCE_SCALE`: Guidance scale for sampling (default: 3.0)

## Generating Samples

### Method 1: Using the `generate()` Function

```python
from ddpm import generate

# Generate samples from a trained model
samples = generate(
    model_path="./out/ddpm/ddpm_model.pth",
    num_samples=16,
    guidance_scale=3.0,
    image_size=40,
    num_classes=2,
    model_channels=64,
    num_timesteps=1000,
    out_dir="./out/ddpm/samples",
    save_images=True,
    device="cuda",
)

# samples is a torch.Tensor with shape (num_samples, 3, image_size, image_size)
# Values are in range [-1, 1]
```

### Method 2: Using the Command-Line Script

```bash
# Generate 16 samples (balanced across classes)
python generate_samples.py --model_path ./out/ddpm/ddpm_model.pth --num_samples 16 --guidance_scale 3.0

# Generate 8 samples from class 0 only
python generate_samples.py --model_path ./out/ddpm/ddpm_model.pth --num_samples 8 --class_label 0 --guidance_scale 5.0

# Generate 32 samples with high guidance scale
python generate_samples.py --model_path ./out/ddpm/ddpm_model.pth --num_samples 32 --guidance_scale 7.0 --out_dir ./my_samples
```

### Command-Line Options

```
--model_path       Path to trained model checkpoint (default: ./out/ddpm/ddpm_model.pth)
--num_samples      Number of samples to generate (default: 16)
--guidance_scale   Classifier-free guidance scale (default: 3.0)
                   - 0.0: No guidance (unconditional generation)
                   - 3.0-7.0: Typical range for conditional generation
                   - Higher values: Stronger class conditioning
--num_classes      Number of classes in the model (default: 2)
--class_label      Generate all samples from specific class (optional)
--image_size       Image size (default: 40)
--model_channels   Base U-Net channels (default: 64)
--num_timesteps    Number of diffusion timesteps (default: 1000)
--out_dir          Output directory for generated images (default: ./out/ddpm/samples)
--device           Device to run on: cuda or cpu (default: cuda)
```

## Classifier-Free Guidance

The DDPM implementation supports classifier-free guidance for conditional generation:

- **Guidance Scale = 0.0**: Unconditional generation (ignores class labels)
- **Guidance Scale = 1.0**: Standard conditional generation
- **Guidance Scale = 3.0-7.0**: Enhanced conditional generation (typical range)
- **Higher values**: Stronger adherence to class labels, but may reduce sample diversity

### Example: Different Guidance Scales

```python
from ddpm import generate

for guidance in [0.0, 3.0, 5.0, 7.0]:
    samples = generate(
        model_path="./out/ddpm/ddpm_model.pth",
        num_samples=8,
        class_labels=[0] * 4 + [1] * 4,  # 4 samples per class
        guidance_scale=guidance,
        out_dir=f"./samples_guidance_{guidance}",
    )
    print(f"Generated samples with guidance_scale={guidance}")
```

## Programmatic Usage

### Creating a Model

```python
from ddpm import create_ddpm

# Create an unconditional model
model = create_ddpm(
    image_size=40,
    in_channels=3,
    model_channels=64,
    num_classes=None,  # None for unconditional
    num_timesteps=1000,
    device="cuda",
)

# Create a conditional model with 2 classes
model = create_ddpm(
    image_size=40,
    in_channels=3,
    model_channels=64,
    num_classes=2,
    num_timesteps=1000,
    class_dropout_prob=0.1,  # For classifier-free guidance
    device="cuda",
)
```

### Training Step

```python
import torch
from torch.utils.data import DataLoader

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    # Compute loss
    loss = model.training_step(images, class_labels=labels, criterion=criterion)
    
    # Update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Sampling

```python
import torch

# Unconditional sampling
model.eval()
with torch.no_grad():
    samples = model.sample(batch_size=16)

# Conditional sampling
class_labels = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
samples = model.sample(
    batch_size=4,
    class_labels=class_labels,
    guidance_scale=3.0,
)

# Sample range is [-1, 1]
# Convert to [0, 1] for visualization
samples_normalized = (samples + 1.0) / 2.0
```

### Saving and Loading Models

```python
import torch

# Save model
torch.save(model.state_dict(), "./my_model.pth")

# Load model
model = create_ddpm(image_size=40, num_classes=2, device="cuda")
model.load_state_dict(torch.load("./my_model.pth"))
model.eval()
```

## Output Files

When using `generate()` with `save_images=True`, the following files are created:

1. **Grid images**: `generated_grid_guidance{scale}.png` - All samples in a grid layout
2. **Class-specific grids**: `generated_class{idx}_guidance{scale}.png` - Samples grouped by class
3. **Individual samples**: `sample_{idx:03d}_class{label}_guidance{scale}.png` - Each sample separately

All images are automatically denormalized from [-1, 1] to [0, 1] for proper visualization.

## Testing

Run the comprehensive test suite:

```bash
# Run all DDPM tests
pytest tests/test_ddpm.py -v

# Run only generate() function tests
pytest tests/test_ddpm.py::TestGenerateFunction -v

# Run with Docker
docker run --rm --gpus all -v $(pwd):/workspace kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 \
    bash -c "cd /workspace && python -m pytest tests/test_ddpm.py -v"
```

## Model Architecture

- **Image Size**: 40x40x3 (configurable)
- **U-Net Architecture**:
  - Base channels: 64
  - Channel multipliers: (1, 2, 4)
  - 2 residual blocks per level
  - Self-attention on deepest level
- **Time Encoding**: 256-dim sinusoidal embeddings
- **Class Conditioning**: Embedding layer for class tokens
- **Parameters**: ~13.4M (conditional) or ~12.8M (unconditional)
- **Noise Schedule**: Linear from 0.0001 to 0.02

## Tips

1. **Guidance Scale**: Start with 3.0 and adjust based on sample quality
2. **Training Time**: Expect 5-10 minutes per epoch on GPU for ~8,000 images
3. **Memory**: Requires ~2GB GPU memory for batch size 16
4. **Timesteps**: 1000 timesteps gives good quality, but you can reduce to 100-500 for faster sampling
5. **Class Balance**: When `class_labels=None`, samples are automatically balanced across classes
