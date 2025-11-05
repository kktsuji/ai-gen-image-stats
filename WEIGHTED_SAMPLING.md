# Weighted Sampling for Class Imbalance

## Overview

Weighted sampling is a technique to handle imbalanced datasets by adjusting the probability of sampling each class. This ensures the model sees both classes equally during training, even when one class has significantly more samples than the other.

## Configuration

In `ddpm.py`, the `train()` function includes a flag to enable/disable weighted sampling:

```python
USE_WEIGHTED_SAMPLING = True  # Enable/disable weighted sampling for class imbalance
```

- **`True`**: Enables weighted sampling (recommended for imbalanced datasets)
- **`False`**: Uses standard random sampling (for balanced datasets)

## How It Works

### 1. Calculate Class Distribution
```python
# Count samples per class
class_counts = [4063, 148]  # Example: [Normal, Abnormal]
```

### 2. Calculate Class Weights (Inverse Frequency)
```python
# Weight = Total samples / Class count
num_samples = 4211
class_weights = [4211/4063, 4211/148]  # [1.036, 28.453]
```

The minority class (Abnormal) gets ~27x higher weight, making it 27x more likely to be sampled.

### 3. Assign Weights to Each Sample
```python
# Each sample gets the weight of its class
sample_weights = [1.036, 1.036, ..., 28.453, 28.453, ...]
```

### 4. Create Weighted Sampler
```python
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),  # Same epoch length
    replacement=True  # Allow duplicates within epoch
)
```

### 5. Use Sampler in DataLoader
```python
# Don't use shuffle=True with sampler!
train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
```

## Effect on Training

### Without Weighted Sampling (Random)
```
Batch 1: [N, N, N, N, N, N, N, N]  ← 100% Normal
Batch 2: [N, N, N, N, N, N, N, N]  ← 100% Normal
Batch 3: [N, A, N, N, N, N, N, N]  ← 12.5% Abnormal (rare!)
Batch 4: [N, N, N, N, N, N, N, N]  ← 100% Normal
...
Expected distribution: 96.5% Normal, 3.5% Abnormal
```

### With Weighted Sampling
```
Batch 1: [N, A, N, A, N, N, A, N]  ← 37.5% Abnormal
Batch 2: [A, N, N, A, N, A, N, A]  ← 50% Abnormal
Batch 3: [N, N, A, N, A, N, N, A]  ← 37.5% Abnormal
Batch 4: [N, A, A, N, N, N, A, A]  ← 50% Abnormal
...
Expected distribution: ~50% Normal, ~50% Abnormal
```

## Output During Training

When enabled, you'll see this output:

```
Training set: 4211 images
  - Class distribution:
    - 0.Normal: 4063 images (96.48%)
    - 1.Abnormal: 148 images (3.52%)

  - Weighted sampling: ENABLED
    - Class weights: ['1.036', '28.453']
  - Number of batches: 526
  - Batch size: 8
```

When disabled:

```
Training set: 4211 images
  - Class distribution:
    - 0.Normal: 4063 images (96.48%)
    - 1.Abnormal: 148 images (3.52%)

  - Weighted sampling: DISABLED (using random sampling)
  - Number of batches: 526
  - Batch size: 8
```

## Testing

You can test the weighted sampling implementation with:

```bash
python test_weighted_sampling.py
```

This script will:
1. Show original dataset distribution
2. Test random sampling (baseline)
3. Test weighted sampling
4. Compare the results

Expected output:
```
Class distribution comparison:
Class           Original        Random          Weighted
------------------------------------------------------------
0.Normal         96.48%         96.25%          48.75%
1.Abnormal        3.52%          3.75%          51.25%

✅ SUCCESS: Weighted sampling is more balanced!
   Improvement: 92.50% better balance
```

## When to Use

### Use Weighted Sampling (`True`) When:
- ✅ Dataset is imbalanced (ratio > 5:1)
- ✅ You want equal representation of all classes
- ✅ Minority class performance is important
- ✅ Training a generative model (like DDPM)

### Don't Use Weighted Sampling (`False`) When:
- ❌ Dataset is already balanced (ratio < 2:1)
- ❌ You want to preserve natural class distribution
- ❌ Training for classification with class priors
- ❌ Memory/compute constraints (balanced sampling is faster)

## Impact on Model Performance

### Benefits:
- **Better minority class learning**: Model sees abnormal samples ~27x more often
- **Balanced generation**: Can generate both classes equally well
- **Prevents majority class bias**: Model doesn't just learn to generate normal images

### Trade-offs:
- **Overfitting risk**: Minority class samples are repeated many times
- **Longer effective training**: Each unique minority sample seen multiple times per epoch
- **Slightly slower convergence**: Model needs to balance both classes

## Recommended Hyperparameters

When using weighted sampling with imbalanced data (27:1 ratio):

```python
EPOCHS = 150-200              # Longer training
BATCH_SIZE = 8                # Smaller batches
LEARNING_RATE = 5e-5          # Lower learning rate
CLASS_DROPOUT_PROB = 0.3      # Higher dropout for unconditional learning
GUIDANCE_SCALE = 5.0-7.0      # Stronger guidance
USE_WEIGHTED_SAMPLING = True  # Enable balancing
```

## Alternatives

If weighted sampling doesn't work well, you can try:

1. **Class-conditional training**: Train separate models for each class
2. **Focal loss**: Use loss function that focuses on hard examples
3. **Data augmentation**: Generate more synthetic minority class samples
4. **Undersample majority**: Reduce normal class samples to match abnormal
5. **Hybrid approach**: Combine weighted sampling with data augmentation

## References

- PyTorch Documentation: [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
- Paper: "A survey on deep learning with class imbalance" (2019)
- DDPM Paper: "Denoising Diffusion Probabilistic Models" (2020)
