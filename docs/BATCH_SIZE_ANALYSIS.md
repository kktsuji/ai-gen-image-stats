# Batch Size Analysis for Small Datasets

## Dataset Context
- **Total images**: 522 (437 normal, 85 abnormal)
- **Class imbalance ratio**: ~5.1:1 (normal:abnormal)
- **Training approach**: Weighted sampling enabled

## Recommended Batch Size: 8

### Current Configuration
- **DDPM training**: `DDPM_TRAIN_BATCH_SIZE=8` ✓ Optimal
- **Classification training**: `TRAIN_BATCH_SIZE=16` → Should reduce to 8

## Why Batch Size 8 is Optimal

### 1. Sufficient Gradient Updates
- **Batch 8**: 65 updates per epoch (522 ÷ 8)
- **Batch 16**: 33 updates per epoch (522 ÷ 16)
- **Batch 32**: 16 updates per epoch (522 ÷ 32)

More gradient updates = faster convergence and better learning on small datasets

### 2. Minority Class Representation
| Batch Size | Total Batches/Epoch | Abnormal Batches (~16%) | Updates/Epoch | Recommendation  |
| ---------- | ------------------- | ----------------------- | ------------- | --------------- |
| 4          | 130                 | ~21                     | 130           | ✓ Good (slower) |
| **8**      | **65**              | **~10**                 | **65**        | **✓ Optimal**   |
| 16         | 33                  | ~5                      | 33            | ✓ Acceptable    |
| 32         | 16                  | ~2-3                    | 16            | ✗ Too large     |

### 3. Gradient Noise Benefits
Small batches provide beneficial gradient noise for small datasets:
- Acts as implicit regularization
- Helps escape local minima
- Improves generalization
- Reduces overfitting risk

### 4. Training Iterations Comparison
For 100 epochs:
- **Batch 8**: 100 × 65 = **6,500 gradient updates**
- **Batch 32**: 100 × 16 = **1,600 gradient updates**

Batch 32 would need ~400 epochs to match batch 8's learning progress.

## Why Not Batch Size 32?

### Common Misconception
"The ratio of abnormal batches stays constant, so batch size doesn't matter"

### Reality
While weighted sampling maintains the **proportion** of minority class samples:
- Batch 8: 65 batches, ~10 abnormal → 15.4%
- Batch 32: 16 batches, ~2.5 abnormal → 15.6%

The issue is **not the ratio**, but the **total number of learning opportunities**:

#### 1. Fewer Gradient Updates (Primary Issue)

Each batch triggers one weight update (gradient descent step):

- **Batch 8**: 65 weight updates per epoch
- **Batch 32**: 16 weight updates per epoch

Small datasets require more frequent weight adjustments to learn effectively. With only 522 images, fewer updates means slower convergence and potentially worse final performance.

#### 2. Gradient Noise as Regularization

**Batch 8 (noisier gradients)**:

- Acts as implicit regularization
- Helps escape local minima
- Better exploration of loss landscape
- Improves generalization on small datasets

**Batch 32 (smoother gradients)**:

- More stable but can overfit easily
- Less exploration, may get stuck in suboptimal solutions
- Requires explicit regularization techniques

#### 3. Effective Training Iterations

To achieve equivalent learning:

- **100 epochs @ batch 8** = 6,500 gradient updates
- **100 epochs @ batch 32** = 1,600 gradient updates
- **Batch 32 needs ~400 epochs** to match batch 8's 100 epochs

#### 4. Learning Retention for Rare Classes

Think of it like studying vocabulary:

- **Batch 8**: 65 study sessions per "epoch" with frequent exposure to rare words
- **Batch 32**: 16 study sessions per "epoch" with same frequency ratio
- Same ratio, but **4× fewer reinforcement cycles** = weaker retention

**Analogy**: Learning 10 common words + 2 rare words

- **More sessions (batch 8)**: See rare words ~10 times across 65 reviews → better retention
- **Fewer sessions (batch 32)**: See rare words ~2.5 times across 16 reviews → worse retention
- The **absolute number of exposures matters**, not just the ratio

## Weighted Sampling (Current Implementation)

### What It Does
- Oversamples minority class based on inverse frequency
- Normal weight: 522/437 ≈ 1.19
- Abnormal weight: 522/85 ≈ 6.14

### Why It's Sufficient
With batch size 8 and weighted sampling:
- **Expected abnormal samples per batch**: ~1.3
- **Probability of ≥1 abnormal per batch**: ~90%+
- Natural variation (sometimes 0, sometimes 2-3) helps learning

### Alternative Approaches (Not Recommended)
❌ **Force ≥1 sample per batch**: 
- More complex implementation
- Reduces beneficial randomness
- Weighted sampling already achieves this probabilistically
- Current approach is more elegant and effective

## General Guidelines

### Batch Size by Dataset Size
| Dataset Size | Recommended Batch Size |
| ------------ | ---------------------- |
| < 1,000      | 4-16                   |
| 1,000-5,000  | 16-32                  |
| 5,000-10,000 | 32-64                  |
| > 10,000     | 64-128+                |

### For Your Dataset (522 images)
- **Optimal**: 8
- **Acceptable**: 4-16
- **Not recommended**: 32+

## Current Training Strategy (Already Optimal)

### DDPM Training (`ddpm_train.py`)
```python
batch_size = 8                    # ✓ Optimal
use_weighted_sampling = True      # ✓ Enabled
class_dropout_prob = 0.3          # ✓ For classifier-free guidance
```

### Classification Training (`train.py`)
```python
batch_size = 16                   # → Should reduce to 8
use_weighted_sampling = True      # ✓ Can enable via env var
use_class_weights = True          # ✓ Loss function weighted
```

## Recommendations

1. **Keep DDPM batch size at 8** - already optimal
2. **Reduce classification batch size from 16 to 8** - better for minority class
3. **Keep weighted sampling enabled** - no need for forced sampling tricks
4. **Monitor training logs** - verify minority class accuracy is improving

## Key Takeaway

For small datasets with class imbalance:
- **Batch size 8 is optimal** for your 522-image dataset
- **Weighted sampling** already handles minority class effectively
- **More gradient updates > larger batch size** for learning efficiency
- Focus on generating synthetic data (DDPM) to increase dataset size, rather than adjusting batch size further

---

*Analysis Date: November 24, 2025*
*Dataset: 437 normal, 85 abnormal (522 total)*
