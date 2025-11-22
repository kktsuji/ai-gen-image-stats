# Hardware-Specific Tuning Guide

## Your System Configuration

```
CPU: Intel i7-8700K (6 cores, 12 threads @ 3.70GHz)
GPU: NVIDIA RTX 5070 (12GB VRAM, 60 SMs, Tensor Cores)
RAM: 16GB DDR4
OS:  WSL2 on Windows
```

---

## Optimal Settings by Task

### 1. Image Classification Training (`train.py`)

#### Recommended Configuration
```bash
python train.py \
  --model-type inception_v3 \
  --batch-size 24 \
  --num-workers 6 \
  --epochs 10 \
  --learning-rate 0.00005
```

**Rationale:**
- **batch-size=24**: Fits in 12GB VRAM with InceptionV3 (299×299 input)
- **num-workers=6**: Half of your CPU threads, leaves room for OS/other processes
- With these settings, expect **~40s/epoch** (down from ~120s)

#### If Using WRN28-CIFAR10
```bash
python train.py \
  --model-type wrn28_cifar10 \
  --batch-size 64 \
  --num-workers 6 \
  --epochs 10
```

**Rationale:**
- **batch-size=64**: WRN28 uses 40×40 images, can fit larger batches
- Even faster training due to smaller images

---

### 2. DDPM Training (`ddpm_train.py`)

#### Recommended Configuration
```bash
python ddpm_train.py \
  --batch-size 16 \
  --num-workers 6 \
  --epochs 100 \
  --learning-rate 0.005 \
  --img-size 40
```

**Rationale:**
- **batch-size=16**: DDPM U-Net is memory-intensive, 16 is safe for 40×40 images
- **num-workers=6**: Same as training
- With these settings, expect **~30-40s/epoch** for 40×40 images

#### For Larger Images (64×64)
```bash
python ddpm_train.py \
  --batch-size 8 \
  --num-workers 6 \
  --img-size 64
```

---

## Finding Your Optimal `num_workers`

### Quick Benchmark Script

```bash
#!/bin/bash
# Save as benchmark_workers.sh

echo "Benchmarking num_workers for your hardware..."

for nw in 0 2 4 6 8 10; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing num_workers=$nw"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run 2 epochs and measure time
    /usr/bin/time -f "Time: %E, CPU: %P" \
      python train.py \
        --epochs 2 \
        --batch-size 16 \
        --num-workers $nw \
        2>&1 | tail -n 5
done

echo ""
echo "Choose the configuration with lowest time and reasonable CPU%"
```

**How to use:**
```bash
chmod +x benchmark_workers.sh
./benchmark_workers.sh
```

**Expected results:**
- `num_workers=0`: Baseline (slow, ~100% CPU on 1 core)
- `num_workers=2`: ~1.5× faster
- `num_workers=4`: ~2× faster
- `num_workers=6`: **~2.4× faster** ← Recommended
- `num_workers=8`: ~2.5× faster (diminishing returns)
- `num_workers=10`: May be slower (overhead)

---

## Docker Shared Memory Sizing

### Current Setting (in `.env`)
```bash
--shm-size=4g
```

### When to Increase

If you see this error:
```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal
```

**Solution:** Increase shared memory
```bash
--shm-size=8g  # or even 12g if you have RAM
```

### Rule of Thumb
- **Minimum:** `num_workers × batch_size × image_memory × 2`
- For your config: `6 × 16 × 0.5MB × 2 = 96MB` (4GB is plenty)

---

## Batch Size Optimization

### How to Find Maximum Batch Size

```bash
# Start small and increase until OOM (Out of Memory)
for bs in 8 16 24 32 48 64; do
    echo "Testing batch_size=$bs"
    python train.py --epochs 1 --batch-size $bs --num-workers 6 \
      && echo "✓ $bs works" \
      || echo "✗ $bs OOM"
done
```

### Memory Guidelines (RTX 5070 - 12GB)

| Model         | Input Size | Max Batch Size | Recommended |
| ------------- | ---------- | -------------- | ----------- |
| InceptionV3   | 299×299    | ~32            | 24          |
| WRN28-CIFAR10 | 40×40      | ~128           | 64          |
| DDPM U-Net    | 40×40      | ~24            | 16          |
| DDPM U-Net    | 64×64      | ~12            | 8           |

---

## Advanced: Mixed Precision Training

Your RTX 5070 has **Tensor Cores** optimized for FP16 operations.

### Enable Automatic Mixed Precision

Add this to your training loop:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
for images, labels in train_loader:
    optimizer.zero_grad()
    
    # Automatic mixed precision context
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- **2-3× faster training**
- **~40% less memory** (can increase batch size)
- Minimal accuracy impact

**Try these batch sizes with AMP:**
- InceptionV3: 48-64 (up from 24)
- DDPM: 32 (up from 16)

---

## CPU Affinity (Advanced)

If you notice inconsistent performance, pin workers to specific cores:

```python
import os
import torch

# Set thread affinity
torch.set_num_threads(6)  # Match num_workers
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
```

Add to top of `train.py` and `ddpm_train.py`.

---

## Environment Variables for Performance

Add these to your `.env`:

```bash
# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# cuDNN performance
export CUDNN_BENCHMARK=1

# Disable debug overhead in production
export PYTHONOPTIMIZE=1
```

---

## Monitoring Commands

### Real-Time Performance Dashboard

```bash
# Terminal 1: GPU monitoring
watch -n 0.5 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Terminal 2: CPU monitoring
htop -d 5

# Terminal 3: Training
python train.py --num-workers 6 --batch-size 24
```

### What Good Performance Looks Like

**GPU (nvidia-smi):**
```
GPU Utilization: 85-100%
Memory Used: 8-10GB / 12GB
Temperature: 60-75°C
Power: 180-200W (near TDP)
```

**CPU (htop):**
```
CPU 1-6:  [||||||||40%    ]  ← Data loading workers
CPU 7-12: [|||10%          ]  ← Available for system
Load Average: 4.0-6.0
```

**If GPU utilization < 80%:**
1. Increase `num_workers` (more parallel data loading)
2. Increase `batch_size` (if memory allows)
3. Check if CPU is bottleneck (should see <80% per core)

**If CPU at 100% on all cores:**
- Reduce `num_workers`
- Simplify data augmentation
- Consider preprocessing data offline

---

## Incremental Testing Plan

### Phase 1: Baseline (Current)
```bash
python train.py --epochs 2 --num-workers 0
# Expected: ~240s total, 1 core at 100%
```

### Phase 2: Add Workers
```bash
python train.py --epochs 2 --num-workers 6
# Expected: ~100s total, 6 cores at 40-60%
# Speedup: 2.4×
```

### Phase 3: Optimize Batch Size
```bash
python train.py --epochs 2 --num-workers 6 --batch-size 24
# Expected: ~80s total (larger batches = fewer iterations)
# Speedup: 3×
```

### Phase 4: (Optional) Mixed Precision
```bash
# After implementing AMP in code
python train.py --epochs 2 --num-workers 6 --batch-size 48
# Expected: ~50s total
# Speedup: 4.8×
```

---

## Troubleshooting

### Issue: Workers not starting
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**Solution:** Set `multiprocessing_context='spawn'`
```python
train_loader = DataLoader(
    dataset,
    num_workers=6,
    multiprocessing_context='spawn'  # Add this
)
```

### Issue: Slow first epoch
- **Normal:** Workers are initialized during first epoch
- **With `persistent_workers=True`:** Overhead only occurs once

### Issue: CPU usage lower than expected
- Docker may not have access to all cores
- Check: `docker run --cpus=12 ...` (add CPU limit)

---

## Summary: Your Optimal Configuration

```bash
# .env file
TRAIN_BATCH_SIZE=24
TRAIN_NUM_WORKERS=6
DDPM_TRAIN_BATCH_SIZE=16
DDPM_TRAIN_NUM_WORKERS=6

# Docker command
DOCKER_COMMAND_PREFIX="docker run --rm --gpus all --shm-size=4g --cpus=12 -v $PWD:/work -w /work kktsuji/pytorch-2.9.0-cuda12.8-cudnn9 python3"
```

**Expected Performance:**
- Classification: **40s/epoch** (was 120s)
- DDPM: **30s/epoch** (was 90s)
- GPU Utilization: **85-95%** (was 40%)

---

**Last Updated:** 2025-11-22  
**Optimized for:** i7-8700K + RTX 5070 + WSL2
