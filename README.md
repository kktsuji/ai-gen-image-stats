# AI Gen Image Stats

## Prerequisites

1. Install NVIDIA GPU driver (see [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/))
2. Install WSL2 to Windows (see [my post](https://tsuji.tech/install-uninstall-wsl/))
3. Install Docker to WSL (see [my post](https://tsuji.tech/install-docker-to-wsl/))
4. Install NVIDIA Container Toolkit (see [my post](https://tsuji.tech/use-nvidia-gpu-with-wsl-docker/))

## Usage (Use PyTorch Image)

Execute the following commands on WSL.

```bash
# Check compatibility for your system
docker pull pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Build the docker image
docker build -t kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 .

# Evaluate
# Avoid Numba cache issue and Matplotlib config issue by setting environment variables
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) -e NUMBA_DISABLE_CACHE=1 -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp -e XDG_CACHE_HOME=/tmp kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 stats.py

# Cleansing
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 cleansing.py

# Train
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 prepare_dataset.py
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 train.py
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 ddpm.py

# Test
docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work --user $(id -u):$(id -g) kktsuji/pytorch-1.7.1-cuda11.0-cudnn8 python3 -m pytest
```
