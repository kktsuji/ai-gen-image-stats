"""
Vendored subset of OpenAI's guided-diffusion (ADM).

Source: https://github.com/openai/guided-diffusion
License: MIT (Copyright (c) 2021 OpenAI) — see the LICENSE file in this directory.

Only the modules required to *build* the ADM U-Net and its Gaussian-diffusion
process and to *load* the public ImageNet-64 checkpoint are vendored:

    nn.py, losses.py, fp16_util.py (trimmed), gaussian_diffusion.py,
    respace.py, unet.py

The training scripts, distributed helpers, dataset readers, and logger from the
upstream package are intentionally omitted; fine-tuning is driven by this
project's own trainer. These files are vendored (rather than added as a
dependency) for reproducibility and offline Docker training, and are excluded
from this repo's ruff/pyright checks.

Vendor edits (search for "VENDOR EDIT") -- minimal changes for transfer-learning
compatibility, since upstream never froze parameters:

- fp16_util.py: trimmed to the two conversion helpers (drops the logger dep).
- unet.py (AttentionBlock.forward): respect ``self.use_checkpoint`` instead of
  hardcoding gradient checkpointing on.
- nn.py (CheckpointFunction.backward): only differentiate inputs that require
  grad, so checkpointing composes with a frozen backbone.
"""
