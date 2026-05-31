"""
Helpers to convert modules between 16-bit and 32-bit precision.

Trimmed vendored copy of guided_diffusion/fp16_util.py from
https://github.com/openai/guided-diffusion (MIT License, Copyright (c) 2021 OpenAI).

Only ``convert_module_to_f16`` / ``convert_module_to_f32`` are retained, since
``unet.py`` imports them at module load time. The original mixed-precision
training helpers (and their ``logger`` dependency) are intentionally omitted:
this project drives fine-tuning through its own trainer and AMP scaler.
"""

import torch.nn as nn


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
