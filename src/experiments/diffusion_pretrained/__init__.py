"""Pretrained-transfer diffusion experiment slice (ADM / guided-diffusion).

Transfers OpenAI's class-conditional ImageNet-64 ADM checkpoint onto small,
domain-specific datasets via a re-initialized class head and frozen/staged
backbone fine-tuning. See ``model.py`` for the wrapper and ``config.py`` for the
strict configuration schema.
"""
