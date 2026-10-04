"""Anomaly Detection - Representation Adaptation on Normal Images (mode: adapt)

Adapts a pretrained backbone to the normal training images of an extended AD
split with the Mean-Shifted Contrastive (MSC) loss (Reiss & Hoshen, "Mean-Shifted
Contrastive Loss for Anomaly Detection", AAAI 2023). No abnormal image and no
label is used: only ``select_partitions(...)["train_normals"]`` of the chosen
``data.normal_pool``.

MSC, as implemented here:

1. ``center``: the L2-normalized mean of the L2-normalized features of the
   training normals, computed once with the initial (ImageNet) backbone and
   deterministic preprocessing.
2. Each step takes two random augmentations of a batch. Their normalized
   features are shifted by the center and normalized again, and the NT-Xent
   contrastive loss (temperature ``tau``) pulls the two views of each image
   together and pushes other images apart, in the mean-shifted space.

Only the parameters matching ``adaptation.trainable_layers`` are updated (the
classification head ``fc.*``, unused by ``extract_features``, never is).
BatchNorm running statistics are a separate choice,
``adaptation.update_frozen_bn_stats``: ``true`` lets every BatchNorm layer
re-estimate them on the normal images during training (as the classifier
training of the other campaigns does), ``false`` keeps the layers without
trainable parameters in eval mode so they stay exactly as initialised. The
result is saved in the classifier checkpoint format
(``checkpoints/final_model.pth``, written last and atomically), so the
``run`` mode can use it through ``feature_extraction.checkpoint``.
"""

import fnmatch
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.experiments.anomaly_detection.features import HEAD_PREFIX, PathListDataset
from src.experiments.anomaly_detection.runner import (
    load_extended_split,
    select_partitions,
)
from src.experiments.sample_selection.selector import create_feature_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import resolve_output_path
from src.utils.data.transforms import get_normalization_transform, get_val_transforms

logger = logging.getLogger(__name__)


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees (cells have no orientation)."""

    def __call__(self, image: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return image.rotate(90 * k) if k else image


def build_augmentation(fe: Dict[str, Any], aug: Dict[str, Any]) -> transforms.Compose:
    """Training augmentation: geometry and blur only, never color.

    Stain color is what tells several normal subclasses apart, so color
    jitter or grayscale would remove the very information to be learned.
    """
    steps: List[Any] = [
        transforms.Resize(fe["image_size"]),
        transforms.RandomResizedCrop(
            fe["crop_size"], scale=tuple(aug["crop_scale"]), ratio=(1.0, 1.0)
        ),
    ]
    if aug["horizontal_flip"]:
        steps.append(transforms.RandomHorizontalFlip())
    if aug["vertical_flip"]:
        steps.append(transforms.RandomVerticalFlip())
    if aug["rotate90"]:
        steps.append(RandomRotate90())
    if aug["blur_probability"] > 0:
        steps.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))],
                p=aug["blur_probability"],
            )
        )
    steps += [transforms.ToTensor(), get_normalization_transform("imagenet")]
    return transforms.Compose(steps)


class TwoViewDataset(Dataset):
    """Two independent augmentations of each image (for contrastive learning)."""

    def __init__(self, paths: Sequence[str], transform: Any) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), self.transform(image)


def mean_shift(features: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Normalize, shift by the (normalized) center, and normalize again."""
    return F.normalize(F.normalize(features, dim=-1) - center, dim=-1)


def msc_loss(
    view1: torch.Tensor, view2: torch.Tensor, temperature: float
) -> torch.Tensor:
    """NT-Xent loss between two views, on already mean-shifted features.

    For 2B embeddings, each one's positive is the other view of the same image;
    the other 2B - 2 embeddings are negatives (self-similarity excluded). The
    inputs are normalized again here on purpose: ``mean_shift`` already returns
    unit vectors (so this is a no-op in training), but the loss stays correct
    for any input, as in the MSC reference implementation.
    """
    z = torch.cat([view1, view2], dim=0)
    z = F.normalize(z, dim=-1)
    n = view1.shape[0]
    logits = z @ z.t() / temperature
    logits = logits.masked_fill(
        torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf")
    )
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(logits, targets)


def set_trainable(model: torch.nn.Module, patterns: Sequence[str]) -> int:
    """Train only parameters whose name matches a pattern; returns their count.

    The classification head (``fc.*``) is never trained: ``extract_features``
    does not use it, so it would get no gradient.

    Raises:
        ValueError: If no backbone parameter matches.
    """
    n_trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith(HEAD_PREFIX) and any(
            fnmatch.fnmatch(name, p) for p in patterns
        )
        n_trainable += param.numel() if param.requires_grad else 0
    if n_trainable == 0:
        raise ValueError(
            f"No backbone parameter matches adaptation.trainable_layers {patterns} "
            f"(the head '{HEAD_PREFIX}*' is not used by extract_features)"
        )
    return n_trainable


def freeze_untrained_batchnorm(model: torch.nn.Module) -> int:
    """Put BatchNorm layers without trainable parameters in eval mode.

    Call after ``model.train()``: their running statistics then stay fixed.
    Returns the number of layers frozen this way.
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()
            frozen += 1
    return frozen


@torch.no_grad()
def compute_center(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> torch.Tensor:
    """Normalized mean of the normalized features (eval mode, no augmentation)."""
    model.eval()
    feats = [
        F.normalize(model.extract_features(x.to(device)), dim=-1)  # type: ignore[operator]
        for x in loader
    ]
    return F.normalize(torch.cat(feats).mean(dim=0), dim=-1)


def run_adaptation(config: Dict[str, Any], device: str) -> Path:
    """Adapt the backbone on normal training images; returns the checkpoint path."""
    fe = config["feature_extraction"]
    ad = config["adaptation"]

    split = load_extended_split(config["data"]["split_file"])
    paths = [
        e["path"]
        for e in select_partitions(split, config["data"]["normal_pool"])[
            "train_normals"
        ]
    ]
    if len(paths) < 2 * ad["batch_size"]:
        raise ValueError(
            f"{len(paths)} training normals are too few for batch size "
            f"{ad['batch_size']} (need at least two full batches)"
        )
    logger.info(f"Adapting on {len(paths)} training normals ({ad['method']})")

    model = create_feature_model(fe["model"], device)
    n_trainable = set_trainable(model, ad["trainable_layers"])
    logger.info(f"Trainable parameters: {n_trainable:,} ({ad['trainable_layers']})")

    center_loader = DataLoader(
        PathListDataset(
            paths,
            get_val_transforms(fe["image_size"], fe["crop_size"], "imagenet"),
        ),
        # Gradient-free feature extraction, batched as in run mode.
        batch_size=fe["batch_size"],
        shuffle=False,
        num_workers=fe["num_workers"],
    )
    center = compute_center(model, center_loader, device)

    train_loader = DataLoader(
        TwoViewDataset(paths, build_augmentation(fe, ad["augmentation"])),
        batch_size=ad["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=fe["num_workers"],
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=ad["learning_rate"],
        momentum=ad["momentum"],
        weight_decay=ad["weight_decay"],
    )

    history = []
    start = time.time()
    for epoch in range(1, ad["epochs"] + 1):
        model.train()
        if not ad["update_frozen_bn_stats"]:
            freeze_untrained_batchnorm(model)
        losses = []
        for view1, view2 in train_loader:
            f1 = model.extract_features(view1.to(device))  # type: ignore[operator]
            f2 = model.extract_features(view2.to(device))  # type: ignore[operator]
            loss = msc_loss(
                mean_shift(f1, center), mean_shift(f2, center), ad["temperature"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        if not np.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite MSC loss at epoch {epoch}")
        history.append({"epoch": epoch, "loss": mean_loss, "steps": len(losses)})
        logger.info(
            f"Epoch {epoch}/{ad['epochs']} - MSC loss {mean_loss:.4f} "
            f"({time.time() - start:.0f}s)"
        )

    reports_dir = resolve_output_path(config, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(reports_dir / "adaptation_history.csv", index=False)
    summary = {
        "method": ad["method"],
        "normal_pool": config["data"]["normal_pool"],
        "n_train_normals": len(paths),
        "epochs": ad["epochs"],
        "steps": int(sum(h["steps"] for h in history)),
        "trainable_parameters": n_trainable,
        "update_frozen_bn_stats": ad["update_frozen_bn_stats"],
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "split_file": config["data"]["split_file"],
    }
    with open(reports_dir / "adaptation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # The checkpoint is the done marker, so it is written last and atomically.
    checkpoints_dir = resolve_output_path(config, "checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoints_dir / "final_model.pth"
    tmp_path = checkpoints_dir / "final_model.pth.tmp"
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=ad["epochs"],
        global_step=summary["steps"],
        is_best=False,
        metrics={"msc_loss": summary["final_loss"]},
        best_metric=None,
        best_metric_name=None,
        trainer_class="MSCAdaptation",
        save_optimizer=False,
    )
    os.replace(tmp_path, final_path)
    logger.info(f"Adapted backbone saved: {final_path}")
    return final_path
