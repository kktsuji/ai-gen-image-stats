"""Shared fixtures for anomaly-detection tests: a tiny fake backbone and a
miniature extended split on disk."""

import json

import pytest
import torch
import torch.nn as nn
from PIL import Image


class TinyBackbone(nn.Module):
    """Stand-in for ResNetClassifier: named layers + extract_features()."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layer2 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(4, 6, 3, stride=2, padding=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(x))
        return x.mean(dim=(2, 3))


@pytest.fixture
def tiny_backbone(monkeypatch):
    """Patch the feature model factory so no pretrained weights are loaded."""
    model = TinyBackbone().eval()
    monkeypatch.setattr(
        "src.experiments.anomaly_detection.features.create_feature_model",
        lambda name, device: model,
    )
    return model


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


@pytest.fixture
def ad_split_file(tmp_path):
    """Extended split JSON with bright 'abnormal' and dark normal images."""
    root = tmp_path / "img"

    def entries(prefix, n, label, subclass, value):
        out = []
        for i in range(n):
            path = root / subclass / f"{prefix}{i}.png"
            _save(path, value + i)
            out.append({"path": str(path), "label": label, "subclass": subclass})
        return out

    split = {
        "metadata": {"classes": {"suspicious": 0, "abnormal": 1}},
        "train": entries("tr", 6, 0, "suspicious", 20)
        + entries("tr", 2, 1, "abnormal", 230),
        "val": entries("va", 3, 0, "suspicious", 25),
        "test": entries("te", 4, 0, "suspicious", 22)
        + entries("te", 3, 1, "abnormal", 235),
        "normal_extra_train": entries("xtr", 5, 0, "red", 30),
        "normal_extra_val": entries("xva", 2, 0, "red", 32),
        "normal_extra_test": entries("xte", 3, 0, "junk", 40),
    }
    path = tmp_path / "cv_binary_ad_split0.json"
    path.write_text(json.dumps(split))
    return path
