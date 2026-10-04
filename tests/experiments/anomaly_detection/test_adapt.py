"""Tests for representation adaptation on normal images (mode: adapt, MSC)."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.experiments.anomaly_detection.adapt import (
    RandomRotate90,
    build_augmentation,
    mean_shift,
    msc_loss,
    run_adaptation,
    set_trainable,
)
from src.experiments.anomaly_detection.config import validate_config
from src.experiments.anomaly_detection.features import load_backbone_weights
from tests.experiments.anomaly_detection.conftest import TinyBackbone


def _reference_nt_xent(out_1, out_2, temperature):
    """The MSC reference implementation's formulation (exp / masked sum)."""
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    out = torch.cat([out_1, out_2], dim=0)
    sim = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim) - torch.eye(2 * bs)).bool()
    sim = sim.masked_select(mask).view(2 * bs, -1)
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    return (-torch.log(pos / sim.sum(dim=-1))).mean()


@pytest.mark.unit
class TestLoss:
    def test_matches_reference_formulation(self):
        torch.manual_seed(0)
        a, b = torch.randn(5, 8), torch.randn(5, 8)
        assert msc_loss(a, b, 0.25).item() == pytest.approx(
            _reference_nt_xent(a, b, 0.25).item(), rel=1e-5
        )

    def test_aligned_views_give_lower_loss(self):
        torch.manual_seed(0)
        a = torch.randn(6, 8)
        noisy = a + 0.05 * torch.randn(6, 8)
        assert msc_loss(a, noisy, 0.25) < msc_loss(a, noisy[torch.randperm(6)], 0.25)

    def test_mean_shift(self):
        torch.manual_seed(0)
        feats = torch.randn(4, 8) * 3
        center = F.normalize(torch.randn(8), dim=-1)
        out = mean_shift(feats, center)
        expected = F.normalize(F.normalize(feats, dim=-1) - center, dim=-1)
        assert torch.allclose(out, expected)
        assert torch.allclose(out.norm(dim=-1), torch.ones(4))


@pytest.mark.unit
class TestHelpers:
    def test_set_trainable(self):
        model = TinyBackbone()
        n = set_trainable(model, ["layer3*"])
        assert n == sum(p.numel() for p in model.layer3.parameters())
        assert not any(p.requires_grad for p in model.layer2.parameters())
        assert all(p.requires_grad for p in model.layer3.parameters())

    def test_set_trainable_requires_a_match(self):
        with pytest.raises(ValueError, match="No backbone parameter matches"):
            set_trainable(TinyBackbone(), ["layer9*"])

    def test_rotate90_keeps_size(self):
        image = Image.new("RGB", (8, 6))
        for _ in range(8):
            assert sorted(RandomRotate90()(image).size) == [6, 8]

    def test_augmentation_shape_and_no_color_ops(self):
        aug = {
            "crop_scale": [0.5, 1.0],
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotate90": True,
            "blur_probability": 0.5,
        }
        t = build_augmentation({"image_size": 20, "crop_size": 16}, aug)
        out = t(Image.new("RGB", (40, 40), (200, 30, 30)))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 16, 16)
        names = [type(step).__name__ for step in t.transforms]
        assert not any("Color" in n or "Gray" in n for n in names)


def _adapt_config(split_file, tmp_path, pool="all"):
    return {
        "experiment": "anomaly_detection",
        "mode": "adapt",
        "compute": {"device": "cpu", "seed": 0},
        "data": {"split_file": str(split_file), "normal_pool": pool},
        "feature_extraction": {
            "model": "resnet50",
            "image_size": 20,
            "crop_size": 16,
            "batch_size": 4,
            "num_workers": 0,
            "cache_dir": None,
            "checkpoint": None,
        },
        "adaptation": {
            "method": "msc",
            "trainable_layers": ["layer3*"],
            "update_frozen_bn_stats": True,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.05,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "temperature": 0.25,
            "augmentation": {
                "crop_scale": [0.5, 1.0],
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotate90": True,
                "blur_probability": 0.2,
            },
        },
        "output": {
            "base_dir": str(tmp_path / "out"),
            "subdirs": {
                "logs": "logs",
                "checkpoints": "checkpoints",
                "reports": "reports",
            },
        },
        "logging": {"console_level": "INFO", "file_level": "INFO"},
    }


@pytest.mark.unit
class TestAdaptConfig:
    def test_valid(self, tmp_path):
        validate_config(_adapt_config("s.json", tmp_path))

    def test_run_sections_not_required(self, tmp_path):
        config = _adapt_config("s.json", tmp_path)
        assert "method" not in config and "threshold" not in config
        validate_config(config)

    @pytest.mark.parametrize(
        "path",
        [
            ("adaptation",),
            ("adaptation", "method"),
            ("adaptation", "trainable_layers"),
            ("adaptation", "update_frozen_bn_stats"),
            ("adaptation", "temperature"),
            ("adaptation", "augmentation"),
            ("adaptation", "augmentation", "crop_scale"),
            ("adaptation", "augmentation", "rotate90"),
            ("output", "subdirs", "checkpoints"),
        ],
    )
    def test_missing(self, tmp_path, path):
        config = _adapt_config("s.json", tmp_path)
        node = config
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        with pytest.raises((KeyError, ValueError)):
            validate_config(config)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("method", "panda"),
            ("trainable_layers", []),
            ("trainable_layers", [""]),
            ("epochs", 0),
            ("batch_size", 1),
            ("learning_rate", 0),
            ("temperature", -1),
            ("momentum", 1.0),
            ("weight_decay", -0.1),
            ("update_frozen_bn_stats", "no"),
        ],
    )
    def test_invalid_adaptation(self, tmp_path, key, value):
        config = _adapt_config("s.json", tmp_path)
        config["adaptation"][key] = value
        with pytest.raises(ValueError):
            validate_config(config)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("crop_scale", [0.8, 0.5]),
            ("crop_scale", [0.0, 1.0]),
            ("crop_scale", [0.5]),
            ("horizontal_flip", "yes"),
            ("blur_probability", 1.5),
        ],
    )
    def test_invalid_augmentation(self, tmp_path, key, value):
        config = _adapt_config("s.json", tmp_path)
        config["adaptation"]["augmentation"][key] = value
        with pytest.raises(ValueError):
            validate_config(config)

    def test_example_config_is_valid(self):
        import yaml

        with open("configs/examples/anomaly-detection-adapt.yaml") as f:
            validate_config(yaml.safe_load(f))

    def test_checkpoint_must_be_null(self, tmp_path):
        config = _adapt_config("s.json", tmp_path)
        config["feature_extraction"]["checkpoint"] = "x.pth"
        with pytest.raises(ValueError, match="must be null in adapt mode"):
            validate_config(config)


@pytest.fixture
def tiny_adapt_model(monkeypatch):
    model = TinyBackbone().eval()
    monkeypatch.setattr(
        "src.experiments.anomaly_detection.adapt.create_feature_model",
        lambda name, device: model,
    )
    return model


@pytest.mark.component
class TestRunAdaptation:
    def test_end_to_end(self, ad_split_file, tmp_path, tiny_adapt_model):
        before = copy.deepcopy(tiny_adapt_model.state_dict())
        ckpt = run_adaptation(_adapt_config(ad_split_file, tmp_path), "cpu")
        assert ckpt.name == "final_model.pth" and ckpt.exists()
        assert not (ckpt.parent / "final_model.pth.tmp").exists()

        after = tiny_adapt_model.state_dict()
        assert torch.equal(before["layer2.weight"], after["layer2.weight"])
        assert not torch.equal(before["layer3.weight"], after["layer3.weight"])

        reports = tmp_path / "out" / "reports"
        summary = json.loads((reports / "adaptation.json").read_text())
        # pool "all": 6 suspicious + 5 red training normals, batch 4, drop_last
        assert summary["n_train_normals"] == 11
        assert summary["steps"] == 2 * 2
        assert (reports / "adaptation_history.csv").read_text().count("\n") == 3

        fresh = TinyBackbone()
        load_backbone_weights(fresh, str(ckpt))
        assert torch.equal(fresh.layer3.weight, after["layer3.weight"])

    def test_suspicious_pool_uses_only_suspicious(
        self, ad_split_file, tmp_path, tiny_adapt_model
    ):
        config = _adapt_config(ad_split_file, tmp_path, pool="suspicious")
        config["adaptation"]["batch_size"] = 3
        run_adaptation(config, "cpu")
        summary = json.loads(
            (tmp_path / "out" / "reports" / "adaptation.json").read_text()
        )
        assert summary["n_train_normals"] == 6

    def test_too_few_normals(self, ad_split_file, tmp_path, tiny_adapt_model):
        config = _adapt_config(ad_split_file, tmp_path, pool="suspicious")
        config["adaptation"]["batch_size"] = 4  # 6 normals < 2 full batches
        with pytest.raises(ValueError, match="too few"):
            run_adaptation(config, "cpu")


class _BNBackbone(torch.nn.Module):
    """Tiny backbone with BatchNorm in a frozen and in a trainable block."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, stride=2, padding=1), torch.nn.BatchNorm2d(4)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, 3, stride=2, padding=1), torch.nn.BatchNorm2d(6)
        )
        self.fc = torch.nn.Linear(6, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(x)).mean(dim=(2, 3))


@pytest.mark.unit
class TestReviewFixes:
    def test_head_never_trainable(self):
        model = _BNBackbone()
        set_trainable(model, ["*"])
        assert not any(p.requires_grad for p in model.fc.parameters())
        assert all(p.requires_grad for p in model.layer3.parameters())

    def test_head_only_pattern_rejected(self):
        with pytest.raises(ValueError, match="not used by extract_features"):
            set_trainable(_BNBackbone(), ["fc*"])

    def test_freeze_untrained_batchnorm(self):
        from src.experiments.anomaly_detection.adapt import freeze_untrained_batchnorm

        model = _BNBackbone()
        set_trainable(model, ["layer3*"])
        model.train()
        assert freeze_untrained_batchnorm(model) == 1
        assert not model.layer2[1].training
        assert model.layer3[1].training


@pytest.mark.component
class TestBatchNormStats:
    @pytest.mark.parametrize("update", [True, False])
    def test_frozen_bn_stats_follow_the_setting(
        self, ad_split_file, tmp_path, monkeypatch, update
    ):
        model = _BNBackbone().eval()
        monkeypatch.setattr(
            "src.experiments.anomaly_detection.adapt.create_feature_model",
            lambda name, device: model,
        )
        before = copy.deepcopy(model.state_dict())
        config = _adapt_config(ad_split_file, tmp_path)
        config["adaptation"]["update_frozen_bn_stats"] = update
        run_adaptation(config, "cpu")
        after = model.state_dict()
        frozen_same = torch.equal(
            before["layer2.1.running_mean"], after["layer2.1.running_mean"]
        )
        assert frozen_same is (not update)
        # The frozen block's weights never change; the trained block's stats do.
        assert torch.equal(before["layer2.0.weight"], after["layer2.0.weight"])
        assert not torch.equal(
            before["layer3.1.running_mean"], after["layer3.1.running_mean"]
        )
        summary = json.loads(
            (tmp_path / "out" / "reports" / "adaptation.json").read_text()
        )
        assert summary["update_frozen_bn_stats"] is update


@pytest.mark.component
class TestBatchSizes:
    def test_center_uses_feature_extraction_batch_size(
        self, ad_split_file, tmp_path, tiny_adapt_model, monkeypatch
    ):
        import src.experiments.anomaly_detection.adapt as adapt_module

        seen = []
        real_loader = adapt_module.DataLoader

        def recording_loader(dataset, **kwargs):
            seen.append((type(dataset).__name__, kwargs["batch_size"]))
            return real_loader(dataset, **kwargs)

        monkeypatch.setattr(adapt_module, "DataLoader", recording_loader)
        config = _adapt_config(ad_split_file, tmp_path)
        config["feature_extraction"]["batch_size"] = 5
        config["adaptation"]["batch_size"] = 3
        run_adaptation(config, "cpu")
        assert seen == [("PathListDataset", 5), ("TwoViewDataset", 3)]
