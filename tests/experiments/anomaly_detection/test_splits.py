"""Tests for anomaly-detection extended split generation."""

import json

import pytest

from src.experiments.anomaly_detection.splits import (
    EXTRA_SPLIT_KEYS,
    _remap_path,
    build_extended_split,
    generate_extended_splits,
    main,
    split_extra_subclass,
)


def _binary_split(fold=0, repeat_seed=42):
    """Minimal kfold binary split dict (suspicious=0, abnormal=1)."""
    return {
        "metadata": {
            "mode": "kfold",
            "classes": {"suspicious": 0, "abnormal": 1},
            "repeat_seed": repeat_seed,
            "fold": fold,
            "n_folds": 5,
            "n_repeats": 2,
            "val_fraction": 0.15,
        },
        "train": [
            {"path": "data/normal/suspicious/s0.png", "label": 0},
            {"path": "data/abnormal/a0.png", "label": 1},
        ],
        "val": [{"path": "data/normal/suspicious/s1.png", "label": 0}],
        "test": [
            {"path": "data/normal/suspicious/s2.png", "label": 0},
            {"path": "data/abnormal/a1.png", "label": 1},
        ],
    }


def _extra_files(n=23):
    return {
        "red": [f"data/normal/red/r{i}.png" for i in range(n)],
        "junk": [f"data/normal/junk/j{i}.png" for i in range(7)],
    }


@pytest.mark.unit
class TestRemapPath:
    def test_none_keeps_path(self):
        assert _remap_path("data/a.png", None) == "data/a.png"

    def test_prefix_replaced(self):
        assert _remap_path("data/a.png", ("data/", "data/in-house/")) == (
            "data/in-house/a.png"
        )

    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError, match="remap prefix"):
            _remap_path("other/a.png", ("data/", "x/"))


@pytest.mark.unit
class TestSplitExtraSubclass:
    def test_folds_partition_subclass_within_repeat(self):
        files = [f"f{i}" for i in range(23)]
        tests = [
            split_extra_subclass(files, "red", 42, fold, 5, 0.15)["test"]
            for fold in range(5)
        ]
        flat = [p for t in tests for p in t]
        assert sorted(flat) == sorted(files)
        assert len(set(flat)) == len(flat)

    def test_parts_are_disjoint_and_complete(self):
        files = [f"f{i}" for i in range(23)]
        parts = split_extra_subclass(files, "red", 42, 2, 5, 0.15)
        all_paths = parts["train"] + parts["val"] + parts["test"]
        assert sorted(all_paths) == sorted(files)
        assert len(parts["val"]) == round(len(parts["train"] + parts["val"]) * 0.15)

    def test_deterministic_and_input_order_invariant(self):
        files = [f"f{i}" for i in range(23)]
        a = split_extra_subclass(files, "red", 42, 1, 5, 0.15)
        b = split_extra_subclass(list(reversed(files)), "red", 42, 1, 5, 0.15)
        assert a == b

    def test_seed_changes_assignment(self):
        files = [f"f{i}" for i in range(23)]
        a = split_extra_subclass(files, "red", 42, 0, 5, 0.15)
        b = split_extra_subclass(files, "red", 43, 0, 5, 0.15)
        assert a["test"] != b["test"]


@pytest.mark.unit
class TestBuildExtendedSplit:
    def test_inherits_binary_entries_unchanged(self):
        binary = _binary_split()
        ext = build_extended_split(binary, _extra_files())
        for key in ("train", "val", "test"):
            assert [(e["path"], e["label"]) for e in ext[key]] == [
                (e["path"], e["label"]) for e in binary[key]
            ]

    def test_subclass_tags(self):
        ext = build_extended_split(_binary_split(), _extra_files())
        assert {e["subclass"] for e in ext["test"]} == {"suspicious", "abnormal"}
        extra_subclasses = {e["subclass"] for k in EXTRA_SPLIT_KEYS for e in ext[k]}
        assert extra_subclasses == {"red", "junk"}

    def test_extra_entries_are_normal_label(self):
        ext = build_extended_split(_binary_split(), _extra_files())
        assert all(e["label"] == 0 for k in EXTRA_SPLIT_KEYS for e in ext[k])

    def test_no_leakage_between_partitions(self):
        ext = build_extended_split(_binary_split(), _extra_files())
        keys = ("train", "val", "test", *EXTRA_SPLIT_KEYS)
        paths = [e["path"] for k in keys for e in ext[k]]
        assert len(paths) == len(set(paths))

    def test_duplicate_path_raises(self):
        extra = {"red": ["data/normal/suspicious/s2.png"]}
        with pytest.raises(ValueError, match="Leakage"):
            build_extended_split(_binary_split(), extra)

    def test_path_remap_applied_to_inherited(self):
        ext = build_extended_split(
            _binary_split(), _extra_files(), path_remap=("data/", "root/")
        )
        assert all(e["path"].startswith("root/") for e in ext["train"])

    def test_metadata_records_extras(self):
        ext = build_extended_split(
            _binary_split(), _extra_files(), source_file="s.json"
        )
        ad_meta = ext["metadata"]["anomaly_detection"]
        assert ad_meta["source_split_file"] == "s.json"
        assert set(ad_meta["extra_subclasses"]) == {"red", "junk"}
        assert ext["metadata"]["fold"] == 0

    def test_rejects_non_kfold(self):
        binary = _binary_split()
        binary["metadata"]["mode"] = "single"
        with pytest.raises(ValueError, match="kfold"):
            build_extended_split(binary, _extra_files())

    def test_rejects_non_binary_labels(self):
        binary = _binary_split()
        binary["metadata"]["classes"] = {"a": 0, "b": 2}
        with pytest.raises(ValueError, match="binary"):
            build_extended_split(binary, _extra_files())


def _write_images(directory, names):
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (8, 8)).save(directory / name)


@pytest.mark.component
class TestGenerateExtendedSplits:
    def _setup(self, tmp_path):
        root = tmp_path / "data"
        _write_images(root / "normal" / "suspicious", ["s0.png", "s1.png", "s2.png"])
        _write_images(root / "abnormal", ["a0.png", "a1.png"])
        _write_images(root / "normal" / "red", [f"r{i}.png" for i in range(12)])
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for fold in range(2):
            binary = _binary_split(fold=fold)
            with open(src_dir / f"cv_binary_split{fold}.json", "w") as f:
                json.dump(binary, f)
        return root, src_dir

    def test_generates_files_and_skips_existing(self, tmp_path):
        root, src_dir = self._setup(tmp_path)
        out_dir = tmp_path / "out"
        src_files = sorted(str(p) for p in src_dir.glob("*.json"))
        remap = ("data/", f"{root}/")
        outputs = generate_extended_splits(
            src_files, str(root / "normal"), str(out_dir), ["red"], remap
        )
        assert [p.split("/")[-1] for p in outputs] == [
            "cv_binary_ad_split0.json",
            "cv_binary_ad_split1.json",
        ]
        with open(outputs[0]) as f:
            ext = json.load(f)
        assert len(ext["normal_extra_test"]) > 0

        # Second call without force keeps the file untouched.
        mtime = (out_dir / "cv_binary_ad_split0.json").stat().st_mtime_ns
        generate_extended_splits(
            src_files, str(root / "normal"), str(out_dir), ["red"], remap
        )
        assert (out_dir / "cv_binary_ad_split0.json").stat().st_mtime_ns == mtime

    def test_missing_files_raise(self, tmp_path):
        root, src_dir = self._setup(tmp_path)
        src_files = sorted(str(p) for p in src_dir.glob("*.json"))
        with pytest.raises(FileNotFoundError, match="path-remap"):
            generate_extended_splits(
                src_files, str(root / "normal"), str(tmp_path / "out"), ["red"]
            )

    def test_empty_sources_raise(self, tmp_path):
        with pytest.raises(ValueError, match="No source"):
            generate_extended_splits([], str(tmp_path), str(tmp_path / "out"))

    def test_cli(self, tmp_path):
        root, src_dir = self._setup(tmp_path)
        out_dir = tmp_path / "out"
        main(
            [
                "--src-dir",
                str(src_dir),
                "--normal-dir",
                str(root / "normal"),
                "--out-dir",
                str(out_dir),
                "--extra-subclasses",
                "red",
                "--path-remap",
                "data/",
                f"{root}/",
            ]
        )
        assert len(list(out_dir.glob("cv_binary_ad_split*.json"))) == 2
