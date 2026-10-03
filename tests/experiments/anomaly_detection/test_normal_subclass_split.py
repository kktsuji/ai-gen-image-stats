"""Tests for the normal-subclass classification split derivation."""

import hashlib
import json

import pytest

from src.experiments.anomaly_detection.normal_subclass_split import (
    build_normal_subclass_split,
    generate_normal_subclass_splits,
    main,
)
from src.utils.data.datasets import SplitFileDataset


def _entries(prefix, n, subclass):
    label = 1 if subclass == "abnormal" else 0
    return [
        {"path": f"{prefix}/{subclass}_{i}.png", "label": label, "subclass": subclass}
        for i in range(n)
    ]


def _ad_split():
    return {
        "train": _entries("tr", 4, "suspicious") + _entries("tr", 2, "abnormal"),
        "val": _entries("va", 2, "suspicious") + _entries("va", 1, "abnormal"),
        "test": _entries("te", 3, "suspicious") + _entries("te", 1, "abnormal"),
        "normal_extra_train": _entries("tr", 5, "red") + _entries("tr", 2, "junk"),
        "normal_extra_val": _entries("va", 1, "red") + _entries("va", 1, "junk"),
        "normal_extra_test": _entries("te", 2, "red") + _entries("te", 1, "junk"),
        "metadata": {
            "classes": {"suspicious": 0, "abnormal": 1},
            "split_index": 3,
            "repeat": 0,
            "fold": 3,
            "n_folds": 5,
            "n_repeats": 2,
            "repeat_seed": 42,
            "val_fraction": 0.15,
            "unrelated": "dropped",
        },
    }


@pytest.mark.unit
class TestBuildNormalSubclassSplit:
    def test_no_abnormal_anywhere(self):
        out = build_normal_subclass_split(_ad_split())
        for part in ("train", "val", "test"):
            assert all(e["subclass"] != "abnormal" for e in out[part])
        assert "abnormal" not in out["metadata"]["classes"]

    def test_partitions_follow_the_ad_split(self):
        out = build_normal_subclass_split(_ad_split())
        assert len(out["train"]) == 4 + 5 + 2
        assert len(out["val"]) == 2 + 1 + 1
        assert len(out["test"]) == 3 + 2 + 1
        assert all(e["path"].startswith("tr/") for e in out["train"])
        assert all(e["path"].startswith("te/") for e in out["test"])

    def test_alphabetical_labels_and_metadata(self):
        out = build_normal_subclass_split(_ad_split())
        meta = out["metadata"]
        assert meta["classes"] == {"junk": 0, "red": 1, "suspicious": 2}
        for e in out["train"]:
            assert e["label"] == meta["classes"][e["subclass"]]
        assert meta["class_samples"]["train"] == {"junk": 2, "red": 5, "suspicious": 4}
        assert meta["split_index"] == 3 and meta["repeat_seed"] == 42
        assert "unrelated" not in meta

    def test_missing_subclass_tag_rejected(self):
        split = _ad_split()
        del split["val"][0]["subclass"]
        with pytest.raises(ValueError, match="subclass tag"):
            build_normal_subclass_split(split)

    def test_empty_partition_rejected(self):
        split = _ad_split()
        split["val"] = _entries("va", 1, "abnormal")
        split["normal_extra_val"] = []
        with pytest.raises(ValueError, match="'val'"):
            build_normal_subclass_split(split)

    def test_overlap_rejected(self):
        split = _ad_split()
        split["normal_extra_test"].append(dict(split["normal_extra_train"][0]))
        with pytest.raises(ValueError, match="in both 'train' and 'test'"):
            build_normal_subclass_split(split)


@pytest.mark.unit
class TestGenerate:
    def _write_source(self, tmp_path, index=3):
        src = tmp_path / "src" / f"cv_binary_ad_split{index}.json"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(json.dumps(_ad_split()))
        return src

    def test_writes_file_with_provenance(self, tmp_path):
        src = self._write_source(tmp_path)
        (target,) = generate_normal_subclass_splits([str(src)], str(tmp_path / "out"))
        assert target.name == "normal_subclass_split3.json"
        meta = json.loads(target.read_text())["metadata"]
        assert meta["source_split_file"] == str(src)
        assert meta["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()

    def test_no_overwrite_without_force(self, tmp_path):
        src = self._write_source(tmp_path)
        out = str(tmp_path / "out")
        generate_normal_subclass_splits([str(src)], out)
        with pytest.raises(FileExistsError):
            generate_normal_subclass_splits([str(src)], out)
        generate_normal_subclass_splits([str(src)], out, force=True)

    def test_bad_inputs(self, tmp_path):
        with pytest.raises(ValueError, match="No source"):
            generate_normal_subclass_splits([], str(tmp_path))
        bad = tmp_path / "noindex.json"
        bad.write_text(json.dumps(_ad_split()))
        with pytest.raises(ValueError, match="split index"):
            generate_normal_subclass_splits([str(bad)], str(tmp_path / "out"))

    def test_cli_and_dataset_compatibility(self, tmp_path):
        self._write_source(tmp_path, index=0)
        self._write_source(tmp_path, index=1)
        out = tmp_path / "out"
        main(["--src-dir", str(tmp_path / "src"), "--out-dir", str(out)])
        assert sorted(p.name for p in out.iterdir()) == [
            "normal_subclass_split0.json",
            "normal_subclass_split1.json",
        ]
        dataset = SplitFileDataset(str(out / "normal_subclass_split0.json"), "train")
        assert dataset.get_classes() == ["junk", "red", "suspicious"]
        assert len(dataset) == 11
