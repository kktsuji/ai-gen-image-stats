"""Tests for the k-abnormal subsampled AD splits."""

import hashlib
import json
from pathlib import Path

import pytest

from src.experiments.anomaly_detection.abnormal_subsample_split import (
    build_subsampled_split,
    draw_order,
    generate_subsampled_splits,
    main,
)
from src.utils.data.datasets import SplitFileDataset


def _entries(prefix, n, subclass):
    label = 1 if subclass == "abnormal" else 0
    return [
        {"path": f"{prefix}/{subclass}_{i}.png", "label": label, "subclass": subclass}
        for i in range(n)
    ]


def _ad_split(index=3):
    return {
        "train": _entries("tr", 6, "suspicious") + _entries("tr", 8, "abnormal"),
        "val": _entries("va", 2, "suspicious") + _entries("va", 2, "abnormal"),
        "test": _entries("te", 3, "suspicious") + _entries("te", 2, "abnormal"),
        "normal_extra_train": _entries("tr", 4, "red"),
        "normal_extra_val": _entries("va", 1, "red"),
        "normal_extra_test": _entries("te", 1, "red"),
        "metadata": {
            "classes": {"suspicious": 0, "abnormal": 1},
            "split_index": index,
            "repeat_seed": 42,
        },
    }


def _abnormal(entries):
    return [e["path"] for e in entries if e["label"] == 1]


@pytest.mark.unit
class TestBuildSubsampledSplit:
    def test_val_keeps_its_abnormals(self):
        out = build_subsampled_split(_ad_split(), 1, 0)
        assert len(_abnormal(out["val"])) == 2

    def test_only_train_abnormals_change(self):
        src = _ad_split()
        out = build_subsampled_split(src, 3, 0)
        assert len(_abnormal(out["train"])) == 3
        assert [e for e in out["train"] if e["label"] == 0] == [
            e for e in src["train"] if e["label"] == 0
        ]
        for key in (
            "val",
            "test",
            "normal_extra_train",
            "normal_extra_val",
            "normal_extra_test",
        ):
            assert out[key] == src[key]
        assert len(_abnormal(src["train"])) == 8  # source untouched

    def test_metadata_records_the_draw(self):
        out = build_subsampled_split(_ad_split(), 2, 1)
        sub = out["metadata"]["abnormal_subsample"]
        assert sub["k"] == 2 and sub["draw"] == 1
        assert sub["n_abnormal_train_available"] == 8
        assert sorted(sub["kept_paths"]) == sorted(_abnormal(out["train"]))
        assert out["metadata"]["split_index"] == 3

    def test_draws_are_nested_and_deterministic(self):
        split = _ad_split()
        small = set(_abnormal(build_subsampled_split(split, 2, 0)["train"]))
        large = set(_abnormal(build_subsampled_split(split, 5, 0)["train"]))
        assert small < large
        assert draw_order(split, 0) == draw_order(_ad_split(), 0)
        assert draw_order(split, 0) != draw_order(split, 1)
        assert draw_order(split, 0) != draw_order(_ad_split(index=4), 0)

    def test_order_ignores_entry_order(self):
        split = _ad_split()
        shuffled = _ad_split()
        shuffled["train"].reverse()
        assert draw_order(split, 2) == draw_order(shuffled, 2)

    def test_k_equal_to_available_keeps_all(self):
        out = build_subsampled_split(_ad_split(), 8, 0)
        assert sorted(_abnormal(out["train"])) == sorted(
            _abnormal(_ad_split()["train"])
        )

    @pytest.mark.parametrize("k", [0, -1, 9, 2.0, True])
    def test_bad_k_rejected(self, k):
        with pytest.raises(ValueError, match="k"):
            build_subsampled_split(_ad_split(), k, 0)

    def test_bad_source_rejected(self):
        with pytest.raises(ValueError, match="draw must be"):
            draw_order(_ad_split(), -1)
        split = _ad_split()
        del split["metadata"]["repeat_seed"]
        with pytest.raises(ValueError, match="repeat_seed"):
            draw_order(split, 0)
        split = _ad_split()
        del split["train"]
        with pytest.raises(ValueError, match="'train'"):
            draw_order(split, 0)


@pytest.mark.unit
class TestGenerate:
    def _write(self, tmp_path, index):
        src = tmp_path / "src" / f"cv_binary_ad_split{index}.json"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(json.dumps(_ad_split(index)))
        return str(src)

    def test_writes_every_k_and_draw_with_provenance(self, tmp_path):
        srcs = [self._write(tmp_path, i) for i in (0, 1)]
        out = tmp_path / "out"
        written = generate_subsampled_splits(srcs, str(out), [1, 4], 2)
        assert sorted(p.name for p in written) == sorted(
            f"cv_binary_ad_split{i}_k{k}_d{d}.json"
            for i in (0, 1)
            for k in (1, 4)
            for d in (0, 1)
        )
        meta = json.loads((out / "cv_binary_ad_split1_k4_d1.json").read_text())
        sub = meta["metadata"]["abnormal_subsample"]
        assert sub["source_split_file"] == srcs[1]
        assert (
            sub["source_sha256"]
            == hashlib.sha256(Path(srcs[1]).read_bytes()).hexdigest()
        )
        assert sub["k"] == 4 and sub["draw"] == 1

    def test_too_large_k_writes_nothing(self, tmp_path):
        srcs = [self._write(tmp_path, 0)]
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="exceeds"):
            generate_subsampled_splits(srcs, str(out), [1, 20], 1)
        assert not out.exists()

    def test_no_overwrite_without_force(self, tmp_path):
        srcs = [self._write(tmp_path, 0)]
        out = tmp_path / "out"
        out.mkdir()
        (out / "cv_binary_ad_split0_k1_d0.json").write_text("old")
        with pytest.raises(FileExistsError, match="nothing was written"):
            generate_subsampled_splits(srcs, str(out), [1, 2], 1)
        assert sorted(p.name for p in out.iterdir()) == [
            "cv_binary_ad_split0_k1_d0.json"
        ]
        generate_subsampled_splits(srcs, str(out), [1, 2], 1, force=True)
        assert len(list(out.iterdir())) == 2

    @pytest.mark.parametrize(
        "ks,draws,match",
        [([], 1, "ks"), ([1, 1], 1, "ks"), ([1], 0, "draws"), ([1], True, "draws")],
    )
    def test_bad_arguments(self, tmp_path, ks, draws, match):
        srcs = [self._write(tmp_path, 0)]
        with pytest.raises(ValueError, match=match):
            generate_subsampled_splits(srcs, str(tmp_path / "out"), ks, draws)

    def test_bad_sources(self, tmp_path):
        with pytest.raises(ValueError, match="No source"):
            generate_subsampled_splits([], str(tmp_path), [1], 1)
        bad = tmp_path / "cv_binary_ad_split0_k1_d0.json"
        bad.write_text(json.dumps(_ad_split()))
        with pytest.raises(ValueError, match="split index"):
            generate_subsampled_splits([str(bad)], str(tmp_path / "out"), [1], 1)
        a = self._write(tmp_path, 2)
        b = tmp_path / "other" / "cv_binary_ad_split2.json"
        b.parent.mkdir()
        b.write_text(json.dumps(_ad_split(2)))
        with pytest.raises(ValueError, match="Two source files"):
            generate_subsampled_splits([a, str(b)], str(tmp_path / "out"), [1], 1)
        padded = tmp_path / "padded" / "cv_binary_ad_split02.json"
        padded.parent.mkdir()
        padded.write_text(json.dumps(_ad_split(2)))
        with pytest.raises(ValueError, match="Two source files"):
            generate_subsampled_splits([a, str(padded)], str(tmp_path / "out"), [1], 1)

    def test_cli_and_dataset_compatibility(self, tmp_path):
        self._write(tmp_path, 0)
        out = tmp_path / "out"
        main(
            ["--src-dir", str(tmp_path / "src"), "--out-dir", str(out)]
            + ["--k", "2", "--draws", "1"]
        )
        dataset = SplitFileDataset(str(out / "cv_binary_ad_split0_k2_d0.json"), "train")
        assert len(dataset) == 6 + 2
        assert dataset.get_classes() == ["suspicious", "abnormal"]


@pytest.mark.unit
class TestCliSkipsOutputs:
    def test_rerun_on_a_folder_holding_outputs(self, tmp_path):
        src = tmp_path / "splits"
        src.mkdir()
        (src / "cv_binary_ad_split0.json").write_text(json.dumps(_ad_split(0)))
        args = ["--src-dir", str(src), "--out-dir", str(src), "--k", "2"]
        main(args + ["--draws", "1"])
        assert (src / "cv_binary_ad_split0_k2_d0.json").exists()
        # The outputs now match the default pattern but are not read as sources.
        main(args + ["--draws", "2", "--force"])
        assert sorted(p.name for p in src.iterdir()) == [
            "cv_binary_ad_split0.json",
            "cv_binary_ad_split0_k2_d0.json",
            "cv_binary_ad_split0_k2_d1.json",
        ]
