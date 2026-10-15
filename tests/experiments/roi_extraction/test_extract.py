"""Tests for roi_extraction's 16-bit ROI extraction."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.experiments.roi_extraction import extract as extract_module
from src.experiments.roi_extraction.catalog import run_catalog
from src.experiments.roi_extraction.extract import (
    find_image_dir,
    find_tile,
    read_uint16,
    roi_window,
    run_extract,
    write_uint16_png,
)
from tests.experiments.roi_extraction.conftest import tile_values


@pytest.mark.unit
class TestRoiWindow:
    @pytest.mark.parametrize(
        "center, expected",
        [
            (20, (0, 40)),  # centered: [center - 20, center + 20)
            (100, (80, 120)),
            (5, (0, 40)),  # top/left edge: shifted inward
            (0, (0, 40)),
            (995, (960, 1000)),  # bottom/right edge: shifted inward
            (999, (960, 1000)),
        ],
    )
    def test_window(self, center, expected):
        assert roi_window(center, 40, 1000) == expected

    def test_odd_size_window(self):
        # The original code takes size // 2 on both sides, then pads the start.
        assert roi_window(10, 5, 100) == (7, 12)

    @pytest.mark.parametrize("center", [-1, 1000, 5000])
    def test_center_outside_axis_raises(self, center):
        with pytest.raises(ValueError, match="outside"):
            roi_window(center, 40, 1000)

    def test_axis_shorter_than_roi_raises(self):
        with pytest.raises(ValueError, match="Cannot fit"):
            roi_window(5, 40, 30)


@pytest.mark.unit
class TestFiles:
    def test_find_image_dir(self, tmp_path):
        (tmp_path / "S" / "2 (after) tif").mkdir(parents=True)
        (tmp_path / "S" / "1 (before)").mkdir()
        assert find_image_dir(str(tmp_path), "S", "*after*").name == "2 (after) tif"

    def test_find_image_dir_requires_exactly_one(self, tmp_path):
        (tmp_path / "S" / "a after").mkdir(parents=True)
        (tmp_path / "S" / "b after").mkdir()
        with pytest.raises(FileNotFoundError, match="exactly one"):
            find_image_dir(str(tmp_path), "S", "*after*")

    def test_find_tile_by_suffix_not_by_order(self, tmp_path):
        for ch in ("ch00", "ch01", "ch02", "ch03"):
            (tmp_path / f"x_s007_{ch}.tif").touch()
        (tmp_path / "x_s070_ch01.tif").touch()
        assert find_tile(tmp_path, 7, "ch01").name == "x_s007_ch01.tif"
        with pytest.raises(FileNotFoundError):
            find_tile(tmp_path, 8, "ch01")

    def test_uint16_png_round_trip(self, tmp_path):
        array = np.array([[0, 1], [4096, 65535]], dtype=np.uint16)
        write_uint16_png(tmp_path / "a.png", array)
        restored = read_uint16(tmp_path / "a.png")
        assert restored.dtype == np.uint16
        np.testing.assert_array_equal(restored, array)

    def test_read_uint16_accepts_big_endian(self, tmp_path):
        values = np.array([[1, 258], [4096, 65535]], dtype=">u2")
        Image.frombytes("I;16B", (2, 2), values.tobytes()).save(tmp_path / "a.tif")
        restored = read_uint16(tmp_path / "a.tif")
        assert restored.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(restored, [[1, 258], [4096, 65535]])

    def test_read_uint16_rejects_8bit(self, tmp_path):
        Image.new("L", (2, 2)).save(tmp_path / "a.png")
        with pytest.raises(ValueError, match="16-bit"):
            read_uint16(tmp_path / "a.png")


@pytest.mark.component
class TestRunExtract:
    def test_extracts_every_cell_losslessly(self, roi_inputs):
        run_catalog(roi_inputs)
        report = run_extract(roi_inputs)
        out = roi_inputs["data"]["raw16_dir"]

        # S 1, tile 1, cell at x=5, y=6, roi_size 4 -> rows 4:8, cols 3:7
        for key, channel_index in (("b", 1), ("g", 2), ("r", 3)):
            roi = read_uint16(Path(out) / f"S-1_img-no001_y6_x5_{key}.png")
            expected = tile_values(0, 1, channel_index)[4:8, 3:7]
            np.testing.assert_array_equal(roi, expected)

        # corner cell (x=15, y=11) is shifted inside the 12 x 16 tile
        roi = read_uint16(Path(out) / "S-1_img-no001_y11_x15_b.png")
        np.testing.assert_array_equal(roi, tile_values(0, 1, 1)[8:12, 12:16])

        assert report["cell_counts"] == {"ctcs": 2, "normal_cells": 3}
        assert report["image_dirs"] == {"S 1": "S 1/2 (after)", "S 2": "S 2/2 (after)"}
        with open(f"{out}/extraction.json") as f:
            assert json.load(f)["catalog_sha256"] == report["catalog_sha256"]

    def test_reads_each_tile_once(self, roi_inputs, mocker):
        run_catalog(roi_inputs)
        spy = mocker.spy(extract_module, "read_uint16")
        run_extract(roi_inputs)
        # 4 distinct tiles (S 1: 1, 2, 3; S 2: 1) x 3 channels, although tile
        # (S 1, 1) is used by both sources
        assert spy.call_count == 12

    def test_channel_tiles_of_different_shape_raise(self, roi_inputs):
        run_catalog(roi_inputs)
        tile_dir = Path(roi_inputs["data"]["image_root"]) / "S 1" / "2 (after)"
        Image.fromarray(np.zeros((10, 16), dtype=np.uint16)).save(
            tile_dir / "S 1 after.lif_s001_ch02.tif"
        )
        with pytest.raises(ValueError, match="differ in shape"):
            run_extract(roi_inputs)

    def test_output_overlapping_an_input_raises(self, roi_inputs):
        run_catalog(roi_inputs)
        roi_inputs["data"]["raw16_dir"] = str(
            Path(roi_inputs["data"]["catalog"]).parent
        )
        roi_inputs["output"]["force"] = True
        with pytest.raises(ValueError, match="overlaps the input"):
            run_extract(roi_inputs)
        assert Path(roi_inputs["data"]["catalog"]).is_file()

    def test_requires_catalog(self, roi_inputs):
        with pytest.raises(FileNotFoundError, match="--mode catalog"):
            run_extract(roi_inputs)

    def test_refuses_non_empty_output(self, roi_inputs):
        run_catalog(roi_inputs)
        run_extract(roi_inputs)
        with pytest.raises(FileExistsError):
            run_extract(roi_inputs)
        roi_inputs["output"]["force"] = True
        run_extract(roi_inputs)
