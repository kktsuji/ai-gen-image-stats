"""Tests for roi_extraction's rendering (pseudo_rgb8_vstretch)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.experiments.roi_extraction.catalog import run_catalog
from src.experiments.roi_extraction.extract import run_extract, write_uint16_png
from src.experiments.roi_extraction.render import (
    load_bgr16,
    render_pseudo_rgb8_vstretch,
    run_render,
    to_uint8,
)
from tests.experiments.roi_extraction.conftest import tile_values

# Golden values from OpenCV 4.11.0 (the version that made the in-house images):
# vstretch(GOLDEN_IN, 50, 99) == GOLDEN_OUT, BGR uint8.
GOLDEN_IN = np.array(
    [
        [[95, 130, 194], [217, 207, 235], [15, 163, 33], [215, 217, 130]],
        [[248, 189, 16], [69, 184, 232], [205, 78, 169], [61, 125, 10]],
        [[29, 240, 66], [19, 182, 39], [59, 4, 59], [81, 222, 44]],
        [[120, 122, 50], [208, 78, 28], [64, 166, 121], [89, 170, 233]],
    ],
    dtype=np.uint8,
)
GOLDEN_OUT = np.array(
    [
        [[0, 0, 0], [167, 159, 180], [0, 0, 0], [65, 66, 40]],
        [[255, 191, 16], [48, 127, 161], [0, 0, 0], [0, 0, 0]],
        [[26, 211, 57], [0, 0, 0], [0, 0, 0], [35, 98, 20]],
        [[0, 0, 0], [9, 3, 1], [0, 0, 0], [64, 122, 167]],
    ],
    dtype=np.uint8,
)


@pytest.mark.unit
class TestToUint8:
    def test_rounds_to_nearest(self):
        # x * 255 / 65535 = x / 257: 128 -> 0.498, 129 -> 0.502, 33024 -> 128.498,
        # 33025 -> 128.502 (exact halves cannot occur, 257 being odd)
        values = np.array([0, 128, 129, 33024, 33025, 65535], dtype=np.uint16)
        np.testing.assert_array_equal(to_uint8(values), [0, 0, 1, 128, 129, 255])

    def test_dtype(self):
        assert to_uint8(np.zeros((2, 2, 3), dtype=np.uint16)).dtype == np.uint8


@pytest.mark.unit
class TestVstretch:
    def test_golden_values(self):
        pytest.importorskip("cv2")
        from src.experiments.roi_extraction.render import vstretch

        np.testing.assert_array_equal(vstretch(GOLDEN_IN, 50, 99), GOLDEN_OUT)

    def test_flat_image_raises(self):
        pytest.importorskip("cv2")
        from src.experiments.roi_extraction.render import vstretch

        with pytest.raises(ValueError, match="Cannot stretch"):
            vstretch(np.full((4, 4, 3), 7, dtype=np.uint8), 50, 99)

    def test_without_stretch_is_plain_8bit(self):
        bgr16 = (GOLDEN_IN.astype(np.uint16) * 257).astype(np.uint16)
        np.testing.assert_array_equal(
            render_pseudo_rgb8_vstretch(bgr16, None), GOLDEN_IN
        )

    def test_with_stretch(self):
        pytest.importorskip("cv2")
        bgr16 = (GOLDEN_IN.astype(np.uint16) * 257).astype(np.uint16)
        stretched = render_pseudo_rgb8_vstretch(
            bgr16, {"lower_percentile": 50, "upper_percentile": 99}
        )
        np.testing.assert_array_equal(stretched, GOLDEN_OUT)


@pytest.mark.unit
def test_load_bgr16_stacks_b_g_r(tmp_path):
    for i, key in enumerate("bgr"):
        write_uint16_png(tmp_path / f"x_{key}.png", np.full((2, 2), i, np.uint16))
    np.testing.assert_array_equal(load_bgr16(tmp_path, "x")[0, 0], [0, 1, 2])


def _rgb(path):
    return np.array(Image.open(path))


@pytest.mark.component
class TestRunRender:
    @pytest.fixture
    def extracted(self, roi_inputs, tmp_path):
        layout = tmp_path / "layout"
        for class_path, name in [
            ("abnormal", "S-1_img-no001_y6_x5"),
            ("normal/blue", "S-2_img-no001_y7_x7"),
            ("normal/red", "S-2_img-no001_y7_x7"),
        ]:
            (layout / class_path).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2)).save(layout / class_path / f"{name}.png")
        roi_inputs["catalog"]["class_layout_dir"] = str(layout)
        run_catalog(roi_inputs)
        run_extract(roi_inputs)
        return roi_inputs

    def test_class_paths_layout(self, extracted):
        extracted["rendering"]["vstretch"] = None
        report = run_render(extracted)
        out = Path(extracted["rendering"]["output_dir"])
        assert report["written_counts"] == {
            "abnormal": 1,
            "normal/blue": 1,
            "normal/red": 1,
        }
        assert report["skipped_count"] == 3
        assert "S-1_img-no002_y0_x0" in report["skipped"]
        # file is RGB: channel 0 = r (ch03), channel 2 = b (ch01)
        rgb = _rgb(out / "abnormal" / "S-1_img-no001_y6_x5.png")
        assert rgb.shape == (4, 4, 3)
        np.testing.assert_array_equal(
            rgb[..., 0], to_uint8(tile_values(0, 1, 3)[4:8, 3:7])
        )
        np.testing.assert_array_equal(
            rgb[..., 2], to_uint8(tile_values(0, 1, 1)[4:8, 3:7])
        )
        np.testing.assert_array_equal(
            _rgb(out / "normal" / "blue" / "S-2_img-no001_y7_x7.png"),
            _rgb(out / "normal" / "red" / "S-2_img-no001_y7_x7.png"),
        )
        with open(out / "rendering.json") as f:
            assert json.load(f)["method"] == "pseudo_rgb8_vstretch"

    def test_pool_layout_writes_every_cell(self, extracted):
        extracted["rendering"]["vstretch"] = None
        extracted["rendering"]["layout"] = "pool"
        report = run_render(extracted)
        assert report["written_counts"] == {"ctcs": 2, "normal_cells": 3}
        assert report["skipped_count"] == 0

    def test_stretched_render(self, extracted):
        pytest.importorskip("cv2")
        report = run_render(extracted)
        assert report["vstretch"] == {"lower_percentile": 50, "upper_percentile": 99}
        out = Path(extracted["rendering"]["output_dir"])
        rgb = _rgb(out / "abnormal" / "S-1_img-no001_y6_x5.png")
        assert rgb.min() == 0  # the lower half of the value channel is zeroed
