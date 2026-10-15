"""Tests for the roi_extraction cell catalog."""

import json

import pytest
from PIL import Image

from src.experiments.roi_extraction.catalog import (
    build_catalog,
    read_coordinates,
    roi_name,
    run_catalog,
    scan_class_layout,
)

SOURCES = {"ctcs": "groundTruthCTCs.csv", "normal_cells": "normalCells.csv"}


def _touch_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


@pytest.mark.unit
class TestHelpers:
    def test_roi_name_matches_original_naming(self):
        assert (
            roi_name("160520 L-2 1-2 tif", 71, 771, 417)
            == "160520-L-2-1-2-tif_img-no071_y417_x771"
        )

    def test_roi_name_keeps_wide_img_no(self):
        assert roi_name("S", 1234, 1, 2) == "S_img-no1234_y2_x1"

    def test_read_coordinates_skips_blank_lines(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("1,2,3\n\n4,5,6")  # no trailing newline, as in CAD files
        assert read_coordinates(path) == [
            {"img_no": 1, "x": 2, "y": 3},
            {"img_no": 4, "x": 5, "y": 6},
        ]

    def test_read_coordinates_ignores_bom(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_bytes(b"\xef\xbb\xbf1,2,3\n")
        assert read_coordinates(path) == [{"img_no": 1, "x": 2, "y": 3}]

    def test_read_coordinates_empty_file(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("")
        assert read_coordinates(path) == []

    @pytest.mark.parametrize("line", ["1,2", "1,2,3,4", "a,2,3"])
    def test_read_coordinates_rejects_bad_rows(self, tmp_path, line):
        path = tmp_path / "c.csv"
        path.write_text(f"1,2,3\n{line}\n")
        with pytest.raises(ValueError, match=":2:"):
            read_coordinates(path)

    def test_scan_class_layout_maps_stems_to_dirs(self, tmp_path):
        _touch_png(tmp_path / "abnormal" / "a.png")
        _touch_png(tmp_path / "normal" / "red" / "b.png")
        _touch_png(tmp_path / "normal" / "blue" / "b.png")
        assert scan_class_layout(str(tmp_path)) == {
            "a": ["abnormal"],
            "b": ["normal/blue", "normal/red"],
        }

    def test_scan_class_layout_matches_extension_case_insensitively(self, tmp_path):
        _touch_png(tmp_path / "abnormal" / "a.PNG")
        (tmp_path / "abnormal" / "b.txt").write_text("x")
        assert scan_class_layout(str(tmp_path)) == {"a": ["abnormal"]}

    def test_scan_class_layout_rejects_top_level_images(self, tmp_path):
        _touch_png(tmp_path / "abnormal" / "a.png")
        _touch_png(tmp_path / "stray.png")
        with pytest.raises(ValueError, match="outside any class directory"):
            scan_class_layout(str(tmp_path))

    def test_scan_class_layout_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_class_layout(str(tmp_path / "missing"))

    def test_scan_class_layout_without_images(self, tmp_path):
        with pytest.raises(ValueError, match="No PNG"):
            scan_class_layout(str(tmp_path))


@pytest.mark.unit
class TestBuildCatalog:
    def test_cells_in_sample_source_row_order(self, roi_inputs):
        catalog = build_catalog(roi_inputs["data"]["coordinates_root"], SOURCES, None)
        cells = catalog["cells"]
        assert [(c["sample"], c["source"], c["img_no"]) for c in cells] == [
            ("S 1", "ctcs", 1),
            ("S 1", "ctcs", 2),
            ("S 1", "normal_cells", 1),
            ("S 1", "normal_cells", 3),
            ("S 2", "normal_cells", 1),
        ]
        assert cells[0] == {
            "sample": "S 1",
            "img_no": 1,
            "x": 5,
            "y": 6,
            "source": "ctcs",
            "roi_name": "S-1_img-no001_y6_x5",
            "class_paths": [],
        }

    def test_metadata_without_layout(self, roi_inputs):
        meta = build_catalog(roi_inputs["data"]["coordinates_root"], SOURCES, None)[
            "metadata"
        ]
        assert meta["cell_counts"] == {"ctcs": 2, "normal_cells": 3}
        assert meta["class_counts"] == {}
        assert meta["unassigned_counts"] is None
        assert sorted(meta["source_files_sha256"]) == [
            "S 1/groundTruthCTCs.csv",
            "S 1/normalCells.csv",
            "S 2/groundTruthCTCs.csv",
            "S 2/normalCells.csv",
        ]
        assert all(len(h) == 64 for h in meta["source_files_sha256"].values())

    def test_class_paths_from_layout(self, roi_inputs, tmp_path):
        layout = tmp_path / "layout"
        _touch_png(layout / "abnormal" / "S-1_img-no001_y6_x5.png")
        _touch_png(layout / "normal" / "blue" / "S-2_img-no001_y7_x7.png")
        _touch_png(layout / "normal" / "red" / "S-2_img-no001_y7_x7.png")
        catalog = build_catalog(
            roi_inputs["data"]["coordinates_root"], SOURCES, str(layout)
        )
        by_name = {c["roi_name"]: c["class_paths"] for c in catalog["cells"]}
        assert by_name["S-1_img-no001_y6_x5"] == ["abnormal"]
        assert by_name["S-2_img-no001_y7_x7"] == ["normal/blue", "normal/red"]
        assert by_name["S-1_img-no002_y0_x0"] == []
        meta = catalog["metadata"]
        assert meta["class_counts"] == {
            "abnormal": 1,
            "normal/blue": 1,
            "normal/red": 1,
        }
        assert meta["unassigned_counts"] == {"ctcs": 1, "normal_cells": 2}

    def test_layout_image_without_cell_raises(self, roi_inputs, tmp_path):
        _touch_png(tmp_path / "layout" / "abnormal" / "unknown.png")
        with pytest.raises(ValueError, match="match no cell"):
            build_catalog(
                roi_inputs["data"]["coordinates_root"],
                SOURCES,
                str(tmp_path / "layout"),
            )

    def test_duplicate_roi_name_raises(self, roi_inputs):
        with pytest.raises(ValueError, match="unique"):
            build_catalog(
                roi_inputs["data"]["coordinates_root"],
                {"a": "normalCells.csv", "b": "normalCells.csv"},
                None,
            )

    def test_missing_csv_raises(self, roi_inputs):
        with pytest.raises(FileNotFoundError, match="missing.csv"):
            build_catalog(
                roi_inputs["data"]["coordinates_root"], {"x": "missing.csv"}, None
            )

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_catalog(str(tmp_path / "missing"), SOURCES, None)

    def test_root_without_samples_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No sample"):
            build_catalog(str(tmp_path), SOURCES, None)


@pytest.mark.unit
class TestRunCatalog:
    def test_writes_json_and_refuses_overwrite(self, roi_inputs):
        catalog = run_catalog(roi_inputs)
        with open(roi_inputs["data"]["catalog"]) as f:
            assert json.load(f)["cells"] == catalog["cells"]
        with pytest.raises(FileExistsError):
            run_catalog(roi_inputs)
        roi_inputs["output"]["force"] = True
        run_catalog(roi_inputs)

    def test_catalog_inside_an_input_raises(self, roi_inputs):
        coords = roi_inputs["data"]["coordinates_root"]
        roi_inputs["data"]["catalog"] = f"{coords}/cells.json"
        with pytest.raises(ValueError, match="overlaps data.coordinates_root"):
            run_catalog(roi_inputs)
