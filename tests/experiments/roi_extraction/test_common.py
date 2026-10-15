"""Tests for the helpers shared by the roi_extraction modes."""

import pytest

from src.experiments.roi_extraction.common import (
    prepare_output_dir,
    protected_paths,
    rois_sha256,
)
from tests.experiments.roi_extraction.test_config import make_config


@pytest.mark.unit
class TestPrepareOutputDir:
    def test_creates_missing_dir(self, tmp_path):
        assert prepare_output_dir(str(tmp_path / "a" / "b"), False, {}).is_dir()

    def test_refuses_non_empty_dir_unless_forced(self, tmp_path):
        (tmp_path / "out" / "sub").mkdir(parents=True)
        (tmp_path / "out" / "f.txt").write_text("x")
        with pytest.raises(FileExistsError):
            prepare_output_dir(str(tmp_path / "out"), False, {})
        out = prepare_output_dir(str(tmp_path / "out"), True, {})
        assert list(out.iterdir()) == []

    def test_symlinked_output_is_emptied_not_removed(self, tmp_path):
        target = tmp_path / "target"
        (target / "sub").mkdir(parents=True)
        (tmp_path / "link").symlink_to(target)
        prepare_output_dir(str(tmp_path / "link"), True, {})
        assert (tmp_path / "link").is_symlink()
        assert list(target.iterdir()) == []

    def test_file_output_raises(self, tmp_path):
        (tmp_path / "f").write_text("x")
        with pytest.raises(NotADirectoryError):
            prepare_output_dir(str(tmp_path / "f"), True, {})

    @pytest.mark.parametrize("output", ["in", "in/sub", "."])
    def test_overlap_raises(self, tmp_path, output):
        (tmp_path / "in").mkdir()
        with pytest.raises(ValueError, match="overlaps data.x"):
            prepare_output_dir(
                str(tmp_path / output), True, {"data.x": str(tmp_path / "in")}
            )

    def test_overlap_through_a_symlink_raises(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")
        with pytest.raises(ValueError, match="overlaps"):
            prepare_output_dir(
                str(tmp_path / "real"), True, {"data.x": str(tmp_path / "link")}
            )


@pytest.mark.unit
def test_protected_paths_exclude_only_the_output():
    config = make_config()
    config["catalog"]["class_layout_dir"] = "layout"
    paths = protected_paths(config, "data.raw16_dir")
    assert set(paths) == {
        "data.coordinates_root",
        "data.image_root",
        "data.catalog",
        "rendering.output_dir",
        "catalog.class_layout_dir",
    }
    config["catalog"]["class_layout_dir"] = None
    assert "catalog.class_layout_dir" not in protected_paths(
        config, "rendering.output_dir"
    )


@pytest.mark.unit
def test_rois_sha256_ignores_class_paths():
    cell = {"sample": "S", "img_no": 1, "x": 2, "y": 3, "roi_name": "n"}
    base = rois_sha256([{**cell, "class_paths": []}])
    assert rois_sha256([{**cell, "class_paths": ["abnormal"]}]) == base
    assert rois_sha256([{**cell, "x": 4, "class_paths": []}]) != base
    other = {**cell, "img_no": 2, "roi_name": "m"}
    assert rois_sha256([cell, other]) == rois_sha256([other, cell])
