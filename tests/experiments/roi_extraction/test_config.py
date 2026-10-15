"""Tests for the roi_extraction config validation."""

import copy

import pytest

from src.experiments.roi_extraction.config import validate_config


def make_config(**overrides):
    config = {
        "experiment": "roi_extraction",
        "mode": "catalog",
        "data": {
            "coordinates_root": "coords",
            "image_root": "raw",
            "catalog": "out/cells.json",
            "raw16_dir": "out/raw16",
        },
        "sources": {"ctcs": "groundTruthCTCs.csv", "normal_cells": "normalCells.csv"},
        "catalog": {"class_layout_dir": None},
        "extraction": {
            "roi_size": 4,
            "image_dir_glob": "*after*",
            "channels": {"b": "ch01", "g": "ch02", "r": "ch03"},
        },
        "rendering": {
            "method": "pseudo_rgb8_vstretch",
            "vstretch": {"lower_percentile": 50, "upper_percentile": 99},
            "layout": "class_paths",
            "output_dir": "out/rendered",
        },
        "output": {"force": False},
    }
    config.update(overrides)
    return config


_DELETE = object()


def _with(path, value):
    config = make_config()
    node = config
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    if value is _DELETE:
        del node[last]
    else:
        node[last] = value
    return config


@pytest.mark.unit
class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config(make_config())

    @pytest.mark.parametrize("mode", ["catalog", "extract", "render"])
    def test_all_modes_pass(self, mode):
        validate_config(make_config(mode=mode))

    def test_null_vstretch_and_pool_layout_pass(self):
        config = _with("rendering.vstretch", None)
        config["rendering"]["layout"] = "pool"
        validate_config(config)

    def test_class_layout_dir_string_passes(self):
        validate_config(_with("catalog.class_layout_dir", "data/in-house"))

    def test_wrong_experiment_raises(self):
        with pytest.raises(ValueError, match="roi_extraction"):
            validate_config(make_config(experiment="classifier"))

    @pytest.mark.parametrize(
        "path",
        [
            "mode",
            "data",
            "data.catalog",
            "sources",
            "catalog.class_layout_dir",
            "extraction.channels",
            "rendering.vstretch",
            "rendering.vstretch.upper_percentile",
            "output.force",
        ],
    )
    def test_missing_key_raises(self, path):
        with pytest.raises(KeyError, match=path.split(".")[-1]):
            validate_config(_with(path, _DELETE))

    @pytest.mark.parametrize(
        "path, value, match",
        [
            ("mode", "train", "mode must be one of"),
            ("data", [], "'data' must be a dictionary"),
            ("data.image_root", "", "data.image_root"),
            ("data.catalog", "cells.yaml", "end with '.json'"),
            ("sources", {}, "at least one pool"),
            ("sources", {"a/b": "x.csv"}, "sources keys"),
            ("sources", {"..": "x.csv"}, "sources keys"),
            ("sources", {"a\\b": "x.csv"}, "sources keys"),
            ("sources", {"_x": "x.csv"}, "sources keys"),
            ("sources", {"ctcs": 3}, "sources.ctcs"),
            ("catalog.class_layout_dir", "", "class_layout_dir"),
            ("extraction.roi_size", 0, "roi_size"),
            ("extraction.roi_size", 41, "even"),
            ("sources", {"ctc": "a.csv", "CTC": "b.csv"}, "more than case"),
            ("extraction.roi_size", True, "roi_size"),
            ("extraction.image_dir_glob", None, "image_dir_glob"),
            ("extraction.channels", {"b": "ch01", "g": "ch02"}, "exactly"),
            (
                "extraction.channels",
                {"b": "ch01", "g": "ch01", "r": "ch03"},
                "unique",
            ),
            ("extraction.channels", {"b": "ch01", "g": "", "r": "ch03"}, "channels.g"),
            ("rendering.method", "float", "rendering.method"),
            ("rendering.vstretch", 50, "dictionary or null"),
            ("rendering.vstretch.lower_percentile", "50", "must be a number"),
            ("rendering.vstretch.upper_percentile", 101, "between 0 and 100"),
            ("rendering.vstretch.lower_percentile", 99, "smaller than"),
            ("rendering.layout", "flat", "rendering.layout"),
            ("rendering.output_dir", "", "output_dir"),
            ("output.force", "no", "boolean"),
        ],
    )
    def test_invalid_value_raises(self, path, value, match):
        with pytest.raises(ValueError, match=match):
            validate_config(_with(path, copy.deepcopy(value)))
