"""ROI Extraction Configuration

Strict validation of the roi_extraction config: every key must be present,
whatever the mode, so one config file drives all three modes
(``--mode catalog``, ``--mode extract``, ``--mode render``).
"""

import re
from typing import Any, Dict

MODES = ("catalog", "extract", "render")
RENDERING_METHODS = ("pseudo_rgb8_vstretch",)
LAYOUTS = ("class_paths", "pool")
CHANNEL_KEYS = ("b", "g", "r")

_DATA_KEYS = ("coordinates_root", "image_root", "catalog", "raw16_dir")
# Pool names become directory names (rendering.layout: pool)
_POOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config key: {where}.{key}")
    return section[key]


def _require_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    section = config[key]
    if not isinstance(section, dict):
        raise ValueError(f"'{key}' must be a dictionary")
    return section


def _check_str(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")


def _check_int(value: Any, where: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")


def _check_percentile(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    if not 0 <= value <= 100:
        raise ValueError(f"{where} must be between 0 and 100, got {value}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate a roi_extraction configuration.

    Raises:
        KeyError: If a required key is missing
        ValueError: If a value is invalid
    """
    if config.get("experiment") != "roi_extraction":
        raise ValueError(
            f"Invalid experiment type: {config.get('experiment')}. "
            "Must be 'roi_extraction'"
        )

    mode = _require(config, "mode", "<root>")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")

    data = _require_section(config, "data")
    for key in _DATA_KEYS:
        _check_str(_require(data, key, "data"), f"data.{key}")
    if not data["catalog"].endswith(".json"):
        raise ValueError("data.catalog must end with '.json'")

    sources = _require_section(config, "sources")
    if not sources:
        raise ValueError("'sources' must map at least one pool name to a CSV file")
    lowered = [str(pool).lower() for pool in sources]
    if len(set(lowered)) != len(lowered):
        # case-insensitive file systems (NTFS) would merge their directories
        raise ValueError(f"sources keys must differ in more than case: {list(sources)}")
    for pool, csv_name in sources.items():
        if not isinstance(pool, str) or not _POOL_NAME.fullmatch(pool):
            raise ValueError(
                "sources keys must be letters, digits, '_' or '-' (starting with "
                f"a letter or digit), got {pool!r}"
            )
        _check_str(csv_name, f"sources.{pool}")

    catalog = _require_section(config, "catalog")
    layout_dir = _require(catalog, "class_layout_dir", "catalog")
    if layout_dir is not None:
        _check_str(layout_dir, "catalog.class_layout_dir")

    extraction = _require_section(config, "extraction")
    roi_size = _require(extraction, "roi_size", "extraction")
    _check_int(roi_size, "extraction.roi_size", 2)
    if roi_size % 2:
        # An odd window cannot be centred on the cell pixel by the original rule
        raise ValueError(f"extraction.roi_size must be even, got {roi_size}")
    _check_str(
        _require(extraction, "image_dir_glob", "extraction"),
        "extraction.image_dir_glob",
    )
    channels = _require(extraction, "channels", "extraction")
    if not isinstance(channels, dict) or set(channels) != set(CHANNEL_KEYS):
        raise ValueError(
            f"extraction.channels must map exactly {list(CHANNEL_KEYS)} to "
            f"file-name suffixes, got {channels!r}"
        )
    for key, suffix in channels.items():
        _check_str(suffix, f"extraction.channels.{key}")
    if len(set(channels.values())) != len(channels):
        raise ValueError("extraction.channels suffixes must be unique")

    rendering = _require_section(config, "rendering")
    method = _require(rendering, "method", "rendering")
    if method not in RENDERING_METHODS:
        raise ValueError(
            f"rendering.method must be one of {list(RENDERING_METHODS)}, got {method!r}"
        )
    vstretch = _require(rendering, "vstretch", "rendering")
    if vstretch is not None:
        if not isinstance(vstretch, dict):
            raise ValueError("rendering.vstretch must be a dictionary or null")
        lower = _require(vstretch, "lower_percentile", "rendering.vstretch")
        upper = _require(vstretch, "upper_percentile", "rendering.vstretch")
        _check_percentile(lower, "rendering.vstretch.lower_percentile")
        _check_percentile(upper, "rendering.vstretch.upper_percentile")
        if lower >= upper:
            raise ValueError(
                "rendering.vstretch.lower_percentile must be smaller than "
                f"upper_percentile, got {lower} >= {upper}"
            )
    layout = _require(rendering, "layout", "rendering")
    if layout not in LAYOUTS:
        raise ValueError(
            f"rendering.layout must be one of {list(LAYOUTS)}, got {layout!r}"
        )
    _check_str(_require(rendering, "output_dir", "rendering"), "rendering.output_dir")

    output = _require_section(config, "output")
    if not isinstance(_require(output, "force", "output"), bool):
        raise ValueError("output.force must be a boolean")
