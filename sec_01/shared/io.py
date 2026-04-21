"""YAML/JSON configuration readers for benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML file into a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed dictionary.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError:
        LOGGER.exception("Failed to open YAML file: %s", path)
        raise

    return data or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload to file.

    Args:
        path: Output file path.
        payload: JSON-serializable payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError:
        LOGGER.exception("Failed to write JSON file: %s", path)
        raise
