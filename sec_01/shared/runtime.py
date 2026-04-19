"""Runtime helpers for deterministic benchmark execution."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Container for one method execution result.

    Args:
        benchmark: Benchmark identifier.
        method: Method identifier.
        metrics: Scalar metrics for comparison.
        metadata: Extra run information.
    """

    benchmark: str
    method: str
    metrics: dict[str, float]
    metadata: dict[str, Any]


class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed wall time in seconds."""
        return time.perf_counter() - self._start


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-wide logging.

    Args:
        level: Logging level.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    """Seed deterministic random generators.

    Args:
        seed: Random seed.
    """

    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist.

    Args:
        path: Target directory.
    """

    path.mkdir(parents=True, exist_ok=True)


def save_result(result: MethodResult, output_dir: Path) -> Path:
    """Serialize one method result to JSON.

    Args:
        result: Method result object.
        output_dir: Directory where JSON will be saved.

    Returns:
        Path to written file.
    """

    ensure_dir(output_dir)
    file_path = output_dir / f"{result.benchmark}_{result.method}_result.json"
    try:
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    except OSError:
        LOGGER.exception("Failed to save result: %s", file_path)
        raise
    return file_path


def save_metrics_table(results: list[MethodResult], csv_path: Path) -> Path:
    """Save comparison metrics as CSV.

    Args:
        results: Collection of method results.
        csv_path: Destination CSV path.

    Returns:
        Written CSV path.
    """

    ensure_dir(csv_path.parent)

    all_keys: list[str] = sorted({key for item in results for key in item.metrics})
    header = ["benchmark", "method", *all_keys]
    lines = [",".join(header)]

    for item in results:
        row = [item.benchmark, item.method]
        for key in all_keys:
            row.append(f"{item.metrics.get(key, float('nan')):.8g}")
        lines.append(",".join(row))

    try:
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except OSError:
        LOGGER.exception("Failed to save metrics CSV: %s", csv_path)
        raise

    return csv_path
