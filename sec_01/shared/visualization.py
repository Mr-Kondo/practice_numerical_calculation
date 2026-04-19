"""Matplotlib helpers for benchmark result visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def ensure_fig_dir(output_dir: Path) -> Path:
    """Create and return figure output directory."""

    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def ensure_animation_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create and return animation and frames directories."""

    animation_dir = output_dir / "animations"
    frames_dir = animation_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return animation_dir, frames_dir


def load_result_jsons(output_dir: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    """Load all method JSON result files for a benchmark."""

    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob(f"{benchmark}_*_result.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded[payload["method"]] = payload
    return loaded


def save_bar_chart(
    labels: list[str],
    values: list[float],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Save a simple bar chart."""

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
