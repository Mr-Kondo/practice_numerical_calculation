"""Generate matplotlib visualizations for B02 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sec_01.shared.visualization import ensure_fig_dir, load_result_jsons, save_bar_chart


def _plot_fvm(ax: plt.Axes, payload: dict) -> None:
    viz = payload.get("metadata", {}).get("viz", {})
    x_idx = viz.get("x_index", [])
    height = viz.get("height_centerline", [])
    if x_idx and height:
        ax.plot(x_idx, height, color="#4C78A8")
    ax.set_title("FVM centerline height")


def _plot_fem(ax: plt.Axes, payload: dict) -> None:
    viz = payload.get("metadata", {}).get("viz", {})
    quality = np.array(viz.get("quality_sample", []), dtype=float)
    if quality.size:
        im = ax.imshow(quality, cmap="viridis", origin="lower")
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("FEM mesh quality sample")


def _plot_sph(ax: plt.Axes, payload: dict) -> None:
    viz = payload.get("metadata", {}).get("viz", {})
    px = viz.get("particle_x", [])
    py = viz.get("particle_y", [])
    if px and py:
        ax.scatter(px, py, s=4, alpha=0.5, color="#E45756")
    ax.set_title("SPH particles (sample)")


def _plot_dem(ax: plt.Axes, payload: dict) -> None:
    viz = payload.get("metadata", {}).get("viz", {})
    px = viz.get("point_x", [])
    py = viz.get("point_y", [])
    if px and py:
        ax.scatter(px, py, s=8, alpha=0.6, color="#72B7B2")
    ax.set_title("DEM points")


def main() -> None:
    """Create B02 plots from saved method JSON outputs."""

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B02"
    fig_dir = ensure_fig_dir(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B02")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=140)
    axes_map = {
        "FVM": axes[0, 0],
        "FEM": axes[0, 1],
        "SPH": axes[1, 0],
        "DEM": axes[1, 1],
    }

    for method, ax in axes_map.items():
        payload = result_map.get(method, {})
        if method == "FVM":
            _plot_fvm(ax, payload)
        elif method == "FEM":
            _plot_fem(ax, payload)
        elif method == "SPH":
            _plot_sph(ax, payload)
        elif method == "DEM":
            _plot_dem(ax, payload)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_dir / "B02_method_overview.png")
    plt.close(fig)

    labels: list[str] = []
    completion_values: list[float] = []
    for method, payload in sorted(result_map.items()):
        labels.append(method)
        completion_values.append(float(payload.get("metrics", {}).get("completion_flag", 0.0)))

    save_bar_chart(
        labels=labels,
        values=completion_values,
        output_path=fig_dir / "B02_completion_comparison.png",
        title="B02: Completion Flag by Method",
        ylabel="completion_flag",
    )


if __name__ == "__main__":
    main()
