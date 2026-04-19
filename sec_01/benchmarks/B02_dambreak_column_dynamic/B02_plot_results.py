"""Generate matplotlib visualizations for B02 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sec_01.shared.visualization import ensure_fig_dir, load_result_jsons, save_bar_chart


def _plot_fvm(ax: plt.Axes, payload: dict) -> None:
    viz = payload.get("metadata", {}).get("viz", {})
    depth_field = np.asarray(viz.get("depth_field", []), dtype=float)
    vorticity = np.asarray(viz.get("vorticity", []), dtype=float)
    x_edges = np.asarray(viz.get("x_edges", []), dtype=float)
    y_edges = np.asarray(viz.get("y_edges", []), dtype=float)
    x_coord = np.asarray(viz.get("x_coord", []), dtype=float)
    y_coord = np.asarray(viz.get("y_coord", []), dtype=float)
    if depth_field.size and x_edges.size and y_edges.size:
        ax.pcolormesh(x_edges, y_edges, depth_field, cmap="Blues", shading="flat")
        if vorticity.size and x_coord.size and y_coord.size:
            omega_max = float(np.max(np.abs(vorticity)))
            if omega_max > 1.0e-8:
                levels = np.linspace(-omega_max, omega_max, 7)
                ax.contour(x_coord, y_coord, vorticity, levels=levels, colors="black", linewidths=0.5, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / L [-]")
        ax.set_ylabel("y / H [-]")
        ax.set_title("FVM depth + vorticity")
        return

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


def main() -> None:
    """Create B02 plots from saved method JSON outputs."""

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B02"
    fig_dir = ensure_fig_dir(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B02")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=140)
    axes_map = {
        "FVM": axes[0],
        "SPH": axes[1],
        "FEM": axes[2],
    }

    for method, ax in axes_map.items():
        payload = result_map.get(method, {})
        if method == "FVM":
            _plot_fvm(ax, payload)
        elif method == "FEM":
            _plot_fem(ax, payload)
        elif method == "SPH":
            _plot_sph(ax, payload)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_dir / "B02_method_overview.png")
    plt.close(fig)

    labels: list[str] = []
    completion_values: list[float] = []
    for method in ("FVM", "SPH", "FEM"):
        payload = result_map.get(method, {})
        if not payload:
            continue
        labels.append(method)
        completion_values.append(float(payload.get("metrics", {}).get("completion_flag", 0.0)))

    save_bar_chart(
        labels=labels,
        values=completion_values,
        output_path=fig_dir / "B02_completion_comparison.png",
        title="B02: Completion Flag by Method",
        ylabel="completion_flag",
    )

    fig_front, ax_front = plt.subplots(1, 1, figsize=(7, 4), dpi=140)
    has_series = False
    for method, color in (("FVM", "#4C78A8"), ("SPH", "#E45756")):
        payload = result_map.get(method, {})
        metrics = payload.get("metrics", {})
        ts = payload.get("metadata", {}).get("viz_timeseries", {})
        times = ts.get("frame_times", [])
        fronts = ts.get("front_position_series", [])
        if times and fronts and len(times) == len(fronts):
            ax_front.plot(times, fronts, label=method, color=color, linewidth=1.8)
            has_series = True
            arrival = float(metrics.get("front_arrival_time", float("nan")))
            if not np.isnan(arrival):
                ax_front.axvline(arrival, color=color, linestyle="--", alpha=0.5)
            impact = float(metrics.get("impact_time", float("nan")))
            rebound = float(metrics.get("rebound_time", float("nan")))
            if not np.isnan(impact):
                ax_front.axvline(impact, color=color, linestyle=":", alpha=0.6)
            if not np.isnan(rebound):
                ax_front.axvline(rebound, color=color, linestyle="-.", alpha=0.6)

    if has_series:
        ax_front.set_title("B02 front position comparison")
        ax_front.set_xlabel("time [s]")
        ax_front.set_ylabel("front position x/L [-]")
        ax_front.set_ylim(0.0, 1.05)
        ax_front.grid(alpha=0.2)
        ax_front.legend()
        fvm_pass = float(result_map.get("FVM", {}).get("metrics", {}).get("acceptance_pass", 0.0)) > 0.5
        sph_pass = float(result_map.get("SPH", {}).get("metrics", {}).get("acceptance_pass", 0.0)) > 0.5
        ax_front.text(
            0.02,
            0.98,
            f"FVM={'PASS' if fvm_pass else 'FAIL'} / SPH={'PASS' if sph_pass else 'FAIL'}",
            transform=ax_front.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    else:
        ax_front.text(0.5, 0.5, "front_position_series not available", ha="center", va="center")
        ax_front.set_axis_off()

    fig_front.tight_layout()
    fig_front.savefig(fig_dir / "B02_front_trajectory.png")
    plt.close(fig_front)

    fvm_ts = result_map.get("FVM", {}).get("metadata", {}).get("viz_timeseries", {})
    fvm_times = fvm_ts.get("frame_times", [])
    fvm_depth_series = fvm_ts.get("depth_field_series", [])
    fvm_vorticity_series = fvm_ts.get("vorticity_series", [])
    x_edges = np.asarray(fvm_ts.get("x_edges", []), dtype=float)
    y_edges = np.asarray(fvm_ts.get("y_edges", []), dtype=float)
    x_coord = np.asarray(fvm_ts.get("x_coord", []), dtype=float)
    y_coord = np.asarray(fvm_ts.get("y_coord", []), dtype=float)
    if fvm_times and fvm_depth_series and fvm_vorticity_series and x_edges.size and y_edges.size:
        target_times = [0.20, 0.40, min(float(fvm_times[-1]), 0.60)]
        fig_vortex, axes_vortex = plt.subplots(1, 3, figsize=(14, 4), dpi=140)
        for ax, target_time in zip(axes_vortex, target_times):
            idx = int(np.argmin(np.abs(np.asarray(fvm_times, dtype=float) - target_time)))
            depth_now = np.asarray(fvm_depth_series[idx], dtype=float)
            omega_now = np.asarray(fvm_vorticity_series[idx], dtype=float)
            ax.pcolormesh(x_edges, y_edges, depth_now, cmap="Blues", shading="flat")
            omega_max = float(np.max(np.abs(omega_now)))
            if omega_max > 1.0e-8 and x_coord.size and y_coord.size:
                levels = np.linspace(-omega_max, omega_max, 7)
                ax.contour(x_coord, y_coord, omega_now, levels=levels, colors="black", linewidths=0.45, alpha=0.55)
            ax.set_title(f"t={float(fvm_times[idx]):.3f}s")
            ax.set_xlabel("x / L [-]")
            ax.set_ylabel("y / H [-]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.15)

        fig_vortex.suptitle("B02 FVM vortex snapshots")
        fig_vortex.tight_layout()
        fig_vortex.savefig(fig_dir / "B02_fvm_vortex_snapshots.png")
        plt.close(fig_vortex)


if __name__ == "__main__":
    main()
