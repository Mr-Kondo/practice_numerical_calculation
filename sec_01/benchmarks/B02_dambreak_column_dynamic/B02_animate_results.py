"""Generate B02 timeseries frames and animation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sec_01.shared.animation_encode import encode_gif_from_frames, encode_mp4_with_ffmpeg
from sec_01.shared.visualization import ensure_animation_dirs, load_result_jsons


def _frame_count(result_map: dict[str, dict]) -> int:
    counts: list[int] = []
    for method in ("FVM", "FEM", "SPH", "DEM"):
        ts = result_map.get(method, {}).get("metadata", {}).get("viz_timeseries", {})
        series_len = 0
        if method == "FVM":
            series_len = len(ts.get("height_centerline_series", []))
        elif method == "FEM":
            series_len = len(ts.get("min_quality_series", []))
        elif method == "SPH":
            series_len = len(ts.get("particle_x_series", []))
        elif method == "DEM":
            series_len = len(ts.get("point_x_series", []))
        if series_len:
            counts.append(series_len)
    return min(counts) if counts else 0


def main() -> None:
    """Render B02 frame sequence and encode GIF/MP4."""

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B02"
    animation_dir, frames_dir = ensure_animation_dirs(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B02")

    frame_count = _frame_count(result_map)

    # --- Pre-compute fixed axis limits for each subplot ---
    # FVM: x-axis is x_index list; height values are normalised 0-1.
    fvm_ts = result_map.get("FVM", {}).get("metadata", {}).get("viz_timeseries", {})
    x_idx = fvm_ts.get("x_index", [])
    h_series = fvm_ts.get("height_centerline_series", [])
    fvm_xlim = (0, max(x_idx) if x_idx else frame_count - 1)
    fvm_ylim = (0.0, 1.05)

    # FEM: cumulative history plotted up to current frame.
    # min_quality can go below zero when the mesh collapses, so derive ylim
    # from the actual series data rather than assuming a fixed [0, 1] range.
    fem_ts = result_map.get("FEM", {}).get("metadata", {}).get("viz_timeseries", {})
    mq_series = fem_ts.get("min_quality_series", [])
    cr_series = fem_ts.get("collapsed_ratio_series", [])
    fem_xlim = (0, frame_count - 1)
    all_fem_vals = mq_series + cr_series
    fem_ymin = min(0.0, float(min(all_fem_vals)) * 1.1) if all_fem_vals else 0.0
    fem_ylim = (fem_ymin, 1.05)

    # SPH: scan all frames to find global bounding box; add 5% margin.
    sph_ts = result_map.get("SPH", {}).get("metadata", {}).get("viz_timeseries", {})
    px_series = sph_ts.get("particle_x_series", [])
    py_series = sph_ts.get("particle_y_series", [])
    if px_series and py_series:
        all_px = [v for frame_data in px_series for v in frame_data]
        all_py = [v for frame_data in py_series for v in frame_data]
        px_span = max(all_px) - min(all_px) if all_px else 1.0
        py_span = max(all_py) - min(all_py) if all_py else 1.0
        sph_xlim = (min(all_px) - px_span * 0.05, max(all_px) + px_span * 0.05)
        sph_ylim = (min(all_py) - py_span * 0.05, max(all_py) + py_span * 0.05)
    else:
        sph_xlim = (0.0, 1.0)
        sph_ylim = (0.0, 1.0)

    # DEM: broken bond count is monotonically non-decreasing.
    dem_ts = result_map.get("DEM", {}).get("metadata", {}).get("viz_timeseries", {})
    broken_series = dem_ts.get("broken_bonds_series", [])
    dem_xlim = (0, frame_count - 1)
    dem_ymax = max(broken_series) * 1.1 + 1 if broken_series else 1.0
    dem_ylim = (0.0, dem_ymax)

    frames: list[Path] = []

    for frame_idx in range(frame_count):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)

        if x_idx and h_series:
            axes[0, 0].plot(x_idx, h_series[frame_idx], color="#4C78A8")
        axes[0, 0].set_title("FVM centerline")
        axes[0, 0].set_xlim(*fvm_xlim)
        axes[0, 0].set_ylim(*fvm_ylim)
        axes[0, 0].grid(alpha=0.2)

        if mq_series:
            axes[0, 1].plot(mq_series[: frame_idx + 1], label="min_quality", color="#E45756")
        if cr_series:
            axes[0, 1].plot(cr_series[: frame_idx + 1], label="collapsed_ratio", color="#72B7B2")
        axes[0, 1].set_title("FEM quality history")
        axes[0, 1].set_xlim(*fem_xlim)
        axes[0, 1].set_ylim(*fem_ylim)
        axes[0, 1].grid(alpha=0.2)
        axes[0, 1].legend()

        if px_series and py_series:
            axes[1, 0].scatter(px_series[frame_idx], py_series[frame_idx], s=7, alpha=0.45, color="#F58518")
        axes[1, 0].set_title("SPH particles")
        axes[1, 0].set_xlim(*sph_xlim)
        axes[1, 0].set_ylim(*sph_ylim)
        axes[1, 0].grid(alpha=0.2)

        if broken_series:
            axes[1, 1].plot(broken_series[: frame_idx + 1], color="#54A24B")
        axes[1, 1].set_title("DEM broken bonds")
        axes[1, 1].set_xlim(*dem_xlim)
        axes[1, 1].set_ylim(*dem_ylim)
        axes[1, 1].grid(alpha=0.2)

        fig.suptitle(f"B02 Dynamics frame={frame_idx}")
        fig.tight_layout()

        frame_path = frames_dir / f"B02_{frame_idx:04d}.png"
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    gif_path = animation_dir / "B02_timeseries.gif"
    encode_gif_from_frames(frames=frames, output_path=gif_path, fps=10)

    mp4_path = animation_dir / "B02_timeseries.mp4"
    encode_mp4_with_ffmpeg(
        frames_pattern=str(frames_dir / "B02_%04d.png"),
        output_path=mp4_path,
        fps=10,
    )


if __name__ == "__main__":
    main()
