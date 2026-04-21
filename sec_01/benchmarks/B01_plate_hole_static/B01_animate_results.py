"""Generate B01 timeseries frames and animation outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sec_01.shared.animation_encode import encode_gif_from_frames, encode_mp4_with_ffmpeg
from sec_01.shared.visualization import ensure_animation_dirs, load_result_jsons

LOGGER = logging.getLogger(__name__)

# Fixed color scale in units of remote_stress.
# Kirsch solution ranges from -1 * sigma_inf (theta=0) to 3 * sigma_inf (theta=pi/2).
_VMIN = -1.0
_VMAX = 3.0

# Display order and subtitle labels for each method panel.
_METHOD_ORDER = ["FDM", "FEM", "BEM"]
_METHOD_SUBTITLES = {
    "FDM": "FDM  (stair-step boundary)",
    "FEM": "FEM  (smooth mesh + noise)",
    "BEM": "BEM  (exact boundary)",
}


def _load_vis_data(ts: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract visualization arrays from a viz_timeseries payload.

    Args:
        ts: viz_timeseries dictionary from one method's JSON result.

    Returns:
        Tuple of (vis_x, vis_y, sigma_2d_base, hole_mask, load_factors) where
        sigma_2d_base has shape (vis_ny, vis_nx) and hole_mask is boolean with
        the same shape (True = inside hole, should not be displayed).
    """

    vis_nx = int(ts["vis_nx"])
    vis_ny = int(ts["vis_ny"])
    vis_x = np.asarray(ts["vis_x"], dtype=float)
    vis_y = np.asarray(ts["vis_y"], dtype=float)
    sigma_flat = np.asarray(ts["sigma_2d_base"], dtype=float)
    hole_flat = np.asarray(ts["hole_mask_flat"], dtype=bool)
    load_factors = np.asarray(ts["load_factors"], dtype=float)

    sigma_2d = sigma_flat.reshape(vis_ny, vis_nx)
    hole_mask = hole_flat.reshape(vis_ny, vis_nx)
    return vis_x, vis_y, sigma_2d, hole_mask, load_factors


def main() -> None:
    """Render B01 frame sequence and encode GIF/MP4.

    Each frame shows a 1x3 grid of pcolormesh panels — one per method —
    displaying the 2-D circumferential stress field σ_θθ normalized by σ_∞.
    The hole interior is masked white. Load is ramped linearly from 0.05 to 1.0
    over 40 frames so the stress growth is visible.
    """

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B01"
    animation_dir, frames_dir = ensure_animation_dirs(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B01")

    # Verify all expected methods are present.
    missing = [m for m in _METHOD_ORDER if m not in result_map]
    if missing:
        LOGGER.warning("B01 animate: missing methods %s — skipping animation.", missing)
        return

    # Load visualization data for each method once.
    vis_data: dict[str, tuple] = {}
    for method in _METHOD_ORDER:
        ts = result_map[method].get("metadata", {}).get("viz_timeseries", {})
        vis_data[method] = _load_vis_data(ts)

    hole_radius = float(result_map["BEM"].get("metadata", {}).get("hole_radius", 0.15))
    load_factors = vis_data["FDM"][4]
    frame_count = len(load_factors)

    frames: list[Path] = []
    for frame_idx in range(frame_count):
        lf = float(load_factors[frame_idx])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
        fig.subplots_adjust(left=0.04, right=0.88, top=0.88, bottom=0.08, wspace=0.12)

        pcm_last = None
        for col, method in enumerate(_METHOD_ORDER):
            ax = axes[col]
            vis_x, vis_y, sigma_2d_base, hole_mask, _ = vis_data[method]

            # Scale base field by current load factor.
            sigma_frame = lf * sigma_2d_base

            # Mask the hole interior so it renders as the bad-color (white).
            sigma_masked = np.where(hole_mask, np.nan, sigma_frame)

            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad(color="white")

            pcm = ax.pcolormesh(
                vis_x,
                vis_y,
                sigma_masked,
                cmap=cmap,
                vmin=_VMIN,
                vmax=_VMAX,
                shading="auto",
            )
            pcm_last = pcm

            # Draw a crisp white fill and black outline for the hole.
            if method == "FDM":
                # For FDM the stair-step hole pattern is visible through the masked
                # pcolormesh cells. Draw only a dashed reference circle to indicate
                # the true circular hole boundary without hiding the staircasing.
                ax.add_patch(
                    mpatches.Circle(
                        (0.0, 0.0),
                        hole_radius,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=0.9,
                        linestyle="--",
                        zorder=4,
                    )
                )
            else:
                # FEM and BEM use a smooth circular mask; draw a solid white fill
                # with a crisp black outline.
                ax.add_patch(
                    mpatches.Circle(
                        (0.0, 0.0),
                        hole_radius,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=3,
                    )
                )

            # Tension arrows on left and right edges.
            arrow_kw = dict(
                arrowstyle="->",
                color="black",
                lw=1.2,
                mutation_scale=10,
            )
            plate_half_w = float(vis_x[-1])
            ax.annotate(
                "",
                xy=(plate_half_w + 0.12, 0.0),
                xytext=(plate_half_w - 0.05, 0.0),
                arrowprops=arrow_kw,
            )
            ax.annotate(
                "",
                xy=(-plate_half_w - 0.12, 0.0),
                xytext=(-plate_half_w + 0.05, 0.0),
                arrowprops=arrow_kw,
            )
            ax.text(plate_half_w + 0.14, 0.0, "σ∞", va="center", ha="left", fontsize=7)
            ax.text(-plate_half_w - 0.14, 0.0, "σ∞", va="center", ha="right", fontsize=7)

            ax.set_aspect("equal")
            ax.set_xlim(float(vis_x[0]) - 0.18, float(vis_x[-1]) + 0.18)
            ax.set_ylim(float(vis_y[0]) - 0.04, float(vis_y[-1]) + 0.04)
            ax.set_title(_METHOD_SUBTITLES[method], fontsize=9, pad=4)
            ax.set_xlabel("x [m]", fontsize=8)
            if col == 0:
                ax.set_ylabel("y [m]", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            f"B01  Plate Stress Field  σ_θθ / σ∞   (load factor = {lf:.2f})",
            fontsize=11,
            y=0.97,
        )

        # Shared colorbar on the right.
        if pcm_last is not None:
            cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.72])
            cbar = fig.colorbar(pcm_last, cax=cbar_ax)
            cbar.set_label("σ_θθ / σ∞", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        frame_path = frames_dir / f"B01_{frame_idx:04d}.png"
        fig.savefig(frame_path, bbox_inches="tight")
        plt.close(fig)
        frames.append(frame_path)

    gif_path = animation_dir / "B01_timeseries.gif"
    encode_gif_from_frames(frames=frames, output_path=gif_path, fps=10)

    mp4_path = animation_dir / "B01_timeseries.mp4"
    encode_mp4_with_ffmpeg(
        frames_pattern=str(frames_dir / "B01_%04d.png"),
        output_path=mp4_path,
        fps=10,
    )


if __name__ == "__main__":
    main()
