"""Render B02 side-view timeseries animation with FVM and SPH."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sec_01.shared.animation_encode import encode_gif_from_frames, encode_mp4_with_ffmpeg
from sec_01.shared.io import read_yaml
from sec_01.shared.visualization import ensure_animation_dirs

LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON payload from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _series_times(ts: dict[str, Any], length: int) -> np.ndarray:
    """Return time array aligned to a series length."""

    raw = np.asarray(ts.get("frame_times", []), dtype=float)
    if raw.size >= length:
        return raw[:length]
    if raw.size > 0 and float(raw[-1]) > 0.0:
        return np.linspace(0.0, float(raw[-1]), length, dtype=float)
    return np.linspace(0.0, float(max(length - 1, 0)), length, dtype=float)


def _nearest_indices(times: np.ndarray, targets: np.ndarray, max_len: int) -> list[int]:
    """Map target times to nearest available frame indices."""

    clipped = times[:max_len]
    indices: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(clipped - target))) if clipped.size else 0
        indices.append(min(max(idx, 0), max(max_len - 1, 0)))
    return indices


def _build_target_times(
    fvm_times: np.ndarray,
    sph_times: np.ndarray,
    compare_end_s: float,
    compare_frames: int,
) -> np.ndarray:
    """Build synchronized target times shared by FVM and SPH."""

    common_end = min(float(fvm_times[-1]), float(sph_times[-1]))
    if compare_end_s > 0.0:
        common_end = min(common_end, compare_end_s)
    frame_count = compare_frames if compare_frames > 1 else int(min(fvm_times.size, sph_times.size))
    frame_count = max(frame_count, 2)
    return np.linspace(0.0, max(common_end, 1.0e-9), frame_count, dtype=float)


def _plot_event_inset(
    ax: plt.Axes,
    times: np.ndarray,
    front: np.ndarray,
    current_time: float,
    impact_time: float,
    rebound_time: float,
    end_time: float,
    y_min: float,
    y_max: float,
    color: str,
) -> None:
    """Plot front trajectory inset with impact/rebound markers."""

    n = min(times.size, front.size)
    if n <= 1:
        return

    inset = ax.inset_axes([0.56, 0.58, 0.41, 0.36])
    inset.plot(times[:n], front[:n], color=color, linewidth=1.0)
    if np.isfinite(impact_time):
        inset.axvline(impact_time, color=color, linestyle="--", linewidth=0.9)
    if np.isfinite(rebound_time):
        inset.axvline(rebound_time, color=color, linestyle=":", linewidth=0.9)
    inset.axvline(current_time, color="#2CA02C", linewidth=1.0)
    inset.set_xlim(0.0, max(end_time, 1.0e-6))
    inset.set_ylim(y_min, y_max)
    inset.set_title("x_front(t)", fontsize=8)
    inset.grid(alpha=0.2)
    inset.tick_params(labelsize=7)


def main() -> None:
    """Generate B02 side-view timeseries animation for FVM and SPH."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B02"
    animation_dir, frames_dir = ensure_animation_dirs(output_dir)

    fvm_path = output_dir / "B02_FVM_result.json"
    sph_path = output_dir / "B02_SPH_result.json"

    if not fvm_path.exists():
        raise FileNotFoundError(f"Missing FVM result: {fvm_path}")
    if not sph_path.exists():
        raise ValueError("Missing B02_SPH_result.json. Run sec01-b02 before animation.")

    fvm_payload = _load_json(fvm_path)
    sph_payload = _load_json(sph_path)

    fvm_ts = fvm_payload.get("metadata", {}).get("viz_timeseries", {})
    sph_ts = sph_payload.get("metadata", {}).get("viz_timeseries", {})

    fvm_profiles = fvm_ts.get("free_surface_y_series", []) or fvm_ts.get("height_centerline_series", [])
    fvm_x_axis = np.asarray(fvm_ts.get("x_coord", []), dtype=float)
    if not fvm_profiles:
        raise ValueError("FVM free-surface timeseries is missing in B02_FVM_result.json")
    if fvm_x_axis.size == 0:
        fvm_x_axis = np.arange(len(fvm_profiles[0]), dtype=float)

    sph_x_series = sph_ts.get("sideview_particle_x_series", []) or sph_ts.get("particle_x_series", [])
    sph_y_series = sph_ts.get("sideview_particle_y_series", []) or sph_ts.get("particle_y_series", [])
    if not sph_x_series or not sph_y_series:
        raise ValueError("SPH particle timeseries is missing in B02_SPH_result.json")

    fvm_len = len(fvm_profiles)
    sph_sideview_times = np.asarray(sph_ts.get("sideview_frame_times", []), dtype=float)
    sph_len = min(len(sph_x_series), len(sph_y_series))
    fvm_times = _series_times(fvm_ts, fvm_len)
    if sph_sideview_times.size >= sph_len and sph_len > 0:
        sph_times = sph_sideview_times[:sph_len]
    else:
        sph_times = _series_times(sph_ts, sph_len)

    compare_end_s = float(cfg.get("animation_compare_end_s", 0.60))
    compare_frames = int(cfg.get("animation_compare_frames", 180))
    fps = int(cfg.get("animation_fps", 8))

    targets = _build_target_times(
        fvm_times=fvm_times, sph_times=sph_times, compare_end_s=compare_end_s, compare_frames=compare_frames
    )
    fvm_indices = _nearest_indices(times=fvm_times, targets=targets, max_len=fvm_len)
    sph_indices = _nearest_indices(times=sph_times, targets=targets, max_len=sph_len)

    axis_x_min = float(cfg.get("plot_axis_x_min", 0.0))
    axis_x_max = float(cfg.get("plot_axis_x_max", 1.0))
    axis_y_min = float(cfg.get("plot_axis_y_min", 0.0))
    axis_y_max = float(cfg.get("plot_axis_y_max", 1.0))
    front_y_min = float(cfg.get("plot_front_y_min", 0.0))
    front_y_max = float(cfg.get("plot_front_y_max", 1.05))

    fvm_front = np.asarray(fvm_ts.get("front_position_series", []), dtype=float)
    sph_front = np.asarray(sph_ts.get("front_position_series", []), dtype=float)
    fvm_impact = float(fvm_payload.get("metrics", {}).get("impact_time", float("nan")))
    fvm_rebound = float(fvm_payload.get("metrics", {}).get("rebound_time", float("nan")))
    sph_impact = float(sph_payload.get("metrics", {}).get("impact_time", float("nan")))
    sph_rebound = float(sph_payload.get("metrics", {}).get("rebound_time", float("nan")))

    frames: list[Path] = []
    for frame_idx, target_time in enumerate(targets):
        fi = fvm_indices[frame_idx]
        si = sph_indices[frame_idx]
        fvm_time_now = float(fvm_times[fi]) if fi < fvm_times.size else float(target_time)
        sph_time_now = float(sph_times[si]) if si < sph_times.size else float(target_time)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        profile = np.asarray(fvm_profiles[fi], dtype=float)
        if profile.size != fvm_x_axis.size:
            x_axis = np.linspace(axis_x_min, axis_x_max, profile.size, dtype=float)
        else:
            x_axis = fvm_x_axis
        profile = np.clip(profile, axis_y_min, axis_y_max)

        axes[0].fill_between(x_axis, axis_y_min, profile, color="#9ECAE1", alpha=0.65)
        axes[0].plot(x_axis, profile, color="#1F4E79", linewidth=1.6)
        axes[0].axvline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[0].set_title("FVM side-view")
        axes[0].set_xlabel("x / L [-]")
        axes[0].set_ylabel("y / H [-]")
        axes[0].set_xlim(axis_x_min, axis_x_max)
        axes[0].set_ylim(axis_y_min, axis_y_max)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(alpha=0.2)

        _plot_event_inset(
            ax=axes[0],
            times=fvm_times,
            front=fvm_front,
            current_time=fvm_time_now,
            impact_time=fvm_impact,
            rebound_time=fvm_rebound,
            end_time=float(targets[-1]),
            y_min=front_y_min,
            y_max=front_y_max,
            color="#1F4E79",
        )

        sx = np.asarray(sph_x_series[si], dtype=float)
        sy = np.asarray(sph_y_series[si], dtype=float)
        axes[1].scatter(sx, sy, s=6, alpha=0.45, color="#E45756")
        axes[1].axvline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[1].set_title("SPH side-view")
        axes[1].set_xlabel("x / L [-]")
        axes[1].set_ylabel("y / H [-]")
        axes[1].set_xlim(axis_x_min, axis_x_max)
        axes[1].set_ylim(axis_y_min, axis_y_max)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(alpha=0.2)

        _plot_event_inset(
            ax=axes[1],
            times=sph_times,
            front=sph_front,
            current_time=sph_time_now,
            impact_time=sph_impact,
            rebound_time=sph_rebound,
            end_time=float(targets[-1]),
            y_min=front_y_min,
            y_max=front_y_max,
            color="#E45756",
        )

        axes[1].text(
            0.02,
            0.98,
            f"target={target_time:.3f}s\nactual={sph_time_now:.3f}s",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

        fig.suptitle(
            f"B02 timeseries side-view FVM vs SPH  target={target_time:.3f}s  fvm={fvm_time_now:.3f}s  sph={sph_time_now:.3f}s"
        )
        fig.tight_layout()

        frame_path = frames_dir / f"B02_{frame_idx:04d}.png"
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    gif_path = animation_dir / "B02_timeseries.gif"
    encode_gif_from_frames(frames=frames, output_path=gif_path, fps=fps)

    mp4_path = animation_dir / "B02_timeseries.mp4"
    encode_mp4_with_ffmpeg(
        frames_pattern=str(frames_dir / "B02_%04d.png"),
        output_path=mp4_path,
        fps=fps,
    )
    LOGGER.info("Saved %s and %s", gif_path, mp4_path)


if __name__ == "__main__":
    main()
