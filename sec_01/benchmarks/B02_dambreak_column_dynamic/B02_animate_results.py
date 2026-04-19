"""Generate B02 timeseries frames and animation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sec_01.shared.animation_encode import encode_gif_from_frames, encode_mp4_with_ffmpeg
from sec_01.shared.io import read_yaml
from sec_01.shared.visualization import ensure_animation_dirs, load_result_jsons


def _series_length(ts: dict, method: str) -> int:
    if method == "FVM":
        return len(ts.get("volume_fraction_series", [])) or len(ts.get("height_centerline_series", []))
    if method == "SPH":
        return len(ts.get("particle_x_series", []))
    if method == "FEM":
        return len(ts.get("min_quality_series", []))
    return 0


def _build_frame_plan(
    result_map: dict[str, dict],
    compare_end_s: float,
    compare_frames: int,
) -> tuple[list[dict[str, int]], list[float] | None]:
    methods = ("FVM", "SPH", "FEM")
    lengths: dict[str, int] = {}
    times_map: dict[str, np.ndarray] = {}

    for method in methods:
        ts = result_map.get(method, {}).get("metadata", {}).get("viz_timeseries", {})
        series_len = _series_length(ts, method)
        if series_len <= 0:
            continue
        lengths[method] = series_len

        frame_times = ts.get("frame_times", [])
        if frame_times:
            times = np.asarray(frame_times[:series_len], dtype=float)
            if times.size:
                times_map[method] = times

    if not lengths:
        return [], None

    frame_count = min(lengths.values())
    if frame_count <= 0:
        return [], None

    if len(times_map) == len(lengths):
        if compare_end_s > 0.0 and compare_frames > 1:
            end_time = compare_end_s
            frame_count = compare_frames
        else:
            end_time = min(float(times[-1]) for times in times_map.values())
        if end_time > 0.0:
            target_times = np.linspace(0.0, end_time, frame_count, dtype=float)
            frame_plan: list[dict[str, int]] = []
            for target_time in target_times:
                frame_sel: dict[str, int] = {}
                for method in methods:
                    if method not in lengths:
                        continue
                    times = times_map[method]
                    index = int(np.argmin(np.abs(times - target_time)))
                    frame_sel[method] = min(index, lengths[method] - 1)
                frame_plan.append(frame_sel)
            return frame_plan, target_times.tolist()

    frame_plan = []
    for index in range(frame_count):
        frame_plan.append({method: index for method in methods if method in lengths})
    return frame_plan, None


def main() -> None:
    """Render B02 frame sequence and encode GIF/MP4."""

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B02"
    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)
    compare_end_s = float(cfg.get("animation_compare_end_s", 0.0))
    compare_frames = int(cfg.get("animation_compare_frames", 0))
    fps = int(cfg.get("animation_fps", 6))

    animation_dir, frames_dir = ensure_animation_dirs(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B02")

    frame_plan, target_times = _build_frame_plan(
        result_map=result_map,
        compare_end_s=compare_end_s,
        compare_frames=compare_frames,
    )
    frame_count = len(frame_plan)

    # --- Pre-compute fixed axis limits for each subplot ---
    # FVM: x-axis is physical coordinate x/L when available; fallback to index.
    fvm_ts = result_map.get("FVM", {}).get("metadata", {}).get("viz_timeseries", {})
    x_idx = fvm_ts.get("x_index", [])
    x_coord = fvm_ts.get("x_coord", [])
    x_edges = fvm_ts.get("x_edges", [])
    y_coord = fvm_ts.get("y_coord", [])
    y_edges = fvm_ts.get("y_edges", [])
    x_axis = x_coord if x_coord else x_idx
    h_series = fvm_ts.get("height_centerline_series", [])
    free_surface_series = fvm_ts.get("free_surface_y_series", h_series)
    volume_fraction_series = fvm_ts.get("volume_fraction_series", [])
    depth_field_series = fvm_ts.get("depth_field_series", [])
    velocity_x_series = fvm_ts.get("velocity_x_series", [])
    velocity_y_series = fvm_ts.get("velocity_y_series", [])
    vorticity_series = fvm_ts.get("vorticity_series", [])
    vorticity_peak_series = fvm_ts.get("vorticity_peak_series", [])
    y_floor = float(fvm_ts.get("y_floor", 0.0))
    y_top_ref = float(fvm_ts.get("y_top_ref", 1.0))
    fvm_steps = fvm_ts.get("frame_steps", [])
    fvm_times = fvm_ts.get("frame_times", [])
    fvm_front_series = fvm_ts.get("front_position_series", [])
    dt_internal = fvm_ts.get("dt_internal")
    fvm_depth_max = 1.0
    if depth_field_series:
        fvm_depth_max = max(
            1.0e-6,
            max(float(np.max(np.asarray(frame, dtype=float))) for frame in depth_field_series),
        )
    fvm_vorticity_max = 0.0
    if vorticity_series:
        fvm_vorticity_max = max(
            0.0,
            max(float(np.max(np.abs(np.asarray(frame, dtype=float)))) for frame in vorticity_series),
        )

    # FEM: prefer collapse geometry; keep quality histories for inset/fallback.
    fem_ts = result_map.get("FEM", {}).get("metadata", {}).get("viz_timeseries", {})
    mq_series = fem_ts.get("min_quality_series", [])
    cr_series = fem_ts.get("collapsed_ratio_series", [])
    fem_x_series = fem_ts.get("node_x_series", [])
    fem_y_series = fem_ts.get("node_y_series", [])
    fem_steps = fem_ts.get("frame_steps", [])
    fem_times = fem_ts.get("frame_times", [])
    all_fem_vals = mq_series + cr_series
    fem_ymin = min(0.0, float(min(all_fem_vals)) * 1.1) if all_fem_vals else 0.0
    fem_metric_xlim = (0, frame_count - 1)
    fem_metric_ylim = (fem_ymin, 1.05)
    if fem_x_series and fem_y_series:
        all_fx = [v for frame_data in fem_x_series for v in frame_data]
        all_fy = [v for frame_data in fem_y_series for v in frame_data]
        fx_span = max(all_fx) - min(all_fx) if all_fx else 1.0
        fy_span = max(all_fy) - min(all_fy) if all_fy else 1.0
        fem_data_xlim = (min(all_fx) - fx_span * 0.05, max(all_fx) + fx_span * 0.05)
        fem_data_ylim = (min(all_fy) - fy_span * 0.05, max(all_fy) + fy_span * 0.05)
    else:
        fem_data_xlim = fem_metric_xlim
        fem_data_ylim = fem_metric_ylim

    # SPH: scan all frames to find global bounding box; add 5% margin.
    sph_ts = result_map.get("SPH", {}).get("metadata", {}).get("viz_timeseries", {})
    px_series = sph_ts.get("particle_x_series", [])
    py_series = sph_ts.get("particle_y_series", [])
    sph_times = sph_ts.get("frame_times", [])
    sph_front_series = sph_ts.get("front_position_series", [])
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

    # Keep geometry panels comparable by sharing one fixed axis range.
    if px_series and py_series:
        geom_xlim = sph_xlim
        geom_ylim = sph_ylim
    elif fem_x_series and fem_y_series:
        geom_xlim = fem_data_xlim
        geom_ylim = fem_data_ylim
    else:
        geom_xlim = (0.0, 1.0)
        geom_ylim = (0.0, 1.0)

    frames: list[Path] = []

    fvm_metrics = result_map.get("FVM", {}).get("metrics", {})
    fvm_bc = result_map.get("FVM", {}).get("metadata", {}).get("boundary_conditions", {})
    fvm_impact = float(fvm_metrics.get("impact_time", float("nan")))
    fvm_rebound = float(fvm_metrics.get("rebound_time", float("nan")))
    fvm_pass = float(fvm_metrics.get("acceptance_pass", 0.0)) > 0.5

    sph_metrics = result_map.get("SPH", {}).get("metrics", {})
    sph_impact = float(sph_metrics.get("impact_time", float("nan")))
    sph_rebound = float(sph_metrics.get("rebound_time", float("nan")))
    sph_pass = float(sph_metrics.get("acceptance_pass", 0.0)) > 0.5

    for frame_idx, frame_sel in enumerate(frame_plan):
        fvm_idx = frame_sel.get("FVM", 0)
        sph_idx = frame_sel.get("SPH", 0)
        fem_idx = frame_sel.get("FEM", 0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=120)

        fvm_representation = "2-D depth field"
        if x_edges and y_edges and depth_field_series:
            depth_now = np.asarray(depth_field_series[fvm_idx], dtype=float)
            axes[0].pcolormesh(
                np.asarray(x_edges, dtype=float),
                np.asarray(y_edges, dtype=float),
                depth_now,
                cmap="Blues",
                vmin=0.0,
                vmax=fvm_depth_max,
                shading="flat",
            )
            wet_fraction = float(np.mean(depth_now > 0.01 * fvm_depth_max) * 100.0)
            if vorticity_series and fvm_idx < len(vorticity_series) and fvm_vorticity_max > 1.0e-8:
                omega_now = np.asarray(vorticity_series[fvm_idx], dtype=float)
                levels = np.linspace(-fvm_vorticity_max, fvm_vorticity_max, 7)
                axes[0].contour(
                    np.asarray(x_coord, dtype=float),
                    np.asarray(y_coord, dtype=float),
                    omega_now,
                    levels=levels,
                    colors="black",
                    linewidths=0.45,
                    alpha=0.45,
                )
            if (
                velocity_x_series
                and velocity_y_series
                and fvm_idx < len(velocity_x_series)
                and fvm_idx < len(velocity_y_series)
            ):
                u_now = np.asarray(velocity_x_series[fvm_idx], dtype=float)
                v_now = np.asarray(velocity_y_series[fvm_idx], dtype=float)
                stride_y = max(1, u_now.shape[0] // 12)
                stride_x = max(1, u_now.shape[1] // 18)
                axes[0].quiver(
                    np.asarray(x_coord, dtype=float)[::stride_x],
                    np.asarray(y_coord, dtype=float)[::stride_y],
                    u_now[::stride_y, ::stride_x],
                    v_now[::stride_y, ::stride_x],
                    color="#0B3C5D",
                    alpha=0.35,
                    scale=12.0,
                    width=0.002,
                )
        elif x_axis and free_surface_series:
            h_now = np.asarray(free_surface_series[fvm_idx], dtype=float)
            h_now = np.clip(h_now, y_floor, y_top_ref)
            axes[0].fill_between(x_axis, y_floor, h_now, color="#9ECAE1", alpha=0.65)
            axes[0].plot(x_axis, h_now, color="#1F4E79", linewidth=1.6)
            wet_fraction = float(np.mean(h_now > 0.05) * 100.0)
        else:
            wet_fraction = 0.0
        axes[0].set_title("FVM depth field / vorticity contours")
        axes[0].set_xlabel("x / L [-]")
        axes[0].set_ylabel("y / H [-]")
        axes[0].set_xlim(*geom_xlim)
        axes[0].set_ylim(*geom_ylim)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(alpha=0.2)
        fvm_front_text = ""
        if fvm_front_series and fvm_idx < len(fvm_front_series):
            fvm_front_x = float(fvm_front_series[fvm_idx])
            axes[0].axvline(fvm_front_x, color="#2CA02C", linestyle="--", linewidth=1.0, alpha=0.9)
            fvm_front_text = f"front={fvm_front_x:.3f}"
        time_text = "t=n/a"
        if fvm_times and fvm_idx < len(fvm_times):
            time_text = f"t={float(fvm_times[fvm_idx]):.3f}"
        elif dt_internal is not None and fvm_steps and fvm_idx < len(fvm_steps):
            time_text = f"t={fvm_steps[fvm_idx] * float(dt_internal):.3f}"
        boundary_text = f"right_bc={fvm_bc.get('right', 'n/a')}"
        escaped_text = f"escaped={float(fvm_metrics.get('escaped_mass_fraction', 0.0)):.3f}"
        omega_text = ""
        if vorticity_peak_series and fvm_idx < len(vorticity_peak_series):
            omega_text = f"omega_max={float(vorticity_peak_series[fvm_idx]):.2f}"
        axes[0].text(
            0.02,
            0.95,
            f"wet={wet_fraction:.1f}%\n{time_text}\n{boundary_text}\n{escaped_text}\n{omega_text}\n{fvm_representation}",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

        if px_series and py_series:
            axes[1].scatter(px_series[sph_idx], py_series[sph_idx], s=7, alpha=0.45, color="#F58518")
        axes[1].set_title("SPH particles (particle coordinates)")
        axes[1].set_xlabel("x / L [-]")
        axes[1].set_ylabel("y / H [-]")
        axes[1].set_xlim(*geom_xlim)
        axes[1].set_ylim(*geom_ylim)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(alpha=0.2)
        sph_front_text = ""
        if sph_front_series and sph_idx < len(sph_front_series):
            sph_front_x = float(sph_front_series[sph_idx])
            axes[1].axvline(sph_front_x, color="#2CA02C", linestyle="--", linewidth=1.0, alpha=0.9)
            sph_front_text = f"front={sph_front_x:.3f}"
        sph_text = ""
        if sph_times and sph_idx < len(sph_times):
            sph_text = f"t={float(sph_times[sph_idx]):.3f}"
        if sph_front_text:
            sph_text = f"{sph_text} {sph_front_text}".strip()
        if sph_text:
            axes[1].text(
                0.02,
                0.95,
                sph_text,
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

        if fem_x_series and fem_y_series:
            axes[2].scatter(fem_x_series[fem_idx], fem_y_series[fem_idx], s=9, alpha=0.5, color="#E45756")
            axes[2].set_title("FEM collapse geometry")
            axes[2].set_xlabel("x / L [-]")
            axes[2].set_ylabel("y / H [-]")
            axes[2].set_xlim(*geom_xlim)
            axes[2].set_ylim(*geom_ylim)
            axes[2].set_aspect("equal", adjustable="box")
            axes[2].grid(alpha=0.2)
            inset = axes[2].inset_axes([0.56, 0.56, 0.42, 0.4])
            if mq_series:
                inset.plot(mq_series[: fem_idx + 1], color="#E45756", linewidth=1.0)
            if cr_series:
                inset.plot(cr_series[: fem_idx + 1], color="#72B7B2", linewidth=1.0)
            inset.set_xlim(*fem_metric_xlim)
            inset.set_ylim(*fem_metric_ylim)
            inset.set_title("quality", fontsize=8)
            inset.grid(alpha=0.2)
            inset.tick_params(labelsize=7)
            quality_text = ""
            if mq_series and fem_idx < len(mq_series):
                quality_text += f"minQ={mq_series[fem_idx]:.2f} "
            if cr_series and fem_idx < len(cr_series):
                quality_text += f"collapse={cr_series[fem_idx]:.2f}"
            if fem_times and fem_idx < len(fem_times):
                quality_text = f"t={float(fem_times[fem_idx]):.3f} {quality_text}".strip()
            elif fem_steps and fem_idx < len(fem_steps):
                quality_text = f"step={fem_steps[fem_idx]} {quality_text}".strip()
            if quality_text:
                axes[2].text(
                    0.02,
                    0.95,
                    quality_text,
                    transform=axes[2].transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                )

        if target_times and frame_idx < len(target_times):
            current_t = float(target_times[frame_idx])
            event_ax = axes[2].inset_axes([0.52, 0.02, 0.46, 0.22])
            if fvm_times and fvm_front_series and len(fvm_times) == len(fvm_front_series):
                event_ax.plot(fvm_times, fvm_front_series, color="#4C78A8", linewidth=1.0, label="FVM")
            if fvm_times and vorticity_peak_series and len(fvm_times) == len(vorticity_peak_series):
                peak_scale = max(max(vorticity_peak_series), 1.0e-8)
                event_ax.plot(
                    fvm_times,
                    [value / peak_scale for value in vorticity_peak_series],
                    color="#54A24B",
                    linewidth=1.0,
                    linestyle="-.",
                    label="omega",
                )
            if sph_times and sph_front_series and len(sph_times) == len(sph_front_series):
                event_ax.plot(sph_times, sph_front_series, color="#E45756", linewidth=1.0, label="SPH")
            if not np.isnan(fvm_impact):
                event_ax.axvline(fvm_impact, color="#4C78A8", linestyle="--", alpha=0.7)
            if not np.isnan(fvm_rebound):
                event_ax.axvline(fvm_rebound, color="#4C78A8", linestyle=":", alpha=0.7)
            if not np.isnan(sph_impact):
                event_ax.axvline(sph_impact, color="#E45756", linestyle="--", alpha=0.7)
            if not np.isnan(sph_rebound):
                event_ax.axvline(sph_rebound, color="#E45756", linestyle=":", alpha=0.7)
            event_ax.axvline(current_t, color="#2CA02C", linewidth=1.0)
            event_ax.set_xlim(0.0, max(current_t, compare_end_s if compare_end_s > 0.0 else current_t + 1.0e-6))
            event_ax.set_ylim(0.0, 1.05)
            event_ax.set_title("events", fontsize=8)
            event_ax.grid(alpha=0.2)
            event_ax.tick_params(labelsize=7)
            event_ax.text(
                0.02,
                0.95,
                f"FVM={'PASS' if fvm_pass else 'FAIL'} SPH={'PASS' if sph_pass else 'FAIL'}",
                transform=event_ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        else:
            if mq_series:
                axes[2].plot(mq_series[: fem_idx + 1], label="min_quality", color="#E45756")
            if cr_series:
                axes[2].plot(cr_series[: fem_idx + 1], label="collapsed_ratio", color="#72B7B2")
            axes[2].set_title("FEM quality history")
            axes[2].set_xlabel("frame index [-]")
            axes[2].set_ylabel("quality metric [-]")
            axes[2].set_xlim(*fem_metric_xlim)
            axes[2].set_ylim(*fem_metric_ylim)
            axes[2].grid(alpha=0.2)
            axes[2].legend()

        if target_times and frame_idx < len(target_times):
            if fvm_front_text:
                fig.suptitle(f"B02 Dynamics frame={frame_idx} target_t={target_times[frame_idx]:.3f} {fvm_front_text}")
            else:
                fig.suptitle(f"B02 Dynamics frame={frame_idx} target_t={target_times[frame_idx]:.3f}")
        else:
            fig.suptitle(f"B02 Dynamics frame={frame_idx}")
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


if __name__ == "__main__":
    main()
