"""B02 FVM solver using 2-D shallow-water equations.

The solver advances the conservative variables ``h``, ``hu``, and ``hv``
on a Cartesian grid with first-order HLL fluxes in both x and y directions.
The initial water column is localised in both directions so that wall impact
can generate transverse shear and recirculation-like vorticity structures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)

DRY_TOL = 1.0e-8


def _safe_velocity(h: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Compute velocity from momentum while masking dry cells."""

    velocity = np.zeros_like(momentum)
    wet = h > DRY_TOL
    velocity[wet] = momentum[wet] / h[wet]
    return velocity


def _hll_flux_x(
    h_left: np.ndarray,
    hu_left: np.ndarray,
    hv_left: np.ndarray,
    h_right: np.ndarray,
    hu_right: np.ndarray,
    hv_right: np.ndarray,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute HLL fluxes across x-normal interfaces."""

    u_left = _safe_velocity(h_left, hu_left)
    v_left = _safe_velocity(h_left, hv_left)
    u_right = _safe_velocity(h_right, hu_right)
    v_right = _safe_velocity(h_right, hv_right)

    c_left = np.sqrt(gravity * np.maximum(h_left, 0.0))
    c_right = np.sqrt(gravity * np.maximum(h_right, 0.0))
    s_left = np.minimum(u_left - c_left, u_right - c_right)
    s_right = np.maximum(u_left + c_left, u_right + c_right)

    flux_h_left = hu_left
    flux_hu_left = hu_left * u_left + 0.5 * gravity * h_left * h_left
    flux_hv_left = hu_left * v_left

    flux_h_right = hu_right
    flux_hu_right = hu_right * u_right + 0.5 * gravity * h_right * h_right
    flux_hv_right = hu_right * v_right

    denom = np.maximum(s_right - s_left, 1.0e-12)

    flux_h = np.where(
        s_left >= 0.0,
        flux_h_left,
        np.where(
            s_right <= 0.0,
            flux_h_right,
            (s_right * flux_h_left - s_left * flux_h_right + s_left * s_right * (h_right - h_left)) / denom,
        ),
    )
    flux_hu = np.where(
        s_left >= 0.0,
        flux_hu_left,
        np.where(
            s_right <= 0.0,
            flux_hu_right,
            (s_right * flux_hu_left - s_left * flux_hu_right + s_left * s_right * (hu_right - hu_left)) / denom,
        ),
    )
    flux_hv = np.where(
        s_left >= 0.0,
        flux_hv_left,
        np.where(
            s_right <= 0.0,
            flux_hv_right,
            (s_right * flux_hv_left - s_left * flux_hv_right + s_left * s_right * (hv_right - hv_left)) / denom,
        ),
    )
    return flux_h, flux_hu, flux_hv


def _hll_flux_y(
    h_bottom: np.ndarray,
    hu_bottom: np.ndarray,
    hv_bottom: np.ndarray,
    h_top: np.ndarray,
    hu_top: np.ndarray,
    hv_top: np.ndarray,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute HLL fluxes across y-normal interfaces."""

    u_bottom = _safe_velocity(h_bottom, hu_bottom)
    v_bottom = _safe_velocity(h_bottom, hv_bottom)
    u_top = _safe_velocity(h_top, hu_top)
    v_top = _safe_velocity(h_top, hv_top)

    c_bottom = np.sqrt(gravity * np.maximum(h_bottom, 0.0))
    c_top = np.sqrt(gravity * np.maximum(h_top, 0.0))
    s_bottom = np.minimum(v_bottom - c_bottom, v_top - c_top)
    s_top = np.maximum(v_bottom + c_bottom, v_top + c_top)

    flux_h_bottom = hv_bottom
    flux_hu_bottom = hv_bottom * u_bottom
    flux_hv_bottom = hv_bottom * v_bottom + 0.5 * gravity * h_bottom * h_bottom

    flux_h_top = hv_top
    flux_hu_top = hv_top * u_top
    flux_hv_top = hv_top * v_top + 0.5 * gravity * h_top * h_top

    denom = np.maximum(s_top - s_bottom, 1.0e-12)

    flux_h = np.where(
        s_bottom >= 0.0,
        flux_h_bottom,
        np.where(
            s_top <= 0.0,
            flux_h_top,
            (s_top * flux_h_bottom - s_bottom * flux_h_top + s_bottom * s_top * (h_top - h_bottom)) / denom,
        ),
    )
    flux_hu = np.where(
        s_bottom >= 0.0,
        flux_hu_bottom,
        np.where(
            s_top <= 0.0,
            flux_hu_top,
            (s_top * flux_hu_bottom - s_bottom * flux_hu_top + s_bottom * s_top * (hu_top - hu_bottom)) / denom,
        ),
    )
    flux_hv = np.where(
        s_bottom >= 0.0,
        flux_hv_bottom,
        np.where(
            s_top <= 0.0,
            flux_hv_top,
            (s_top * flux_hv_bottom - s_bottom * flux_hv_top + s_bottom * s_top * (hv_top - hv_bottom)) / denom,
        ),
    )
    return flux_h, flux_hu, flux_hv


def _laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return a second-order Laplacian with mirrored edge padding."""

    padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    lap_x = (padded[1:-1, 2:] - 2.0 * field + padded[1:-1, :-2]) / max(dx * dx, 1.0e-12)
    lap_y = (padded[2:, 1:-1] - 2.0 * field + padded[:-2, 1:-1]) / max(dy * dy, 1.0e-12)
    return lap_x + lap_y


def _compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute depth-averaged vorticity omega = dv/dx - du/dy."""

    dvdx = np.gradient(v, dx, axis=1, edge_order=1)
    dudy = np.gradient(u, dy, axis=0, edge_order=1)
    return dvdx - dudy


def _front_position(
    depth: np.ndarray,
    threshold: float,
    x_edges: np.ndarray,
    percentile: float,
) -> float:
    """Return a robust front position from the column-mass percentile."""

    column_mass = np.sum(np.maximum(depth - threshold, 0.0), axis=0)
    total_mass = float(np.sum(column_mass))
    if total_mass <= 1.0e-12:
        wet_columns = np.max(depth, axis=0) > threshold
        if not np.any(wet_columns):
            return 0.0
        last_wet = int(np.where(wet_columns)[0][-1])
        return float(x_edges[last_wet + 1])

    cumulative = np.cumsum(column_mass) / total_mass
    index = int(np.searchsorted(cumulative, percentile, side="left"))
    index = min(max(index, 0), len(x_edges) - 2)
    return float(x_edges[index + 1])


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run a 2-D shallow-water finite-volume dam-break solver."""

    backend = select_backend(prefer_gpu=prefer_gpu)
    nx = int(config["grid_nx"])
    ny = int(config.get("grid_ny", 80))
    steps = int(config["steps"])
    gravity = float(config.get("gravity", 9.81))
    dt_base = float(config.get("dt", 0.004))
    target_sim_time = float(config.get("target_sim_time_s", steps * dt_base))
    max_steps = int(config.get("max_steps", max(steps, int(np.ceil(target_sim_time / 5.0e-4)))))
    cfl = float(config.get("fvm_cfl", 0.30))
    viscosity = float(config.get("fvm_viscosity", 7.5e-4))
    ambient_depth = float(config.get("fvm_ambient_depth", 1.0e-4))
    transverse_perturbation = float(config.get("fvm_transverse_perturbation", 0.10))
    vortex_threshold = float(config.get("fvm_vorticity_threshold", 0.75))
    vortex_min_duration = float(config.get("fvm_vortex_min_duration_s", 0.05))
    vortex_min_area = float(config.get("fvm_vortex_min_area", 0.01))
    runup_band_x = float(config.get("runup_band_x", 0.95))
    front_threshold = float(config.get("front_threshold", 2.5e-3))
    front_percentile = float(config.get("fvm_front_percentile", 0.995))
    front_target_x = float(config.get("column_x_fraction", 0.72))
    impact_x = float(config.get("accept_front_reach_x", 0.95))
    rebound_window = float(config.get("rebound_window_s", 0.10))
    rebound_min_drop = float(config.get("rebound_min_drop", 0.08))
    rebound_velocity_threshold = float(config.get("fvm_rebound_velocity_threshold", -0.02))
    rebound_clip_cells = max(0, int(config.get("fvm_rebound_clip_cells", 2)))
    rebound_interior_band_width_cells = max(1, int(config.get("fvm_rebound_interior_band_width_cells", 4)))
    rebound_velocity_hold_s = float(config.get("fvm_rebound_velocity_hold_s", 0.02))
    rebound_wet_mass_min = float(config.get("fvm_rebound_wet_mass_min", 0.003))
    rebound_clipped_min_drop = float(config.get("fvm_rebound_clipped_min_drop", max(rebound_min_drop, 0.02)))
    initial_x_fraction = float(config.get("dam_width_fraction", 0.25))
    initial_center_y = float(config.get("fvm_initial_column_center_y_fraction", config.get("column_y_fraction", 0.35)))
    initial_height_fraction = float(
        config.get("fvm_initial_column_height_fraction", config.get("column_height_fraction", 0.35))
    )
    initial_depth = float(config.get("fvm_initial_depth", 1.0))
    sample_interval = max(1, int(config.get("fvm_sample_interval", max_steps // 120)))
    right_boundary = str(config.get("right_boundary", "reflective")).strip().lower()
    if right_boundary not in {"reflective", "open"}:
        raise ValueError(f"Unsupported B02 FVM right_boundary={right_boundary!r}")

    dx = 1.0 / nx
    dy = 1.0 / ny
    x_edges = np.linspace(0.0, 1.0, nx + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, ny + 1, dtype=np.float64)
    x_coord = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_coord = 0.5 * (y_edges[:-1] + y_edges[1:])
    clip_cells_effective = min(rebound_clip_cells, max(nx - 1, 0))
    clipped_nx = max(nx - clip_cells_effective, 1)
    clipped_x_edges = x_edges[: clipped_nx + 1]
    inner_band_end = clipped_nx
    inner_band_start = max(0, inner_band_end - rebound_interior_band_width_cells)
    xx, yy = np.meshgrid(x_coord, y_coord)

    half_span_y = 0.5 * initial_height_fraction
    inside_column = (xx <= initial_x_fraction) & (np.abs(yy - initial_center_y) <= half_span_y)
    h = np.full((ny, nx), ambient_depth, dtype=np.float64)
    h[inside_column] = initial_depth
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    if transverse_perturbation > 0.0:
        band_height = max(initial_height_fraction, 1.0e-3)
        transverse_profile = np.sin(np.pi * (yy - (initial_center_y - half_span_y)) / band_height)
        transverse_profile *= inside_column
        hv += h * transverse_perturbation * transverse_profile

    initial_mass = float(np.sum(h) * dx * dy)
    sampled_steps: list[int] = []
    sampled_times: list[float] = []
    front_position_series: list[float] = []
    depth_field_series: list[list[list[float]]] = []
    height_centerline_series: list[list[float]] = []
    volume_fraction_series: list[list[list[float]]] = []
    velocity_x_series: list[list[list[float]]] = []
    velocity_y_series: list[list[list[float]]] = []
    vorticity_series: list[list[list[float]]] = []
    vorticity_peak_series: list[float] = []
    vortex_area_series: list[float] = []
    right_band_velocity_series: list[float] = []
    clipped_front_series: list[float] = []
    inner_band_velocity_series: list[float] = []
    inner_band_wet_mass_series: list[float] = []

    simulated_time = 0.0
    max_runup_like_height = 0.0

    for step in range(max_steps):
        if simulated_time >= target_sim_time:
            break

        u = _safe_velocity(h, hu)
        v = _safe_velocity(h, hv)
        wave_speed = np.max(np.abs(u) + np.sqrt(gravity * np.maximum(h, 0.0)))
        transverse_speed = np.max(np.abs(v) + np.sqrt(gravity * np.maximum(h, 0.0)))
        dt_x = dx / max(float(wave_speed), 1.0e-8)
        dt_y = dy / max(float(transverse_speed), 1.0e-8)
        dt_step = min(dt_base, cfl * min(dt_x, dt_y), target_sim_time - simulated_time)
        dt_step = max(dt_step, 1.0e-6)

        h_left = np.empty((ny, nx + 1), dtype=np.float64)
        hu_left = np.empty((ny, nx + 1), dtype=np.float64)
        hv_left = np.empty((ny, nx + 1), dtype=np.float64)
        h_right = np.empty((ny, nx + 1), dtype=np.float64)
        hu_right = np.empty((ny, nx + 1), dtype=np.float64)
        hv_right = np.empty((ny, nx + 1), dtype=np.float64)

        h_left[:, 1:nx] = h[:, :-1]
        hu_left[:, 1:nx] = hu[:, :-1]
        hv_left[:, 1:nx] = hv[:, :-1]
        h_right[:, 1:nx] = h[:, 1:]
        hu_right[:, 1:nx] = hu[:, 1:]
        hv_right[:, 1:nx] = hv[:, 1:]

        h_left[:, 0] = h[:, 0]
        hu_left[:, 0] = -hu[:, 0]
        hv_left[:, 0] = hv[:, 0]
        h_right[:, 0] = h[:, 0]
        hu_right[:, 0] = hu[:, 0]
        hv_right[:, 0] = hv[:, 0]

        h_left[:, nx] = h[:, -1]
        hu_left[:, nx] = hu[:, -1]
        hv_left[:, nx] = hv[:, -1]
        h_right[:, nx] = h[:, -1]
        hv_right[:, nx] = hv[:, -1]
        if right_boundary == "reflective":
            hu_right[:, nx] = -hu[:, -1]
        else:
            hu_right[:, nx] = np.maximum(0.0, hu[:, -1])

        flux_h_x, flux_hu_x, flux_hv_x = _hll_flux_x(
            h_left=h_left,
            hu_left=hu_left,
            hv_left=hv_left,
            h_right=h_right,
            hu_right=hu_right,
            hv_right=hv_right,
            gravity=gravity,
        )

        h_bottom = np.empty((ny + 1, nx), dtype=np.float64)
        hu_bottom = np.empty((ny + 1, nx), dtype=np.float64)
        hv_bottom = np.empty((ny + 1, nx), dtype=np.float64)
        h_top = np.empty((ny + 1, nx), dtype=np.float64)
        hu_top = np.empty((ny + 1, nx), dtype=np.float64)
        hv_top = np.empty((ny + 1, nx), dtype=np.float64)

        h_bottom[1:ny, :] = h[:-1, :]
        hu_bottom[1:ny, :] = hu[:-1, :]
        hv_bottom[1:ny, :] = hv[:-1, :]
        h_top[1:ny, :] = h[1:, :]
        hu_top[1:ny, :] = hu[1:, :]
        hv_top[1:ny, :] = hv[1:, :]

        h_bottom[0, :] = h[0, :]
        hu_bottom[0, :] = hu[0, :]
        hv_bottom[0, :] = -hv[0, :]
        h_top[0, :] = h[0, :]
        hu_top[0, :] = hu[0, :]
        hv_top[0, :] = hv[0, :]

        h_bottom[ny, :] = h[-1, :]
        hu_bottom[ny, :] = hu[-1, :]
        hv_bottom[ny, :] = hv[-1, :]
        h_top[ny, :] = h[-1, :]
        hu_top[ny, :] = hu[-1, :]
        hv_top[ny, :] = -hv[-1, :]

        flux_h_y, flux_hu_y, flux_hv_y = _hll_flux_y(
            h_bottom=h_bottom,
            hu_bottom=hu_bottom,
            hv_bottom=hv_bottom,
            h_top=h_top,
            hu_top=hu_top,
            hv_top=hv_top,
            gravity=gravity,
        )

        h -= dt_step * ((flux_h_x[:, 1:] - flux_h_x[:, :-1]) / dx + (flux_h_y[1:, :] - flux_h_y[:-1, :]) / dy)
        hu -= dt_step * ((flux_hu_x[:, 1:] - flux_hu_x[:, :-1]) / dx + (flux_hu_y[1:, :] - flux_hu_y[:-1, :]) / dy)
        hv -= dt_step * ((flux_hv_x[:, 1:] - flux_hv_x[:, :-1]) / dx + (flux_hv_y[1:, :] - flux_hv_y[:-1, :]) / dy)

        if viscosity > 0.0:
            hu += dt_step * viscosity * _laplacian(hu, dx=dx, dy=dy)
            hv += dt_step * viscosity * _laplacian(hv, dx=dx, dy=dy)

        h = np.maximum(h, ambient_depth)
        dry_mask = h <= max(ambient_depth * 1.1, DRY_TOL)
        hu[dry_mask] = 0.0
        hv[dry_mask] = 0.0
        simulated_time += dt_step

        runup_start = int(max(0, np.floor(runup_band_x * nx)))
        max_runup_like_height = max(max_runup_like_height, float(np.max(h[:, runup_start:])))

        if step % sample_interval == 0 or step == max_steps - 1 or simulated_time >= target_sim_time:
            u = _safe_velocity(h, hu)
            v = _safe_velocity(h, hv)
            omega = _compute_vorticity(u, v, dx=dx, dy=dy)
            vortex_area = float(np.mean(np.abs(omega) >= vortex_threshold))
            right_band_u = u[:, int(np.floor(impact_x * nx)) :]
            right_band_depth = h[:, int(np.floor(impact_x * nx)) :]
            if right_band_u.size and float(np.sum(right_band_depth)) > 1.0e-12:
                right_band_velocity = float(np.sum(right_band_u * right_band_depth) / np.sum(right_band_depth))
            else:
                right_band_velocity = 0.0

            clipped_front = _front_position(
                h[:, :clipped_nx],
                threshold=front_threshold,
                x_edges=clipped_x_edges,
                percentile=front_percentile,
            )
            inner_band_depth = h[:, inner_band_start:inner_band_end]
            inner_band_u = u[:, inner_band_start:inner_band_end]
            wet_band_depth = np.where(inner_band_depth > front_threshold, inner_band_depth, 0.0)
            wet_band_sum = float(np.sum(wet_band_depth))
            inner_band_wet_mass = float(wet_band_sum * dx * dy)
            if inner_band_u.size and wet_band_sum > 1.0e-12:
                inner_band_velocity = float(np.sum(inner_band_u * wet_band_depth) / wet_band_sum)
            else:
                inner_band_velocity = 0.0

            sampled_steps.append(step)
            sampled_times.append(float(simulated_time))
            front_position_series.append(
                _front_position(
                    h,
                    threshold=front_threshold,
                    x_edges=x_edges,
                    percentile=front_percentile,
                )
            )
            depth_field_series.append(h.tolist())
            height_centerline_series.append(h[ny // 2, :].tolist())
            volume_fraction_series.append(np.clip(h / max(initial_depth, 1.0e-6), 0.0, 1.0).tolist())
            velocity_x_series.append(u.tolist())
            velocity_y_series.append(v.tolist())
            vorticity_series.append(omega.tolist())
            vorticity_peak_series.append(float(np.max(np.abs(omega))))
            vortex_area_series.append(vortex_area)
            right_band_velocity_series.append(right_band_velocity)
            clipped_front_series.append(float(clipped_front))
            inner_band_velocity_series.append(float(inner_band_velocity))
            inner_band_wet_mass_series.append(float(inner_band_wet_mass))

    final_mass = float(np.sum(h) * dx * dy)
    mass_error = abs(final_mass - initial_mass) / max(initial_mass, 1.0e-12)
    retained_mass_fraction = final_mass / max(initial_mass, 1.0e-12)
    escaped_mass_fraction = max(initial_mass - final_mass, 0.0) / max(initial_mass, 1.0e-12)
    spread_width = float(np.count_nonzero(np.max(h, axis=0) > front_threshold) * dx)

    front_arrival_time = float("nan")
    for position, time_value in zip(front_position_series, sampled_times):
        if position >= front_target_x:
            front_arrival_time = float(time_value)
            break

    impact_time = float("nan")
    rebound_time = float("nan")
    rebound_drop = 0.0
    front_reached_flag = 0.0
    rebound_flag = 0.0
    if front_position_series and sampled_times:
        impact_idx = -1
        for idx, position in enumerate(front_position_series):
            if position >= impact_x:
                impact_idx = idx
                impact_time = float(sampled_times[idx])
                front_reached_flag = 1.0
                break
        if impact_idx >= 0 and impact_idx < len(clipped_front_series):
            baseline_front = float(clipped_front_series[impact_idx])
            min_front = baseline_front
            clipped_drop_time = float("nan")
            hold_start_time = float("nan")
            hold_reached_time = float("nan")
            rebound_end = impact_time + rebound_window
            for idx in range(impact_idx, len(front_position_series)):
                if sampled_times[idx] > rebound_end:
                    break
                clipped_front_now = float(clipped_front_series[idx])
                if clipped_front_now < min_front:
                    min_front = clipped_front_now
                if np.isnan(clipped_drop_time) and baseline_front - clipped_front_now >= rebound_clipped_min_drop:
                    clipped_drop_time = float(sampled_times[idx])

                reverse_and_wet = (
                    idx < len(inner_band_velocity_series)
                    and idx < len(inner_band_wet_mass_series)
                    and inner_band_wet_mass_series[idx] >= rebound_wet_mass_min
                    and inner_band_velocity_series[idx] <= rebound_velocity_threshold
                )
                if reverse_and_wet:
                    if np.isnan(hold_start_time):
                        hold_start_time = float(sampled_times[idx])
                    if sampled_times[idx] - hold_start_time >= rebound_velocity_hold_s:
                        hold_reached_time = float(sampled_times[idx])
                        break
                else:
                    hold_start_time = float("nan")
            rebound_drop = baseline_front - min_front
            if not np.isnan(clipped_drop_time) and not np.isnan(hold_reached_time):
                rebound_flag = 1.0
                rebound_time = max(clipped_drop_time, hold_reached_time)

    vortex_peak = max(vorticity_peak_series) if vorticity_peak_series else 0.0
    vortex_area_peak = max(vortex_area_series) if vortex_area_series else 0.0
    vortex_duration_s = 0.0
    for idx in range(1, len(sampled_times)):
        if vorticity_peak_series[idx] >= vortex_threshold and vortex_area_series[idx] >= vortex_min_area:
            vortex_duration_s += sampled_times[idx] - sampled_times[idx - 1]
    vortex_pass = (
        1.0
        if vortex_peak >= vortex_threshold and vortex_area_peak >= vortex_min_area and vortex_duration_s >= vortex_min_duration
        else 0.0
    )

    acceptance_pass = 0.0
    if (
        simulated_time >= target_sim_time
        and front_reached_flag > 0.5
        and rebound_flag > 0.5
        and vortex_pass > 0.5
        and mass_error <= float(config.get("accept_mass_error_max", 5.0e-3))
        and retained_mass_fraction >= float(config.get("accept_retained_mass_min", 0.995))
        and escaped_mass_fraction <= float(config.get("accept_escaped_mass_max", 0.005))
        and max_runup_like_height >= float(config.get("accept_runup_min", 0.05))
    ):
        acceptance_pass = 1.0

    u_final = _safe_velocity(h, hu)
    v_final = _safe_velocity(h, hv)
    omega_final = _compute_vorticity(u_final, v_final, dx=dx, dy=dy)

    metrics = {
        "dof": float(nx * ny),
        "mass_error": float(mass_error),
        "retained_mass_fraction": float(retained_mass_fraction),
        "escaped_mass_fraction": float(escaped_mass_fraction),
        "splash_spread_width": spread_width,
        "front_arrival_time": front_arrival_time,
        "max_runup_like_height": float(max_runup_like_height),
        "simulated_time_end": float(simulated_time),
        "impact_time": impact_time,
        "rebound_time": rebound_time,
        "rebound_drop": float(rebound_drop),
        "front_reached_flag": front_reached_flag,
        "rebound_flag": rebound_flag,
        "vorticity_peak": float(vortex_peak),
        "vortex_area_peak": float(vortex_area_peak),
        "vortex_duration_s": float(vortex_duration_s),
        "vortex_pass": float(vortex_pass),
        "acceptance_pass": acceptance_pass,
        "completion_flag": 1.0,
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": (
            "2-D shallow-water conservative finite-volume scheme (HLL fluxes in x/y) "
            f"with viscosity={viscosity:.3e}, right boundary={right_boundary}, "
            "and exported vorticity diagnostics."
        ),
        "boundary_conditions": {
            "left": "reflective",
            "right": right_boundary,
            "bottom": "reflective",
            "top": "reflective",
        },
        "viz": {
            "representation": "depth_field_2d",
            "x_index": list(range(nx)),
            "y_index": list(range(ny)),
            "x_coord": x_coord.tolist(),
            "y_coord": y_coord.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_shape": [ny, nx],
            "height_centerline": h[ny // 2, :].tolist(),
            "depth_field": h.tolist(),
            "volume_fraction": np.clip(h / max(initial_depth, 1.0e-6), 0.0, 1.0).tolist(),
            "velocity_x": u_final.tolist(),
            "velocity_y": v_final.tolist(),
            "vorticity": omega_final.tolist(),
            "front_position": float(front_position_series[-1]) if front_position_series else 0.0,
        },
        "viz_timeseries": {
            "representation": "depth_field_2d",
            "frame_steps": sampled_steps,
            "frame_times": sampled_times,
            "simulated_time_end": float(simulated_time),
            "x_index": list(range(nx)),
            "y_index": list(range(ny)),
            "x_coord": x_coord.tolist(),
            "y_coord": y_coord.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_shape": [ny, nx],
            "depth_field_series": depth_field_series,
            "height_centerline_series": height_centerline_series,
            "free_surface_y_series": height_centerline_series,
            "volume_fraction_series": volume_fraction_series,
            "velocity_x_series": velocity_x_series,
            "velocity_y_series": velocity_y_series,
            "vorticity_series": vorticity_series,
            "vorticity_peak_series": vorticity_peak_series,
            "vortex_area_series": vortex_area_series,
            "right_band_velocity_series": right_band_velocity_series,
            "clipped_front_series": clipped_front_series,
            "inner_band_velocity_series": inner_band_velocity_series,
            "inner_band_wet_mass_series": inner_band_wet_mass_series,
            "front_position_series": front_position_series,
        },
    }

    LOGGER.info(
        "B02 FVM run finished: mass_error=%.4e, vorticity_peak=%.3f, rebound=%.0f",
        metrics["mass_error"],
        metrics["vorticity_peak"],
        metrics["rebound_flag"],
    )
    return MethodResult(benchmark="B02", method="FVM", metrics=metrics, metadata=metadata)
