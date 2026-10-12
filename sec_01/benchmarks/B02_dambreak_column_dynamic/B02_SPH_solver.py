"""B02 SPH solver in 2-D.

This module implements a minimum viable 2-D weakly-compressible SPH model
for the B02
dam-break benchmark using:
  - cell-linked neighbour search,
  - cubic-spline kernel interactions,
  - Tait equation of state pressure,
  - Monaghan-style artificial viscosity,
  - reflective wall boundaries.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _build_cell_linked_list(
    positions: np.ndarray, cell_size: float, active: np.ndarray
) -> dict[tuple[int, int], list[int]]:
    """Build cell-linked list for neighbour search."""

    cell_map: dict[tuple[int, int], list[int]] = {}
    for idx in np.flatnonzero(active):
        pos = positions[idx]
        cx = int(np.floor(pos[0] / cell_size))
        cy = int(np.floor(pos[1] / cell_size))
        cell_map.setdefault((cx, cy), []).append(idx)
    return cell_map


def _neighbour_candidates(cell_map: dict[tuple[int, int], list[int]], pos: np.ndarray, cell_size: float) -> list[int]:
    """Return neighbour candidate indices from adjacent cells."""

    cx = int(np.floor(pos[0] / cell_size))
    cy = int(np.floor(pos[1] / cell_size))
    neighbours: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            neighbours.extend(cell_map.get((cx + dx, cy + dy), []))
    return neighbours


def _project_particles_to_grid(
    positions: np.ndarray,
    velocities: np.ndarray,
    mass: float,
    rho0: float,
    nx: int,
    ny: int,
    domain_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project particle data to Eulerian grid for diagnostics."""

    density = np.zeros((ny, nx), dtype=np.float64)
    vel_x = np.zeros((ny, nx), dtype=np.float64)
    vel_y = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.float64)

    cell_x = np.clip((positions[:, 0] * nx).astype(np.int32), 0, nx - 1)
    cell_y = np.clip((positions[:, 1] / domain_y * ny).astype(np.int32), 0, ny - 1)

    np.add.at(density, (cell_y, cell_x), mass)
    np.add.at(vel_x, (cell_y, cell_x), velocities[:, 0])
    np.add.at(vel_y, (cell_y, cell_x), velocities[:, 1])
    np.add.at(counts, (cell_y, cell_x), 1.0)

    wet = counts > 0.0
    vel_x[wet] /= counts[wet]
    vel_y[wet] /= counts[wet]

    cell_area = (1.0 / nx) * (domain_y / ny)
    density_ratio = density / (rho0 * cell_area)
    return density_ratio, vel_x, vel_y, counts


def _compute_vorticity(
    u: np.ndarray, v: np.ndarray, counts: np.ndarray, dx: float, dy: float, h: float
) -> np.ndarray:
    """Compute curl after count-weighted smoothing of the sparse particle grid."""

    # shortcut: grid smoothing is diagnostic only; use particle-space curl for validated vorticity.
    sigma = (max(h / dy, 1.0), max(h / dx, 1.0))
    weight = gaussian_filter(counts, sigma=sigma, mode="constant")
    divisor = np.maximum(weight, 1.0e-12)
    smooth_u = gaussian_filter(u * counts, sigma=sigma, mode="constant") / divisor
    smooth_v = gaussian_filter(v * counts, sigma=sigma, mode="constant") / divisor
    omega = np.gradient(smooth_v, dx, axis=1) - np.gradient(smooth_u, dy, axis=0)
    return np.where(weight >= 0.05, omega, 0.0)


def _stable_dt(
    dt_base: float,
    velocities: np.ndarray,
    h: float,
    c0: float,
    cfl: float,
) -> float:
    """Compute time step with acoustic and advective constraints."""

    speed = np.linalg.norm(velocities, axis=1)
    vmax = float(np.max(speed)) if speed.size else 0.0
    dt_acoustic = cfl * h / max(c0 + vmax, 1.0e-8)
    return float(min(dt_base, dt_acoustic))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run 2-D weakly-compressible SPH simulation for B02 benchmark."""

    rng = np.random.default_rng(int(config["seed"]))

    particles = int(config.get("sph_particles", 1200))
    dt_base = float(config.get("dt", 0.004))
    gravity = float(config.get("gravity", 9.81))
    cfl = float(config.get("sph_cfl", 0.25))

    h_smooth = float(config.get("sph_smoothing_length", 0.028))
    rho0 = float(config.get("sph_rho0", 1000.0))
    c0 = float(config.get("sph_c0", 30.0))
    gamma = float(config.get("sph_gamma", 7.0))
    alpha_visc = float(config.get("sph_alpha_visc", 0.06))
    beta_visc = float(config.get("sph_beta_visc", 0.0))

    domain_x = 1.0
    domain_y = float(config.get("sph_domain_y", 1.0))
    right_boundary = str(config.get("right_boundary", "reflective")).strip().lower()
    if right_boundary not in {"reflective", "open"}:
        raise ValueError(f"Unsupported B02 SPH right_boundary={right_boundary!r}")

    target_sim_time = float(config.get("target_sim_time_s", 0.60))
    max_steps = int(config.get("max_steps", 2600))
    front_quantile = float(config.get("sph_front_quantile", 0.995))
    front_target_x = float(config.get("column_x_fraction", 0.72))
    impact_x = float(config.get("impact_x", 0.95))
    rebound_window = float(config.get("rebound_window_s", 0.45))
    rebound_min_drop = float(config.get("rebound_min_drop", 0.08))
    runup_band_x = float(config.get("runup_band_x", 0.95))
    rebound_wall_band_x = float(config.get("sph_rebound_wall_band_x", 0.92))
    rebound_vx_negative_threshold = float(config.get("sph_rebound_vx_negative_threshold", -0.02))
    rebound_negative_fraction_min = float(config.get("sph_rebound_negative_fraction_min", 0.20))

    floor_restitution = float(config.get("sph_floor_restitution", 0.12))
    floor_friction = float(config.get("sph_floor_friction", 0.86))
    left_wall_restitution = float(config.get("sph_left_wall_restitution", 0.18))
    right_wall_restitution = float(config.get("sph_right_wall_restitution", 0.18))
    right_wall_restitution = min(max(right_wall_restitution, 0.0), 1.0)
    right_wall_restitution_min = float(config.get("sph_right_wall_restitution_min", 0.08))
    right_wall_restitution_min = min(max(right_wall_restitution_min, 0.0), right_wall_restitution)
    right_wall_speed_ref = max(float(config.get("sph_right_wall_speed_ref", 1.0)), 1.0e-6)
    right_wall_tangent_damping = float(config.get("sph_right_wall_tangent_damping", 0.08))
    right_wall_tangent_damping = min(max(right_wall_tangent_damping, 0.0), 0.99)
    front_speed_threshold = max(float(config.get("sph_front_speed_threshold", 0.02)), 0.0)
    front_mobile_min_particles = max(int(config.get("sph_front_mobile_min_particles", 24)), 1)
    front_wall_exclusion = max(float(config.get("sph_front_wall_exclusion", 0.01)), 0.0)

    nx = int(config.get("grid_nx", 140))
    ny = int(config.get("grid_ny", 80))
    sample_interval = max(1, int(config.get("sph_sample_interval", max_steps // 140)))
    vortex_threshold = float(config.get("sph_vorticity_threshold", 0.60))
    vortex_min_area = float(config.get("sph_vortex_min_area", 0.01))

    dam_fraction = float(config.get("dam_width_fraction", 0.25))
    dam_height = float(config.get("sph_initial_height", 0.8))
    if (
        particles <= 0
        or nx < 2
        or ny < 2
        or dt_base <= 0.0
        or cfl <= 0.0
        or h_smooth <= 0.0
        or rho0 <= 0.0
        or c0 <= 0.0
        or gamma <= 0.0
        or domain_y <= 0.0
        or not 0.0 < dam_fraction <= 1.0
        or not 0.0 < dam_height <= domain_y
    ):
        raise ValueError("Invalid B02 SPH particle, grid, time-step, material, or domain parameter")

    fluid_area = dam_fraction * dam_height
    spacing = float(np.sqrt(fluid_area / max(float(particles), 1.0)))
    nx_part = max(1, int(round(dam_fraction / spacing)))
    ny_part = max(1, int(round(dam_height / spacing)))

    particles = nx_part * ny_part
    x_part = np.linspace(spacing / 2.0, dam_fraction - spacing / 2.0, nx_part, dtype=np.float64)
    y_part = np.linspace(spacing / 2.0, dam_height - spacing / 2.0, ny_part, dtype=np.float64)
    xv, yv = np.meshgrid(x_part, y_part)
    positions = np.column_stack([xv.ravel(), yv.ravel()])
    velocities = np.zeros((particles, 2), dtype=np.float64)
    active = np.ones(particles, dtype=bool)

    particle_mass = rho0 * fluid_area / max(float(particles), 1.0)
    initial_mass = particle_mass * particles

    sampled_steps: list[int] = []
    sampled_times: list[float] = []
    front_position_series: list[float] = []
    max_speed_series: list[float] = []
    density_ratio_series: list[list[list[float]]] = []
    velocity_x_series: list[list[list[float]]] = []
    velocity_y_series: list[list[list[float]]] = []
    vorticity_series: list[list[list[float]]] = []
    vorticity_peak_series: list[float] = []
    vortex_area_series: list[float] = []
    wall_band_mean_vx_series: list[float] = []
    wall_band_negative_fraction_series: list[float] = []
    particle_x_series: list[list[float]] = []
    particle_y_series: list[list[float]] = []
    startup_diagnostics: list[dict[str, float]] = []

    sample_count = min(particles, int(config.get("sph_viz_sample_particles", 900)))
    sample_indices = rng.choice(particles, size=sample_count, replace=False)
    startup_diag_steps = max(0, int(config.get("sph_startup_diag_steps", 10)))

    simulated_time = 0.0
    max_runup_like_height = 0.0
    x_edges = np.linspace(0.0, 1.0, nx + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, domain_y, ny + 1, dtype=np.float64)
    x_coord = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_coord = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Save a true pre-step snapshot so the first visual frame is not mislabeled as t=0.
    initial_speed = np.linalg.norm(velocities, axis=1)
    density_ratio, vel_x_grid, vel_y_grid, counts = _project_particles_to_grid(
        positions=positions,
        velocities=velocities,
        mass=particle_mass,
        rho0=rho0,
        nx=nx,
        ny=ny,
        domain_y=domain_y,
    )
    omega = _compute_vorticity(vel_x_grid, vel_y_grid, counts, dx=1.0 / nx, dy=domain_y / ny, h=h_smooth)
    sampled_steps.append(-1)
    sampled_times.append(0.0)
    front_position_series.append(float(np.quantile(positions[:, 0], front_quantile)))
    max_speed_series.append(float(np.max(initial_speed) if initial_speed.size else 0.0))
    initial_wall_band = positions[:, 0] >= rebound_wall_band_x
    if np.any(initial_wall_band):
        wall_vx = velocities[initial_wall_band, 0]
        wall_band_mean_vx_series.append(float(np.mean(wall_vx)))
        wall_band_negative_fraction_series.append(float(np.mean(wall_vx <= rebound_vx_negative_threshold)))
    else:
        wall_band_mean_vx_series.append(0.0)
        wall_band_negative_fraction_series.append(0.0)
    density_ratio_series.append(density_ratio.tolist())
    velocity_x_series.append(vel_x_grid.tolist())
    velocity_y_series.append(vel_y_grid.tolist())
    vorticity_series.append(omega.tolist())
    vorticity_peak_series.append(float(np.max(np.abs(omega))))
    vortex_area_series.append(float(np.mean(np.abs(omega) >= vortex_threshold)))
    particle_x_series.append(positions[sample_indices, 0].tolist())
    particle_y_series.append(positions[sample_indices, 1].tolist())

    for step in range(max_steps):
        if simulated_time >= target_sim_time:
            break
        active_indices = np.flatnonzero(active)

        dt_step = _stable_dt(
            dt_base=dt_base,
            velocities=velocities[active],
            h=h_smooth,
            c0=c0,
            cfl=cfl,
        )
        dt_step = min(dt_step, target_sim_time - simulated_time)

        cell_map = _build_cell_linked_list(positions, cell_size=2.0 * h_smooth, active=active)
        density = np.zeros(particles, dtype=np.float64)

        sigma = 10.0 / (7.0 * np.pi * h_smooth * h_smooth)
        w0 = sigma * 1.0

        for i in active_indices:
            rho_i = particle_mass * w0
            neighbours = _neighbour_candidates(cell_map, positions[i], 2.0 * h_smooth)
            if neighbours:
                j_idx = np.array(neighbours, dtype=np.int32)
                j_idx = j_idx[j_idx != i]
                if len(j_idx) > 0:
                    r_vecs = positions[i] - positions[j_idx]
                    rs = np.linalg.norm(r_vecs, axis=1)
                    valid = rs < 2.0 * h_smooth
                    rs = rs[valid]
                    if len(rs) > 0:
                        qs = rs / h_smooth
                        w = np.zeros_like(qs)
                        mask1 = qs < 1.0
                        w[mask1] = sigma * (1.0 - 1.5 * qs[mask1] ** 2 + 0.75 * qs[mask1] ** 3)
                        mask2 = (qs >= 1.0) & (qs < 2.0)
                        w[mask2] = sigma * 0.25 * (2.0 - qs[mask2]) ** 3
                        rho_i += np.sum(particle_mass * w)
            density[i] = max(rho_i, 0.1 * rho0)

        pressure = (c0 * c0 * rho0 / gamma) * (np.power(density / rho0, gamma) - 1.0)
        pressure = np.maximum(pressure, 0.0)

        acc = np.zeros_like(velocities)
        acc[active, 1] -= gravity
        eps = 0.01 * h_smooth * h_smooth
        for i in active_indices:
            neighbours = _neighbour_candidates(cell_map, positions[i], 2.0 * h_smooth)
            if not neighbours:
                continue
            j_idx = np.array(neighbours, dtype=np.int32)
            j_idx = j_idx[j_idx != i]
            if len(j_idx) == 0:
                continue

            r_vecs = positions[i] - positions[j_idx]
            rs = np.linalg.norm(r_vecs, axis=1)
            valid = (rs > 1.0e-12) & (rs < 2.0 * h_smooth)
            if not np.any(valid):
                continue

            j_idx = j_idx[valid]
            r_vecs = r_vecs[valid]
            rs = rs[valid]

            qs = rs / h_smooth
            dwdq = np.zeros_like(qs)
            mask1 = qs < 1.0
            dwdq[mask1] = sigma * (-3.0 * qs[mask1] + 2.25 * qs[mask1] ** 2)
            mask2 = (qs >= 1.0) & (qs < 2.0)
            dwdq[mask2] = -sigma * 0.75 * (2.0 - qs[mask2]) ** 2

            grad_w = (dwdq / h_smooth)[:, np.newaxis] * (r_vecs / rs[:, np.newaxis])

            vij = velocities[i] - velocities[j_idx]
            rij_dot_vij = np.sum(r_vecs * vij, axis=1)
            rho_ij = 0.5 * (density[i] + density[j_idx])

            visc_term = np.zeros_like(rs)
            v_mask = rij_dot_vij < 0.0
            if np.any(v_mask):
                mu_ij = h_smooth * rij_dot_vij[v_mask] / (rs[v_mask] ** 2 + eps)
                rho_ij_safe = np.maximum(rho_ij[v_mask], 1.0e-12)
                visc_term[v_mask] = (-alpha_visc * c0 * mu_ij + beta_visc * mu_ij**2) / rho_ij_safe

            pressure_term = (
                pressure[i] / (density[i] * density[i])
                + pressure[j_idx] / (density[j_idx] * density[j_idx])
            )
            pair_coeff = -particle_mass * (pressure_term + visc_term)

            acc[i] += np.sum(pair_coeff[:, np.newaxis] * grad_w, axis=0)

        velocities += dt_step * acc
        positions += dt_step * velocities
        simulated_time += dt_step

        # shortcut: wall projection enforces containment; use ghost particles for pressure-accurate walls.
        below = active & (positions[:, 1] < 0.0)
        positions[below, 1] = 0.0
        velocities[below, 1] = np.abs(velocities[below, 1]) * floor_restitution
        velocities[below, 0] *= floor_friction

        above = active & (positions[:, 1] > domain_y)
        positions[above, 1] = domain_y
        velocities[above, 1] = -np.abs(velocities[above, 1]) * floor_restitution

        left = active & (positions[:, 0] < 0.0)
        positions[left, 0] = 0.0
        velocities[left, 0] = np.abs(velocities[left, 0]) * left_wall_restitution

        if right_boundary == "reflective":
            right = active & (positions[:, 0] > domain_x)
            positions[right, 0] = domain_x
            if np.any(right):
                # Use stronger damping at high impact speed to avoid overly uniform rebounds.
                impact_speed = np.abs(velocities[right, 0])
                restitution_scale = np.exp(-impact_speed / right_wall_speed_ref)
                effective_restitution = (
                    right_wall_restitution_min
                    + (right_wall_restitution - right_wall_restitution_min) * restitution_scale
                )
                velocities[right, 0] = -np.abs(velocities[right, 0]) * effective_restitution
                velocities[right, 1] *= 1.0 - right_wall_tangent_damping
        else:
            escaped = active & (positions[:, 0] > domain_x)
            active[escaped] = False
            velocities[escaped] = 0.0

        runup_particles = active & (positions[:, 0] > runup_band_x)
        if np.any(runup_particles):
            max_runup_like_height = max(
                max_runup_like_height, float(np.max(positions[runup_particles, 1]))
            )

        if step < startup_diag_steps:
            right_hits = int(np.count_nonzero(right if right_boundary == "reflective" else escaped))
            startup_diagnostics.append(
                {
                    "step": float(step),
                    "time": float(simulated_time),
                    "density_min": float(np.min(density[active_indices])) if active_indices.size else 0.0,
                    "density_max": float(np.max(density[active_indices])) if active_indices.size else 0.0,
                    "pressure_min": float(np.min(pressure[active_indices])) if active_indices.size else 0.0,
                    "pressure_max": float(np.max(pressure[active_indices])) if active_indices.size else 0.0,
                    "zero_pressure_fraction": (
                        float(np.mean(pressure[active_indices] <= 0.0)) if active_indices.size else 0.0
                    ),
                    "max_speed": float(np.max(np.linalg.norm(velocities[active], axis=1))) if np.any(active) else 0.0,
                    "max_abs_vy": float(np.max(np.abs(velocities[active, 1]))) if np.any(active) else 0.0,
                    "floor_hits": float(np.count_nonzero(below)),
                    "ceiling_hits": float(np.count_nonzero(above)),
                    "left_hits": float(np.count_nonzero(left)),
                    "right_hits": float(right_hits),
                }
            )

        should_sample = (
            (step > 0 and step % sample_interval == 0) or step == max_steps - 1 or simulated_time >= target_sim_time
        )
        if should_sample:
            # Prefer moving particles for front tracking to avoid wall-sticking tails masking rebound.
            moving_front = active & (np.abs(velocities[:, 0]) >= front_speed_threshold)
            if int(np.count_nonzero(moving_front)) >= front_mobile_min_particles:
                front_source = positions[moving_front, 0]
            else:
                front_source = positions[active, 0]

            # Exclude wall-clamped particles from front detection when possible.
            front_bulk = front_source[front_source < (domain_x - front_wall_exclusion)]
            if front_bulk.size >= front_mobile_min_particles:
                front_source = front_bulk
            front_position = float(np.quantile(front_source, front_quantile)) if front_source.size else float("nan")
            speed = np.linalg.norm(velocities[active], axis=1)
            density_ratio, vel_x_grid, vel_y_grid, counts = _project_particles_to_grid(
                positions=positions[active],
                velocities=velocities[active],
                mass=particle_mass,
                rho0=rho0,
                nx=nx,
                ny=ny,
                domain_y=domain_y,
            )
            omega = _compute_vorticity(vel_x_grid, vel_y_grid, counts, dx=1.0 / nx, dy=domain_y / ny, h=h_smooth)
            omega_peak = float(np.max(np.abs(omega)))
            vortex_area = float(np.mean(np.abs(omega) >= vortex_threshold))

            sampled_steps.append(step)
            sampled_times.append(float(simulated_time))
            front_position_series.append(front_position)
            max_speed_series.append(float(np.max(speed) if speed.size else 0.0))

            wall_band = active & (positions[:, 0] >= rebound_wall_band_x)
            if np.any(wall_band):
                wall_vx = velocities[wall_band, 0]
                wall_band_mean_vx_series.append(float(np.mean(wall_vx)))
                wall_band_negative_fraction_series.append(float(np.mean(wall_vx <= rebound_vx_negative_threshold)))
            else:
                wall_band_mean_vx_series.append(0.0)
                wall_band_negative_fraction_series.append(0.0)

            density_ratio_series.append(density_ratio.tolist())
            velocity_x_series.append(vel_x_grid.tolist())
            velocity_y_series.append(vel_y_grid.tolist())
            vorticity_series.append(omega.tolist())
            vorticity_peak_series.append(omega_peak)
            vortex_area_series.append(vortex_area)
            visible_indices = sample_indices[active[sample_indices]]
            particle_x_series.append(positions[visible_indices, 0].tolist())
            particle_y_series.append(positions[visible_indices, 1].tolist())

    front_arrival_time = float("nan")
    for front_now, t_now in zip(front_position_series, sampled_times):
        if front_now >= front_target_x:
            front_arrival_time = float(t_now)
            break

    impact_time = float("nan")
    rebound_time = float("nan")
    rebound_drop = 0.0
    rebound_velocity_negative_fraction_peak = 0.0
    rebound_velocity_mean_vx_min = 0.0
    rebound_drop_flag = 0.0
    rebound_velocity_flag = 0.0
    front_reached_flag = 0.0
    rebound_flag = 0.0
    if front_position_series and sampled_times:
        impact_idx = -1
        for idx, pos in enumerate(front_position_series):
            if pos >= impact_x:
                impact_idx = idx
                impact_time = float(sampled_times[idx])
                front_reached_flag = 1.0
                break
        if impact_idx >= 0:
            baseline = float(front_position_series[impact_idx])
            min_front = baseline
            min_idx = impact_idx
            end_t = impact_time + rebound_window
            for idx in range(impact_idx, len(front_position_series)):
                if sampled_times[idx] > end_t:
                    break
                if front_position_series[idx] < min_front:
                    min_front = float(front_position_series[idx])
                    min_idx = idx
            rebound_drop = baseline - min_front
            if rebound_drop >= rebound_min_drop:
                rebound_drop_flag = 1.0
                rebound_time = float(sampled_times[min_idx])

            neg_frac_peak = 0.0
            min_mean_vx = 0.0
            for idx in range(impact_idx, len(sampled_times)):
                if sampled_times[idx] > end_t:
                    break
                neg_frac_peak = max(neg_frac_peak, float(wall_band_negative_fraction_series[idx]))
                min_mean_vx = min(min_mean_vx, float(wall_band_mean_vx_series[idx]))

            rebound_velocity_negative_fraction_peak = neg_frac_peak
            rebound_velocity_mean_vx_min = min_mean_vx
            if neg_frac_peak >= rebound_negative_fraction_min:
                rebound_velocity_flag = 1.0

            if rebound_drop_flag > 0.5 or rebound_velocity_flag > 0.5:
                rebound_flag = 1.0
                if np.isnan(rebound_time):
                    rebound_time = float(impact_time)

    vortex_peak = max(vorticity_peak_series) if vorticity_peak_series else 0.0
    vortex_area_peak = max(vortex_area_series) if vortex_area_series else 0.0
    vortex_duration_s = 0.0
    for idx in range(1, len(sampled_times)):
        if vorticity_peak_series[idx] >= vortex_threshold and vortex_area_series[idx] >= vortex_min_area:
            vortex_duration_s += sampled_times[idx] - sampled_times[idx - 1]
    final_mass = particle_mass * int(np.count_nonzero(active))
    escaped_mass = particle_mass * int(np.count_nonzero(~active))
    mass_error = abs(final_mass + escaped_mass - initial_mass) / max(initial_mass, 1.0e-12)
    retained_mass_fraction = final_mass / max(initial_mass, 1.0e-12)
    escaped_mass_fraction = escaped_mass / max(initial_mass, 1.0e-12)
    completion_flag = float(
        simulated_time >= target_sim_time and np.isfinite(positions).all() and np.isfinite(velocities).all()
    )

    metrics = {
        "dof": float(particles * 2),
        "mass_error": float(mass_error),
        "retained_mass_fraction": float(retained_mass_fraction),
        "escaped_mass_fraction": float(escaped_mass_fraction),
        "splash_spread_width": (
            float(np.quantile(positions[active, 0], 0.95) - np.quantile(positions[active, 0], 0.05))
            if np.any(active) else 0.0
        ),
        "front_arrival_time": float(front_arrival_time),
        "max_runup_like_height": float(max_runup_like_height),
        "simulated_time_end": float(simulated_time),
        "impact_time": float(impact_time),
        "rebound_time": float(rebound_time),
        "rebound_drop": float(rebound_drop),
        "rebound_drop_flag": float(rebound_drop_flag),
        "rebound_velocity_flag": float(rebound_velocity_flag),
        "rebound_velocity_negative_fraction_peak": float(rebound_velocity_negative_fraction_peak),
        "rebound_velocity_mean_vx_min": float(rebound_velocity_mean_vx_min),
        "front_reached_flag": float(front_reached_flag),
        "rebound_flag": float(rebound_flag),
        "vorticity_peak": float(vortex_peak),
        "vortex_area_peak": float(vortex_area_peak),
        "vortex_duration_s": float(vortex_duration_s),
        "peak_particle_speed": float(np.max(np.linalg.norm(velocities[active], axis=1))) if np.any(active) else 0.0,
        "completion_flag": completion_flag,
    }

    viz_density_ratio = density_ratio_series[-1] if density_ratio_series else []
    viz_vx = velocity_x_series[-1] if velocity_x_series else []
    viz_vy = velocity_y_series[-1] if velocity_y_series else []
    viz_omega = vorticity_series[-1] if vorticity_series else []
    metadata = {
        "status": "success" if completion_flag else "incomplete",
        "backend": "numpy",
        "notes": (
            "2-D weakly-compressible SPH with cell-linked neighbour search, "
            "Tait EOS, and artificial viscosity "
            f"(right boundary={right_boundary}, "
            f"right wall restitution={right_wall_restitution:.3f}, "
            f"min restitution={right_wall_restitution_min:.3f}, "
            f"speed ref={right_wall_speed_ref:.3f}, "
            f"tangential damping={right_wall_tangent_damping:.3f}, "
            f"front speed threshold={front_speed_threshold:.3f}, "
            f"front mobile min particles={front_mobile_min_particles}, "
            f"front wall exclusion={front_wall_exclusion:.3f}, "
            f"rebound wall band x={rebound_wall_band_x:.3f}, "
            f"rebound vx threshold={rebound_vx_negative_threshold:.3f}, "
            f"rebound negative fraction min={rebound_negative_fraction_min:.3f})."
        ),
        "boundary_conditions": {
            "left": "reflective",
            "right": right_boundary,
            "bottom": "reflective",
            "top": "reflective",
        },
        "viz": {
            "representation": "particle_and_projected_grid",
            "x_coord": x_coord.tolist(),
            "y_coord": y_coord.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_shape": [ny, nx],
            "projected_density_ratio": viz_density_ratio,
            "velocity_x": viz_vx,
            "velocity_y": viz_vy,
            "vorticity": viz_omega,
            "particle_x": positions[sample_indices[active[sample_indices]], 0].tolist(),
            "particle_y": positions[sample_indices[active[sample_indices]], 1].tolist(),
            "front_position": float(front_position_series[-1]) if front_position_series else 0.0,
        },
        "viz_timeseries": {
            "representation": "particle_and_projected_grid",
            "frame_steps": sampled_steps,
            "frame_times": sampled_times,
            "simulated_time_end": float(simulated_time),
            "x_coord": x_coord.tolist(),
            "y_coord": y_coord.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_shape": [ny, nx],
            "projected_density_ratio_series": density_ratio_series,
            "velocity_x_series": velocity_x_series,
            "velocity_y_series": velocity_y_series,
            "vorticity_series": vorticity_series,
            "vorticity_peak_series": vorticity_peak_series,
            "vortex_area_series": vortex_area_series,
            "front_position_series": front_position_series,
            "particle_x_series": particle_x_series,
            "particle_y_series": particle_y_series,
            "max_speed_series": max_speed_series,
            "startup_diagnostics": startup_diagnostics,
        },
    }

    LOGGER.info(
        "B02 SPH finished: mass_error=%.4e, vorticity_peak=%.3f, rebound=%.0f",
        metrics["mass_error"],
        metrics["vorticity_peak"],
        metrics["rebound_flag"],
    )
    return MethodResult(benchmark="B02", method="SPH", metrics=metrics, metadata=metadata)
