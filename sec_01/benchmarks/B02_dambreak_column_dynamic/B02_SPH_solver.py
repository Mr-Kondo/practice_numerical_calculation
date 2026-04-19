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

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _cubic_spline_weight(r: float, h: float) -> float:
    """Return cubic-spline kernel value in 2-D."""

    q = r / max(h, 1.0e-12)
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    if q < 2.0:
        val = 2.0 - q
        return sigma * 0.25 * val * val * val
    return 0.0


def _cubic_spline_grad(r_vec: np.ndarray, h: float) -> np.ndarray:
    """Return gradient of cubic-spline kernel in 2-D."""

    r = float(np.linalg.norm(r_vec))
    if r <= 1.0e-12:
        return np.zeros(2, dtype=np.float64)

    q = r / max(h, 1.0e-12)
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q < 1.0:
        dwdq = sigma * (-3.0 * q + 2.25 * q * q)
    elif q < 2.0:
        val = 2.0 - q
        dwdq = -sigma * 0.75 * val * val
    else:
        return np.zeros(2, dtype=np.float64)

    return (dwdq / max(h, 1.0e-12)) * (r_vec / r)


def _build_cell_linked_list(positions: np.ndarray, cell_size: float) -> dict[tuple[int, int], list[int]]:
    """Build cell-linked list for neighbour search."""

    cell_map: dict[tuple[int, int], list[int]] = {}
    for idx, pos in enumerate(positions):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project particle data to Eulerian grid for diagnostics."""

    density = np.zeros((ny, nx), dtype=np.float64)
    vel_x = np.zeros((ny, nx), dtype=np.float64)
    vel_y = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.float64)

    cell_x = np.clip((positions[:, 0] * nx).astype(np.int32), 0, nx - 1)
    cell_y = np.clip((positions[:, 1] * ny).astype(np.int32), 0, ny - 1)

    np.add.at(density, (cell_y, cell_x), mass)
    np.add.at(vel_x, (cell_y, cell_x), velocities[:, 0])
    np.add.at(vel_y, (cell_y, cell_x), velocities[:, 1])
    np.add.at(counts, (cell_y, cell_x), 1.0)

    wet = counts > 0.0
    vel_x[wet] /= counts[wet]
    vel_y[wet] /= counts[wet]

    cell_area = (1.0 / nx) * (1.0 / ny)
    depth = density / max(rho0 * cell_area, 1.0e-12)
    return depth, vel_x, vel_y, counts


def _compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute vorticity omega = dv/dx - du/dy on grid."""

    dvdx = np.gradient(v, dx, axis=1, edge_order=1)
    dudy = np.gradient(u, dy, axis=0, edge_order=1)
    return dvdx - dudy


def _stable_dt(
    dt_base: float,
    velocities: np.ndarray,
    h: float,
    c0: float,
    cfl: float,
    min_ratio: float,
) -> float:
    """Compute time step with acoustic and advective constraints."""

    speed = np.linalg.norm(velocities, axis=1)
    vmax = float(np.max(speed)) if speed.size else 0.0
    dt_adv = cfl * h / max(vmax, 1.0e-8)
    dt_acoustic = cfl * h / max(c0 + vmax, 1.0e-8)
    dt_min = max(min_ratio * dt_base, 1.0e-6)
    return float(min(dt_base, max(dt_min, min(dt_adv, dt_acoustic))))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run 2-D weakly-compressible SPH simulation for B02 benchmark."""

    backend = select_backend(prefer_gpu=prefer_gpu)
    rng = np.random.default_rng(int(config["seed"]))

    particles = int(config.get("sph_particles", 1200))
    dt_base = float(config.get("dt", 0.004))
    gravity = float(config.get("gravity", 9.81))
    cfl = float(config.get("sph_cfl", 0.25))
    dt_min_ratio = float(config.get("sph_dt_min_ratio", 0.12))

    h_smooth = float(config.get("sph_smoothing_length", 0.028))
    rho0 = float(config.get("sph_rho0", 1000.0))
    c0 = float(config.get("sph_c0", 30.0))
    gamma = float(config.get("sph_gamma", 7.0))
    alpha_visc = float(config.get("sph_alpha_visc", 0.06))
    beta_visc = float(config.get("sph_beta_visc", 0.0))

    domain_x = 1.0
    domain_y = 1.0
    right_boundary = str(config.get("right_boundary", "reflective")).strip().lower()
    if right_boundary not in {"reflective", "open"}:
        raise ValueError(f"Unsupported B02 SPH right_boundary={right_boundary!r}")

    target_sim_time = float(config.get("target_sim_time_s", 0.60))
    max_steps = int(config.get("max_steps", 2600))
    front_quantile = float(config.get("sph_front_quantile", 0.995))
    front_target_x = float(config.get("column_x_fraction", 0.72))
    impact_x = float(config.get("accept_front_reach_x", 0.95))
    rebound_window = float(config.get("rebound_window_s", 0.45))
    rebound_min_drop = float(config.get("rebound_min_drop", 0.08))
    runup_band_x = float(config.get("runup_band_x", 0.95))

    floor_restitution = float(config.get("sph_floor_restitution", 0.12))
    floor_friction = float(config.get("sph_floor_friction", 0.86))
    left_wall_restitution = float(config.get("sph_left_wall_restitution", 0.18))
    right_wall_restitution = float(config.get("sph_right_wall_restitution", 0.18))

    nx = int(config.get("grid_nx", 140))
    ny = int(config.get("grid_ny", 80))
    sample_interval = max(1, int(config.get("sph_sample_interval", max_steps // 140)))
    vortex_threshold = float(config.get("sph_vorticity_threshold", 0.60))
    vortex_min_duration = float(config.get("sph_vortex_min_duration_s", 0.05))
    vortex_min_area = float(config.get("sph_vortex_min_area", 0.01))

    dam_fraction = float(config.get("dam_width_fraction", 0.25))
    dam_height = float(config.get("sph_initial_height", 0.8))

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

    particle_mass = rho0 * fluid_area / max(float(particles), 1.0)
    initial_mass = particle_mass * particles

    sampled_steps: list[int] = []
    sampled_times: list[float] = []
    front_position_series: list[float] = []
    max_speed_series: list[float] = []
    depth_field_series: list[list[list[float]]] = []
    velocity_x_series: list[list[list[float]]] = []
    velocity_y_series: list[list[list[float]]] = []
    vorticity_series: list[list[list[float]]] = []
    vorticity_peak_series: list[float] = []
    vortex_area_series: list[float] = []
    particle_x_series: list[list[float]] = []
    particle_y_series: list[list[float]] = []
    startup_diagnostics: list[dict[str, float]] = []

    sample_count = min(particles, int(config.get("sph_viz_sample_particles", 900)))
    sample_indices = rng.choice(particles, size=sample_count, replace=False)
    startup_diag_steps = max(0, int(config.get("sph_startup_diag_steps", 10)))

    simulated_time = 0.0
    max_runup_like_height = 0.0
    escaped_mass_fraction = 0.0
    x_edges = np.linspace(0.0, 1.0, nx + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, ny + 1, dtype=np.float64)
    x_coord = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_coord = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Save a true pre-step snapshot so the first visual frame is not mislabeled as t=0.
    initial_speed = np.linalg.norm(velocities, axis=1)
    depth_field, vel_x_grid, vel_y_grid, _ = _project_particles_to_grid(
        positions=positions,
        velocities=velocities,
        mass=particle_mass,
        rho0=rho0,
        nx=nx,
        ny=ny,
    )
    omega = _compute_vorticity(vel_x_grid, vel_y_grid, dx=1.0 / nx, dy=1.0 / ny)
    sampled_steps.append(-1)
    sampled_times.append(0.0)
    front_position_series.append(float(np.quantile(positions[:, 0], front_quantile)))
    max_speed_series.append(float(np.max(initial_speed) if initial_speed.size else 0.0))
    depth_field_series.append(depth_field.tolist())
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

        dt_step = _stable_dt(
            dt_base=dt_base,
            velocities=velocities,
            h=h_smooth,
            c0=c0,
            cfl=cfl,
            min_ratio=dt_min_ratio,
        )
        dt_step = min(dt_step, target_sim_time - simulated_time)
        dt_step = max(dt_step, 1.0e-6)

        cell_map = _build_cell_linked_list(positions, cell_size=2.0 * h_smooth)
        density = np.zeros(particles, dtype=np.float64)
        pressure = np.zeros(particles, dtype=np.float64)

        sigma = 10.0 / (7.0 * np.pi * h_smooth * h_smooth)
        w0 = sigma * 1.0

        for i in range(particles):
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
        acc[:, 1] -= gravity
        eps = 0.01 * h_smooth * h_smooth
        for i in range(particles):
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

            pressure_term = pressure[i] / (density[i] * density[i]) + pressure[j_idx] / (density[j_idx] * density[j_idx])
            pair_coeff = -particle_mass * (pressure_term + visc_term)

            acc[i] += np.sum(pair_coeff[:, np.newaxis] * grad_w, axis=0)

        velocities += dt_step * acc
        positions += dt_step * velocities
        simulated_time += dt_step

        below = positions[:, 1] < 0.0
        positions[below, 1] = 0.0
        velocities[below, 1] = np.abs(velocities[below, 1]) * floor_restitution
        velocities[below, 0] *= floor_friction

        above = positions[:, 1] > domain_y
        positions[above, 1] = domain_y
        velocities[above, 1] = -np.abs(velocities[above, 1]) * floor_restitution

        left = positions[:, 0] < 0.0
        positions[left, 0] = 0.0
        velocities[left, 0] = np.abs(velocities[left, 0]) * left_wall_restitution

        if right_boundary == "reflective":
            right = positions[:, 0] > domain_x
            positions[right, 0] = domain_x
            velocities[right, 0] = -np.abs(velocities[right, 0]) * right_wall_restitution
        else:
            escaped = positions[:, 0] > domain_x
            escaped_mass_fraction = float(np.mean(escaped))
            positions[escaped, 0] = domain_x + 0.02

        if np.any(positions[:, 0] > runup_band_x):
            max_runup_like_height = max(max_runup_like_height, float(np.max(positions[positions[:, 0] > runup_band_x, 1])))

        if step < startup_diag_steps:
            right_hits = int(np.count_nonzero(right)) if right_boundary == "reflective" else int(np.count_nonzero(escaped))
            startup_diagnostics.append(
                {
                    "step": float(step),
                    "time": float(simulated_time),
                    "density_min": float(np.min(density)),
                    "density_max": float(np.max(density)),
                    "pressure_min": float(np.min(pressure)),
                    "pressure_max": float(np.max(pressure)),
                    "zero_pressure_fraction": float(np.mean(pressure <= 0.0)),
                    "max_speed": float(np.max(np.linalg.norm(velocities, axis=1))),
                    "max_abs_vy": float(np.max(np.abs(velocities[:, 1]))),
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
            front_position = float(np.quantile(positions[:, 0], front_quantile))
            speed = np.linalg.norm(velocities, axis=1)
            depth_field, vel_x_grid, vel_y_grid, counts = _project_particles_to_grid(
                positions=positions,
                velocities=velocities,
                mass=particle_mass,
                rho0=rho0,
                nx=nx,
                ny=ny,
            )
            omega = _compute_vorticity(vel_x_grid, vel_y_grid, dx=1.0 / nx, dy=1.0 / ny)
            omega_peak = float(np.max(np.abs(omega)))
            vortex_area = float(np.mean(np.abs(omega) >= vortex_threshold))

            sampled_steps.append(step)
            sampled_times.append(float(simulated_time))
            front_position_series.append(front_position)
            max_speed_series.append(float(np.max(speed) if speed.size else 0.0))
            depth_field_series.append(depth_field.tolist())
            velocity_x_series.append(vel_x_grid.tolist())
            velocity_y_series.append(vel_y_grid.tolist())
            vorticity_series.append(omega.tolist())
            vorticity_peak_series.append(omega_peak)
            vortex_area_series.append(vortex_area)
            particle_x_series.append(positions[sample_indices, 0].tolist())
            particle_y_series.append(positions[sample_indices, 1].tolist())

    front_arrival_time = float("nan")
    for front_now, t_now in zip(front_position_series, sampled_times):
        if front_now >= front_target_x:
            front_arrival_time = float(t_now)
            break

    impact_time = float("nan")
    rebound_time = float("nan")
    rebound_drop = 0.0
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
                rebound_flag = 1.0
                rebound_time = float(sampled_times[min_idx])

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

    final_mass = initial_mass * (1.0 - escaped_mass_fraction)
    mass_error = abs(final_mass - initial_mass) / max(initial_mass, 1.0e-12)
    retained_mass_fraction = final_mass / max(initial_mass, 1.0e-12)

    acceptance_pass = 0.0
    if (
        simulated_time >= target_sim_time
        and front_reached_flag > 0.5
        and rebound_flag > 0.5
        and vortex_pass > 0.5
        and retained_mass_fraction >= float(config.get("accept_retained_mass_min", 0.995))
        and escaped_mass_fraction <= float(config.get("accept_escaped_mass_max", 0.005))
        and max_runup_like_height >= float(config.get("accept_runup_min", 0.05))
    ):
        acceptance_pass = 1.0

    metrics = {
        "dof": float(particles * 2),
        "mass_error": float(mass_error),
        "retained_mass_fraction": float(retained_mass_fraction),
        "escaped_mass_fraction": float(escaped_mass_fraction),
        "splash_spread_width": float(np.quantile(positions[:, 0], 0.95) - np.quantile(positions[:, 0], 0.05)),
        "front_arrival_time": float(front_arrival_time),
        "max_runup_like_height": float(max_runup_like_height),
        "simulated_time_end": float(simulated_time),
        "impact_time": float(impact_time),
        "rebound_time": float(rebound_time),
        "rebound_drop": float(rebound_drop),
        "front_reached_flag": float(front_reached_flag),
        "rebound_flag": float(rebound_flag),
        "vorticity_peak": float(vortex_peak),
        "vortex_area_peak": float(vortex_area_peak),
        "vortex_duration_s": float(vortex_duration_s),
        "vortex_pass": float(vortex_pass),
        "peak_particle_speed": float(np.max(np.linalg.norm(velocities, axis=1))),
        "acceptance_pass": float(acceptance_pass),
        "completion_flag": 1.0,
    }

    viz_depth = depth_field_series[-1] if depth_field_series else []
    viz_vx = velocity_x_series[-1] if velocity_x_series else []
    viz_vy = velocity_y_series[-1] if velocity_y_series else []
    viz_omega = vorticity_series[-1] if vorticity_series else []
    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": (
            "2-D weakly-compressible SPH with cell-linked neighbour search, "
            "Tait EOS, and artificial viscosity "
            f"(right boundary={right_boundary})."
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
            "depth_field": viz_depth,
            "velocity_x": viz_vx,
            "velocity_y": viz_vy,
            "vorticity": viz_omega,
            "particle_x": positions[sample_indices, 0].tolist(),
            "particle_y": positions[sample_indices, 1].tolist(),
            "front_position": float(front_position_series[-1]) if front_position_series else 0.0,
        },
        "viz_timeseries": {
            "representation": "particle_and_projected_grid",
            "frame_steps": sampled_steps,
            "frame_times": sampled_times,
            "sideview_frame_times": sampled_times,
            "simulated_time_end": float(simulated_time),
            "x_coord": x_coord.tolist(),
            "y_coord": y_coord.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid_shape": [ny, nx],
            "depth_field_series": depth_field_series,
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
