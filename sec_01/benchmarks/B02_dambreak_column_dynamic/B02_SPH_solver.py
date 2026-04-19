"""B02 SPH solver for dam-break particle dynamics.

Physics:
  - Gravity acts in the negative-y direction on every particle.
  - A grid-based density estimate (O(N * n_bins)) drives horizontal pressure
    acceleration without the O(N^2) cost of full neighbour-pair SPH.
  - Floor (y = 0) and left wall (x = 0) use inelastic-reflective boundary
    conditions.  No right-wall constraint: particles spread freely.

Initial condition: N particles uniformly distributed in a rectangular dam
cell x in [0, dam_width_fraction], y in [0, 0.8], at rest.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _compute_depth_proxy(
    particles: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    domain_x: float,
    domain_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build column density proxy from particle distribution.

    Args:
        particles: Particle coordinates, shape (N, 2).
        n_bins_x: Number of bins in x.
        n_bins_y: Number of bins in y.
        domain_x: Domain size in x.
        domain_y: Domain size in y.

    Returns:
        Tuple (xi, yi, depth_norm).
    """

    xi = np.clip(
        (particles[:, 0] / domain_x * n_bins_x).astype(np.int32),
        0,
        n_bins_x - 1,
    )
    yi = np.clip(
        (particles[:, 1] / domain_y * n_bins_y).astype(np.int32),
        0,
        n_bins_y - 1,
    )
    grid = np.zeros((n_bins_y, n_bins_x), dtype=np.float64)
    np.add.at(grid, (yi, xi), 1.0)
    depth_col = grid.sum(axis=0)
    depth_norm = depth_col / max(float(depth_col.max()), 1.0)
    return xi, yi, depth_norm


def _compute_acceleration_field(depth_norm: np.ndarray, gravity: float, n_bins_x: int, nu: float) -> np.ndarray:
    """Compute 1-D acceleration field from depth profile.

    Args:
        depth_norm: Normalized column depth.
        gravity: Gravity constant.
        n_bins_x: Number of x bins.
        nu: Artificial viscosity coefficient.

    Returns:
        Acceleration values per x-bin.
    """

    grad_depth = np.zeros(n_bins_x, dtype=np.float64)
    grad_depth[1:-1] = 0.5 * (depth_norm[2:] - depth_norm[:-2])
    grad_depth[0] = depth_norm[1] - depth_norm[0]
    grad_depth[-1] = depth_norm[-1] - depth_norm[-2]

    lap_depth = np.zeros(n_bins_x, dtype=np.float64)
    lap_depth[1:-1] = depth_norm[2:] - 2.0 * depth_norm[1:-1] + depth_norm[:-2]
    lap_depth[0] = depth_norm[1] - depth_norm[0]
    lap_depth[-1] = depth_norm[-2] - depth_norm[-1]

    dx_bin = 1.0 / max(n_bins_x, 1)
    pressure_acc = -gravity * grad_depth / max(dx_bin, 1.0e-8)
    viscous_acc = nu * lap_depth / max(dx_bin * dx_bin, 1.0e-8)
    return pressure_acc + viscous_acc


def _apply_xsph(velocities: np.ndarray, xi: np.ndarray, xsph_coeff: float, n_bins_x: int) -> None:
    """Apply simple XSPH-like velocity smoothing by x-bin means.

    Args:
        velocities: Particle velocities, shape (N, 2).
        xi: x-bin index per particle.
        xsph_coeff: Smoothing weight in [0, 1].
        n_bins_x: Number of x bins.
    """

    if xsph_coeff <= 0.0:
        return

    counts = np.bincount(xi, minlength=n_bins_x).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    mean_vx = np.bincount(xi, weights=velocities[:, 0], minlength=n_bins_x) / counts
    mean_vy = np.bincount(xi, weights=velocities[:, 1], minlength=n_bins_x) / counts

    velocities[:, 0] = (1.0 - xsph_coeff) * velocities[:, 0] + xsph_coeff * mean_vx[xi]
    velocities[:, 1] = (1.0 - xsph_coeff) * velocities[:, 1] + xsph_coeff * mean_vy[xi]


def _compute_dt(
    dt_base: float,
    velocities: np.ndarray,
    cfl: float,
    dx_bin: float,
    dy_bin: float,
    min_ratio: float,
) -> float:
    """Compute CFL-limited time step with a lower safety bound.

    Args:
        dt_base: Configured base dt.
        velocities: Particle velocities.
        cfl: CFL number.
        dx_bin: Grid spacing in x.
        dy_bin: Grid spacing in y.
        min_ratio: Lower ratio relative to dt_base.

    Returns:
        Stable time step.
    """

    speed = np.linalg.norm(velocities, axis=1)
    max_speed = float(np.max(speed)) if speed.size else 0.0
    dt_min = max(min_ratio, 0.05) * dt_base
    if max_speed < 1.0e-8:
        return dt_base

    dt_cfl = cfl * min(dx_bin, dy_bin) / max_speed
    return float(min(dt_base, max(dt_min, dt_cfl)))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run particle-based SPH dam-break proxy with gravity.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for SPH proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    rng = np.random.default_rng(int(config["seed"]))
    n_particles = int(config["sph_particles"])
    steps = int(config["steps"])
    dt = float(config["dt"])
    g = float(config.get("gravity", 9.81))
    dam_fraction = float(config["dam_width_fraction"])
    target_sim_time = float(config.get("target_sim_time_s", steps * dt))
    max_steps = int(config.get("max_steps", max(steps, int(np.ceil(target_sim_time / max(dt * 0.5, 1.0e-6))))))
    right_boundary = str(config.get("right_boundary", "reflective")).strip().lower()

    if right_boundary not in {"reflective", "open"}:
        raise ValueError(f"Unsupported B02 SPH right_boundary={right_boundary!r}")

    n_bins_x = int(config.get("sph_bins_x", 60))
    n_bins_y = int(config.get("sph_bins_y", 30))
    cfl = float(config.get("sph_cfl", 0.35))
    dt_min_ratio = float(config.get("sph_dt_min_ratio", 0.25))
    artificial_viscosity = float(config.get("sph_artificial_viscosity", 0.08))
    xsph_coeff = float(config.get("sph_xsph_coeff", 0.12))
    floor_restitution = float(config.get("sph_floor_restitution", 0.15))
    floor_friction = float(config.get("sph_floor_friction", 0.85))
    left_wall_restitution = float(config.get("sph_left_wall_restitution", 0.2))
    right_wall_restitution = float(config.get("sph_right_wall_restitution", 0.2))
    front_quantile = float(config.get("sph_front_quantile", 0.98))
    front_target_x = float(config.get("column_x_fraction", 0.72))
    impact_x = float(config.get("accept_front_reach_x", 0.95))
    rebound_window = float(config.get("rebound_window_s", 0.10))
    rebound_min_drop = float(config.get("rebound_min_drop", 0.08))

    # Initial rectangular dam: x in [0, dam_fraction], y in [0, 0.8].
    particles = np.column_stack(
        [
            rng.uniform(0.0, dam_fraction, n_particles),
            rng.uniform(0.0, 0.8, n_particles),
        ]
    )
    velocities = np.zeros_like(particles)

    # Grid-based density for pressure estimation (avoids O(N^2)).
    domain_x = 1.0
    domain_y = 1.0
    dx_bin = domain_x / max(n_bins_x, 1)
    dy_bin = domain_y / max(n_bins_y, 1)

    sample_interval = max(1, max_steps // 120)
    sampled_steps: list[int] = []
    particle_x_series: list[list[float]] = []
    particle_y_series: list[list[float]] = []
    sampled_times: list[float] = []
    front_position_series: list[float] = []
    max_speed_series: list[float] = []
    mean_density_proxy_series: list[float] = []
    simulated_time = 0.0
    max_runup_like_height = 0.0
    base_sample_size = min(240, n_particles)
    front_sample_size = min(80, max(1, n_particles - base_sample_size))
    base_indices = rng.choice(n_particles, size=base_sample_size, replace=False)

    for step in range(max_steps):
        if simulated_time >= target_sim_time:
            break

        dt_step = _compute_dt(
            dt_base=dt,
            velocities=velocities,
            cfl=cfl,
            dx_bin=dx_bin,
            dy_bin=dy_bin,
            min_ratio=dt_min_ratio,
        )

        # --- Gravity ---
        velocities[:, 1] -= g * dt_step

        # --- Grid-based pressure: horizontal acceleration from depth gradient ---
        xi, _, depth_norm = _compute_depth_proxy(
            particles=particles,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            domain_x=domain_x,
            domain_y=domain_y,
        )
        ax_field = _compute_acceleration_field(
            depth_norm=depth_norm,
            gravity=g,
            n_bins_x=n_bins_x,
            nu=artificial_viscosity,
        )
        velocities[:, 0] += dt_step * ax_field[xi]
        _apply_xsph(velocities=velocities, xi=xi, xsph_coeff=xsph_coeff, n_bins_x=n_bins_x)

        # --- Advect ---
        particles += dt_step * velocities
        simulated_time += dt_step

        # --- Boundary conditions ---
        # Floor (y = 0): inelastic reflection.
        below = particles[:, 1] < 0.0
        particles[below, 1] = 0.0
        velocities[below, 1] = np.abs(velocities[below, 1]) * floor_restitution
        velocities[below, 0] *= floor_friction

        # Left wall (x = 0): inelastic reflection.
        left = particles[:, 0] < 0.0
        particles[left, 0] = 0.0
        velocities[left, 0] = np.abs(velocities[left, 0]) * left_wall_restitution

        if right_boundary == "reflective":
            right = particles[:, 0] > domain_x
            particles[right, 0] = domain_x
            velocities[right, 0] = -np.abs(velocities[right, 0]) * right_wall_restitution

        max_runup_like_height = max(
            max_runup_like_height,
            float(np.max(particles[particles[:, 0] > 0.95, 1])) if np.any(particles[:, 0] > 0.95) else 0.0,
        )

        if step % sample_interval == 0 or step == max_steps - 1:
            sampled_steps.append(step)
            sampled_times.append(simulated_time)
            front_indices = np.argpartition(particles[:, 0], -front_sample_size)[-front_sample_size:]
            sampled_indices = np.unique(np.concatenate([base_indices, front_indices]))
            particle_x_series.append(particles[sampled_indices, 0].tolist())
            particle_y_series.append(particles[sampled_indices, 1].tolist())
            front_position_series.append(float(np.quantile(particles[:, 0], front_quantile)))
            max_speed_series.append(float(np.max(np.linalg.norm(velocities, axis=1))))
            mean_density_proxy_series.append(float(np.mean(depth_norm)))

    spread = float(np.quantile(particles[:, 0], 0.95) - np.quantile(particles[:, 0], 0.05))
    speed = np.linalg.norm(velocities, axis=1)
    front_arrival_time = float("nan")
    for pos, t_val in zip(front_position_series, sampled_times):
        if pos >= front_target_x:
            front_arrival_time = float(t_val)
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
            baseline_front = float(front_position_series[impact_idx])
            min_front = baseline_front
            min_idx = impact_idx
            rebound_end = impact_time + rebound_window
            for idx in range(impact_idx, len(front_position_series)):
                if sampled_times[idx] > rebound_end:
                    break
                if front_position_series[idx] < min_front:
                    min_front = float(front_position_series[idx])
                    min_idx = idx
            rebound_drop = baseline_front - min_front
            if rebound_drop >= rebound_min_drop:
                rebound_flag = 1.0
                rebound_time = float(sampled_times[min_idx])

    acceptance_pass = 0.0
    if (
        simulated_time >= target_sim_time
        and front_reached_flag > 0.5
        and rebound_flag > 0.5
        and max_runup_like_height >= float(config.get("accept_runup_min", 0.05))
    ):
        acceptance_pass = 1.0

    metrics = {
        "dof": float(n_particles * 2),
        "mass_error": 0.0,
        "splash_spread_width": spread,
        "completion_flag": 1.0,
        "peak_particle_speed": float(np.max(speed)),
        "front_arrival_time": front_arrival_time,
        "max_runup_like_height": float(max_runup_like_height),
        "simulated_time_end": float(simulated_time),
        "impact_time": impact_time,
        "rebound_time": rebound_time,
        "rebound_drop": float(rebound_drop),
        "front_reached_flag": front_reached_flag,
        "rebound_flag": rebound_flag,
        "acceptance_pass": acceptance_pass,
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": (
            "Stabilized SPH proxy with depth-gradient pressure, artificial viscosity, "
            f"XSPH smoothing, and right boundary={right_boundary}."
        ),
        "boundary_conditions": {
            "left": "reflective",
            "right": right_boundary,
            "floor": "reflective",
        },
        "viz": {
            "particle_x": particles[: min(900, n_particles), 0].tolist(),
            "particle_y": particles[: min(900, n_particles), 1].tolist(),
            "front_position": float(front_position_series[-1]) if front_position_series else 0.0,
        },
        "viz_timeseries": {
            "frame_steps": sampled_steps,
            "frame_times": sampled_times,
            "dt": dt,
            "simulated_time_end": float(simulated_time),
            "front_position_series": front_position_series,
            "max_speed_series": max_speed_series,
            "mean_density_proxy_series": mean_density_proxy_series,
            "particle_x_series": particle_x_series,
            "particle_y_series": particle_y_series,
        },
    }

    LOGGER.info("B02 SPH run finished: spread=%.4f", spread)
    return MethodResult(benchmark="B02", method="SPH", metrics=metrics, metadata=metadata)
