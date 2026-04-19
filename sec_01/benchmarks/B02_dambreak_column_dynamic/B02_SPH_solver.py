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

    # Initial rectangular dam: x in [0, dam_fraction], y in [0, 0.8].
    particles = np.column_stack(
        [
            rng.uniform(0.0, dam_fraction, n_particles),
            rng.uniform(0.0, 0.8, n_particles),
        ]
    )
    velocities = np.zeros_like(particles)

    # Grid-based density for pressure estimation (avoids O(N^2)).
    n_bins_x = 60
    n_bins_y = 30
    domain_x = 1.0
    domain_y = 1.0

    sample_interval = max(1, steps // 40)
    sampled_steps: list[int] = []
    particle_x_series: list[list[float]] = []
    particle_y_series: list[list[float]] = []
    sample_size = min(320, n_particles)
    sampled_indices = rng.choice(n_particles, size=sample_size, replace=False)

    for step in range(steps):
        # --- Gravity ---
        velocities[:, 1] -= g * dt

        # --- Grid-based pressure: horizontal acceleration from depth gradient ---
        # Count particles in each (x, y) bin; column-sum gives a depth proxy.
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
        depth_col = grid.sum(axis=0)  # (n_bins_x,)
        depth_norm = depth_col / max(depth_col.max(), 1.0)

        # Pressure gradient: du_x/dt = -g * d(depth)/dx
        grad_depth = np.empty(n_bins_x)
        grad_depth[1:] = depth_norm[1:] - depth_norm[:-1]
        grad_depth[0] = 0.0
        ax_field = -g * grad_depth * n_bins_x  # multiply by 1/dx_bin
        velocities[:, 0] += dt * ax_field[xi]

        # --- Advect ---
        particles += dt * velocities

        # --- Boundary conditions ---
        # Floor (y = 0): inelastic reflection.
        below = particles[:, 1] < 0.0
        particles[below, 1] = 0.0
        velocities[below, 1] = np.abs(velocities[below, 1]) * 0.15
        velocities[below, 0] *= 0.85  # horizontal friction

        # Left wall (x = 0): inelastic reflection.
        left = particles[:, 0] < 0.0
        particles[left, 0] = 0.0
        velocities[left, 0] = np.abs(velocities[left, 0]) * 0.2

        if step % sample_interval == 0 or step == steps - 1:
            sampled_steps.append(step)
            particle_x_series.append(particles[sampled_indices, 0].tolist())
            particle_y_series.append(particles[sampled_indices, 1].tolist())

    spread = float(np.quantile(particles[:, 0], 0.95) - np.quantile(particles[:, 0], 0.05))
    speed = np.linalg.norm(velocities, axis=1)

    metrics = {
        "dof": float(n_particles * 2),
        "mass_error": 0.0,
        "splash_spread_width": spread,
        "completion_flag": 1.0,
        "peak_particle_speed": float(np.max(speed)),
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": "Gravity-driven SPH proxy with grid-based pressure and reflective walls.",
        "viz": {
            "particle_x": particles[: min(900, n_particles), 0].tolist(),
            "particle_y": particles[: min(900, n_particles), 1].tolist(),
        },
        "viz_timeseries": {
            "frame_steps": sampled_steps,
            "particle_x_series": particle_x_series,
            "particle_y_series": particle_y_series,
        },
    }

    LOGGER.info("B02 SPH run finished: spread=%.4f", spread)
    return MethodResult(benchmark="B02", method="SPH", metrics=metrics, metadata=metadata)
