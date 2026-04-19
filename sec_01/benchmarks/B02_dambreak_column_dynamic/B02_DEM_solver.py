"""B02 DEM solver for bond-particle column collapse under gravity.

A rectangular lattice of bonded particles represents the initial dam
column.  Gravity accelerates all particles downward.  Floor (y = 0)
contact uses an inelastic-reflective condition.  Lateral spreading is
driven by inter-particle bond forces once bonds are strained beyond
the break threshold.

Bond breakage is irreversible.  The cumulative broken-bond count is
the primary diagnostic signal, rising from zero as the column collapses.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run bond-particle DEM dam-break proxy under gravity.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for DEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    rng = np.random.default_rng(int(config["seed"]))

    n_rows = 36
    n_cols = 8
    spacing = 0.018
    break_strain = float(config["break_strain"])
    g = float(config.get("gravity", 9.81))

    x0 = np.repeat(np.arange(n_cols), n_rows) * spacing
    y0 = np.tile(np.arange(n_rows), n_cols) * spacing
    points = np.column_stack([x0, y0])

    # Build initial bond list: horizontal and vertical nearest neighbours.
    neighbors: list[tuple[int, int, float]] = []
    for col in range(n_cols):
        for row in range(n_rows):
            idx = col * n_rows + row
            if row + 1 < n_rows:
                j = col * n_rows + (row + 1)
                neighbors.append((idx, j, spacing))
            if col + 1 < n_cols:
                j = (col + 1) * n_rows + row
                neighbors.append((idx, j, spacing))

    steps = int(config["steps"])
    dt = float(config["dt"])
    velocity = np.zeros_like(points)
    sample_interval = max(1, steps // 40)
    sampled_steps: list[int] = []
    point_x_series: list[list[float]] = []
    point_y_series: list[list[float]] = []
    broken_series: list[float] = []

    broken = 0
    first_break_step = -1

    for step in range(steps):
        # --- Gravity ---
        velocity[:, 1] -= g * dt

        # --- Bond spring forces ---
        force = np.zeros_like(points)
        new_neighbors: list[tuple[int, int, float]] = []
        for i, j, rest in neighbors:
            delta = points[j] - points[i]
            current = float(np.linalg.norm(delta))
            strain = (current - rest) / rest
            if strain > break_strain:
                broken += 1
                if first_break_step < 0:
                    first_break_step = step
            else:
                # Hooke spring force proportional to strain.
                k = 500.0  # stiffness (normalised units)
                f = k * strain * delta / (current + 1.0e-12)
                force[i] += f
                force[j] -= f
                new_neighbors.append((i, j, rest))
        neighbors = new_neighbors

        velocity += dt * force
        points += dt * velocity

        # --- Floor boundary (y = 0): inelastic reflection ---
        below = points[:, 1] < 0.0
        points[below, 1] = 0.0
        velocity[below, 1] = np.abs(velocity[below, 1]) * 0.1
        velocity[below, 0] *= 0.85  # horizontal friction at floor

        if step % sample_interval == 0 or step == steps - 1:
            sampled_steps.append(step)
            point_x_series.append(points[:, 0].tolist())
            point_y_series.append(points[:, 1].tolist())
            broken_series.append(float(broken))

    tip_disp = float(np.max(points[:, 0] - x0))

    metrics = {
        "dof": float(points.shape[0] * 2),
        "first_failure_time": float(max(first_break_step, 0) * dt),
        "broken_bonds": float(broken),
        "tip_displacement": tip_disp,
        "completion_flag": 1.0,
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "remaining_bonds": len(neighbors),
        "notes": "Gravity-driven column collapse; Hooke bonds break on strain threshold.",
        "viz": {
            "point_x": points[:, 0].tolist(),
            "point_y": points[:, 1].tolist(),
        },
        "viz_timeseries": {
            "frame_steps": sampled_steps,
            "point_x_series": point_x_series,
            "point_y_series": point_y_series,
            "broken_bonds_series": broken_series,
        },
    }

    LOGGER.info("B02 DEM run finished: broken_bonds=%s", broken)
    return MethodResult(benchmark="B02", method="DEM", metrics=metrics, metadata=metadata)
