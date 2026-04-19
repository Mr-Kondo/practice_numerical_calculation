"""B02 DEM proxy solver for deformable-column fracture behavior."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run bond-break DEM proxy for column fragmentation.

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

    x0 = np.repeat(np.arange(n_cols), n_rows) * spacing
    y0 = np.tile(np.arange(n_rows), n_cols) * spacing
    points = np.column_stack([x0, y0])

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

    broken = 0
    first_break_step = -1

    for step in range(steps):
        impact = 0.12 * np.exp(-((points[:, 1] - points[:, 1].mean()) ** 2) / 0.02)
        impact *= 0.5 + rng.random(points.shape[0])
        velocity[:, 0] += dt * impact
        points += dt * velocity

        new_neighbors: list[tuple[int, int, float]] = []
        for i, j, rest in neighbors:
            current = np.linalg.norm(points[i] - points[j])
            strain = (current - rest) / rest
            if strain > break_strain:
                broken += 1
                if first_break_step < 0:
                    first_break_step = step
            else:
                new_neighbors.append((i, j, rest))
        neighbors = new_neighbors

    tip_disp = float(np.max(points[:, 0] - x0))

    metrics = {
        "dof": float(points.shape[0] * 2),
        "first_failure_time": float(max(first_break_step, 0) * dt),
        "broken_bonds": float(broken),
        "tip_displacement": tip_disp,
        "completion_flag": 1.0,
    }

    metadata = {
        "backend": backend.name,
        "remaining_bonds": len(neighbors),
        "notes": "Bond breakage models column tearing without mesh collapse.",
    }

    LOGGER.info("B02 DEM run finished: broken_bonds=%s", broken)
    return MethodResult(benchmark="B02", method="DEM", metrics=metrics, metadata=metadata)
