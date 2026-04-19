"""B02 FVM solver using 1-D shallow-water equations (inviscid).

Governing equations:

    h_t + (h u)_x = 0
    u_t + g h_x   = 0

An explicit first-order upwind scheme advances the solution with a
CFL-stable internal time step computed from the grid spacing and
wave speed.  Both ends use reflective (no-flow) wall conditions.
The spatial domain is normalised to x in [0, 1] with h_0 = 1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run 1-D shallow-water finite-volume dam-break solver.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FVM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    nx = int(config["grid_nx"])
    steps = int(config["steps"])
    g = float(config.get("gravity", 9.81))
    dam_fraction = float(config["dam_width_fraction"])

    # Normalised grid: domain x in [0, 1], cell width dx = 1/nx.
    dx = 1.0 / nx
    dam_nx = max(2, int(nx * dam_fraction))

    h = np.zeros(nx, dtype=np.float64)
    h[:dam_nx] = 1.0
    u = np.zeros(nx, dtype=np.float64)

    # CFL-stable internal time step (Courant number 0.45).
    dt_fvm = 0.45 * dx / (np.sqrt(g) + 1.0e-8)

    initial_mass = float(np.sum(h) * dx)
    sample_interval = max(1, steps // 40)
    height_series: list[list[float]] = []
    sampled_steps: list[int] = []

    for step in range(steps):
        # --- Momentum: u_t = -g * h_x (forward difference) ---
        dh = np.empty(nx)
        dh[:-1] = h[1:] - h[:-1]
        dh[-1] = 0.0  # right-wall zero-gradient
        u = u - dt_fvm * g * dh / dx

        # Reflective wall boundary conditions.
        u[0] = max(0.0, u[0])  # left wall: block inflow
        u[-1] = min(0.0, u[-1])  # right wall: block outflow

        # --- Continuity: h_t = -(h u)_x (first-order upwind) ---
        # Right-face upwind flux for each cell.
        h_right = np.empty(nx)
        h_right[:-1] = h[1:]
        h_right[-1] = h[-1]  # ghost cell at right wall

        u_pos = np.maximum(u, 0.0)
        u_neg = np.minimum(u, 0.0)
        flux_right = h * u_pos + h_right * u_neg

        # Left-face flux is the right-face flux of the left neighbour.
        flux_left = np.empty(nx)
        flux_left[0] = 0.0  # left wall: no inflow
        flux_left[1:] = flux_right[:-1]

        h = h - dt_fvm * (flux_right - flux_left) / dx
        h = np.maximum(h, 0.0)

        if step % sample_interval == 0 or step == steps - 1:
            sampled_steps.append(step)
            height_series.append(h.tolist())

    final_mass = float(np.sum(h) * dx)
    mass_error = abs(final_mass - initial_mass) / max(initial_mass, 1.0e-12)
    spread_width = float(np.count_nonzero(h > 1.0e-3) * dx)

    metrics = {
        "dof": float(nx),
        "mass_error": float(mass_error),
        "splash_spread_width": spread_width,
        "completion_flag": 1.0,
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": "1-D shallow-water upwind scheme; CFL-stable internal dt.",
        "viz": {
            "x_index": list(range(nx)),
            "height_centerline": h.tolist(),
        },
        "viz_timeseries": {
            "frame_steps": sampled_steps,
            "x_index": list(range(nx)),
            "height_centerline_series": height_series,
        },
    }

    LOGGER.info("B02 FVM run finished: mass_error=%.4e", metrics["mass_error"])
    return MethodResult(benchmark="B02", method="FVM", metrics=metrics, metadata=metadata)
