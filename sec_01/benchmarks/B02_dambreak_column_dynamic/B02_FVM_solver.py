"""B02 FVM proxy solver for conservative fluid transport."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _initial_height(nx: int, ny: int, dam_fraction: float) -> np.ndarray:
    """Build initial dam-break height field."""

    h = np.zeros((ny, nx), dtype=np.float64)
    dam_nx = max(2, int(nx * dam_fraction))
    h[:, :dam_nx] = 1.0
    return h


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run finite-volume proxy with simple upwind fluxes.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FVM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    nx = int(config["grid_nx"])
    ny = int(config["grid_ny"])
    steps = int(config["steps"])
    dt = float(config["dt"])

    h = _initial_height(nx, ny, float(config["dam_width_fraction"]))
    u = np.zeros_like(h)

    initial_mass = float(np.sum(h))

    for _ in range(steps):
        flux = np.roll(h * u, -1, axis=1) - (h * u)
        h = h - dt * flux
        h = np.maximum(h, 0.0)
        u = u + 0.1 * (np.roll(h, 1, axis=1) - h)

    final_mass = float(np.sum(h))
    mass_error = abs(final_mass - initial_mass) / max(initial_mass, 1.0e-12)

    metrics = {
        "dof": float(nx * ny),
        "mass_error": float(mass_error),
        "splash_spread_width": float(np.count_nonzero(h > 1.0e-3) / ny),
        "completion_flag": 1.0,
    }

    metadata = {
        "backend": backend.name,
        "notes": "Conservative-grid proxy emphasizing mass transport.",
    }

    LOGGER.info("B02 FVM run finished: mass_error=%.4e", metrics["mass_error"])
    return MethodResult(benchmark="B02", method="FVM", metrics=metrics, metadata=metadata)
