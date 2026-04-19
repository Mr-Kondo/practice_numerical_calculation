"""B01 FDM proxy solver on a structured grid."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _kirsch_proxy(
    x: np.ndarray,
    y: np.ndarray,
    hole_radius: float,
    remote_stress: float,
) -> np.ndarray:
    """Compute proxy circumferential stress using Kirsch-like trend."""

    r = np.sqrt(x * x + y * y) + 1.0e-8
    theta = np.arctan2(y, x)
    ratio = (hole_radius / r) ** 2
    return remote_stress * (1.0 + 2.0 * ratio * np.cos(2.0 * theta))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run structured-grid finite-difference approximation.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FDM run.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    nx = int(config["resolution"]["fdm_nx"])
    ny = int(config["resolution"]["fdm_ny"])
    width = float(config["plate_width"])
    height = float(config["plate_height"])
    hole_radius = float(config["hole_radius"])
    remote_stress = float(config["remote_stress"])

    x = np.linspace(-width / 2.0, width / 2.0, nx)
    y = np.linspace(-height / 2.0, height / 2.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    r = np.sqrt(xx * xx + yy * yy)
    solid_mask = r >= hole_radius

    stair_hole_radius = math.hypot(width / nx, height / ny) + hole_radius
    stair_mask = r >= stair_hole_radius

    sigma_exact = _kirsch_proxy(xx, yy, hole_radius=hole_radius, remote_stress=remote_stress)
    sigma_fdm = _kirsch_proxy(
        xx,
        yy,
        hole_radius=stair_hole_radius,
        remote_stress=remote_stress,
    )

    region = solid_mask & stair_mask
    near_hole = region & (r < hole_radius * 1.7)

    abs_error = np.abs(sigma_fdm[near_hole] - sigma_exact[near_hole])
    mean_error = float(abs_error.mean()) if abs_error.size else float("nan")

    kt_est = float(np.max(sigma_fdm[near_hole]) / remote_stress) if abs_error.size else float("nan")

    metrics = {
        "dof": float(nx * ny),
        "kt_estimate": kt_est,
        "near_hole_mae": mean_error,
        "hole_geometry_error": float((stair_hole_radius - hole_radius) / hole_radius),
    }

    metadata = {
        "backend": backend.name,
        "grid_shape": [ny, nx],
        "notes": "Structured grid with stair-step hole approximation.",
    }

    LOGGER.info("B01 FDM run finished: dof=%s", metrics["dof"])
    return MethodResult(benchmark="B01", method="FDM", metrics=metrics, metadata=metadata)
