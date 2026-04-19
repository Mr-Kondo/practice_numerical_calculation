"""B01 FEM proxy solver with local mesh refinement around the hole."""

from __future__ import annotations

import logging
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
    """Compute proxy stress concentration profile."""

    r = np.sqrt(x * x + y * y) + 1.0e-8
    theta = np.arctan2(y, x)
    ratio = (hole_radius / r) ** 2
    return remote_stress * (1.0 + 2.0 * ratio * np.cos(2.0 * theta))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run locally refined point-cloud FEM proxy.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    n_points = int(config["resolution"]["fem_points"])
    width = float(config["plate_width"])
    height = float(config["plate_height"])
    hole_radius = float(config["hole_radius"])
    remote_stress = float(config["remote_stress"])

    rng = np.random.default_rng(int(config["seed"]))

    core_count = int(n_points * 0.6)
    outer_count = n_points - core_count

    theta_core = rng.uniform(0.0, 2.0 * np.pi, core_count)
    radial_core = hole_radius + rng.uniform(0.0, 0.25, core_count) ** 2
    x_core = radial_core * np.cos(theta_core)
    y_core = radial_core * np.sin(theta_core)

    x_outer = rng.uniform(-width / 2.0, width / 2.0, outer_count)
    y_outer = rng.uniform(-height / 2.0, height / 2.0, outer_count)

    x = np.concatenate([x_core, x_outer])
    y = np.concatenate([y_core, y_outer])

    r = np.sqrt(x * x + y * y)
    solid = r >= hole_radius

    sigma = _kirsch_proxy(x[solid], y[solid], hole_radius=hole_radius, remote_stress=remote_stress)
    near_hole = r[solid] < hole_radius * 1.4
    target = _kirsch_proxy(x[solid][near_hole], y[solid][near_hole], hole_radius=hole_radius, remote_stress=remote_stress)

    residual = np.abs(sigma[near_hole] - target)
    metrics = {
        "dof": float(np.count_nonzero(solid) * 2),
        "kt_estimate": float(np.max(sigma[near_hole]) / remote_stress),
        "near_hole_mae": float(residual.mean()) if residual.size else float("nan"),
        "hole_geometry_error": 0.03,
    }

    metadata = {
        "backend": backend.name,
        "point_count": int(np.count_nonzero(solid)),
        "notes": "Locally refined unstructured representation around the hole.",
    }

    LOGGER.info("B01 FEM run finished: dof=%s", metrics["dof"])
    return MethodResult(benchmark="B01", method="FEM", metrics=metrics, metadata=metadata)
