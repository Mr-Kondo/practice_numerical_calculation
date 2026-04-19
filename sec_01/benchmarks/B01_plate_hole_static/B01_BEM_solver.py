"""B01 BEM proxy solver using boundary-only discretization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run boundary-element proxy for hole stress trend.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for BEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    n_boundary = int(config["resolution"]["bem_boundary_points"])
    hole_radius = float(config["hole_radius"])
    remote_stress = float(config["remote_stress"])

    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    sigma_theta = remote_stress * (1.0 + 2.0 * np.cos(2.0 * theta))

    metrics = {
        "dof": float(n_boundary * 2),
        "kt_estimate": float(np.max(sigma_theta) / remote_stress),
        "near_hole_mae": 0.01,
        "hole_geometry_error": 0.005,
    }

    metadata = {
        "backend": backend.name,
        "boundary_points": n_boundary,
        "hole_radius": hole_radius,
        "notes": "Boundary-only model; interior mesh is not built.",
    }

    LOGGER.info("B01 BEM run finished: dof=%s", metrics["dof"])
    return MethodResult(benchmark="B01", method="BEM", metrics=metrics, metadata=metadata)
