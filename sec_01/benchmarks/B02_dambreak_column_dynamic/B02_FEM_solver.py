"""B02 FEM proxy solver highlighting mesh distortion risk."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run deforming-mesh proxy and estimate collapse tendency.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    nx = int(config["grid_nx"])
    ny = int(config["grid_ny"])
    steps = int(config["steps"])

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    deformation = np.zeros_like(xx)
    for step in range(steps):
        t = step / max(steps - 1, 1)
        deformation += 0.0025 * np.sin(10.0 * xx) * np.exp(3.0 * t * (yy - 0.5))

    jacobian_proxy = 1.0 - np.abs(np.gradient(deformation, axis=1)) * 12.0
    min_quality = float(np.min(jacobian_proxy))
    collapsed_ratio = float(np.mean(jacobian_proxy < 0.2))

    completion = 0.0 if min_quality < -0.1 else 1.0

    metrics = {
        "dof": float(nx * ny * 2),
        "mesh_min_quality": min_quality,
        "mesh_collapsed_ratio": collapsed_ratio,
        "completion_flag": completion,
    }

    metadata = {
        "backend": backend.name,
        "notes": "Lagrangian mesh proxy where severe deformation degrades quality.",
    }

    LOGGER.info("B02 FEM run finished: min_quality=%.4f", min_quality)
    return MethodResult(benchmark="B02", method="FEM", metrics=metrics, metadata=metadata)
