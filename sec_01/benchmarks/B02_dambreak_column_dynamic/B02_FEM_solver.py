"""B02 FEM solver for Lagrangian-mesh dam-break dynamics.

A structured rectangular mesh represents the fluid column.  Nodes fall
under gravity; floor (y = 0) nodes are constrained with a soft
inelastic reflective condition.  A horizontal pressure-like force
(proportional to local column depth) drives lateral spreading.

The mesh quality metric is a proxy Jacobian computed from the gradient
of vertical node displacement.  When nodes pile up at the floor the
Jacobian degrades below zero, indicating mesh collapse --- the same
phenomenon that triggers remeshing in real ALE codes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run Lagrangian-mesh dam-break proxy under gravity.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for FEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    steps = int(config["steps"])
    dt = float(config["dt"])
    g = float(config.get("gravity", 9.81))
    dam_fraction = float(config["dam_width_fraction"])

    # Structured mesh for the dam column: 16 x 24 nodes.
    n_cols = 16
    n_rows = 24
    col_width = dam_fraction  # normalised x
    col_height = 0.8  # normalised y

    x0 = np.linspace(0.0, col_width, n_cols)
    y0 = np.linspace(0.0, col_height, n_rows)
    xx0, yy0 = np.meshgrid(x0, y0, indexing="ij")  # shape (n_cols, n_rows)
    nodes = np.column_stack([xx0.ravel(), yy0.ravel()])  # (N, 2)
    vel = np.zeros_like(nodes)

    n_nodes = nodes.shape[0]

    sample_interval = max(1, steps // 40)
    min_quality_series: list[float] = []
    collapsed_ratio_series: list[float] = []
    sampled_steps: list[int] = []

    for step in range(steps):
        # Gravity.
        vel[:, 1] -= g * dt

        # Lateral pressure proxy: proportional to local depth estimate.
        # Each column of nodes (same x-index) has n_rows nodes; depth is
        # approximated by the mean y of nodes in that column.
        node_xi = np.arange(n_nodes) // n_rows  # column index per node
        col_mean_y = np.array([nodes[node_xi == ci, 1].mean() for ci in range(n_cols)])
        # Horizontal pressure gradient: -g * d(depth)/dx
        depth_grad = np.empty(n_cols)
        depth_grad[1:] = col_mean_y[1:] - col_mean_y[:-1]
        depth_grad[0] = 0.0
        vel[:, 0] -= dt * g * depth_grad[node_xi] / (col_width / (n_cols - 1))

        nodes += dt * vel

        # Floor boundary (y = 0): inelastic reflection + horizontal friction.
        below = nodes[:, 1] < 0.0
        nodes[below, 1] = 0.0
        vel[below, 1] = np.abs(vel[below, 1]) * 0.1
        vel[below, 0] *= 0.9

        # Left wall (x = 0): inelastic reflection.
        left = nodes[:, 0] < 0.0
        nodes[left, 0] = 0.0
        vel[left, 0] = np.abs(vel[left, 0]) * 0.2

        if step % sample_interval == 0 or step == steps - 1:
            # Jacobian proxy: based on y-displacement gradient over initial mesh.
            y_disp = nodes[:, 1] - np.linspace(0.0, col_height, n_rows).tolist() * n_cols
            y_disp_grid = y_disp.reshape(n_cols, n_rows)
            jac = 1.0 + np.gradient(y_disp_grid, axis=1) / (col_height / (n_rows - 1))
            sampled_steps.append(step)
            min_quality_series.append(float(np.min(jac)))
            collapsed_ratio_series.append(float(np.mean(jac < 0.2)))

    # Final mesh quality.
    y_disp_final = nodes[:, 1] - np.linspace(0.0, col_height, n_rows).tolist() * n_cols
    jac_final = 1.0 + np.gradient(y_disp_final.reshape(n_cols, n_rows), axis=1) / (col_height / (n_rows - 1))
    min_quality = float(np.min(jac_final))
    collapsed_ratio = float(np.mean(jac_final < 0.2))
    completion = 0.0 if min_quality < -0.5 else 1.0

    metrics = {
        "dof": float(n_nodes * 2),
        "mesh_min_quality": min_quality,
        "mesh_collapsed_ratio": collapsed_ratio,
        "completion_flag": completion,
    }

    metadata = {
        "status": "success",
        "backend": backend.name,
        "notes": "Lagrangian mesh under gravity; floor BC; quality = proxy Jacobian.",
        "viz": {
            "quality_sample": jac_final[::2, ::4].tolist(),
        },
        "viz_timeseries": {
            "frame_steps": sampled_steps,
            "min_quality_series": min_quality_series,
            "collapsed_ratio_series": collapsed_ratio_series,
        },
    }

    LOGGER.info("B02 FEM run finished: min_quality=%.4f", min_quality)
    return MethodResult(benchmark="B02", method="FEM", metrics=metrics, metadata=metadata)
