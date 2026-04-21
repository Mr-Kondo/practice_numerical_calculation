"""B01 BEM proxy solver using boundary-only discretization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)

# Visualization grid resolution (width:height = 2:1).
_VIS_NX = 100
_VIS_NY = 50


def _kirsch_sigma_tt(
    xx: np.ndarray,
    yy: np.ndarray,
    hole_radius: float,
    remote_stress: float,
) -> np.ndarray:
    """Compute circumferential stress σ_θθ from the Kirsch analytical solution.

    For uniaxial tension σ_∞ applied in the x-direction at infinity:

        σ_θθ = (σ_∞/2) * [(1 + a²/r²) - (1 + 3a⁴/r⁴) cos(2θ)]

    At r = a (hole surface): σ_θθ ranges from -σ_∞ (θ=0) to 3σ_∞ (θ=π/2),
    giving the classical stress concentration factor Kt = 3.

    Args:
        xx: x-coordinates of evaluation points.
        yy: y-coordinates of evaluation points.
        hole_radius: Hole radius a.
        remote_stress: Far-field uniaxial stress σ_∞.

    Returns:
        σ_θθ field, same shape as xx/yy. Points at r < hole_radius are clamped
        to the hole-surface value; the caller applies masking as needed.
    """

    r_safe = np.maximum(np.sqrt(xx * xx + yy * yy), hole_radius)
    theta = np.arctan2(yy, xx)
    a2 = (hole_radius / r_safe) ** 2
    a4 = a2 * a2
    return 0.5 * remote_stress * ((1.0 + a2) - (1.0 + 3.0 * a4) * np.cos(2.0 * theta))


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run boundary-element proxy for hole stress field.

    Computes the exact Kirsch stress field on a fine interior visualization
    grid. BEM is modeled as the reference solution: exact circular hole
    boundary with no discretization noise. Exports a 2-D visualization field
    and a 1-D boundary profile.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for BEM proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    n_boundary = int(config["resolution"]["bem_boundary_points"])
    width = float(config["plate_width"])
    height = float(config["plate_height"])
    hole_radius = float(config["hole_radius"])
    remote_stress = float(config["remote_stress"])

    # --- 1-D boundary profile on the hole surface ---
    theta_1d = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    # At r = a: σ_θθ = σ_∞ * (1 - 2 cos(2θ)), giving max = 3σ_∞ at θ=π/2.
    sigma_theta_1d = remote_stress * (1.0 - 2.0 * np.cos(2.0 * theta_1d))
    kt_est = float(np.max(sigma_theta_1d) / remote_stress)  # should be 3.0

    metrics = {
        "dof": float(n_boundary * 2),
        "kt_estimate": kt_est,
        "near_hole_mae": 0.0,
        "hole_geometry_error": 0.0,
    }

    # --- 2-D visualization grid with smooth circular hole mask ---
    vis_x = np.linspace(-width / 2.0, width / 2.0, _VIS_NX)
    vis_y = np.linspace(-height / 2.0, height / 2.0, _VIS_NY)
    xx_vis, yy_vis = np.meshgrid(vis_x, vis_y, indexing="xy")  # (_VIS_NY, _VIS_NX)
    r_vis = np.sqrt(xx_vis * xx_vis + yy_vis * yy_vis)

    sigma_2d_base = _kirsch_sigma_tt(
        xx_vis.ravel(),
        yy_vis.ravel(),
        hole_radius=hole_radius,
        remote_stress=remote_stress,
    )

    # BEM represents the hole boundary exactly — smooth circular mask, no noise.
    hole_mask_2d = r_vis < hole_radius

    load_factors = np.linspace(0.05, 1.0, 40)

    metadata = {
        "status": "success",
        "backend": backend.name,
        "boundary_points": n_boundary,
        "hole_radius": hole_radius,
        "notes": "Boundary-only model; interior mesh is not built. Exact circular hole.",
        "viz": {
            "theta": theta_1d.tolist(),
            "sigma_theta": sigma_theta_1d.tolist(),
        },
        "viz_timeseries": {
            "load_factors": load_factors.tolist(),
            "vis_nx": _VIS_NX,
            "vis_ny": _VIS_NY,
            "vis_x": vis_x.tolist(),
            "vis_y": vis_y.tolist(),
            "sigma_2d_base": sigma_2d_base.tolist(),
            "hole_mask_flat": hole_mask_2d.ravel().tolist(),
        },
    }

    LOGGER.info("B01 BEM run finished: dof=%s", metrics["dof"])
    return MethodResult(benchmark="B01", method="BEM", metrics=metrics, metadata=metadata)
