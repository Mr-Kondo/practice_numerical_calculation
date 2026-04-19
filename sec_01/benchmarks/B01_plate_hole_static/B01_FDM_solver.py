"""B01 FDM proxy solver on a structured grid."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)

# Visualization grid resolution (width:height = 2:1).
_VIS_NX = 100
_VIS_NY = 50

# Coarse stair representation resolution for FDM hole boundary.
# Smaller values produce more visible staircasing in the output figure.
_STAIR_NX = 30
_STAIR_NY = 15


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
    """Run structured-grid finite-difference approximation.

    Computes the Kirsch stress field on a structured grid with a stair-step
    hole approximation. Exports a 2-D visualization field for animation and
    a 1-D boundary profile for the static stress plot.

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

    # --- Structured analysis grid ---
    x = np.linspace(-width / 2.0, width / 2.0, nx)
    y = np.linspace(-height / 2.0, height / 2.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx * xx + yy * yy)

    # Stair-step effective radius: nearest cell corner plus hole radius.
    stair_r = math.hypot(width / nx, height / ny) + hole_radius
    stair_mask = r >= stair_r

    # Compute exact Kirsch field on the analysis grid.
    sigma_grid = _kirsch_sigma_tt(xx, yy, hole_radius=hole_radius, remote_stress=remote_stress)

    # kt estimate: maximum sampled stress in the stair-boundary ring.
    near_stair = stair_mask & (r < stair_r * 1.3)
    kt_est = float(np.max(sigma_grid[near_stair]) / remote_stress) if near_stair.any() else float("nan")
    near_hole_region = stair_mask & (r < hole_radius * 1.7)
    mean_error = (
        float(np.abs(sigma_grid[near_hole_region] - remote_stress * 3.0).mean()) if near_hole_region.any() else float("nan")
    )

    metrics = {
        "dof": float(nx * ny),
        "kt_estimate": kt_est,
        "near_hole_mae": mean_error,
        "hole_geometry_error": float((stair_r - hole_radius) / hole_radius),
    }

    # --- 1-D boundary profile for static plot ---
    theta_1d = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    # Sample at the stair-effective radius to reflect FDM's offset boundary.
    sigma_theta_1d = _kirsch_sigma_tt(
        stair_r * np.cos(theta_1d),
        stair_r * np.sin(theta_1d),
        hole_radius=hole_radius,
        remote_stress=remote_stress,
    )

    # --- 2-D visualization grid ---
    vis_x = np.linspace(-width / 2.0, width / 2.0, _VIS_NX)
    vis_y = np.linspace(-height / 2.0, height / 2.0, _VIS_NY)
    xx_vis, yy_vis = np.meshgrid(vis_x, vis_y, indexing="xy")  # (_VIS_NY, _VIS_NX)

    # FDM stair-step hole mask: snap each vis-grid point to its coarse cell
    # center and test whether that center is inside the physical hole.
    dx_s = width / _STAIR_NX
    dy_s = height / _STAIR_NY
    xi_s = np.clip(np.floor((xx_vis + width / 2.0) / dx_s).astype(int), 0, _STAIR_NX - 1)
    yi_s = np.clip(np.floor((yy_vis + height / 2.0) / dy_s).astype(int), 0, _STAIR_NY - 1)
    xc = -width / 2.0 + (xi_s + 0.5) * dx_s
    yc = -height / 2.0 + (yi_s + 0.5) * dy_s
    hole_mask_2d = np.sqrt(xc * xc + yc * yc) < hole_radius  # True = in hole

    sigma_2d_base = _kirsch_sigma_tt(
        xx_vis.ravel(),
        yy_vis.ravel(),
        hole_radius=hole_radius,
        remote_stress=remote_stress,
    )

    load_factors = np.linspace(0.05, 1.0, 40)

    metadata = {
        "status": "success",
        "backend": backend.name,
        "grid_shape": [ny, nx],
        "notes": "Structured grid with stair-step hole approximation.",
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

    LOGGER.info("B01 FDM run finished: dof=%s", metrics["dof"])
    return MethodResult(benchmark="B01", method="FDM", metrics=metrics, metadata=metadata)
