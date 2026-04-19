"""B02 SPH proxy solver with optional MPS tensor acceleration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sec_01.shared.gpu_backend import select_backend
from sec_01.shared.runtime import MethodResult

LOGGER = logging.getLogger(__name__)


def _run_numpy(particles: np.ndarray, velocities: np.ndarray, steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance particles with simple pressure-repulsion proxy."""

    for _ in range(steps):
        center = particles.mean(axis=0)
        direction = particles - center
        norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1.0e-8
        force = direction / norm
        velocities = 0.995 * velocities + dt * force
        particles = particles + dt * velocities
    return particles, velocities


def _run_torch_mps(particles: np.ndarray, velocities: np.ndarray, steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance particles using torch backend when available."""

    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    p = torch.tensor(particles, dtype=torch.float32, device=device)
    v = torch.tensor(velocities, dtype=torch.float32, device=device)

    for _ in range(steps):
        center = p.mean(dim=0, keepdim=True)
        direction = p - center
        norm = torch.linalg.norm(direction, dim=1, keepdim=True) + 1.0e-8
        force = direction / norm
        v = 0.995 * v + dt * force
        p = p + dt * v

    return p.cpu().numpy(), v.cpu().numpy()


def run(config: dict[str, Any], prefer_gpu: bool) -> MethodResult:
    """Run particle-based SPH proxy.

    Args:
        config: Benchmark configuration.
        prefer_gpu: Whether GPU should be attempted.

    Returns:
        Method result for SPH proxy.
    """

    backend = select_backend(prefer_gpu=prefer_gpu)
    rng = np.random.default_rng(int(config["seed"]))
    n_particles = int(config["sph_particles"])
    steps = int(config["steps"])
    dt = float(config["dt"])

    particles = np.column_stack(
        [
            rng.uniform(0.0, float(config["dam_width_fraction"]), n_particles),
            rng.uniform(0.0, 0.8, n_particles),
        ]
    )
    velocities = np.zeros_like(particles)

    if backend.name == "torch-mps":
        particles, velocities = _run_torch_mps(particles, velocities, steps=steps, dt=dt)
    else:
        particles, velocities = _run_numpy(particles, velocities, steps=steps, dt=dt)

    spread = float(np.quantile(particles[:, 0], 0.95) - np.quantile(particles[:, 0], 0.05))
    speed = np.linalg.norm(velocities, axis=1)

    metrics = {
        "dof": float(n_particles * 2),
        "mass_error": 0.0,
        "splash_spread_width": spread,
        "completion_flag": 1.0,
        "peak_particle_speed": float(np.max(speed)),
    }

    metadata = {
        "backend": backend.name,
        "notes": "Particle method naturally handles topology changes.",
    }

    LOGGER.info("B02 SPH run finished: spread=%.4f", spread)
    return MethodResult(benchmark="B02", method="SPH", metrics=metrics, metadata=metadata)
