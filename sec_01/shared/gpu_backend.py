"""Backend utility for optional Apple Silicon MPS acceleration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

LOGGER = logging.getLogger(__name__)

BackendName = Literal["numpy", "torch-cpu", "torch-mps"]


@dataclass
class BackendInfo:
    """Selected computational backend."""

    name: BackendName
    torch_available: bool
    mps_available: bool


def select_backend(prefer_gpu: bool) -> BackendInfo:
    """Select numerical backend with safe fallback.

    Args:
        prefer_gpu: Whether GPU acceleration should be attempted.

    Returns:
        Backend selection summary.
    """

    try:
        import torch
    except ImportError:
        return BackendInfo(name="numpy", torch_available=False, mps_available=False)

    mps_available = bool(torch.backends.mps.is_available())
    if prefer_gpu and mps_available:
        LOGGER.info("Using torch MPS backend.")
        return BackendInfo(name="torch-mps", torch_available=True, mps_available=True)

    LOGGER.info("Using torch CPU backend.")
    return BackendInfo(name="torch-cpu", torch_available=True, mps_available=mps_available)
