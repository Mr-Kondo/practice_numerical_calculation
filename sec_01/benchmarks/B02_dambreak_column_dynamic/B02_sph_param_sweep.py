"""Run an acceptance-focused SPH parameter sweep for B02.

This script targets wall reflection and artificial viscosity parameters and
compares each SPH case against one FVM baseline arrival time.
"""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_FVM_solver import run as run_fvm
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import run as run_sph
from sec_01.shared.io import read_yaml
from sec_01.shared.runtime import Timer, configure_logging, save_result, seed_everything

LOGGER = logging.getLogger(__name__)


def _is_nan(value: float) -> bool:
    """Return True if the input is NaN.

    Args:
        value: Scalar float value.

    Returns:
        True when value is NaN.
    """

    return value != value


def _wall_profiles() -> list[dict[str, float]]:
    """Return wall-parameter profile list.

    Returns:
        List of wall parameter dictionaries.
    """

    return [
        {
            "sph_floor_restitution": 0.00,
            "sph_left_wall_restitution": 0.00,
            "sph_right_wall_restitution": 0.00,
            "sph_floor_friction": 0.98,
        },
        {
            "sph_floor_restitution": 0.02,
            "sph_left_wall_restitution": 0.04,
            "sph_right_wall_restitution": 0.04,
            "sph_floor_friction": 0.95,
        },
        {
            "sph_floor_restitution": 0.05,
            "sph_left_wall_restitution": 0.08,
            "sph_right_wall_restitution": 0.08,
            "sph_floor_friction": 0.90,
        },
        {
            "sph_floor_restitution": 0.02,
            "sph_left_wall_restitution": 0.00,
            "sph_right_wall_restitution": 0.00,
            "sph_floor_friction": 0.98,
        },
    ]


def _viscosity_profiles() -> list[dict[str, float]]:
    """Return viscosity-parameter profile list.

    Returns:
        List of viscosity parameter dictionaries.
    """

    return [
        {"sph_alpha_visc": 0.03, "sph_beta_visc": 0.00},
        {"sph_alpha_visc": 0.05, "sph_beta_visc": 0.10},
        {"sph_alpha_visc": 0.08, "sph_beta_visc": 0.20},
        {"sph_alpha_visc": 0.05, "sph_beta_visc": 0.00},
    ]


def _build_cases() -> list[tuple[str, dict[str, float]]]:
    """Create 16 sweep cases from wall and viscosity profiles.

    Returns:
        List of tuples containing case name and parameter override map.
    """

    cases: list[tuple[str, dict[str, float]]] = []
    walls = _wall_profiles()
    viscs = _viscosity_profiles()
    for wall_idx, visc_idx in product(range(len(walls)), range(len(viscs))):
        case_name = f"W{wall_idx + 1:02d}_V{visc_idx + 1:02d}"
        params: dict[str, float] = {}
        params.update(walls[wall_idx])
        params.update(viscs[visc_idx])
        cases.append((case_name, params))
    return cases


def _run_fvm_baseline(config: dict[str, Any], prefer_gpu: bool) -> tuple[float, float]:
    """Run FVM once and return arrival time and wall clock.

    Args:
        config: B02 configuration dictionary.
        prefer_gpu: Backend preference flag.

    Returns:
        Tuple of (front_arrival_time, wall_time_s).
    """

    with Timer() as timer:
        fvm_result = run_fvm(config, prefer_gpu=prefer_gpu)
    arrival = float(fvm_result.metrics.get("front_arrival_time", float("nan")))
    return arrival, timer.elapsed_seconds


def _final_acceptance(sph_metrics: dict[str, float], fvm_arrival: float, max_diff: float) -> float:
    """Compute final acceptance using existing B02 post criteria.

    Args:
        sph_metrics: SPH scalar metrics.
        fvm_arrival: FVM baseline arrival time.
        max_diff: Allowed arrival-time difference.

    Returns:
        Final acceptance flag as float in {0.0, 1.0}.
    """

    base_accept = float(sph_metrics.get("acceptance_pass", 0.0))
    sph_arrival = float(sph_metrics.get("front_arrival_time", float("nan")))
    if base_accept <= 0.5 or _is_nan(fvm_arrival) or _is_nan(sph_arrival):
        return 0.0
    diff = abs(fvm_arrival - sph_arrival)
    return 1.0 if diff <= max_diff else 0.0


def main() -> None:
    """Execute SPH sweep and write a compact summary CSV."""

    configure_logging()

    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)

    seed_everything(int(cfg["seed"]))

    output_root = Path(__file__).resolve().parents[2] / "outputs" / cfg["output_subdir"]
    sweep_dir = output_root / "sweeps"
    summary_path = sweep_dir / "B02_sph_sweep_summary.csv"

    prefer_gpu = bool(cfg.get("prefer_gpu", False))
    accept_arrival_diff = float(cfg.get("accept_arrival_diff_s", 0.05))

    fvm_arrival, fvm_wall_time = _run_fvm_baseline(cfg, prefer_gpu=prefer_gpu)
    LOGGER.info("B02 sweep baseline FVM done: arrival=%.6f, wall_time=%.3fs", fvm_arrival, fvm_wall_time)

    cases = _build_cases()
    rows: list[dict[str, float | str]] = []

    for case_name, override in cases:
        case_cfg = deepcopy(cfg)
        case_cfg.update(override)
        with Timer() as timer:
            result = run_sph(case_cfg, prefer_gpu=prefer_gpu)
        result.metrics["wall_time_s"] = timer.elapsed_seconds
        result.method = f"SPH_{case_name}"
        save_result(result, sweep_dir)

        metrics = result.metrics
        arrival = float(metrics.get("front_arrival_time", float("nan")))
        arrival_diff = abs(fvm_arrival - arrival) if not (_is_nan(fvm_arrival) or _is_nan(arrival)) else float("nan")
        accept_final = _final_acceptance(
            sph_metrics=metrics,
            fvm_arrival=fvm_arrival,
            max_diff=accept_arrival_diff,
        )

        rows.append(
            {
                "case": case_name,
                "method": result.method,
                "fvm_front_arrival_time": fvm_arrival,
                "sph_front_arrival_time": arrival,
                "arrival_diff_fvm_sph": arrival_diff,
                "acceptance_pass": float(metrics.get("acceptance_pass", 0.0)),
                "acceptance_pass_final": accept_final,
                "rebound_flag": float(metrics.get("rebound_flag", 0.0)),
                "rebound_drop": float(metrics.get("rebound_drop", float("nan"))),
                "retained_mass_fraction": float(metrics.get("retained_mass_fraction", float("nan"))),
                "mass_error": float(metrics.get("mass_error", float("nan"))),
                "peak_particle_speed": float(metrics.get("peak_particle_speed", float("nan"))),
                "wall_time_s": float(metrics.get("wall_time_s", float("nan"))),
                "sph_floor_restitution": float(override["sph_floor_restitution"]),
                "sph_left_wall_restitution": float(override["sph_left_wall_restitution"]),
                "sph_right_wall_restitution": float(override["sph_right_wall_restitution"]),
                "sph_floor_friction": float(override["sph_floor_friction"]),
                "sph_alpha_visc": float(override["sph_alpha_visc"]),
                "sph_beta_visc": float(override["sph_beta_visc"]),
            }
        )
        LOGGER.info(
            "Sweep case %s: accept=%.0f final=%.0f arrival_diff=%.6f rebound=%.0f",
            case_name,
            float(metrics.get("acceptance_pass", 0.0)),
            accept_final,
            arrival_diff,
            float(metrics.get("rebound_flag", 0.0)),
        )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "method",
        "fvm_front_arrival_time",
        "sph_front_arrival_time",
        "arrival_diff_fvm_sph",
        "acceptance_pass",
        "acceptance_pass_final",
        "rebound_flag",
        "rebound_drop",
        "retained_mass_fraction",
        "mass_error",
        "peak_particle_speed",
        "wall_time_s",
        "sph_floor_restitution",
        "sph_left_wall_restitution",
        "sph_right_wall_restitution",
        "sph_floor_friction",
        "sph_alpha_visc",
        "sph_beta_visc",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passing = sum(1 for row in rows if float(row["acceptance_pass_final"]) > 0.5)
    LOGGER.info("B02 SPH sweep done: %d/%d cases pass final acceptance.", passing, len(rows))
    print(f"B02 SPH sweep done: {passing}/{len(rows)} cases pass final acceptance.")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
