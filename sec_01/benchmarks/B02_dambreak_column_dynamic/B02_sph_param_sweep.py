"""Explore B02 SPH wall and artificial-viscosity diagnostics."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path

from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import run as run_sph
from sec_01.shared.io import read_yaml
from sec_01.shared.runtime import Timer, configure_logging, save_result, seed_everything

LOGGER = logging.getLogger(__name__)


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


def main() -> None:
    """Execute SPH sweep and write a compact summary CSV."""

    configure_logging()

    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)

    seed_everything(int(cfg["seed"]))

    output_root = Path(__file__).resolve().parents[2] / "outputs" / cfg["output_subdir"]
    sweep_dir = output_root / "sweeps"
    summary_path = sweep_dir / "B02_sph_sweep_summary.csv"


    cases = _build_cases()
    rows: list[dict[str, float | str]] = []

    for case_name, override in cases:
        case_cfg = deepcopy(cfg)
        case_cfg.update(override)
        with Timer() as timer:
            result = run_sph(case_cfg, prefer_gpu=False)
        result.metrics["wall_time_s"] = timer.elapsed_seconds
        result.method = f"SPH_{case_name}"
        save_result(result, sweep_dir)

        metrics = result.metrics
        rows.append(
            {
                "case": case_name,
                "method": result.method,
                "front_arrival_time": float(metrics.get("front_arrival_time", float("nan"))),
                "completion_flag": float(metrics.get("completion_flag", 0.0)),
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
            "Sweep case %s: completion=%.0f rebound=%.0f",
            case_name,
            float(metrics.get("completion_flag", 0.0)),
            float(metrics.get("rebound_flag", 0.0)),
        )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "method",
        "front_arrival_time",
        "completion_flag",
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

    completed = sum(1 for row in rows if float(row["completion_flag"]) > 0.5)
    LOGGER.info("B02 SPH sweep done: %d/%d cases completed.", completed, len(rows))
    print(f"B02 SPH sweep done: {completed}/{len(rows)} cases completed.")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
