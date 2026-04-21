"""Run a focused SPH right-wall parameter scan for B02.

This utility sweeps only right-wall parameters and writes a CSV summary
for quick calibration of rebound behavior.
"""

from __future__ import annotations

import argparse
import csv
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import run as run_sph
from sec_01.shared.io import read_yaml
from sec_01.shared.runtime import Timer, configure_logging, seed_everything

LOGGER = logging.getLogger(__name__)


def _parse_values(text: str) -> list[float]:
    """Parse comma-separated float values.

    Args:
        text: Comma-separated float string.

    Returns:
        Parsed float list.
    """

    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""

    parser = argparse.ArgumentParser(description="Focused B02 SPH right-wall scan")
    parser.add_argument(
        "--mins",
        type=str,
        default="0.05,0.08,0.12",
        help="Comma-separated values for sph_right_wall_restitution_min",
    )
    parser.add_argument(
        "--speed-refs",
        type=str,
        default="0.6,1.0,1.4",
        help="Comma-separated values for sph_right_wall_speed_ref",
    )
    parser.add_argument(
        "--dampings",
        type=str,
        default="0.08",
        help="Comma-separated values for sph_right_wall_tangent_damping",
    )
    parser.add_argument(
        "--target-sim-time",
        type=float,
        default=None,
        help="Optional temporary override for target_sim_time_s during scan",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional temporary override for max_steps during scan",
    )
    parser.add_argument(
        "--front-quantile",
        type=float,
        default=None,
        help="Optional temporary override for sph_front_quantile during scan",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    return parser


def _to_row(
    min_restitution: float,
    speed_ref: float,
    tangential_damping: float,
    metrics: dict[str, float],
    wall_time_s: float,
) -> dict[str, float]:
    """Build a CSV row from one scan result."""

    return {
        "sph_right_wall_restitution_min": min_restitution,
        "sph_right_wall_speed_ref": speed_ref,
        "sph_right_wall_tangent_damping": tangential_damping,
        "acceptance_pass": float(metrics.get("acceptance_pass", 0.0)),
        "rebound_flag": float(metrics.get("rebound_flag", 0.0)),
        "rebound_drop": float(metrics.get("rebound_drop", float("nan"))),
        "front_arrival_time": float(metrics.get("front_arrival_time", float("nan"))),
        "mass_error": float(metrics.get("mass_error", float("nan"))),
        "retained_mass_fraction": float(metrics.get("retained_mass_fraction", float("nan"))),
        "peak_particle_speed": float(metrics.get("peak_particle_speed", float("nan"))),
        "wall_time_s": wall_time_s,
    }


def main() -> None:
    """Run a focused right-wall parameter scan and print summary."""

    configure_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg: dict[str, Any] = read_yaml(cfg_path)
    seed_everything(int(cfg["seed"]))

    mins = _parse_values(args.mins)
    speed_refs = _parse_values(args.speed_refs)
    dampings = _parse_values(args.dampings)

    if args.target_sim_time is not None:
        cfg["target_sim_time_s"] = float(args.target_sim_time)
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.front_quantile is not None:
        cfg["sph_front_quantile"] = float(args.front_quantile)

    output_root = Path(__file__).resolve().parents[2] / "outputs" / str(cfg["output_subdir"]) / "sweeps"
    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or (output_root / "B02_right_wall_scan.csv")

    prefer_gpu = bool(cfg.get("prefer_gpu", False))
    rows: list[dict[str, float]] = []

    for min_restitution, speed_ref, tangential_damping in product(mins, speed_refs, dampings):
        case_cfg = deepcopy(cfg)
        case_cfg["sph_right_wall_restitution_min"] = float(min_restitution)
        case_cfg["sph_right_wall_speed_ref"] = float(speed_ref)
        case_cfg["sph_right_wall_tangent_damping"] = float(tangential_damping)

        with Timer() as timer:
            result = run_sph(case_cfg, prefer_gpu=prefer_gpu)

        row = _to_row(
            min_restitution=float(min_restitution),
            speed_ref=float(speed_ref),
            tangential_damping=float(tangential_damping),
            metrics=result.metrics,
            wall_time_s=timer.elapsed_seconds,
        )
        rows.append(row)

        LOGGER.info(
            "scan min=%.3f speed_ref=%.3f damping=%.3f -> accept=%.0f rebound=%.0f drop=%.3f",
            row["sph_right_wall_restitution_min"],
            row["sph_right_wall_speed_ref"],
            row["sph_right_wall_tangent_damping"],
            row["acceptance_pass"],
            row["rebound_flag"],
            row["rebound_drop"],
        )

    rows.sort(key=lambda item: (item["acceptance_pass"], item["rebound_flag"], item["rebound_drop"]), reverse=True)

    fieldnames = [
        "sph_right_wall_restitution_min",
        "sph_right_wall_speed_ref",
        "sph_right_wall_tangent_damping",
        "acceptance_pass",
        "rebound_flag",
        "rebound_drop",
        "front_arrival_time",
        "mass_error",
        "retained_mass_fraction",
        "peak_particle_speed",
        "wall_time_s",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["acceptance_pass"] > 0.5)
    rebounds = sum(1 for row in rows if row["rebound_flag"] > 0.5)

    print(f"scan cases={len(rows)} accept_pass={passed} rebound_pass={rebounds}")
    print(f"summary_csv={output_csv}")
    if rows:
        top = rows[0]
        print(
            "best="
            f"min={top['sph_right_wall_restitution_min']:.3f},"
            f"speed_ref={top['sph_right_wall_speed_ref']:.3f},"
            f"damping={top['sph_right_wall_tangent_damping']:.3f},"
            f"accept={top['acceptance_pass']:.0f},"
            f"rebound={top['rebound_flag']:.0f},"
            f"drop={top['rebound_drop']:.3f}"
        )


if __name__ == "__main__":
    main()
