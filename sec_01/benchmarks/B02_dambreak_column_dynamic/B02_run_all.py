"""Run and compare all B02 methods."""

from __future__ import annotations

from pathlib import Path

from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_DEM_solver import run as run_dem
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_FEM_solver import run as run_fem
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_FVM_solver import run as run_fvm
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import run as run_sph
from sec_01.shared.io import read_yaml
from sec_01.shared.runtime import (
    Timer,
    configure_logging,
    save_metrics_table,
    save_result,
    seed_everything,
)


def main() -> None:
    """Execute B02 comparison and persist outputs."""

    configure_logging()
    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)

    seed_everything(int(cfg["seed"]))

    outputs = Path(__file__).resolve().parents[2] / "outputs" / cfg["output_subdir"]
    prefer_gpu = bool(cfg.get("prefer_gpu", False))

    results = []
    for runner in (run_fem, run_fvm, run_sph, run_dem):
        with Timer() as timer:
            result = runner(cfg, prefer_gpu=prefer_gpu)
        result.metrics["wall_time_s"] = timer.elapsed_seconds
        save_result(result, outputs)
        results.append(result)

    save_metrics_table(results, outputs / "B02_metrics.csv")


if __name__ == "__main__":
    main()
