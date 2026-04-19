"""Run and compare all B01 methods."""

from __future__ import annotations

from pathlib import Path

from sec_01.benchmarks.B01_plate_hole_static.B01_BEM_solver import run as run_bem
from sec_01.benchmarks.B01_plate_hole_static.B01_FDM_solver import run as run_fdm
from sec_01.benchmarks.B01_plate_hole_static.B01_FEM_solver import run as run_fem
from sec_01.shared.io import read_yaml
from sec_01.shared.runtime import (
    Timer,
    configure_logging,
    create_failure_result,
    save_metrics_table,
    save_result,
    seed_everything,
)


def main() -> None:
    """Execute B01 comparison and persist outputs."""

    configure_logging()
    cfg_path = Path(__file__).with_name("B01_common_cfg.yaml")
    cfg = read_yaml(cfg_path)

    seed_everything(int(cfg["seed"]))

    outputs = Path(__file__).resolve().parents[2] / "outputs" / cfg["output_subdir"]
    prefer_gpu = bool(cfg.get("prefer_gpu", False))

    results = []
    for runner in (run_fdm, run_fem, run_bem):
        method_name = runner.__name__.removeprefix("run_").upper()
        try:
            with Timer() as timer:
                result = runner(cfg, prefer_gpu=prefer_gpu)
            result.metrics["wall_time_s"] = timer.elapsed_seconds
            result.metrics.setdefault("completion_flag", 1.0)
            result.metadata.setdefault("status", "success")
        except Exception as exc:  # pylint: disable=broad-except
            result = create_failure_result(
                benchmark="B01",
                method=method_name,
                error_message=str(exc),
            )
        save_result(result, outputs)
        results.append(result)

    save_metrics_table(results, outputs / "B01_metrics.csv")

    success_count = sum(int(item.metrics.get("completion_flag", 0.0) > 0.0) for item in results)
    print(f"B01 done: {success_count}/{len(results)} methods succeeded.")


if __name__ == "__main__":
    main()
