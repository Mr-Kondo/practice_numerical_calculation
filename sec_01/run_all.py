"""Run both benchmarks and generate method-comparison outputs."""

from __future__ import annotations

from sec_01.benchmarks.B01_plate_hole_static.B01_post_metrics import main as b01_post
from sec_01.benchmarks.B01_plate_hole_static.B01_run_all import main as b01_run
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_post_metrics import main as b02_post
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_run_all import main as b02_run


def main() -> None:
    """Run all benchmarks and print concise summaries."""

    b01_run()
    b02_run()

    print("=== B01 Summary ===")
    b01_post()
    print("=== B02 Summary ===")
    b02_post()


if __name__ == "__main__":
    main()
