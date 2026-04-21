"""Run both benchmarks and generate method-comparison outputs."""

from __future__ import annotations

from sec_01.benchmarks.B01_plate_hole_static.B01_post_metrics import main as b01_post
from sec_01.benchmarks.B01_plate_hole_static.B01_animate_results import main as b01_animate
from sec_01.benchmarks.B01_plate_hole_static.B01_plot_results import main as b01_plot
from sec_01.benchmarks.B01_plate_hole_static.B01_run_all import main as b01_run
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_animate_sideview import main as b02_animate
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_post_metrics import main as b02_post
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_plot_results import main as b02_plot
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_run_all import main as b02_run


def main() -> None:
    """Run all benchmarks and print concise summaries."""

    b01_run()
    b02_run()
    b01_plot()
    b02_plot()
    b01_animate()
    b02_animate()

    print("=== B01 Summary ===")
    b01_post()
    print("=== B02 Summary ===")
    b02_post()
    print("Saved figures:")
    print("- sec_01/outputs/B01/figs")
    print("- sec_01/outputs/B02/figs")
    print("Saved animations:")
    print("- sec_01/outputs/B01/animations")
    print("- sec_01/outputs/B02/animations")


if __name__ == "__main__":
    main()
