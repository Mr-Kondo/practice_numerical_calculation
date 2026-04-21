"""Generate matplotlib visualizations for B01 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sec_01.shared.visualization import ensure_fig_dir, load_result_jsons, save_bar_chart


def main() -> None:
    """Create B01 plots from saved method JSON outputs."""

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "B01"
    fig_dir = ensure_fig_dir(output_dir)
    result_map = load_result_jsons(output_dir, benchmark="B01")

    kt_labels: list[str] = []
    kt_values: list[float] = []

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for method, payload in sorted(result_map.items()):
        viz = payload.get("metadata", {}).get("viz", {})
        theta = viz.get("theta", [])
        sigma = viz.get("sigma_theta", [])
        if theta and sigma:
            ax.plot(theta, sigma, label=method)
        kt_labels.append(method)
        kt_values.append(float(payload.get("metrics", {}).get("kt_estimate", float("nan"))))

    ax.set_title("B01: Circumferential Stress Trend Near Hole")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("proxy sigma_theta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "B01_method_stress_profiles.png")
    plt.close(fig)

    save_bar_chart(
        labels=kt_labels,
        values=kt_values,
        output_path=fig_dir / "B01_kt_comparison.png",
        title="B01: Stress Concentration Estimate Comparison",
        ylabel="kt_estimate",
    )


if __name__ == "__main__":
    main()
