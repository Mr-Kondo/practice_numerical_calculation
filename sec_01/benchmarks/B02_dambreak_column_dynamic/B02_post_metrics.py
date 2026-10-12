"""Print each B02 model's diagnostics without cross-model acceptance."""

from __future__ import annotations

import csv
from pathlib import Path


def _value(row: dict[str, str], key: str) -> float:
    """Get float value from optional CSV column."""

    raw = row.get(key)
    if raw is None or raw == "nan":
        return float("nan")
    return float(raw)


def main() -> None:
    """Print method summaries sorted by completion and runtime."""

    csv_path = Path(__file__).resolve().parents[2] / "outputs" / "B02" / "B02_metrics.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("B02 model-specific diagnostics (completion and SPH mass balance are not physical validation):")
    ranked = sorted(rows, key=lambda x: (-_value(x, "completion_flag"), _value(x, "wall_time_s")))
    for row in ranked:
        retained_mass = _value(row, "retained_mass_fraction")
        escaped_mass = _value(row, "escaped_mass_fraction")
        print(
            f"{row['method']}: completion={_value(row, 'completion_flag'):.0f}, "
            f"mass_budget_error={_value(row, 'mass_error'):.3g}, "
            f"retained_mass={retained_mass:.3g}, "
            f"escaped_mass={escaped_mass:.3g}, "
            f"front_arrival_time={_value(row, 'front_arrival_time'):.3g}, "
            f"max_runup_like_height={_value(row, 'max_runup_like_height'):.3g}, "
            f"vorticity_peak={_value(row, 'vorticity_peak'):.3g}, "
            f"vortex_area_peak={_value(row, 'vortex_area_peak'):.3g}, "
            f"vortex_duration_s={_value(row, 'vortex_duration_s'):.3g}, "
            f"peak_particle_speed={_value(row, 'peak_particle_speed'):.3g}"
        )


if __name__ == "__main__":
    main()
