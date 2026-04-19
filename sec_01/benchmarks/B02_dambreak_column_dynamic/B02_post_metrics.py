"""Post-process B02 method metrics for robustness comparison."""

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

    ranked = sorted(rows, key=lambda x: (-_value(x, "completion_flag"), _value(x, "wall_time_s")))
    for row in ranked:
        print(
            f"{row['method']}: completion={_value(row, 'completion_flag'):.0f}, "
            f"mass_error={_value(row, 'mass_error'):.3g}, "
            f"broken_bonds={_value(row, 'broken_bonds'):.0f}, "
            f"mesh_min_quality={_value(row, 'mesh_min_quality'):.3g}"
        )


if __name__ == "__main__":
    main()
