"""Post-process B01 method metrics for human-readable ranking."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    """Print sorted B01 ranking by near-hole error."""

    csv_path = Path(__file__).resolve().parents[2] / "outputs" / "B01" / "B01_metrics.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ranked = sorted(rows, key=lambda x: float(x["near_hole_mae"]))
    for row in ranked:
        print(
            f"{row['method']}: near_hole_mae={float(row['near_hole_mae']):.4g}, "
            f"kt_estimate={float(row['kt_estimate']):.4g}, dof={float(row['dof']):.0f}"
        )


if __name__ == "__main__":
    main()
