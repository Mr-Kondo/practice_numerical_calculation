"""Post-process B02 method metrics for robustness comparison."""

from __future__ import annotations

import csv
from pathlib import Path

from sec_01.shared.io import read_yaml


def _value(row: dict[str, str], key: str) -> float:
    """Get float value from optional CSV column."""

    raw = row.get(key)
    if raw is None or raw == "nan":
        return float("nan")
    return float(raw)


def main() -> None:
    """Print method summaries sorted by completion and runtime."""

    cfg_path = Path(__file__).with_name("B02_common_cfg.yaml")
    cfg = read_yaml(cfg_path)
    accept_arrival_diff = float(cfg.get("accept_arrival_diff_s", 0.05))

    csv_path = Path(__file__).resolve().parents[2] / "outputs" / "B02" / "B02_metrics.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    rows_by_method = {row.get("method", ""): row for row in rows}
    fvm_arrival = _value(rows_by_method.get("FVM", {}), "front_arrival_time")
    sph_arrival = _value(rows_by_method.get("SPH", {}), "front_arrival_time")
    arrival_diff = float("nan")
    if not (fvm_arrival != fvm_arrival or sph_arrival != sph_arrival):
        arrival_diff = abs(fvm_arrival - sph_arrival)

    for row in rows:
        method = row.get("method", "")
        row["arrival_diff_fvm_sph"] = "nan" if arrival_diff != arrival_diff else f"{arrival_diff:.8g}"
        if method in {"FVM", "SPH"}:
            base_accept = _value(row, "acceptance_pass")
            vortex_pass = _value(row, "vortex_pass")
            final_accept = 0.0
            if method == "FVM" and base_accept > 0.5 and arrival_diff != arrival_diff:
                final_accept = 1.0 if vortex_pass > 0.5 else 0.0
            elif base_accept > 0.5 and arrival_diff <= accept_arrival_diff:
                final_accept = 1.0
            row["acceptance_pass_final"] = f"{final_accept:.8g}"
        else:
            row["acceptance_pass_final"] = "nan"

    fields = original_fields[:]
    for extra in ("arrival_diff_fvm_sph", "acceptance_pass_final"):
        if extra not in fields:
            fields.append(extra)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda x: (-_value(x, "completion_flag"), _value(x, "wall_time_s")))
    for row in ranked:
        retained_mass = _value(row, "retained_mass_fraction")
        escaped_mass = _value(row, "escaped_mass_fraction")
        print(
            f"{row['method']}: completion={_value(row, 'completion_flag'):.0f}, "
            f"mass_error={_value(row, 'mass_error'):.3g}, "
            f"retained_mass={retained_mass:.3g}, "
            f"escaped_mass={escaped_mass:.3g}, "
            f"front_arrival_time={_value(row, 'front_arrival_time'):.3g}, "
            f"arrival_diff_fvm_sph={_value(row, 'arrival_diff_fvm_sph'):.3g}, "
            f"max_runup_like_height={_value(row, 'max_runup_like_height'):.3g}, "
            f"vorticity_peak={_value(row, 'vorticity_peak'):.3g}, "
            f"vortex_area_peak={_value(row, 'vortex_area_peak'):.3g}, "
            f"vortex_duration_s={_value(row, 'vortex_duration_s'):.3g}, "
            f"vortex_pass={_value(row, 'vortex_pass'):.0f}, "
            f"peak_particle_speed={_value(row, 'peak_particle_speed'):.3g}, "
            f"acceptance_pass={_value(row, 'acceptance_pass'):.0f}, "
            f"acceptance_pass_final={_value(row, 'acceptance_pass_final'):.0f}, "
            f"mesh_min_quality={_value(row, 'mesh_min_quality'):.3g}"
        )


if __name__ == "__main__":
    main()
