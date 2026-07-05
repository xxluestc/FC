"""Audit which causal feature groups actually exist in processed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GROUPS = {
    "base_motion": [
        "timestamp",
        "segment_id",
        "speed_kmh",
        "speed_mps",
        "acceleration_mps2",
        "speed_smooth_mps",
        "acceleration_smooth_mps2",
        "p_dem_measured_kw",
        "soc_pct",
    ],
    "route_stop_source": ["odometer_km"],
    "control_intent": [
        "target_power_kw",
        "loadable_power_kw",
        "dcdc_target_current_a",
        "fuel_cell_state",
    ],
    "power_chain": [
        "fc_voltage_v",
        "fc_current_a",
        "dcdc_output_voltage_v",
        "dcdc_output_current_a",
        "battery_voltage_v",
        "battery_current_a",
        "motor_voltage_v",
        "motor_current_a",
        "fc_input_power_kw",
        "dcdc_output_power_kw",
        "battery_power_kw_raw_sign",
        "motor_power_kw_raw_sign",
    ],
    "aux_thermal_air": [
        "air_flow",
        "air_demand_flow",
        "air_compressor_power",
        "water_pump_power",
        "hydrogen_pump_power",
        "temperature",
        "pressure",
        "fan_power",
        "electric_heater_power",
    ],
    "health_state": [
        "mean_cell_voltage_v",
        "min_cell_voltage_v",
        "max_cell_voltage_v",
        "cell_voltage_deviation_v",
        "cell_voltage_array",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(args.canonical)
    processed = pd.read_csv(args.processed)
    rows = []
    for group, requested in GROUPS.items():
        for field in requested:
            source = (
                "processed"
                if field in processed
                else "canonical" if field in canonical else "absent"
            )
            values = (
                processed[field]
                if field in processed
                else canonical[field] if field in canonical else None
            )
            rows.append(
                {
                    "feature_group": group,
                    "field": field,
                    "availability": source != "absent",
                    "source_table": source,
                    "dtype": str(values.dtype) if values is not None else "",
                    "missing_pct": (
                        100 * values.isna().mean() if values is not None else None
                    ),
                    "unique_values": (
                        values.nunique(dropna=True) if values is not None else None
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(args.out_dir / "processed_feature_availability.csv", index=False)
    summary = {
        "canonical_rows": len(canonical),
        "processed_rows": len(processed),
        "canonical_columns": canonical.columns.tolist(),
        "processed_only_columns": [c for c in processed if c not in canonical],
        "available_by_group": {
            group: audit.loc[
                (audit.feature_group == group) & audit.availability, "field"
            ].tolist()
            for group in GROUPS
        },
        "absent_by_group": {
            group: audit.loc[
                (audit.feature_group == group) & ~audit.availability, "field"
            ].tolist()
            for group in GROUPS
        },
        "derived_at_prediction_time": [
            "stop_duration",
            "time_since_last_stop",
            "distance_since_last_stop",
            "segment_elapsed_time",
            "segment_elapsed_distance",
        ],
    }
    (args.out_dir / "processed_feature_availability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
