"""Validate that the committed/local experiment artifacts form a consistent run."""

from pathlib import Path
import argparse

import pandas as pd


def require_columns(path: Path, expected: set[str]) -> None:
    frame = pd.read_csv(path, nrows=10)
    missing = expected.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-local",
        action="store_true",
        help="Also validate large generated CSV files that are intentionally not committed.",
    )
    args = parser.parse_args()
    required = {
        Path("data/results/prediction_metrics.csv"): {
            "method",
            "horizon_s",
            "model_family",
            "point_nmae_range_pct",
            "point_nrmse_range_pct",
            "window_energy_mae_kwh",
            "high_power_f1",
            "braking_f1",
        },
        Path("data/results/allocation/allocation_metrics.csv"): {
            "strategy",
            "horizon_s",
            "degradation_proxy_sum",
            "soc_error",
        },
        Path("data/results/allocation/demand_clipping_audit.json"): set(),
    }
    if args.full_local:
        required.update(
            {
                Path("data/processed/liu_vehicle_canonical_1s.csv"): {
                    "timestamp",
                    "speed_kmh",
                    "soc_pct",
                },
                Path("data/processed/power_demand_from_dynamics.csv"): {
                    "p_dem_measured_kw",
                    "p_dem_dyn_calibrated_kw",
                },
                Path("data/processed/prediction_results.csv"): {
                    "method",
                    "forecast_horizon_s",
                    "step_ahead_s",
                    "power_pred_kw",
                },
            }
        )
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise SystemExit(
            "Missing local generated artifacts. Run numbered scripts in order: "
            + ", ".join(missing_files)
        )
    for path, columns in required.items():
        if columns:
            require_columns(path, columns)

    prediction_metrics = pd.read_csv("data/results/prediction_metrics.csv")
    if set(prediction_metrics["method"]) != {"state_direct_power"}:
        raise RuntimeError("Prediction metrics contain an unexpected semantic head.")
    if set(prediction_metrics["horizon_s"]) != {1, 3, 5, 10}:
        raise RuntimeError("Prediction horizons are incomplete.")
    if set(prediction_metrics["model_family"]) != {
        "extratrees",
        "hist_gradient_boosting",
        "xgboost",
        "brake_aware_extratrees",
    }:
        raise RuntimeError("Horizon model-family comparison is incomplete.")

    allocation = pd.read_csv("data/results/allocation/allocation_metrics.csv")
    if not (allocation["soc_error"].abs() <= 0.02).all():
        raise RuntimeError("At least one allocation violates the terminal SOC band.")
    mode = "full local" if args.full_local else "committed-artifact"
    print(f"Reproducibility {mode} checks passed.")


if __name__ == "__main__":
    main()
