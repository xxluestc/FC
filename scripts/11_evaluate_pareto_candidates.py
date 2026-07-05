"""Evaluate calibration-selected Pareto controllers on the untouched later run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_controller_module(path: Path):
    spec = importlib.util.spec_from_file_location("controller_optimization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_from_row(row):
    return {
        "horizon": int(row.horizon),
        "forecast_mode": row.forecast_mode,
        "confidence_form": row.confidence_form,
        "confidence_rate": float(row.confidence_rate),
        "trend_weight": float(row.trend_weight),
        "trusted_steps": int(row.trusted_steps),
        "forecast_block_size": int(row.forecast_block_size),
        "move_block_size": int(row.move_block_size),
        "max_horizon_switches": int(row.max_horizon_switches),
        "min_dwell": int(row.min_dwell),
        "weights": {
            "hydrogen": float(row.w_hydrogen),
            "degradation_proxy": float(row.w_degradation_proxy),
            "battery_use": float(row.w_battery_use),
            "soc": float(row.w_soc),
            "switch": float(row.w_switch),
            "smooth": float(row.w_smooth),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-script", type=Path, required=True)
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-metrics", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    module = load_controller_module(args.controller_script)
    vehicle = pd.read_csv(args.vehicle)
    predictions = pd.read_csv(args.predictions)
    prediction_metrics = pd.read_csv(args.prediction_metrics)
    stack = pd.read_csv(args.stack_map)
    pareto = pd.read_csv(args.pareto)
    runs = module.consecutive_runs(
        predictions.loc[predictions.forecast_horizon_s.eq(10), "origin_index"]
    )
    evaluation_indices = [run for run in runs if len(run) >= 3600][-1][:3600]
    raw_demand = vehicle.loc[evaluation_indices, "p_dem_measured_kw"].to_numpy()
    lower, upper = -75.0, 120 + stack.stack_power_kw.max()
    demand = np.clip(raw_demand, lower, upper)
    prediction_map = {
        (
            int(row.forecast_horizon_s),
            int(row.origin_index),
            int(row.step_ahead_s),
        ): float(np.clip(row.power_pred_kw, lower, upper))
        for row in predictions.itertuples()
    }
    validation_nrmse = {
        int(row.horizon_s): float(row.point_nrmse_range_pct)
        for row in prediction_metrics[
            prediction_metrics.split.eq("validation") & prediction_metrics.selected
        ].itertuples()
    }
    actions = stack.stack_power_kw.to_numpy()
    hydrogen = stack.faraday_h2_g_s.to_numpy()
    proxy = stack.performance_loss_cost_normalized.to_numpy()
    rows = []
    for row in pareto.itertuples():
        config = config_from_row(row)
        _, metrics = module.run_controller(
            f"Pareto_{int(row.candidate_id)}",
            "predicted",
            config["horizon"],
            demand,
            evaluation_indices,
            prediction_map,
            actions,
            hydrogen,
            proxy,
            config,
            validation_nrmse[config["horizon"]],
        )
        deficit_kwh = max(0, 0.70 - metrics["soc_final"]) * 37.0
        equivalent_seconds = deficit_kwh / (actions[-1] * 0.95 / 3600)
        metrics.update(
            {
                "candidate_id": int(row.candidate_id),
                "calibration_selection_score": float(row.selection_score),
                "soc_equivalent_recharge_s": equivalent_seconds,
                "soc_equivalent_h2_kg": metrics["h2_kg"]
                + equivalent_seconds * hydrogen[-1] / 1000,
                "soc_equivalent_proxy": metrics["degradation_proxy_sum"]
                + equivalent_seconds * proxy[-1],
                **{
                    key: value
                    for key, value in config.items()
                    if key not in ("weights",)
                },
                **{f"w_{key}": value for key, value in config["weights"].items()},
            }
        )
        rows.append(metrics)
    result = pd.DataFrame(rows).sort_values("calibration_selection_score")
    result.to_csv(args.out_dir / "pareto_late_run_generalization.csv", index=False)
    (args.out_dir / "pareto_evaluation_scope.json").write_text(
        json.dumps(
            {
                "selection_rule": "Candidates were selected on earlier runs only; late-run results are diagnostic and do not change the default controller.",
                "evaluation_start_index": evaluation_indices[0],
                "evaluation_end_index": evaluation_indices[-1],
                "evaluation_points": len(evaluation_indices),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
