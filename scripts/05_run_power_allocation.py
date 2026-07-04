"""Fair horizon/weight comparison for degradation-proxy-aware allocation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_power.battery_model import next_soc
from fc_power.power_allocation.mpc_allocator import DEFAULT_WEIGHTS, choose


def run_strategy(
    strategy,
    horizon,
    demand,
    source_indices,
    prediction_map,
    actions,
    hydrogen,
    degradation_proxy,
    weights,
):
    soc = 0.70
    previous_tier = 0
    dwell = 15
    rows = []
    started = time.perf_counter()

    for step, (source_index, current_demand) in enumerate(zip(source_indices, demand)):
        available_horizon = min(horizon, len(demand) - step)
        if strategy == "instant":
            preview = np.asarray([current_demand])
        elif strategy == "constant":
            preview = np.repeat(current_demand, available_horizon)
        elif strategy == "perfect":
            preview = demand[step : step + available_horizon]
        elif strategy == "predicted":
            preview = np.r_[
                current_demand,
                [
                    prediction_map.get((source_index, offset), current_demand)
                    for offset in range(1, available_horizon)
                ],
            ]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        tier = choose(
            preview,
            soc,
            previous_tier,
            dwell,
            actions,
            hydrogen,
            degradation_proxy,
            weights=weights,
        )
        fuel_cell_power = actions[tier]
        battery_power = current_demand - fuel_cell_power
        next_state = float(next_soc(soc, battery_power))
        rows.append(
            {
                "step": step,
                "source_index": source_index,
                "strategy": strategy,
                "horizon_s": horizon,
                "demand_kw": current_demand,
                "p_fc_kw": fuel_cell_power,
                "p_bat_kw": battery_power,
                "soc": next_state,
                "tier": tier,
                "h2_g": hydrogen[tier],
                "degradation_proxy": degradation_proxy[tier],
            }
        )
        dwell = min(15, dwell + 1) if tier == previous_tier else 1
        previous_tier = tier
        soc = next_state

    trajectory = pd.DataFrame(rows)
    switches = int(trajectory["tier"].diff().fillna(0).ne(0).sum())
    metrics = {
        "strategy": strategy,
        "horizon_s": horizon,
        "n": len(trajectory),
        "h2_kg": trajectory["h2_g"].sum() / 1000,
        "degradation_proxy_sum": trajectory["degradation_proxy"].sum(),
        "soc_final": trajectory["soc"].iloc[-1],
        "soc_error": trajectory["soc"].iloc[-1] - 0.70,
        "battery_throughput_kwh": trajectory["p_bat_kw"].abs().sum() / 3600,
        "switch_count": switches,
        "fc_total_variation_kw": trajectory["p_fc_kw"].diff().fillna(0).abs().sum(),
        "runtime_s": time.perf_counter() - started,
    }
    return trajectory, metrics


def consecutive_test_sequence(origins, maximum_length=3600):
    runs = []
    current = []
    for origin in np.sort(origins):
        if current and origin != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(int(origin))
    if current:
        runs.append(current)
    return max(runs, key=len)[:maximum_length]


def relative_improvements(metrics):
    frame = pd.DataFrame(metrics)
    instant = frame[frame["strategy"].eq("instant")].iloc[0]
    rows = []
    minimized = [
        "h2_kg",
        "degradation_proxy_sum",
        "battery_throughput_kwh",
        "switch_count",
        "fc_total_variation_kw",
    ]
    for row in frame.itertuples():
        constant = frame[
            frame["strategy"].eq("constant") & frame["horizon_s"].eq(row.horizon_s)
        ]
        for baseline_name, baseline in (
            ("instant", instant),
            ("constant", constant.iloc[0] if len(constant) else None),
        ):
            if baseline is None or row.strategy in ("instant", baseline_name):
                continue
            result = {
                "strategy": row.strategy,
                "horizon_s": row.horizon_s,
                "baseline": baseline_name,
            }
            for metric in minimized:
                result[f"{metric}_improvement_pct"] = (
                    100
                    * (baseline[metric] - getattr(row, metric))
                    / max(abs(baseline[metric]), 1e-12)
                )
            result["absolute_soc_error_improvement_pct"] = (
                100
                * (abs(baseline["soc_error"]) - abs(row.soc_error))
                / max(abs(baseline["soc_error"]), 1e-12)
            )
            rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    vehicle = pd.read_csv(args.vehicle)
    predictions = pd.read_csv(args.predictions)
    stack_map = pd.read_csv(args.stack_map)
    causal = predictions[predictions["method"].eq("state_direct_power")]
    sequence = consecutive_test_sequence(causal["origin_index"].unique())

    raw_demand = vehicle.loc[sequence, "p_dem_measured_kw"].to_numpy()
    lower_bound = -75.0
    upper_bound = 120 + stack_map["stack_power_kw"].max()
    feasible_demand = np.clip(raw_demand, lower_bound, upper_bound)
    clipping_delta = raw_demand - feasible_demand
    clipped = np.abs(clipping_delta) > 1e-12

    prediction_map = {
        (int(row.origin_index), int(row.horizon_s)): float(
            np.clip(row.power_pred_kw, lower_bound, upper_bound)
        )
        for row in causal.itertuples()
    }
    actions = stack_map["stack_power_kw"].to_numpy()
    hydrogen = stack_map["faraday_h2_g_s"].to_numpy()
    degradation_proxy = stack_map["performance_loss_cost_normalized"].to_numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    metrics = []
    experiment_grid = [("instant", 1)] + [
        (strategy, horizon)
        for horizon in (3, 5, 10)
        for strategy in ("constant", "perfect", "predicted")
    ]
    for strategy, horizon in experiment_grid:
        trajectory, result = run_strategy(
            strategy,
            horizon,
            feasible_demand,
            sequence,
            prediction_map,
            actions,
            hydrogen,
            degradation_proxy,
            DEFAULT_WEIGHTS.copy(),
        )
        trajectory["raw_demand_kw"] = raw_demand
        trajectory["was_clipped"] = clipped
        trajectories.append(trajectory)
        metrics.append(result)
        print(result, flush=True)

    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.out_dir / "allocation_metrics.csv", index=False)
    trajectory_frame.to_csv(args.out_dir / "allocation_trajectory.csv", index=False)
    relative_improvements(metrics).to_csv(
        args.out_dir / "allocation_relative_improvement.csv", index=False
    )

    clipping_audit = {
        "n": len(raw_demand),
        "clipped_points": int(clipped.sum()),
        "clipped_share_pct": 100 * float(clipped.mean()),
        "absolute_clipped_energy_kwh": float(np.abs(clipping_delta).sum() / 3600),
        "signed_clipped_energy_kwh": float(clipping_delta.sum() / 3600),
        "maximum_clipped_power_kw": float(np.abs(clipping_delta).max()),
        "raw_absolute_energy_kwh": float(np.abs(raw_demand).sum() / 3600),
        "feasible_absolute_energy_kwh": float(np.abs(feasible_demand).sum() / 3600),
        "interpretation": "If clipping is material, allocation results apply only inside the FC+battery feasible domain.",
    }
    (args.out_dir / "demand_clipping_audit.json").write_text(
        json.dumps(clipping_audit, indent=2), encoding="utf-8"
    )

    impact_rows = []
    for (strategy, horizon), group in trajectory_frame.groupby(
        ["strategy", "horizon_s"]
    ):
        for label, mask in (
            ("clipped", group["was_clipped"]),
            ("unclipped", ~group["was_clipped"]),
        ):
            part = group[mask]
            impact_rows.append(
                {
                    "strategy": strategy,
                    "horizon_s": horizon,
                    "subset": label,
                    "n": len(part),
                    "h2_kg": part["h2_g"].sum() / 1000,
                    "degradation_proxy_sum": part["degradation_proxy"].sum(),
                    "battery_throughput_kwh": part["p_bat_kw"].abs().sum() / 3600,
                }
            )
    pd.DataFrame(impact_rows).to_csv(
        args.out_dir / "clipping_strategy_impact.csv", index=False
    )

    sensitivity_rows = []
    sensitivity_grid = {
        "hydrogen": (0.25, 0.45, 0.75),
        "degradation_proxy": (0.5, 1.0, 2.0),
        "smooth": (0.002, 0.005, 0.02),
        "soc": (1.5, 3.0, 6.0),
    }
    sensitivity_length = min(1800, len(sequence))
    for parameter, values in sensitivity_grid.items():
        for value in values:
            weights = DEFAULT_WEIGHTS.copy()
            weights[parameter] = value
            _, result = run_strategy(
                "predicted",
                5,
                feasible_demand[:sensitivity_length],
                sequence[:sensitivity_length],
                prediction_map,
                actions,
                hydrogen,
                degradation_proxy,
                weights,
            )
            result.update(
                {
                    "varied_parameter": parameter,
                    "varied_value": value,
                    "sensitivity_n": sensitivity_length,
                }
            )
            sensitivity_rows.append(result)
    pd.DataFrame(sensitivity_rows).to_csv(
        args.out_dir / "allocation_weight_sensitivity.csv", index=False
    )

    pd.DataFrame(
        {
            "source_index": sequence,
            "raw_demand_kw": raw_demand,
            "feasible_demand_kw": feasible_demand,
            "clipping_delta_kw": clipping_delta,
        }
    ).to_csv(args.out_dir / "test_demand.csv", index=False)


if __name__ == "__main__":
    main()
