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
    confidence_decay=0.0,
    min_dwell=15,
):
    soc = 0.70
    previous_tier = 0
    dwell = min_dwell
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
                    np.exp(-confidence_decay * offset)
                    * prediction_map.get(
                        (horizon, source_index, offset), current_demand
                    )
                    + (1 - np.exp(-confidence_decay * offset)) * current_demand
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
            min_dwell=min_dwell,
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
                "weighted_h2_cost": weights["hydrogen"]
                * hydrogen[tier]
                / max(hydrogen.max(), 1e-9),
                "weighted_proxy_cost": weights["degradation_proxy"]
                * degradation_proxy[tier],
                "weighted_switch_cost": weights["switch"] * (tier != previous_tier),
                "weighted_smooth_cost": weights["smooth"]
                * abs(fuel_cell_power - actions[previous_tier])
                / max(np.diff(actions).max(), 1),
                "is_braking": current_demand < -5,
                "confidence_decay": confidence_decay,
            }
        )
        dwell = min(min_dwell, dwell + 1) if tier == previous_tier else 1
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


def search_weights(
    demand,
    source_indices,
    prediction_map,
    actions,
    hydrogen,
    degradation_proxy,
):
    """Deterministic compact grid around w_deg=2 on a calibration prefix."""

    anchors = [
        (0.45, 2.0, 0.10, 3.0, 0.10, 0.08),
        (0.45, 2.0, 0.05, 3.0, 0.05, 0.00),
        (0.45, 2.0, 0.20, 3.0, 0.20, 0.16),
    ]
    rng = np.random.default_rng(2026)
    choices = [
        (0.30, 0.45, 0.60),
        (1.5, 2.0, 2.5),
        (0.05, 0.10, 0.20),
        (2.0, 3.0, 5.0),
        (0.05, 0.10, 0.20),
        (0.00, 0.08, 0.16),
    ]
    while len(anchors) < 24:
        candidate = tuple(float(rng.choice(values)) for values in choices)
        if candidate not in anchors:
            anchors.append(candidate)
    length = min(1200, len(demand))
    reference_weights = DEFAULT_WEIGHTS.copy()
    reference_weights.update({"degradation_proxy": 2.0, "smooth": 0.1, "switch": 0.1})
    _, reference = run_strategy(
        "constant",
        5,
        demand[:length],
        source_indices[:length],
        prediction_map,
        actions,
        hydrogen,
        degradation_proxy,
        reference_weights,
    )
    rows = []
    for hydrogen_w, deg_w, smooth_w, soc_w, switch_w, decay in anchors:
        weights = DEFAULT_WEIGHTS.copy()
        weights.update(
            {
                "hydrogen": hydrogen_w,
                "degradation_proxy": deg_w,
                "smooth": smooth_w,
                "soc": soc_w,
                "switch": switch_w,
            }
        )
        _, result = run_strategy(
            "predicted",
            5,
            demand[:length],
            source_indices[:length],
            prediction_map,
            actions,
            hydrogen,
            degradation_proxy,
            weights,
            confidence_decay=decay,
        )
        _, instant_result = run_strategy(
            "instant",
            1,
            demand[:length],
            source_indices[:length],
            prediction_map,
            actions,
            hydrogen,
            degradation_proxy,
            weights,
        )
        score = (
            result["h2_kg"] / max(reference["h2_kg"], 1e-9)
            + result["degradation_proxy_sum"]
            / max(reference["degradation_proxy_sum"], 1e-9)
            + 0.3
            * result["battery_throughput_kwh"]
            / max(reference["battery_throughput_kwh"], 1e-9)
            + 0.2
            * result["fc_total_variation_kw"]
            / max(reference["fc_total_variation_kw"], 1e-9)
            + 30 * abs(result["soc_error"])
        )
        # A weight set is not eligible if the H=1 baseline cannot sustain SOC;
        # otherwise comparisons would mix controller and feasibility effects.
        if abs(result["soc_error"]) > 0.02 or abs(instant_result["soc_error"]) > 0.02:
            score += 100
        rows.append(
            {
                **result,
                "w_h2": hydrogen_w,
                "w_deg": deg_w,
                "w_smooth": smooth_w,
                "w_soc": soc_w,
                "w_switch": switch_w,
                "confidence_decay": decay,
                "selection_score": score,
                "instant_soc_error": instant_result["soc_error"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("selection_score")
    best = frame.iloc[0]
    best_weights = DEFAULT_WEIGHTS.copy()
    best_weights.update(
        {
            "hydrogen": best.w_h2,
            "degradation_proxy": best.w_deg,
            "smooth": best.w_smooth,
            "soc": best.w_soc,
            "switch": best.w_switch,
        }
    )
    return frame, best_weights, float(best.confidence_decay)


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
    horizon_10 = causal[causal["forecast_horizon_s"].eq(10)]
    sequence = consecutive_test_sequence(horizon_10["origin_index"].unique())

    raw_demand = vehicle.loc[sequence, "p_dem_measured_kw"].to_numpy()
    lower_bound = -75.0
    upper_bound = 120 + stack_map["stack_power_kw"].max()
    feasible_demand = np.clip(raw_demand, lower_bound, upper_bound)
    clipping_delta = raw_demand - feasible_demand
    clipped = np.abs(clipping_delta) > 1e-12

    prediction_map = {
        (
            int(row.forecast_horizon_s),
            int(row.origin_index),
            int(row.step_ahead_s),
        ): float(np.clip(row.power_pred_kw, lower_bound, upper_bound))
        for row in causal.itertuples()
    }
    actions = stack_map["stack_power_kw"].to_numpy()
    hydrogen = stack_map["faraday_h2_g_s"].to_numpy()
    degradation_proxy = stack_map["performance_loss_cost_normalized"].to_numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    search, optimized_weights, optimized_decay = search_weights(
        feasible_demand,
        sequence,
        prediction_map,
        actions,
        hydrogen,
        degradation_proxy,
    )
    search.to_csv(args.out_dir / "mpc_weight_search.csv", index=False)

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
            optimized_weights,
            optimized_decay if strategy == "predicted" else 0.0,
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

    occupancy = trajectory_frame.groupby(
        ["strategy", "horizon_s", "tier"], as_index=False
    ).agg(count=("tier", "size"), p_fc_kw=("p_fc_kw", "first"))
    occupancy["share_pct"] = (
        100
        * occupancy["count"]
        / occupancy.groupby(["strategy", "horizon_s"])["count"].transform("sum")
    )
    occupancy.to_csv(args.out_dir / "strategy_tier_occupancy.csv", index=False)
    proxy_by_tier = trajectory_frame.groupby(
        ["strategy", "horizon_s", "tier"], as_index=False
    ).agg(
        count=("tier", "size"),
        proxy_sum=("degradation_proxy", "sum"),
        weighted_proxy_sum=("weighted_proxy_cost", "sum"),
    )
    proxy_by_tier.to_csv(args.out_dir / "tier_proxy_contribution.csv", index=False)
    brake_rows = []
    for keys, group in trajectory_frame.groupby(
        ["strategy", "horizon_s", "is_braking"]
    ):
        brake_rows.append(
            {
                "strategy": keys[0],
                "horizon_s": keys[1],
                "subset": "braking" if keys[2] else "non_braking",
                "n": len(group),
                "mean_demand_kw": group["demand_kw"].mean(),
                "mean_fc_kw": group["p_fc_kw"].mean(),
                "mean_battery_kw": group["p_bat_kw"].mean(),
                "battery_throughput_kwh": group["p_bat_kw"].abs().sum() / 3600,
                "proxy_sum": group["degradation_proxy"].sum(),
            }
        )
    pd.DataFrame(brake_rows).to_csv(
        args.out_dir / "braking_allocation_diagnostics.csv", index=False
    )
    paired = trajectory_frame[
        trajectory_frame["strategy"].isin(["predicted", "perfect"])
    ].pivot_table(index=["step", "horizon_s"], columns="strategy", values="tier")
    paired = paired.dropna().reset_index()
    paired["action_difference_tiers"] = paired["predicted"] - paired["perfect"]
    paired["actions_differ"] = paired["action_difference_tiers"].ne(0)
    paired.to_csv(args.out_dir / "predicted_vs_perfect_actions.csv", index=False)

    sensitivity_rows = []
    sensitivity_grid = {
        "hydrogen": (0.30, 0.45, 0.60),
        "degradation_proxy": (1.5, 2.0, 2.5),
        "smooth": (0.05, 0.10, 0.20),
        "soc": (1.5, 3.0, 6.0),
        "switch": (0.05, 0.10, 0.20),
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
                optimized_decay,
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
