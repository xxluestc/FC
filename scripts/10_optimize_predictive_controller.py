"""Jointly search forecast handling, move blocking, dwell, and MPC weights."""

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


def consecutive_sequence(origins, maximum_length=3600):
    runs, current = [], []
    for origin in np.sort(np.unique(origins)):
        if current and origin != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(int(origin))
    if current:
        runs.append(current)
    return max(runs, key=len)[:maximum_length]


def consecutive_runs(origins):
    """Return chronological contiguous runs without joining across gaps."""

    runs, current = [], []
    for origin in np.sort(np.unique(origins)):
        if current and origin != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(int(origin))
    if current:
        runs.append(current)
    return runs


def confidence_alpha(offsets, form, rate, validation_nrmse, trusted_steps):
    offsets = np.asarray(offsets, dtype=float)
    if form == "exponential":
        return np.exp(-rate * offsets)
    if form == "linear":
        return np.maximum(0, 1 - rate * offsets)
    if form == "adaptive":
        return np.exp(-rate * offsets * max(1, validation_nrmse / 5))
    if form == "hard_fallback":
        return (offsets <= trusted_steps).astype(float)
    raise ValueError(form)


def transform_forecast(raw, current, config, validation_nrmse):
    if len(raw) == 0:
        return raw
    offsets = np.arange(1, len(raw) + 1)
    alpha = confidence_alpha(
        offsets,
        config["confidence_form"],
        config["confidence_rate"],
        validation_nrmse,
        config["trusted_steps"],
    )
    forecast = alpha * raw + (1 - alpha) * current
    mode = config["forecast_mode"]
    blend = config["trend_weight"]
    if mode == "window_mean":
        trend = np.repeat(forecast.mean(), len(forecast))
    elif mode == "cumulative_mean":
        trend = np.cumsum(forecast) / np.arange(1, len(forecast) + 1)
    elif mode == "block_mean":
        trend = forecast.copy()
        width = max(1, config["forecast_block_size"])
        for start in range(0, len(forecast), width):
            trend[start : start + width] = forecast[start : start + width].mean()
    elif mode == "raw":
        trend = forecast
    else:
        raise ValueError(mode)
    return (1 - blend) * forecast + blend * trend


def run_controller(
    label,
    strategy,
    horizon,
    demand,
    source_indices,
    prediction_map,
    actions,
    hydrogen,
    proxy,
    config,
    validation_nrmse,
):
    soc = 0.70
    previous_tier = 0
    dwell = config["min_dwell"]
    rows = []
    started = time.perf_counter()
    max_step = max(np.diff(actions).max(), 1)
    for step, (source_index, current_demand) in enumerate(zip(source_indices, demand)):
        available = min(horizon, len(demand) - step)
        if strategy == "instant":
            preview = np.asarray([current_demand])
        elif strategy == "constant":
            preview = np.repeat(current_demand, available)
        elif strategy == "perfect":
            preview = demand[step : step + available]
        elif strategy == "predicted":
            raw = np.asarray(
                [
                    prediction_map.get((horizon, source_index, offset), current_demand)
                    for offset in range(1, available)
                ]
            )
            future = transform_forecast(raw, current_demand, config, validation_nrmse)
            preview = np.r_[current_demand, future]
        else:
            raise ValueError(strategy)
        tier = choose(
            preview,
            soc,
            previous_tier,
            dwell,
            actions,
            hydrogen,
            proxy,
            min_dwell=config["min_dwell"],
            weights=config["weights"],
            block_size=config["move_block_size"],
            max_horizon_switches=config["max_horizon_switches"],
        )
        p_fc = actions[tier]
        p_bat = current_demand - p_fc
        soc_next = float(next_soc(soc, p_bat))
        switched = tier != previous_tier
        preferred_fc = np.clip(
            max(current_demand, 0) + 1200 * (0.70 - soc), actions.min(), actions.max()
        )
        weights = config["weights"]
        rows.append(
            {
                "step": step,
                "source_index": source_index,
                "strategy": label,
                "horizon_s": horizon,
                "demand_kw": current_demand,
                "p_fc_kw": p_fc,
                "p_bat_kw": p_bat,
                "soc": soc_next,
                "tier": tier,
                "is_braking": current_demand < -5,
                "cost_hydrogen": weights["hydrogen"]
                * hydrogen[tier]
                / max(hydrogen.max(), 1e-9),
                "cost_proxy": weights["degradation_proxy"] * proxy[tier],
                "cost_battery": weights["battery_use"] * abs(p_bat) / 120,
                "cost_soc": weights["soc"] * abs(p_fc - preferred_fc) / actions.max()
                + 0.5 * abs(soc_next - 0.70) / 0.1,
                "cost_switch": weights["switch"] * switched,
                "cost_smooth": weights["smooth"]
                * abs(p_fc - actions[previous_tier])
                / max_step,
                "degradation_proxy": proxy[tier],
                "h2_g": hydrogen[tier],
            }
        )
        dwell = min(config["min_dwell"], dwell + 1) if not switched else 1
        previous_tier = tier
        soc = soc_next
    frame = pd.DataFrame(rows)
    cost_columns = [column for column in frame if column.startswith("cost_")]
    metrics = {
        "strategy": label,
        "horizon_s": horizon,
        "n": len(frame),
        "h2_kg": frame.h2_g.sum() / 1000,
        "degradation_proxy_sum": frame.degradation_proxy.sum(),
        "soc_final": frame.soc.iloc[-1],
        "soc_error": frame.soc.iloc[-1] - 0.70,
        "battery_throughput_kwh": frame.p_bat_kw.abs().sum() / 3600,
        "switch_count": int(frame.tier.diff().fillna(0).ne(0).sum()),
        "fc_total_variation_kw": frame.p_fc_kw.diff().fillna(0).abs().sum(),
        "internal_objective_sum": frame[cost_columns].sum().sum(),
        "runtime_s": time.perf_counter() - started,
    }
    return frame, metrics


def candidate_configs(center, count=48):
    rng = np.random.default_rng(2026)
    candidates = []
    anchor = {
        **center,
        "horizon": 5,
        "forecast_mode": "raw",
        "confidence_form": "exponential",
        "confidence_rate": 0.08,
        "trend_weight": 0.0,
        "trusted_steps": 3,
        "forecast_block_size": 2,
        "move_block_size": 1,
        "max_horizon_switches": 2,
        "min_dwell": 15,
    }
    candidates.append(anchor)
    choices = {
        "horizon": (3, 5, 10),
        "forecast_mode": ("raw", "cumulative_mean", "window_mean", "block_mean"),
        "confidence_form": ("exponential", "linear", "adaptive", "hard_fallback"),
        "confidence_rate": (0.02, 0.05, 0.10, 0.20, 0.35),
        "trend_weight": (0.25, 0.5, 0.75, 1.0),
        "trusted_steps": (1, 2, 3, 5),
        "forecast_block_size": (2, 3, 5),
        "move_block_size": (1, 2, 3, 5),
        "max_horizon_switches": (1, 2),
        "min_dwell": (10, 15, 25, 40, 60),
    }
    weight_choices = {
        "hydrogen": (0.15, 0.3, 0.45, 0.7, 1.0),
        "degradation_proxy": (0.75, 1.25, 2.0, 3.5, 6.0),
        "battery_use": (0.5, 1.0, 1.5, 2.5, 4.0),
        "soc": (1.0, 2.0, 3.0, 5.0, 8.0),
        "switch": (0.03, 0.08, 0.15, 0.3, 0.6),
        "smooth": (0.01, 0.04, 0.1, 0.3, 0.8),
    }
    while len(candidates) < count:
        config = {key: rng.choice(values).item() for key, values in choices.items()}
        config["weights"] = {
            key: float(rng.choice(values)) for key, values in weight_choices.items()
        }
        candidates.append(config)
    return candidates


def flatten_config(config):
    row = {key: value for key, value in config.items() if key != "weights"}
    row.update({f"w_{key}": value for key, value in config["weights"].items()})
    return row


def pareto_mask(frame, columns):
    values = frame[columns].to_numpy(float)
    keep = np.ones(len(frame), dtype=bool)
    for index, value in enumerate(values):
        dominated = np.any(
            np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        )
        keep[index] = not dominated
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-metrics", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--previous-search", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vehicle = pd.read_csv(args.vehicle)
    predictions = pd.read_csv(args.predictions)
    prediction_metrics = pd.read_csv(args.prediction_metrics)
    stack = pd.read_csv(args.stack_map)
    previous = pd.read_csv(args.previous_search).sort_values("selection_score").iloc[0]
    horizon_10 = predictions[predictions.forecast_horizon_s.eq(10)]
    runs = consecutive_runs(horizon_10.origin_index)
    eligible_runs = [run for run in runs if len(run) >= 3600]
    if len(eligible_runs) < 3:
        raise RuntimeError(
            "Need three chronological contiguous runs of at least 3600 s"
        )
    calibration_index_sets = [run[:3600] for run in eligible_runs[:-1]]
    evaluation_indices = eligible_runs[-1][:3600]
    raw_calibration_demands = [
        vehicle.loc[indices, "p_dem_measured_kw"].to_numpy()
        for indices in calibration_index_sets
    ]
    raw_evaluation_demand = vehicle.loc[
        evaluation_indices, "p_dem_measured_kw"
    ].to_numpy()
    lower, upper = -75.0, 120 + stack.stack_power_kw.max()
    calibration_demands = [
        np.clip(values, lower, upper) for values in raw_calibration_demands
    ]
    evaluation_demand = np.clip(raw_evaluation_demand, lower, upper)
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
    center_weights = {
        "hydrogen": float(previous.w_h2),
        "degradation_proxy": float(previous.w_deg),
        "battery_use": 1.5,
        "soc": float(previous.w_soc),
        "switch": float(previous.w_switch),
        "smooth": float(previous.w_smooth),
    }
    center = {"weights": center_weights}
    baseline_config = {
        "weights": center_weights,
        "min_dwell": 15,
        "move_block_size": 1,
        "max_horizon_switches": 2,
        "forecast_mode": "raw",
        "confidence_form": "exponential",
        "confidence_rate": 0.08,
        "trend_weight": 0.0,
        "trusted_steps": 3,
        "forecast_block_size": 2,
    }
    calibration_instant_rows = []
    for run_id, (indices, run_demand) in enumerate(
        zip(calibration_index_sets, calibration_demands)
    ):
        _, run_metrics = run_controller(
            f"Instant_calibration_{run_id}",
            "instant",
            1,
            run_demand,
            indices,
            prediction_map,
            actions,
            hydrogen,
            proxy,
            baseline_config,
            validation_nrmse[1],
        )
        calibration_instant_rows.append(run_metrics)
    calibration_instant = pd.DataFrame(calibration_instant_rows).sum(numeric_only=True)
    search_rows = []
    configs = candidate_configs(center)
    for candidate_id, config in enumerate(configs):
        print(
            f"Searching controller structure candidate {candidate_id + 1}/{len(configs)}",
            flush=True,
        )
        run_metrics_rows = []
        for run_id, (indices, run_demand) in enumerate(
            zip(calibration_index_sets, calibration_demands)
        ):
            _, run_metrics = run_controller(
                f"candidate_{candidate_id}_run_{run_id}",
                "predicted",
                int(config["horizon"]),
                run_demand,
                indices,
                prediction_map,
                actions,
                hydrogen,
                proxy,
                config,
                validation_nrmse[int(config["horizon"])],
            )
            run_metrics_rows.append(run_metrics)
        run_metrics_frame = pd.DataFrame(run_metrics_rows)
        metrics = run_metrics_frame.sum(numeric_only=True).to_dict()
        metrics["soc_error"] = float(
            run_metrics_frame.loc[
                run_metrics_frame.soc_error.abs().idxmax(), "soc_error"
            ]
        )
        metrics["soc_final"] = 0.70 + metrics["soc_error"]
        metrics["calibration_runs"] = len(run_metrics_frame)
        normalized = {
            metric: metrics[metric] / max(calibration_instant[metric], 1e-9)
            for metric in (
                "h2_kg",
                "degradation_proxy_sum",
                "battery_throughput_kwh",
                "switch_count",
                "fc_total_variation_kw",
            )
        }
        score = (
            normalized["h2_kg"]
            + normalized["degradation_proxy_sum"]
            + 0.35 * normalized["battery_throughput_kwh"]
            + 0.55 * normalized["switch_count"]
            + 0.55 * normalized["fc_total_variation_kw"]
            + 40 * abs(metrics["soc_error"])
        )
        if (run_metrics_frame.soc_error.abs() > 0.02).any():
            score += 100
        search_rows.append(
            {
                "candidate_id": candidate_id,
                **flatten_config(config),
                **metrics,
                "selection_score": score,
            }
        )
    search = pd.DataFrame(search_rows).sort_values("selection_score")
    pareto_columns = [
        "h2_kg",
        "degradation_proxy_sum",
        "battery_throughput_kwh",
        "switch_count",
        "fc_total_variation_kw",
    ]
    feasible = search[search.soc_error.abs() <= 0.02].copy()
    feasible["pareto"] = pareto_mask(feasible, pareto_columns)
    search = search.merge(
        feasible[["candidate_id", "pareto"]], on="candidate_id", how="left"
    )
    search["pareto"] = search["pareto"].eq(True)
    search.to_csv(args.out_dir / "controller_structure_weight_search.csv", index=False)
    feasible[feasible.pareto].to_csv(
        args.out_dir / "controller_pareto_candidates.csv", index=False
    )
    best_id = int(feasible.sort_values("selection_score").iloc[0].candidate_id)
    best_config = configs[best_id]
    (args.out_dir / "selected_controller_config.json").write_text(
        json.dumps(best_config, indent=2), encoding="utf-8"
    )

    eval_demand = evaluation_demand
    eval_indices = evaluation_indices
    best_horizon = int(best_config["horizon"])
    experiments = [
        ("Instant", "instant", 1, baseline_config),
        ("Constant", "constant", best_horizon, best_config),
        ("Perfect", "perfect", best_horizon, best_config),
        ("Predicted_original", "predicted", 5, baseline_config),
        ("Predicted_optimized", "predicted", best_horizon, best_config),
    ]
    trajectories, metrics_rows = [], []
    for label, strategy, horizon, config in experiments:
        frame, metrics = run_controller(
            label,
            strategy,
            horizon,
            eval_demand,
            eval_indices,
            prediction_map,
            actions,
            hydrogen,
            proxy,
            config,
            validation_nrmse[horizon],
        )
        frame["raw_demand_kw"] = raw_evaluation_demand
        frame["was_clipped"] = np.abs(raw_evaluation_demand - eval_demand) > 1e-12
        trajectories.append(frame)
        metrics_rows.append(metrics)
    trajectory = pd.concat(trajectories, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    trajectory.to_csv(args.out_dir / "optimized_strategy_trajectories.csv", index=False)
    metrics.to_csv(args.out_dir / "optimized_strategy_metrics.csv", index=False)

    relative_rows = []
    minimized = [
        "h2_kg",
        "degradation_proxy_sum",
        "battery_throughput_kwh",
        "switch_count",
        "fc_total_variation_kw",
    ]
    optimized = metrics[metrics.strategy.eq("Predicted_optimized")].iloc[0]
    for baseline_name in ("Instant", "Constant", "Perfect", "Predicted_original"):
        baseline = metrics[metrics.strategy.eq(baseline_name)].iloc[0]
        row = {"strategy": "Predicted_optimized", "baseline": baseline_name}
        for metric in minimized:
            row[f"{metric}_improvement_pct"] = (
                100
                * (baseline[metric] - optimized[metric])
                / max(abs(baseline[metric]), 1e-9)
            )
        row["absolute_soc_error_improvement_pct"] = (
            100
            * (abs(baseline.soc_error) - abs(optimized.soc_error))
            / max(abs(baseline.soc_error), 1e-9)
        )
        relative_rows.append(row)
    pd.DataFrame(relative_rows).to_csv(
        args.out_dir / "optimized_relative_improvements.csv", index=False
    )
    clipping = {
        "evaluation_points": len(eval_demand),
        "clipped_points": int(
            (np.abs(raw_evaluation_demand - eval_demand) > 1e-12).sum()
        ),
        "clipped_share_pct": float(
            100 * (np.abs(raw_evaluation_demand - eval_demand) > 1e-12).mean()
        ),
        "absolute_clipped_energy_kwh": float(
            np.abs(raw_evaluation_demand - eval_demand).sum() / 3600
        ),
    }
    (args.out_dir / "optimized_clipping_audit.json").write_text(
        json.dumps(clipping, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
