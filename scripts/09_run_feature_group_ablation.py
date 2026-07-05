"""Compare causal feature groups using validation-only model selection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_power.prediction.horizon_models import build_regressor

HISTORY = 30
HORIZONS = (1, 3, 5, 10)


def causal_state_arrays(data: pd.DataFrame):
    speed = data["speed_smooth_mps"].to_numpy()
    segment = data["segment_id"].to_numpy()
    odometer = data["odometer_km"].interpolate().ffill().bfill().to_numpy()
    stopped = speed < 0.3
    n = len(data)
    stop_duration = np.zeros(n)
    time_since_stop = np.zeros(n)
    distance_since_stop = np.zeros(n)
    segment_elapsed = np.zeros(n)
    segment_distance = np.zeros(n)
    for index in range(1, n):
        if segment[index] != segment[index - 1]:
            continue
        segment_elapsed[index] = segment_elapsed[index - 1] + 1
        delta = odometer[index] - odometer[index - 1]
        if not np.isfinite(delta) or delta < 0 or delta > 1:
            delta = speed[index] / 1000
        segment_distance[index] = segment_distance[index - 1] + delta
        if stopped[index]:
            stop_duration[index] = stop_duration[index - 1] + 1
        else:
            time_since_stop[index] = time_since_stop[index - 1] + 1
            distance_since_stop[index] = distance_since_stop[index - 1] + delta
    return np.c_[
        stop_duration,
        np.log1p(time_since_stop),
        np.log1p(distance_since_stop),
        np.log1p(segment_elapsed),
        np.log1p(segment_distance),
    ]


def histories_with_summary(data, columns, indices, width=10):
    blocks = []
    rows = indices - width + 1
    for column in columns:
        values = data[column].interpolate().ffill().bfill().to_numpy(dtype=float)
        history = sliding_window_view(values, width)[rows]
        blocks.extend(
            [
                history,
                history.mean(1)[:, None],
                history.std(1)[:, None],
                history.min(1)[:, None],
                history.max(1)[:, None],
            ]
        )
    return np.column_stack(blocks) if blocks else np.empty((len(indices), 0))


def build_feature_groups(data: pd.DataFrame):
    segment = data["segment_id"].to_numpy()
    indices = np.asarray(
        [
            index
            for index in range(HISTORY, len(data) - max(HORIZONS))
            if segment[index - HISTORY] == segment[index + max(HORIZONS)]
        ]
    )
    speed = data["speed_smooth_mps"].to_numpy()
    acceleration = data["acceleration_smooth_mps2"].to_numpy()
    power = data["p_dem_measured_kw"].to_numpy()
    rows = indices - HISTORY
    base_histories = [
        sliding_window_view(values, HISTORY + 1)[rows]
        for values in (speed, acceleration, power)
    ]
    summary = []
    for history in base_histories:
        for width in (3, 5, 10, 20, 30):
            window = history[:, -width:]
            summary.extend(
                [
                    window.mean(1)[:, None],
                    window.std(1)[:, None],
                    window.min(1)[:, None],
                    window.max(1)[:, None],
                ]
            )
    timestamp = pd.DatetimeIndex(data["timestamp"])
    seconds = (
        timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    ).to_numpy()[indices]
    modes = np.c_[
        speed[indices] < 0.3,
        acceleration[indices] > 0.3,
        acceleration[indices] < -0.3,
        np.abs(acceleration[indices]) <= 0.1,
        power[indices] < -5,
    ]
    base = np.column_stack(
        [
            *base_histories,
            *summary,
            np.sin(2 * np.pi * seconds / 86400),
            np.cos(2 * np.pi * seconds / 86400),
            modes,
            data["soc_pct"].to_numpy()[indices] / 100,
        ]
    )
    available = set(data.columns)
    intent_columns = [
        column
        for column in ("target_power_kw", "loadable_power_kw")
        if column in available
    ]
    power_chain_columns = [
        column
        for column in (
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
        )
        if column in available
    ]
    health_columns = [
        column
        for column in (
            "mean_cell_voltage_v",
            "min_cell_voltage_v",
            "max_cell_voltage_v",
        )
        if column in available
    ]
    blocks = {
        "base": base.astype(np.float32),
        "route_stop": causal_state_arrays(data)[indices].astype(np.float32),
        "intent": histories_with_summary(data, intent_columns, indices).astype(
            np.float32
        ),
        "power_chain": histories_with_summary(
            data, power_chain_columns, indices, width=5
        ).astype(np.float32),
        "health": histories_with_summary(data, health_columns, indices, width=5).astype(
            np.float32
        ),
    }
    group_specs = {
        "base": ("base",),
        "base_route": ("base", "route_stop"),
        "base_intent": ("base", "intent"),
        "base_power_chain": ("base", "power_chain"),
        "base_health": ("base", "health"),
        "base_route_intent": ("base", "route_stop", "intent"),
        "all_available": (
            "base",
            "route_stop",
            "intent",
            "power_chain",
            "health",
        ),
    }
    targets = sliding_window_view(power, max(HORIZONS))[indices + 1]
    metadata = {
        "intent_columns": intent_columns,
        "power_chain_columns": power_chain_columns,
        "health_columns": health_columns,
        "aux_thermal_columns": [],
        "feature_dimensions": {
            name: sum(blocks[block].shape[1] for block in spec)
            for name, spec in group_specs.items()
        },
    }
    return indices, blocks, group_specs, targets, metadata


def split_positions(indices, n_rows, horizon):
    train_end = int(0.70 * n_rows)
    validation_end = int(0.85 * n_rows)
    train = np.flatnonzero(indices + horizon < train_end - horizon)
    validation = np.flatnonzero(
        (indices >= train_end + horizon)
        & (indices + horizon < validation_end - horizon)
    )
    test = np.flatnonzero(indices >= validation_end + horizon)
    if len(train) > 35_000:
        train = train[np.linspace(0, len(train) - 1, 35_000).astype(int)]
    return train, validation, test


def evaluate(actual, predicted, origin_power, threshold, full_scale):
    actual = np.asarray(actual).reshape(len(actual), -1)
    predicted = np.asarray(predicted).reshape(len(predicted), -1)
    error = predicted - actual
    actual_energy = actual.sum(1) / 3600
    predicted_energy = predicted.sum(1) / 3600
    actual_ramp = np.abs(np.diff(np.c_[origin_power, actual], axis=1)).sum(1)
    predicted_ramp = np.abs(np.diff(np.c_[origin_power, predicted], axis=1)).sum(1)
    return {
        "point_mae_kw": np.mean(np.abs(error)),
        "point_rmse_kw": np.sqrt(np.mean(error**2)),
        "point_nrmse_range_pct": 100 * np.sqrt(np.mean(error**2)) / full_scale,
        "window_mean_rmse_kw": np.sqrt(
            np.mean((predicted.mean(1) - actual.mean(1)) ** 2)
        ),
        "window_energy_mae_kwh": np.mean(np.abs(predicted_energy - actual_energy)),
        "high_power_f1": f1_score(
            actual.max(1) >= threshold, predicted.max(1) >= threshold, zero_division=0
        ),
        "braking_f1": f1_score(
            actual.min(1) < -5, predicted.min(1) < -5, zero_division=0
        ),
        "ramp_risk_mae_kw": np.mean(np.abs(predicted_ramp - actual_ramp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, parse_dates=["timestamp"])
    indices, blocks, group_specs, targets, metadata = build_feature_groups(data)
    power = data["p_dem_measured_kw"].to_numpy()
    full_scale = power.max() - power.min()
    rows = []
    prediction_rows = []
    selections = []
    for horizon in HORIZONS:
        train, validation, test = split_positions(indices, len(data), horizon)
        target = targets[:, :horizon]
        threshold = np.quantile(target[train], 0.9)
        candidates = {}
        for group_index, (group_name, group_blocks) in enumerate(group_specs.items()):
            features = np.column_stack([blocks[name] for name in group_blocks])
            print(
                f"Testing whether causal feature group {group_name} improves H={horizon}",
                flush=True,
            )
            started = time.perf_counter()
            try:
                model = build_regressor("xgboost", 3000 + 10 * group_index + horizon)
            except (ImportError, ModuleNotFoundError):
                model = ExtraTreesRegressor(
                    n_estimators=80, min_samples_leaf=2, n_jobs=-1, random_state=2026
                )
            model.fit(features[train], target[train])
            val_prediction = np.asarray(model.predict(features[validation])).reshape(
                len(validation), -1
            )
            test_prediction = np.asarray(model.predict(features[test])).reshape(
                len(test), -1
            )
            split_metrics = {}
            for split, positions, prediction in (
                ("validation", validation, val_prediction),
                ("test", test, test_prediction),
            ):
                metric = evaluate(
                    target[positions],
                    prediction,
                    power[indices[positions]][:, None],
                    threshold,
                    full_scale,
                )
                metric.update(
                    {
                        "feature_group": group_name,
                        "horizon_s": horizon,
                        "split": split,
                        "n": len(positions),
                        "feature_count": features.shape[1],
                        "runtime_s": time.perf_counter() - started,
                    }
                )
                rows.append(metric)
                split_metrics[split] = metric
            validation_score = (
                split_metrics["validation"]["point_nrmse_range_pct"]
                + 0.05 * split_metrics["validation"]["ramp_risk_mae_kw"]
                + 2 * (1 - split_metrics["validation"]["high_power_f1"])
                + 2 * (1 - split_metrics["validation"]["braking_f1"])
            )
            candidates[group_name] = (validation_score, test_prediction)
            del features
        selected_group = min(candidates, key=lambda name: candidates[name][0])
        selections.append(
            {
                "horizon_s": horizon,
                "selected_feature_group": selected_group,
                "validation_score": candidates[selected_group][0],
            }
        )
        selected_prediction = candidates[selected_group][1]
        for test_row, position in enumerate(test):
            origin = int(indices[position])
            for step in range(horizon):
                prediction_rows.append(
                    {
                        "origin_index": origin,
                        "target_index": origin + step + 1,
                        "forecast_horizon_s": horizon,
                        "step_ahead_s": step + 1,
                        "method": "state_direct_power_feature_selected",
                        "feature_group": selected_group,
                        "origin_power_kw": power[origin],
                        "power_pred_kw": selected_prediction[test_row, step],
                        "power_actual_kw": target[position, step],
                    }
                )
    pd.DataFrame(rows).to_csv(args.out_dir / "feature_group_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(
        args.out_dir / "feature_group_selection.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.out_dir / "feature_selected_predictions.csv", index=False
    )
    (args.out_dir / "feature_group_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(selections).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
