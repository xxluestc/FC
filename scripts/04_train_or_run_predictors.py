"""Train horizon-specific direct-power predictors without future leakage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_power.prediction.horizon_models import BrakeAwareExtraTrees, build_regressor
from fc_power.vehicle_dynamics import VehicleParams, force_power

HISTORY = 30
HORIZONS = (1, 3, 5, 10)
FAMILIES = ("extratrees", "hist_gradient_boosting", "xgboost", "brake_aware_extratrees")


def stop_state_features(speed: np.ndarray, odometer: np.ndarray, segment: np.ndarray):
    """Return causal stop duration/time/distance states for every row."""

    n = len(speed)
    stop_duration = np.zeros(n)
    time_since_stop = np.zeros(n)
    distance_since_stop = np.zeros(n)
    stopped = speed < 0.3
    for index in range(1, n):
        if segment[index] != segment[index - 1]:
            continue
        if stopped[index]:
            stop_duration[index] = stop_duration[index - 1] + 1
            time_since_stop[index] = 0
            distance_since_stop[index] = 0
        else:
            stop_duration[index] = 0
            time_since_stop[index] = time_since_stop[index - 1] + 1
            delta = odometer[index] - odometer[index - 1]
            if not np.isfinite(delta) or delta < 0 or delta > 1:
                delta = speed[index] / 1000
            distance_since_stop[index] = distance_since_stop[index - 1] + delta
    return stop_duration, time_since_stop, distance_since_stop


def build_supervised(data: pd.DataFrame):
    """Build causal history/state features and future power targets."""

    segment = data["segment_id"].to_numpy()
    speed = data["speed_smooth_mps"].to_numpy()
    acceleration = data["acceleration_smooth_mps2"].to_numpy()
    power = data["p_dem_measured_kw"].to_numpy()
    soc = data["soc_pct"].to_numpy() / 100
    odometer = data["odometer_km"].interpolate().ffill().bfill().to_numpy()
    timestamp = pd.DatetimeIndex(data["timestamp"])
    stop_duration, time_since_stop, distance_since_stop = stop_state_features(
        speed, odometer, segment
    )

    indices = np.asarray(
        [
            index
            for index in range(HISTORY, len(data) - max(HORIZONS))
            if segment[index - HISTORY] == segment[index + max(HORIZONS)]
        ]
    )
    rows = indices - HISTORY
    histories = [
        sliding_window_view(values, HISTORY + 1)[rows]
        for values in (speed, acceleration, power)
    ]
    summaries = []
    for values in histories:
        for width in (3, 5, 10, 20, 30):
            window = values[:, -width:]
            summaries.extend(
                (window.mean(1), window.std(1), window.min(1), window.max(1))
            )
    seconds = (
        timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    ).to_numpy()[indices]
    modes = np.c_[
        speed[indices] < 0.3,
        acceleration[indices] > 0.3,
        acceleration[indices] < -0.3,
        np.abs(acceleration[indices]) <= 0.1,
        power[indices] < -5,
        power[indices] > np.quantile(power[indices], 0.9),
    ].astype(float)
    features = np.c_[
        *histories,
        *summaries,
        np.sin(2 * np.pi * seconds / 86400),
        np.cos(2 * np.pi * seconds / 86400),
        modes,
        soc[indices],
        stop_duration[indices],
        np.log1p(time_since_stop[indices]),
        np.log1p(distance_since_stop[indices]),
    ]
    targets = sliding_window_view(power, max(HORIZONS))[indices + 1]
    return indices, features, targets


def split_positions(indices: np.ndarray, n_rows: int, horizon: int):
    """Chronological 70/15/15 split with a horizon-sized purge gap."""

    train_end = int(0.70 * n_rows)
    validation_end = int(0.85 * n_rows)
    train = np.flatnonzero(indices + horizon < train_end - horizon)
    validation = np.flatnonzero(
        (indices >= train_end + horizon)
        & (indices + horizon < validation_end - horizon)
    )
    test = np.flatnonzero(indices >= validation_end + horizon)
    if len(train) > 40_000:
        train = train[np.linspace(0, len(train) - 1, 40_000).astype(int)]
    return train, validation, test


def metric_row(actual, predicted, train_power, full_scale, family, horizon, split):
    """Evaluate point trajectory, window energy, and event classification."""

    actual = np.asarray(actual).reshape(len(actual), -1)
    predicted = np.asarray(predicted).reshape(len(predicted), -1)
    error = predicted - actual
    actual_energy = actual.sum(1) / 3600
    predicted_energy = predicted.sum(1) / 3600
    high_threshold = np.quantile(train_power, 0.9)
    actual_high = actual.max(1) >= high_threshold
    predicted_high = predicted.max(1) >= high_threshold
    actual_brake = actual.min(1) < -5
    predicted_brake = predicted.min(1) < -5
    mean_actual = actual.mean(1)
    mean_predicted = predicted.mean(1)
    return {
        "model_family": family,
        "method": "state_direct_power",
        "horizon_s": horizon,
        "split": split,
        "n": len(actual),
        "point_mae_kw": np.mean(np.abs(error)),
        "point_rmse_kw": np.sqrt(np.mean(error**2)),
        "point_nmae_range_pct": 100 * np.mean(np.abs(error)) / full_scale,
        "point_nrmse_range_pct": 100 * np.sqrt(np.mean(error**2)) / full_scale,
        "window_mean_mae_kw": mean_absolute_error(mean_actual, mean_predicted),
        "window_mean_rmse_kw": mean_squared_error(mean_actual, mean_predicted) ** 0.5,
        "window_energy_mae_kwh": np.mean(np.abs(predicted_energy - actual_energy)),
        "window_energy_rmse_kwh": np.sqrt(
            np.mean((predicted_energy - actual_energy) ** 2)
        ),
        "high_power_f1": f1_score(actual_high, predicted_high, zero_division=0),
        "braking_f1": f1_score(actual_brake, predicted_brake, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dynamics-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("data/results/horizon_model_comparison.csv"),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    data = pd.read_csv(args.input, parse_dates=["timestamp"])
    indices, features, targets = build_supervised(data)
    full_scale = float(
        data["p_dem_measured_kw"].max() - data["p_dem_measured_kw"].min()
    )
    result_rows = []
    selected_rows = []
    selections = []

    for horizon in HORIZONS:
        train, validation, test = split_positions(indices, len(data), horizon)
        y = targets[:, :horizon]
        candidates = {}
        for family_index, family in enumerate(FAMILIES):
            model = (
                BrakeAwareExtraTrees(2100 + horizon)
                if family == "brake_aware_extratrees"
                else build_regressor(family, 2100 + 10 * family_index + horizon)
            )
            fit_start = time.perf_counter()
            model.fit(features[train], y[train])
            validation_prediction = np.asarray(
                model.predict(features[validation])
            ).reshape(len(validation), -1)
            test_prediction = np.asarray(model.predict(features[test])).reshape(
                len(test), -1
            )
            validation_metrics = metric_row(
                y[validation],
                validation_prediction,
                y[train].ravel(),
                full_scale,
                family,
                horizon,
                "validation",
            )
            test_metrics = metric_row(
                y[test],
                test_prediction,
                y[train].ravel(),
                full_scale,
                family,
                horizon,
                "test",
            )
            runtime = time.perf_counter() - fit_start
            validation_metrics["training_runtime_s"] = runtime
            test_metrics["training_runtime_s"] = runtime
            result_rows.extend((validation_metrics, test_metrics))
            # Selection is validation-only; event errors prevent a low-RMSE model
            # from ignoring braking/high-power windows.
            score = (
                validation_metrics["point_nrmse_range_pct"]
                + 2 * (1 - validation_metrics["high_power_f1"])
                + 2 * (1 - validation_metrics["braking_f1"])
            )
            candidates[family] = (score, test_prediction, test_metrics)
            print(
                f"H={horizon} family={family} val_score={score:.4f} runtime={runtime:.1f}s",
                flush=True,
            )

        selected_family = min(candidates, key=lambda name: candidates[name][0])
        prediction = candidates[selected_family][1]
        selections.append(
            {
                "horizon_s": horizon,
                "selected_family": selected_family,
                "validation_score": candidates[selected_family][0],
            }
        )
        for row_index, position in enumerate(test):
            origin = int(indices[position])
            for step in range(horizon):
                selected_rows.append(
                    {
                        "origin_index": origin,
                        "target_index": origin + step + 1,
                        "forecast_horizon_s": horizon,
                        "step_ahead_s": step + 1,
                        "method": "state_direct_power",
                        "model_family": selected_family,
                        "origin_power_kw": data["p_dem_measured_kw"].iat[origin],
                        "power_pred_kw": prediction[row_index, step],
                        "power_actual_kw": y[position, step],
                    }
                )

    comparison = pd.DataFrame(result_rows)
    selection = pd.DataFrame(selections)
    metrics = comparison.merge(selection, on="horizon_s", how="left")
    metrics["selected"] = metrics["model_family"].eq(metrics["selected_family"])
    predictions = pd.DataFrame(selected_rows)
    for path in (args.output, args.metrics, args.comparison):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    metrics.to_csv(args.metrics, index=False)
    comparison.to_csv(args.comparison, index=False)
    Path("data/results/horizon_model_selection.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    print(selection.to_string(index=False))
    print(f"total_runtime_s={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
