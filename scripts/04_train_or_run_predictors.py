"""Train and evaluate explicitly separated short-horizon predictors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fc_power.prediction.condition_aware_predictor import build_model
from fc_power.vehicle_dynamics import VehicleParams, force_power

H_MAX = 15
HISTORY = 30
HORIZONS = (1, 3, 5, 10, 15)


def operating_mode(speed: float, acceleration: float) -> np.ndarray:
    """Return transparent rule-state flags used as model inputs."""

    return np.asarray(
        [
            speed < 0.3,
            acceleration > 0.3,
            acceleration < -0.3,
            abs(acceleration) <= 0.1,
            speed > 13.9,
        ],
        dtype=float,
    )


def feature_vector(
    speed_all: np.ndarray,
    acceleration_all: np.ndarray,
    power_all: np.ndarray,
    timestamps: pd.DatetimeIndex,
    soc_all: np.ndarray,
    index: int,
) -> np.ndarray:
    """Build a causal state vector from the preceding 30 seconds."""

    sl = slice(index - HISTORY, index + 1)
    speed = speed_all[sl]
    acceleration = acceleration_all[sl]
    power = power_all[sl]
    timestamp = timestamps[index]

    summary = []
    for values in (speed, acceleration, power):
        for width in (3, 5, 10, 20, 30):
            window = values[-width:]
            summary.extend((window.mean(), window.std(), window.min(), window.max()))

    seconds_of_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    periodic = [
        np.sin(2 * np.pi * seconds_of_day / 86400),
        np.cos(2 * np.pi * seconds_of_day / 86400),
    ]
    return np.r_[
        speed,
        acceleration,
        power,
        summary,
        periodic,
        operating_mode(speed[-1], acceleration[-1]),
        soc_all[index] / 100,
    ]


def calibrated_dynamics_power(
    initial_speed: float,
    future_speed: np.ndarray,
    traction: dict,
    braking: dict,
) -> np.ndarray:
    acceleration = np.diff(np.r_[initial_speed, future_speed])
    wheel_power = force_power(future_speed, acceleration, VehicleParams())["p_wheel_kw"]
    return np.where(
        wheel_power >= 0,
        traction["intercept_kw"] + traction["slope"] * wheel_power,
        braking["intercept_kw"] + braking["slope"] * wheel_power,
    )


def build_supervised(data: pd.DataFrame, traction: dict, braking: dict):
    segment = data["segment_id"].to_numpy()
    speed = data["speed_smooth_mps"].to_numpy()
    acceleration = data["acceleration_smooth_mps2"].to_numpy()
    power = data["p_dem_measured_kw"].to_numpy()
    timestamps = pd.DatetimeIndex(data["timestamp"])
    soc = data["soc_pct"].to_numpy()
    indices = []
    for index in range(HISTORY, len(data) - H_MAX):
        if segment[index - HISTORY] == segment[index + H_MAX]:
            indices.append(index)

    indices = np.asarray(indices)
    history_rows = indices - HISTORY
    speed_history = sliding_window_view(speed, HISTORY + 1)[history_rows]
    acceleration_history = sliding_window_view(acceleration, HISTORY + 1)[history_rows]
    power_history = sliding_window_view(power, HISTORY + 1)[history_rows]

    summary_columns = []
    for values in (speed_history, acceleration_history, power_history):
        for width in (3, 5, 10, 20, 30):
            window = values[:, -width:]
            summary_columns.extend(
                (
                    window.mean(axis=1),
                    window.std(axis=1),
                    window.min(axis=1),
                    window.max(axis=1),
                )
            )
    seconds_of_day = (
        timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second
    ).to_numpy()[indices]
    periodic = np.c_[
        np.sin(2 * np.pi * seconds_of_day / 86400),
        np.cos(2 * np.pi * seconds_of_day / 86400),
    ]
    current_speed = speed[indices]
    current_acceleration = acceleration[indices]
    modes = np.c_[
        current_speed < 0.3,
        current_acceleration > 0.3,
        current_acceleration < -0.3,
        np.abs(current_acceleration) <= 0.1,
        current_speed > 13.9,
    ].astype(float)
    features = np.c_[
        speed_history,
        acceleration_history,
        power_history,
        *summary_columns,
        periodic,
        modes,
        soc[indices] / 100,
    ]

    speed_targets = sliding_window_view(speed, H_MAX)[indices + 1]
    power_targets = sliding_window_view(power, H_MAX)[indices + 1]
    future_acceleration = np.diff(np.c_[speed[indices], speed_targets], axis=1)
    wheel_power = force_power(speed_targets, future_acceleration, VehicleParams())[
        "p_wheel_kw"
    ]
    physics_targets = np.where(
        wheel_power >= 0,
        traction["intercept_kw"] + traction["slope"] * wheel_power,
        braking["intercept_kw"] + braking["slope"] * wheel_power,
    )
    residual_targets = power_targets - physics_targets
    return indices, features, speed_targets, power_targets, residual_targets


def error_summary(actual: np.ndarray, predicted: np.ndarray, full_scale: float) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    rms = np.sqrt(np.mean(np.asarray(actual) ** 2))
    return {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "nmae_range_pct": 100 * mae / full_scale,
        "nrmse_range_pct": 100 * rmse / full_scale,
        "rmse_over_actual_rms_pct": 100 * rmse / rms,
    }


def evaluate(predictions: pd.DataFrame, train_power: np.ndarray) -> pd.DataFrame:
    full_scale = float(
        predictions["power_actual_kw"].max() - predictions["power_actual_kw"].min()
    )
    high_threshold = float(np.quantile(train_power, 0.90))
    rows = []
    for method, group in predictions.groupby("method"):
        actual_matrix = group.pivot(
            index="origin_index", columns="horizon_s", values="power_actual_kw"
        )
        predicted_matrix = group.pivot(
            index="origin_index", columns="horizon_s", values="power_pred_kw"
        )
        speed_actual = group.pivot(
            index="origin_index", columns="horizon_s", values="speed_actual_mps"
        )
        speed_predicted = group.pivot(
            index="origin_index", columns="horizon_s", values="speed_pred_mps"
        )
        current_power = group.groupby("origin_index")["origin_power_kw"].first()

        for horizon in HORIZONS:
            actual = actual_matrix.loc[:, 1:horizon]
            predicted = predicted_matrix.loc[:, 1:horizon]
            actual_mean = actual.mean(axis=1).to_numpy()
            predicted_mean = (
                group.loc[
                    group["horizon_s"].eq(horizon),
                    ["origin_index", "window_mean_power_pred_kw"],
                ]
                .set_index("origin_index")
                .reindex(actual.index)["window_mean_power_pred_kw"]
                .to_numpy()
            )
            metrics = error_summary(actual_mean, predicted_mean, full_scale)

            actual_energy = actual.sum(axis=1).to_numpy() / 3600
            predicted_energy = predicted_mean * horizon / 3600
            energy_error = predicted_energy - actual_energy

            actual_high = actual.max(axis=1).to_numpy() >= high_threshold
            predicted_high = predicted.max(axis=1).to_numpy() >= high_threshold
            actual_braking = actual.min(axis=1).to_numpy() < -5
            predicted_braking = predicted.min(axis=1).to_numpy() < -5

            actual_with_origin = np.c_[current_power.to_numpy(), actual.to_numpy()]
            predicted_with_origin = np.c_[
                current_power.to_numpy(), predicted.to_numpy()
            ]
            actual_ramp = np.abs(np.diff(actual_with_origin, axis=1)).sum(axis=1)
            predicted_ramp = np.abs(np.diff(predicted_with_origin, axis=1)).sum(axis=1)
            ramp_error = predicted_ramp - actual_ramp

            rows.append(
                {
                    "method": method,
                    "horizon_s": horizon,
                    "speed_mae_mps": mean_absolute_error(
                        speed_actual[horizon], speed_predicted[horizon]
                    ),
                    "speed_rmse_mps": mean_squared_error(
                        speed_actual[horizon], speed_predicted[horizon]
                    )
                    ** 0.5,
                    **metrics,
                    "energy_mae_kwh": np.mean(np.abs(energy_error)),
                    "energy_rmse_kwh": np.sqrt(np.mean(energy_error**2)),
                    "high_power_accuracy": accuracy_score(actual_high, predicted_high),
                    "high_power_balanced_accuracy": balanced_accuracy_score(
                        actual_high, predicted_high
                    ),
                    "high_power_f1": f1_score(
                        actual_high, predicted_high, zero_division=0
                    ),
                    "braking_accuracy": accuracy_score(
                        actual_braking, predicted_braking
                    ),
                    "braking_balanced_accuracy": balanced_accuracy_score(
                        actual_braking, predicted_braking
                    ),
                    "braking_f1": f1_score(
                        actual_braking, predicted_braking, zero_division=0
                    ),
                    "ramp_risk_mae_kw": np.mean(np.abs(ramp_error)),
                    "ramp_risk_rmse_kw": np.sqrt(np.mean(ramp_error**2)),
                    "normalization_full_scale_kw": full_scale,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dynamics-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    pipeline_started = time.perf_counter()
    data = pd.read_csv(args.input, parse_dates=["timestamp"])
    dynamics = json.loads(args.dynamics_metrics.read_text(encoding="utf-8"))
    traction = dynamics["models"]["traction"]
    braking = dynamics["models"]["braking"]
    indices, features, speed_targets, power_targets, residual_targets = (
        build_supervised(data, traction, braking)
    )
    print(
        f"feature_build_s={time.perf_counter() - pipeline_started:.1f}, "
        f"samples={len(indices)}, features={features.shape[1]}",
        flush=True,
    )

    train_mask = indices < int(0.70 * len(data))
    validation_mask = (indices >= int(0.70 * len(data))) & (
        indices < int(0.85 * len(data))
    )
    test_mask = indices >= int(0.85 * len(data))
    train_positions = np.flatnonzero(train_mask)
    train_positions = train_positions[
        np.linspace(
            0, len(train_positions) - 1, min(60_000, len(train_positions))
        ).astype(int)
    ]
    validation_positions = np.flatnonzero(validation_mask)
    test_positions = np.flatnonzero(test_mask)
    current_speed = data["speed_smooth_mps"].to_numpy()[indices]
    current_acceleration = data["acceleration_smooth_mps2"].to_numpy()[indices]
    regime = np.select(
        [
            current_speed < 0.3,
            current_acceleration > 0.3,
            current_acceleration < -0.3,
            np.abs(current_acceleration) <= 0.1,
        ],
        ["idle", "acceleration", "deceleration", "cruise"],
        default="other",
    )

    started = time.perf_counter()
    # A shared state partition avoids fitting three expensive forests. Each
    # semantic head is standardized independently so power-scale variance does
    # not dominate speed or residual targets.
    power_mean_targets = np.column_stack(
        [power_targets[:, :horizon].mean(axis=1) for horizon in HORIZONS]
    )
    residual_mean_targets = np.column_stack(
        [residual_targets[:, :horizon].mean(axis=1) for horizon in HORIZONS]
    )
    joint_targets = np.c_[
        speed_targets,
        power_targets,
        residual_targets,
        power_mean_targets,
        residual_mean_targets,
    ]
    target_mean = joint_targets[train_positions].mean(axis=0)
    target_scale = joint_targets[train_positions].std(axis=0)
    target_scale[target_scale < 1e-9] = 1.0
    standardized_targets = (joint_targets - target_mean) / target_scale
    shared_model = build_model(seed=2026).fit(
        features[train_positions], standardized_targets[train_positions]
    )
    print(f"model_fit_s={time.perf_counter() - started:.1f}", flush=True)
    joint_predictions = (
        shared_model.predict(features[test_positions]) * target_scale + target_mean
    )
    speed_predictions = np.maximum(0, joint_predictions[:, :H_MAX])
    direct_power_predictions = joint_predictions[:, H_MAX : 2 * H_MAX]
    residual_predictions = joint_predictions[:, 2 * H_MAX : 3 * H_MAX]
    direct_mean_predictions = joint_predictions[:, 3 * H_MAX : 3 * H_MAX + 5]
    residual_mean_predictions = joint_predictions[:, 3 * H_MAX + 5 :]
    # Dedicated window heads optimize the quantities consumed by MPC. They do
    # not replace the per-second trajectory used for ramp/event evaluation.
    compact_columns = [
        30,
        61,
        92,
        93,
        97,
        101,
        105,
        109,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
    ]
    scaler = StandardScaler().fit(features[train_positions][:, compact_columns])
    neighbor_model = KNeighborsRegressor(
        n_neighbors=25,
        weights="distance",
        p=2,
        algorithm="ball_tree",
        leaf_size=40,
        n_jobs=-1,
    ).fit(
        scaler.transform(features[train_positions][:, compact_columns]),
        power_mean_targets[train_positions],
    )
    neighbor_validation = neighbor_model.predict(
        scaler.transform(features[validation_positions][:, compact_columns])
    )
    neighbor_test = neighbor_model.predict(
        scaler.transform(features[test_positions][:, compact_columns])
    )
    boosted_mean_predictions = []
    blend_weights = []
    for horizon_index, horizon in enumerate(HORIZONS):
        model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.04,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=2030 + horizon,
            early_stopping_rounds=30,
        )
        model.fit(
            features[train_positions],
            power_mean_targets[train_positions, horizon_index],
            eval_set=[
                (
                    features[validation_positions],
                    power_mean_targets[validation_positions, horizon_index],
                )
            ],
            verbose=False,
        )
        xgb_validation = model.predict(features[validation_positions])
        xgb_test = model.predict(features[test_positions])
        validation_target = power_mean_targets[validation_positions, horizon_index]
        candidates = np.linspace(0, 1, 11)
        validation_rmse = [
            np.sqrt(
                np.mean(
                    (
                        weight * xgb_validation
                        + (1 - weight) * neighbor_validation[:, horizon_index]
                        - validation_target
                    )
                    ** 2
                )
            )
            for weight in candidates
        ]
        weight = float(candidates[int(np.argmin(validation_rmse))])
        blend_weights.append(weight)
        horizon_prediction = (
            weight * xgb_test + (1 - weight) * neighbor_test[:, horizon_index]
        )
        boosted_mean_predictions.append(horizon_prediction)
    boosted_mean_predictions = np.column_stack(boosted_mean_predictions)
    print(f"xgb_blend_weights={dict(zip(HORIZONS, blend_weights))}", flush=True)
    rows = []
    measured_power = data["p_dem_measured_kw"].to_numpy()
    measured_speed = data["speed_smooth_mps"].to_numpy()

    for test_row, position in enumerate(test_positions):
        origin = int(indices[position])
        predicted_speed = speed_predictions[test_row]
        dynamics_power = calibrated_dynamics_power(
            measured_speed[origin], predicted_speed, traction, braking
        )
        dynamics_means = {
            horizon: dynamics_power[:horizon].mean() for horizon in HORIZONS
        }
        methods = {
            "speed_only_dynamics": (dynamics_power, dynamics_means),
            "state_direct_power": (
                direct_power_predictions[test_row],
                dict(zip(HORIZONS, boosted_mean_predictions[test_row])),
            ),
            "hybrid_physics_corrected": (
                dynamics_power + residual_predictions[test_row],
                {
                    horizon: dynamics_means[horizon]
                    + residual_mean_predictions[test_row, horizon_index]
                    for horizon_index, horizon in enumerate(HORIZONS)
                },
            ),
        }
        for method, (predicted_power, predicted_means) in methods.items():
            for step in range(H_MAX):
                rows.append(
                    {
                        "origin_index": origin,
                        "target_index": origin + step + 1,
                        "horizon_s": step + 1,
                        "method": method,
                        "origin_power_kw": measured_power[origin],
                        "speed_pred_mps": predicted_speed[step],
                        "speed_actual_mps": speed_targets[position, step],
                        "power_pred_kw": predicted_power[step],
                        "power_actual_kw": power_targets[position, step],
                        "window_mean_power_pred_kw": predicted_means.get(
                            step + 1, np.nan
                        ),
                    }
                )

    predictions = pd.DataFrame(rows)
    metrics = evaluate(predictions, measured_power[indices[train_mask]])
    metrics["training_runtime_s"] = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    metrics.to_csv(args.metrics, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
