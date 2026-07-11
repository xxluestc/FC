"""Train event-conditioned point correction and conformal multi-step intervals."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from fc_power.prediction.event_conformal import EventConditionedResidualConformal


ROOT = Path(__file__).resolve().parents[1]


def write_report(metrics: pd.DataFrame, metadata: dict, out_dir: Path) -> None:
    rows = []
    for horizon in sorted(metrics.horizon_s.unique()):
        test = metrics[
            metrics.horizon_s.eq(horizon) & metrics.split.eq("test")
        ].set_index("method")
        base = test.loc["xgboost_point"]
        event = test.loc["event_residual_center"]
        rows.append(
            {
                "horizon_s": int(horizon),
                "base_mae_kw": base.point_mae_kw,
                "event_mae_kw": event.point_mae_kw,
                "mae_improvement_pct": 100
                * (base.point_mae_kw - event.point_mae_kw)
                / base.point_mae_kw,
                "base_rmse_kw": base.point_rmse_kw,
                "event_rmse_kw": event.point_rmse_kw,
                "high_f1_change": event.high_power_f1 - base.high_power_f1,
                "brake_f1_change": event.braking_f1 - base.braking_f1,
                "fixed_coverage": event.fixed_pointwise_90_coverage,
                "adaptive_coverage": event.pointwise_90_coverage,
                "adaptive_width_kw": event.mean_interval_width_kw,
            }
        )
    summary = pd.DataFrame(rows)
    table_lines = [
        "| H(s) | base MAE | event MAE | MAE改善 | RMSE变化 | high F1变化 | brake F1变化 | 固定覆盖 | 在线覆盖 | 区间宽度 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        table_lines.append(
            f"| {row.horizon_s} | {row.base_mae_kw:.4f} | {row.event_mae_kw:.4f} | "
            f"{row.mae_improvement_pct:.2f}% | {row.event_rmse_kw-row.base_rmse_kw:+.4f} | "
            f"{row.high_f1_change:+.4f} | {row.brake_f1_change:+.4f} | "
            f"{row.fixed_coverage:.2%} | {row.adaptive_coverage:.2%} | "
            f"{row.adaptive_width_kw:.2f} kW |"
        )
    report = f"""# 事件条件化概率功率预测报告

## 方法

XGBoost给出直接多步点预测；未来制动/高功率分类器提供事件概率；验证集前半拟合事件交互残差，后半标定分组conformal区间。测试阶段的自适应区间只在完整H秒结果已经到达后更新，未使用当前或未来标签。

## 测试结果

{chr(10).join(table_lines)}

## 决策

- 事件残差中心小幅改善H=5/10 MAE和高功率F1，可保留为候选中心预测。
- H=10制动F1和窗口均值RMSE没有同步改善，因此不能仅凭总体MAE替换冻结XGBoost。
- 在线自适应覆盖率高于固定区间，但仍低于名义90%，表明测试段存在分布漂移；当前区间不升级为默认鲁棒MPC输入。
- 下一步以相同多堆world model比较base/event/perfect preview的控制后悔值，并把制动概率直接作为场景或安全裕度，而不是强迫中心轨迹跨过-5 kW阈值。

运行时间：{metadata['runtime_s']:.1f} s。
"""
    (out_dir / "event_probabilistic_report.md").write_text(
        report, encoding="utf-8"
    )


def load_baseline_helpers():
    path = ROOT / "scripts/04_train_or_run_predictors.py"
    spec = importlib.util.spec_from_file_location("prediction_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classifier(seed):
    return ExtraTreesClassifier(
        n_estimators=80,
        max_depth=22,
        min_samples_leaf=3,
        max_features=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def probabilistic_metrics(actual, forecast):
    covered = (actual >= forecast.lower) & (actual <= forecast.upper)
    return {
        "pointwise_90_coverage": float(covered.mean()),
        "simultaneous_window_90_coverage": float(covered.all(axis=1).mean()),
        "mean_interval_width_kw": float((forecast.upper - forecast.lower).mean()),
        "p95_interval_width_kw": float(
            np.quantile(forecast.upper - forecast.lower, 0.95)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/baseline_power_demand.csv",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10])
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/prediction_event"
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        metrics = pd.read_csv(args.out_dir / "event_probabilistic_metrics.csv")
        metadata = json.loads(
            (args.out_dir / "event_probabilistic_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        write_report(metrics, metadata, args.out_dir)
        return
    from fc_power.prediction.horizon_models import build_regressor
    unsupported = set(args.horizons).difference({1, 3, 5, 10})
    if unsupported:
        raise ValueError(f"unsupported horizons: {sorted(unsupported)}")
    if not args.input.exists():
        fallback = ROOT / "data/key/baseline_power_demand.csv.gz"
        if not fallback.exists():
            raise FileNotFoundError("materialize baseline power demand first")
        args.input = fallback

    helpers = load_baseline_helpers()
    data = pd.read_csv(args.input, parse_dates=["timestamp"])
    indices, features, targets = helpers.build_supervised(data)
    full_scale = float(data.p_dem_measured_kw.max() - data.p_dem_measured_kw.min())
    rows, prediction_rows, calibration_summaries = [], [], []
    started = time.perf_counter()

    for horizon in args.horizons:
        train, validation, test = helpers.split_positions(indices, len(data), horizon)
        y = targets[:, :horizon]
        high_threshold = float(np.quantile(y[train], 0.9))
        train_brake = y[train].min(axis=1) < -5.0
        train_high = y[train].max(axis=1) >= high_threshold

        point_model = build_regressor("xgboost", 3100 + horizon)
        brake_model = classifier(3200 + horizon)
        high_model = classifier(3300 + horizon)
        point_model.fit(features[train], y[train])
        brake_model.fit(features[train], train_brake)
        high_model.fit(features[train], train_high)

        val_point = np.asarray(point_model.predict(features[validation])).reshape(
            len(validation), horizon
        )
        test_point = np.asarray(point_model.predict(features[test])).reshape(
            len(test), horizon
        )
        val_brake_probability = brake_model.predict_proba(features[validation])[:, 1]
        val_high_probability = high_model.predict_proba(features[validation])[:, 1]
        test_brake_probability = brake_model.predict_proba(features[test])[:, 1]
        test_high_probability = high_model.predict_proba(features[test])[:, 1]

        calibrator = EventConditionedResidualConformal().fit(
            val_point,
            y[validation],
            val_brake_probability,
            val_high_probability,
        )
        val_tail = slice(calibrator.calibration_split_, None)
        validation_forecast = calibrator.predict(
            val_point[val_tail],
            val_brake_probability[val_tail],
            val_high_probability[val_tail],
        )
        test_forecast = calibrator.predict(
            test_point, test_brake_probability, test_high_probability
        )
        adaptive_test_forecast = calibrator.predict_adaptive(
            test_point,
            y[test],
            test_brake_probability,
            test_high_probability,
            delay_steps=horizon,
            rolling_window=2000,
        )

        for split_name, positions, actual, base, forecast in (
            (
                "validation_calibration_tail",
                validation[val_tail],
                y[validation][val_tail],
                val_point[val_tail],
                validation_forecast,
            ),
            ("test", test, y[test], test_point, adaptive_test_forecast),
        ):
            for method, prediction in (
                ("xgboost_point", base),
                ("event_residual_center", forecast.center),
            ):
                metric = helpers.metric_row(
                    actual,
                    prediction,
                    y[train].ravel(),
                    full_scale,
                    method,
                    horizon,
                    split_name,
                )
                metric["method"] = method
                if method == "event_residual_center":
                    metric.update(probabilistic_metrics(actual, forecast))
                    metric["interval_mode"] = (
                        "online_adaptive_delayed" if split_name == "test" else "fixed"
                    )
                    if split_name == "test":
                        fixed_metrics = probabilistic_metrics(actual, test_forecast)
                        metric.update(
                            {
                                f"fixed_{name}": value
                                for name, value in fixed_metrics.items()
                            }
                        )
                rows.append(metric)

        actual_test_brake = y[test].min(axis=1) < -5.0
        actual_test_high = y[test].max(axis=1) >= high_threshold
        calibration_summaries.append(
            {
                "horizon_s": horizon,
                "brake_classifier_f1": f1_score(
                    actual_test_brake,
                    test_brake_probability >= 0.5,
                    zero_division=0,
                ),
                "high_classifier_f1": f1_score(
                    actual_test_high,
                    test_high_probability >= 0.5,
                    zero_division=0,
                ),
                "event_group_counts": calibrator.group_counts_,
            }
        )
        for row_index, position in enumerate(test):
            for step in range(horizon):
                prediction_rows.append(
                    {
                        "origin_index": int(indices[position]),
                        "target_index": int(indices[position] + step + 1),
                        "forecast_horizon_s": horizon,
                        "step_ahead_s": step + 1,
                        "power_actual_kw": y[position, step],
                        "power_base_kw": test_point[row_index, step],
                        "power_center_kw": adaptive_test_forecast.center[
                            row_index, step
                        ],
                        "power_p05_kw": adaptive_test_forecast.lower[
                            row_index, step
                        ],
                        "power_p95_kw": adaptive_test_forecast.upper[
                            row_index, step
                        ],
                        "brake_probability": test_brake_probability[row_index],
                        "high_probability": test_high_probability[row_index],
                        "event_code": int(
                            adaptive_test_forecast.event_code[row_index]
                        ),
                    }
                )
        print(f"completed H={horizon}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.out_dir / "event_probabilistic_metrics.csv", index=False)
    predictions.to_csv(
        args.out_dir / "event_probabilistic_predictions.csv", index=False
    )
    metadata = {
        "interpretation": (
            "XGBoost point forecasts are corrected using validation-only event "
            "residual structure; intervals use the untouched chronological half "
            "of validation. Test observations are never used for fitting."
        ),
        "horizons": args.horizons,
        "nominal_pointwise_coverage": 0.90,
        "test_interval_mode": (
            "causal online adaptive conformal with horizon-delayed outcomes"
        ),
        "calibration": calibration_summaries,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "event_probabilistic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(metrics, metadata, args.out_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
