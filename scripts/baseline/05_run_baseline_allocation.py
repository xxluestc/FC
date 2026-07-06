"""运行固定配置的四策略功率分配基线，不进行任何参数搜索。

中文名：05_运行基线功率分配。
输入：需求功率表、固定预测结果、电堆档位表和 baseline.yaml。
输出：四种策略逐秒轨迹、总指标、相对变化和需求裁剪审计。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_allocation_helpers():
    """加载现有05脚本的已验证函数，避免维护两套控制逻辑。"""

    path = ROOT / "scripts" / "05_run_power_allocation.py"
    spec = importlib.util.spec_from_file_location("allocation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    allocation = config["allocation"]
    horizon = int(allocation["default_horizon_s"])
    helpers = load_allocation_helpers()
    vehicle = pd.read_csv(args.vehicle)
    predictions = pd.read_csv(args.predictions)
    stack = pd.read_csv(args.stack_map)

    # H=10的起点集合用于确定所有预测时域共同可用的连续测试段。
    causal = predictions[predictions["method"].eq("state_direct_power")]
    horizon_10 = causal[causal["forecast_horizon_s"].eq(10)]
    sequence = helpers.consecutive_test_sequence(
        horizon_10["origin_index"].unique(),
        maximum_length=int(allocation["test_length_s"]),
    )
    raw_demand = vehicle.loc[sequence, "p_dem_measured_kw"].to_numpy()
    lower = float(config["constraints"]["battery_charge_limit_kw"])
    upper = float(
        config["constraints"]["battery_discharge_limit_kw"] + stack.stack_power_kw.max()
    )
    demand = np.clip(raw_demand, lower, upper)
    clipped = np.abs(raw_demand - demand) > 1e-12

    prediction_map = {
        (
            int(row.forecast_horizon_s),
            int(row.origin_index),
            int(row.step_ahead_s),
        ): float(np.clip(row.power_pred_kw, lower, upper))
        for row in causal.itertuples()
    }
    actions = stack["stack_power_kw"].to_numpy()
    hydrogen = stack["faraday_h2_g_s"].to_numpy()
    degradation_proxy = stack["performance_loss_cost_normalized"].to_numpy()
    weights = {key: float(value) for key, value in allocation["weights"].items()}

    # Instant只看当前；Constant假定未来恒定；Perfect使用真实未来；Predicted使用因果预测。
    experiments = [
        ("instant", 1),
        ("constant", horizon),
        ("perfect", horizon),
        ("predicted", horizon),
    ]
    trajectories, metrics = [], []
    for strategy, strategy_horizon in experiments:
        trajectory, result = helpers.run_strategy(
            strategy,
            strategy_horizon,
            demand,
            sequence,
            prediction_map,
            actions,
            hydrogen,
            degradation_proxy,
            weights,
            confidence_decay=(
                float(allocation["confidence_decay"])
                if strategy == "predicted"
                else 0.0
            ),
            min_dwell=int(allocation["minimum_dwell_s"]),
        )
        trajectory["raw_demand_kw"] = raw_demand
        trajectory["was_clipped"] = clipped
        trajectories.append(trajectory)
        metrics.append(result)

    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    trajectory_frame.to_csv(args.out_dir / "baseline_trajectory.csv", index=False)
    metrics_frame.to_csv(args.out_dir / "baseline_allocation_metrics.csv", index=False)
    helpers.relative_improvements(metrics).to_csv(
        args.out_dir / "baseline_relative_improvements.csv", index=False
    )
    clipping = {
        "points": len(demand),
        "clipped_points": int(clipped.sum()),
        "clipped_share_pct": 100 * float(clipped.mean()),
        "absolute_clipped_energy_kwh": float(np.abs(raw_demand - demand).sum() / 3600),
        "maximum_clipped_power_kw": float(np.abs(raw_demand - demand).max()),
        "scope": "裁剪非零时，结果属于当前FC+电池可行域基线。",
    }
    (args.out_dir / "baseline_clipping_audit.json").write_text(
        json.dumps(clipping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
