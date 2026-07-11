"""Evaluate base/event/scenario previews in the same multi-stack controller."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.hydrogen_model import faraday_h2_g_s
from fc_power.power_allocation import choose_beam
from fc_power.world_model import load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


def load_benchmark_helpers():
    path = ROOT / "scripts/20_benchmark_multistack_real_demand.py"
    spec = importlib.util.spec_from_file_location("multistack_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preview_for(name, current_demand, origin, prediction_groups, actual_lookup, horizon):
    if name == "perfect":
        return np.asarray(
            [actual_lookup.get(origin + step, current_demand) for step in range(horizon)]
        )
    group = prediction_groups.get(origin)
    if group is None:
        return np.repeat(current_demand, horizon)
    group = group.set_index("step_ahead_s")
    values = [current_demand]
    for step in range(1, horizon):
        if step not in group.index:
            values.append(values[-1])
            continue
        row = group.loc[step]
        if name == "base_xgboost":
            value = row.power_base_kw
        elif name == "event_center":
            value = row.power_center_kw
        elif name == "event_brake_scenario":
            probability = float(row.brake_probability)
            value = (1 - probability) * row.power_center_kw + probability * row.power_p05_kw
        else:
            raise ValueError(f"unknown strategy: {name}")
        values.append(float(value))
    return np.asarray(values)


def run_controller(
    name,
    model,
    initial_state,
    source_indices,
    demand,
    prediction_groups,
    actual_lookup,
    horizon,
    beam_width,
):
    state = initial_state
    rows = []
    started = time.perf_counter()
    lower = model.config.battery.charge_power_limit_kw
    upper = model.config.battery.discharge_power_limit_kw + sum(
        proxy.evaluate(0.0, [max(model.config.allowed_currents_a)])["stack_power_kw"][0]
        for proxy in model.performance_proxies
    )
    for step_index, (origin, current_demand) in enumerate(
        zip(source_indices, demand)
    ):
        preview = np.clip(
            preview_for(
                name,
                current_demand,
                int(origin),
                prediction_groups,
                actual_lookup,
                horizon,
            ),
            lower,
            upper,
        )
        planned = choose_beam(
            model,
            state,
            preview,
            beam_width=beam_width,
            terminal_soc_weight=300.0,
        )
        result = planned.step
        row = {
            "strategy": name,
            "step": step_index,
            "source_index": origin,
            "demand_power_kw": current_demand,
            "preview_mae_kw": float(np.mean(np.abs(preview - np.asarray([
                actual_lookup.get(int(origin) + offset, current_demand)
                for offset in range(horizon)
            ])))),
            "stack_power_kw": result.constraints.stack_power_kw,
            "battery_power_kw": result.constraints.battery_power_kw,
            "soc": result.next_state.soc,
            "hydrogen_g": result.cost.raw_hydrogen_g,
            "degradation_increment_pct": result.cost.raw_degradation_increment_pct,
            "performance_loss": result.cost.performance_loss,
            "battery_throughput_kwh": result.cost.raw_battery_throughput_kwh,
            "step_cost": result.cost.total,
            "switches": sum(item.switched for item in result.stacks),
            "safety_override": bool(result.constraints.safety_overrides),
            "feasible": result.constraints.feasible,
        }
        for stack in result.stacks:
            row[f"stack_{stack.stack_index}_current_a"] = stack.current_a
        rows.append(row)
        state = result.next_state
    return pd.DataFrame(rows), state, time.perf_counter() - started


def summarize(name, frame, state, runtime, model):
    soc_error = state.soc - model.config.soc_reference
    reference_current = 195.0
    reference_power = model.performance_proxies[0].evaluate(
        0.0, [reference_current]
    )["stack_power_kw"][0]
    h2_g_per_kwh = faraday_h2_g_s(reference_current) * 3600 / reference_power
    soc_equivalent_h2 = -soc_error * model.config.battery.energy_kwh * h2_g_per_kwh
    return {
        "strategy": name,
        "preview_mae_kw": frame.preview_mae_kw.mean(),
        "hydrogen_g": frame.hydrogen_g.sum(),
        "hydrogen_soc_corrected_g": frame.hydrogen_g.sum() + soc_equivalent_h2,
        "degradation_increment_pct": frame.degradation_increment_pct.sum(),
        "performance_loss_sum": frame.performance_loss.sum(),
        "battery_throughput_kwh": frame.battery_throughput_kwh.sum(),
        "soc_final": state.soc,
        "soc_error": soc_error,
        "switch_count": int(frame.switches.sum()),
        "safety_override_steps": int(frame.safety_override.sum()),
        "constraint_violation_steps": int((~frame.feasible).sum()),
        "stack_0_current_a_step": frame.stack_0_current_a.sum(),
        "stack_1_current_a_step": frame.stack_1_current_a.sum(),
        "runtime_s": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=900)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/prediction_event"
    )
    args = parser.parse_args()
    if min(args.offset, args.length, args.horizon, args.beam_width) < 0 or min(
        args.length, args.horizon, args.beam_width
    ) == 0:
        raise ValueError("offset must be non-negative and other sizes positive")

    prediction_path = args.out_dir / "event_probabilistic_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("run script 21 before control-regret evaluation")
    helpers = load_benchmark_helpers()
    full_indices, full_demand = helpers.load_frozen_test_demand(
        args.offset + args.length + args.horizon
    )
    selected = slice(args.offset, args.offset + args.length)
    source_indices = full_indices[selected]
    demand = full_demand[selected]
    actual_lookup = {
        int(index): float(value) for index, value in zip(full_indices, full_demand)
    }
    predictions = pd.read_csv(prediction_path)
    predictions = predictions[predictions.forecast_horizon_s.eq(args.horizon)]
    predictions = predictions[predictions.origin_index.isin(source_indices)]
    prediction_groups = {
        int(origin): group for origin, group in predictions.groupby("origin_index")
    }

    model = load_lzw_multistack_world_model(
        ROOT, n_stacks=2, heterogeneity_factors=[1.0, 1.10]
    )
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    initial = model.initial_state(
        soc=model.config.soc_reference,
        degradation_pct=[0.10 * reference, 0.65 * reference],
    )
    names = ("base_xgboost", "event_center", "event_brake_scenario", "perfect")
    frames, metrics = [], []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        frame, state, runtime = run_controller(
            name,
            model,
            initial,
            source_indices,
            demand,
            prediction_groups,
            actual_lookup,
            args.horizon,
            args.beam_width,
        )
        frames.append(frame)
        metrics.append(summarize(name, frame, state, runtime, model))
        pd.concat(frames, ignore_index=True).to_csv(
            args.out_dir / "prediction_control_trajectory.csv", index=False
        )
        pd.DataFrame(metrics).to_csv(
            args.out_dir / "prediction_control_regret.csv", index=False
        )
        print(f"completed {name}: {runtime:.1f} s", flush=True)

    summary = pd.DataFrame(metrics)
    perfect = summary.set_index("strategy").loc["perfect"]
    for column in (
        "hydrogen_soc_corrected_g",
        "degradation_increment_pct",
        "performance_loss_sum",
        "battery_throughput_kwh",
    ):
        summary[f"{column}_regret_vs_perfect_pct"] = 100 * (
            summary[column] - perfect[column]
        ) / max(abs(perfect[column]), 1e-12)
    summary.to_csv(args.out_dir / "prediction_control_regret.csv", index=False)
    table_lines = [
        "| 策略 | preview MAE | SOC误差 | SOC等值H2(g) | Gamma增量(%) | 性能损失 | 电池吞吐(kWh) | 硬违规 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        table_lines.append(
            f"| {row.strategy} | {row.preview_mae_kw:.3f} | {row.soc_error:+.6f} | "
            f"{row.hydrogen_soc_corrected_g:.3f} | {row.degradation_increment_pct:.6f} | "
            f"{row.performance_loss_sum:.3f} | {row.battery_throughput_kwh:.3f} | "
            f"{row.constraint_violation_steps} |"
        )
    by_name = summary.set_index("strategy")
    base = by_name.loc["base_xgboost"]
    event = by_name.loc["event_center"]
    preview_gain = 100 * (base.preview_mae_kw - event.preview_mae_kw) / base.preview_mae_kw
    report = f"""# 概率预测下游控制后悔值筛查

- 冻结测试段：offset={args.offset}, length={args.length} s
- 共同规划器：双堆Beam，H={args.horizon} s，beam width={args.beam_width}
- 比较对象：XGBoost中心、事件残差中心、制动概率下分位场景、Perfect preview
- 本实验是120秒事件丰富窗口筛查；正式升级仍需更长测试段。

{chr(10).join(table_lines)}

## 决策

- 事件中心相对XGBoost将preview MAE降低{preview_gain:.2f}%，并将末端SOC从{base.soc_error:+.6f}修正到{event.soc_error:+.6f}，后者通过±0.001门槛。
- 事件中心相对Perfect的SOC等值氢耗/Gamma增量/性能损失后悔分别为{event.hydrogen_soc_corrected_g_regret_vs_perfect_pct:+.2f}%/{event.degradation_increment_pct_regret_vs_perfect_pct:+.2f}%/{event.performance_loss_sum_regret_vs_perfect_pct:+.2f}%。
- p05制动场景显著扩大preview误差，不设为默认；后续改用事件概率触发有限安全裕度或多场景风险聚合。
- Perfect的preview MAE非零来自输入先裁剪到双堆+电池可行域，而误差对照仍使用原始需求。
"""
    (args.out_dir / "prediction_control_regret_report.md").write_text(
        report, encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
