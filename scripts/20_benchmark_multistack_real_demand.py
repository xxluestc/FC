"""Compare multi-stack baselines on the frozen real-demand test sequence."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.hydrogen_model import faraday_h2_g_s
from fc_power.power_allocation import (
    choose_average,
    choose_beam,
    choose_instant,
    choose_rotating,
)
from fc_power.world_model import load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def longest_consecutive(values):
    runs, current = [], []
    for value in np.sort(np.unique(values)):
        value = int(value)
        if current and value != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(value)
    if current:
        runs.append(current)
    if not runs:
        raise ValueError("no test origins available")
    return np.asarray(max(runs, key=len), dtype=int)


def load_frozen_test_demand(length: int) -> tuple[np.ndarray, np.ndarray]:
    processed = ROOT / "data/processed"
    key = ROOT / "data/key"
    vehicle_path = processed / "baseline_power_demand.csv"
    prediction_path = processed / "baseline_prediction_results.csv"
    if not vehicle_path.exists():
        vehicle_path = key / "baseline_power_demand.csv.gz"
    if not prediction_path.exists():
        prediction_path = key / "baseline_prediction_results.csv.gz"
    vehicle = pd.read_csv(vehicle_path, usecols=["p_dem_measured_kw"])
    predictions = pd.read_csv(
        prediction_path,
        usecols=["origin_index", "forecast_horizon_s", "method"],
    )
    origins = predictions.loc[
        predictions.method.eq("state_direct_power")
        & predictions.forecast_horizon_s.eq(10),
        "origin_index",
    ]
    sequence = longest_consecutive(origins)[:length]
    if len(sequence) < length:
        raise ValueError(f"requested {length} samples but only {len(sequence)} are available")
    return sequence, vehicle.loc[sequence, "p_dem_measured_kw"].to_numpy(dtype=float)


def run_strategy(
    name,
    model,
    initial_state,
    demand,
    horizon,
    rotation_period,
    beam_terminal_soc_weight,
):
    state = initial_state
    rows = []
    started = time.perf_counter()
    for index, current_demand in enumerate(demand):
        if name == "average":
            planned = choose_average(model, state, current_demand)
        elif name == "rotating":
            lead = (index // rotation_period) % model.n_stacks
            planned = choose_rotating(model, state, current_demand, lead)
        elif name == "instant_health":
            planned = choose_instant(model, state, current_demand)
        elif name == "beam_perfect":
            preview = demand[index : min(index + horizon, len(demand))]
            planned = choose_beam(
                model,
                state,
                preview,
                beam_width=8,
                terminal_soc_weight=beam_terminal_soc_weight,
            )
        else:
            raise ValueError(f"unknown strategy: {name}")

        step = planned.step
        row = {
            "strategy": name,
            "step": index,
            "demand_power_kw": current_demand,
            "stack_power_kw": step.constraints.stack_power_kw,
            "battery_power_kw": step.constraints.battery_power_kw,
            "soc": step.next_state.soc,
            "hydrogen_g": step.cost.raw_hydrogen_g,
            "degradation_increment_pct": step.cost.raw_degradation_increment_pct,
            "performance_loss": step.cost.performance_loss,
            "battery_throughput_kwh": step.cost.raw_battery_throughput_kwh,
            "switches": sum(item.switched for item in step.stacks),
            "power_balance_error_kw": step.constraints.power_balance_error_kw,
            "feasible": step.constraints.feasible,
            "violations": "|".join(step.constraints.violations),
            "safety_overrides": "|".join(step.constraints.safety_overrides),
        }
        for stack in step.stacks:
            row[f"stack_{stack.stack_index}_current_a"] = stack.current_a
            row[f"stack_{stack.stack_index}_damage_pct"] = stack.degradation_after_pct
        rows.append(row)
        state = step.next_state
    frame = pd.DataFrame(rows)
    return frame, state, time.perf_counter() - started


def summarize(name, frame, final_state, runtime_s, model):
    final_damage = np.asarray(
        [stack.health.degradation for stack in final_state.stacks], dtype=float
    )
    soc_delta = final_state.soc - model.config.soc_reference
    battery_energy_credit_kwh = soc_delta * model.config.battery.energy_kwh
    reference_current = 195.0
    reference_power = model.performance_proxies[0].evaluate(
        0.0, [reference_current]
    )["stack_power_kw"][0]
    reference_h2_g_per_kwh = float(
        faraday_h2_g_s(reference_current) * 3600.0 / reference_power
    )
    soc_equivalent_h2_g = -battery_energy_credit_kwh * reference_h2_g_per_kwh
    return {
        "strategy": name,
        "n_steps": len(frame),
        "hydrogen_g": float(frame.hydrogen_g.sum()),
        "soc_equivalent_hydrogen_g": soc_equivalent_h2_g,
        "hydrogen_soc_corrected_g": float(frame.hydrogen_g.sum())
        + soc_equivalent_h2_g,
        "degradation_increment_pct": float(frame.degradation_increment_pct.sum()),
        "performance_loss_sum": float(frame.performance_loss.sum()),
        "battery_throughput_kwh": float(frame.battery_throughput_kwh.sum()),
        "soc_final": final_state.soc,
        "soc_error": soc_delta,
        "switch_count": int(frame.switches.sum()),
        "constraint_violation_steps": int((~frame.feasible).sum()),
        "safety_override_steps": int(frame.safety_overrides.ne("").sum()),
        "max_power_balance_error_kw": float(frame.power_balance_error_kw.abs().max()),
        "final_damage_mean_pct": float(final_damage.mean()),
        "final_damage_range_pct": float(final_damage.max() - final_damage.min()),
        "stack_0_current_a_step": float(frame.stack_0_current_a.sum()),
        "stack_1_current_a_step": float(frame.stack_1_current_a.sum()),
        "runtime_s": runtime_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--beam-terminal-soc-weight", type=float, default=300.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("average", "rotating", "instant_health", "beam_perfect"),
        default=None,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/multistack"
    )
    args = parser.parse_args()
    if args.length <= 0 or args.horizon <= 0 or args.rotation_period <= 0:
        raise ValueError("length, horizon and rotation period must be positive")
    if args.beam_terminal_soc_weight < 0:
        raise ValueError("beam terminal SOC weight must be non-negative")

    source_index, raw_demand = load_frozen_test_demand(args.length)
    model = load_lzw_multistack_world_model(
        ROOT, n_stacks=2, heterogeneity_factors=[1.0, 1.10]
    )
    maximum_stack_power = sum(
        proxy.evaluate(0.0, [max(model.config.allowed_currents_a)])["stack_power_kw"][0]
        for proxy in model.performance_proxies
    )
    demand = np.clip(
        raw_demand,
        model.config.battery.charge_power_limit_kw,
        model.config.battery.discharge_power_limit_kw + maximum_stack_power,
    )
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    initial_state = model.initial_state(
        soc=model.config.soc_reference,
        degradation_pct=[0.10 * reference, 0.65 * reference],
    )

    all_strategies = ("average", "rotating", "instant_health", "beam_perfect")
    selected = all_strategies if args.strategies is None else tuple(args.strategies)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.out_dir / "allocation_trajectory.csv"
    summary_path = args.out_dir / "real_demand_benchmark.csv"
    if set(selected) != set(all_strategies) and trajectory_path.exists():
        existing_trajectory = pd.read_csv(trajectory_path)
        existing_trajectory = existing_trajectory[
            ~existing_trajectory.strategy.isin(selected)
        ]
        trajectories = [existing_trajectory]
    else:
        trajectories = []
    if set(selected) != set(all_strategies) and summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_summary = existing_summary[~existing_summary.strategy.isin(selected)]
        summaries = existing_summary.to_dict("records")
    else:
        summaries = []

    for name in selected:
        frame, final_state, runtime_s = run_strategy(
            name,
            model,
            initial_state,
            demand,
            args.horizon,
            args.rotation_period,
            args.beam_terminal_soc_weight,
        )
        frame["source_index"] = source_index
        frame["raw_demand_power_kw"] = raw_demand
        trajectories.append(frame)
        summaries.append(summarize(name, frame, final_state, runtime_s, model))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(trajectories, ignore_index=True).to_csv(
            args.out_dir / "allocation_trajectory.csv", index=False
        )
        pd.DataFrame(summaries).to_csv(
            args.out_dir / "real_demand_benchmark.csv", index=False
        )
        print(f"completed {name}: {runtime_s:.2f} s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.concat(trajectories, ignore_index=True)
    summary = pd.DataFrame(summaries)
    ordering = {name: index for index, name in enumerate(all_strategies)}
    summary = summary.sort_values(
        "strategy", key=lambda values: values.map(ordering)
    ).reset_index(drop=True)
    trajectory.to_csv(args.out_dir / "allocation_trajectory.csv", index=False)
    summary.to_csv(args.out_dir / "real_demand_benchmark.csv", index=False)
    fair = summary.soc_error.abs().le(0.001)
    comparison = ""
    by_name = summary.set_index("strategy")
    if (
        "average" in by_name.index
        and "beam_perfect" in by_name.index
        and abs(by_name.loc["average", "soc_error"]) <= 0.001
        and abs(by_name.loc["beam_perfect", "soc_error"]) <= 0.001
    ):
        average = by_name.loc["average"]
        beam = by_name.loc["beam_perfect"]

        def reduction(column):
            return 100 * (average[column] - beam[column]) / average[column]

        comparison = f"""
## 严格SOC公平结果

Average与Beam满足末端SOC门槛。相对Average，Beam的SOC等值氢耗降低{reduction('hydrogen_soc_corrected_g'):.2f}%，累计Gamma损伤增量降低{reduction('degradation_increment_pct'):.2f}%，性能损失降低{reduction('performance_loss_sum'):.2f}%，电池吞吐降低{reduction('battery_throughput_kwh'):.2f}%。
"""
    report = f"""# 多堆真实需求段公平基准

## 设置

- 冻结连续测试段长度：{args.length} s
- Beam perfect-preview时域：{args.horizon} s
- Beam终端SOC权重：{args.beam_terminal_soc_weight:g}
- Rotating轮换周期：{args.rotation_period} s
- 初始SOC：{model.config.soc_reference:.3f}
- 初始损伤：10% / 65% LZW late参考状态
- 需求裁剪点：{int(np.count_nonzero(np.abs(raw_demand - demand) > 1e-12))}

## 公平性门槛

- 功率平衡、电池功率和SOC约束不可放宽且要求零违规；
- 驻留规则优先满足；急制动无可行动作时允许安全覆盖并单独计数；
- 主结论要求 `|SOC_end-SOC_ref|<=0.001`；
- 未达到门槛时仅报告原始指标和SOC等值氢耗筛查值，不做优劣结论。

当前满足末端SOC门槛的策略数：{int(fair.sum())}/{len(fair)}。

## 结果

{frame_to_markdown(summary)}

{comparison}
"""
    (args.out_dir / "real_demand_benchmark_report.md").write_text(
        report, encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
