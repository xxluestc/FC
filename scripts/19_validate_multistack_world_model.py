"""Run a compact deterministic multi-stack closed-loop validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.power_allocation.multistack_allocator import choose_beam
from fc_power.world_model import MultiStackAction, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = ROOT / "data/results/multistack"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_lzw_multistack_world_model(
        ROOT, n_stacks=2, heterogeneity_factors=[1.0, 1.10]
    )
    damage_reference = model.performance_proxies[0].mapping.damage_reference_pct
    state = model.initial_state(
        soc=0.70,
        degradation_pct=[0.10 * damage_reference, 0.65 * damage_reference],
    )
    demand = np.tile(
        np.array([45.0, 55.0, 70.0, 85.0, 60.0, 35.0, 10.0, -10.0]),
        4,
    )
    rows = []
    violations = []
    for index, current_demand in enumerate(demand):
        # The preview covers at least one 15 s dwell interval whenever enough
        # future samples remain; shorter horizons can make a locally attractive
        # shutdown lock the healthier stack out of the next load peak.
        preview = demand[index : min(index + 16, len(demand))]
        planned = choose_beam(model, state, preview, beam_width=8)
        step = planned.step
        if not step.constraints.feasible:
            violations.extend(step.constraints.violations)
        row = {
            "step": index,
            "demand_power_kw": current_demand,
            "stack_power_kw": step.constraints.stack_power_kw,
            "battery_power_kw": step.constraints.battery_power_kw,
            "power_balance_error_kw": step.constraints.power_balance_error_kw,
            "soc_before": state.soc,
            "soc_after": step.next_state.soc,
            "step_cost": step.cost.total,
            "hydrogen_g": step.cost.raw_hydrogen_g,
            "feasible": step.constraints.feasible,
        }
        for stack in step.stacks:
            prefix = f"stack_{stack.stack_index}"
            row[f"{prefix}_current_a"] = stack.current_a
            row[f"{prefix}_on"] = stack.is_on
            row[f"{prefix}_power_kw"] = stack.power_kw
            row[f"{prefix}_damage_increment_pct"] = (
                stack.degradation_increment_pct
            )
            row[f"{prefix}_damage_after_pct"] = stack.degradation_after_pct
            row[f"{prefix}_performance_loss"] = (
                stack.normalized_performance_loss
            )
        rows.append(row)
        state = step.next_state

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "mechanistic_closed_loop.csv", index=False)
    max_balance_error = float(frame.power_balance_error_kw.abs().max())
    total_hydrogen = float(frame.hydrogen_g.sum())
    damage_end = [stack.health.degradation for stack in state.stacks]
    current_share = [
        float(frame[f"stack_{index}_current_a"].sum())
        for index in range(model.n_stacks)
    ]

    triple = load_lzw_multistack_world_model(ROOT, n_stacks=3)
    triple_state = triple.initial_state(
        degradation_pct=[0.0, 0.5 * damage_reference, damage_reference]
    )
    triple_action = tuple(195.0 for _ in range(3))
    triple_step = triple.step(
        triple_state,
        MultiStackAction.from_currents(triple_action),
        demand_power_kw=70.0,
    )
    triple_powers = [item.power_kw for item in triple_step.stacks]

    report = f"""# 多堆机理世界模型闭环验证

## 验证结论

- 双堆滚动步数：{len(frame)}
- 硬约束违规数：{len(violations)}
- 最大功率平衡误差：{max_balance_error:.3e} kW
- SOC：{frame.soc_before.iloc[0]:.6f} → {state.soc:.6f}
- 累计理论氢耗：{total_hydrogen:.6f} g
- 两堆末端损伤指数：{damage_end[0]:.6f}% / {damage_end[1]:.6f}%
- 两堆累计电流分配：{current_share[0]:.1f} A·step / {current_share[1]:.1f} A·step

## 三堆异质健康检查

三堆在相同195 A动作下，初始损伤分别为0%、50%和100% LZW late参考状态，输出功率依次为：{triple_powers[0]:.6f} / {triple_powers[1]:.6f} / {triple_powers[2]:.6f} kW。健康更差的堆在相同电流下可输出功率更低。

## 结论边界

该实验验证统一 `step(state, action, demand)`、滚动健康响应、SOC、电池补偿、氢耗、离散动作、驻留与功率守恒。需求序列是接口回归用的合成短序列，不作为节能或寿命提升结论；下一步需在真实需求测试段上比较Average、Rotating、Instant和Beam策略。
"""
    (output_dir / "mechanistic_closed_loop_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print("Triple-stack constraints:", asdict(triple_step.constraints))


if __name__ == "__main__":
    main()
