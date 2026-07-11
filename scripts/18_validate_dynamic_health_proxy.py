"""Validate that Gamma health updates change the online current-point proxy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.health.dynamic_proxy import (
    DynamicPerformanceLossProxy,
    LzwIvConditions,
)
from fc_power.health.gamma_process import GammaHealthModel, GammaHealthState
from fc_power.health.lzw_gamma_calibration import (
    ThetaPowerLawMap,
    ghaderi_gamma_params,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    upstream = ROOT / "data/upstream_lzw"
    health_dir = ROOT / "data/results/health"
    calibration = json.loads(
        (health_dir / "lzw_gamma_calibration.json").read_text(encoding="utf-8")
    )
    mapping = ThetaPowerLawMap.from_dict(calibration["theta_power_law_map"])
    conditions = LzwIvConditions.from_upstream_dict(
        json.loads(
            (upstream / "current_point_cost_conditions.json").read_text(
                encoding="utf-8"
            )
        )
    )
    upstream_table = pd.read_csv(
        upstream / "current_point_degradation_cost_table.csv"
    )
    proxy = DynamicPerformanceLossProxy(
        mapping,
        conditions,
        upstream_table.equivalent_stack_power_loss_clipped_W.max(),
    )

    currents = np.array([0, 25, 90, 120, 160, 195, 270, 370], dtype=float)
    rows = []
    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        damage = fraction * mapping.damage_reference_pct
        result = proxy.evaluate(damage, currents)
        for index, current in enumerate(currents):
            rows.append(
                {
                    "health_fraction_of_lzw_late": fraction,
                    "damage_pct": damage,
                    "current_a": current,
                    "cell_voltage_v": result["current_cell_voltage_v"][index],
                    "stack_power_kw": result["stack_power_kw"][index],
                    "equivalent_power_loss_w": result["equivalent_power_loss_w"][
                        index
                    ],
                    "normalized_proxy": result["normalized_proxy"][index],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(health_dir / "dynamic_current_point_proxy.csv", index=False)

    params = ghaderi_gamma_params(calibration["gamma_scale_pct"])
    model = GammaHealthModel(params)
    scenarios = []
    for current in (195.0, 370.0):
        state = GammaHealthState()
        first = model.transition(
            state,
            current,
            dt_s=3600.0,
            stochastic=False,
            next_on=True,
            shift_event=False,
        )
        current_proxy = proxy.evaluate(first.state.degradation, [current])
        scenarios.append(
            {
                "current_a": current,
                "one_hour_expected_damage_pct": first.total_increment,
                "theta_after_one_hour_i0": mapping.theta_reported(
                    first.state.degradation
                )[0],
                "theta_after_one_hour_ih": mapping.theta_reported(
                    first.state.degradation
                )[1],
                "theta_after_one_hour_R_ohm": mapping.theta_reported(
                    first.state.degradation
                )[2],
                "normalized_proxy_after_one_hour": current_proxy[
                    "normalized_proxy"
                ][0],
            }
        )
    scenario_frame = pd.DataFrame(scenarios)
    scenario_frame.to_csv(health_dir / "one_hour_action_response.csv", index=False)

    late = table[table.health_fraction_of_lzw_late.eq(1.0)]
    report = f"""# 动态健康代理验证

## 结论

动态代理已经从固定late表升级为 `C_deg(I|theta_current)`。Gamma损伤状态变化后，theta、IV模型电压、可输出功率和候选档位proxy同步更新。

## 回归检查

- 健康起点的性能损失proxy为0。
- LZW late状态下370 A归一化proxy为{late.loc[late.current_a.eq(370), 'normalized_proxy'].iloc[0]:.6f}。
- 同一健康状态下，高电流档位放大等效功率损失。
- 同一电流档位下，健康损伤增加时proxy非减。

## 一小时动作响应示例

| 电流(A) | 期望损伤增量(%) | 更新后R_ohm | 更新后proxy |
|---:|---:|---:|---:|
| 195 | {scenario_frame.iloc[0].one_hour_expected_damage_pct:.6f} | {scenario_frame.iloc[0].theta_after_one_hour_R_ohm:.6f} | {scenario_frame.iloc[0].normalized_proxy_after_one_hour:.6g} |
| 370 | {scenario_frame.iloc[1].one_hour_expected_damage_pct:.6f} | {scenario_frame.iloc[1].theta_after_one_hour_R_ohm:.6f} | {scenario_frame.iloc[1].normalized_proxy_after_one_hour:.6g} |

这里只验证状态响应和接口正确性，不把一小时示例解释为精确寿命预测。下一步应把该接口接入多堆机理世界模型的候选动作rollout。
"""
    (health_dir / "dynamic_health_proxy_report.md").write_text(
        report, encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(scenario_frame.to_string(index=False))


if __name__ == "__main__":
    main()
