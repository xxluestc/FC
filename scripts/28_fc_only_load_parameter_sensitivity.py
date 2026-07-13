"""Audit Markov time bases and FC-only capacity reserve candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    ZUO_FAST_TRANSITION,
    ZUO_LOAD_LEVEL_FRACTIONS,
    ZUO_SLOW_TRANSITION,
    blend_transition_matrices,
)
from fc_power.power_allocation.multistack_allocator import choose_instant
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/load_zuo_calibration"
OUTPUT = ROOT / "data/results/fc_only_load_sensitivity"
STRIDES_S = (1, 5, 10, 30, 60)
BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
RESERVE_FRACTIONS = (0.0, 0.02, 0.05, 0.10, 0.15)
TRACKING_TOLERANCE_KW = 5.5


def stationary_distribution(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    design = np.vstack([matrix.T - np.eye(4), np.ones(4)])
    target = np.r_[np.zeros(4), 1.0]
    distribution = np.linalg.lstsq(design, target, rcond=None)[0]
    return distribution / distribution.sum()


def event_rate_per_hour(matrix, interval_s: int) -> float:
    distribution = stationary_distribution(matrix)
    change_probability = 1 - float(np.dot(distribution, np.diag(matrix)))
    return change_probability * 3600 / interval_s


def empirical_matrix(table: pd.DataFrame, stride_s: int) -> np.ndarray:
    selected = table[table.stride_s == stride_s]
    matrix = selected.pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise ValueError(f"empirical matrix is incomplete at stride {stride_s}")
    return matrix


def main() -> None:
    transition_table = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    timescale_rows = []
    empirical = {}
    for stride in STRIDES_S:
        matrix = empirical_matrix(transition_table, stride)
        empirical[stride] = matrix
        stationary = stationary_distribution(matrix)
        timescale_rows.append(
            {
                "matrix": f"empirical_{stride}s",
                "decision_interval_s": stride,
                "event_rate_per_hour": event_rate_per_hour(matrix, stride),
                **{
                    f"stationary_state_{index}": value
                    for index, value in enumerate(stationary)
                },
                **{
                    f"self_transition_state_{index}": matrix[index, index]
                    for index in range(4)
                },
            }
        )
    for name, matrix in (
        ("zuo_fast_30s", np.asarray(ZUO_FAST_TRANSITION)),
        ("zuo_slow_30s", np.asarray(ZUO_SLOW_TRANSITION)),
    ):
        stationary = stationary_distribution(matrix)
        timescale_rows.append(
            {
                "matrix": name,
                "decision_interval_s": 30,
                "event_rate_per_hour": event_rate_per_hour(matrix, 30),
                **{
                    f"stationary_state_{index}": value
                    for index, value in enumerate(stationary)
                },
                **{
                    f"self_transition_state_{index}": matrix[index, index]
                    for index in range(4)
                },
            }
        )

    blend_rows = []
    for literature_name, literature in (
        ("zuo_fast", ZUO_FAST_TRANSITION),
        ("zuo_slow", ZUO_SLOW_TRANSITION),
    ):
        for weight in BLEND_WEIGHTS:
            matrix = blend_transition_matrices(empirical[30], literature, weight)
            blend_rows.append(
                {
                    "literature_matrix": literature_name,
                    "empirical_weight": weight,
                    "assumed_common_interval_s": 30,
                    "event_rate_per_hour": event_rate_per_hour(matrix, 30),
                    "time_base_warning": (
                        "convex blend assumes a common Markov interval; not selected "
                        "as the calibrated baseline"
                    ),
                }
            )

    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=(1.0, 1.05, 1.10),
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    damage_reference = model.performance_proxies[0].mapping.damage_reference_pct
    health_cases = {
        "healthy": (0.0, 0.0, 0.0),
        "heterogeneous": (0.10, 0.40, 0.80),
        "late": (1.0, 1.0, 1.0),
    }
    capacity = model.fc_power_reference_kw()
    coverage_rows = []
    for health_name, fractions in health_cases.items():
        state = model.initial_state(
            degradation_pct=np.asarray(fractions) * damage_reference
        )
        for reserve in RESERVE_FRACTIONS:
            reference = capacity * (1 - reserve)
            for load_state, fraction in enumerate(ZUO_LOAD_LEVEL_FRACTIONS):
                demand = float(reference * fraction)
                try:
                    result = choose_instant(model, state, demand)
                    feasible = result.step.constraints.feasible
                    error = result.step.cost.raw_power_tracking_error_kw
                    online = int(sum(result.action.is_on))
                except RuntimeError:
                    feasible, error, online = False, np.nan, 0
                coverage_rows.append(
                    {
                        "health_case": health_name,
                        "reserve_fraction": reserve,
                        "load_state": load_state,
                        "demand_power_kw": demand,
                        "feasible": feasible,
                        "tracking_error_kw": error,
                        "online_stacks": online,
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    timescales = pd.DataFrame(timescale_rows)
    blends = pd.DataFrame(blend_rows)
    coverage = pd.DataFrame(coverage_rows)
    timescales.to_csv(OUTPUT / "markov_timescale_audit.csv", index=False)
    blends.to_csv(OUTPUT / "matrix_blend_warning.csv", index=False)
    coverage.to_csv(OUTPUT / "capacity_reserve_coverage.csv", index=False)

    empirical_1s_rate = float(
        timescales.loc[timescales.matrix == "empirical_1s", "event_rate_per_hour"].iloc[0]
    )
    slow_rate = float(
        timescales.loc[timescales.matrix == "zuo_slow_30s", "event_rate_per_hour"].iloc[0]
    )
    fast_rate = float(
        timescales.loc[timescales.matrix == "zuo_fast_30s", "event_rate_per_hour"].iloc[0]
    )
    recommendation = {
        "calibrated_baseline": {
            "matrix": "empirical_1s",
            "decision_interval_s": 1,
            "reason": (
                "preserves the native one-second transition time base and observed "
                "state-change rate"
            ),
        },
        "literature_stress_scenarios": [
            {"matrix": "zuo_slow", "candidate_interval_s": 30},
            {"matrix": "zuo_fast", "candidate_interval_s": 30},
        ],
        "matrix_blending": "not selected because transition time bases differ",
        "capacity_reserve_fraction": 0.05,
        "capacity_reserve_status": (
            "conservative baseline candidate; scan retained for sensitivity"
        ),
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
    }
    (OUTPUT / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    grouped = coverage.groupby(["health_case", "reserve_fraction"]).agg(
        feasible_share=("feasible", "mean"),
        max_abs_error_kw=("tracking_error_kw", lambda x: float(np.nanmax(np.abs(x)))),
        min_online_stacks=("online_stacks", "min"),
        max_online_stacks=("online_stacks", "max"),
    )
    all_feasible = grouped[grouped.feasible_share == 1.0]
    report = f"""# FC-only负载参数敏感性

## Markov时间基准

- 实车1秒矩阵的稳态状态变化率约为{empirical_1s_rate:.2f}次/h。
- 同一实车序列下采样到30秒后只剩{event_rate_per_hour(empirical[30], 30):.2f}次/h，说明下采样会漏掉中间变化，不能把30秒经验矩阵当作1秒过程的等价替代。
- 若将Zuo慢变/快变矩阵的一个转移步暂按30秒解释，对应约{slow_rate:.2f}/{fast_rate:.2f}次/h；30秒是工程压力场景假设，不是论文直接给定值。
- 经验矩阵与Zuo矩阵不再直接融合为主标定矩阵，因为凸组合要求相同的Markov时间基准。

## 基线选择

- 主标定负载：实车1秒经验矩阵，保持原生1秒时间基准。
- 文献压力场景：Zuo慢变30秒候选、Zuo快变30秒候选，分别报告，不伪装成实车标定结果。
- 容量余量：继续保留5%作为保守候选；0%-15%扫描结果完整保存。三种健康状态下共有{len(all_feasible)}个“健康状态×余量”组合达到四状态全可行。

该选择优先保证时间单位一致和来源可解释。下一步应使用这三个独立场景做确定性多策略比较，而不是继续调未来预测或电池外层。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
