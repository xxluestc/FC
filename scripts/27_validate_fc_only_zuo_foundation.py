"""Validate Zuo/real-load candidates on the deterministic FC-only foundation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    TestScenario,
    ZUO_FAST_TRANSITION,
    ZUO_LOAD_LEVEL_FRACTIONS,
    ZUO_SLOW_TRANSITION,
    generate_zuo_markov_system_load,
    run_policy,
)
from fc_power.power_allocation.multistack_allocator import choose_beam, choose_instant
from fc_power.power_allocation.multistack_baselines import choose_average, choose_rotating
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/load_zuo_calibration"
OUTPUT = ROOT / "data/results/fc_only_zuo_foundation"
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)


def load_empirical_matrix(stride_s: int) -> np.ndarray:
    table = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    selected = table[table.stride_s == stride_s]
    matrix = selected.pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise ValueError(f"{stride_s}-second empirical transition matrix is incomplete")
    return matrix


def load_initial_probabilities(stride_s: int) -> np.ndarray:
    table = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    selected = table[table.stride_s == stride_s].sort_values("state")
    probabilities = selected.occupancy_fraction.to_numpy(dtype=float)
    if probabilities.shape != (4,) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"{stride_s}-second state occupancy is incomplete")
    return probabilities


def main() -> None:
    initial_probabilities = load_initial_probabilities(1)
    scenarios = {
        "empirical_1s": {
            "matrix": load_empirical_matrix(1),
            "decision_interval_s": 1,
        },
        "zuo_slow_30s": {
            "matrix": np.asarray(ZUO_SLOW_TRANSITION),
            "decision_interval_s": 30,
        },
        "zuo_fast_30s": {
            "matrix": np.asarray(ZUO_FAST_TRANSITION),
            "decision_interval_s": 30,
        },
    }
    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY_FACTORS,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    system_power_reference_kw = (
        model.fc_power_reference_kw() * (1 - CAPACITY_RESERVE_FRACTION)
    )
    damage_reference = model.performance_proxies[0].mapping.damage_reference_pct
    state = model.initial_state(
        degradation_pct=np.asarray(INITIAL_DAMAGE_FRACTION) * damage_reference
    )

    coverage_rows = []
    selectors = {
        "average": lambda demand: choose_average(model, state, demand),
        "rotating": lambda demand: choose_rotating(
            model, state, demand, lead_stack=0
        ),
        "instant_health": lambda demand: choose_instant(model, state, demand),
        "beam_health": lambda demand: choose_beam(
            model, state, [demand], beam_width=4
        ),
    }
    for load_state, fraction in enumerate(ZUO_LOAD_LEVEL_FRACTIONS):
        demand = float(fraction * system_power_reference_kw)
        for strategy, select in selectors.items():
            result = select(demand)
            coverage_rows.append(
                {
                    "load_state": load_state,
                    "demand_power_kw": demand,
                    "strategy": strategy,
                    "current_a": ";".join(
                        f"{value:g}" for value in result.action.current_a
                    ),
                    "is_on": ";".join(
                        str(int(value)) for value in result.action.is_on
                    ),
                    "online_stacks": int(sum(result.action.is_on)),
                    "stack_power_kw": result.step.constraints.stack_power_kw,
                    "tracking_error_kw": (
                        result.step.cost.raw_power_tracking_error_kw
                    ),
                    "feasible": result.step.constraints.feasible,
                }
            )

    run_rows = []
    for index, (name, scenario_config) in enumerate(scenarios.items()):
        matrix = scenario_config["matrix"]
        decision_interval_s = scenario_config["decision_interval_s"]
        profile = generate_zuo_markov_system_load(
            2026 + index,
            length_s=120,
            decision_interval_s=decision_interval_s,
            system_power_reference_kw=system_power_reference_kw,
            transition_matrix=matrix,
            initial_probabilities=initial_probabilities,
            source=name,
        )
        scenario = TestScenario(
            name,
            profile,
            INITIAL_DAMAGE_FRACTION,
            stochastic_health=False,
        )
        run = run_policy(model, scenario, "instant_health")
        run_rows.append(
            {
                "load_source": name,
                "load_seed": 2026 + index,
                "decision_interval_s": decision_interval_s,
                "n_steps": run.metrics["n_steps"],
                "constraint_violation_steps": run.metrics[
                    "constraint_violation_steps"
                ],
                "fc_tracking_mae_kw": run.metrics["fc_tracking_mae_kw"],
                "fc_tracking_max_abs_kw": run.metrics[
                    "fc_tracking_max_abs_kw"
                ],
                "fc_tracking_within_tolerance_share": run.metrics[
                    "fc_tracking_within_tolerance_share"
                ],
                "online_stack_count_max": run.metrics["online_stack_count_max"],
                "online_stack_count_mean": run.metrics["online_stack_count_mean"],
                "hydrogen_g": run.metrics["hydrogen_g"],
                "expected_damage_increment_pct": run.metrics[
                    "expected_damage_increment_pct"
                ],
                "final_damage_range_pct": run.metrics["final_damage_range_pct"],
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    runs = pd.DataFrame(run_rows)
    coverage.to_csv(OUTPUT / "state_action_coverage.csv", index=False)
    runs.to_csv(OUTPUT / "deterministic_profile_runs.csv", index=False)
    metadata = {
        "literature_source": (
            "Zuo et al., Reliability Engineering & System Safety 241 (2024) "
            "109660, Appendix A, Eqs. A.5-A.6"
        ),
        "empirical_source": "data/results/load_zuo_calibration",
        "calibrated_baseline": {
            "matrix": "empirical_1s",
            "decision_interval_s": 1,
        },
        "literature_stress_scenarios": {
            "zuo_slow_30s": "30 s is an engineering time-base candidate",
            "zuo_fast_30s": "30 s is an engineering time-base candidate",
        },
        "matrix_blending": "not used because transition time bases differ",
        "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
        "positive_demand_online_stack_rule": "exactly two stacks online",
        "healthy_two_stack_capacity_kw": model.fc_power_reference_kw(),
        "mapped_system_power_reference_kw": system_power_reference_kw,
        "initial_damage_fraction": list(INITIAL_DAMAGE_FRACTION),
        "heterogeneity_factors": list(HETEROGENEITY_FACTORS),
        "stochastic_health": False,
        "scenarios": {
            name: {
                "decision_interval_s": values["decision_interval_s"],
                "matrix": values["matrix"].tolist(),
            }
            for name, values in scenarios.items()
        },
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# FC-only Zuo负载基础验证

- 功率接口：FC-only；三堆N+1，正需求时恰好两堆在线、一堆休息；确定性健康更新。
- 两台健康堆容量：{model.fc_power_reference_kw():.3f} kW；预留{CAPACITY_RESERVE_FRACTION:.0%}后负载参考上限：{system_power_reference_kw:.3f} kW。
- 主标定场景：实车1秒经验矩阵；压力场景：Zuo慢变/快变矩阵，暂按30秒一个转移步解释。
- Zuo场景的30秒时间基准、5%容量余量和{TRACKING_TOLERANCE_KW:.1f} kW跟踪容差均为显式工程候选，不是论文直接给定值。

四个状态、四种基础策略共{len(coverage)}个静态动作检查均可行；最大绝对跟踪误差为{coverage.tracking_error_kw.abs().max():.3f} kW，在线堆数范围为{int(coverage.online_stacks.min())}-{int(coverage.online_stacks.max())}。
三个独立场景各运行120步，硬约束违规总数为{int(runs.constraint_violation_steps.sum())}，最大绝对跟踪误差为{runs.fc_tracking_max_abs_kw.max():.3f} kW，平均/最大在线堆数均为{runs.online_stack_count_mean.mean():.1f}/{int(runs.online_stack_count_max.max())}。

该结果只证明基础接口和三个负载场景在短确定性轨迹上可执行，不证明Zuo场景30秒时间基准或5%容量余量最优，也不构成策略延寿结论。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
