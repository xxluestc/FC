"""Compare deterministic FC-only policies on time-consistent load scenarios."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    TestScenario,
    ZUO_FAST_TRANSITION,
    ZUO_SLOW_TRANSITION,
    generate_zuo_markov_system_load,
    paired_strategy_comparison,
    run_policy,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/load_zuo_calibration"
DEFAULT_OUTPUT = ROOT / "data/results/fc_only_deterministic_comparison"
STRATEGIES = ("average", "rotating", "instant_health", "beam_health")
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
PAIR_METRICS = (
    "hydrogen_g",
    "hydrogen_g_per_fc_kwh",
    "main_expected_damage_increment_pct",
    "main_performance_loss_sum",
    "damage_increment_range_pct",
    "fc_tracking_mae_kw",
    "total_switch_count",
    "total_load_shift_count",
    "online_step_range",
    "planning_runtime_s",
)


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


def load_initial_probabilities() -> np.ndarray:
    table = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    selected = table[table.stride_s == 1].sort_values("state")
    probabilities = selected.occupancy_fraction.to_numpy(dtype=float)
    if probabilities.shape != (4,) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("one-second state occupancy is incomplete")
    return probabilities


def aggregate_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "hydrogen_g",
        "fc_energy_kwh",
        "hydrogen_g_per_fc_kwh",
        "main_expected_damage_increment_pct",
        "main_performance_loss_sum",
        "damage_increment_range_pct",
        "fc_tracking_mae_kw",
        "fc_tracking_max_abs_kw",
        "total_switch_count",
        "total_load_shift_count",
        "online_step_range",
        "planning_runtime_s",
        "planning_expanded_nodes",
    )
    rows = []
    for (source, strategy), group in per_run.groupby(["load_source", "strategy"]):
        row = {
            "load_source": source,
            "strategy": strategy,
            "n_runs": len(group),
            "zero_violation_share": float(group.constraint_violation_steps.eq(0).mean()),
            "tracking_within_tolerance_share": float(
                group.fc_tracking_within_tolerance_share.mean()
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_case(task):
    model, scenario, strategy, beam_horizon, beam_width, rotation_period = task
    run = run_policy(
        model,
        scenario,
        strategy,
        beam_horizon=beam_horizon,
        beam_width=beam_width,
        rotation_period=rotation_period,
    )
    return scenario.name, strategy, run.metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--beam-horizon", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(
        args.length,
        args.beam_horizon,
        args.beam_width,
        args.rotation_period,
        args.jobs,
    ) <= 0:
        raise ValueError("length and planner settings must be positive")
    if not args.seeds:
        raise ValueError("at least one load seed is required")

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
        1 - CAPACITY_RESERVE_FRACTION
    ) * model.fc_power_reference_kw()
    initial_probabilities = load_initial_probabilities()
    scenarios = {
        "empirical_1s": (load_empirical_matrix(1), 1),
        "zuo_slow_30s": (np.asarray(ZUO_SLOW_TRANSITION), 30),
        "zuo_fast_30s": (np.asarray(ZUO_FAST_TRANSITION), 30),
    }

    started = time.perf_counter()
    tasks = []
    for seed in args.seeds:
        for source, (matrix, decision_interval_s) in scenarios.items():
            demand = generate_zuo_markov_system_load(
                seed,
                length_s=args.length,
                decision_interval_s=decision_interval_s,
                system_power_reference_kw=system_power_reference_kw,
                transition_matrix=matrix,
                initial_probabilities=initial_probabilities,
                source=source,
            )
            scenario = TestScenario(
                name=f"{source}_seed_{seed}",
                demand=demand,
                initial_damage_fraction=INITIAL_DAMAGE_FRACTION,
                health_seed=10_000 + seed,
                stochastic_health=False,
            )
            for strategy in STRATEGIES:
                tasks.append(
                    (
                        model,
                        scenario,
                        strategy,
                        args.beam_horizon,
                        args.beam_width,
                        args.rotation_period,
                    )
                )

    if args.jobs == 1:
        completed = map(run_case, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        completed = executor.map(run_case, tasks, chunksize=1)
    rows = []
    try:
        for scenario_name, strategy, metrics in completed:
            rows.append(metrics)
            print(f"completed {scenario_name} strategy={strategy}", flush=True)
    finally:
        if args.jobs > 1:
            executor.shutdown()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    aggregate = aggregate_metrics(per_run)
    paired = paired_strategy_comparison(
        per_run,
        reference_strategy="average",
        metrics=PAIR_METRICS,
    )
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_deltas.csv", index=False)

    runtime_s = time.perf_counter() - started
    metadata = {
        "scope": "deterministic FC-only three-stack N+1 inner allocation",
        "strategies": list(STRATEGIES),
        "load_seeds": args.seeds,
        "length_s": args.length,
        "initial_damage_fraction": list(INITIAL_DAMAGE_FRACTION),
        "heterogeneity_factors": list(HETEROGENEITY_FACTORS),
        "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
        "beam_horizon": args.beam_horizon,
        "beam_width": args.beam_width,
        "beam_preview": "current demand held constant; no future demand access",
        "rotation_period": args.rotation_period,
        "parallel_jobs": args.jobs,
        "stochastic_health": False,
        "runtime_s": runtime_s,
        "scenario_intervals_s": {
            name: interval for name, (_, interval) in scenarios.items()
        },
        "zuo_interval_status": "30 s engineering stress assumption, not a paper fact",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "| 场景 | 策略 | n | 零违规 | 容差内 | H2(g) | FC电量(kWh) | H2强度(g/kWh) | 期望退化(%) | 退化增量极差(%) | 跟踪MAE(kW) | 启停 | 变载 | 在线步数极差 | 规划时间(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.load_source} | {row.strategy} | {row.n_runs} | "
            f"{row.zero_violation_share:.0%} | "
            f"{row.tracking_within_tolerance_share:.0%} | "
            f"{row.hydrogen_g_mean:.3f} | "
            f"{row.fc_energy_kwh_mean:.3f} | "
            f"{row.hydrogen_g_per_fc_kwh_mean:.3f} | "
            f"{row.main_expected_damage_increment_pct_mean:.6f} | "
            f"{row.damage_increment_range_pct_mean:.6f} | "
            f"{row.fc_tracking_mae_kw_mean:.3f} | "
            f"{row.total_switch_count_mean:.1f} | "
            f"{row.total_load_shift_count_mean:.1f} | "
            f"{row.online_step_range_mean:.1f} | "
            f"{row.planning_runtime_s_mean:.3f} |"
        )
    report = f"""# FC-only确定性策略基础比较

- 三堆N+1，正需求时恰好两堆在线；确定性动作驱动健康更新。
- 负载为实车1秒主基线及Zuo慢变/快变独立压力场景；Zuo场景30秒转移步是工程假设。
- 所有策略使用相同负载与初始健康。Beam只使用当前需求保持预览，不访问真实未来需求。
- 轨迹长度{args.length}秒，种子{args.seeds}，Beam时域/宽度={args.beam_horizon}/{args.beam_width}。

{chr(10).join(lines)}

总氢耗必须与FC实际输出电量和跟踪误差联合解释，较低氢耗不能自动解释为较高效率。
本报告是G4基础可执行性与方向检查，不是寿命结论；当前种子数和轨迹长度只支持初步配对区间，规划时间受当前机器影响，Zuo时间尺度、动作网格和目标权重仍需后续消融。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
