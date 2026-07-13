"""Run paired FC-only Gamma-health experiments on frozen load scenarios."""

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
OUTPUT = ROOT / "data/results/fc_only_gamma_paired"
STRATEGIES = ("average", "rotating", "instant_health")
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
PAIR_METRICS = (
    "main_sampled_damage_increment_pct",
    "main_expected_damage_increment_pct",
    "damage_increment_range_pct",
    "hydrogen_g_per_fc_kwh",
    "fc_tracking_mae_kw",
    "total_switch_count",
    "online_step_range",
)


def load_empirical_matrix() -> np.ndarray:
    table = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    selected = table[table.stride_s == 1]
    matrix = selected.pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise ValueError("one-second empirical transition matrix is incomplete")
    return matrix


def load_initial_probabilities() -> np.ndarray:
    table = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    selected = table[table.stride_s == 1].sort_values("state")
    probabilities = selected.occupancy_fraction.to_numpy(dtype=float)
    if probabilities.shape != (4,) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("one-second state occupancy is incomplete")
    return probabilities


def run_case(task):
    model, scenario, strategy, rotation_period = task
    run = run_policy(
        model,
        scenario,
        strategy,
        rotation_period=rotation_period,
    )
    return scenario.name, strategy, run.metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument(
        "--pair-seeds", nargs="+", type=int, default=list(range(2026, 2036))
    )
    parser.add_argument("--gamma-terminal-cv", type=float, default=0.10)
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.length, args.rotation_period, args.jobs) <= 0:
        raise ValueError("length, rotation period and jobs must be positive")
    if not np.isfinite(args.gamma_terminal_cv) or args.gamma_terminal_cv <= 0:
        raise ValueError("Gamma terminal CV must be finite and positive")
    if not args.pair_seeds or len(set(args.pair_seeds)) != len(args.pair_seeds):
        raise ValueError("pair seeds must be non-empty and unique")

    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY_FACTORS,
        gamma_terminal_cv=args.gamma_terminal_cv,
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
        "empirical_1s": (load_empirical_matrix(), 1),
        "zuo_slow_30s": (np.asarray(ZUO_SLOW_TRANSITION), 30),
        "zuo_fast_30s": (np.asarray(ZUO_FAST_TRANSITION), 30),
    }
    gamma_scale_pct = model.health_models[0].params.gamma_scale

    tasks = []
    for pair_seed in args.pair_seeds:
        for source, (matrix, decision_interval_s) in scenarios.items():
            demand = generate_zuo_markov_system_load(
                pair_seed,
                length_s=args.length,
                decision_interval_s=decision_interval_s,
                system_power_reference_kw=system_power_reference_kw,
                transition_matrix=matrix,
                initial_probabilities=initial_probabilities,
                source=source,
            )
            scenario = TestScenario(
                name=f"{source}_pair_{pair_seed}",
                demand=demand,
                initial_damage_fraction=INITIAL_DAMAGE_FRACTION,
                health_seed=30_000 + pair_seed,
                stochastic_health=True,
            )
            for strategy in STRATEGIES:
                tasks.append((model, scenario, strategy, args.rotation_period))

    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for scenario_name, strategy, metrics in executor.map(
            run_case, tasks, chunksize=1
        ):
            metrics["pair_seed"] = metrics["load_seed"]
            metrics["gamma_residual_pct"] = (
                metrics["main_sampled_damage_increment_pct"]
                - metrics["main_expected_damage_increment_pct"]
            )
            metrics["gamma_shape_sum"] = (
                metrics["main_expected_continuous_damage_pct"] / gamma_scale_pct
            )
            metrics["sampled_continuous_near_zero"] = bool(
                metrics["main_sampled_continuous_damage_pct"] < 1e-15
            )
            rows.append(metrics)
            print(f"completed {scenario_name} strategy={strategy}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    expected_count = len(args.pair_seeds) * len(scenarios)
    pairing = per_run.groupby(["load_source", "load_seed", "health_seed"]).agg(
        strategies=("strategy", "nunique"),
        runs=("strategy", "size"),
    )
    if (
        len(pairing) != expected_count
        or not pairing.strategies.eq(len(STRATEGIES)).all()
        or not pairing.runs.eq(len(STRATEGIES)).all()
    ):
        raise AssertionError("load/health pairing is incomplete")

    paired = paired_strategy_comparison(
        per_run,
        reference_strategy="average",
        metrics=PAIR_METRICS,
    )
    aggregate_rows = []
    numeric = (
        "main_sampled_damage_increment_pct",
        "main_expected_damage_increment_pct",
        "gamma_residual_pct",
        "damage_increment_range_pct",
        "hydrogen_g_per_fc_kwh",
        "fc_tracking_mae_kw",
        "total_switch_count",
        "online_step_range",
        "planning_runtime_s",
    )
    for (source, strategy), group in per_run.groupby(["load_source", "strategy"]):
        row = {
            "load_source": source,
            "strategy": strategy,
            "n_pairs": len(group),
            "zero_violation_share": float(group.constraint_violation_steps.eq(0).mean()),
            "tracking_within_tolerance_share": float(
                group.fc_tracking_within_tolerance_share.mean()
            ),
        }
        for metric in numeric:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        residual = group.gamma_residual_pct.to_numpy(dtype=float)
        row["gamma_residual_q05_pct"] = float(np.quantile(residual, 0.05))
        row["gamma_residual_median_pct"] = float(np.median(residual))
        row["gamma_residual_q95_pct"] = float(np.quantile(residual, 0.95))
        row["gamma_residual_skew"] = float(pd.Series(residual).skew())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    pairing.reset_index().to_csv(args.out_dir / "pairing_audit.csv", index=False)

    metadata = {
        "scope": "FC-only stochastic Gamma health with online action feedback",
        "load_scenarios": list(scenarios),
        "pair_seeds": args.pair_seeds,
        "health_seed_rule": "30000 + pair_seed, shared across strategies",
        "pairing_keys": ["load_source", "load_seed", "health_seed"],
        "strategies": list(STRATEGIES),
        "excluded_strategy": (
            "beam_health deferred to sensitivity because deterministic benefit was "
            "not stable and planning cost was about twice instant_health"
        ),
        "length_s": args.length,
        "gamma_terminal_cv": args.gamma_terminal_cv,
        "gamma_scale_pct": gamma_scale_pct,
        "stochastic_health": True,
        "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
        "initial_damage_fraction": list(INITIAL_DAMAGE_FRACTION),
        "initial_health_interpretation": (
            "one heterogeneous three-stack initial state per run; stack-level values "
            "are aggregated into one run record and also retained in stack-specific fields"
        ),
        "parallel_jobs": args.jobs,
        "runtime_s": time.perf_counter() - started,
        "interpretation": (
            "Each load seed is paired with one health seed. This estimates joint "
            "load-and-health variability and does not separate their variance components. "
            "The health seed is a deterministic one-to-one mapping of the load seed, not "
            "an independent two-factor factorial design."
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    instant = paired[paired.strategy == "instant_health"]
    near_zero_share = float(per_run.sampled_continuous_near_zero.mean())
    shape_min = float(per_run.gamma_shape_sum.min())
    shape_max = float(per_run.gamma_shape_sum.max())
    lines = [
        "| 场景 | 采样退化差值(%) | 95%区间 | 改善率 | 跟踪MAE相对变化 | H2强度相对变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in scenarios:
        selected = instant[instant.load_source == source].set_index("metric")
        damage = selected.loc["main_sampled_damage_increment_pct"]
        tracking = selected.loc["fc_tracking_mae_kw"]
        hydrogen = selected.loc["hydrogen_g_per_fc_kwh"]
        lines.append(
            f"| {source} | {damage.mean_delta:+.6f} | "
            f"±{damage.ci95:.6f} | "
            f"{damage.lower_is_better_win_share:.0%} | "
            f"{tracking.mean_relative_pct:+.1f}% | "
            f"{hydrogen.mean_relative_pct:+.1f}% |"
        )
    report = f"""# FC-only Gamma同种子配对实验

- 三类冻结负载，每类{len(args.pair_seeds)}个联合配对种子；每个配对内负载种子和健康种子对所有策略相同。
- Gamma终点CV假设为{args.gamma_terminal_cv:.0%}；健康状态按实际执行动作逐步随机更新，并反馈到下一步控制。
- 比较Average、Rotating和Instant-health；Beam留到时域/CV敏感性阶段。
- 所有结果必须同时解释采样退化、条件期望退化和`sampled-expected` Gamma残差。
- 每次运行从同一个异质三堆初始状态开始，三堆结果聚合为一条运行记录，同时保留逐堆字段。

## Instant-health相对Average

{chr(10).join(lines)}

当前{len(args.pair_seeds)}个联合配对采用负载种子到健康种子的一一映射，只估计联合变化，不分解两个方差来源。残差偏度和分位数保存在聚合表中；本实验不构成寿命外推。

## 时间尺度诊断

120秒运行的连续Gamma shape总量仅为{shape_min:.3e}-{shape_max:.3e}，采样连续增量低于`1e-15%`的运行占{near_zero_share:.1%}。因此本表的短时采样退化差值不能用于策略延寿排序；控制比较继续使用条件期望，Gamma不确定性转入聚合暴露分析。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
