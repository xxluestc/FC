"""Run synthetic/real-bootstrap loads through the unified multi-stack testbed."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    SyntheticLoadConfig,
    TestScenario,
    append_soc_recovery_tail,
    clip_profile_to_feasible_envelope,
    generate_event_load,
    generate_real_block_bootstrap,
    paired_strategy_comparison,
    run_policy,
)
from fc_power.world_model import load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("average", "rotating", "instant_health", "beam_health")


def load_real_power():
    processed = ROOT / "data/processed/baseline_power_demand.csv"
    key = ROOT / "data/key/baseline_power_demand.csv.gz"
    path = processed if processed.exists() else key
    if not path.exists():
        raise FileNotFoundError("baseline demand is not materialized or tracked")
    return pd.read_csv(path, usecols=["p_dem_measured_kw"])[
        "p_dem_measured_kw"
    ].to_numpy(dtype=float)


def aggregate_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "hydrogen_soc_corrected_g",
        "sampled_damage_increment_pct",
        "main_sampled_damage_increment_pct",
        "expected_damage_increment_pct",
        "main_expected_damage_increment_pct",
        "performance_loss_sum",
        "main_performance_loss_sum",
        "battery_throughput_kwh",
        "soc_error",
        "safety_override_steps",
        "aged_stack_current_share",
        "main_aged_stack_current_share",
        "damage_increment_range_pct",
    ]
    rows = []
    for (source, strategy), group in per_run.groupby(["load_source", "strategy"]):
        row = {
            "load_source": source,
            "strategy": strategy,
            "n_runs": len(group),
            "soc_fair_share": float(group.soc_error.abs().le(0.001).mean()),
            "zero_violation_share": float(
                group.constraint_violation_steps.eq(0).mean()
            ),
        }
        for column in numeric:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_ci95"] = (
                1.96 * row[f"{column}_std"] / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--n-stacks", type=int, choices=[2, 3], default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--beam-horizon", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--stochastic-health", action="store_true")
    parser.add_argument("--gamma-terminal-cv", type=float, default=0.10)
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--soc-recovery-length", type=int, default=120)
    parser.add_argument("--soc-recovery-demand-kw", type=float, default=30.0)
    parser.add_argument("--stack-capacity-reserve-fraction", type=float, default=0.01)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/testbed"
    )
    args = parser.parse_args()
    if args.length <= 0 or args.beam_horizon <= 0 or args.beam_width <= 0:
        raise ValueError("length and planner sizes must be positive")
    if args.soc_recovery_length < 0:
        raise ValueError("SOC recovery length must be non-negative")
    if args.gamma_terminal_cv <= 0:
        raise ValueError("gamma terminal CV must be positive")
    if not 0 <= args.stack_capacity_reserve_fraction < 1:
        raise ValueError("stack capacity reserve fraction must lie in [0, 1)")
    if not args.seeds:
        raise ValueError("at least one seed is required")

    config = SyntheticLoadConfig(length_s=args.length)
    real_power = load_real_power()
    initial_damage_fraction = (
        (0.10, 0.70) if args.n_stacks == 2 else (0.10, 0.40, 0.70)
    )
    heterogeneity_factors = (
        (1.0, 1.10) if args.n_stacks == 2 else (1.0, 1.05, 1.10)
    )
    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=args.n_stacks,
        heterogeneity_factors=heterogeneity_factors,
        gamma_terminal_cv=args.gamma_terminal_cv,
    )
    trajectories, metrics = [], []
    started = time.perf_counter()
    for seed in args.seeds:
        profiles = (
            generate_event_load(seed, config),
            generate_real_block_bootstrap(
                real_power,
                args.length,
                seed,
                block_length_s=30,
                boundary_candidates=32,
            ),
        )
        for profile in profiles:
            source = str(profile.source.iloc[0])
            profile = append_soc_recovery_tail(
                profile,
                duration_s=args.soc_recovery_length,
                demand_power_kw=args.soc_recovery_demand_kw,
            )
            profile = clip_profile_to_feasible_envelope(
                model,
                profile,
                initial_damage_fraction,
                stack_capacity_reserve_fraction=args.stack_capacity_reserve_fraction,
            )
            scenario = TestScenario(
                name=f"{source}_seed_{seed}",
                demand=profile,
                initial_damage_fraction=initial_damage_fraction,
                health_seed=10_000 + seed,
                stochastic_health=args.stochastic_health,
            )
            for strategy in args.strategies:
                run = run_policy(
                    model,
                    scenario,
                    strategy,
                    beam_horizon=args.beam_horizon,
                    beam_width=args.beam_width,
                )
                trajectories.append(run.trajectory)
                metrics.append(run.metrics)
                args.out_dir.mkdir(parents=True, exist_ok=True)
                pd.concat(trajectories, ignore_index=True).to_csv(
                    args.out_dir / "testbed_trajectory.csv", index=False
                )
                pd.DataFrame(metrics).to_csv(
                    args.out_dir / "testbed_per_run_metrics.csv", index=False
                )
                print(
                    f"completed source={source} seed={seed} strategy={strategy}",
                    flush=True,
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.concat(trajectories, ignore_index=True)
    per_run = pd.DataFrame(metrics)
    aggregate = aggregate_metrics(per_run)
    paired = (
        paired_strategy_comparison(per_run)
        if "average" in set(per_run.strategy) and per_run.strategy.nunique() > 1
        else pd.DataFrame()
    )
    trajectory.to_csv(args.out_dir / "testbed_trajectory.csv", index=False)
    per_run.to_csv(args.out_dir / "testbed_per_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "testbed_aggregate_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "testbed_paired_deltas.csv", index=False)
    metadata = {
        "synthetic_load_config": asdict(config),
        "seeds": args.seeds,
        "strategies": args.strategies,
        "n_stacks": args.n_stacks,
        "initial_damage_fraction": initial_damage_fraction,
        "heterogeneity_factors": heterogeneity_factors,
        "stochastic_health": args.stochastic_health,
        "gamma_terminal_cv": args.gamma_terminal_cv,
        "soc_recovery_length": args.soc_recovery_length,
        "soc_recovery_demand_kw": args.soc_recovery_demand_kw,
        "stack_capacity_reserve_fraction": args.stack_capacity_reserve_fraction,
        "beam_horizon": args.beam_horizon,
        "beam_width": args.beam_width,
        "runtime_s": time.perf_counter() - started,
        "scope": (
            "Synthetic settings are stress-test assumptions. Real-block profiles "
            "preserve measured samples but not original trip-level chronology."
        ),
    }
    (args.out_dir / "testbed_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    key_columns = [
        "load_source",
        "strategy",
        "n_runs",
        "soc_fair_share",
        "zero_violation_share",
        "hydrogen_soc_corrected_g_mean",
        "main_expected_damage_increment_pct_mean",
        "main_sampled_damage_increment_pct_mean",
        "main_performance_loss_sum_mean",
        "battery_throughput_kwh_mean",
        "main_aged_stack_current_share_mean",
    ]
    lines = [
        "| 负载 | 策略 | n | SOC公平率 | 零违规率 | 总SOC等值H2(g) | 主段期望退化(%) | 主段采样退化(%) | 主段性能损失 | 总电池吞吐 | 主段老化堆电流份额 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate[key_columns].itertuples(index=False):
        lines.append(
            f"| {row.load_source} | {row.strategy} | {row.n_runs} | "
            f"{row.soc_fair_share:.0%} | {row.zero_violation_share:.0%} | "
            f"{row.hydrogen_soc_corrected_g_mean:.3f} | "
            f"{row.main_expected_damage_increment_pct_mean:.6f} | "
            f"{row.main_sampled_damage_increment_pct_mean:.6f} | "
            f"{row.main_performance_loss_sum_mean:.3f} | "
            f"{row.battery_throughput_kwh_mean:.3f} | "
            f"{row.main_aged_stack_current_share_mean:.1%} |"
        )
    paired_lines = [
        "| 负载 | 策略 | 指标 | n | 相对Average均值 | 95% CI | 配对改善率 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    metric_labels = {
        "hydrogen_soc_corrected_g": "SOC等值H2(g)",
        "main_expected_damage_increment_pct": "主段期望退化(%)",
        "main_performance_loss_sum": "主段性能损失",
        "battery_throughput_kwh": "总电池吞吐(kWh)",
        "main_aged_stack_current_share": "老化堆电流份额",
    }
    for row in paired.itertuples(index=False):
        paired_lines.append(
            f"| {row.load_source} | {row.strategy} | "
            f"{metric_labels.get(row.metric, row.metric)} | {row.n_pairs} | "
            f"{row.mean_delta:.6g} ({row.mean_relative_pct:+.2f}%) | "
            f"±{row.ci95:.6g} | {row.lower_is_better_win_share:.0%} |"
        )
    report = f"""# 多堆统一随机测试框架报告

## 范围

- 堆数：{args.n_stacks}；两类负载：事件型半马尔可夫压力测试、实测连续块重采样；
- 种子：{args.seeds}；主随机负载{args.length} s，统一SOC恢复尾段{args.soc_recovery_length} s @ {args.soc_recovery_demand_kw:g} kW；
- 策略：{', '.join(args.strategies)}；
- 每一步均执行“动作→Gamma/均值损伤→theta→IV/功率→下一状态”，并由代码检查状态连续性和不可逆性；
- stochastic_health={args.stochastic_health}，Gamma终点CV={args.gamma_terminal_cv:.0%}。确定性均值用于策略公平比较；随机Gamma模式用于敏感性分析。
- 统一负载上界预留初始燃料电池容量的{args.stack_capacity_reserve_fraction:.1%}，避免在线老化后“初始极限点”变为无解；所有裁剪均保留审计列。
- SOC恢复尾段是所有策略共同执行的标准化负载，用于把初始/末端电池能量拉回同一口径，不属于原始随机负载。

## 汇总

{chr(10).join(lines)}

## 同种子配对比较（候选策略 - Average）

所有指标均以越低越好解释；区间跨0时不视为稳定收益。

{chr(10).join(paired_lines) if len(paired_lines) > 2 else '本次未包含可与Average配对的候选策略。'}

## 结论边界

合成转移概率和功率范围属于透明压力测试假设，不代表某辆车的实测分布；真实块重采样保留块内动态但打破跨块行程语义。只有在两类负载、多种子、SOC公平且零硬违规时稳定出现的方向，才进入后续算法比较。
"""
    (args.out_dir / "testbed_report.md").write_text(report, encoding="utf-8")
    print(aggregate[key_columns].to_string(index=False))


if __name__ == "__main__":
    main()
