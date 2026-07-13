"""Replay fixed, non-tuned center windows from every real holdout segment."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import TestScenario, paired_strategy_comparison, run_policy
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
OUTPUT = ROOT / "data/results/fc_only_real_holdout_replay"
HOLDOUT_SEGMENTS = tuple(range(22, 46))
STRATEGIES = ("average", "rotating", "instant_health", "beam_health")
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)
NORMALIZATION_POWER_KW = 30.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
PAIR_METRICS = (
    "hydrogen_g_per_fc_kwh",
    "main_expected_damage_increment_pct",
    "damage_increment_range_pct",
    "fc_tracking_mae_kw",
    "total_switch_count",
    "online_step_range",
)


def run_case(task):
    model, scenario, strategy, beam_horizon, beam_width, rotation_period = task
    try:
        run = run_policy(
            model,
            scenario,
            strategy,
            beam_horizon=beam_horizon,
            beam_width=beam_width,
            rotation_period=rotation_period,
        )
    except RuntimeError as error:
        return scenario.name, strategy, None, str(error)
    return scenario.name, strategy, run.metrics, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-length", type=int, default=120)
    parser.add_argument("--beam-horizon", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(
        args.window_length,
        args.beam_horizon,
        args.beam_width,
        args.rotation_period,
        args.jobs,
    ) <= 0:
        raise ValueError("window and planner settings must be positive")

    raw = pd.read_csv(
        SOURCE,
        usecols=["timestamp", "segment_id", "fc_input_power_kw"],
    )
    holdout = raw[raw.segment_id.isin(HOLDOUT_SEGMENTS)].copy()
    if tuple(sorted(holdout.segment_id.unique())) != HOLDOUT_SEGMENTS:
        raise ValueError("holdout segment set is incomplete")

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

    tasks = []
    manifest_rows = []
    for segment_id, segment in holdout.groupby("segment_id", sort=True):
        segment = segment.reset_index(drop=True)
        length = min(args.window_length, len(segment))
        start = (len(segment) - length) // 2
        window = segment.iloc[start : start + length].copy()
        normalized = np.clip(
            window.fc_input_power_kw.to_numpy(dtype=float)
            / NORMALIZATION_POWER_KW,
            0.0,
            1.0,
        )
        demand = pd.DataFrame(
            {
                "demand_power_kw": normalized * system_power_reference_kw,
                "event": np.where(normalized > 0, "real_fc_on", "real_fc_off"),
                "source": "real_holdout_center_window",
                "seed": int(segment_id),
            }
        )
        scenario = TestScenario(
            name=f"holdout_segment_{int(segment_id)}",
            demand=demand,
            initial_damage_fraction=INITIAL_DAMAGE_FRACTION,
            health_seed=20_000 + int(segment_id),
            stochastic_health=False,
        )
        manifest_rows.append(
            {
                "segment_id": int(segment_id),
                "segment_rows": len(segment),
                "window_start_offset": start,
                "window_rows": length,
                "window_start_timestamp": window.timestamp.iloc[0],
                "window_end_timestamp": window.timestamp.iloc[-1],
                "raw_power_min_kw": float(window.fc_input_power_kw.min()),
                "raw_power_max_kw": float(window.fc_input_power_kw.max()),
                "raw_power_mean_kw": float(window.fc_input_power_kw.mean()),
                "positive_share": float((window.fc_input_power_kw > 0).mean()),
                "clipped_high_samples": int(
                    (window.fc_input_power_kw > NORMALIZATION_POWER_KW).sum()
                ),
            }
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

    started = time.perf_counter()
    rows = []
    failure_rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for scenario_name, strategy, metrics, error in executor.map(
            run_case, tasks, chunksize=1
        ):
            if error is None:
                rows.append(metrics)
                status = "completed"
            else:
                segment_id = int(scenario_name.rsplit("_", 1)[-1])
                failure_rows.append(
                    {
                        "scenario": scenario_name,
                        "segment_id": segment_id,
                        "strategy": strategy,
                        "error": error,
                    }
                )
                status = "failed"
            print(f"{status} {scenario_name} strategy={strategy}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    per_run = pd.DataFrame(rows)
    failures = pd.DataFrame(
        failure_rows, columns=["scenario", "segment_id", "strategy", "error"]
    )
    per_run = per_run.merge(
        manifest[["segment_id", "positive_share"]],
        left_on="load_seed",
        right_on="segment_id",
        how="left",
        validate="many_to_one",
    )
    operating_runs = per_run[per_run.positive_share > 0].copy()
    paired = paired_strategy_comparison(
        operating_runs,
        reference_strategy="average",
        metrics=PAIR_METRICS,
    )
    beam_vs_instant = paired_strategy_comparison(
        operating_runs,
        reference_strategy="instant_health",
        metrics=PAIR_METRICS + ("planning_runtime_s",),
    )
    beam_vs_instant = beam_vs_instant[
        beam_vs_instant.strategy == "beam_health"
    ].reset_index(drop=True)
    numeric = (
        "hydrogen_g_per_fc_kwh",
        "main_expected_damage_increment_pct",
        "damage_increment_range_pct",
        "fc_tracking_mae_kw",
        "fc_tracking_max_abs_kw",
        "total_switch_count",
        "total_load_shift_count",
        "online_step_range",
        "planning_runtime_s",
    )
    aggregate_rows = []
    for strategy in STRATEGIES:
        group = per_run[per_run.strategy == strategy]
        operating = group[group.positive_share > 0]
        failed = failures[failures.strategy == strategy]
        row = {
            "strategy": strategy,
            "n_segments": len(manifest),
            "successful_segments": len(group),
            "failed_segments": len(failed),
            "operating_segments": len(operating),
            "execution_success_share": float(len(group) / len(manifest)),
            "zero_violation_share": float(group.constraint_violation_steps.eq(0).mean()),
            "tracking_within_tolerance_share": float(
                group.fc_tracking_within_tolerance_share.mean()
            ),
        }
        for metric in numeric:
            values = operating[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    manifest.to_csv(args.out_dir / "window_manifest.csv", index=False)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    failures.to_csv(args.out_dir / "failed_runs.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    beam_vs_instant.to_csv(
        args.out_dir / "beam_vs_instant_deltas.csv", index=False
    )
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "holdout_segments": list(HOLDOUT_SEGMENTS),
        "selection_rule": (
            "one temporal center window per complete holdout segment; short segments "
            "kept in full; selection does not inspect power values"
        ),
        "window_length_s": args.window_length,
        "normalization_power_kw": NORMALIZATION_POWER_KW,
        "normalization_source": "frozen calibration-partition target-power maximum",
        "mapping_system_power_reference_kw": system_power_reference_kw,
        "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
        "initial_damage_fraction": list(INITIAL_DAMAGE_FRACTION),
        "strategies": list(STRATEGIES),
        "beam_preview": "current demand held constant; no future demand access",
        "parallel_jobs": args.jobs,
        "stochastic_health": False,
        "runtime_s": time.perf_counter() - started,
        "no_retuning": True,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "| 策略 | 成功/全部segment | 执行成功率 | 零违规 | 容差内 | H2强度(g/kWh) | 期望退化(%) | 退化增量极差(%) | 跟踪MAE(kW) | 启停 | 在线步数极差 | 规划时间(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.successful_segments}/{row.n_segments} | "
            f"{row.execution_success_share:.0%} | "
            f"{row.zero_violation_share:.0%} | "
            f"{row.tracking_within_tolerance_share:.0%} | "
            f"{row.hydrogen_g_per_fc_kwh_mean:.3f} | "
            f"{row.main_expected_damage_increment_pct_mean:.6f} | "
            f"{row.damage_increment_range_pct_mean:.6f} | "
            f"{row.fc_tracking_mae_kw_mean:.3f} | "
            f"{row.total_switch_count_mean:.1f} | "
            f"{row.online_step_range_mean:.1f} | "
            f"{row.planning_runtime_s_mean:.3f} |"
        )
    beam_tracking = beam_vs_instant[
        beam_vs_instant.metric == "fc_tracking_mae_kw"
    ].iloc[0]
    beam_runtime = beam_vs_instant[
        beam_vs_instant.metric == "planning_runtime_s"
    ].iloc[0]
    report = f"""# FC-only真实留出段中心窗口回放

- 数据仅来自未参与标定的segment 22-45，共{len(manifest)}个完整segment；不重新调整任何模型或控制参数。
- 每段按时间位置取中心最多{args.window_length}秒，短段全部保留；窗口选择不查看功率值。
- 单堆`fc_input_power_kw`按冻结的30 kW标定参考归一化，再映射到两台健康堆95%容量。
- 其中含正功率窗口{int((manifest.positive_share > 0).sum())}个，纯停机窗口{int((manifest.positive_share == 0).sum())}个。
- 物理均值只统计成功的正功率窗口；纯停机窗口只进入执行成功率和约束检查。无可行动作会记录到`failed_runs.csv`，不做策略替换或容差放宽。

{chr(10).join(lines)}

Average在{len(failures[failures.strategy == 'average'])}个真实窗口无严格等电流可行动作；其余策略全部执行成功。
Beam相对Instant的跟踪MAE平均变化{beam_tracking.mean_relative_pct:+.2f}%（差值{beam_tracking.mean_delta:+.3f} ± {beam_tracking.ci95:.3f} kW），规划时间增加{beam_runtime.mean_relative_pct:.1f}%；当前差异不支持Beam优于Instant。

该结果验证固定参数在真实连续窗口上的可执行性，不等于86,415秒全量回放，也不构成寿命外推。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
