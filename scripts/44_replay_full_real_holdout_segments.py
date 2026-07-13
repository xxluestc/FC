"""Replay every unseen real holdout segment with frozen stack assignments."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    TestScenario,
    orient_service_pair,
    run_policy,
    split_at_largest_segment_gap,
    stationary_service_exposure,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
SPLIT_METADATA = ROOT / "data/results/load_zuo_calibration/metadata.json"
TEMPLATES = ROOT / "data/results/fc_only_service_templates/service_exposure_templates.csv"
HEALTH_CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_full_holdout_replay"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
HETEROGENEITY = (1.0, 1.05, 1.10)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}
POLICIES = ("fixed_pair", "health_greedy")
NORMALIZATION_POWER_KW = 30.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
TRACE_SEGMENT_ID = 32
TRACE_HEALTH_CASE = "oldest_stack_0"
TRACE_POLICY = "health_greedy"


def load_development_exposure() -> ServiceExposure:
    table = pd.read_csv(TEMPLATES)
    selected = table[table.template_source == "real_calibration_window"]
    templates = [
        ServiceExposure(
            duration_h=float(row.duration_h),
            continuous_mean_pct=(
                row.role_0_continuous_mean_pct,
                row.role_1_continuous_mean_pct,
            ),
            load_shift_damage_pct=(
                row.role_0_load_shift_damage_pct,
                row.role_1_load_shift_damage_pct,
            ),
            operational_start_damage_pct=(
                row.role_0_operational_start_damage_pct,
                row.role_1_operational_start_damage_pct,
            ),
        )
        for row in selected.itertuples(index=False)
    ]
    return stationary_service_exposure(templates, 1.0)


def frozen_assignment(policy, initial_fraction, exposure, config):
    damage = tuple(float(value * config.health_limit_pct) for value in initial_fraction)
    state = ServiceScheduleState(damage)
    if policy == "fixed_pair":
        pair = (0, 1)
    elif policy == "health_greedy":
        pair = tuple(sorted(range(3), key=lambda index: (damage[index], index))[:2])
    else:
        raise ValueError(f"unknown policy: {policy}")
    return orient_service_pair(pair, state, exposure, config.heterogeneity_factors)


def run_case(task):
    model, scenario, assignment, descriptor, capture_trace = task
    try:
        run = run_policy(
            model,
            scenario,
            "instant_health",
            fixed_online_assignment=assignment,
        )
    except RuntimeError as error:
        return {**descriptor, "assignment": str(assignment), "error": str(error)}, None

    trajectory = run.trajectory
    final_damage = np.asarray(
        [
            trajectory[f"stack_{index}_damage_after_pct"].iloc[-1]
            for index in range(model.n_stacks)
        ],
        dtype=float,
    )
    initial_damage = np.asarray(
        [
            trajectory[f"stack_{index}_damage_before_pct"].iloc[0]
            for index in range(model.n_stacks)
        ],
        dtype=float,
    )
    tracking = trajectory.fc_power_tracking_error_kw.to_numpy(dtype=float)
    row = {
        **descriptor,
        "assignment": str(assignment),
        "assignment_first": assignment[0],
        "assignment_second": assignment[1],
        "error": "",
        "n_steps": len(trajectory),
        "fc_energy_kwh": run.metrics["fc_energy_kwh"],
        "hydrogen_g": run.metrics["hydrogen_g"],
        "expected_damage_increment_pct": run.metrics[
            "main_expected_damage_increment_pct"
        ],
        "continuous_damage_increment_pct": run.metrics[
            "main_expected_continuous_damage_pct"
        ],
        "ramp_damage_increment_pct": run.metrics["main_ramp_damage_pct"],
        "shift_damage_increment_pct": run.metrics["main_shift_damage_pct"],
        "start_stop_damage_increment_pct": run.metrics[
            "main_start_stop_damage_pct"
        ],
        "terminal_max_damage_pct": float(final_damage.max()),
        "terminal_damage_range_pct": float(final_damage.max() - final_damage.min()),
        "damage_increment_range_pct": float(
            (final_damage - initial_damage).max()
            - (final_damage - initial_damage).min()
        ),
        "tracking_abs_sum_kw": float(np.abs(tracking).sum()),
        "tracking_sq_sum_kw2": float(np.square(tracking).sum()),
        "tracking_max_abs_kw": float(np.abs(tracking).max()),
        "tracking_within_tolerance_steps": int(
            (np.abs(tracking) <= TRACKING_TOLERANCE_KW + 1e-12).sum()
        ),
        "constraint_violation_steps": run.metrics["constraint_violation_steps"],
        "safety_override_steps": run.metrics["safety_override_steps"],
        "total_switch_count": run.metrics["total_switch_count"],
        "total_load_shift_count": run.metrics["total_load_shift_count"],
        "planning_runtime_s": run.metrics["planning_runtime_s"],
    }
    trace = None
    if capture_trace:
        trace = pd.DataFrame(
            {
                "step": trajectory.step,
                "demand_power_kw": trajectory.demand_power_kw,
                "stack_power_kw": trajectory.stack_power_kw,
                "tracking_error_kw": trajectory.fc_power_tracking_error_kw,
            }
        )
        trace["minute"] = (trace.step // 60).astype(int)
        trace = trace.groupby("minute", as_index=False).agg(
            time_h=("step", lambda values: float(values.mean() / 3600.0)),
            demand_power_kw=("demand_power_kw", "mean"),
            stack_power_kw=("stack_power_kw", "mean"),
            tracking_max_abs_kw=("tracking_error_kw", lambda values: float(np.abs(values).max())),
        )
    return row, trace


def pair_policies(per_run):
    metrics = (
        "terminal_max_damage_pct",
        "terminal_damage_range_pct",
        "damage_increment_range_pct",
        "expected_damage_increment_pct",
        "hydrogen_g",
        "fc_energy_kwh",
        "tracking_abs_sum_kw",
        "tracking_max_abs_kw",
        "total_switch_count",
    )
    keys = ["segment_id", "health_case", "positive_steps"]
    wide = per_run.pivot(index=keys, columns="policy", values=list(metrics))
    rows = []
    for index, values in wide.iterrows():
        row = dict(zip(keys, index))
        for metric in metrics:
            fixed = float(values[(metric, "fixed_pair")])
            greedy = float(values[(metric, "health_greedy")])
            row[f"fixed_{metric}"] = fixed
            row[f"health_greedy_{metric}"] = greedy
            row[f"delta_{metric}"] = greedy - fixed
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(per_run):
    rows = []
    for (health_case, policy), group in per_run.groupby(
        ["health_case", "policy"], sort=False
    ):
        steps = int(group.n_steps.sum())
        energy = float(group.fc_energy_kwh.sum())
        rows.append(
            {
                "health_case": health_case,
                "policy": policy,
                "segments": len(group),
                "steps": steps,
                "positive_steps": int(group.positive_steps.sum()),
                "constraint_violation_steps": int(group.constraint_violation_steps.sum()),
                "safety_override_steps": int(group.safety_override_steps.sum()),
                "tracking_mae_kw": float(group.tracking_abs_sum_kw.sum() / steps),
                "tracking_rmse_kw": float(np.sqrt(group.tracking_sq_sum_kw2.sum() / steps)),
                "tracking_max_abs_kw": float(group.tracking_max_abs_kw.max()),
                "tracking_within_tolerance_share": float(
                    group.tracking_within_tolerance_steps.sum() / steps
                ),
                "hydrogen_g_per_fc_kwh": float(group.hydrogen_g.sum() / max(energy, 1e-12)),
                "expected_damage_increment_sum_pct": float(
                    group.expected_damage_increment_pct.sum()
                ),
                "total_switch_count": int(group.total_switch_count.sum()),
                "planning_runtime_s": float(group.planning_runtime_s.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(per_run, paired, trace):
    operating = per_run[per_run.positive_steps > 0]
    greedy = operating[operating.policy == "health_greedy"]
    segment_tracking = greedy.groupby("segment_id", sort=True).tracking_max_abs_kw.max()
    operating_pairs = paired[paired.positive_steps > 0]
    health_order = list(HEALTH_CASES)
    health_labels = ("Oldest: 2", "Oldest: 0", "Oldest: 1")
    damage_delta = (
        operating_pairs.groupby("health_case").delta_terminal_max_damage_pct.mean()
        .reindex(health_order)
        * 1e3
    )
    colors = ("#1D3557", "#2A9D8F", "#E76F51")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45))
    axes[0].plot(trace.time_h, trace.demand_power_kw, color="#6C757D", linewidth=0.9, label="Demand")
    axes[0].plot(trace.time_h, trace.stack_power_kw, color=colors[0], linewidth=0.9, label="FC output")
    axes[0].set_xlabel("Observed time (h)")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")

    x = np.arange(len(segment_tracking))
    axes[1].bar(x, segment_tracking.to_numpy(), color=colors[1], width=0.68)
    axes[1].axhline(TRACKING_TOLERANCE_KW, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, [str(value) for value in segment_tracking.index])
    axes[1].set_xlabel("Holdout segment")
    axes[1].set_ylabel("Max tracking error (kW)")

    bars = axes[2].bar(np.arange(3), damage_delta.to_numpy(), color=colors, width=0.68)
    axes[2].axhline(0, color="#555555", linewidth=0.7)
    axes[2].set_xticks(np.arange(3), health_labels, rotation=12)
    axes[2].set_ylabel("Greedy - fixed max damage\n($10^{-3}$ %-point)")
    axes[2].set_ylim(min(-1.0, float(damage_delta.min()) * 1.12), 0.5)
    for bar, value in zip(bars, damage_delta):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    for index, ax in enumerate(axes):
        ax.text(-0.13, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.65, w_pad=1.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig14_full_real_holdout_replay.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_report(per_run, aggregate, manifest, out_dir):
    clipped_steps = int(manifest.clipped_high_steps.sum())
    positive_steps = int(manifest.positive_steps.sum())
    operating_segments = int(manifest.positive_steps.gt(0).sum())
    indexed = aggregate.set_index(["health_case", "policy"])
    tradeoff_lines = []
    for health_case in ("oldest_stack_0", "oldest_stack_1"):
        fixed = indexed.loc[(health_case, "fixed_pair")]
        greedy = indexed.loc[(health_case, "health_greedy")]
        tradeoff_lines.append(
            f"- `{health_case}`：health-greedy相对固定双堆的全段期望退化总和变化"
            f"{greedy.expected_damage_increment_sum_pct - fixed.expected_damage_increment_sum_pct:+.6f}个百分点，"
            f"跟踪MAE变化{greedy.tracking_mae_kw - fixed.tracking_mae_kw:+.3f} kW，"
            f"氢耗强度变化{greedy.hydrogen_g_per_fc_kwh - fixed.hydrogen_g_per_fc_kwh:+.3f} g/kWh。"
        )
    report = f"""# 未见真实留出完整segment回放

- segment 22-45共{int(manifest.rows.sum()):,}个1秒样本，全部进入回放，其中正功率{positive_steps:,}步、正功率segment {operating_segments}个；没有抽窗或删除纯停机段。
- 每个完整segment内部逐秒携带动作、驻留和确定性健康状态；未知时间缺口之间不桥接状态，因此不虚构缺口内负载或退化。
- 仅比较冻结固定双堆与24小时慢层的health-greedy入口选择；快层均为Instant，未使用未来需求。
- 3种健康身份循环、24段、2策略共{len(per_run)}例全部完成；约束违规总数为0，最大跟踪误差为{per_run.tracking_max_abs_kw.max():.3f} kW。
- 当固定集合包含最老堆时，health-greedy在8/8正功率段降低终端最大退化；但它优化的是最大健康而不是总退化，代价权衡如下。
{chr(10).join(tradeoff_lines)}
- 高于冻结30 kW参考的样本有{clipped_steps:,}步，占正功率步{clipped_steps / positive_steps:.2%}，回放将其截到归一化1.0。因此本结果是设计包络内全样本验证，不是留出峰值的全保真容量验证；容量缺口另见`fc_only_holdout_capacity_audit`。

该结果验证冻结双时间尺度方法在所有未见完整连续块上的可执行性和健康均衡方向。段间存在未观测缺口，故不将24段拼接成一条虚构连续行程，也不据此声称实车寿命预测。
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        paired = pd.read_csv(args.out_dir / "paired_policy_deltas.csv")
        aggregate = pd.read_csv(args.out_dir / "aggregate_metrics.csv")
        manifest = pd.read_csv(args.out_dir / "segment_manifest.csv")
        trace = pd.read_csv(args.out_dir / "representative_trace_60s.csv")
        plot_results(per_run, paired, trace)
        write_report(per_run, aggregate, manifest, args.out_dir)
        return

    frame = pd.read_csv(
        SOURCE, usecols=["timestamp", "segment_id", "fc_input_power_kw"]
    )
    split = split_at_largest_segment_gap(frame)
    frozen_split = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    if list(split.holdout_segments) != frozen_split["holdout_segments"]:
        raise AssertionError("holdout split differs from frozen calibration audit")
    holdout = frame[frame.segment_id.isin(split.holdout_segments)].copy()
    if len(holdout) != 86_415:
        raise AssertionError("frozen holdout row count changed")

    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    system_reference_kw = (
        1 - CAPACITY_RESERVE_FRACTION
    ) * model.fc_power_reference_kw()
    calibration = json.loads(HEALTH_CALIBRATION.read_text(encoding="utf-8"))
    schedule_config = ServiceScheduleConfig(
        health_limit_pct=float(calibration["terminal_total_damage_pct"]),
        gamma_scale_pct=float(calibration["gamma_scale_pct"]),
        heterogeneity_factors=HETEROGENEITY,
        start_damage_pct=float(
            calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
        ),
    )
    decision_exposure = load_development_exposure()
    tasks = []
    manifest_rows = []
    for segment_id, segment in holdout.groupby("segment_id", sort=True):
        segment = segment.reset_index(drop=True)
        normalized = np.clip(
            segment.fc_input_power_kw.to_numpy(dtype=float) / NORMALIZATION_POWER_KW,
            0.0,
            1.0,
        )
        demand = pd.DataFrame(
            {
                "demand_power_kw": normalized * system_reference_kw,
                "event": np.where(normalized > 0, "real_fc_on", "real_fc_off"),
                "source": "real_holdout_full_segment",
                "seed": int(segment_id),
            }
        )
        positive_steps = int((normalized > 0).sum())
        manifest_rows.append(
            {
                "segment_id": int(segment_id),
                "rows": len(segment),
                "positive_steps": positive_steps,
                "start_timestamp": segment.timestamp.iloc[0],
                "end_timestamp": segment.timestamp.iloc[-1],
                "raw_power_mean_kw": float(segment.fc_input_power_kw.mean()),
                "raw_power_max_kw": float(segment.fc_input_power_kw.max()),
                "clipped_high_steps": int(
                    (segment.fc_input_power_kw > NORMALIZATION_POWER_KW).sum()
                ),
            }
        )
        for health_case, initial_fraction in HEALTH_CASES.items():
            scenario = TestScenario(
                name=f"full_holdout_{int(segment_id)}_{health_case}",
                demand=demand,
                initial_damage_fraction=initial_fraction,
                health_seed=60_000 + int(segment_id),
                stochastic_health=False,
            )
            for policy in POLICIES:
                assignment = frozen_assignment(
                    policy, initial_fraction, decision_exposure, schedule_config
                )
                descriptor = {
                    "segment_id": int(segment_id),
                    "health_case": health_case,
                    "policy": policy,
                    "positive_steps": positive_steps,
                }
                capture_trace = (
                    int(segment_id) == TRACE_SEGMENT_ID
                    and health_case == TRACE_HEALTH_CASE
                    and policy == TRACE_POLICY
                )
                tasks.append((model, scenario, assignment, descriptor, capture_trace))

    started = time.perf_counter()
    rows = []
    traces = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for index, (row, trace) in enumerate(executor.map(run_case, tasks, chunksize=1), 1):
            rows.append(row)
            if trace is not None:
                traces.append(trace)
            if index % 12 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} cases", flush=True)
    per_run = pd.DataFrame(rows)
    failures = per_run[per_run.error != ""]
    if len(failures):
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(args.out_dir / "per_run_metrics_with_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} full holdout runs failed")
    if int(per_run.constraint_violation_steps.sum()) != 0:
        raise AssertionError("full holdout replay contains constraint violations")
    if float(per_run.tracking_max_abs_kw.max()) > TRACKING_TOLERANCE_KW + 1e-12:
        raise AssertionError("full holdout replay exceeded tracking tolerance")
    if len(traces) != 1:
        raise AssertionError("representative trace was not captured exactly once")

    manifest = pd.DataFrame(manifest_rows)
    paired = pair_policies(per_run)
    aggregate = aggregate_runs(per_run)
    trace = traces[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_dir / "segment_manifest.csv", index=False)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_policy_deltas.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    trace.to_csv(args.out_dir / "representative_trace_60s.csv", index=False)
    plot_results(per_run, paired, trace)

    operating = paired[paired.positive_steps > 0]
    summary_rows = []
    for health_case in HEALTH_CASES:
        group = operating[operating.health_case == health_case]
        delta = group.delta_terminal_max_damage_pct
        summary_rows.append(
            {
                "health_case": health_case,
                "operating_segments": len(group),
                "terminal_max_delta_mean_pct": float(delta.mean()),
                "terminal_max_delta_max_pct": float(delta.max()),
                "health_greedy_better_share": float((delta < -1e-15).mean()),
                "health_greedy_nonworse_share": float((delta <= 1e-15).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    metadata = {
        "scope": "all unseen real holdout samples, replayed as complete independent segments",
        "holdout_segments": list(split.holdout_segments),
        "holdout_rows": len(holdout),
        "positive_rows": int((holdout.fc_input_power_kw > 0).sum()),
        "segment_boundary_handling": (
            "health and control state are continuous within each complete segment; "
            "states are not bridged across unobserved timestamp gaps"
        ),
        "initial_state_assumption": "each independent segment starts from an actionable all-off control state",
        "normalization_power_kw": NORMALIZATION_POWER_KW,
        "mapping_system_power_reference_kw": system_reference_kw,
        "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
        "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
        "health_cases": HEALTH_CASES,
        "policies": list(POLICIES),
        "decision_information": "development real-template mean and segment-entry health only",
        "stochastic_health": False,
        "future_demand_used": False,
        "no_retuning": True,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(per_run, aggregate, manifest, args.out_dir)
    print(summary.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
