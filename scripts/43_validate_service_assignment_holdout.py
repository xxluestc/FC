"""Validate frozen slow assignments on unseen real holdout windows."""

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
    candidate_assignments,
    choose_service_assignment,
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
OUTPUT = ROOT / "data/results/fc_only_service_holdout_assignment"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
HETEROGENEITY = (1.0, 1.05, 1.10)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}
NORMALIZATION_POWER_KW = 30.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5


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


def run_assignment(task):
    model, scenario, assignment, descriptor = task
    try:
        run = run_policy(
            model,
            scenario,
            "instant_health",
            fixed_online_assignment=assignment,
        )
    except RuntimeError as error:
        return {**descriptor, "assignment": str(assignment), "feasible": False, "error": str(error)}
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    initial = np.asarray(scenario.initial_damage_fraction, dtype=float) * reference
    increments = np.zeros(model.n_stacks, dtype=float)
    entry_start_removed = 0.0
    for stack in range(model.n_stacks):
        increments[stack] = float(
            run.trajectory[f"stack_{stack}_expected_continuous_increment_pct"].sum()
            + run.trajectory[f"stack_{stack}_ramp_increment_pct"].sum()
            + run.trajectory[f"stack_{stack}_shift_increment_pct"].sum()
            + run.trajectory[f"stack_{stack}_start_stop_increment_pct"].iloc[1:].sum()
        )
        entry_start_removed += float(
            run.trajectory[f"stack_{stack}_start_stop_increment_pct"].iloc[0]
        )
    adjusted = initial + increments
    return {
        **descriptor,
        "assignment": str(assignment),
        "assignment_first": assignment[0],
        "assignment_second": assignment[1],
        "feasible": True,
        "error": "",
        "adjusted_terminal_max_pct": float(adjusted.max()),
        "adjusted_terminal_range_pct": float(adjusted.max() - adjusted.min()),
        "adjusted_increment_total_pct": float(increments.sum()),
        "entry_start_removed_pct": entry_start_removed,
        "tracking_mae_kw": run.metrics["fc_tracking_mae_kw"],
        "tracking_max_abs_kw": run.metrics["fc_tracking_max_abs_kw"],
        "tracking_within_tolerance_share": run.metrics[
            "fc_tracking_within_tolerance_share"
        ],
        "constraint_violation_steps": run.metrics["constraint_violation_steps"],
        "hydrogen_g_per_fc_kwh": run.metrics["hydrogen_g_per_fc_kwh"],
    }


def frozen_decisions(initial_fraction, decision_exposure, config):
    damage = tuple(float(value * config.health_limit_pct) for value in initial_fraction)
    state = ServiceScheduleState(damage)
    healthiest = tuple(
        sorted(range(len(damage)), key=lambda index: (damage[index], index))[:2]
    )
    return {
        "fixed_pair": orient_service_pair(
            (0, 1), state, decision_exposure, config.heterogeneity_factors
        ),
        "health_greedy": orient_service_pair(
            healthiest, state, decision_exposure, config.heterogeneity_factors
        ),
        "expected_max": choose_service_assignment(
            state, decision_exposure, config, objective="expected_max"
        ).assignment,
    }


def compare_with_oracle(results, decision_exposure, config):
    rows = []
    keys = ["segment_id", "health_case"]
    for key, group in results.groupby(keys, sort=True):
        feasible = group[group.feasible].copy()
        if feasible.empty:
            continue
        oracle = feasible.sort_values(
            ["adjusted_terminal_max_pct", "tracking_mae_kw", "assignment"]
        ).iloc[0]
        initial_fraction = HEALTH_CASES[key[1]]
        decisions = frozen_decisions(initial_fraction, decision_exposure, config)
        fixed_value = None
        selected_rows = {}
        for policy, assignment in decisions.items():
            match = feasible[
                (feasible.assignment_first == assignment[0])
                & (feasible.assignment_second == assignment[1])
            ]
            selected_rows[policy] = None if match.empty else match.iloc[0]
            if policy == "fixed_pair" and len(match):
                fixed_value = float(match.iloc[0].adjusted_terminal_max_pct)
        for policy, assignment in decisions.items():
            selected = selected_rows[policy]
            rows.append(
                {
                    "segment_id": key[0],
                    "health_case": key[1],
                    "policy": policy,
                    "selected_assignment": str(assignment),
                    "selected_online_set": str(tuple(sorted(assignment))),
                    "oracle_assignment": oracle.assignment,
                    "oracle_online_set": str(
                        tuple(sorted((int(oracle.assignment_first), int(oracle.assignment_second))))
                    ),
                    "feasible": selected is not None,
                    "exact_assignment_hit": bool(
                        selected is not None and selected.assignment == oracle.assignment
                    ),
                    "online_set_hit": bool(
                        selected is not None
                        and {int(selected.assignment_first), int(selected.assignment_second)}
                        == {int(oracle.assignment_first), int(oracle.assignment_second)}
                    ),
                    "adjusted_terminal_max_pct": (
                        np.nan if selected is None else selected.adjusted_terminal_max_pct
                    ),
                    "oracle_terminal_max_pct": oracle.adjusted_terminal_max_pct,
                    "regret_pct": (
                        np.nan
                        if selected is None
                        else selected.adjusted_terminal_max_pct
                        - oracle.adjusted_terminal_max_pct
                    ),
                    "delta_vs_fixed_pct": (
                        np.nan
                        if selected is None or fixed_value is None
                        else selected.adjusted_terminal_max_pct - fixed_value
                    ),
                    "tracking_mae_kw": np.nan if selected is None else selected.tracking_mae_kw,
                    "tracking_max_abs_kw": (
                        np.nan if selected is None else selected.tracking_max_abs_kw
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(comparison):
    return comparison.groupby("policy", sort=False).agg(
        cases=("segment_id", "count"),
        feasible_share=("feasible", "mean"),
        exact_assignment_hit_share=("exact_assignment_hit", "mean"),
        online_set_hit_share=("online_set_hit", "mean"),
        regret_mean_pct=("regret_pct", "mean"),
        regret_max_pct=("regret_pct", "max"),
        delta_vs_fixed_mean_pct=("delta_vs_fixed_pct", "mean"),
        delta_vs_fixed_improvement_share=("delta_vs_fixed_pct", lambda values: float((values < 0).mean())),
        tracking_mae_mean_kw=("tracking_mae_kw", "mean"),
        tracking_max_abs_kw=("tracking_max_abs_kw", "max"),
    ).reset_index()


def plot_summary(summary):
    order = ("fixed_pair", "health_greedy", "expected_max")
    labels = ("Fixed", "Health-greedy", "Expected-max")
    colors = ("#6C757D", "#7A5195", "#1D3557")
    selected = summary.set_index("policy").loc[list(order)]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    x = np.arange(len(order))
    bars0 = axes[0].bar(x, selected.online_set_hit_share, color=colors, width=0.65)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Oracle online-set hit share")
    bars1 = axes[1].bar(x, selected.regret_mean_pct * 1e6, color=colors, width=0.65)
    axes[1].set_ylabel("Mean regret ($10^{-6}$ %-point)")
    bars2 = axes[2].bar(x, selected.tracking_mae_mean_kw, color=colors, width=0.65)
    axes[2].axhline(TRACKING_TOLERANCE_KW, color="#666666", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("FC tracking MAE (kW)")
    for ax, bars, values, formatter in (
        (axes[0], bars0, selected.online_set_hit_share, lambda value: f"{value:.2f}"),
        (axes[1], bars1, selected.regret_mean_pct * 1e6, lambda value: f"{value:.1f}"),
        (axes[2], bars2, selected.tracking_mae_mean_kw, lambda value: f"{value:.2f}"),
    ):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(float(value), 0) + 0.02 * max(1.0, float(np.max(values))),
                formatter(float(value)),
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    for index, ax in enumerate(axes):
        ax.set_xticks(x, labels, rotation=12)
        ax.text(-0.13, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.65, w_pad=1.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig13_holdout_assignment_validation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-length", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.window_length <= 0 or args.jobs <= 0:
        raise ValueError("window length and jobs must be positive")
    if args.summarize_only:
        summary_path = args.out_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError("--summarize-only requires summary.csv")
        plot_summary(pd.read_csv(summary_path))
        return
    frame = pd.read_csv(
        SOURCE,
        usecols=["timestamp", "segment_id", "fc_input_power_kw"],
    )
    split = split_at_largest_segment_gap(frame)
    split_metadata = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    if list(split.holdout_segments) != split_metadata["holdout_segments"]:
        raise AssertionError("holdout split differs from frozen audit")
    holdout = frame[frame.segment_id.isin(split.holdout_segments)].copy()

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
    power_reference = (1 - CAPACITY_RESERVE_FRACTION) * model.fc_power_reference_kw()
    health = json.loads(HEALTH_CALIBRATION.read_text(encoding="utf-8"))
    config = ServiceScheduleConfig(
        health_limit_pct=float(health["terminal_total_damage_pct"]),
        gamma_scale_pct=float(health["gamma_scale_pct"]),
        heterogeneity_factors=HETEROGENEITY,
        start_damage_pct=float(
            health["coefficients_percent_units"]["start_stop_pct_per_cycle"]
        ),
        risk_horizon_h=100.0,
        risk_samples=128,
    )
    decision_exposure = load_development_exposure()
    manifest_rows = []
    tasks = []
    for segment_id, segment in holdout.groupby("segment_id", sort=True):
        segment = segment.reset_index(drop=True)
        length = min(args.window_length, len(segment))
        start = (len(segment) - length) // 2
        window = segment.iloc[start : start + length].copy()
        normalized = np.clip(
            window.fc_input_power_kw.to_numpy(dtype=float) / NORMALIZATION_POWER_KW,
            0.0,
            1.0,
        )
        positive_share = float((normalized > 0).mean())
        manifest_rows.append(
            {
                "segment_id": int(segment_id),
                "segment_rows": len(segment),
                "window_start_offset": start,
                "window_rows": length,
                "window_start_timestamp": window.timestamp.iloc[0],
                "window_end_timestamp": window.timestamp.iloc[-1],
                "positive_share": positive_share,
            }
        )
        if positive_share == 0:
            continue
        demand = pd.DataFrame(
            {
                "demand_power_kw": normalized * power_reference,
                "event": np.where(normalized > 0, "real_fc_on", "real_fc_off"),
                "source": "real_holdout_center_window",
                "seed": int(segment_id),
            }
        )
        for health_case, initial_fraction in HEALTH_CASES.items():
            scenario = TestScenario(
                name=f"holdout_{segment_id}_{health_case}",
                demand=demand,
                initial_damage_fraction=initial_fraction,
                health_seed=50_000 + int(segment_id),
                stochastic_health=False,
            )
            for assignment in candidate_assignments(3):
                tasks.append(
                    (
                        model,
                        scenario,
                        assignment,
                        {
                            "segment_id": int(segment_id),
                            "health_case": health_case,
                            "positive_share": positive_share,
                        },
                    )
                )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(run_assignment, tasks, chunksize=1))
    results = pd.DataFrame(rows)
    comparison = compare_with_oracle(results, decision_exposure, config)
    summary = summarize(comparison)
    if not np.allclose(
        comparison.loc[comparison.feasible, "tracking_max_abs_kw"].le(
            TRACKING_TOLERANCE_KW + 1e-12
        ),
        True,
    ):
        raise AssertionError("a feasible holdout assignment exceeded tracking tolerance")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(args.out_dir / "window_manifest.csv", index=False)
    results.to_csv(args.out_dir / "candidate_assignment_runs.csv", index=False)
    comparison.to_csv(args.out_dir / "policy_vs_oracle.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    plot_summary(summary)
    metadata = {
        "scope": "frozen slow-assignment validation on unseen real center windows",
        "holdout_segments": list(split.holdout_segments),
        "operating_windows": int(pd.DataFrame(manifest_rows).positive_share.gt(0).sum()),
        "health_cases": HEALTH_CASES,
        "candidate_assignments": [list(value) for value in candidate_assignments(3)],
        "decision_information": "development-template mean and pre-window health only",
        "oracle_information": "post-window candidate outcomes; evaluation only",
        "entry_start_handling": "first-sample start damage removed from assignment ranking",
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 冻结慢层真实留出分配验证\n\n"
    report += (
        "策略只使用开发模板均值和窗口前健康状态；事后oracle枚举6个有序双堆分配，"
        "仅用于计算留出遗憾。三个健康案例是同一老化分布的堆身份循环，不参与调参。"
        "窗口入口的人为启动损伤从排名指标中删除。\n\n"
    )
    report += summary.to_markdown(index=False) + "\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
