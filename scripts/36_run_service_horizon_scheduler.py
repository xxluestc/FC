"""Screen slow N+1 stack scheduling with real-calibrated exposure templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    choose_service_assignment,
    stationary_service_exposure,
    transition_service_epoch,
)
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/results/fc_only_mechanism_ablation/per_run_metrics.csv"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_service_scheduler"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
POLICIES = ("fixed_pair", "periodic_rotation", "expected_max", "gamma_cvar")
HETEROGENEITY = np.asarray((1.0, 1.05, 1.10))
INITIAL_DAMAGE_FRACTION = np.asarray((0.10, 0.40, 0.80))


def load_templates() -> list[ServiceExposure]:
    table = pd.read_csv(INPUT)
    selected = table[
        (table.load_source == "empirical_1s")
        & (table.strategy == "instant_health")
    ]
    if len(selected) < 3:
        raise ValueError("at least three empirical Instant templates are required")
    templates = []
    for row in selected.itertuples(index=False):
        templates.append(
            ServiceExposure(
                duration_h=float(row.n_steps) / 3600.0,
                continuous_mean_pct=(
                    row.stack_0_main_continuous_damage_pct / HETEROGENEITY[0],
                    row.stack_1_main_continuous_damage_pct / HETEROGENEITY[1],
                ),
                load_shift_damage_pct=(
                    row.stack_0_main_shift_damage_pct / HETEROGENEITY[0],
                    row.stack_1_main_shift_damage_pct / HETEROGENEITY[1],
                ),
            )
        )
    return templates


def aggregate_epoch(
    templates: list[ServiceExposure],
    rng: np.random.Generator,
    epoch_h: float,
) -> ServiceExposure:
    base_duration = templates[0].duration_h
    if not all(np.isclose(item.duration_h, base_duration) for item in templates):
        raise ValueError("service templates must have equal duration")
    blocks = int(round(epoch_h / base_duration))
    if blocks <= 0 or not np.isclose(blocks * base_duration, epoch_h):
        raise ValueError("epoch_h must be an integer multiple of template duration")
    sampled = rng.integers(0, len(templates), size=blocks)
    continuous = np.sum(
        [templates[index].continuous_mean_pct for index in sampled], axis=0
    )
    shifts = np.sum(
        [templates[index].load_shift_damage_pct for index in sampled], axis=0
    )
    return ServiceExposure(
        duration_h=epoch_h,
        continuous_mean_pct=tuple(float(value) for value in continuous),
        load_shift_damage_pct=tuple(float(value) for value in shifts),
    )


def orient_pair(pair, state, exposure):
    """Map the heavier role to the stack with more residual health."""

    first, second = pair
    role_damage = np.asarray(exposure.continuous_mean_pct) + np.asarray(
        exposure.load_shift_damage_pct
    )
    candidates = ((first, second), (second, first))
    return min(
        candidates,
        key=lambda assignment: (
            max(
                state.damage_pct[stack] + role_damage[role] * HETEROGENEITY[stack]
                for role, stack in enumerate(assignment)
            ),
            assignment,
        ),
    )


def choose_policy_assignment(
    policy,
    state,
    exposure,
    config,
    epoch,
    rotation_epochs,
    reschedule_epochs,
):
    if policy == "fixed_pair":
        return orient_pair((0, 1), state, exposure)
    if policy == "periodic_rotation":
        pairs = ((0, 1), (1, 2), (2, 0))
        pair = pairs[(epoch // rotation_epochs) % len(pairs)]
        return orient_pair(pair, state, exposure)
    if (
        policy in {"expected_max", "gamma_cvar"}
        and state.online_assignment is not None
        and epoch % reschedule_epochs != 0
    ):
        return orient_pair(state.online_assignment, state, exposure)
    if policy == "expected_max":
        return choose_service_assignment(
            state, exposure, config, objective="expected_max"
        ).assignment
    if policy == "gamma_cvar":
        return choose_service_assignment(
            state, exposure, config, objective="gamma_cvar"
        ).assignment
    raise ValueError(f"unknown policy: {policy}")


def simulate_case(task):
    (
        policy,
        seed,
        exposures,
        decision_exposure,
        health_uniforms,
        config,
        max_hours,
        rotation_epochs,
        reschedule_epochs,
    ) = task
    state = ServiceScheduleState(
        tuple(float(value) for value in INITIAL_DAMAGE_FRACTION * config.health_limit_pct)
    )
    crossed_h = max_hours
    crossed = False
    max_damage_history = []
    assignments = []
    for epoch, exposure in enumerate(exposures):
        assignment = choose_policy_assignment(
            policy,
            state,
            decision_exposure,
            config,
            epoch,
            rotation_epochs,
            reschedule_epochs,
        )
        transition = transition_service_epoch(
            state,
            exposure,
            config,
            assignment,
            stochastic=True,
            continuous_uniforms=health_uniforms[epoch],
        )
        state = transition.state
        assignments.append(assignment)
        max_damage_history.append(max(state.damage_pct))
        if max(state.damage_pct) >= config.health_limit_pct:
            crossed_h = state.elapsed_h
            crossed = True
            break
    assignment_changes = sum(
        set(current) != set(previous)
        for previous, current in zip(assignments, assignments[1:])
    )
    return {
        "policy": policy,
        "seed": seed,
        "crossed_health_limit": crossed,
        "time_to_health_limit_h": crossed_h,
        "final_elapsed_h": state.elapsed_h,
        "final_max_damage_pct": max(state.damage_pct),
        "final_damage_range_pct": max(state.damage_pct) - min(state.damage_pct),
        "start_count": state.start_count,
        "assignment_change_count": assignment_changes,
        **{
            f"stack_{index}_final_damage_pct": value
            for index, value in enumerate(state.damage_pct)
        },
    }


def summarize(per_run):
    rows = []
    for policy, group in per_run.groupby("policy"):
        values = group.time_to_health_limit_h.to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "n_runs": len(group),
                "health_limit_crossing_share": float(
                    group.crossed_health_limit.mean()
                ),
                "time_to_limit_mean_h": float(values.mean()),
                "time_to_limit_std_h": float(values.std(ddof=1)),
                "time_to_limit_q10_h": float(np.quantile(values, 0.10)),
                "time_to_limit_median_h": float(np.median(values)),
                "time_to_limit_q90_h": float(np.quantile(values, 0.90)),
                "start_count_mean": float(group.start_count.mean()),
                "assignment_change_mean": float(
                    group.assignment_change_count.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_vs_fixed(per_run):
    fixed = per_run[per_run.policy == "fixed_pair"].set_index("seed")
    rows = []
    for policy in POLICIES:
        if policy == "fixed_pair":
            continue
        selected = per_run[per_run.policy == policy].set_index("seed")
        delta = (
            selected.time_to_health_limit_h - fixed.time_to_health_limit_h
        ).to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "mean_gain_h": float(delta.mean()),
                "minimum_gain_h": float(delta.min()),
                "maximum_gain_h": float(delta.max()),
                "win_share": float((delta > 0).mean()),
                "nonworse_share": float((delta >= 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(per_run, summary):
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "fixed_pair": "#4C78A8",
        "periodic_rotation": "#F2A541",
        "expected_max": "#2A9D8F",
        "gamma_cvar": "#D65F5F",
    }
    labels = {
        "fixed_pair": "Fixed pair",
        "periodic_rotation": "Periodic rotation",
        "expected_max": "Expected max",
        "gamma_cvar": "Gamma-CVaR",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    maximum = int(per_run.time_to_health_limit_h.max())
    times = np.linspace(0, maximum, 200)
    for policy in POLICIES:
        values = per_run[per_run.policy == policy].time_to_health_limit_h.to_numpy()
        survival = np.asarray([(values > time).mean() for time in times])
        axes[0].step(
            times,
            survival,
            where="post",
            color=colors[policy],
            label=labels[policy],
        )
    axes[0].set_xlabel("Service exposure (h)")
    axes[0].set_ylabel("Health-boundary survival")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")

    ordered = summary.set_index("policy").loc[list(POLICIES)]
    positions = np.arange(len(POLICIES))
    lower = ordered.time_to_limit_median_h - ordered.time_to_limit_q10_h
    upper = ordered.time_to_limit_q90_h - ordered.time_to_limit_median_h
    for index, policy in enumerate(POLICIES):
        axes[1].errorbar(
            positions[index],
            ordered.time_to_limit_median_h.iloc[index],
            yerr=np.asarray(
                [[lower.iloc[index]], [upper.iloc[index]]], dtype=float
            ),
            fmt="o",
            color=colors[policy],
            ecolor=colors[policy],
            capsize=3,
        )
    axes[1].set_xticks(
        positions,
        [labels[policy].replace(" ", "\n") for policy in POLICIES],
    )
    axes[1].set_ylabel("Time to health boundary (h)")
    axes[1].text(-0.12, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8, pad=0.55)
    fig.savefig(
        FIGURES / "fig07_service_horizon_screen.png", dpi=320, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=float, default=2000.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--rotation-hours", type=float, default=24.0)
    parser.add_argument("--risk-horizon-h", type=float, default=100.0)
    parser.add_argument("--reschedule-hours", type=float, default=24.0)
    parser.add_argument("--risk-samples", type=int, default=128)
    parser.add_argument("--gamma-terminal-cv", type=float, default=0.10)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.max_hours <= 0 or args.epoch_h <= 0 or not args.seeds:
        raise ValueError("time settings and seeds must be positive")
    rotation_epochs = int(round(args.rotation_hours / args.epoch_h))
    reschedule_epochs = int(round(args.reschedule_hours / args.epoch_h))
    epochs = int(np.ceil(args.max_hours / args.epoch_h))
    if rotation_epochs <= 0 or reschedule_epochs <= 0:
        raise ValueError("scheduling periods must cover at least one epoch")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    health_limit = float(calibration["terminal_total_damage_pct"])
    gamma_scale = gamma_scale_for_terminal_cv(
        float(calibration["terminal_total_damage_pct"]),
        float(calibration["terminal_continuous_damage_pct"]),
        args.gamma_terminal_cv,
    )
    config = ServiceScheduleConfig(
        health_limit_pct=health_limit,
        gamma_scale_pct=gamma_scale,
        heterogeneity_factors=tuple(HETEROGENEITY),
        start_damage_pct=float(
            calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
        ),
        risk_horizon_h=args.risk_horizon_h,
        risk_samples=args.risk_samples,
    )
    templates = load_templates()
    decision_exposure = stationary_service_exposure(templates, args.epoch_h)
    tasks = []
    for seed in args.seeds:
        load_rng = np.random.default_rng(seed)
        health_rng = np.random.default_rng(100_000 + seed)
        exposures = tuple(
            aggregate_epoch(templates, load_rng, args.epoch_h) for _ in range(epochs)
        )
        health_uniforms = np.clip(
            health_rng.random((epochs, 2)), 1e-12, 1 - 1e-12
        )
        tasks.extend(
            (
                policy,
                seed,
                exposures,
                decision_exposure,
                health_uniforms,
                config,
                args.max_hours,
                rotation_epochs,
                reschedule_epochs,
            )
            for policy in POLICIES
        )
    per_run = pd.DataFrame(map(simulate_case, tasks))
    summary = summarize(per_run)
    paired = paired_vs_fixed(per_run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_fixed.csv", index=False)
    if not args.skip_plot:
        plot_results(per_run, summary)
    metadata = {
        "scope": "development screen using real-calibrated Markov exposure templates; not holdout validation",
        "policies": list(POLICIES),
        "seeds": args.seeds,
        "epoch_h": args.epoch_h,
        "max_hours": args.max_hours,
        "rotation_hours": args.rotation_hours,
        "reschedule_hours": args.reschedule_hours,
        "risk_horizon_h": args.risk_horizon_h,
        "risk_samples": args.risk_samples,
        "health_limit_pct": health_limit,
        "health_limit_interpretation": "LZW calibrated trajectory endpoint, not an identified failure threshold",
        "gamma_scale_pct": config.gamma_scale_pct,
        "gamma_terminal_cv": args.gamma_terminal_cv,
        "template_count": len(templates),
        "decision_information": "stationary mean of development templates; no upcoming epoch exposure",
        "health_pairing": "common inverse-CDF uniforms by seed, epoch and online role",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 小时级N+1调度开发筛查\n\n"
    report += (
        "本实验使用实车标定Markov开发场景的120秒Instant动作暴露作为块模板，"
        "决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层"
        "在线集合实际变化时计入。健康边界是LZW"
        "标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。\n\n"
    )
    report += summary.to_markdown(index=False) + "\n"
    report += "\n## 相对固定双堆的配对结果\n\n"
    report += paired.to_markdown(index=False) + "\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
