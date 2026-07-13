"""Audit first-stack versus N+1 two-stack service boundaries."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    choose_baseline_protected_assignment,
    eligible_service_assignments,
    evaluate_service_assignment,
    evaluate_service_continuity_assignment,
    orient_service_pair,
    select_guarded_blend_policy,
    stationary_service_exposure,
    transition_service_epoch,
)
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = (
    ROOT
    / "data/results/fc_only_service_templates_norm40/service_exposure_templates.csv"
)
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_n_plus_one_boundary"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
POLICIES = (
    "fixed_pair",
    "periodic_rotation",
    "health_greedy",
    "expected_max",
    "expected_total",
    "expected_n_plus_one",
)
BLEND_WEIGHTS = {
    "order_blend_25": 0.25,
    "order_blend_50": 0.50,
    "order_blend_75": 0.75,
    "order_blend_90": 0.90,
    "order_blend_99": 0.99,
}
CONTINUITY_THRESHOLDS_H = {
    "continuity_second_0": 0.0,
    "continuity_second_24": 24.0,
    "continuity_second_48": 48.0,
    "continuity_second_100": 100.0,
}
AVAILABLE_POLICIES = (
    POLICIES
    + tuple(BLEND_WEIGHTS)
    + (
        "continuity_rul",
        "guarded_blend",
        "rate_bounded_blend",
        "protected_blend_50",
    )
    + tuple(CONTINUITY_THRESHOLDS_H)
)
HETEROGENEITY = np.asarray((1.0, 1.05, 1.10))
INITIAL_DAMAGE_FRACTION = np.asarray((0.10, 0.40, 0.80))
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260714


def load_real_templates(path: Path) -> list[ServiceExposure]:
    table = pd.read_csv(path)
    table = table[table.template_source.eq("real_calibration_window")]
    if len(table) != 48:
        raise AssertionError("N+1 audit requires 48 frozen real templates")
    return [
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
        for row in table.itertuples(index=False)
    ]


def aggregate_epoch(templates, rng, epoch_h: float) -> ServiceExposure:
    base_duration = templates[0].duration_h
    if not all(np.isclose(item.duration_h, base_duration) for item in templates):
        raise ValueError("service templates must have equal duration")
    blocks = int(round(epoch_h / base_duration))
    if blocks <= 0 or not np.isclose(blocks * base_duration, epoch_h):
        raise ValueError("epoch_h must be an integer multiple of template duration")
    selected = rng.integers(0, len(templates), size=blocks)
    continuous = np.sum(
        [templates[index].continuous_mean_pct for index in selected], axis=0
    )
    shifts = np.sum(
        [templates[index].load_shift_damage_pct for index in selected], axis=0
    )
    starts = np.sum(
        [templates[index].operational_start_damage_pct for index in selected],
        axis=0,
    )
    return ServiceExposure(
        duration_h=epoch_h,
        continuous_mean_pct=tuple(float(value) for value in continuous),
        load_shift_damage_pct=tuple(float(value) for value in shifts),
        operational_start_damage_pct=tuple(float(value) for value in starts),
    )


def orient_pair(pair, state, exposure, config):
    return orient_service_pair(
        pair,
        state,
        exposure,
        config.heterogeneity_factors,
    )


def choose_assignment(
    policy,
    state,
    exposure,
    config,
    epoch,
    rotation_epochs,
    reschedule_epochs,
):
    eligible = eligible_service_assignments(state, config.health_limit_pct)
    if not eligible:
        raise RuntimeError("fewer than two stacks remain below the service boundary")
    eligible_sets = {frozenset(value) for value in eligible}
    current_set = (
        None
        if state.online_assignment is None
        else frozenset(state.online_assignment)
    )
    if (
        policy
        in {
            "health_greedy",
            "expected_max",
            "expected_total",
            "expected_n_plus_one",
            "continuity_rul",
            "protected_blend_50",
            *BLEND_WEIGHTS,
            *CONTINUITY_THRESHOLDS_H,
        }
        and current_set in eligible_sets
        and epoch % reschedule_epochs != 0
    ):
        if policy == "continuity_rul" or policy in CONTINUITY_THRESHOLDS_H:
            return state.online_assignment
        return orient_pair(state.online_assignment, state, exposure, config)

    alive = sorted({index for assignment in eligible for index in assignment})
    if policy == "fixed_pair":
        preferred = (0, 1)
        pair = preferred if frozenset(preferred) in eligible_sets else tuple(alive[:2])
        return orient_pair(pair, state, exposure, config)
    if policy == "periodic_rotation":
        cycle = ((0, 1), (1, 2), (2, 0))
        start = (epoch // rotation_epochs) % len(cycle)
        pair = next(
            value
            for offset in range(len(cycle))
            if frozenset(value := cycle[(start + offset) % len(cycle)])
            in eligible_sets
        )
        return orient_pair(pair, state, exposure, config)
    if policy == "health_greedy":
        pair = tuple(
            sorted(alive, key=lambda index: (state.damage_pct[index], index))[:2]
        )
        return orient_pair(pair, state, exposure, config)
    if policy == "protected_blend_50":
        preferred = (0, 1)
        baseline_pair = (
            preferred if frozenset(preferred) in eligible_sets else tuple(alive[:2])
        )
        baseline_assignment = orient_pair(
            baseline_pair,
            state,
            exposure,
            config,
        )
        return choose_baseline_protected_assignment(
            state,
            exposure,
            replace(config, n_plus_one_weight=0.50),
            baseline_assignment,
            assignments=eligible,
        ).assignment
    if policy == "continuity_rul":
        decisions = [
            evaluate_service_continuity_assignment(
                state,
                exposure,
                config,
                assignment,
                first_boundary_weight=0.5,
            )
            for assignment in eligible
        ]
        return min(
            decisions,
            key=lambda item: (
                -item.objective_h,
                -item.expected_second_boundary_h,
                -item.expected_first_boundary_h,
                item.new_starts,
                item.assignment,
            ),
        ).assignment
    if policy in CONTINUITY_THRESHOLDS_H:
        decisions = [
            evaluate_service_continuity_assignment(
                state,
                exposure,
                config,
                assignment,
                first_boundary_weight=0.0,
            )
            for assignment in eligible
        ]
        best = min(
            decisions,
            key=lambda item: (
                -item.expected_second_boundary_h,
                item.new_starts,
                item.assignment,
            ),
        )
        current = next(
            (
                item
                for item in decisions
                if item.assignment == state.online_assignment
            ),
            None,
        )
        threshold = CONTINUITY_THRESHOLDS_H[policy]
        if (
            current is not None
            and best.expected_second_boundary_h
            <= current.expected_second_boundary_h + threshold
        ):
            return current.assignment
        return best.assignment
    if policy in {
        "expected_max",
        "expected_total",
        "expected_n_plus_one",
        *BLEND_WEIGHTS,
    }:
        objective = (
            "expected_order_blend" if policy in BLEND_WEIGHTS else policy
        )
        decision_config = (
            replace(config, n_plus_one_weight=BLEND_WEIGHTS[policy])
            if policy in BLEND_WEIGHTS
            else config
        )
        decisions = [
            evaluate_service_assignment(
                state,
                exposure,
                decision_config,
                assignment,
                objective=objective,
            )
            for assignment in eligible
        ]
        return min(
            decisions,
            key=lambda item: (item.objective, item.new_starts, item.assignment),
        ).assignment
    raise ValueError(f"unknown policy: {policy}")


def simulate_case(task):
    (
        policy,
        seed,
        exposures,
        decision_exposure,
        uniforms,
        config,
        initial_damage_fraction,
        max_hours,
        rotation_epochs,
        reschedule_epochs,
        capture_trace,
    ) = task
    initial_fraction = np.asarray(initial_damage_fraction, dtype=float)
    if initial_fraction.shape != (len(config.heterogeneity_factors),):
        raise ValueError("initial_damage_fraction must match the stack count")
    if np.any(~np.isfinite(initial_fraction)) or np.any(
        (initial_fraction < 0) | (initial_fraction >= 1)
    ):
        raise ValueError("initial damage fractions must lie in [0, 1)")
    initial = initial_fraction * config.health_limit_pct
    effective_policy = policy
    if policy in {"guarded_blend", "rate_bounded_blend"}:
        effective_policy = select_guarded_blend_policy(
            initial,
            config.heterogeneity_factors,
            max_rate_ratio=(
                float(HETEROGENEITY.max() / HETEROGENEITY.min())
                if policy == "rate_bounded_blend"
                else None
            ),
        )
    state = ServiceScheduleState(tuple(float(value) for value in initial))
    crossing_times = np.full(len(initial), np.nan)
    assignments = []
    trace_rows = []
    first_total = np.nan
    first_range = np.nan
    for epoch, exposure in enumerate(exposures):
        assignment = choose_assignment(
            effective_policy,
            state,
            decision_exposure,
            config,
            epoch,
            rotation_epochs,
            reschedule_epochs,
        )
        if any(
            state.damage_pct[index] >= config.health_limit_pct
            for index in assignment
        ):
            raise AssertionError("a stack beyond the boundary was scheduled")
        before = np.asarray(state.damage_pct)
        transition = transition_service_epoch(
            state,
            exposure,
            config,
            assignment,
            stochastic=True,
            continuous_uniforms=uniforms[epoch],
        )
        state = transition.state
        after = np.asarray(state.damage_pct)
        assignments.append(assignment)
        newly_crossed = (
            (before < config.health_limit_pct)
            & (after >= config.health_limit_pct)
            & np.isnan(crossing_times)
        )
        crossing_times[newly_crossed] = state.elapsed_h
        crossed_count = int(np.isfinite(crossing_times).sum())
        if crossed_count >= 1 and not np.isfinite(first_total):
            first_total = float(after.sum())
            first_range = float(after.max() - after.min())
        if capture_trace:
            trace_rows.append(
                {
                    "policy": policy,
                    "effective_policy": effective_policy,
                    "seed": seed,
                    "elapsed_h": state.elapsed_h,
                    "assignment": str(assignment),
                    "crossed_count": crossed_count,
                    **{
                        f"stack_{index}_damage_pct": value
                        for index, value in enumerate(after)
                    },
                }
            )
        if crossed_count >= 2:
            break

    finite_crossings = np.sort(crossing_times[np.isfinite(crossing_times)])
    first_crossed = len(finite_crossings) >= 1
    second_crossed = len(finite_crossings) >= 2
    first_h = float(finite_crossings[0]) if first_crossed else float(max_hours)
    second_h = float(finite_crossings[1]) if second_crossed else float(max_hours)
    changes = sum(
        set(current) != set(previous)
        for previous, current in zip(assignments, assignments[1:])
    )
    row = {
        "policy": policy,
        "effective_policy": effective_policy,
        "seed": seed,
        "first_boundary_crossed": first_crossed,
        "second_boundary_crossed": second_crossed,
        "time_to_first_boundary_h": first_h,
        "time_to_second_boundary_h": second_h,
        "post_first_n_plus_one_reserve_h": second_h - first_h,
        "first_boundary_total_damage_pct": first_total,
        "first_boundary_damage_range_pct": first_range,
        "final_elapsed_h": state.elapsed_h,
        "final_total_damage_pct": float(sum(state.damage_pct)),
        "final_damage_range_pct": float(max(state.damage_pct) - min(state.damage_pct)),
        "start_count": state.start_count,
        "assignment_change_count": changes,
        "crossing_order": str(tuple(np.argsort(np.nan_to_num(crossing_times, nan=np.inf)))),
        **{
            f"stack_{index}_crossing_h": crossing_times[index]
            for index in range(len(crossing_times))
        },
        **{
            f"stack_{index}_final_damage_pct": value
            for index, value in enumerate(state.damage_pct)
        },
    }
    return row, trace_rows


def simulate_seed(task):
    (
        seed,
        policies,
        templates,
        decision_exposure,
        config,
        initial_damage_fraction,
        max_hours,
        epoch_h,
        rotation_epochs,
        reschedule_epochs,
        capture_trace,
    ) = task
    epochs = int(np.ceil(max_hours / epoch_h))
    load_rng = np.random.default_rng(seed)
    health_rng = np.random.default_rng(100_000 + seed)
    exposures = tuple(
        aggregate_epoch(templates, load_rng, epoch_h) for _ in range(epochs)
    )
    uniforms = np.clip(health_rng.random((epochs, 2)), 1e-12, 1 - 1e-12)
    rows, traces = [], []
    for policy in policies:
        row, trace = simulate_case(
            (
                policy,
                seed,
                exposures,
                decision_exposure,
                uniforms,
                config,
                initial_damage_fraction,
                max_hours,
                rotation_epochs,
                reschedule_epochs,
                capture_trace,
            )
        )
        rows.append(row)
        traces.extend(trace)
    return rows, traces


def mean_bca(values, seed):
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0]):
        value = float(values[0])
        return value, value, value
    result = bootstrap(
        (values,),
        np.mean,
        method="BCa",
        n_resamples=BOOTSTRAP_SAMPLES,
        batch=2000,
        rng=np.random.default_rng(seed),
    )
    return (
        float(values.mean()),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def summarize(per_run):
    rows = []
    for policy, group in per_run.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "runs": len(group),
                "first_boundary_mean_h": group.time_to_first_boundary_h.mean(),
                "second_boundary_mean_h": group.time_to_second_boundary_h.mean(),
                "second_boundary_q10_h": group.time_to_second_boundary_h.quantile(0.10),
                "second_boundary_median_h": group.time_to_second_boundary_h.median(),
                "second_boundary_q90_h": group.time_to_second_boundary_h.quantile(0.90),
                "post_first_reserve_mean_h": group.post_first_n_plus_one_reserve_h.mean(),
                "first_boundary_total_damage_mean_pct": group.first_boundary_total_damage_pct.mean(),
                "start_count_mean": group.start_count.mean(),
                "assignment_change_mean": group.assignment_change_count.mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(per_run):
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index("seed")
    rows = []
    for index, policy in enumerate(POLICIES[1:]):
        selected = per_run[per_run.policy.eq(policy)].set_index("seed")
        first = selected.time_to_first_boundary_h - fixed.time_to_first_boundary_h
        second = selected.time_to_second_boundary_h - fixed.time_to_second_boundary_h
        total = (
            selected.first_boundary_total_damage_pct
            - fixed.first_boundary_total_damage_pct
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + index
        )
        wins = int((second > 0).sum())
        losses = int((second < 0).sum())
        informative = wins + losses
        sign_p = (
            float(binomtest(wins, informative, 0.5, alternative="greater").pvalue)
            if informative
            else 1.0
        )
        rows.append(
            {
                "policy": policy,
                "reference": "fixed_pair",
                "first_boundary_gain_mean_h": first.mean(),
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_gain_median_h": second.median(),
                "second_boundary_wins": wins,
                "second_boundary_ties": int((second == 0).sum()),
                "second_boundary_losses": losses,
                "second_boundary_sign_p_one_sided": sign_p,
                "second_boundary_win_share": (second > 0).mean(),
                "second_boundary_nonworse_share": (second >= 0).mean(),
                "first_boundary_total_damage_delta_mean_pct": total.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_results(per_run, summary, paired, output_path):
    colors = {
        "fixed_pair": "#33658A",
        "periodic_rotation": "#F6AE2D",
        "health_greedy": "#2A9D8F",
        "expected_max": "#D1495B",
        "expected_total": "#8C6BB1",
        "expected_n_plus_one": "#0077B6",
        "order_blend_25": "#7A5195",
        "order_blend_50": "#5F6CAF",
        "order_blend_75": "#2E86AB",
        "order_blend_90": "#008F95",
        "order_blend_99": "#00A676",
        "continuity_rul": "#C44E52",
        "continuity_second_0": "#B56576",
        "continuity_second_24": "#A44A3F",
        "continuity_second_48": "#8C3B32",
        "continuity_second_100": "#6F2D28",
        "guarded_blend": "#D55E00",
        "protected_blend_50": "#0072B2",
    }
    labels = {
        "fixed_pair": "Fixed pair",
        "periodic_rotation": "Periodic rotation",
        "health_greedy": "Health-greedy",
        "expected_max": "Expected-max",
        "expected_total": "Expected-total",
        "expected_n_plus_one": "N+1 objective",
        "order_blend_25": "Blend 0.25",
        "order_blend_50": "Blend 0.50",
        "order_blend_75": "Blend 0.75",
        "order_blend_90": "Blend 0.90",
        "order_blend_99": "Blend 0.99",
        "continuity_rul": "Two-stage RUL",
        "continuity_second_0": "Continuity h=0",
        "continuity_second_24": "Continuity h=24",
        "continuity_second_48": "Continuity h=48",
        "continuity_second_100": "Continuity h=100",
        "guarded_blend": "Guarded Blend",
        "protected_blend_50": "Protected Blend",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))

    maximum = int(per_run.time_to_second_boundary_h.max())
    x = np.arange(maximum + 1)
    for policy in POLICIES:
        group = per_run[per_run.policy.eq(policy)]
        first_survival = np.asarray(
            [(group.time_to_first_boundary_h > hour).mean() for hour in x]
        )
        second_survival = np.asarray(
            [(group.time_to_second_boundary_h > hour).mean() for hour in x]
        )
        axes[0].step(
            x,
            second_survival,
            where="post",
            color=colors[policy],
            lw=1.2,
            label=labels[policy],
        )
        axes[0].step(
            x,
            first_survival,
            where="post",
            color=colors[policy],
            lw=0.65,
            ls="--",
            alpha=0.65,
        )
    axes[0].set_xlabel("Operating exposure (h)")
    axes[0].set_ylabel("Boundary survival probability")
    axes[0].legend(frameon=False, fontsize=6.3, handlelength=1.6)
    axes[0].text(
        0.98,
        0.05,
        "solid: second stack\ndashed: first stack",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color="#5F6368",
    )

    positions = np.arange(len(POLICIES))
    ordered = summary.set_index("policy").loc[list(POLICIES)]
    axes[1].plot(
        positions,
        ordered.first_boundary_mean_h,
        "o--",
        color="#6C757D",
        ms=3.5,
        lw=0.8,
        label="First stack",
    )
    axes[1].plot(
        positions,
        ordered.second_boundary_mean_h,
        "s-",
        color="#1B263B",
        ms=3.5,
        lw=1.0,
        label="Second stack (N+1)",
    )
    axes[1].set_xticks(
        positions,
        [labels[value] for value in POLICIES],
        rotation=32,
        ha="right",
    )
    axes[1].set_ylabel("Mean time to boundary (h)")
    axes[1].legend(frameon=False, fontsize=6.3)

    plot_paired = paired.set_index("policy").loc[list(POLICIES[1:])]
    y = np.arange(len(plot_paired))
    mean = plot_paired.second_boundary_gain_mean_h.to_numpy()
    lower = mean - plot_paired.second_boundary_gain_ci95_low_h.to_numpy()
    upper = plot_paired.second_boundary_gain_ci95_high_h.to_numpy() - mean
    axes[2].errorbar(
        mean,
        y,
        xerr=np.vstack((lower, upper)),
        fmt="o",
        color="#1B263B",
        ecolor="#6C757D",
        capsize=2.2,
        ms=4,
        lw=0.9,
    )
    axes[2].axvline(0, color="#6C757D", lw=0.8, ls="--")
    axes[2].set_yticks(y, [labels[value] for value in plot_paired.index])
    axes[2].set_xlabel("Paired N+1 boundary gain (h)")
    axes[2].invert_yaxis()

    for index, axis in enumerate(axes):
        axis.text(
            -0.14,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.65, w_pad=1.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(out_dir, health_limit, per_run, summary, paired):
    report = f"""# N+1 first/second calibration-boundary audit

- Evaluated {per_run.seed.nunique()} paired health seeds without future demand.
- When one stack reaches {health_limit:.6f}%, it is removed from service and the remaining two continue until a second stack reaches the same boundary.
- This boundary is the endpoint of the LZW calibration trajectory, not a physical failure threshold or a claimed system lifetime.

## Summary

{summary.to_markdown(index=False)}

## Paired against fixed pair

{paired.to_markdown(index=False)}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    global POLICIES
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=AVAILABLE_POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=int, default=6000)
    parser.add_argument("--rotation-hours", type=float, default=24.0)
    parser.add_argument("--reschedule-hours", type=float, default=24.0)
    parser.add_argument("--gamma-terminal-cv", type=float, default=0.10)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    POLICIES = tuple(args.policies)
    if not POLICIES or POLICIES[0] != "fixed_pair":
        raise ValueError("policies must start with fixed_pair for paired reporting")
    if not args.seeds or min(args.epoch_h, args.max_hours, args.jobs) <= 0:
        raise ValueError("seeds and time settings must be positive")
    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        POLICIES = tuple(per_run.policy.drop_duplicates())
        summary = summarize(per_run)
        paired = paired_summary(per_run)
        calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        health_limit = float(calibration["terminal_total_damage_pct"])
        summary.to_csv(args.out_dir / "summary.csv", index=False)
        paired.to_csv(args.out_dir / "paired_vs_fixed.csv", index=False)
        figure = args.out_dir / "fig21_n_plus_one_service_boundary.png"
        plot_results(per_run, summary, paired, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        write_report(args.out_dir, health_limit, per_run, summary, paired)
        print(summary.to_string(index=False))
        print(paired.to_string(index=False))
        return
    rotation_epochs = int(round(args.rotation_hours / args.epoch_h))
    reschedule_epochs = int(round(args.reschedule_hours / args.epoch_h))
    if rotation_epochs <= 0 or reschedule_epochs <= 0:
        raise ValueError("scheduling periods must cover at least one epoch")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    health_limit = float(calibration["terminal_total_damage_pct"])
    gamma_scale = gamma_scale_for_terminal_cv(
        health_limit,
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
        risk_horizon_h=100.0,
        risk_samples=512,
    )
    templates = load_real_templates(TEMPLATES)
    decision_exposure = stationary_service_exposure(templates, args.epoch_h)
    tasks = []
    for seed in args.seeds:
        tasks.append(
            (
                seed,
                POLICIES,
                templates,
                decision_exposure,
                config,
                tuple(float(value) for value in INITIAL_DAMAGE_FRACTION),
                args.max_hours,
                args.epoch_h,
                rotation_epochs,
                reschedule_epochs,
                seed == args.seeds[0],
            )
        )

    rows, traces = [], []
    if args.jobs == 1:
        results = map(simulate_seed, tasks)
        for seed_rows, seed_traces in results:
            rows.extend(seed_rows)
            traces.extend(seed_traces)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for seed_rows, seed_traces in executor.map(
                simulate_seed, tasks, chunksize=1
            ):
                rows.extend(seed_rows)
                traces.extend(seed_traces)
    per_run = pd.DataFrame(rows)
    if not per_run.second_boundary_crossed.all():
        raise AssertionError("max_hours did not cover every second-stack boundary")
    summary = summarize(per_run)
    paired = paired_summary(per_run)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_boundary_trajectories.csv", index=False
    )
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_fixed.csv", index=False)
    figure = args.out_dir / "fig21_n_plus_one_service_boundary.png"
    plot_results(per_run, summary, paired, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_results(
        per_run,
        summary,
        paired,
        FIGURES / "fig21_n_plus_one_service_boundary.png",
    )

    metadata = {
        "scope": "development-only N+1 service-boundary audit",
        "health_boundary_pct": health_limit,
        "health_boundary_interpretation": (
            "LZW calibration trajectory endpoint, not a physical failure threshold"
        ),
        "n_plus_one_boundary": (
            "second stack reaches the declared calibration boundary; one crossed "
            "stack is removed and the remaining two continue"
        ),
        "system_failure_claimed": False,
        "policies": list(POLICIES),
        "seeds": args.seeds,
        "epoch_h": args.epoch_h,
        "max_hours": args.max_hours,
        "rotation_hours": args.rotation_hours,
        "reschedule_hours": args.reschedule_hours,
        "parallel_jobs": args.jobs,
        "template_input": str(TEMPLATES.relative_to(ROOT)),
        "template_source": "real_calibration_window",
        "template_count": len(templates),
        "gamma_terminal_cv": args.gamma_terminal_cv,
        "gamma_scale_pct": gamma_scale,
        "decision_information": (
            "stationary mean of development templates; no future epoch exposure"
        ),
        "pairing": "same sampled exposure and inverse-CDF uniforms by seed and epoch",
        "literature_boundary": (
            "Zuo 2024 states that one failed stack leaves two operating stacks and "
            "system failure occurs when two stacks fail"
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.out_dir, health_limit, per_run, summary, paired)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
