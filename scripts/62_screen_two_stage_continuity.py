"""Screen a two-stage RUL proxy on heterogeneity failure cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from fc_power.evaluation import ServiceScheduleConfig
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_SCRIPT = ROOT / "scripts/61_audit_n_plus_one_parameter_robustness.py"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_two_stage_continuity_screen"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
POLICIES = (
    "fixed_pair",
    "health_greedy",
    "order_blend_50",
    "expected_n_plus_one",
    "continuity_rul",
    "continuity_second_0",
    "continuity_second_24",
    "continuity_second_48",
    "continuity_second_100",
    "guarded_blend",
)
PROTECTED_POLICIES = (
    "fixed_pair",
    "order_blend_50",
    "guarded_blend",
    "protected_blend_50",
)
PLOT_POLICIES = (
    "health_greedy",
    "order_blend_50",
    "continuity_rul",
    "guarded_blend",
    "protected_blend_50",
)
TARGETS = (
    "reference",
    "heterogeneity_current_perm_3",
    "heterogeneity_gp_re_high_perm_1",
    "heterogeneity_gp_re_high_perm_3",
    "heterogeneity_gp_re_increased_perm_1",
    "heterogeneity_gp_re_increased_perm_3",
)
PROTECTED_TARGETS = TARGETS + (
    "boundary_scale_0.80",
    "boundary_scale_1.20",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260715


def load_robustness_module():
    spec = importlib.util.spec_from_file_location(
        "fc_n_plus_one_robustness", ROBUSTNESS_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ROBUSTNESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROBUST = load_robustness_module()
AUDIT = ROBUST.AUDIT


def run_target_seed(task):
    scenario, local_seed, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    for row in rows:
        row["scenario_id"] = scenario["scenario_id"]
        row["local_seed"] = local_seed
    for row in traces:
        row["scenario_id"] = scenario["scenario_id"]
        row["local_seed"] = local_seed
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


def summarize(per_run, policies=POLICIES, targets=TARGETS):
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(
        ["scenario_id", "local_seed"]
    )
    rows = []
    for index, policy in enumerate(policies[1:]):
        selected = per_run[per_run.policy.eq(policy)].set_index(
            ["scenario_id", "local_seed"]
        )
        for scenario_index, scenario_id in enumerate(targets):
            candidate = selected.loc[scenario_id]
            reference = fixed.loc[scenario_id]
            first = (
                candidate.time_to_first_boundary_h
                - reference.time_to_first_boundary_h
            )
            second = (
                candidate.time_to_second_boundary_h
                - reference.time_to_second_boundary_h
            )
            first_mean, first_low, first_high = mean_bca(
                first, BOOTSTRAP_SEED + 100 * index + 2 * scenario_index
            )
            second_mean, second_low, second_high = mean_bca(
                second, BOOTSTRAP_SEED + 100 * index + 2 * scenario_index + 1
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "policy": policy,
                    "first_boundary_gain_mean_h": first_mean,
                    "first_boundary_gain_ci95_low_h": first_low,
                    "first_boundary_gain_ci95_high_h": first_high,
                    "second_boundary_gain_mean_h": second_mean,
                    "second_boundary_gain_ci95_low_h": second_low,
                    "second_boundary_gain_ci95_high_h": second_high,
                    "second_boundary_wins": int((second > 0).sum()),
                    "second_boundary_ties": int((second == 0).sum()),
                    "second_boundary_losses": int((second < 0).sum()),
                    "second_boundary_nonworse_share": float((second >= 0).mean()),
                    "start_count_delta_mean": (
                        candidate.start_count - reference.start_count
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_results(summary, output_path, targets=TARGETS, focused_protected=False):
    labels = {
        "health_greedy": "Health-greedy",
        "order_blend_50": "Blend 0.50",
        "expected_n_plus_one": "Pure N+1",
        "continuity_rul": "Two-stage RUL",
        "continuity_second_0": "Continuity h=0",
        "continuity_second_24": "Continuity h=24",
        "continuity_second_48": "Continuity h=48",
        "continuity_second_100": "Continuity h=100",
        "guarded_blend": "Guarded Blend",
        "protected_blend_50": "Protected Blend",
    }
    colors = {
        "health_greedy": "#2A9D8F",
        "order_blend_50": "#5F6CAF",
        "expected_n_plus_one": "#0077B6",
        "continuity_rul": "#C44E52",
        "continuity_second_0": "#B56576",
        "continuity_second_24": "#A44A3F",
        "continuity_second_48": "#8C3B32",
        "continuity_second_100": "#6F2D28",
        "guarded_blend": "#D55E00",
        "protected_blend_50": "#0072B2",
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
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.6))
    for axis, scenario_id, title in (
        (axes[0], "reference", "Reference"),
        (
            axes[1],
            "heterogeneity_gp_re_increased_perm_3",
            "Worst heterogeneity",
        ),
    ):
        selected = summary[
            summary.scenario_id.eq(scenario_id)
            & summary.policy.isin(PLOT_POLICIES)
        ]
        for row in selected.itertuples(index=False):
            axis.scatter(
                row.second_boundary_gain_mean_h,
                row.first_boundary_gain_mean_h,
                s=25,
                color=colors[row.policy],
                label=labels[row.policy],
                zorder=3,
            )
        axis.axhline(0, color="#7A8288", lw=0.8, ls="--")
        axis.axvline(0, color="#7A8288", lw=0.8, ls="--")
        axis.set_xlabel("N+1 second-boundary gain (h)")
        axis.set_ylabel("First-boundary gain (h)")
        axis.text(0.03, 0.96, title, transform=axis.transAxes, va="top", fontsize=7)
    axes[0].legend(frameon=False, fontsize=5.8, loc="best")

    focus_policy = "protected_blend_50" if focused_protected else "guarded_blend"
    selected = (
        summary[summary.policy.eq(focus_policy)]
        .set_index("scenario_id")
        .loc[list(targets)]
    )
    y = np.arange(len(selected))
    mean = selected.second_boundary_gain_mean_h.to_numpy(dtype=float)
    lower = mean - selected.second_boundary_gain_ci95_low_h.to_numpy(dtype=float)
    upper = selected.second_boundary_gain_ci95_high_h.to_numpy(dtype=float) - mean
    axes[2].errorbar(
        mean,
        y,
        xerr=np.vstack((lower, upper)),
        fmt="o",
        ms=3.8,
        color="#1B263B",
        ecolor="#6C757D",
        capsize=2.0,
        lw=0.9,
    )
    axes[2].axvline(0, color="#7A8288", lw=0.8, ls="--")
    short_labels = {
        "reference": "Reference",
        "heterogeneity_current_perm_3": "Current perm. 3",
        "heterogeneity_gp_re_high_perm_1": "GP-RE high p. 1",
        "heterogeneity_gp_re_high_perm_3": "GP-RE high p. 3",
        "heterogeneity_gp_re_increased_perm_1": "GP-RE inc. p. 1",
        "heterogeneity_gp_re_increased_perm_3": "GP-RE inc. p. 3",
        "boundary_scale_0.80": "Boundary x0.8",
        "boundary_scale_1.20": "Boundary x1.2",
    }
    short = [short_labels[value] for value in targets]
    axes[2].set_yticks(y, short, fontsize=6.2)
    axes[2].set_xlabel(
        ("Protected Blend" if focused_protected else "Guarded Blend")
        + " N+1 gain (h)"
    )
    axes[2].invert_yaxis()

    for index, axis in enumerate(axes):
        axis.text(
            -0.14,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.65, w_pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=int, default=6000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--focused-protected", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    policies = PROTECTED_POLICIES if args.focused_protected else POLICIES
    targets = PROTECTED_TARGETS if args.focused_protected else TARGETS
    if min(args.seeds, args.jobs, args.epoch_h, args.max_hours) <= 0:
        raise ValueError("seeds, jobs and time settings must be positive")

    figure_name = (
        "fig25_protected_blend_screen.png"
        if args.focused_protected
        else "fig24_two_stage_continuity_screen.png"
    )
    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        summary = summarize(per_run, policies, targets)
        summary.to_csv(args.out_dir / "scenario_summary.csv", index=False)
        figure = args.out_dir / figure_name
        plot_results(summary, figure, targets, args.focused_protected)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        printed_policy = (
            "protected_blend_50" if args.focused_protected else "guarded_blend"
        )
        print(summary[summary.policy.eq(printed_policy)].to_string(index=False))
        return

    all_scenarios = {
        item["scenario_id"]: item for item in ROBUST.build_scenarios()
    }
    scenarios = [all_scenarios[value] for value in targets]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    base_limit = float(calibration["terminal_total_damage_pct"])
    base_continuous = float(calibration["terminal_continuous_damage_pct"])
    start_damage = float(
        calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
    )
    templates = AUDIT.load_real_templates(AUDIT.TEMPLATES)
    decision_exposure = AUDIT.stationary_service_exposure(templates, args.epoch_h)
    rotation_epochs = int(round(24.0 / args.epoch_h))
    tasks = []
    for scenario in scenarios:
        scale = scenario["boundary_scale"]
        config = ServiceScheduleConfig(
            health_limit_pct=base_limit * scale,
            gamma_scale_pct=gamma_scale_for_terminal_cv(
                base_limit * scale,
                base_continuous * scale,
                scenario["gamma_terminal_cv"],
            ),
            heterogeneity_factors=scenario["heterogeneity_factors"],
            start_damage_pct=start_damage,
            risk_horizon_h=scenario["risk_horizon_h"],
            risk_samples=512,
            n_plus_one_weight=0.50,
        )
        reschedule_epochs = int(round(scenario["reschedule_h"] / args.epoch_h))
        for local_seed in range(args.seeds):
            simulation_task = (
                local_seed,
                policies,
                templates,
                decision_exposure,
                config,
                scenario["initial_damage_fraction"],
                args.max_hours,
                args.epoch_h,
                rotation_epochs,
                reschedule_epochs,
                scenario["scenario_id"] == "reference" and local_seed == 0,
            )
            tasks.append((scenario, local_seed, simulation_task))

    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for target_rows, target_traces in executor.map(
            run_target_seed, tasks, chunksize=1
        ):
            rows.extend(target_rows)
            traces.extend(target_traces)
    per_run = pd.DataFrame(rows)
    if len(per_run) != len(targets) * args.seeds * len(policies):
        raise AssertionError("two-stage screen result matrix is incomplete")
    if not per_run.second_boundary_crossed.all():
        raise AssertionError("max_hours did not cover every second boundary")
    summary = summarize(per_run, policies, targets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_boundary_trajectories.csv", index=False
    )
    figure = args.out_dir / figure_name
    plot_results(summary, figure, targets, args.focused_protected)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "development screen of analytic two-stage RUL scheduling",
        "target_scenarios": list(targets),
        "seeds_per_scenario": args.seeds,
        "policies": list(policies),
        "future_demand_used": False,
        "continuity_projection": "stationary development exposure mean",
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printed_policy = (
        "protected_blend_50" if args.focused_protected else "guarded_blend"
    )
    print(summary[summary.policy.eq(printed_policy)].to_string(index=False))


if __name__ == "__main__":
    main()
