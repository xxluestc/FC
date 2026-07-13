"""Audit frozen N+1 scheduling across health and heterogeneity assumptions."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap, gamma

from fc_power.evaluation import ServiceScheduleConfig
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts/58_audit_n_plus_one_service_boundary.py"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_n_plus_one_parameter_robustness"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
SIMULATED_POLICIES = (
    "fixed_pair",
    "health_greedy",
    "order_blend_50",
)
POLICIES = SIMULATED_POLICIES + ("guarded_blend",)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260715


def load_audit_module():
    spec = importlib.util.spec_from_file_location("fc_n_plus_one_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()


def normalized_gamma_quantiles(delta):
    values = gamma.ppf((0.10, 0.50, 0.90), a=delta, scale=1.0 / delta)
    values /= values.mean()
    return tuple(float(value) for value in values)


def build_scenarios():
    baseline_initial = (0.10, 0.40, 0.80)
    baseline_factors = (1.00, 1.05, 1.10)
    scenarios = []

    def add(
        scenario_id,
        category,
        initial=baseline_initial,
        factors=baseline_factors,
        boundary_scale=1.0,
        gamma_cv=0.10,
        risk_horizon_h=100.0,
        reschedule_h=24.0,
        evidence="predeclared project setting",
    ):
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "category": category,
                "initial_damage_fraction": tuple(float(value) for value in initial),
                "heterogeneity_factors": tuple(float(value) for value in factors),
                "boundary_scale": float(boundary_scale),
                "gamma_terminal_cv": float(gamma_cv),
                "risk_horizon_h": float(risk_horizon_h),
                "reschedule_h": float(reschedule_h),
                "evidence": evidence,
            }
        )

    add("reference", "reference")

    for index, values in enumerate(
        sorted(set(itertools.permutations(baseline_initial)))
    ):
        if values != baseline_initial:
            add(
                f"initial_stagger_perm_{index}",
                "initial_health",
                initial=values,
                evidence="identity permutation of the predeclared stagger",
            )
    for aged_stack in range(3):
        initial = [0.0, 0.0, 0.0]
        initial[aged_stack] = 0.01 / (0.278 - 0.1803)
        add(
            f"initial_zuo_one_aged_stack_{aged_stack}",
            "initial_health",
            initial=initial,
            evidence="Zuo 2024 Eq. 21 and Table 1, normalized to fresh-to-FT margin",
        )
    add(
        "initial_equal_fresh",
        "initial_health",
        initial=(0.0, 0.0, 0.0),
        evidence="symmetry control",
    )
    add(
        "initial_equal_midlife",
        "initial_health",
        initial=(0.40, 0.40, 0.40),
        evidence="symmetry control at nonzero damage",
    )

    for index, values in enumerate(
        sorted(set(itertools.permutations(baseline_factors)))
    ):
        if values != baseline_factors:
            add(
                f"heterogeneity_current_perm_{index}",
                "heterogeneity",
                factors=values,
                evidence="identity permutation of predeclared deterministic factors",
            )
    add(
        "heterogeneity_identical",
        "heterogeneity",
        factors=(1.0, 1.0, 1.0),
        evidence="no-heterogeneity control",
    )
    random_effects = {
        "low": 43.8 * 5,
        "base": 43.8,
        "high": 43.8 / 5,
        "increased": 43.8 / 10,
    }
    for level, delta in random_effects.items():
        quantiles = normalized_gamma_quantiles(delta)
        permutations = sorted(set(itertools.permutations(quantiles)))
        if level in {"low", "base"}:
            permutations = [permutations[0], permutations[-1]]
        for index, values in enumerate(permutations):
            add(
                f"heterogeneity_gp_re_{level}_perm_{index}",
                "heterogeneity",
                factors=values,
                evidence=(
                    "Zuo 2025 GP-RE delta/phi sensitivity; normalized "
                    "10/50/90 percentiles, not fitted vehicle factors"
                ),
            )

    for scale in (0.80, 1.20):
        add(
            f"boundary_scale_{scale:.2f}",
            "boundary",
            boundary_scale=scale,
            evidence="predeclared +/-20% calibration-boundary sensitivity",
        )
    for cv in (0.05, 0.20):
        add(
            f"gamma_cv_{cv:.2f}",
            "stochasticity",
            gamma_cv=cv,
            evidence="predeclared aggregate Gamma CV sensitivity",
        )
    for horizon in (24.0, 240.0):
        add(
            f"risk_horizon_{int(horizon)}h",
            "scheduler",
            risk_horizon_h=horizon,
            evidence="one-day and ten-day projection sensitivity",
        )
    for period in (12.0, 48.0):
        add(
            f"reschedule_{int(period)}h",
            "scheduler",
            reschedule_h=period,
            evidence="half-day and two-day rescheduling sensitivity",
        )

    identifiers = [item["scenario_id"] for item in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("scenario identifiers must be unique")
    return scenarios


def run_scenario_seed(task):
    scenario, local_seed, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    descriptor = {
        key: value
        for key, value in scenario.items()
        if key not in {"initial_damage_fraction", "heterogeneity_factors"}
    }
    for row in rows:
        row.update(descriptor)
        row["local_seed"] = local_seed
        row["initial_damage_fraction"] = str(scenario["initial_damage_fraction"])
        row["heterogeneity_factors"] = str(scenario["heterogeneity_factors"])
    for row in traces:
        row.update(descriptor)
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


def pair_runs(per_run):
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(
        ["scenario_id", "local_seed"]
    )
    rows = []
    for policy in POLICIES[1:]:
        selected = per_run[per_run.policy.eq(policy)].set_index(
            ["scenario_id", "local_seed"]
        )
        for key, row in selected.iterrows():
            reference = fixed.loc[key]
            rows.append(
                {
                    "scenario_id": key[0],
                    "local_seed": key[1],
                    "category": row.category,
                    "policy": policy,
                    "first_boundary_gain_h": (
                        row.time_to_first_boundary_h
                        - reference.time_to_first_boundary_h
                    ),
                    "second_boundary_gain_h": (
                        row.time_to_second_boundary_h
                        - reference.time_to_second_boundary_h
                    ),
                    "start_count_delta": row.start_count - reference.start_count,
                    "assignment_change_delta": (
                        row.assignment_change_count
                        - reference.assignment_change_count
                    ),
                }
            )
    return pd.DataFrame(rows)


def materialize_guarded_blend(per_run, scenarios):
    rows = []
    for scenario in scenarios:
        initial = np.asarray(scenario["initial_damage_fraction"], dtype=float)
        factors = np.asarray(scenario["heterogeneity_factors"], dtype=float)
        separated = not np.allclose(initial, initial[0])
        aligned = int(np.argmax(initial)) == int(np.argmax(factors))
        source_policy = "order_blend_50" if separated and aligned else "fixed_pair"
        selected = per_run[
            per_run.scenario_id.eq(scenario["scenario_id"])
            & per_run.policy.eq(source_policy)
        ].copy()
        selected["policy"] = "guarded_blend"
        selected["effective_policy"] = source_policy
        rows.append(selected)
    return pd.concat([per_run, *rows], ignore_index=True)


def summarize_scenarios(paired):
    rows = []
    for index, ((scenario_id, policy), group) in enumerate(
        paired.groupby(["scenario_id", "policy"], sort=False)
    ):
        first = group.first_boundary_gain_h.to_numpy(dtype=float)
        second = group.second_boundary_gain_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 2 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 2 * index + 1
        )
        wins = int((second > 0).sum())
        losses = int((second < 0).sum())
        informative = wins + losses
        rows.append(
            {
                "scenario_id": scenario_id,
                "category": group.category.iloc[0],
                "policy": policy,
                "seeds": len(group),
                "first_boundary_gain_mean_h": first_mean,
                "first_boundary_gain_ci95_low_h": first_low,
                "first_boundary_gain_ci95_high_h": first_high,
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_gain_median_h": float(np.median(second)),
                "second_boundary_wins": wins,
                "second_boundary_ties": int((second == 0).sum()),
                "second_boundary_losses": losses,
                "second_boundary_sign_p_one_sided": (
                    float(
                        binomtest(
                            wins,
                            informative,
                            0.5,
                            alternative="greater",
                        ).pvalue
                    )
                    if informative
                    else 1.0
                ),
                "second_boundary_nonworse_share": float((second >= 0).mean()),
                "start_count_delta_mean": group.start_count_delta.mean(),
                "assignment_change_delta_mean": group.assignment_change_delta.mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_robustness(summary):
    rows = []
    for policy, group in summary.groupby("policy", sort=False):
        worst = group.loc[group.second_boundary_gain_mean_h.idxmin()]
        rows.append(
            {
                "policy": policy,
                "scenarios": len(group),
                "positive_first_mean_scenarios": int(
                    (group.first_boundary_gain_mean_h > 0).sum()
                ),
                "nonnegative_second_mean_scenarios": int(
                    (group.second_boundary_gain_mean_h >= 0).sum()
                ),
                "positive_second_ci_scenarios": int(
                    (group.second_boundary_gain_ci95_low_h > 0).sum()
                ),
                "negative_second_ci_scenarios": int(
                    (group.second_boundary_gain_ci95_high_h < 0).sum()
                ),
                "second_gain_mean_across_scenarios_h": (
                    group.second_boundary_gain_mean_h.mean()
                ),
                "second_gain_min_scenario_h": worst.second_boundary_gain_mean_h,
                "second_gain_min_scenario_id": worst.scenario_id,
                "minimum_seed_nonworse_share": (
                    group.second_boundary_nonworse_share.min()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_results(summary, output_path):
    policy_style = {
        "health_greedy": ("#2A9D8F", "o", "Health-greedy"),
        "order_blend_50": ("#5F6CAF", "s", "Blend 0.50"),
        "guarded_blend": ("#D55E00", "D", "Guarded Blend"),
    }
    category_markers = {
        "reference": "o",
        "initial_health": "^",
        "heterogeneity": "s",
        "boundary": "D",
        "stochasticity": "P",
        "scheduler": "X",
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
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75))

    for policy, (color, _, label) in policy_style.items():
        selected = summary[summary.policy.eq(policy)]
        for category, group in selected.groupby("category", sort=False):
            axes[0].scatter(
                group.second_boundary_gain_mean_h,
                group.first_boundary_gain_mean_h,
                s=18,
                marker=category_markers[category],
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )
        axes[0].scatter([], [], s=20, color=color, label=label)
    axes[0].axhline(0, color="#7A8288", lw=0.8, ls="--")
    axes[0].axvline(0, color="#7A8288", lw=0.8, ls="--")
    axes[0].set_xlabel("N+1 second-boundary gain (h)")
    axes[0].set_ylabel("First-boundary gain (h)")
    axes[0].legend(frameon=False, fontsize=6.4)

    blend = summary[summary.policy.eq("order_blend_50")]
    categories = list(
        dict.fromkeys(
            [
                "reference",
                "initial_health",
                "heterogeneity",
                "boundary",
                "stochasticity",
                "scheduler",
            ]
        )
    )
    arrays = [
        blend.loc[blend.category.eq(category), "second_boundary_gain_mean_h"]
        .to_numpy(dtype=float)
        for category in categories
    ]
    axes[1].boxplot(
        arrays,
        positions=np.arange(len(categories)),
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#D9DDEF", "edgecolor": "#5F6CAF", "linewidth": 0.8},
        medianprops={"color": "#1B263B", "linewidth": 1.0},
        whiskerprops={"color": "#6C757D", "linewidth": 0.8},
        capprops={"color": "#6C757D", "linewidth": 0.8},
    )
    for position, values in enumerate(arrays):
        jitter = np.linspace(-0.12, 0.12, max(1, len(values)))
        axes[1].scatter(
            position + jitter,
            values,
            s=9,
            color="#5F6CAF",
            alpha=0.65,
            edgecolor="none",
        )
    axes[1].axhline(0, color="#7A8288", lw=0.8, ls="--")
    axes[1].set_xticks(
        np.arange(len(categories)),
        ["Ref.", "Initial", "Hetero.", "Boundary", "Gamma", "Schedule"],
        rotation=32,
        ha="right",
    )
    axes[1].set_ylabel("Blend 0.50 N+1 gain (h)")

    worst = blend.nsmallest(8, "second_boundary_gain_mean_h").sort_values(
        "second_boundary_gain_mean_h"
    )
    y = np.arange(len(worst))
    mean = worst.second_boundary_gain_mean_h.to_numpy(dtype=float)
    lower = mean - worst.second_boundary_gain_ci95_low_h.to_numpy(dtype=float)
    upper = worst.second_boundary_gain_ci95_high_h.to_numpy(dtype=float) - mean
    axes[2].errorbar(
        mean,
        y,
        xerr=np.vstack((lower, upper)),
        fmt="s",
        ms=3.5,
        color="#1B263B",
        ecolor="#6C757D",
        capsize=2.0,
        lw=0.9,
    )
    axes[2].axvline(0, color="#7A8288", lw=0.8, ls="--")
    def compact_scenario_label(value):
        replacements = (
            ("heterogeneity_gp_re_increased_perm_", "GP-RE increased p."),
            ("heterogeneity_gp_re_high_perm_", "GP-RE high p."),
            ("heterogeneity_current_perm_", "Current factors p."),
            ("initial_stagger_perm_", "Initial stagger p."),
            ("boundary_scale_", "Boundary x"),
        )
        for prefix, label in replacements:
            if value.startswith(prefix):
                return label + value.removeprefix(prefix)
        return value.replace("_", " ")

    labels = [compact_scenario_label(value) for value in worst.scenario_id]
    axes[2].set_yticks(y, labels, fontsize=5.6)
    axes[2].set_xlabel("Worst-case N+1 gain (h)")

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
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=int, default=6000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.jobs, args.epoch_h, args.max_hours) <= 0:
        raise ValueError("seeds, jobs and time settings must be positive")

    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        paired = pair_runs(per_run)
        scenario_summary = summarize_scenarios(paired)
        robustness = summarize_robustness(scenario_summary)
        paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
        scenario_summary.to_csv(args.out_dir / "scenario_summary.csv", index=False)
        robustness.to_csv(args.out_dir / "robustness_summary.csv", index=False)
        figure = args.out_dir / "fig23_n_plus_one_parameter_robustness.png"
        plot_results(scenario_summary, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        print(robustness.to_string(index=False))
        return

    scenarios = build_scenarios()
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
        health_limit = base_limit * scale
        gamma_scale = gamma_scale_for_terminal_cv(
            health_limit,
            base_continuous * scale,
            scenario["gamma_terminal_cv"],
        )
        config = ServiceScheduleConfig(
            health_limit_pct=health_limit,
            gamma_scale_pct=gamma_scale,
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
                SIMULATED_POLICIES,
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
        for scenario_rows, scenario_traces in executor.map(
            run_scenario_seed, tasks, chunksize=1
        ):
            rows.extend(scenario_rows)
            traces.extend(scenario_traces)
    per_run = pd.DataFrame(rows)
    expected_simulated = len(scenarios) * args.seeds * len(SIMULATED_POLICIES)
    if len(per_run) != expected_simulated:
        raise AssertionError("parameter robustness result matrix is incomplete")
    per_run = materialize_guarded_blend(per_run, scenarios)
    expected = len(scenarios) * args.seeds * len(POLICIES)
    if len(per_run) != expected:
        raise AssertionError("parameter robustness result matrix is incomplete")
    if not per_run.second_boundary_crossed.all():
        raise AssertionError("max_hours did not cover every second boundary")

    paired = pair_runs(per_run)
    scenario_summary = summarize_scenarios(paired)
    robustness = summarize_robustness(scenario_summary)
    manifest = pd.DataFrame(scenarios)
    manifest["initial_damage_fraction"] = manifest.initial_damage_fraction.astype(str)
    manifest["heterogeneity_factors"] = manifest.heterogeneity_factors.astype(str)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
    scenario_summary.to_csv(args.out_dir / "scenario_summary.csv", index=False)
    robustness.to_csv(args.out_dir / "robustness_summary.csv", index=False)
    manifest.to_csv(args.out_dir / "scenario_manifest.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_boundary_trajectories.csv", index=False
    )
    figure = args.out_dir / "fig23_n_plus_one_parameter_robustness.png"
    plot_results(scenario_summary, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "frozen N+1 objective parameter robustness",
        "scenarios": len(scenarios),
        "seeds_per_scenario": args.seeds,
        "evaluated_runs": len(per_run),
        "policies": list(POLICIES),
        "blend_weight": 0.50,
        "blend_weight_retuned": False,
        "heterogeneity_interpretation": (
            "literature-driven normalized quantile stress cases; not fitted vehicle factors"
        ),
        "failure_boundary_interpretation": (
            "LZW calibration endpoint and +/-20% sensitivity; not physical EOL"
        ),
        "future_demand_used": False,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Frozen N+1 parameter robustness

- {len(scenarios)} predeclared scenarios, {args.seeds} paired seeds per scenario.
- Literature random-effect quantiles are normalized stress cases, not identified vehicle-specific stack factors.
- Blend 0.50 is frozen and is not retuned in this audit.
- The health boundary is a calibration endpoint, not physical EOL.

## Robustness summary

{robustness.to_markdown(index=False)}
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
