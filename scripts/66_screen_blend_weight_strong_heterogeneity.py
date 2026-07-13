"""Screen predeclared order-statistic weights under strong aligned rates."""

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
AUDIT_SCRIPT = ROOT / "scripts/58_audit_n_plus_one_service_boundary.py"
ROBUSTNESS_SCRIPT = ROOT / "scripts/61_audit_n_plus_one_parameter_robustness.py"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_blend_weight_strong_heterogeneity"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
POLICIES = (
    "fixed_pair",
    "order_blend_25",
    "order_blend_50",
    "order_blend_75",
    "order_blend_90",
    "order_blend_99",
)
SCENARIO_IDS = (
    "reference",
    "heterogeneity_gp_re_increased_perm_2",
)
WEIGHTS = {
    "order_blend_25": 0.25,
    "order_blend_50": 0.50,
    "order_blend_75": 0.75,
    "order_blend_90": 0.90,
    "order_blend_99": 0.99,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260718


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module("fc_n_plus_one_audit_weight_screen", AUDIT_SCRIPT)
ROBUST = load_module("fc_n_plus_one_robust_weight_screen", ROBUSTNESS_SCRIPT)


def run_task(task):
    scenario_id, local_seed, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    for row in rows:
        row["scenario_id"] = scenario_id
        row["local_seed"] = local_seed
        row["simulation_seed"] = row.pop("seed")
    for row in traces:
        row["scenario_id"] = scenario_id
        row["local_seed"] = local_seed
    return rows, traces


def pair_runs(per_run):
    keys = ["scenario_id", "local_seed"]
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(keys)
    rows = []
    for policy, weight in WEIGHTS.items():
        selected = per_run[per_run.policy.eq(policy)].set_index(keys)
        for key, row in selected.iterrows():
            reference = fixed.loc[key]
            rows.append(
                {
                    "scenario_id": key[0],
                    "local_seed": key[1],
                    "policy": policy,
                    "n_plus_one_weight": weight,
                    "first_boundary_gain_h": (
                        row.time_to_first_boundary_h
                        - reference.time_to_first_boundary_h
                    ),
                    "second_boundary_gain_h": (
                        row.time_to_second_boundary_h
                        - reference.time_to_second_boundary_h
                    ),
                    "start_count_delta": row.start_count - reference.start_count,
                }
            )
    return pd.DataFrame(rows)


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


def summarize(paired):
    rows = []
    grouped = paired.groupby(["scenario_id", "policy"], sort=False)
    for index, ((scenario_id, policy), group) in enumerate(grouped):
        first = group.first_boundary_gain_h.to_numpy(dtype=float)
        second = group.second_boundary_gain_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 2 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 2 * index + 1
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "n_plus_one_weight": float(group.n_plus_one_weight.iloc[0]),
                "seeds": len(group),
                "first_boundary_gain_mean_h": first_mean,
                "first_boundary_gain_ci95_low_h": first_low,
                "first_boundary_gain_ci95_high_h": first_high,
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_nonworse_share": float((second >= 0).mean()),
                "start_count_delta_mean": float(group.start_count_delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def select_candidates(summary):
    rows = []
    for scenario_id, group in summary.groupby("scenario_id", sort=False):
        feasible = group[group.second_boundary_gain_ci95_low_h >= 0]
        if feasible.empty:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "selected_policy": "fixed_pair",
                    "selection_status": "no blend weight passed N+1 lower-CI gate",
                }
            )
            continue
        selected = feasible.loc[feasible.first_boundary_gain_mean_h.idxmax()]
        rows.append(
            {
                "scenario_id": scenario_id,
                "selected_policy": selected.policy,
                "selected_weight": selected.n_plus_one_weight,
                "first_boundary_gain_mean_h": selected.first_boundary_gain_mean_h,
                "second_boundary_gain_mean_h": selected.second_boundary_gain_mean_h,
                "second_boundary_gain_ci95_low_h": (
                    selected.second_boundary_gain_ci95_low_h
                ),
                "selection_status": "development candidate; external validation required",
            }
        )
    return pd.DataFrame(rows)


def plot_results(summary, output_path):
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
    styles = {
        "reference": ("#0072B2", "o", "Reference rates"),
        "heterogeneity_gp_re_increased_perm_2": (
            "#D55E00",
            "s",
            "Strong, aligned",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.6))
    for scenario_id, (color, marker, label) in styles.items():
        selected = summary[summary.scenario_id.eq(scenario_id)].sort_values(
            "n_plus_one_weight"
        )
        x = selected.n_plus_one_weight.to_numpy(dtype=float)
        first = selected.first_boundary_gain_mean_h.to_numpy(dtype=float)
        second = selected.second_boundary_gain_mean_h.to_numpy(dtype=float)
        axes[0].plot(x, first, color=color, marker=marker, ms=3.5, lw=1.0, label=label)
        axes[1].plot(x, second, color=color, marker=marker, ms=3.5, lw=1.0)
        axes[1].fill_between(
            x,
            selected.second_boundary_gain_ci95_low_h.to_numpy(dtype=float),
            selected.second_boundary_gain_ci95_high_h.to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            lw=0,
        )
        axes[2].plot(
            first,
            second,
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.0,
            label=label,
        )
    axes[0].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[0].set_xlabel("N+1 objective weight")
    axes[0].set_ylabel("First-boundary gain (h)")
    axes[0].legend(frameon=False, fontsize=6.2)
    axes[1].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[1].set_xlabel("N+1 objective weight")
    axes[1].set_ylabel("N+1 boundary gain (h)")
    axes[2].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[2].axvline(0, color="#6C757D", lw=0.8, ls="--")
    axes[2].set_xlabel("First-boundary gain (h)")
    axes[2].set_ylabel("N+1 boundary gain (h)")
    for index, axis in enumerate(axes):
        axis.text(
            0.0,
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
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.jobs, args.epoch_h, args.max_hours) <= 0:
        raise ValueError("seeds, jobs and time settings must be positive")

    if args.summarize_only:
        summary = pd.read_csv(args.out_dir / "weight_summary.csv")
        candidates = pd.read_csv(args.out_dir / "candidate_selection.csv")
        figure = args.out_dir / "fig29_blend_weight_strong_heterogeneity.png"
        plot_results(summary, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        print(candidates.to_string(index=False))
        return

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    health_limit = float(calibration["terminal_total_damage_pct"])
    continuous = float(calibration["terminal_continuous_damage_pct"])
    gamma_scale = gamma_scale_for_terminal_cv(
        health_limit, continuous, 0.10
    )
    start_damage = float(
        calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
    )
    templates = AUDIT.load_real_templates(AUDIT.TEMPLATES)
    decision_exposure = AUDIT.stationary_service_exposure(templates, args.epoch_h)
    scenario_map = {
        item["scenario_id"]: item for item in ROBUST.build_scenarios()
    }
    scenarios = [scenario_map[value] for value in SCENARIO_IDS]
    rotation_epochs = int(round(24.0 / args.epoch_h))
    reschedule_epochs = int(round(24.0 / args.epoch_h))
    tasks = []
    for scenario_index, scenario in enumerate(scenarios):
        config = ServiceScheduleConfig(
            health_limit_pct=health_limit,
            gamma_scale_pct=gamma_scale,
            heterogeneity_factors=scenario["heterogeneity_factors"],
            start_damage_pct=start_damage,
            risk_horizon_h=100.0,
            risk_samples=512,
            n_plus_one_weight=0.50,
        )
        for local_seed in range(args.seeds):
            simulation_seed = scenario_index * 100_000 + local_seed
            simulation_task = (
                simulation_seed,
                POLICIES,
                templates,
                decision_exposure,
                config,
                scenario["initial_damage_fraction"],
                args.max_hours,
                args.epoch_h,
                rotation_epochs,
                reschedule_epochs,
                scenario_index == 0 and local_seed == 0,
            )
            tasks.append((scenario["scenario_id"], local_seed, simulation_task))

    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for task_rows, task_traces in executor.map(run_task, tasks, chunksize=1):
            rows.extend(task_rows)
            traces.extend(task_traces)
    per_run = pd.DataFrame(rows)
    expected = len(scenarios) * args.seeds * len(POLICIES)
    if len(per_run) != expected or not per_run.second_boundary_crossed.all():
        raise AssertionError("blend-weight screen matrix is incomplete")
    paired = pair_runs(per_run)
    summary = summarize(paired)
    candidates = select_candidates(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
    summary.to_csv(args.out_dir / "weight_summary.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_selection.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_weight_trajectories.csv", index=False
    )
    figure = args.out_dir / "fig29_blend_weight_strong_heterogeneity.png"
    plot_results(summary, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "development screen of predeclared order-statistic weights",
        "scenario_ids": list(SCENARIO_IDS),
        "weights": list(WEIGHTS.values()),
        "seeds_per_scenario": args.seeds,
        "selection_gate": "N+1 gain lower 95% BCa confidence bound >= 0",
        "selection_tiebreak": "maximum mean first-boundary gain",
        "external_validation_required": True,
        "future_demand_used": False,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report = f"""# Blend-weight screen under strong aligned heterogeneity

- The five weights were already declared in the N+1 audit code.
- A weight is eligible only when its paired N+1 lower confidence bound is
  nonnegative. Selection remains development-only until external validation.

## Summary

{summary.to_markdown(index=False)}

## Candidate selection

{candidates.to_markdown(index=False)}
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
