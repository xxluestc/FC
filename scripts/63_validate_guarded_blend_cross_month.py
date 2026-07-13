"""Validate Guarded Blend across held-out months and heterogeneity alignments."""

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

from fc_power.evaluation import ServiceExposure, ServiceScheduleConfig
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts/58_audit_n_plus_one_service_boundary.py"
ROBUSTNESS_SCRIPT = ROOT / "scripts/61_audit_n_plus_one_parameter_robustness.py"
EXTERNAL_TEMPLATES = (
    ROOT
    / "data/results/fc_only_external_service_templates/service_exposure_templates.csv"
)
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_guarded_blend_cross_month"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
SCENARIO_IDS = (
    "reference",
    "heterogeneity_gp_re_increased_perm_2",
    "heterogeneity_gp_re_increased_perm_3",
)
SIMULATED_POLICIES = ("fixed_pair", "order_blend_50")
POLICIES = SIMULATED_POLICIES + ("guarded_blend",)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260716


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module("fc_n_plus_one_audit_guarded", AUDIT_SCRIPT)
ROBUST = load_module("fc_n_plus_one_robustness_guarded", ROBUSTNESS_SCRIPT)


def row_to_exposure(row) -> ServiceExposure:
    return ServiceExposure(
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


def run_scenario_month_seed(task):
    scenario_id, month, local_seed, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    for row in rows:
        row["scenario_id"] = scenario_id
        row["month"] = month
        row["local_seed"] = local_seed
        row["simulation_seed"] = row.pop("seed")
    for row in traces:
        row["scenario_id"] = scenario_id
        row["month"] = month
        row["local_seed"] = local_seed
    return rows, traces


def guard_source(scenario):
    initial = np.asarray(scenario["initial_damage_fraction"], dtype=float)
    factors = np.asarray(scenario["heterogeneity_factors"], dtype=float)
    separated = not np.allclose(initial, initial[0])
    aligned = int(np.argmax(initial)) == int(np.argmax(factors))
    return "order_blend_50" if separated and aligned else "fixed_pair"


def materialize_guarded(per_run, scenarios):
    rows = [per_run]
    for scenario in scenarios:
        source = guard_source(scenario)
        selected = per_run[
            per_run.scenario_id.eq(scenario["scenario_id"])
            & per_run.policy.eq(source)
        ].copy()
        selected["policy"] = "guarded_blend"
        selected["effective_policy"] = source
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def pair_runs(per_run):
    keys = ["scenario_id", "month", "local_seed"]
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(keys)
    rows = []
    for policy in POLICIES[1:]:
        selected = per_run[per_run.policy.eq(policy)].set_index(keys)
        first = selected.time_to_first_boundary_h - fixed.time_to_first_boundary_h
        second = selected.time_to_second_boundary_h - fixed.time_to_second_boundary_h
        for key, first_gain in first.items():
            rows.append(
                {
                    "scenario_id": key[0],
                    "month": key[1],
                    "local_seed": key[2],
                    "policy": policy,
                    "first_boundary_gain_h": first_gain,
                    "second_boundary_gain_h": second.loc[key],
                }
            )
    return pd.DataFrame(rows)


def summarize_months(paired):
    return (
        paired.groupby(["scenario_id", "policy", "month"], sort=True)
        .agg(
            seeds=("local_seed", "size"),
            first_boundary_gain_mean_h=("first_boundary_gain_h", "mean"),
            second_boundary_gain_mean_h=("second_boundary_gain_h", "mean"),
            second_boundary_nonworse_share=(
                "second_boundary_gain_h", lambda values: (values >= 0).mean()
            ),
        )
        .reset_index()
    )


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


def summarize_primary(monthly, paired):
    rows = []
    for scenario_index, scenario_id in enumerate(SCENARIO_IDS):
        for policy_index, policy in enumerate(POLICIES[1:]):
            selected = monthly[
                monthly.scenario_id.eq(scenario_id) & monthly.policy.eq(policy)
            ]
            first = selected.first_boundary_gain_mean_h.to_numpy(dtype=float)
            second = selected.second_boundary_gain_mean_h.to_numpy(dtype=float)
            seed = BOOTSTRAP_SEED + 20 * scenario_index + 2 * policy_index
            first_mean, first_low, first_high = mean_bca(first, seed)
            second_mean, second_low, second_high = mean_bca(second, seed + 1)
            seed_rows = paired[
                paired.scenario_id.eq(scenario_id) & paired.policy.eq(policy)
            ]
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "policy": policy,
                    "months": len(selected),
                    "first_boundary_gain_mean_h": first_mean,
                    "first_boundary_gain_ci95_low_h": first_low,
                    "first_boundary_gain_ci95_high_h": first_high,
                    "second_boundary_gain_mean_h": second_mean,
                    "second_boundary_gain_ci95_low_h": second_low,
                    "second_boundary_gain_ci95_high_h": second_high,
                    "second_boundary_better_months": int((second > 0).sum()),
                    "second_boundary_tied_months": int((second == 0).sum()),
                    "second_boundary_worse_months": int((second < 0).sum()),
                    "seed_level_nonworse_share": float(
                        (seed_rows.second_boundary_gain_h >= 0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_results(monthly, primary, output_path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))
    months = sorted(monthly.month.unique())
    x = np.arange(len(months))
    reference = (
        monthly[
            monthly.scenario_id.eq("reference")
            & monthly.policy.eq("guarded_blend")
        ]
        .set_index("month")
        .loc[months]
    )
    tick_positions = x[::2]
    tick_labels = [months[index][2:] for index in tick_positions]
    for axis, column, ylabel in (
        (axes[0], "first_boundary_gain_mean_h", "First-boundary gain (h)"),
        (axes[1], "second_boundary_gain_mean_h", "N+1 second-boundary gain (h)"),
    ):
        axis.plot(
            x,
            reference[column],
            marker="o",
            ms=3.0,
            lw=1.1,
            color="#D55E00",
        )
        axis.axhline(0, color="#7A8288", lw=0.8, ls="--")
        axis.set_xticks(tick_positions, tick_labels, rotation=35, ha="right")
        axis.set_xlabel("Held-out month")
        axis.set_ylabel(ylabel)

    selected = (
        primary[primary.policy.eq("guarded_blend")]
        .set_index("scenario_id")
        .loc[list(SCENARIO_IDS)]
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
        ms=4.0,
        color="#1B263B",
        ecolor="#6C757D",
        capsize=2.2,
        lw=1.0,
    )
    axes[2].axvline(0, color="#7A8288", lw=0.8, ls="--")
    axes[2].set_yticks(
        y,
        ("Reference", "Strong RE aligned", "Strong RE mismatched"),
    )
    axes[2].set_xlabel("Month-level N+1 gain (h)")
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
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=int, default=6000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.jobs, args.epoch_h, args.max_hours) <= 0:
        raise ValueError("seeds, jobs and time settings must be positive")

    figure = args.out_dir / "fig26_guarded_blend_cross_month.png"
    if args.summarize_only:
        monthly = pd.read_csv(args.out_dir / "monthly_effects.csv")
        primary = pd.read_csv(args.out_dir / "primary_statistics.csv")
        plot_results(monthly, primary, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        print(primary[primary.policy.eq("guarded_blend")].to_string(index=False))
        return

    external = pd.read_csv(EXTERNAL_TEMPLATES)
    if len(external) != 39 or external.month.nunique() != 13:
        raise AssertionError("guarded validation requires 13 months and 39 templates")
    scenario_map = {
        item["scenario_id"]: item for item in ROBUST.build_scenarios()
    }
    scenarios = [scenario_map[value] for value in SCENARIO_IDS]
    development = AUDIT.load_real_templates(AUDIT.TEMPLATES)
    decision_exposure = AUDIT.stationary_service_exposure(development, args.epoch_h)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    base_limit = float(calibration["terminal_total_damage_pct"])
    base_continuous = float(calibration["terminal_continuous_damage_pct"])
    start_damage = float(
        calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
    )
    rotation_epochs = int(round(24.0 / args.epoch_h))
    reschedule_epochs = int(round(24.0 / args.epoch_h))
    tasks = []
    for scenario_index, scenario in enumerate(scenarios):
        config = ServiceScheduleConfig(
            health_limit_pct=base_limit,
            gamma_scale_pct=gamma_scale_for_terminal_cv(
                base_limit, base_continuous, scenario["gamma_terminal_cv"]
            ),
            heterogeneity_factors=scenario["heterogeneity_factors"],
            start_damage_pct=start_damage,
            risk_horizon_h=scenario["risk_horizon_h"],
            risk_samples=512,
            n_plus_one_weight=0.50,
        )
        for month_index, (month, table) in enumerate(
            external.groupby("month", sort=True)
        ):
            templates = [
                row_to_exposure(row) for row in table.itertuples(index=False)
            ]
            for local_seed in range(args.seeds):
                simulation_seed = scenario_index * 100_000 + month_index * 1000 + local_seed
                simulation_task = (
                    simulation_seed,
                    SIMULATED_POLICIES,
                    templates,
                    decision_exposure,
                    config,
                    scenario["initial_damage_fraction"],
                    args.max_hours,
                    args.epoch_h,
                    rotation_epochs,
                    reschedule_epochs,
                    scenario_index == 0 and month_index == 0 and local_seed == 0,
                )
                tasks.append(
                    (scenario["scenario_id"], month, local_seed, simulation_task)
                )

    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for task_rows, task_traces in executor.map(
            run_scenario_month_seed, tasks, chunksize=1
        ):
            rows.extend(task_rows)
            traces.extend(task_traces)
    per_run = pd.DataFrame(rows)
    expected_simulated = len(SCENARIO_IDS) * 13 * args.seeds * len(SIMULATED_POLICIES)
    if len(per_run) != expected_simulated:
        raise AssertionError("guarded cross-month simulated matrix is incomplete")
    per_run = materialize_guarded(per_run, scenarios)
    expected = len(SCENARIO_IDS) * 13 * args.seeds * len(POLICIES)
    if len(per_run) != expected or not per_run.second_boundary_crossed.all():
        raise AssertionError("guarded cross-month result matrix is incomplete")
    paired = pair_runs(per_run)
    monthly = summarize_months(paired)
    primary = summarize_primary(monthly, paired)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
    monthly.to_csv(args.out_dir / "monthly_effects.csv", index=False)
    primary.to_csv(args.out_dir / "primary_statistics.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_boundary_trajectories.csv", index=False
    )
    plot_results(monthly, primary, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "held-out cross-month validation of the Guarded Blend fallback",
        "scenario_ids": list(SCENARIO_IDS),
        "scenario_guard_sources": {
            item["scenario_id"]: guard_source(item) for item in scenarios
        },
        "months": 13,
        "templates": 39,
        "seeds_per_month": args.seeds,
        "policies": list(POLICIES),
        "decision_exposure": "frozen mean of 48 development templates",
        "realized_exposure": "held-out monthly three-block bootstrap",
        "future_demand_used": False,
        "heterogeneity_status": "literature-driven stress factors, not vehicle-fitted",
        "health_boundary_interpretation": (
            "LZW calibration trajectory endpoint, not a physical failure threshold"
        ),
        "bootstrap_unit": "calendar month",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Guarded Blend held-out cross-month validation

- Three predeclared health-rate alignment scenarios were evaluated over 13 held-out months.
- The guard used current health and literature-driven rate factors, not future demand.
- The LZW endpoint remains a calibration boundary rather than a physical EOL threshold.

## Month-level statistics

{primary.to_markdown(index=False)}
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(primary[primary.policy.eq("guarded_blend")].to_string(index=False))


if __name__ == "__main__":
    main()
