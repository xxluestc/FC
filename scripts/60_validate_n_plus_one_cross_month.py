"""Validate the frozen N+1 trade-off objective on held-out monthly exposures."""

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
from scipy.stats import binomtest, bootstrap

from fc_power.evaluation import ServiceExposure, ServiceScheduleConfig
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts/58_audit_n_plus_one_service_boundary.py"
EXTERNAL_TEMPLATES = (
    ROOT
    / "data/results/fc_only_external_service_templates/service_exposure_templates.csv"
)
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_n_plus_one_cross_month"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
POLICIES = ("fixed_pair", "order_blend_50", "expected_n_plus_one")
LABELS = {
    "order_blend_50": "Blend 0.50",
    "expected_n_plus_one": "Pure N+1",
}
COLORS = {
    "order_blend_50": "#5F6CAF",
    "expected_n_plus_one": "#0077B6",
}
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260714


def load_audit_module():
    spec = importlib.util.spec_from_file_location("fc_n_plus_one_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()


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


def run_month_seed(task):
    month, local_seed, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    for row in rows:
        row["month"] = month
        row["local_seed"] = local_seed
        row["simulation_seed"] = row.pop("seed")
    for row in traces:
        row["month"] = month
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
        ["month", "local_seed"]
    )
    rows = []
    for policy in POLICIES[1:]:
        selected = per_run[per_run.policy.eq(policy)].set_index(
            ["month", "local_seed"]
        )
        first = selected.time_to_first_boundary_h - fixed.time_to_first_boundary_h
        second = selected.time_to_second_boundary_h - fixed.time_to_second_boundary_h
        for (month, seed), first_gain in first.items():
            rows.append(
                {
                    "policy": policy,
                    "month": month,
                    "local_seed": seed,
                    "first_boundary_gain_h": first_gain,
                    "second_boundary_gain_h": second.loc[(month, seed)],
                }
            )
    return pd.DataFrame(rows)


def summarize_months(paired):
    return (
        paired.groupby(["policy", "month"], sort=True)
        .agg(
            seeds=("local_seed", "size"),
            first_boundary_gain_mean_h=("first_boundary_gain_h", "mean"),
            second_boundary_gain_mean_h=("second_boundary_gain_h", "mean"),
            second_boundary_gain_median_h=("second_boundary_gain_h", "median"),
            second_boundary_win_share=("second_boundary_gain_h", lambda x: (x > 0).mean()),
            second_boundary_nonworse_share=(
                "second_boundary_gain_h", lambda x: (x >= 0).mean()
            ),
        )
        .reset_index()
    )


def summarize_primary(monthly, paired):
    rows = []
    for index, policy in enumerate(POLICIES[1:]):
        selected = monthly[monthly.policy.eq(policy)]
        first = selected.first_boundary_gain_mean_h.to_numpy(dtype=float)
        second = selected.second_boundary_gain_mean_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 10 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 10 * index + 1
        )
        wins = int((second > 0).sum())
        losses = int((second < 0).sum())
        informative = wins + losses
        rows.append(
            {
                "policy": policy,
                "months": len(selected),
                "first_boundary_gain_mean_h": first_mean,
                "first_boundary_gain_ci95_low_h": first_low,
                "first_boundary_gain_ci95_high_h": first_high,
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_gain_median_h": float(np.median(second)),
                "second_boundary_better_months": wins,
                "second_boundary_tied_months": int((second == 0).sum()),
                "second_boundary_worse_months": losses,
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
                "seed_level_nonworse_share": float(
                    (
                        paired.loc[
                            paired.policy.eq(policy), "second_boundary_gain_h"
                        ]
                        >= 0
                    ).mean()
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
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45))
    months = sorted(monthly.month.unique())
    x = np.arange(len(months))
    selected = (
        monthly[monthly.policy.eq("order_blend_50")]
        .set_index("month")
        .loc[months]
    )
    axes[0].plot(
        x,
        selected.first_boundary_gain_mean_h,
        marker="o",
        ms=3.0,
        lw=1.1,
        color=COLORS["order_blend_50"],
    )
    axes[1].plot(
        x,
        selected.second_boundary_gain_mean_h,
        marker="o",
        ms=3.0,
        lw=1.1,
        color=COLORS["order_blend_50"],
    )
    for axis in axes[:2]:
        axis.axhline(0, color="#7A8288", lw=0.8, ls="--")
        axis.set_xticks(x, [value[2:] for value in months], rotation=45, ha="right")
        axis.set_xlabel("Held-out month (2025-06 to 2026-06)")
    axes[0].set_ylabel("First-boundary gain (h)")
    axes[1].set_ylabel("N+1 second-boundary gain (h)")

    ordered = primary.set_index("policy").loc[list(POLICIES[1:])]
    y = np.arange(len(ordered))
    mean = ordered.second_boundary_gain_mean_h.to_numpy(dtype=float)
    low = mean - ordered.second_boundary_gain_ci95_low_h.to_numpy(dtype=float)
    high = ordered.second_boundary_gain_ci95_high_h.to_numpy(dtype=float) - mean
    axes[2].errorbar(
        mean,
        y,
        xerr=np.vstack((low, high)),
        fmt="o",
        ms=4.2,
        color="#1B263B",
        ecolor="#6C757D",
        capsize=2.2,
        lw=1.0,
    )
    axes[2].axvline(0, color="#7A8288", lw=0.8, ls="--")
    axes[2].set_yticks(y, [LABELS[value] for value in POLICIES[1:]])
    axes[2].set_xlabel("Month-level mean N+1 gain (h)")
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
    if not EXTERNAL_TEMPLATES.exists():
        raise FileNotFoundError("run script 59 before cross-month validation")

    if args.summarize_only:
        monthly = pd.read_csv(args.out_dir / "monthly_effects.csv")
        primary = pd.read_csv(args.out_dir / "primary_statistics.csv")
        figure = args.out_dir / "fig22_n_plus_one_cross_month.png"
        plot_results(monthly, primary, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        print(primary.to_string(index=False))
        return

    external = pd.read_csv(EXTERNAL_TEMPLATES)
    if len(external) != 39 or external.month.nunique() != 13:
        raise AssertionError("cross-month validation requires 39 templates")
    development = AUDIT.load_real_templates(AUDIT.TEMPLATES)
    decision_exposure = AUDIT.stationary_service_exposure(
        development, args.epoch_h
    )
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    health_limit = float(calibration["terminal_total_damage_pct"])
    gamma_scale = gamma_scale_for_terminal_cv(
        health_limit,
        float(calibration["terminal_continuous_damage_pct"]),
        0.10,
    )
    config = ServiceScheduleConfig(
        health_limit_pct=health_limit,
        gamma_scale_pct=gamma_scale,
        heterogeneity_factors=tuple(AUDIT.HETEROGENEITY),
        start_damage_pct=float(
            calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
        ),
        risk_horizon_h=100.0,
        risk_samples=512,
    )
    rotation_epochs = int(round(24.0 / args.epoch_h))
    reschedule_epochs = int(round(24.0 / args.epoch_h))
    tasks = []
    for month_index, (month, table) in enumerate(
        external.groupby("month", sort=True)
    ):
        templates = [row_to_exposure(row) for row in table.itertuples(index=False)]
        for local_seed in range(args.seeds):
            simulation_seed = month_index * 1000 + local_seed
            simulation_task = (
                simulation_seed,
                POLICIES,
                templates,
                decision_exposure,
                config,
                args.max_hours,
                args.epoch_h,
                rotation_epochs,
                reschedule_epochs,
                month_index == 0 and local_seed == 0,
            )
            tasks.append((month, local_seed, simulation_task))

    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for month_rows, month_traces in executor.map(
            run_month_seed, tasks, chunksize=1
        ):
            rows.extend(month_rows)
            traces.extend(month_traces)
    per_run = pd.DataFrame(rows)
    if len(per_run) != 13 * args.seeds * len(POLICIES):
        raise AssertionError("cross-month result matrix is incomplete")
    if not per_run.second_boundary_crossed.all():
        raise AssertionError("max_hours did not cover every second boundary")
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
    figure = args.out_dir / "fig22_n_plus_one_cross_month.png"
    plot_results(monthly, primary, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "held-out cross-month N+1 long-horizon validation",
        "months": int(external.month.nunique()),
        "templates": len(external),
        "seeds_per_month": args.seeds,
        "evaluated_runs": len(per_run),
        "policies": list(POLICIES),
        "blend_weight": 0.50,
        "blend_weight_retuned_on_external_data": False,
        "decision_exposure": "frozen mean of 48 development templates",
        "realized_exposure": "sampled only from each held-out month's three blocks",
        "future_demand_used": False,
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
    report = f"""# Held-out cross-month N+1 validation

- Frozen Blend 0.50 was evaluated on 13 calendar months with {args.seeds} paired seeds per month.
- The scheduler used only the development-template mean; monthly future exposure was hidden.
- The boundary is the LZW calibration endpoint, not a physical failure threshold.

## Primary calendar-month statistics

{primary.to_markdown(index=False)}
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
