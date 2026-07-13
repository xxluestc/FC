"""Stress-test N+1 scheduling while changing only the stopping boundary."""

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


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts/58_audit_n_plus_one_service_boundary.py"
ROBUSTNESS_SCRIPT = ROOT / "scripts/61_audit_n_plus_one_parameter_robustness.py"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
BOUNDARY_MAPPING = (
    ROOT
    / "data/results/fc_only_physical_boundary_mapping/voltage_loss_boundary_mapping.csv"
)
OUTPUT = ROOT / "data/results/fc_only_frozen_process_physical_boundaries"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
SIMULATED_POLICIES = ("fixed_pair", "order_blend_50")
POLICIES = SIMULATED_POLICIES + ("guarded_blend",)
SCENARIO_IDS = (
    "reference",
    "heterogeneity_gp_re_increased_perm_2",
    "heterogeneity_gp_re_increased_perm_3",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260717


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module("fc_n_plus_one_audit_physical_boundary", AUDIT_SCRIPT)
ROBUST = load_module("fc_n_plus_one_robust_physical_boundary", ROBUSTNESS_SCRIPT)


def select_mapping(mapping, current_a, loss_fraction):
    selected = mapping[
        np.isclose(mapping.current_a, current_a)
        & np.isclose(mapping.target_voltage_loss_fraction, loss_fraction)
    ]
    if len(selected) != 1:
        raise AssertionError("physical-boundary mapping row must be unique")
    return float(selected.inferred_damage_boundary_pct.iloc[0])


def build_boundaries(base_limit, mapping):
    rows = [
        {
            "boundary_id": "lzw_calibration_endpoint",
            "boundary_label": "LZW endpoint",
            "health_limit_pct": base_limit,
            "diagnostic_current_a": np.nan,
            "voltage_loss_fraction": np.nan,
            "extrapolated": False,
        },
        {
            "boundary_id": "370a_5pct_voltage_loss",
            "boundary_label": "370 A, 5% loss",
            "health_limit_pct": select_mapping(mapping, 370.0, 0.05),
            "diagnostic_current_a": 370.0,
            "voltage_loss_fraction": 0.05,
            "extrapolated": True,
        },
        {
            "boundary_id": "195a_5pct_voltage_loss",
            "boundary_label": "195 A, 5% loss",
            "health_limit_pct": select_mapping(mapping, 195.0, 0.05),
            "diagnostic_current_a": 195.0,
            "voltage_loss_fraction": 0.05,
            "extrapolated": True,
        },
        {
            "boundary_id": "370a_10pct_voltage_loss",
            "boundary_label": "370 A, 10% loss",
            "health_limit_pct": select_mapping(mapping, 370.0, 0.10),
            "diagnostic_current_a": 370.0,
            "voltage_loss_fraction": 0.10,
            "extrapolated": True,
        },
    ]
    rows.sort(key=lambda item: item["health_limit_pct"])
    for index, row in enumerate(rows):
        row["boundary_order"] = index
        row["boundary_over_calibration"] = row["health_limit_pct"] / base_limit
    return rows


def run_task(task):
    descriptor, simulation_task = task
    rows, traces = AUDIT.simulate_seed(simulation_task)
    for row in rows:
        row.update(descriptor)
        row["simulation_seed"] = row.pop("seed")
    for row in traces:
        row.update(descriptor)
    return rows, traces


def guard_source(initial_fraction, factors):
    initial = np.asarray(initial_fraction, dtype=float)
    rates = np.asarray(factors, dtype=float)
    separated = not np.allclose(initial, initial[0])
    aligned = int(np.argmax(initial)) == int(np.argmax(rates))
    return "order_blend_50" if separated and aligned else "fixed_pair"


def materialize_guarded(per_run):
    rows = [per_run]
    for (_, _), group in per_run.groupby(
        ["boundary_id", "scenario_id"], sort=False
    ):
        source = group.guard_source.iloc[0]
        selected = group[group.policy.eq(source)].copy()
        selected["policy"] = "guarded_blend"
        selected["effective_policy"] = source
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def pair_runs(per_run):
    keys = ["boundary_id", "scenario_id", "local_seed"]
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(keys)
    rows = []
    for policy in POLICIES[1:]:
        selected = per_run[per_run.policy.eq(policy)].set_index(keys)
        for key, row in selected.iterrows():
            reference = fixed.loc[key]
            rows.append(
                {
                    "boundary_id": key[0],
                    "scenario_id": key[1],
                    "local_seed": key[2],
                    "boundary_label": row.boundary_label,
                    "boundary_order": row.boundary_order,
                    "health_limit_pct": row.health_limit_pct,
                    "boundary_over_calibration": row.boundary_over_calibration,
                    "extrapolated": row.extrapolated,
                    "policy": policy,
                    "effective_policy": row.effective_policy,
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
    grouped = paired.groupby(["boundary_id", "scenario_id", "policy"], sort=False)
    for index, ((boundary_id, scenario_id, policy), group) in enumerate(grouped):
        first = group.first_boundary_gain_h.to_numpy(dtype=float)
        second = group.second_boundary_gain_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 2 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 2 * index + 1
        )
        start_delta = float(group.start_count_delta.mean())
        rows.append(
            {
                "boundary_id": boundary_id,
                "boundary_label": group.boundary_label.iloc[0],
                "boundary_order": int(group.boundary_order.iloc[0]),
                "health_limit_pct": float(group.health_limit_pct.iloc[0]),
                "boundary_over_calibration": float(
                    group.boundary_over_calibration.iloc[0]
                ),
                "extrapolated": bool(group.extrapolated.iloc[0]),
                "scenario_id": scenario_id,
                "policy": policy,
                "effective_policy": group.effective_policy.iloc[0],
                "seeds": len(group),
                "first_boundary_gain_mean_h": first_mean,
                "first_boundary_gain_ci95_low_h": first_low,
                "first_boundary_gain_ci95_high_h": first_high,
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_nonworse_share": float((second >= 0).mean()),
                "start_count_delta_mean": start_delta,
                "assignment_change_delta_mean": float(
                    group.assignment_change_delta.mean()
                ),
                "first_gain_per_extra_start_h": (
                    first_mean / start_delta if start_delta > 0 else np.nan
                ),
                "second_gain_per_extra_start_h": (
                    second_mean / start_delta if start_delta > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def pair_boundary_effects(paired):
    keys = ["scenario_id", "policy", "local_seed"]
    reference = paired[paired.boundary_order.eq(0)].set_index(keys)
    rows = []
    for _, row in paired[paired.boundary_order.gt(0)].iterrows():
        key = (row.scenario_id, row.policy, row.local_seed)
        baseline = reference.loc[key]
        rows.append(
            {
                "boundary_id": row.boundary_id,
                "boundary_label": row.boundary_label,
                "boundary_order": row.boundary_order,
                "boundary_over_calibration": row.boundary_over_calibration,
                "scenario_id": row.scenario_id,
                "policy": row.policy,
                "local_seed": row.local_seed,
                "first_gain_change_vs_lzw_h": (
                    row.first_boundary_gain_h - baseline.first_boundary_gain_h
                ),
                "second_gain_change_vs_lzw_h": (
                    row.second_boundary_gain_h - baseline.second_boundary_gain_h
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_boundary_effects(contrasts):
    rows = []
    grouped = contrasts.groupby(
        ["boundary_id", "scenario_id", "policy"], sort=False
    )
    for index, ((boundary_id, scenario_id, policy), group) in enumerate(grouped):
        first = group.first_gain_change_vs_lzw_h.to_numpy(dtype=float)
        second = group.second_gain_change_vs_lzw_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 1000 + 2 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 1001 + 2 * index
        )
        rows.append(
            {
                "boundary_id": boundary_id,
                "boundary_label": group.boundary_label.iloc[0],
                "boundary_order": int(group.boundary_order.iloc[0]),
                "boundary_over_calibration": float(
                    group.boundary_over_calibration.iloc[0]
                ),
                "scenario_id": scenario_id,
                "policy": policy,
                "seeds": len(group),
                "first_gain_change_vs_lzw_mean_h": first_mean,
                "first_gain_change_vs_lzw_ci95_low_h": first_low,
                "first_gain_change_vs_lzw_ci95_high_h": first_high,
                "second_gain_change_vs_lzw_mean_h": second_mean,
                "second_gain_change_vs_lzw_ci95_low_h": second_low,
                "second_gain_change_vs_lzw_ci95_high_h": second_high,
            }
        )
    return pd.DataFrame(rows)


def plot_metric(axis, table, metric, low, high, ylabel, show_ci=True):
    styles = {
        "reference": ("#0072B2", "o", "Reference rates"),
        "heterogeneity_gp_re_increased_perm_2": (
            "#D55E00",
            "s",
            "Strong, aligned",
        ),
        "heterogeneity_gp_re_increased_perm_3": (
            "#009E73",
            "^",
            "Strong, misaligned",
        ),
    }
    for scenario_id, (color, marker, label) in styles.items():
        selected = table[table.scenario_id.eq(scenario_id)].sort_values(
            "boundary_order"
        )
        x = selected.boundary_over_calibration.to_numpy(dtype=float)
        y = selected[metric].to_numpy(dtype=float)
        axis.plot(x, y, color=color, marker=marker, ms=3.5, lw=1.0, label=label)
        if show_ci:
            axis.fill_between(
                x,
                selected[low].to_numpy(dtype=float),
                selected[high].to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                lw=0,
            )
    axis.axhline(0, color="#6C757D", lw=0.8, ls="--")
    axis.set_xlabel("Boundary / LZW endpoint")
    axis.set_ylabel(ylabel)


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
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.85))
    guarded = summary[summary.policy.eq("guarded_blend")]
    blend = summary[summary.policy.eq("order_blend_50")]
    plot_metric(
        axes[0, 0],
        guarded,
        "first_boundary_gain_mean_h",
        "first_boundary_gain_ci95_low_h",
        "first_boundary_gain_ci95_high_h",
        "First-boundary gain (h)",
    )
    axes[0, 0].legend(frameon=False, fontsize=6.3)
    plot_metric(
        axes[0, 1],
        blend,
        "second_boundary_gain_mean_h",
        "second_boundary_gain_ci95_low_h",
        "second_boundary_gain_ci95_high_h",
        "Raw Blend N+1 gain (h)",
    )
    plot_metric(
        axes[1, 0],
        guarded,
        "second_boundary_gain_mean_h",
        "second_boundary_gain_ci95_low_h",
        "second_boundary_gain_ci95_high_h",
        "Guarded Blend N+1 gain (h)",
    )

    active = guarded[guarded.start_count_delta_mean > 0]
    for metric, color, marker, label in (
        ("first_gain_per_extra_start_h", "#0072B2", "o", "First boundary"),
        ("second_gain_per_extra_start_h", "#D55E00", "s", "N+1 boundary"),
    ):
        for scenario_id, line_style in (
            ("reference", "-"),
            ("heterogeneity_gp_re_increased_perm_2", "--"),
        ):
            selected = active[active.scenario_id.eq(scenario_id)].sort_values(
                "boundary_order"
            )
            axes[1, 1].plot(
                selected.boundary_over_calibration,
                selected[metric],
                color=color,
                marker=marker,
                ms=3.3,
                lw=1.0,
                ls=line_style,
                label=label if scenario_id == "reference" else None,
            )
    axes[1, 1].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[1, 1].set_xlabel("Boundary / LZW endpoint")
    axes[1, 1].set_ylabel("Gain per extra start (h/start)")
    axes[1, 1].legend(frameon=False, fontsize=6.1, loc="upper right")
    axes[1, 1].text(
        0.98,
        0.43,
        "solid: reference rates\ndashed: strong aligned",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="center",
        fontsize=6.0,
        color="#5F6368",
    )
    for index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.7, w_pad=1.1, h_pad=1.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    out_dir,
    summary,
    boundary_effect_summary,
    boundaries,
    base_limit,
    initial_absolute,
):
    guarded = summary[summary.policy.eq("guarded_blend")]
    blend = summary[summary.policy.eq("order_blend_50")]
    negative_guard = guarded[guarded.second_boundary_gain_ci95_high_h < 0]
    negative_blend = blend[blend.second_boundary_gain_ci95_high_h < 0]
    report = f"""# Frozen-process physical-boundary stress audit

- Only the stopping boundary changes. Initial absolute damage, degradation rates,
  Gamma scale, load templates, and policy parameters remain frozen.
- Initial absolute stack damage is `{tuple(round(x, 6) for x in initial_absolute)}`%.
- The LZW observed endpoint is `{base_limit:.6f}`%. All voltage-loss boundaries
  are model extrapolations, not validated physical EOL values.
- Raw Blend has {len(negative_blend)} scenario-boundary combinations with a
  strictly negative N+1 confidence interval; Guarded Blend has
  {len(negative_guard)}.

## Boundary manifest

{pd.DataFrame(boundaries).to_markdown(index=False)}

## Guarded Blend

{guarded.to_markdown(index=False)}

## Paired boundary-effect contrasts

{boundary_effect_summary[boundary_effect_summary.policy.eq("guarded_blend")].to_markdown(index=False)}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--epoch-h", type=float, default=1.0)
    parser.add_argument("--max-hours", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.jobs, args.epoch_h, args.max_hours) <= 0:
        raise ValueError("seeds, jobs and time settings must be positive")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    base_limit = float(calibration["terminal_total_damage_pct"])
    initial_absolute = base_limit * np.asarray((0.10, 0.40, 0.80))
    mapping = pd.read_csv(BOUNDARY_MAPPING)
    boundaries = build_boundaries(base_limit, mapping)
    figure = args.out_dir / "fig28_frozen_process_physical_boundaries.png"

    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        paired = pair_runs(per_run)
        summary = summarize(paired)
        boundary_effects = pair_boundary_effects(paired)
        boundary_effect_summary = summarize_boundary_effects(boundary_effects)
        paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
        summary.to_csv(args.out_dir / "boundary_scenario_summary.csv", index=False)
        boundary_effects.to_csv(
            args.out_dir / "paired_boundary_effect_deltas.csv", index=False
        )
        boundary_effect_summary.to_csv(
            args.out_dir / "boundary_effect_summary.csv", index=False
        )
        plot_results(summary, figure)
        FIGURES.mkdir(parents=True, exist_ok=True)
        (FIGURES / figure.name).write_bytes(figure.read_bytes())
        write_report(
            args.out_dir,
            summary,
            boundary_effect_summary,
            boundaries,
            base_limit,
            initial_absolute,
        )
        print(summary[summary.policy.eq("guarded_blend")].to_string(index=False))
        return

    scenario_map = {
        item["scenario_id"]: item for item in ROBUST.build_scenarios()
    }
    scenarios = [scenario_map[value] for value in SCENARIO_IDS]
    templates = AUDIT.load_real_templates(AUDIT.TEMPLATES)
    decision_exposure = AUDIT.stationary_service_exposure(templates, args.epoch_h)
    gamma_scale = float(calibration["gamma_scale_pct"])
    start_damage = float(
        calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
    )
    rotation_epochs = int(round(24.0 / args.epoch_h))
    reschedule_epochs = int(round(24.0 / args.epoch_h))
    tasks = []
    for boundary in boundaries:
        limit = boundary["health_limit_pct"]
        initial_fraction = tuple(float(value / limit) for value in initial_absolute)
        for scenario_index, scenario in enumerate(scenarios):
            factors = scenario["heterogeneity_factors"]
            source = guard_source(initial_fraction, factors)
            config = ServiceScheduleConfig(
                health_limit_pct=limit,
                gamma_scale_pct=gamma_scale,
                heterogeneity_factors=factors,
                start_damage_pct=start_damage,
                risk_horizon_h=100.0,
                risk_samples=512,
                n_plus_one_weight=0.50,
            )
            for local_seed in range(args.seeds):
                simulation_seed = scenario_index * 100_000 + local_seed
                descriptor = {
                    **boundary,
                    "scenario_id": scenario["scenario_id"],
                    "local_seed": local_seed,
                    "initial_damage_fraction": str(initial_fraction),
                    "initial_damage_absolute_pct": str(tuple(initial_absolute)),
                    "heterogeneity_factors": str(factors),
                    "guard_source": source,
                }
                simulation_task = (
                    simulation_seed,
                    SIMULATED_POLICIES,
                    templates,
                    decision_exposure,
                    config,
                    initial_fraction,
                    args.max_hours,
                    args.epoch_h,
                    rotation_epochs,
                    reschedule_epochs,
                    boundary["boundary_order"] == 0
                    and scenario_index == 0
                    and local_seed == 0,
                )
                tasks.append((descriptor, simulation_task))

    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for task_rows, task_traces in executor.map(run_task, tasks, chunksize=1):
            rows.extend(task_rows)
            traces.extend(task_traces)
    per_run = pd.DataFrame(rows)
    expected_simulated = (
        len(boundaries) * len(scenarios) * args.seeds * len(SIMULATED_POLICIES)
    )
    if len(per_run) != expected_simulated:
        raise AssertionError("physical-boundary result matrix is incomplete")
    per_run = materialize_guarded(per_run)
    if not per_run.second_boundary_crossed.all():
        args.out_dir.mkdir(parents=True, exist_ok=True)
        censored = per_run[~per_run.second_boundary_crossed]
        censored.to_csv(args.out_dir / "censored_runs.csv", index=False)
        raise AssertionError(
            f"max_hours={args.max_hours} censored {len(censored)} second boundaries"
        )
    paired = pair_runs(per_run)
    summary = summarize(paired)
    boundary_effects = pair_boundary_effects(paired)
    boundary_effect_summary = summarize_boundary_effects(boundary_effects)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_seed_deltas.csv", index=False)
    summary.to_csv(args.out_dir / "boundary_scenario_summary.csv", index=False)
    boundary_effects.to_csv(
        args.out_dir / "paired_boundary_effect_deltas.csv", index=False
    )
    boundary_effect_summary.to_csv(
        args.out_dir / "boundary_effect_summary.csv", index=False
    )
    pd.DataFrame(boundaries).to_csv(args.out_dir / "boundary_manifest.csv", index=False)
    pd.DataFrame(traces).to_csv(
        args.out_dir / "representative_boundary_trajectories.csv", index=False
    )
    plot_results(summary, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())
    metadata = {
        "scope": "frozen-process stopping-boundary stress audit",
        "boundary_count": len(boundaries),
        "scenario_ids": list(SCENARIO_IDS),
        "seeds_per_scenario_boundary": args.seeds,
        "policies": list(POLICIES),
        "initial_absolute_damage_pct": list(initial_absolute),
        "gamma_scale_pct_frozen": gamma_scale,
        "degradation_coefficients_frozen": True,
        "load_templates_frozen": True,
        "policy_parameters_frozen": True,
        "future_demand_used": False,
        "voltage_loss_boundaries_validated": False,
        "voltage_loss_boundary_status": "model-extrapolation stress scenarios",
        "pairing": "same sampled exposure and inverse-CDF uniforms within each seed",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        args.out_dir,
        summary,
        boundary_effect_summary,
        boundaries,
        base_limit,
        initial_absolute,
    )
    print(summary[summary.policy.eq("guarded_blend")].to_string(index=False))


if __name__ == "__main__":
    main()
