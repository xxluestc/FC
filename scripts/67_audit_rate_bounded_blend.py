"""Replay the nominal-rate applicability guard on existing paired runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from fc_power.evaluation import select_guarded_blend_policy


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_SCRIPT = ROOT / "scripts/61_audit_n_plus_one_parameter_robustness.py"
PARAMETER_DIR = ROOT / "data/results/fc_only_n_plus_one_parameter_robustness"
PHYSICAL_DIR = ROOT / "data/results/fc_only_frozen_process_physical_boundaries"
CROSS_MONTH_DIR = ROOT / "data/results/fc_only_guarded_blend_cross_month"
OUTPUT = ROOT / "data/results/fc_only_rate_bounded_blend"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
RATE_RATIO_LIMIT = 1.10
POLICIES = ("guarded_blend", "rate_bounded_blend")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260719


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROBUST = load_module("fc_n_plus_one_robust_rate_guard", ROBUSTNESS_SCRIPT)


def scenario_map():
    return {item["scenario_id"]: item for item in ROBUST.build_scenarios()}


def materialize_rate_bounded(per_run, scenarios):
    rows = [per_run]
    sources = []
    for scenario_id in per_run.scenario_id.drop_duplicates():
        scenario = scenarios[scenario_id]
        source = select_guarded_blend_policy(
            scenario["initial_damage_fraction"],
            scenario["heterogeneity_factors"],
            max_rate_ratio=RATE_RATIO_LIMIT,
        )
        selected = per_run[
            per_run.scenario_id.eq(scenario_id) & per_run.policy.eq(source)
        ].copy()
        selected["policy"] = "rate_bounded_blend"
        selected["effective_policy"] = source
        rows.append(selected)
        factors = np.asarray(scenario["heterogeneity_factors"], dtype=float)
        sources.append(
            {
                "scenario_id": scenario_id,
                "rate_ratio": float(factors.max() / factors.min()),
                "rate_bounded_source": source,
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sources)


def paired_deltas(per_run, keys):
    fixed = per_run[per_run.policy.eq("fixed_pair")].set_index(keys)
    rows = []
    for policy in POLICIES:
        selected = per_run[per_run.policy.eq(policy)].set_index(keys)
        for key, row in selected.iterrows():
            if not isinstance(key, tuple):
                key = (key,)
            reference = fixed.loc[key]
            record = {name: value for name, value in zip(keys, key)}
            record.update(
                {
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
                }
            )
            rows.append(record)
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


def summarize_groups(paired, group_columns, cluster_column=None):
    rows = []
    for index, (key, group) in enumerate(paired.groupby(group_columns, sort=False)):
        if not isinstance(key, tuple):
            key = (key,)
        if cluster_column is None:
            first = group.first_boundary_gain_h.to_numpy(dtype=float)
            second = group.second_boundary_gain_h.to_numpy(dtype=float)
        else:
            clustered = group.groupby(cluster_column)[
                ["first_boundary_gain_h", "second_boundary_gain_h"]
            ].mean()
            first = clustered.first_boundary_gain_h.to_numpy(dtype=float)
            second = clustered.second_boundary_gain_h.to_numpy(dtype=float)
        first_mean, first_low, first_high = mean_bca(
            first, BOOTSTRAP_SEED + 2 * index
        )
        second_mean, second_low, second_high = mean_bca(
            second, BOOTSTRAP_SEED + 2 * index + 1
        )
        row = {name: value for name, value in zip(group_columns, key)}
        row.update(
            {
                "samples": len(first),
                "first_boundary_gain_mean_h": first_mean,
                "first_boundary_gain_ci95_low_h": first_low,
                "first_boundary_gain_ci95_high_h": first_high,
                "second_boundary_gain_mean_h": second_mean,
                "second_boundary_gain_ci95_low_h": second_low,
                "second_boundary_gain_ci95_high_h": second_high,
                "second_boundary_nonworse_share": float(
                    (group.second_boundary_gain_h >= 0).mean()
                ),
                "start_count_delta_mean": float(group.start_count_delta.mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(parameter, physical, cross_month, sources, output_path):
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

    wide = parameter.pivot(
        index="scenario_id", columns="policy", values="second_boundary_gain_mean_h"
    )
    axes[0, 0].scatter(
        wide.guarded_blend,
        wide.rate_bounded_blend,
        s=15,
        color="#0072B2",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    bound = max(abs(wide.to_numpy()).max(), 1.0)
    axes[0, 0].plot([-bound, bound], [-bound, bound], color="#7A8288", lw=0.8, ls=":")
    axes[0, 0].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[0, 0].axvline(0, color="#6C757D", lw=0.8, ls="--")
    axes[0, 0].set_xscale("symlog", linthresh=10)
    axes[0, 0].set_yscale("symlog", linthresh=10)
    axes[0, 0].set_xlabel("Original Guarded N+1 gain (h)")
    axes[0, 0].set_ylabel("Rate-bounded N+1 gain (h)")

    scenario_styles = {
        "reference": ("#0072B2", "o", "Reference rates"),
        "heterogeneity_gp_re_increased_perm_2": (
            "#D55E00",
            "s",
            "Strong, aligned",
        ),
    }
    for scenario_id, (color, marker, label) in scenario_styles.items():
        for policy, line_style in (
            ("guarded_blend", "--"),
            ("rate_bounded_blend", "-"),
        ):
            selected = physical[
                physical.scenario_id.eq(scenario_id) & physical.policy.eq(policy)
            ].sort_values("boundary_order")
            axes[0, 1].plot(
                selected.boundary_over_calibration,
                selected.second_boundary_gain_mean_h,
                color=color,
                marker=marker,
                ms=3.4,
                lw=1.0,
                ls=line_style,
                label=(label if policy == "rate_bounded_blend" else None),
            )
    axes[0, 1].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[0, 1].set_xlabel("Boundary / LZW endpoint")
    axes[0, 1].set_ylabel("N+1 boundary gain (h)")
    axes[0, 1].legend(frameon=False, fontsize=6.2)
    axes[0, 1].text(
        0.52,
        0.96,
        "solid: rate-bounded\ndashed: original guard",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color="#5F6368",
    )

    focus = cross_month[
        cross_month.scenario_id.isin(scenario_styles)
    ].copy()
    labels = []
    means = []
    errors = []
    colors = []
    markers = []
    for scenario_id in scenario_styles:
        for policy in POLICIES:
            row = focus[
                focus.scenario_id.eq(scenario_id) & focus.policy.eq(policy)
            ].iloc[0]
            labels.append(
                ("Ref." if scenario_id == "reference" else "Strong")
                + (" / old" if policy == "guarded_blend" else " / bounded")
            )
            means.append(row.second_boundary_gain_mean_h)
            errors.append(
                (
                    row.second_boundary_gain_mean_h
                    - row.second_boundary_gain_ci95_low_h,
                    row.second_boundary_gain_ci95_high_h
                    - row.second_boundary_gain_mean_h,
                )
            )
            colors.append(scenario_styles[scenario_id][0])
            markers.append("o" if policy == "guarded_blend" else "s")
    for index, (mean, error, color, marker) in enumerate(
        zip(means, errors, colors, markers)
    ):
        axes[1, 0].errorbar(
            mean,
            index,
            xerr=np.asarray(error).reshape(2, 1),
            fmt=marker,
            color=color,
            ecolor=color,
            ms=4,
            capsize=2,
            lw=0.9,
        )
    axes[1, 0].axvline(0, color="#6C757D", lw=0.8, ls="--")
    axes[1, 0].set_yticks(range(len(labels)), labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Cross-month N+1 gain (h)")

    merged = parameter[parameter.policy.eq("guarded_blend")].merge(
        sources, on="scenario_id"
    )
    active = merged.rate_bounded_source.eq("order_blend_50")
    axes[1, 1].scatter(
        merged.loc[~active, "rate_ratio"],
        merged.loc[~active, "second_boundary_gain_mean_h"],
        s=17,
        facecolor="white",
        edgecolor="#6C757D",
        linewidth=0.7,
        label="Fallback",
    )
    axes[1, 1].scatter(
        merged.loc[active, "rate_ratio"],
        merged.loc[active, "second_boundary_gain_mean_h"],
        s=18,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.3,
        label="Blend enabled",
    )
    axes[1, 1].axvline(RATE_RATIO_LIMIT, color="#D55E00", lw=0.9, ls="--")
    axes[1, 1].axhline(0, color="#6C757D", lw=0.8, ls=":")
    axes[1, 1].set_yscale("symlog", linthresh=1, linscale=1.5)
    axes[1, 1].set_xlabel("Estimated max/min degradation rate")
    axes[1, 1].set_ylabel("Original Guarded N+1 gain (h)")
    axes[1, 1].legend(frameon=False, fontsize=6.2)

    for index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.7, w_pad=1.05, h_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    scenarios = scenario_map()
    parameter_runs, sources = materialize_rate_bounded(
        pd.read_csv(PARAMETER_DIR / "per_run_metrics.csv"), scenarios
    )
    physical_runs, _ = materialize_rate_bounded(
        pd.read_csv(PHYSICAL_DIR / "per_run_metrics.csv"), scenarios
    )
    cross_runs, _ = materialize_rate_bounded(
        pd.read_csv(CROSS_MONTH_DIR / "per_run_metrics.csv"), scenarios
    )

    parameter_paired = paired_deltas(
        parameter_runs, ["scenario_id", "local_seed"]
    )
    parameter_summary = summarize_groups(
        parameter_paired, ["scenario_id", "policy"]
    )
    physical_paired = paired_deltas(
        physical_runs, ["boundary_id", "scenario_id", "local_seed"]
    )
    physical_summary = summarize_groups(
        physical_paired, ["boundary_id", "scenario_id", "policy"]
    ).merge(
        physical_runs[
            [
                "boundary_id",
                "boundary_label",
                "boundary_order",
                "boundary_over_calibration",
            ]
        ].drop_duplicates(),
        on="boundary_id",
    )
    cross_paired = paired_deltas(
        cross_runs, ["scenario_id", "month", "local_seed"]
    )
    cross_summary = summarize_groups(
        cross_paired,
        ["scenario_id", "policy"],
        cluster_column="month",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources.to_csv(OUTPUT / "scenario_gate_sources.csv", index=False)
    parameter_paired.to_csv(OUTPUT / "parameter_paired_deltas.csv", index=False)
    parameter_summary.to_csv(OUTPUT / "parameter_summary.csv", index=False)
    physical_paired.to_csv(OUTPUT / "physical_boundary_paired_deltas.csv", index=False)
    physical_summary.to_csv(OUTPUT / "physical_boundary_summary.csv", index=False)
    cross_paired.to_csv(OUTPUT / "cross_month_paired_deltas.csv", index=False)
    cross_summary.to_csv(OUTPUT / "cross_month_summary.csv", index=False)
    figure = OUTPUT / "fig30_rate_bounded_blend_audit.png"
    plot_results(
        parameter_summary, physical_summary, cross_summary, sources, figure
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())

    robustness = []
    for policy, group in parameter_summary.groupby("policy", sort=False):
        robustness.append(
            {
                "policy": policy,
                "scenarios": len(group),
                "blend_enabled_scenarios": (
                    int(
                    (
                        sources.set_index("scenario_id")
                        .loc[group.scenario_id, "rate_bounded_source"]
                        .eq("order_blend_50")
                        .sum()
                    )
                    )
                    if policy == "rate_bounded_blend"
                    else np.nan
                ),
                "negative_n_plus_one_ci_scenarios": int(
                    (group.second_boundary_gain_ci95_high_h < 0).sum()
                ),
                "positive_first_gain_scenarios": int(
                    (group.first_boundary_gain_mean_h > 0).sum()
                ),
                "minimum_n_plus_one_mean_h": float(
                    group.second_boundary_gain_mean_h.min()
                ),
            }
        )
    robustness = pd.DataFrame(robustness)
    robustness.to_csv(OUTPUT / "robustness_summary.csv", index=False)
    metadata = {
        "scope": "post hoc replay of a nominal-envelope applicability guard",
        "rate_ratio_limit": RATE_RATIO_LIMIT,
        "threshold_source": "predeclared nominal factors 1.00/1.05/1.10",
        "threshold_tuned_on_outcomes": False,
        "new_simulations": False,
        "parameter_scenarios": int(parameter_runs.scenario_id.nunique()),
        "physical_boundaries": int(physical_runs.boundary_id.nunique()),
        "cross_months": int(cross_runs.month.nunique()),
        "future_demand_used": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report = f"""# Rate-bounded Blend replay audit

- The rate-ratio limit is `{RATE_RATIO_LIMIT:.2f}`, taken from the predeclared
  nominal factors `(1.00, 1.05, 1.10)` rather than tuned on these outcomes.
- Existing paired simulations are replayed; no trajectories are regenerated.

## Parameter robustness

{robustness.to_markdown(index=False)}

## Cross-month summary

{cross_summary.to_markdown(index=False)}
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(robustness.to_string(index=False))
    print(cross_summary.to_string(index=False))


if __name__ == "__main__":
    main()
