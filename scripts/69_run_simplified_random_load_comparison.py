"""Compare simplified three-stack allocation policies on random dynamic loads."""

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
    TestScenario,
    ZUO_FAST_TRANSITION,
    ZUO_SLOW_TRANSITION,
    generate_zuo_markov_system_load,
    paired_strategy_comparison,
    run_policy,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/load_zuo_calibration_norm40"
OUTPUT = ROOT / "data/results/simplified_random_load_comparison"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"

STRATEGIES = (
    "average",
    "daisy_chain",
    "rotating",
    "instant_no_health",
    "instant_health",
)
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)
SYSTEM_POWER_REFERENCE_KW = 40.0
TRACKING_TOLERANCE_KW = 5.5
PAIR_METRICS = (
    "max_stack_damage_increment_pct",
    "main_expected_damage_increment_pct",
    "damage_increment_range_pct",
    "hydrogen_g_per_fc_kwh",
    "fc_tracking_mae_kw",
    "total_switch_count",
    "online_step_range",
)


def load_empirical_matrix() -> np.ndarray:
    table = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    selected = table[table.stride_s.eq(1)]
    matrix = selected.pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("one-second empirical transition matrix is incomplete")
    for row in range(4):
        if not np.all(np.isfinite(matrix[row])):
            matrix[row] = 0.0
            matrix[row, row] = 1.0
    return matrix


def load_initial_probabilities() -> np.ndarray:
    table = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    selected = table[table.stride_s.eq(1)].sort_values("state")
    probabilities = selected.occupancy_fraction.to_numpy(dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    if probabilities.shape != (4,) or probabilities.sum() <= 0:
        raise ValueError("one-second state occupancy is incomplete")
    return probabilities / probabilities.sum()


def build_model():
    return load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY_FACTORS,
        config=WorldModelConfig(
            allowed_currents_a=(0.0, 25.0, 60.0, 90.0, 120.0, 160.0, 195.0, 270.0, 370.0),
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )


def build_scenarios(length_s: int):
    return {
        "empirical_random_1s": (load_empirical_matrix(), 1),
        "zuo_slow_random_30s": (np.asarray(ZUO_SLOW_TRANSITION), 30),
        "zuo_fast_random_30s": (np.asarray(ZUO_FAST_TRANSITION), 30),
    }


def run_case(task):
    model, scenario, strategy, rotation_period, keep_trajectory = task
    run = run_policy(
        model,
        scenario,
        strategy,
        rotation_period=rotation_period,
    )
    metrics = dict(run.metrics)
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    initial = np.asarray(INITIAL_DAMAGE_FRACTION, dtype=float) * reference
    final = np.asarray(
        [metrics[f"stack_{index}_final_damage_pct"] for index in range(3)],
        dtype=float,
    )
    increments = final - initial
    metrics.update(
        {
            "max_stack_damage_increment_pct": float(increments.max()),
            "mean_stack_damage_increment_pct": float(increments.mean()),
            "min_stack_damage_increment_pct": float(increments.min()),
            "final_max_damage_pct": float(final.max()),
            "pair_seed": metrics["load_seed"],
        }
    )
    trajectory = None
    if keep_trajectory:
        columns = [
            "step",
            "demand_power_kw",
            "stack_power_kw",
            "strategy",
        ] + [f"stack_{index}_current_a" for index in range(3)]
        trajectory = run.trajectory.loc[:, columns].copy()
    return metrics, trajectory


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = PAIR_METRICS + (
        "mean_stack_damage_increment_pct",
        "final_max_damage_pct",
        "planning_runtime_s",
    )
    rows = []
    for (source, strategy), group in per_run.groupby(
        ["load_source", "strategy"], sort=False
    ):
        row = {
            "load_source": source,
            "strategy": strategy,
            "n_pairs": len(group),
            "zero_violation_share": float(
                group.constraint_violation_steps.eq(0).mean()
            ),
            "tracking_within_tolerance_share": float(
                group.fc_tracking_within_tolerance_share.mean()
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_closed_loop(per_run: pd.DataFrame, expected_runs: int, length_s: int) -> dict:
    checks = {
        "run_rows": int(len(per_run)),
        "expected_run_rows": int(expected_runs),
        "wrong_step_count_runs": int(per_run.n_steps.ne(length_s).sum()),
        "constraint_violation_runs": int(
            per_run.constraint_violation_steps.ne(0).sum()
        ),
        "tracking_failure_runs": int(
            per_run.fc_tracking_within_tolerance_share.lt(1.0).sum()
        ),
        "wrong_online_count_runs": int(
            (
                ~np.isclose(per_run.online_stack_count_mean, 2.0)
                | per_run.online_stack_count_max.ne(2)
            ).sum()
        ),
        "missing_health_update_runs": int(
            per_run.health_changed_steps.ne(length_s).sum()
        ),
        "nonpositive_damage_runs": int(
            per_run.main_expected_damage_increment_pct.le(0.0).sum()
        ),
        "clipped_runs": int(per_run.clipped_points.ne(0).sum()),
        "safety_override_runs": int(per_run.safety_override_steps.ne(0).sum()),
    }
    failed = [
        key
        for key, value in checks.items()
        if key not in {"run_rows", "expected_run_rows"} and value != 0
    ]
    if checks["run_rows"] != checks["expected_run_rows"]:
        failed.append("run_rows")
    if failed:
        raise AssertionError(
            "closed-loop route validation failed: " + ", ".join(failed)
        )
    checks["status"] = "passed"
    return checks


def plot_results(
    example: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired_average: pd.DataFrame,
    health_ablation: pd.DataFrame,
    output_path: Path,
) -> None:
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
    colors = {
        "average": "#7A7A7A",
        "daisy_chain": "#E69F00",
        "rotating": "#56B4E9",
        "instant_no_health": "#CC79A7",
        "instant_health": "#009E73",
    }
    labels = {
        "average": "Average",
        "daisy_chain": "DC-average",
        "rotating": "Rotating",
        "instant_no_health": "Instant, no health",
        "instant_health": "Instant, health-aware",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.05))

    dc = example[example.strategy.eq("daisy_chain")]
    axes[0, 0].step(
        dc.step,
        dc.demand_power_kw,
        where="post",
        color="#2F3437",
        lw=1.1,
        label="Demand",
    )
    for index, color in enumerate(("#0072B2", "#D55E00", "#009E73")):
        axes[0, 0].step(
            dc.step,
            dc[f"stack_{index}_current_a"] / 10.0,
            where="post",
            color=color,
            lw=0.9,
            label=f"Stack {index + 1} current / 10",
        )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Demand (kW) or scaled current")
    axes[0, 0].legend(frameon=False, fontsize=5.8, ncol=2)

    selected = paired_average[
        paired_average.metric.eq("max_stack_damage_increment_pct")
    ].copy()
    sources = list(aggregate.load_source.drop_duplicates())
    candidates = [strategy for strategy in STRATEGIES if strategy != "average"]
    width = 0.18
    x = np.arange(len(sources))
    for offset, strategy in enumerate(candidates):
        rows = selected[selected.strategy.eq(strategy)].set_index("load_source")
        values = [rows.loc[source, "mean_relative_pct"] for source in sources]
        errors = [
            100
            * rows.loc[source, "ci95"]
            / max(
                abs(
                    aggregate[
                        aggregate.load_source.eq(source)
                        & aggregate.strategy.eq("average")
                    ].max_stack_damage_increment_pct_mean.iloc[0]
                ),
                1e-12,
            )
            for source in sources
        ]
        axes[0, 1].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=colors[strategy],
            label=labels[strategy],
        )
    axes[0, 1].axhline(0, color="#6C757D", lw=0.8)
    axes[0, 1].set_xticks(x, ["Empirical", "Zuo slow", "Zuo fast"])
    axes[0, 1].set_ylabel("Max-stack damage vs Average (%)")
    axes[0, 1].legend(frameon=False, fontsize=5.7, ncol=2)

    marker = {sources[0]: "o", sources[1]: "s", sources[2]: "^"}
    for strategy in STRATEGIES:
        rows = aggregate[aggregate.strategy.eq(strategy)]
        for row in rows.itertuples():
            axes[1, 0].scatter(
                row.hydrogen_g_per_fc_kwh_mean,
                row.max_stack_damage_increment_pct_mean * 1e3,
                s=27,
                marker=marker[row.load_source],
                color=colors[strategy],
                edgecolor="white",
                linewidth=0.35,
            )
    for strategy in STRATEGIES:
        axes[1, 0].scatter([], [], color=colors[strategy], label=labels[strategy])
    axes[1, 0].set_xlabel("Hydrogen intensity (g/kWh)")
    axes[1, 0].set_ylabel("Max-stack damage increment (x10$^{-3}$ pp)")
    axes[1, 0].legend(frameon=False, fontsize=5.5, ncol=2)

    ablation_metrics = [
        ("max_stack_damage_increment_pct", "Max-stack damage"),
        ("damage_increment_range_pct", "Damage imbalance"),
        ("hydrogen_g_per_fc_kwh", "Hydrogen intensity"),
        ("fc_tracking_mae_kw", "Tracking MAE"),
    ]
    health = health_ablation[health_ablation.strategy.eq("instant_health")]
    y = np.arange(len(ablation_metrics))
    scenario_offsets = (-0.18, 0.0, 0.18)
    scenario_colors = ("#0072B2", "#E69F00", "#D55E00")
    for source, offset, color in zip(sources, scenario_offsets, scenario_colors):
        indexed = health[health.load_source.eq(source)].set_index("metric")
        values = [indexed.loc[metric, "mean_relative_pct"] for metric, _ in ablation_metrics]
        axes[1, 1].scatter(
            values,
            y + offset,
            s=23,
            color=color,
            label={sources[0]: "Empirical", sources[1]: "Zuo slow", sources[2]: "Zuo fast"}[source],
        )
    axes[1, 1].axvline(0, color="#6C757D", lw=0.8, ls="--")
    axes[1, 1].set_yticks(y, [label for _, label in ablation_metrics])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Health-aware vs no-health objective (%)")
    axes[1, 1].legend(frameon=False, fontsize=6.0)

    for index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=180)
    parser.add_argument(
        "--pair-seeds", nargs="+", type=int, default=list(range(2026, 2036))
    )
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.length, args.rotation_period, args.jobs) <= 0:
        raise ValueError("length, rotation period and jobs must be positive")
    if not args.pair_seeds or len(set(args.pair_seeds)) != len(args.pair_seeds):
        raise ValueError("pair seeds must be non-empty and unique")

    model = build_model()
    scenario_specs = build_scenarios(args.length)
    initial_probabilities = load_initial_probabilities()
    tasks = []
    first_seed = args.pair_seeds[0]
    for pair_seed in args.pair_seeds:
        for source, (matrix, interval_s) in scenario_specs.items():
            demand = generate_zuo_markov_system_load(
                pair_seed,
                length_s=args.length,
                decision_interval_s=interval_s,
                system_power_reference_kw=SYSTEM_POWER_REFERENCE_KW,
                transition_matrix=matrix,
                initial_probabilities=initial_probabilities,
                source=source,
            )
            scenario = TestScenario(
                name=f"{source}_pair_{pair_seed}",
                demand=demand,
                initial_damage_fraction=INITIAL_DAMAGE_FRACTION,
                health_seed=40_000 + pair_seed,
                stochastic_health=False,
            )
            for strategy in STRATEGIES:
                keep = pair_seed == first_seed and source == "zuo_fast_random_30s"
                tasks.append((model, scenario, strategy, args.rotation_period, keep))

    started = time.perf_counter()
    metric_rows = []
    example_rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for metrics, trajectory in executor.map(run_case, tasks, chunksize=1):
            metric_rows.append(metrics)
            if trajectory is not None:
                example_rows.append(trajectory)
            print(
                f"completed {metrics['load_source']} seed={metrics['load_seed']} "
                f"strategy={metrics['strategy']}",
                flush=True,
            )

    per_run = pd.DataFrame(metric_rows)
    expected_runs = len(args.pair_seeds) * len(scenario_specs) * len(STRATEGIES)
    if len(per_run) != expected_runs:
        raise AssertionError("paired random-load runs are incomplete")
    pairing = per_run.groupby(["load_source", "load_seed"]).strategy.nunique()
    if not pairing.eq(len(STRATEGIES)).all():
        raise AssertionError("each load seed must contain every strategy")
    route_validation = validate_closed_loop(per_run, expected_runs, args.length)

    aggregate = summarize(per_run)
    paired_average = paired_strategy_comparison(
        per_run,
        reference_strategy="average",
        metrics=PAIR_METRICS,
    )
    health_ablation = paired_strategy_comparison(
        per_run,
        reference_strategy="instant_no_health",
        metrics=PAIR_METRICS,
    )
    health_ablation = health_ablation[
        health_ablation.strategy.eq("instant_health")
    ].reset_index(drop=True)
    example = pd.concat(example_rows, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    paired_average.to_csv(args.out_dir / "paired_vs_average.csv", index=False)
    health_ablation.to_csv(args.out_dir / "paired_health_ablation.csv", index=False)
    example.to_csv(args.out_dir / "example_trajectory.csv", index=False)
    figure = args.out_dir / "fig32_simplified_random_load_comparison.png"
    plot_results(example, aggregate, paired_average, health_ablation, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())

    decision = {
        "scope": "first-pass closed-loop comparison on random dynamic loads",
        "strategies": list(STRATEGIES),
        "random_load_scenarios": list(scenario_specs),
        "paired_seeds": args.pair_seeds,
        "health_model": (
            "deterministic action-driven engineering proxy; online D-to-theta-to-IV "
            "updates retained; no precise physical-rate claim"
        ),
        "health_ablation": (
            "same plant health evolution; planner degradation and performance-loss "
            "weights disabled only for instant_no_health"
        ),
        "zuo_role": "system N+1 logic and slow/fast random-load matrices only",
        "action_grid_note": (
            "60 A is an IV-model interpolation point added between audited 25 A and "
            "90 A points so equal-load baselines can track the 16.57 kW low state"
        ),
        "future_demand_used": False,
        "battery_used": False,
        "closed_loop_validation": route_validation,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    ablation = health_ablation.set_index(["load_source", "metric"])
    report_lines = []
    for source in scenario_specs:
        damage = ablation.loc[(source, "max_stack_damage_increment_pct")]
        imbalance = ablation.loc[(source, "damage_increment_range_pct")]
        hydrogen = ablation.loc[(source, "hydrogen_g_per_fc_kwh")]
        report_lines.append(
            f"- {source}: health-aware vs no-health max-stack damage "
            f"{damage.mean_relative_pct:+.2f}%, damage imbalance "
            f"{imbalance.mean_relative_pct:+.2f}%, hydrogen intensity "
            f"{hydrogen.mean_relative_pct:+.2f}%."
        )
    report = f"""# Simplified random-load allocation comparison

This is the first-pass chain-completion experiment. The degradation block is
kept as an online engineering proxy. It is not presented as a precise physical
rate calibration.

## Setup

- Three stacks, exactly two online, FC-only power interface.
- Random loads: empirical 1 s Markov, Zuo slow 30 s, Zuo fast 30 s.
- Strategies: Average, Zuo-style DC-average, Rotating, Instant without health
  terms, and Instant with health terms.
- All strategies execute the same health state transition. The ablation changes
  only what the planner reads.

## Health-objective ablation

{chr(10).join(report_lines)}

See `aggregate_metrics.csv`, `paired_vs_average.csv`, and
`paired_health_ablation.csv` for paired results and uncertainty.
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(health_ablation.to_string(index=False))


if __name__ == "__main__":
    main()
