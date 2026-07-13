"""Audit coefficient sensitivity on frozen FC-only policy action paths."""

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
    run_policy,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/results/load_zuo_calibration"
OUTPUT = ROOT / "data/results/fc_only_mechanism_ablation"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
STRATEGIES = ("average", "rotating", "instant_health")
REFERENCES = ("average", "rotating")
INITIAL_DAMAGE_FRACTION = (0.10, 0.40, 0.80)
HETEROGENEITY_FACTORS = (1.0, 1.05, 1.10)
CONTINUOUS_MULTIPLIERS = (0.5, 1.0, 2.0)
EVENT_MULTIPLIERS = (0.0, 0.1, 0.5, 1.0, 2.0)


def load_empirical_inputs() -> tuple[np.ndarray, np.ndarray]:
    transitions = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    matrix = transitions[transitions.stride_s == 1].pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    coverage = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    probabilities = (
        coverage[coverage.stride_s == 1]
        .sort_values("state")
        .occupancy_fraction.to_numpy(dtype=float)
    )
    if matrix.shape != (4, 4) or probabilities.shape != (4,):
        raise ValueError("empirical load audit is incomplete")
    return matrix, probabilities


def run_case(task):
    model, scenario, strategy, rotation_period = task
    return run_policy(
        model, scenario, strategy, rotation_period=rotation_period
    ).metrics


def mechanism_grid(per_run: pd.DataFrame, damage_reference: float) -> pd.DataFrame:
    initial = np.asarray(INITIAL_DAMAGE_FRACTION) * damage_reference
    rows = []
    for run in per_run.to_dict(orient="records"):
        for continuous in CONTINUOUS_MULTIPLIERS:
            for start_stop in EVENT_MULTIPLIERS:
                for shift in EVENT_MULTIPLIERS:
                    increments = np.asarray(
                        [
                            continuous
                            * run[f"stack_{i}_main_continuous_damage_pct"]
                            + start_stop
                            * run[f"stack_{i}_main_start_stop_damage_pct"]
                            + shift * run[f"stack_{i}_main_shift_damage_pct"]
                            for i in range(3)
                        ]
                    )
                    rows.append(
                        {
                            "load_source": run["load_source"],
                            "load_seed": run["load_seed"],
                            "strategy": run["strategy"],
                            "continuous_multiplier": continuous,
                            "start_stop_multiplier": start_stop,
                            "shift_multiplier": shift,
                            "total_damage_pct": float(increments.sum()),
                            "aged_stack_increment_pct": float(increments[2]),
                            "max_terminal_damage_pct": float(
                                np.max(initial + increments)
                            ),
                            "increment_range_pct": float(
                                increments.max() - increments.min()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def paired_deltas(grid: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "load_source",
        "load_seed",
        "continuous_multiplier",
        "start_stop_multiplier",
        "shift_multiplier",
    ]
    metrics = (
        "total_damage_pct",
        "aged_stack_increment_pct",
        "max_terminal_damage_pct",
        "increment_range_pct",
    )
    instant = grid[grid.strategy == "instant_health"].set_index(keys)
    rows = []
    for reference in REFERENCES:
        baseline = grid[grid.strategy == reference].set_index(keys)
        for key, current in instant.iterrows():
            compared = baseline.loc[key]
            row = dict(zip(keys, key))
            row["reference_strategy"] = reference
            for metric in metrics:
                delta = float(current[metric] - compared[metric])
                row[f"{metric}_delta"] = delta
                row[f"{metric}_relative_pct"] = float(
                    100.0 * delta / max(abs(compared[metric]), 1e-12)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_sensitivity(paired: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    sources = ("empirical_1s", "zuo_slow_30s", "zuo_fast_30s")
    source_labels = ("Real-calibrated", "Zuo slow", "Zuo fast")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15), sharex=True, sharey=True)
    images = []
    for row_index, reference in enumerate(REFERENCES):
        row_values = paired[
            (paired.reference_strategy == reference)
            & np.isclose(paired.continuous_multiplier, 1.0)
        ]["total_damage_pct_relative_pct"].to_numpy()
        limit = max(5.0, float(np.nanmax(np.abs(row_values))))
        row_image = None
        for column_index, (source, label) in enumerate(zip(sources, source_labels)):
            ax = axes[row_index, column_index]
            selected = paired[
                (paired.reference_strategy == reference)
                & (paired.load_source == source)
                & np.isclose(paired.continuous_multiplier, 1.0)
            ]
            pivot = selected.pivot_table(
                index="start_stop_multiplier",
                columns="shift_multiplier",
                values="total_damage_pct_relative_pct",
                aggfunc="mean",
            ).sort_index(ascending=False)
            row_image = ax.imshow(
                pivot.to_numpy(),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            ax.set_xticks(
                range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns]
            )
            ax.set_yticks(
                range(len(pivot.index)), [f"{value:g}" for value in pivot.index]
            )
            if row_index == 0:
                ax.set_title(label)
            if row_index == 1:
                ax.set_xlabel("Load-shift multiplier")
            if column_index == 0:
                ax.set_ylabel(
                    f"Start-stop multiplier\nvs {reference.title()}"
                )
            panel = row_index * 3 + column_index
            ax.text(
                -0.13,
                1.03,
                chr(ord("a") + panel),
                transform=ax.transAxes,
                fontweight="bold",
            )
        images.append(row_image)
    for row_index, (image, reference) in enumerate(zip(images, REFERENCES)):
        colorbar = fig.colorbar(
            image, ax=axes[row_index, :], fraction=0.022, pad=0.02
        )
        colorbar.ax.set_title(
            f"Instant vs\n{reference.title()} (%)", fontsize=7.5, pad=4
        )
    fig.subplots_adjust(
        left=0.11,
        right=0.90,
        bottom=0.13,
        top=0.94,
        wspace=0.14,
        hspace=0.16,
    )
    fig.savefig(FIGURES / "fig06_mechanism_ablation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2031)))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--rotation-period", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.length <= 0 or args.jobs <= 0 or not args.seeds:
        raise ValueError("length, jobs and seeds must be positive")

    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY_FACTORS,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=5.5,
        ),
    )
    matrix, probabilities = load_empirical_inputs()
    scenarios = {
        "empirical_1s": (matrix, 1),
        "zuo_slow_30s": (np.asarray(ZUO_SLOW_TRANSITION), 30),
        "zuo_fast_30s": (np.asarray(ZUO_FAST_TRANSITION), 30),
    }
    power_reference = 0.95 * model.fc_power_reference_kw()
    tasks = []
    for seed in args.seeds:
        for source, (transition, interval) in scenarios.items():
            demand = generate_zuo_markov_system_load(
                seed,
                length_s=args.length,
                decision_interval_s=interval,
                system_power_reference_kw=power_reference,
                transition_matrix=transition,
                initial_probabilities=probabilities,
                source=source,
            )
            scenario = TestScenario(
                f"{source}_seed_{seed}",
                demand,
                INITIAL_DAMAGE_FRACTION,
                health_seed=10_000 + seed,
                stochastic_health=False,
            )
            tasks.extend(
                (model, scenario, strategy, args.rotation_period)
                for strategy in STRATEGIES
            )

    started = time.perf_counter()
    if args.jobs == 1:
        results = list(map(run_case, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(run_case, tasks, chunksize=1))
    per_run = pd.DataFrame(results)
    damage_reference = model.performance_proxies[0].mapping.damage_reference_pct
    grid = mechanism_grid(per_run, damage_reference)
    paired = paired_deltas(grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    grid.to_csv(args.out_dir / "frozen_policy_75point_grid.csv", index=False)
    paired.to_csv(args.out_dir / "instant_paired_sensitivity.csv", index=False)
    plot_sensitivity(paired)

    default = paired[
        np.isclose(paired.continuous_multiplier, 1.0)
        & np.isclose(paired.start_stop_multiplier, 1.0)
        & np.isclose(paired.shift_multiplier, 1.0)
    ]
    summary = default.groupby(["load_source", "reference_strategy"]).agg(
        total_damage_relative_pct=("total_damage_pct_relative_pct", "mean"),
        aged_stack_relative_pct=("aged_stack_increment_pct_relative_pct", "mean"),
        increment_range_relative_pct=("increment_range_pct_relative_pct", "mean"),
    )
    metadata = {
        "scope": "frozen-policy coefficient attribution; closed-loop reruns required at ranking boundaries",
        "strategies": list(STRATEGIES),
        "seeds": args.seeds,
        "length_s": args.length,
        "grid_points": 75,
        "continuous_multipliers": list(CONTINUOUS_MULTIPLIERS),
        "event_multipliers": list(EVENT_MULTIPLIERS),
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# FC-only degradation-mechanism ablation\n\n"
    report += (
        "This 75-point grid rescales continuous, start-stop and load-shift damage "
        "on frozen action paths. It diagnoses coefficient-driven conclusions and "
        "does not claim that controller actions remain optimal after rescaling.\n\n"
    )
    report += summary.to_markdown() + "\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
