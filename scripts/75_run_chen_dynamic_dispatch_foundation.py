"""Run the corrected Chen dynamic-dispatch foundation without plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation.chen_dynamic_load import (
    derive_chen_load_levels,
    generate_chen_random_dynamic_load,
)
from fc_power.evaluation.zuo_load_calibration import ZUO_FAST_TRANSITION
from fc_power.power_allocation.chen_dispatch import ChenDispatchModel
from fc_power.power_allocation.chen_dispatch_policies import (
    precompute_chen_solution_tables,
    run_chen_policy,
)


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data/processed/chen_efficiency_curves_audited.csv"
OUTPUT = ROOT / "data/results/chen_dynamic_dispatch_foundation"
STRATEGIES = (
    "average",
    "daisy_chain",
    "instantaneous",
    "sticky",
    "one_step_greedy",
    "break_even_hysteresis",
    "offline_dp",
)
METRICS = (
    "total_hydrogen_g",
    "hydrogen_g_per_net_kwh",
    "energy_weighted_efficiency_lhv_pct",
    "total_stack_state_changes",
    "total_evaluated_objective_g",
)


def derive_break_even_switch_penalty(
    model: ChenDispatchModel,
    low_demand_kw: float,
    median_dwell_s: float,
) -> tuple[float, dict[str, float]]:
    instantaneous = model.solve_instantaneous(low_demand_kw)
    pair = model.solve_mode(low_demand_kw, ("stack_2", "stack_3"))
    if pair is None:
        raise ValueError("the strongest pair is infeasible at the low load center")
    extra_hydrogen_g_per_s = (
        pair.hydrogen_g_per_s - instantaneous.hydrogen_g_per_s
    )
    if extra_hydrogen_g_per_s <= 0:
        raise ValueError("low-load two-stack operation must cost more hydrogen")
    penalty = extra_hydrogen_g_per_s * median_dwell_s / 2.0
    return penalty, {
        "low_demand_kw": low_demand_kw,
        "instantaneous_mode": "+".join(instantaneous.mode),
        "pair_mode": "+".join(pair.mode),
        "extra_hydrogen_g_per_s": extra_hydrogen_g_per_s,
        "median_dwell_s": median_dwell_s,
        "round_trip_stack_changes": 2.0,
        "break_even_penalty_g_per_change": penalty,
    }


def aggregate_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in per_run.groupby("strategy", sort=False):
        row: dict[str, float | int | str] = {
            "strategy": strategy,
            "n_seeds": len(group),
            "all_power_balanced": bool(
                group["power_balance_max_abs_kw"].lt(1e-8).all()
            ),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(
    per_run: pd.DataFrame,
    reference_strategy: str,
) -> pd.DataFrame:
    reference = per_run[per_run.strategy.eq(reference_strategy)].set_index("seed")
    rows = []
    for strategy, group in per_run.groupby("strategy", sort=False):
        if strategy == reference_strategy:
            continue
        candidate = group.set_index("seed")
        if not candidate.index.equals(reference.index):
            candidate = candidate.reindex(reference.index)
        for metric in METRICS:
            difference = (
                candidate[metric].to_numpy(dtype=float)
                - reference[metric].to_numpy(dtype=float)
            )
            relative = 100.0 * difference / reference[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "reference_strategy": reference_strategy,
                    "strategy": strategy,
                    "metric": metric,
                    "mean_difference": float(difference.mean()),
                    "difference_std": float(difference.std(ddof=1)),
                    "difference_ci95": float(
                        1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
                    ),
                    "mean_relative_pct": float(relative.mean()),
                    "relative_pct_std": float(relative.std(ddof=1)),
                    "relative_pct_ci95": float(
                        1.96 * relative.std(ddof=1) / np.sqrt(len(relative))
                    ),
                    "win_share": float((difference < 0).mean()),
                    "n_pairs": len(difference),
                }
            )
    return pd.DataFrame(rows)


def validate_offline_lower_bound(per_run: pd.DataFrame) -> None:
    for seed, group in per_run.groupby("seed"):
        offline = float(
            group.loc[
                group.strategy.eq("offline_dp"),
                "total_evaluated_objective_g",
            ].iloc[0]
        )
        online = group.loc[
            ~group.strategy.eq("offline_dp"),
            "total_evaluated_objective_g",
        ].to_numpy(dtype=float)
        if np.any(offline > online + 1e-9):
            raise AssertionError(f"offline DP is not a lower bound for seed {seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=900)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--split-label", default="development")
    args = parser.parse_args()
    if args.length <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("length must be positive and seeds must be non-empty and unique")

    curves = pd.read_csv(CURVES)
    model = ChenDispatchModel(curves)
    levels = derive_chen_load_levels(curves)
    dwell_range_s = (8, 25)
    median_dwell_s = float(np.mean(dwell_range_s))
    main_penalty, penalty_derivation = derive_break_even_switch_penalty(
        model,
        levels.single_peak_kw,
        median_dwell_s,
    )
    sensitivity_penalties = [
        0.0,
        0.5 * main_penalty,
        main_penalty,
        2.0 * main_penalty,
        4.0 * main_penalty,
    ]

    metric_rows = []
    sensitivity_rows = []
    trajectory_rows_by_seed: dict[int, list[pd.DataFrame]] = {}
    for seed in args.seeds:
        load = generate_chen_random_dynamic_load(
            seed,
            length_s=args.length,
            levels=levels,
            transition_matrix=ZUO_FAST_TRANSITION,
            dwell_range_s=dwell_range_s,
            target_variation_fraction=0.03,
        )
        demand = load["demand_net_power_kw"].to_numpy(dtype=float)
        solution_tables = precompute_chen_solution_tables(model, demand)
        for strategy in STRATEGIES:
            run = run_chen_policy(
                model,
                demand,
                strategy,
                switch_penalty_g_per_change=main_penalty,
                solution_tables=solution_tables,
            )
            row = dict(run.metrics)
            row["seed"] = seed
            row["split_label"] = args.split_label
            row["load_event_count"] = int(load.event_id.nunique())
            row["unique_demand_count"] = int(load.demand_net_power_kw.nunique())
            metric_rows.append(row)
            trajectory = run.trajectory.copy()
            trajectory["seed"] = seed
            trajectory["load_state"] = load["load_state"].to_numpy()
            trajectory["event_id"] = load["event_id"].to_numpy()
            trajectory["event_boundary"] = load["event_boundary"].to_numpy()
            trajectory_rows_by_seed.setdefault(seed, []).append(trajectory)

        for penalty in sensitivity_penalties:
            for strategy in (
                "one_step_greedy",
                "break_even_hysteresis",
                "offline_dp",
            ):
                run = run_chen_policy(
                    model,
                    demand,
                    strategy,
                    switch_penalty_g_per_change=penalty,
                    solution_tables=solution_tables,
                )
                row = dict(run.metrics)
                row["seed"] = seed
                row["split_label"] = args.split_label
                row["penalty_multiple"] = (
                    penalty / main_penalty if main_penalty > 0 else 0.0
                )
                sensitivity_rows.append(row)

    per_run = pd.DataFrame(metric_rows)
    validate_offline_lower_bound(per_run)
    aggregate = aggregate_metrics(per_run)
    vs_average = paired_comparison(per_run, "average")
    vs_instantaneous = paired_comparison(per_run, "instantaneous")
    sensitivity = pd.DataFrame(sensitivity_rows)
    changes = per_run.pivot(
        index="seed",
        columns="strategy",
        values="total_stack_state_changes",
    )
    changes["hysteresis_reduction"] = (
        changes["instantaneous"] - changes["break_even_hysteresis"]
    )
    median_reduction = float(changes["hysteresis_reduction"].median())
    representative_seed = int(
        (changes["hysteresis_reduction"] - median_reduction).abs().idxmin()
    )
    representative = pd.concat(
        trajectory_rows_by_seed[representative_seed],
        ignore_index=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    vs_average.to_csv(args.out_dir / "paired_vs_average.csv", index=False)
    vs_instantaneous.to_csv(
        args.out_dir / "paired_vs_instantaneous.csv",
        index=False,
    )
    sensitivity.to_csv(args.out_dir / "switch_penalty_sensitivity.csv", index=False)
    representative.to_csv(args.out_dir / "representative_trajectory.csv", index=False)

    metadata = {
        "scope": "corrected Chen net-power/LHV dynamic dispatch foundation",
        "curve_source": str(CURVES.relative_to(ROOT)),
        "load_amplitude_source": {
            "single_peak_kw": levels.single_peak_kw,
            "dual_peak_kw": levels.dual_peak_kw,
            "high_load_kw": levels.high_load_kw,
            "reserve_peak_kw": levels.reserve_peak_kw,
            "n_plus_one_max_kw": levels.n_plus_one_max_kw,
        },
        "random_load": {
            "transition_matrix_role": "Zuo fast matrix controls event order only",
            "amplitudes_use_zuo_levels": False,
            "dwell_range_s": list(dwell_range_s),
            "dwell_status": "engineering switching-stress setting, not identified vehicle physics",
            "target_variation_fraction": 0.03,
            "physical_ramp_constraint_used": False,
        },
        "main_switch_penalty": penalty_derivation,
        "hysteresis_thresholds": {
            "start_multiplier": 1.0,
            "stop_multiplier": 2.0,
            "stop_multiplier_reason": (
                "an optional stop must repay the current stop and one expected "
                "future restart; this is a two-state-change round trip"
            ),
        },
        "switch_penalty_interpretation": (
            "normalized equivalent-hydrogen trade-off parameter, not a physical degradation coefficient"
        ),
        "strategies": list(STRATEGIES),
        "seeds": args.seeds,
        "split_label": args.split_label,
        "representative_trajectory": {
            "seed": representative_seed,
            "selection": (
                "seed whose online-hysteresis stack-state-change reduction is "
                "closest to the split median"
            ),
            "median_change_reduction": median_reduction,
        },
        "length_s": args.length,
        "future_demand_used_by_online_strategies": False,
        "offline_dp_role": "post-hoc lower bound using the complete demand sequence",
        "battery_used": False,
        "degradation_used": False,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print("\nPaired vs instantaneous:")
    print(vs_instantaneous.to_string(index=False))


if __name__ == "__main__":
    main()
