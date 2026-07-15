"""Validate immediate N+1 reconfiguration after a permanent single-stack fault."""

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
OUTPUT = ROOT / "data/results/chen_n_plus_one_fault_reconfiguration"
STRATEGIES = (
    "average",
    "daisy_chain",
    "instantaneous",
    "sticky",
    "one_step_greedy",
    "break_even_hysteresis",
    "offline_dp",
)


def derive_switch_penalty(
    model: ChenDispatchModel,
    low_demand_kw: float,
) -> float:
    single = model.solve_instantaneous(low_demand_kw)
    pair = model.solve_mode(low_demand_kw, ("stack_2", "stack_3"))
    if pair is None:
        raise ValueError("strongest pair is infeasible at the low-load center")
    return (
        (pair.hydrogen_g_per_s - single.hydrogen_g_per_s)
        * 16.5
        / 2.0
    )


def choose_fault_step(load: pd.DataFrame, earliest_step: int) -> int:
    candidates = load.loc[
        load["event_boundary"]
        & load["step"].ge(earliest_step)
        & load["step"].lt(len(load) - 1)
        & load["load_state"].ge(2),
        "step",
    ]
    if candidates.empty:
        raise ValueError("load has no high-demand event boundary after fault onset")
    return int(candidates.iloc[0]) + 1


def post_fault_hydrogen_g(trajectory: pd.DataFrame, fault_step: int) -> float:
    return float(trajectory.loc[trajectory["step"].ge(fault_step), "hydrogen_g"].sum())


def summarize_fault_runs(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "incremental_post_fault_hydrogen_g",
        "incremental_post_fault_hydrogen_pct",
        "total_stack_state_changes",
        "fault_step_state_changes",
        "reserve_stack_1_post_fault_share",
        "power_balance_max_abs_kw",
        "faulted_stack_post_fault_max_kw",
    )
    rows = []
    for (strategy, failed_stack), group in per_run.groupby(
        ["strategy", "failed_stack"],
        sort=False,
    ):
        row: dict[str, float | int | str | bool] = {
            "strategy": strategy,
            "failed_stack": failed_stack,
            "n_seeds": len(group),
            "all_power_balanced": bool(
                group["power_balance_max_abs_kw"].lt(1e-8).all()
            ),
            "all_faulted_power_zero": bool(
                group["faulted_stack_post_fault_max_kw"].lt(1e-10).all()
            ),
            "all_immediate_reconfiguration": bool(
                group["reconfiguration_delay_steps"].eq(0).all()
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_offline_lower_bound(per_run: pd.DataFrame) -> None:
    for (seed, failed_stack), group in per_run.groupby(["seed", "failed_stack"]):
        offline = float(
            group.loc[
                group["strategy"].eq("offline_dp"),
                "total_evaluated_objective_g",
            ].iloc[0]
        )
        online = group.loc[
            ~group["strategy"].eq("offline_dp"),
            "total_evaluated_objective_g",
        ].to_numpy(dtype=float)
        if np.any(offline > online + 1e-9):
            raise AssertionError(
                f"offline DP is not a lower bound for seed {seed}, {failed_stack}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=900)
    parser.add_argument("--fault-after-step", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3026, 3036)))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (
        args.length <= 0
        or args.fault_after_step <= 0
        or args.fault_after_step >= args.length
        or not args.seeds
    ):
        raise ValueError("invalid length, fault step, or seed list")

    curves = pd.read_csv(CURVES)
    model = ChenDispatchModel(curves)
    levels = derive_chen_load_levels(curves)
    switch_penalty = derive_switch_penalty(model, levels.single_peak_kw)
    metric_rows = []
    trajectory_rows: dict[tuple[int, str], list[pd.DataFrame]] = {}

    for seed in args.seeds:
        load = generate_chen_random_dynamic_load(
            seed,
            length_s=args.length,
            levels=levels,
            transition_matrix=ZUO_FAST_TRANSITION,
            dwell_range_s=(8, 25),
            target_variation_fraction=0.03,
        )
        demand = load["demand_net_power_kw"].to_numpy(dtype=float)
        fault_step = choose_fault_step(load, args.fault_after_step)
        healthy_tables = precompute_chen_solution_tables(model, demand)
        healthy_runs = {
            strategy: run_chen_policy(
                model,
                demand,
                strategy,
                switch_penalty_g_per_change=switch_penalty,
                solution_tables=healthy_tables,
            )
            for strategy in STRATEGIES
        }

        for failed_stack in model.stack_ids:
            remaining = tuple(
                stack_id
                for stack_id in model.stack_ids
                if stack_id != failed_stack
            )
            availability = [
                model.stack_ids if step < fault_step else remaining
                for step in range(args.length)
            ]
            fault_tables = precompute_chen_solution_tables(
                model,
                demand,
                available_stack_ids_by_step=availability,
            )
            for strategy in STRATEGIES:
                healthy = healthy_runs[strategy]
                faulted = run_chen_policy(
                    model,
                    demand,
                    strategy,
                    switch_penalty_g_per_change=switch_penalty,
                    solution_tables=fault_tables,
                )
                healthy_post_hydrogen = post_fault_hydrogen_g(
                    healthy.trajectory,
                    fault_step,
                )
                faulted_post_hydrogen = post_fault_hydrogen_g(
                    faulted.trajectory,
                    fault_step,
                )
                failed_power_column = f"{failed_stack}_net_power_kw"
                failed_post_power = faulted.trajectory.loc[
                    faulted.trajectory["step"].ge(fault_step),
                    failed_power_column,
                ]
                isolated_steps = failed_post_power.abs().le(1e-10).to_numpy()
                reconfiguration_delay = int(np.argmax(isolated_steps))
                if not isolated_steps.any():
                    reconfiguration_delay = args.length - fault_step
                pre_fault_active = bool(
                    healthy.trajectory.loc[
                        fault_step - 1,
                        failed_power_column,
                    ]
                    > 1e-10
                )
                post_fault_rows = faulted.trajectory["step"].ge(fault_step)
                stack_1_share = float(
                    faulted.trajectory.loc[
                        post_fault_rows,
                        "stack_1_net_power_kw",
                    ].gt(1e-10).mean()
                )
                row = dict(faulted.metrics)
                row.update(
                    {
                        "seed": seed,
                        "failed_stack": failed_stack,
                        "fault_step": fault_step,
                        "fault_demand_net_power_kw": float(demand[fault_step]),
                        "faulted_stack_active_before_fault": pre_fault_active,
                        "faulted_stack_post_fault_max_kw": float(
                            failed_post_power.abs().max()
                        ),
                        "reconfiguration_delay_steps": reconfiguration_delay,
                        "fault_step_state_changes": int(
                            faulted.trajectory.loc[
                                fault_step,
                                "stack_state_changes",
                            ]
                        ),
                        "reserve_stack_1_post_fault_share": stack_1_share,
                        "healthy_post_fault_hydrogen_g": healthy_post_hydrogen,
                        "faulted_post_fault_hydrogen_g": faulted_post_hydrogen,
                        "incremental_post_fault_hydrogen_g": (
                            faulted_post_hydrogen - healthy_post_hydrogen
                        ),
                        "incremental_post_fault_hydrogen_pct": 100.0
                        * (faulted_post_hydrogen / healthy_post_hydrogen - 1.0),
                    }
                )
                metric_rows.append(row)

                if strategy in ("break_even_hysteresis", "instantaneous"):
                    trajectory = faulted.trajectory.copy()
                    trajectory["seed"] = seed
                    trajectory["failed_stack"] = failed_stack
                    trajectory["fault_step"] = fault_step
                    trajectory["fault_active"] = trajectory["step"].ge(fault_step)
                    trajectory["load_state"] = load["load_state"].to_numpy()
                    trajectory_rows.setdefault((seed, failed_stack), []).append(
                        trajectory
                    )

    per_run = pd.DataFrame(metric_rows)
    validate_offline_lower_bound(per_run)
    aggregate = summarize_fault_runs(per_run)
    if not aggregate["all_power_balanced"].all():
        raise AssertionError("at least one fault run failed power balance")
    if not aggregate["all_faulted_power_zero"].all():
        raise AssertionError("a failed stack produced power after isolation")

    representative_candidates = per_run.loc[
        per_run["strategy"].eq("break_even_hysteresis")
        & per_run["failed_stack"].eq("stack_3")
    ].copy()
    median_increment = float(
        representative_candidates["incremental_post_fault_hydrogen_pct"].median()
    )
    representative_seed = int(
        representative_candidates.loc[
            (
                representative_candidates["incremental_post_fault_hydrogen_pct"]
                - median_increment
            ).abs().idxmin(),
            "seed",
        ]
    )
    representative = pd.concat(
        trajectory_rows[(representative_seed, "stack_3")],
        ignore_index=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_fault_run_metrics.csv", index=False)
    aggregate.to_csv(
        args.out_dir / "aggregate_by_strategy_and_fault.csv",
        index=False,
    )
    representative.to_csv(
        args.out_dir / "representative_stack_3_fault_trajectory.csv",
        index=False,
    )
    metadata = {
        "scope": "permanent single-stack fault with immediate perfect isolation",
        "fault_timing": (
            "one step after the first high-load event boundary at or after "
            "the configured onset, so online high-load allocation is active first"
        ),
        "fault_after_step": args.fault_after_step,
        "failed_stack_identities": list(model.stack_ids),
        "guaranteed_n_plus_one_power_kw": levels.guaranteed_n_plus_one_power_kw,
        "maximum_two_stack_power_kw": levels.maximum_two_stack_power_kw,
        "maximum_fault_injection_demand_kw": float(
            per_run["fault_demand_net_power_kw"].max()
        ),
        "controller_fault_information": "available in the fault step",
        "fault_detection_delay_modeled": False,
        "fault_recovery_modeled": False,
        "future_demand_used_by_online_strategies": False,
        "switch_penalty_g_per_change": switch_penalty,
        "seeds": args.seeds,
        "representative": {
            "failed_stack": "stack_3",
            "seed": representative_seed,
            "selection": "closest to median post-fault hydrogen increment",
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
