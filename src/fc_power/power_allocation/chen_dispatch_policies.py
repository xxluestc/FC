"""Auditable dynamic policies built on the exact Chen inner dispatch oracle."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from fc_power.power_allocation.chen_dispatch import (
    ChenDispatchModel,
    ChenDispatchSolution,
    changed_stack_states,
)


@dataclass(frozen=True)
class ChenPolicyRun:
    strategy: str
    trajectory: pd.DataFrame
    metrics: dict[str, float | int | str]


def precompute_chen_solution_tables(
    model: ChenDispatchModel,
    demand_net_power_kw,
) -> list[dict[tuple[str, ...], ChenDispatchSolution]]:
    demand = np.asarray(demand_net_power_kw, dtype=float)
    if demand.ndim != 1 or len(demand) == 0 or np.any(~np.isfinite(demand)):
        raise ValueError("demand_net_power_kw must be a finite non-empty vector")
    tables = [model.solve_all_modes(value) for value in demand]
    if any(not table for table in tables):
        first = next(index for index, table in enumerate(tables) if not table)
        raise ValueError(f"demand at step {first} has no feasible N+1 dispatch")
    return tables


def run_chen_policy(
    model: ChenDispatchModel,
    demand_net_power_kw,
    strategy: str,
    *,
    dt_s: float = 1.0,
    switch_penalty_g_per_change: float = 0.0,
    solution_tables: list[dict[tuple[str, ...], ChenDispatchSolution]] | None = None,
    hysteresis_start_multiplier: float = 1.0,
    hysteresis_stop_multiplier: float = 2.0,
) -> ChenPolicyRun:
    """Run one online policy or the offline DP comparator on a demand vector.

    The default stop threshold represents a complete stop/restart round trip:
    an optional stop is delayed until accumulated hydrogen savings repay both
    the current stop and one expected future restart.
    """

    demand = np.asarray(demand_net_power_kw, dtype=float)
    if demand.ndim != 1 or len(demand) == 0 or np.any(~np.isfinite(demand)):
        raise ValueError("demand_net_power_kw must be a finite non-empty vector")
    if np.any(demand < 0):
        raise ValueError("Chen FC-only dispatch does not accept negative demand")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(switch_penalty_g_per_change) or switch_penalty_g_per_change < 0:
        raise ValueError("switch penalty must be finite and non-negative")
    if (
        not np.isfinite(hysteresis_start_multiplier)
        or not np.isfinite(hysteresis_stop_multiplier)
        or hysteresis_start_multiplier < 0
        or hysteresis_stop_multiplier < 0
    ):
        raise ValueError("hysteresis multipliers must be finite and non-negative")
    allowed = {
        "average",
        "daisy_chain",
        "instantaneous",
        "sticky",
        "one_step_greedy",
        "break_even_hysteresis",
        "offline_dp",
    }
    if strategy not in allowed:
        raise ValueError(f"unknown Chen dispatch strategy: {strategy}")

    started = time.perf_counter()
    if solution_tables is None:
        solution_tables = precompute_chen_solution_tables(model, demand)
    elif len(solution_tables) != len(demand) or any(not table for table in solution_tables):
        raise ValueError("precomputed solution tables do not match the demand vector")
    else:
        for value, table in zip(demand, solution_tables):
            sample = next(iter(table.values()))
            if not np.isclose(sample.demand_net_power_kw, value, atol=1e-8):
                raise ValueError("precomputed solution table demand is misaligned")

    if strategy == "offline_dp":
        selected = _offline_dynamic_programming(
            solution_tables,
            dt_s,
            switch_penalty_g_per_change,
        )
    else:
        selected = []
        previous_mode: tuple[str, ...] = ()
        cumulative_advantage_g: dict[tuple[str, ...], float] = {}
        average_pair = _strongest_pair(model)
        daisy_order = _efficiency_order(model)
        for value, solutions in zip(demand, solution_tables):
            if strategy == "average":
                solution = _average_solution(model, float(value), average_pair)
            elif strategy == "daisy_chain":
                solution = _daisy_solution(model, float(value), daisy_order)
            elif strategy == "instantaneous":
                solution = _minimum_hydrogen_solution(solutions)
            elif strategy == "sticky" and previous_mode in solutions:
                solution = solutions[previous_mode]
            elif strategy == "sticky":
                solution = _minimum_hydrogen_solution(solutions)
            elif strategy == "break_even_hysteresis":
                if previous_mode not in solutions:
                    solution = _minimum_hydrogen_solution(solutions)
                    cumulative_advantage_g = {}
                else:
                    current = solutions[previous_mode]
                    cumulative_advantage_g = {
                        mode: value
                        for mode, value in cumulative_advantage_g.items()
                        if mode in solutions and mode != previous_mode
                    }
                    for mode, alternative in solutions.items():
                        if mode == previous_mode:
                            continue
                        incremental_saving = (
                            current.hydrogen_g_per_s
                            - alternative.hydrogen_g_per_s
                        ) * dt_s
                        cumulative_advantage_g[mode] = max(
                            0.0,
                            cumulative_advantage_g.get(mode, 0.0)
                            + incremental_saving,
                        )
                    eligible = [
                        (
                            cumulative_advantage_g.get(mode, 0.0)
                            - (
                                hysteresis_stop_multiplier
                                if len(mode) < len(previous_mode)
                                else hysteresis_start_multiplier
                            )
                            * switch_penalty_g_per_change
                            * changed_stack_states(previous_mode, mode),
                            alternative,
                            cumulative_advantage_g.get(mode, 0.0),
                        )
                        for mode, alternative in solutions.items()
                        if mode != previous_mode
                    ]
                    best_surplus, best_alternative = max(
                        eligible,
                        key=lambda item: (
                            item[0],
                            -item[1].total_chemical_input_lhv_kw,
                            item[1].mode,
                        ),
                    )[:2]
                    if (
                        best_surplus >= -1e-12
                        and cumulative_advantage_g.get(
                            best_alternative.mode,
                            0.0,
                        )
                        > 1e-12
                    ):
                        solution = best_alternative
                        cumulative_advantage_g = {}
                    else:
                        solution = current
            else:
                solution = min(
                    solutions.values(),
                    key=lambda item: (
                        item.hydrogen_g_per_s * dt_s
                        + switch_penalty_g_per_change
                        * changed_stack_states(previous_mode, item.mode),
                        item.total_chemical_input_lhv_kw,
                        item.mode,
                    ),
                )
            selected.append(solution)
            if solution.mode != previous_mode and strategy != "break_even_hysteresis":
                cumulative_advantage_g = {}
            previous_mode = solution.mode

    rows = []
    previous_mode = ()
    cumulative_hydrogen = 0.0
    total_changes = 0
    total_starts = 0
    total_stops = 0
    for step, solution in enumerate(selected):
        current = set(solution.mode)
        previous = set(previous_mode)
        starts = len(current - previous)
        stops = len(previous - current)
        changes = starts + stops
        hydrogen = solution.hydrogen_g_per_s * dt_s
        cumulative_hydrogen += hydrogen
        total_changes += changes
        total_starts += starts
        total_stops += stops
        row = solution.as_record()
        row.update(
            {
                "step": step,
                "time_s": step * dt_s,
                "strategy": strategy,
                "stack_state_changes": changes,
                "stack_starts": starts,
                "stack_stops": stops,
                "hydrogen_g": hydrogen,
                "cumulative_hydrogen_g": cumulative_hydrogen,
            }
        )
        rows.append(row)
        previous_mode = solution.mode

    trajectory = pd.DataFrame(rows)
    output_energy_kwh = float(demand.sum() * dt_s / 3600.0)
    chemical_energy_kwh = float(
        trajectory["total_chemical_input_lhv_kw"].sum() * dt_s / 3600.0
    )
    total_objective = (
        cumulative_hydrogen
        + switch_penalty_g_per_change * total_changes
    )
    metrics: dict[str, float | int | str] = {
        "strategy": strategy,
        "n_steps": len(demand),
        "dt_s": dt_s,
        "switch_penalty_g_per_change": switch_penalty_g_per_change,
        "hysteresis_start_multiplier": hysteresis_start_multiplier,
        "hysteresis_stop_multiplier": hysteresis_stop_multiplier,
        "total_hydrogen_g": cumulative_hydrogen,
        "hydrogen_g_per_net_kwh": cumulative_hydrogen / output_energy_kwh,
        "energy_weighted_efficiency_lhv_pct": (
            100.0 * output_energy_kwh / chemical_energy_kwh
        ),
        "total_stack_state_changes": total_changes,
        "total_stack_starts": total_starts,
        "total_stack_stops": total_stops,
        "mode_change_events": int(
            trajectory["mode"].ne(trajectory["mode"].shift()).sum()
        ),
        "one_stack_share": float(trajectory["active_stack_count"].eq(1).mean()),
        "two_stack_share": float(trajectory["active_stack_count"].eq(2).mean()),
        "power_balance_mae_kw": float(
            trajectory["power_balance_error_kw"].abs().mean()
        ),
        "power_balance_max_abs_kw": float(
            trajectory["power_balance_error_kw"].abs().max()
        ),
        "total_evaluated_objective_g": total_objective,
        "runtime_s": time.perf_counter() - started,
    }
    return ChenPolicyRun(strategy=strategy, trajectory=trajectory, metrics=metrics)


def _minimum_hydrogen_solution(
    solutions: dict[tuple[str, ...], ChenDispatchSolution],
) -> ChenDispatchSolution:
    return min(
        solutions.values(),
        key=lambda item: (item.total_chemical_input_lhv_kw, item.mode),
    )


def _strongest_pair(model: ChenDispatchModel) -> tuple[str, str]:
    ranked = sorted(
        model.stack_ids,
        key=lambda stack_id: (
            model.curves[stack_id].maximum_net_power_kw,
            float(model.curves[stack_id].efficiency_lhv_pct.max()),
        ),
        reverse=True,
    )
    return tuple(sorted(ranked[:2]))


def _efficiency_order(model: ChenDispatchModel) -> tuple[str, ...]:
    return tuple(
        sorted(
            model.stack_ids,
            key=lambda stack_id: float(
                model.curves[stack_id].efficiency_lhv_pct.max()
            ),
            reverse=True,
        )
    )


def _average_solution(
    model: ChenDispatchModel,
    demand: float,
    pair: tuple[str, str],
) -> ChenDispatchSolution:
    if np.isclose(demand, 0.0):
        return model.evaluate_allocation(demand, {})
    first, second = (model.curves[stack_id] for stack_id in pair)
    lower = max(first.minimum_net_power_kw, demand - second.maximum_net_power_kw)
    upper = min(first.maximum_net_power_kw, demand - second.minimum_net_power_kw)
    if lower > upper + 1e-10:
        raise ValueError(f"fixed average pair cannot supply {demand:.6g} kW")
    first_power = float(np.clip(demand / 2.0, lower, upper))
    return model.evaluate_allocation(
        demand,
        {pair[0]: first_power, pair[1]: demand - first_power},
    )


def _daisy_solution(
    model: ChenDispatchModel,
    demand: float,
    order: tuple[str, ...],
) -> ChenDispatchSolution:
    if np.isclose(demand, 0.0):
        return model.evaluate_allocation(demand, {})
    for stack_id in order:
        solution = model.solve_mode(demand, (stack_id,))
        if solution is not None:
            return solution
    for first_id in order:
        for second_id in order:
            if second_id == first_id:
                continue
            first = model.curves[first_id]
            second = model.curves[second_id]
            lower = max(
                first.minimum_net_power_kw,
                demand - second.maximum_net_power_kw,
            )
            upper = min(
                first.maximum_net_power_kw,
                demand - second.minimum_net_power_kw,
            )
            if lower <= upper + 1e-10:
                first_power = upper
                return model.evaluate_allocation(
                    demand,
                    {
                        first_id: first_power,
                        second_id: demand - first_power,
                    },
                )
    raise ValueError(f"daisy chain cannot supply {demand:.6g} kW")


def _offline_dynamic_programming(
    solution_tables: list[dict[tuple[str, ...], ChenDispatchSolution]],
    dt_s: float,
    switch_penalty_g_per_change: float,
) -> list[ChenDispatchSolution]:
    costs: dict[tuple[str, ...], float] = {(): 0.0}
    parents: list[dict[tuple[str, ...], tuple[str, ...]]] = []
    for solutions in solution_tables:
        next_costs: dict[tuple[str, ...], float] = {}
        next_parents: dict[tuple[str, ...], tuple[str, ...]] = {}
        for mode, solution in solutions.items():
            best_previous, best_cost = min(
                (
                    (
                        previous_mode,
                        previous_cost
                        + switch_penalty_g_per_change
                        * changed_stack_states(previous_mode, mode),
                    )
                    for previous_mode, previous_cost in costs.items()
                ),
                key=lambda item: (item[1], item[0]),
            )
            next_costs[mode] = best_cost + solution.hydrogen_g_per_s * dt_s
            next_parents[mode] = best_previous
        costs = next_costs
        parents.append(next_parents)

    final_mode = min(costs, key=lambda mode: (costs[mode], mode))
    modes = [final_mode]
    for step in range(len(solution_tables) - 1, 0, -1):
        modes.append(parents[step][modes[-1]])
    modes.reverse()
    return [
        solution_tables[step][mode]
        for step, mode in enumerate(modes)
    ]
