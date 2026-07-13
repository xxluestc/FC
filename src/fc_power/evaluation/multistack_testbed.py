"""Unified multi-policy, multi-seed testbed with online health execution."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from fc_power.hydrogen_model import faraday_h2_g_s
from fc_power.power_allocation import (
    choose_average,
    choose_beam,
    choose_instant,
    choose_rotating,
    choose_terminal_soc_recovery,
)
from fc_power.world_model import MechanisticMultiStackWorldModel


@dataclass(frozen=True)
class TestScenario:
    name: str
    demand: pd.DataFrame
    initial_damage_fraction: tuple[float, ...]
    initial_soc: float = 0.70
    health_seed: int = 2026
    stochastic_health: bool = True

    def __post_init__(self) -> None:
        required = {"demand_power_kw", "event"}
        if not required.issubset(self.demand.columns) or len(self.demand) == 0:
            raise ValueError("demand frame is empty or missing required columns")
        if any(not np.isfinite(value) or value < 0 for value in self.initial_damage_fraction):
            raise ValueError("initial damage fractions must be finite and non-negative")


@dataclass(frozen=True)
class TestRun:
    trajectory: pd.DataFrame
    metrics: dict


def clip_profile_to_feasible_envelope(
    model: MechanisticMultiStackWorldModel,
    demand: pd.DataFrame,
    initial_damage_fraction: tuple[float, ...],
    *,
    stack_capacity_reserve_fraction: float = 0.01,
) -> pd.DataFrame:
    """Clip to a shared FC+battery envelope and audit every change.

    The stack reserve prevents a demand clipped exactly at the initial maximum
    from becoming infeasible after the online health state advances.  It is
    applied only to fuel-cell capacity; the battery limits are unchanged.
    """

    if len(initial_damage_fraction) != model.n_stacks:
        raise ValueError("initial health does not match n_stacks")
    if "demand_power_kw" not in demand:
        raise ValueError("demand frame is missing demand_power_kw")
    if (
        not np.isfinite(stack_capacity_reserve_fraction)
        or not 0 <= stack_capacity_reserve_fraction < 1
    ):
        raise ValueError("stack capacity reserve fraction must lie in [0, 1)")
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    maximum_current = max(model.config.allowed_currents_a)
    initial_maximum_stack_power = sum(
        proxy.evaluate(fraction * reference, [maximum_current])["stack_power_kw"][0]
        for proxy, fraction in zip(
            model.performance_proxies, initial_damage_fraction
        )
    )
    reserved_stack_power = (
        1.0 - stack_capacity_reserve_fraction
    ) * initial_maximum_stack_power
    lower = model.config.battery.charge_power_limit_kw
    upper = model.config.battery.discharge_power_limit_kw + reserved_stack_power
    result = demand.copy()
    raw = result.demand_power_kw.to_numpy(dtype=float)
    clipped = np.clip(raw, lower, upper)
    result["raw_demand_power_kw"] = raw
    result["demand_power_kw"] = clipped
    result["demand_clip_delta_kw"] = raw - clipped
    result["was_clipped"] = np.abs(raw - clipped) > 1e-12
    result.attrs["feasible_lower_kw"] = float(lower)
    result.attrs["feasible_upper_kw"] = float(upper)
    result.attrs["initial_maximum_stack_power_kw"] = float(
        initial_maximum_stack_power
    )
    result.attrs["stack_capacity_reserve_fraction"] = float(
        stack_capacity_reserve_fraction
    )
    return result


def run_policy(
    model: MechanisticMultiStackWorldModel,
    scenario: TestScenario,
    strategy: str,
    *,
    beam_horizon: int = 16,
    beam_width: int = 4,
    rotation_period: int = 30,
    fixed_online_assignment=None,
) -> TestRun:
    """Plan deterministically, execute Gamma health online, and record every step."""

    if len(scenario.initial_damage_fraction) != model.n_stacks:
        raise ValueError("scenario initial health does not match n_stacks")
    if beam_horizon <= 0 or beam_width <= 0 or rotation_period <= 0:
        raise ValueError("planner sizes must be positive")
    if fixed_online_assignment is not None and strategy != "instant_health":
        raise ValueError("fixed_online_assignment is only supported by instant_health")
    reference = model.performance_proxies[0].mapping.damage_reference_pct
    initial_damage = np.asarray(scenario.initial_damage_fraction) * reference
    state = model.initial_state(
        soc=scenario.initial_soc, degradation_pct=initial_damage
    )
    rng = np.random.default_rng(scenario.health_seed)
    demand = scenario.demand.demand_power_kw.to_numpy(dtype=float)
    if (
        model.config.power_interface == "fc_only"
        and "is_soc_recovery" in scenario.demand
        and bool(scenario.demand.is_soc_recovery.any())
    ):
        raise ValueError("fc_only scenarios cannot contain an SOC recovery tail")
    rows = []
    for step_index, current_demand in enumerate(demand):
        is_recovery = (
            "is_soc_recovery" in scenario.demand
            and bool(scenario.demand.is_soc_recovery.iloc[step_index])
        )
        planning_started = perf_counter()
        if is_recovery:
            planned = choose_terminal_soc_recovery(model, state, current_demand)
        elif strategy == "average":
            planned = choose_average(model, state, current_demand)
        elif strategy == "rotating":
            lead = (step_index // rotation_period) % model.n_stacks
            planned = choose_rotating(model, state, current_demand, lead)
        elif strategy == "instant_health":
            planned = choose_instant(
                model,
                state,
                current_demand,
                online_assignment=fixed_online_assignment,
            )
        elif strategy == "beam_health":
            preview_length = min(beam_horizon, len(demand) - step_index)
            preview = np.full(preview_length, current_demand, dtype=float)
            planned = choose_beam(
                model,
                state,
                preview,
                beam_width=beam_width,
                terminal_soc_weight=300.0,
            )
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        planning_elapsed_s = perf_counter() - planning_started

        executed = model.step(
            state,
            planned.action,
            current_demand,
            stochastic_health=scenario.stochastic_health,
            rng=rng,
            allow_dwell_override=bool(planned.step.constraints.safety_overrides),
        )
        row = {
            "scenario": scenario.name,
            "strategy": strategy,
            "health_seed": scenario.health_seed,
            "step": step_index,
            "demand_power_kw": current_demand,
            "event": scenario.demand.event.iloc[step_index],
            "is_soc_recovery": is_recovery,
            "power_interface": executed.constraints.power_interface,
            "stack_power_kw": executed.constraints.stack_power_kw,
            "battery_power_kw": executed.constraints.battery_power_kw,
            "soc": executed.next_state.soc,
            "hydrogen_g": executed.cost.raw_hydrogen_g,
            "battery_throughput_kwh": executed.cost.raw_battery_throughput_kwh,
            "performance_loss": executed.cost.performance_loss,
            "objective_total": executed.cost.total,
            "objective_hydrogen": executed.cost.hydrogen,
            "objective_degradation_increment": executed.cost.degradation_increment,
            "objective_degradation_reference_pct": (
                executed.cost.raw_degradation_reference_pct
            ),
            "objective_performance_loss": executed.cost.performance_loss,
            "objective_power_tracking": executed.cost.power_tracking,
            "objective_battery_use": executed.cost.battery_use,
            "objective_soc": executed.cost.soc,
            "objective_switch": executed.cost.switch,
            "objective_ramp": executed.cost.ramp,
            "expected_damage_increment_pct": planned.step.cost.raw_degradation_increment_pct,
            "sampled_damage_increment_pct": executed.cost.raw_degradation_increment_pct,
            "expected_continuous_damage_increment_pct": sum(
                item.expected_load_increment_pct + item.natural_increment_pct
                for item in planned.step.stacks
            ),
            "sampled_continuous_damage_increment_pct": sum(
                item.degradation_increment_pct
                - item.ramp_increment_pct
                - item.shift_increment_pct
                - item.start_stop_increment_pct
                - item.natural_increment_pct
                for item in executed.stacks
            )
            + sum(item.natural_increment_pct for item in executed.stacks),
            "discrete_damage_increment_pct": sum(
                item.ramp_increment_pct
                + item.shift_increment_pct
                + item.start_stop_increment_pct
                for item in executed.stacks
            ),
            "ramp_damage_increment_pct": sum(
                item.ramp_increment_pct for item in executed.stacks
            ),
            "shift_damage_increment_pct": sum(
                item.shift_increment_pct for item in executed.stacks
            ),
            "start_stop_damage_increment_pct": sum(
                item.start_stop_increment_pct for item in executed.stacks
            ),
            "constraint_feasible": executed.constraints.feasible,
            "safety_override": bool(executed.constraints.safety_overrides),
            "power_balance_error_kw": executed.constraints.power_balance_error_kw,
            "fc_power_tracking_error_kw": (
                executed.cost.raw_power_tracking_error_kw
            ),
            "planning_elapsed_s": planning_elapsed_s,
            "planning_expanded_nodes": planned.expanded_nodes,
            "planning_feasible_candidates": planned.feasible_candidates,
        }
        for index, (before, expected, after) in enumerate(
            zip(state.stacks, planned.step.stacks, executed.stacks)
        ):
            prefix = f"stack_{index}"
            row[f"{prefix}_current_a"] = after.current_a
            row[f"{prefix}_on"] = after.is_on
            row[f"{prefix}_damage_before_pct"] = before.health.degradation
            row[f"{prefix}_expected_increment_pct"] = expected.degradation_increment_pct
            row[f"{prefix}_sampled_increment_pct"] = after.degradation_increment_pct
            row[f"{prefix}_expected_continuous_increment_pct"] = (
                expected.expected_load_increment_pct
                + expected.natural_increment_pct
            )
            row[f"{prefix}_discrete_increment_pct"] = (
                after.ramp_increment_pct
                + after.shift_increment_pct
                + after.start_stop_increment_pct
            )
            row[f"{prefix}_ramp_increment_pct"] = after.ramp_increment_pct
            row[f"{prefix}_shift_increment_pct"] = after.shift_increment_pct
            row[f"{prefix}_start_stop_increment_pct"] = (
                after.start_stop_increment_pct
            )
            row[f"{prefix}_damage_after_pct"] = after.degradation_after_pct
            row[f"{prefix}_theta_i0"] = after.theta_reported[0]
            row[f"{prefix}_theta_ih"] = after.theta_reported[1]
            row[f"{prefix}_theta_R_ohm"] = after.theta_reported[2]
            row[f"{prefix}_cell_voltage_v"] = after.cell_voltage_v
            row[f"{prefix}_power_kw"] = after.power_kw
            row[f"{prefix}_switched"] = after.switched
            row[f"{prefix}_shifted_load"] = after.shifted_load
        rows.append(row)
        state = executed.next_state

    trajectory = pd.DataFrame(rows)
    _assert_online_health_invariants(trajectory, model.n_stacks)
    metrics = summarize_run(model, scenario, strategy, trajectory, state)
    return TestRun(trajectory, metrics)


def summarize_run(model, scenario, strategy, trajectory, final_state):
    main = (
        trajectory[~trajectory.is_soc_recovery]
        if "is_soc_recovery" in trajectory
        else trajectory
    )
    recovery = (
        trajectory[trajectory.is_soc_recovery]
        if "is_soc_recovery" in trajectory
        else trajectory.iloc[0:0]
    )
    final_damage = np.asarray(
        [stack.health.degradation for stack in final_state.stacks], dtype=float
    )
    initial_damage = np.asarray(
        [trajectory[f"stack_{i}_damage_before_pct"].iloc[0] for i in range(model.n_stacks)]
    )
    current_sum = np.asarray(
        [trajectory[f"stack_{i}_current_a"].sum() for i in range(model.n_stacks)]
    )
    main_current_sum = np.asarray(
        [main[f"stack_{i}_current_a"].sum() for i in range(model.n_stacks)]
    )
    aged_index = int(np.argmax(initial_damage))
    fc_only = model.config.power_interface == "fc_only"
    soc_error = final_state.soc - model.config.soc_reference
    reference_current = 195.0
    reference_power = model.performance_proxies[0].evaluate(
        0.0, [reference_current]
    )["stack_power_kw"][0]
    h2_g_per_kwh = faraday_h2_g_s(reference_current) * 3600 / reference_power
    corrected_h2 = (
        np.nan
        if fc_only
        else trajectory.hydrogen_g.sum()
        - soc_error * model.config.battery.energy_kwh * h2_g_per_kwh
    )
    tracking_error = trajectory.fc_power_tracking_error_kw.to_numpy(dtype=float)
    fc_energy_kwh = float(
        trajectory.stack_power_kw.sum() * model.config.dt_s / 3600.0
    )
    online = np.column_stack(
        [
            trajectory[f"stack_{i}_on"].to_numpy(dtype=bool)
            for i in range(model.n_stacks)
        ]
    )
    switch_counts = np.asarray(
        [trajectory[f"stack_{i}_switched"].sum() for i in range(model.n_stacks)],
        dtype=int,
    )
    load_shift_counts = np.asarray(
        [
            trajectory[f"stack_{i}_shifted_load"].sum()
            for i in range(model.n_stacks)
        ],
        dtype=int,
    )
    return {
        "scenario": scenario.name,
        "load_source": str(scenario.demand.source.iloc[0])
        if "source" in scenario.demand
        else "unspecified",
        "load_seed": int(scenario.demand.seed.iloc[0])
        if "seed" in scenario.demand
        else -1,
        "strategy": strategy,
        "power_interface": model.config.power_interface,
        "health_seed": scenario.health_seed,
        "n_steps": len(trajectory),
        "hydrogen_g": float(trajectory.hydrogen_g.sum()),
        "fc_energy_kwh": fc_energy_kwh,
        "hydrogen_g_per_fc_kwh": float(
            trajectory.hydrogen_g.sum() / max(fc_energy_kwh, 1e-12)
        ),
        "hydrogen_soc_corrected_g": float(corrected_h2),
        "sampled_damage_increment_pct": float(
            trajectory.sampled_damage_increment_pct.sum()
        ),
        "main_sampled_damage_increment_pct": float(
            main.sampled_damage_increment_pct.sum()
        ),
        "recovery_sampled_damage_increment_pct": float(
            recovery.sampled_damage_increment_pct.sum()
        ),
        "main_expected_continuous_damage_pct": float(
            main.expected_continuous_damage_increment_pct.sum()
        ),
        "main_sampled_continuous_damage_pct": float(
            main.sampled_continuous_damage_increment_pct.sum()
        ),
        "main_discrete_damage_pct": float(
            main.discrete_damage_increment_pct.sum()
        ),
        "main_ramp_damage_pct": float(main.ramp_damage_increment_pct.sum()),
        "main_shift_damage_pct": float(main.shift_damage_increment_pct.sum()),
        "main_start_stop_damage_pct": float(
            main.start_stop_damage_increment_pct.sum()
        ),
        "expected_damage_increment_pct": float(
            trajectory.expected_damage_increment_pct.sum()
        ),
        "main_expected_damage_increment_pct": float(
            main.expected_damage_increment_pct.sum()
        ),
        "recovery_expected_damage_increment_pct": float(
            recovery.expected_damage_increment_pct.sum()
        ),
        "performance_loss_sum": float(trajectory.performance_loss.sum()),
        "objective_total_sum": float(trajectory.objective_total.sum()),
        "objective_degradation_increment_sum": float(
            trajectory.objective_degradation_increment.sum()
        ),
        "main_performance_loss_sum": float(main.performance_loss.sum()),
        "recovery_performance_loss_sum": float(recovery.performance_loss.sum()),
        "main_hydrogen_g": float(main.hydrogen_g.sum()),
        "recovery_hydrogen_g": float(recovery.hydrogen_g.sum()),
        "battery_throughput_kwh": (
            np.nan if fc_only else float(trajectory.battery_throughput_kwh.sum())
        ),
        "soc_final": np.nan if fc_only else final_state.soc,
        "soc_error": np.nan if fc_only else soc_error,
        "soc_metrics_applicable": not fc_only,
        "fc_tracking_mae_kw": (
            float(np.mean(np.abs(tracking_error))) if fc_only else np.nan
        ),
        "fc_tracking_rmse_kw": (
            float(np.sqrt(np.mean(tracking_error**2))) if fc_only else np.nan
        ),
        "fc_tracking_bias_kw": (
            float(np.mean(tracking_error)) if fc_only else np.nan
        ),
        "fc_tracking_max_abs_kw": (
            float(np.max(np.abs(tracking_error))) if fc_only else np.nan
        ),
        "fc_tracking_within_tolerance_share": (
            float(
                np.mean(
                    np.abs(tracking_error)
                    <= model.config.fc_power_tracking_tolerance_kw + 1e-12
                )
            )
            if fc_only
            else np.nan
        ),
        "online_stack_count_mean": float(online.sum(axis=1).mean()),
        "online_stack_count_max": int(online.sum(axis=1).max()),
        "online_step_range": int(online.sum(axis=0).max() - online.sum(axis=0).min()),
        "total_switch_count": int(switch_counts.sum()),
        "total_load_shift_count": int(load_shift_counts.sum()),
        "constraint_violation_steps": int((~trajectory.constraint_feasible).sum()),
        "safety_override_steps": int(trajectory.safety_override.sum()),
        "planning_runtime_s": float(trajectory.planning_elapsed_s.sum()),
        "planning_runtime_mean_ms": float(
            1000.0 * trajectory.planning_elapsed_s.mean()
        ),
        "planning_runtime_max_ms": float(
            1000.0 * trajectory.planning_elapsed_s.max()
        ),
        "planning_expanded_nodes": int(trajectory.planning_expanded_nodes.sum()),
        "planning_feasible_candidates": int(
            trajectory.planning_feasible_candidates.sum()
        ),
        "max_power_balance_error_kw": float(
            trajectory.power_balance_error_kw.abs().max()
        ),
        "clipped_points": int(scenario.demand.was_clipped.sum())
        if "was_clipped" in scenario.demand
        else 0,
        "clipped_share": float(scenario.demand.was_clipped.mean())
        if "was_clipped" in scenario.demand
        else 0.0,
        "soc_recovery_steps": int(scenario.demand.is_soc_recovery.sum())
        if "is_soc_recovery" in scenario.demand
        else 0,
        "health_changed_steps": int(
            (trajectory.sampled_damage_increment_pct > 0).sum()
        ),
        "final_damage_mean_pct": float(final_damage.mean()),
        "final_damage_range_pct": float(final_damage.max() - final_damage.min()),
        "damage_increment_range_pct": float(
            (final_damage - initial_damage).max()
            - (final_damage - initial_damage).min()
        ),
        "aged_stack_index": aged_index,
        "aged_stack_current_share": float(
            current_sum[aged_index] / max(current_sum.sum(), 1e-12)
        ),
        "main_aged_stack_current_share": float(
            main_current_sum[aged_index] / max(main_current_sum.sum(), 1e-12)
        ),
        **{
            f"stack_{index}_current_a_step": float(value)
            for index, value in enumerate(current_sum)
        },
        **{
            f"stack_{index}_final_damage_pct": float(value)
            for index, value in enumerate(final_damage)
        },
        **{
            f"stack_{index}_main_continuous_damage_pct": float(
                main[f"stack_{index}_expected_continuous_increment_pct"].sum()
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_main_shift_damage_pct": float(
                main[f"stack_{index}_shift_increment_pct"].sum()
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_main_start_stop_damage_pct": float(
                main[f"stack_{index}_start_stop_increment_pct"].sum()
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_online_steps": int(online[:, index].sum())
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_switch_count": int(
                trajectory[f"stack_{index}_switched"].sum()
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_load_shift_count": int(
                trajectory[f"stack_{index}_shifted_load"].sum()
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_max_consecutive_online_steps": _max_consecutive_true(
                online[:, index]
            )
            for index in range(model.n_stacks)
        },
        **{
            f"stack_{index}_max_consecutive_off_steps": _max_consecutive_true(
                ~online[:, index]
            )
            for index in range(model.n_stacks)
        },
    }


def paired_strategy_comparison(
    per_run: pd.DataFrame,
    *,
    reference_strategy: str = "average",
    metrics: tuple[str, ...] = (
        "hydrogen_soc_corrected_g",
        "main_expected_damage_increment_pct",
        "main_performance_loss_sum",
        "battery_throughput_kwh",
        "main_aged_stack_current_share",
    ),
) -> pd.DataFrame:
    """Compute paired candidate-minus-reference effects across load seeds.

    All default metrics use the convention that a negative difference is
    better. Pairing prevents differences between random loads from being
    misreported as controller effects.
    """

    keys = ("load_source", "load_seed", "health_seed")
    required = {*keys, "strategy", *metrics}
    missing = required.difference(per_run.columns)
    if missing:
        raise ValueError(f"per-run metrics are missing columns: {sorted(missing)}")
    reference = per_run[per_run.strategy == reference_strategy]
    if reference.empty:
        raise ValueError(f"reference strategy {reference_strategy!r} is absent")
    rows = []
    for strategy in sorted(set(per_run.strategy) - {reference_strategy}):
        candidate = per_run[per_run.strategy == strategy]
        paired = candidate.merge(
            reference,
            on=list(keys),
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        for source, group in paired.groupby("load_source"):
            for metric in metrics:
                candidate_values = group[f"{metric}_candidate"].to_numpy(dtype=float)
                reference_values = group[f"{metric}_reference"].to_numpy(dtype=float)
                delta = candidate_values - reference_values
                relative = 100.0 * delta / np.maximum(
                    np.abs(reference_values), 1e-12
                )
                std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
                rows.append(
                    {
                        "load_source": source,
                        "strategy": strategy,
                        "reference_strategy": reference_strategy,
                        "metric": metric,
                        "n_pairs": len(delta),
                        "mean_delta": float(delta.mean()),
                        "ci95": 1.96 * std / np.sqrt(len(delta)),
                        "mean_relative_pct": float(relative.mean()),
                        "lower_is_better_win_share": float(np.mean(delta < 0)),
                    }
                )
    return pd.DataFrame(rows)


def _assert_online_health_invariants(trajectory: pd.DataFrame, n_stacks: int):
    tolerance = 1e-12
    for index in range(n_stacks):
        before = trajectory[f"stack_{index}_damage_before_pct"].to_numpy()
        increment = trajectory[f"stack_{index}_sampled_increment_pct"].to_numpy()
        after = trajectory[f"stack_{index}_damage_after_pct"].to_numpy()
        if np.any(increment < -tolerance) or np.any(after < before - tolerance):
            raise AssertionError("health degradation must be irreversible")
        if not np.allclose(after, before + increment, atol=tolerance, rtol=1e-10):
            raise AssertionError("damage state does not equal prior plus action increment")
        if len(after) > 1 and not np.allclose(
            before[1:], after[:-1], atol=tolerance, rtol=1e-10
        ):
            raise AssertionError("updated health was not carried into the next decision")


def _max_consecutive_true(values) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)
