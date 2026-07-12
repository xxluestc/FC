"""Enumerative and Beam-search control for the shared multi-stack world model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from fc_power.world_model.mechanistic import (
    MechanisticMultiStackWorldModel,
    MultiStackAction,
    MultiStackState,
    WorldStep,
)


def power_balance(stack_powers_kw, battery_power_kw):
    """Compatibility helper retained from the original placeholder module."""

    return sum(stack_powers_kw) + battery_power_kw


@dataclass(frozen=True)
class PlanningResult:
    action: MultiStackAction
    step: WorldStep
    objective: float
    expanded_nodes: int
    feasible_candidates: int


@dataclass(frozen=True)
class _BeamNode:
    objective: float
    state: MultiStackState
    first_action: MultiStackAction
    first_step: WorldStep


def enumerate_actions(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    *,
    include_energized_idle: bool = True,
    respect_dwell: bool = True,
):
    """Yield the discrete Cartesian action grid allowed by dwell memory.

    Zero current has two physically different modes: fully off and energized
    idle.  Positive-current levels are always energized.
    """

    if len(state.stacks) != model.n_stacks:
        raise ValueError("state stack count does not match model")
    per_stack = []
    for stack in state.stacks:
        if respect_dwell and stack.dwell_s + 1e-12 < model.config.min_dwell_s:
            per_stack.append(((stack.health.current_a, stack.health.is_on),))
            continue
        candidates = [(0.0, False)]
        if include_energized_idle:
            candidates.append((0.0, True))
        candidates.extend(
            (float(current), True)
            for current in model.config.allowed_currents_a
            if current > 0
        )
        per_stack.append(tuple(candidates))

    for combined in product(*per_stack):
        yield MultiStackAction(
            current_a=tuple(item[0] for item in combined),
            is_on=tuple(item[1] for item in combined),
        )


def choose_instant(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    demand_power_kw: float,
    *,
    allow_dwell_override: bool = True,
) -> PlanningResult:
    """Choose the minimum-cost feasible one-step action."""

    best = None
    expanded = 0
    feasible = 0
    passes = [(True, False)]
    if allow_dwell_override:
        passes.append((False, True))
    for respect_dwell, override in passes:
        for action in enumerate_actions(model, state, respect_dwell=respect_dwell):
            expanded += 1
            step = model.step(
                state,
                action,
                demand_power_kw,
                allow_dwell_override=override,
            )
            if not step.constraints.feasible:
                continue
            feasible += 1
            key = (step.cost.total, _action_tiebreak(action))
            if best is None or key < best[0]:
                best = (key, action, step)
        if best is not None:
            break
    if best is None:
        raise RuntimeError("no feasible multi-stack action for the requested demand")
    return PlanningResult(best[1], best[2], best[0][0], expanded, feasible)


def choose_beam(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    demand_preview_kw,
    *,
    beam_width: int = 16,
    terminal_soc_weight: float = 3.0,
    allow_dwell_override: bool = True,
    dwell_override_penalty: float = 1.0,
) -> PlanningResult:
    """Plan over a deterministic demand preview and execute its first action."""

    preview = np.asarray(demand_preview_kw, dtype=float)
    if preview.ndim != 1 or preview.size == 0 or np.any(~np.isfinite(preview)):
        raise ValueError("demand_preview_kw must be a finite non-empty vector")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if not np.isfinite(terminal_soc_weight) or terminal_soc_weight < 0:
        raise ValueError("terminal_soc_weight must be finite and non-negative")
    if not np.isfinite(dwell_override_penalty) or dwell_override_penalty < 0:
        raise ValueError("dwell_override_penalty must be finite and non-negative")

    beam: list[_BeamNode | None] = [None]
    expanded = 0
    feasible_count = 0
    for horizon_index, demand in enumerate(preview):
        candidates: list[_BeamNode] = []
        passes = [(True, False)]
        if allow_dwell_override:
            passes.append((False, True))
        for respect_dwell, override in passes:
            for node in beam:
                node_state = state if node is None else node.state
                base_objective = 0.0 if node is None else node.objective
                for action in enumerate_actions(
                    model, node_state, respect_dwell=respect_dwell
                ):
                    expanded += 1
                    step = model.step(
                        node_state,
                        action,
                        float(demand),
                        allow_dwell_override=override,
                    )
                    if not step.constraints.feasible:
                        continue
                    feasible_count += 1
                    first_action = action if node is None else node.first_action
                    first_step = step if node is None else node.first_step
                    objective = (
                        base_objective
                        + step.cost.total
                        + dwell_override_penalty
                        * len(step.constraints.safety_overrides)
                    )
                    candidates.append(
                        _BeamNode(
                            objective, step.next_state, first_action, first_step
                        )
                    )
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                f"no feasible multi-stack beam node at preview index {horizon_index}"
            )
        beam = sorted(
            candidates,
            key=lambda node: (
                node.objective
                + terminal_soc_weight
                * abs(node.state.soc - model.config.soc_reference),
                _action_tiebreak(node.first_action),
            ),
        )[:beam_width]

    best = min(
        beam,
        key=lambda node: (
            node.objective
            + terminal_soc_weight
            * abs(node.state.soc - model.config.soc_reference),
            _action_tiebreak(node.first_action),
        ),
    )
    final_objective = best.objective + terminal_soc_weight * abs(
        best.state.soc - model.config.soc_reference
    )
    return PlanningResult(
        best.first_action,
        best.first_step,
        float(final_objective),
        expanded,
        feasible_count,
    )


def project_to_feasible(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    requested_action: MultiStackAction,
    demand_power_kw: float,
    *,
    on_mismatch_penalty: float = 1.0,
    allow_dwell_override: bool = True,
) -> PlanningResult:
    """Project an arbitrary policy output to the nearest feasible grid action."""

    if len(requested_action.current_a) != model.n_stacks:
        raise ValueError("requested action stack count does not match model")
    if not np.isfinite(on_mismatch_penalty) or on_mismatch_penalty < 0:
        raise ValueError("on_mismatch_penalty must be finite and non-negative")
    current_scale = max(model.config.allowed_currents_a)
    best = None
    expanded = 0
    feasible = 0
    passes = [(True, False)]
    if allow_dwell_override:
        passes.append((False, True))
    for respect_dwell, override in passes:
        for action in enumerate_actions(model, state, respect_dwell=respect_dwell):
            expanded += 1
            step = model.step(
                state,
                action,
                demand_power_kw,
                allow_dwell_override=override,
            )
            if not step.constraints.feasible:
                continue
            feasible += 1
            current_distance = sum(
                ((actual - requested) / current_scale) ** 2
                for actual, requested in zip(action.current_a, requested_action.current_a)
            )
            on_distance = on_mismatch_penalty * sum(
                actual != requested
                for actual, requested in zip(action.is_on, requested_action.is_on)
            )
            distance = float(current_distance + on_distance)
            key = (distance, step.cost.total, _action_tiebreak(action))
            if best is None or key < best[0]:
                best = (key, action, step)
        if best is not None:
            break
    if best is None:
        raise RuntimeError("no feasible action exists for safety projection")
    return PlanningResult(best[1], best[2], best[0][0], expanded, feasible)


def choose_terminal_soc_recovery(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    demand_power_kw: float,
) -> PlanningResult:
    """Common terminal controller minimizing next-step SOC error.

    It is intentionally strategy-independent and is used only in an explicitly
    labelled recovery tail.  Hydrogen and degradation incurred during recovery
    remain in each strategy's total metrics.
    """

    best = None
    expanded = 0
    feasible = 0
    hold_steps = max(
        1, int(np.ceil(model.config.min_dwell_s / model.config.dt_s))
    )
    for respect_dwell, override in ((True, False), (False, True)):
        for action in enumerate_actions(model, state, respect_dwell=respect_dwell):
            expanded += 1
            step = model.step(
                state,
                action,
                demand_power_kw,
                allow_dwell_override=override,
            )
            if not step.constraints.feasible:
                continue
            rollout_state = step.next_state
            rollout_cost = step.cost.total
            rollout_feasible = True
            for _ in range(1, hold_steps):
                future = model.step(
                    rollout_state,
                    action,
                    demand_power_kw,
                )
                expanded += 1
                if not future.constraints.feasible:
                    rollout_feasible = False
                    break
                rollout_state = future.next_state
                rollout_cost += future.cost.total
            if not rollout_feasible:
                continue
            feasible += 1
            soc_error = abs(
                rollout_state.soc - model.config.soc_reference
            )
            key = (
                soc_error,
                rollout_cost,
                _action_tiebreak(action),
            )
            if best is None or key < best[0]:
                best = (key, action, step)
        if best is not None:
            break
    if best is None:
        raise RuntimeError("no feasible action for terminal SOC recovery")
    return PlanningResult(best[1], best[2], best[0][0], expanded, feasible)


def _action_tiebreak(action: MultiStackAction):
    return tuple(action.current_a) + tuple(int(value) for value in action.is_on)
