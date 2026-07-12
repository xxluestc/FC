"""Health-blind rule baselines for fair multi-stack controller comparison."""

from __future__ import annotations

import numpy as np

from fc_power.power_allocation.multistack_allocator import (
    PlanningResult,
    enumerate_actions,
)
from fc_power.world_model.mechanistic import (
    MechanisticMultiStackWorldModel,
    MultiStackState,
)


def choose_average(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    demand_power_kw: float,
    *,
    soc_feedback_kw_per_soc: float = 1200.0,
) -> PlanningResult:
    """Choose equal stack currents with charge-sustaining SOC feedback."""

    target = _target_stack_power(model, state, demand_power_kw, soc_feedback_kw_per_soc)
    candidates, expanded = _rule_candidates(
        model,
        state,
        demand_power_kw,
        lambda action: _is_average_action(model, action),
        lambda step, action: _tracking_score(model, step, target),
    )
    if not candidates:
        raise RuntimeError("no feasible equal-load action")
    best = min(candidates)
    return PlanningResult(best[2], best[3], best[0], expanded, len(candidates))


def choose_rotating(
    model: MechanisticMultiStackWorldModel,
    state: MultiStackState,
    demand_power_kw: float,
    lead_stack: int,
    *,
    soc_feedback_kw_per_soc: float = 1200.0,
) -> PlanningResult:
    """Prefer a rotating lead stack while tracking total requested power."""

    if not 0 <= lead_stack < model.n_stacks:
        raise ValueError("lead_stack is out of range")
    target = _target_stack_power(model, state, demand_power_kw, soc_feedback_kw_per_soc)
    max_current = max(model.config.allowed_currents_a)
    def score(step, action):
        total_current = sum(action.current_a)
        non_lead_current = total_current - action.current_a[lead_stack]
        lead_penalty = non_lead_current / max(model.n_stacks * max_current, 1e-12)
        return _tracking_score(model, step, target) + 0.08 * lead_penalty

    candidates, expanded = _rule_candidates(
        model, state, demand_power_kw, lambda action: True, score
    )
    if not candidates:
        raise RuntimeError("no feasible rotating-load action")
    best = min(candidates)
    return PlanningResult(best[2], best[3], best[0], expanded, len(candidates))


def _target_stack_power(model, state, demand_power_kw, feedback_gain):
    if not np.isfinite(feedback_gain) or feedback_gain < 0:
        raise ValueError("soc feedback gain must be finite and non-negative")
    correction = (
        0.0
        if model.config.power_interface == "fc_only"
        else feedback_gain * (model.config.soc_reference - state.soc)
    )
    capacities = sorted(
        (
            float(
                proxy.evaluate(
                    stack.health.degradation,
                    [max(model.config.allowed_currents_a)],
                )["stack_power_kw"][0]
            )
            for proxy, stack in zip(model.performance_proxies, state.stacks)
        ),
        reverse=True,
    )
    online_limit = model.config.max_online_stacks or model.n_stacks
    maximum = sum(capacities[:online_limit])
    return float(np.clip(max(demand_power_kw, 0.0) + correction, 0.0, maximum))


def _is_average_action(model, action):
    online_limit = model.config.max_online_stacks or model.n_stacks
    if online_limit >= model.n_stacks:
        return len(set(action.current_a)) == 1 and len(set(action.is_on)) == 1
    online_currents = [
        current for current, is_on in zip(action.current_a, action.is_on) if is_on
    ]
    offline_currents_are_zero = all(
        current == 0.0
        for current, is_on in zip(action.current_a, action.is_on)
        if not is_on
    )
    return (
        offline_currents_are_zero
        and len(online_currents) <= online_limit
        and len(set(online_currents)) <= 1
    )


def _tracking_score(model, step, target_stack_power_kw):
    if model.config.power_interface == "fc_only":
        power_scale = max(model.fc_power_reference_kw(), 1.0)
    else:
        power_scale = max(
            abs(model.config.battery.charge_power_limit_kw),
            abs(model.config.battery.discharge_power_limit_kw),
            1.0,
        )
    tracking = abs(step.constraints.stack_power_kw - target_stack_power_kw) / power_scale
    hydrogen = step.cost.hydrogen
    switches = step.cost.switch
    return float(tracking + 0.03 * hydrogen + 0.02 * switches)


def _action_key(action):
    return tuple(action.current_a) + tuple(int(value) for value in action.is_on)


def _rule_candidates(model, state, demand_power_kw, action_filter, score):
    """Try normal dwell first, then audited emergency dwell relaxation."""

    expanded = 0
    for respect_dwell, override in ((True, False), (False, True)):
        candidates = []
        for action in enumerate_actions(model, state, respect_dwell=respect_dwell):
            if not action_filter(action):
                continue
            expanded += 1
            step = model.step(
                state,
                action,
                demand_power_kw,
                allow_dwell_override=override,
            )
            if not step.constraints.feasible:
                continue
            value = score(step, action)
            candidates.append((value, _action_key(action), action, step))
        if candidates:
            return candidates, expanded
    return [], expanded
