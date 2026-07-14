"""Coefficient-free health-adaptive allocation for a final feasibility screen.

The allocator does not estimate a physical degradation rate.  A fixed,
dimensionless fleet health-progress budget is distributed among stacks in
proportion to their executed electrical energy.  The policy can use current
LZW health progress to update stack performance and, optionally, prefer less
progressed stacks among actions with essentially identical tracking quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from fc_power.health.dynamic_proxy import LzwIvConditions
from fc_power.health.lzw_health_progress import LzwHealthProgressMap
from fc_power.hydrogen_model import faraday_h2_g_s
from fc_power.lzw_iv_model import iv_model, reported_to_model_theta


@dataclass(frozen=True)
class RelativeHealthWeights:
    """Dimensionless preferences after the best tracking band is fixed."""

    tracking: float = 20.0
    hydrogen: float = 0.25
    switch: float = 0.02
    ramp: float = 0.01
    health_loading: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RelativeHealthAction:
    current_a: tuple[float, ...]
    predicted_power_kw: tuple[float, ...]
    predicted_total_power_kw: float
    tracking_error_kw: float
    score: float
    hydrogen_term: float
    switch_term: float
    ramp_term: float
    health_loading_term: float


def build_n_plus_one_action_grid(
    allowed_currents_a,
    *,
    n_stacks: int = 3,
    max_online_stacks: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return action indices and currents for a finite N+1 action space."""

    levels = np.asarray(allowed_currents_a, dtype=float)
    if (
        levels.ndim != 1
        or len(levels) < 2
        or levels[0] != 0.0
        or np.any(~np.isfinite(levels))
        or np.any(np.diff(levels) <= 0)
    ):
        raise ValueError(
            "allowed currents must be finite, strictly increasing, and start at zero"
        )
    if isinstance(n_stacks, bool) or not isinstance(n_stacks, int) or n_stacks <= 0:
        raise ValueError("n_stacks must be a positive integer")
    if (
        isinstance(max_online_stacks, bool)
        or not isinstance(max_online_stacks, int)
        or not 0 < max_online_stacks <= n_stacks
    ):
        raise ValueError("max_online_stacks must be in [1, n_stacks]")

    indices = np.asarray(
        [
            item
            for item in product(range(len(levels)), repeat=n_stacks)
            if np.count_nonzero(item) <= max_online_stacks
        ],
        dtype=int,
    )
    return indices, levels[indices]


def lzw_power_table_kw(
    mapping: LzwHealthProgressMap,
    conditions: LzwIvConditions,
    health_progress,
    allowed_currents_a,
) -> np.ndarray:
    """Evaluate every stack/current power on the data-derived theta manifold."""

    progress = np.asarray(health_progress, dtype=float)
    currents = np.asarray(allowed_currents_a, dtype=float)
    if progress.ndim != 1 or np.any(~np.isfinite(progress)):
        raise ValueError("health_progress must be a finite vector")
    if np.any((progress < 0) | (progress > 1)):
        raise ValueError("health_progress must lie in [0, 1]")
    if currents.ndim != 1 or np.any(~np.isfinite(currents)) or np.any(currents < 0):
        raise ValueError("allowed currents must be finite and non-negative")

    theta = reported_to_model_theta(
        mapping.theta_at(progress), conditions.active_area_cm2
    )[:, None, :]
    density = currents[None, :] / conditions.active_area_cm2
    voltage = iv_model(
        temperature_c=conditions.temperature_c,
        theta_model=theta,
        current_density_a_cm2=density,
        a=conditions.a,
        b=conditions.b,
        inner=[
            conditions.concentration_b,
            conditions.limiting_current_a_cm2,
        ],
        active_area_cm2=conditions.active_area_cm2,
    )
    table = currents[None, :] * voltage * conditions.stack_cells / 1000.0
    table[:, currents == 0.0] = 0.0
    return table


def interpolate_lzw_power_table_kw(
    health_progress,
    lookup_health_progress,
    lookup_power_kw,
) -> np.ndarray:
    """Linearly interpolate a precomputed LZW progress/current power table."""

    progress = np.asarray(health_progress, dtype=float)
    grid = np.asarray(lookup_health_progress, dtype=float)
    values = np.asarray(lookup_power_kw, dtype=float)
    if progress.ndim != 1 or np.any(~np.isfinite(progress)):
        raise ValueError("health_progress must be a finite vector")
    if np.any((progress < 0) | (progress > 1)):
        raise ValueError("health_progress must lie in [0, 1]")
    if (
        grid.ndim != 1
        or len(grid) < 2
        or grid[0] != 0.0
        or grid[-1] != 1.0
        or np.any(np.diff(grid) <= 0)
        or values.ndim != 2
        or values.shape[0] != len(grid)
    ):
        raise ValueError("lookup grid and power values have invalid dimensions")
    output = np.empty((len(progress), values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(progress, grid, values[:, column])
    return output


def choose_relative_health_action(
    *,
    action_indices: np.ndarray,
    action_currents_a: np.ndarray,
    power_table_kw: np.ndarray,
    decision_health_progress,
    previous_currents_a,
    demand_power_kw: float,
    max_online_stacks: int,
    tracking_slack_kw: float = 0.5,
    weights: RelativeHealthWeights = RelativeHealthWeights(),
) -> RelativeHealthAction:
    """Choose within a narrow best-tracking band, then apply soft preferences."""

    indices = np.asarray(action_indices, dtype=int)
    currents = np.asarray(action_currents_a, dtype=float)
    table = np.asarray(power_table_kw, dtype=float)
    health = np.asarray(decision_health_progress, dtype=float)
    previous = np.asarray(previous_currents_a, dtype=float)
    if indices.ndim != 2 or currents.shape != indices.shape:
        raise ValueError("action index/current grids must have matching 2-D shapes")
    n_actions, n_stacks = indices.shape
    if n_actions == 0 or table.shape[0] != n_stacks or health.shape != (n_stacks,):
        raise ValueError("action grid, power table, and health dimensions must agree")
    if previous.shape != (n_stacks,):
        raise ValueError("previous_currents_a must match the stack count")
    if table.shape[1] <= int(indices.max(initial=0)):
        raise ValueError("power table does not cover all action indices")
    if not np.isfinite(demand_power_kw) or demand_power_kw < 0:
        raise ValueError("demand_power_kw must be finite and non-negative")
    if not np.isfinite(tracking_slack_kw) or tracking_slack_kw < 0:
        raise ValueError("tracking_slack_kw must be finite and non-negative")

    selected_power = table[np.arange(n_stacks)[None, :], indices]
    total_power = selected_power.sum(axis=1)
    errors = np.abs(total_power - demand_power_kw)
    minimum_error = float(errors.min())
    eligible = errors <= minimum_error + tracking_slack_kw + 1e-12

    max_current = max(float(currents.max(initial=0.0)), 1e-12)
    capacity_reference = max(float(np.max(total_power)), 1.0)
    hydrogen = currents.sum(axis=1) / max(max_online_stacks * max_current, 1e-12)
    switches = np.mean((currents > 0) != (previous[None, :] > 0), axis=1)
    ramps = np.sum(np.abs(currents - previous[None, :]), axis=1) / (
        n_stacks * max_current
    )
    health_loading = np.divide(
        np.sum(selected_power * health[None, :], axis=1),
        total_power,
        out=np.zeros_like(total_power),
        where=total_power > 0,
    )
    scores = (
        weights.tracking * errors / capacity_reference
        + weights.hydrogen * hydrogen
        + weights.switch * switches
        + weights.ramp * ramps
        + weights.health_loading * health_loading
    )
    scores[~eligible] = np.inf
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise RuntimeError("no action remains inside the best-tracking band")
    return RelativeHealthAction(
        current_a=tuple(float(value) for value in currents[best]),
        predicted_power_kw=tuple(float(value) for value in selected_power[best]),
        predicted_total_power_kw=float(total_power[best]),
        tracking_error_kw=float(errors[best]),
        score=float(scores[best]),
        hydrogen_term=float(hydrogen[best]),
        switch_term=float(switches[best]),
        ramp_term=float(ramps[best]),
        health_loading_term=float(health_loading[best]),
    )


def allocate_relative_health_budget(
    health_progress,
    executed_stack_power_kw,
    *,
    demand_power_kw: float,
    dt_s: float,
    episode_demand_energy_kwh: float,
    fleet_progress_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a fixed episode budget by executed stack-energy share.

    The sum of increments over a perfectly tracked episode equals the supplied
    fleet budget.  The budget is a dimensionless sensitivity setting, not a
    degradation coefficient or a physical time rate.
    """

    health = np.asarray(health_progress, dtype=float)
    power = np.asarray(executed_stack_power_kw, dtype=float)
    if health.ndim != 1 or power.shape != health.shape:
        raise ValueError("health and stack power must be same-length vectors")
    if np.any(~np.isfinite(health)) or np.any((health < 0) | (health > 1)):
        raise ValueError("health progress must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(power)) or np.any(power < 0):
        raise ValueError("executed stack power must be finite and non-negative")
    scalars = (demand_power_kw, dt_s, episode_demand_energy_kwh, fleet_progress_budget)
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("budget inputs must be finite")
    if demand_power_kw < 0 or dt_s <= 0 or episode_demand_energy_kwh <= 0:
        raise ValueError("demand must be non-negative and time/energy positive")
    if fleet_progress_budget < 0:
        raise ValueError("fleet_progress_budget must be non-negative")

    total_power = float(power.sum())
    if demand_power_kw > 0.0 and total_power == 0.0 and fleet_progress_budget > 0.0:
        raise ValueError("positive demand cannot receive a relative-health budget with zero output")
    if demand_power_kw == 0.0 or fleet_progress_budget == 0.0:
        increments = np.zeros_like(health)
    else:
        demand_energy_kwh = demand_power_kw * dt_s / 3600.0
        step_budget = fleet_progress_budget * (
            demand_energy_kwh / episode_demand_energy_kwh
        )
        increments = step_budget * power / total_power
    next_health = health + increments
    if np.any(next_health > 1.0 + 1e-12):
        raise ValueError("relative health budget would exceed the LZW endpoint")
    return np.minimum(next_health, 1.0), increments


def executed_hydrogen_g(current_a, dt_s: float, *, stack_cells: int = 170) -> float:
    """Return Faraday-law hydrogen for one multi-stack action."""

    currents = np.asarray(current_a, dtype=float)
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    return float(faraday_h2_g_s(currents, n_cells=stack_cells).sum() * dt_s)


__all__ = [
    "RelativeHealthAction",
    "RelativeHealthWeights",
    "allocate_relative_health_budget",
    "build_n_plus_one_action_grid",
    "choose_relative_health_action",
    "executed_hydrogen_g",
    "interpolate_lzw_power_table_kw",
    "lzw_power_table_kw",
]
