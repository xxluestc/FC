"""Instant and receding-horizon power allocators."""

from fc_power.power_allocation.multistack_allocator import (
    PlanningResult,
    choose_beam,
    choose_instant,
    choose_terminal_soc_recovery,
    enumerate_actions,
    power_balance,
    project_to_feasible,
)
from fc_power.power_allocation.multistack_baselines import (
    choose_average,
    choose_rotating,
)

__all__ = [
    "PlanningResult",
    "choose_beam",
    "choose_average",
    "choose_instant",
    "choose_rotating",
    "choose_terminal_soc_recovery",
    "enumerate_actions",
    "power_balance",
    "project_to_feasible",
]
