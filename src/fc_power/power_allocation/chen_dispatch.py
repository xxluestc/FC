"""Exact inner dispatch on Chen Peng's audited efficiency curves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


AUDITED_COLUMNS = (
    "stack_id",
    "cell_count",
    "current_density_a_cm2",
    "gross_stack_power_kw",
    "efficiency_lhv_pct",
    "chemical_input_lhv_kw",
    "net_system_power_kw",
    "auxiliary_power_kw",
)


@dataclass(frozen=True)
class ChenStackCurve:
    stack_id: str
    cell_count: int
    current_density_a_cm2: np.ndarray
    gross_power_kw: np.ndarray
    efficiency_lhv_pct: np.ndarray
    chemical_input_lhv_kw: np.ndarray
    net_power_kw: np.ndarray
    auxiliary_power_kw: np.ndarray

    @property
    def minimum_net_power_kw(self) -> float:
        return float(self.net_power_kw[0])

    @property
    def maximum_net_power_kw(self) -> float:
        return float(self.net_power_kw[-1])


@dataclass(frozen=True)
class ChenDispatchSolution:
    demand_net_power_kw: float
    mode: tuple[str, ...]
    stack_ids: tuple[str, ...]
    stack_net_power_kw: tuple[float, ...]
    stack_gross_power_kw: tuple[float, ...]
    stack_efficiency_lhv_pct: tuple[float, ...]
    stack_chemical_input_lhv_kw: tuple[float, ...]
    stack_current_density_a_cm2: tuple[float, ...]
    total_chemical_input_lhv_kw: float
    hydrogen_g_per_s: float
    system_efficiency_lhv_pct: float
    power_balance_error_kw: float

    def as_record(self) -> dict[str, float | int | str]:
        record: dict[str, float | int | str] = {
            "demand_net_power_kw": self.demand_net_power_kw,
            "mode": "+".join(self.mode) if self.mode else "OFF",
            "active_stack_count": len(self.mode),
            "total_chemical_input_lhv_kw": self.total_chemical_input_lhv_kw,
            "hydrogen_g_per_s": self.hydrogen_g_per_s,
            "system_efficiency_lhv_pct": self.system_efficiency_lhv_pct,
            "power_balance_error_kw": self.power_balance_error_kw,
        }
        for index, stack_id in enumerate(self.stack_ids):
            record[f"{stack_id}_net_power_kw"] = self.stack_net_power_kw[index]
            record[f"{stack_id}_gross_power_kw"] = self.stack_gross_power_kw[index]
            record[f"{stack_id}_efficiency_lhv_pct"] = (
                self.stack_efficiency_lhv_pct[index]
            )
            record[f"{stack_id}_current_density_a_cm2"] = (
                self.stack_current_density_a_cm2[index]
            )
        return record


class ChenDispatchModel:
    """Interpolate audited stack curves and solve one/two-stack dispatch exactly."""

    def __init__(self, audited_curves: pd.DataFrame):
        missing = set(AUDITED_COLUMNS).difference(audited_curves.columns)
        if missing:
            raise ValueError(f"missing audited Chen columns: {sorted(missing)}")
        self.stack_ids = tuple(sorted(audited_curves["stack_id"].unique()))
        if len(self.stack_ids) < 2:
            raise ValueError("at least two Chen stack curves are required")

        curves: dict[str, ChenStackCurve] = {}
        for stack_id in self.stack_ids:
            group = audited_curves[audited_curves["stack_id"].eq(stack_id)].copy()
            group = group.sort_values("net_system_power_kw", kind="stable")
            net_power = group["net_system_power_kw"].to_numpy(dtype=float)
            if len(group) < 2 or np.any(~np.isfinite(net_power)):
                raise ValueError(f"{stack_id} has an invalid net-power curve")
            if np.any(np.diff(net_power) <= 0):
                raise ValueError(f"{stack_id} net power must be strictly increasing")
            if group["cell_count"].nunique() != 1:
                raise ValueError(f"{stack_id} must have one cell count")
            curves[stack_id] = ChenStackCurve(
                stack_id=stack_id,
                cell_count=int(group["cell_count"].iloc[0]),
                current_density_a_cm2=group[
                    "current_density_a_cm2"
                ].to_numpy(dtype=float),
                gross_power_kw=group["gross_stack_power_kw"].to_numpy(dtype=float),
                efficiency_lhv_pct=group["efficiency_lhv_pct"].to_numpy(
                    dtype=float
                ),
                chemical_input_lhv_kw=group[
                    "chemical_input_lhv_kw"
                ].to_numpy(dtype=float),
                net_power_kw=net_power,
                auxiliary_power_kw=group["auxiliary_power_kw"].to_numpy(
                    dtype=float
                ),
            )
        self.curves = curves

    def modes(self, max_active_stacks: int = 2) -> tuple[tuple[str, ...], ...]:
        if not 1 <= max_active_stacks <= len(self.stack_ids):
            raise ValueError("max_active_stacks is outside the stack count")
        return tuple(
            mode
            for size in range(1, max_active_stacks + 1)
            for mode in combinations(self.stack_ids, size)
        )

    def solve_mode(
        self,
        demand_net_power_kw: float,
        mode: Iterable[str],
    ) -> ChenDispatchSolution | None:
        demand = float(demand_net_power_kw)
        if not np.isfinite(demand) or demand < 0:
            raise ValueError("demand_net_power_kw must be finite and non-negative")
        mode = self._canonical_mode(mode)
        if not mode:
            return (
                self._build_solution(demand, (), ())
                if np.isclose(demand, 0.0)
                else None
            )
        if len(mode) > 2:
            raise ValueError("the current exact solver supports at most two active stacks")
        if np.isclose(demand, 0.0):
            return None

        if len(mode) == 1:
            curve = self.curves[mode[0]]
            if not self._within(
                demand,
                curve.minimum_net_power_kw,
                curve.maximum_net_power_kw,
            ):
                return None
            return self._build_solution(demand, mode, (demand,))

        first, second = (self.curves[stack_id] for stack_id in mode)
        lower = max(
            first.minimum_net_power_kw,
            demand - second.maximum_net_power_kw,
        )
        upper = min(
            first.maximum_net_power_kw,
            demand - second.minimum_net_power_kw,
        )
        if lower > upper + 1e-10:
            return None

        candidates = [lower, upper]
        candidates.extend(
            first.net_power_kw[
                (first.net_power_kw >= lower - 1e-10)
                & (first.net_power_kw <= upper + 1e-10)
            ]
        )
        reflected = demand - second.net_power_kw
        candidates.extend(
            reflected[
                (reflected >= lower - 1e-10)
                & (reflected <= upper + 1e-10)
            ]
        )
        candidates = np.unique(np.clip(candidates, lower, upper))
        objective = np.asarray(
            [
                self._interpolate(first, value, "chemical_input_lhv_kw")
                + self._interpolate(
                    second,
                    demand - value,
                    "chemical_input_lhv_kw",
                )
                for value in candidates
            ]
        )
        best_index = int(np.argmin(objective))
        first_power = float(candidates[best_index])
        return self._build_solution(
            demand,
            mode,
            (first_power, demand - first_power),
        )

    def evaluate_allocation(
        self,
        demand_net_power_kw: float,
        stack_net_power_kw: dict[str, float],
    ) -> ChenDispatchSolution:
        """Evaluate an externally selected feasible allocation on the same curves."""

        demand = float(demand_net_power_kw)
        if not np.isfinite(demand) or demand < 0:
            raise ValueError("demand_net_power_kw must be finite and non-negative")
        unknown = set(stack_net_power_kw).difference(self.stack_ids)
        if unknown:
            raise ValueError(f"unknown Chen stacks: {sorted(unknown)}")
        if any(not np.isfinite(value) or value < 0 for value in stack_net_power_kw.values()):
            raise ValueError("stack powers must be finite and non-negative")
        active = {
            stack_id: float(stack_net_power_kw.get(stack_id, 0.0))
            for stack_id in self.stack_ids
            if stack_net_power_kw.get(stack_id, 0.0) > 1e-10
        }
        if len(active) > 2:
            raise ValueError("the current N+1 dispatch allows at most two active stacks")
        if not np.isclose(sum(active.values()), demand, atol=1e-8):
            raise ValueError("external allocation does not balance net power demand")
        for stack_id, power in active.items():
            curve = self.curves[stack_id]
            if not self._within(
                power,
                curve.minimum_net_power_kw,
                curve.maximum_net_power_kw,
            ):
                raise ValueError(f"{stack_id} power is outside its interpolation domain")
        mode = tuple(active)
        return self._build_solution(
            demand,
            mode,
            tuple(active.values()),
        )

    def solve_all_modes(
        self,
        demand_net_power_kw: float,
        *,
        max_active_stacks: int = 2,
    ) -> dict[tuple[str, ...], ChenDispatchSolution]:
        solutions = {}
        if np.isclose(demand_net_power_kw, 0.0):
            off = self.solve_mode(demand_net_power_kw, ())
            if off is not None:
                solutions[()] = off
            return solutions
        for mode in self.modes(max_active_stacks):
            solution = self.solve_mode(demand_net_power_kw, mode)
            if solution is not None:
                solutions[mode] = solution
        return solutions

    def solve_instantaneous(
        self,
        demand_net_power_kw: float,
        *,
        max_active_stacks: int = 2,
    ) -> ChenDispatchSolution:
        solutions = self.solve_all_modes(
            demand_net_power_kw,
            max_active_stacks=max_active_stacks,
        )
        if not solutions:
            raise ValueError(
                f"no feasible Chen dispatch for demand {demand_net_power_kw:.6g} kW"
            )
        return min(
            solutions.values(),
            key=lambda item: (item.total_chemical_input_lhv_kw, item.mode),
        )

    def _canonical_mode(self, mode: Iterable[str]) -> tuple[str, ...]:
        requested = tuple(mode)
        if len(set(requested)) != len(requested):
            raise ValueError("mode contains a duplicate stack")
        unknown = set(requested).difference(self.stack_ids)
        if unknown:
            raise ValueError(f"unknown Chen stacks: {sorted(unknown)}")
        return tuple(stack_id for stack_id in self.stack_ids if stack_id in requested)

    @staticmethod
    def _within(value: float, lower: float, upper: float) -> bool:
        return lower - 1e-10 <= value <= upper + 1e-10

    @staticmethod
    def _interpolate(
        curve: ChenStackCurve,
        net_power_kw: float,
        field: str,
    ) -> float:
        values = getattr(curve, field)
        return float(np.interp(net_power_kw, curve.net_power_kw, values))

    def _build_solution(
        self,
        demand_net_power_kw: float,
        mode: tuple[str, ...],
        active_net_power_kw: tuple[float, ...],
    ) -> ChenDispatchSolution:
        active = dict(zip(mode, active_net_power_kw))
        net_power = []
        gross_power = []
        efficiency = []
        chemical_input = []
        current_density = []
        for stack_id in self.stack_ids:
            power = float(active.get(stack_id, 0.0))
            net_power.append(power)
            if power == 0.0:
                gross_power.append(0.0)
                efficiency.append(0.0)
                chemical_input.append(0.0)
                current_density.append(0.0)
                continue
            curve = self.curves[stack_id]
            gross_power.append(
                self._interpolate(curve, power, "gross_power_kw")
            )
            efficiency.append(
                self._interpolate(curve, power, "efficiency_lhv_pct")
            )
            chemical_input.append(
                self._interpolate(curve, power, "chemical_input_lhv_kw")
            )
            current_density.append(
                self._interpolate(curve, power, "current_density_a_cm2")
            )

        total_net = float(sum(net_power))
        total_chemical = float(sum(chemical_input))
        system_efficiency = (
            100.0 * total_net / total_chemical if total_chemical > 0 else 0.0
        )
        return ChenDispatchSolution(
            demand_net_power_kw=demand_net_power_kw,
            mode=mode,
            stack_ids=self.stack_ids,
            stack_net_power_kw=tuple(net_power),
            stack_gross_power_kw=tuple(gross_power),
            stack_efficiency_lhv_pct=tuple(efficiency),
            stack_chemical_input_lhv_kw=tuple(chemical_input),
            stack_current_density_a_cm2=tuple(current_density),
            total_chemical_input_lhv_kw=total_chemical,
            hydrogen_g_per_s=total_chemical / 120.0,
            system_efficiency_lhv_pct=system_efficiency,
            power_balance_error_kw=total_net - demand_net_power_kw,
        )


def changed_stack_states(
    previous_mode: Iterable[str],
    next_mode: Iterable[str],
) -> int:
    """Count binary on/off state changes between two active-stack sets."""

    return len(set(previous_mode).symmetric_difference(next_mode))
