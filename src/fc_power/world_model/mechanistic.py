"""Auditable multi-stack PEMFC/battery one-step world model.

The model is intentionally controller-agnostic.  Enumerative MPC, Beam search,
Dreamer and other policies must all call the same ``step`` method so that power
balance, health dynamics and safety constraints remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fc_power.battery_model import BatteryParams, next_soc, throughput_cost
from fc_power.health.dynamic_proxy import DynamicPerformanceLossProxy
from fc_power.health.gamma_process import GammaHealthModel, GammaHealthState
from fc_power.hydrogen_model import faraday_h2_g_s


@dataclass(frozen=True)
class StackControlState:
    """Health and discrete-control memory for one stack."""

    health: GammaHealthState = field(default_factory=GammaHealthState)
    dwell_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dwell_s) or self.dwell_s < 0:
            raise ValueError("dwell_s must be finite and non-negative")


@dataclass(frozen=True)
class MultiStackState:
    """State carried between online power-allocation decisions."""

    soc: float
    stacks: tuple[StackControlState, ...]
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.soc):
            raise ValueError("soc must be finite")
        if not self.stacks:
            raise ValueError("at least one stack state is required")
        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and non-negative")


@dataclass(frozen=True)
class MultiStackAction:
    """Requested current and energized state for every stack."""

    current_a: tuple[float, ...]
    is_on: tuple[bool, ...]

    @classmethod
    def from_currents(cls, current_a, is_on=None):
        currents = tuple(float(value) for value in current_a)
        if is_on is None:
            on = tuple(value > 0 for value in currents)
        else:
            on = tuple(bool(value) for value in is_on)
        return cls(currents, on)

    def __post_init__(self) -> None:
        if not self.current_a or len(self.current_a) != len(self.is_on):
            raise ValueError("current_a and is_on must have the same non-zero length")
        if any(not np.isfinite(value) or value < 0 for value in self.current_a):
            raise ValueError("stack currents must be finite and non-negative")


@dataclass(frozen=True)
class WorldCostWeights:
    """Weights applied to dimensionless, explicitly reported cost terms."""

    hydrogen: float = 0.45
    degradation_increment: float = 1.0
    performance_loss: float = 0.25
    battery_use: float = 1.5
    soc: float = 3.0
    switch: float = 0.08
    ramp: float = 0.005

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")


@dataclass(frozen=True)
class WorldModelConfig:
    """Time scale, action space and hard operating constraints."""

    dt_s: float = 1.0
    battery: BatteryParams = field(default_factory=BatteryParams)
    allowed_currents_a: tuple[float, ...] = (
        0.0,
        25.0,
        90.0,
        120.0,
        160.0,
        195.0,
        270.0,
        370.0,
    )
    min_dwell_s: float = 15.0
    max_ramp_a_per_s: float | None = None
    soc_reference: float = 0.70
    soc_feedback_kw_per_soc: float = 1200.0
    power_balance_tolerance_kw: float = 1e-9
    current_match_tolerance_a: float = 1e-9
    weights: WorldCostWeights = field(default_factory=WorldCostWeights)

    def __post_init__(self) -> None:
        currents = np.asarray(self.allowed_currents_a, dtype=float)
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if currents.ndim != 1 or currents.size < 2:
            raise ValueError("allowed_currents_a must contain at least two levels")
        if np.any(~np.isfinite(currents)) or np.any(currents < 0):
            raise ValueError("allowed currents must be finite and non-negative")
        if np.any(np.diff(currents) <= 0):
            raise ValueError("allowed currents must be strictly increasing")
        if not np.isfinite(self.min_dwell_s) or self.min_dwell_s < 0:
            raise ValueError("min_dwell_s must be finite and non-negative")
        if self.max_ramp_a_per_s is not None and (
            not np.isfinite(self.max_ramp_a_per_s) or self.max_ramp_a_per_s <= 0
        ):
            raise ValueError("max_ramp_a_per_s must be finite and positive")
        if not self.battery.soc_min <= self.soc_reference <= self.battery.soc_max:
            raise ValueError("soc_reference must be within battery SOC limits")
        if (
            not np.isfinite(self.soc_feedback_kw_per_soc)
            or self.soc_feedback_kw_per_soc < 0
        ):
            raise ValueError("soc feedback gain must be finite and non-negative")
        if self.power_balance_tolerance_kw < 0:
            raise ValueError("power balance tolerance must be non-negative")
        if self.current_match_tolerance_a < 0:
            raise ValueError("current match tolerance must be non-negative")


@dataclass(frozen=True)
class StackStep:
    stack_index: int
    current_a: float
    is_on: bool
    cell_voltage_v: float
    power_kw: float
    hydrogen_g: float
    degradation_increment_pct: float
    degradation_after_pct: float
    theta_reported: tuple[float, float, float]
    normalized_performance_loss: float
    switched: bool
    shifted_load: bool


@dataclass(frozen=True)
class CostBreakdown:
    total: float
    hydrogen: float
    degradation_increment: float
    performance_loss: float
    battery_use: float
    soc: float
    switch: float
    ramp: float
    raw_hydrogen_g: float
    raw_degradation_increment_pct: float
    raw_battery_throughput_kwh: float


@dataclass(frozen=True)
class ConstraintInfo:
    feasible: bool
    violations: tuple[str, ...]
    demand_power_kw: float
    stack_power_kw: float
    battery_power_kw: float
    power_balance_error_kw: float
    next_soc: float
    safety_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldStep:
    next_state: MultiStackState
    stacks: tuple[StackStep, ...]
    cost: CostBreakdown
    constraints: ConstraintInfo


class MechanisticMultiStackWorldModel:
    """Joint PEMFC health, IV, hydrogen and battery transition model."""

    def __init__(
        self,
        health_models: tuple[GammaHealthModel, ...],
        performance_proxies: tuple[DynamicPerformanceLossProxy, ...],
        config: WorldModelConfig = WorldModelConfig(),
    ):
        if not health_models or len(health_models) != len(performance_proxies):
            raise ValueError("one health model and proxy are required per stack")
        self.health_models = tuple(health_models)
        self.performance_proxies = tuple(performance_proxies)
        self.config = config

    @property
    def n_stacks(self) -> int:
        return len(self.health_models)

    def initial_state(
        self,
        soc: float | None = None,
        degradation_pct=None,
        current_a=None,
        is_on=None,
    ) -> MultiStackState:
        """Create an immediately actionable initial state."""

        n = self.n_stacks
        damages = np.zeros(n) if degradation_pct is None else np.asarray(degradation_pct)
        currents = np.zeros(n) if current_a is None else np.asarray(current_a)
        on = currents > 0 if is_on is None else np.asarray(is_on, dtype=bool)
        if damages.shape != (n,) or currents.shape != (n,) or on.shape != (n,):
            raise ValueError("initial stack vectors must match n_stacks")
        stacks = tuple(
            StackControlState(
                health=GammaHealthState(
                    degradation=float(damages[index]),
                    current_a=float(currents[index]),
                    is_on=bool(on[index]),
                ),
                dwell_s=self.config.min_dwell_s,
            )
            for index in range(n)
        )
        return MultiStackState(
            soc=self.config.soc_reference if soc is None else float(soc),
            stacks=stacks,
        )

    def step(
        self,
        state: MultiStackState,
        action: MultiStackAction,
        demand_power_kw: float,
        *,
        stochastic_health: bool = False,
        rng: np.random.Generator | None = None,
        allow_dwell_override: bool = False,
    ) -> WorldStep:
        """Advance all stacks and the battery by one control interval."""

        if not isinstance(state, MultiStackState):
            raise TypeError("state must be MultiStackState")
        if not isinstance(action, MultiStackAction):
            raise TypeError("action must be MultiStackAction")
        if len(state.stacks) != self.n_stacks or len(action.current_a) != self.n_stacks:
            raise ValueError("state/action stack count does not match the model")
        if not np.isfinite(demand_power_kw):
            raise ValueError("demand_power_kw must be finite")

        violations: list[str] = []
        safety_overrides: list[str] = []
        stack_steps: list[StackStep] = []
        next_stack_states: list[StackControlState] = []
        allowed = np.asarray(self.config.allowed_currents_a)
        generator = np.random.default_rng() if rng is None else rng

        for index, (stack_state, requested_current, requested_on) in enumerate(
            zip(state.stacks, action.current_a, action.is_on)
        ):
            on = bool(requested_on or requested_current > 0)
            if requested_current > 0 and not requested_on:
                violations.append(f"stack_{index}:positive_current_while_off")
            if np.min(np.abs(allowed - requested_current)) > self.config.current_match_tolerance_a:
                violations.append(f"stack_{index}:current_not_in_action_grid")

            previous = stack_state.health
            changed = on != previous.is_on or not np.isclose(
                requested_current,
                previous.current_a,
                atol=self.config.current_match_tolerance_a,
                rtol=0,
            )
            if changed and stack_state.dwell_s + 1e-12 < self.config.min_dwell_s:
                event = f"stack_{index}:minimum_dwell"
                if allow_dwell_override:
                    safety_overrides.append(event)
                else:
                    violations.append(event)
            ramp = abs(requested_current - previous.current_a)
            if (
                self.config.max_ramp_a_per_s is not None
                and ramp / self.config.dt_s > self.config.max_ramp_a_per_s + 1e-12
            ):
                violations.append(f"stack_{index}:ramp_limit")

            shifted_load = previous.is_on and on and ramp > self.config.current_match_tolerance_a
            transition = self.health_models[index].transition(
                previous,
                requested_current,
                dt_s=self.config.dt_s,
                stochastic=stochastic_health,
                rng=generator,
                next_on=on,
                shift_event=shifted_load,
            )
            proxy_result = self.performance_proxies[index].evaluate(
                transition.state.degradation,
                [requested_current],
                dt_s=self.config.dt_s,
            )
            power_kw = float(proxy_result["stack_power_kw"][0]) if on else 0.0
            voltage_v = float(proxy_result["current_cell_voltage_v"][0])
            hydrogen_g = (
                float(faraday_h2_g_s(requested_current) * self.config.dt_s)
                if on
                else 0.0
            )
            theta = tuple(
                float(value) for value in proxy_result["theta_reported"].tolist()
            )
            stack_steps.append(
                StackStep(
                    stack_index=index,
                    current_a=requested_current,
                    is_on=on,
                    cell_voltage_v=voltage_v,
                    power_kw=power_kw,
                    hydrogen_g=hydrogen_g,
                    degradation_increment_pct=transition.total_increment,
                    degradation_after_pct=transition.state.degradation,
                    theta_reported=theta,
                    normalized_performance_loss=float(
                        proxy_result["normalized_proxy"][0]
                    ),
                    switched=on != previous.is_on,
                    shifted_load=shifted_load,
                )
            )
            next_stack_states.append(
                StackControlState(
                    health=transition.state,
                    dwell_s=(
                        self.config.dt_s
                        if changed
                        else stack_state.dwell_s + self.config.dt_s
                    ),
                )
            )

        total_stack_power = float(sum(item.power_kw for item in stack_steps))
        battery_power = float(demand_power_kw - total_stack_power)
        next_battery_soc = float(
            next_soc(state.soc, battery_power, self.config.dt_s, self.config.battery)
        )
        balance_error = total_stack_power + battery_power - demand_power_kw
        battery = self.config.battery
        if battery_power < battery.charge_power_limit_kw - 1e-12:
            violations.append("battery:charge_power_limit")
        if battery_power > battery.discharge_power_limit_kw + 1e-12:
            violations.append("battery:discharge_power_limit")
        if next_battery_soc < battery.soc_min - 1e-12:
            violations.append("battery:soc_min")
        if next_battery_soc > battery.soc_max + 1e-12:
            violations.append("battery:soc_max")
        if abs(balance_error) > self.config.power_balance_tolerance_kw:
            violations.append("system:power_balance")

        next_state = MultiStackState(
            soc=next_battery_soc,
            stacks=tuple(next_stack_states),
            elapsed_s=state.elapsed_s + self.config.dt_s,
        )
        cost = self._cost(state, next_state, stack_steps, battery_power)
        constraints = ConstraintInfo(
            feasible=not violations,
            violations=tuple(violations),
            demand_power_kw=float(demand_power_kw),
            stack_power_kw=total_stack_power,
            battery_power_kw=battery_power,
            power_balance_error_kw=float(balance_error),
            next_soc=next_battery_soc,
            safety_overrides=tuple(safety_overrides),
        )
        return WorldStep(next_state, tuple(stack_steps), cost, constraints)

    def _cost(
        self,
        state: MultiStackState,
        next_state: MultiStackState,
        stack_steps: list[StackStep],
        battery_power_kw: float,
    ) -> CostBreakdown:
        n = self.n_stacks
        weights = self.config.weights
        max_current = max(self.config.allowed_currents_a)
        max_h2 = float(
            faraday_h2_g_s(max_current) * self.config.dt_s * max(n, 1)
        )
        hydrogen_raw = sum(item.hydrogen_g for item in stack_steps)
        degradation_raw = sum(
            item.degradation_increment_pct for item in stack_steps
        )
        damage_reference = sum(
            proxy.mapping.damage_reference_pct
            for proxy in self.performance_proxies
        )
        throughput_raw = float(
            throughput_cost(battery_power_kw, self.config.dt_s)
        )
        battery_power_reference = max(
            abs(self.config.battery.charge_power_limit_kw),
            abs(self.config.battery.discharge_power_limit_kw),
        )
        soc_range = self.config.battery.soc_max - self.config.battery.soc_min
        switches = sum(item.switched for item in stack_steps) / n
        ramp = sum(
            abs(item.current_a - previous.health.current_a)
            for item, previous in zip(stack_steps, state.stacks)
        ) / (n * max_current)
        desired_battery_power = self.config.soc_feedback_kw_per_soc * (
            state.soc - self.config.soc_reference
        )
        components = {
            "hydrogen": hydrogen_raw / max(max_h2, 1e-12),
            "degradation_increment": degradation_raw
            / max(damage_reference, 1e-12),
            "performance_loss": sum(
                item.normalized_performance_loss for item in stack_steps
            )
            / n,
            "battery_use": abs(battery_power_kw - desired_battery_power)
            / battery_power_reference,
            "soc": abs(next_state.soc - self.config.soc_reference) / soc_range,
            "switch": switches,
            "ramp": ramp,
        }
        total = sum(getattr(weights, name) * value for name, value in components.items())
        return CostBreakdown(
            total=float(total),
            **{name: float(value) for name, value in components.items()},
            raw_hydrogen_g=float(hydrogen_raw),
            raw_degradation_increment_pct=float(degradation_raw),
            raw_battery_throughput_kwh=throughput_raw,
        )
