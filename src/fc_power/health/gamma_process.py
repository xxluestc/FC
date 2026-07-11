"""Action-dependent Gamma-process health-state transitions.

The cumulative state ``degradation`` is a non-negative, irreversible health
indicator.  It is deliberately kept separate from the existing performance
loss proxy: the proxy scores an action at the current health state, whereas
this module predicts how an action changes the future health state.

No literature coefficients are embedded here.  Rates and event increments
must be supplied explicitly after their units and provenance are audited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadRateMap:
    """Mean load-amplitude degradation rate at reference current points.

    ``mean_rate_per_hour`` and the cumulative degradation state must use the
    same user-defined degradation unit.  Linear interpolation is used between
    current points and endpoint values are used outside the table.
    """

    current_a: tuple[float, ...]
    mean_rate_per_hour: tuple[float, ...]

    def __post_init__(self) -> None:
        current = np.asarray(self.current_a, dtype=float)
        rate = np.asarray(self.mean_rate_per_hour, dtype=float)
        if current.ndim != 1 or rate.ndim != 1 or current.size != rate.size:
            raise ValueError("current_a and mean_rate_per_hour must be equal 1-D arrays")
        if current.size < 2:
            raise ValueError("at least two current reference points are required")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(rate)):
            raise ValueError("rate map values must be finite")
        if np.any(np.diff(current) <= 0):
            raise ValueError("current_a must be strictly increasing")
        if np.any(current < 0) or np.any(rate < 0):
            raise ValueError("current and degradation rates must be non-negative")

    def rate_at(self, current_a: float) -> float:
        if not np.isfinite(current_a) or current_a < 0:
            raise ValueError("current_a must be finite and non-negative")
        return float(np.interp(current_a, self.current_a, self.mean_rate_per_hour))


@dataclass(frozen=True)
class GammaHealthParams:
    """Explicit parameters for one stack's health transition model."""

    load_rate_map: LoadRateMap
    gamma_scale: float
    natural_rate_per_hour: float = 0.0
    off_rate_per_hour: float = 0.0
    ramp_increment_per_amp: float = 0.0
    shift_increment: float = 0.0
    shift_threshold_a: float | None = None
    start_increment: float = 0.0
    stop_increment: float = 0.0
    on_threshold_a: float = 1e-9
    heterogeneity_factor: float = 1.0
    failure_threshold: float | None = None

    def __post_init__(self) -> None:
        non_negative = {
            "gamma_scale": self.gamma_scale,
            "natural_rate_per_hour": self.natural_rate_per_hour,
            "off_rate_per_hour": self.off_rate_per_hour,
            "ramp_increment_per_amp": self.ramp_increment_per_amp,
            "shift_increment": self.shift_increment,
            "start_increment": self.start_increment,
            "stop_increment": self.stop_increment,
            "on_threshold_a": self.on_threshold_a,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gamma_scale <= 0:
            raise ValueError("gamma_scale must be positive")
        if self.shift_threshold_a is not None and (
            not np.isfinite(self.shift_threshold_a) or self.shift_threshold_a < 0
        ):
            raise ValueError("shift_threshold_a must be finite and non-negative")
        if not np.isfinite(self.heterogeneity_factor) or self.heterogeneity_factor <= 0:
            raise ValueError("heterogeneity_factor must be finite and positive")
        if self.failure_threshold is not None and (
            not np.isfinite(self.failure_threshold) or self.failure_threshold <= 0
        ):
            raise ValueError("failure_threshold must be finite and positive")


@dataclass(frozen=True)
class GammaHealthState:
    """Minimal observable state carried between controller decisions."""

    degradation: float = 0.0
    current_a: float = 0.0
    is_on: bool = False
    start_count: int = 0
    stop_count: int = 0
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.degradation) or self.degradation < 0:
            raise ValueError("degradation must be finite and non-negative")
        if not np.isfinite(self.current_a) or self.current_a < 0:
            raise ValueError("current_a must be finite and non-negative")
        if self.start_count < 0 or self.stop_count < 0:
            raise ValueError("event counts must be non-negative")
        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and non-negative")


@dataclass(frozen=True)
class HealthTransition:
    """Auditable decomposition of one health-state transition."""

    state: GammaHealthState
    total_increment: float
    load_increment: float
    expected_load_increment: float
    natural_increment: float
    ramp_increment: float
    shift_increment: float
    start_stop_increment: float
    stochastic: bool


class GammaHealthModel:
    """Predict a stack's next cumulative degradation under one action."""

    def __init__(self, params: GammaHealthParams):
        self.params = params

    def expected_load_increment(
        self, current_a: float, dt_s: float, *, is_on: bool | None = None
    ) -> float:
        """Return the expected intrinsic increment before random sampling."""

        self._validate_step(current_a, dt_s)
        operating = current_a > self.params.on_threshold_a if is_on is None else is_on
        if operating:
            rate = self.params.load_rate_map.rate_at(current_a)
        else:
            rate = self.params.off_rate_per_hour
        return rate * self.params.heterogeneity_factor * dt_s / 3600.0

    def transition(
        self,
        state: GammaHealthState,
        current_a: float,
        dt_s: float = 1.0,
        *,
        stochastic: bool = True,
        rng: np.random.Generator | None = None,
        next_on: bool | None = None,
        shift_event: bool | None = None,
    ) -> HealthTransition:
        """Predict one action-dependent transition without mutating ``state``.

        Load-amplitude deterioration is sampled from a Gamma distribution.
        Load-change and on/off event increments are deterministic because the
        Zuo-style decomposition models these mechanisms separately.  Setting
        ``stochastic=False`` returns the exact conditional mean and is intended
        for controller debugging and deterministic baselines.
        """

        self._validate_step(current_a, dt_s)
        if not isinstance(state, GammaHealthState):
            raise TypeError("state must be GammaHealthState")

        operating = (
            current_a > self.params.on_threshold_a if next_on is None else bool(next_on)
        )
        if not operating and current_a > self.params.on_threshold_a:
            raise ValueError("a stopped stack cannot have positive current")
        expected_load = self.expected_load_increment(
            current_a, dt_s, is_on=operating
        )
        if stochastic and expected_load > 0:
            generator = np.random.default_rng() if rng is None else rng
            shape = expected_load / self.params.gamma_scale
            load_increment = float(generator.gamma(shape, self.params.gamma_scale))
        else:
            load_increment = expected_load

        natural_increment = 0.0
        if operating:
            natural_increment = (
                self.params.natural_rate_per_hour
                * self.params.heterogeneity_factor
                * dt_s
                / 3600.0
            )
        ramp_increment = (
            self.params.ramp_increment_per_amp
            * abs(current_a - state.current_a)
            * self.params.heterogeneity_factor
        )

        if shift_event is None:
            shifted = (
                self.params.shift_threshold_a is not None
                and abs(current_a - state.current_a) > self.params.shift_threshold_a
            )
        else:
            shifted = bool(shift_event)
        shift_increment = (
            self.params.shift_increment
            * int(shifted)
            * self.params.heterogeneity_factor
        )

        started = operating and not state.is_on
        stopped = state.is_on and not operating
        start_stop_increment = self.params.heterogeneity_factor * (
            self.params.start_increment * int(started)
            + self.params.stop_increment * int(stopped)
        )

        total = (
            load_increment
            + natural_increment
            + ramp_increment
            + shift_increment
            + start_stop_increment
        )
        next_state = GammaHealthState(
            degradation=state.degradation + total,
            current_a=float(current_a),
            is_on=operating,
            start_count=state.start_count + int(started),
            stop_count=state.stop_count + int(stopped),
            elapsed_s=state.elapsed_s + dt_s,
        )
        return HealthTransition(
            state=next_state,
            total_increment=total,
            load_increment=load_increment,
            expected_load_increment=expected_load,
            natural_increment=natural_increment,
            ramp_increment=ramp_increment,
            shift_increment=shift_increment,
            start_stop_increment=start_stop_increment,
            stochastic=stochastic,
        )

    def soh(self, state: GammaHealthState) -> float | None:
        """Map degradation to a normalized SOH if a failure threshold is known."""

        if self.params.failure_threshold is None:
            return None
        return float(np.clip(1.0 - state.degradation / self.params.failure_threshold, 0, 1))

    @staticmethod
    def _validate_step(current_a: float, dt_s: float) -> None:
        if not np.isfinite(current_a) or current_a < 0:
            raise ValueError("current_a must be finite and non-negative")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
