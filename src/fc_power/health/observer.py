"""Auditable prediction/correction boundary for stack degradation observations.

The observer is deliberately state-free.  Its belief is carried explicitly so
candidate-action rollouts cannot mutate or consume measurements.  Measurements
are expressed as degradation-proxy observations; a future voltage/current UKF
adapter must establish that mapping before real-vehicle correction is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from fc_power.health.gamma_process import GammaHealthState


@dataclass(frozen=True)
class DegradationObservation:
    """One time-aligned scalar observation of cumulative degradation."""

    degradation_pct: float
    variance_pct2: float
    elapsed_s: float
    source: str
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.degradation_pct) or self.degradation_pct < 0:
            raise ValueError("degradation_pct must be finite and non-negative")
        if not np.isfinite(self.variance_pct2) or self.variance_pct2 <= 0:
            raise ValueError("variance_pct2 must be finite and positive")
        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and non-negative")
        if not self.source.strip():
            raise ValueError("source must be non-empty")


@dataclass(frozen=True)
class HealthBelief:
    """Observer belief carried alongside a world-model health state."""

    state: GammaHealthState
    variance_pct2: float
    correction_count: int = 0
    last_observation_elapsed_s: float | None = None
    last_observation_source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, GammaHealthState):
            raise TypeError("state must be GammaHealthState")
        if not np.isfinite(self.variance_pct2) or self.variance_pct2 < 0:
            raise ValueError("variance_pct2 must be finite and non-negative")
        if self.correction_count < 0:
            raise ValueError("correction_count must be non-negative")
        if self.last_observation_elapsed_s is not None and (
            not np.isfinite(self.last_observation_elapsed_s)
            or self.last_observation_elapsed_s < 0
        ):
            raise ValueError(
                "last_observation_elapsed_s must be finite and non-negative"
            )
        if self.last_observation_source is not None and not str(
            self.last_observation_source
        ).strip():
            raise ValueError("last_observation_source must be non-empty or None")


@dataclass(frozen=True)
class HealthCorrectionAudit:
    """Scalar correction details retained for provenance and diagnostics."""

    observation: DegradationObservation
    predicted_degradation_pct: float
    innovation_pct: float
    kalman_gain: float
    raw_posterior_degradation_pct: float
    posterior_degradation_pct: float
    predicted_variance_pct2: float
    posterior_variance_pct2: float
    monotonic_projection_applied: bool


@dataclass(frozen=True)
class HealthObserverUpdate:
    """Prediction and optional observation correction for one executed step."""

    prediction: HealthBelief
    posterior: HealthBelief
    correction: HealthCorrectionAudit | None


class HealthObserver(Protocol):
    """Pure prediction/correction protocol for one stack."""

    def initialize(self, state: GammaHealthState) -> HealthBelief: ...

    def predict(
        self,
        prior: HealthBelief,
        predicted_state: GammaHealthState,
        *,
        expected_gamma_increment_pct: float,
    ) -> HealthBelief: ...

    def correct(
        self,
        prediction: HealthBelief,
        observation: DegradationObservation,
        *,
        monotonic_lower_bound_pct: float,
    ) -> tuple[HealthBelief, HealthCorrectionAudit]: ...


@dataclass(frozen=True)
class GaussianDegradationObserver:
    """Scalar Gaussian belief update around the Gamma conditional mean.

    Only the continuous Gamma load term contributes ``mean * scale`` process
    variance.  Start/stop, ramp and load-shift increments remain deterministic
    because the current degradation model treats those event channels as fixed.
    """

    gamma_scale_pct: float
    initial_variance_pct2: float = 0.0
    process_variance_rate_pct2_per_hour: float = 0.0
    timestamp_tolerance_s: float = 1e-6

    def __post_init__(self) -> None:
        positive = {
            "gamma_scale_pct": self.gamma_scale_pct,
            "timestamp_tolerance_s": self.timestamp_tolerance_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "initial_variance_pct2": self.initial_variance_pct2,
            "process_variance_rate_pct2_per_hour": (
                self.process_variance_rate_pct2_per_hour
            ),
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def initialize(self, state: GammaHealthState) -> HealthBelief:
        return HealthBelief(state=state, variance_pct2=self.initial_variance_pct2)

    def predict(
        self,
        prior: HealthBelief,
        predicted_state: GammaHealthState,
        *,
        expected_gamma_increment_pct: float,
    ) -> HealthBelief:
        if not isinstance(prior, HealthBelief):
            raise TypeError("prior must be HealthBelief")
        if not isinstance(predicted_state, GammaHealthState):
            raise TypeError("predicted_state must be GammaHealthState")
        if (
            not np.isfinite(expected_gamma_increment_pct)
            or expected_gamma_increment_pct < 0
        ):
            raise ValueError(
                "expected_gamma_increment_pct must be finite and non-negative"
            )
        dt_s = predicted_state.elapsed_s - prior.state.elapsed_s
        if dt_s <= 0:
            raise ValueError("predicted_state must advance observer time")
        if predicted_state.degradation + 1e-15 < prior.state.degradation:
            raise ValueError("predicted cumulative degradation cannot decrease")

        gamma_variance = expected_gamma_increment_pct * self.gamma_scale_pct
        model_variance = self.process_variance_rate_pct2_per_hour * dt_s / 3600.0
        return HealthBelief(
            state=predicted_state,
            variance_pct2=prior.variance_pct2 + gamma_variance + model_variance,
            correction_count=prior.correction_count,
            last_observation_elapsed_s=prior.last_observation_elapsed_s,
            last_observation_source=prior.last_observation_source,
        )

    def correct(
        self,
        prediction: HealthBelief,
        observation: DegradationObservation,
        *,
        monotonic_lower_bound_pct: float,
    ) -> tuple[HealthBelief, HealthCorrectionAudit]:
        if not isinstance(prediction, HealthBelief):
            raise TypeError("prediction must be HealthBelief")
        if not isinstance(observation, DegradationObservation):
            raise TypeError("observation must be DegradationObservation")
        if (
            not np.isfinite(monotonic_lower_bound_pct)
            or monotonic_lower_bound_pct < 0
        ):
            raise ValueError(
                "monotonic_lower_bound_pct must be finite and non-negative"
            )
        if (
            abs(observation.elapsed_s - prediction.state.elapsed_s)
            > self.timestamp_tolerance_s
        ):
            raise ValueError("observation timestamp does not match prediction")
        if monotonic_lower_bound_pct > prediction.state.degradation + 1e-12:
            raise ValueError("monotonic lower bound cannot exceed prediction")

        predicted_variance = prediction.variance_pct2
        gain = predicted_variance / (
            predicted_variance + observation.variance_pct2
        )
        innovation = observation.degradation_pct - prediction.state.degradation
        raw_posterior = prediction.state.degradation + gain * innovation
        posterior_damage = max(monotonic_lower_bound_pct, raw_posterior)
        projected = posterior_damage > raw_posterior + 1e-15
        posterior_variance = (1.0 - gain) * predicted_variance
        posterior = HealthBelief(
            state=replace(prediction.state, degradation=float(posterior_damage)),
            variance_pct2=float(posterior_variance),
            correction_count=prediction.correction_count + 1,
            last_observation_elapsed_s=observation.elapsed_s,
            last_observation_source=observation.source,
        )
        audit = HealthCorrectionAudit(
            observation=observation,
            predicted_degradation_pct=prediction.state.degradation,
            innovation_pct=float(innovation),
            kalman_gain=float(gain),
            raw_posterior_degradation_pct=float(raw_posterior),
            posterior_degradation_pct=float(posterior_damage),
            predicted_variance_pct2=float(predicted_variance),
            posterior_variance_pct2=float(posterior_variance),
            monotonic_projection_applied=projected,
        )
        return posterior, audit

    def update(
        self,
        prior: HealthBelief,
        predicted_state: GammaHealthState,
        *,
        expected_gamma_increment_pct: float,
        observation: DegradationObservation | None = None,
    ) -> HealthObserverUpdate:
        prediction = self.predict(
            prior,
            predicted_state,
            expected_gamma_increment_pct=expected_gamma_increment_pct,
        )
        if observation is None:
            return HealthObserverUpdate(prediction, prediction, None)
        posterior, audit = self.correct(
            prediction,
            observation,
            monotonic_lower_bound_pct=prior.state.degradation,
        )
        return HealthObserverUpdate(prediction, posterior, audit)
