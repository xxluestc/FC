"""Slow-timescale N+1 stack scheduling under aggregated Gamma exposure."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.stats import gamma


@dataclass(frozen=True)
class ServiceExposure:
    """Base damage exposure for two online roles during one service epoch."""

    duration_h: float
    continuous_mean_pct: tuple[float, float]
    load_shift_damage_pct: tuple[float, float]
    operational_start_damage_pct: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_h) or self.duration_h <= 0:
            raise ValueError("duration_h must be finite and positive")
        for name, values in (
            ("continuous_mean_pct", self.continuous_mean_pct),
            ("load_shift_damage_pct", self.load_shift_damage_pct),
            ("operational_start_damage_pct", self.operational_start_damage_pct),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (2,) or np.any(~np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} must contain two non-negative values")


@dataclass(frozen=True)
class ServiceScheduleConfig:
    """Auditable assumptions for the slow supervisory scheduler."""

    health_limit_pct: float
    gamma_scale_pct: float
    heterogeneity_factors: tuple[float, ...]
    start_damage_pct: float
    risk_horizon_h: float = 100.0
    cvar_alpha: float = 0.95
    risk_samples: int = 512
    risk_seed: int = 2026
    n_plus_one_weight: float = 0.5

    def __post_init__(self) -> None:
        positive = {
            "health_limit_pct": self.health_limit_pct,
            "gamma_scale_pct": self.gamma_scale_pct,
            "risk_horizon_h": self.risk_horizon_h,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        factors = np.asarray(self.heterogeneity_factors, dtype=float)
        if factors.size < 3 or np.any(~np.isfinite(factors)) or np.any(factors <= 0):
            raise ValueError("at least three positive heterogeneity factors are required")
        if not np.isfinite(self.start_damage_pct) or self.start_damage_pct < 0:
            raise ValueError("start_damage_pct must be finite and non-negative")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must lie in (0, 1)")
        if self.risk_samples < 20:
            raise ValueError("risk_samples must be at least 20")
        if not 0 <= self.n_plus_one_weight <= 1:
            raise ValueError("n_plus_one_weight must lie in [0, 1]")


@dataclass(frozen=True)
class ServiceScheduleState:
    damage_pct: tuple[float, ...]
    online_assignment: tuple[int, int] | None = None
    elapsed_h: float = 0.0
    start_count: int = 0

    def __post_init__(self) -> None:
        damage = np.asarray(self.damage_pct, dtype=float)
        if damage.size < 3 or np.any(~np.isfinite(damage)) or np.any(damage < 0):
            raise ValueError("damage_pct must contain at least three non-negative values")
        if self.online_assignment is not None:
            if len(set(self.online_assignment)) != 2:
                raise ValueError("online_assignment must contain two distinct stacks")
            if min(self.online_assignment) < 0 or max(self.online_assignment) >= damage.size:
                raise ValueError("online_assignment contains an invalid stack index")
        if not np.isfinite(self.elapsed_h) or self.elapsed_h < 0:
            raise ValueError("elapsed_h must be finite and non-negative")
        if self.start_count < 0:
            raise ValueError("start_count must be non-negative")


@dataclass(frozen=True)
class ServiceScheduleDecision:
    assignment: tuple[int, int]
    objective: float
    expected_max_health_fraction: float
    expected_n_plus_one_health_fraction: float
    expected_mean_health_fraction: float
    cvar_max_health_fraction: float | None
    new_starts: int


@dataclass(frozen=True)
class ServiceScheduleTransition:
    state: ServiceScheduleState
    continuous_damage_pct: tuple[float, ...]
    load_shift_damage_pct: tuple[float, ...]
    operational_start_damage_pct: tuple[float, ...]
    start_damage_pct: tuple[float, ...]


def candidate_assignments(n_stacks: int) -> tuple[tuple[int, int], ...]:
    if n_stacks < 3:
        raise ValueError("N+1 scheduling requires at least three stacks")
    return tuple(permutations(range(n_stacks), 2))


def eligible_service_assignments(
    state: ServiceScheduleState,
    health_limit_pct: float,
) -> tuple[tuple[int, int], ...]:
    """Return ordered two-stack assignments below a declared service boundary."""

    if not np.isfinite(health_limit_pct) or health_limit_pct <= 0:
        raise ValueError("health_limit_pct must be finite and positive")
    eligible = tuple(
        index
        for index, damage in enumerate(state.damage_pct)
        if damage < health_limit_pct
    )
    if len(eligible) < 2:
        return ()
    return tuple(permutations(eligible, 2))


def stationary_service_exposure(
    templates: list[ServiceExposure] | tuple[ServiceExposure, ...],
    duration_h: float,
) -> ServiceExposure:
    """Scale the mean development-template exposure to a decision epoch."""

    if not templates:
        raise ValueError("at least one service exposure template is required")
    base_duration = templates[0].duration_h
    if not all(np.isclose(item.duration_h, base_duration) for item in templates):
        raise ValueError("service exposure templates must have equal duration")
    blocks = duration_h / base_duration
    if blocks <= 0 or not np.isclose(blocks, round(blocks)):
        raise ValueError("duration_h must be an integer multiple of template duration")
    continuous = blocks * np.mean(
        [item.continuous_mean_pct for item in templates], axis=0
    )
    shifts = blocks * np.mean(
        [item.load_shift_damage_pct for item in templates], axis=0
    )
    operational_starts = blocks * np.mean(
        [item.operational_start_damage_pct for item in templates], axis=0
    )
    return ServiceExposure(
        duration_h=duration_h,
        continuous_mean_pct=tuple(float(value) for value in continuous),
        load_shift_damage_pct=tuple(float(value) for value in shifts),
        operational_start_damage_pct=tuple(
            float(value) for value in operational_starts
        ),
    )


def orient_service_pair(
    pair: tuple[int, int],
    state: ServiceScheduleState,
    exposure: ServiceExposure,
    heterogeneity_factors,
) -> tuple[int, int]:
    """Map the heavier fast-layer role to the better residual-health stack."""

    if len(set(pair)) != 2 or min(pair) < 0 or max(pair) >= len(state.damage_pct):
        raise ValueError("pair must contain two valid distinct stack indices")
    factors = np.asarray(heterogeneity_factors, dtype=float)
    if factors.shape != (len(state.damage_pct),) or np.any(factors <= 0):
        raise ValueError("heterogeneity_factors must match the service state")
    role_damage = (
        np.asarray(exposure.continuous_mean_pct)
        + np.asarray(exposure.load_shift_damage_pct)
        + np.asarray(exposure.operational_start_damage_pct)
    )
    candidates = (pair, (pair[1], pair[0]))
    return min(
        candidates,
        key=lambda assignment: (
            max(
                state.damage_pct[stack] + role_damage[role] * factors[stack]
                for role, stack in enumerate(assignment)
            ),
            assignment,
        ),
    )


def choose_service_assignment(
    state: ServiceScheduleState,
    exposure: ServiceExposure,
    config: ServiceScheduleConfig,
    *,
    objective: str = "gamma_cvar",
) -> ServiceScheduleDecision:
    """Choose two online stacks without using a future demand trajectory."""

    if len(state.damage_pct) != len(config.heterogeneity_factors):
        raise ValueError("state and heterogeneity_factors must have equal length")
    valid_objectives = {
        "expected_max",
        "expected_n_plus_one",
        "expected_order_blend",
        "expected_total",
        "gamma_cvar",
    }
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {sorted(valid_objectives)}")

    best = None
    for assignment in candidate_assignments(len(state.damage_pct)):
        decision = evaluate_service_assignment(
            state, exposure, config, assignment, objective=objective
        )
        key = (decision.objective, decision.new_starts, assignment)
        if best is None or key < best[0]:
            best = (key, decision)
    return best[1]


def evaluate_service_assignment(
    state: ServiceScheduleState,
    exposure: ServiceExposure,
    config: ServiceScheduleConfig,
    assignment: tuple[int, int],
    *,
    objective: str = "expected_max",
) -> ServiceScheduleDecision:
    """Evaluate one auditable candidate without selecting among assignments."""

    if assignment not in candidate_assignments(len(state.damage_pct)):
        raise ValueError("assignment is not a valid two-stack role mapping")
    valid_objectives = {
        "expected_max",
        "expected_n_plus_one",
        "expected_order_blend",
        "expected_total",
        "gamma_cvar",
    }
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {sorted(valid_objectives)}")
    expected_damage, starts = _project_expected_damage(
        state, exposure, config, assignment
    )
    expected_max = float(expected_damage.max() / config.health_limit_pct)
    expected_n_plus_one = float(
        np.partition(expected_damage, -2)[-2] / config.health_limit_pct
    )
    expected_mean = float(expected_damage.mean() / config.health_limit_pct)
    cvar = None
    if objective == "gamma_cvar":
        uniforms = _common_uniforms(
            config.risk_samples, len(state.damage_pct), config.risk_seed
        )
        cvar = _project_cvar(state, exposure, config, assignment, uniforms)
    score = {
        "expected_max": expected_max,
        "expected_n_plus_one": expected_n_plus_one,
        "expected_order_blend": (
            (1 - config.n_plus_one_weight) * expected_max
            + config.n_plus_one_weight * expected_n_plus_one
        ),
        "expected_total": expected_mean,
        "gamma_cvar": cvar,
    }[objective]
    return ServiceScheduleDecision(
        assignment=assignment,
        objective=float(score),
        expected_max_health_fraction=expected_max,
        expected_n_plus_one_health_fraction=expected_n_plus_one,
        expected_mean_health_fraction=expected_mean,
        cvar_max_health_fraction=cvar,
        new_starts=starts,
    )


def transition_service_epoch(
    state: ServiceScheduleState,
    exposure: ServiceExposure,
    config: ServiceScheduleConfig,
    assignment: tuple[int, int],
    *,
    stochastic: bool = True,
    rng: np.random.Generator | None = None,
    continuous_uniforms=None,
) -> ServiceScheduleTransition:
    """Execute one epoch and charge starts only when the online set changes."""

    if assignment not in candidate_assignments(len(state.damage_pct)):
        raise ValueError("assignment is not a valid two-stack role mapping")
    factors = np.asarray(config.heterogeneity_factors)
    continuous = np.zeros(len(state.damage_pct), dtype=float)
    shifts = np.zeros_like(continuous)
    operational_starts = np.zeros_like(continuous)
    starts = np.zeros_like(continuous)
    previous = set() if state.online_assignment is None else set(state.online_assignment)
    generator = np.random.default_rng() if rng is None else rng
    uniforms = None
    if continuous_uniforms is not None:
        uniforms = np.asarray(continuous_uniforms, dtype=float)
        if uniforms.shape != (2,) or np.any(~np.isfinite(uniforms)):
            raise ValueError("continuous_uniforms must contain two finite values")
        if np.any((uniforms <= 0) | (uniforms >= 1)):
            raise ValueError("continuous_uniforms must lie strictly inside (0, 1)")
    for role, stack in enumerate(assignment):
        mean = exposure.continuous_mean_pct[role] * factors[stack]
        if stochastic and mean > 0:
            if uniforms is None:
                continuous[stack] = generator.gamma(
                    shape=mean / config.gamma_scale_pct,
                    scale=config.gamma_scale_pct,
                )
            else:
                continuous[stack] = gamma.ppf(
                    uniforms[role],
                    a=mean / config.gamma_scale_pct,
                    scale=config.gamma_scale_pct,
                )
        else:
            continuous[stack] = mean
        shifts[stack] = exposure.load_shift_damage_pct[role] * factors[stack]
        operational_starts[stack] = (
            exposure.operational_start_damage_pct[role] * factors[stack]
        )
        if stack not in previous:
            starts[stack] = config.start_damage_pct * factors[stack]
    next_damage = (
        np.asarray(state.damage_pct)
        + continuous
        + shifts
        + operational_starts
        + starts
    )
    next_state = ServiceScheduleState(
        damage_pct=tuple(float(value) for value in next_damage),
        online_assignment=assignment,
        elapsed_h=state.elapsed_h + exposure.duration_h,
        start_count=state.start_count + int(np.count_nonzero(starts)),
    )
    return ServiceScheduleTransition(
        state=next_state,
        continuous_damage_pct=tuple(float(value) for value in continuous),
        load_shift_damage_pct=tuple(float(value) for value in shifts),
        operational_start_damage_pct=tuple(
            float(value) for value in operational_starts
        ),
        start_damage_pct=tuple(float(value) for value in starts),
    )


def _project_expected_damage(state, exposure, config, assignment):
    factor = config.risk_horizon_h / exposure.duration_h
    projected = np.asarray(state.damage_pct, dtype=float).copy()
    heterogeneity = np.asarray(config.heterogeneity_factors)
    previous = set() if state.online_assignment is None else set(state.online_assignment)
    starts = 0
    for role, stack in enumerate(assignment):
        projected[stack] += factor * heterogeneity[stack] * (
            exposure.continuous_mean_pct[role]
            + exposure.load_shift_damage_pct[role]
            + exposure.operational_start_damage_pct[role]
        )
        if stack not in previous:
            projected[stack] += config.start_damage_pct * heterogeneity[stack]
            starts += 1
    return projected, starts


def _project_cvar(state, exposure, config, assignment, uniforms):
    factor = config.risk_horizon_h / exposure.duration_h
    samples = np.broadcast_to(
        np.asarray(state.damage_pct, dtype=float), uniforms.shape
    ).copy()
    heterogeneity = np.asarray(config.heterogeneity_factors)
    previous = set() if state.online_assignment is None else set(state.online_assignment)
    for role, stack in enumerate(assignment):
        mean = factor * exposure.continuous_mean_pct[role] * heterogeneity[stack]
        if mean > 0:
            samples[:, stack] += gamma.ppf(
                uniforms[:, stack],
                a=mean / config.gamma_scale_pct,
                scale=config.gamma_scale_pct,
            )
        samples[:, stack] += (
            factor
            * (
                exposure.load_shift_damage_pct[role]
                + exposure.operational_start_damage_pct[role]
            )
            * heterogeneity[stack]
        )
        if stack not in previous:
            samples[:, stack] += config.start_damage_pct * heterogeneity[stack]
    maximum = samples.max(axis=1) / config.health_limit_pct
    tail_count = max(1, int(np.ceil((1 - config.cvar_alpha) * len(maximum))))
    return float(np.partition(maximum, -tail_count)[-tail_count:].mean())


def _common_uniforms(samples: int, n_stacks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((samples, n_stacks)), 1e-12, 1 - 1e-12)
