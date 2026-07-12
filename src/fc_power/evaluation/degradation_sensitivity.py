"""Fixed-action degradation exposure for coefficient robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fc_power.health.lzw_gamma_calibration import GhaderiPeiCoefficients


@dataclass(frozen=True)
class ActionExposure:
    """Mutually exclusive operating hours and additive event counts by stack."""

    natural_on_h: tuple[float, ...]
    low_load_h: tuple[float, ...]
    high_load_h: tuple[float, ...]
    start_count: tuple[int, ...]
    load_shift_count: tuple[int, ...]
    heterogeneity_factors: tuple[float, ...]
    duration_s: float

    def __post_init__(self) -> None:
        lengths = {
            len(self.natural_on_h),
            len(self.low_load_h),
            len(self.high_load_h),
            len(self.start_count),
            len(self.load_shift_count),
            len(self.heterogeneity_factors),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("all exposure vectors must have the same non-zero length")
        continuous = (*self.natural_on_h, *self.low_load_h, *self.high_load_h)
        if any(not np.isfinite(value) or value < 0 for value in continuous):
            raise ValueError("operating exposure must be finite and non-negative")
        if any(value < 0 for value in (*self.start_count, *self.load_shift_count)):
            raise ValueError("event counts must be non-negative")
        if any(
            not np.isfinite(value) or value <= 0
            for value in self.heterogeneity_factors
        ):
            raise ValueError("heterogeneity factors must be finite and positive")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")

    @property
    def n_stacks(self) -> int:
        return len(self.natural_on_h)

    def damage_by_stack(
        self, coefficients: GhaderiPeiCoefficients
    ) -> np.ndarray:
        """Evaluate literature damage for the recorded action exposure."""

        natural = np.asarray(self.natural_on_h) * coefficients.natural_on_pct_per_hour
        low = np.asarray(self.low_load_h) * coefficients.low_load_pct_per_hour
        high = np.asarray(self.high_load_h) * (
            coefficients.natural_on_pct_per_hour
            + coefficients.high_load_pct_per_hour
        )
        starts = np.asarray(self.start_count) * coefficients.start_stop_pct_per_cycle
        shifts = (
            np.asarray(self.load_shift_count) * coefficients.load_shift_pct_per_cycle
        )
        heterogeneity = np.asarray(self.heterogeneity_factors)
        return heterogeneity * (natural + low + high + starts + shifts)

    def total_damage(self, coefficients: GhaderiPeiCoefficients) -> float:
        return float(self.damage_by_stack(coefficients).sum())


def extract_action_exposure(
    trajectory: pd.DataFrame,
    n_stacks: int,
    heterogeneity_factors,
    *,
    dt_s: float = 1.0,
    maximum_current_a: float = 370.0,
    include_soc_recovery: bool = False,
    tolerance_a: float = 1e-9,
) -> ActionExposure:
    """Extract mutually exclusive regimes from one scenario-policy trajectory.

    The initial stack state is assumed off at 0 A, matching the unified
    testbed.  Recovery samples are excluded by default so every controller is
    compared on the original task exposure.
    """

    if n_stacks <= 0:
        raise ValueError("n_stacks must be positive")
    factors = tuple(float(value) for value in heterogeneity_factors)
    if len(factors) != n_stacks:
        raise ValueError("heterogeneity factors must match n_stacks")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(maximum_current_a) or maximum_current_a <= 0:
        raise ValueError("maximum_current_a must be finite and positive")
    frame = trajectory.copy()
    if not include_soc_recovery and "is_soc_recovery" in frame:
        frame = frame[~frame.is_soc_recovery]
    if len(frame) == 0:
        raise ValueError("trajectory contains no selected samples")
    if "step" in frame:
        frame = frame.sort_values("step")

    natural_h, low_h, high_h, starts, shifts = [], [], [], [], []
    high_threshold = 0.90 * maximum_current_a
    for index in range(n_stacks):
        current_column = f"stack_{index}_current_a"
        on_column = f"stack_{index}_on"
        if current_column not in frame or on_column not in frame:
            raise ValueError(f"trajectory is missing stack {index} action columns")
        current = frame[current_column].to_numpy(dtype=float)
        on = frame[on_column].to_numpy(dtype=bool)
        if np.any(~np.isfinite(current)) or np.any(current < 0):
            raise ValueError("stack currents must be finite and non-negative")
        previous_on = np.r_[False, on[:-1]]
        previous_current = np.r_[0.0, current[:-1]]
        start = on & ~previous_on
        shift = (
            on
            & previous_on
            & (np.abs(current - previous_current) > tolerance_a)
        )
        low = on & (current <= tolerance_a)
        high = on & (current >= high_threshold - tolerance_a)
        natural = on & ~low & ~high
        natural_h.append(float(natural.sum() * dt_s / 3600.0))
        low_h.append(float(low.sum() * dt_s / 3600.0))
        high_h.append(float(high.sum() * dt_s / 3600.0))
        starts.append(int(start.sum()))
        shifts.append(int(shift.sum()))

    return ActionExposure(
        tuple(natural_h),
        tuple(low_h),
        tuple(high_h),
        tuple(starts),
        tuple(shifts),
        factors,
        float(len(frame) * dt_s),
    )
