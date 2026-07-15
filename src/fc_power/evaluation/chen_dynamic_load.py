"""Random dynamic loads whose amplitudes are derived from Chen stack curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChenLoadLevels:
    single_peak_kw: float
    dual_peak_kw: float
    high_load_kw: float
    reserve_peak_kw: float
    n_plus_one_max_kw: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.single_peak_kw,
                self.dual_peak_kw,
                self.high_load_kw,
                self.reserve_peak_kw,
            ],
            dtype=float,
        )


def derive_chen_load_levels(audited_curves: pd.DataFrame) -> ChenLoadLevels:
    """Derive four system-load centers from efficiency peaks and N+1 capacity."""

    required = {
        "stack_id",
        "net_system_power_kw",
        "efficiency_lhv_pct",
    }
    missing = required.difference(audited_curves.columns)
    if missing:
        raise ValueError(f"missing Chen load-level columns: {sorted(missing)}")
    peaks = []
    maxima = []
    for _, group in audited_curves.groupby("stack_id", sort=True):
        if len(group) < 2:
            raise ValueError("each Chen stack needs at least two curve points")
        peak = group.loc[group["efficiency_lhv_pct"].idxmax()]
        peaks.append(float(peak["net_system_power_kw"]))
        maxima.append(float(group["net_system_power_kw"].max()))
    if len(peaks) < 3:
        raise ValueError("three Chen curves are required for N+1 load derivation")

    single_peak = float(np.median(peaks))
    dual_peak = float(sum(sorted(peaks, reverse=True)[:2]))
    n_plus_one_max = float(sum(sorted(maxima, reverse=True)[:2]))
    high_load = 0.75 * n_plus_one_max
    reserve_peak = 0.90 * n_plus_one_max
    levels = np.asarray(
        [single_peak, dual_peak, high_load, reserve_peak],
        dtype=float,
    )
    if np.any(np.diff(levels) <= 0):
        raise ValueError("derived Chen load centers must be strictly increasing")
    return ChenLoadLevels(
        single_peak_kw=single_peak,
        dual_peak_kw=dual_peak,
        high_load_kw=high_load,
        reserve_peak_kw=reserve_peak,
        n_plus_one_max_kw=n_plus_one_max,
    )


def generate_chen_random_dynamic_load(
    seed: int,
    *,
    length_s: int,
    levels: ChenLoadLevels,
    transition_matrix,
    dwell_range_s: tuple[int, int] = (8, 25),
    target_variation_fraction: float = 0.03,
    ensure_state_coverage: bool = True,
) -> pd.DataFrame:
    """Generate a semi-Markov stress load with continuous event targets.

    The transition matrix controls event order only.  Demand amplitudes come
    from Chen's efficiency peaks and N+1 net capacity.  Dwell time and the
    small within-state target variation are explicit engineering stress
    settings because a physical vehicle ramp/dwell law is not available.
    """

    if length_s <= 0:
        raise ValueError("length_s must be positive")
    low_dwell, high_dwell = dwell_range_s
    if low_dwell <= 0 or high_dwell < low_dwell:
        raise ValueError("dwell_range_s must be positive and ordered")
    if not 0 <= target_variation_fraction < 0.2:
        raise ValueError("target variation must lie in [0, 0.2)")
    matrix = np.asarray(transition_matrix, dtype=float)
    if (
        matrix.shape != (4, 4)
        or np.any(~np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.allclose(matrix.sum(axis=1), 1.0)
    ):
        raise ValueError("transition_matrix must be a stochastic 4x4 matrix")
    centers = levels.as_array()
    if np.any(centers <= 0) or np.any(np.diff(centers) <= 0):
        raise ValueError("Chen load centers must be positive and increasing")

    rng = np.random.default_rng(seed)
    warmup = list(range(4)) if ensure_state_coverage else []
    state = 0
    rows = []
    event_id = 0
    while len(rows) < length_s:
        if warmup:
            state = warmup.pop(0)
        dwell = int(rng.integers(low_dwell, high_dwell + 1))
        target = float(
            centers[state]
            * rng.uniform(
                1.0 - target_variation_fraction,
                1.0 + target_variation_fraction,
            )
        )
        target = min(target, 0.95 * levels.n_plus_one_max_kw)
        take = min(dwell, length_s - len(rows))
        for within_event in range(take):
            rows.append(
                {
                    "step": len(rows),
                    "time_s": float(len(rows)),
                    "demand_net_power_kw": target,
                    "load_state": state,
                    "load_center_kw": centers[state],
                    "event_id": event_id,
                    "event_boundary": within_event == 0,
                    "source": "chen_curve_calibrated_random_dynamic",
                    "seed": seed,
                }
            )
        event_id += 1
        state = int(rng.choice(4, p=matrix[state]))
    return pd.DataFrame(rows)
