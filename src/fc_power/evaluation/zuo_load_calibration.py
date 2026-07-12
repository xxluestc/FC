"""Time-isolated calibration helpers for Zuo-style four-state loads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ZUO_LOAD_LEVELS_KW = (2.9, 4.1, 5.8, 7.0)
ZUO_LOAD_LEVEL_FRACTIONS = tuple(value / 7.0 for value in ZUO_LOAD_LEVELS_KW)
ZUO_FAST_TRANSITION = (
    (0.10, 0.35, 0.35, 0.20),
    (0.35, 0.10, 0.35, 0.20),
    (0.20, 0.35, 0.10, 0.35),
    (0.20, 0.35, 0.35, 0.10),
)
ZUO_SLOW_TRANSITION = (
    (0.70, 0.10, 0.15, 0.05),
    (0.05, 0.80, 0.05, 0.10),
    (0.05, 0.05, 0.85, 0.05),
    (0.05, 0.15, 0.20, 0.60),
)


@dataclass(frozen=True)
class TemporalSegmentSplit:
    calibration_segments: tuple[int, ...]
    holdout_segments: tuple[int, ...]
    gap_seconds: float
    calibration_end: str
    holdout_start: str


@dataclass(frozen=True)
class TransitionEstimate:
    stride_s: int
    counts: np.ndarray
    probabilities: np.ndarray
    ci95_lower: np.ndarray
    ci95_upper: np.ndarray
    occupancy: np.ndarray
    segment_counts: np.ndarray


def split_at_largest_segment_gap(
    frame: pd.DataFrame,
    *,
    segment_column: str = "segment_id",
    timestamp_column: str = "timestamp",
) -> TemporalSegmentSplit:
    """Choose a temporal holdout boundary without consulting power values."""

    required = {segment_column, timestamp_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing split columns: {sorted(required - set(frame.columns))}")
    summary = frame[[segment_column, timestamp_column]].copy()
    summary[timestamp_column] = pd.to_datetime(summary[timestamp_column], errors="raise")
    summary = (
        summary.groupby(segment_column, sort=False)[timestamp_column]
        .agg(["min", "max"])
        .sort_values("min")
    )
    if len(summary) < 2:
        raise ValueError("at least two temporal segments are required")
    gaps = summary["min"].iloc[1:].to_numpy() - summary["max"].iloc[:-1].to_numpy()
    split_after = int(np.argmax(gaps))
    gap_seconds = float(gaps[split_after] / np.timedelta64(1, "s"))
    calibration = tuple(int(value) for value in summary.index[: split_after + 1])
    holdout = tuple(int(value) for value in summary.index[split_after + 1 :])
    if not calibration or not holdout or gap_seconds <= 0:
        raise ValueError("largest segment gap does not define a valid temporal holdout")
    return TemporalSegmentSplit(
        calibration_segments=calibration,
        holdout_segments=holdout,
        gap_seconds=gap_seconds,
        calibration_end=summary["max"].iloc[split_after].isoformat(),
        holdout_start=summary["min"].iloc[split_after + 1].isoformat(),
    )


def quantize_zuo_states(power_kw, normalization_power_kw: float) -> np.ndarray:
    """Map positive single-stack power to Zuo's normalized four states.

    Non-positive samples are labelled -1 and remain explicit gaps; they are not
    bridged when transition counts are formed.
    """

    power = np.asarray(power_kw, dtype=float)
    if power.ndim != 1 or np.any(~np.isfinite(power)):
        raise ValueError("power_kw must be a finite vector")
    if not np.isfinite(normalization_power_kw) or normalization_power_kw <= 0:
        raise ValueError("normalization_power_kw must be finite and positive")
    states = np.full(power.shape, -1, dtype=int)
    positive = power > 0
    normalized = np.clip(power[positive] / normalization_power_kw, 0.0, 1.0)
    levels = np.asarray(ZUO_LOAD_LEVEL_FRACTIONS, dtype=float)
    states[positive] = np.abs(normalized[:, None] - levels[None, :]).argmin(axis=1)
    return states


def estimate_segmented_transitions(
    frame: pd.DataFrame,
    *,
    normalization_power_kw: float,
    stride_s: int,
    power_column: str = "fc_input_power_kw",
    segment_column: str = "segment_id",
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2026,
) -> TransitionEstimate:
    """Estimate transitions without crossing segment or off-period boundaries."""

    required = {power_column, segment_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"missing transition columns: {sorted(required - set(frame.columns))}"
        )
    if isinstance(stride_s, bool) or not isinstance(stride_s, int) or stride_s <= 0:
        raise ValueError("stride_s must be a positive integer")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    counts_by_segment = []
    occupancy = np.zeros(4, dtype=int)
    for _, segment in frame.groupby(segment_column, sort=False):
        power = segment[power_column].to_numpy(dtype=float)[::stride_s]
        states = quantize_zuo_states(power, normalization_power_kw)
        valid_states = states[states >= 0]
        occupancy += np.bincount(valid_states, minlength=4)
        counts = np.zeros((4, 4), dtype=int)
        if len(states) > 1:
            valid_pairs = (states[:-1] >= 0) & (states[1:] >= 0)
            np.add.at(counts, (states[:-1][valid_pairs], states[1:][valid_pairs]), 1)
        counts_by_segment.append(counts)

    segment_counts = np.asarray(counts_by_segment, dtype=int)
    counts = segment_counts.sum(axis=0)
    probabilities = _row_probabilities(counts)
    lower, upper = _segment_bootstrap_intervals(
        segment_counts, bootstrap_samples, bootstrap_seed
    )
    return TransitionEstimate(
        stride_s=stride_s,
        counts=counts,
        probabilities=probabilities,
        ci95_lower=lower,
        ci95_upper=upper,
        occupancy=occupancy,
        segment_counts=segment_counts,
    )


def blend_transition_matrices(empirical, literature, empirical_weight: float):
    """Convexly blend two audited stochastic matrices with an explicit weight."""

    if not np.isfinite(empirical_weight) or not 0 <= empirical_weight <= 1:
        raise ValueError("empirical_weight must lie in [0, 1]")
    empirical = _validated_transition_matrix(empirical, "empirical")
    literature = _validated_transition_matrix(literature, "literature")
    blended = empirical_weight * empirical + (1 - empirical_weight) * literature
    return blended / blended.sum(axis=1, keepdims=True)


def generate_zuo_markov_system_load(
    seed: int,
    *,
    length_s: int,
    decision_interval_s: int,
    system_power_reference_kw: float,
    transition_matrix,
    initial_probabilities=None,
    state_fractions=ZUO_LOAD_LEVEL_FRACTIONS,
    source: str = "zuo_calibrated_markov",
) -> pd.DataFrame:
    """Generate a piecewise-constant N+1 system demand from four load states."""

    if length_s <= 0 or decision_interval_s <= 0:
        raise ValueError("length_s and decision_interval_s must be positive")
    if not np.isfinite(system_power_reference_kw) or system_power_reference_kw <= 0:
        raise ValueError("system_power_reference_kw must be finite and positive")
    matrix = _validated_transition_matrix(transition_matrix, "transition")
    fractions = np.asarray(state_fractions, dtype=float)
    if (
        fractions.shape != (4,)
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0)
        or np.any(np.diff(fractions) <= 0)
        or fractions[-1] > 1
    ):
        raise ValueError("state_fractions must be four increasing values in (0, 1]")
    if initial_probabilities is None:
        initial = np.full(4, 0.25)
    else:
        initial = np.asarray(initial_probabilities, dtype=float)
        if (
            initial.shape != (4,)
            or np.any(~np.isfinite(initial))
            or np.any(initial < 0)
            or initial.sum() <= 0
        ):
            raise ValueError("initial_probabilities must be four non-negative values")
        initial = initial / initial.sum()

    rng = np.random.default_rng(seed)
    state = int(rng.choice(4, p=initial))
    previous_state = None
    event_id = -1
    rows = []
    for block_start in range(0, length_s, decision_interval_s):
        changed = previous_state is None or state != previous_state
        if changed:
            event_id += 1
        block_stop = min(block_start + decision_interval_s, length_s)
        demand = float(fractions[state] * system_power_reference_kw)
        for step in range(block_start, block_stop):
            rows.append(
                {
                    "step": step,
                    "time_s": float(step),
                    "demand_power_kw": demand,
                    "load_state": state,
                    "event": f"load_state_{state}",
                    "event_id": event_id,
                    "event_boundary": changed and step == block_start,
                    "markov_decision": step == block_start,
                    "source": source,
                    "seed": seed,
                }
            )
        previous_state = state
        state = int(rng.choice(4, p=matrix[state]))
    return pd.DataFrame(rows)


def _row_probabilities(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        totals,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=totals > 0,
    )


def _validated_transition_matrix(values, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} transition matrix must be finite non-negative 4x4")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError(f"{name} transition matrix rows must sum to one")
    return matrix


def _segment_bootstrap_intervals(segment_counts, samples: int, seed: int):
    segment_counts = np.asarray(segment_counts, dtype=int)
    if segment_counts.ndim != 3 or segment_counts.shape[1:] != (4, 4):
        raise ValueError("segment_counts must have shape (n_segments, 4, 4)")
    if len(segment_counts) == 0:
        raise ValueError("at least one segment count matrix is required")
    rng = np.random.default_rng(seed)
    draws = np.full((samples, 4, 4), np.nan, dtype=float)
    for sample in range(samples):
        indices = rng.integers(0, len(segment_counts), size=len(segment_counts))
        draws[sample] = _row_probabilities(segment_counts[indices].sum(axis=0))
    lower = np.full((4, 4), np.nan, dtype=float)
    upper = np.full((4, 4), np.nan, dtype=float)
    for row in range(4):
        for column in range(4):
            finite = draws[:, row, column][np.isfinite(draws[:, row, column])]
            if len(finite):
                lower[row, column], upper[row, column] = np.quantile(
                    finite, [0.025, 0.975]
                )
    return lower, upper
