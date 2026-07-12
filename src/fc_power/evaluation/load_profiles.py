"""Reproducible synthetic and real-block multi-stack demand profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_NAMES = ("idle", "cruise", "high", "braking")


@dataclass(frozen=True)
class SyntheticLoadConfig:
    """Transparent semi-Markov stress-load configuration.

    Defaults are engineering stress ranges, not parameters identified from a
    specific vehicle.  Paper-level experiments must report sensitivity or use
    the real-block bootstrap alongside this generator.
    """

    length_s: int = 300
    dt_s: float = 1.0
    power_min_kw: float = -75.0
    power_max_kw: float = 180.0
    smoothing: float = 0.35
    noise_std_kw: float = 2.0
    max_ramp_kw_per_s: float = 35.0
    duration_ranges_s: tuple[tuple[int, int], ...] = (
        (8, 35),
        (15, 55),
        (8, 30),
        (4, 18),
    )
    target_ranges_kw: tuple[tuple[float, float], ...] = (
        (-2.0, 5.0),
        (25.0, 80.0),
        (95.0, 175.0),
        (-70.0, -10.0),
    )
    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.10, 0.75, 0.15, 0.00),
        (0.20, 0.15, 0.40, 0.25),
        (0.05, 0.45, 0.15, 0.35),
        (0.45, 0.45, 0.10, 0.00),
    )
    ensure_event_coverage: bool = True

    def __post_init__(self) -> None:
        if self.length_s <= 0 or not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("length_s and dt_s must be positive")
        if self.power_min_kw >= self.power_max_kw:
            raise ValueError("power bounds are invalid")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must lie in (0, 1]")
        if self.noise_std_kw < 0 or self.max_ramp_kw_per_s <= 0:
            raise ValueError("noise/ramp parameters are invalid")
        if len(self.duration_ranges_s) != len(EVENT_NAMES):
            raise ValueError("one duration range is required per event")
        if len(self.target_ranges_kw) != len(EVENT_NAMES):
            raise ValueError("one target range is required per event")
        matrix = np.asarray(self.transition_matrix, dtype=float)
        if matrix.shape != (len(EVENT_NAMES), len(EVENT_NAMES)):
            raise ValueError("transition matrix has an invalid shape")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("transition rows must be non-negative and sum to one")
        if any(low <= 0 or high < low for low, high in self.duration_ranges_s):
            raise ValueError("duration ranges must be positive and ordered")


def generate_event_load(
    seed: int,
    config: SyntheticLoadConfig = SyntheticLoadConfig(),
) -> pd.DataFrame:
    """Generate a bounded event-labelled semi-Markov load sequence."""

    rng = np.random.default_rng(seed)
    samples = int(np.ceil(config.length_s / config.dt_s))
    warmup = list(range(len(EVENT_NAMES))) if config.ensure_event_coverage else []
    state = 0
    previous = 0.0
    rows = []
    event_id = 0
    while len(rows) < samples:
        if warmup:
            state = warmup.pop(0)
        low_duration, high_duration = config.duration_ranges_s[state]
        duration = int(rng.integers(low_duration, high_duration + 1))
        low_target, high_target = config.target_ranges_kw[state]
        target = float(rng.uniform(low_target, high_target))
        response = max(config.smoothing, 0.80) if state == 3 else config.smoothing
        for within_event in range(duration):
            if len(rows) >= samples:
                break
            desired = target + 0.05 * (high_target - low_target) * np.sin(
                2 * np.pi * within_event / max(duration, 1)
            )
            delta = response * (desired - previous) + rng.normal(
                0.0, config.noise_std_kw
            )
            ramp_limit = config.max_ramp_kw_per_s * config.dt_s
            demand = float(
                np.clip(
                    previous + np.clip(delta, -ramp_limit, ramp_limit),
                    config.power_min_kw,
                    config.power_max_kw,
                )
            )
            rows.append(
                {
                    "step": len(rows),
                    "time_s": len(rows) * config.dt_s,
                    "demand_power_kw": demand,
                    "event": EVENT_NAMES[state],
                    "event_id": event_id,
                    "event_boundary": within_event == 0,
                    "source": "synthetic_event_markov",
                    "seed": seed,
                }
            )
            previous = demand
        event_id += 1
        state = int(rng.choice(len(EVENT_NAMES), p=config.transition_matrix[state]))
    return pd.DataFrame(rows)


def classify_power_events(power_kw, high_threshold_kw: float | None = None):
    power = np.asarray(power_kw, dtype=float)
    if power.ndim != 1 or np.any(~np.isfinite(power)):
        raise ValueError("power_kw must be a finite vector")
    if high_threshold_kw is None:
        positive = power[power > 5]
        high_threshold_kw = float(np.quantile(positive, 0.90)) if len(positive) else 80.0
    labels = np.full(len(power), "cruise", dtype=object)
    labels[np.abs(power) <= 5] = "idle"
    labels[power < -5] = "braking"
    labels[power >= high_threshold_kw] = "high"
    return labels


def generate_real_block_bootstrap(
    real_power_kw,
    length_s: int,
    seed: int,
    *,
    block_length_s: int = 30,
    boundary_candidates: int = 32,
) -> pd.DataFrame:
    """Resample real contiguous blocks while reducing artificial joins."""

    source = np.asarray(real_power_kw, dtype=float)
    if source.ndim != 1:
        raise ValueError("real_power_kw must be a vector")
    if length_s <= 0 or block_length_s <= 1 or len(source) < block_length_s:
        raise ValueError("length/block/source sizes are invalid")
    if boundary_candidates <= 0:
        raise ValueError("boundary_candidates must be positive")
    finite_count = np.convolve(
        np.isfinite(source).astype(int),
        np.ones(block_length_s, dtype=int),
        mode="valid",
    )
    valid_starts = np.flatnonzero(finite_count == block_length_s)
    if len(valid_starts) == 0:
        raise ValueError("source contains no fully finite contiguous block")
    rng = np.random.default_rng(seed)
    demand, source_indices, block_ids = [], [], []
    block_id = 0
    while len(demand) < length_s:
        starts = rng.choice(valid_starts, size=boundary_candidates, replace=True)
        if demand:
            mismatch = np.abs(source[starts] - demand[-1])
            best = np.argsort(mismatch)[: max(1, boundary_candidates // 4)]
            start = int(starts[rng.choice(best)])
        else:
            start = int(starts[0])
        stop = min(start + block_length_s, len(source))
        block = source[start:stop]
        take = min(len(block), length_s - len(demand))
        demand.extend(block[:take].tolist())
        source_indices.extend(range(start, start + take))
        block_ids.extend([block_id] * take)
        block_id += 1
    labels = classify_power_events(demand)
    boundaries = np.r_[True, np.diff(block_ids) != 0]
    return pd.DataFrame(
        {
            "step": np.arange(length_s),
            "time_s": np.arange(length_s, dtype=float),
            "demand_power_kw": demand,
            "event": labels,
            "event_id": block_ids,
            "event_boundary": boundaries,
            "source_index": source_indices,
            "source": "real_block_bootstrap",
            "seed": seed,
        }
    )


def append_soc_recovery_tail(
    profile: pd.DataFrame,
    duration_s: int = 120,
    demand_power_kw: float = 30.0,
) -> pd.DataFrame:
    """Append a shared controllable load window for terminal-SOC equalization."""

    if duration_s < 0 or not np.isfinite(demand_power_kw):
        raise ValueError("recovery duration/demand is invalid")
    result = profile.copy()
    result["is_soc_recovery"] = False
    if duration_s == 0:
        return result
    start_step = len(result)
    event_id = int(result.event_id.max()) + 1 if "event_id" in result else 0
    tail = pd.DataFrame(
        {
            "step": np.arange(start_step, start_step + duration_s),
            "time_s": np.arange(start_step, start_step + duration_s, dtype=float),
            "demand_power_kw": demand_power_kw,
            "event": "soc_recovery",
            "event_id": event_id,
            "event_boundary": np.r_[True, np.zeros(duration_s - 1, dtype=bool)],
            "source": str(result.source.iloc[0]) if "source" in result else "unknown",
            "seed": int(result.seed.iloc[0]) if "seed" in result else -1,
            "is_soc_recovery": True,
        }
    )
    return pd.concat([result, tail], ignore_index=True, sort=False)
