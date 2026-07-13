"""Leakage-safe development templates for slow service scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fc_power.evaluation.service_scheduler import ServiceExposure


@dataclass(frozen=True)
class CalibrationWindow:
    segment_id: int
    start_offset: int
    length_s: int

    def __post_init__(self) -> None:
        if self.segment_id < 0 or self.start_offset < 0 or self.length_s <= 0:
            raise ValueError("calibration window fields must be non-negative")


def select_calibration_windows(
    frame: pd.DataFrame,
    calibration_segments,
    *,
    length_s: int,
    count: int,
    seed: int,
) -> tuple[CalibrationWindow, ...]:
    """Sample uniformly from valid starts without crossing segment boundaries."""

    if "segment_id" not in frame:
        raise ValueError("frame is missing segment_id")
    if length_s <= 0 or count <= 0:
        raise ValueError("length_s and count must be positive")
    allowed = frozenset(int(value) for value in calibration_segments)
    if not allowed:
        raise ValueError("calibration_segments must not be empty")
    groups = []
    for segment_id, segment in frame[frame.segment_id.isin(allowed)].groupby(
        "segment_id", sort=True
    ):
        valid_starts = len(segment) - length_s + 1
        if valid_starts > 0:
            groups.append((int(segment_id), valid_starts))
    if not groups:
        raise ValueError("no calibration segment is long enough for a window")
    counts = np.asarray([item[1] for item in groups], dtype=np.int64)
    total = int(counts.sum())
    if count > total:
        raise ValueError("requested more unique windows than valid starts")
    rng = np.random.default_rng(seed)
    draws = rng.choice(total, size=count, replace=False)
    cumulative = np.cumsum(counts)
    windows = []
    for draw in draws:
        group_index = int(np.searchsorted(cumulative, draw, side="right"))
        previous = 0 if group_index == 0 else int(cumulative[group_index - 1])
        windows.append(
            CalibrationWindow(
                segment_id=groups[group_index][0],
                start_offset=int(draw) - previous,
                length_s=length_s,
            )
        )
    return tuple(windows)


def materialize_calibration_window(
    frame: pd.DataFrame,
    window: CalibrationWindow,
) -> pd.DataFrame:
    segment = frame[frame.segment_id == window.segment_id].reset_index(drop=True)
    stop = window.start_offset + window.length_s
    if stop > len(segment):
        raise ValueError("calibration window exceeds its segment")
    result = segment.iloc[window.start_offset:stop].copy().reset_index(drop=True)
    if len(result) != window.length_s or result.segment_id.nunique() != 1:
        raise AssertionError("materialized window crossed a segment boundary")
    return result


def service_exposure_from_trajectory(
    trajectory: pd.DataFrame,
    *,
    duration_h: float,
    assigned_stacks: tuple[int, int] = (0, 1),
) -> tuple[ServiceExposure, tuple[int, int]]:
    """Extract role exposure, excluding artificial starts at block entry."""

    if len(set(assigned_stacks)) != 2:
        raise ValueError("assigned_stacks must contain two distinct indices")
    if len(trajectory) == 0:
        raise ValueError("trajectory must not be empty")
    outside = [
        int(column.split("_")[1])
        for column in trajectory.columns
        if column.startswith("stack_") and column.endswith("_on")
        and int(column.split("_")[1]) not in assigned_stacks
    ]
    if any(bool(trajectory[f"stack_{index}_on"].any()) for index in set(outside)):
        raise AssertionError("fast allocator used a stack outside the slow assignment")
    rows = []
    for stack in assigned_stacks:
        continuous = float(
            trajectory[f"stack_{stack}_expected_continuous_increment_pct"].sum()
        )
        shift = float(
            trajectory[f"stack_{stack}_ramp_increment_pct"].sum()
            + trajectory[f"stack_{stack}_shift_increment_pct"].sum()
        )
        operational_start = float(
            trajectory[f"stack_{stack}_start_stop_increment_pct"].iloc[1:].sum()
        )
        rows.append((stack, continuous, shift, operational_start))
    rows.sort(key=lambda item: (-(item[1] + item[2] + item[3]), item[0]))
    exposure = ServiceExposure(
        duration_h=duration_h,
        continuous_mean_pct=(rows[0][1], rows[1][1]),
        load_shift_damage_pct=(rows[0][2], rows[1][2]),
        operational_start_damage_pct=(rows[0][3], rows[1][3]),
    )
    return exposure, (rows[0][0], rows[1][0])
