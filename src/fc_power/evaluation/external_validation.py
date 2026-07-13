"""Deterministic extraction of external real-power validation blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd


POWER_PACKET_COLUMNS = ("timestamp", "fc_voltage_v", "fc_current_a")


def canonicalize_power_packets(
    raw: pd.DataFrame,
    *,
    target_dt_s: int = 1,
    gap_s: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicate packets and resample without crossing telemetry gaps."""

    missing = set(POWER_PACKET_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"missing power packet columns: {sorted(missing)}")
    if target_dt_s <= 0 or gap_s <= target_dt_s:
        raise ValueError("target_dt_s must be positive and smaller than gap_s")

    frame = raw.loc[:, POWER_PACKET_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("fc_voltage_v", "fc_current_a"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("no valid timestamped power packets")

    duplicate_timestamps = int(frame.timestamp.duplicated().sum())
    frame = (
        frame.groupby("timestamp", as_index=False)[
            ["fc_voltage_v", "fc_current_a"]
        ]
        .mean()
        .sort_values("timestamp")
    )
    dt_s = frame.timestamp.diff().dt.total_seconds()
    frame["source_segment_id"] = (dt_s.isna() | (dt_s > gap_s)).cumsum() - 1

    pieces = []
    interpolation_limit = max(1, int(gap_s / target_dt_s) - 1)
    for segment_id, group in frame.groupby("source_segment_id", sort=True):
        sampled = (
            group.set_index("timestamp")[["fc_voltage_v", "fc_current_a"]]
            .resample(f"{target_dt_s}s")
            .mean()
        )
        missing_before_interpolation = sampled.isna().any(axis=1)
        sampled = sampled.interpolate(
            "time",
            limit=interpolation_limit,
            limit_area="inside",
        )
        sampled["interpolated_power"] = (
            missing_before_interpolation & sampled.notna().all(axis=1)
        )
        sampled["source_segment_id"] = int(segment_id)
        pieces.append(sampled.reset_index())

    canonical = pd.concat(pieces, ignore_index=True).sort_values("timestamp")
    canonical["fc_input_power_kw"] = (
        canonical.fc_voltage_v * canonical.fc_current_a / 1000.0
    )
    audit = {
        "raw_rows": int(len(raw)),
        "valid_timestamp_rows": int(len(frame) + duplicate_timestamps),
        "duplicate_timestamps": duplicate_timestamps,
        "unique_packet_rows": int(len(frame)),
        "canonical_rows": int(len(canonical)),
        "source_segments": int(canonical.source_segment_id.nunique()),
        "interpolated_power_rows": int(canonical.interpolated_power.sum()),
        "target_dt_s": int(target_dt_s),
        "gap_s": int(gap_s),
    }
    return canonical.reset_index(drop=True), audit


def select_first_operating_block(
    canonical: pd.DataFrame,
    month: str | pd.Period,
    *,
    block_steps: int = 1800,
    minimum_positive_share: float = 0.5,
    positive_threshold_kw: float = 0.5,
) -> pd.DataFrame:
    """Return the first qualifying block in a month without crossing a gap."""

    required = {"timestamp", "source_segment_id", "fc_input_power_kw"}
    missing = required.difference(canonical.columns)
    if missing:
        raise ValueError(f"missing canonical columns: {sorted(missing)}")
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    if not 0 <= minimum_positive_share <= 1:
        raise ValueError("minimum_positive_share must lie in [0, 1]")
    if not np.isfinite(positive_threshold_kw) or positive_threshold_kw < 0:
        raise ValueError("positive_threshold_kw must be finite and non-negative")

    period = pd.Period(month, freq="M")
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
    frame = frame[
        (frame.timestamp >= period.start_time)
        & (frame.timestamp <= period.end_time)
    ].sort_values("timestamp")
    required_positive = int(np.ceil(block_steps * minimum_positive_share))

    for _, segment in frame.groupby("source_segment_id", sort=True):
        segment = segment.reset_index(drop=True)
        if len(segment) < block_steps:
            continue
        values = segment.fc_input_power_kw.to_numpy(dtype=float)
        valid = np.isfinite(values)
        positive = valid & (values >= positive_threshold_kw)
        kernel = np.ones(block_steps, dtype=int)
        valid_count = np.convolve(valid.astype(int), kernel, mode="valid")
        positive_count = np.convolve(positive.astype(int), kernel, mode="valid")
        candidates = np.flatnonzero(
            (valid_count == block_steps) & (positive_count >= required_positive)
        )
        if not len(candidates):
            continue
        start = int(candidates[0])
        block = segment.iloc[start : start + block_steps].copy()
        cadence = block.timestamp.diff().dt.total_seconds().dropna()
        if not np.allclose(cadence, 1.0):
            raise AssertionError("selected external block is not one-second cadence")
        block.insert(0, "block_step", np.arange(block_steps, dtype=int))
        return block.reset_index(drop=True)

    raise ValueError(f"no qualifying {block_steps}-step block in {period}")
