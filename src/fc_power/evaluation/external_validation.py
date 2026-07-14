"""Deterministic extraction of external real-power validation blocks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


POWER_PACKET_COLUMNS = ("timestamp", "fc_voltage_v", "fc_current_a")


@dataclass(frozen=True)
class DataExclusionInterval:
    """One half-open interval removed from all derived sequences."""

    start_inclusive: pd.Timestamp
    end_exclusive: pd.Timestamp
    reason: str = ""
    status: str = "exclude"
    preserve_segment_break: bool = True


@dataclass(frozen=True)
class DataExclusionRules:
    """Validated, source-auditable data exclusion rules."""

    intervals: tuple[DataExclusionInterval, ...]
    vehicle_id: str | None = None
    time_basis: str | None = None
    raw_files_are_immutable: bool = True
    transition_counts_must_not_cross_exclusions: bool = True
    source_path: str | None = None


def load_data_exclusions(path: str | Path) -> DataExclusionRules:
    """Read and validate a UTF-8 JSON exclusion configuration."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("data exclusion configuration must be a JSON object")
    return _parse_data_exclusion_rules(payload, source_path=str(source))


def apply_data_exclusions(
    frame: pd.DataFrame,
    rules: DataExclusionRules | Mapping[str, Any] | str | Path,
    *,
    timestamp_column: str = "timestamp",
    segment_column: str | None = None,
    output_segment_column: str = "model_segment_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove configured intervals and make every interval a hard segment break.

    The original segment/block identifier is retained.  The returned model
    segment combines that identifier with the side of every exclusion interval,
    so transition or dwell calculations cannot bridge removed time even when no
    excluded row happened to be present in the supplied frame.
    """

    parsed = _coerce_data_exclusion_rules(rules)
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    if output_segment_column in frame.columns:
        raise ValueError(f"output segment column already exists: {output_segment_column}")
    resolved_segment = _resolve_segment_column(frame, segment_column)
    if frame[resolved_segment].isna().any():
        raise ValueError("data exclusions require non-missing segment identifiers")

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("data exclusions require valid timestamps in every row")
    result[timestamp_column] = timestamps

    excluded = np.zeros(len(result), dtype=bool)
    interval_audit: list[dict[str, Any]] = []
    aligned_intervals = []
    for interval in parsed.intervals:
        start = _align_rule_timestamp(interval.start_inclusive, timestamps)
        end = _align_rule_timestamp(interval.end_exclusive, timestamps)
        mask = ((timestamps >= start) & (timestamps < end)).to_numpy()
        excluded |= mask
        aligned_intervals.append((start, end))
        interval_audit.append(
            {
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "reason": interval.reason,
                "status": interval.status,
                "excluded_rows": int(mask.sum()),
                "hard_segment_break": True,
            }
        )

    kept = result.loc[~excluded].copy()
    kept_timestamps = timestamps.loc[~excluded]
    key_frame = pd.DataFrame(
        {"source_segment": kept[resolved_segment].astype("object").to_numpy()},
        index=kept.index,
    )
    for index, (_, end) in enumerate(aligned_intervals):
        key_frame[f"exclusion_side_{index}"] = (kept_timestamps >= end).to_numpy()
    keys = pd.MultiIndex.from_frame(key_frame)
    kept[output_segment_column] = pd.factorize(keys, sort=False)[0].astype(int)

    source_segments = int(result[resolved_segment].nunique(dropna=False))
    model_segments = int(kept[output_segment_column].nunique()) if len(kept) else 0
    audit: dict[str, Any] = {
        "source_path": parsed.source_path,
        "vehicle_id": parsed.vehicle_id,
        "timestamp_column": timestamp_column,
        "source_segment_column": resolved_segment,
        "output_segment_column": output_segment_column,
        "input_rows": int(len(result)),
        "excluded_rows": int(excluded.sum()),
        "output_rows": int(len(kept)),
        "source_segments": source_segments,
        "model_segments": model_segments,
        "hard_segment_breaks_added": max(0, model_segments - source_segments),
        "transition_counts_must_not_cross_exclusions": True,
        "intervals": interval_audit,
    }
    return kept.reset_index(drop=True), audit


def canonicalize_power_packets(
    raw: pd.DataFrame,
    *,
    target_dt_s: int = 1,
    gap_s: int = 10,
    step_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicate packets and resample without crossing telemetry gaps.

    Voltage and current are linearly interpolated inside short packet gaps.
    Declared step columns, such as a target-power command, are forward-filled
    instead so resampling cannot invent intermediate command levels.
    """

    numeric_columns = ("fc_voltage_v", "fc_current_a", *step_columns)
    required_columns = ("timestamp", *numeric_columns)
    missing = set(required_columns).difference(raw.columns)
    if missing:
        raise ValueError(f"missing power packet columns: {sorted(missing)}")
    if len(set(step_columns)) != len(step_columns):
        raise ValueError("step_columns must not contain duplicates")
    if {"fc_voltage_v", "fc_current_a"}.intersection(step_columns):
        raise ValueError("voltage and current cannot also be step columns")
    if target_dt_s <= 0 or gap_s <= target_dt_s:
        raise ValueError("target_dt_s must be positive and smaller than gap_s")

    frame = raw.loc[:, required_columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("no valid timestamped power packets")

    duplicate_timestamps = int(frame.timestamp.duplicated().sum())
    frame = (
        frame.groupby("timestamp", as_index=False)[list(numeric_columns)]
        .mean()
        .sort_values("timestamp")
    )
    dt_s = frame.timestamp.diff().dt.total_seconds()
    frame["source_segment_id"] = (dt_s.isna() | (dt_s > gap_s)).cumsum() - 1

    pieces = []
    interpolation_limit = max(1, int(gap_s / target_dt_s) - 1)
    for segment_id, group in frame.groupby("source_segment_id", sort=True):
        indexed = group.set_index("timestamp")
        sampled_power = indexed[["fc_voltage_v", "fc_current_a"]].resample(
            f"{target_dt_s}s"
        ).mean()
        missing_before_interpolation = sampled_power.isna().any(axis=1)
        sampled_power = sampled_power.interpolate(
            "time",
            limit=interpolation_limit,
            limit_area="inside",
        )
        sampled = sampled_power
        sampled["interpolated_power"] = (
            missing_before_interpolation & sampled_power.notna().all(axis=1)
        )
        for column in step_columns:
            sampled_step = indexed[column].resample(f"{target_dt_s}s").last()
            missing_step = sampled_step.isna()
            sampled[column] = sampled_step.ffill(limit=interpolation_limit)
            sampled[f"{column}_forward_filled"] = missing_step & sampled[column].notna()
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
        "step_columns": list(step_columns),
        "forward_filled_step_rows": {
            column: int(canonical[f"{column}_forward_filled"].sum())
            for column in step_columns
        },
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
    segment_column: str = "source_segment_id",
) -> pd.DataFrame:
    """Return the first qualifying block in a month without crossing a gap."""

    required = {"timestamp", segment_column, "fc_input_power_kw"}
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

    for _, segment in frame.groupby(segment_column, sort=True):
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


def extract_target_events(
    canonical: pd.DataFrame,
    month: str | pd.Period,
    *,
    operating_threshold_kw: float = 0.5,
    target_threshold_kw: float = 0.0,
    target_rounding_kw: float = 0.1,
    segment_column: str = "model_segment_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compress a calendar month into positive-command operating events.

    Interpolated rows may be retained only inside a run whose nearest observed
    anchors are both operating.  They can therefore never create an active-run
    or target-event boundary.  Event censoring is reserved for a calendar or
    telemetry-segment edge; a contiguous observed off boundary is complete.
    """

    required = {
        "timestamp",
        segment_column,
        "target_power_kw",
        "fc_input_power_kw",
        "interpolated_power",
        "target_power_kw_forward_filled",
    }
    missing = required.difference(canonical.columns)
    if missing:
        raise ValueError(f"missing target-event columns: {sorted(missing)}")
    if not np.isfinite(operating_threshold_kw) or operating_threshold_kw < 0:
        raise ValueError("operating_threshold_kw must be finite and non-negative")
    if not np.isfinite(target_threshold_kw) or target_threshold_kw < 0:
        raise ValueError("target_threshold_kw must be finite and non-negative")
    if not np.isfinite(target_rounding_kw) or target_rounding_kw <= 0:
        raise ValueError("target_rounding_kw must be finite and positive")

    period = pd.Period(month, freq="M")
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
    if frame.timestamp.isna().any():
        raise ValueError("target-event extraction requires valid timestamps")
    frame = frame[
        (frame.timestamp >= period.start_time)
        & (frame.timestamp <= period.end_time)
    ].sort_values([segment_column, "timestamp"], kind="mergesort")
    if frame.empty:
        raise ValueError(f"no rows in target-event month {period}")
    if frame.duplicated([segment_column, "timestamp"]).any():
        raise ValueError("duplicate timestamps within target-event segment")

    target = pd.to_numeric(frame.target_power_kw, errors="coerce")
    power = pd.to_numeric(frame.fc_input_power_kw, errors="coerce")
    interpolated = frame.interpolated_power.astype(bool)
    observed = ~interpolated
    finite = np.isfinite(target) & np.isfinite(power)
    positive_command = finite & (target > target_threshold_kw)
    raw_anchor_operating = (
        observed
        & positive_command
        & (power > operating_threshold_kw)
    )

    anchor_state = pd.Series(
        np.where(observed, raw_anchor_operating.astype(float), np.nan),
        index=frame.index,
    )
    previous_anchor = anchor_state.groupby(frame[segment_column], sort=False).ffill()
    next_anchor = anchor_state.groupby(frame[segment_column], sort=False).bfill()
    interpolation_inside_operating = (
        interpolated
        & positive_command
        & previous_anchor.eq(1.0)
        & next_anchor.eq(1.0)
    )
    operating = raw_anchor_operating | interpolation_inside_operating

    segment_changed = frame[segment_column].ne(frame[segment_column].shift())
    dt_s = frame.timestamp.diff().dt.total_seconds()
    previous_same = ~segment_changed & dt_s.eq(1.0)
    next_same = previous_same.shift(-1, fill_value=False)
    run_start = operating & (
        ~operating.shift(fill_value=False) | ~previous_same
    )
    frame["_active_run"] = run_start.cumsum()
    frame["_operating"] = operating

    active = frame.loc[operating].copy()
    if active.empty:
        raise ValueError(f"no positive-command active rows in {period}")
    active["_start_interpolated"] = interpolated.loc[active.index].to_numpy()
    active["_run_left_censored"] = False
    active["_run_right_censored"] = False
    for run_id, run in active.groupby("_active_run", sort=False):
        first_index = run.index[0]
        last_index = run.index[-1]
        active.loc[run.index, "_run_left_censored"] = not bool(
            previous_same.loc[first_index]
        )
        active.loc[run.index, "_run_right_censored"] = not bool(
            next_same.loc[last_index]
        )

    decimals = max(0, int(np.ceil(-np.log10(target_rounding_kw))))
    active["_target_rounded"] = (
        (active.target_power_kw / target_rounding_kw).round()
        * target_rounding_kw
    ).round(decimals)
    active["archive_event_segment_id"] = (
        str(period)
        + ":"
        + active[segment_column].astype(str)
        + ":"
        + active._active_run.astype(str)
    )
    event_start = active.archive_event_segment_id.ne(
        active.archive_event_segment_id.shift()
    ) | active._target_rounded.ne(active._target_rounded.shift())
    if (event_start & active._start_interpolated).any():
        raise AssertionError("an interpolated row attempted to start a target event")
    active["_raw_event"] = event_start.cumsum()
    events = (
        active.groupby("_raw_event", sort=False)
        .agg(
            archive_event_segment_id=("archive_event_segment_id", "first"),
            start_timestamp=("timestamp", "first"),
            end_timestamp=("timestamp", "last"),
            dwell_time_s=("timestamp", "size"),
            target_power_kw=("_target_rounded", "first"),
            fc_input_power_kw=("fc_input_power_kw", "mean"),
            fc_input_power_p10_kw=(
                "fc_input_power_kw",
                lambda value: value.quantile(0.10),
            ),
            fc_input_power_p90_kw=(
                "fc_input_power_kw",
                lambda value: value.quantile(0.90),
            ),
            interpolated_power_share=("interpolated_power", "mean"),
            forward_filled_target_share=(
                "target_power_kw_forward_filled",
                "mean",
            ),
            _run_left_censored=("_run_left_censored", "first"),
            _run_right_censored=("_run_right_censored", "first"),
            start_interpolated=("_start_interpolated", "first"),
        )
        .reset_index(drop=True)
    )
    events.insert(0, "month", str(period))
    events["event_order"] = events.groupby(
        "archive_event_segment_id", sort=False
    ).cumcount()
    segment_sizes = events.groupby(
        "archive_event_segment_id", sort=False
    ).event_order.transform("size")
    events["left_censored"] = (
        events.event_order.eq(0) & events.pop("_run_left_censored").astype(bool)
    )
    events["right_censored"] = (
        events.event_order.eq(segment_sizes - 1)
        & events.pop("_run_right_censored").astype(bool)
    )
    events["complete_dwell"] = ~(
        events.left_censored | events.right_censored
    )
    if events.start_interpolated.any():
        raise AssertionError("interpolated target-event start survived extraction")
    events.insert(
        0,
        "archive_event_id",
        [f"archive_{period}_e{index:05d}" for index in range(len(events))],
    )

    zero_target_tail = finite & (power > operating_threshold_kw) & (
        target <= target_threshold_kw
    )
    audit = {
        "month": str(period),
        "calendar_rows": int(len(frame)),
        "active_rows": int(len(active)),
        "observed_active_rows": int((operating & observed).sum()),
        "interpolated_internal_active_rows": int(
            interpolation_inside_operating.sum()
        ),
        "interpolated_event_starts": int(events.start_interpolated.sum()),
        "zero_target_positive_power_rows_excluded": int(zero_target_tail.sum()),
        "event_segments": int(events.archive_event_segment_id.nunique()),
        "raw_target_events": int(len(events)),
        "complete_dwell_events": int(events.complete_dwell.sum()),
        "left_censored_events": int(events.left_censored.sum()),
        "right_censored_events": int(events.right_censored.sum()),
        "transitions": int(
            len(events) - events.archive_event_segment_id.nunique()
        ),
        "target_rounding_kw": float(target_rounding_kw),
    }
    return events, audit


def _parse_data_exclusion_rules(
    payload: Mapping[str, Any], *, source_path: str | None = None
) -> DataExclusionRules:
    raw_intervals = payload.get("intervals", [])
    if not isinstance(raw_intervals, list):
        raise ValueError("data exclusion intervals must be a list")
    intervals = []
    for index, raw in enumerate(raw_intervals):
        if not isinstance(raw, Mapping):
            raise ValueError(f"data exclusion interval {index} must be an object")
        if "start_inclusive" not in raw or "end_exclusive" not in raw:
            raise ValueError(
                f"data exclusion interval {index} requires start_inclusive and end_exclusive"
            )
        start = pd.Timestamp(raw["start_inclusive"])
        end = pd.Timestamp(raw["end_exclusive"])
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError(f"data exclusion interval {index} is not a valid range")
        intervals.append(
            DataExclusionInterval(
                start_inclusive=start,
                end_exclusive=end,
                reason=str(raw.get("reason", "")),
                status=str(raw.get("status", "exclude")),
                preserve_segment_break=bool(raw.get("preserve_segment_break", True)),
            )
        )
    intervals.sort(key=lambda value: value.start_inclusive)
    return DataExclusionRules(
        intervals=tuple(intervals),
        vehicle_id=(
            None if payload.get("vehicle_id") is None else str(payload["vehicle_id"])
        ),
        time_basis=(
            None if payload.get("time_basis") is None else str(payload["time_basis"])
        ),
        raw_files_are_immutable=bool(payload.get("raw_files_are_immutable", True)),
        transition_counts_must_not_cross_exclusions=bool(
            payload.get("transition_counts_must_not_cross_exclusions", True)
        ),
        source_path=source_path,
    )


def _coerce_data_exclusion_rules(
    rules: DataExclusionRules | Mapping[str, Any] | str | Path,
) -> DataExclusionRules:
    if isinstance(rules, DataExclusionRules):
        return rules
    if isinstance(rules, Mapping):
        return _parse_data_exclusion_rules(rules)
    return load_data_exclusions(rules)


def _resolve_segment_column(
    frame: pd.DataFrame, requested: str | None
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"missing segment column: {requested}")
        return requested
    for candidate in ("segment_id", "block_id", "source_segment_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("one of segment_id, block_id, or source_segment_id is required")


def _align_rule_timestamp(
    value: pd.Timestamp, timestamps: pd.Series
) -> pd.Timestamp:
    timestamp_tz = timestamps.dt.tz
    value = pd.Timestamp(value)
    if timestamp_tz is None:
        if value.tzinfo is not None:
            raise ValueError("timezone-aware exclusion cannot be applied to naive timestamps")
        return value
    if value.tzinfo is None:
        return value.tz_localize(timestamp_tz)
    return value.tz_convert(timestamp_tz)
