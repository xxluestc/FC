"""Empirical event-level semi-Markov load fitting and generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from fc_power.evaluation.external_validation import (
    DataExclusionRules,
    apply_data_exclusions,
)


@dataclass(frozen=True)
class EmpiricalEventLoadModel:
    """Auditable event-level model learned only from vehicle power data."""

    n_states: int
    state_centers_kw: np.ndarray
    state_signal_centers_kw: np.ndarray
    transition_counts: np.ndarray
    transition_probabilities: np.ndarray
    initial_probabilities: np.ndarray
    dwell_times_s: tuple[np.ndarray, ...]
    event_power_samples_kw: tuple[np.ndarray, ...]
    event_table: pd.DataFrame
    statistics: dict[str, Any]
    power_column: str
    state_signal_column: str
    sampling_interval_s: float
    fit_random_state: int

    def audit_statistics(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the fitted statistics."""

        return deepcopy(self.statistics)


@dataclass
class _CandidateFit:
    n_states: int
    bic: float
    labels: np.ndarray
    signal_centers: np.ndarray
    events: pd.DataFrame
    transition_counts: np.ndarray
    statistics: dict[str, Any]


def fit_empirical_event_load(
    frame: pd.DataFrame,
    *,
    candidate_states: Sequence[int] = (2, 3, 4, 5, 6),
    min_state_occupancy: float = 0.02,
    min_events_per_state: int = 3,
    operating_power_threshold_kw: float = 0.5,
    gap_s: float = 10.0,
    power_column: str = "fc_input_power_kw",
    target_power_column: str | None = "target_power_kw",
    timestamp_column: str = "timestamp",
    segment_column: str | None = None,
    interpolated_column: str = "interpolated_power",
    exclusion_rules: DataExclusionRules | Mapping[str, Any] | str | Path | None = None,
    random_state: int = 2026,
) -> EmpiricalEventLoadModel:
    """Fit a BIC-selected empirical event semi-Markov model.

    Candidate state models are eligible only when every state passes the
    declared occupancy and event-count thresholds and has at least one observed
    outgoing event transition.  State inference may use ``target_power_kw``
    when supplied, while generated event power always comes from the measured
    ``fc_input_power_kw`` events.
    """

    candidates = _validated_candidates(candidate_states)
    if not np.isfinite(min_state_occupancy) or not 0 < min_state_occupancy < 1:
        raise ValueError("min_state_occupancy must lie in (0, 1)")
    if (
        isinstance(min_events_per_state, bool)
        or not isinstance(min_events_per_state, (int, np.integer))
        or min_events_per_state <= 0
    ):
        raise ValueError("min_events_per_state must be a positive integer")
    min_events_per_state = int(min_events_per_state)
    if (
        not np.isfinite(operating_power_threshold_kw)
        or operating_power_threshold_kw < 0
    ):
        raise ValueError("operating_power_threshold_kw must be finite and non-negative")
    if not np.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("gap_s must be finite and positive")

    rows, preparation_audit, exclusion_audit, signal_column = _prepare_rows(
        frame,
        power_column=power_column,
        target_power_column=target_power_column,
        timestamp_column=timestamp_column,
        segment_column=segment_column,
        interpolated_column=interpolated_column,
        exclusion_rules=exclusion_rules,
        operating_power_threshold_kw=operating_power_threshold_kw,
        gap_s=gap_s,
    )
    fit_mask = ~rows["_interpolated"].to_numpy(dtype=bool)
    fit_values = rows.loc[fit_mask, "_signal_kw"].to_numpy(dtype=float)
    if len(fit_values) < min(candidates):
        raise ValueError("too few non-interpolated operating rows for state fitting")
    sampling_interval_s = _infer_sampling_interval(rows)

    candidate_audit: list[dict[str, Any]] = []
    eligible: list[_CandidateFit] = []
    for n_states in candidates:
        candidate = _fit_candidate(
            rows,
            fit_values,
            fit_mask,
            n_states=n_states,
            min_state_occupancy=min_state_occupancy,
            min_events_per_state=min_events_per_state,
            sampling_interval_s=sampling_interval_s,
            random_state=random_state,
        )
        candidate_audit.append(candidate.statistics)
        if candidate.statistics["eligible"]:
            eligible.append(candidate)

    if not eligible:
        reasons = "; ".join(
            f"K={item['n_states']}: {', '.join(item['rejection_reasons'])}"
            for item in candidate_audit
        )
        raise ValueError(f"no candidate state model passed the declared gates ({reasons})")
    selected = min(eligible, key=lambda value: (value.bic, value.n_states))

    labels = selected.labels
    n_states = selected.n_states
    observed_for_centers = fit_mask
    state_centers = np.asarray(
        [
            rows.loc[observed_for_centers & (labels == state), "_power_kw"].mean()
            for state in range(n_states)
        ],
        dtype=float,
    )
    events = selected.events.copy()
    dwell_times = tuple(
        events.loc[events.state == state, "dwell_time_s"].to_numpy(dtype=float)
        for state in range(n_states)
    )
    event_power_samples = tuple(
        events.loc[events.state == state, "power_mean_kw"].to_numpy(dtype=float)
        for state in range(n_states)
    )
    initial_counts = np.bincount(
        events.groupby("model_segment_id", sort=False).first().state.to_numpy(dtype=int),
        minlength=n_states,
    )
    initial_probabilities = initial_counts / initial_counts.sum()
    transition_probabilities = _row_probabilities(selected.transition_counts)

    sample_counts = np.bincount(labels, minlength=n_states)
    event_counts = np.bincount(events.state.to_numpy(dtype=int), minlength=n_states)
    dwell_totals = np.asarray([values.sum() for values in dwell_times], dtype=float)
    total_dwell_s = float(dwell_totals.sum())
    transition_events = int(selected.transition_counts.sum())
    model_statistics: dict[str, Any] = {
        **preparation_audit,
        "exclusions": exclusion_audit,
        "state_signal_column": signal_column,
        "candidate_states": list(candidates),
        "candidate_models": candidate_audit,
        "selection_criterion": "minimum BIC among candidates passing all gates",
        "minimum_state_occupancy": float(min_state_occupancy),
        "minimum_events_per_state": int(min_events_per_state),
        "selected_n_states": int(n_states),
        "selected_bic": float(selected.bic),
        "state_centers_kw": state_centers.tolist(),
        "state_signal_centers_kw": selected.signal_centers.tolist(),
        "state_sample_counts": sample_counts.tolist(),
        "state_sample_occupancy": (sample_counts / sample_counts.sum()).tolist(),
        "state_event_counts": event_counts.tolist(),
        "state_dwell_occupancy": (dwell_totals / total_dwell_s).tolist(),
        "event_count": int(len(events)),
        "event_transition_count": transition_events,
        "transition_counts": selected.transition_counts.tolist(),
        "transition_probabilities": transition_probabilities.tolist(),
        "initial_event_counts": initial_counts.tolist(),
        "initial_probabilities": initial_probabilities.tolist(),
        "sampling_interval_s": float(sampling_interval_s),
        "observed_duration_s": total_dwell_s,
        "event_rate_per_180_s": (
            float(180.0 * transition_events / total_dwell_s)
            if total_dwell_s > 0
            else None
        ),
        "zero_transition_window_probability_180_s": _zero_transition_window_share(
            rows, events, horizon_s=180.0, sampling_interval_s=sampling_interval_s
        ),
        "dwell_time_s": {
            "overall": _distribution_summary(events.dwell_time_s.to_numpy(dtype=float)),
            "by_state": [
                _distribution_summary(values) for values in dwell_times
            ],
        },
        "event_power_kw": {
            "overall": _distribution_summary(
                events.power_mean_kw.to_numpy(dtype=float)
            ),
            "by_state": [
                _distribution_summary(values) for values in event_power_samples
            ],
        },
        "fit_random_state": int(random_state),
        "transition_unit": "compressed adjacent events within model_segment_id",
        "dwell_sampling": "state-conditional empirical event dwell times",
        "engineering_stress_transform": False,
    }
    public_events = events.rename(
        columns={"_timestamp_start": "start_timestamp", "_timestamp_end": "end_timestamp"}
    )
    return EmpiricalEventLoadModel(
        n_states=n_states,
        state_centers_kw=state_centers,
        state_signal_centers_kw=selected.signal_centers.copy(),
        transition_counts=selected.transition_counts.copy(),
        transition_probabilities=transition_probabilities,
        initial_probabilities=initial_probabilities,
        dwell_times_s=tuple(values.copy() for values in dwell_times),
        event_power_samples_kw=tuple(values.copy() for values in event_power_samples),
        event_table=public_events.reset_index(drop=True),
        statistics=model_statistics,
        power_column=power_column,
        state_signal_column=signal_column,
        sampling_interval_s=float(sampling_interval_s),
        fit_random_state=int(random_state),
    )


def fit_empirical_event_table(
    event_table: pd.DataFrame,
    *,
    candidate_states: Sequence[int] = (2, 3, 4, 5, 6),
    min_state_occupancy: float = 0.02,
    min_events_per_state: int = 3,
    min_complete_events_per_state: int = 1,
    min_outgoing_transitions_per_state: int = 1,
    max_fit_samples: int = 200_000,
    segment_column: str | None = None,
    dwell_column: str = "dwell_time_s",
    target_power_column: str = "target_power_kw",
    power_column: str = "fc_input_power_kw",
    order_column: str | None = None,
    left_censored_column: str | None = None,
    right_censored_column: str | None = None,
    random_state: int = 2026,
) -> EmpiricalEventLoadModel:
    """Fit directly from ordered, compressed raw-command power events.

    The GMM fit uses a deterministic systematic-stratified sample whose event
    inclusion frequency is proportional to dwell time.  At most
    ``max_fit_samples`` equivalent seconds are materialized, independent of the
    full archive duration.  The fitted GMM then classifies every input event;
    adjacent equal-state events are merged within, but never across, segments.

    Input row order is authoritative and is never sorted internally.  Segment
    identifiers must form contiguous blocks.  When an order column is supplied
    (or a conventional event/timestamp column is detected), it must increase
    strictly within each segment.
    """

    candidates = _validated_candidates(candidate_states)
    if not np.isfinite(min_state_occupancy) or not 0 < min_state_occupancy < 1:
        raise ValueError("min_state_occupancy must lie in (0, 1)")
    min_events_per_state = _positive_integer(
        min_events_per_state, "min_events_per_state"
    )
    min_complete_events_per_state = _positive_integer(
        min_complete_events_per_state, "min_complete_events_per_state"
    )
    min_outgoing_transitions_per_state = _positive_integer(
        min_outgoing_transitions_per_state,
        "min_outgoing_transitions_per_state",
    )
    max_fit_samples = _positive_integer(max_fit_samples, "max_fit_samples")
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be an integer")
    random_state = int(random_state)

    rows, preparation_audit = _prepare_compressed_event_rows(
        event_table,
        segment_column=segment_column,
        dwell_column=dwell_column,
        target_power_column=target_power_column,
        power_column=power_column,
        order_column=order_column,
        left_censored_column=left_censored_column,
        right_censored_column=right_censored_column,
    )
    fit_values, sampling_audit = _dwell_weighted_fit_sample(
        rows._signal_kw.to_numpy(dtype=float),
        rows._dwell_time_s.to_numpy(dtype=float),
        max_fit_samples=max_fit_samples,
        random_state=random_state,
    )

    candidate_audit: list[dict[str, Any]] = []
    eligible: list[_CandidateFit] = []
    for n_states in candidates:
        candidate = _fit_event_table_candidate(
            rows,
            fit_values,
            n_states=n_states,
            min_state_occupancy=min_state_occupancy,
            min_events_per_state=min_events_per_state,
            min_complete_events_per_state=min_complete_events_per_state,
            min_outgoing_transitions_per_state=(
                min_outgoing_transitions_per_state
            ),
            random_state=random_state,
        )
        candidate_audit.append(candidate.statistics)
        if candidate.statistics["eligible"]:
            eligible.append(candidate)

    if not eligible:
        reasons = "; ".join(
            f"K={item['n_states']}: {', '.join(item['rejection_reasons'])}"
            for item in candidate_audit
        )
        raise ValueError(f"no candidate state model passed the declared gates ({reasons})")
    selected = min(eligible, key=lambda value: (value.bic, value.n_states))

    events = selected.events.copy()
    n_states = selected.n_states
    complete_events = events.loc[
        ~(events.left_censored | events.right_censored)
    ].copy()
    dwell_times = tuple(
        complete_events.loc[
            complete_events.state == state, "dwell_time_s"
        ].to_numpy(dtype=float)
        for state in range(n_states)
    )
    event_power_samples = tuple(
        complete_events.loc[
            complete_events.state == state, "power_mean_kw"
        ].to_numpy(dtype=float)
        for state in range(n_states)
    )
    dwell_totals = np.bincount(
        events.state.to_numpy(dtype=int),
        weights=events.dwell_time_s.to_numpy(dtype=float),
        minlength=n_states,
    )
    total_dwell_s = float(events.dwell_time_s.sum())
    state_centers = np.asarray(
        [
            np.average(
                events.loc[events.state == state, "power_mean_kw"],
                weights=events.loc[events.state == state, "dwell_time_s"],
            )
            for state in range(n_states)
        ],
        dtype=float,
    )
    event_counts = np.bincount(
        events.state.to_numpy(dtype=int), minlength=n_states
    )
    initial_counts = np.bincount(
        events.groupby("model_segment_id", sort=False).first().state.to_numpy(dtype=int),
        minlength=n_states,
    )
    initial_probabilities = initial_counts / initial_counts.sum()
    transition_probabilities = _row_probabilities(selected.transition_counts)
    transition_events = int(selected.transition_counts.sum())
    selected_candidate = selected.statistics
    sample_counts = selected_candidate["weighted_fit_sample_state_counts"]
    sample_count_total = int(sum(sample_counts))

    model_statistics: dict[str, Any] = {
        **preparation_audit,
        "input_kind": "compressed_raw_command_event_table",
        "exclusions": None,
        "state_signal_column": target_power_column,
        "candidate_states": list(candidates),
        "candidate_models": candidate_audit,
        "selection_criterion": "minimum BIC among candidates passing all gates",
        "minimum_state_occupancy": float(min_state_occupancy),
        "minimum_events_per_state": int(min_events_per_state),
        "minimum_complete_events_per_state": int(
            min_complete_events_per_state
        ),
        "minimum_outgoing_transitions_per_state": int(
            min_outgoing_transitions_per_state
        ),
        "weighted_gmm_sampling": sampling_audit,
        "selected_n_states": int(n_states),
        "selected_bic": float(selected.bic),
        "state_centers_kw": state_centers.tolist(),
        "state_signal_centers_kw": selected.signal_centers.tolist(),
        "state_sample_counts": list(sample_counts),
        "state_sample_occupancy": [
            float(value / sample_count_total) for value in sample_counts
        ],
        "state_dwell_seconds": dwell_totals.tolist(),
        "state_dwell_occupancy": (dwell_totals / total_dwell_s).tolist(),
        "state_event_counts": event_counts.tolist(),
        "state_complete_event_counts": np.bincount(
            complete_events.state.to_numpy(dtype=int), minlength=n_states
        ).tolist(),
        "left_censored_event_count": int(events.left_censored.sum()),
        "right_censored_event_count": int(events.right_censored.sum()),
        "merged_event_count": int(len(events)),
        "event_count": int(len(events)),
        "event_transition_count": transition_events,
        "transition_counts": selected.transition_counts.tolist(),
        "transition_probabilities": transition_probabilities.tolist(),
        "initial_event_counts": initial_counts.tolist(),
        "initial_probabilities": initial_probabilities.tolist(),
        "sampling_interval_s": 1.0,
        "observed_duration_s": total_dwell_s,
        "event_rate_per_180_s": float(180.0 * transition_events / total_dwell_s),
        "zero_transition_window_probability_180_s": (
            _compressed_zero_transition_window_share(events, horizon_s=180.0)
        ),
        "dwell_time_s": {
            "overall": _distribution_summary(
                events.dwell_time_s.to_numpy(dtype=float)
            ),
            "by_state": [_distribution_summary(values) for values in dwell_times],
        },
        "event_power_kw": {
            "overall": _distribution_summary(
                events.power_mean_kw.to_numpy(dtype=float)
            ),
            "by_state": [
                _distribution_summary(values) for values in event_power_samples
            ],
        },
        "fit_random_state": random_state,
        "transition_unit": "merged adjacent states within model_segment_id",
        "dwell_sampling": (
            "paired state-conditional dwell and realized power from complete "
            "uncensored merged events"
        ),
        "engineering_stress_transform": False,
    }
    return EmpiricalEventLoadModel(
        n_states=n_states,
        state_centers_kw=state_centers,
        state_signal_centers_kw=selected.signal_centers.copy(),
        transition_counts=selected.transition_counts.copy(),
        transition_probabilities=transition_probabilities,
        initial_probabilities=initial_probabilities,
        dwell_times_s=tuple(values.copy() for values in dwell_times),
        event_power_samples_kw=tuple(
            values.copy() for values in event_power_samples
        ),
        event_table=events.reset_index(drop=True),
        statistics=model_statistics,
        power_column=power_column,
        state_signal_column=target_power_column,
        sampling_interval_s=1.0,
        fit_random_state=random_state,
    )


def generate_empirical_event_load(
    model: EmpiricalEventLoadModel,
    *,
    length_s: float,
    seed: int,
    dt_s: float = 1.0,
    mode: str = "empirical",
    dwell_time_scale: float | None = None,
) -> pd.DataFrame:
    """Generate by event, with an explicit opt-in dwell-only stress transform.

    ``mode="empirical"`` samples fitted dwell times without modification.
    ``mode="stress"`` requires a declared ``0 < dwell_time_scale < 1``.  The
    stress mode does not alter vehicle-fitted powers or event transitions.
    """

    if not np.isfinite(length_s) or length_s <= 0:
        raise ValueError("length_s must be finite and positive")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if mode == "empirical":
        if dwell_time_scale is not None and not np.isclose(dwell_time_scale, 1.0):
            raise ValueError("empirical mode does not permit dwell-time scaling")
        scale = 1.0
        stress_transform = False
    elif mode == "stress":
        if (
            dwell_time_scale is None
            or not np.isfinite(dwell_time_scale)
            or not 0 < dwell_time_scale < 1
        ):
            raise ValueError("stress mode requires an explicit dwell_time_scale in (0, 1)")
        scale = float(dwell_time_scale)
        stress_transform = True
    else:
        raise ValueError("mode must be 'empirical' or 'stress'")

    samples = int(np.ceil(length_s / dt_s))
    rng = np.random.default_rng(seed)
    state = int(rng.choice(model.n_states, p=model.initial_probabilities))
    rows: list[dict[str, Any]] = []
    event_id = 0
    while len(rows) < samples:
        event_sample_index = int(rng.integers(len(model.dwell_times_s[state])))
        empirical_dwell_s = float(model.dwell_times_s[state][event_sample_index])
        effective_dwell_s = empirical_dwell_s * scale
        event_steps = max(1, int(np.ceil(effective_dwell_s / dt_s)))
        event_power_kw = float(
            model.event_power_samples_kw[state][event_sample_index]
        )
        take = min(event_steps, samples - len(rows))
        for within_event in range(take):
            step = len(rows)
            rows.append(
                {
                    "step": step,
                    "time_s": float(step * dt_s),
                    "demand_power_kw": event_power_kw,
                    "load_state": state,
                    "event": f"empirical_state_{state}",
                    "event_id": event_id,
                    "event_boundary": within_event == 0,
                    "semi_markov_decision": within_event == 0,
                    "sampled_empirical_dwell_s": empirical_dwell_s,
                    "effective_dwell_time_s": float(event_steps * dt_s),
                    "generation_mode": mode,
                    "engineering_stress_transform": stress_transform,
                    "dwell_time_scale": scale,
                    "source": (
                        "vehicle_empirical_event_semi_markov_stress"
                        if stress_transform
                        else "vehicle_empirical_event_semi_markov"
                    ),
                    "seed": seed,
                }
            )
        event_id += 1
        if len(rows) < samples:
            state = int(rng.choice(model.n_states, p=model.transition_probabilities[state]))

    profile = pd.DataFrame(rows)
    profile.attrs["generation_audit"] = {
        "mode": mode,
        "engineering_stress_transform": stress_transform,
        "dwell_time_scale": scale,
        "length_s": float(length_s),
        "dt_s": float(dt_s),
        "generated_events": int(profile.event_boundary.sum()),
        "generated_transitions": int(max(0, profile.event_boundary.sum() - 1)),
        "model_selected_n_states": int(model.n_states),
        "power_sampling": "state-conditional vehicle event means",
        "transition_sampling": "vehicle event-level transition rows",
    }
    return profile


def _prepare_rows(
    frame: pd.DataFrame,
    *,
    power_column: str,
    target_power_column: str | None,
    timestamp_column: str,
    segment_column: str | None,
    interpolated_column: str,
    exclusion_rules: DataExclusionRules | Mapping[str, Any] | str | Path | None,
    operating_power_threshold_kw: float,
    gap_s: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None, str]:
    required = {timestamp_column, power_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing empirical event columns: {sorted(missing)}")
    resolved_segment = _resolve_segment_column(frame, segment_column)

    exclusion_audit = None
    work = frame.copy().reset_index(drop=True)
    if exclusion_rules is not None:
        work, exclusion_audit = apply_data_exclusions(
            work,
            exclusion_rules,
            timestamp_column=timestamp_column,
            segment_column=resolved_segment,
            output_segment_column="_exclusion_segment_id",
        )
        resolved_segment = "_exclusion_segment_id"

    timestamps = pd.to_datetime(work[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("empirical event fitting requires valid timestamps")
    if work[resolved_segment].isna().any():
        raise ValueError("empirical event fitting requires non-missing segment identifiers")
    power = pd.to_numeric(work[power_column], errors="coerce")
    use_target = (
        target_power_column is not None and target_power_column in work.columns
    )
    signal_column = target_power_column if use_target else power_column
    signal = (
        pd.to_numeric(work[target_power_column], errors="coerce")
        if use_target
        else power.copy()
    )
    interpolated = _coerce_interpolated_flags(work, interpolated_column)

    prepared = pd.DataFrame(
        {
            "_timestamp": timestamps,
            "_source_segment": work[resolved_segment].to_numpy(),
            "_power_kw": power,
            "_signal_kw": signal,
            "_interpolated": interpolated,
        }
    ).sort_values(["_source_segment", "_timestamp"], kind="mergesort")
    duplicate = prepared.duplicated(["_source_segment", "_timestamp"])
    if duplicate.any():
        raise ValueError("duplicate timestamps within an empirical source segment")

    finite = np.isfinite(prepared._power_kw) & np.isfinite(prepared._signal_kw)
    operating = finite & (prepared._power_kw > operating_power_threshold_kw)
    source_changed = prepared._source_segment.ne(prepared._source_segment.shift())
    dt = prepared._timestamp.diff().dt.total_seconds()
    cadence_break = dt.le(0) | dt.gt(gap_s)
    previous_not_operating = ~operating.shift(fill_value=False)
    continuity_start = source_changed | cadence_break | previous_not_operating
    prepared["_continuity_run"] = continuity_start.cumsum()
    prepared = prepared.loc[operating].copy()
    if prepared.empty:
        raise ValueError("no finite operating power rows remain for event fitting")
    model_keys = pd.MultiIndex.from_frame(
        prepared[["_source_segment", "_continuity_run"]]
    )
    prepared["model_segment_id"] = pd.factorize(model_keys, sort=False)[0].astype(int)
    prepared = prepared.reset_index(drop=True)

    audit = {
        "input_rows": int(len(frame)),
        "rows_after_exclusions": int(len(work)),
        "model_rows": int(len(prepared)),
        "nonfinite_or_missing_signal_rows": int((~finite).sum()),
        "off_or_below_threshold_rows": int((finite & ~operating).sum()),
        "operating_power_threshold_kw": float(operating_power_threshold_kw),
        "interpolated_model_rows": int(prepared._interpolated.sum()),
        "noninterpolated_fit_rows": int((~prepared._interpolated).sum()),
        "source_segment_column": resolved_segment,
        "model_segment_count": int(prepared.model_segment_id.nunique()),
        "gap_threshold_s": float(gap_s),
        "interpolated_points_can_create_event_boundaries": False,
    }
    return prepared, audit, exclusion_audit, str(signal_column)


def _prepare_compressed_event_rows(
    event_table: pd.DataFrame,
    *,
    segment_column: str | None,
    dwell_column: str,
    target_power_column: str,
    power_column: str,
    order_column: str | None,
    left_censored_column: str | None,
    right_censored_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_segment = _resolve_segment_column(event_table, segment_column)
    required = {
        resolved_segment,
        dwell_column,
        target_power_column,
        power_column,
    }
    missing = required.difference(event_table.columns)
    if missing:
        raise ValueError(f"missing compressed event columns: {sorted(missing)}")
    if event_table.empty:
        raise ValueError("compressed event table must not be empty")
    for column in (left_censored_column, right_censored_column):
        if column is not None and column not in event_table.columns:
            raise ValueError(f"missing compressed event censor column: {column}")

    segment = event_table[resolved_segment]
    if segment.isna().any() or any(
        isinstance(value, str) and not value.strip() for value in segment
    ):
        raise ValueError("compressed event segment identifiers must be non-empty")
    segment_codes, _ = pd.factorize(segment, sort=False)
    block_starts = np.r_[True, segment_codes[1:] != segment_codes[:-1]]
    block_codes = segment_codes[block_starts]
    if len(np.unique(block_codes)) != len(block_codes):
        raise ValueError(
            "compressed event rows must be ordered with each segment contiguous"
        )

    resolved_order = _resolve_event_order_column(event_table, order_column)
    ordering_validation = "input row order with contiguous segment blocks"
    if resolved_order is not None:
        _validate_event_order(event_table[resolved_order], segment_codes, resolved_order)
        ordering_validation = f"strictly increasing {resolved_order} within segment"

    dwell = pd.to_numeric(event_table[dwell_column], errors="coerce").to_numpy(
        dtype=float
    )
    signal = pd.to_numeric(
        event_table[target_power_column], errors="coerce"
    ).to_numpy(dtype=float)
    power = pd.to_numeric(event_table[power_column], errors="coerce").to_numpy(
        dtype=float
    )
    if np.any(~np.isfinite(dwell)) or np.any(dwell <= 0):
        raise ValueError("compressed event dwell_time_s must be finite and positive")
    total_dwell_s = float(dwell.sum())
    if not np.isfinite(total_dwell_s):
        raise ValueError("compressed event total dwell_time_s must be finite")
    if np.any(~np.isfinite(signal)):
        raise ValueError("compressed event target power must be finite")
    if np.any(~np.isfinite(power)):
        raise ValueError("compressed event realized power must be finite")
    left_censored = (
        np.zeros(len(event_table), dtype=bool)
        if left_censored_column is None
        else event_table[left_censored_column].astype(bool).to_numpy()
    )
    right_censored = (
        np.zeros(len(event_table), dtype=bool)
        if right_censored_column is None
        else event_table[right_censored_column].astype(bool).to_numpy()
    )

    rows = pd.DataFrame(
        {
            "_source_segment": segment.to_numpy(),
            "_model_segment_id": segment_codes.astype(int),
            "_dwell_time_s": dwell,
            "_signal_kw": signal,
            "_power_kw": power,
            "_raw_event_index": np.arange(len(event_table), dtype=int),
            "_left_censored": left_censored,
            "_right_censored": right_censored,
        }
    )
    audit = {
        "input_rows": int(len(event_table)),
        "raw_command_event_count": int(len(event_table)),
        "source_segment_column": resolved_segment,
        "model_segment_count": int(len(block_codes)),
        "dwell_column": dwell_column,
        "power_column": power_column,
        "state_signal_column": target_power_column,
        "order_column": resolved_order,
        "ordering_validation": ordering_validation,
        "input_total_dwell_s": total_dwell_s,
        "left_censored_column": left_censored_column,
        "right_censored_column": right_censored_column,
        "left_censored_raw_event_count": int(left_censored.sum()),
        "right_censored_raw_event_count": int(right_censored.sum()),
    }
    return rows, audit


def _dwell_weighted_fit_sample(
    signal: np.ndarray,
    dwell_time_s: np.ndarray,
    *,
    max_fit_samples: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_dwell_s = float(dwell_time_s.sum())
    sample_count = min(max_fit_samples, max(1, int(np.ceil(total_dwell_s))))
    rng = np.random.default_rng(random_state)
    random_offset = float(rng.random())
    positions = (
        (np.arange(sample_count, dtype=float) + random_offset)
        * total_dwell_s
        / sample_count
    )
    event_indices = np.searchsorted(
        np.cumsum(dwell_time_s), positions, side="right"
    )
    event_indices = np.minimum(event_indices, len(signal) - 1)
    sample = signal[event_indices]
    audit = {
        "method": "systematic_stratified_dwell_proportional",
        "weight_column": "dwell_time_s",
        "maximum_effective_seconds": int(max_fit_samples),
        "total_effective_seconds": total_dwell_s,
        "actual_sample_count": int(len(sample)),
        "capped": bool(sample_count < int(np.ceil(total_dwell_s))),
        "random_state": int(random_state),
        "random_offset_fraction": random_offset,
        "full_event_table_expanded_to_rows": False,
    }
    return sample, audit


def _fit_event_table_candidate(
    rows: pd.DataFrame,
    fit_values: np.ndarray,
    *,
    n_states: int,
    min_state_occupancy: float,
    min_events_per_state: int,
    min_complete_events_per_state: int,
    min_outgoing_transitions_per_state: int,
    random_state: int,
) -> _CandidateFit:
    rejected: list[str] = []
    base_statistics: dict[str, Any] = {
        "n_states": int(n_states),
        "bic": None,
        "eligible": False,
        "rejection_reasons": rejected,
    }
    if len(fit_values) < n_states or len(np.unique(fit_values)) < n_states:
        rejected.append("fewer distinct weighted fit values than states")
        return _CandidateFit(
            n_states=n_states,
            bic=float("inf"),
            labels=np.empty(0, dtype=int),
            signal_centers=np.empty(0),
            events=pd.DataFrame(),
            transition_counts=np.zeros((n_states, n_states), dtype=int),
            statistics=base_statistics,
        )

    mixture = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=3,
        init_params="k-means++",
        max_iter=500,
        random_state=random_state,
    )
    fit_matrix = fit_values.reshape(-1, 1)
    mixture.fit(fit_matrix)
    bic = float(mixture.bic(fit_matrix))
    raw_centers = mixture.means_.ravel()
    order = np.argsort(raw_centers)
    remap = np.empty(n_states, dtype=int)
    remap[order] = np.arange(n_states)
    labels = remap[
        mixture.predict(rows[["_signal_kw"]].to_numpy(dtype=float))
    ]
    fit_labels = remap[mixture.predict(fit_matrix)]
    centers = raw_centers[order]
    events = _merge_classified_event_rows(rows, labels)
    transition_counts = _transition_counts(events, n_states)
    dwell_totals = np.bincount(
        labels,
        weights=rows._dwell_time_s.to_numpy(dtype=float),
        minlength=n_states,
    )
    occupancy = dwell_totals / dwell_totals.sum()
    event_counts = np.bincount(
        events.state.to_numpy(dtype=int), minlength=n_states
    )
    complete_events = events.loc[
        ~(events.left_censored | events.right_censored)
    ]
    complete_event_counts = np.bincount(
        complete_events.state.to_numpy(dtype=int), minlength=n_states
    )
    outgoing_counts = transition_counts.sum(axis=1)
    fit_sample_counts = np.bincount(fit_labels, minlength=n_states)

    if np.any(occupancy < min_state_occupancy):
        rejected.append("state dwell occupancy below minimum")
    if np.any(event_counts < min_events_per_state):
        rejected.append("merged state event count below minimum")
    if np.any(complete_event_counts < min_complete_events_per_state):
        rejected.append("complete merged state event count below minimum")
    if np.any(outgoing_counts < min_outgoing_transitions_per_state):
        rejected.append("state outgoing event transitions below minimum")
    statistics = {
        "n_states": int(n_states),
        "bic": bic,
        "eligible": not rejected,
        "rejection_reasons": rejected,
        "converged": bool(mixture.converged_),
        "state_signal_centers_kw": centers.tolist(),
        "mixture_weights": mixture.weights_[order].tolist(),
        "mixture_variances": mixture.covariances_.reshape(-1)[order].tolist(),
        "weighted_fit_sample_state_counts": fit_sample_counts.tolist(),
        "state_dwell_seconds": dwell_totals.tolist(),
        "state_occupancy": occupancy.tolist(),
        "state_event_counts": event_counts.tolist(),
        "state_complete_event_counts": complete_event_counts.tolist(),
        "state_outgoing_event_counts": outgoing_counts.tolist(),
        "raw_command_event_count": int(len(rows)),
        "merged_event_count": int(len(events)),
        "event_transition_count": int(transition_counts.sum()),
    }
    return _CandidateFit(
        n_states=n_states,
        bic=bic,
        labels=labels,
        signal_centers=centers,
        events=events,
        transition_counts=transition_counts,
        statistics=statistics,
    )


def _merge_classified_event_rows(
    rows: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    segments = rows._model_segment_id.to_numpy(dtype=int)
    boundaries = np.r_[
        True,
        (segments[1:] != segments[:-1]) | (labels[1:] != labels[:-1]),
    ]
    grouped = rows.copy()
    grouped["_state"] = labels
    grouped["_merged_event_id"] = np.cumsum(boundaries) - 1
    event_rows: list[dict[str, Any]] = []
    for event_id, group in grouped.groupby("_merged_event_id", sort=False):
        dwell = group._dwell_time_s.to_numpy(dtype=float)
        total_dwell = float(dwell.sum())
        event_rows.append(
            {
                "event_id": int(event_id),
                "model_segment_id": int(group._model_segment_id.iloc[0]),
                "source_segment_id": group._source_segment.iloc[0],
                "state": int(group._state.iloc[0]),
                "dwell_time_s": total_dwell,
                "raw_event_count": int(len(group)),
                "start_raw_event_index": int(group._raw_event_index.iloc[0]),
                "end_raw_event_index": int(group._raw_event_index.iloc[-1]),
                "left_censored": bool(group._left_censored.iloc[0]),
                "right_censored": bool(group._right_censored.iloc[-1]),
                "power_mean_kw": float(
                    np.average(group._power_kw.to_numpy(dtype=float), weights=dwell)
                ),
                "state_signal_mean_kw": float(
                    np.average(group._signal_kw.to_numpy(dtype=float), weights=dwell)
                ),
            }
        )
    return pd.DataFrame(event_rows)


def _compressed_zero_transition_window_share(
    events: pd.DataFrame, *, horizon_s: float
) -> float | None:
    zero_weight = 0.0
    eligible_weight = 0.0
    for _, segment_events in events.groupby("model_segment_id", sort=False):
        dwell = segment_events.dwell_time_s.to_numpy(dtype=float)
        eligible_weight += max(float(dwell.sum()) - horizon_s + 1.0, 0.0)
        zero_weight += float(np.maximum(dwell - horizon_s + 1.0, 0.0).sum())
    if eligible_weight <= 0:
        return None
    return float(min(1.0, zero_weight / eligible_weight))


def _resolve_event_order_column(
    frame: pd.DataFrame, requested: str | None
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"missing compressed event order column: {requested}")
        return requested
    for candidate in (
        "event_order",
        "event_index",
        "event_id",
        "start_timestamp",
        "timestamp",
    ):
        if candidate in frame.columns:
            return candidate
    return None


def _validate_event_order(
    values: pd.Series, segment_codes: np.ndarray, column: str
) -> None:
    if "timestamp" in column.lower() or pd.api.types.is_datetime64_any_dtype(values):
        ordered = pd.to_datetime(values, errors="coerce")
    else:
        ordered = pd.to_numeric(values, errors="coerce")
    if ordered.isna().any():
        raise ValueError(f"compressed event order column must be finite: {column}")
    for segment in np.unique(segment_codes):
        segment_values = ordered.iloc[np.flatnonzero(segment_codes == segment)]
        differences = segment_values.diff().iloc[1:]
        threshold = (
            pd.Timedelta(0)
            if pd.api.types.is_timedelta64_dtype(differences)
            else 0
        )
        if len(segment_values) > 1 and not (differences > threshold).all():
            raise ValueError(
                f"compressed events must be strictly ordered within segment by {column}"
            )


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fit_candidate(
    rows: pd.DataFrame,
    fit_values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_states: int,
    min_state_occupancy: float,
    min_events_per_state: int,
    sampling_interval_s: float,
    random_state: int,
) -> _CandidateFit:
    rejected: list[str] = []
    base_statistics: dict[str, Any] = {
        "n_states": int(n_states),
        "bic": None,
        "eligible": False,
        "rejection_reasons": rejected,
    }
    unique_values = np.unique(fit_values)
    if len(fit_values) < n_states or len(unique_values) < n_states:
        rejected.append("fewer distinct fit values than states")
        return _CandidateFit(
            n_states=n_states,
            bic=float("inf"),
            labels=np.empty(0, dtype=int),
            signal_centers=np.empty(0),
            events=pd.DataFrame(),
            transition_counts=np.zeros((n_states, n_states), dtype=int),
            statistics=base_statistics,
        )

    mixture = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=3,
        init_params="k-means++",
        max_iter=500,
        random_state=random_state,
    )
    mixture.fit(fit_values.reshape(-1, 1))
    bic = float(mixture.bic(fit_values.reshape(-1, 1)))
    raw_labels = mixture.predict(rows[["_signal_kw"]].to_numpy(dtype=float))
    raw_centers = mixture.means_.ravel()
    order = np.argsort(raw_centers)
    remap = np.empty(n_states, dtype=int)
    remap[order] = np.arange(n_states)
    labels = remap[raw_labels]
    labels = _suppress_interpolated_boundaries(rows, labels)
    centers = raw_centers[order]
    events = _compress_events(rows, labels, sampling_interval_s)
    transition_counts = _transition_counts(events, n_states)
    sample_counts = np.bincount(labels, minlength=n_states)
    occupancy = sample_counts / sample_counts.sum()
    event_counts = np.bincount(events.state.to_numpy(dtype=int), minlength=n_states)
    outgoing_counts = transition_counts.sum(axis=1)

    if np.any(occupancy < min_state_occupancy):
        rejected.append("state occupancy below minimum")
    if np.any(event_counts < min_events_per_state):
        rejected.append("state event count below minimum")
    if np.any(outgoing_counts == 0):
        rejected.append("state has no observed outgoing event transition")
    interpolation_boundaries = int(
        (
            rows._interpolated.to_numpy(dtype=bool)
            & np.r_[False, labels[1:] != labels[:-1]]
            & ~rows.model_segment_id.ne(rows.model_segment_id.shift()).to_numpy()
        ).sum()
    )
    statistics = {
        "n_states": int(n_states),
        "bic": bic,
        "eligible": not rejected,
        "rejection_reasons": rejected,
        "converged": bool(mixture.converged_),
        "state_signal_centers_kw": centers.tolist(),
        "mixture_weights": mixture.weights_[order].tolist(),
        "mixture_variances": mixture.covariances_.reshape(-1)[order].tolist(),
        "state_sample_counts": sample_counts.tolist(),
        "state_occupancy": occupancy.tolist(),
        "state_event_counts": event_counts.tolist(),
        "state_outgoing_event_counts": outgoing_counts.tolist(),
        "event_count": int(len(events)),
        "event_transition_count": int(transition_counts.sum()),
        "interpolated_event_boundaries": interpolation_boundaries,
    }
    return _CandidateFit(
        n_states=n_states,
        bic=bic,
        labels=labels,
        signal_centers=centers,
        events=events,
        transition_counts=transition_counts,
        statistics=statistics,
    )


def _suppress_interpolated_boundaries(
    rows: pd.DataFrame, labels: np.ndarray
) -> np.ndarray:
    corrected = labels.copy()
    for _, positions in rows.groupby("model_segment_id", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        interpolated = rows.loc[positions, "_interpolated"].to_numpy(dtype=bool)
        observed_positions = np.flatnonzero(~interpolated)
        if len(observed_positions) == 0:
            corrected[positions] = corrected[positions[0]]
            continue
        first_observed = int(observed_positions[0])
        corrected[positions[:first_observed]] = corrected[positions[first_observed]]
        for local in range(first_observed + 1, len(positions)):
            if interpolated[local]:
                corrected[positions[local]] = corrected[positions[local - 1]]
    return corrected


def _compress_events(
    rows: pd.DataFrame, labels: np.ndarray, sampling_interval_s: float
) -> pd.DataFrame:
    segment_values = rows.model_segment_id.to_numpy(dtype=int)
    boundaries = np.r_[
        True,
        (segment_values[1:] != segment_values[:-1]) | (labels[1:] != labels[:-1]),
    ]
    event_ids = np.cumsum(boundaries) - 1
    segment_cadence: dict[int, float] = {}
    for segment, segment_rows in rows.groupby("model_segment_id", sort=False):
        segment_dt = segment_rows._timestamp.diff().dt.total_seconds()
        positive_dt = segment_dt[segment_dt > 0]
        segment_cadence[int(segment)] = (
            float(positive_dt.median()) if len(positive_dt) else sampling_interval_s
        )
    grouped_rows = rows.copy()
    grouped_rows["_event_id"] = event_ids
    grouped_rows["_state"] = labels
    event_rows = []
    for event_id, group in grouped_rows.groupby("_event_id", sort=False):
        segment = int(group.model_segment_id.iloc[0])
        dt_s = segment_cadence[segment]
        duration_s = float(
            (group._timestamp.iloc[-1] - group._timestamp.iloc[0]).total_seconds()
            + dt_s
        )
        event_rows.append(
            {
                "event_id": int(event_id),
                "model_segment_id": segment,
                "state": int(group._state.iloc[0]),
                "_timestamp_start": group._timestamp.iloc[0],
                "_timestamp_end": group._timestamp.iloc[-1],
                "dwell_time_s": duration_s,
                "sample_count": int(len(group)),
                "interpolated_sample_count": int(group._interpolated.sum()),
                "power_mean_kw": float(group._power_kw.mean()),
                "power_median_kw": float(group._power_kw.median()),
                "state_signal_mean_kw": float(group._signal_kw.mean()),
            }
        )
    return pd.DataFrame(event_rows)


def _transition_counts(events: pd.DataFrame, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=int)
    for _, segment_events in events.groupby("model_segment_id", sort=False):
        states = segment_events.state.to_numpy(dtype=int)
        if len(states) > 1:
            np.add.at(counts, (states[:-1], states[1:]), 1)
    return counts


def _row_probabilities(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("every fitted state requires an observed outgoing event transition")
    return counts / totals


def _zero_transition_window_share(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_s: float,
    sampling_interval_s: float,
) -> float | None:
    zero_transition = 0
    eligible_windows = 0
    horizon = np.timedelta64(int(round(horizon_s * 1e9)), "ns")
    for segment, segment_rows in rows.groupby("model_segment_id", sort=False):
        timestamps = segment_rows._timestamp.to_numpy(dtype="datetime64[ns]")
        if len(timestamps) == 0:
            continue
        end = timestamps[-1] + np.timedelta64(
            int(round(sampling_interval_s * 1e9)), "ns"
        )
        starts = timestamps[timestamps + horizon <= end]
        if not len(starts):
            continue
        event_starts = events.loc[
            events.model_segment_id == segment, "_timestamp_start"
        ].to_numpy(dtype="datetime64[ns]")[1:]
        indices = np.searchsorted(event_starts, starts, side="right")
        next_starts = np.full(len(starts), np.datetime64("NaT"), dtype="datetime64[ns]")
        has_next = indices < len(event_starts)
        next_starts[has_next] = event_starts[indices[has_next]]
        no_change = ~has_next | (next_starts >= starts + horizon)
        zero_transition += int(no_change.sum())
        eligible_windows += int(len(starts))
    if eligible_windows == 0:
        return None
    return float(zero_transition / eligible_windows)


def _distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "maximum": None,
        }
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "maximum": float(values.max()),
    }


def _infer_sampling_interval(rows: pd.DataFrame) -> float:
    differences = rows.groupby("model_segment_id", sort=False)._timestamp.diff()
    seconds = differences.dt.total_seconds()
    positive = seconds[seconds > 0]
    return float(positive.median()) if len(positive) else 1.0


def _coerce_interpolated_flags(
    frame: pd.DataFrame, column: str
) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    values = frame[column]
    if values.isna().any():
        raise ValueError("interpolated_power flags must not be missing")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("interpolated_power flags must be boolean or 0/1")
    return numeric.to_numpy(dtype=bool)


def _resolve_segment_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"missing segment column: {requested}")
        return requested
    for candidate in (
        "segment_id",
        "block_id",
        "source_segment_id",
        "archive_event_segment_id",
        "model_segment_id",
    ):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "one of segment_id, block_id, source_segment_id, archive_event_segment_id, "
        "or model_segment_id is required"
    )


def _validated_candidates(values: Sequence[int]) -> tuple[int, ...]:
    candidates = tuple(values)
    if not candidates:
        raise ValueError("candidate_states must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2
        for value in candidates
    ):
        raise ValueError("candidate_states must contain integers of at least two")
    if len(set(int(value) for value in candidates)) != len(candidates):
        raise ValueError("candidate_states must not contain duplicates")
    return tuple(sorted(int(value) for value in candidates))
