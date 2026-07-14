"""Data-derived progress coordinate for the recorded LZW theta trajectory.

The coordinate is fitted only from the ordered theta observations.  It does
not consume action exposure, damage coefficients, or a stochastic process
calibration.  The terminal value ``h=1`` is reserved for the endpoint of the
recorded LZW trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


THETA_COLUMNS = (
    "i0_A_per_cm2",
    "ih_A_per_cm2",
    "R_ohm_reported_ohm_cm2",
)

# A positive sign means that the reported component increases along the LZW
# degradation trajectory.  i0 moves in the opposite direction.
DEGRADATION_DIRECTIONS = (-1, 1, 1)
DEGRADATION_DIRECTION_LABELS = ("decreasing", "increasing", "increasing")

_TERMINAL_NONMATCH_CAP = 1.0 - 1e-12


def _canonical_row_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        column
        for column in frame.columns
        if column == "canonical_row" or column.startswith("canonical_row_")
    ]
    if len(candidates) > 1:
        raise ValueError(
            "key table has multiple canonical_row columns; exactly one is required"
        )
    return candidates[0] if candidates else None


def _validated_key_values(
    frame: pd.DataFrame, *, table_name: str
) -> tuple[tuple[str, str, str] | None, np.ndarray | None]:
    canonical = _canonical_row_column(frame)
    present = {
        "event_id": "event_id" in frame.columns,
        "canonical_row": canonical is not None,
        "original_index": "original_index" in frame.columns,
    }
    if not any(present.values()):
        return None, None
    if not all(present.values()):
        missing = sorted(name for name, exists in present.items() if not exists)
        raise ValueError(
            f"{table_name} provides an incomplete LZW key; missing {missing}"
        )

    assert canonical is not None
    columns = ("event_id", canonical, "original_index")
    keys = frame.loc[:, columns]
    if keys.isna().any().any():
        raise ValueError(f"{table_name} LZW keys must not contain missing values")
    for column in columns:
        if keys[column].duplicated().any():
            raise ValueError(f"{table_name} key column {column!r} must be unique")
    if keys.duplicated().any():
        raise ValueError(f"{table_name} composite LZW key must be unique")

    numeric_keys: dict[str, np.ndarray] = {}
    for column in (canonical, "original_index"):
        converted = pd.to_numeric(keys[column], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(converted)):
            raise ValueError(f"{table_name} key column {column!r} must be numeric")
        if len(converted) > 1 and np.any(np.diff(converted) <= 0):
            raise ValueError(
                f"{table_name} key column {column!r} must be strictly increasing"
            )
        numeric_keys[column] = converted

    event_ids = keys["event_id"].astype(str)
    if not event_ids.is_monotonic_increasing:
        raise ValueError(
            f"{table_name} event_id order must agree with canonical row order"
        )

    values = np.empty((len(keys), 3), dtype=object)
    values[:, 0] = event_ids.to_numpy()
    values[:, 1] = numeric_keys[canonical]
    values[:, 2] = numeric_keys["original_index"]
    return columns, values


def validate_lzw_theta_keys(
    theta: pd.DataFrame, reference: pd.DataFrame | None = None
) -> tuple[str, str, str] | None:
    """Validate complete, unique, ordered LZW row keys when they are supplied.

    If ``reference`` is supplied, both tables must carry complete keys and the
    three-key sequence must match exactly row for row.  The canonical row
    column may be named ``canonical_row`` or use a suffixed upstream name such
    as ``canonical_row_6104``.
    """

    if not isinstance(theta, pd.DataFrame):
        raise TypeError("theta must be a pandas DataFrame")
    theta_columns, theta_values = _validated_key_values(theta, table_name="theta")
    if reference is None:
        return theta_columns
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pandas DataFrame")

    _, reference_values = _validated_key_values(reference, table_name="reference")
    if theta_values is None or reference_values is None:
        raise ValueError(
            "theta and reference must both provide event_id, canonical_row, "
            "and original_index"
        )
    if theta_values.shape != reference_values.shape or not np.array_equal(
        theta_values, reference_values
    ):
        raise ValueError("theta and reference LZW keys are not aligned in row order")
    return theta_columns


def _as_theta_array(theta, *, allow_vector: bool) -> tuple[np.ndarray, bool]:
    if isinstance(theta, pd.DataFrame):
        missing = sorted(set(THETA_COLUMNS).difference(theta.columns))
        if missing:
            raise ValueError(f"theta table is missing columns: {missing}")
        values = theta.loc[:, THETA_COLUMNS].to_numpy(dtype=float)
        scalar = False
    else:
        values = np.asarray(theta, dtype=float)
        scalar = values.ndim == 1
        if scalar:
            if not allow_vector or values.shape != (len(THETA_COLUMNS),):
                raise ValueError(
                    f"theta must have a final dimension of {len(THETA_COLUMNS)}"
                )
            values = values[None, :]
        elif values.ndim != 2 or values.shape[1] != len(THETA_COLUMNS):
            raise ValueError(
                f"theta must have shape (n, {len(THETA_COLUMNS)})"
            )
    if np.any(~np.isfinite(values)):
        raise ValueError("theta values must be finite")
    return values, scalar


def _strictly_positive_span(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    directions = np.asarray(DEGRADATION_DIRECTIONS, dtype=float)
    spans = directions * (end - start)
    scale = np.maximum.reduce(
        [np.abs(start), np.abs(end), np.full_like(start, np.finfo(float).tiny)]
    )
    invalid = spans <= 64.0 * np.finfo(float).eps * scale
    if np.any(invalid):
        details = ", ".join(THETA_COLUMNS[index] for index in np.flatnonzero(invalid))
        raise ValueError(
            "theta endpoint signal is constant or opposes its explicit "
            f"degradation direction: {details}"
        )
    return spans


def _compress_common_knots(
    h: np.ndarray, projected_theta: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique_h, first, counts = np.unique(h, return_index=True, return_counts=True)
    theta_knots = np.empty((len(unique_h), projected_theta.shape[1]), dtype=float)
    for index, (offset, count) in enumerate(zip(first, counts)):
        theta_knots[index] = np.median(
            projected_theta[offset : offset + count], axis=0
        )
    theta_knots[0] = start
    theta_knots[-1] = end
    return unique_h, theta_knots


def _inverse_knots(
    directed_theta: np.ndarray, h_knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique_theta, first, counts = np.unique(
        directed_theta, return_index=True, return_counts=True
    )
    inverse_h = np.empty(len(unique_theta), dtype=float)
    for index, (offset, count) in enumerate(zip(first, counts)):
        inverse_h[index] = float(np.median(h_knots[offset : offset + count]))
    inverse_h[0] = 0.0
    inverse_h[-1] = 1.0
    return unique_theta, inverse_h


@dataclass(frozen=True)
class LzwHealthProgressMap:
    """Serializable bidirectional map between LZW theta and progress ``h``."""

    theta_start: tuple[float, float, float]
    theta_end: tuple[float, float, float]
    h_knots: tuple[float, ...]
    theta_knots: tuple[tuple[float, float, float], ...]
    endpoint_window: int
    theta_columns: tuple[str, str, str] = THETA_COLUMNS
    degradation_directions: tuple[int, int, int] = DEGRADATION_DIRECTIONS

    def __post_init__(self) -> None:
        if tuple(self.theta_columns) != THETA_COLUMNS:
            raise ValueError(f"theta_columns must be {THETA_COLUMNS}")
        if tuple(self.degradation_directions) != DEGRADATION_DIRECTIONS:
            raise ValueError(
                f"degradation_directions must be {DEGRADATION_DIRECTIONS}"
            )
        if not isinstance(self.endpoint_window, int) or self.endpoint_window <= 0:
            raise ValueError("endpoint_window must be a positive integer")

        start = np.asarray(self.theta_start, dtype=float)
        end = np.asarray(self.theta_end, dtype=float)
        h = np.asarray(self.h_knots, dtype=float)
        knots = np.asarray(self.theta_knots, dtype=float)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("theta endpoints must each contain three values")
        if h.ndim != 1 or len(h) < 2 or knots.shape != (len(h), 3):
            raise ValueError("serialized interpolation knots have invalid dimensions")
        if np.any(~np.isfinite(np.r_[start, end, h, knots.ravel()])):
            raise ValueError("serialized mapping values must be finite")
        if h[0] != 0.0 or h[-1] != 1.0 or np.any(np.diff(h) <= 0):
            raise ValueError("h knots must increase strictly from 0 to 1")
        _strictly_positive_span(start, end)
        directions = np.asarray(DEGRADATION_DIRECTIONS, dtype=float)
        directed_knots = knots * directions
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(
            np.max(np.abs(directed_knots), axis=0), np.finfo(float).tiny
        )
        if np.any(np.diff(directed_knots, axis=0) < -tolerance):
            raise ValueError("theta knots must follow the degradation directions")
        if not np.array_equal(knots[0], start) or not np.array_equal(knots[-1], end):
            raise ValueError("theta knot endpoints must match theta_start/theta_end")

    def theta_at(self, h):
        """Interpolate theta at scalar or array-like progress values in [0, 1]."""

        progress = np.asarray(h, dtype=float)
        if np.any(~np.isfinite(progress)) or np.any((progress < 0) | (progress > 1)):
            raise ValueError("h must contain finite values in [0, 1]")
        h_knots = np.asarray(self.h_knots, dtype=float)
        theta_knots = np.asarray(self.theta_knots, dtype=float)
        return np.stack(
            [
                np.interp(progress, h_knots, theta_knots[:, index])
                for index in range(len(THETA_COLUMNS))
            ],
            axis=-1,
        )

    def theta(self, h):
        """Alias for :meth:`theta_at`."""

        return self.theta_at(h)

    def component_progress(self, theta) -> np.ndarray:
        """Invert each monotone theta component separately."""

        values, scalar = _as_theta_array(theta, allow_vector=True)
        h_knots = np.asarray(self.h_knots, dtype=float)
        theta_knots = np.asarray(self.theta_knots, dtype=float)
        directions = np.asarray(self.degradation_directions, dtype=float)
        estimates = np.empty_like(values)
        for index, direction in enumerate(directions):
            inverse_theta, inverse_h = _inverse_knots(
                direction * theta_knots[:, index], h_knots
            )
            estimates[:, index] = np.interp(
                direction * values[:, index], inverse_theta, inverse_h
            )
        return estimates[0] if scalar else estimates

    def h_from_theta(self, theta):
        """Robustly aggregate component inversions into progress in [0, 1]."""

        values, scalar = _as_theta_array(theta, allow_vector=True)
        component_h = self.component_progress(values)
        progress = np.median(component_h, axis=1)

        start = np.asarray(self.theta_start, dtype=float)
        end = np.asarray(self.theta_end, dtype=float)
        scale = np.maximum.reduce(
            [np.abs(start), np.abs(end), np.abs(end - start)]
        )
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(
            scale, np.finfo(float).tiny
        )
        terminal_match = np.all(np.abs(values - end) <= tolerance, axis=1)
        progress = np.clip(progress, 0.0, _TERMINAL_NONMATCH_CAP)
        progress[terminal_match] = 1.0
        return float(progress[0]) if scalar else progress

    def progress(self, theta):
        """Alias for :meth:`h_from_theta`."""

        return self.h_from_theta(theta)

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation without fitted estimators."""

        return {
            "schema_version": 1,
            "theta_columns": list(self.theta_columns),
            "degradation_directions": list(self.degradation_directions),
            "endpoint_window": self.endpoint_window,
            "theta_start": list(self.theta_start),
            "theta_end": list(self.theta_end),
            "h_knots": list(self.h_knots),
            "theta_knots": [list(row) for row in self.theta_knots],
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> "LzwHealthProgressMap":
        """Reconstruct and validate a serialized progress map."""

        if int(values.get("schema_version", 1)) != 1:
            raise ValueError("unsupported LZW health-progress schema version")
        return cls(
            theta_start=tuple(float(value) for value in values["theta_start"]),
            theta_end=tuple(float(value) for value in values["theta_end"]),
            h_knots=tuple(float(value) for value in values["h_knots"]),
            theta_knots=tuple(
                tuple(float(value) for value in row)
                for row in values["theta_knots"]
            ),
            endpoint_window=int(values["endpoint_window"]),
            theta_columns=tuple(values.get("theta_columns", THETA_COLUMNS)),
            degradation_directions=tuple(
                int(value)
                for value in values.get(
                    "degradation_directions", DEGRADATION_DIRECTIONS
                )
            ),
        )


def _error_metrics(observed: np.ndarray, reconstructed: np.ndarray, span: float) -> dict:
    residual = observed - reconstructed
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "normalized_rmse": float(
            np.sqrt(np.mean(np.square(residual))) / abs(span)
        ),
    }


def fit_lzw_health_progress(
    theta: pd.DataFrame,
    *,
    endpoint_window: int = 50,
    key_reference: pd.DataFrame | None = None,
) -> tuple[LzwHealthProgressMap, np.ndarray, dict]:
    """Fit a coefficient-free monotone theta/progress map.

    Returns ``(mapping, h, diagnostics)``.  ``h`` follows the input row order,
    starts at zero, is non-decreasing, and reaches one only on the final row.
    """

    if not isinstance(theta, pd.DataFrame):
        raise TypeError("theta must be a pandas DataFrame")
    key_columns = validate_lzw_theta_keys(theta, key_reference)
    observed, _ = _as_theta_array(theta, allow_vector=False)
    count = len(observed)
    if not isinstance(endpoint_window, int) or endpoint_window <= 0:
        raise ValueError("endpoint_window must be a positive integer")
    if count < 2 or 2 * endpoint_window > count:
        raise ValueError("endpoint windows must be non-empty and non-overlapping")

    start = np.median(observed[:endpoint_window], axis=0)
    end = np.median(observed[-endpoint_window:], axis=0)
    signed_spans = _strictly_positive_span(start, end)
    directions = np.asarray(DEGRADATION_DIRECTIONS, dtype=float)
    normalized = directions * (observed - start) / signed_spans

    row_order = np.arange(count, dtype=float)
    component_h = np.empty_like(normalized)
    for index in range(len(THETA_COLUMNS)):
        component_h[:, index] = IsotonicRegression(
            increasing=True, out_of_bounds="clip"
        ).fit_transform(row_order, normalized[:, index])
        component_h[:, index] = np.clip(component_h[:, index], 0.0, 1.0)
        component_h[0, index] = 0.0
        component_h[-1, index] = 1.0

    h = np.median(component_h, axis=1)
    h = np.maximum.accumulate(np.clip(h, 0.0, 1.0))
    h[0] = 0.0
    if count > 1:
        preterminal_cap = 1.0 - 0.5 / (count - 1)
        h[:-1] = np.minimum(h[:-1], preterminal_cap)
    h[-1] = 1.0

    projected_theta = start + component_h * (end - start)
    h_knots, theta_knots = _compress_common_knots(
        h, projected_theta, start, end
    )
    mapping = LzwHealthProgressMap(
        theta_start=tuple(float(value) for value in start),
        theta_end=tuple(float(value) for value in end),
        h_knots=tuple(float(value) for value in h_knots),
        theta_knots=tuple(
            tuple(float(value) for value in row) for row in theta_knots
        ),
        endpoint_window=endpoint_window,
    )

    reconstructed = mapping.theta_at(h)
    component_diagnostics = {}
    for index, column in enumerate(THETA_COLUMNS):
        raw_row_rho = float(spearmanr(row_order, observed[:, index]).statistic)
        raw_h_rho = float(spearmanr(h, observed[:, index]).statistic)
        component_diagnostics[column] = {
            "direction": DEGRADATION_DIRECTION_LABELS[index],
            "direction_sign": DEGRADATION_DIRECTIONS[index],
            "start_endpoint": float(start[index]),
            "end_endpoint": float(end[index]),
            "signed_endpoint_span": float(signed_spans[index]),
            "spearman_vs_row": raw_row_rho,
            "spearman_vs_h": raw_h_rho,
            "degradation_aligned_spearman_vs_row": float(
                directions[index] * raw_row_rho
            ),
            "degradation_aligned_spearman_vs_h": float(
                directions[index] * raw_h_rho
            ),
            "reconstruction": _error_metrics(
                observed[:, index], reconstructed[:, index], signed_spans[index]
            ),
        }

    h_roundtrip = mapping.h_from_theta(reconstructed)
    h_residual = h - h_roundtrip
    diagnostics = {
        "method": "robust_endpoints_isotonic_components_median_aggregation",
        "row_count": count,
        "endpoint_window": endpoint_window,
        "key_columns": list(key_columns) if key_columns is not None else None,
        "components": component_diagnostics,
        "progress": {
            "minimum": float(h.min()),
            "maximum": float(h.max()),
            "is_monotone": bool(np.all(np.diff(h) >= 0.0)),
            "terminal_value_count": int(np.count_nonzero(h == 1.0)),
            "roundtrip_rmse": float(np.sqrt(np.mean(np.square(h_residual)))),
            "roundtrip_mae": float(np.mean(np.abs(h_residual))),
            "roundtrip_max_abs_error": float(np.max(np.abs(h_residual))),
        },
    }
    return mapping, h, diagnostics


__all__ = [
    "DEGRADATION_DIRECTIONS",
    "LzwHealthProgressMap",
    "THETA_COLUMNS",
    "fit_lzw_health_progress",
    "validate_lzw_theta_keys",
]
