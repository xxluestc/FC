"""Event-conditioned residual correction and conformal forecast intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbabilisticForecast:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    event_code: np.ndarray


class EventConditionedResidualConformal:
    """Calibrate a point forecaster without leaking test observations.

    The first chronological half of the calibration set trains a compact
    event-aware residual correction.  The second half estimates group-wise
    residual quantiles, leaving the held-out test sequence untouched.
    """

    def __init__(
        self,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        ridge_alpha: float = 10.0,
        minimum_group_samples: int = 100,
    ):
        if not 0 < lower_quantile < upper_quantile < 1:
            raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and non-negative")
        if minimum_group_samples <= 0:
            raise ValueError("minimum_group_samples must be positive")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.ridge_alpha = ridge_alpha
        self.minimum_group_samples = minimum_group_samples

    def fit(self, point_prediction, actual, brake_probability, high_probability):
        point, actual, brake, high = self._validate(
            point_prediction, actual, brake_probability, high_probability
        )
        if len(point) < 2 * self.minimum_group_samples:
            raise ValueError("calibration set is too small for chronological splitting")
        split = len(point) // 2
        self.correction_model_ = make_pipeline(
            StandardScaler(), Ridge(alpha=self.ridge_alpha)
        )
        self.correction_model_.fit(
            self._correction_features(point[:split], brake[:split], high[:split]),
            actual[:split] - point[:split],
        )
        corrected = point[split:] + self.correction_model_.predict(
            self._correction_features(point[split:], brake[split:], high[split:])
        )
        residual = actual[split:] - corrected
        codes = self.event_codes(brake[split:], high[split:])
        self.global_quantiles_ = self._quantiles(residual)
        self.global_residuals_ = residual.copy()
        self.group_quantiles_ = {}
        self.group_residuals_ = {}
        self.group_counts_ = {}
        for code in range(4):
            selected = codes == code
            self.group_counts_[code] = int(selected.sum())
            if selected.sum() >= self.minimum_group_samples:
                self.group_quantiles_[code] = self._quantiles(residual[selected])
                self.group_residuals_[code] = residual[selected].copy()
        self.calibration_split_ = split
        self.horizon_ = point.shape[1]
        return self

    def predict(self, point_prediction, brake_probability, high_probability):
        if not hasattr(self, "correction_model_"):
            raise RuntimeError("fit must be called before predict")
        point = np.asarray(point_prediction, dtype=float)
        brake = np.asarray(brake_probability, dtype=float)
        high = np.asarray(high_probability, dtype=float)
        if point.ndim != 2 or point.shape[1] != self.horizon_:
            raise ValueError("point_prediction has an incompatible horizon")
        if brake.shape != (len(point),) or high.shape != (len(point),):
            raise ValueError("event probability vectors must align with predictions")
        if np.any(~np.isfinite(point)) or np.any(~np.isfinite(brake)) or np.any(~np.isfinite(high)):
            raise ValueError("forecast inputs must be finite")
        if np.any((brake < 0) | (brake > 1) | (high < 0) | (high > 1)):
            raise ValueError("event probabilities must lie in [0, 1]")

        center = point + self.correction_model_.predict(
            self._correction_features(point, brake, high)
        )
        codes = self.event_codes(brake, high)
        lower = np.empty_like(center)
        upper = np.empty_like(center)
        for code in range(4):
            selected = codes == code
            quantiles = self.group_quantiles_.get(code, self.global_quantiles_)
            lower[selected] = center[selected] + quantiles[0]
            upper[selected] = center[selected] + quantiles[1]
        lower = np.minimum(lower, center)
        upper = np.maximum(upper, center)
        return ProbabilisticForecast(center, lower, upper, codes)

    def predict_adaptive(
        self,
        point_prediction,
        actual,
        brake_probability,
        high_probability,
        *,
        delay_steps: int | None = None,
        rolling_window: int = 2000,
    ):
        """Produce causal online intervals updated only after outcomes arrive.

        For a horizon ``H``, the residual from origin ``i-H`` is added before
        forecasting origin ``i``.  Thus no component of the current or future
        target window is used to construct its own interval.
        """

        fixed = self.predict(point_prediction, brake_probability, high_probability)
        actual = np.asarray(actual, dtype=float)
        if actual.shape != fixed.center.shape or np.any(~np.isfinite(actual)):
            raise ValueError("actual must be a finite matrix aligned with predictions")
        delay = self.horizon_ if delay_steps is None else int(delay_steps)
        if delay <= 0:
            raise ValueError("delay_steps must be positive")
        if rolling_window < self.minimum_group_samples:
            raise ValueError("rolling_window is smaller than minimum_group_samples")

        global_pool = self.global_residuals_[-rolling_window:].copy()
        group_pools = {
            code: residual[-rolling_window:].copy()
            for code, residual in self.group_residuals_.items()
        }
        lower = np.empty_like(fixed.center)
        upper = np.empty_like(fixed.center)
        for index in range(len(fixed.center)):
            observed_index = index - delay
            if observed_index >= 0:
                residual = (
                    actual[observed_index] - fixed.center[observed_index]
                )[None, :]
                global_pool = np.r_[global_pool, residual][-rolling_window:]
                observed_code = int(fixed.event_code[observed_index])
                previous = group_pools.get(
                    observed_code, np.empty((0, self.horizon_))
                )
                group_pools[observed_code] = np.r_[previous, residual][
                    -rolling_window:
                ]
            code = int(fixed.event_code[index])
            pool = group_pools.get(code, global_pool)
            if len(pool) < self.minimum_group_samples:
                pool = global_pool
            quantiles = self._quantiles(pool)
            lower[index] = fixed.center[index] + quantiles[0]
            upper[index] = fixed.center[index] + quantiles[1]
        lower = np.minimum(lower, fixed.center)
        upper = np.maximum(upper, fixed.center)
        return ProbabilisticForecast(
            fixed.center, lower, upper, fixed.event_code
        )

    @staticmethod
    def event_codes(brake_probability, high_probability):
        brake = np.asarray(brake_probability) >= 0.5
        high = np.asarray(high_probability) >= 0.5
        return brake.astype(int) + 2 * high.astype(int)

    def _quantiles(self, residual):
        return (
            np.quantile(residual, self.lower_quantile, axis=0),
            np.quantile(residual, self.upper_quantile, axis=0),
        )

    @staticmethod
    def _correction_features(point, brake, high):
        return np.c_[
            point,
            brake,
            high,
            point * brake[:, None],
            point * high[:, None],
        ]

    @staticmethod
    def _validate(point_prediction, actual, brake_probability, high_probability):
        point = np.asarray(point_prediction, dtype=float)
        actual = np.asarray(actual, dtype=float)
        brake = np.asarray(brake_probability, dtype=float)
        high = np.asarray(high_probability, dtype=float)
        if point.ndim != 2 or actual.shape != point.shape:
            raise ValueError("point_prediction and actual must be aligned matrices")
        if brake.shape != (len(point),) or high.shape != (len(point),):
            raise ValueError("event probability vectors must align with predictions")
        arrays = (point, actual, brake, high)
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("calibration arrays must be finite")
        if np.any((brake < 0) | (brake > 1) | (high < 0) | (high > 1)):
            raise ValueError("event probabilities must lie in [0, 1]")
        return point, actual, brake, high
