"""Calibrate a literature-structured Gamma health state to LZW theta data.

The Ghaderi/Pei coefficients define relative action exposure in percent-like
damage units.  They are not re-identified from the LZW trajectory.  Instead,
the recorded cumulative exposure is evaluated with those fixed coefficients,
then a compact monotone map links the resulting damage index to the observed
UKF-PF health parameters ``[i0, ih, R_ohm]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from fc_power.health.gamma_process import GammaHealthParams, LoadRateMap


CURRENT_LEVELS_A = (0.0, 25.0, 90.0, 120.0, 160.0, 195.0, 270.0, 370.0)
TIME_COLUMNS = tuple(f"time_{int(current)}A_cum" for current in CURRENT_LEVELS_A)
THETA_COLUMNS = (
    "i0_A_per_cm2",
    "ih_A_per_cm2",
    "R_ohm_reported_ohm_cm2",
)


@dataclass(frozen=True)
class GhaderiPeiCoefficients:
    """Fixed PEMFC degradation coefficients reported by Ghaderi/Pei.

    Values are percentage degradation per cycle or per hour.  They specify
    the mean action-exposure structure only; they are not claimed to be
    directly identified for the LZW stack.
    """

    start_stop_pct_per_cycle: float = 0.00196
    high_load_pct_per_hour: float = 0.001470
    low_load_pct_per_hour: float = 0.00126
    natural_on_pct_per_hour: float = 0.002
    load_shift_pct_per_cycle: float = 5.93e-5


@dataclass(frozen=True)
class ThetaPowerLawMap:
    """Monotone map from cumulative damage exposure to LZW theta."""

    damage_reference_pct: float
    theta_start: tuple[float, float, float]
    theta_end: tuple[float, float, float]
    exponents: tuple[float, float, float]
    active_area_cm2: float = 406.0

    def theta_reported(self, damage_pct):
        damage = np.maximum(np.asarray(damage_pct, dtype=float), 0.0)
        fraction = damage / self.damage_reference_pct
        start = np.asarray(self.theta_start)
        end = np.asarray(self.theta_end)
        exponent = np.asarray(self.exponents)
        return start + (end - start) * np.power(fraction[..., None], exponent)

    def theta_model(self, damage_pct):
        theta = np.asarray(self.theta_reported(damage_pct)).copy()
        theta[..., 2] /= self.active_area_cm2
        return theta

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict):
        return cls(
            damage_reference_pct=float(values["damage_reference_pct"]),
            theta_start=tuple(float(value) for value in values["theta_start"]),
            theta_end=tuple(float(value) for value in values["theta_end"]),
            exponents=tuple(float(value) for value in values["exponents"]),
            active_area_cm2=float(values.get("active_area_cm2", 406.0)),
        )


def validate_lzw_alignment(events: pd.DataFrame, theta: pd.DataFrame) -> None:
    if len(events) != len(theta):
        raise ValueError("event and theta tables must have equal row counts")
    required_events = {
        "canonical_row_6104",
        "qt_num_cum",
        "bz_num_cum",
        *TIME_COLUMNS,
    }
    required_theta = {"canonical_row_6104", *THETA_COLUMNS}
    missing_events = required_events.difference(events.columns)
    missing_theta = required_theta.difference(theta.columns)
    if missing_events or missing_theta:
        raise ValueError(
            f"missing columns: events={sorted(missing_events)}, theta={sorted(missing_theta)}"
        )
    if not np.array_equal(
        events["canonical_row_6104"].to_numpy(),
        theta["canonical_row_6104"].to_numpy(),
    ):
        raise ValueError("event and theta canonical rows are not aligned")


def cumulative_damage_components(
    events: pd.DataFrame,
    coefficients: GhaderiPeiCoefficients = GhaderiPeiCoefficients(),
) -> pd.DataFrame:
    """Evaluate fixed literature coefficients on recorded cumulative exposure."""

    missing = {"qt_num_cum", "bz_num_cum", *TIME_COLUMNS}.difference(events.columns)
    if missing:
        raise ValueError(f"event table is missing columns: {sorted(missing)}")

    operating_columns = [column for column in TIME_COLUMNS if column != "time_0A_cum"]
    operating_h = events[operating_columns].sum(axis=1).to_numpy(dtype=float) / 3600.0
    low_h = events["time_0A_cum"].to_numpy(dtype=float) / 3600.0
    high_h = events["time_370A_cum"].to_numpy(dtype=float) / 3600.0

    frame = pd.DataFrame(
        {
            "start_stop_damage_pct": coefficients.start_stop_pct_per_cycle
            * events["qt_num_cum"].to_numpy(dtype=float),
            "high_load_damage_pct": coefficients.high_load_pct_per_hour * high_h,
            "low_load_damage_pct": coefficients.low_load_pct_per_hour * low_h,
            "natural_on_damage_pct": coefficients.natural_on_pct_per_hour
            * operating_h,
            "load_shift_damage_pct": coefficients.load_shift_pct_per_cycle
            * events["bz_num_cum"].to_numpy(dtype=float),
        }
    )
    frame -= frame.iloc[0]
    frame["continuous_damage_pct"] = frame[
        ["high_load_damage_pct", "low_load_damage_pct", "natural_on_damage_pct"]
    ].sum(axis=1)
    frame["discrete_event_damage_pct"] = frame[
        ["start_stop_damage_pct", "load_shift_damage_pct"]
    ].sum(axis=1)
    frame["total_damage_pct"] = (
        frame["continuous_damage_pct"] + frame["discrete_event_damage_pct"]
    )
    if (frame.diff().iloc[1:] < -1e-12).any().any():
        raise ValueError("cumulative damage components must be non-decreasing")
    return frame


def _fit_exponent(z: np.ndarray, observed: np.ndarray, start: float, end: float):
    def predict(exponent: float):
        return start + (end - start) * np.power(z, exponent)

    result = minimize_scalar(
        lambda exponent: np.mean(np.square(observed - predict(exponent))),
        bounds=(0.1, 6.0),
        method="bounded",
    )
    fitted = predict(float(result.x))
    residual = observed - fitted
    denominator = np.sum(np.square(observed - observed.mean()))
    r2 = 1.0 - np.sum(np.square(residual)) / denominator
    return float(result.x), fitted, {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "r2": float(r2),
    }


def fit_theta_power_law(
    damage_pct,
    theta: pd.DataFrame,
    endpoint_events: int = 50,
    active_area_cm2: float = 406.0,
):
    """Fit one monotone power exponent for each theta component."""

    damage = np.asarray(damage_pct, dtype=float)
    if damage.ndim != 1 or len(damage) != len(theta):
        raise ValueError("damage and theta must be aligned one-dimensional trajectories")
    if np.any(np.diff(damage) < -1e-12) or damage[-1] <= 0:
        raise ValueError("damage trajectory must be non-decreasing with a positive endpoint")
    if endpoint_events <= 0 or 2 * endpoint_events > len(theta):
        raise ValueError("invalid endpoint_events")

    observed = theta.loc[:, THETA_COLUMNS].to_numpy(dtype=float)
    start = observed[:endpoint_events].mean(axis=0)
    end = observed[-endpoint_events:].mean(axis=0)
    z = np.clip(damage / damage[-1], 0.0, 1.0)

    exponents = []
    fitted_columns = []
    metrics = {}
    for index, column in enumerate(THETA_COLUMNS):
        exponent, fitted, column_metrics = _fit_exponent(
            z, observed[:, index], start[index], end[index]
        )
        exponents.append(exponent)
        fitted_columns.append(fitted)
        metrics[column] = {"exponent": exponent, **column_metrics}

    mapping = ThetaPowerLawMap(
        damage_reference_pct=float(damage[-1]),
        theta_start=tuple(float(value) for value in start),
        theta_end=tuple(float(value) for value in end),
        exponents=tuple(exponents),
        active_area_cm2=float(active_area_cm2),
    )
    return mapping, np.column_stack(fitted_columns), metrics


def gamma_scale_for_terminal_cv(
    terminal_total_damage_pct: float,
    terminal_continuous_damage_pct: float,
    terminal_cv: float,
) -> float:
    """Choose Gamma scale so total terminal damage has a target CV.

    Discrete start/shift increments remain deterministic.  If the continuous
    Gamma contribution has mean ``mu`` and scale ``beta``, its variance is
    ``mu*beta``.  Therefore ``beta=(cv*total_mean)^2/mu``.
    """

    values = (terminal_total_damage_pct, terminal_continuous_damage_pct, terminal_cv)
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("damage levels and terminal_cv must be finite and positive")
    return float(
        np.square(terminal_cv * terminal_total_damage_pct)
        / terminal_continuous_damage_pct
    )


def ghaderi_gamma_params(
    gamma_scale: float,
    coefficients: GhaderiPeiCoefficients = GhaderiPeiCoefficients(),
    heterogeneity_factor: float = 1.0,
) -> GammaHealthParams:
    """Build candidate-action parameters in literature damage-percent units.

    Zero current while ``next_on=True`` represents energized idle/OCV.  Zero
    current with ``next_on=False`` represents a fully stopped stack and uses
    ``off_rate_per_hour=0``.  This distinction is made at transition time.
    """

    rates = []
    for current in CURRENT_LEVELS_A:
        rate = coefficients.natural_on_pct_per_hour
        if current == 0:
            rate += coefficients.low_load_pct_per_hour
        if current == max(CURRENT_LEVELS_A):
            rate += coefficients.high_load_pct_per_hour
        rates.append(rate)
    return GammaHealthParams(
        load_rate_map=LoadRateMap(CURRENT_LEVELS_A, tuple(rates)),
        gamma_scale=gamma_scale,
        off_rate_per_hour=0.0,
        shift_increment=coefficients.load_shift_pct_per_cycle,
        start_increment=coefficients.start_stop_pct_per_cycle,
        stop_increment=0.0,
        heterogeneity_factor=heterogeneity_factor,
    )


def monte_carlo_recorded_exposure(
    components: pd.DataFrame,
    mapping: ThetaPowerLawMap,
    gamma_scale: float,
    samples: int = 512,
    seed: int = 2026,
):
    """Simulate Gamma uncertainty along the already recorded exposure path."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    continuous = components["continuous_damage_pct"].to_numpy(dtype=float)
    discrete = components["discrete_event_damage_pct"].to_numpy(dtype=float)
    means = np.diff(np.r_[0.0, continuous])
    if np.any(means < -1e-12):
        raise ValueError("continuous damage must be non-decreasing")

    rng = np.random.default_rng(seed)
    increments = np.zeros((samples, len(means)), dtype=float)
    positive = means > 0
    increments[:, positive] = rng.gamma(
        shape=means[positive] / gamma_scale,
        scale=gamma_scale,
        size=(samples, int(positive.sum())),
    )
    damage = np.cumsum(increments, axis=1) + discrete[None, :]
    theta = mapping.theta_reported(damage)
    return damage, theta
