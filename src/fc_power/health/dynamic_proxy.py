"""Health-dependent performance-loss proxy for candidate FC currents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fc_power.health.lzw_gamma_calibration import ThetaPowerLawMap
from fc_power.lzw_iv_model import iv_model, reported_to_model_theta


@dataclass(frozen=True)
class LzwIvConditions:
    temperature_c: float
    a: float
    b: float
    concentration_b: float
    limiting_current_a_cm2: float
    active_area_cm2: float = 406.0
    stack_cells: int = 170

    @classmethod
    def from_upstream_dict(cls, values: dict):
        return cls(
            temperature_c=float(values["T_ref_C"]),
            a=float(values["a_ref"]),
            b=float(values["b_ref"]),
            concentration_b=float(values["B"]),
            limiting_current_a_cm2=float(values["i_lim_A_cm2"]),
            active_area_cm2=float(values["active_area_cm2"]),
            stack_cells=int(values["stack_cells"]),
        )


class DynamicPerformanceLossProxy:
    """Evaluate ``C_deg(I | theta(damage))`` at the current health state."""

    def __init__(
        self,
        mapping: ThetaPowerLawMap,
        conditions: LzwIvConditions,
        normalization_power_loss_w: float,
    ):
        if not np.isfinite(normalization_power_loss_w) or normalization_power_loss_w <= 0:
            raise ValueError("normalization_power_loss_w must be finite and positive")
        self.mapping = mapping
        self.conditions = conditions
        self.normalization_power_loss_w = float(normalization_power_loss_w)
        self.healthy_theta_reported = np.asarray(mapping.theta_start, dtype=float)

    def evaluate(self, damage_pct, current_a, dt_s=1.0) -> dict[str, np.ndarray]:
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        current = np.asarray(current_a, dtype=float)
        if np.any(~np.isfinite(current)) or np.any(current < 0):
            raise ValueError("current_a must be finite and non-negative")

        theta_reported = np.asarray(self.mapping.theta_reported(damage_pct), dtype=float)
        if theta_reported.ndim != 1:
            raise ValueError("evaluate expects one scalar health state")
        theta_model = reported_to_model_theta(
            theta_reported, self.conditions.active_area_cm2
        )
        healthy_model = reported_to_model_theta(
            self.healthy_theta_reported, self.conditions.active_area_cm2
        )
        density = current / self.conditions.active_area_cm2
        model_kwargs = {
            "temperature_c": self.conditions.temperature_c,
            "current_density_a_cm2": density,
            "a": self.conditions.a,
            "b": self.conditions.b,
            "inner": [
                self.conditions.concentration_b,
                self.conditions.limiting_current_a_cm2,
            ],
            "active_area_cm2": self.conditions.active_area_cm2,
        }
        healthy_voltage = iv_model(theta_model=healthy_model, **model_kwargs)
        current_voltage = iv_model(theta_model=theta_model, **model_kwargs)
        voltage_loss = np.maximum(healthy_voltage - current_voltage, 0.0)
        power_loss_w = voltage_loss * current * self.conditions.stack_cells
        energy_loss_wh = power_loss_w * dt_s / 3600.0
        normalized = power_loss_w / self.normalization_power_loss_w
        stack_power_kw = (
            current * current_voltage * self.conditions.stack_cells / 1000.0
        )
        return {
            "current_a": current,
            "theta_reported": theta_reported,
            "healthy_cell_voltage_v": healthy_voltage,
            "current_cell_voltage_v": current_voltage,
            "voltage_loss_v_per_cell": voltage_loss,
            "equivalent_power_loss_w": power_loss_w,
            "equivalent_energy_loss_wh": energy_loss_wh,
            "normalized_proxy": normalized,
            "stack_power_kw": stack_power_kw,
        }
