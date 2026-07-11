"""Package-level port of the LZW MATLAB ``IV_model``.

The source copy remains under ``scripts/upstream_lzw_matlab/iv_model.py`` for
provenance.  This module exposes the same equations to controllers without
making runtime code depend on the scripts directory.
"""

from __future__ import annotations

import numpy as np


R_GAS = 8.314
FARADAY = 9.64853399e4
ACTIVE_AREA_CM2 = 406.0


def reported_to_model_theta(theta_reported, active_area_cm2=ACTIVE_AREA_CM2):
    """Convert reported ``[i0, ih, R]`` to the embedded IV-model state."""

    theta = np.asarray(theta_reported, dtype=float)
    output = theta.copy()
    output[..., 2] /= active_area_cm2
    return output


def iv_model(
    temperature_c,
    theta_model,
    current_density_a_cm2,
    a,
    b,
    inner,
    *,
    active_area_cm2=ACTIVE_AREA_CM2,
    clip_output=True,
):
    """Evaluate the UKF-PF embedded cell-voltage equation."""

    state = np.maximum(np.asarray(theta_model, dtype=float), 1e-12)
    current_density = np.maximum(
        np.asarray(current_density_a_cm2, dtype=float), 1e-12
    )
    temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    inner = np.asarray(inner, dtype=float)

    activation = (R_GAS * temperature_k) / FARADAY * np.arcsinh(
        (state[..., 1] + current_density) / (2 * state[..., 0])
    )
    ohmic = current_density * state[..., 2] * active_area_cm2
    denominator = inner[..., 1] - current_density - state[..., 1]
    safe_denominator = np.where(denominator <= 0, 1e-12, denominator)
    concentration = inner[..., 0] * np.log(inner[..., 1] / safe_denominator)
    reversible = (
        1.229
        - 0.85e-3 * (temperature_k - 298.15)
        + 4.3085e-5
        * temperature_k
        * (np.log(a) + 0.5 * np.log(b))
    )
    voltage = reversible - activation - ohmic - concentration
    return np.clip(voltage, 0, 2) if clip_output else voltage
