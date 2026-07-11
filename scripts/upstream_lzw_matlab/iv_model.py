"""Faithful Python port of the local IV_model in UKF_PF.m."""
from __future__ import annotations
import numpy as np

R_GAS = 8.314
FARADAY = 9.64853399e4
ACTIVE_AREA_CM2 = 406.0


def reported_to_model_theta(theta_reported):
    """Adapt plotted/report-table theta to the state expected by UKF_PF IV_model.

    Report table: i0 stored near 1.9 (display factor 1e7), ih near .002,
    R near .04. The embedded model multiplies x3 by current density and 406;
    therefore R_reported/406 is used as the model state. This R mapping is an
    audited inference, not an explicit conversion found in the MATLAB files.
    """
    t = np.asarray(theta_reported, dtype=float)
    out = t.copy()
    out[..., 0] *= 1e-7
    out[..., 2] /= ACTIVE_AREA_CM2
    return out


def iv_model(temperature_c, theta_model, current_density_a_cm2, a, b, inner,
             *, clip_output=True):
    """Evaluate the UKF_PF.m nested IV_model.

    theta_model[..., :] = [i0, ih, R_state]; inner=[B, i_lim].
    Returns cell voltage in V. Broadcasting is supported.
    """
    x = np.maximum(np.asarray(theta_model, dtype=float), 1e-12)
    i = np.maximum(np.asarray(current_density_a_cm2, dtype=float), 1e-12)
    T = np.asarray(temperature_c, dtype=float) + 273.15
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    inner = np.asarray(inner, dtype=float)
    e_act = (R_GAS*T)/FARADAY * np.arcsinh((x[...,1]+i)/(2*x[...,0]))
    e_om = i*x[...,2]*ACTIVE_AREA_CM2
    denominator = inner[...,1] - i - x[...,1]
    safe_denominator = np.where(denominator <= 0, 1e-12, denominator)
    e_con = inner[...,0] * np.log(inner[...,1]/safe_denominator)
    E = 1.229 - .85e-3*(T-298.15) + 4.3085e-5*T*(np.log(a)+.5*np.log(b))
    out = E-e_act-e_om-e_con
    return np.clip(out, 0, 2) if clip_output else out

