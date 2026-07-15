"""Audit and normalize Chen Peng's three-stack efficiency curves."""

from __future__ import annotations

import pandas as pd


ACTIVE_AREA_CM2 = 406.0
FARADAY_CONSTANT_C_PER_MOL = 96_485.0
HYDROGEN_MOLAR_MASS_KG_PER_MOL = 2.02e-3
LHV_J_PER_KG = 120.0e6
LHV_KJ_PER_MOL = 241.98
HHV_KJ_PER_MOL = 286.02

SOURCE_COLUMNS = (
    "stack_id",
    "cell_count",
    "current_density_a_cm2",
    "gross_stack_power_kw",
    "efficiency_lhv_pct",
)


def audit_chen_efficiency_curves(source: pd.DataFrame) -> pd.DataFrame:
    """Convert the Origin snapshot to a consistent net-power/LHV basis.

    Chen's MATLAB code returns gross electrochemical stack power on the curve
    x-axis, while its efficiency numerator subtracts compressor power.  The
    hydrogen-flow equation and 120 MJ/kg denominator make the stored efficiency
    LHV-based.  This function reconstructs net power from that same definition.
    """

    missing = set(SOURCE_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"missing Chen curve columns: {sorted(missing)}")

    frame = source.loc[:, SOURCE_COLUMNS].copy()
    numeric_columns = SOURCE_COLUMNS[1:]
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if frame.duplicated(["stack_id", "current_density_a_cm2"]).any():
        raise ValueError("stack/current-density samples must be unique")
    if (frame.loc[:, numeric_columns] <= 0).any().any():
        raise ValueError("Chen curve inputs must be positive")

    frame = frame.sort_values(
        ["stack_id", "current_density_a_cm2"],
        kind="stable",
    ).reset_index(drop=True)
    frame["stack_current_a"] = (
        frame["current_density_a_cm2"] * ACTIVE_AREA_CM2
    )
    frame["hydrogen_flow_kg_s"] = (
        frame["stack_current_a"]
        * HYDROGEN_MOLAR_MASS_KG_PER_MOL
        * frame["cell_count"]
        / (2.0 * FARADAY_CONSTANT_C_PER_MOL)
    )
    frame["chemical_input_lhv_kw"] = (
        frame["hydrogen_flow_kg_s"] * LHV_J_PER_KG / 1_000.0
    )
    frame["net_system_power_kw"] = (
        frame["chemical_input_lhv_kw"] * frame["efficiency_lhv_pct"] / 100.0
    )
    frame["auxiliary_power_kw"] = (
        frame["gross_stack_power_kw"] - frame["net_system_power_kw"]
    )
    frame["efficiency_hhv_pct"] = (
        frame["efficiency_lhv_pct"] * LHV_KJ_PER_MOL / HHV_KJ_PER_MOL
    )

    if (frame["auxiliary_power_kw"] < 0).any():
        raise ValueError("reconstructed net power cannot exceed gross stack power")
    if not frame.groupby("stack_id")["current_density_a_cm2"].apply(
        lambda values: values.is_monotonic_increasing
    ).all():
        raise ValueError("current-density samples must increase within each stack")
    return frame


def summarize_chen_efficiency_curves(frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    """Return one traceable domain and peak-efficiency record per stack."""

    summaries: list[dict[str, float | int | str]] = []
    for stack_id, group in frame.groupby("stack_id", sort=True):
        peak = group.loc[group["efficiency_lhv_pct"].idxmax()]
        summaries.append(
            {
                "stack_id": str(stack_id),
                "cell_count": int(group["cell_count"].iloc[0]),
                "samples": int(len(group)),
                "gross_power_min_kw": float(group["gross_stack_power_kw"].min()),
                "gross_power_max_kw": float(group["gross_stack_power_kw"].max()),
                "net_power_min_kw": float(group["net_system_power_kw"].min()),
                "net_power_max_kw": float(group["net_system_power_kw"].max()),
                "auxiliary_power_min_kw": float(group["auxiliary_power_kw"].min()),
                "auxiliary_power_max_kw": float(group["auxiliary_power_kw"].max()),
                "peak_efficiency_lhv_pct": float(peak["efficiency_lhv_pct"]),
                "peak_efficiency_current_density_a_cm2": float(
                    peak["current_density_a_cm2"]
                ),
                "peak_efficiency_gross_power_kw": float(
                    peak["gross_stack_power_kw"]
                ),
                "peak_efficiency_net_power_kw": float(peak["net_system_power_kw"]),
            }
        )
    return summaries
