"""Map the LZW damage proxy to fixed-condition voltage-loss thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fc_power.health.dynamic_proxy import (
    DynamicPerformanceLossProxy,
    LzwIvConditions,
)
from fc_power.health.lzw_gamma_calibration import ThetaPowerLawMap


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
CONDITIONS = ROOT / "data/upstream_lzw/current_point_cost_conditions.json"
OUTPUT = ROOT / "data/results/fc_only_physical_boundary_mapping"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
CURRENTS_A = (90.0, 120.0, 160.0, 195.0, 270.0, 340.0, 370.0)
VOLTAGE_LOSS_THRESHOLDS = (0.05, 0.10, 0.15)
CALIBRATION_SCALES = (0.80, 1.00, 1.20)


def voltage_loss_fraction(proxy, damage_pct, current_a):
    result = proxy.evaluate(damage_pct, current_a)
    healthy = float(result["healthy_cell_voltage_v"])
    current = float(result["current_cell_voltage_v"])
    return (healthy - current) / healthy


def solve_damage_boundary(proxy, reference_damage, current_a, threshold):
    upper = reference_damage
    while voltage_loss_fraction(proxy, upper, current_a) < threshold:
        upper *= 1.5
        if upper > 10 * reference_damage:
            return float("nan")
    return float(
        brentq(
            lambda damage: voltage_loss_fraction(proxy, damage, current_a)
            - threshold,
            0.0,
            upper,
        )
    )


def build_tables(proxy, mapping):
    reference = mapping.damage_reference_pct
    boundary_rows = []
    for current in CURRENTS_A:
        healthy = float(proxy.evaluate(0.0, current)["healthy_cell_voltage_v"])
        at_reference = float(
            proxy.evaluate(reference, current)["current_cell_voltage_v"]
        )
        for threshold in VOLTAGE_LOSS_THRESHOLDS:
            damage = solve_damage_boundary(proxy, reference, current, threshold)
            theta = mapping.theta_reported(damage)
            boundary_rows.append(
                {
                    "current_a": current,
                    "healthy_cell_voltage_v": healthy,
                    "calibration_endpoint_cell_voltage_v": at_reference,
                    "calibration_endpoint_voltage_loss_fraction": (
                        healthy - at_reference
                    )
                    / healthy,
                    "target_voltage_loss_fraction": threshold,
                    "inferred_damage_boundary_pct": damage,
                    "damage_boundary_over_calibration": damage / reference,
                    "within_lzw_calibration_range": damage <= reference,
                    "theta_i0": theta[0],
                    "theta_ih": theta[1],
                    "theta_R_ohm": theta[2],
                }
            )

    scale_rows = []
    for scale in CALIBRATION_SCALES:
        damage = reference * scale
        for current in CURRENTS_A:
            scale_rows.append(
                {
                    "boundary_scale": scale,
                    "damage_boundary_pct": damage,
                    "current_a": current,
                    "voltage_loss_fraction": voltage_loss_fraction(
                        proxy, damage, current
                    ),
                    "within_lzw_calibration_range": damage <= reference,
                }
            )
    return pd.DataFrame(boundary_rows), pd.DataFrame(scale_rows)


def plot_results(proxy, mapping, boundaries, scales, output_path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        195.0: "#0072B2",
        270.0: "#009E73",
        340.0: "#D55E00",
        370.0: "#CC79A7",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))
    reference = mapping.damage_reference_pct
    damage = np.linspace(0.0, 3.1 * reference, 500)
    for current, color in colors.items():
        loss = [100 * voltage_loss_fraction(proxy, value, current) for value in damage]
        axes[0].plot(
            damage / reference,
            loss,
            color=color,
            lw=1.1,
            label=f"{current:.0f} A",
        )
    axes[0].axvspan(0, 1, color="#D9E5D6", alpha=0.35, lw=0)
    axes[0].axvline(1, color="#6C757D", lw=0.8, ls="--")
    axes[0].axhline(5, color="#7A8288", lw=0.8, ls=":")
    axes[0].axhline(10, color="#7A8288", lw=0.8, ls="--")
    axes[0].set_xlabel("Damage / LZW calibration endpoint")
    axes[0].set_ylabel("Fixed-condition voltage loss (%)")
    axes[0].legend(frameon=False, fontsize=6.2, ncol=2)

    threshold_style = {
        0.05: ("#0072B2", "o", "5% loss"),
        0.10: ("#D55E00", "s", "10% loss"),
        0.15: ("#CC79A7", "^", "15% loss"),
    }
    axes[1].axhspan(0, 1, color="#D9E5D6", alpha=0.35, lw=0)
    for threshold, (color, marker, label) in threshold_style.items():
        selected = boundaries[boundaries.target_voltage_loss_fraction.eq(threshold)]
        axes[1].plot(
            selected.current_a,
            selected.damage_boundary_over_calibration,
            color=color,
            marker=marker,
            ms=3.4,
            lw=1.0,
            label=label,
        )
    axes[1].axhline(1, color="#6C757D", lw=0.8, ls="--")
    axes[1].set_xlabel("Diagnostic current (A)")
    axes[1].set_ylabel("Damage threshold / calibration endpoint")
    axes[1].legend(frameon=False, fontsize=6.2)

    scale_colors = {0.8: "#56B4E9", 1.0: "#0072B2", 1.2: "#D55E00"}
    for scale, color in scale_colors.items():
        selected = scales[scales.boundary_scale.eq(scale)]
        axes[2].plot(
            selected.current_a,
            100 * selected.voltage_loss_fraction,
            color=color,
            marker="o",
            ms=3.0,
            lw=1.0,
            label=f"Boundary x{scale:.1f}",
        )
    axes[2].axhline(5, color="#7A8288", lw=0.8, ls=":")
    axes[2].axhline(10, color="#7A8288", lw=0.8, ls="--")
    axes[2].set_xlabel("Diagnostic current (A)")
    axes[2].set_ylabel("Voltage loss at boundary (%)")
    axes[2].legend(frameon=False, fontsize=6.2)

    for index, axis in enumerate(axes):
        axis.text(
            0.0,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.65, w_pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    conditions = json.loads(CONDITIONS.read_text(encoding="utf-8"))
    mapping = ThetaPowerLawMap.from_dict(calibration["theta_power_law_map"])
    proxy = DynamicPerformanceLossProxy(
        mapping,
        LzwIvConditions.from_upstream_dict(conditions),
        normalization_power_loss_w=1.0,
    )
    boundaries, scales = build_tables(proxy, mapping)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    boundaries.to_csv(args.out_dir / "voltage_loss_boundary_mapping.csv", index=False)
    scales.to_csv(args.out_dir / "calibration_boundary_voltage_loss.csv", index=False)
    figure = args.out_dir / "fig27_voltage_loss_boundary_mapping.png"
    plot_results(proxy, mapping, boundaries, scales, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())

    selected = boundaries[
        boundaries.current_a.isin((195.0, 370.0))
        & boundaries.target_voltage_loss_fraction.isin((0.05, 0.10))
    ]
    metadata = {
        "scope": "diagnostic mapping from LZW damage proxy to voltage-loss thresholds",
        "calibration_damage_endpoint_pct": mapping.damage_reference_pct,
        "diagnostic_currents_a": list(CURRENTS_A),
        "voltage_loss_thresholds": list(VOLTAGE_LOSS_THRESHOLDS),
        "calibration_boundary_scales": list(CALIBRATION_SCALES),
        "extrapolation_warning": (
            "Every 5%, 10%, and 15% voltage-loss threshold lies beyond the "
            "observed LZW theta trajectory and is only a stress scenario."
        ),
        "physical_eol_claimed": False,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report = f"""# Voltage-loss boundary mapping audit

- The LZW calibration endpoint is `{mapping.damage_reference_pct:.6f}%` damage.
- At that endpoint, fixed-condition voltage loss ranges from
  `{100 * boundaries.calibration_endpoint_voltage_loss_fraction.min():.3f}%` to
  `{100 * boundaries.calibration_endpoint_voltage_loss_fraction.max():.3f}%`
  over the audited current points.
- Every 5%, 10%, and 15% voltage-loss threshold lies outside the observed LZW
  theta trajectory. These mappings are extrapolation stress scenarios, not
  identified physical EOL thresholds.

## Selected mappings

{selected.to_markdown(index=False)}
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
