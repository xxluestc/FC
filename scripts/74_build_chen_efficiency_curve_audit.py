"""Build the audited Chen efficiency-curve dataset and provenance summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fc_power.power_allocation.chen_efficiency_curves import (
    ACTIVE_AREA_CM2,
    FARADAY_CONSTANT_C_PER_MOL,
    HHV_KJ_PER_MOL,
    HYDROGEN_MOLAR_MASS_KG_PER_MOL,
    LHV_J_PER_KG,
    LHV_KJ_PER_MOL,
    audit_chen_efficiency_curves,
    summarize_chen_efficiency_curves,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/upstream_chen/chen_efficiency_curves_origin_sheet5.csv"
OUTPUT = ROOT / "data/processed/chen_efficiency_curves_audited.csv"
SUMMARY = ROOT / "data/processed/chen_efficiency_curves_audit.json"
FIGURE = (
    ROOT
    / "data/results/chen_efficiency_curve_audit/fig35_chen_curve_basis_audit.png"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_curve_basis_audit(frame: pd.DataFrame, output_path: Path) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = ("270 cells", "300 cells", "330 cells")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    lower = float(frame[["gross_stack_power_kw", "net_system_power_kw"]].min().min())
    upper = float(frame[["gross_stack_power_kw", "net_system_power_kw"]].max().max())
    axes[0].plot(
        [lower, upper],
        [lower, upper],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Gross = net",
    )
    for (_, group), color, label in zip(
        frame.groupby("stack_id", sort=True), colors, labels
    ):
        axes[0].plot(
            group["gross_stack_power_kw"],
            group["net_system_power_kw"],
            marker="o",
            markersize=2.8,
            color=color,
            label=label,
        )
        axes[1].plot(
            group["net_system_power_kw"],
            group["efficiency_lhv_pct"],
            marker="o",
            markersize=2.8,
            color=color,
            label=label,
        )

    axes[0].set(
        xlabel="Gross stack power (kW)",
        ylabel="Reconstructed net power (kW)",
    )
    axes[0].set_title("(a) Power-basis correction", loc="left")
    axes[1].set(
        xlabel="Net system power (kW)",
        ylabel="System efficiency, LHV (%)",
    )
    axes[1].set_title("(b) Audited efficiency curves", loc="left")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    source = pd.read_csv(SOURCE)
    audited = audit_chen_efficiency_curves(source)
    summary = {
        "decision_basis": {
            "allocation_power_axis": "net_system_power_kw",
            "efficiency_basis": "LHV",
            "off_state_is_separate": True,
            "sampled_domains_are_physical_limits": False,
        },
        "constants_reproducing_chen_matlab": {
            "active_area_cm2": ACTIVE_AREA_CM2,
            "faraday_constant_c_per_mol": FARADAY_CONSTANT_C_PER_MOL,
            "hydrogen_molar_mass_kg_per_mol": HYDROGEN_MOLAR_MASS_KG_PER_MOL,
            "lhv_j_per_kg": LHV_J_PER_KG,
            "lhv_kj_per_mol": LHV_KJ_PER_MOL,
            "hhv_kj_per_mol": HHV_KJ_PER_MOL,
        },
        "source": {
            "repository_snapshot": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "origin_project": (
                "G:/大论文/2025陈鹏/论文/实验数据/obsidian/第四章.opju"
            ),
            "origin_workbook": "Book7/Sheet5",
            "matlab_curve_generator": (
                "G:/大论文/2025陈鹏/论文/仿真模型/Untitled3_6.m"
            ),
            "matlab_efficiency_function": (
                "G:/大论文/2025陈鹏/论文/仿真模型/cal_eff.m"
            ),
        },
        "stacks": summarize_chen_efficiency_curves(audited),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(OUTPUT, index=False)
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_curve_basis_audit(audited, FIGURE)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
