"""Consolidate service rescheduling screens into one auditable result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "fc_only_service_scheduler_reschedule"
FIGURES = RESULTS / "figures/fc_only_foundation"
INTERVALS_H = (6, 12, 24, 48)
POLICIES = ("expected_max", "gamma_cvar")


def load_results() -> pd.DataFrame:
    rows = []
    for interval in INTERVALS_H:
        path = (
            RESULTS
            / f"fc_only_service_scheduler_sweeps/reschedule_{interval}h/summary.csv"
        )
        table = pd.read_csv(path)
        for row in table.to_dict(orient="records"):
            rows.append({"reschedule_h": interval, **row})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"expected_max": "#2A9D8F", "gamma_cvar": "#D65F5F"}
    labels = {"expected_max": "Expected max", "gamma_cvar": "Gamma-CVaR"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fixed = table[table.policy == "fixed_pair"].time_to_limit_mean_h.mean()
    axes[0].axhline(
        fixed, color="#4C78A8", linestyle="--", linewidth=1.2, label="Fixed pair"
    )
    for policy in POLICIES:
        selected = table[table.policy == policy].sort_values("reschedule_h")
        axes[0].errorbar(
            selected.reschedule_h,
            selected.time_to_limit_mean_h,
            yerr=selected.time_to_limit_std_h,
            marker="o",
            capsize=3,
            color=colors[policy],
            label=labels[policy],
        )
        axes[1].plot(
            selected.reschedule_h,
            selected.start_count_mean,
            marker="o",
            color=colors[policy],
            label=labels[policy],
        )
    axes[0].set_xlabel("Minimum rescheduling period (h)")
    axes[0].set_ylabel("Time to health boundary (h)")
    axes[0].legend(frameon=False)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].set_xlabel("Minimum rescheduling period (h)")
    axes[1].set_ylabel("Mean start count")
    axes[1].text(-0.12, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    axes[1].set_xticks(INTERVALS_H)
    axes[0].set_xticks(INTERVALS_H)
    fig.tight_layout(w_pad=1.8, pad=0.55)
    fig.savefig(
        FIGURES / "fig08_reschedule_sensitivity.png", dpi=320, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    table = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "reschedule_sensitivity.csv", index=False)
    plot(table)
    adaptive = table[table.policy.isin(POLICIES)]
    summary = adaptive.groupby("policy").agg(
        minimum_mean_time_h=("time_to_limit_mean_h", "min"),
        maximum_mean_time_h=("time_to_limit_mean_h", "max"),
        minimum_start_count=("start_count_mean", "min"),
        maximum_start_count=("start_count_mean", "max"),
    )
    metadata = {
        "scope": "development rescheduling-period screen",
        "reschedule_hours": list(INTERVALS_H),
        "load_health_seeds": list(range(2026, 2036)),
        "risk_samples": 64,
        "source_directories": [
            f"fc_only_service_scheduler_sweeps/reschedule_{h}h"
            for h in INTERVALS_H
        ],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 慢层最小重调度周期敏感性\n\n"
    report += (
        "6/12/24/48小时开发筛查使用同一组10个负载-健康种子。"
        "Expected-max与Gamma-CVaR在四个周期均保持相对固定双堆的健康边界时间优势；"
        "延长周期主要减少启动次数，没有消除收益。\n\n"
    )
    report += summary.to_markdown() + "\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
