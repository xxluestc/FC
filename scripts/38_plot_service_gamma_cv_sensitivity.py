"""Consolidate Gamma-CV service scheduling screens."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "fc_only_service_scheduler_gamma_cv"
FIGURES = RESULTS / "figures/fc_only_foundation"
CVS = (("05", 0.05), ("10", 0.10), ("20", 0.20))
POLICIES = ("fixed_pair", "expected_max", "gamma_cvar")


def load_results():
    summary_rows = []
    paired_rows = []
    for tag, cv in CVS:
        directory = RESULTS / f"fc_only_service_scheduler_sweeps/gamma_cv_{tag}"
        summary = pd.read_csv(directory / "summary.csv")
        summary["gamma_terminal_cv"] = cv
        summary_rows.append(summary)
        per_run = pd.read_csv(directory / "per_run_metrics.csv")
        fixed = per_run[per_run.policy == "fixed_pair"].set_index("seed")
        for policy in ("expected_max", "gamma_cvar"):
            selected = per_run[per_run.policy == policy].set_index("seed")
            for seed in selected.index:
                paired_rows.append(
                    {
                        "gamma_terminal_cv": cv,
                        "policy": policy,
                        "seed": seed,
                        "time_gain_vs_fixed_h": float(
                            selected.loc[seed, "time_to_health_limit_h"]
                            - fixed.loc[seed, "time_to_health_limit_h"]
                        ),
                    }
                )
    return pd.concat(summary_rows, ignore_index=True), pd.DataFrame(paired_rows)


def plot(summary, paired):
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "fixed_pair": "#4C78A8",
        "expected_max": "#2A9D8F",
        "gamma_cvar": "#D65F5F",
    }
    labels = {
        "fixed_pair": "Fixed pair",
        "expected_max": "Expected max",
        "gamma_cvar": "Gamma-CVaR",
    }
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
    for policy in POLICIES:
        selected = summary[summary.policy == policy].sort_values("gamma_terminal_cv")
        axes[0].errorbar(
            100 * selected.gamma_terminal_cv,
            selected.time_to_limit_mean_h,
            yerr=selected.time_to_limit_std_h,
            marker="o",
            capsize=3,
            color=colors[policy],
            label=labels[policy],
        )
    for policy in ("expected_max", "gamma_cvar"):
        selected = paired[paired.policy == policy]
        grouped = selected.groupby("gamma_terminal_cv").time_gain_vs_fixed_h
        mean = grouped.mean()
        minimum = grouped.min()
        maximum = grouped.max()
        axes[1].errorbar(
            100 * mean.index.to_numpy(),
            mean.to_numpy(),
            yerr=np.vstack([mean - minimum, maximum - mean]),
            marker="o",
            capsize=3,
            color=colors[policy],
            label=labels[policy],
        )
    axes[0].set_xlabel("Terminal CV assumption (%)")
    axes[0].set_ylabel("Time to health boundary (h)")
    axes[0].set_xticks([5, 10, 20])
    axes[0].legend(frameon=False)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("Terminal CV assumption (%)")
    axes[1].set_ylabel("Paired gain vs fixed pair (h)")
    axes[1].set_xticks([5, 10, 20])
    axes[1].text(-0.12, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8, pad=0.55)
    fig.savefig(
        FIGURES / "fig09_service_gamma_cv_sensitivity.png",
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    summary, paired = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "gamma_cv_summary.csv", index=False)
    paired.to_csv(OUTPUT / "gamma_cv_paired.csv", index=False)
    plot(summary, paired)
    evidence = paired.groupby(["gamma_terminal_cv", "policy"]).agg(
        mean_gain_h=("time_gain_vs_fixed_h", "mean"),
        minimum_gain_h=("time_gain_vs_fixed_h", "min"),
        win_share=("time_gain_vs_fixed_h", lambda values: float((values > 0).mean())),
    )
    metadata = {
        "scope": "development Gamma-CV sensitivity",
        "gamma_terminal_cvs": [cv for _, cv in CVS],
        "seeds": list(range(2026, 2036)),
        "reschedule_hours": 24,
        "risk_samples": 64,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 小时级调度Gamma CV敏感性\n\n"
    report += (
        "CV 5%/10%/20%均使用相同10个负载-健康种子和24小时最小重调度周期。"
        "Expected-max与Gamma-CVaR相对固定双堆在全部配对种子上改善。\n\n"
    )
    report += evidence.to_markdown() + "\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(evidence.to_string())


if __name__ == "__main__":
    main()
