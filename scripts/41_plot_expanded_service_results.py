"""Plot expanded fast-layer templates and long-service results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data/results/fc_only_service_templates/service_exposure_templates.csv"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
RESULTS = {
    "Real": ROOT / "data/results/fc_only_service_scheduler_strong_baseline_real",
    "Empirical\nMarkov": ROOT / "data/results/fc_only_service_scheduler_strong_baseline_markov",
    "Zuo\nslow": ROOT / "data/results/fc_only_service_scheduler_strong_baseline_zuo_slow",
    "Zuo\nfast": ROOT / "data/results/fc_only_service_scheduler_strong_baseline_zuo_fast",
}
SOURCE_KEYS = {
    "Real": "real_calibration_window",
    "Empirical\nMarkov": "empirical_markov_1s",
    "Zuo\nslow": "zuo_slow_30s",
    "Zuo\nfast": "zuo_fast_30s",
}
POLICIES = ("fixed_pair", "health_greedy", "expected_max")
POLICY_LABELS = ("Fixed", "Health-greedy", "Expected-max")
POLICY_COLORS = ("#6C757D", "#7A5195", "#1D3557")


def exposure_rates():
    table = pd.read_csv(TEMPLATES)
    rows = []
    for label, source in SOURCE_KEYS.items():
        selected = table[table.template_source == source]
        duration = selected.duration_h.to_numpy(dtype=float)
        rows.append(
            {
                "source": label,
                "continuous": float(
                    np.mean(
                        (
                            selected.role_0_continuous_mean_pct
                            + selected.role_1_continuous_mean_pct
                        )
                        / duration
                        * 1000
                    )
                ),
                "load_shift": float(
                    np.mean(
                        (
                            selected.role_0_load_shift_damage_pct
                            + selected.role_1_load_shift_damage_pct
                        )
                        / duration
                        * 1000
                    )
                ),
                "operational_start": float(
                    np.mean(
                        (
                            selected.role_0_operational_start_damage_pct
                            + selected.role_1_operational_start_damage_pct
                        )
                        / duration
                        * 1000
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def load_service_results():
    summaries = []
    paired = []
    for source, directory in RESULTS.items():
        summary = pd.read_csv(directory / "summary.csv")
        summary["source"] = source
        summaries.append(summary)
        comparison = pd.read_csv(directory / "paired_vs_fixed.csv")
        comparison["source"] = source
        paired.append(comparison)
    return pd.concat(summaries, ignore_index=True), pd.concat(paired, ignore_index=True)


def plot():
    exposure = exposure_rates()
    summary, paired = load_service_results()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    x = np.arange(len(RESULTS))

    bottom = np.zeros(len(exposure))
    components = (
        ("continuous", "Continuous", "#457B9D"),
        ("load_shift", "Load shift", "#F4A261"),
        ("operational_start", "Operational start", "#E76F51"),
    )
    for column, label, color in components:
        values = exposure[column].to_numpy(dtype=float)
        axes[0].bar(x, values, bottom=bottom, width=0.68, color=color, label=label)
        bottom += values
    axes[0].set_xticks(x, exposure.source)
    axes[0].set_ylabel("Two-role damage per 1000 h (%)")
    axes[0].set_title("Fast-layer exposure")
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")

    width = 0.22
    for index, (policy, label, color) in enumerate(
        zip(POLICIES, POLICY_LABELS, POLICY_COLORS)
    ):
        selected = summary[summary.policy == policy].set_index("source").loc[list(RESULTS)]
        mean = selected.time_to_limit_mean_h.to_numpy(dtype=float)
        low = mean - selected.time_to_limit_q10_h.to_numpy(dtype=float)
        high = selected.time_to_limit_q90_h.to_numpy(dtype=float) - mean
        axes[1].bar(
            x + (index - 1.0) * width,
            mean,
            width=width,
            color=color,
            label=label,
            yerr=np.vstack([low, high]),
            capsize=1.5,
            error_kw={"linewidth": 0.7},
        )
    axes[1].set_xticks(x, list(RESULTS))
    axes[1].set_ylabel("Time to health boundary (h)")
    axes[1].set_title("20 paired stochastic runs")
    axes[1].legend(
        frameon=False,
        fontsize=6.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
    )

    for index, (policy, label, color) in enumerate(
        (("health_greedy", "Health-greedy", "#7A5195"), ("expected_max", "Expected-max", "#1D3557"))
    ):
        selected = paired[paired.policy == policy].set_index("source").loc[list(RESULTS)]
        mean = selected.mean_gain_h.to_numpy(dtype=float)
        low = mean - selected.minimum_gain_h.to_numpy(dtype=float)
        high = selected.maximum_gain_h.to_numpy(dtype=float) - mean
        axes[2].errorbar(
            x + (index - 0.5) * 0.12,
            mean,
            yerr=np.vstack([low, high]),
            fmt="o",
            color=color,
            markersize=4,
            capsize=2,
            linewidth=1,
            label=label,
        )
    axes[2].axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(x, list(RESULTS))
    axes[2].set_ylabel("Gain vs fixed pair (h)")
    axes[2].set_title("Paired range across seeds")
    axes[2].legend(
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
    )

    for index, ax in enumerate(axes):
        ax.text(-0.13, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.8)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.25, wspace=0.43)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig11_expanded_service_results.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
