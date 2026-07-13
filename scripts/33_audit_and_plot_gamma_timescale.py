"""Audit and plot the exposure time scale implied by the Gamma calibration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import gamma

from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "fc_only_gamma_timescale"
FIGURES = RESULTS / "figures/fc_only_foundation"
CALIBRATION = RESULTS / "health/lzw_gamma_calibration.json"
DETERMINISTIC = RESULTS / "fc_only_deterministic_comparison/per_run_metrics.csv"
HORIZONS_H = np.asarray([1 / 30, 1, 10, 100, 1000], dtype=float)
TERMINAL_CV = 0.10
COLORS = {
    "empirical_1s": "#2A9D8F",
    "zuo_slow_30s": "#4C78A8",
    "zuo_fast_30s": "#D65F5F",
}
LABELS = {
    "empirical_1s": "Real-calibrated",
    "zuo_slow_30s": "Zuo slow (30 s*)",
    "zuo_fast_30s": "Zuo fast (30 s*)",
}


def set_paper_style() -> None:
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
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
        }
    )


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    gamma_scale = gamma_scale_for_terminal_cv(
        calibration["terminal_total_damage_pct"],
        calibration["terminal_continuous_damage_pct"],
        TERMINAL_CV,
    )
    per_run = pd.read_csv(DETERMINISTIC)
    per_run = per_run[per_run.strategy == "instant_health"].copy()
    per_run["continuous_rate_pct_per_hour"] = (
        per_run.main_expected_continuous_damage_pct
        * 3600.0
        / per_run.n_steps
    )

    rows = []
    for run in per_run.itertuples(index=False):
        for horizon_h in HORIZONS_H:
            mean = run.continuous_rate_pct_per_hour * horizon_h
            shape = mean / gamma_scale
            near_zero_probability = float(gamma.cdf(0.01 * mean, shape, scale=gamma_scale))
            rows.append(
                {
                    "load_source": run.load_source,
                    "load_seed": run.load_seed,
                    "horizon_h": horizon_h,
                    "expected_continuous_damage_pct": mean,
                    "gamma_scale_pct": gamma_scale,
                    "gamma_shape": shape,
                    "probability_below_one_percent_of_mean": near_zero_probability,
                    "q05_pct": float(gamma.ppf(0.05, shape, scale=gamma_scale)),
                    "median_pct": float(gamma.ppf(0.50, shape, scale=gamma_scale)),
                    "q95_pct": float(gamma.ppf(0.95, shape, scale=gamma_scale)),
                }
            )
    audit = pd.DataFrame(rows)
    summary = audit.groupby(["load_source", "horizon_h"]).agg(
        shape_median=("gamma_shape", "median"),
        shape_q05=("gamma_shape", lambda values: float(np.quantile(values, 0.05))),
        shape_q95=("gamma_shape", lambda values: float(np.quantile(values, 0.95))),
        near_zero_probability_median=(
            "probability_below_one_percent_of_mean",
            "median",
        ),
        near_zero_probability_q05=(
            "probability_below_one_percent_of_mean",
            lambda values: float(np.quantile(values, 0.05)),
        ),
        near_zero_probability_q95=(
            "probability_below_one_percent_of_mean",
            lambda values: float(np.quantile(values, 0.95)),
        ),
    ).reset_index()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT / "gamma_timescale_audit.csv", index=False)
    summary.to_csv(OUTPUT / "gamma_timescale_summary.csv", index=False)
    metadata = {
        "source": str(DETERMINISTIC.relative_to(ROOT)),
        "strategy": "instant_health deterministic exposure paths",
        "gamma_terminal_cv": TERMINAL_CV,
        "gamma_scale_pct": gamma_scale,
        "horizons_h": HORIZONS_H.tolist(),
        "near_zero_definition": "Gamma increment below 1% of its conditional mean",
        "interpretation": (
            "Analytical aggregation uses Gamma additivity with a frozen action-exposure "
            "rate. It diagnoses time scale and is not a service-life prediction."
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    for source in LABELS:
        selected = summary[summary.load_source == source].sort_values("horizon_h")
        x = selected.horizon_h.to_numpy(dtype=float)
        color = COLORS[source]
        axes[0].plot(
            x,
            selected.shape_median,
            marker="o",
            color=color,
            label=LABELS[source],
        )
        axes[0].fill_between(
            x,
            selected.shape_q05,
            selected.shape_q95,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axes[1].plot(
            x,
            selected.near_zero_probability_median,
            marker="o",
            color=color,
            label=LABELS[source],
        )
        axes[1].fill_between(
            x,
            selected.near_zero_probability_q05,
            selected.near_zero_probability_q95,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Aggregated exposure (h)")
    axes[0].set_ylabel("Effective Gamma shape")
    axes[0].legend(loc="upper left")
    axes[0].text(-0.1, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")

    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Aggregated exposure (h)")
    axes[1].set_ylabel("P[increment < 1% of mean]")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].text(-0.1, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")

    fig.tight_layout(w_pad=1.6, pad=0.55)
    fig.savefig(
        FIGURES / "fig04_gamma_timescale_diagnostic.png",
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)

    short = summary[np.isclose(summary.horizon_h, 1 / 30)]
    long = summary[np.isclose(summary.horizon_h, 1000)]
    report = f"""# FC-only Gamma时间尺度诊断

- 终点CV假设：{TERMINAL_CV:.0%}；固定Gamma scale：{gamma_scale:.6f}% damage。
- 120秒等效暴露下，三类负载的中位shape范围为{short.shape_median.min():.3e}-{short.shape_median.max():.3e}，采样增量低于条件均值1%的概率为{short.near_zero_probability_median.min():.1%}-{short.near_zero_probability_median.max():.1%}。
- 1000小时聚合暴露下，中位shape范围增至{long.shape_median.min():.2f}-{long.shape_median.max():.2f}，近零概率降至{long.near_zero_probability_median.min():.1%}-{long.near_zero_probability_median.max():.1%}。

当前Gamma参数适合在聚合暴露层讨论不确定性，不适合用10条120秒在线采样轨迹估计均值或宣称策略延寿。控制层继续使用条件期望；随机Gamma作为长时间尺度敏感性和风险分布。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
