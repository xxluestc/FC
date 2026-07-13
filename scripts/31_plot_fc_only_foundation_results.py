"""Generate publication-quality figures for the FC-only foundation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "figures/fc_only_foundation"

COLORS = {
    "average": "#4C78A8",
    "rotating": "#F2A541",
    "instant_health": "#2A9D8F",
    "beam_health": "#8E6C8A",
    "empirical": "#2A9D8F",
    "zuo_slow": "#4C78A8",
    "zuo_fast": "#D65F5F",
}
STRATEGY_LABELS = {
    "average": "Average",
    "rotating": "Rotating",
    "instant_health": "Instant-health",
    "beam_health": "Beam-health",
}
SCENARIO_LABELS = {
    "empirical_1s": "Real-calibrated (1 s)",
    "zuo_slow_30s": "Zuo slow (30 s*)",
    "zuo_fast_30s": "Zuo fast (30 s*)",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "legend.frameon": False,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_markov_timescale() -> None:
    table = pd.read_csv(
        RESULTS / "fc_only_load_sensitivity/markov_timescale_audit.csv"
    )
    empirical = table[table.matrix.str.startswith("empirical")].sort_values(
        "decision_interval_s"
    )
    slow = table[table.matrix == "zuo_slow_30s"].iloc[0]
    fast = table[table.matrix == "zuo_fast_30s"].iloc[0]

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.plot(
        empirical.decision_interval_s,
        empirical.event_rate_per_hour,
        marker="o",
        color=COLORS["empirical"],
        label="Real-data estimate",
    )
    ax.scatter(
        [slow.decision_interval_s],
        [slow.event_rate_per_hour],
        marker="s",
        s=35,
        color=COLORS["zuo_slow"],
        label="Zuo slow (30 s*)",
        zorder=3,
    )
    ax.scatter(
        [fast.decision_interval_s],
        [fast.event_rate_per_hour],
        marker="^",
        s=42,
        color=COLORS["zuo_fast"],
        label="Zuo fast (30 s*)",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 5, 10, 30, 60], labels=["1", "5", "10", "30", "60"])
    ax.set_xlabel("Transition sampling interval (s)")
    ax.set_ylabel("Estimated state changes (h$^{-1}$)")
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.03,
        "* Engineering time-base assumption",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(pad=0.5)
    save_figure(fig, "fig01_markov_timescale_audit")


def plot_deterministic_tradeoffs() -> None:
    paired = pd.read_csv(
        RESULTS / "fc_only_deterministic_comparison/paired_deltas.csv"
    )
    paired = paired[paired.strategy == "instant_health"].copy()
    panels = (
        ("fc_tracking_mae_kw", "Tracking MAE delta (kW)", 1.0),
        ("hydrogen_g_per_fc_kwh", "H$_2$ intensity delta (g kWh$^{-1}$)", 1.0),
        ("main_expected_damage_increment_pct", "Damage delta ($10^{-4}$ %)", 1e4),
    )
    scenarios = list(SCENARIO_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), sharey=True)
    y = np.arange(len(scenarios))
    for panel_index, (metric, label, scale) in enumerate(panels):
        ax = axes[panel_index]
        selected = paired[paired.metric == metric].set_index("load_source").loc[
            scenarios
        ]
        delta = scale * selected.mean_delta.to_numpy(dtype=float)
        interval = scale * selected.ci95.to_numpy(dtype=float)
        colors = [
            COLORS["empirical"],
            COLORS["zuo_slow"],
            COLORS["zuo_fast"],
        ]
        ax.errorbar(
            delta,
            y,
            xerr=interval,
            fmt="none",
            ecolor="#555555",
            elinewidth=0.9,
            capsize=2.5,
            zorder=2,
        )
        ax.scatter(
            delta,
            y,
            c=colors,
            s=34,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel(label)
        ax.text(
            -0.12,
            1.03,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontweight="bold",
        )
        if panel_index == 0:
            ax.set_yticks(y, labels=[SCENARIO_LABELS[item] for item in scenarios])
        else:
            ax.tick_params(axis="y", left=False)
        ax.invert_yaxis()
    fig.tight_layout(w_pad=1.0, pad=0.55)
    save_figure(fig, "fig02_deterministic_tradeoff_forest")


def plot_real_holdout_validation() -> None:
    aggregate = pd.read_csv(
        RESULTS / "fc_only_real_holdout_replay/aggregate_metrics.csv"
    ).set_index("strategy").loc[list(STRATEGY_LABELS)]
    labels = [STRATEGY_LABELS[item] for item in aggregate.index]
    colors = [COLORS[item] for item in aggregate.index]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    axes[0].bar(
        x,
        aggregate.execution_success_share * 100,
        color=colors,
        width=0.68,
    )
    axes[0].set_ylabel("Execution success (%)")
    axes[0].set_ylim(80, 102)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    axes[1].bar(
        x,
        aggregate.fc_tracking_mae_kw_mean,
        yerr=aggregate.fc_tracking_mae_kw_std,
        color=colors,
        width=0.68,
        capsize=2.5,
        error_kw={"elinewidth": 0.8},
    )
    axes[1].set_ylabel("Tracking MAE (kW)")
    axes[1].set_ylim(bottom=0)

    axes[2].bar(
        x,
        aggregate.planning_runtime_s_mean,
        yerr=aggregate.planning_runtime_s_std,
        color=colors,
        width=0.68,
        capsize=2.5,
        error_kw={"elinewidth": 0.8},
    )
    axes[2].set_ylabel("Planning time per window (s)")

    for panel_index, ax in enumerate(axes):
        ax.set_xticks(x, labels=labels, rotation=22, ha="right")
        ax.text(
            -0.12,
            1.03,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(w_pad=1.2, pad=0.55)
    save_figure(fig, "fig03_real_holdout_validation")


def write_manifest() -> None:
    manifest = """# FC-only基础结果图清单

| 文件 | 建议图注 |
|---|---|
| `fig01_markov_timescale_audit` | 实车经验Markov矩阵在不同采样间隔下的等效状态变化率，以及暂按30秒解释的Zuo慢变/快变压力场景。下采样会漏掉中间变化，因此不同时间基准的矩阵不直接融合。 |
| `fig02_deterministic_tradeoff_forest` | Instant-health相对Average在10个配对负载种子上的功率跟踪、单位输出电量氢耗和期望退化变化。点为配对均值，误差线为95%区间；负值表示降低。 |
| `fig03_real_holdout_validation` | 冻结参数在segment 22-45固定中心窗口上的执行成功率、正功率成功窗口跟踪误差和规划时间。Average在两个窗口无严格等电流可行动作。柱为均值，误差线为跨正功率窗口标准差。 |

每张图保存为320 DPI PNG。图内`30 s*`表示工程时间基准假设，不是Zuo论文直接给定值。
"""
    (OUTPUT / "figure_manifest.md").write_text(manifest, encoding="utf-8")


def main() -> None:
    set_paper_style()
    plot_markov_timescale()
    plot_deterministic_tradeoffs()
    plot_real_holdout_validation()
    write_manifest()


if __name__ == "__main__":
    main()
