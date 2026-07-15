"""Plot validated N+1 single-stack fault results as 320 DPI PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data/results/chen_n_plus_one_fault_reconfiguration"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures"
STACK_COLORS = ("#0072B2", "#D55E00", "#009E73")
STACK_LABELS = ("Stack 1", "Stack 2", "Stack 3")
POLICY_COLORS = {
    "average": "#CC79A7",
    "instantaneous": "#000000",
    "break_even_hysteresis": "#009E73",
    "offline_dp": "#6C757D",
}
POLICY_LABELS = {
    "average": "Average",
    "instantaneous": "Instantaneous",
    "break_even_hysteresis": "Online hysteresis",
    "offline_dp": "Offline DP",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 320,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_fault_trajectory(frame: pd.DataFrame, output: Path) -> None:
    strategies = ("instantaneous", "break_even_hysteresis")
    reference = frame.loc[frame["strategy"].eq(strategies[0])].sort_values("step")
    seed = int(reference["seed"].iloc[0])
    fault_step = int(reference["fault_step"].iloc[0])
    demand = reference["demand_net_power_kw"].to_numpy(dtype=float)
    time_s = reference["time_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": (0.72, 1.0, 1.0)},
        constrained_layout=True,
    )
    axes[0].step(time_s, demand, where="post", color="#111111", linewidth=1.2)
    axes[0].fill_between(
        time_s,
        0.0,
        demand,
        step="post",
        color="#BDBDBD",
        alpha=0.25,
        linewidth=0.0,
    )
    axes[0].set_ylabel("Demand\n(kW net)")
    axes[0].set_title(
        f"(a) Holdout seed {seed}: permanent Stack 3 fault",
        loc="left",
    )
    axes[0].set_ylim(bottom=0.0)
    style_axis(axes[0])

    for index, strategy in enumerate(strategies, start=1):
        selected = frame.loc[frame["strategy"].eq(strategy)].sort_values("step")
        powers = [
            selected[f"stack_{stack}_net_power_kw"].to_numpy(dtype=float)
            for stack in range(1, 4)
        ]
        axis = axes[index]
        axis.stackplot(
            time_s,
            *powers,
            labels=STACK_LABELS,
            colors=STACK_COLORS,
            step="post",
            alpha=0.92,
            linewidth=0.0,
        )
        axis.step(
            time_s,
            demand,
            where="post",
            color="#111111",
            linewidth=0.9,
            label="Demand",
        )
        axis.set_ylabel("Allocated power\n(kW net)")
        axis.set_ylim(bottom=0.0)
        display = POLICY_LABELS[strategy]
        total_hydrogen = float(selected["hydrogen_g"].sum())
        total_changes = int(selected["stack_state_changes"].sum())
        axis.set_title(
            f"({chr(ord('a') + index)}) {display}: "
            f"{total_hydrogen:.1f} g H$_2$, {total_changes} state changes",
            loc="left",
        )
        style_axis(axis)

    for axis in axes:
        axis.axvline(fault_step, color="#D62728", linewidth=1.1, linestyle="--")
    axes[0].annotate(
        "fault isolated",
        xy=(fault_step, 0.93),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        color="#D62728",
        fontsize=7.5,
        va="top",
    )
    axes[1].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.8,
        columnspacing=1.1,
    )
    axes[-1].set_xlabel("Time (s)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fault_cost(aggregate: pd.DataFrame, output: Path) -> None:
    strategies = tuple(POLICY_COLORS)
    failed_stacks = ("stack_1", "stack_2", "stack_3")
    x = np.arange(len(failed_stacks), dtype=float)
    width = 0.19
    offsets = (np.arange(len(strategies)) - 1.5) * width
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    for offset, strategy in zip(offsets, strategies):
        selected = aggregate.loc[aggregate["strategy"].eq(strategy)].set_index(
            "failed_stack"
        ).reindex(failed_stacks)
        axes[0].bar(
            x + offset,
            selected["incremental_post_fault_hydrogen_pct_mean"],
            width=width,
            yerr=selected["incremental_post_fault_hydrogen_pct_ci95"],
            color=POLICY_COLORS[strategy],
            label=POLICY_LABELS[strategy],
            capsize=2.0,
        )
        axes[1].bar(
            x + offset,
            100.0 * selected["reserve_stack_1_post_fault_share_mean"],
            width=width,
            yerr=100.0 * selected["reserve_stack_1_post_fault_share_ci95"],
            color=POLICY_COLORS[strategy],
            label=POLICY_LABELS[strategy],
            capsize=2.0,
        )

    axes[0].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[0].set(
        ylabel="Post-fault hydrogen increase (%)",
        title="(a) Cost of permanent single-stack loss",
    )
    axes[1].set(
        ylabel="Stack 1 active after fault (%)",
        title="(b) Reserve-stack utilization",
        ylim=(0.0, 108.0),
    )
    for axis in axes:
        axis.set_xticks(x, ("Stack 1", "Stack 2", "Stack 3"))
        axis.set_xlabel("Failed stack")
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    trajectory = pd.read_csv(
        args.results_dir / "representative_stack_3_fault_trajectory.csv"
    )
    aggregate = pd.read_csv(
        args.results_dir / "aggregate_by_strategy_and_fault.csv"
    )
    plot_fault_trajectory(
        trajectory,
        args.out_dir / "fig38_chen_n_plus_one_fault_reconfiguration.png",
    )
    plot_fault_cost(
        aggregate,
        args.out_dir / "fig39_chen_n_plus_one_fault_cost.png",
    )


if __name__ == "__main__":
    main()
