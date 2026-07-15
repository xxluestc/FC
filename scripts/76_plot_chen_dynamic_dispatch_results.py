"""Plot validated Chen dynamic-dispatch results as publication-style PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data/results/chen_dynamic_dispatch_holdout"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures"

STACK_COLORS = ("#0072B2", "#D55E00", "#009E73")
STACK_LABELS = ("Stack 1", "Stack 2", "Stack 3")
STRATEGY_STYLES = {
    "average": ("#CC79A7", "s", "Average"),
    "daisy_chain": ("#E69F00", "X", "Daisy chain"),
    "instantaneous": ("#000000", "o", "Instantaneous"),
    "sticky": ("#56B4E9", "^", "Sticky mode"),
    "one_step_greedy": ("#0072B2", "v", "One-step greedy"),
    "break_even_hysteresis": ("#009E73", "D", "Online hysteresis"),
    "offline_dp": ("#6C757D", "P", "Offline DP bound"),
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


def strategy_summary(frame: pd.DataFrame, strategy: str) -> tuple[float, int]:
    selected = frame.loc[frame["strategy"] == strategy]
    hydrogen = float(selected["hydrogen_g"].sum())
    changes = int(selected["stack_state_changes"].sum())
    return hydrogen, changes


def plot_representative_trajectory(frame: pd.DataFrame, output: Path) -> None:
    strategies = ("instantaneous", "break_even_hysteresis")
    reference = frame.loc[frame["strategy"] == strategies[0]].sort_values("step")
    if reference.empty:
        raise ValueError("representative trajectory has no instantaneous rows")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": (0.72, 1.0, 1.0)},
        constrained_layout=True,
    )
    time_s = reference["time_s"].to_numpy(dtype=float)
    demand = reference["demand_net_power_kw"].to_numpy(dtype=float)
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
    seed = int(reference["seed"].iloc[0]) if "seed" in reference else None
    seed_label = f"Holdout seed {seed}: " if seed is not None else ""
    axes[0].set_title(
        f"(a) {seed_label}random dynamic net-power request",
        loc="left",
    )
    axes[0].set_ylim(bottom=0.0)
    style_axis(axes[0])

    for index, strategy in enumerate(strategies, start=1):
        selected = frame.loc[frame["strategy"] == strategy].sort_values("step")
        if len(selected) != len(reference):
            raise ValueError(f"incomplete representative rows for {strategy}")
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
        event_times = selected.loc[
            selected["stack_state_changes"] > 0, "time_s"
        ].to_numpy(dtype=float)
        for event_time in event_times:
            axis.axvline(event_time, color="#333333", linewidth=0.45, alpha=0.22)
        hydrogen, changes = strategy_summary(selected, strategy)
        display = STRATEGY_STYLES[strategy][2]
        axis.set_title(
            f"({chr(ord('a') + index)}) {display}: "
            f"{hydrogen:.1f} g H$_2$, {changes} stack-state changes",
            loc="left",
        )
        axis.set_ylabel("Allocated power\n(kW net)")
        axis.set_ylim(bottom=0.0)
        style_axis(axis)

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


def paired_tradeoff(
    policies: pd.DataFrame,
    strategies: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    instant = policies.loc[
        policies["strategy"] == "instantaneous",
        ["seed", "total_hydrogen_g", "total_stack_state_changes"],
    ].rename(
        columns={
            "total_hydrogen_g": "instant_hydrogen_g",
            "total_stack_state_changes": "instant_changes",
        }
    )
    selected = policies.copy()
    if strategies is not None:
        selected = selected.loc[selected["strategy"].isin(strategies)]
    paired = selected.merge(instant, on="seed", validate="many_to_one")
    paired["hydrogen_increase_pct"] = 100.0 * (
        paired["total_hydrogen_g"] / paired["instant_hydrogen_g"] - 1.0
    )
    paired["change_reduction_pct"] = 100.0 * (
        1.0
        - paired["total_stack_state_changes"]
        / paired["instant_changes"].replace(0, np.nan)
    )
    return paired


def aggregate_tradeoff(paired: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for key, group in paired.groupby(groups, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record: dict[str, float | str] = dict(zip(groups, key_values))
        for column in ("hydrogen_increase_pct", "change_reduction_pct"):
            values = group[column].dropna().to_numpy(dtype=float)
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def plot_tradeoff(
    policy_runs: pd.DataFrame,
    sensitivity_runs: pd.DataFrame,
    output: Path,
) -> None:
    main = aggregate_tradeoff(paired_tradeoff(policy_runs), ["strategy"])

    instant = policy_runs.loc[
        policy_runs["strategy"] == "instantaneous",
        ["seed", "total_hydrogen_g", "total_stack_state_changes"],
    ].copy()
    expanded = sensitivity_runs.merge(
        instant,
        on="seed",
        suffixes=("", "_instant"),
        validate="many_to_one",
    )
    expanded["hydrogen_increase_pct"] = 100.0 * (
        expanded["total_hydrogen_g"] / expanded["total_hydrogen_g_instant"] - 1.0
    )
    expanded["change_reduction_pct"] = 100.0 * (
        1.0
        - expanded["total_stack_state_changes"]
        / expanded["total_stack_state_changes_instant"].replace(0, np.nan)
    )
    sweep = aggregate_tradeoff(
        expanded,
        ["strategy", "penalty_multiple"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    axis = axes[0]
    for row in main.itertuples(index=False):
        color, marker, label = STRATEGY_STYLES[row.strategy]
        axis.errorbar(
            row.change_reduction_pct_mean,
            row.hydrogen_increase_pct_mean,
            xerr=row.change_reduction_pct_ci95,
            yerr=row.hydrogen_increase_pct_ci95,
            marker=marker,
            markersize=5.5,
            color=color,
            linestyle="none",
            capsize=2.0,
            label=label,
        )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axis.axvline(0.0, color="#999999", linestyle=":", linewidth=0.7)
    axis.set(
        xlabel="Stack-state changes reduced vs instantaneous (%)",
        ylabel="Hydrogen increase vs instantaneous (%)",
        title="(a) Holdout comparison: all strategies",
    )
    axis.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(axis)

    axis = axes[1]
    for strategy, linestyle in (
        ("break_even_hysteresis", "-"),
        ("offline_dp", "--"),
        ("one_step_greedy", ":"),
    ):
        selected = sweep.loc[sweep["strategy"] == strategy].sort_values(
            "penalty_multiple"
        )
        color, marker, label = STRATEGY_STYLES[strategy]
        axis.plot(
            selected["change_reduction_pct_mean"],
            selected["hydrogen_increase_pct_mean"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.5,
            label=label,
        )
        for row in selected.itertuples(index=False):
            if (
                strategy == "break_even_hysteresis"
                and row.penalty_multiple in (0.5, 1.0, 2.0, 4.0)
            ):
                offset = {
                    0.5: (3, 3),
                    1.0: (3, 3),
                    2.0: (4, -12),
                    4.0: (4, 5),
                }[row.penalty_multiple]
                axis.annotate(
                    f"{row.penalty_multiple:g}x",
                    (row.change_reduction_pct_mean, row.hydrogen_increase_pct_mean),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="Stack-state changes reduced vs instantaneous (%)",
        ylabel="Hydrogen increase vs instantaneous (%)",
        title="(b) Penalty sensitivity and offline bound",
        xlim=(-5.0, 100.0),
        ylim=(-0.03, 0.42),
    )
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_style()
    trajectory = pd.read_csv(args.results_dir / "representative_trajectory.csv")
    policy_runs = pd.read_csv(args.results_dir / "per_run_metrics.csv")
    sensitivity = pd.read_csv(args.results_dir / "switch_penalty_sensitivity.csv")
    plot_representative_trajectory(
        trajectory,
        args.out_dir / "fig36_chen_dynamic_dispatch_trajectory.png",
    )
    plot_tradeoff(
        policy_runs,
        sensitivity,
        args.out_dir / "fig37_chen_dynamic_dispatch_tradeoff.png",
    )


if __name__ == "__main__":
    main()
