"""Create complete controller diagnostic tables and publication-style figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


COLORS = {
    "Instant": "#4C78A8",
    "Constant": "#F58518",
    "Perfect": "#54A24B",
    "Predicted_original": "#B279A2",
    "Predicted_optimized": "#E45756",
}


def save_figure(fig, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def regime(demand: pd.Series) -> pd.Series:
    ramp = demand.diff().fillna(0).abs()
    return pd.Series(
        np.select(
            [
                demand < -5,
                ramp >= ramp.quantile(0.9),
                demand >= demand.quantile(0.9),
                demand.abs() <= 5,
            ],
            ["Braking", "Large ramp", "High power", "Idle/low"],
            default="Normal",
        ),
        index=demand.index,
    )


def block_metric_table(group: pd.DataFrame, block_size: int = 60) -> pd.DataFrame:
    group = group.sort_values("step").copy()
    group["switch_event"] = group.tier.diff().fillna(0).ne(0).astype(float)
    group["fc_variation"] = group.p_fc_kw.diff().fillna(0).abs()
    group["block"] = np.arange(len(group)) // block_size
    return group.groupby("block").agg(
        h2_g=("h2_g", "sum"),
        proxy=("degradation_proxy", "sum"),
        battery_throughput_kwh=("p_bat_kw", lambda values: values.abs().sum() / 3600),
        switches=("switch_event", "sum"),
        fc_variation_kw=("fc_variation", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--pareto-search", type=Path, required=True)
    parser.add_argument("--pareto-evaluation", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--clipping", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(args.trajectory)
    metrics = pd.read_csv(args.metrics)
    pareto_search = pd.read_csv(args.pareto_search)
    pareto_evaluation = pd.read_csv(args.pareto_evaluation)
    stack = pd.read_csv(args.stack_map)
    clipping = json.loads(args.clipping.read_text(encoding="utf-8"))

    cost_columns = [column for column in trajectory if column.startswith("cost_")]
    objective = trajectory.groupby("strategy", as_index=False)[cost_columns].sum()
    objective["internal_objective_sum"] = objective[cost_columns].sum(axis=1)
    objective.to_csv(args.out_dir / "final_objective_components.csv", index=False)

    occupancy = trajectory.groupby(["strategy", "tier"], as_index=False).agg(
        count=("tier", "size"),
        proxy_sum=("degradation_proxy", "sum"),
        mean_fc_kw=("p_fc_kw", "mean"),
    )
    occupancy["share_pct"] = (
        100
        * occupancy["count"]
        / occupancy.groupby("strategy")["count"].transform("sum")
    )
    occupancy = occupancy.merge(
        stack[["stack_power_kw", "performance_loss_cost_normalized"]],
        left_on="tier",
        right_index=True,
        how="left",
    )
    occupancy.to_csv(args.out_dir / "final_tier_occupancy_proxy.csv", index=False)

    switching_rows = []
    for strategy, group in trajectory.groupby("strategy"):
        group = group.sort_values("step")
        changed = group.tier.diff().fillna(0).ne(0)
        run_lengths = changed.cumsum().value_counts().to_numpy()
        switching_rows.append(
            {
                "strategy": strategy,
                "switch_count": int(changed.sum()),
                "mean_dwell_s": run_lengths.mean(),
                "median_dwell_s": np.median(run_lengths),
                "minimum_dwell_s": run_lengths.min(),
                "fc_total_variation_kw": group.p_fc_kw.diff().fillna(0).abs().sum(),
            }
        )
    pd.DataFrame(switching_rows).to_csv(
        args.out_dir / "final_switching_statistics.csv", index=False
    )

    subset_rows = []
    for (strategy, braking), group in trajectory.groupby(["strategy", "is_braking"]):
        subset_rows.append(
            {
                "strategy": strategy,
                "subset": "Braking" if braking else "Non-braking",
                "n": len(group),
                "h2_kg": group.h2_g.sum() / 1000,
                "proxy_sum": group.degradation_proxy.sum(),
                "battery_throughput_kwh": group.p_bat_kw.abs().sum() / 3600,
                "mean_fc_kw": group.p_fc_kw.mean(),
                "mean_battery_kw": group.p_bat_kw.mean(),
            }
        )
    pd.DataFrame(subset_rows).to_csv(
        args.out_dir / "final_braking_nonbraking.csv", index=False
    )

    pred = trajectory[trajectory.strategy.eq("Predicted_optimized")].set_index("step")
    perfect = trajectory[trajectory.strategy.eq("Perfect")].set_index("step")
    action = (
        pred[["tier", "demand_kw", "degradation_proxy"]]
        .add_suffix("_predicted")
        .join(perfect[["tier", "degradation_proxy"]].add_suffix("_perfect"))
    )
    action["regime"] = regime(action.demand_kw_predicted)
    action["actions_differ"] = action.tier_predicted.ne(action.tier_perfect)
    action["absolute_tier_difference"] = (
        action.tier_predicted - action.tier_perfect
    ).abs()
    action["proxy_difference"] = (
        action.degradation_proxy_predicted - action.degradation_proxy_perfect
    )
    action_summary = action.groupby("regime", as_index=False).agg(
        n=("actions_differ", "size"),
        action_difference_rate=("actions_differ", "mean"),
        mean_absolute_tier_difference=("absolute_tier_difference", "mean"),
        proxy_difference_sum=("proxy_difference", "sum"),
    )
    action_summary.to_csv(
        args.out_dir / "optimized_vs_perfect_by_regime.csv", index=False
    )

    max_power = stack.stack_power_kw.iloc[-1]
    max_h2 = stack.faraday_h2_g_s.iloc[-1]
    max_proxy = stack.performance_loss_cost_normalized.iloc[-1]
    deficit_kwh = (0.70 - metrics.soc_final).clip(lower=0) * 37.0
    recharge_seconds = deficit_kwh / (max_power * 0.95 / 3600)
    equivalent = metrics.copy()
    equivalent["soc_deficit_kwh"] = deficit_kwh
    equivalent["soc_equivalent_recharge_s"] = recharge_seconds
    equivalent["soc_equivalent_h2_kg"] = (
        metrics.h2_kg + recharge_seconds * max_h2 / 1000
    )
    equivalent["soc_equivalent_proxy"] = (
        metrics.degradation_proxy_sum + recharge_seconds * max_proxy
    )
    equivalent["clipped_share_pct"] = clipping["clipped_share_pct"]
    equivalent.to_csv(
        args.out_dir / "final_metrics_with_soc_equivalence.csv", index=False
    )

    optimized = equivalent[equivalent.strategy.eq("Predicted_optimized")].iloc[0]
    relative_rows = []
    for baseline in equivalent.itertuples():
        if baseline.strategy == "Predicted_optimized":
            continue
        relative_rows.append(
            {
                "baseline": baseline.strategy,
                "soc_equivalent_h2_improvement_pct": 100
                * (baseline.soc_equivalent_h2_kg - optimized.soc_equivalent_h2_kg)
                / baseline.soc_equivalent_h2_kg,
                "soc_equivalent_proxy_improvement_pct": 100
                * (baseline.soc_equivalent_proxy - optimized.soc_equivalent_proxy)
                / baseline.soc_equivalent_proxy,
                "switch_improvement_pct": 100
                * (baseline.switch_count - optimized.switch_count)
                / max(baseline.switch_count, 1),
                "fc_variation_improvement_pct": 100
                * (baseline.fc_total_variation_kw - optimized.fc_total_variation_kw)
                / max(baseline.fc_total_variation_kw, 1e-9),
            }
        )
    pd.DataFrame(relative_rows).to_csv(
        args.out_dir / "soc_equivalent_relative_improvement.csv", index=False
    )

    optimized_blocks = block_metric_table(
        trajectory[trajectory.strategy.eq("Predicted_optimized")]
    )
    comparison_rows = []
    rng = np.random.default_rng(2026)
    block_metrics = optimized_blocks.columns.tolist()
    for baseline_name in (
        "Instant",
        "Constant",
        "Perfect",
        "Predicted_original",
    ):
        baseline_blocks = block_metric_table(
            trajectory[trajectory.strategy.eq(baseline_name)]
        )
        for metric in block_metrics:
            difference = (optimized_blocks[metric] - baseline_blocks[metric]).to_numpy()
            bootstrap = np.asarray(
                [
                    difference[rng.integers(0, len(difference), len(difference))].mean()
                    for _ in range(2000)
                ]
            )
            statistic, pvalue = (
                wilcoxon(difference) if np.any(difference) else (0.0, 1.0)
            )
            comparison_rows.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "optimized_minus_baseline_mean_per_60s": difference.mean(),
                    "ci95_low": np.quantile(bootstrap, 0.025),
                    "ci95_high": np.quantile(bootstrap, 0.975),
                    "wilcoxon_statistic": statistic,
                    "pvalue_raw": pvalue,
                    "blocks": len(difference),
                }
            )
    block_comparison = pd.DataFrame(comparison_rows)
    block_comparison["pvalue_holm"] = multipletests(
        block_comparison.pvalue_raw, method="holm"
    )[1]
    block_comparison.to_csv(
        args.out_dir / "paired_60s_block_comparison.csv", index=False
    )

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    metric_columns = [
        "soc_equivalent_h2_kg",
        "soc_equivalent_proxy",
        "switch_count",
        "fc_total_variation_kw",
    ]
    labels = [
        "SOC-equivalent H$_2$",
        "SOC-equivalent proxy",
        "Switches",
        "FC variation",
    ]
    normalized = equivalent.set_index("strategy")[metric_columns]
    normalized = normalized / normalized.loc["Instant"]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4))
    for axis, column, label in zip(axes, metric_columns, labels):
        values = normalized[column]
        axis.bar(
            np.arange(len(values)),
            values,
            color=[COLORS[name] for name in values.index],
            width=0.72,
        )
        axis.axhline(1, color="#333333", lw=0.8, ls="--")
        axis.set_ylabel(f"{label}\n(relative to Instant)")
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(
            [name.replace("Predicted_", "P-") for name in values.index],
            rotation=50,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.2)
    save_figure(fig, args.figure_dir / "optimized_strategy_metric_comparison")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.8), sharex=True)
    display = trajectory[trajectory.strategy.eq("Predicted_optimized")].iloc[:900]
    time_minutes = display.step / 60
    axes[0].plot(
        time_minutes, display.demand_kw, color="#222222", lw=0.7, label="Vehicle demand"
    )
    axes[0].plot(
        time_minutes,
        display.p_fc_kw,
        color=COLORS["Predicted_optimized"],
        lw=0.9,
        label="Fuel cell",
    )
    axes[0].plot(
        time_minutes, display.p_bat_kw, color="#4C78A8", lw=0.7, label="Battery"
    )
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(ncol=3, frameon=False)
    for name in ("Instant", "Predicted_original", "Predicted_optimized"):
        part = trajectory[trajectory.strategy.eq(name)].iloc[:900]
        axes[1].step(
            part.step / 60,
            part.tier,
            where="post",
            lw=0.8,
            color=COLORS[name],
            label=name,
        )
        axes[2].plot(part.step / 60, part.soc, lw=0.9, color=COLORS[name], label=name)
    axes[1].set_ylabel("FC tier")
    axes[2].set_ylabel("SOC")
    axes[2].set_xlabel("Time (min)")
    axes[1].legend(ncol=3, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    save_figure(fig, args.figure_dir / "optimized_power_split_and_switching")

    fig, axis = plt.subplots(figsize=(4.6, 3.2))
    scatter = axis.scatter(
        pareto_evaluation.fc_total_variation_kw,
        pareto_evaluation.soc_equivalent_proxy,
        s=25 + 1.2 * pareto_evaluation.switch_count,
        c=pareto_evaluation.soc_error.abs(),
        cmap="viridis_r",
        edgecolor="white",
        linewidth=0.5,
    )
    for row in pareto_evaluation.itertuples():
        axis.annotate(
            str(row.candidate_id),
            (row.fc_total_variation_kw, row.soc_equivalent_proxy),
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set(
        xlabel="FC total variation (kW)", ylabel="SOC-equivalent degradation proxy"
    )
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label="|Terminal SOC error|")
    save_figure(fig, args.figure_dir / "pareto_generalization_tradeoff")

    fig, axis = plt.subplots(figsize=(4.8, 3.0))
    ordered = action_summary.sort_values("action_difference_rate")
    axis.barh(ordered.regime, 100 * ordered.action_difference_rate, color="#4C78A8")
    axis.set(
        xlabel="Predicted–Perfect action difference (%)", ylabel="Operating regime"
    )
    axis.grid(axis="x", alpha=0.2)
    save_figure(fig, args.figure_dir / "action_difference_by_regime")


if __name__ == "__main__":
    main()
