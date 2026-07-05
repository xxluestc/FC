"""Diagnose why the original predicted MPC differs from baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def block_bootstrap_ci(values: np.ndarray, block: int = 60, repeats: int = 1000):
    """Return a 95% moving-block bootstrap interval for an additive mean."""

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(2026)
    starts = np.arange(0, len(values), block)
    blocks = [values[start : start + block] for start in starts]
    estimates = []
    for _ in range(repeats):
        sampled = [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        estimates.append(np.concatenate(sampled)[: len(values)].mean())
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def operating_regime(frame: pd.DataFrame) -> pd.Series:
    demand = frame["demand_kw"]
    ramp = demand.diff().fillna(0).abs()
    high = demand.quantile(0.9)
    large_ramp = ramp.quantile(0.9)
    return pd.Series(
        np.select(
            [demand < -5, ramp >= large_ramp, demand >= high, demand.abs() <= 5],
            ["braking", "large_ramp", "high_power", "idle_low"],
            default="normal",
        ),
        index=frame.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--weight-search", type=Path, required=True)
    parser.add_argument("--stack-map", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = pd.read_csv(args.trajectory)
    metrics = pd.read_csv(args.metrics)
    search = pd.read_csv(args.weight_search).sort_values("selection_score")
    stack = pd.read_csv(args.stack_map)
    predictions = pd.read_csv(args.predictions)
    best = search.iloc[0]
    weights = {
        "hydrogen": best.w_h2,
        "degradation_proxy": best.w_deg,
        "battery_use": 1.5,
        "soc": best.w_soc,
        "switch": best.w_switch,
        "smooth": best.w_smooth,
    }
    action_max = stack["stack_power_kw"].max()
    max_step = stack["stack_power_kw"].diff().max()

    objective_rows = []
    switch_rows = []
    enriched = []
    for (strategy, horizon), group in trajectory.groupby(["strategy", "horizon_s"]):
        group = group.sort_values("step").copy()
        previous_soc = group["soc"].shift(fill_value=0.70)
        preferred_fc = np.clip(
            np.maximum(group["demand_kw"], 0) + 1200 * (0.70 - previous_soc),
            stack["stack_power_kw"].min(),
            action_max,
        )
        group["cost_hydrogen"] = (
            weights["hydrogen"] * group["h2_g"] / stack["faraday_h2_g_s"].max()
        )
        group["cost_proxy"] = weights["degradation_proxy"] * group["degradation_proxy"]
        group["cost_battery"] = weights["battery_use"] * group["p_bat_kw"].abs() / 120
        group["cost_soc_tracking"] = (
            weights["soc"] * (group["p_fc_kw"] - preferred_fc).abs() / action_max
            + 0.5 * (group["soc"] - 0.70).abs() / 0.1
        )
        group["cost_switch"] = group["weighted_switch_cost"]
        group["cost_smooth"] = group["weighted_smooth_cost"]
        cost_columns = [column for column in group if column.startswith("cost_")]
        group["cost_internal_sum"] = group[cost_columns].sum(axis=1)
        group["regime"] = operating_regime(group)
        enriched.append(group)
        objective_rows.append(
            {
                "strategy": strategy,
                "horizon_s": horizon,
                **{column: group[column].sum() for column in cost_columns},
                "internal_objective_sum": group["cost_internal_sum"].sum(),
            }
        )
        changed = group["tier"].diff().fillna(0).ne(0)
        runs = changed.cumsum().value_counts().to_numpy()
        switch_rows.append(
            {
                "strategy": strategy,
                "horizon_s": horizon,
                "switch_count": int(changed.sum()),
                "mean_dwell_s": runs.mean(),
                "median_dwell_s": np.median(runs),
                "minimum_dwell_s": runs.min(),
                "large_tier_jump_count": int(
                    group["tier"].diff().abs().fillna(0).gt(1).sum()
                ),
            }
        )
    enriched = pd.concat(enriched, ignore_index=True)
    pd.DataFrame(objective_rows).to_csv(
        args.out_dir / "objective_component_audit.csv", index=False
    )
    pd.DataFrame(switch_rows).to_csv(args.out_dir / "switching_audit.csv", index=False)

    occupancy = enriched.groupby(["strategy", "horizon_s", "tier"], as_index=False).agg(
        count=("tier", "size"),
        proxy_sum=("degradation_proxy", "sum"),
        internal_cost_sum=("cost_internal_sum", "sum"),
    )
    occupancy["share_pct"] = (
        100
        * occupancy["count"]
        / occupancy.groupby(["strategy", "horizon_s"])["count"].transform("sum")
    )
    occupancy = occupancy.merge(
        stack[["stack_power_kw", "performance_loss_cost_normalized"]],
        left_on="tier",
        right_index=True,
        how="left",
    )
    occupancy.to_csv(args.out_dir / "tier_occupancy_and_proxy.csv", index=False)

    instant = enriched[enriched["strategy"].eq("instant")]
    decomposition_rows = []
    for horizon in (3, 5, 10):
        predicted = enriched[
            enriched["strategy"].eq("predicted") & enriched["horizon_s"].eq(horizon)
        ]
        instant_counts = (
            instant["tier"].value_counts().reindex(range(len(stack)), fill_value=0)
        )
        predicted_counts = (
            predicted["tier"].value_counts().reindex(range(len(stack)), fill_value=0)
        )
        for tier in range(len(stack)):
            delta_count = int(predicted_counts[tier] - instant_counts[tier])
            decomposition_rows.append(
                {
                    "horizon_s": horizon,
                    "tier": tier,
                    "delta_occupancy_s": delta_count,
                    "proxy_per_s": stack.performance_loss_cost_normalized.iloc[tier],
                    "proxy_delta_contribution": delta_count
                    * stack.performance_loss_cost_normalized.iloc[tier],
                }
            )
    pd.DataFrame(decomposition_rows).to_csv(
        args.out_dir / "proxy_excess_decomposition_vs_instant.csv", index=False
    )

    comparison_rows = []
    for horizon in (3, 5, 10):
        pred = enriched[
            enriched.strategy.eq("predicted") & enriched.horizon_s.eq(horizon)
        ].set_index("step")
        perfect = enriched[
            enriched.strategy.eq("perfect") & enriched.horizon_s.eq(horizon)
        ].set_index("step")
        joined = pred.add_suffix("_pred").join(perfect.add_suffix("_perfect"))
        joined["action_diff"] = joined["tier_pred"] - joined["tier_perfect"]
        joined["actions_differ"] = joined["action_diff"].ne(0)
        for regime, part in joined.groupby("regime_pred"):
            difference = part["actions_differ"].astype(float).to_numpy()
            low, high = block_bootstrap_ci(difference)
            comparison_rows.append(
                {
                    "horizon_s": horizon,
                    "regime": regime,
                    "n": len(part),
                    "action_difference_rate": difference.mean(),
                    "difference_rate_ci_low": low,
                    "difference_rate_ci_high": high,
                    "mean_abs_tier_difference": part["action_diff"].abs().mean(),
                    "proxy_delta_pred_minus_perfect": (
                        part["degradation_proxy_pred"]
                        - part["degradation_proxy_perfect"]
                    ).sum(),
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        args.out_dir / "predicted_vs_perfect_by_regime.csv", index=False
    )

    # Link per-origin forecast errors to H5 action disagreement. The Wilcoxon
    # p-value is descriptive because adjacent seconds remain autocorrelated.
    h5_prediction = predictions[predictions.forecast_horizon_s.eq(5)].copy()
    forecast_error = h5_prediction.groupby("origin_index").apply(
        lambda part: np.sqrt(np.mean((part.power_pred_kw - part.power_actual_kw) ** 2)),
        include_groups=False,
    )
    pred_h5 = enriched[
        enriched.strategy.eq("predicted") & enriched.horizon_s.eq(5)
    ].set_index("source_index")
    perfect_h5 = enriched[
        enriched.strategy.eq("perfect") & enriched.horizon_s.eq(5)
    ].set_index("source_index")
    link = (
        pd.DataFrame({"forecast_rmse_kw": forecast_error})
        .join(
            pd.DataFrame(
                {
                    "actions_differ": pred_h5.tier.ne(perfect_h5.tier),
                    "predicted_tier": pred_h5.tier,
                    "perfect_tier": perfect_h5.tier,
                }
            )
        )
        .dropna()
    )
    action_mask = link["actions_differ"].astype(bool)
    differ = link.loc[action_mask, "forecast_rmse_kw"]
    same = link.loc[~action_mask, "forecast_rmse_kw"]
    sample = min(len(differ), len(same))
    statistic, pvalue = wilcoxon(
        differ.iloc[:sample].to_numpy(), same.iloc[:sample].to_numpy()
    )
    summary = {
        "h5_mean_forecast_rmse_when_actions_differ_kw": float(differ.mean()),
        "h5_mean_forecast_rmse_when_actions_match_kw": float(same.mean()),
        "paired_descriptive_wilcoxon_statistic": float(statistic),
        "paired_descriptive_pvalue": float(pvalue),
        "autocorrelation_warning": "Adjacent origins overlap; p-value is descriptive, not an independent-sample causal test.",
        "internal_vs_evaluation_alignment": {
            "aligned": [
                "hydrogen",
                "degradation proxy",
                "battery throughput proxy",
                "switch",
                "FC variation proxy",
                "SOC",
            ],
            "mismatches": [
                "internal objective normalizes and weights components, final table reports raw sums",
                "SOC uses per-step preferred-FC tracking and a short terminal beam penalty rather than final-trip SOC equality",
                "weight search was tuned for Predicted H5, not Perfect or Instant",
                "beam search and adjacent-tier restriction are approximate, so Perfect is not a global optimum",
            ],
        },
    }
    (args.out_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metrics.to_csv(args.out_dir / "strategy_total_metrics.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
