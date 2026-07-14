"""Run the coefficient-free health-adaptive allocation stop-gate experiment."""

from __future__ import annotations

import json
import shutil
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.health.dynamic_proxy import LzwIvConditions
from fc_power.health.lzw_health_progress import LzwHealthProgressMap
from fc_power.power_allocation.relative_health_allocator import (
    RelativeHealthWeights,
    allocate_relative_health_budget,
    build_n_plus_one_action_grid,
    choose_relative_health_action,
    executed_hydrogen_g,
    interpolate_lzw_power_table_kw,
    lzw_power_table_kw,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/relative_health_adaptive_replay"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
SOURCE = ROOT / "data/results/fc_only_external_monthly_blocks/external_monthly_power_blocks.csv"
MAPPING = ROOT / "data/results/lzw_health_progress/mapping.json"
CONDITIONS = ROOT / "data/upstream_lzw/current_point_cost_conditions.json"

ALLOWED_CURRENTS_A = (0.0, 90.0, 120.0, 160.0, 195.0, 270.0, 370.0)
MAX_ONLINE_STACKS = 2
CONTROL_INTERVAL_S = 10
TRACKING_SLACK_KW = 0.5
SYSTEM_CAPACITY_FRACTION = 0.90
INITIAL_HEALTH_LEVELS = (0.30, 0.36, 0.42)
PRIMARY_FLEET_PROGRESS_BUDGET = 0.03
SENSITIVITY_BUDGETS = (0.01, 0.06)
PRIMARY_HEALTH_WEIGHT = 0.20
HEALTH_WEIGHT_SWEEP = (0.10, 0.20, 0.40)
LATER_MONTHS = ("2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
HYDROGEN_LHV_KWH_PER_KG = 33.33
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 7319
HEALTH_LOOKUP_POINTS = 10001


def load_inputs():
    frame = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    mapping = LzwHealthProgressMap.from_dict(
        json.loads(MAPPING.read_text(encoding="utf-8"))
    )
    conditions = LzwIvConditions.from_upstream_dict(
        json.loads(CONDITIONS.read_text(encoding="utf-8"))
    )
    frame["target_power_kw"] = pd.to_numeric(
        frame["target_power_kw"], errors="coerce"
    )
    if frame["target_power_kw"].isna().any():
        raise ValueError("external replay contains missing target power")
    calibration = frame.loc[frame["month"] < LATER_MONTHS[0], "target_power_kw"]
    calibration = calibration[calibration > 0.5]
    if calibration.empty:
        raise ValueError("no positive pre-evaluation target power is available")
    vehicle_reference_kw = float(calibration.quantile(0.99))
    evaluation = frame.loc[frame["month"].isin(LATER_MONTHS)].copy()
    if evaluation.empty:
        raise ValueError("later development replay is empty")
    return evaluation, mapping, conditions, vehicle_reference_kw


def prepare_episodes(
    evaluation: pd.DataFrame,
    *,
    vehicle_reference_kw: float,
    mapped_capacity_kw: float,
) -> dict[str, pd.DataFrame]:
    episodes = {}
    for block_id, block in evaluation.groupby("block_id", sort=True):
        sampled = block.loc[block["block_step"] % CONTROL_INTERVAL_S == 0].copy()
        per_unit = np.clip(
            sampled["target_power_kw"].to_numpy(dtype=float)
            / vehicle_reference_kw,
            0.0,
            1.0,
        )
        sampled["mapped_demand_kw"] = per_unit * mapped_capacity_kw
        energy = float(
            sampled["mapped_demand_kw"].sum() * CONTROL_INTERVAL_S / 3600.0
        )
        if energy > 0:
            episodes[str(block_id)] = sampled.reset_index(drop=True)
    if not episodes:
        raise ValueError("all later replay blocks have zero mapped demand")
    return episodes


def policy_definitions():
    base = {
        "frozen_health": {
            "decision": "healthy_fixed",
            "health_weight": 0.0,
        },
        "performance_adaptive": {
            "decision": "online",
            "health_weight": 0.0,
        },
    }
    for value in HEALTH_WEIGHT_SWEEP:
        base[f"health_reserve_{value:.2f}"] = {
            "decision": "online",
            "health_weight": value,
        }
    return base


def simulate_episode(
    *,
    block_id: str,
    episode: pd.DataFrame,
    mapping: LzwHealthProgressMap,
    conditions: LzwIvConditions,
    health_lookup_grid: np.ndarray,
    health_lookup_power_kw: np.ndarray,
    action_indices: np.ndarray,
    action_currents: np.ndarray,
    initial_health,
    policy_name: str,
    policy: dict,
    fleet_progress_budget: float,
    keep_trace: bool,
):
    health = np.asarray(initial_health, dtype=float).copy()
    initial = health.copy()
    previous_currents = np.zeros(3, dtype=float)
    demand = episode["mapped_demand_kw"].to_numpy(dtype=float)
    episode_energy = float(demand.sum() * CONTROL_INTERVAL_S / 3600.0)
    rows = []
    demand_energy = 0.0
    delivered_energy = 0.0
    useful_delivered_energy = 0.0
    absolute_error_energy = 0.0
    hydrogen_g = 0.0
    switches = 0
    starts = 0
    health_loaded_energy = 0.0
    stack_energy = np.zeros(3, dtype=float)
    total_progress_increment = np.zeros(3, dtype=float)

    for step, row in episode.iterrows():
        requested = float(row["mapped_demand_kw"])
        decision_health = (
            np.zeros(3, dtype=float)
            if policy["decision"] == "healthy_fixed"
            else health
        )
        decision_power_table = interpolate_lzw_power_table_kw(
            decision_health, health_lookup_grid, health_lookup_power_kw
        )
        action = choose_relative_health_action(
            action_indices=action_indices,
            action_currents_a=action_currents,
            power_table_kw=decision_power_table,
            decision_health_progress=decision_health,
            previous_currents_a=previous_currents,
            demand_power_kw=requested,
            max_online_stacks=MAX_ONLINE_STACKS,
            tracking_slack_kw=TRACKING_SLACK_KW,
            weights=RelativeHealthWeights(
                health_loading=float(policy["health_weight"])
            ),
        )
        currents = np.asarray(action.current_a, dtype=float)
        current_indices = np.searchsorted(ALLOWED_CURRENTS_A, currents)
        actual_power_table = interpolate_lzw_power_table_kw(
            health, health_lookup_grid, health_lookup_power_kw
        )
        actual_stack_power = actual_power_table[np.arange(3), current_indices]
        actual_total = float(actual_stack_power.sum())
        next_health, increments = allocate_relative_health_budget(
            health,
            actual_stack_power,
            demand_power_kw=requested,
            dt_s=CONTROL_INTERVAL_S,
            episode_demand_energy_kwh=episode_energy,
            fleet_progress_budget=fleet_progress_budget,
        )

        dt_h = CONTROL_INTERVAL_S / 3600.0
        demand_energy += requested * dt_h
        delivered_energy += actual_total * dt_h
        useful_delivered_energy += min(actual_total, requested) * dt_h
        absolute_error_energy += abs(actual_total - requested) * dt_h
        hydrogen_g += executed_hydrogen_g(
            currents, CONTROL_INTERVAL_S, stack_cells=conditions.stack_cells
        )
        previous_on = previous_currents > 0
        current_on = currents > 0
        switches += int(np.count_nonzero(previous_on != current_on))
        starts += int(np.count_nonzero((~previous_on) & current_on))
        health_loaded_energy += float(np.sum(actual_stack_power * health) * dt_h)
        stack_energy += actual_stack_power * dt_h
        total_progress_increment += increments

        if keep_trace:
            trace = {
                "block_id": block_id,
                "policy": policy_name,
                "fleet_progress_budget": fleet_progress_budget,
                "step": int(step),
                "elapsed_s": int(step * CONTROL_INTERVAL_S),
                "timestamp": row["timestamp"],
                "vehicle_target_power_kw": float(row["target_power_kw"]),
                "mapped_demand_kw": requested,
                "actual_total_power_kw": actual_total,
                "tracking_error_kw": actual_total - requested,
            }
            for index in range(3):
                trace[f"stack_{index + 1}_current_a"] = currents[index]
                trace[f"stack_{index + 1}_power_kw"] = actual_stack_power[index]
                trace[f"stack_{index + 1}_h_before"] = health[index]
                trace[f"stack_{index + 1}_h_increment"] = increments[index]
                trace[f"stack_{index + 1}_h_after"] = next_health[index]
            rows.append(trace)
        health = next_health
        previous_currents = currents

    gross_efficiency_pct = 100.0 * delivered_energy / max(
        hydrogen_g / 1000.0 * HYDROGEN_LHV_KWH_PER_KG, 1e-12
    )
    useful_efficiency_pct = 100.0 * useful_delivered_energy / max(
        hydrogen_g / 1000.0 * HYDROGEN_LHV_KWH_PER_KG, 1e-12
    )
    metrics = {
        "block_id": block_id,
        "month": str(episode["month"].iloc[0]),
        "initial_health": "|".join(f"{value:.2f}" for value in initial),
        "policy": policy_name,
        "health_weight": float(policy["health_weight"]),
        "fleet_progress_budget": fleet_progress_budget,
        "steps": len(episode),
        "demand_energy_kwh": demand_energy,
        "delivered_energy_kwh": delivered_energy,
        "useful_delivered_energy_kwh": useful_delivered_energy,
        "tracking_mae_kw": absolute_error_energy / max(
            len(episode) * CONTROL_INTERVAL_S / 3600.0, 1e-12
        ),
        "energy_tracking_error_pct": 100.0
        * (delivered_energy - demand_energy)
        / max(demand_energy, 1e-12),
        "hydrogen_g": hydrogen_g,
        "gross_electrical_efficiency_pct": gross_efficiency_pct,
        "useful_electrical_efficiency_pct": useful_efficiency_pct,
        "switch_count": switches,
        "start_count": starts,
        "health_weighted_loading": health_loaded_energy
        / max(delivered_energy, 1e-12),
        "final_max_health_progress": float(health.max()),
        "final_min_health_progress": float(health.min()),
        "final_health_range": float(health.max() - health.min()),
        "final_health_std": float(health.std()),
        "total_progress_increment": float(total_progress_increment.sum()),
    }
    for index in range(3):
        metrics[f"stack_{index + 1}_initial_h"] = initial[index]
        metrics[f"stack_{index + 1}_final_h"] = health[index]
        metrics[f"stack_{index + 1}_energy_kwh"] = stack_energy[index]
    return metrics, pd.DataFrame(rows)


def mean_ci(values, *, rng: np.random.Generator):
    data = np.asarray(values, dtype=float)
    observed = float(data.mean())
    draws = rng.choice(data, size=(BOOTSTRAP_RESAMPLES, len(data)), replace=True)
    boot = draws.mean(axis=1)
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_block_effects(metrics: pd.DataFrame, policy: str, reference: str):
    primary = metrics.loc[
        np.isclose(metrics["fleet_progress_budget"], PRIMARY_FLEET_PROGRESS_BUDGET)
    ]
    averaged = (
        primary.groupby(["block_id", "policy"], as_index=False)
        .mean(numeric_only=True)
    )
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in (
        "final_max_health_progress",
        "final_health_range",
        "health_weighted_loading",
        "useful_electrical_efficiency_pct",
        "tracking_mae_kw",
        "switch_count",
    ):
        pivot = averaged.pivot(index="block_id", columns="policy", values=metric)
        paired = (pivot[policy] - pivot[reference]).dropna()
        effect, low, high = mean_ci(paired.to_numpy(), rng=rng)
        rows.append(
            {
                "policy": policy,
                "reference": reference,
                "metric": metric,
                "block_count": len(paired),
                "mean_paired_difference": effect,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            }
        )
    return rows


def choose_example_block(
    episodes: dict[str, pd.DataFrame], *, single_stack_capacity_kw: float
) -> str:
    ranked = []
    for block_id, frame in episodes.items():
        target = frame["target_power_kw"].to_numpy(dtype=float)
        demand = frame["mapped_demand_kw"].to_numpy(dtype=float)
        transitions = int(np.count_nonzero(np.abs(np.diff(target)) > 0.01))
        requires_two_stacks = int(float(demand.max()) > single_stack_capacity_kw)
        ranked.append(
            (requires_two_stacks, transitions, float(target.max()), block_id)
        )
    return max(ranked)[3]


def plot_results(
    trace: pd.DataFrame,
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)

    reserve = trace.loc[trace["policy"] == f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}"]
    time_min = reserve["elapsed_s"] / 60.0
    axes[0, 0].plot(time_min, reserve["mapped_demand_kw"], color="#222222", lw=1.2, label="Demand")
    for index, color in enumerate(colors, start=1):
        axes[0, 0].plot(
            time_min,
            reserve[f"stack_{index}_power_kw"],
            color=color,
            lw=0.9,
            label=f"Stack {index}",
        )
    axes[0, 0].set(xlabel="Time (min)", ylabel="Power (kW)", title="(a) Real-load mapped N+1 replay")
    axes[0, 0].legend(ncol=2, frameon=False)

    for policy_name, style, alpha in (
        ("performance_adaptive", "--", 0.75),
        (f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}", "-", 1.0),
    ):
        selected = trace.loc[trace["policy"] == policy_name]
        t = selected["elapsed_s"] / 60.0
        for index, color in enumerate(colors, start=1):
            axes[0, 1].plot(
                t,
                selected[f"stack_{index}_h_after"],
                color=color,
                ls=style,
                alpha=alpha,
                lw=1.0,
            )
    axes[0, 1].set(
        xlabel="Time (min)",
        ylabel="Normalized progress, h",
        title="(b) Online relative health update",
    )
    axes[0, 1].text(0.02, 0.78, "solid: health reserve\ndashed: adaptive IV", transform=axes[0, 1].transAxes, va="top", fontsize=7)

    primary = metrics.loc[
        np.isclose(metrics["fleet_progress_budget"], PRIMARY_FLEET_PROGRESS_BUDGET)
    ]
    order = [
        "frozen_health",
        "performance_adaptive",
        f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}",
    ]
    labels = ["Frozen", "Adaptive IV", "Health reserve"]
    values = [
        primary.loc[primary["policy"] == name, "final_health_range"].to_numpy()
        for name in order
    ]
    box = axes[1, 0].boxplot(values, tick_labels=labels, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ("#999999", "#56B4E9", "#009E73")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 0].set(ylabel="Final health-progress range", title="(c) Paired episode distribution")

    summary = (
        primary.groupby(["policy", "health_weight"], as_index=False)
        .agg(
            final_max=("final_max_health_progress", "mean"),
            tracking_mae=("tracking_mae_kw", "mean"),
        )
    )
    reserve_summary = summary.loc[summary["policy"].str.startswith("health_reserve")].sort_values("health_weight")
    axes[1, 1].plot(
        reserve_summary["final_max"],
        reserve_summary["tracking_mae"],
        color="#CC79A7",
        marker="o",
        lw=1.1,
    )
    for row in reserve_summary.itertuples():
        offset = {0.1: (4, 3), 0.2: (5, 5), 0.4: (5, -10)}[
            round(float(row.health_weight), 1)
        ]
        axes[1, 1].annotate(
            f"w={row.health_weight:.1f}",
            (row.final_max, row.tracking_mae),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.8,
        )
    for name, marker, color in (("frozen_health", "s", "#666666"), ("performance_adaptive", "^", "#0072B2")):
        row = summary.loc[summary["policy"] == name].iloc[0]
        axes[1, 1].scatter(row["final_max"], row["tracking_mae"], marker=marker, color=color, s=28, label=name.replace("_", " "))
    axes[1, 1].set(
        xlabel="Mean final maximum h",
        ylabel="Tracking MAE (kW)",
        title="(d) Health-tracking trade-off",
    )
    axes[1, 1].legend(frameon=False)

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    evaluation, mapping, conditions, vehicle_reference_kw = load_inputs()
    action_indices, action_currents = build_n_plus_one_action_grid(
        ALLOWED_CURRENTS_A,
        n_stacks=3,
        max_online_stacks=MAX_ONLINE_STACKS,
    )
    health_lookup_grid = np.linspace(0.0, 1.0, HEALTH_LOOKUP_POINTS)
    health_lookup_power_kw = lzw_power_table_kw(
        mapping,
        conditions,
        health_lookup_grid,
        ALLOWED_CURRENTS_A,
    )
    audit_progress = np.linspace(0.0037, 0.9963, 101)
    lookup_audit = interpolate_lzw_power_table_kw(
        audit_progress, health_lookup_grid, health_lookup_power_kw
    )
    direct_audit = lzw_power_table_kw(
        mapping, conditions, audit_progress, ALLOWED_CURRENTS_A
    )
    lookup_max_abs_error_kw = float(np.max(np.abs(lookup_audit - direct_audit)))
    healthy_power = lzw_power_table_kw(
        mapping, conditions, [0.0, 0.0, 0.0], ALLOWED_CURRENTS_A
    )
    healthy_n_plus_one_capacity = float(
        np.sort(healthy_power[:, -1])[-MAX_ONLINE_STACKS:].sum()
    )
    mapped_capacity = SYSTEM_CAPACITY_FRACTION * healthy_n_plus_one_capacity
    episodes = prepare_episodes(
        evaluation,
        vehicle_reference_kw=vehicle_reference_kw,
        mapped_capacity_kw=mapped_capacity,
    )
    single_stack_capacity = float(healthy_power[:, -1].max())
    example_block = choose_example_block(
        episodes, single_stack_capacity_kw=single_stack_capacity
    )
    initial_orders = sorted(set(permutations(INITIAL_HEALTH_LEVELS)))
    policies = policy_definitions()

    metric_rows = []
    trace_frames = []
    for budget in (PRIMARY_FLEET_PROGRESS_BUDGET,) + SENSITIVITY_BUDGETS:
        active_policies = (
            policies
            if np.isclose(budget, PRIMARY_FLEET_PROGRESS_BUDGET)
            else {
                name: value
                for name, value in policies.items()
                if name in {"frozen_health", f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}"}
            }
        )
        for block_id, episode in episodes.items():
            for initial_health in initial_orders:
                for policy_name, policy in active_policies.items():
                    keep_trace = (
                        np.isclose(budget, PRIMARY_FLEET_PROGRESS_BUDGET)
                        and block_id == example_block
                        and initial_health == initial_orders[0]
                        and policy_name
                        in {
                            "frozen_health",
                            "performance_adaptive",
                            f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}",
                        }
                    )
                    metrics, trace = simulate_episode(
                        block_id=block_id,
                        episode=episode,
                        mapping=mapping,
                        conditions=conditions,
                        health_lookup_grid=health_lookup_grid,
                        health_lookup_power_kw=health_lookup_power_kw,
                        action_indices=action_indices,
                        action_currents=action_currents,
                        initial_health=initial_health,
                        policy_name=policy_name,
                        policy=policy,
                        fleet_progress_budget=budget,
                        keep_trace=keep_trace,
                    )
                    metric_rows.append(metrics)
                    if keep_trace:
                        trace_frames.append(trace)

    metrics = pd.DataFrame(metric_rows)
    trace = pd.concat(trace_frames, ignore_index=True)
    aggregate = (
        metrics.groupby(["fleet_progress_budget", "policy", "health_weight"], as_index=False)
        .agg(
            episodes=("block_id", "size"),
            blocks=("block_id", "nunique"),
            tracking_mae_kw=("tracking_mae_kw", "mean"),
            gross_electrical_efficiency_pct=("gross_electrical_efficiency_pct", "mean"),
            useful_electrical_efficiency_pct=("useful_electrical_efficiency_pct", "mean"),
            final_max_health_progress=("final_max_health_progress", "mean"),
            final_health_range=("final_health_range", "mean"),
            health_weighted_loading=("health_weighted_loading", "mean"),
            switch_count=("switch_count", "mean"),
            total_progress_increment=("total_progress_increment", "mean"),
        )
    )
    effect_rows = []
    primary_policy = f"health_reserve_{PRIMARY_HEALTH_WEIGHT:.2f}"
    for reference in ("frozen_health", "performance_adaptive"):
        effect_rows.extend(paired_block_effects(metrics, primary_policy, reference))
    effects = pd.DataFrame(effect_rows)

    budget_error = float(
        np.max(
            np.abs(
                metrics["total_progress_increment"]
                - metrics["fleet_progress_budget"]
            )
        )
    )
    primary_reference = "performance_adaptive"
    against_reference = effects.loc[
        effects["reference"] == primary_reference
    ].set_index("metric")
    max_health_effect = float(
        against_reference.loc[
            "final_max_health_progress", "mean_paired_difference"
        ]
    )
    loading_effect = float(
        against_reference.loc[
            "health_weighted_loading", "mean_paired_difference"
        ]
    )
    efficiency_effect = float(
        against_reference.loc[
            "useful_electrical_efficiency_pct", "mean_paired_difference"
        ]
    )
    tracking_effect = float(
        against_reference.loc["tracking_mae_kw", "mean_paired_difference"]
    )
    checks = {
        "fleet_budget_is_policy_invariant": budget_error <= 1e-10,
        "final_max_health_is_lower_than_performance_adaptive": max_health_effect < 0.0,
        "health_weighted_loading_is_lower_than_performance_adaptive": loading_effect < 0.0,
        "useful_efficiency_loss_within_0p25_point": efficiency_effect >= -0.25,
        "tracking_mae_increase_within_0p25_kw": tracking_effect <= 0.25,
    }
    decision = "LIMITED_GO_MECHANISM_ONLY" if all(checks.values()) else "NO_GO"

    metrics.to_csv(OUTPUT / "episode_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    effects.to_csv(OUTPUT / "paired_block_effects.csv", index=False)
    trace.to_csv(OUTPUT / "example_trajectory.csv", index=False)
    figure = OUTPUT / "fig35_relative_health_adaptive_replay.png"
    plot_results(trace, metrics, effects, figure)
    shutil.copy2(figure, FIGURES / figure.name)

    metadata = {
        "decision": decision,
        "interpretation": (
            "fixed-budget relative-health redistribution mechanism only; not "
            "independent method effectiveness, physical degradation-rate "
            "identification, or lifetime-extension validation"
        ),
        "primary_reference": primary_reference,
        "source": str(SOURCE.resolve()),
        "evaluation_months": list(LATER_MONTHS),
        "evaluation_role": "nested later-time development replay, not untouched final holdout",
        "vehicle_power_field": "target_power_kw",
        "vehicle_reference_kw_pre_2026_02_p99": vehicle_reference_kw,
        "healthy_n_plus_one_capacity_kw": healthy_n_plus_one_capacity,
        "mapped_capacity_kw": mapped_capacity,
        "system_capacity_fraction": SYSTEM_CAPACITY_FRACTION,
        "control_interval_s": CONTROL_INTERVAL_S,
        "allowed_currents_a": list(ALLOWED_CURRENTS_A),
        "max_online_stacks": MAX_ONLINE_STACKS,
        "tracking_slack_kw": TRACKING_SLACK_KW,
        "health_lookup_points": HEALTH_LOOKUP_POINTS,
        "health_lookup_max_abs_error_kw": lookup_max_abs_error_kw,
        "initial_health_levels": list(INITIAL_HEALTH_LEVELS),
        "initial_order_count": len(initial_orders),
        "episode_block_count": len(episodes),
        "primary_fleet_progress_budget": PRIMARY_FLEET_PROGRESS_BUDGET,
        "sensitivity_budgets": list(SENSITIVITY_BUDGETS),
        "primary_health_weight": PRIMARY_HEALTH_WEIGHT,
        "health_weight_sweep": list(HEALTH_WEIGHT_SWEEP),
        "example_block_selection": (
            "prefer blocks exceeding healthy single-stack capacity, then maximum "
            "target transition count and maximum target power"
        ),
        "example_block_id": example_block,
        "checks": checks,
        "maximum_budget_conservation_error": budget_error,
        "known_limits": [
            "fleet progress budget is a normalized scenario setting, not a rate",
            "energy throughput only allocates relative progress; start, stop, ramp, and load-amplitude damage are not identified",
            "three virtual stacks are positions on one LZW theta manifold, not three measured stacks",
            "the real vehicle trace supplies demand only and is not a multi-stack control record",
            "the online health observation is idealized in this feasibility screen",
            "health improvement is partly mechanical because the policy directly minimizes health-weighted loading",
            "the 0.10 health weight is worse than the health-blind controller on final maximum h, so weak weights are not uniformly robust",
            "useful electrical efficiency excludes generation above instantaneous demand",
        ],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def effect(metric, reference=primary_reference):
        row = effects.loc[
            (effects["metric"] == metric) & (effects["reference"] == reference)
        ].iloc[0]
        return (
            float(row["mean_paired_difference"]),
            float(row["bootstrap_ci95_low"]),
            float(row["bootstrap_ci95_high"]),
        )

    max_delta = effect("final_max_health_progress")
    range_delta = effect("final_health_range")
    load_delta = effect("health_weighted_loading")
    efficiency_delta = effect("useful_electrical_efficiency_pct")
    track_delta = effect("tracking_mae_kw")
    report = f"""# Relative health-adaptive real-load replay

## Decision

`{decision}` for an internal, coefficient-free mechanism demonstration only.

The experiment fixes the total fleet health-progress budget for every policy
and allocates that budget by executed stack-energy share.  The health-aware
policy may prefer a less-progressed stack only among actions within
{TRACKING_SLACK_KW:.2f} kW of the best available tracking error.  It therefore
cannot improve the health result by withholding demanded power.

This is not independent method-effectiveness evidence, a physical
degradation-rate model, an action-causal ageing model, or evidence of
real-vehicle lifetime extension.

## Primary paired effects

Primary normalized fleet budget: `{PRIMARY_FLEET_PROGRESS_BUDGET:.3f}`.  Primary
health-loading weight: `{PRIMARY_HEALTH_WEIGHT:.2f}`.  The statistical unit is
one later-time development block after averaging all six assignments of the
same three initial health levels.

| Metric, health reserve minus performance-adaptive IV | Mean difference | Block bootstrap 95% interval |
|---|---:|---:|
| Final maximum `h` | {max_delta[0]:+.6f} | [{max_delta[1]:+.6f}, {max_delta[2]:+.6f}] |
| Final `h` range | {range_delta[0]:+.6f} | [{range_delta[1]:+.6f}, {range_delta[2]:+.6f}] |
| Energy-weighted loaded `h` | {load_delta[0]:+.6f} | [{load_delta[1]:+.6f}, {load_delta[2]:+.6f}] |
| Useful electrical efficiency (percentage points) | {efficiency_delta[0]:+.4f} | [{efficiency_delta[1]:+.4f}, {efficiency_delta[2]:+.4f}] |
| Tracking MAE (kW) | {track_delta[0]:+.4f} | [{track_delta[1]:+.4f}, {track_delta[2]:+.4f}] |

Maximum fleet-budget conservation error across all episodes and policies:
`{budget_error:.3e}`.

## Evidence boundary

- LZW supplies only `h -> theta(h) -> IV/power`.
- The 21UBE0022-derived archive supplies `target_power_kw` demand replay only.
- The total health budget is identical across policies; only its allocation can change.
- Start/stop, ramp, and load-amplitude degradation coefficients are absent.
- The three stacks are virtual health positions on one measured theta manifold.
- February--June 2026 are nested later-time development replay, not an untouched final holdout.
- The policy directly minimizes health-weighted loading, so part of the health
  improvement is mechanical and only verifies the intended redistribution.
- A weight of 0.10 is worse than health-blind control on final maximum `h`;
  weak health weights are not uniformly robust.

The allowed claim is that online health degree can change stack selection and
redistribute a fixed relative health budget while preserving hard constraints.
The prohibited claim is that this proves an independently effective algorithm,
measured efficiency improvement, or reduced physical PEMFC degradation on the
real vehicle.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
