"""Audit the current deterministic action-to-health-to-power chain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.world_model import (
    MultiStackAction,
    WorldModelConfig,
    load_lzw_multistack_world_model,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/online_health_chain_audit"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
CURRENT_SEQUENCE_A = (0.0, 25.0, 90.0, 120.0, 160.0, 195.0, 270.0, 370.0, 0.0)
BLOCK_SECONDS = 60
INITIAL_DAMAGE_FRACTION = 0.40
DIAGNOSTIC_CURRENT_A = 270.0


def build_model():
    return load_lzw_multistack_world_model(
        ROOT,
        n_stacks=1,
        heterogeneity_factors=(1.0,),
        config=WorldModelConfig(
            allowed_currents_a=CURRENT_SEQUENCE_A[:-1],
            min_online_stacks=0,
            max_online_stacks=1,
            power_interface="battery",
        ),
    )


def candidate_action_audit(model) -> pd.DataFrame:
    health_model = model.health_models[0]
    proxy = model.performance_proxies[0]
    damage_reference = proxy.mapping.damage_reference_pct
    damage = INITIAL_DAMAGE_FRACTION * damage_reference
    rows = []
    for current in CURRENT_SEQUENCE_A[:-1]:
        steady_on = current >= 0.0
        steady_rate = health_model.params.load_rate_map.rate_at(current)
        steady_increment = health_model.expected_load_increment(
            current, 1.0, is_on=steady_on
        )
        off_state = model.initial_state(
            degradation_pct=[damage], current_a=[0.0], is_on=[False]
        ).stacks[0].health
        start = health_model.transition(
            off_state,
            current,
            dt_s=1.0,
            stochastic=False,
            next_on=True,
            shift_event=False,
        )
        on_120_state = model.initial_state(
            degradation_pct=[damage], current_a=[120.0], is_on=[True]
        ).stacks[0].health
        shift = health_model.transition(
            on_120_state,
            current,
            dt_s=1.0,
            stochastic=False,
            next_on=True,
            shift_event=current != 120.0,
        )
        after = proxy.evaluate(start.state.degradation, [current])
        rows.append(
            {
                "current_a": current,
                "steady_rate_pct_per_hour": steady_rate,
                "steady_increment_pct_per_s": steady_increment,
                "from_off_total_increment_pct": start.total_increment,
                "from_off_start_increment_pct": start.start_stop_increment,
                "from_120a_total_increment_pct": shift.total_increment,
                "from_120a_shift_increment_pct": shift.shift_increment,
                "damage_after_start_pct": start.state.degradation,
                "cell_voltage_after_start_v": float(after["current_cell_voltage_v"][0]),
                "stack_power_after_start_kw": float(after["stack_power_kw"][0]),
            }
        )
    return pd.DataFrame(rows)


def controlled_online_trace(model) -> pd.DataFrame:
    proxy = model.performance_proxies[0]
    damage_reference = proxy.mapping.damage_reference_pct
    initial_damage = INITIAL_DAMAGE_FRACTION * damage_reference
    state = model.initial_state(
        degradation_pct=[initial_damage], current_a=[0.0], is_on=[False]
    )
    rows = []
    for block, current in enumerate(CURRENT_SEQUENCE_A):
        is_on = current > 0.0
        for offset in range(BLOCK_SECONDS):
            action = MultiStackAction((current,), (is_on,))
            previous = state.stacks[0].health
            shifted = previous.is_on and is_on and current != previous.current_a
            predicted = model.health_models[0].transition(
                previous,
                current,
                dt_s=model.config.dt_s,
                stochastic=False,
                next_on=is_on,
                shift_event=shifted,
            )
            predicted_power = proxy.evaluate(
                predicted.state.degradation, [current], dt_s=model.config.dt_s
            )
            demand = (
                float(predicted_power["stack_power_kw"][0]) if is_on else 0.0
            )
            executed = model.step(
                state,
                action,
                demand,
                stochastic_health=False,
            )
            if not executed.constraints.feasible:
                raise AssertionError(executed.constraints.violations)
            stack = executed.stacks[0]
            diagnostic = proxy.evaluate(
                stack.degradation_after_pct,
                [DIAGNOSTIC_CURRENT_A],
                dt_s=model.config.dt_s,
            )
            rows.append(
                {
                    "step": block * BLOCK_SECONDS + offset,
                    "block": block,
                    "requested_current_a": current,
                    "is_on": is_on,
                    "load_increment_pct": stack.expected_load_increment_pct,
                    "natural_increment_pct": stack.natural_increment_pct,
                    "ramp_increment_pct": stack.ramp_increment_pct,
                    "shift_increment_pct": stack.shift_increment_pct,
                    "start_stop_increment_pct": stack.start_stop_increment_pct,
                    "total_increment_pct": stack.degradation_increment_pct,
                    "damage_pct": stack.degradation_after_pct,
                    "theta_i0_a_per_cm2": stack.theta_reported[0],
                    "theta_ih_a_per_cm2": stack.theta_reported[1],
                    "theta_r_ohm_cm2": stack.theta_reported[2],
                    "actual_cell_voltage_v": stack.cell_voltage_v,
                    "actual_stack_power_kw": stack.power_kw,
                    "diagnostic_cell_voltage_270a_v": float(
                        diagnostic["current_cell_voltage_v"][0]
                    ),
                    "diagnostic_stack_power_270a_kw": float(
                        diagnostic["stack_power_kw"][0]
                    ),
                }
            )
            state = executed.next_state
    return pd.DataFrame(rows)


def plot_audit(candidates: pd.DataFrame, trace: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 6.8,
            "savefig.dpi": 320,
        }
    )
    colors = {
        "load": "#0072B2",
        "shift": "#E69F00",
        "start": "#D55E00",
        "damage": "#333333",
        "i0": "#0072B2",
        "ih": "#009E73",
        "r": "#CC79A7",
        "voltage": "#0072B2",
        "power": "#D55E00",
    }
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.0), constrained_layout=True)

    axes[0, 0].step(
        trace.step,
        trace.requested_current_a,
        where="post",
        color=colors["load"],
        lw=1.1,
    )
    axes[0, 0].set_ylabel("Executed current (A)")
    axes[0, 0].set_xlabel("Time (s)")

    axes[0, 1].plot(
        candidates.current_a,
        candidates.steady_rate_pct_per_hour,
        "o-",
        color=colors["damage"],
        lw=1.0,
        ms=3.5,
    )
    axes[0, 1].axvspan(25, 270, color="#D9D9D9", alpha=0.45, zorder=0)
    axes[0, 1].set_xlabel("Candidate current (A)")
    axes[0, 1].set_ylabel("Steady damage rate (% h$^{-1}$)")

    axes[1, 0].semilogy(
        trace.step,
        np.maximum(trace.load_increment_pct, 1e-12),
        color=colors["load"],
        lw=0.9,
        label="Load amplitude",
    )
    axes[1, 0].semilogy(
        trace.step,
        np.maximum(trace.shift_increment_pct, 1e-12),
        color=colors["shift"],
        lw=0.9,
        label="Load shift",
    )
    axes[1, 0].semilogy(
        trace.step,
        np.maximum(trace.start_stop_increment_pct, 1e-12),
        color=colors["start"],
        lw=0.9,
        label="Start/stop",
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel(r"One-step $\Delta D$ (%)")
    axes[1, 0].legend(frameon=False, ncol=1)

    initial_damage = trace.damage_pct.iloc[0] - trace.total_increment_pct.iloc[0]
    axes[1, 1].plot(
        trace.step,
        1e3 * (trace.damage_pct - initial_damage),
        color=colors["damage"],
        lw=1.1,
    )
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel(r"Cumulative $D-D_0$ ($10^{-3}$ %)")

    theta_columns = (
        ("theta_i0_a_per_cm2", "$i_0$", colors["i0"]),
        ("theta_ih_a_per_cm2", "$i_h$", colors["ih"]),
        ("theta_r_ohm_cm2", "$R$", colors["r"]),
    )
    for column, label, color in theta_columns:
        initial = trace[column].iloc[0]
        relative_ppm = 1e6 * (trace[column] / initial - 1.0)
        axes[2, 0].plot(trace.step, relative_ppm, color=color, lw=1.0, label=label)
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Parameter change (ppm from $t=0$)")
    axes[2, 0].legend(frameon=False, ncol=3)

    power_axis = axes[2, 1].twinx()
    voltage_change_mv = 1e3 * (
        trace.diagnostic_cell_voltage_270a_v
        - trace.diagnostic_cell_voltage_270a_v.iloc[0]
    )
    power_change_w = 1e3 * (
        trace.diagnostic_stack_power_270a_kw
        - trace.diagnostic_stack_power_270a_kw.iloc[0]
    )
    voltage_line = axes[2, 1].plot(
        trace.step,
        voltage_change_mv,
        color=colors["voltage"],
        lw=1.0,
        label=r"$\Delta V$",
    )
    power_line = power_axis.plot(
        trace.step,
        power_change_w,
        color=colors["power"],
        lw=1.0,
        label=r"$\Delta P$",
    )
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel(r"$\Delta V$ at 270 A (mV)", color=colors["voltage"])
    power_axis.set_ylabel(r"$\Delta P$ at 270 A (W)", color=colors["power"])
    axes[2, 1].tick_params(axis="y", colors=colors["voltage"])
    power_axis.tick_params(axis="y", colors=colors["power"])
    axes[2, 1].legend(
        voltage_line + power_line,
        [line.get_label() for line in voltage_line + power_line],
        frameon=False,
        loc="lower left",
    )

    for index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.03,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    model = build_model()
    candidates = candidate_action_audit(model)
    trace = controlled_online_trace(model)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT / "candidate_action_audit.csv", index=False)
    trace.to_csv(OUTPUT / "controlled_online_trace.csv", index=False)
    figure = OUTPUT / "fig33_online_health_chain_audit.png"
    plot_audit(candidates, trace, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())

    params = model.health_models[0].params
    metadata = {
        "status": "diagnostic_current_implementation_not_method_validation",
        "initial_damage_fraction": INITIAL_DAMAGE_FRACTION,
        "initial_damage_pct": float(trace.damage_pct.iloc[0] - trace.total_increment_pct.iloc[0]),
        "final_damage_pct": float(trace.damage_pct.iloc[-1]),
        "damage_increment_pct": float(
            trace.damage_pct.iloc[-1]
            - (trace.damage_pct.iloc[0] - trace.total_increment_pct.iloc[0])
        ),
        "sequence_a": list(CURRENT_SEQUENCE_A),
        "block_seconds": BLOCK_SECONDS,
        "diagnostic_current_a": DIAGNOSTIC_CURRENT_A,
        "steady_rate_map_pct_per_hour": dict(
            zip(params.load_rate_map.current_a, params.load_rate_map.mean_rate_per_hour)
        ),
        "shift_increment_pct": params.shift_increment,
        "start_increment_pct": params.start_increment,
        "stop_increment_pct": params.stop_increment,
        "ramp_increment_per_amp": params.ramp_increment_per_amp,
        "stochastic_health": False,
        "known_failures_exposed": [
            "steady rates are identical from 25 A through 270 A",
            "start cost dominates a short trace",
            "stop and ramp costs are zero",
        ],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
