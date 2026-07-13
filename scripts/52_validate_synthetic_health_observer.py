"""Validate the health observer with a synthetic, explicitly non-vehicle chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.health import (
    DegradationObservation,
    GaussianDegradationObserver,
    GammaHealthState,
)
from fc_power.world_model import load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/synthetic_health_observer"
FIGURE = (
    ROOT
    / "data/results/figures/fc_only_foundation/fig17_synthetic_health_observer.png"
)
HOURS = 720
DT_S = 3600.0
OBSERVATION_INTERVAL_H = 24
OBSERVATION_STD_PCT = 0.03
TRUTH_HETEROGENEITY = 1.15
SEED = 20260713


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def operating_current(hour: int) -> float:
    """Predeclared repeating load with a weekly four-hour rest event."""

    hour_of_week = hour % 168
    if hour_of_week < 4:
        return 0.0
    level = (hour // 8) % 3
    return (90.0, 195.0, 270.0)[level]


def simulate() -> tuple[pd.DataFrame, dict]:
    nominal_model = load_lzw_multistack_world_model(ROOT, n_stacks=1)
    truth_model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=1,
        heterogeneity_factors=(TRUTH_HETEROGENEITY,),
    )
    nominal = nominal_model.health_models[0]
    truth = truth_model.health_models[0]
    observer = GaussianDegradationObserver(
        gamma_scale_pct=nominal.params.gamma_scale,
        initial_variance_pct2=OBSERVATION_STD_PCT**2,
        process_variance_rate_pct2_per_hour=2e-5,
    )
    truth_state = GammaHealthState()
    open_loop_state = GammaHealthState()
    belief = observer.initialize(GammaHealthState())
    rng = np.random.default_rng(SEED)
    rows = []

    for hour in range(1, HOURS + 1):
        current_a = operating_current(hour - 1)
        on = current_a > 0
        truth_transition = truth.transition(
            truth_state,
            current_a,
            dt_s=DT_S,
            stochastic=False,
            next_on=on,
        )
        open_loop_transition = nominal.transition(
            open_loop_state,
            current_a,
            dt_s=DT_S,
            stochastic=False,
            next_on=on,
        )
        prediction_transition = nominal.transition(
            belief.state,
            current_a,
            dt_s=DT_S,
            stochastic=False,
            next_on=on,
        )
        observation = None
        observation_value = np.nan
        if hour % OBSERVATION_INTERVAL_H == 0:
            observation_value = max(
                0.0,
                truth_transition.state.degradation
                + rng.normal(0.0, OBSERVATION_STD_PCT),
            )
            observation = DegradationObservation(
                degradation_pct=float(observation_value),
                variance_pct2=OBSERVATION_STD_PCT**2,
                elapsed_s=hour * DT_S,
                source="synthetic-direct-degradation-proxy",
                synthetic=True,
            )
        update = observer.update(
            belief,
            prediction_transition.state,
            expected_gamma_increment_pct=(
                prediction_transition.expected_load_increment
            ),
            observation=observation,
        )
        truth_state = truth_transition.state
        open_loop_state = open_loop_transition.state
        belief = update.posterior
        rows.append(
            {
                "hour": hour,
                "current_a": current_a,
                "truth_degradation_pct": truth_state.degradation,
                "open_loop_degradation_pct": open_loop_state.degradation,
                "prediction_degradation_pct": (
                    update.prediction.state.degradation
                ),
                "posterior_degradation_pct": belief.state.degradation,
                "posterior_std_pct": np.sqrt(belief.variance_pct2),
                "observation_degradation_pct": observation_value,
                "corrected": observation is not None,
                "monotonic_projection": bool(
                    update.correction
                    and update.correction.monotonic_projection_applied
                ),
            }
        )

    frame = pd.DataFrame(rows)
    truth_values = frame.truth_degradation_pct.to_numpy(dtype=float)
    open_error = frame.open_loop_degradation_pct.to_numpy(dtype=float) - truth_values
    posterior_error = (
        frame.posterior_degradation_pct.to_numpy(dtype=float) - truth_values
    )
    posterior_std = frame.posterior_std_pct.to_numpy(dtype=float)
    metrics = {
        "hours": HOURS,
        "observation_interval_h": OBSERVATION_INTERVAL_H,
        "observation_count": int(frame.corrected.sum()),
        "synthetic_truth_heterogeneity_factor": TRUTH_HETEROGENEITY,
        "observation_std_pct": OBSERVATION_STD_PCT,
        "open_loop_rmse_pct": float(np.sqrt(np.mean(open_error**2))),
        "posterior_rmse_pct": float(np.sqrt(np.mean(posterior_error**2))),
        "rmse_reduction_fraction": float(
            1.0
            - np.sqrt(np.mean(posterior_error**2))
            / np.sqrt(np.mean(open_error**2))
        ),
        "open_loop_terminal_abs_error_pct": float(abs(open_error[-1])),
        "posterior_terminal_abs_error_pct": float(abs(posterior_error[-1])),
        "posterior_95pct_interval_coverage": float(
            np.mean(np.abs(posterior_error) <= 1.96 * posterior_std)
        ),
        "monotonic_projection_count": int(frame.monotonic_projection.sum()),
    }
    if metrics["posterior_rmse_pct"] >= metrics["open_loop_rmse_pct"]:
        raise AssertionError("synthetic observer did not reduce model-drift RMSE")
    return frame, metrics


def plot(frame: pd.DataFrame) -> None:
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
    hours = frame.hour.to_numpy(dtype=float)
    posterior = frame.posterior_degradation_pct.to_numpy(dtype=float)
    uncertainty = 1.96 * frame.posterior_std_pct.to_numpy(dtype=float)
    truth = frame.truth_degradation_pct.to_numpy(dtype=float)
    observed = frame.corrected.to_numpy(dtype=bool)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.3, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.6, 1.0)},
    )
    axes[0].fill_between(
        hours,
        np.maximum(0.0, posterior - uncertainty),
        posterior + uncertainty,
        color="#457B9D",
        alpha=0.16,
        linewidth=0,
        label="Posterior 95% interval",
    )
    axes[0].plot(hours, truth, color="#1D3557", linewidth=1.3, label="Synthetic truth")
    axes[0].plot(
        hours,
        frame.open_loop_degradation_pct,
        color="#E76F51",
        linewidth=1.0,
        linestyle="--",
        label="Open-loop nominal",
    )
    axes[0].plot(
        hours,
        posterior,
        color="#2A9D8F",
        linewidth=1.1,
        label="Corrected belief",
    )
    axes[0].scatter(
        hours[observed],
        frame.loc[observed, "observation_degradation_pct"],
        s=10,
        facecolor="white",
        edgecolor="#6D597A",
        linewidth=0.7,
        zorder=4,
        label="24 h synthetic observation",
    )
    axes[0].set_ylabel("Cumulative degradation (%-point)")
    axes[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left")

    axes[1].plot(
        hours,
        np.abs(frame.open_loop_degradation_pct - truth),
        color="#E76F51",
        linewidth=1.0,
        linestyle="--",
        label="Open-loop absolute error",
    )
    axes[1].plot(
        hours,
        np.abs(posterior - truth),
        color="#2A9D8F",
        linewidth=1.0,
        label="Posterior absolute error",
    )
    axes[1].set_ylabel("Absolute error (%-point)")
    axes[1].set_xlabel("Synthetic operating time (h)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.075,
            1.02,
            label,
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.8, h_pad=1.0)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_outputs(frame: pd.DataFrame, metrics: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.round(12).to_csv(OUTPUT / "trajectory.csv", index=False)
    calibration = ROOT / "data/results/health/lzw_gamma_calibration.json"
    metadata = {
        "status": "synthetic interface validation only",
        "real_vehicle_observer_complete": False,
        "observation_semantics": (
            "direct noisy degradation proxy; not measured voltage/current and not "
            "linked to 21UBE0022"
        ),
        "truth_process": (
            "deterministic conditional mean with a predeclared 15% stack "
            "heterogeneity; Gamma variance is propagated by the observer"
        ),
        "random_seed": SEED,
        "parameters": {
            "hours": HOURS,
            "dt_s": DT_S,
            "observation_interval_h": OBSERVATION_INTERVAL_H,
            "observation_std_pct": OBSERVATION_STD_PCT,
            "truth_heterogeneity_factor": TRUTH_HETEROGENEITY,
        },
        "source_sha256": {str(calibration.relative_to(ROOT)): sha256(calibration)},
        "metrics": metrics,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# Synthetic health-observer interface validation

This experiment validates software timing and uncertainty propagation only. The observation is a noisy direct degradation proxy generated from a deterministic conditional-mean truth model with a {TRUTH_HETEROGENEITY:.2f} heterogeneity factor. Gamma uncertainty is propagated in the observer belief rather than sampled into the synthetic truth. This is not a 21UBE0022 measurement and does not validate an online SOH posterior.

- Horizon: {HOURS} h; observation interval: {OBSERVATION_INTERVAL_H} h.
- Open-loop RMSE: {metrics['open_loop_rmse_pct']:.6f} %-point.
- Corrected-belief RMSE: {metrics['posterior_rmse_pct']:.6f} %-point.
- Synthetic RMSE reduction: {100 * metrics['rmse_reduction_fraction']:.2f}%.
- Posterior 95% interval coverage: {100 * metrics['posterior_95pct_interval_coverage']:.2f}%.
- Monotonic projections: {metrics['monotonic_projection_count']}.

The only admissible conclusion is that the explicit `predict -> execute -> correct -> next decision` interface can reduce an injected model drift under its synthetic direct-observation assumption. Real correction remains blocked on a vehicle/stack-linked MAT observation chain and a validated voltage/current-to-health measurement model.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    frame, metrics = simulate()
    plot(frame)
    write_outputs(frame, metrics)
    print(json.dumps(metrics, indent=2))
    print(FIGURE.relative_to(ROOT))


if __name__ == "__main__":
    main()
