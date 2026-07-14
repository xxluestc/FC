"""Screen whether LZW exposure can calibrate an action-resolved aging law.

This is a stop-gated diagnostic. It reconstructs interval exposure from the
raw current/voltage samples and never uses the flawed cumulative time columns
in the upstream event table. Zuo is not used for any degradation coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from fc_power.lzw_iv_model import iv_model, reported_to_model_theta


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LZW_ROOT = ROOT.parent
EVENTS = ROOT / "data/upstream_lzw/canonical_event_table_6104.csv"
CONDITIONS = ROOT / "data/upstream_lzw/current_point_cost_conditions.json"
OUTPUT = ROOT / "data/results/lzw_overall_exposure_screen"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"

REFERENCE_CURRENT_A = 370.0
HEALTH_BASELINE_EVENTS = 50
DELETION_START = 3700
DELETION_STOP = 3900
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260720

MODEL_FEATURES = {
    "zero_increment": (),
    "constant_segment": ("constant",),
    "event_count_clock": ("event_count",),
    "elapsed_clock": ("elapsed_samples",),
    "stack_on_clock": ("stack_on_samples",),
    "load_on_clock": ("load_on_samples",),
    "charge_throughput": ("charge_ampere_samples",),
    "stack_on_plus_charge": (
        "stack_on_samples",
        "charge_ampere_samples",
    ),
}
TIME_MODELS = ("elapsed_clock", "stack_on_clock", "load_on_clock")
LOAD_MODELS = ("charge_throughput", "stack_on_plus_charge")
PHYSICAL_MODELS = TIME_MODELS + LOAD_MODELS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_reference_health_loss(events: pd.DataFrame, conditions: dict) -> np.ndarray:
    """Return normalized power-capability loss at one observed current point."""

    area = float(conditions["active_area_cm2"])
    theta_reported = events[["i0", "ih", "R_ohm"]].to_numpy(dtype=float)
    theta_model = reported_to_model_theta(theta_reported, area)
    healthy_theta = theta_model[:HEALTH_BASELINE_EVENTS].mean(axis=0)
    kwargs = {
        "temperature_c": float(conditions["T_ref_C"]),
        "current_density_a_cm2": REFERENCE_CURRENT_A / area,
        "a": float(conditions["a_ref"]),
        "b": float(conditions["b_ref"]),
        "inner": [
            float(conditions["B"]),
            float(conditions["i_lim_A_cm2"]),
        ],
        "active_area_cm2": area,
    }
    healthy_voltage = float(iv_model(theta_model=healthy_theta, **kwargs))
    event_voltage = iv_model(theta_model=theta_model, **kwargs)
    return (healthy_voltage - event_voltage) / healthy_voltage


def build_transition_table(
    events: pd.DataFrame,
    health_loss: np.ndarray,
    raw_data: np.ndarray,
) -> pd.DataFrame:
    """Aggregate realized exposure between consecutive run-segment anchors."""

    first_positions = (
        events.groupby("source_segment_row_1based", sort=False).head(1).index.to_numpy()
    )
    anchors = events.loc[
        first_positions,
        [
            "source_segment_row_1based",
            "raw_event_start_index_1based",
            "original_index",
        ],
    ].reset_index(drop=True)
    anchor_health = np.asarray(health_loss, dtype=float)[first_positions]
    rows = []
    for index in range(len(anchors) - 1):
        raw_start = int(anchors.raw_event_start_index_1based.iloc[index] - 1)
        raw_stop = int(anchors.raw_event_start_index_1based.iloc[index + 1] - 1)
        current_raw = np.asarray(raw_data[raw_start:raw_stop, 1], dtype=float)
        voltage_raw = np.asarray(raw_data[raw_start:raw_stop, 0], dtype=float)
        current_valid = np.isfinite(current_raw)
        voltage_valid = np.isfinite(voltage_raw)
        current = np.where(current_valid & (current_raw > 5.0), current_raw, 0.0)
        voltage_on = voltage_valid & (voltage_raw > 5.0)
        load_on = current > 0.0
        original_start = int(anchors.original_index.iloc[index])
        original_stop = int(anchors.original_index.iloc[index + 1])
        crosses_deletion = (
            original_start <= DELETION_START - 1
            and original_stop >= DELETION_STOP + 1
        )
        rows.append(
            {
                "transition_id": index,
                "source_segment_start": int(
                    anchors.source_segment_row_1based.iloc[index]
                ),
                "source_segment_stop": int(
                    anchors.source_segment_row_1based.iloc[index + 1]
                ),
                "raw_start_index_1based": raw_start + 1,
                "raw_stop_index_1based_exclusive": raw_stop + 1,
                "original_event_start": original_start,
                "original_event_stop": original_stop,
                "event_count": int(
                    first_positions[index + 1] - first_positions[index]
                ),
                "elapsed_samples": raw_stop - raw_start,
                "stack_on_samples": int(voltage_on.sum()),
                "load_on_samples": int(load_on.sum()),
                "charge_ampere_samples": float(current.sum()),
                "mean_loaded_current_A": (
                    float(current[load_on].mean()) if load_on.any() else 0.0
                ),
                "nonfinite_current_samples": int((~current_valid).sum()),
                "health_loss_start": float(anchor_health[index]),
                "health_loss_stop": float(anchor_health[index + 1]),
                "delta_health_loss": float(
                    anchor_health[index + 1] - anchor_health[index]
                ),
                "crosses_deleted_event_gap": crosses_deletion,
                "eligible_for_model": not crosses_deletion,
            }
        )
    table = pd.DataFrame(rows)
    table["constant"] = 1.0
    return table


def fit_nonnegative(
    frame: pd.DataFrame,
    target: np.ndarray,
    features: tuple[str, ...],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a no-intercept nonnegative exposure model with robust scaling."""

    if not features:
        return np.zeros(0, dtype=float), np.zeros(len(frame), dtype=float)
    design = frame.loc[:, list(features)].to_numpy(dtype=float)
    train_design = design[train_indices]
    scale = np.median(train_design, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    scaled_coefficients, _ = nnls(
        train_design / scale,
        np.asarray(target, dtype=float)[train_indices],
    )
    coefficients = scaled_coefficients / scale
    return coefficients, design @ coefficients


def residual_lag1(residual: np.ndarray) -> float:
    if len(residual) < 3 or np.std(residual[:-1]) == 0 or np.std(residual[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(residual[:-1], residual[1:])[0, 1])


def evaluate_split(
    frame: pd.DataFrame,
    target: np.ndarray,
    split_name: str,
    train_stop: int,
    eval_start: int,
    eval_stop: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    train_indices = np.arange(train_stop)
    eval_indices = np.arange(eval_start, eval_stop)
    zero_rmse = float(np.sqrt(np.mean(np.square(target[eval_indices]))))
    zero_mae = float(np.mean(np.abs(target[eval_indices])))
    metric_rows = []
    coefficient_rows = []
    predictions = {}
    for model, features in MODEL_FEATURES.items():
        coefficients, prediction = fit_nonnegative(
            frame, target, features, train_indices
        )
        predictions[model] = prediction
        residual = target[eval_indices] - prediction[eval_indices]
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))
        metric_rows.append(
            {
                "split": split_name,
                "model": model,
                "train_rows": len(train_indices),
                "eval_rows": len(eval_indices),
                "eval_start": eval_start,
                "eval_stop_exclusive": eval_stop,
                "rmse": rmse,
                "mae": mae,
                "rmse_improvement_vs_zero": 1.0 - rmse / zero_rmse,
                "mae_improvement_vs_zero": 1.0 - mae / zero_mae,
                "r2_zero_increment": 1.0 - (rmse / zero_rmse) ** 2,
                "residual_lag1": residual_lag1(residual),
                "prediction_mean": float(prediction[eval_indices].mean()),
                "target_mean": float(target[eval_indices].mean()),
            }
        )
        for feature, coefficient in zip(features, coefficients):
            coefficient_rows.append(
                {
                    "split": split_name,
                    "model": model,
                    "feature": feature,
                    "coefficient": float(coefficient),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows), predictions


def moving_block_improvement_ci(
    model_error: np.ndarray,
    reference_error: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float]:
    """Moving-block interval for paired relative RMSE improvement."""

    model_error = np.asarray(model_error, dtype=float)
    reference_error = np.asarray(reference_error, dtype=float)
    if model_error.shape != reference_error.shape or model_error.ndim != 1:
        raise ValueError("paired errors must be one-dimensional and equally sized")
    n = len(model_error)
    block_length = max(5, int(round(np.sqrt(n))))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + np.arange(block_length)) % n)
        selected = np.asarray(indices[:n], dtype=int)
        model_rmse = np.sqrt(np.mean(np.square(model_error[selected])))
        reference_rmse = np.sqrt(np.mean(np.square(reference_error[selected])))
        samples[draw] = 1.0 - model_rmse / reference_rmse
    observed = 1.0 - np.sqrt(np.mean(np.square(model_error))) / np.sqrt(
        np.mean(np.square(reference_error))
    )
    return {
        "observed": float(observed),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "block_length": block_length,
        "resamples": n_resamples,
    }


def select_on_validation(validation_metrics: pd.DataFrame) -> dict[str, str]:
    indexed = validation_metrics.set_index("model")
    best_time = min(TIME_MODELS, key=lambda name: indexed.loc[name, "rmse"])
    best_load = min(LOAD_MODELS, key=lambda name: indexed.loc[name, "rmse"])
    best_physical = min(
        PHYSICAL_MODELS, key=lambda name: indexed.loc[name, "rmse"]
    )
    return {
        "best_time_model": best_time,
        "best_load_model": best_load,
        "best_physical_model": best_physical,
    }


def make_figure(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    selection: dict[str, str],
    rolling: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "time": "#0072B2",
        "load": "#D55E00",
        "event": "#009E73",
        "observed": "#2F3437",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.0))

    test_start = int(round(0.85 * len(table)))
    x = np.arange(test_start, len(table) + 1)
    observed_path = np.r_[
        table.health_loss_start.iloc[test_start],
        table.health_loss_start.iloc[test_start]
        + np.cumsum(table.delta_health_loss.iloc[test_start:].to_numpy()),
    ]
    axes[0, 0].plot(x, observed_path * 100, color=colors["observed"], lw=1.2, label="Observed")
    for model, color, label in (
        (selection["best_time_model"], colors["time"], "Time exposure"),
        (selection["best_load_model"], colors["load"], "Load exposure"),
        ("event_count_clock", colors["event"], "Event-count control"),
    ):
        path = np.r_[
            observed_path[0],
            observed_path[0] + np.cumsum(predictions[model][test_start:]),
        ]
        axes[0, 0].plot(x, path * 100, color=color, lw=1.0, label=label)
    axes[0, 0].set_xlabel("Segment transition")
    axes[0, 0].set_ylabel("Normalized 370 A power loss (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.2)

    test_metrics = metrics[metrics.split.eq("train85_test15")].set_index("model")
    shown = [
        "constant_segment",
        "event_count_clock",
        selection["best_time_model"],
        selection["best_load_model"],
    ]
    labels = ["Constant", "Event count", "Best time", "Best load"]
    values = [test_metrics.loc[name, "rmse_improvement_vs_zero"] * 100 for name in shown]
    axes[0, 1].barh(
        np.arange(len(shown)),
        values,
        color=["#999999", colors["event"], colors["time"], colors["load"]],
    )
    axes[0, 1].axvline(0, color="#6C757D", lw=0.8)
    axes[0, 1].set_yticks(np.arange(len(shown)), labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Test RMSE improvement vs zero (%)")

    corr_columns = [
        "event_count",
        "elapsed_samples",
        "stack_on_samples",
        "load_on_samples",
        "charge_ampere_samples",
    ]
    corr = table.loc[: test_start - 1, corr_columns].corr().to_numpy()
    image = axes[1, 0].imshow(corr, vmin=0.85, vmax=1.0, cmap="viridis")
    short = ["Events", "Elapsed", "Stack on", "Loaded", "Charge"]
    axes[1, 0].set_xticks(range(len(short)), short, rotation=35, ha="right")
    axes[1, 0].set_yticks(range(len(short)), short)
    for row in range(len(short)):
        for column in range(len(short)):
            axes[1, 0].text(
                column,
                row,
                f"{corr[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if corr[row, column] < 0.94 else "black",
            )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.03)

    rolling_plot = rolling.sort_values("train_fraction")
    axes[1, 1].plot(
        rolling_plot.train_fraction * 100,
        rolling_plot.load_vs_time_rmse_improvement * 100,
        marker="o",
        ms=3.8,
        color=colors["load"],
        lw=1.0,
    )
    axes[1, 1].axhline(0, color="#6C757D", lw=0.8, ls="--")
    axes[1, 1].axhline(10, color="#0072B2", lw=0.8, ls=":")
    axes[1, 1].set_xlabel("Expanding training endpoint (%)")
    axes[1, 1].set_ylabel("Load vs time RMSE gain (%)")

    for index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lzw-root", type=Path, default=DEFAULT_LZW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    raw_path = args.lzw_root / "converted_data/raw/data_full.npy"
    events = pd.read_csv(EVENTS)
    conditions = json.loads(CONDITIONS.read_text(encoding="utf-8"))
    health_loss = fixed_reference_health_loss(events, conditions)
    raw_data = np.load(raw_path, mmap_mode="r")
    complete_table = build_transition_table(events, health_loss, raw_data)
    table = complete_table[complete_table.eligible_for_model].reset_index(drop=True)
    target = table.delta_health_loss.to_numpy(dtype=float)

    n = len(table)
    train70 = int(round(0.70 * n))
    train85 = int(round(0.85 * n))
    validation_metrics, validation_coefficients, _ = evaluate_split(
        table, target, "train70_validation15", train70, train70, train85
    )
    selection = select_on_validation(validation_metrics)
    test_metrics, test_coefficients, test_predictions = evaluate_split(
        table, target, "train85_test15", train85, train85, n
    )

    rolling_metrics = []
    rolling_coefficients = []
    rolling_rows = []
    for fraction in (0.60, 0.70, 0.80, 0.90):
        train_stop = int(round(fraction * n))
        eval_stop = min(n, train_stop + int(round(0.10 * n)))
        split_name = f"rolling_train{int(fraction * 100)}_next10"
        split_metrics, split_coefficients, _ = evaluate_split(
            table,
            target,
            split_name,
            train_stop,
            train_stop,
            eval_stop,
        )
        rolling_metrics.append(split_metrics)
        rolling_coefficients.append(split_coefficients)
        indexed = split_metrics.set_index("model")
        time_rmse = float(indexed.loc[selection["best_time_model"], "rmse"])
        load_rmse = float(indexed.loc[selection["best_load_model"], "rmse"])
        rolling_rows.append(
            {
                "split": split_name,
                "train_fraction": fraction,
                "best_time_model": selection["best_time_model"],
                "best_load_model": selection["best_load_model"],
                "load_vs_time_rmse_improvement": 1.0 - load_rmse / time_rmse,
            }
        )
    rolling = pd.DataFrame(rolling_rows)
    metrics = pd.concat(
        [validation_metrics, test_metrics, *rolling_metrics], ignore_index=True
    )
    coefficients = pd.concat(
        [validation_coefficients, test_coefficients, *rolling_coefficients],
        ignore_index=True,
    )

    test_indices = np.arange(train85, n)
    selected_time = selection["best_time_model"]
    selected_load = selection["best_load_model"]
    selected_physical = selection["best_physical_model"]
    load_vs_time_ci = moving_block_improvement_ci(
        target[test_indices] - test_predictions[selected_load][test_indices],
        target[test_indices] - test_predictions[selected_time][test_indices],
        seed=BOOTSTRAP_SEED,
    )
    physical_vs_zero_ci = moving_block_improvement_ci(
        target[test_indices] - test_predictions[selected_physical][test_indices],
        target[test_indices],
        seed=BOOTSTRAP_SEED + 1,
    )
    test_indexed = test_metrics.set_index("model")
    train_correlation = float(
        table.loc[: train85 - 1, "stack_on_samples"].corr(
            table.loc[: train85 - 1, "charge_ampere_samples"]
        )
    )
    selected_load_coefficients = test_coefficients[
        test_coefficients.model.eq(selected_load)
    ]
    charge_coefficient = float(
        selected_load_coefficients.loc[
            selected_load_coefficients.feature.eq("charge_ampere_samples"),
            "coefficient",
        ].iloc[0]
    )
    gates = {
        "physical_trend_improves_zero_by_10pct": (
            physical_vs_zero_ci["observed"] >= 0.10
            and physical_vs_zero_ci["ci95_low"] > 0.0
        ),
        "load_improves_time_by_10pct": (
            load_vs_time_ci["observed"] >= 0.10
            and load_vs_time_ci["ci95_low"] > 0.0
        ),
        "charge_coefficient_retained": charge_coefficient > 0.0,
        "load_beats_time_in_three_of_four_rolling_windows": int(
            (rolling.load_vs_time_rmse_improvement > 0).sum()
        )
        >= 3,
        "event_count_control_not_better_than_physical": float(
            test_indexed.loc["event_count_clock", "rmse"]
        )
        >= float(test_indexed.loc[selected_physical, "rmse"]),
        "stack_on_charge_correlation_below_0_95": abs(train_correlation) < 0.95,
        "sampling_period_verified": False,
    }
    trend_only_pass = bool(gates["physical_trend_improves_zero_by_10pct"])
    action_model_pass = bool(all(gates.values()))
    decision = {
        "status": (
            "action_resolved_model_supported"
            if action_model_pass
            else "rejected_for_action_resolved_degradation"
        ),
        "trend_only_pass": trend_only_pass,
        "action_model_pass": action_model_pass,
        "selection_on_validation": selection,
        "gates": gates,
        "diagnostics": {
            "eligible_segment_transitions": n,
            "excluded_deleted_gap_transitions": int(
                complete_table.crosses_deleted_event_gap.sum()
            ),
            "positive_target_increment_share": float((target > 0).mean()),
            "stack_on_charge_train_correlation": train_correlation,
            "load_beats_time_rolling_windows": int(
                (rolling.load_vs_time_rmse_improvement > 0).sum()
            ),
            "load_vs_time_test": load_vs_time_ci,
            "best_physical_vs_zero_test": physical_vs_zero_ci,
            "event_count_test_rmse": float(
                test_indexed.loc["event_count_clock", "rmse"]
            ),
            "best_physical_test_rmse": float(
                test_indexed.loc[selected_physical, "rmse"]
            ),
        },
        "allowed_role": (
            "normalized endpoint/trend sensitivity only; not an online rate law"
        ),
        "zuo_role": "system topology and random-load generation only",
        "controller_changed": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    complete_table.to_csv(args.out_dir / "segment_transition_exposure.csv", index=False)
    metrics.to_csv(args.out_dir / "model_metrics.csv", index=False)
    coefficients.to_csv(args.out_dir / "model_coefficients.csv", index=False)
    rolling.to_csv(args.out_dir / "rolling_load_vs_time.csv", index=False)
    (args.out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = {
        "events": {"path": str(EVENTS), "sha256": sha256_file(EVENTS)},
        "conditions": {
            "path": str(CONDITIONS),
            "sha256": sha256_file(CONDITIONS),
        },
        "raw_data": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "reference_current_A": REFERENCE_CURRENT_A,
        "health_baseline_events": HEALTH_BASELINE_EVENTS,
        "sampling_unit": "raw samples; seconds are not asserted",
        "future_demand_used": False,
        "zuo_degradation_parameters_used": False,
    }
    (args.out_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    figure_path = args.out_dir / "fig31_lzw_overall_exposure_screen.png"
    make_figure(
        table,
        metrics,
        test_predictions,
        selection,
        rolling,
        figure_path,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure_path.name).write_bytes(figure_path.read_bytes())

    report = f"""# LZW overall-exposure stop-gate screen

## Decision

`{decision['status']}`. The physical exposure trend can be retained only as an
endpoint/trend sensitivity. It is not accepted as an action-resolved online
degradation law, and no controller code was changed.

## Primary chronological test

- Validation-selected time model: `{selected_time}`
- Validation-selected load model: `{selected_load}`
- Validation-selected physical model: `{selected_physical}`
- Best physical vs zero RMSE improvement: {physical_vs_zero_ci['observed']:.3%}
  (moving-block 95% interval {physical_vs_zero_ci['ci95_low']:.3%} to
  {physical_vs_zero_ci['ci95_high']:.3%})
- Load vs time RMSE improvement: {load_vs_time_ci['observed']:.3%}
  (moving-block 95% interval {load_vs_time_ci['ci95_low']:.3%} to
  {load_vs_time_ci['ci95_high']:.3%})
- Stack-on/charge correlation in the first 85%: {train_correlation:.6f}
- Load beats time in {int((rolling.load_vs_time_rmse_improvement > 0).sum())}/4
  expanding-window checks.

## Interpretation boundary

The LZW theta/IV chain supplies the health target. Raw LZW current and voltage
samples supply exposure. Zuo supplies neither a coefficient nor a health
target in this screen. A high cumulative fit is not accepted when event-count
controls are stronger, load adds no stable value beyond time, exposure terms
are collinear, or the raw sampling period is unverified.
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
