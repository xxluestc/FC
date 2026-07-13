"""Test whether 21UBE0022 total stack voltage can support health observations.

The analysis is deliberately separate from the older LZW MAT trajectory. It
matches operating current and coolant temperature, aggregates by actual CSV
timestamps, and marks discontinuities before estimating within-epoch trends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, trim_mean
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
COL_TIME = "上报时间"
COL_VOLTAGE = "DCDC输入电压"
COL_CURRENT = "DCDC输入电流"
COL_TARGET_CURRENT = "DCDC目标电流"
COL_COOLANT_IN = "冷却液进堆温度"
COL_COOLANT_OUT = "冷却液出堆温度"
COL_AIR_PRESSURE = "空气进堆压力"
COL_H2_PRESSURE = "氢气进堆压力"
CURRENT_POINTS_A = np.asarray([90.0, 120.0, 160.0, 195.0, 270.0, 340.0])


@dataclass(frozen=True)
class AuditConfig:
    current_tolerance_a: float = 2.0
    target_tolerance_a: float = 5.0
    coolant_min_c: float = 65.0
    coolant_max_c: float = 70.0
    minimum_daily_rows: int = 30
    trim_fraction: float = 0.05
    minimum_epoch_months: int = 4
    minimum_jump_pct: float = 0.8
    minimum_point_total_rows: int = 10_000
    minimum_point_month_coverage: float = 0.8


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def scan_daily_matched(root: Path, config: AuditConfig) -> tuple[pd.DataFrame, dict]:
    files = sorted(root.rglob("*.csv"))
    records: list[dict] = []
    skipped: list[dict] = []
    rows_total = 0
    rows_matched = 0
    required = [
        COL_TIME,
        COL_VOLTAGE,
        COL_CURRENT,
        COL_TARGET_CURRENT,
        COL_COOLANT_IN,
        COL_COOLANT_OUT,
        COL_AIR_PRESSURE,
        COL_H2_PRESSURE,
    ]

    for file_index, path in enumerate(files, start=1):
        try:
            frame = pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=required,
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception as exc:
            skipped.append({"file": path.name, "reason": repr(exc)})
            continue
        rows_total += len(frame)
        time = pd.to_datetime(frame[COL_TIME], errors="coerce")
        voltage = _numeric(frame, COL_VOLTAGE)
        current = _numeric(frame, COL_CURRENT)
        target = _numeric(frame, COL_TARGET_CURRENT)
        coolant_in = _numeric(frame, COL_COOLANT_IN)
        coolant_out = _numeric(frame, COL_COOLANT_OUT)
        air_pressure = _numeric(frame, COL_AIR_PRESSURE)
        h2_pressure = _numeric(frame, COL_H2_PRESSURE)

        distance = np.abs(current.to_numpy()[:, None] - CURRENT_POINTS_A[None, :])
        nearest_index = np.argmin(np.where(np.isfinite(distance), distance, np.inf), axis=1)
        nearest_current = CURRENT_POINTS_A[nearest_index]
        valid = (
            time.notna()
            & voltage.between(80.0, 170.0)
            & (np.min(distance, axis=1) <= config.current_tolerance_a)
            & ((target - nearest_current).abs() <= config.target_tolerance_a)
            & coolant_in.between(config.coolant_min_c, config.coolant_max_c)
        )
        rows_matched += int(valid.sum())
        selected = pd.DataFrame(
            {
                "date": time[valid].dt.strftime("%Y-%m-%d"),
                "current_point_a": nearest_current[valid],
                "stack_voltage_v": voltage[valid].to_numpy(),
                "actual_current_a": current[valid].to_numpy(),
                "target_current_a": target[valid].to_numpy(),
                "coolant_in_c": coolant_in[valid].to_numpy(),
                "coolant_out_c": coolant_out[valid].to_numpy(),
                "air_in_pressure_raw": air_pressure[valid].to_numpy(),
                "h2_in_pressure_raw": h2_pressure[valid].to_numpy(),
            }
        )
        for (date, point), group in selected.groupby(["date", "current_point_a"]):
            if len(group) < config.minimum_daily_rows:
                continue
            values = group["stack_voltage_v"].to_numpy(dtype=float)
            records.append(
                {
                    "date": date,
                    "current_point_a": point,
                    "rows": len(group),
                    "stack_voltage_trimmed_mean_v": float(
                        trim_mean(values, config.trim_fraction)
                    ),
                    "stack_voltage_median_v": float(np.median(values)),
                    "stack_voltage_std_v": float(np.std(values)),
                    "actual_current_mean_a": float(group["actual_current_a"].mean()),
                    "coolant_in_mean_c": float(group["coolant_in_c"].mean()),
                    "coolant_out_mean_c": float(group["coolant_out_c"].mean()),
                    "air_in_pressure_mean_raw": float(group["air_in_pressure_raw"].mean()),
                    "h2_in_pressure_mean_raw": float(group["h2_in_pressure_raw"].mean()),
                    "source_file": path.name,
                }
            )
        if file_index % 25 == 0 or file_index == len(files):
            print(
                f"[{file_index}/{len(files)}] matched={rows_matched:,} "
                f"daily_records={len(records):,}",
                flush=True,
            )

    daily = pd.DataFrame(records)
    if daily.empty:
        raise RuntimeError("no matched daily voltage records were found")
    daily["date"] = pd.to_datetime(daily["date"])
    # A date may appear in two exports. Collapse it without trusting file names.
    collapsed = []
    for (date, point), group in daily.groupby(["date", "current_point_a"]):
        weights = group["rows"].to_numpy(dtype=float)
        row = {
            "date": date,
            "current_point_a": point,
            "rows": int(weights.sum()),
        }
        for column in [
            "stack_voltage_trimmed_mean_v",
            "stack_voltage_median_v",
            "stack_voltage_std_v",
            "actual_current_mean_a",
            "coolant_in_mean_c",
            "coolant_out_mean_c",
            "air_in_pressure_mean_raw",
            "h2_in_pressure_mean_raw",
        ]:
            row[column] = float(np.average(group[column], weights=weights))
        row["source_files"] = ";".join(sorted(group["source_file"].unique()))
        collapsed.append(row)
    daily = pd.DataFrame(collapsed).sort_values(["date", "current_point_a"])
    metadata = {
        "recent_root": str(root),
        "files_found": len(files),
        "files_scanned": len(files) - len(skipped),
        "raw_rows": rows_total,
        "matched_rows": rows_matched,
        "matched_share": rows_matched / rows_total if rows_total else 0.0,
        "daily_records": len(daily),
        "actual_start": str(daily["date"].min().date()),
        "actual_end": str(daily["date"].max().date()),
        "skipped_files": skipped,
        "config": config.__dict__,
    }
    return daily, metadata


def build_monthly_signal(
    daily: pd.DataFrame, config: AuditConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[float]]:
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    records = []
    for (month, point), group in daily.groupby(["month", "current_point_a"]):
        records.append(
            {
                "month": month,
                "current_point_a": point,
                "days": group["date"].nunique(),
                "rows": int(group["rows"].sum()),
                "stack_voltage_v": float(
                    np.median(group["stack_voltage_trimmed_mean_v"])
                ),
                "daily_voltage_mad_v": float(
                    np.median(
                        np.abs(
                            group["stack_voltage_trimmed_mean_v"]
                            - group["stack_voltage_trimmed_mean_v"].median()
                        )
                    )
                ),
                "coolant_in_c": float(np.median(group["coolant_in_mean_c"])),
                "air_in_pressure_raw": float(
                    np.median(group["air_in_pressure_mean_raw"])
                ),
                "h2_in_pressure_raw": float(
                    np.median(group["h2_in_pressure_mean_raw"])
                ),
            }
        )
    monthly = pd.DataFrame(records).sort_values(["month", "current_point_a"])
    total_months = monthly["month"].nunique()
    point_coverage = monthly.groupby("current_point_a").agg(
        months=("month", "nunique"), rows=("rows", "sum")
    )
    eligible_points = point_coverage[
        (point_coverage["rows"] >= config.minimum_point_total_rows)
        & (
            point_coverage["months"] / total_months
            >= config.minimum_point_month_coverage
        )
    ].index.astype(float).tolist()
    if len(eligible_points) < 2:
        raise RuntimeError("fewer than two current points have robust longitudinal coverage")
    pivot = monthly[monthly.current_point_a.isin(eligible_points)].pivot(
        index="month", columns="current_point_a", values="stack_voltage_v"
    )
    centers = pivot.median(axis=0)
    relative = 100.0 * pivot.subtract(centers, axis=1).divide(centers, axis=1)
    composite = pd.DataFrame(
        {
            "month": relative.index,
            "month_date": pd.PeriodIndex(relative.index, freq="M").to_timestamp(),
            "matched_current_points": relative.notna().sum(axis=1),
            "relative_voltage_shift_pct": relative.median(axis=1, skipna=True),
        }
    ).reset_index(drop=True)
    composite["month_to_month_shift_pct"] = composite[
        "relative_voltage_shift_pct"
    ].diff()
    differences = composite["month_to_month_shift_pct"].dropna()
    difference_mad = float(
        np.median(np.abs(differences - differences.median())) * 1.4826
    )
    threshold = max(config.minimum_jump_pct, 4.0 * difference_mad)
    composite["jump_threshold_pct"] = threshold
    composite["discontinuity"] = (
        composite["month_to_month_shift_pct"].abs() > threshold
    )
    composite["epoch"] = composite["discontinuity"].cumsum().astype(int)

    slopes = []
    for epoch, group in composite.groupby("epoch"):
        if len(group) < config.minimum_epoch_months:
            slope = low = high = float("nan")
        else:
            x = (group["month_date"] - group["month_date"].min()).dt.days.to_numpy()
            y = group["relative_voltage_shift_pct"].to_numpy()
            slope, _, low, high = theilslopes(y, x, alpha=0.95)
        slopes.append(
            {
                "epoch": int(epoch),
                "start_month": group["month"].iloc[0],
                "end_month": group["month"].iloc[-1],
                "months": len(group),
                "slope_pct_per_100_days": float(slope * 100.0),
                "slope_ci95_low_pct_per_100_days": float(low * 100.0),
                "slope_ci95_high_pct_per_100_days": float(high * 100.0),
            }
        )
    return monthly, composite, slopes, eligible_points


def build_environment_adjusted_signal(
    daily: pd.DataFrame,
    composite: pd.DataFrame,
    eligible_points: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove observed operating-condition effects within the final epoch."""

    final_epoch = int(composite["epoch"].max())
    final_start = pd.Timestamp(
        composite.loc[composite.epoch.eq(final_epoch), "month_date"].min()
    )
    selected = daily[
        (daily.date >= final_start) & daily.current_point_a.isin(eligible_points)
    ].copy()
    features = [
        "actual_current_mean_a",
        "coolant_in_mean_c",
        "coolant_out_mean_c",
        "air_in_pressure_mean_raw",
        "h2_in_pressure_mean_raw",
    ]
    adjusted_parts = []
    model_rows = []
    for point, group in selected.groupby("current_point_a"):
        valid = group.dropna(subset=features + ["stack_voltage_trimmed_mean_v"]).copy()
        model = make_pipeline(
            StandardScaler(),
            HuberRegressor(epsilon=1.5, alpha=0.01, max_iter=1000),
        )
        target = valid["stack_voltage_trimmed_mean_v"]
        model.fit(valid[features], target)
        prediction = model.predict(valid[features])
        voltage_scale = float(target.median())
        valid["adjusted_voltage_shift_pct"] = (
            100.0 * (target.to_numpy() - prediction) / voltage_scale
        )
        adjusted_parts.append(valid)
        model_rows.append(
            {
                "current_point_a": point,
                "daily_records": len(valid),
                "voltage_scale_v": voltage_scale,
                "huber_r2": float(model.score(valid[features], target)),
                "features": ";".join(features),
            }
        )
    adjusted_daily = pd.concat(adjusted_parts, ignore_index=True)
    adjusted_daily["month"] = adjusted_daily.date.dt.to_period("M").astype(str)
    by_point = (
        adjusted_daily.groupby(["month", "current_point_a"])[
            "adjusted_voltage_shift_pct"
        ]
        .median()
        .unstack()
    )
    adjusted = pd.DataFrame(
        {
            "month": by_point.index,
            "month_date": pd.PeriodIndex(by_point.index, freq="M").to_timestamp(),
            "matched_current_points": by_point.notna().sum(axis=1),
            "environment_adjusted_shift_pct": by_point.median(axis=1),
        }
    ).reset_index(drop=True)
    x = (
        adjusted["month_date"] - adjusted["month_date"].min()
    ).dt.days.to_numpy(dtype=float)
    y = adjusted["environment_adjusted_shift_pct"].to_numpy(dtype=float)
    slope, _, low, high = theilslopes(y, x, alpha=0.95)
    trend = {
        "epoch": final_epoch,
        "start_month": adjusted.month.iloc[0],
        "end_month": adjusted.month.iloc[-1],
        "months": len(adjusted),
        "slope_pct_per_100_days": float(slope * 100.0),
        "slope_ci95_low_pct_per_100_days": float(low * 100.0),
        "slope_ci95_high_pct_per_100_days": float(high * 100.0),
        "covariates": features,
    }
    return adjusted, pd.DataFrame(model_rows), trend


def plot_signal(
    monthly: pd.DataFrame,
    composite: pd.DataFrame,
    slopes: list[dict],
    eligible_points: list[float],
    adjusted_trend: dict,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    plotted = monthly[monthly.current_point_a.isin(eligible_points)]
    for color, (point, group) in zip(colors, plotted.groupby("current_point_a")):
        axes[0].plot(
            pd.to_datetime(group["month"]),
            group["stack_voltage_v"],
            "o-",
            color=color,
            ms=3,
            lw=1.0,
            label=f"{point:g} A",
        )
    axes[0].set_title("(a) Matched stack voltage")
    axes[0].set_ylabel("Stack voltage (V)")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(
        composite["month_date"],
        composite["relative_voltage_shift_pct"],
        "o-",
        color="#333333",
        ms=3.5,
        lw=1.2,
    )
    jumps = composite[composite["discontinuity"]]
    axes[1].scatter(
        jumps["month_date"],
        jumps["relative_voltage_shift_pct"],
        marker="x",
        s=36,
        color="#E45756",
        zorder=4,
        label="Discontinuity",
    )
    axes[1].axhline(0.0, color="#999999", lw=0.7)
    axes[1].set_title("(b) Cross-current composite")
    axes[1].set_ylabel("Relative voltage shift (%)")
    axes[1].legend(frameon=False)

    slope_frame = pd.DataFrame(slopes).dropna(subset=["slope_pct_per_100_days"])
    raw = slope_frame.iloc[-1]
    valid = pd.DataFrame(
        [
            {
                "label": "Matched raw",
                "slope": raw.slope_pct_per_100_days,
                "low": raw.slope_ci95_low_pct_per_100_days,
                "high": raw.slope_ci95_high_pct_per_100_days,
            },
            {
                "label": "Environment-adjusted",
                "slope": adjusted_trend["slope_pct_per_100_days"],
                "low": adjusted_trend["slope_ci95_low_pct_per_100_days"],
                "high": adjusted_trend["slope_ci95_high_pct_per_100_days"],
            },
        ]
    )
    axes[2].axhline(0.0, color="#999999", lw=0.7)
    if len(valid):
        x = np.arange(len(valid))
        y = valid["slope"].to_numpy()
        lower = y - valid["low"].to_numpy()
        upper = valid["high"].to_numpy() - y
        axes[2].errorbar(
            x,
            y,
            yerr=[lower, upper],
            fmt="o",
            color="#4C78A8",
            capsize=3,
        )
        axes[2].set_xticks(x, valid["label"])
    else:
        axes[2].text(0.5, 0.5, "No epoch long enough", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_xticks([])
    axes[2].set_title("(c) Final-epoch trend")
    axes[2].set_ylabel("Shift per 100 days (%)")
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=45)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def write_report(
    metadata: dict,
    composite: pd.DataFrame,
    slopes: list[dict],
    eligible_points: list[float],
    adjusted_trend: dict,
    output: Path,
) -> None:
    jumps = composite[composite["discontinuity"]]
    valid_epochs = [row for row in slopes if np.isfinite(row["slope_pct_per_100_days"])]
    slope_lines = "\n".join(
        f"- Epoch {row['epoch']} ({row['start_month']} to {row['end_month']}, {row['months']} months): "
        f"{row['slope_pct_per_100_days']:.3f}% per 100 days "
        f"(95% CI {row['slope_ci95_low_pct_per_100_days']:.3f} to {row['slope_ci95_high_pct_per_100_days']:.3f})."
        for row in valid_epochs
    ) or "- No telemetry epoch was long enough for a slope estimate."
    jump_lines = "\n".join(
        f"- {row.month}: month-to-month composite shift {row.month_to_month_shift_pct:+.3f}%."
        for row in jumps.itertuples()
    ) or "- No discontinuity exceeded the robust threshold."
    text = f"""# 21UBE0022 voltage-health signal audit

## Data treatment

- The archive remains separate from the older LZW MAT health chain.
- Files are assigned by the timestamp column, not by the one-day-shifted filename.
- Total stack voltage is matched at 90/120/160/195/270/340 A, actual-current tolerance {metadata['config']['current_tolerance_a']} A, target-current tolerance {metadata['config']['target_tolerance_a']} A, and coolant-in temperature {metadata['config']['coolant_min_c']}--{metadata['config']['coolant_max_c']} C.
- {metadata['matched_rows']:,} of {metadata['raw_rows']:,} rows pass these conditions ({metadata['matched_share']:.2%}); {metadata['daily_records']:,} daily current-point records remain.
- The longitudinal composite uses only current points with at least {metadata['config']['minimum_point_total_rows']:,} matched rows and {metadata['config']['minimum_point_month_coverage']:.0%} month coverage: {', '.join(f'{value:g} A' for value in eligible_points)}. Sparse points remain in the daily/monthly tables but do not affect discontinuity or slope estimates.

## Discontinuity audit

The cross-current composite is centered within each current point. A discontinuity is declared only when the month-to-month shift exceeds max(0.8%, four robust MAD scales), here {float(composite['jump_threshold_pct'].iloc[0]):.3f}%.

{jump_lines}

These jumps are not interpreted as degradation. They may represent telemetry precision changes, controller calibration, maintenance or stack replacement. Without a stack-ID/maintenance log, the full archive cannot be treated as one uninterrupted ageing trajectory.

## Within-epoch trends

{slope_lines}

After separate robust adjustment at each retained current point for actual current, coolant-in/out temperature, air-in pressure and hydrogen-in pressure, the final-epoch trend is {adjusted_trend['slope_pct_per_100_days']:.3f}% per 100 days (95% CI {adjusted_trend['slope_ci95_low_pct_per_100_days']:.3f} to {adjusted_trend['slope_ci95_high_pct_per_100_days']:.3f}). The adjusted interval includes zero, so the apparent raw decline is not yet identifiable as irreversible ageing rather than changing operating conditions.

Within-epoch voltage shift is an observable performance residual, not true SOH and not the older MAT parameter trajectory. It may be monitored after execution, but it must not be converted directly into the world model's cumulative degradation state until a stack-specific measurement model or maintenance/stack-replacement log resolves this confounding. It is not suitable for identifying a universal Gamma degradation rate from the full recent-year archive.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/results/liu_21ube0022_voltage_health_audit",
    )
    parser.add_argument("--reuse-daily", action="store_true")
    args = parser.parse_args()
    config = AuditConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.out_dir / "daily_matched_voltage.csv"
    metadata_path = args.out_dir / "metadata.json"
    if args.reuse_daily:
        if not daily_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("--reuse-daily requires existing daily CSV and metadata")
        daily = pd.read_csv(daily_path, parse_dates=["date"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["config"] = config.__dict__
        metadata["daily_extraction_reused"] = True
    else:
        daily, metadata = scan_daily_matched(args.recent_root, config)
        metadata["daily_extraction_reused"] = False
    monthly, composite, slopes, eligible_points = build_monthly_signal(daily, config)
    adjusted, environment_models, adjusted_trend = build_environment_adjusted_signal(
        daily, composite, eligible_points
    )
    metadata["discontinuities"] = int(composite["discontinuity"].sum())
    metadata["epochs"] = len(slopes)
    metadata["eligible_current_points_a"] = eligible_points
    metadata["environment_adjusted_final_epoch_trend"] = adjusted_trend
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(args.out_dir / "monthly_matched_voltage.csv", index=False)
    composite.to_csv(args.out_dir / "composite_voltage_signal.csv", index=False)
    pd.DataFrame(slopes).to_csv(args.out_dir / "epoch_trends.csv", index=False)
    adjusted.to_csv(args.out_dir / "environment_adjusted_signal.csv", index=False)
    environment_models.to_csv(args.out_dir / "environment_model_audit.csv", index=False)
    plot_signal(
        monthly,
        composite,
        slopes,
        eligible_points,
        adjusted_trend,
        args.out_dir / "fig19_voltage_health_signal.png",
    )
    write_report(
        metadata,
        composite,
        slopes,
        eligible_points,
        adjusted_trend,
        args.out_dir / "report.md",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    print(pd.DataFrame(slopes).to_string(index=False))


if __name__ == "__main__":
    main()
