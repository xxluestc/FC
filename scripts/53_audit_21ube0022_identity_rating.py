"""Audit 21UBE0022 source identity and its empirical FC power envelope.

The recent-year archive is read only. It is never concatenated with the current
development canonical table. Outputs are aggregate evidence under data/results.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import heapq
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COL_TIME = "上报时间"
COL_STATE = "燃料电池状态(自动控制模式下)"
COL_TARGET = "目标功率"
COL_FC_V = "DCDC输入电压"
COL_FC_I = "DCDC输入电流"
COL_OUT_V = "DCDC输出电压"
COL_OUT_I = "DCDC输出电流"
COL_TARGET_I = "DCDC目标电流"
COL_LOADABLE = "可加载功率"
COL_CELL_MEAN = "节电压平均值"
COL_CELL_MIN = "最低节电压"
COL_CELL_MAX = "最高节电压"

CORE_COLUMNS = [COL_TIME, COL_TARGET, COL_FC_V, COL_FC_I]
OPTIONAL_COLUMNS = [
    COL_STATE,
    COL_OUT_V,
    COL_OUT_I,
    COL_TARGET_I,
    COL_LOADABLE,
    COL_CELL_MEAN,
    COL_CELL_MIN,
    COL_CELL_MAX,
]
METRIC_RANGES = {
    "target_power_kw": (-100.0, 150.0),
    "fc_input_power_kw": (-100.0, 150.0),
    "dcdc_output_power_kw": (-100.0, 150.0),
    # The source does not define the unit of this field. Keep it as a raw
    # diagnostic and never present it as kW evidence.
    "loadable_raw": (-100.0, 5_000.0),
    "fc_current_a": (-500.0, 1000.0),
}


@dataclass
class StreamingHistogram:
    low: float
    high: float
    bins: int = 25_000
    counts: np.ndarray = field(init=False)
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    underflow: int = 0
    overflow: int = 0

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.bins, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += int(len(values))
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.underflow += int((values < self.low).sum())
        self.overflow += int((values > self.high).sum())
        clipped = np.clip(values, self.low, np.nextafter(self.high, self.low))
        index = ((clipped - self.low) / (self.high - self.low) * self.bins).astype(int)
        self.counts += np.bincount(index, minlength=self.bins)

    def quantile(self, q: float) -> float:
        if not self.count:
            return float("nan")
        rank = max(1, int(np.ceil(q * self.count)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        width = (self.high - self.low) / self.bins
        estimate = self.low + (index + 0.5) * width
        return float(np.clip(estimate, self.minimum, self.maximum))

    def summary(self) -> dict[str, float | int]:
        mean = self.total / self.count if self.count else float("nan")
        variance = self.total_sq / self.count - mean**2 if self.count else float("nan")
        return {
            "count": self.count,
            "min": self.minimum if self.count else None,
            "mean": mean,
            "std": float(np.sqrt(max(0.0, variance))) if self.count else None,
            "p50": self.quantile(0.5),
            "p90": self.quantile(0.9),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "p999": self.quantile(0.999),
            "max": self.maximum if self.count else None,
            "underflow": self.underflow,
            "overflow": self.overflow,
        }


def new_metric_histograms() -> dict[str, StreamingHistogram]:
    return {name: StreamingHistogram(*limits) for name, limits in METRIC_RANGES.items()}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def counter_quantile(counts: Counter[float], q: float) -> float:
    total = sum(counts.values())
    if not total:
        return float("nan")
    rank = max(1, int(np.ceil(q * total)))
    cumulative = 0
    for value, count in sorted(counts.items()):
        cumulative += count
        if cumulative >= rank:
            return float(value)
    raise RuntimeError("counter quantile rank was not reached")


def parse_filename_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return pd.Timestamp(match.group(1)) if match else None


def push_top(
    heap: list[tuple[float, int, dict]],
    values: pd.Series,
    times: pd.Series,
    source: str,
    target: pd.Series,
    voltage: pd.Series,
    current: pd.Series,
    limit: int = 30,
) -> None:
    candidate_index = values.nlargest(min(limit, values.notna().sum())).index
    for idx in candidate_index:
        value = float(values.loc[idx])
        row = {
            "timestamp": str(times.loc[idx]),
            "source_file": source,
            "fc_input_power_kw": value,
            "target_power_kw": float(target.loc[idx]) if pd.notna(target.loc[idx]) else None,
            "fc_voltage_v": float(voltage.loc[idx]) if pd.notna(voltage.loc[idx]) else None,
            "fc_current_a": float(current.loc[idx]) if pd.notna(current.loc[idx]) else None,
        }
        item = (value, id(row), row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif value > heap[0][0]:
            heapq.heapreplace(heap, item)


def scan_recent_archive(
    root: Path, chunksize: int
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(root.rglob("*.csv"))
    global_hist = new_metric_histograms()
    monthly_hist: dict[str, dict[str, StreamingHistogram]] = defaultdict(new_metric_histograms)
    monthly_running = Counter()
    command_counts: Counter[float] = Counter()
    top_heap: list[tuple[float, int, dict]] = []
    file_records: list[dict] = []
    declared_cell_numbers: set[int] = set()
    cell_validity = Counter()
    skipped: list[dict] = []
    total_rows = 0

    for file_index, path in enumerate(files, start=1):
        try:
            header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
        except Exception as exc:
            skipped.append({"file": path.name, "reason": repr(exc)})
            continue
        missing = [column for column in CORE_COLUMNS if column not in header]
        if missing:
            skipped.append({"file": path.name, "reason": f"missing columns: {missing}"})
            continue
        declared_cell_numbers.update(
            int(match.group(1))
            for column in header
            if (match := re.fullmatch(r"节电压(\d+)", column))
        )
        usecols = CORE_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in header]
        file_rows = 0
        file_running = 0
        file_start: pd.Timestamp | None = None
        file_end: pd.Timestamp | None = None

        try:
            iterator = pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=usecols,
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines="skip",
            )
            for frame in iterator:
                file_rows += len(frame)
                total_rows += len(frame)
                times = pd.to_datetime(frame[COL_TIME], errors="coerce")
                valid_times = times.dropna()
                if len(valid_times):
                    chunk_start, chunk_end = valid_times.min(), valid_times.max()
                    file_start = chunk_start if file_start is None else min(file_start, chunk_start)
                    file_end = chunk_end if file_end is None else max(file_end, chunk_end)

                voltage = numeric(frame, COL_FC_V)
                current = numeric(frame, COL_FC_I)
                target = numeric(frame, COL_TARGET)
                output_voltage = numeric(frame, COL_OUT_V)
                output_current = numeric(frame, COL_OUT_I)
                loadable = numeric(frame, COL_LOADABLE)
                input_power = voltage * current / 1000.0
                output_power = output_voltage * output_current / 1000.0
                running = current >= 5.0
                file_running += int(running.sum())

                metrics = {
                    "target_power_kw": target[running].to_numpy(),
                    "fc_input_power_kw": input_power[running].to_numpy(),
                    "dcdc_output_power_kw": output_power[running].to_numpy(),
                    "loadable_raw": loadable[running].to_numpy(),
                    "fc_current_a": current[running].to_numpy(),
                }
                for name, values in metrics.items():
                    global_hist[name].update(values)

                valid_months = times.dt.strftime("%Y-%m")
                for month in valid_months[running & times.notna()].dropna().unique():
                    mask = running & valid_months.eq(month)
                    monthly_running[month] += int(mask.sum())
                    for name, series in {
                        "target_power_kw": target,
                        "fc_input_power_kw": input_power,
                        "dcdc_output_power_kw": output_power,
                        "loadable_raw": loadable,
                        "fc_current_a": current,
                    }.items():
                        monthly_hist[month][name].update(series[mask].to_numpy())

                rounded_target = target[running & target.notna()].round(1)
                command_counts.update(rounded_target.tolist())
                push_top(top_heap, input_power[running], times, path.name, target, voltage, current)

                if COL_CELL_MEAN in frame:
                    cell_mean = numeric(frame, COL_CELL_MEAN)
                    voltage_per_declared_channel = voltage / 85.0
                    valid_mean = (
                        running & cell_mean.notna() & voltage_per_declared_channel.notna()
                    )
                    cell_validity["running_rows_checked"] += int(valid_mean.sum())
                    cell_validity["reported_mean_gt_0_5"] += int(
                        (valid_mean & (cell_mean > 0.5)).sum()
                    )
                    cell_validity["reported_mean_close_to_v_over_85"] += int(
                        (
                            valid_mean
                            & ((cell_mean - voltage_per_declared_channel).abs() <= 0.06)
                        ).sum()
                    )
        except Exception as exc:
            skipped.append({"file": path.name, "reason": repr(exc)})
            continue

        filename_date = parse_filename_date(path)
        actual_date = file_start.normalize() if file_start is not None else None
        offset_days = (
            int((filename_date - actual_date).days)
            if filename_date is not None and actual_date is not None
            else None
        )
        file_records.append(
            {
                "source_file": path.name,
                "relative_path": str(path.relative_to(root)),
                "rows": file_rows,
                "running_rows": file_running,
                "actual_start": str(file_start) if file_start is not None else None,
                "actual_end": str(file_end) if file_end is not None else None,
                "actual_date": str(actual_date.date()) if actual_date is not None else None,
                "filename_date": str(filename_date.date()) if filename_date is not None else None,
                "filename_minus_actual_days": offset_days,
            }
        )
        if file_index % 25 == 0 or file_index == len(files):
            print(f"[{file_index}/{len(files)}] rows={total_rows:,} file={path.name}", flush=True)

    monthly_records = []
    for month in sorted(monthly_hist):
        record = {"month": month, "running_rows": monthly_running[month]}
        for name, histogram in monthly_hist[month].items():
            summary = histogram.summary()
            for key in ("p50", "p95", "p99", "p999", "max"):
                record[f"{name}_{key}"] = summary[key]
        monthly_records.append(record)

    command_frame = pd.DataFrame(
        [
            {"target_power_kw_rounded_0p1": value, "running_rows": count}
            for value, count in command_counts.most_common()
        ]
    )
    top_frame = pd.DataFrame([item[2] for item in sorted(top_heap, reverse=True)])
    file_frame = pd.DataFrame(file_records)
    metric_summaries = {
        name: histogram.summary() for name, histogram in global_hist.items()
    }
    for label, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999)):
        metric_summaries["target_power_kw"][label] = counter_quantile(command_counts, q)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recent_root": str(root),
        "csv_files_found": len(files),
        "csv_files_scanned": len(file_records),
        "total_rows": total_rows,
        "declared_cell_voltage_columns": sorted(declared_cell_numbers),
        "declared_cell_count": max(declared_cell_numbers) if declared_cell_numbers else 0,
        "metrics_running_current_ge_5a": metric_summaries,
        "cell_mean_audit": dict(cell_validity),
        "skipped_files": skipped,
    }
    return metadata, file_frame, pd.DataFrame(monthly_records), command_frame, top_frame


def audit_overlap(half_month_root: Path, recent_root: Path, file_frame: pd.DataFrame) -> pd.DataFrame:
    recent_by_date: dict[str, list[Path]] = defaultdict(list)
    for record in file_frame.to_dict("records"):
        if record.get("actual_date"):
            recent_by_date[record["actual_date"]].append(recent_root / record["relative_path"])

    records = []
    for old_path in sorted(half_month_root.glob("*.csv")):
        old = pd.read_csv(old_path, encoding="utf-8-sig", usecols=CORE_COLUMNS, low_memory=False)
        old[COL_TIME] = pd.to_datetime(old[COL_TIME], errors="coerce")
        valid = old.dropna(subset=[COL_TIME])
        if valid.empty:
            continue
        actual_date = str(valid[COL_TIME].min().date())
        candidates = recent_by_date.get(actual_date, [])
        best = None
        for recent_path in candidates:
            recent = pd.read_csv(recent_path, encoding="utf-8-sig", usecols=CORE_COLUMNS, low_memory=False)
            recent[COL_TIME] = pd.to_datetime(recent[COL_TIME], errors="coerce")
            old_unique = valid.drop_duplicates(COL_TIME).set_index(COL_TIME)
            recent_unique = recent.dropna(subset=[COL_TIME]).drop_duplicates(COL_TIME).set_index(COL_TIME)
            index = old_unique.index.intersection(recent_unique.index)
            left = old_unique.loc[index, CORE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
            right = recent_unique.loc[index, CORE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
            exact = ((left == right) | (left.isna() & right.isna())).all(axis=1)
            score = int(exact.sum())
            candidate = {
                "liu_source_file": old_path.name,
                "recent_archive_file": recent_path.name,
                "actual_date": actual_date,
                "liu_rows": len(old),
                "recent_rows": len(recent),
                "shared_unique_timestamps": len(index),
                "exact_core_signal_rows": score,
                "exact_core_fraction": float(exact.mean()) if len(exact) else 0.0,
                "recent_filename_minus_actual_days": (
                    parse_filename_date(recent_path) - pd.Timestamp(actual_date)
                ).days,
            }
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            records.append(
                {
                    "liu_source_file": old_path.name,
                    "recent_archive_file": None,
                    "actual_date": actual_date,
                    "liu_rows": len(old),
                    "recent_rows": 0,
                    "shared_unique_timestamps": 0,
                    "exact_core_signal_rows": 0,
                    "exact_core_fraction": 0.0,
                    "recent_filename_minus_actual_days": None,
                }
            )
        else:
            records.append(best[1])
    return pd.DataFrame(records)


def audit_cell_channel_samples(root: Path, file_frame: pd.DataFrame) -> pd.DataFrame:
    """Sample one file per actual month to audit channel availability.

    The numbered fields are telemetry channels. This audit deliberately avoids
    interpreting their count as the physical single-cell count.
    """

    candidates = file_frame[file_frame["running_rows"] > 0].copy()
    candidates["month"] = candidates["actual_date"].astype(str).str[:7]
    records = []
    for month, group in candidates.groupby("month", sort=True):
        record = group.sort_values("actual_start").iloc[0]
        path = root / record["relative_path"]
        header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
        cell_columns = [
            column
            for index in range(1, 86)
            if (column := f"节电压{index}") in header
        ]
        usecols = [COL_FC_V, COL_FC_I, COL_CELL_MEAN] + cell_columns
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
        current = numeric(frame, COL_FC_I)
        running = frame[current >= 5.0]
        channel_values = running[cell_columns].apply(pd.to_numeric, errors="coerce")
        nonzero_per_row = (channel_values.abs() > 1e-9).sum(axis=1)
        active_per_column = (channel_values.abs() > 1e-9).any(axis=0)
        stack_voltage = numeric(running, COL_FC_V)
        channel_mean = numeric(running, COL_CELL_MEAN)
        ratio = (stack_voltage / channel_mean).replace([np.inf, -np.inf], np.nan).dropna()
        records.append(
            {
                "month": month,
                "source_file": path.name,
                "running_rows": len(running),
                "declared_channels": len(cell_columns),
                "channels_ever_nonzero": int(active_per_column.sum()),
                "modal_nonzero_channels_per_row": int(nonzero_per_row.mode().iloc[0]),
                "channel_mean_voltage_v_median": float(channel_mean.median()),
                "stack_voltage_v_median": float(stack_voltage.median()),
                "stack_voltage_over_channel_mean_median": float(ratio.median()),
            }
        )
    return pd.DataFrame(records)


def plot_power_envelope(metadata: dict, monthly: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    metrics = metadata["metrics_running_current_ge_5a"]
    labels = ["Target command", "Stack input", "DC/DC output"]
    names = ["target_power_kw", "fc_input_power_kw", "dcdc_output_power_kw"]
    p99 = [metrics[name]["p99"] for name in names]
    p999 = [metrics[name]["p999"] for name in names]
    maxima = [metrics[name]["max"] for name in names]
    x = np.arange(len(names))
    axes[0].plot(x, maxima, "o", color="#333333", ms=4, label="Maximum")
    axes[0].plot(x, p999, "s", color="#E45756", ms=4, label="99.9th percentile")
    axes[0].plot(x, p99, "^", color="#4C78A8", ms=4, label="99th percentile")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("(a) Empirical power envelope")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False)

    month_x = np.arange(len(monthly))
    axes[1].plot(month_x, monthly["fc_input_power_kw_p999"], "o-", color="#E45756", ms=3, lw=1.2, label="99.9th percentile")
    axes[1].plot(month_x, monthly["fc_input_power_kw_max"], "s-", color="#333333", ms=3, lw=1.0, label="Maximum")
    axes[1].set_xticks(month_x, monthly["month"], rotation=55, ha="right")
    axes[1].set_ylabel("Stack input power (kW)")
    axes[1].set_title("(b) Month-wise observed upper tail")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def write_report(
    metadata: dict,
    overlap: pd.DataFrame,
    channel_samples: pd.DataFrame,
    output: Path,
) -> None:
    metrics = metadata["metrics_running_current_ge_5a"]
    matched = overlap[overlap["exact_core_fraction"] == 1.0]
    offsets = overlap["recent_filename_minus_actual_days"].dropna().unique().tolist()
    cell = metadata["cell_mean_audit"]
    checked = cell.get("running_rows_checked", 0)
    close = cell.get("reported_mean_close_to_v_over_85", 0)
    ratio_median = float(channel_samples["stack_voltage_over_channel_mean_median"].median())
    channel_min = int(channel_samples["modal_nonzero_channels_per_row"].min())
    channel_max = int(channel_samples["modal_nonzero_channels_per_row"].max())
    text = f"""# 21UBE0022 identity and power audit

## Scope

The recent-year archive was scanned read-only and was not concatenated with the seven-day development canonical dataset. All power statistics below use rows with DC/DC input current at least 5 A.

## Identity evidence

- {len(matched)}/{len(overlap)} Liu half-month files have a recent-archive counterpart with identical timestamp, target-power, DC/DC input-voltage and input-current rows.
- The archive filename date minus the actual first timestamp date is {offsets}; file names must not be treated as the measurement date without reading the timestamp column.
- This proves that the current seven-day load source belongs to vehicle `21UBE0022_苏E02625F` and is contained in the recent archive at the level of the four control-critical signals.
- It does not prove byte identity for every telemetry field, and it does not prove that the stack installed after April 2025 is the same physical stack as Liu's 170-cell 2022--2024 ageing stack.

## Empirical power evidence

| Signal | p99 (kW) | p99.9 (kW) | observed max (kW) |
|---|---:|---:|---:|
| Target command | {metrics['target_power_kw']['p99']:.3f} | {metrics['target_power_kw']['p999']:.3f} | {metrics['target_power_kw']['max']:.3f} |
| Stack-side DC/DC input | {metrics['fc_input_power_kw']['p99']:.3f} | {metrics['fc_input_power_kw']['p999']:.3f} | {metrics['fc_input_power_kw']['max']:.3f} |
| Bus-side DC/DC output | {metrics['dcdc_output_power_kw']['p99']:.3f} | {metrics['dcdc_output_power_kw']['p999']:.3f} | {metrics['dcdc_output_power_kw']['max']:.3f} |
These values establish an empirical operating envelope and repeated controller commands, not a nameplate rated net power. A physical rating still requires a controller calibration table, vehicle specification or nameplate.

The source field `可加载功率` is excluded from the power table because its observed values reach {metrics['loadable_raw']['max']:.0f} and no unit definition is present in the available files. It is retained only as `loadable_raw` in machine-readable diagnostics.

## Health-observation consequence

- Headers declare {metadata['declared_cell_count']} numbered voltage channels; this is not evidence of {metadata['declared_cell_count']} physical PEMFC cells.
- On {checked:,} running rows, the source's reported mean-voltage field agrees numerically with stack voltage divided by 85 declared channels within 0.06 V for {close:,} rows ({100 * close / checked if checked else 0:.2f}%). This is a telemetry-consistency check, not a physical cell-count inference.
- Across one sampled file per actual month, the median ratio of stack voltage to reported channel mean is {ratio_median:.2f}. The modal number of nonzero numbered channels changes from {channel_min} to {channel_max}, proving that channel availability/format changes within the archive.
- If the reported 1.3--1.5 V field is a channel-level mean, the stack/channel ratio near 85 is compatible with two-cell grouped channels and a 170-cell stack. If it is intended as a physical single-cell mean, its scale is inconsistent with normal loaded PEMFC voltage. Because the CSVs contain neither a channel definition nor a stack-ID field, physical cell count and stack continuity remain unproven.
- A separate 21UBE0022 voltage-trend measurement model may use total stack voltage after current/temperature matching and explicit telemetry-epoch filtering. It must not row-wise mix these records with the older MAT chain.
- The MAT chain remains a cross-dataset degradation prior. The recent archive is a candidate source for vehicle-specific correction and independent validation, not proof that both sources measured the same physical stack.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-root", type=Path, required=True)
    parser.add_argument("--liu-half-month-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/results/liu_21ube0022_identity_rating",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata, files, monthly, commands, top = scan_recent_archive(args.recent_root, args.chunksize)
    overlap = audit_overlap(args.liu_half_month_root, args.recent_root, files)
    channel_samples = audit_cell_channel_samples(args.recent_root, files)
    if channel_samples.empty:
        raise RuntimeError("no running files were available for channel sampling")
    metadata["liu_half_month_root"] = str(args.liu_half_month_root)
    metadata["identity_overlap"] = {
        "files_checked": len(overlap),
        "files_exact_on_core_signals": int((overlap["exact_core_fraction"] == 1.0).sum()),
    }
    metadata["cell_channel_sample"] = {
        "months_sampled": len(channel_samples),
        "modal_nonzero_channels_min": int(
            channel_samples["modal_nonzero_channels_per_row"].min()
        ),
        "modal_nonzero_channels_max": int(
            channel_samples["modal_nonzero_channels_per_row"].max()
        ),
        "median_stack_voltage_over_channel_mean": float(
            channel_samples["stack_voltage_over_channel_mean_median"].median()
        ),
        "interpretation": (
            "numbered telemetry channels are not a verified physical cell count; "
            "two-cell grouping is compatible with the observed voltage scale"
        ),
    }

    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    files.to_csv(args.out_dir / "file_time_audit.csv", index=False)
    monthly.to_csv(args.out_dir / "monthly_power_summary.csv", index=False)
    commands.to_csv(args.out_dir / "target_command_frequency.csv", index=False)
    top.to_csv(args.out_dir / "top_stack_power_points.csv", index=False)
    overlap.to_csv(args.out_dir / "identity_overlap.csv", index=False)
    channel_samples.to_csv(args.out_dir / "cell_channel_sample_audit.csv", index=False)
    plot_power_envelope(metadata, monthly, args.out_dir / "fig18_21ube0022_power_envelope.png")
    write_report(metadata, overlap, channel_samples, args.out_dir / "report.md")
    print(json.dumps(metadata["identity_overlap"], ensure_ascii=False))
    print(json.dumps(metadata["metrics_running_current_ge_5a"], ensure_ascii=False))


if __name__ == "__main__":
    main()
