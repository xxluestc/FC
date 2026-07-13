"""Build a separate cross-month real-power validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fc_power.evaluation import (
    canonicalize_power_packets,
    select_first_operating_block,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECENT_ROOT = Path(r"G:\大论文\实车数据\21UBE0022_苏E02625F")
FILE_AUDIT = ROOT / "data/results/liu_21ube0022_identity_rating/file_time_audit.csv"
OUTPUT = ROOT / "data/results/fc_only_external_monthly_blocks"
COL_TIME = "上报时间"
COL_VOLTAGE = "DCDC输入电压"
COL_CURRENT = "DCDC输入电流"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_power_packets(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=[COL_TIME, COL_VOLTAGE, COL_CURRENT],
            low_memory=False,
        ).rename(
            columns={
                COL_TIME: "timestamp",
                COL_VOLTAGE: "fc_voltage_v",
                COL_CURRENT: "fc_current_a",
            }
        )
        parts.append(frame)
    if not parts:
        raise ValueError("no raw files selected for external block extraction")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-root", type=Path, default=DEFAULT_RECENT_ROOT)
    parser.add_argument("--start-month", default="2025-06")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--block-steps", type=int, default=1800)
    parser.add_argument("--blocks-per-month", type=int, default=3)
    parser.add_argument("--minimum-positive-share", type=float, default=0.8)
    parser.add_argument("--positive-threshold-kw", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.recent_root.exists():
        raise FileNotFoundError(args.recent_root)
    if args.blocks_per_month <= 0:
        raise ValueError("blocks-per-month must be positive")
    if not FILE_AUDIT.exists():
        raise FileNotFoundError("run script 53 before building external blocks")

    periods = pd.period_range(args.start_month, args.end_month, freq="M")
    if not len(periods):
        raise ValueError("external validation month range is empty")
    audit = pd.read_csv(FILE_AUDIT, parse_dates=["actual_start", "actual_end"])
    blocks = []
    manifest_rows = []
    source_rows = []
    for period in periods:
        intersects = (audit.actual_end >= period.start_time) & (
            audit.actual_start <= period.end_time
        )
        selected_audit = audit.loc[intersects].sort_values("actual_start")
        paths = [args.recent_root / Path(value) for value in selected_audit.relative_path]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing audited archive file: {missing[0]}")

        canonical, packet_audit = canonicalize_power_packets(
            load_power_packets(paths)
        )
        month_duration = pd.Timedelta(days=period.days_in_month)
        for stratum in range(args.blocks_per_month):
            stratum_start = period.start_time + month_duration * (
                stratum / args.blocks_per_month
            )
            stratum_end = period.start_time + month_duration * (
                (stratum + 1) / args.blocks_per_month
            ) - pd.Timedelta(seconds=1)
            stratum_frame = canonical[
                (canonical.timestamp >= stratum_start)
                & (canonical.timestamp <= stratum_end)
            ]
            block = select_first_operating_block(
                stratum_frame,
                period,
                block_steps=args.block_steps,
                minimum_positive_share=args.minimum_positive_share,
                positive_threshold_kw=args.positive_threshold_kw,
            )
            block_id = f"external_{period}_s{stratum + 1}"
            block.insert(0, "block_id", block_id)
            block.insert(1, "month", str(period))
            block.insert(2, "stratum", stratum + 1)
            blocks.append(block)

            start = block.timestamp.iloc[0]
            end = block.timestamp.iloc[-1]
            contributing = selected_audit[
                (selected_audit.actual_end >= start)
                & (selected_audit.actual_start <= end)
            ]
            if contributing.empty:
                raise AssertionError(
                    "selected block is not covered by the file-time audit"
                )
            source_names = []
            for row in contributing.itertuples(index=False):
                path = args.recent_root / Path(row.relative_path)
                source_names.append(str(row.relative_path))
                source_rows.append(
                    {
                        "block_id": block_id,
                        "relative_path": row.relative_path,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

            power = block.fc_input_power_kw
            manifest_rows.append(
                {
                    "block_id": block_id,
                    "month": str(period),
                    "stratum": stratum + 1,
                    "stratum_start": stratum_start,
                    "stratum_end": stratum_end,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "steps": len(block),
                    "positive_steps": int(
                        (power >= args.positive_threshold_kw).sum()
                    ),
                    "positive_share": float(
                        (power >= args.positive_threshold_kw).mean()
                    ),
                    "power_mean_kw": float(power.mean()),
                    "power_p95_kw": float(power.quantile(0.95)),
                    "power_max_kw": float(power.max()),
                    "above_40kw_steps": int((power > 40.0).sum()),
                    "negative_power_steps": int((power < 0.0).sum()),
                    "interpolated_power_steps": int(
                        block.interpolated_power.sum()
                    ),
                    "archive_files_read": len(paths),
                    "selected_source_files": " | ".join(source_names),
                    "raw_rows_read": packet_audit["raw_rows"],
                    "duplicate_timestamps": packet_audit["duplicate_timestamps"],
                    "source_segments": packet_audit["source_segments"],
                }
            )
            print(f"selected {block_id}: {start} to {end}", flush=True)

    block_table = pd.concat(blocks, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    sources = pd.DataFrame(source_rows).drop_duplicates(
        ["block_id", "relative_path"]
    )
    expected_blocks = len(periods) * args.blocks_per_month
    if len(block_table) != expected_blocks * args.block_steps:
        raise AssertionError("external block table has an unexpected row count")
    if block_table.block_id.nunique() != expected_blocks:
        raise AssertionError("the prespecified external strata were not all selected")
    if block_table.timestamp.between(
        pd.Timestamp("2025-05-17"), pd.Timestamp("2025-05-28")
    ).any():
        raise AssertionError("external cohort overlaps the original seven-day source")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    block_table.to_csv(args.out_dir / "external_monthly_power_blocks.csv", index=False)
    manifest.to_csv(args.out_dir / "block_manifest.csv", index=False)
    sources.to_csv(args.out_dir / "selected_source_manifest.csv", index=False)
    metadata = {
        "scope": "separate external real-power trajectory cohort",
        "archive_root": str(args.recent_root),
        "file_time_audit": str(FILE_AUDIT.relative_to(ROOT)),
        "file_time_audit_sha256": sha256(FILE_AUDIT),
        "months": [str(value) for value in periods],
        "block_steps": args.block_steps,
        "blocks_per_month": args.blocks_per_month,
        "target_dt_s": 1,
        "gap_threshold_s": 10,
        "minimum_positive_share": args.minimum_positive_share,
        "positive_threshold_kw": args.positive_threshold_kw,
        "selection_rule": (
            "split each calendar month into equal time strata, then take the first "
            "chronological one-second block in every stratum with finite power at "
            "every step and the prespecified minimum positive-power share"
        ),
        "separation": (
            "months 2025-06 through 2026-06 are not concatenated with the original "
            "May 2025 seven-day modelling chain; every block resets health/control state"
        ),
        "parameter_use_boundary": (
            "the archive previously supported the scalar 40 kW operating reference, "
            "but these blocks did not tune controller weights, assignments or health parameters"
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 独立跨月实车功率块

- 从{periods[0]}至{periods[-1]}共{len(periods)}个月，每月分成{args.blocks_per_month}个等长时间层，每层按同一预声明规则选择第一个{args.block_steps}秒连续块，共{expected_blocks}块。
- 每一步功率均有限，相邻采样为1秒，且正功率占比至少{args.minimum_positive_share:.0%}；不根据控制结果挑块。
- 共{len(block_table):,}个1秒样本，功率范围{block_table.fc_input_power_kw.min():.3f}--{block_table.fc_input_power_kw.max():.3f} kW，高于40 kW的样本{int((block_table.fc_input_power_kw > 40).sum())}个；插值功率步{int(block_table.interpolated_power.sum())}个。
- 这些月份不与原2025-05七天链拼接；每个块单独初始化健康与控制状态，只验证冻结控制器对外部真实功率轨迹的行为。
- 全年归档曾用于确认40 kW经验参考，因此这里不称“归一化完全未见”；但块内轨迹没有参与控制权重、堆选择规则或健康参数调整。

逐块时间、功率和原始文件见`block_manifest.csv`，实际贡献文件的SHA-256见`selected_source_manifest.csv`。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
