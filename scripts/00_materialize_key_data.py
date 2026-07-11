"""Materialize compressed key datasets for local continuation.

中文名：00_解压关键数据。

The repository stores selected processed datasets as ``.csv.gz`` files under
``data/key`` so a new collaborator can continue the baseline/degradation-proxy
workflow without the private raw Excel/MAT files.  This script restores the
CSV files expected by the existing scripts.  It does not download or create
raw data.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "data" / "key"


RESTORE_MAP = {
    KEY / "liu_vehicle_canonical_1s.csv.gz": [
        ROOT / "data" / "processed" / "liu_vehicle_canonical_1s.csv"
    ],
    KEY / "baseline_power_demand.csv.gz": [
        ROOT / "data" / "processed" / "baseline_power_demand.csv",
        ROOT / "data" / "processed" / "power_demand_from_dynamics.csv",
    ],
    KEY / "baseline_prediction_results.csv.gz": [
        ROOT / "data" / "processed" / "baseline_prediction_results.csv",
        ROOT / "data" / "processed" / "prediction_results.csv",
    ],
    KEY / "current_point_degradation_h2.csv.gz": [
        ROOT / "data" / "processed" / "current_point_degradation_h2.csv"
    ],
}

LI_JUNHAO_RESTORE_MAP = {
    KEY / "li_junhao_dual_stack" / "engine3_three_day_processed.csv.gz": [
        ROOT
        / "experiments"
        / "li_junhao_dual_stack_audit"
        / "三号发动机_three_day_processed.csv"
    ],
    KEY / "li_junhao_dual_stack" / "engine4_three_day_processed.csv.gz": [
        ROOT
        / "experiments"
        / "li_junhao_dual_stack_audit"
        / "四号发动机_three_day_processed.csv"
    ],
    KEY / "li_junhao_dual_stack" / "engine5_three_day_processed.csv.gz": [
        ROOT
        / "experiments"
        / "li_junhao_dual_stack_audit"
        / "五号发动机_three_day_processed.csv"
    ],
}


def restore_one(src: Path, dst: Path, overwrite: bool = False) -> str:
    """Restore one gzipped CSV to ``dst``."""
    if not src.exists():
        return f"missing key file: {src.relative_to(ROOT)}"
    if dst.exists() and not overwrite:
        return f"exists, skipped: {dst.relative_to(ROOT)}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return f"restored: {dst.relative_to(ROOT)}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing materialized CSV files",
    )
    parser.add_argument(
        "--include-li-junhao",
        action="store_true",
        help="also restore Li Junhao three-day processed dual-stack CSV files",
    )
    args = parser.parse_args()

    maps = dict(RESTORE_MAP)
    if args.include_li_junhao:
        maps.update(LI_JUNHAO_RESTORE_MAP)

    messages: list[str] = []
    for src, outputs in maps.items():
        for dst in outputs:
            messages.append(restore_one(src, dst, overwrite=args.overwrite))

    print("\n".join(messages))


if __name__ == "__main__":
    main()
