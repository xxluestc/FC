"""Build held-out monthly service exposures through the frozen fast controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    TestScenario,
    run_policy,
    service_exposure_from_trajectory,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = (
    ROOT
    / "data/results/fc_only_external_monthly_blocks/external_monthly_power_blocks.csv"
)
OUTPUT = ROOT / "data/results/fc_only_external_service_templates"
HETEROGENEITY = (1.0, 1.05, 1.10)
ASSIGNMENT = (0, 1)
NORMALIZATION_POWER_KW = 40.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_template(task):
    model, scenario, descriptor = task
    run = run_policy(
        model,
        scenario,
        "instant_health",
        fixed_online_assignment=ASSIGNMENT,
    )
    exposure, role_stacks = service_exposure_from_trajectory(
        run.trajectory,
        duration_h=len(run.trajectory) * model.config.dt_s / 3600.0,
        assigned_stacks=ASSIGNMENT,
    )
    return {
        **descriptor,
        "duration_h": exposure.duration_h,
        "role_0_source_stack": role_stacks[0],
        "role_1_source_stack": role_stacks[1],
        "role_0_continuous_mean_pct": exposure.continuous_mean_pct[0],
        "role_1_continuous_mean_pct": exposure.continuous_mean_pct[1],
        "role_0_load_shift_damage_pct": exposure.load_shift_damage_pct[0],
        "role_1_load_shift_damage_pct": exposure.load_shift_damage_pct[1],
        "role_0_operational_start_damage_pct": (
            exposure.operational_start_damage_pct[0]
        ),
        "role_1_operational_start_damage_pct": (
            exposure.operational_start_damage_pct[1]
        ),
        "tracking_mae_kw": run.metrics["fc_tracking_mae_kw"],
        "tracking_max_abs_kw": run.metrics["fc_tracking_max_abs_kw"],
        "constraint_violation_steps": run.metrics["constraint_violation_steps"],
        "safety_override_steps": run.metrics["safety_override_steps"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if not BLOCKS.exists():
        raise FileNotFoundError("run script 56 before building external templates")

    blocks = pd.read_csv(BLOCKS, parse_dates=["timestamp"])
    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    system_reference_kw = (
        1 - CAPACITY_RESERVE_FRACTION
    ) * model.fc_power_reference_kw()
    tasks = []
    for block_id, block in blocks.groupby("block_id", sort=True):
        block = block.sort_values("block_step").reset_index(drop=True)
        power = block.fc_input_power_kw.to_numpy(dtype=float)
        normalized = np.maximum(power, 0.0) / NORMALIZATION_POWER_KW
        month = str(block.month.iloc[0])
        stratum = int(block.stratum.iloc[0])
        demand = pd.DataFrame(
            {
                "demand_power_kw": normalized * system_reference_kw,
                "event": np.where(
                    normalized > 0,
                    "external_real_on",
                    "external_real_off",
                ),
                "source": "external_monthly_real_power",
                "seed": int(month.replace("-", "")) + stratum,
            }
        )
        scenario = TestScenario(
            name=f"external_service_{block_id}",
            demand=demand,
            initial_damage_fraction=(0.0, 0.0, 0.0),
            health_seed=80_000 + int(month.replace("-", "")) + stratum,
            stochastic_health=False,
        )
        tasks.append(
            (
                model,
                scenario,
                {
                    "template_id": block_id,
                    "template_source": "external_monthly_real_power",
                    "month": month,
                    "stratum": stratum,
                    "raw_power_mean_kw": float(power.mean()),
                    "raw_power_max_kw": float(power.max()),
                    "positive_share": float((power > 0).mean()),
                    "above_40kw_steps": int((power > 40.0).sum()),
                },
            )
        )

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(run_template, tasks, chunksize=1))
    table = pd.DataFrame(rows).sort_values(["month", "stratum"]).reset_index(drop=True)
    if len(table) != 39 or table.month.nunique() != 13:
        raise AssertionError("external service audit requires 39 blocks across 13 months")
    if not table.groupby("month").size().eq(3).all():
        raise AssertionError("each held-out month must provide three time-stratum blocks")
    if int(table.constraint_violation_steps.sum()) != 0:
        raise AssertionError("external service templates contain hard constraint violations")
    if float(table.tracking_max_abs_kw.max()) > TRACKING_TOLERANCE_KW + 1e-12:
        raise AssertionError("external service templates exceeded tracking tolerance")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "service_exposure_templates.csv"
    table.to_csv(output, index=False)
    metadata = {
        "scope": "held-out cross-month service exposure templates",
        "block_input": str(BLOCKS.relative_to(ROOT)),
        "block_input_sha256": sha256(BLOCKS),
        "templates": len(table),
        "months": int(table.month.nunique()),
        "templates_per_month": 3,
        "normalization_power_kw": NORMALIZATION_POWER_KW,
        "mapping_system_power_reference_kw": system_reference_kw,
        "controller_policy": "instant_health",
        "fixed_online_assignment": ASSIGNMENT,
        "controller_retuned": False,
        "future_demand_used": False,
        "runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Cross-month service exposure templates

- Converted 39 held-out real-power blocks from 13 months through the frozen fast controller.
- Each month contributes three continuous 300 s blocks from separate time strata.
- No controller retuning or future demand information was used.
- Maximum tracking error: {table.tracking_max_abs_kw.max():.6f} kW.
- Hard constraint violations: {int(table.constraint_violation_steps.sum())}.
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(table.groupby("month").agg(
        templates=("template_id", "size"),
        mean_power_kw=("raw_power_mean_kw", "mean"),
        max_tracking_error_kw=("tracking_max_abs_kw", "max"),
    ).to_string())


if __name__ == "__main__":
    main()
