"""Audit tracking-tolerance feasibility on the frozen worst holdout case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import TestScenario, run_policy, split_at_largest_segment_gap
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
BASELINE = ROOT / "data/results/fc_only_full_holdout_replay/per_run_metrics.csv"
OUTPUT = ROOT / "data/results/fc_only_tracking_tolerance_audit"
SEGMENT_ID = 42
HEALTH_CASE = "oldest_stack_0"
INITIAL_DAMAGE_FRACTION = (0.80, 0.10, 0.40)
ASSIGNMENT = (1, 0)
HETEROGENEITY = (1.0, 1.05, 1.10)
NORMALIZATION_POWER_KW = 30.0
CAPACITY_RESERVE_FRACTION = 0.05
TOLERANCES_KW = (
    4.0,
    4.5,
    4.6,
    4.7,
    4.8,
    4.9,
    4.95,
    5.0,
    5.25,
    5.4,
    5.45,
    5.49,
    5.499,
    5.5,
)


def demand_for_worst_case(model):
    frame = pd.read_csv(
        SOURCE, usecols=["timestamp", "segment_id", "fc_input_power_kw"]
    )
    split = split_at_largest_segment_gap(frame)
    if SEGMENT_ID not in split.holdout_segments:
        raise AssertionError("worst-case segment is no longer in the holdout split")
    segment = frame[frame.segment_id.eq(SEGMENT_ID)].reset_index(drop=True)
    normalized = np.clip(
        segment.fc_input_power_kw.to_numpy(dtype=float) / NORMALIZATION_POWER_KW,
        0.0,
        1.0,
    )
    system_reference_kw = (
        1.0 - CAPACITY_RESERVE_FRACTION
    ) * model.fc_power_reference_kw()
    return pd.DataFrame(
        {
            "demand_power_kw": normalized * system_reference_kw,
            "event": np.where(normalized > 0, "real_fc_on", "real_fc_off"),
            "source": "real_holdout_tracking_tolerance_audit",
            "seed": SEGMENT_ID,
        }
    )


def run_tolerance(tolerance_kw):
    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=tolerance_kw,
        ),
    )
    scenario = TestScenario(
        name=f"tracking_tolerance_{tolerance_kw:g}_kw",
        demand=demand_for_worst_case(model),
        initial_damage_fraction=INITIAL_DAMAGE_FRACTION,
        health_seed=60_000 + SEGMENT_ID,
        stochastic_health=False,
    )
    try:
        run = run_policy(
            model,
            scenario,
            "instant_health",
            fixed_online_assignment=ASSIGNMENT,
        )
    except RuntimeError as error:
        return {
            "tracking_tolerance_kw": tolerance_kw,
            "success": False,
            "n_steps": 0,
            "tracking_max_abs_kw": np.nan,
            "tracking_mae_kw": np.nan,
            "safety_override_steps": np.nan,
            "expected_damage_increment_pct": np.nan,
            "error": str(error),
        }
    trajectory = run.trajectory
    tracking = trajectory.fc_power_tracking_error_kw.to_numpy(dtype=float)
    return {
        "tracking_tolerance_kw": tolerance_kw,
        "success": True,
        "n_steps": len(trajectory),
        "tracking_max_abs_kw": float(np.abs(tracking).max()),
        "tracking_mae_kw": float(np.abs(tracking).mean()),
        "safety_override_steps": int(run.metrics["safety_override_steps"]),
        "expected_damage_increment_pct": float(
            run.metrics["main_expected_damage_increment_pct"]
        ),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse already written tolerance rows and run only missing levels",
    )
    args = parser.parse_args()
    baseline = pd.read_csv(BASELINE)
    frozen = baseline[
        baseline.segment_id.eq(SEGMENT_ID)
        & baseline.health_case.eq(HEALTH_CASE)
        & baseline.policy.eq("fixed_pair")
    ]
    if len(frozen) != 1 or tuple(
        int(value) for value in frozen.iloc[0].assignment.strip("()").split(",")
    ) != ASSIGNMENT:
        raise AssertionError("frozen worst-case identity changed")

    cached = {}
    if args.resume and (OUTPUT / "tolerance_sweep.csv").exists():
        existing = pd.read_csv(OUTPUT / "tolerance_sweep.csv")
        existing["error"] = existing["error"].fillna("")
        cached = {
            float(row.tracking_tolerance_kw): row._asdict()
            for row in existing.itertuples(index=False)
        }
    rows = []
    for tolerance in TOLERANCES_KW:
        if tolerance in cached:
            rows.append(cached[tolerance])
        else:
            rows.append(run_tolerance(tolerance))
    rows.sort(key=lambda row: float(row["tracking_tolerance_kw"]))
    table = pd.DataFrame(rows)
    reproduced = table[table.tracking_tolerance_kw.eq(5.5)].iloc[0]
    if not reproduced.success or not np.isclose(
        reproduced.tracking_max_abs_kw,
        frozen.iloc[0].tracking_max_abs_kw,
        atol=1e-12,
    ):
        raise AssertionError("5.5 kW targeted replay did not reproduce the baseline")
    successful = table[table.success]
    minimum_tested_success_kw = float(successful.tracking_tolerance_kw.min())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "tolerance_sweep.csv", index=False)
    metadata = {
        "scope": "targeted replay of the frozen maximum-error 30 kW case",
        "segment_id": SEGMENT_ID,
        "health_case": HEALTH_CASE,
        "policy": "fixed_pair",
        "assignment": ASSIGNMENT,
        "normalization_power_kw": NORMALIZATION_POWER_KW,
        "minimum_tested_success_kw": minimum_tested_success_kw,
        "static_healthy_grid_max_error_kw": 4.018,
        "full_holdout_claim": False,
        "interpretation": (
            "Failure means this controller and action grid could not complete the "
            "targeted case at the tested tolerance; it is not a full-holdout sweep."
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 跟踪容差最坏案例边界审计

冻结30 kW主回放的最大误差案例为segment {SEGMENT_ID}、`{HEALTH_CASE}`、固定双堆、分配
`{ASSIGNMENT}`。静态健康动作网格最大误差曾为4.018 kW；本审计保持真实段、在线健康和驻留
演化不变，只改变硬跟踪容差。

{table.to_markdown(index=False)}

当前测试网格中的最小成功容差为{minimum_tested_success_kw:g} kW。该结果说明5.5 kW附近的
容差来自离散动作、健康漂移和驻留共同作用，并非任意宽松常数；但它只审计冻结最坏案例，
不等同于所有留出案例在更紧容差下的成功率曲线。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
