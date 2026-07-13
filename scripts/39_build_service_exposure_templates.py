"""Build leakage-safe fast-layer exposure templates for service scheduling."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    TestScenario,
    ZUO_FAST_TRANSITION,
    ZUO_SLOW_TRANSITION,
    generate_zuo_markov_system_load,
    materialize_calibration_window,
    run_policy,
    select_calibration_windows,
    service_exposure_from_trajectory,
    split_at_largest_segment_gap,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
AUDIT = ROOT / "data/results/load_zuo_calibration"
HEALTH_CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_service_templates"
NORMALIZATION_POWER_KW = 30.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
ASSIGNMENT = (0, 1)


def calibration_start_rate(frame, calibration_segments, *, samples=1000, seed=2026):
    rows = []
    for segment_id, segment in frame[
        frame.segment_id.isin(calibration_segments)
    ].groupby("segment_id", sort=True):
        on = segment.fc_input_power_kw.to_numpy(dtype=float) > 0
        starts = int(on[0]) + int(np.sum((~on[:-1]) & on[1:]))
        rows.append((int(segment_id), len(on), starts))
    values = np.asarray([(length, starts) for _, length, starts in rows], dtype=float)
    observed = float(values[:, 1].sum() / (values[:, 0].sum() / 3600.0))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = values[rng.integers(0, len(values), size=len(values))]
        draws.append(float(selected[:, 1].sum() / (selected[:, 0].sum() / 3600.0)))
    return observed, tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def load_empirical_inputs() -> tuple[np.ndarray, np.ndarray]:
    transitions = pd.read_csv(AUDIT / "transition_scale_audit.csv")
    matrix = transitions[transitions.stride_s == 1].pivot(
        index="source_state",
        columns="target_state",
        values="empirical_probability",
    ).to_numpy(dtype=float)
    coverage = pd.read_csv(AUDIT / "state_coverage_audit.csv")
    probabilities = (
        coverage[coverage.stride_s == 1]
        .sort_values("state")
        .occupancy_fraction.to_numpy(dtype=float)
    )
    if matrix.shape != (4, 4) or probabilities.shape != (4,):
        raise ValueError("empirical load audit is incomplete")
    return matrix, probabilities


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
    row = {
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
        "internal_start_count": int(
            sum(
                run.trajectory[f"stack_{stack}_start_stop_increment_pct"]
                .iloc[1:]
                .gt(0)
                .sum()
                for stack in ASSIGNMENT
            )
        ),
        "tracking_mae_kw": run.metrics["fc_tracking_mae_kw"],
        "tracking_max_abs_kw": run.metrics["fc_tracking_max_abs_kw"],
        "tracking_within_tolerance_share": run.metrics[
            "fc_tracking_within_tolerance_share"
        ],
        "constraint_violation_steps": run.metrics["constraint_violation_steps"],
        "safety_override_steps": run.metrics["safety_override_steps"],
        "planning_runtime_s": run.metrics["planning_runtime_s"],
    }
    return row


def real_tasks(frame, split, model, count, length_s, seed, power_reference):
    windows = select_calibration_windows(
        frame,
        split.calibration_segments,
        length_s=length_s,
        count=count,
        seed=seed,
    )
    tasks = []
    for index, window_spec in enumerate(windows):
        window = materialize_calibration_window(frame, window_spec)
        normalized = np.clip(
            window.fc_input_power_kw.to_numpy(dtype=float) / NORMALIZATION_POWER_KW,
            0.0,
            1.0,
        )
        demand = pd.DataFrame(
            {
                "demand_power_kw": normalized * power_reference,
                "event": np.where(normalized > 0, "real_fc_on", "real_fc_off"),
                "source": "real_calibration_window",
                "seed": seed + index,
            }
        )
        scenario = TestScenario(
            name=f"real_calibration_window_{index:03d}",
            demand=demand,
            initial_damage_fraction=(0.0, 0.0, 0.0),
            health_seed=30_000 + index,
            stochastic_health=False,
        )
        tasks.append(
            (
                model,
                scenario,
                {
                    "template_id": scenario.name,
                    "template_source": "real_calibration_window",
                    "source_seed": seed,
                    "segment_id": window_spec.segment_id,
                    "start_offset": window_spec.start_offset,
                    "start_timestamp": str(window.timestamp.iloc[0]),
                    "end_timestamp": str(window.timestamp.iloc[-1]),
                    "raw_power_mean_kw": float(window.fc_input_power_kw.mean()),
                    "raw_power_max_kw": float(window.fc_input_power_kw.max()),
                    "positive_share": float((window.fc_input_power_kw > 0).mean()),
                },
            )
        )
    return tasks


def markov_tasks(model, count, length_s, seed, power_reference, matrix, probabilities):
    scenarios = (
        ("empirical_markov_1s", matrix, 1),
        ("zuo_slow_30s", np.asarray(ZUO_SLOW_TRANSITION), 30),
        ("zuo_fast_30s", np.asarray(ZUO_FAST_TRANSITION), 30),
    )
    tasks = []
    for source_index, (source, transition, interval) in enumerate(scenarios):
        for index in range(count):
            load_seed = seed + 10_000 * source_index + index
            demand = generate_zuo_markov_system_load(
                load_seed,
                length_s=length_s,
                decision_interval_s=interval,
                system_power_reference_kw=power_reference,
                transition_matrix=transition,
                initial_probabilities=probabilities,
                source=source,
            )
            scenario = TestScenario(
                name=f"{source}_{index:03d}",
                demand=demand,
                initial_damage_fraction=(0.0, 0.0, 0.0),
                health_seed=40_000 + load_seed,
                stochastic_health=False,
            )
            tasks.append(
                (
                    model,
                    scenario,
                    {
                        "template_id": scenario.name,
                        "template_source": source,
                        "source_seed": load_seed,
                        "segment_id": -1,
                        "start_offset": -1,
                        "start_timestamp": "",
                        "end_timestamp": "",
                        "raw_power_mean_kw": np.nan,
                        "raw_power_max_kw": np.nan,
                        "positive_share": 1.0,
                    },
                )
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--real-count", type=int, default=48)
    parser.add_argument("--markov-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.length, args.real_count, args.markov_count, args.jobs) <= 0:
        raise ValueError("length, counts and jobs must be positive")

    frame = pd.read_csv(
        SOURCE,
        usecols=["timestamp", "segment_id", "target_power_kw", "fc_input_power_kw"],
    )
    split = split_at_largest_segment_gap(frame)
    audit_metadata = json.loads((AUDIT / "metadata.json").read_text(encoding="utf-8"))
    if list(split.calibration_segments) != audit_metadata["calibration_segments"]:
        raise AssertionError("calibration segment boundary differs from audited metadata")
    if list(split.holdout_segments) != audit_metadata["holdout_segments"]:
        raise AssertionError("holdout segment boundary differs from audited metadata")
    if not set(frame[frame.segment_id.isin(split.calibration_segments)].segment_id).isdisjoint(
        split.holdout_segments
    ):
        raise AssertionError("calibration and holdout segments overlap")

    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=(1.0, 1.0, 1.0),
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    power_reference = (1 - CAPACITY_RESERVE_FRACTION) * model.fc_power_reference_kw()
    matrix, probabilities = load_empirical_inputs()
    tasks = real_tasks(
        frame,
        split,
        model,
        args.real_count,
        args.length,
        args.seed,
        power_reference,
    )
    tasks.extend(
        markov_tasks(
            model,
            args.markov_count,
            args.length,
            args.seed,
            power_reference,
            matrix,
            probabilities,
        )
    )
    started = time.perf_counter()
    existing_path = args.out_dir / "service_exposure_templates.csv"
    existing_metadata_path = args.out_dir / "metadata.json"
    existing_metadata = {}
    if args.reuse_existing:
        if not existing_path.exists():
            raise FileNotFoundError("--reuse-existing requires an existing template CSV")
        table = pd.read_csv(existing_path)
        if existing_metadata_path.exists():
            existing_metadata = json.loads(
                existing_metadata_path.read_text(encoding="utf-8")
            )
        expected_ids = {task[2]["template_id"] for task in tasks}
        if set(table.template_id) != expected_ids or len(table) != len(tasks):
            raise AssertionError("existing template CSV does not match requested tasks")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            rows = list(executor.map(run_template, tasks, chunksize=1))
        table = pd.DataFrame(rows).sort_values("template_id").reset_index(drop=True)
    if int(table.constraint_violation_steps.sum()) != 0:
        raise AssertionError("template execution contains constraint violations")
    if not np.allclose(table.tracking_within_tolerance_share, 1.0):
        raise AssertionError("template execution exceeded FC tracking tolerance")

    start_rate, start_rate_ci95 = calibration_start_rate(
        frame, split.calibration_segments, seed=args.seed
    )
    health_calibration = json.loads(HEALTH_CALIBRATION.read_text(encoding="utf-8"))
    start_damage = float(
        health_calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
    )
    real_mask = table.template_source == "real_calibration_window"
    operational_start = table.loc[real_mask, "duration_h"] * start_rate * start_damage
    table.loc[real_mask, "role_0_operational_start_damage_pct"] = operational_start
    table.loc[real_mask, "role_1_operational_start_damage_pct"] = operational_start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "service_exposure_templates.csv", index=False)
    table["continuous_total_pct"] = (
        table.role_0_continuous_mean_pct + table.role_1_continuous_mean_pct
    )
    table["operational_start_total_pct"] = (
        table.role_0_operational_start_damage_pct
        + table.role_1_operational_start_damage_pct
    )
    summary = table.groupby("template_source", sort=False).agg(
        template_count=("template_id", "count"),
        duration_mean_h=("duration_h", "mean"),
        positive_share_mean=("positive_share", "mean"),
        continuous_total_mean_pct=("continuous_total_pct", "mean"),
        operational_start_total_mean_pct=("operational_start_total_pct", "mean"),
        tracking_mae_mean_kw=("tracking_mae_kw", "mean"),
    ).reset_index()
    summary.to_csv(args.out_dir / "template_summary.csv", index=False)
    metadata = {
        "scope": "development-only fast-layer exposure templates",
        "source": str(SOURCE.relative_to(ROOT)),
        "calibration_segments": list(split.calibration_segments),
        "forbidden_holdout_segments": list(split.holdout_segments),
        "selection": "uniform without replacement over valid calibration-window starts",
        "window_length_s": args.length,
        "real_window_count": args.real_count,
        "markov_count_per_source": args.markov_count,
        "seed": args.seed,
        "fixed_fast_layer_assignment": list(ASSIGNMENT),
        "initial_damage_fraction": [0.0, 0.0, 0.0],
        "heterogeneity_factors": [1.0, 1.0, 1.0],
        "entry_start_handling": "block-entry starts are excluded from fast trajectories; real operational starts use the complete calibration-segment rate",
        "calibration_start_events": int(
            round(start_rate * audit_metadata["calibration_rows"] / 3600.0)
        ),
        "calibration_start_rate_per_observed_hour": start_rate,
        "calibration_start_rate_segment_bootstrap_ci95": list(start_rate_ci95),
        "real_template_operational_start_model": "complete-calibration-segment mean rate applied to both assigned roles",
        "fast_layer_results_reused_in_this_run": args.reuse_existing,
        "fast_layer_runtime_s": (
            existing_metadata.get(
                "fast_layer_runtime_s", existing_metadata.get("runtime_s")
            )
            if args.reuse_existing
            else time.perf_counter() - started
        ),
        "source_pooling": "sources remain separate and must not be uniformly mixed for primary claims",
        "postprocess_runtime_s": time.perf_counter() - started,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 快层服务暴露模板库\n\n"
    report += (
        "主模板只从segment 0-21连续窗口抽取；segment 22-45未参与窗口选择、"
        "归一化或控制参数调整。Instant快层被慢层角色固定为两个指定堆，"
        "第三堆不得上线。块入口的人为启动不计入模板；实车运行启停使用完整"
        f"校准段统计的{start_rate:.3f}次/观测小时，segment重采样95%区间为"
        f"[{start_rate_ci95[0]:.3f}, {start_rate_ci95[1]:.3f}]。\n\n"
    )
    report += summary.to_markdown(index=False) + "\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
