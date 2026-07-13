"""Replay frozen FC-only control on separate cross-month real-power blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap, wilcoxon

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    TestScenario,
    orient_service_pair,
    run_policy,
    stationary_service_exposure,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]
BLOCK_DIR = ROOT / "data/results/fc_only_external_monthly_blocks"
BLOCKS = BLOCK_DIR / "external_monthly_power_blocks.csv"
BLOCK_MANIFEST = BLOCK_DIR / "block_manifest.csv"
TEMPLATES = ROOT / "data/results/fc_only_service_templates_norm40/service_exposure_templates.csv"
HEALTH_CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_external_monthly_replay"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
HETEROGENEITY = (1.0, 1.05, 1.10)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}
POLICIES = ("fixed_pair", "health_greedy")
NORMALIZATION_POWER_KW = 40.0
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260713


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_exposure() -> ServiceExposure:
    table = pd.read_csv(TEMPLATES)
    selected = table[table.template_source.eq("real_calibration_window")]
    templates = [
        ServiceExposure(
            duration_h=float(row.duration_h),
            continuous_mean_pct=(
                row.role_0_continuous_mean_pct,
                row.role_1_continuous_mean_pct,
            ),
            load_shift_damage_pct=(
                row.role_0_load_shift_damage_pct,
                row.role_1_load_shift_damage_pct,
            ),
            operational_start_damage_pct=(
                row.role_0_operational_start_damage_pct,
                row.role_1_operational_start_damage_pct,
            ),
        )
        for row in selected.itertuples(index=False)
    ]
    if len(templates) != 48:
        raise AssertionError("external replay requires 48 frozen real templates")
    return stationary_service_exposure(templates, 1.0)


def frozen_assignment(policy, initial_fraction, exposure, config):
    damage = tuple(float(value * config.health_limit_pct) for value in initial_fraction)
    state = ServiceScheduleState(damage)
    if policy == "fixed_pair":
        pair = (0, 1)
    elif policy == "health_greedy":
        pair = tuple(sorted(range(3), key=lambda index: (damage[index], index))[:2])
    else:
        raise ValueError(f"unknown policy: {policy}")
    return orient_service_pair(pair, state, exposure, config.heterogeneity_factors)


def run_case(task):
    model, scenario, assignment, descriptor, capture_trace = task
    try:
        run = run_policy(
            model,
            scenario,
            "instant_health",
            fixed_online_assignment=assignment,
        )
    except RuntimeError as error:
        return {**descriptor, "assignment": str(assignment), "error": str(error)}, None

    trajectory = run.trajectory
    final_damage = np.asarray(
        [trajectory[f"stack_{index}_damage_after_pct"].iloc[-1] for index in range(model.n_stacks)]
    )
    initial_damage = np.asarray(
        [trajectory[f"stack_{index}_damage_before_pct"].iloc[0] for index in range(model.n_stacks)]
    )
    tracking = trajectory.fc_power_tracking_error_kw.to_numpy(dtype=float)
    energy = float(run.metrics["fc_energy_kwh"])
    row = {
        **descriptor,
        "assignment": str(assignment),
        "assignment_first": assignment[0],
        "assignment_second": assignment[1],
        "error": "",
        "n_steps": len(trajectory),
        "fc_energy_kwh": energy,
        "hydrogen_g": run.metrics["hydrogen_g"],
        "hydrogen_g_per_fc_kwh": float(run.metrics["hydrogen_g"] / max(energy, 1e-12)),
        "expected_damage_increment_pct": run.metrics["main_expected_damage_increment_pct"],
        "terminal_max_damage_pct": float(final_damage.max()),
        "terminal_damage_range_pct": float(final_damage.max() - final_damage.min()),
        "damage_increment_range_pct": float(
            (final_damage - initial_damage).max() - (final_damage - initial_damage).min()
        ),
        "tracking_mae_kw": float(np.abs(tracking).mean()),
        "tracking_rmse_kw": float(np.sqrt(np.square(tracking).mean())),
        "tracking_max_abs_kw": float(np.abs(tracking).max()),
        "constraint_violation_steps": run.metrics["constraint_violation_steps"],
        "safety_override_steps": run.metrics["safety_override_steps"],
        "total_switch_count": run.metrics["total_switch_count"],
        "planning_runtime_s": run.metrics["planning_runtime_s"],
    }
    trace = None
    if capture_trace:
        trace = pd.DataFrame(
            {
                "step": trajectory.step,
                "demand_power_kw": trajectory.demand_power_kw,
                "stack_power_kw": trajectory.stack_power_kw,
                "tracking_error_kw": trajectory.fc_power_tracking_error_kw,
            }
        )
    return row, trace


def pair_policies(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "terminal_max_damage_pct",
        "terminal_damage_range_pct",
        "damage_increment_range_pct",
        "expected_damage_increment_pct",
        "hydrogen_g_per_fc_kwh",
        "tracking_mae_kw",
        "tracking_rmse_kw",
        "tracking_max_abs_kw",
        "safety_override_steps",
        "total_switch_count",
    )
    keys = ["block_id", "month", "health_case", "positive_steps", "above_40kw_steps"]
    wide = per_run.pivot(index=keys, columns="policy", values=list(metrics))
    rows = []
    for index, values in wide.iterrows():
        row = dict(zip(keys, index))
        for metric in metrics:
            fixed = float(values[(metric, "fixed_pair")])
            greedy = float(values[(metric, "health_greedy")])
            row[f"fixed_{metric}"] = fixed
            row[f"health_greedy_{metric}"] = greedy
            row[f"delta_{metric}"] = greedy - fixed
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_interval(values, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if np.allclose(array, array[0], atol=1e-15):
        value = float(array[0])
        return value, value, value
    result = bootstrap(
        (array,),
        np.mean,
        n_resamples=samples,
        batch=2000,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    return (
        float(array.mean()),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def statistical_summary(paired: pd.DataFrame):
    informative = paired[paired.health_case.isin(["oldest_stack_0", "oldest_stack_1"])]
    block_effects = (
        informative.groupby(["block_id", "month"], sort=True)
        .agg(
            delta_terminal_max_damage_pct=("delta_terminal_max_damage_pct", "mean"),
            delta_expected_damage_increment_pct=("delta_expected_damage_increment_pct", "mean"),
            delta_tracking_mae_kw=("delta_tracking_mae_kw", "mean"),
            delta_hydrogen_g_per_fc_kwh=("delta_hydrogen_g_per_fc_kwh", "mean"),
        )
        .reset_index()
    )
    month_effects = (
        block_effects.groupby("month", sort=True)
        .agg(
            trajectory_blocks=("block_id", "size"),
            delta_terminal_max_damage_pct=("delta_terminal_max_damage_pct", "mean"),
            delta_expected_damage_increment_pct=("delta_expected_damage_increment_pct", "mean"),
            delta_tracking_mae_kw=("delta_tracking_mae_kw", "mean"),
            delta_hydrogen_g_per_fc_kwh=("delta_hydrogen_g_per_fc_kwh", "mean"),
        )
        .reset_index()
    )
    primary = month_effects.delta_terminal_max_damage_pct.to_numpy(dtype=float)
    mean, low, high = bootstrap_mean_interval(primary)
    nonzero = primary[~np.isclose(primary, 0.0, atol=1e-15)]
    wilcoxon_p = (
        float(wilcoxon(nonzero, alternative="less", method="exact").pvalue)
        if len(nonzero)
        else 1.0
    )
    better = int((primary < -1e-15).sum())
    sign_p = float(binomtest(better, len(primary), 0.5, alternative="greater").pvalue)
    summary = pd.DataFrame(
        [
            {
                "analysis_unit": "calendar_month",
                "months": len(primary),
                "trajectory_blocks": len(block_effects),
                "primary_metric": "month mean across three time strata and two informative health-identity rotations",
                "terminal_max_delta_mean_pct": mean,
                "terminal_max_delta_ci95_low_pct": low,
                "terminal_max_delta_ci95_high_pct": high,
                "better_months": better,
                "nonworse_months": int((primary <= 1e-15).sum()),
                "wilcoxon_one_sided_p": wilcoxon_p,
                "sign_test_one_sided_p": sign_p,
            }
        ]
    )
    diagnostics = []
    for health_case, group in paired.groupby("health_case", sort=False):
        monthly = group.groupby("month", sort=True).agg(
            terminal_max_delta=("delta_terminal_max_damage_pct", "mean"),
            total_damage_delta=("delta_expected_damage_increment_pct", "mean"),
            tracking_delta=("delta_tracking_mae_kw", "mean"),
            hydrogen_delta=("delta_hydrogen_g_per_fc_kwh", "mean"),
        )
        values = monthly.terminal_max_delta.to_numpy(dtype=float)
        effect_mean, effect_low, effect_high = bootstrap_mean_interval(
            values, seed=BOOTSTRAP_SEED + len(diagnostics) + 1
        )
        diagnostics.append(
            {
                "health_case": health_case,
                "months": len(values),
                "trajectory_blocks": len(group),
                "terminal_max_delta_mean_pct": effect_mean,
                "terminal_max_delta_ci95_low_pct": effect_low,
                "terminal_max_delta_ci95_high_pct": effect_high,
                "better_share": float((values < -1e-15).mean()),
                "nonworse_share": float((values <= 1e-15).mean()),
                "total_damage_delta_mean_pct": float(
                    monthly.total_damage_delta.mean()
                ),
                "tracking_mae_delta_mean_kw": float(monthly.tracking_delta.mean()),
                "hydrogen_intensity_delta_mean_g_per_kwh": float(
                    monthly.hydrogen_delta.mean()
                ),
            }
        )
    return summary, pd.DataFrame(diagnostics), block_effects, month_effects


def plot_results(manifest, month_effects, per_run, trace, output_path):
    monthly_power = (
        manifest.groupby("month", sort=True)
        .agg(
            power_mean_kw=("power_mean_kw", "mean"),
            power_max_kw=("power_max_kw", "max"),
        )
        .reset_index()
    )
    months = monthly_power.month.astype(str).str[2:]
    x = np.arange(len(monthly_power))
    colors = {"blue": "#33658A", "orange": "#F6AE2D", "red": "#D1495B", "gray": "#6C757D"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))
    axes[0].plot(x, monthly_power.power_mean_kw, "o-", color=colors["blue"], ms=3, lw=1.1)
    axes[0].plot(x, monthly_power.power_max_kw, "s-", color=colors["orange"], ms=3, lw=1.0)
    axes[0].axhline(40, color=colors["gray"], ls="--", lw=0.8)
    axes[0].text(0.15, 40.45, "40 kW reference", color=colors["gray"], fontsize=6.4)
    axes[0].text(x[-1] + 0.15, monthly_power.power_mean_kw.iloc[-1], "Mean", color=colors["blue"], fontsize=6.4, va="center")
    axes[0].text(x[-1] + 0.15, monthly_power.power_max_kw.iloc[-1], "Maximum", color=colors["orange"], fontsize=6.4, va="center")
    axes[0].set_xlim(-0.6, x[-1] + 1.15)
    axes[0].set_xticks(x[::2], months.iloc[::2], rotation=35, ha="right")
    axes[0].set_xlabel("External month")
    axes[0].set_ylabel("Observed stack power (kW)")

    primary = month_effects.delta_terminal_max_damage_pct.to_numpy() * 1e3
    axes[1].bar(x, primary, color=np.where(primary <= 0, colors["blue"], colors["red"]), width=0.72)
    axes[1].axhline(0, color=colors["gray"], lw=0.8)
    axes[1].set_xticks(x[::2], months.iloc[::2], rotation=35, ha="right")
    axes[1].set_xlabel("External month")
    axes[1].set_ylabel("Greedy - fixed max damage\n($10^{-3}$ %-point)")

    total = month_effects.delta_expected_damage_increment_pct.to_numpy() * 1e3
    axes[2].scatter(total, primary, s=25, color=colors["orange"], edgecolor="white", linewidth=0.4)
    axes[2].axhline(0, color=colors["gray"], lw=0.8)
    axes[2].axvline(0, color=colors["gray"], lw=0.8)
    axes[2].set_xlabel("Total-damage change\n($10^{-3}$ %-point)")
    axes[2].set_ylabel("Max-damage change\n($10^{-3}$ %-point)")
    label_indices = tuple(
        dict.fromkeys(
            (
                int(np.argmin(primary)),
                int(np.argmin(total)),
                int(np.argmax(total)),
            )
        )
    )
    offsets = ((4, 4), (4, -10), (-24, 4))
    for index, offset in zip(label_indices, offsets):
        axes[2].annotate(
            months.iloc[index],
            (total[index], primary[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
        )

    for index, axis in enumerate(axes):
        axis.text(-0.14, 1.04, chr(ord("a") + index), transform=axis.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.65, w_pad=1.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")

    if args.summarize_only:
        per_run = pd.read_csv(args.out_dir / "per_run_metrics.csv")
        paired = pd.read_csv(args.out_dir / "paired_policy_deltas.csv")
        manifest = pd.read_csv(BLOCK_MANIFEST)
        trace = pd.read_csv(args.out_dir / "representative_trace.csv")
        summary, diagnostics, block_effects, month_effects = statistical_summary(
            paired
        )
    else:
        if not BLOCKS.exists():
            raise FileNotFoundError("run script 56 before external replay")
        blocks = pd.read_csv(BLOCKS, parse_dates=["timestamp"])
        manifest = pd.read_csv(BLOCK_MANIFEST)
        representative_block_id = str(
            manifest.loc[manifest.power_max_kw.idxmax(), "block_id"]
        )
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
        system_reference_kw = (1 - CAPACITY_RESERVE_FRACTION) * model.fc_power_reference_kw()
        calibration = json.loads(HEALTH_CALIBRATION.read_text(encoding="utf-8"))
        schedule_config = ServiceScheduleConfig(
            health_limit_pct=float(calibration["terminal_total_damage_pct"]),
            gamma_scale_pct=float(calibration["gamma_scale_pct"]),
            heterogeneity_factors=HETEROGENEITY,
            start_damage_pct=float(calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]),
        )
        exposure = load_development_exposure()
        tasks = []
        for block_id, block in blocks.groupby("block_id", sort=True):
            block = block.sort_values("block_step").reset_index(drop=True)
            power = block.fc_input_power_kw.to_numpy(dtype=float)
            normalized = np.maximum(power, 0.0) / NORMALIZATION_POWER_KW
            demand = pd.DataFrame(
                {
                    "demand_power_kw": normalized * system_reference_kw,
                    "event": np.where(normalized > 0, "external_real_on", "external_real_off"),
                    "source": "external_monthly_real_power",
                    "seed": int(block.month.iloc[0].replace("-", "")),
                }
            )
            for health_case, initial_fraction in HEALTH_CASES.items():
                scenario = TestScenario(
                    name=f"{block_id}_{health_case}",
                    demand=demand,
                    initial_damage_fraction=initial_fraction,
                    health_seed=70_000 + int(block.month.iloc[0].replace("-", "")),
                    stochastic_health=False,
                )
                for policy in POLICIES:
                    assignment = frozen_assignment(policy, initial_fraction, exposure, schedule_config)
                    descriptor = {
                        "block_id": block_id,
                        "month": block.month.iloc[0],
                        "stratum": int(block.stratum.iloc[0]),
                        "health_case": health_case,
                        "policy": policy,
                        "positive_steps": int((power >= 0.5).sum()),
                        "above_40kw_steps": int((power > 40.0).sum()),
                        "negative_power_steps": int((power < 0.0).sum()),
                    }
                    capture = (
                        block_id == representative_block_id
                        and health_case == "oldest_stack_0"
                        and policy == "health_greedy"
                    )
                    tasks.append((model, scenario, assignment, descriptor, capture))

        started = time.perf_counter()
        rows, traces = [], []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for index, (row, trace) in enumerate(executor.map(run_case, tasks, chunksize=1), 1):
                rows.append(row)
                if trace is not None:
                    traces.append(trace)
                if index % 13 == 0 or index == len(tasks):
                    print(f"completed {index}/{len(tasks)} external cases", flush=True)
        per_run = pd.DataFrame(rows)
        failures = per_run[per_run.error != ""]
        if len(failures):
            args.out_dir.mkdir(parents=True, exist_ok=True)
            per_run.to_csv(args.out_dir / "per_run_metrics_with_failures.csv", index=False)
            raise RuntimeError(f"{len(failures)} external replay cases failed")
        if int(per_run.constraint_violation_steps.sum()) != 0:
            raise AssertionError("external replay contains hard constraint violations")
        if float(per_run.tracking_max_abs_kw.max()) > TRACKING_TOLERANCE_KW + 1e-12:
            raise AssertionError("external replay exceeded the frozen tracking tolerance")
        if len(traces) != 1:
            raise AssertionError("representative external trace was not captured exactly once")
        paired = pair_policies(per_run)
        summary, diagnostics, block_effects, month_effects = statistical_summary(
            paired
        )
        trace = traces[0]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(args.out_dir / "per_run_metrics.csv", index=False)
        paired.to_csv(args.out_dir / "paired_policy_deltas.csv", index=False)
        trace.to_csv(args.out_dir / "representative_trace.csv", index=False)
        metadata = {
            "scope": "frozen control replay on a separate cross-month real-power cohort",
            "block_input": str(BLOCKS.relative_to(ROOT)),
            "block_input_sha256": sha256(BLOCKS),
            "template_input": str(TEMPLATES.relative_to(ROOT)),
            "template_input_sha256": sha256(TEMPLATES),
            "blocks": int(blocks.block_id.nunique()),
            "months": int(blocks.month.nunique()),
            "base_power_steps": len(blocks),
            "evaluated_cases": len(per_run),
            "evaluated_steps": int(per_run.n_steps.sum()),
            "normalization_power_kw": NORMALIZATION_POWER_KW,
            "power_above_reference_handling": "not clipped; replayed against physical action/constraint model",
            "negative_power_handling": "negative FC input is mapped to zero because the FC system cannot absorb regenerative power",
            "representative_trace_block": representative_block_id,
            "mapping_system_power_reference_kw": system_reference_kw,
            "capacity_reserve_fraction": CAPACITY_RESERVE_FRACTION,
            "tracking_tolerance_kw": TRACKING_TOLERANCE_KW,
            "health_cases": HEALTH_CASES,
            "policies": list(POLICIES),
            "future_demand_used": False,
            "controller_retuned": False,
            "health_state_reset_per_block": True,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_method": "BCa at calendar-month level",
            "runtime_s": time.perf_counter() - started,
        }
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary.to_csv(args.out_dir / "primary_statistics.csv", index=False)
    diagnostics.to_csv(args.out_dir / "health_case_diagnostics.csv", index=False)
    block_effects.to_csv(args.out_dir / "trajectory_block_effects.csv", index=False)
    month_effects.to_csv(args.out_dir / "month_effects.csv", index=False)
    legacy_effects = args.out_dir / "month_block_effects.csv"
    if legacy_effects.exists():
        legacy_effects.unlink()
    figure = args.out_dir / "fig20_external_monthly_replay.png"
    plot_results(manifest, month_effects, per_run, trace, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / figure.name).write_bytes(figure.read_bytes())

    primary = summary.iloc[0]
    total_delta = float(month_effects.delta_expected_damage_increment_pct.mean())
    tracking_delta = float(month_effects.delta_tracking_mae_kw.mean())
    hydrogen_delta = float(month_effects.delta_hydrogen_g_per_fc_kwh.mean())
    safety_overrides = int(per_run.safety_override_steps.sum())
    above_reference_steps = int(manifest.above_40kw_steps.sum())
    above_reference_months = int(
        manifest.loc[manifest.above_40kw_steps > 0, "month"].nunique()
    )
    interpolated_steps = int(manifest.interpolated_power_steps.sum())
    negative_steps = int(manifest.negative_power_steps.sum())
    informative_cases = int(
        per_run.health_case.isin(["oldest_stack_0", "oldest_stack_1"]).sum()
    )
    report = f"""# 独立跨月实车功率回放

- 2025-06至2026-06共13个月，每月前/中/后三个预声明时间层各取1个连续块，共{int(manifest.shape[0])}块、{int(manifest.steps.sum()):,}个基础1秒功率点；3种健康身份和2种冻结策略形成{len(per_run)}例、{int(per_run.n_steps.sum()):,}个闭环评估步，其中{informative_cases}例构成{informative_cases // 2}个有信息的配对比较。
- 这些块不与原七天链拼接，每块重置健康和控制状态；控制器、40 kW参考、健康参数和慢层规则均未重调，也没有使用未来需求。
- 原始遥测按与七天canonical相同的规则去重并在不超过10秒的段内重采样；70,200步中{interpolated_steps:,}步是相邻遥测包之间的插值，任何大于10秒的缺口都没有跨越。负功率噪声{negative_steps}步按燃料电池不可吸收回馈功率映射为0。
- {above_reference_steps}个高于40 kW的点来自{above_reference_months}个月，均没有截断，最高观测功率{manifest.power_max_kw.max():.3f} kW；最大跟踪误差{per_run.tracking_max_abs_kw.max():.3f} kW，硬约束违规{int(per_run.constraint_violation_steps.sum())}步。
- 主统计单位仍是13个日历月：先在每个轨迹块内平均“最老堆为0/1”两个有信息的身份循环，再汇总同月三个时间层，避免把身份复制和同月块当成独立样本。
- Health-greedy相对固定双堆的终端最大退化变化均值为{primary.terminal_max_delta_mean_pct:+.9f}个百分点，月级BCa bootstrap 95%区间[{primary.terminal_max_delta_ci95_low_pct:+.9f}, {primary.terminal_max_delta_ci95_high_pct:+.9f}]；改善{int(primary.better_months)}/{int(primary.months)}个月，单侧精确Wilcoxon和符号检验均为p={primary.wilcoxon_one_sided_p:.6g}。该p值是13/13同号时的精确下界，效应大小以均值和区间为准。
- 对应总期望退化变化{total_delta:+.9f}个百分点、跟踪MAE变化{tracking_delta:+.4f} kW、氢耗强度变化{hydrogen_delta:+.4f} g/kWh；安全驻留覆盖{safety_overrides}步，占全部评估步{safety_overrides / int(per_run.n_steps.sum()):.3%}。
- 该实验补强跨月真实功率可执行性和健康均衡方向，不证明同一物理电堆跨13个月连续老化，也不把30分钟块外推为寿命。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
