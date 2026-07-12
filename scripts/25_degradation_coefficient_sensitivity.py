"""Audit controller ranking across literature degradation coefficient scales."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import extract_action_exposure
from fc_power.health.lzw_gamma_calibration import GhaderiPeiCoefficients


ROOT = Path(__file__).resolve().parents[1]


def scaled_coefficients(
    baseline: GhaderiPeiCoefficients,
    continuous_multiplier: float,
    start_stop_multiplier: float,
    load_shift_multiplier: float,
) -> GhaderiPeiCoefficients:
    return baseline.scaled(
        continuous=continuous_multiplier,
        start_stop=start_stop_multiplier,
        load_shift=load_shift_multiplier,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ROOT / "data/results/testbed_multiseed/testbed_trajectory.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data/results/testbed_multiseed/testbed_metadata.json",
    )
    parser.add_argument(
        "--continuous-multipliers", nargs="+", type=float, default=[0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--start-stop-multipliers",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--load-shift-multipliers",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/results/testbed_coefficient_sensitivity",
    )
    args = parser.parse_args()
    multipliers = (
        args.continuous_multipliers
        + args.start_stop_multipliers
        + args.load_shift_multipliers
    )
    if any(not np.isfinite(value) or value < 0 for value in multipliers):
        raise ValueError("coefficient multipliers must be finite and non-negative")

    trajectory = pd.read_csv(args.trajectory)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    n_stacks = int(metadata["n_stacks"])
    heterogeneity = tuple(metadata["heterogeneity_factors"])
    maximum_current = float(
        max(metadata.get("allowed_currents_a", [0, 25, 90, 120, 160, 195, 270, 370]))
    )
    exposures = {
        (scenario, strategy): extract_action_exposure(
            group,
            n_stacks,
            heterogeneity,
            maximum_current_a=maximum_current,
        )
        for (scenario, strategy), group in trajectory.groupby(
            ["scenario", "strategy"]
        )
        if strategy in {"average", "beam_health"}
    }
    scenarios = sorted({scenario for scenario, _ in exposures})
    if not scenarios:
        raise ValueError("trajectory contains no Average/Beam scenario pairs")

    baseline = GhaderiPeiCoefficients()
    rows = []
    for continuous_multiplier in args.continuous_multipliers:
        for start_stop_multiplier in args.start_stop_multipliers:
            for load_shift_multiplier in args.load_shift_multipliers:
                coefficients = scaled_coefficients(
                    baseline,
                    continuous_multiplier,
                    start_stop_multiplier,
                    load_shift_multiplier,
                )
                scenario_rows = []
                for scenario in scenarios:
                    if not {
                        (scenario, "average"),
                        (scenario, "beam_health"),
                    }.issubset(exposures):
                        continue
                    average = exposures[(scenario, "average")].total_damage(
                        coefficients
                    )
                    beam = exposures[(scenario, "beam_health")].total_damage(
                        coefficients
                    )
                    source = scenario.rsplit("_seed_", 1)[0]
                    scenario_rows.append((source, beam - average, average))
                frame = pd.DataFrame(
                    scenario_rows,
                    columns=["load_source", "delta", "reference"],
                )
                for source, group in frame.groupby("load_source"):
                    delta = group.delta.to_numpy(dtype=float)
                    reference = group.reference.to_numpy(dtype=float)
                    std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
                    rows.append(
                        {
                            "load_source": source,
                            "continuous_multiplier": continuous_multiplier,
                            "start_stop_multiplier": start_stop_multiplier,
                            "load_shift_multiplier": load_shift_multiplier,
                            "n_pairs": len(delta),
                            "mean_beam_minus_average_pct": float(delta.mean()),
                            "ci95_pct": 1.96 * std / np.sqrt(len(delta)),
                            "mean_relative_pct": float(
                                np.mean(100.0 * delta / np.maximum(reference, 1e-12))
                            ),
                            "beam_lower_share": float(np.mean(delta < 0)),
                        }
                    )

    grid = pd.DataFrame(rows)
    robustness_rows = []
    for source, group in grid.groupby("load_source"):
        robust = group.mean_beam_minus_average_pct + group.ci95_pct < 0
        robustness_rows.append(
            {
                "load_source": source,
                "grid_points": len(group),
                "mean_beam_lower_grid_share": float(
                    (group.mean_beam_minus_average_pct < 0).mean()
                ),
                "ci95_beam_lower_grid_share": float(robust.mean()),
                "all_seed_beam_lower_grid_share": float(
                    group.beam_lower_share.eq(1.0).mean()
                ),
                "sign_reversal_grid_share": float(
                    (group.mean_beam_minus_average_pct > 0).mean()
                ),
                "minimum_mean_delta_pct": float(
                    group.mean_beam_minus_average_pct.min()
                ),
                "maximum_mean_delta_pct": float(
                    group.mean_beam_minus_average_pct.max()
                ),
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    baseline_rows = grid[
        grid.continuous_multiplier.eq(1.0)
        & grid.start_stop_multiplier.eq(1.0)
        & grid.load_shift_multiplier.eq(1.0)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "coefficient_grid_pairwise.csv", index=False)
    robustness.to_csv(args.out_dir / "coefficient_grid_robustness.csv", index=False)
    metadata_out = {
        "source_trajectory": str(args.trajectory),
        "source_metadata": str(args.metadata),
        "baseline_coefficients": asdict(baseline),
        "continuous_multipliers": args.continuous_multipliers,
        "start_stop_multipliers": args.start_stop_multipliers,
        "load_shift_multipliers": args.load_shift_multipliers,
        "n_stacks": n_stacks,
        "heterogeneity_factors": heterogeneity,
        "scope": (
            "Fixed-action exposure reweighting. It tests evaluation robustness "
            "but does not re-plan actions or replay health-dependent power."
        ),
    }
    (args.out_dir / "coefficient_sensitivity_metadata.json").write_text(
        json.dumps(metadata_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_lines = [
        "| 负载 | n | Beam-Average退化(%) | 相对变化 | 95% CI | Beam改善率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows.itertuples(index=False):
        baseline_lines.append(
            f"| {row.load_source} | {row.n_pairs} | "
            f"{row.mean_beam_minus_average_pct:.6f} | "
            f"{row.mean_relative_pct:+.2f}% | ±{row.ci95_pct:.6f} | "
            f"{row.beam_lower_share:.0%} |"
        )
    robustness_lines = [
        "| 负载 | 网格数 | 均值有利 | 95%CI有利 | 全种子有利 | 均值反转 | 差值范围(%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in robustness.itertuples(index=False):
        robustness_lines.append(
            f"| {row.load_source} | {row.grid_points} | "
            f"{row.mean_beam_lower_grid_share:.0%} | "
            f"{row.ci95_beam_lower_grid_share:.0%} | "
            f"{row.all_seed_beam_lower_grid_share:.0%} | "
            f"{row.sign_reversal_grid_share:.0%} | "
            f"[{row.minimum_mean_delta_pct:.6f}, "
            f"{row.maximum_mean_delta_pct:.6f}] |"
        )
    report = f"""# 退化事件系数敏感性门槛

## 口径

- 0 A只计低载；正电流计自然运行，90%以上最大电流再叠加高载项；启停与变载为可叠加事件。
- 连续工况系数扫描：{args.continuous_multipliers}；启停：{args.start_stop_multipliers}；变载：{args.load_shift_multipliers}。
- 使用同一10种子Average/Beam主段动作暴露做固定动作重加权；SOC恢复尾段不计入。
- 这是“评价指标对系数的稳健性”审计，不是闭环重新规划，也不证明文献系数适用于LZW电堆。

## 文献基准系数

{chr(10).join(baseline_lines)}

## 联合网格稳健性

{chr(10).join(robustness_lines)}

## 通过门槛

只有当策略方向在合理系数网格中大部分保持一致、配对95%区间不跨0，并在闭环重新规划验证中复现，才允许把“延寿”作为算法收益。固定动作重加权不通过时，应先修正退化目标；通过后仍需做闭环系数敏感性。
"""
    (args.out_dir / "coefficient_sensitivity_report.md").write_text(
        report, encoding="utf-8"
    )
    print(baseline_rows.to_string(index=False))
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
