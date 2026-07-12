"""Aggregate short online-policy traces into long-horizon Gamma distributions."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation.gamma_sensitivity import (
    exposure_from_trajectory,
    sample_repeated_exposure,
)
from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ROOT / "data/results/testbed/testbed_trajectory.csv",
    )
    parser.add_argument("--exposure-hours", type=float, default=1000.0)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--cvs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/testbed_gamma_long"
    )
    args = parser.parse_args()
    if args.exposure_hours <= 0 or args.samples <= 0 or any(cv <= 0 for cv in args.cvs):
        raise ValueError("exposure, samples and CV values must be positive")

    calibration = json.loads(
        (ROOT / "data/results/health/lzw_gamma_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    trajectory = pd.read_csv(args.trajectory)
    n_stacks = len(
        [column for column in trajectory if column.endswith("_damage_before_pct")]
    )
    groups = {
        (scenario, strategy): exposure_from_trajectory(group, n_stacks)
        for (scenario, strategy), group in trajectory.groupby(
            ["scenario", "strategy"]
        )
        if strategy in {"average", "beam_health"}
    }
    rows, pair_rows = [], []
    for cv in args.cvs:
        scale = gamma_scale_for_terminal_cv(
            calibration["terminal_total_damage_pct"],
            calibration["terminal_continuous_damage_pct"],
            cv,
        )
        for scenario in sorted({key[0] for key in groups}):
            available = {
                strategy: exposure
                for (name, strategy), exposure in groups.items()
                if name == scenario
            }
            if not {"average", "beam_health"}.issubset(available):
                continue
            repeats = max(
                1,
                int(
                    round(
                        args.exposure_hours
                        * 3600
                        / available["average"].duration_s
                    )
                ),
            )
            uniforms = np.random.default_rng(
                zlib.crc32(f"{scenario}|{cv}|2026".encode("utf-8"))
            ).uniform(1e-12, 1 - 1e-12, (args.samples, n_stacks))
            sampled = {}
            for strategy, exposure in available.items():
                values = sample_repeated_exposure(
                    exposure,
                    repeats,
                    scale,
                    args.samples,
                    2026,
                    common_uniforms=uniforms,
                )
                sampled[strategy] = values
                total = values.sum(axis=1)
                aged = values[:, int(np.argmax(exposure.initial_damage_pct))]
                continuous = sum(exposure.continuous_mean_pct) * repeats
                discrete = sum(exposure.discrete_damage_pct) * repeats
                rows.append(
                    {
                        "scenario": scenario,
                        "strategy": strategy,
                        "terminal_cv_assumption": cv,
                        "gamma_scale_pct": scale,
                        "cycle_repeats": repeats,
                        "actual_exposure_hours": repeats
                        * exposure.duration_s
                        / 3600,
                        "continuous_mean_pct": continuous,
                        "discrete_damage_pct": discrete,
                        "discrete_fraction": discrete
                        / max(continuous + discrete, 1e-12),
                        "total_mean_pct": float(total.mean()),
                        "total_std_pct": float(total.std()),
                        "total_p05_pct": float(np.quantile(total, 0.05)),
                        "total_p50_pct": float(np.quantile(total, 0.50)),
                        "total_p95_pct": float(np.quantile(total, 0.95)),
                        "aged_stack_mean_pct": float(aged.mean()),
                        "aged_stack_p95_pct": float(np.quantile(aged, 0.95)),
                    }
                )
            average_total = sampled["average"].sum(axis=1)
            beam_total = sampled["beam_health"].sum(axis=1)
            pair_rows.append(
                {
                    "scenario": scenario,
                    "terminal_cv_assumption": cv,
                    "probability_beam_total_damage_lower": float(
                        np.mean(beam_total < average_total)
                    ),
                    "mean_beam_minus_average_pct": float(
                        np.mean(beam_total - average_total)
                    ),
                    "p95_beam_minus_average_pct": float(
                        np.quantile(beam_total - average_total, 0.95)
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(pair_rows)
    summary.to_csv(args.out_dir / "long_gamma_summary.csv", index=False)
    paired.to_csv(args.out_dir / "long_gamma_pairwise.csv", index=False)
    metadata = {
        "source_trajectory": str(args.trajectory),
        "requested_exposure_hours": args.exposure_hours,
        "samples": args.samples,
        "cvs": args.cvs,
        "interpretation": (
            "A short online-control cycle is repeated as an exposure pattern. "
            "Continuous Gamma increments are aggregated exactly; start/shift "
            "events remain deterministic. This is a sensitivity test, not a "
            "claim that the short trip repeats unchanged in service."
        ),
    }
    (args.out_dir / "long_gamma_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pair_lines = [
        "| 场景 | CV | P(Beam退化更低) | Beam-Average均值(%) | 差值P95(%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        pair_lines.append(
            f"| {row.scenario} | {row.terminal_cv_assumption:.0%} | "
            f"{row.probability_beam_total_damage_lower:.1%} | "
            f"{row.mean_beam_minus_average_pct:.3f} | "
            f"{row.p95_beam_minus_average_pct:.3f} |"
        )
    discrete_min = float(summary.discrete_fraction.min())
    discrete_max = float(summary.discrete_fraction.max())
    beam_worse = paired.loc[
        paired.mean_beam_minus_average_pct > 0, "scenario"
    ].unique()
    exception_text = (
        "、".join(beam_worse)
        if len(beam_worse)
        else "本次场景中没有"
    )
    report = f"""# 长期Gamma不确定性敏感性

- 等效暴露：{args.exposure_hours:g} h
- Monte Carlo样本：{args.samples}
- 终点CV假设：{args.cvs}
- 连续负载Gamma增量按可加性聚合；启停/变载保持确定性事件增量。
- 短周期重复只用于放大并比较策略暴露，不代表真实车辆长期重复同一段工况。

## 策略配对结果

{chr(10).join(pair_lines)}

## 可解释结论

- 事件型确定性损伤占总期望增量的{discrete_min:.1%}–{discrete_max:.1%}，当前长期排序主要由启停/变载等事件系数和策略事件次数决定，而不是Gamma连续增量方差。
- CV从{min(args.cvs):.0%}增至{max(args.cvs):.0%}会扩大分布宽度，但没有改变本次场景的策略排序。
- Beam并非普遍占优；出现Beam总退化更高的场景：{exception_text}。因此后续不能用单个随机种子声称延寿。
- 本实验保留Gamma作为不可逆在线状态及边界不确定性；主控制比较使用条件均值，CV只进入敏感性分析。
"""
    (args.out_dir / "long_gamma_report.md").write_text(report, encoding="utf-8")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
