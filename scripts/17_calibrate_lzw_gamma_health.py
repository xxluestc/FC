"""Calibrate the Gamma health index and theta map on LZW event exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.health.lzw_gamma_calibration import (
    GhaderiPeiCoefficients,
    THETA_COLUMNS,
    cumulative_damage_components,
    fit_theta_power_law,
    gamma_scale_for_terminal_cv,
    monte_carlo_recorded_exposure,
    validate_lzw_alignment,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "data/upstream_lzw/canonical_event_table_6104.csv",
    )
    parser.add_argument(
        "--theta",
        type=Path,
        default=ROOT / "data/upstream_lzw/theta_event_trajectory_6104.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/results/health"
    )
    parser.add_argument("--terminal-cv", type=float, default=0.10)
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    theta = pd.read_csv(args.theta)
    validate_lzw_alignment(events, theta)

    coefficients = GhaderiPeiCoefficients()
    components = cumulative_damage_components(events, coefficients)
    mapping, fitted_theta, fit_metrics = fit_theta_power_law(
        components["total_damage_pct"], theta
    )

    terminal_total = float(components["total_damage_pct"].iloc[-1])
    terminal_continuous = float(components["continuous_damage_pct"].iloc[-1])
    sensitivity = {
        f"cv_{int(round(cv * 100)):02d}_pct": gamma_scale_for_terminal_cv(
            terminal_total, terminal_continuous, cv
        )
        for cv in (0.05, 0.10, 0.20)
    }
    gamma_scale = gamma_scale_for_terminal_cv(
        terminal_total, terminal_continuous, args.terminal_cv
    )

    damage_mc, theta_mc = monte_carlo_recorded_exposure(
        components,
        mapping,
        gamma_scale,
        samples=args.mc_samples,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.concat(
        [
            theta[["event_id", "canonical_row_6104", "original_index"]].reset_index(
                drop=True
            ),
            components.reset_index(drop=True),
        ],
        axis=1,
    )
    for index, column in enumerate(THETA_COLUMNS):
        trajectory[f"observed_{column}"] = theta[column].to_numpy()
        trajectory[f"fitted_mean_{column}"] = fitted_theta[:, index]
        trajectory[f"mc_p05_{column}"] = np.quantile(theta_mc[:, :, index], 0.05, axis=0)
        trajectory[f"mc_p50_{column}"] = np.quantile(theta_mc[:, :, index], 0.50, axis=0)
        trajectory[f"mc_p95_{column}"] = np.quantile(theta_mc[:, :, index], 0.95, axis=0)
    trajectory["mc_damage_mean_pct"] = damage_mc.mean(axis=0)
    trajectory["mc_damage_p05_pct"] = np.quantile(damage_mc, 0.05, axis=0)
    trajectory["mc_damage_p50_pct"] = np.quantile(damage_mc, 0.50, axis=0)
    trajectory["mc_damage_p95_pct"] = np.quantile(damage_mc, 0.95, axis=0)
    trajectory.to_csv(args.out_dir / "lzw_gamma_health_trajectory.csv", index=False)

    contribution_columns = [
        "start_stop_damage_pct",
        "high_load_damage_pct",
        "low_load_damage_pct",
        "natural_on_damage_pct",
        "load_shift_damage_pct",
    ]
    contributions = {
        column: float(components[column].iloc[-1]) for column in contribution_columns
    }
    contribution_fraction = {
        column: value / terminal_total for column, value in contributions.items()
    }
    summary = {
        "interpretation": (
            "Literature coefficients define relative mean action exposure; "
            "LZW theta calibrates the monotone state mapping. This is not an "
            "independent causal identification of material degradation rates."
        ),
        "n_events": len(events),
        "coefficients_source": "Ghaderi 2023 Table 4, citing Pei 2008",
        "operating_regime_definition": (
            "0 A uses the low-load term without natural-on time; positive "
            "current uses natural-on decay, with the high-load term added at "
            "370 A. Start/stop and load shift are additive events."
        ),
        "coefficients_percent_units": coefficients.__dict__,
        "terminal_total_damage_pct": terminal_total,
        "terminal_continuous_damage_pct": terminal_continuous,
        "terminal_discrete_damage_pct": float(
            components["discrete_event_damage_pct"].iloc[-1]
        ),
        "terminal_contributions_pct": contributions,
        "terminal_contribution_fraction": contribution_fraction,
        "theta_power_law_map": mapping.to_dict(),
        "theta_fit_metrics": fit_metrics,
        "gamma_terminal_cv_assumption": args.terminal_cv,
        "gamma_scale_pct": gamma_scale,
        "gamma_scale_sensitivity": sensitivity,
        "monte_carlo_samples": args.mc_samples,
        "random_seed": args.seed,
        "terminal_mc_damage_mean_pct": float(damage_mc[:, -1].mean()),
        "terminal_mc_damage_p05_pct": float(np.quantile(damage_mc[:, -1], 0.05)),
        "terminal_mc_damage_p50_pct": float(np.quantile(damage_mc[:, -1], 0.50)),
        "terminal_mc_damage_p95_pct": float(np.quantile(damage_mc[:, -1], 0.95)),
    }
    (args.out_dir / "lzw_gamma_calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# LZW Gamma健康状态标定报告

## 结论

使用Ghaderi 2023 Table 4（源自Pei 2008）的固定退化系数计算动作暴露指数，再将该指数单调映射到刘占伟6104事件UKF-PF健康参数。该方法实现了“候选动作→未来累计健康状态”的可运行结构，但不声称从单条实车轨迹独立辨识了各工况的材料退化因果系数。

## 数据和尺度

- 事件数：{len(events)}
- 记录末端累计动作损伤指数：{terminal_total:.6f}%
- 连续负载部分：{terminal_continuous:.6f}%
- 启停/变载离散部分：{summary['terminal_discrete_damage_pct']:.6f}%
- Gamma终点CV假设：{args.terminal_cv:.1%}
- 对应Gamma scale：{gamma_scale:.6g} %
- 工况口径：0 A只计低载；正电流计自然运行，370 A再叠加高载项；启停和变载作为可叠加离散事件。

## theta映射拟合

| 参数 | 幂指数 | R2 | RMSE |
|---|---:|---:|---:|
| i0 | {fit_metrics[THETA_COLUMNS[0]]['exponent']:.4f} | {fit_metrics[THETA_COLUMNS[0]]['r2']:.4f} | {fit_metrics[THETA_COLUMNS[0]]['rmse']:.4g} |
| ih | {fit_metrics[THETA_COLUMNS[1]]['exponent']:.4f} | {fit_metrics[THETA_COLUMNS[1]]['r2']:.4f} | {fit_metrics[THETA_COLUMNS[1]]['rmse']:.4g} |
| R_ohm | {fit_metrics[THETA_COLUMNS[2]]['exponent']:.4f} | {fit_metrics[THETA_COLUMNS[2]]['r2']:.4f} | {fit_metrics[THETA_COLUMNS[2]]['rmse']:.4g} |

## 不确定性边界

当前只有一条长期theta轨迹，无法可靠识别堆间Gamma方差。终点CV=5%/10%/20%作为敏感性假设，而不是实测参数。默认10%仅用于跑通随机健康rollout；论文结论必须同时报告敏感性结果。

## 下一步

把标定结果接入动态 `C_deg(I|theta_current)`，然后在候选动作rollout中同步递推Gamma损伤、theta、电压、功率、氢耗和SOC。
"""
    (args.out_dir / "lzw_gamma_calibration_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
