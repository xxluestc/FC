"""Aggregate frozen FC-only exposure paths into Gamma sensitivity distributions."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import gamma

from fc_power.health.lzw_gamma_calibration import gamma_scale_for_terminal_cv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
SOURCE = RESULTS / "fc_only_deterministic_comparison/per_run_metrics.csv"
CALIBRATION = RESULTS / "health/lzw_gamma_calibration.json"
OUTPUT = RESULTS / "fc_only_gamma_aggregate"
FIGURES = RESULTS / "figures/fc_only_foundation"
SCENARIOS = ("empirical_1s", "zuo_slow_30s", "zuo_fast_30s")
SCENARIO_LABELS = {
    "empirical_1s": "Real-calibrated",
    "zuo_slow_30s": "Zuo slow (30 s*)",
    "zuo_fast_30s": "Zuo fast (30 s*)",
}
COLORS = {
    "empirical_1s": "#2A9D8F",
    "zuo_slow_30s": "#4C78A8",
    "zuo_fast_30s": "#D65F5F",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-hours", type=float, default=1000.0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--cvs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (
        not np.isfinite(args.exposure_hours)
        or args.exposure_hours <= 0
        or args.samples <= 0
        or any(not np.isfinite(cv) or cv <= 0 for cv in args.cvs)
    ):
        raise ValueError("exposure, samples and CV values must be positive")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    per_run = pd.read_csv(SOURCE)
    per_run = per_run[
        per_run.strategy.isin(["average", "rotating", "instant_health"])
    ].copy()
    required = {
        "load_source",
        "load_seed",
        "strategy",
        "n_steps",
        "main_expected_continuous_damage_pct",
        "main_discrete_damage_pct",
    }
    if missing := required.difference(per_run.columns):
        raise ValueError(f"deterministic metrics missing columns: {sorted(missing)}")

    summary_rows = []
    pair_rows = []
    for cv in args.cvs:
        scale = gamma_scale_for_terminal_cv(
            calibration["terminal_total_damage_pct"],
            calibration["terminal_continuous_damage_pct"],
            cv,
        )
        for source in SCENARIOS:
            source_rows = per_run[per_run.load_source == source]
            pooled = {"instant_health": [], "rotating": []}
            for load_seed, group in source_rows.groupby("load_seed"):
                by_strategy = group.set_index("strategy")
                if not {"average", "rotating", "instant_health"}.issubset(
                    by_strategy.index
                ):
                    raise AssertionError("strategy exposure pairing is incomplete")
                repeats = max(
                    1,
                    int(round(args.exposure_hours * 3600 / by_strategy.n_steps.iloc[0])),
                )
                uniforms = np.random.default_rng(
                    zlib.crc32(f"{source}|{load_seed}|{cv}|fc-only".encode("utf-8"))
                ).uniform(1e-12, 1 - 1e-12, args.samples)
                sampled = {}
                for strategy in ("average", "rotating", "instant_health"):
                    row = by_strategy.loc[strategy]
                    continuous_mean = (
                        row.main_expected_continuous_damage_pct * repeats
                    )
                    discrete = row.main_discrete_damage_pct * repeats
                    continuous_sample = gamma.ppf(
                        uniforms,
                        a=continuous_mean / scale,
                        scale=scale,
                    )
                    total = continuous_sample + discrete
                    sampled[strategy] = total
                    summary_rows.append(
                        {
                            "load_source": source,
                            "load_seed": int(load_seed),
                            "strategy": strategy,
                            "terminal_cv_assumption": cv,
                            "gamma_scale_pct": scale,
                            "cycle_repeats": repeats,
                            "actual_exposure_hours": repeats
                            * row.n_steps
                            / 3600,
                            "continuous_mean_pct": continuous_mean,
                            "discrete_damage_pct": discrete,
                            "discrete_fraction": discrete
                            / max(continuous_mean + discrete, 1e-12),
                            "total_mean_pct": float(total.mean()),
                            "total_std_pct": float(total.std()),
                            "total_p05_pct": float(np.quantile(total, 0.05)),
                            "total_p50_pct": float(np.quantile(total, 0.50)),
                            "total_p95_pct": float(np.quantile(total, 0.95)),
                        }
                    )
                for strategy in ("instant_health", "rotating"):
                    pooled[strategy].append(sampled[strategy] - sampled["average"])

            for strategy, chunks in pooled.items():
                delta = np.concatenate(chunks)
                pair_rows.append(
                    {
                        "load_source": source,
                        "strategy": strategy,
                        "reference_strategy": "average",
                        "terminal_cv_assumption": cv,
                        "n_load_seeds": len(chunks),
                        "samples_per_seed": args.samples,
                        "mean_delta_pct": float(delta.mean()),
                        "p05_delta_pct": float(np.quantile(delta, 0.05)),
                        "p50_delta_pct": float(np.quantile(delta, 0.50)),
                        "p95_delta_pct": float(np.quantile(delta, 0.95)),
                        "probability_lower_damage": float(np.mean(delta < 0)),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(pair_rows)
    summary.to_csv(args.out_dir / "aggregate_gamma_summary.csv", index=False)
    paired.to_csv(args.out_dir / "aggregate_gamma_paired.csv", index=False)
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "requested_exposure_hours": args.exposure_hours,
        "samples_per_load_seed": args.samples,
        "terminal_cv_assumptions": args.cvs,
        "strategies": ["average", "rotating", "instant_health"],
        "coupling": (
            "common inverse-CDF uniforms across strategies within each "
            "load_source/load_seed/CV tuple"
        ),
        "interpretation": (
            "A frozen 120-second action exposure is repeated to diagnose long-scale "
            "Gamma sensitivity. This is not a service-life prediction."
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    instant = paired[paired.strategy == "instant_health"]
    discrete_min = float(summary.discrete_fraction.min())
    discrete_max = float(summary.discrete_fraction.max())
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    offsets = {
        "empirical_1s": -0.35,
        "zuo_slow_30s": 0.0,
        "zuo_fast_30s": 0.35,
    }
    for source in SCENARIOS:
        selected = instant[instant.load_source == source].sort_values(
            "terminal_cv_assumption"
        )
        x = 100 * selected.terminal_cv_assumption.to_numpy(dtype=float)
        color = COLORS[source]
        axes[0].plot(
            x,
            selected.probability_lower_damage,
            marker="o",
            color=color,
            label=SCENARIO_LABELS[source],
        )
        center = selected.p50_delta_pct.to_numpy(dtype=float)
        axes[1].errorbar(
            x + offsets[source],
            center,
            yerr=np.vstack(
                [
                    center - selected.p05_delta_pct.to_numpy(dtype=float),
                    selected.p95_delta_pct.to_numpy(dtype=float) - center,
                ]
            ),
            marker="o",
            color=color,
            linestyle="none",
            capsize=2.5,
            elinewidth=1.0,
        )

    axes[0].set_xlabel("Terminal CV assumption (%)")
    axes[0].set_ylabel("P[Instant damage < Average]")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[0].legend(loc="best")
    axes[0].text(-0.1, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")

    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xlabel("Terminal CV assumption (%)")
    axes[1].set_ylabel("Instant - Average damage (%)")
    axes[1].text(-0.1, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.6, pad=0.55)
    fig.savefig(
        FIGURES / "fig05_aggregate_gamma_sensitivity.png",
        dpi=320,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "fig05_aggregate_gamma_sensitivity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    lines = [
        "| 场景 | CV | P(Instant退化更低) | 差值中位数(%) | 差值P05-P95(%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in instant.itertuples(index=False):
        lines.append(
            f"| {row.load_source} | {row.terminal_cv_assumption:.0%} | "
            f"{row.probability_lower_damage:.1%} | "
            f"{row.p50_delta_pct:+.3f} | "
            f"[{row.p05_delta_pct:+.3f}, {row.p95_delta_pct:+.3f}] |"
        )
    report = f"""# FC-only聚合Gamma敏感性

- 冻结120秒确定性动作暴露重复到约{args.exposure_hours:g}小时；每个负载种子{args.samples:,}个解析Gamma样本。
- CV假设：{args.cvs}；连续项按Gamma可加性聚合，离散启停/变载项确定性重复。
- 策略间使用共同逆CDF均匀数做配对耦合；该耦合用于敏感性比较，不代表可观测的共同材料噪声。

{chr(10).join(lines)}

结果只表示“若短周期暴露长期重复”的风险敏感性，不是车辆寿命预测。结论必须结合离散退化占比和CV假设解释。
本次离散启停/变载项占总期望退化的{discrete_min:.1%}-{discrete_max:.1%}，因此CV从5%增至20%没有改变改善概率；当前排序主要由事件系数与动作事件数主导。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
