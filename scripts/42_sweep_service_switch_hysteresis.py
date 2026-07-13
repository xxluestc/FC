"""Tune a development-only switching hysteresis against health-greedy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/36_run_service_horizon_scheduler.py"
TEMPLATES = ROOT / "data/results/fc_only_service_templates/service_exposure_templates.csv"
OUTPUT = ROOT / "data/results/fc_only_service_hysteresis_sweep"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
DEFAULT_MARGINS = (0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005)


def run_margin(task):
    margin, out_dir, seeds, max_hours = task
    label = f"margin_{margin:.5f}".rstrip("0").rstrip(".").replace(".", "p")
    case_dir = out_dir / label
    command = [
        sys.executable,
        str(RUNNER),
        "--template-input",
        str(TEMPLATES),
        "--template-source",
        "real_calibration_window",
        "--max-hours",
        str(max_hours),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--policies",
        "fixed_pair",
        "health_greedy",
        "expected_hysteresis",
        "--switch-margin-fraction",
        str(margin),
        "--deterministic-health",
        "--skip-plot",
        "--out-dir",
        str(case_dir),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = pd.read_csv(case_dir / "summary.csv").set_index("policy")
    paired = pd.read_csv(case_dir / "paired_vs_health_greedy.csv")
    comparison = paired[paired.policy == "expected_hysteresis"].iloc[0]
    return {
        "margin_fraction": margin,
        "fixed_time_mean_h": summary.loc["fixed_pair", "time_to_limit_mean_h"],
        "fixed_start_mean": summary.loc["fixed_pair", "start_count_mean"],
        "health_greedy_time_mean_h": summary.loc[
            "health_greedy", "time_to_limit_mean_h"
        ],
        "health_greedy_start_mean": summary.loc[
            "health_greedy", "start_count_mean"
        ],
        "hysteresis_time_mean_h": summary.loc[
            "expected_hysteresis", "time_to_limit_mean_h"
        ],
        "hysteresis_start_mean": summary.loc[
            "expected_hysteresis", "start_count_mean"
        ],
        "time_gain_vs_health_mean_h": comparison.mean_gain_h,
        "time_gain_vs_health_min_h": comparison.minimum_gain_h,
        "time_gain_vs_health_max_h": comparison.maximum_gain_h,
        "win_share_vs_health": comparison.win_share,
        "nonworse_share_vs_health": comparison.nonworse_share,
    }


def plot(table):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    colors = plt.cm.viridis(
        np.linspace(0.12, 0.88, len(table))
    )
    health_time = float(table.health_greedy_time_mean_h.iloc[0])
    health_starts = float(table.health_greedy_start_mean.iloc[0])
    fixed_time = float(table.fixed_time_mean_h.iloc[0])
    fixed_starts = float(table.fixed_start_mean.iloc[0])
    axes[0].scatter(
        table.hysteresis_start_mean,
        table.hysteresis_time_mean_h,
        c=colors,
        s=28,
        zorder=3,
    )
    axes[0].plot(
        table.hysteresis_start_mean,
        table.hysteresis_time_mean_h,
        color="#777777",
        linewidth=0.8,
        zorder=1,
    )
    axes[0].scatter(health_starts, health_time, marker="s", color="#7A5195", s=30, label="Health-greedy")
    axes[0].scatter(fixed_starts, fixed_time, marker="^", color="#6C757D", s=34, label="Fixed pair")
    axes[0].set_xlabel("Mean stack starts")
    axes[0].set_ylabel("Mean time to health boundary (h)")
    axes[0].set_title("Development Pareto screen")
    axes[0].legend(frameon=False, fontsize=7)

    starts_avoided = health_starts - table.hysteresis_start_mean
    axes[1].scatter(
        starts_avoided,
        table.time_gain_vs_health_mean_h,
        c=colors,
        s=28,
        zorder=3,
    )
    axes[1].plot(
        starts_avoided,
        table.time_gain_vs_health_mean_h,
        color="#777777",
        linewidth=0.8,
        zorder=1,
    )
    axes[1].axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].axvline(0, color="#999999", linewidth=0.7, linestyle=":")
    offsets = ((3, 3), (3, 3), (3, -10), (3, 8), (3, -10), (3, 3), (3, 3))
    for offset, row, x, y in zip(
        offsets, table.itertuples(), starts_avoided, table.time_gain_vs_health_mean_h
    ):
        axes[1].annotate(
            f"{row.margin_fraction:g}",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.3,
        )
    axes[1].set_xlabel("Starts avoided vs health-greedy")
    axes[1].set_ylabel("Time gain vs health-greedy (h)")
    axes[1].set_title("Hysteresis margin (labels)")
    for index, ax in enumerate(axes):
        ax.text(-0.12, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.7, w_pad=1.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig12_service_hysteresis_pareto.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", nargs="+", type=float, default=list(DEFAULT_MARGINS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    parser.add_argument("--max-hours", type=float, default=4000.0)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.margins or min(args.margins) < 0 or not args.seeds or args.jobs <= 0:
        raise ValueError("margins, seeds and jobs are invalid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "hysteresis_sweep.csv"
    if args.summarize_only:
        if not summary_path.exists():
            raise FileNotFoundError("--summarize-only requires hysteresis_sweep.csv")
        table = pd.read_csv(summary_path)
    else:
        tasks = [
            (margin, args.out_dir, args.seeds, args.max_hours)
            for margin in args.margins
        ]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            rows = list(executor.map(run_margin, tasks, chunksize=1))
        table = pd.DataFrame(rows).sort_values("margin_fraction")
        table.to_csv(summary_path, index=False)
    plot(table)
    health_time = float(table.health_greedy_time_mean_h.iloc[0])
    eligible = table[
        (table.hysteresis_time_mean_h >= 0.99 * health_time)
        & (table.time_gain_vs_health_min_h >= 0)
    ]
    selected_margin = (
        float(eligible.sort_values("hysteresis_start_mean").iloc[0].margin_fraction)
        if len(eligible)
        else None
    )
    metadata = {
        "scope": "development-only deterministic hysteresis Pareto screen",
        "template_source": "real_calibration_window",
        "margins": args.margins,
        "seeds": args.seeds,
        "selection_rule": "smallest starts subject to <=1% mean time loss and nonnegative minimum paired time change",
        "selected_margin": selected_margin,
        "selection_outcome": "no margin passed the frozen rule" if selected_margin is None else "margin selected",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 服务调度切换滞回开发筛查\n\n"
    report += (
        "本扫描仅用于探索寿命-主动启停Pareto。冻结选点规则要求平均到边界时间损失"
        "不超过1%，且所有配对种子的时间变化非负。"
    )
    if selected_margin is None:
        report += "没有阈值通过，因此主方法不采用滞回增强。\n\n"
    else:
        report += f"选中阈值为{selected_margin:g}。\n\n"
    report += table.to_markdown(index=False) + "\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
