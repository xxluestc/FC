"""Run and plot one-factor robustness for the expanded real service model."""

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
TEMPLATE_METADATA = ROOT / "data/results/fc_only_service_templates/metadata.json"
OUTPUT = ROOT / "data/results/fc_only_service_robustness"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"


def run_case(task):
    case, setting, value, extra, out_dir, seeds, max_hours = task
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
        "--deterministic-health",
        "--skip-plot",
        "--out-dir",
        str(out_dir / case),
        *extra,
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
    paired = pd.read_csv(out_dir / case / "paired_vs_fixed.csv")
    result = paired[paired.policy == "health_greedy"].iloc[0].to_dict()
    result.update({"case": case, "setting": setting, "multiplier": value})
    return result


def case_matrix(metadata):
    rate = float(metadata["calibration_start_rate_per_observed_hour"])
    low, high = metadata["calibration_start_rate_segment_bootstrap_ci95"]
    definitions = [("base", "base", 1.0, [])]
    settings = {
        "health_limit": ("--health-limit-multiplier", (0.8, 1.2)),
        "continuous": ("--continuous-multiplier", (0.5, 2.0)),
        "load_shift": ("--load-shift-multiplier", (0.5, 2.0)),
        "operational_start": (
            "--operational-start-multiplier",
            (float(low) / rate, float(high) / rate),
        ),
        "assignment_start": ("--assignment-start-multiplier", (0.5, 2.0)),
    }
    for setting, (argument, values) in settings.items():
        for value in values:
            label = f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            definitions.append(
                (f"{setting}_{label}", setting, float(value), [argument, str(value)])
            )
    return definitions


def plot_robustness(table):
    labels = {
        "health_limit": "Health boundary",
        "continuous": "Continuous damage",
        "load_shift": "Load-shift damage",
        "operational_start": "Operational start rate",
        "assignment_start": "Scheduling-start damage",
    }
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
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    baseline = table[table.setting == "base"].iloc[0]
    settings = list(labels)
    for index, setting in enumerate(settings):
        ax = axes.flat[index]
        selected = pd.concat(
            [
                table[table.setting == setting],
                pd.DataFrame(
                    [{**baseline.to_dict(), "setting": setting, "multiplier": 1.0}]
                ),
            ],
            ignore_index=True,
        ).sort_values("multiplier")
        x = selected.multiplier.to_numpy(dtype=float)
        mean = selected.mean_gain_h.to_numpy(dtype=float)
        lower = selected.minimum_gain_h.to_numpy(dtype=float)
        upper = selected.maximum_gain_h.to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color="#A8DADC", alpha=0.55, linewidth=0)
        ax.plot(x, mean, "o-", color="#1D3557", linewidth=1.4, markersize=3.8)
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.axvline(1, color="#999999", linewidth=0.7, linestyle=":")
        ax.set_title(labels[setting])
        ax.set_xlabel("Multiplier")
        ax.set_ylabel("Health-greedy gain vs fixed (h)")
        ax.text(-0.12, 1.03, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    axes.flat[-1].axis("off")
    fig.tight_layout(pad=1.0, w_pad=1.2, h_pad=1.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig10_service_robustness.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    parser.add_argument("--max-hours", type=float, default=4000.0)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.seeds or args.max_hours <= 0 or args.jobs <= 0:
        raise ValueError("seeds, max-hours and jobs must be positive")
    metadata = json.loads(TEMPLATE_METADATA.read_text(encoding="utf-8"))
    definitions = case_matrix(metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (case, setting, value, extra, args.out_dir, args.seeds, args.max_hours)
        for case, setting, value, extra in definitions
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(run_case, tasks, chunksize=1))
    table = pd.DataFrame(rows).sort_values(["setting", "multiplier"])
    table.to_csv(args.out_dir / "robustness_summary.csv", index=False)
    plot_robustness(table)
    metadata_out = {
        "scope": "one-factor-at-a-time deterministic conditional-mean robustness",
        "template_source": "real_calibration_window",
        "seeds": args.seeds,
        "max_hours": args.max_hours,
        "cases": [item[0] for item in definitions],
        "operational_start_bounds_source": "complete calibration segment bootstrap 95% interval",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    minimum = table.minimum_gain_h.min()
    wins = table.win_share.min()
    report = "# 扩展实车服务模型稳健性\n\n"
    report += (
        "Health-greedy与固定双堆在相同负载种子下配对；使用确定性条件均值执行，"
        "每次只改变一个物理口径。阴影范围为10个负载种子的最小至最大配对增益。\n\n"
    )
    report += table.to_markdown(index=False) + "\n\n"
    report += f"全扫描最小配对增益为{minimum:.2f} h，最低获胜率为{wins:.1%}。\n"
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
