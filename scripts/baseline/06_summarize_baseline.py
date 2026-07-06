"""汇总最小基线的预测、功率分配和裁剪结果，并生成两张说明图。

中文名：06_汇总基线结果。该脚本不训练、不选参，只读取已经生成的结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save(fig, path: Path) -> None:
    """同时保存便于预览的PNG和论文可用的矢量PDF。"""

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-metrics", type=Path, required=True)
    parser.add_argument("--allocation-metrics", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--clipping", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    prediction = pd.read_csv(args.prediction_metrics)
    allocation = pd.read_csv(args.allocation_metrics)
    trajectory = pd.read_csv(args.trajectory)
    clipping = json.loads(args.clipping.read_text(encoding="utf-8"))
    test_prediction = prediction[
        prediction["split"].eq("test") & prediction["selected"]
    ].copy()
    summary = {
        "default_prediction_horizon_s": 5,
        "prediction_test": test_prediction.to_dict(orient="records"),
        "allocation": allocation.to_dict(orient="records"),
        "clipping": clipping,
        "warning": "Raw H2/proxy are not fully comparable when terminal SOC differs.",
    }
    (args.out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].plot(
        test_prediction.horizon_s,
        test_prediction.point_rmse_kw,
        marker="o",
        color="#4C78A8",
    )
    axes[0].set(xlabel="Horizon (s)", ylabel="Point RMSE (kW)")
    axes[1].plot(
        test_prediction.horizon_s,
        test_prediction.window_energy_mae_kwh,
        marker="o",
        color="#F58518",
    )
    axes[1].set(xlabel="Horizon (s)", ylabel="Energy MAE (kWh)")
    axes[2].plot(
        test_prediction.horizon_s,
        test_prediction.high_power_f1,
        marker="o",
        label="High power",
    )
    axes[2].plot(
        test_prediction.horizon_s,
        test_prediction.braking_f1,
        marker="s",
        label="Braking",
    )
    axes[2].set(xlabel="Horizon (s)", ylabel="Event F1", ylim=(0, 1))
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks([1, 3, 5, 10])
    save(fig, args.figure_dir / "baseline_prediction_metrics")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    colors = {
        "instant": "#777777",
        "constant": "#F58518",
        "perfect": "#54A24B",
        "predicted": "#4C78A8",
    }
    for strategy, group in trajectory.groupby("strategy"):
        axes[0].step(
            group.step / 60,
            group.p_fc_kw,
            where="post",
            lw=0.8,
            color=colors[strategy],
            label=strategy,
        )
        axes[1].plot(
            group.step / 60,
            group.soc,
            lw=0.9,
            color=colors[strategy],
            label=strategy,
        )
    axes[0].set_ylabel("FC power (kW)")
    axes[0].legend(ncol=4, frameon=False)
    axes[1].set(xlabel="Time (min)", ylabel="SOC")
    for axis in axes:
        axis.grid(alpha=0.2)
    save(fig, args.figure_dir / "baseline_allocation_comparison")
    print(allocation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
