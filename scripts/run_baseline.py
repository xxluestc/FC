"""最小可运行基线的统一入口。

中文名：运行基线主流程。

模式：
- check：只检查已提交的小型结果和字段，不需要原始大数据；
- run：从现有canonical重新生成动力学、固定XGBoost预测和四策略分配；
- preprocess-run：先从本地原始CSV生成canonical，再执行run。

该入口不会调用07--12优化实验，也不会进行模型族、权重或控制结构搜索。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    """显示并执行子步骤；任一步失败立即停止主流程。"""

    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_columns(path: Path, columns: set[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少基线文件：{path}")
    actual = set(pd.read_csv(path, nrows=5).columns)
    missing = columns - actual
    if missing:
        raise RuntimeError(f"{path} 缺少字段：{sorted(missing)}")


def lightweight_check(config: dict) -> None:
    """无原始数据时验证已提交基线结果的schema和四策略完整性。"""

    result_dir = ROOT / config["outputs"]["result_dir"]
    require_columns(
        result_dir / "baseline_prediction_metrics.csv",
        {"horizon_s", "point_rmse_kw", "window_energy_mae_kwh", "selected"},
    )
    require_columns(
        result_dir / "baseline_allocation_metrics.csv",
        {"strategy", "h2_kg", "degradation_proxy_sum", "soc_final"},
    )
    metrics = pd.read_csv(result_dir / "baseline_allocation_metrics.csv")
    expected = {"instant", "constant", "perfect", "predicted"}
    if set(metrics.strategy) != expected:
        raise RuntimeError("基线四策略不完整")
    print("Baseline lightweight schema check passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("check", "run", "preprocess-run"), default="check"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="仅preprocess-run需要：本地21UBE0022原始日CSV目录，不写入配置或Git。",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.mode == "check":
        lightweight_check(config)
        return

    canonical = ROOT / config["data"]["canonical"]
    demand = ROOT / config["outputs"]["demand_table"]
    predictions = ROOT / config["outputs"]["prediction_results"]
    result_dir = ROOT / config["outputs"]["result_dir"]
    figure_dir = ROOT / config["outputs"]["figure_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "preprocess-run":
        if args.raw_dir is None:
            raise SystemExit("preprocess-run必须提供--raw-dir")
        run(
            [
                sys.executable,
                "scripts/01_preprocess_data.py",
                "--input-dir",
                str(args.raw_dir),
                "--output",
                str(canonical),
                "--summary",
                str(result_dir / "baseline_preprocess_summary.json"),
            ]
        )
    if not canonical.exists():
        raise FileNotFoundError("缺少canonical；请提供本地原始数据运行preprocess-run")

    # 03：由canonical构建实测需求功率和独立车辆动力学基线。
    run(
        [
            sys.executable,
            "scripts/03_vehicle_dynamics_power.py",
            "--input",
            str(canonical),
            "--output",
            str(demand),
            "--metrics",
            str(result_dir / "baseline_dynamics_metrics.json"),
        ]
    )
    # 04：固定XGBoost，四个时域分别训练；不存在模型族搜索。
    run(
        [
            sys.executable,
            "scripts/04_train_or_run_predictors.py",
            "--input",
            str(demand),
            "--dynamics-metrics",
            str(result_dir / "baseline_dynamics_metrics.json"),
            "--output",
            str(predictions),
            "--metrics",
            str(result_dir / "baseline_prediction_metrics.csv"),
            "--comparison",
            str(result_dir / "baseline_prediction_comparison.csv"),
            "--selection",
            str(result_dir / "baseline_prediction_selection.json"),
            "--families",
            config["prediction"]["model_family"],
            "--horizons",
            *[str(value) for value in config["prediction"]["horizons_s"]],
        ]
    )
    # 05：读取冻结权重，只运行四个定义固定的策略。
    run(
        [
            sys.executable,
            "scripts/baseline/05_run_baseline_allocation.py",
            "--vehicle",
            str(demand),
            "--predictions",
            str(predictions),
            "--stack-map",
            str(ROOT / config["data"]["stack_map"]),
            "--config",
            str(config_path),
            "--out-dir",
            str(result_dir),
        ]
    )
    # 06：只汇总和画图，不改模型或参数。
    run(
        [
            sys.executable,
            "scripts/baseline/06_summarize_baseline.py",
            "--prediction-metrics",
            str(result_dir / "baseline_prediction_metrics.csv"),
            "--allocation-metrics",
            str(result_dir / "baseline_allocation_metrics.csv"),
            "--trajectory",
            str(result_dir / "baseline_trajectory.csv"),
            "--clipping",
            str(result_dir / "baseline_clipping_audit.json"),
            "--out-dir",
            str(result_dir),
            "--figure-dir",
            str(figure_dir),
        ]
    )
    lightweight_check(config)


if __name__ == "__main__":
    main()
