"""Audit time-isolated real-load statistics for Zuo-style calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation.zuo_load_calibration import (
    ZUO_FAST_TRANSITION,
    ZUO_LOAD_LEVEL_FRACTIONS,
    ZUO_SLOW_TRANSITION,
    estimate_segmented_transitions,
    split_at_largest_segment_gap,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
OUTPUT = ROOT / "data/results/load_zuo_calibration_norm40"
STRIDES_S = (1, 5, 10, 30, 60)
DEFAULT_NORMALIZATION_POWER_KW = 40.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalization-power-kw",
        type=float,
        default=DEFAULT_NORMALIZATION_POWER_KW,
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.normalization_power_kw <= 0:
        raise ValueError("normalization power must be positive")
    frame = pd.read_csv(
        SOURCE,
        usecols=["timestamp", "segment_id", "target_power_kw", "fc_input_power_kw"],
    )
    split = split_at_largest_segment_gap(frame)
    calibration = frame[frame.segment_id.isin(split.calibration_segments)].copy()
    holdout = frame[frame.segment_id.isin(split.holdout_segments)].copy()
    normalization_power_kw = float(args.normalization_power_kw)
    parsed_time = pd.to_datetime(calibration.timestamp, errors="raise")
    power = calibration.fc_input_power_kw
    missing_power_samples = int(power.isna().sum())
    negative_power_samples = int(power.lt(0).sum())
    clipped_power_samples = int(power.gt(normalization_power_kw).sum())
    cadence_violations = 0
    cadence_intervals = 0
    for _, segment in calibration.assign(_timestamp=parsed_time).groupby(
        "segment_id", sort=False
    ):
        delta_s = segment._timestamp.diff().dt.total_seconds().dropna()
        cadence_violations += int((delta_s != 1.0).sum())
        cadence_intervals += len(delta_s)

    matrix_rows = []
    state_rows = []
    fast = np.asarray(ZUO_FAST_TRANSITION)
    slow = np.asarray(ZUO_SLOW_TRANSITION)
    for stride in STRIDES_S:
        estimate = estimate_segmented_transitions(
            calibration,
            normalization_power_kw=normalization_power_kw,
            stride_s=stride,
        )
        occupancy_total = int(estimate.occupancy.sum())
        for state, (level, count) in enumerate(
            zip(ZUO_LOAD_LEVEL_FRACTIONS, estimate.occupancy)
        ):
            state_rows.append(
                {
                    "stride_s": stride,
                    "state": state,
                    "normalized_level": level,
                    "mapped_single_stack_kw": level * normalization_power_kw,
                    "occupancy_count": int(count),
                    "occupancy_fraction": (
                        float(count / occupancy_total) if occupancy_total else np.nan
                    ),
                }
            )
        for source_state in range(4):
            for target_state in range(4):
                matrix_rows.append(
                    {
                        "stride_s": stride,
                        "source_state": source_state,
                        "target_state": target_state,
                        "transition_count": int(
                            estimate.counts[source_state, target_state]
                        ),
                        "empirical_probability": estimate.probabilities[
                            source_state, target_state
                        ],
                        "ci95_lower": estimate.ci95_lower[
                            source_state, target_state
                        ],
                        "ci95_upper": estimate.ci95_upper[
                            source_state, target_state
                        ],
                        "zuo_fast_probability": fast[source_state, target_state],
                        "zuo_slow_probability": slow[source_state, target_state],
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transitions = pd.DataFrame(matrix_rows)
    states = pd.DataFrame(state_rows)
    transitions.to_csv(args.out_dir / "transition_scale_audit.csv", index=False)
    states.to_csv(args.out_dir / "state_coverage_audit.csv", index=False)

    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "power_column": "fc_input_power_kw",
        "normalization_reference_source": (
            "40 kW repeated controller level from the independent recent-archive "
            "audit; empirical operating reference, not a nameplate rating"
        ),
        "normalization_reference_kw": normalization_power_kw,
        "split_rule": "largest timestamp gap between complete segment_id groups",
        "calibration_segments": list(split.calibration_segments),
        "holdout_segments": list(split.holdout_segments),
        "calibration_rows": len(calibration),
        "holdout_rows": len(holdout),
        "gap_seconds": split.gap_seconds,
        "calibration_end": split.calibration_end,
        "holdout_start": split.holdout_start,
        "strides_s": list(STRIDES_S),
        "one_second_cadence": {
            "checked_intervals": cadence_intervals,
            "non_one_second_intervals": cadence_violations,
        },
        "power_quality": {
            "missing_samples": missing_power_samples,
            "negative_samples": negative_power_samples,
            "samples_above_normalization_reference": clipped_power_samples,
            "above_reference_handling": "clipped to normalized value 1.0",
        },
        "bootstrap": {
            "unit": "complete calibration segment",
            "samples": 1000,
            "seed": 2026,
            "interval": "percentile 95%",
            "row_condition": (
                "each cell interval uses bootstrap draws with at least one "
                "transition from that source-state row"
            ),
        },
        "holdout_usage": (
            "segment 22-45 values are not fitted; the normalization reference comes "
            "from the separate full recent-archive identity/power audit"
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    one_second = states[states.stride_s == 1]
    missing_states = one_second.loc[
        one_second.occupancy_count.eq(0), "state"
    ].astype(int).tolist()
    if missing_states:
        coverage_note = (
            f"状态{missing_states}在标定分区中没有覆盖，因此对应转移行不能称为实车辨识结果；"
            "后续必须明确使用文献先验或额外标定数据。"
        )
    else:
        least_covered = one_second.loc[one_second.occupancy_count.idxmin()]
        coverage_note = (
            "四个状态在标定分区中均有覆盖；其中覆盖最少的是状态"
            f"{int(least_covered.state)}（{least_covered.occupancy_fraction:.2%}），"
            "其转移概率不确定性必须结合segment级置信区间解释。"
        )
    report = f"""# Zuo 2024实车单堆负载标定审计

## 数据隔离

- 单堆功率：`fc_input_power_kw`；标定归一化参考：{normalization_power_kw:.3f} kW。该值来自独立全年归档中重复出现的40 kW控制档，不解释为铭牌额定功率；segment 22-45的数值不参与选择。
- 按完整`segment_id`之间最大时间间隔切分：标定段{split.calibration_segments[0]}-{split.calibration_segments[-1]}，共{len(calibration):,}行；留出段{split.holdout_segments[0]}-{split.holdout_segments[-1]}，共{len(holdout):,}行。
- 间隔为{split.gap_seconds / 86400:.2f}天；留出段不参与状态、转移矩阵或归一化参数拟合。
- 标定分区检查了{cadence_intervals:,}个段内采样间隔，非1秒间隔数为{cadence_violations}。
- 单堆功率缺失值{missing_power_samples}个、负值{negative_power_samples}个；高于{normalization_power_kw:g} kW归一化参考的样本{clipped_power_samples:,}个，量化时显式截到归一化值1.0。

## Zuo状态覆盖

采用Zuo 2024附录A的四状态相对等级`[2.9, 4.1, 5.8, 7.0] / 7.0`。1秒标定样本占比分别为：{', '.join(f'{value:.2%}' for value in one_second.occupancy_fraction.fillna(0))}。

{coverage_note}

## 时间尺度

分别审计1/5/10/30/60秒采样步长。转移仅在同一完整segment内、且相邻两点均为正功率时计数；不会跨越停机区间或segment边界。置信区间按完整segment重采样1000次，保留段内时间相关性；某来源状态在一次bootstrap中没有转移时，该次抽样不进入该行区间计算。

当前输出只做尺度与覆盖审计，不选择最终转移矩阵，也不执行显著性检验。下一步应比较各步长的驻留概率、有效转移数和控制决策周期，再决定如何将实车统计与Zuo快变/慢变矩阵组合。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
