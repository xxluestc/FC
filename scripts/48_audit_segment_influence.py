"""Audit leave-one-segment-out and single-sign-reversal sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/results/fc_only_full_holdout_replay/paired_policy_deltas.csv"
OUTPUT = ROOT / "data/results/fc_only_full_holdout_statistics"
PRIMARY_CASES = ("oldest_stack_0", "oldest_stack_1")
METRIC = "delta_terminal_max_damage_pct"


def one_sided_p(values):
    result = wilcoxon(
        np.asarray(values, dtype=float),
        zero_method="wilcox",
        alternative="less",
        method="exact",
    )
    return float(result.statistic), float(result.pvalue)


def primary_frame():
    paired = pd.read_csv(INPUT)
    frame = paired[
        paired.positive_steps.gt(0) & paired.health_case.isin(PRIMARY_CASES)
    ][["segment_id", "health_case", METRIC]].copy()
    counts = frame.groupby("health_case").segment_id.nunique()
    if not counts.eq(8).all() or frame[METRIC].ge(0).any():
        raise AssertionError("primary eight-segment all-improved evidence changed")
    return frame


def leave_one_out(frame):
    rows = []
    for omitted_segment in sorted(frame.segment_id.unique()):
        scenario_rows = []
        for health_case in PRIMARY_CASES:
            selected = frame[
                frame.health_case.eq(health_case)
                & frame.segment_id.ne(omitted_segment)
            ]
            statistic, p_value = one_sided_p(selected[METRIC])
            scenario_rows.append(
                {
                    "health_case": health_case,
                    "omitted_segment_id": int(omitted_segment),
                    "n_segments": len(selected),
                    "mean_delta_pct": float(selected[METRIC].mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value_one_sided": p_value,
                }
            )
        reject, adjusted, _, _ = multipletests(
            [row["p_value_one_sided"] for row in scenario_rows],
            alpha=0.05,
            method="holm",
        )
        for row, p_holm, rejected in zip(scenario_rows, adjusted, reject):
            row["p_value_holm"] = float(p_holm)
            row["reject_holm_0p05"] = bool(rejected)
            rows.append(row)
    return pd.DataFrame(rows)


def single_sign_reversal(frame):
    original_p = {
        health_case: one_sided_p(
            frame.loc[frame.health_case.eq(health_case), METRIC]
        )[1]
        for health_case in PRIMARY_CASES
    }
    rows = []
    for health_case in PRIMARY_CASES:
        other_case = next(case for case in PRIMARY_CASES if case != health_case)
        selected = frame[frame.health_case.eq(health_case)].sort_values("segment_id")
        absolute_rank = selected[METRIC].abs().rank(method="average").astype(int)
        for row_index, row in selected.iterrows():
            perturbed = selected[METRIC].copy()
            perturbed.loc[row_index] *= -1.0
            statistic, p_value = one_sided_p(perturbed)
            p_values = [p_value, original_p[other_case]]
            reject, adjusted, _, _ = multipletests(
                p_values, alpha=0.05, method="holm"
            )
            rows.append(
                {
                    "health_case": health_case,
                    "reversed_segment_id": int(row.segment_id),
                    "reversed_abs_rank": int(absolute_rank.loc[row_index]),
                    "original_delta_pct": float(row[METRIC]),
                    "perturbed_mean_delta_pct": float(perturbed.mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value_one_sided": p_value,
                    "p_value_holm_with_other_original": float(adjusted[0]),
                    "reject_holm_0p05": bool(reject[0]),
                }
            )
    return pd.DataFrame(rows)


def main():
    frame = primary_frame()
    leave_out = leave_one_out(frame)
    reversals = single_sign_reversal(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    leave_out.round(12).to_csv(OUTPUT / "leave_one_segment_out.csv", index=False)
    reversals.round(12).to_csv(OUTPUT / "single_sign_reversal.csv", index=False)

    summary = {
        "leave_one_out_all_holm_significant": bool(
            leave_out.reject_holm_0p05.all()
        ),
        "leave_one_out_max_holm_p": float(leave_out.p_value_holm.max()),
        "single_reversal_cases": int(len(reversals)),
        "single_reversal_holm_failures": int(
            (~reversals.reject_holm_0p05).sum()
        ),
        "single_reversal_failure_share": float(
            (~reversals.reject_holm_0p05).mean()
        ),
        "interpretation": (
            "Leave-one-out uses observed data; sign reversal is a hypothetical "
            "fragility audit and is not additional experimental evidence."
        ),
    }
    (OUTPUT / "influence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 完整segment影响力与脆弱性审计\n\n"
    report += (
        f"逐段删一后两个主身份的Holm校正检验是否全部保留："
        f"`{summary['leave_one_out_all_holm_significant']}`；最大校正$p$值为"
        f"{summary['leave_one_out_max_holm_p']:.8f}。\n\n"
    )
    report += (
        f"共构造{summary['single_reversal_cases']}个单段符号反转假设，其中"
        f"{summary['single_reversal_holm_failures']}个会使相应主检验在0.05水平不再显著。"
        "符号反转是脆弱性压力测试，不是观测证据。\n"
    )
    (OUTPUT / "influence_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(reversals.groupby("health_case").reject_holm_0p05.value_counts())


if __name__ == "__main__":
    main()
