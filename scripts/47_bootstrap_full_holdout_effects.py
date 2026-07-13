"""Quantify complete-segment uncertainty for frozen full-holdout effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/results/fc_only_full_holdout_replay/paired_policy_deltas.csv"
MANIFEST = ROOT / "data/results/fc_only_full_holdout_replay/segment_manifest.csv"
OUTPUT = ROOT / "data/results/fc_only_full_holdout_statistics"
HEALTH_CASES = ("oldest_stack_2", "oldest_stack_0", "oldest_stack_1")
PRIMARY_CASES = ("oldest_stack_0", "oldest_stack_1")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026


def statistics_for_sample(frame):
    rows = frame.segment_rows.to_numpy(dtype=float)
    fixed_energy = frame.fixed_fc_energy_kwh.to_numpy(dtype=float).sum()
    greedy_energy = frame.health_greedy_fc_energy_kwh.to_numpy(dtype=float).sum()
    fixed_h2 = frame.fixed_hydrogen_g.to_numpy(dtype=float).sum()
    greedy_h2 = frame.health_greedy_hydrogen_g.to_numpy(dtype=float).sum()
    return {
        "terminal_max_delta_mean_pct": float(
            frame.delta_terminal_max_damage_pct.mean()
        ),
        "expected_damage_delta_mean_pct": float(
            frame.delta_expected_damage_increment_pct.mean()
        ),
        "tracking_mae_delta_kw": float(
            frame.health_greedy_tracking_abs_sum_kw.sum() / rows.sum()
            - frame.fixed_tracking_abs_sum_kw.sum() / rows.sum()
        ),
        "hydrogen_intensity_delta_g_per_kwh": float(
            greedy_h2 / max(greedy_energy, 1e-12)
            - fixed_h2 / max(fixed_energy, 1e-12)
        ),
    }


def bootstrap_case(frame, rng):
    point = statistics_for_sample(frame)
    values = {metric: np.empty(BOOTSTRAP_SAMPLES) for metric in point}
    n_segments = len(frame)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, n_segments, size=n_segments)
        sampled = frame.iloc[indices]
        sample_statistics = statistics_for_sample(sampled)
        for metric, value in sample_statistics.items():
            values[metric][sample_index] = value
    rows = []
    for metric, estimate in point.items():
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "bootstrap_mean": float(values[metric].mean()),
                "ci95_lower": float(np.quantile(values[metric], 0.025)),
                "ci95_upper": float(np.quantile(values[metric], 0.975)),
            }
        )
    return rows


def main():
    paired = pd.read_csv(INPUT)
    manifest = pd.read_csv(MANIFEST)[["segment_id", "rows"]].rename(
        columns={"rows": "segment_rows"}
    )
    operating = paired[paired.positive_steps > 0].merge(
        manifest, on="segment_id", how="left", validate="many_to_one"
    )
    if operating.segment_id.nunique() != 8 or operating.segment_rows.isna().any():
        raise AssertionError("frozen operating-segment set changed")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    test_rows = []
    for health_case in HEALTH_CASES:
        frame = operating[operating.health_case == health_case].sort_values(
            "segment_id"
        )
        if len(frame) != 8:
            raise AssertionError(f"{health_case} does not contain eight operating segments")
        for row in bootstrap_case(frame, rng):
            summary_rows.append(
                {"health_case": health_case, "n_segments": len(frame), **row}
            )
        if health_case in PRIMARY_CASES:
            differences = frame.delta_terminal_max_damage_pct.to_numpy(dtype=float)
            test = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="less",
                method="exact",
            )
            test_rows.append(
                {
                    "health_case": health_case,
                    "hypothesis": (
                        "health_greedy_minus_fixed_terminal_max_damage < 0"
                    ),
                    "n_segments": len(differences),
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value_one_sided": float(test.pvalue),
                    "better_segments": int((differences < 0).sum()),
                    "nonworse_segments": int((differences <= 0).sum()),
                }
            )

    tests = pd.DataFrame(test_rows)
    reject, adjusted, _, _ = multipletests(
        tests.p_value_one_sided.to_numpy(dtype=float),
        alpha=0.05,
        method="holm",
    )
    tests["p_value_holm"] = adjusted
    tests["reject_holm_0p05"] = reject
    summary = pd.DataFrame(summary_rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.round(12).to_csv(OUTPUT / "segment_bootstrap_summary.csv", index=False)
    tests.round(12).to_csv(OUTPUT / "primary_wilcoxon_tests.csv", index=False)
    metadata = {
        "resampling_unit": "complete positive-power holdout segment",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval": "percentile 95%",
        "primary_metric": "health_greedy minus fixed terminal maximum damage",
        "primary_alternative": "less than zero",
        "multiplicity": (
            "Holm correction across two predeclared nontrivial health identities"
        ),
        "oldest_stack_2_handling": (
            "zero-effect control; excluded from significance testing"
        ),
        "independence_boundary": (
            "time steps are not treated as independent observations"
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary_summary = summary[
        summary.health_case.isin(PRIMARY_CASES)
        & summary.metric.eq("terminal_max_delta_mean_pct")
    ]
    report = "# 完整留出 segment 级不确定性\n\n"
    report += (
        "重采样单位为 8 个完整正功率 segment，不把逐秒样本视为独立观测。"
        "两个非平凡健康身份是预声明的堆身份循环，分别检验 health-greedy 相对固定双堆的"
        "终端最大退化差是否小于 0；Holm 校正控制两个主检验的多重性。\n\n"
    )
    report += primary_summary.to_markdown(index=False) + "\n\n"
    report += tests.to_markdown(index=False) + "\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(primary_summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
