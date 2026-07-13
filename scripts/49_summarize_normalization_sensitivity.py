"""Summarize post-hoc 30/35/40 kW normalization sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "fc_only_normalization_sensitivity"
FIGURE = RESULTS / "figures/fc_only_foundation/fig16_normalization_sensitivity.png"
RUNS = {
    30.0: RESULTS / "fc_only_full_holdout_replay",
    35.0: RESULTS / "fc_only_full_holdout_norm35",
    40.0: RESULTS / "fc_only_full_holdout_norm40",
}
HEALTH_CASES = ("oldest_stack_2", "oldest_stack_0", "oldest_stack_1")
PRIMARY_CASES = ("oldest_stack_0", "oldest_stack_1")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(reference_kw, directory):
    per_run = pd.read_csv(directory / "per_run_metrics.csv")
    paired = pd.read_csv(directory / "paired_policy_deltas.csv")
    aggregate = pd.read_csv(directory / "aggregate_metrics.csv")
    summary = pd.read_csv(directory / "summary.csv")
    manifest = pd.read_csv(directory / "segment_manifest.csv")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if float(metadata["normalization_power_kw"]) != reference_kw:
        raise AssertionError(f"normalization metadata mismatch for {directory}")
    if len(per_run) != 144 or int(per_run.n_steps.sum()) != 518_490:
        raise AssertionError(f"incomplete replay at {reference_kw:g} kW")
    return per_run, paired, aggregate, summary, manifest, metadata


def build_tables():
    reference_rows = []
    effect_rows = []
    statistic_rows = []
    source_paths = []
    for reference_kw, directory in RUNS.items():
        # Reuse identical segment resamples across references and reproduce
        # the frozen 30 kW audit exactly.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        per_run, paired, aggregate, summary, manifest, metadata = load_run(
            reference_kw, directory
        )
        source_paths.extend(
            directory / name
            for name in (
                "per_run_metrics.csv",
                "paired_policy_deltas.csv",
                "aggregate_metrics.csv",
                "summary.csv",
                "segment_manifest.csv",
                "metadata.json",
            )
        )
        positive_steps = int(manifest.positive_steps.sum())
        clipped_steps = int(manifest.clipped_high_steps.sum())
        reference_rows.append(
            {
                "normalization_power_kw": reference_kw,
                "posthoc_diagnostic": reference_kw != 30.0,
                "cases": len(per_run),
                "evaluated_steps": int(per_run.n_steps.sum()),
                "clipped_high_steps": clipped_steps,
                "clipped_share_positive": clipped_steps / positive_steps,
                "constraint_violation_steps": int(
                    per_run.constraint_violation_steps.sum()
                ),
                "safety_override_steps": int(per_run.safety_override_steps.sum()),
                "tracking_max_abs_kw": float(per_run.tracking_max_abs_kw.max()),
            }
        )
        indexed = aggregate.set_index(["health_case", "policy"])
        summary_indexed = summary.set_index("health_case")
        operating = paired[paired.positive_steps.gt(0)]
        test_rows = []
        for health_case in HEALTH_CASES:
            fixed = indexed.loc[(health_case, "fixed_pair")]
            greedy = indexed.loc[(health_case, "health_greedy")]
            effect_rows.append(
                {
                    "normalization_power_kw": reference_kw,
                    "health_case": health_case,
                    "terminal_max_delta_mean_pct": float(
                        summary_indexed.loc[
                            health_case, "terminal_max_delta_mean_pct"
                        ]
                    ),
                    "better_share": float(
                        summary_indexed.loc[
                            health_case, "health_greedy_better_share"
                        ]
                    ),
                    "expected_damage_delta_sum_pct": float(
                        greedy.expected_damage_increment_sum_pct
                        - fixed.expected_damage_increment_sum_pct
                    ),
                    "tracking_mae_delta_kw": float(
                        greedy.tracking_mae_kw - fixed.tracking_mae_kw
                    ),
                    "hydrogen_intensity_delta_g_per_kwh": float(
                        greedy.hydrogen_g_per_fc_kwh
                        - fixed.hydrogen_g_per_fc_kwh
                    ),
                }
            )
            values = operating.loc[
                operating.health_case.eq(health_case),
                "delta_terminal_max_damage_pct",
            ].to_numpy(dtype=float)
            if len(values) != 8:
                raise AssertionError("normalization sensitivity segment set changed")
            indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
            bootstrap_means = values[indices].mean(axis=1)
            if health_case not in PRIMARY_CASES:
                continue
            test = wilcoxon(
                values,
                zero_method="wilcox",
                alternative="less",
                method="exact",
            )
            test_rows.append(
                {
                    "normalization_power_kw": reference_kw,
                    "health_case": health_case,
                    "n_segments": len(values),
                    "estimate_pct": float(values.mean()),
                    "ci95_lower_pct": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_upper_pct": float(np.quantile(bootstrap_means, 0.975)),
                    "better_segments": int((values < 0).sum()),
                    "p_value_one_sided": float(test.pvalue),
                }
            )
        reject, adjusted, _, _ = multipletests(
            [row["p_value_one_sided"] for row in test_rows],
            alpha=0.05,
            method="holm",
        )
        for row, p_holm, rejected in zip(test_rows, adjusted, reject):
            row["p_value_holm"] = float(p_holm)
            row["reject_holm_0p05"] = bool(rejected)
            statistic_rows.append(row)
    return (
        pd.DataFrame(reference_rows),
        pd.DataFrame(effect_rows),
        pd.DataFrame(statistic_rows),
        source_paths,
    )


def plot_results(reference, effects, statistics):
    refs = reference.normalization_power_kw.to_numpy(dtype=float)
    colors = {"oldest_stack_0": "#2A9D8F", "oldest_stack_1": "#E76F51"}
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
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.7))
    axes[0, 0].plot(
        refs,
        100 * reference.clipped_share_positive,
        marker="o",
        color="#1D3557",
        linewidth=1.2,
    )
    axes[0, 0].set_ylabel("Clipped positive samples (%)")

    axes[0, 1].plot(
        refs,
        reference.tracking_max_abs_kw,
        marker="s",
        color="#457B9D",
        linewidth=1.2,
    )
    axes[0, 1].axhline(5.5, color="#555555", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("Max tracking error (kW)")

    for health_case, label in (
        ("oldest_stack_0", "Oldest: 0"),
        ("oldest_stack_1", "Oldest: 1"),
    ):
        group = statistics[statistics.health_case.eq(health_case)].sort_values(
            "normalization_power_kw"
        )
        estimate = 1e3 * group.estimate_pct.to_numpy(dtype=float)
        lower = 1e3 * group.ci95_lower_pct.to_numpy(dtype=float)
        upper = 1e3 * group.ci95_upper_pct.to_numpy(dtype=float)
        axes[1, 0].errorbar(
            group.normalization_power_kw,
            estimate,
            yerr=np.vstack((estimate - lower, upper - estimate)),
            marker="o",
            capsize=2.5,
            linewidth=1.1,
            color=colors[health_case],
            label=label,
        )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.7)
    axes[1, 0].set_ylabel("Greedy - fixed max damage\n($10^{-3}$ %-point)")
    axes[1, 0].legend(frameon=False, fontsize=7)

    width = 1.5
    for offset, (health_case, label) in zip(
        (-width / 2, width / 2),
        (("oldest_stack_0", "Oldest: 0"), ("oldest_stack_1", "Oldest: 1")),
    ):
        group = effects[effects.health_case.eq(health_case)].sort_values(
            "normalization_power_kw"
        )
        axes[1, 1].bar(
            group.normalization_power_kw + offset,
            group.expected_damage_delta_sum_pct,
            width=width,
            color=colors[health_case],
            label=label,
        )
    axes[1, 1].axhline(0, color="#555555", linewidth=0.7)
    axes[1, 1].set_ylabel("Total expected damage delta (%-point)")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for index, ax in enumerate(axes.flat):
        ax.set_xticks(refs)
        ax.set_xlabel("Normalization reference (kW)")
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.8, w_pad=1.4, h_pad=1.3)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    reference, effects, statistics, sources = build_tables()
    if int(reference.constraint_violation_steps.sum()) != 0:
        raise AssertionError("normalization sensitivity includes hard violations")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference.round(12).to_csv(OUTPUT / "reference_summary.csv", index=False)
    effects.round(12).to_csv(OUTPUT / "health_effects.csv", index=False)
    statistics.round(12).to_csv(OUTPUT / "segment_statistics.csv", index=False)
    metadata = {
        "status": "post-hoc diagnostic; frozen 30 kW result remains primary",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "physical_rating_confirmed": False,
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in sources
        ],
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(reference, effects, statistics)
    report = "# 30/35/40 kW归一化参考事后敏感性\n\n"
    report += (
        "30 kW是冻结主分析；35和40 kW仅用于诊断容量口径对结论的影响，不能替代车辆额定资料。"
        "所有参考使用相同留出segment、健康身份、控制器权重和5.5 kW容差。\n\n"
    )
    report += reference.to_markdown(index=False) + "\n\n"
    report += statistics.to_markdown(index=False) + "\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(reference.to_string(index=False))
    print(effects.to_string(index=False))
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
