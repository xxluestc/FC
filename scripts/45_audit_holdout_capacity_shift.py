"""Audit out-of-range holdout loads without changing the frozen controller."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.evaluation import split_at_largest_segment_gap
from fc_power.world_model import (
    MultiStackAction,
    WorldModelConfig,
    load_lzw_multistack_world_model,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/liu_vehicle_canonical_1s.csv"
OUTPUT = ROOT / "data/results/fc_only_holdout_capacity_audit"
FIGURES = ROOT / "data/results/figures/fc_only_foundation"
HEALTH_CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
HETEROGENEITY = (1.0, 1.05, 1.10)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}
POLICY_ASSIGNMENTS = {
    "fixed_pair": {
        health_case: (0, 1) for health_case in HEALTH_CASES
    },
    "health_greedy": {
        "oldest_stack_2": (0, 1),
        "oldest_stack_0": (1, 2),
        "oldest_stack_1": (2, 0),
    },
}
CAPACITY_RESERVE_FRACTION = 0.05
TRACKING_TOLERANCE_KW = 5.5


def maximum_pair_power(model, initial_fraction, assignment, health_limit_pct):
    state = model.initial_state(
        degradation_pct=np.asarray(initial_fraction, dtype=float) * health_limit_pct
    )
    currents = np.zeros(model.n_stacks, dtype=float)
    online = np.zeros(model.n_stacks, dtype=bool)
    for stack in assignment:
        currents[stack] = max(model.config.allowed_currents_a)
        online[stack] = True
    step = model.step(
        state,
        MultiStackAction(tuple(currents), tuple(online)),
        demand_power_kw=0.0,
    )
    return float(step.constraints.stack_power_kw)


def plot_audit(calibration, holdout, audit, strict_required_kw):
    positive_cal = np.sort(
        calibration.loc[calibration.fc_input_power_kw > 0, "fc_input_power_kw"].to_numpy()
    )[::-1]
    positive_hold = np.sort(
        holdout.loc[holdout.fc_input_power_kw > 0, "fc_input_power_kw"].to_numpy()
    )[::-1]
    labels = ("30 kW frozen", "31.34 kW cal. max", "40 kW candidate")
    colors = ("#6C757D", "#2A9D8F", "#E76F51")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45))
    axes[0].plot(
        np.arange(1, len(positive_cal) + 1) / len(positive_cal),
        positive_cal,
        color="#1D3557",
        linewidth=0.9,
        label="Calibration",
    )
    axes[0].plot(
        np.arange(1, len(positive_hold) + 1) / len(positive_hold),
        positive_hold,
        color="#E76F51",
        linewidth=0.9,
        label="Holdout",
    )
    axes[0].axhline(30.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Positive-sample exceedance share")
    axes[0].set_ylabel("Single-stack power (kW)")
    axes[0].legend(frameon=False, fontsize=6.8)

    x = np.arange(len(audit))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        audit.calibration_clip_share_positive * 100,
        width,
        color="#1D3557",
        label="Calibration",
    )
    axes[1].bar(
        x + width / 2,
        audit.holdout_clip_share_positive * 100,
        width,
        color="#E76F51",
        label="Holdout",
    )
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Samples above reference (%)")
    axes[1].legend(frameon=False, fontsize=6.8)

    axes[2].bar(
        x,
        audit.holdout_strict_capacity_exceedance_share_positive * 100,
        color=colors,
        width=0.66,
    )
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_ylabel("Unclipped holdout above\npair capacity (%)")
    capacity_values = (
        audit.holdout_strict_capacity_exceedance_share_positive * 100
    )
    axes[2].set_ylim(0, max(1.0, float(capacity_values.max()) * 1.15))
    for index, value in enumerate(
        capacity_values
    ):
        axes[2].text(
            index,
            float(value) + 0.02 * max(1.0, float(capacity_values.max())),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    for index, ax in enumerate(axes):
        ax.text(-0.13, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.65, w_pad=1.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig15_holdout_capacity_shift_audit.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    frame = pd.read_csv(
        SOURCE, usecols=["segment_id", "target_power_kw", "fc_input_power_kw"]
    )
    split = split_at_largest_segment_gap(
        pd.read_csv(SOURCE, usecols=["timestamp", "segment_id"])
    )
    calibration = frame[frame.segment_id.isin(split.calibration_segments)].copy()
    holdout = frame[frame.segment_id.isin(split.holdout_segments)].copy()
    positive_cal = calibration[calibration.fc_input_power_kw > 0]
    positive_hold = holdout[holdout.fc_input_power_kw > 0]
    references = (
        (
            "frozen_calibration_target_max",
            float(calibration.target_power_kw.max()),
            "frozen development reference; not a physical rating",
        ),
        (
            "calibration_measured_max",
            float(calibration.fc_input_power_kw.max()),
            "calibration-only diagnostic reference",
        ),
        (
            "controller_40kw_candidate",
            40.0,
            "post-hoc engineering candidate; requires external rating confirmation",
        ),
    )
    model = load_lzw_multistack_world_model(
        ROOT,
        n_stacks=3,
        heterogeneity_factors=HETEROGENEITY,
        config=WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=TRACKING_TOLERANCE_KW,
        ),
    )
    health = json.loads(HEALTH_CALIBRATION.read_text(encoding="utf-8"))
    health_limit_pct = float(health["terminal_total_damage_pct"])
    pair_rows = []
    for health_case, initial_fraction in HEALTH_CASES.items():
        for policy, assignments in POLICY_ASSIGNMENTS.items():
            assignment = assignments[health_case]
            pair_rows.append(
                {
                    "health_case": health_case,
                    "policy": policy,
                    "assignment": str(assignment),
                    "maximum_pair_power_kw": maximum_pair_power(
                        model, initial_fraction, assignment, health_limit_pct
                    ),
                }
            )
    pair_capacity = pd.DataFrame(pair_rows)
    minimum_pair_capacity_kw = float(pair_capacity.maximum_pair_power_kw.min())
    mapping_reference_kw = (
        1 - CAPACITY_RESERVE_FRACTION
    ) * model.fc_power_reference_kw()
    holdout_max_kw = float(holdout.fc_input_power_kw.max())
    strict_required_kw = (
        holdout_max_kw * mapping_reference_kw / minimum_pair_capacity_kw
    )
    tolerance_required_kw = holdout_max_kw * mapping_reference_kw / (
        minimum_pair_capacity_kw + TRACKING_TOLERANCE_KW
    )
    rows = []
    for source, reference_kw, interpretation in references:
        calibration_demand = (
            positive_cal.fc_input_power_kw.to_numpy(dtype=float)
            / reference_kw
            * mapping_reference_kw
        )
        holdout_demand = (
            positive_hold.fc_input_power_kw.to_numpy(dtype=float)
            / reference_kw
            * mapping_reference_kw
        )
        rows.append(
            {
                "reference_source": source,
                "reference_kw": reference_kw,
                "interpretation": interpretation,
                "calibration_clip_share_positive": float(
                    positive_cal.fc_input_power_kw.gt(reference_kw).mean()
                ),
                "holdout_clip_share_positive": float(
                    positive_hold.fc_input_power_kw.gt(reference_kw).mean()
                ),
                "calibration_strict_capacity_exceedance_share_positive": float(
                    (calibration_demand > minimum_pair_capacity_kw + 1e-12).mean()
                ),
                "holdout_strict_capacity_exceedance_share_positive": float(
                    (holdout_demand > minimum_pair_capacity_kw + 1e-12).mean()
                ),
                "holdout_tracking_envelope_exceedance_share_positive": float(
                    (
                        holdout_demand
                        > minimum_pair_capacity_kw + TRACKING_TOLERANCE_KW + 1e-12
                    ).mean()
                ),
                "holdout_unclipped_peak_system_demand_kw": float(
                    holdout_demand.max()
                ),
            }
        )
    audit = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT / "normalization_capacity_audit.csv", index=False)
    pair_capacity.to_csv(OUTPUT / "health_case_pair_capacity.csv", index=False)
    plot_audit(calibration, holdout, audit, strict_required_kw)
    metadata = {
        "frozen_reference_kw": references[0][1],
        "frozen_reference_origin": references[0][2],
        "holdout_target_power_max_kw": float(holdout.target_power_kw.max()),
        "holdout_measured_power_max_kw": holdout_max_kw,
        "mapping_system_power_reference_kw": mapping_reference_kw,
        "minimum_initial_pair_capacity_kw": minimum_pair_capacity_kw,
        "posthoc_minimum_reference_for_strict_capacity_kw": strict_required_kw,
        "posthoc_minimum_reference_with_tracking_tolerance_kw": tolerance_required_kw,
        "interpretation": (
            "minimum references are holdout diagnostics, not admissible calibration "
            "parameters; a physical stack/controller rating is required"
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frozen = audit.iloc[0]
    candidate = audit.iloc[-1]
    report = f"""# 留出负载容量与归一化审计

- 冻结30 kW参考来自标定期`target_power_kw`最大值，不是物理额定功率。
- 留出期出现新的40 kW控制目标，实测单堆峰值{holdout_max_kw:.4f} kW；这是相对开发分区的分布外负载档。
- 当前完整回放把高于30 kW的{frozen.holdout_clip_share_positive:.2%}正功率样本截到归一化1.0，因此只能解释为设计包络内验证。
- 若保持当前三堆N+1映射且完全不截峰，30 kW参考下有{frozen.holdout_strict_capacity_exceedance_share_positive:.2%}正功率样本超过双堆初始物理容量；允许5.5 kW离散跟踪容差后仍有{frozen.holdout_tracking_envelope_exceedance_share_positive:.2%}超包络。
- 事后计算的严格容量最低参考为{strict_required_kw:.3f} kW；该值使用了留出峰值，只能诊断，不能回填标定。
- 40 kW可覆盖当前留出峰值且不截断；独立全年归档已支持它作为经验运行参考，但不能在没有铭牌或控制器设计资料时称为额定净功率。

因此，当前瓶颈是物理容量/归一化依据缺失，不是health-greedy控制失效。论文主结果保留30 kW冻结验证并明确截峰比例；获得额定资料后，再决定是否以40 kW物理参考重建全部负载模板和最终回放。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(audit.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
