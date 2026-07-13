"""Build canonical paper tables and a traceable claim-value registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "paper_evidence"
HEALTH = RESULTS / "health/lzw_gamma_calibration.json"
ROBUSTNESS = RESULTS / "fc_only_service_robustness/robustness_summary.csv"
ASSIGNMENT = RESULTS / "fc_only_service_holdout_assignment/summary.csv"
FULL_AGGREGATE = RESULTS / "fc_only_full_holdout_replay/aggregate_metrics.csv"
FULL_SUMMARY = RESULTS / "fc_only_full_holdout_replay/summary.csv"
FULL_MANIFEST = RESULTS / "fc_only_full_holdout_replay/segment_manifest.csv"
CAPACITY = RESULTS / "fc_only_holdout_capacity_audit/normalization_capacity_audit.csv"
CAPACITY_METADATA = RESULTS / "fc_only_holdout_capacity_audit/metadata.json"
SEGMENT_BOOTSTRAP = RESULTS / "fc_only_full_holdout_statistics/segment_bootstrap_summary.csv"
SEGMENT_TESTS = RESULTS / "fc_only_full_holdout_statistics/primary_wilcoxon_tests.csv"
SEGMENT_LEAVE_ONE_OUT = RESULTS / "fc_only_full_holdout_statistics/leave_one_segment_out.csv"
SEGMENT_SIGN_REVERSAL = RESULTS / "fc_only_full_holdout_statistics/single_sign_reversal.csv"
SEGMENT_INFLUENCE = RESULTS / "fc_only_full_holdout_statistics/influence_summary.json"
NORMALIZATION_REFERENCE = RESULTS / "fc_only_normalization_sensitivity/reference_summary.csv"
NORMALIZATION_EFFECTS = RESULTS / "fc_only_normalization_sensitivity/health_effects.csv"
NORMALIZATION_STATISTICS = RESULTS / "fc_only_normalization_sensitivity/segment_statistics.csv"
NORMALIZATION_METADATA = RESULTS / "fc_only_normalization_sensitivity/metadata.json"
TRACKING_TOLERANCE = RESULTS / "fc_only_tracking_tolerance_audit/tolerance_sweep.csv"
TRACKING_TOLERANCE_METADATA = RESULTS / "fc_only_tracking_tolerance_audit/metadata.json"
SERVICE_OBJECTIVE_SUMMARY = RESULTS / "fc_only_service_objective_audit/summary.csv"
SERVICE_OBJECTIVE_DETAIL = RESULTS / "fc_only_service_objective_audit/per_decision_regret.csv"
SERVICE_OBJECTIVE_METADATA = RESULTS / "fc_only_service_objective_audit/metadata.json"
STRONG_BASELINES = {
    "real_calibrated": RESULTS / "fc_only_service_scheduler_strong_baseline_real/summary.csv",
    "empirical_markov": RESULTS / "fc_only_service_scheduler_strong_baseline_markov/summary.csv",
    "zuo_slow": RESULTS / "fc_only_service_scheduler_strong_baseline_zuo_slow/summary.csv",
    "zuo_fast": RESULTS / "fc_only_service_scheduler_strong_baseline_zuo_fast/summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def method_parameters(health):
    coefficients = health["coefficients_percent_units"]
    return pd.DataFrame(
        [
            ("n_stacks", 3, "count", "three-stack N+1 system"),
            ("online_stacks_at_positive_demand", 2, "count", "one stack rests"),
            ("fast_step", 1.0, "s", "real-load sampling interval"),
            ("allowed_currents", "0;25;90;120;160;195;270;370", "A", "discrete action grid"),
            ("tracking_tolerance", 5.5, "kW", "frozen FC-only hard bound"),
            ("capacity_reserve", 0.05, "fraction", "healthy pair mapping reserve"),
            ("load_normalization_reference", 30.0, "kW", "development target maximum; not rating"),
            ("slow_reschedule_period", 24.0, "h", "minimum health-greedy epoch"),
            ("health_boundary", health["terminal_total_damage_pct"], "%", "LZW endpoint; not failure threshold"),
            ("gamma_scale", health["gamma_scale_pct"], "%", "terminal CV 10% assumption"),
            ("heterogeneity_factors", "1.00;1.05;1.10", "factor", "predeclared stack factors"),
            ("start_damage", coefficients["start_stop_pct_per_cycle"], "%/start", "Ghaderi/Pei structure"),
            ("high_load_damage", coefficients["high_load_pct_per_hour"], "%/h", "370 A regime"),
            ("low_load_damage", coefficients["low_load_pct_per_hour"], "%/h", "energized 0 A regime"),
            ("natural_on_damage", coefficients["natural_on_pct_per_hour"], "%/h", "positive-current operation"),
            ("load_shift_damage", coefficients["load_shift_pct_per_cycle"], "%/shift", "current-level change"),
        ],
        columns=["parameter", "value", "unit", "interpretation"],
    )


def strong_baseline_table():
    frames = []
    for scenario, path in STRONG_BASELINES.items():
        frame = pd.read_csv(path)
        frame.insert(0, "scenario", scenario)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    health_mean = (
        table[table.policy == "health_greedy"]
        .set_index("scenario")
        .time_to_limit_mean_h
    )
    table["delta_vs_health_greedy_mean_h"] = table.apply(
        lambda row: row.time_to_limit_mean_h - health_mean.loc[row.scenario], axis=1
    )
    return table


def full_holdout_table():
    aggregate = pd.read_csv(FULL_AGGREGATE)
    summary = pd.read_csv(FULL_SUMMARY).set_index("health_case")
    aggregate["terminal_max_delta_mean_pct"] = aggregate.apply(
        lambda row: (
            summary.loc[row.health_case, "terminal_max_delta_mean_pct"]
            if row.policy == "health_greedy"
            else 0.0
        ),
        axis=1,
    )
    aggregate["terminal_max_nonworse_share"] = aggregate.apply(
        lambda row: (
            summary.loc[row.health_case, "health_greedy_nonworse_share"]
            if row.policy == "health_greedy"
            else 1.0
        ),
        axis=1,
    )
    return aggregate


def canonical_claims(
    strong,
    robustness,
    assignment,
    full,
    capacity,
    capacity_meta,
    segment_bootstrap,
    segment_tests,
    normalization_reference,
    normalization_effects,
    normalization_statistics,
    influence_summary,
    tracking_tolerance,
    tracking_tolerance_meta,
    service_objective,
):
    real = strong[strong.scenario == "real_calibrated"].set_index("policy")
    full_indexed = full.set_index(["health_case", "policy"])
    frozen_capacity = capacity.set_index("reference_source").loc[
        "frozen_calibration_target_max"
    ]
    manifest = pd.read_csv(FULL_MANIFEST)
    primary_bootstrap = segment_bootstrap[
        segment_bootstrap.metric.eq("terminal_max_delta_mean_pct")
        & segment_bootstrap.health_case.isin(("oldest_stack_0", "oldest_stack_1"))
    ].set_index("health_case")
    primary_tests = segment_tests.set_index("health_case")
    claims = {
        "service_real_mean_boundary_h": {
            policy: float(real.loc[policy, "time_to_limit_mean_h"])
            for policy in ("fixed_pair", "health_greedy", "expected_max", "gamma_cvar")
        },
        "expected_max_minus_health_greedy_mean_h_by_scenario": {
            scenario: round(
                float(
                group.set_index("policy").loc["expected_max", "time_to_limit_mean_h"]
                - group.set_index("policy").loc["health_greedy", "time_to_limit_mean_h"]
                ),
                3,
            )
            for scenario, group in strong.groupby("scenario")
        },
        "health_greedy_minus_fixed_by_scenario": {
            scenario: {
                "gain_h": round(float(
                    group.set_index("policy").loc["health_greedy", "time_to_limit_mean_h"]
                    - group.set_index("policy").loc["fixed_pair", "time_to_limit_mean_h"]
                ), 3),
                "relative_gain_pct": round(float(
                    100.0
                    * (
                        group.set_index("policy").loc["health_greedy", "time_to_limit_mean_h"]
                        - group.set_index("policy").loc["fixed_pair", "time_to_limit_mean_h"]
                    )
                    / group.set_index("policy").loc["fixed_pair", "time_to_limit_mean_h"]
                ), 3),
            }
            for scenario, group in strong.groupby("scenario")
        },
        "robustness": {
            "cases": int(len(robustness)),
            "minimum_paired_gain_h": float(robustness.minimum_gain_h.min()),
            "all_nonworse": bool(robustness.nonworse_share.eq(1.0).all()),
        },
        "holdout_assignment": {
            policy: {
                "cases": int(row.cases),
                "online_set_hit_share": round(float(row.online_set_hit_share), 6),
                "regret_max_pct": round(float(row.regret_max_pct), 9),
                "tracking_max_abs_kw": round(float(row.tracking_max_abs_kw), 6),
            }
            for policy, row in assignment.set_index("policy").iterrows()
        },
        "full_holdout": {
            "segments": int(manifest.segment_id.nunique()),
            "rows": int(manifest.rows.sum()),
            "positive_rows": int(manifest.positive_steps.sum()),
            "evaluated_steps": int(full.steps.sum()),
            "constraint_violations": int(full.constraint_violation_steps.sum()),
            "audited_safety_override_steps": int(full.safety_override_steps.sum()),
            "tracking_max_abs_kw": round(float(full.tracking_max_abs_kw.max()), 6),
            "oldest_stack_0_terminal_max_delta_mean_pct": round(float(
                full_indexed.loc[("oldest_stack_0", "health_greedy"), "terminal_max_delta_mean_pct"]
            ), 9),
            "oldest_stack_1_terminal_max_delta_mean_pct": round(float(
                full_indexed.loc[("oldest_stack_1", "health_greedy"), "terminal_max_delta_mean_pct"]
            ), 9),
            "oldest_stack_0_total_expected_damage_delta_pct": round(float(
                full_indexed.loc[("oldest_stack_0", "health_greedy"), "expected_damage_increment_sum_pct"]
                - full_indexed.loc[("oldest_stack_0", "fixed_pair"), "expected_damage_increment_sum_pct"]
            ), 9),
            "oldest_stack_1_total_expected_damage_delta_pct": round(float(
                full_indexed.loc[("oldest_stack_1", "health_greedy"), "expected_damage_increment_sum_pct"]
                - full_indexed.loc[("oldest_stack_1", "fixed_pair"), "expected_damage_increment_sum_pct"]
            ), 9),
        },
        "full_holdout_segment_statistics": {
            health_case: {
                "n_segments": int(primary_bootstrap.loc[health_case, "n_segments"]),
                "terminal_max_delta_mean_pct": round(float(
                    primary_bootstrap.loc[health_case, "estimate"]
                ), 9),
                "bootstrap_ci95_lower_pct": round(float(
                    primary_bootstrap.loc[health_case, "ci95_lower"]
                ), 9),
                "bootstrap_ci95_upper_pct": round(float(
                    primary_bootstrap.loc[health_case, "ci95_upper"]
                ), 9),
                "better_segments": int(
                    primary_tests.loc[health_case, "better_segments"]
                ),
                "wilcoxon_p_one_sided": round(float(
                    primary_tests.loc[health_case, "p_value_one_sided"]
                ), 9),
                "wilcoxon_p_holm": round(float(
                    primary_tests.loc[health_case, "p_value_holm"]
                ), 9),
            }
            for health_case in ("oldest_stack_0", "oldest_stack_1")
        },
        "segment_influence": {
            "leave_one_out_all_holm_significant": bool(
                influence_summary["leave_one_out_all_holm_significant"]
            ),
            "leave_one_out_max_holm_p": float(
                influence_summary["leave_one_out_max_holm_p"]
            ),
            "single_reversal_cases": int(
                influence_summary["single_reversal_cases"]
            ),
            "single_reversal_holm_failures": int(
                influence_summary["single_reversal_holm_failures"]
            ),
        },
        "normalization_sensitivity": {
            "clipped_share_positive_by_reference": {
                str(int(row.normalization_power_kw)): round(
                    float(row.clipped_share_positive), 9
                )
                for row in normalization_reference.itertuples(index=False)
            },
            "all_constraint_violation_steps": int(
                normalization_reference.constraint_violation_steps.sum()
            ),
            "terminal_max_delta_mean_pct": {
                str(int(reference_kw)): {
                    health_case: round(float(
                        normalization_effects.loc[
                            normalization_effects.normalization_power_kw.eq(reference_kw)
                            & normalization_effects.health_case.eq(health_case),
                            "terminal_max_delta_mean_pct",
                        ].iloc[0]
                    ), 9)
                    for health_case in ("oldest_stack_0", "oldest_stack_1")
                }
                for reference_kw in (30.0, 35.0, 40.0)
            },
            "all_primary_segments_better": bool(
                normalization_statistics.better_segments.eq(8).all()
            ),
            "all_primary_ci95_upper_below_zero": bool(
                normalization_statistics.ci95_upper_pct.lt(0).all()
            ),
            "all_primary_holm_significant": bool(
                normalization_statistics.reject_holm_0p05.all()
            ),
        },
        "tracking_tolerance_targeted_audit": {
            "segment_id": int(tracking_tolerance_meta["segment_id"]),
            "minimum_tested_success_kw": float(
                tracking_tolerance_meta["minimum_tested_success_kw"]
            ),
            "largest_tested_failure_kw": float(
                tracking_tolerance.loc[
                    ~tracking_tolerance.success, "tracking_tolerance_kw"
                ].max()
            ),
            "frozen_tolerance_kw": 5.5,
            "frozen_margin_over_minimum_tested_success_kw": round(
                5.5 - tracking_tolerance_meta["minimum_tested_success_kw"], 6
            ),
            "full_holdout_claim": bool(
                tracking_tolerance_meta["full_holdout_claim"]
            ),
        },
        "service_objective_regret_audit": {
            policy: {
                "decision_points": int(row.decision_points),
                "expected_regret_max": float(row.expected_regret_max),
                "cvar_regret_max": float(row.cvar_regret_max),
                "expected_online_set_match_share": float(
                    row.expected_online_set_match_share
                ),
                "cvar_online_set_match_share": float(
                    row.cvar_online_set_match_share
                ),
                "expected_assignment_match_share": float(
                    row.expected_assignment_match_share
                ),
            }
            for policy, row in service_objective.set_index("policy").iterrows()
        },
        "capacity_boundary": {
            "holdout_clip_share_positive": round(float(frozen_capacity.holdout_clip_share_positive), 9),
            "holdout_strict_exceedance_share_positive": round(float(
                frozen_capacity.holdout_strict_capacity_exceedance_share_positive
            ), 9),
            "holdout_tracking_envelope_exceedance_share_positive": round(float(
                frozen_capacity.holdout_tracking_envelope_exceedance_share_positive
            ), 9),
            "posthoc_strict_reference_lower_bound_kw": round(float(
                capacity_meta["posthoc_minimum_reference_for_strict_capacity_kw"]
            ), 6),
            "controller_candidate_kw": 40.0,
        },
    }
    if claims["full_holdout"]["evaluated_steps"] != 518_490:
        raise AssertionError("paper evidence omitted full holdout steps")
    if claims["full_holdout"]["constraint_violations"] != 0:
        raise AssertionError("paper evidence contains holdout constraint violations")
    if not claims["robustness"]["all_nonworse"]:
        raise AssertionError("robustness summary no longer supports all-nonworse claim")
    if not all(
        row["bootstrap_ci95_upper_pct"] < 0.0
        and row["wilcoxon_p_holm"] < 0.05
        and row["better_segments"] == row["n_segments"] == 8
        for row in claims["full_holdout_segment_statistics"].values()
    ):
        raise AssertionError("segment-level primary evidence no longer supports C8")
    if not (
        claims["segment_influence"]["leave_one_out_all_holm_significant"]
        and claims["segment_influence"]["single_reversal_cases"] == 16
        and claims["normalization_sensitivity"]["all_constraint_violation_steps"] == 0
        and claims["normalization_sensitivity"]["all_primary_segments_better"]
        and claims["normalization_sensitivity"]["all_primary_ci95_upper_below_zero"]
        and claims["normalization_sensitivity"]["all_primary_holm_significant"]
    ):
        raise AssertionError("sensitivity evidence no longer supports bounded C8 claim")
    tolerance_claim = claims["tracking_tolerance_targeted_audit"]
    if not (
        tolerance_claim["largest_tested_failure_kw"] == 4.9
        and tolerance_claim["minimum_tested_success_kw"] == 4.95
        and not tolerance_claim["full_holdout_claim"]
    ):
        raise AssertionError("targeted tracking-tolerance boundary changed")
    objective_claim = claims["service_objective_regret_audit"]
    if not all(
        row["decision_points"] == 12
        and row["expected_regret_max"] == 0.0
        and row["cvar_regret_max"] == 0.0
        and row["expected_online_set_match_share"] == 1.0
        and row["cvar_online_set_match_share"] == 1.0
        for row in objective_claim.values()
    ):
        raise AssertionError("service objective regret audit changed")
    return claims


def main():
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    capacity_meta = json.loads(CAPACITY_METADATA.read_text(encoding="utf-8"))
    tables = {
        "table01_method_parameters.csv": method_parameters(health),
        "table02_service_strong_baselines.csv": strong_baseline_table(),
        "table03_service_robustness.csv": pd.read_csv(ROBUSTNESS),
        "table04_holdout_assignment.csv": pd.read_csv(ASSIGNMENT),
        "table05_full_holdout.csv": full_holdout_table(),
        "table06_capacity_audit.csv": pd.read_csv(CAPACITY),
        "table07_segment_bootstrap.csv": pd.read_csv(SEGMENT_BOOTSTRAP),
        "table08_segment_wilcoxon.csv": pd.read_csv(SEGMENT_TESTS),
        "table09_segment_leave_one_out.csv": pd.read_csv(SEGMENT_LEAVE_ONE_OUT),
        "table10_segment_sign_reversal.csv": pd.read_csv(SEGMENT_SIGN_REVERSAL),
        "table11_normalization_reference.csv": pd.read_csv(NORMALIZATION_REFERENCE),
        "table12_normalization_health_effects.csv": pd.read_csv(NORMALIZATION_EFFECTS),
        "table13_normalization_segment_statistics.csv": pd.read_csv(NORMALIZATION_STATISTICS),
        "table14_tracking_tolerance_targeted.csv": pd.read_csv(TRACKING_TOLERANCE),
        "table15_service_objective_summary.csv": pd.read_csv(SERVICE_OBJECTIVE_SUMMARY),
        "table16_service_objective_detail.csv": pd.read_csv(SERVICE_OBJECTIVE_DETAIL),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(9).to_csv(OUTPUT / name, index=False)

    strong = tables["table02_service_strong_baselines.csv"]
    robustness = tables["table03_service_robustness.csv"]
    assignment = tables["table04_holdout_assignment.csv"]
    full = tables["table05_full_holdout.csv"]
    capacity = tables["table06_capacity_audit.csv"]
    segment_bootstrap = tables["table07_segment_bootstrap.csv"]
    segment_tests = tables["table08_segment_wilcoxon.csv"]
    normalization_reference = tables["table11_normalization_reference.csv"]
    normalization_effects = tables["table12_normalization_health_effects.csv"]
    normalization_statistics = tables["table13_normalization_segment_statistics.csv"]
    influence_summary = json.loads(SEGMENT_INFLUENCE.read_text(encoding="utf-8"))
    tracking_tolerance = tables["table14_tracking_tolerance_targeted.csv"]
    tracking_tolerance_meta = json.loads(
        TRACKING_TOLERANCE_METADATA.read_text(encoding="utf-8")
    )
    service_objective = tables["table15_service_objective_summary.csv"]
    claims = canonical_claims(
        strong,
        robustness,
        assignment,
        full,
        capacity,
        capacity_meta,
        segment_bootstrap,
        segment_tests,
        normalization_reference,
        normalization_effects,
        normalization_statistics,
        influence_summary,
        tracking_tolerance,
        tracking_tolerance_meta,
        service_objective,
    )
    (OUTPUT / "claim_values.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = [
        HEALTH,
        ROBUSTNESS,
        ASSIGNMENT,
        FULL_AGGREGATE,
        FULL_SUMMARY,
        FULL_MANIFEST,
        CAPACITY,
        CAPACITY_METADATA,
        SEGMENT_BOOTSTRAP,
        SEGMENT_TESTS,
        SEGMENT_LEAVE_ONE_OUT,
        SEGMENT_SIGN_REVERSAL,
        SEGMENT_INFLUENCE,
        NORMALIZATION_REFERENCE,
        NORMALIZATION_EFFECTS,
        NORMALIZATION_STATISTICS,
        NORMALIZATION_METADATA,
        TRACKING_TOLERANCE,
        TRACKING_TOLERANCE_METADATA,
        SERVICE_OBJECTIVE_SUMMARY,
        SERVICE_OBJECTIVE_DETAIL,
        SERVICE_OBJECTIVE_METADATA,
        *STRONG_BASELINES.values(),
    ]
    source_manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in sources
        ],
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = """# 论文规范证据表

本目录由`scripts/46_build_paper_evidence_tables.py`从冻结实验结果自动生成。

- `table01`：方法参数与解释边界；
- `table02`：四种负载场景的长期强基线；
- `table03`：11组单因素稳健性；
- `table04`：冻结窗口oracle选择；
- `table05`：全部真实留出段可行性和代价权衡；
- `table06`：归一化参考与N+1容量审计；
- `table07`：完整留出segment bootstrap点估计与95%区间；
- `table08`：预声明最差堆主检验与Holm校正；
- `table09-10`：逐段删一与单段符号反转影响力审计；
- `table11-13`：30/35/40 kW事后归一化敏感性；
- `table14`：冻结最大误差案例的跟踪容差边界；
- `table15-16`：12个慢层决策点的自身目标遗憾和分配明细；
- `claim_values.json`：正文可引用的规范数值；
- `source_manifest.json`：每个输入文件的SHA-256。

任何正文数字应先进入`claim_values.json`，不得从图上估读或手工改写。40 kW仍是待物理资料
确认的候选，不因出现在规范表中而成为已验证额定值。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(claims, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
