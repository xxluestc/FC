"""Audit slow-layer decision regret under Expected-max and Gamma-CVaR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    candidate_assignments,
    choose_service_assignment,
    evaluate_service_assignment,
    orient_service_pair,
    stationary_service_exposure,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data/results/fc_only_service_templates/service_exposure_templates.csv"
CALIBRATION = ROOT / "data/results/health/lzw_gamma_calibration.json"
OUTPUT = ROOT / "data/results/fc_only_service_objective_audit"
SCENARIOS = (
    "real_calibration_window",
    "empirical_markov_1s",
    "zuo_fast_30s",
    "zuo_slow_30s",
)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}
HETEROGENEITY = (1.0, 1.05, 1.10)
POLICIES = ("health_greedy", "expected_max", "gamma_cvar")


def load_exposure(source):
    table = pd.read_csv(TEMPLATES)
    selected = table[table.template_source.eq(source)]
    if len(selected) < 3:
        raise AssertionError(f"too few frozen templates for {source}")
    templates = [
        ServiceExposure(
            duration_h=float(row.duration_h),
            continuous_mean_pct=(
                row.role_0_continuous_mean_pct,
                row.role_1_continuous_mean_pct,
            ),
            load_shift_damage_pct=(
                row.role_0_load_shift_damage_pct,
                row.role_1_load_shift_damage_pct,
            ),
            operational_start_damage_pct=(
                row.role_0_operational_start_damage_pct,
                row.role_1_operational_start_damage_pct,
            ),
        )
        for row in selected.itertuples(index=False)
    ]
    return stationary_service_exposure(templates, 1.0)


def policy_assignments(state, exposure, config):
    pair = tuple(
        sorted(
            range(len(state.damage_pct)),
            key=lambda index: (state.damage_pct[index], index),
        )[:2]
    )
    return {
        "health_greedy": orient_service_pair(
            pair, state, exposure, config.heterogeneity_factors
        ),
        "expected_max": choose_service_assignment(
            state, exposure, config, objective="expected_max"
        ).assignment,
        "gamma_cvar": choose_service_assignment(
            state, exposure, config, objective="gamma_cvar"
        ).assignment,
    }


def main():
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    config = ServiceScheduleConfig(
        health_limit_pct=float(calibration["terminal_total_damage_pct"]),
        gamma_scale_pct=float(calibration["gamma_scale_pct"]),
        heterogeneity_factors=HETEROGENEITY,
        start_damage_pct=float(
            calibration["coefficients_percent_units"]["start_stop_pct_per_cycle"]
        ),
        risk_horizon_h=100.0,
        risk_samples=128,
    )
    rows = []
    for scenario in SCENARIOS:
        exposure = load_exposure(scenario)
        for health_case, fractions in HEALTH_CASES.items():
            state = ServiceScheduleState(
                tuple(config.health_limit_pct * value for value in fractions)
            )
            candidate_rows = []
            for assignment in candidate_assignments(len(fractions)):
                expected = evaluate_service_assignment(
                    state,
                    exposure,
                    config,
                    assignment,
                    objective="expected_max",
                )
                cvar = evaluate_service_assignment(
                    state,
                    exposure,
                    config,
                    assignment,
                    objective="gamma_cvar",
                )
                candidate_rows.append(
                    {
                        "assignment": assignment,
                        "expected_max_fraction": expected.expected_max_health_fraction,
                        "gamma_cvar_fraction": cvar.cvar_max_health_fraction,
                    }
                )
            expected_min = min(row["expected_max_fraction"] for row in candidate_rows)
            cvar_min = min(row["gamma_cvar_fraction"] for row in candidate_rows)
            assignments = policy_assignments(state, exposure, config)
            for policy, assignment in assignments.items():
                selected = next(
                    row for row in candidate_rows if row["assignment"] == assignment
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "health_case": health_case,
                        "policy": policy,
                        "assignment": str(assignment),
                        "online_set": str(tuple(sorted(assignment))),
                        "expected_max_fraction": selected["expected_max_fraction"],
                        "expected_regret_fraction": (
                            selected["expected_max_fraction"] - expected_min
                        ),
                        "gamma_cvar_fraction": selected["gamma_cvar_fraction"],
                        "gamma_cvar_regret_fraction": (
                            selected["gamma_cvar_fraction"] - cvar_min
                        ),
                    }
                )
    per_decision = pd.DataFrame(rows)
    keys = ["scenario", "health_case"]
    expected_reference = per_decision[per_decision.policy.eq("expected_max")][
        keys + ["assignment", "online_set"]
    ].rename(
        columns={
            "assignment": "expected_optimal_assignment",
            "online_set": "expected_optimal_online_set",
        }
    )
    cvar_reference = per_decision[per_decision.policy.eq("gamma_cvar")][
        keys + ["assignment", "online_set"]
    ].rename(
        columns={
            "assignment": "cvar_optimal_assignment",
            "online_set": "cvar_optimal_online_set",
        }
    )
    per_decision = per_decision.merge(
        expected_reference, on=keys, validate="many_to_one"
    ).merge(cvar_reference, on=keys, validate="many_to_one")
    per_decision["expected_assignment_match"] = (
        per_decision.assignment == per_decision.expected_optimal_assignment
    )
    per_decision["expected_online_set_match"] = (
        per_decision.online_set == per_decision.expected_optimal_online_set
    )
    per_decision["cvar_assignment_match"] = (
        per_decision.assignment == per_decision.cvar_optimal_assignment
    )
    per_decision["cvar_online_set_match"] = (
        per_decision.online_set == per_decision.cvar_optimal_online_set
    )
    summary = (
        per_decision.groupby("policy", sort=False)
        .agg(
            decision_points=("policy", "size"),
            expected_regret_mean=("expected_regret_fraction", "mean"),
            expected_regret_max=("expected_regret_fraction", "max"),
            expected_optimal_share=(
                "expected_regret_fraction",
                lambda values: float(np.isclose(values, 0.0, atol=1e-15).mean()),
            ),
            cvar_regret_mean=("gamma_cvar_regret_fraction", "mean"),
            cvar_regret_max=("gamma_cvar_regret_fraction", "max"),
            cvar_optimal_share=(
                "gamma_cvar_regret_fraction",
                lambda values: float(np.isclose(values, 0.0, atol=1e-15).mean()),
            ),
            expected_assignment_match_share=("expected_assignment_match", "mean"),
            expected_online_set_match_share=("expected_online_set_match", "mean"),
            cvar_assignment_match_share=("cvar_assignment_match", "mean"),
            cvar_online_set_match_share=("cvar_online_set_match", "mean"),
        )
        .reset_index()
    )
    if len(per_decision) != 36:
        raise AssertionError("decision audit did not cover 4 scenarios x 3 states x 3 policies")
    if not np.isclose(
        summary.loc[summary.policy.eq("expected_max"), "expected_regret_max"].iloc[0],
        0.0,
        atol=1e-15,
    ):
        raise AssertionError("Expected-max failed to minimize its declared objective")
    if not np.isclose(
        summary.loc[summary.policy.eq("gamma_cvar"), "cvar_regret_max"].iloc[0],
        0.0,
        atol=1e-15,
    ):
        raise AssertionError("Gamma-CVaR failed to minimize its declared objective")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_decision.round(12).to_csv(OUTPUT / "per_decision_regret.csv", index=False)
    summary.round(12).to_csv(OUTPUT / "summary.csv", index=False)
    metadata = {
        "scope": "12 frozen development decision points; not long-horizon performance",
        "scenarios": SCENARIOS,
        "health_cases": HEALTH_CASES,
        "assignments_enumerated_per_point": 6,
        "risk_horizon_h": config.risk_horizon_h,
        "risk_samples": config.risk_samples,
        "future_demand_used": False,
        "interpretation": (
            "Own-objective optimality is a consistency check. Only cross-objective "
            "regret and downstream boundary time can support comparative claims."
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "# 慢层自身目标遗憾审计\n\n"
    report += (
        "四种冻结开发暴露与三种健康身份形成12个决策点，每点事后枚举六个有序分配。"
        "Expected-max和Gamma-CVaR在自己的目标上最优是实现一致性检查，不单独构成性能证据。\n\n"
    )
    report += summary.to_markdown(index=False) + "\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
