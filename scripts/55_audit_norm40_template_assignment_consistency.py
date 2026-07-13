"""Prove whether norm40 template rebuilding changes frozen replay assignments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fc_power.evaluation import (
    ServiceExposure,
    ServiceScheduleState,
    orient_service_pair,
    stationary_service_exposure,
)


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/results/fc_only_service_templates/service_exposure_templates.csv"
NEW = ROOT / "data/results/fc_only_service_templates_norm40/service_exposure_templates.csv"
REPLAY = ROOT / "data/results/fc_only_full_holdout_norm40/metadata.json"
OUTPUT = ROOT / "data/results/fc_only_norm40_template_consistency"
HETEROGENEITY = (1.0, 1.05, 1.10)
HEALTH_CASES = {
    "oldest_stack_2": (0.10, 0.40, 0.80),
    "oldest_stack_0": (0.80, 0.10, 0.40),
    "oldest_stack_1": (0.40, 0.80, 0.10),
}


def load_exposure(path: Path) -> ServiceExposure:
    table = pd.read_csv(path)
    selected = table[table.template_source == "real_calibration_window"]
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


def role_damage(exposure: ServiceExposure) -> tuple[float, float]:
    return tuple(
        exposure.continuous_mean_pct[index]
        + exposure.load_shift_damage_pct[index]
        + exposure.operational_start_damage_pct[index]
        for index in range(2)
    )


def assignments(exposure: ServiceExposure) -> pd.DataFrame:
    rows = []
    for health_case, damage in HEALTH_CASES.items():
        state = ServiceScheduleState(damage)
        health_pair = tuple(sorted(range(3), key=lambda index: (damage[index], index))[:2])
        for policy, pair in (("fixed_pair", (0, 1)), ("health_greedy", health_pair)):
            assignment = orient_service_pair(pair, state, exposure, HETEROGENEITY)
            rows.append(
                {
                    "health_case": health_case,
                    "policy": policy,
                    "stack_for_role_0": assignment[0],
                    "stack_for_role_1": assignment[1],
                }
            )
    return pd.DataFrame(rows).sort_values(["health_case", "policy"]).reset_index(drop=True)


def main() -> None:
    old_exposure = load_exposure(OLD)
    new_exposure = load_exposure(NEW)
    old_assignments = assignments(old_exposure)
    new_assignments = assignments(new_exposure)
    merged = old_assignments.merge(
        new_assignments,
        on=["health_case", "policy"],
        suffixes=("_norm30_template", "_norm40_template"),
        validate="one_to_one",
    )
    merged["identical"] = (
        merged.stack_for_role_0_norm30_template
        == merged.stack_for_role_0_norm40_template
    ) & (
        merged.stack_for_role_1_norm30_template
        == merged.stack_for_role_1_norm40_template
    )
    if not bool(merged.identical.all()):
        raise AssertionError("norm40 template rebuild changes a frozen replay assignment")
    replay_metadata = json.loads(REPLAY.read_text(encoding="utf-8"))
    if float(replay_metadata["normalization_power_kw"]) != 40.0:
        raise AssertionError("reused replay is not the 40 kW replay")

    old_role = role_damage(old_exposure)
    new_role = role_damage(new_exposure)
    metadata = {
        "old_template": str(OLD.relative_to(ROOT)),
        "new_template": str(NEW.relative_to(ROOT)),
        "reused_replay": str(REPLAY.parent.relative_to(ROOT)),
        "normalization_power_kw": 40.0,
        "cases_checked": len(merged),
        "identical_assignments": int(merged.identical.sum()),
        "old_role_damage_pct_per_hour": list(old_role),
        "new_role_damage_pct_per_hour": list(new_role),
        "reuse_reason": (
            "the existing norm40 replay already uses 40 kW demand mapping; the only "
            "template-dependent replay input is the frozen ordered assignment, which "
            "is identical for every predeclared health case and policy"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT / "assignment_comparison.csv", index=False)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 40 kW模板与已有全量回放一致性

- 40 kW重新标定和120个快层模板已经完成。
- 旧/新实车模板的一小时角色总退化暴露分别为`{old_role}`和`{new_role}`个百分点。
- 3种健康身份 x 2种冻结策略共{len(merged)}个入口决策中，在线堆集合和角色顺序{int(merged.identical.sum())}/{len(merged)}完全一致。
- 已有`fc_only_full_holdout_norm40`本身已经使用40 kW需求映射。完整段回放中模板只影响冻结入口角色；其余输入是同一世界模型、同一真实段、同一初始健康和同一Instant快层。
- 因此重建模板不会改变任何已有40 kW完整段动作轨迹，复用现有144例结果成立，无需进行约一小时的重复计算。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
