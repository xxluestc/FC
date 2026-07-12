"""Metrics and publication-style plotting helpers."""
from fc_power.evaluation.load_profiles import (
    EVENT_NAMES,
    SyntheticLoadConfig,
    append_soc_recovery_tail,
    classify_power_events,
    generate_event_load,
    generate_real_block_bootstrap,
)
from fc_power.evaluation.multistack_testbed import (
    TestRun,
    TestScenario,
    clip_profile_to_feasible_envelope,
    paired_strategy_comparison,
    run_policy,
    summarize_run,
)
from fc_power.evaluation.gamma_sensitivity import (
    GammaExposure,
    exposure_from_trajectory,
    sample_repeated_exposure,
)

__all__ = [
    "EVENT_NAMES",
    "GammaExposure",
    "SyntheticLoadConfig",
    "TestRun",
    "TestScenario",
    "append_soc_recovery_tail",
    "classify_power_events",
    "clip_profile_to_feasible_envelope",
    "generate_event_load",
    "generate_real_block_bootstrap",
    "exposure_from_trajectory",
    "paired_strategy_comparison",
    "run_policy",
    "sample_repeated_exposure",
    "summarize_run",
]
