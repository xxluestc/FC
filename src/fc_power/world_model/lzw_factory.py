"""Construct the multi-stack world model from tracked LZW calibration files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fc_power.health.dynamic_proxy import DynamicPerformanceLossProxy, LzwIvConditions
from fc_power.health.gamma_process import GammaHealthModel
from fc_power.health.lzw_gamma_calibration import (
    ThetaPowerLawMap,
    ghaderi_gamma_params,
)
from fc_power.world_model.mechanistic import (
    MechanisticMultiStackWorldModel,
    WorldModelConfig,
)


def load_lzw_multistack_world_model(
    repo_root: str | Path,
    n_stacks: int = 2,
    *,
    heterogeneity_factors=None,
    config: WorldModelConfig | None = None,
) -> MechanisticMultiStackWorldModel:
    """Load a deterministic model definition; stochasticity is chosen per step."""

    if n_stacks <= 0:
        raise ValueError("n_stacks must be positive")
    root = Path(repo_root)
    upstream = root / "data/upstream_lzw"
    health_results = root / "data/results/health"
    calibration = json.loads(
        (health_results / "lzw_gamma_calibration.json").read_text(encoding="utf-8")
    )
    conditions = LzwIvConditions.from_upstream_dict(
        json.loads(
            (upstream / "current_point_cost_conditions.json").read_text(
                encoding="utf-8"
            )
        )
    )
    table = pd.read_csv(upstream / "current_point_degradation_cost_table.csv")
    normalization = float(table.equivalent_stack_power_loss_clipped_W.max())
    mapping = ThetaPowerLawMap.from_dict(calibration["theta_power_law_map"])
    proxy = DynamicPerformanceLossProxy(mapping, conditions, normalization)

    factors = (
        tuple(1.0 for _ in range(n_stacks))
        if heterogeneity_factors is None
        else tuple(float(value) for value in heterogeneity_factors)
    )
    if len(factors) != n_stacks:
        raise ValueError("heterogeneity_factors must match n_stacks")
    health_models = tuple(
        GammaHealthModel(
            ghaderi_gamma_params(
                calibration["gamma_scale_pct"], heterogeneity_factor=factor
            )
        )
        for factor in factors
    )
    return MechanisticMultiStackWorldModel(
        health_models,
        tuple(proxy for _ in range(n_stacks)),
        WorldModelConfig() if config is None else config,
    )
