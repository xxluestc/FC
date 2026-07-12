"""Long-horizon Gamma uncertainty from online-policy exposure traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_distribution


@dataclass(frozen=True)
class GammaExposure:
    continuous_mean_pct: tuple[float, ...]
    discrete_damage_pct: tuple[float, ...]
    initial_damage_pct: tuple[float, ...]
    duration_s: float

    @property
    def n_stacks(self):
        return len(self.continuous_mean_pct)


def exposure_from_trajectory(trajectory: pd.DataFrame, n_stacks: int) -> GammaExposure:
    """Extract only the main-task exposure, excluding SOC recovery."""

    main = (
        trajectory[~trajectory.is_soc_recovery]
        if "is_soc_recovery" in trajectory
        else trajectory
    )
    if len(main) == 0:
        raise ValueError("trajectory contains no main-task samples")
    continuous, discrete, initial = [], [], []
    for index in range(n_stacks):
        required = (
            f"stack_{index}_expected_continuous_increment_pct",
            f"stack_{index}_discrete_increment_pct",
            f"stack_{index}_damage_before_pct",
        )
        if any(column not in main for column in required):
            raise ValueError(f"trajectory is missing stack {index} exposure columns")
        continuous.append(float(main[required[0]].sum()))
        discrete.append(float(main[required[1]].sum()))
        initial.append(float(main[required[2]].iloc[0]))
    return GammaExposure(
        tuple(continuous), tuple(discrete), tuple(initial), float(len(main))
    )


def sample_repeated_exposure(
    exposure: GammaExposure,
    repeats: int,
    gamma_scale_pct: float,
    samples: int,
    seed: int,
    *,
    common_uniforms: np.ndarray | None = None,
) -> np.ndarray:
    """Sample exact aggregated Gamma increments plus deterministic events."""

    if repeats <= 0 or samples <= 0 or gamma_scale_pct <= 0:
        raise ValueError("repeats, samples and gamma scale must be positive")
    if common_uniforms is None:
        common_uniforms = np.random.default_rng(seed).uniform(
            1e-12, 1 - 1e-12, size=(samples, exposure.n_stacks)
        )
    uniforms = np.asarray(common_uniforms, dtype=float)
    if uniforms.shape != (samples, exposure.n_stacks):
        raise ValueError("common_uniforms has an incompatible shape")
    output = np.empty_like(uniforms)
    for index, (continuous, discrete) in enumerate(
        zip(exposure.continuous_mean_pct, exposure.discrete_damage_pct)
    ):
        total_mean = continuous * repeats
        if total_mean > 0:
            sampled_continuous = gamma_distribution.ppf(
                uniforms[:, index],
                a=total_mean / gamma_scale_pct,
                scale=gamma_scale_pct,
            )
        else:
            sampled_continuous = np.zeros(samples)
        output[:, index] = sampled_continuous + discrete * repeats
    return output
