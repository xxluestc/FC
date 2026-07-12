import unittest

import numpy as np

from fc_power.evaluation.gamma_sensitivity import (
    GammaExposure,
    sample_repeated_exposure,
)


class GammaSensitivityTest(unittest.TestCase):
    def test_aggregated_mean_matches_continuous_plus_discrete_exposure(self):
        exposure = GammaExposure((0.02, 0.03), (0.01, 0.02), (0.0, 1.0), 60)
        sampled = sample_repeated_exposure(exposure, 100, 0.1, 20000, 7)
        expected = np.array([3.0, 5.0])
        np.testing.assert_allclose(sampled.mean(axis=0), expected, rtol=0.015)

    def test_larger_scale_increases_variance_without_changing_mean(self):
        exposure = GammaExposure((0.04,), (0.01,), (0.0,), 60)
        uniforms = np.random.default_rng(9).uniform(1e-9, 1 - 1e-9, (10000, 1))
        low = sample_repeated_exposure(
            exposure, 100, 0.03, 10000, 9, common_uniforms=uniforms
        )
        high = sample_repeated_exposure(
            exposure, 100, 0.5, 10000, 9, common_uniforms=uniforms
        )
        self.assertGreater(high.std(), low.std())
        self.assertAlmostEqual(high.mean(), low.mean(), delta=0.08)


if __name__ == "__main__":
    unittest.main()
