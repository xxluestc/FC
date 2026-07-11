import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.health.lzw_gamma_calibration import (
    THETA_COLUMNS,
    cumulative_damage_components,
    fit_theta_power_law,
    gamma_scale_for_terminal_cv,
    ghaderi_gamma_params,
    validate_lzw_alignment,
)


ROOT = Path(__file__).resolve().parents[1]


class LzwGammaCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = pd.read_csv(
            ROOT / "data/upstream_lzw/canonical_event_table_6104.csv"
        )
        cls.theta = pd.read_csv(
            ROOT / "data/upstream_lzw/theta_event_trajectory_6104.csv"
        )

    def test_upstream_tables_are_aligned(self):
        validate_lzw_alignment(self.events, self.theta)

    def test_fixed_literature_damage_index_is_monotone(self):
        components = cumulative_damage_components(self.events)
        self.assertTrue((components.diff().iloc[1:] >= -1e-12).all().all())
        self.assertAlmostEqual(components.total_damage_pct.iloc[0], 0.0)
        self.assertAlmostEqual(
            components.total_damage_pct.iloc[-1], 9.353028530555553
        )

    def test_theta_mapping_matches_endpoints_and_trend(self):
        components = cumulative_damage_components(self.events)
        mapping, fitted, metrics = fit_theta_power_law(
            components.total_damage_pct, self.theta
        )
        np.testing.assert_allclose(
            mapping.theta_reported(0.0),
            self.theta.loc[:49, THETA_COLUMNS].mean().to_numpy(),
        )
        np.testing.assert_allclose(
            mapping.theta_reported(mapping.damage_reference_pct),
            self.theta.loc[len(self.theta) - 50 :, THETA_COLUMNS].mean().to_numpy(),
        )
        self.assertEqual(fitted.shape, (6104, 3))
        self.assertGreater(metrics[THETA_COLUMNS[0]]["r2"], 0.97)
        self.assertGreater(metrics[THETA_COLUMNS[1]]["r2"], 0.93)
        self.assertGreater(metrics[THETA_COLUMNS[2]]["r2"], 0.97)

    def test_gamma_scale_targets_requested_terminal_cv(self):
        total, continuous, cv = 9.35, 6.5, 0.1
        beta = gamma_scale_for_terminal_cv(total, continuous, cv)
        realized = np.sqrt(continuous * beta) / total
        self.assertAlmostEqual(realized, cv)

    def test_action_parameters_keep_idle_and_off_distinct(self):
        params = ghaderi_gamma_params(gamma_scale=0.1)
        idle_rate = params.load_rate_map.rate_at(0.0)
        nominal_rate = params.load_rate_map.rate_at(195.0)
        high_rate = params.load_rate_map.rate_at(370.0)
        self.assertGreater(idle_rate, nominal_rate)
        self.assertGreater(high_rate, nominal_rate)
        self.assertEqual(params.off_rate_per_hour, 0.0)


if __name__ == "__main__":
    unittest.main()
