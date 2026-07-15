import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation.chen_dynamic_load import (
    derive_chen_load_levels,
    generate_chen_random_dynamic_load,
)
from fc_power.evaluation.zuo_load_calibration import ZUO_FAST_TRANSITION


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data/processed/chen_efficiency_curves_audited.csv"


class ChenDynamicLoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = derive_chen_load_levels(pd.read_csv(CURVES))

    def test_levels_come_from_curve_peaks_and_n_plus_one_capacity(self):
        self.assertAlmostEqual(self.levels.single_peak_kw, 24.819747513573567)
        self.assertAlmostEqual(self.levels.dual_peak_kw, 50.86374796665447)
        self.assertAlmostEqual(
            self.levels.maximum_two_stack_power_kw,
            125.88539697716798,
        )
        self.assertAlmostEqual(
            self.levels.guaranteed_n_plus_one_power_kw,
            114.34203706331125,
        )
        self.assertAlmostEqual(
            self.levels.reserve_peak_kw,
            0.9 * self.levels.guaranteed_n_plus_one_power_kw,
        )
        self.assertTrue(np.all(np.diff(self.levels.as_array()) > 0))
        maxima = (
            pd.read_csv(CURVES)
            .groupby("stack_id")["net_system_power_kw"]
            .max()
        )
        remaining_capacities = [
            float(maxima.drop(failed_stack).sum())
            for failed_stack in maxima.index
        ]
        self.assertAlmostEqual(
            min(remaining_capacities),
            self.levels.guaranteed_n_plus_one_power_kw,
        )

    def test_random_dynamic_load_is_reproducible_and_actually_changes(self):
        first = generate_chen_random_dynamic_load(
            2026,
            length_s=600,
            levels=self.levels,
            transition_matrix=ZUO_FAST_TRANSITION,
        )
        second = generate_chen_random_dynamic_load(
            2026,
            length_s=600,
            levels=self.levels,
            transition_matrix=ZUO_FAST_TRANSITION,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 600)
        self.assertEqual(first.load_state.nunique(), 4)
        self.assertGreater(first.event_id.nunique(), 20)
        self.assertGreater(first.demand_net_power_kw.nunique(), 20)
        self.assertGreater(first.event_boundary.sum(), 20)
        self.assertLessEqual(
            first.demand_net_power_kw.max(),
            0.95 * self.levels.guaranteed_n_plus_one_power_kw,
        )

    def test_invalid_transition_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stochastic"):
            generate_chen_random_dynamic_load(
                7,
                length_s=30,
                levels=self.levels,
                transition_matrix=np.ones((4, 4)),
            )


if __name__ == "__main__":
    unittest.main()
