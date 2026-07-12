import unittest

import numpy as np
import pandas as pd

from fc_power.evaluation import extract_action_exposure
from fc_power.health.lzw_gamma_calibration import GhaderiPeiCoefficients


class DegradationSensitivityTest(unittest.TestCase):
    def test_exposure_regimes_are_exclusive_and_events_are_counted(self):
        trajectory = pd.DataFrame(
            {
                "step": np.arange(5),
                "is_soc_recovery": False,
                "stack_0_current_a": [0, 90, 195, 370, 0],
                "stack_0_on": [False, True, True, True, False],
                "stack_1_current_a": [0, 0, 0, 25, 25],
                "stack_1_on": [True, True, False, True, True],
            }
        )
        exposure = extract_action_exposure(
            trajectory, 2, (1.0, 1.1), maximum_current_a=370.0
        )
        self.assertEqual(exposure.natural_on_h[0], 2 / 3600)
        self.assertEqual(exposure.high_load_h[0], 1 / 3600)
        self.assertEqual(exposure.low_load_h[0], 0.0)
        self.assertEqual(exposure.start_count, (1, 2))
        self.assertEqual(exposure.load_shift_count, (2, 0))
        self.assertEqual(exposure.low_load_h[1], 2 / 3600)
        self.assertEqual(exposure.natural_on_h[1], 2 / 3600)

    def test_damage_evaluation_uses_heterogeneity(self):
        trajectory = pd.DataFrame(
            {
                "stack_0_current_a": [195.0],
                "stack_0_on": [True],
                "stack_1_current_a": [195.0],
                "stack_1_on": [True],
            }
        )
        exposure = extract_action_exposure(trajectory, 2, (1.0, 2.0))
        damage = exposure.damage_by_stack(GhaderiPeiCoefficients())
        self.assertAlmostEqual(damage[1], 2 * damage[0])


if __name__ == "__main__":
    unittest.main()
