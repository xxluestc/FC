import unittest

import numpy as np
import pandas as pd

from fc_power.evaluation.service_templates import (
    CalibrationWindow,
    materialize_calibration_window,
    select_calibration_windows,
    service_exposure_from_trajectory,
)


class ServiceTemplateTest(unittest.TestCase):
    def test_window_selection_is_reproducible_and_calibration_only(self):
        frame = pd.DataFrame(
            {
                "segment_id": np.repeat([0, 1, 2], [10, 20, 30]),
                "power": np.arange(60),
            }
        )
        first = select_calibration_windows(
            frame, (0, 1), length_s=5, count=8, seed=9
        )
        second = select_calibration_windows(
            frame, (0, 1), length_s=5, count=8, seed=9
        )
        self.assertEqual(first, second)
        self.assertTrue(all(item.segment_id in {0, 1} for item in first))
        self.assertEqual(len(set(first)), len(first))
        self.assertTrue(all(item.segment_id != 2 for item in first))

    def test_materialization_never_crosses_segments(self):
        frame = pd.DataFrame(
            {"segment_id": [0] * 5 + [1] * 5, "value": np.arange(10)}
        )
        result = materialize_calibration_window(
            frame, CalibrationWindow(segment_id=0, start_offset=3, length_s=2)
        )
        self.assertEqual(result.segment_id.unique().tolist(), [0])
        self.assertEqual(result.value.tolist(), [3, 4])

    def test_exposure_sorts_roles_and_excludes_entry_start(self):
        trajectory = pd.DataFrame()
        for stack in range(3):
            trajectory[f"stack_{stack}_on"] = [stack < 2] * 3
            trajectory[f"stack_{stack}_expected_continuous_increment_pct"] = (
                [0.2, 0.2, 0.2] if stack == 1 else [0.1, 0.1, 0.1]
            )
            trajectory[f"stack_{stack}_ramp_increment_pct"] = [0.0, 0.0, 0.0]
            trajectory[f"stack_{stack}_shift_increment_pct"] = (
                [0.0, 0.02, 0.0] if stack == 1 else [0.0, 0.01, 0.0]
            )
            trajectory[f"stack_{stack}_start_stop_increment_pct"] = (
                [0.5, 0.03, 0.0] if stack < 2 else [0.0, 0.0, 0.0]
            )
        exposure, role_stacks = service_exposure_from_trajectory(
            trajectory, duration_h=3 / 3600
        )
        self.assertEqual(role_stacks, (1, 0))
        self.assertTrue(np.allclose(exposure.continuous_mean_pct, (0.6, 0.3)))
        self.assertTrue(np.allclose(exposure.load_shift_damage_pct, (0.02, 0.01)))
        self.assertTrue(
            np.allclose(exposure.operational_start_damage_pct, (0.03, 0.03))
        )


if __name__ == "__main__":
    unittest.main()
