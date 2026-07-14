import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/68_screen_lzw_overall_exposure.py"
SPEC = importlib.util.spec_from_file_location("lzw_exposure_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LzwOverallExposureScreenTest(unittest.TestCase):
    def test_fit_is_nonnegative(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "constant": 1.0})
        target = np.array([-1.0, -2.0, -3.0])
        coefficients, prediction = MODULE.fit_nonnegative(
            frame, target, ("x",), np.arange(3)
        )
        self.assertGreaterEqual(coefficients[0], 0.0)
        np.testing.assert_array_equal(prediction, np.zeros(3))

    def test_transition_exposure_uses_raw_intervals(self):
        events = pd.DataFrame(
            {
                "source_segment_row_1based": [2, 3, 4],
                "raw_event_start_index_1based": [1, 4, 7],
                "original_index": [1, 2, 3],
            }
        )
        health = np.array([0.0, 0.1, 0.3])
        raw = np.zeros((9, 2), dtype=float)
        raw[:, 0] = [10, 10, 0, 10, 10, 10, 0, 0, 0]
        raw[:, 1] = [0, 10, 20, 0, 30, 40, 0, 0, 0]
        table = MODULE.build_transition_table(events, health, raw)
        self.assertEqual(table.elapsed_samples.tolist(), [3, 3])
        self.assertEqual(table.stack_on_samples.tolist(), [2, 3])
        self.assertEqual(table.load_on_samples.tolist(), [2, 2])
        self.assertEqual(table.charge_ampere_samples.tolist(), [30.0, 70.0])
        np.testing.assert_allclose(table.delta_health_loss, [0.1, 0.2])

    def test_moving_block_interval_is_reproducible(self):
        model = np.linspace(0.1, 0.2, 30)
        reference = np.linspace(0.2, 0.4, 30)
        first = MODULE.moving_block_improvement_ci(
            model, reference, seed=7, n_resamples=100
        )
        second = MODULE.moving_block_improvement_ci(
            model, reference, seed=7, n_resamples=100
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed"], 0.5)


if __name__ == "__main__":
    unittest.main()
