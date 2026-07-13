import unittest

import numpy as np
import pandas as pd

from fc_power.evaluation.external_validation import (
    canonicalize_power_packets,
    select_first_operating_block,
)


class ExternalValidationTest(unittest.TestCase):
    def test_duplicate_packets_are_averaged_before_power_derivation(self):
        raw = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:20",
                ],
                "fc_voltage_v": [100.0, 200.0, 150.0, 120.0],
                "fc_current_a": [10.0, 20.0, 20.0, 10.0],
            }
        )
        canonical, audit = canonicalize_power_packets(raw)
        self.assertEqual(audit["duplicate_timestamps"], 1)
        self.assertEqual(audit["source_segments"], 2)
        first = canonical.iloc[0]
        self.assertAlmostEqual(first.fc_voltage_v, 150.0)
        self.assertAlmostEqual(first.fc_current_a, 15.0)
        self.assertAlmostEqual(first.fc_input_power_kw, 2.25)
        self.assertEqual(int(canonical.interpolated_power.sum()), 0)

    def test_short_internal_gap_is_interpolated_and_flagged(self):
        raw = pd.DataFrame(
            {
                "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:00:02"],
                "fc_voltage_v": [100.0, 300.0],
                "fc_current_a": [10.0, 30.0],
            }
        )
        canonical, audit = canonicalize_power_packets(raw)
        self.assertEqual(len(canonical), 3)
        self.assertTrue(canonical.interpolated_power.iloc[1])
        self.assertEqual(audit["interpolated_power_rows"], 1)

    def test_first_block_is_selected_without_crossing_segments(self):
        timestamps = pd.date_range("2026-02-01", periods=12, freq="s")
        canonical = pd.DataFrame(
            {
                "timestamp": timestamps,
                "source_segment_id": [0] * 5 + [1] * 7,
                "fc_input_power_kw": [0.0] * 5 + [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            }
        )
        block = select_first_operating_block(
            canonical,
            "2026-02",
            block_steps=4,
            minimum_positive_share=0.75,
        )
        self.assertEqual(block.source_segment_id.nunique(), 1)
        self.assertEqual(block.source_segment_id.iloc[0], 1)
        self.assertEqual(block.timestamp.iloc[0], timestamps[5])
        np.testing.assert_array_equal(block.block_step, np.arange(4))

    def test_missing_qualifying_block_is_rejected(self):
        canonical = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01", periods=5, freq="s"),
                "source_segment_id": [0] * 5,
                "fc_input_power_kw": [0.0] * 5,
            }
        )
        with self.assertRaises(ValueError):
            select_first_operating_block(
                canonical,
                "2026-03",
                block_steps=4,
                minimum_positive_share=0.5,
            )


if __name__ == "__main__":
    unittest.main()
