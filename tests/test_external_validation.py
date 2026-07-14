import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation.external_validation import (
    apply_data_exclusions,
    canonicalize_power_packets,
    extract_target_events,
    load_data_exclusions,
    select_first_operating_block,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalValidationTest(unittest.TestCase):
    def test_configured_exclusion_removes_rows_and_splits_original_segment(self):
        rules = load_data_exclusions(
            ROOT / "configs/21ube0022_data_exclusions.json"
        )
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-29 23:59:59",
                        "2026-01-30 00:00:00",
                        "2026-02-06 23:59:59",
                        "2026-02-07 00:00:00",
                    ]
                ),
                "segment_id": [9, 9, 9, 9],
                "fc_input_power_kw": [10.0, 20.0, 30.0, 40.0],
            }
        )
        filtered, audit = apply_data_exclusions(frame, rules)

        self.assertEqual(filtered.fc_input_power_kw.tolist(), [10.0, 40.0])
        self.assertEqual(filtered.segment_id.nunique(), 1)
        self.assertEqual(filtered.model_segment_id.nunique(), 2)
        self.assertEqual(audit["excluded_rows"], 2)
        self.assertEqual(audit["hard_segment_breaks_added"], 1)
        self.assertTrue(audit["transition_counts_must_not_cross_exclusions"])

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

    def test_step_command_is_forward_filled_without_inventing_a_level(self):
        raw = pd.DataFrame(
            {
                "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:00:02"],
                "fc_voltage_v": [100.0, 300.0],
                "fc_current_a": [10.0, 30.0],
                "target_power_kw": [10.0, 30.0],
            }
        )
        canonical, audit = canonicalize_power_packets(
            raw, step_columns=("target_power_kw",)
        )

        self.assertEqual(canonical.target_power_kw.tolist(), [10.0, 10.0, 30.0])
        self.assertTrue(canonical.target_power_kw_forward_filled.iloc[1])
        self.assertEqual(audit["forward_filled_step_rows"]["target_power_kw"], 1)

    def test_target_events_exclude_zero_tail_and_cannot_start_on_interpolation(self):
        canonical = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01", periods=9, freq="s"),
                "model_segment_id": 0,
                "target_power_kw": [0, 10, 10, 10, 10, 10, 0, 0, 0],
                "fc_input_power_kw": [0, 5, 10, 10, 10, 8, 8, 4, 0],
                "interpolated_power": [False, True, False, True, False, True, False, True, False],
                "target_power_kw_forward_filled": [False, True, False, True, False, True, False, True, False],
            }
        )

        events, audit = extract_target_events(canonical, "2026-03")

        self.assertEqual(len(events), 1)
        self.assertEqual(events.start_timestamp.iloc[0], canonical.timestamp.iloc[2])
        self.assertEqual(events.end_timestamp.iloc[0], canonical.timestamp.iloc[4])
        self.assertEqual(events.dwell_time_s.iloc[0], 3)
        self.assertFalse(events.start_interpolated.iloc[0])
        self.assertFalse(events.left_censored.iloc[0])
        self.assertFalse(events.right_censored.iloc[0])
        self.assertTrue(events.complete_dwell.iloc[0])
        self.assertEqual(audit["interpolated_event_starts"], 0)
        self.assertEqual(audit["zero_target_positive_power_rows_excluded"], 2)

    def test_target_event_at_telemetry_edges_is_censored(self):
        canonical = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-01", periods=3, freq="s"),
                "model_segment_id": 0,
                "target_power_kw": [10.0] * 3,
                "fc_input_power_kw": [8.0] * 3,
                "interpolated_power": False,
                "target_power_kw_forward_filled": False,
            }
        )

        events, audit = extract_target_events(canonical, "2026-03")

        self.assertTrue(events.left_censored.iloc[0])
        self.assertTrue(events.right_censored.iloc[0])
        self.assertFalse(events.complete_dwell.iloc[0])
        self.assertEqual(audit["complete_dwell_events"], 0)

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
