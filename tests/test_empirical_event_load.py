import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fc_power.evaluation.empirical_event_load import (
    fit_empirical_event_load,
    fit_empirical_event_table,
    generate_empirical_event_load,
)
from fc_power.evaluation import zuo_load_calibration


def _frame_from_events(values, *, start="2026-01-01", block_id="block-a"):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(values), freq="s"),
            "block_id": block_id,
            "fc_input_power_kw": values,
            "target_power_kw": values,
            "interpolated_power": False,
        }
    )


def _fit_long_dwell_model():
    values = np.r_[
        np.full(240, 10.0),
        np.full(240, 50.0),
        np.full(240, 10.0),
        np.full(240, 50.0),
    ]
    return fit_empirical_event_load(
        _frame_from_events(values),
        candidate_states=(2, 3),
        min_state_occupancy=0.1,
        min_events_per_state=2,
    )


def _compressed_event_table():
    return pd.DataFrame(
        {
            "segment_id": ["a"] * 5 + ["b"] * 6,
            "event_index": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
            "dwell_time_s": [2, 3, 4, 2, 2, 5, 1, 2, 2, 2, 2],
            "target_power_kw": [10, 10, 50, 10, 50, 50, 50, 10, 50, 10, 50],
            "fc_input_power_kw": [8, 12, 48, 11, 52, 49, 55, 9, 51, 10, 50],
        }
    )


class EmpiricalEventLoadTest(unittest.TestCase):
    def test_compressed_table_uses_dwell_weight_for_occupancy_and_power_centers(self):
        events = pd.DataFrame(
            {
                "segment_id": ["archive-a"] * 4,
                "event_index": [0, 1, 2, 3],
                "dwell_time_s": [180, 10, 20, 10],
                "target_power_kw": [8, 48, 12, 52],
                "fc_input_power_kw": [10, 50, 30, 70],
            }
        )
        model = fit_empirical_event_table(
            events,
            candidate_states=(2,),
            min_state_occupancy=0.05,
            min_events_per_state=2,
            random_state=7,
        )

        np.testing.assert_allclose(model.state_signal_centers_kw, [8.4, 50.0])
        np.testing.assert_allclose(model.state_centers_kw, [12.0, 60.0])
        np.testing.assert_allclose(
            model.statistics["state_dwell_occupancy"],
            [200 / 220, 20 / 220],
        )
        sampling = model.statistics["weighted_gmm_sampling"]
        self.assertEqual(sampling["method"], "systematic_stratified_dwell_proportional")
        self.assertEqual(sampling["maximum_effective_seconds"], 200_000)
        self.assertEqual(sampling["actual_sample_count"], 220)
        self.assertEqual(sampling["random_state"], 7)

    def test_compressed_table_merges_adjacent_states_without_crossing_segments(self):
        source = _compressed_event_table()
        model = fit_empirical_event_table(
            source,
            candidate_states=(2,),
            min_state_occupancy=0.1,
            min_events_per_state=2,
            random_state=11,
        )

        self.assertEqual(len(model.event_table), 9)
        self.assertEqual(
            model.event_table.groupby("source_segment_id", sort=False).size().tolist(),
            [4, 5],
        )
        self.assertEqual(model.event_table.raw_event_count.iloc[0], 2)
        self.assertEqual(model.event_table.raw_event_count.iloc[4], 2)
        self.assertEqual(model.event_table.dwell_time_s.iloc[0], 5.0)
        self.assertEqual(model.event_table.dwell_time_s.iloc[4], 6.0)
        self.assertEqual(model.transition_counts.sum(), 7)
        self.assertEqual(
            model.event_table.dwell_time_s.sum(), source.dwell_time_s.sum()
        )

    def test_compressed_table_fit_and_generation_are_seed_reproducible(self):
        source = _compressed_event_table()
        kwargs = {
            "candidate_states": (2,),
            "min_state_occupancy": 0.1,
            "min_events_per_state": 2,
            "max_fit_samples": 10,
            "random_state": 29,
        }
        first = fit_empirical_event_table(source, **kwargs)
        second = fit_empirical_event_table(source, **kwargs)

        np.testing.assert_array_equal(first.state_centers_kw, second.state_centers_kw)
        np.testing.assert_array_equal(first.transition_counts, second.transition_counts)
        pd.testing.assert_frame_equal(first.event_table, second.event_table)
        self.assertEqual(first.audit_statistics(), second.audit_statistics())
        self.assertEqual(
            first.statistics["weighted_gmm_sampling"]["actual_sample_count"], 10
        )
        self.assertTrue(first.statistics["weighted_gmm_sampling"]["capped"])
        json.dumps(first.audit_statistics())

        generated_first = generate_empirical_event_load(
            first, length_s=60, seed=101
        )
        generated_second = generate_empirical_event_load(
            second, length_s=60, seed=101
        )
        pd.testing.assert_frame_equal(generated_first, generated_second)
        self.assertEqual(len(generated_first), 60)

    def test_censored_dwells_are_excluded_and_complete_power_dwell_pairs_stay_coupled(self):
        source = pd.DataFrame(
            {
                "segment_id": ["a"] * 8,
                "event_index": list(range(8)),
                "dwell_time_s": [999, 11, 12, 13, 14, 15, 16, 888],
                "target_power_kw": [10, 50, 10, 50, 10, 50, 10, 50],
                "fc_input_power_kw": [99, 21, 22, 23, 24, 25, 26, 88],
                "left_censored": [True, False, False, False, False, False, False, False],
                "right_censored": [False, False, False, False, False, False, False, True],
            }
        )
        model = fit_empirical_event_table(
            source,
            candidate_states=(2,),
            min_state_occupancy=0.01,
            min_events_per_state=3,
            min_complete_events_per_state=3,
            left_censored_column="left_censored",
            right_censored_column="right_censored",
            random_state=13,
        )

        self.assertNotIn(999.0, np.concatenate(model.dwell_times_s))
        self.assertNotIn(888.0, np.concatenate(model.dwell_times_s))
        self.assertEqual(model.statistics["state_complete_event_counts"], [3, 3])
        generated = generate_empirical_event_load(model, length_s=120, seed=3)
        expected_pairs = {
            11.0: 21.0,
            12.0: 22.0,
            13.0: 23.0,
            14.0: 24.0,
            15.0: 25.0,
            16.0: 26.0,
        }
        starts = generated.loc[generated.event_boundary]
        for row in starts.itertuples(index=False):
            self.assertEqual(
                row.demand_power_kw,
                expected_pairs[row.sampled_empirical_dwell_s],
            )

        with self.assertRaisesRegex(ValueError, "complete merged"):
            fit_empirical_event_table(
                source,
                candidate_states=(2,),
                min_state_occupancy=0.01,
                min_events_per_state=3,
                min_complete_events_per_state=4,
                left_censored_column="left_censored",
                right_censored_column="right_censored",
            )

    def test_compressed_table_candidate_gates_use_merged_events_and_outgoing_counts(self):
        source = pd.DataFrame(
            {
                "segment_id": ["a"] * 7,
                "event_index": list(range(7)),
                "dwell_time_s": [10, 10, 10, 10, 10, 10, 1],
                "target_power_kw": [10, 50, 10, 50, 10, 50, 90],
                "fc_input_power_kw": [10, 50, 10, 50, 10, 50, 90],
            }
        )
        model = fit_empirical_event_table(
            source,
            candidate_states=(2, 3),
            min_state_occupancy=0.01,
            min_events_per_state=2,
            min_outgoing_transitions_per_state=1,
            random_state=4,
        )

        self.assertEqual(model.n_states, 2)
        rejected = model.statistics["candidate_models"][1]
        self.assertFalse(rejected["eligible"])
        self.assertIn(
            "merged state event count below minimum", rejected["rejection_reasons"]
        )
        self.assertIn(
            "state outgoing event transitions below minimum",
            rejected["rejection_reasons"],
        )
        self.assertEqual(rejected["state_event_counts"], [3, 3, 1])
        self.assertEqual(rejected["state_outgoing_event_counts"], [3, 3, 0])

    def test_compressed_table_rejects_invalid_dwell_segments_and_order(self):
        valid = _compressed_event_table()
        invalid_dwell = valid.copy()
        invalid_dwell.loc[0, "dwell_time_s"] = 0
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            fit_empirical_event_table(invalid_dwell, candidate_states=(2,))

        empty_segment = valid.copy()
        empty_segment.loc[0, "segment_id"] = " "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fit_empirical_event_table(empty_segment, candidate_states=(2,))

        noncontiguous = valid.iloc[[0, 5, 1]].copy()
        with self.assertRaisesRegex(ValueError, "each segment contiguous"):
            fit_empirical_event_table(noncontiguous, candidate_states=(2,))

        unordered = valid.iloc[:5].copy()
        unordered["event_index"] = [0, 2, 1, 3, 4]
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            fit_empirical_event_table(unordered, candidate_states=(2,))

    def test_state_fit_is_vehicle_derived_and_not_fixed_to_zuo_levels(self):
        values = np.tile(np.r_[np.full(20, 11.0), np.full(20, 47.0)], 4)
        with mock.patch.object(
            zuo_load_calibration,
            "ZUO_LOAD_LEVELS_KW",
            (1000.0, 2000.0, 3000.0, 4000.0),
        ):
            model = fit_empirical_event_load(
                _frame_from_events(values),
                candidate_states=(2, 3),
                min_state_occupancy=0.1,
                min_events_per_state=3,
            )

        self.assertEqual(model.n_states, 2)
        np.testing.assert_allclose(model.state_centers_kw, [11.0, 47.0], atol=0.1)
        rejected_k3 = model.statistics["candidate_models"][1]
        self.assertFalse(rejected_k3["eligible"])
        self.assertIn("fewer distinct fit values", rejected_k3["rejection_reasons"][0])

    def test_exclusion_is_a_hard_break_for_event_transitions(self):
        values = np.asarray(
            [10, 10, 50, 50, 10, 10, 50, 50, 30, 30, 10, 10, 50, 50, 10, 10, 50, 50],
            dtype=float,
        )
        frame = _frame_from_events(values)
        rules = {
            "intervals": [
                {
                    "start_inclusive": "2026-01-01T00:00:08",
                    "end_exclusive": "2026-01-01T00:00:10",
                    "reason": "test exclusion",
                    "preserve_segment_break": True,
                }
            ],
            "transition_counts_must_not_cross_exclusions": True,
        }
        model = fit_empirical_event_load(
            frame,
            candidate_states=(2,),
            min_state_occupancy=0.1,
            min_events_per_state=2,
            exclusion_rules=rules,
        )

        self.assertEqual(model.statistics["exclusions"]["excluded_rows"], 2)
        self.assertEqual(model.statistics["model_segment_count"], 2)
        np.testing.assert_array_equal(model.transition_counts, [[0, 4], [2, 0]])
        self.assertEqual(model.transition_counts.sum(), 6)

    def test_interpolated_power_cannot_start_an_event(self):
        target = np.asarray([10, 10, 10, 50, 50, 50, 10, 10, 50, 50, 10, 10])
        frame = _frame_from_events(target)
        frame.loc[3, "interpolated_power"] = True
        frame.loc[3, "fc_input_power_kw"] = 30.0
        model = fit_empirical_event_load(
            frame,
            candidate_states=(2,),
            min_state_occupancy=0.1,
            min_events_per_state=2,
        )

        interpolated_timestamp = frame.timestamp.iloc[3]
        self.assertNotIn(
            interpolated_timestamp,
            model.event_table.start_timestamp.iloc[1:].tolist(),
        )
        self.assertEqual(model.event_table.start_timestamp.iloc[1], frame.timestamp.iloc[4])
        self.assertEqual(
            model.statistics["candidate_models"][0]["interpolated_event_boundaries"],
            0,
        )

    def test_empirical_mode_preserves_long_dwell_statistics_and_seed(self):
        model = _fit_long_dwell_model()
        first = generate_empirical_event_load(model, length_s=180, seed=17)
        second = generate_empirical_event_load(model, length_s=180, seed=17)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(int(first.event_boundary.sum()), 1)
        self.assertFalse(first.engineering_stress_transform.any())
        self.assertTrue((first.dwell_time_scale == 1.0).all())
        self.assertEqual(model.statistics["event_count"], len(model.event_table))
        self.assertEqual(model.statistics["event_transition_count"], 3)
        self.assertEqual(model.statistics["dwell_time_s"]["overall"]["median"], 240.0)
        self.assertEqual(int(np.trace(model.transition_counts)), 0)

    def test_stress_mode_explicitly_scales_only_dwell_and_has_multiple_jumps(self):
        model = _fit_long_dwell_model()
        stress = generate_empirical_event_load(
            model,
            length_s=180,
            seed=17,
            mode="stress",
            dwell_time_scale=0.05,
        )

        self.assertGreater(int(stress.event_boundary.sum()), 2)
        self.assertTrue(stress.engineering_stress_transform.all())
        self.assertTrue((stress.dwell_time_scale == 0.05).all())
        self.assertEqual(stress.generation_mode.unique().tolist(), ["stress"])
        self.assertTrue(
            stress.sampled_empirical_dwell_s.isin(
                np.concatenate(model.dwell_times_s)
            ).all()
        )
        self.assertTrue(
            stress.demand_power_kw.isin(
                np.concatenate(model.event_power_samples_kw)
            ).all()
        )
        self.assertTrue(
            stress.attrs["generation_audit"]["engineering_stress_transform"]
        )
        self.assertEqual(stress.attrs["generation_audit"]["dwell_time_scale"], 0.05)
        with self.assertRaises(ValueError):
            generate_empirical_event_load(model, length_s=180, seed=17, mode="stress")
        with self.assertRaises(ValueError):
            generate_empirical_event_load(
                model,
                length_s=180,
                seed=17,
                mode="empirical",
                dwell_time_scale=0.5,
            )


if __name__ == "__main__":
    unittest.main()
