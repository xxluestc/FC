import unittest

import numpy as np
import pandas as pd

from fc_power.evaluation.load_profiles import (
    EVENT_NAMES,
    SyntheticLoadConfig,
    append_soc_recovery_tail,
    generate_event_load,
    generate_real_block_bootstrap,
)
from fc_power.evaluation.zuo_load_calibration import (
    ZUO_FAST_TRANSITION,
    ZUO_SLOW_TRANSITION,
    estimate_segmented_transitions,
    split_at_largest_segment_gap,
)


class LoadProfilesTest(unittest.TestCase):
    def test_event_load_is_reproducible_bounded_and_event_complete(self):
        config = SyntheticLoadConfig(length_s=180)
        first = generate_event_load(2026, config)
        second = generate_event_load(2026, config)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 180)
        self.assertGreaterEqual(first.demand_power_kw.min(), config.power_min_kw)
        self.assertLessEqual(first.demand_power_kw.max(), config.power_max_kw)
        self.assertEqual(set(first.event), set(EVENT_NAMES))
        self.assertTrue(first.event_boundary.any())

    def test_different_seeds_change_event_load(self):
        config = SyntheticLoadConfig(length_s=120)
        first = generate_event_load(1, config)
        second = generate_event_load(2, config)
        self.assertFalse(np.allclose(first.demand_power_kw, second.demand_power_kw))

    def test_real_block_bootstrap_preserves_real_samples(self):
        source = np.sin(np.arange(500) / 20) * 60
        first = generate_real_block_bootstrap(source, 120, 2026, block_length_s=20)
        second = generate_real_block_bootstrap(source, 120, 2026, block_length_s=20)
        self.assertTrue(first.equals(second))
        np.testing.assert_allclose(
            first.demand_power_kw.to_numpy(),
            source[first.source_index.to_numpy()],
        )
        self.assertEqual(len(first), 120)

    def test_real_block_bootstrap_skips_nan_boundaries(self):
        source = np.arange(200, dtype=float)
        source[50:55] = np.nan
        result = generate_real_block_bootstrap(
            source, 80, 7, block_length_s=20
        )
        self.assertTrue(np.isfinite(result.demand_power_kw).all())
        np.testing.assert_allclose(
            result.demand_power_kw.to_numpy(),
            source[result.source_index.to_numpy()],
        )

    def test_soc_recovery_tail_is_explicit_and_common(self):
        profile = generate_event_load(3, SyntheticLoadConfig(length_s=20))
        extended = append_soc_recovery_tail(profile, 15, 30.0)
        self.assertEqual(len(extended), 35)
        self.assertFalse(extended.is_soc_recovery.iloc[:20].any())
        self.assertTrue(extended.is_soc_recovery.iloc[20:].all())
        self.assertTrue((extended.demand_power_kw.iloc[20:] == 30.0).all())

    def test_temporal_split_uses_complete_segments_around_largest_gap(self):
        frame = pd.DataFrame(
            {
                "segment_id": [0, 0, 1, 1, 2, 2],
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 00:00:01",
                        "2025-01-01 00:00:10",
                        "2025-01-01 00:00:11",
                        "2025-01-03 00:00:00",
                        "2025-01-03 00:00:01",
                    ]
                ),
            }
        )
        split = split_at_largest_segment_gap(frame)
        self.assertEqual(split.calibration_segments, (0, 1))
        self.assertEqual(split.holdout_segments, (2,))
        self.assertGreater(split.gap_seconds, 86_000)

    def test_transition_counts_do_not_bridge_off_or_segment_boundaries(self):
        frame = pd.DataFrame(
            {
                "segment_id": [0, 0, 0, 0, 1, 1],
                "fc_input_power_kw": [16.0, 16.0, 0.0, 40.0, 40.0, 40.0],
            }
        )
        estimate = estimate_segmented_transitions(
            frame,
            normalization_power_kw=40.0,
            stride_s=1,
            bootstrap_samples=50,
            bootstrap_seed=7,
        )
        self.assertEqual(int(estimate.counts.sum()), 2)
        self.assertEqual(estimate.counts[0, 0], 1)
        self.assertEqual(estimate.counts[3, 3], 1)
        self.assertEqual(estimate.counts[0, 3], 0)
        self.assertAlmostEqual(estimate.probabilities[0, 0], 1.0)
        self.assertAlmostEqual(estimate.probabilities[3, 3], 1.0)

    def test_zuo_reference_matrices_are_stochastic(self):
        for matrix in (ZUO_FAST_TRANSITION, ZUO_SLOW_TRANSITION):
            values = np.asarray(matrix)
            self.assertEqual(values.shape, (4, 4))
            self.assertTrue((values >= 0).all())
            np.testing.assert_allclose(values.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
