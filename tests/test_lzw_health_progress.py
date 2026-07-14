import json
import unittest

import numpy as np
import pandas as pd

from fc_power.health.lzw_health_progress import (
    DEGRADATION_DIRECTIONS,
    THETA_COLUMNS,
    LzwHealthProgressMap,
    fit_lzw_health_progress,
    validate_lzw_theta_keys,
)


def synthetic_theta(count=301, seed=2026):
    rng = np.random.default_rng(seed)
    latent = np.linspace(0.0, 1.0, count)
    smooth = np.column_stack(
        [
            1.91e-7 - 1.05e-8 * latent**1.25,
            0.0020 + 0.00165 * latent**0.85,
            0.0406 + 0.0228 * latent**1.10,
        ]
    )
    noise = rng.normal(size=smooth.shape) * np.array([1.2e-11, 2.0e-6, 3.0e-5])
    observed = smooth + noise
    observed[count // 3] += np.array([7e-10, -1.2e-4, -8e-4])
    return pd.DataFrame(
        {
            "event_id": [f"LZW-E{index:06d}" for index in range(1, count + 1)],
            "canonical_row_6104": np.arange(1, count + 1),
            "original_index": np.arange(1001, 1001 + count),
            **{
                column: observed[:, index]
                for index, column in enumerate(THETA_COLUMNS)
            },
        }
    )


class LzwHealthProgressTest(unittest.TestCase):
    def setUp(self):
        self.theta = synthetic_theta()

    def test_progress_is_monotone_and_only_terminal_row_is_one(self):
        mapping, h, diagnostics = fit_lzw_health_progress(
            self.theta, endpoint_window=15
        )
        self.assertEqual(h[0], 0.0)
        self.assertEqual(h[-1], 1.0)
        self.assertTrue(np.all(np.diff(h) >= 0.0))
        self.assertTrue(np.all(h[:-1] < 1.0))
        self.assertEqual(diagnostics["progress"]["terminal_value_count"], 1)
        self.assertIsInstance(mapping, LzwHealthProgressMap)

    def test_endpoints_are_window_medians_and_directions_are_explicit(self):
        window = 17
        mapping, _, diagnostics = fit_lzw_health_progress(
            self.theta, endpoint_window=window
        )
        expected_start = self.theta.loc[:, THETA_COLUMNS].iloc[:window].median()
        expected_end = self.theta.loc[:, THETA_COLUMNS].iloc[-window:].median()
        np.testing.assert_allclose(mapping.theta_at(0.0), expected_start.to_numpy())
        np.testing.assert_allclose(mapping.theta_at(1.0), expected_end.to_numpy())
        self.assertEqual(mapping.degradation_directions, DEGRADATION_DIRECTIONS)

        grid_theta = mapping.theta_at(np.linspace(0.0, 1.0, 1001))
        directed_diff = np.diff(grid_theta, axis=0) * np.asarray(
            DEGRADATION_DIRECTIONS
        )
        self.assertTrue(np.all(directed_diff >= -1e-15))
        for column in THETA_COLUMNS:
            component = diagnostics["components"][column]
            self.assertIn("spearman_vs_h", component)
            self.assertIn("start_endpoint", component)
            self.assertIn("reconstruction", component)
            self.assertGreater(
                component["degradation_aligned_spearman_vs_h"], 0.95
            )

    def test_mapping_roundtrips_through_json(self):
        mapping, _, _ = fit_lzw_health_progress(self.theta, endpoint_window=15)
        restored = LzwHealthProgressMap.from_dict(
            json.loads(json.dumps(mapping.to_dict()))
        )
        grid = np.linspace(0.0, 1.0, 51)
        np.testing.assert_array_equal(restored.theta_at(grid), mapping.theta_at(grid))
        np.testing.assert_allclose(
            restored.h_from_theta(restored.theta_at(grid)), grid, atol=2e-2
        )
        self.assertEqual(restored.h_from_theta(restored.theta_end), 1.0)
        beyond_end = np.asarray(restored.theta_end) + np.array([-1e-8, 1.0, 1.0])
        self.assertLess(restored.h_from_theta(beyond_end), 1.0)

    def test_bad_or_misaligned_keys_fail(self):
        reference = self.theta.loc[
            :, ["event_id", "canonical_row_6104", "original_index"]
        ].copy()
        validate_lzw_theta_keys(self.theta, reference)

        incomplete = self.theta.drop(columns="original_index")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fit_lzw_health_progress(incomplete, endpoint_window=15)

        duplicate = self.theta.copy()
        duplicate.loc[1, "event_id"] = duplicate.loc[0, "event_id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            fit_lzw_health_progress(duplicate, endpoint_window=15)

        misaligned = reference.copy()
        misaligned.loc[[20, 21], "event_id"] = misaligned.loc[
            [21, 20], "event_id"
        ].to_numpy()
        with self.assertRaises(ValueError):
            validate_lzw_theta_keys(self.theta, misaligned)

    def test_constant_or_wrong_direction_signal_fails(self):
        constant = self.theta.copy()
        constant[THETA_COLUMNS[0]] = 1.0
        with self.assertRaisesRegex(ValueError, "constant or opposes"):
            fit_lzw_health_progress(constant, endpoint_window=15)

        wrong_direction = self.theta.copy()
        wrong_direction[THETA_COLUMNS[1]] = wrong_direction[
            THETA_COLUMNS[1]
        ].iloc[::-1].to_numpy()
        with self.assertRaisesRegex(ValueError, "constant or opposes"):
            fit_lzw_health_progress(wrong_direction, endpoint_window=15)

    def test_synthetic_trajectory_reconstruction_is_accurate(self):
        mapping, h, diagnostics = fit_lzw_health_progress(
            self.theta, endpoint_window=15
        )
        reconstructed = mapping.theta_at(h)
        spans = np.abs(np.asarray(mapping.theta_end) - np.asarray(mapping.theta_start))
        normalized_rmse = np.sqrt(
            np.mean(np.square(self.theta.loc[:, THETA_COLUMNS] - reconstructed), axis=0)
        ) / spans
        self.assertTrue(np.all(normalized_rmse < 0.045), normalized_rmse)
        for column in THETA_COLUMNS:
            reported = diagnostics["components"][column]["reconstruction"]
            self.assertLess(reported["normalized_rmse"], 0.045)
        self.assertLess(diagnostics["progress"]["roundtrip_rmse"], 0.02)


if __name__ == "__main__":
    unittest.main()
