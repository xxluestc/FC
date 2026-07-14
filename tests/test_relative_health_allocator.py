import unittest

import numpy as np

from fc_power.health.dynamic_proxy import LzwIvConditions
from fc_power.health.lzw_health_progress import LzwHealthProgressMap
from fc_power.power_allocation.relative_health_allocator import (
    RelativeHealthWeights,
    allocate_relative_health_budget,
    build_n_plus_one_action_grid,
    choose_relative_health_action,
    interpolate_lzw_power_table_kw,
    lzw_power_table_kw,
)


def simple_mapping():
    return LzwHealthProgressMap(
        theta_start=(1.90e-7, 0.0020, 0.0406),
        theta_end=(1.80e-7, 0.0037, 0.0634),
        h_knots=(0.0, 1.0),
        theta_knots=(
            (1.90e-7, 0.0020, 0.0406),
            (1.80e-7, 0.0037, 0.0634),
        ),
        endpoint_window=1,
    )


def conditions():
    return LzwIvConditions(
        temperature_c=60.0,
        a=1.1,
        b=0.2,
        concentration_b=0.012,
        limiting_current_a_cm2=1.2,
    )


class RelativeHealthAllocatorTest(unittest.TestCase):
    def test_action_grid_enforces_n_plus_one_limit(self):
        indices, currents = build_n_plus_one_action_grid(
            [0.0, 100.0, 200.0], n_stacks=3, max_online_stacks=2
        )
        self.assertEqual(indices.shape, currents.shape)
        self.assertTrue(np.all(np.count_nonzero(currents, axis=1) <= 2))
        self.assertTrue(np.any(np.all(currents == 0.0, axis=1)))

    def test_aged_stack_has_lower_power_at_same_current(self):
        table = lzw_power_table_kw(
            simple_mapping(), conditions(), [0.0, 1.0], [0.0, 195.0]
        )
        self.assertEqual(table[0, 0], 0.0)
        self.assertGreater(table[0, 1], table[1, 1])

    def test_lookup_interpolation_matches_direct_model(self):
        mapping = simple_mapping()
        condition = conditions()
        grid = np.linspace(0.0, 1.0, 101)
        lookup = lzw_power_table_kw(
            mapping, condition, grid, [0.0, 120.0, 270.0]
        )
        progress = np.asarray([0.123, 0.456, 0.789])
        interpolated = interpolate_lzw_power_table_kw(progress, grid, lookup)
        direct = lzw_power_table_kw(
            mapping, condition, progress, [0.0, 120.0, 270.0]
        )
        np.testing.assert_allclose(interpolated, direct, atol=2e-5, rtol=0.0)

    def test_health_loading_breaks_equal_tracking_tie(self):
        indices, currents = build_n_plus_one_action_grid(
            [0.0, 100.0], n_stacks=3, max_online_stacks=1
        )
        power = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        chosen = choose_relative_health_action(
            action_indices=indices,
            action_currents_a=currents,
            power_table_kw=power,
            decision_health_progress=[0.7, 0.2, 0.5],
            previous_currents_a=[0.0, 0.0, 0.0],
            demand_power_kw=10.0,
            max_online_stacks=1,
            tracking_slack_kw=0.0,
            weights=RelativeHealthWeights(
                tracking=20.0,
                hydrogen=0.0,
                switch=0.0,
                ramp=0.0,
                health_loading=1.0,
            ),
        )
        self.assertEqual(chosen.current_a, (0.0, 100.0, 0.0))

    def test_budget_is_policy_invariant_in_total_and_power_weighted(self):
        next_health, increments = allocate_relative_health_budget(
            [0.2, 0.4, 0.6],
            [30.0, 10.0, 0.0],
            demand_power_kw=40.0,
            dt_s=10.0,
            episode_demand_energy_kwh=1.0,
            fleet_progress_budget=0.09,
        )
        expected_total = 0.09 * (40.0 * 10.0 / 3600.0)
        self.assertAlmostEqual(float(increments.sum()), expected_total)
        self.assertAlmostEqual(increments[0] / increments[1], 3.0)
        np.testing.assert_allclose(next_health, np.asarray([0.2, 0.4, 0.6]) + increments)

    def test_budget_raises_when_lzw_endpoint_would_be_exceeded(self):
        with self.assertRaisesRegex(ValueError, "endpoint"):
            allocate_relative_health_budget(
                [0.99],
                [40.0],
                demand_power_kw=40.0,
                dt_s=3600.0,
                episode_demand_energy_kwh=40.0,
                fleet_progress_budget=0.1,
            )

    def test_budget_raises_for_positive_demand_with_zero_output(self):
        with self.assertRaisesRegex(ValueError, "zero output"):
            allocate_relative_health_budget(
                [0.2, 0.4, 0.6],
                [0.0, 0.0, 0.0],
                demand_power_kw=1.0,
                dt_s=10.0,
                episode_demand_energy_kwh=1.0,
                fleet_progress_budget=0.03,
            )


if __name__ == "__main__":
    unittest.main()
