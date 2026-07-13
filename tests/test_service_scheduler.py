import unittest

import numpy as np

from fc_power.evaluation.service_scheduler import (
    ServiceExposure,
    ServiceScheduleConfig,
    ServiceScheduleState,
    candidate_assignments,
    choose_service_assignment,
    stationary_service_exposure,
    transition_service_epoch,
)


class ServiceSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.exposure = ServiceExposure(
            duration_h=1.0,
            continuous_mean_pct=(0.0020, 0.0022),
            load_shift_damage_pct=(0.0002, 0.0002),
        )
        self.config = ServiceScheduleConfig(
            health_limit_pct=10.0,
            gamma_scale_pct=0.1,
            heterogeneity_factors=(1.0, 1.05, 1.10),
            start_damage_pct=0.002,
            risk_horizon_h=100.0,
            risk_samples=128,
        )

    def test_candidate_assignments_map_two_roles(self):
        assignments = candidate_assignments(3)
        self.assertEqual(len(assignments), 6)
        self.assertTrue(all(len(set(value)) == 2 for value in assignments))

    def test_expected_scheduler_rests_most_damaged_stack(self):
        state = ServiceScheduleState((1.0, 4.0, 8.0))
        decision = choose_service_assignment(
            state, self.exposure, self.config, objective="expected_max"
        )
        self.assertNotIn(2, decision.assignment)
        self.assertEqual(decision.new_starts, 2)

    def test_stationary_exposure_uses_template_mean_only(self):
        second = ServiceExposure(
            duration_h=1.0,
            continuous_mean_pct=(0.0040, 0.0042),
            load_shift_damage_pct=(0.0004, 0.0006),
        )
        exposure = stationary_service_exposure([self.exposure, second], 2.0)
        self.assertTrue(
            np.allclose(exposure.continuous_mean_pct, (0.0060, 0.0064))
        )
        self.assertTrue(
            np.allclose(exposure.load_shift_damage_pct, (0.0006, 0.0008))
        )

    def test_gamma_cvar_scheduler_is_deterministic(self):
        state = ServiceScheduleState((1.0, 4.0, 8.0))
        first = choose_service_assignment(state, self.exposure, self.config)
        second = choose_service_assignment(state, self.exposure, self.config)
        self.assertEqual(first, second)
        self.assertIsNotNone(first.cvar_max_health_fraction)

    def test_transition_updates_only_online_stacks_and_counts_real_starts(self):
        state = ServiceScheduleState((1.0, 4.0, 8.0))
        first = transition_service_epoch(
            state,
            self.exposure,
            self.config,
            (0, 1),
            stochastic=False,
        )
        self.assertGreater(first.state.damage_pct[0], state.damage_pct[0])
        self.assertGreater(first.state.damage_pct[1], state.damage_pct[1])
        self.assertEqual(first.state.damage_pct[2], state.damage_pct[2])
        self.assertEqual(first.state.start_count, 2)

        second = transition_service_epoch(
            first.state,
            self.exposure,
            self.config,
            (1, 0),
            stochastic=False,
        )
        self.assertEqual(second.state.start_count, 2)
        self.assertTrue(np.allclose(second.start_damage_pct, 0.0))

    def test_seeded_stochastic_transition_is_reproducible(self):
        state = ServiceScheduleState((1.0, 4.0, 8.0))
        first = transition_service_epoch(
            state,
            self.exposure,
            self.config,
            (0, 1),
            rng=np.random.default_rng(44),
        )
        second = transition_service_epoch(
            state,
            self.exposure,
            self.config,
            (0, 1),
            rng=np.random.default_rng(44),
        )
        self.assertEqual(first, second)

    def test_explicit_uniforms_couple_gamma_by_online_role(self):
        state = ServiceScheduleState((1.0, 4.0, 8.0))
        first = transition_service_epoch(
            state,
            self.exposure,
            self.config,
            (0, 1),
            continuous_uniforms=(0.25, 0.75),
        )
        second = transition_service_epoch(
            state,
            self.exposure,
            self.config,
            (1, 0),
            continuous_uniforms=(0.25, 0.75),
        )
        self.assertNotEqual(
            first.continuous_damage_pct, second.continuous_damage_pct
        )
        with self.assertRaisesRegex(ValueError, "strictly"):
            transition_service_epoch(
                state,
                self.exposure,
                self.config,
                (0, 1),
                continuous_uniforms=(0.0, 0.5),
            )


if __name__ == "__main__":
    unittest.main()
