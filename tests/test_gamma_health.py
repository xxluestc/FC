import unittest

import numpy as np

from fc_power.health.gamma_process import (
    GammaHealthModel,
    GammaHealthParams,
    GammaHealthState,
    LoadRateMap,
)


def model(**overrides):
    values = {
        "load_rate_map": LoadRateMap(
            current_a=(0.0, 100.0, 200.0, 400.0),
            mean_rate_per_hour=(0.0, 1.0, 2.0, 6.0),
        ),
        "gamma_scale": 0.01,
        "natural_rate_per_hour": 0.1,
        "off_rate_per_hour": 0.0,
        "ramp_increment_per_amp": 0.001,
        "shift_increment": 0.05,
        "shift_threshold_a": 50.0,
        "start_increment": 0.2,
        "stop_increment": 0.1,
        "failure_threshold": 100.0,
    }
    values.update(overrides)
    return GammaHealthModel(GammaHealthParams(**values))


class GammaHealthModelTest(unittest.TestCase):
    def test_deterministic_transition_is_auditable_and_monotone(self):
        result = model().transition(
            GammaHealthState(), current_a=100.0, dt_s=3600.0, stochastic=False
        )
        self.assertAlmostEqual(result.expected_load_increment, 1.0)
        self.assertAlmostEqual(result.load_increment, 1.0)
        self.assertAlmostEqual(result.natural_increment, 0.1)
        self.assertAlmostEqual(result.ramp_increment, 0.1)
        self.assertAlmostEqual(result.start_stop_increment, 0.2)
        self.assertAlmostEqual(result.shift_increment, 0.05)
        self.assertAlmostEqual(result.total_increment, 1.45)
        self.assertAlmostEqual(result.state.degradation, 1.45)
        self.assertTrue(result.state.is_on)
        self.assertEqual(result.state.start_count, 1)

    def test_high_current_has_higher_expected_increment(self):
        health = model(ramp_increment_per_amp=0.0, start_increment=0.0)
        low = health.expected_load_increment(100.0, 3600.0)
        high = health.expected_load_increment(400.0, 3600.0)
        self.assertGreater(high, low)

    def test_same_seed_reproduces_gamma_increment(self):
        health = model(ramp_increment_per_amp=0.0, start_increment=0.0)
        first = health.transition(
            GammaHealthState(),
            200.0,
            60.0,
            stochastic=True,
            rng=np.random.default_rng(2026),
        )
        second = health.transition(
            GammaHealthState(),
            200.0,
            60.0,
            stochastic=True,
            rng=np.random.default_rng(2026),
        )
        self.assertEqual(first.load_increment, second.load_increment)
        self.assertGreaterEqual(first.total_increment, 0.0)

    def test_stop_event_does_not_reverse_degradation(self):
        health = model(ramp_increment_per_amp=0.0)
        initial = GammaHealthState(degradation=3.0, current_a=100.0, is_on=True)
        result = health.transition(initial, 0.0, dt_s=1.0, stochastic=False)
        self.assertGreaterEqual(result.state.degradation, initial.degradation)
        self.assertAlmostEqual(result.start_stop_increment, 0.1)
        self.assertFalse(result.state.is_on)
        self.assertEqual(result.state.stop_count, 1)

    def test_zero_current_can_represent_idle_or_fully_stopped(self):
        health = model(
            off_rate_per_hour=0.0,
            ramp_increment_per_amp=0.0,
            shift_increment=0.0,
            stop_increment=0.0,
        )
        stopped = health.transition(
            GammaHealthState(), 0.0, 3600.0, stochastic=False, next_on=False
        )
        idle = health.transition(
            GammaHealthState(is_on=True),
            0.0,
            3600.0,
            stochastic=False,
            next_on=True,
        )
        self.assertAlmostEqual(stopped.total_increment, 0.0)
        self.assertAlmostEqual(idle.total_increment, 0.1)

    def test_explicit_shift_event_can_override_current_threshold(self):
        health = model(
            ramp_increment_per_amp=0.0,
            start_increment=0.0,
            natural_rate_per_hour=0.0,
        )
        state = GammaHealthState(current_a=100.0, is_on=True)
        no_shift = health.transition(
            state, 200.0, stochastic=False, shift_event=False
        )
        shift = health.transition(state, 200.0, stochastic=False, shift_event=True)
        self.assertAlmostEqual(no_shift.shift_increment, 0.0)
        self.assertAlmostEqual(shift.shift_increment, 0.05)

    def test_stack_heterogeneity_scales_expected_damage(self):
        nominal = model(
            heterogeneity_factor=1.0,
            ramp_increment_per_amp=0.0,
            start_increment=0.0,
            natural_rate_per_hour=0.0,
        )
        aged = model(
            heterogeneity_factor=1.5,
            ramp_increment_per_amp=0.0,
            start_increment=0.0,
            natural_rate_per_hour=0.0,
        )
        self.assertAlmostEqual(
            aged.expected_load_increment(200.0, 3600.0),
            1.5 * nominal.expected_load_increment(200.0, 3600.0),
        )

    def test_soh_requires_an_explicit_failure_threshold(self):
        state = GammaHealthState(degradation=25.0)
        self.assertAlmostEqual(model().soh(state), 0.75)
        self.assertIsNone(model(failure_threshold=None).soh(state))

    def test_invalid_rate_map_is_rejected(self):
        with self.assertRaises(ValueError):
            LoadRateMap(current_a=(0.0, 100.0, 90.0), mean_rate_per_hour=(0, 1, 2))


if __name__ == "__main__":
    unittest.main()
