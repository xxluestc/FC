import unittest
from pathlib import Path

from fc_power.health import (
    DegradationObservation,
    GaussianDegradationObserver,
    GammaHealthState,
    HealthBelief,
)
from fc_power.world_model import (
    MultiStackAction,
    ObservedHealthExecutionLoop,
    WorldModelConfig,
    load_lzw_multistack_world_model,
)


ROOT = Path(__file__).resolve().parents[1]


class GaussianDegradationObserverTest(unittest.TestCase):
    def setUp(self):
        self.observer = GaussianDegradationObserver(
            gamma_scale_pct=0.2,
            initial_variance_pct2=0.04,
            process_variance_rate_pct2_per_hour=0.01,
        )

    def test_predict_propagates_only_gamma_and_model_variance(self):
        prior = self.observer.initialize(GammaHealthState(degradation=1.0))
        predicted_state = GammaHealthState(
            degradation=1.5,
            current_a=100.0,
            is_on=True,
            elapsed_s=3600.0,
        )
        prediction = self.observer.predict(
            prior,
            predicted_state,
            expected_gamma_increment_pct=0.3,
        )
        self.assertAlmostEqual(prediction.variance_pct2, 0.04 + 0.3 * 0.2 + 0.01)
        self.assertEqual(prediction.state, predicted_state)
        self.assertEqual(prior.variance_pct2, 0.04)

    def test_correct_is_time_aligned_and_audited(self):
        prediction = HealthBelief(
            GammaHealthState(degradation=2.0, elapsed_s=10.0),
            variance_pct2=0.09,
        )
        observation = DegradationObservation(
            degradation_pct=2.4,
            variance_pct2=0.01,
            elapsed_s=10.0,
            source="synthetic-unit-test",
            synthetic=True,
        )
        posterior, audit = self.observer.correct(
            prediction,
            observation,
            monotonic_lower_bound_pct=1.8,
        )
        self.assertAlmostEqual(audit.kalman_gain, 0.9)
        self.assertAlmostEqual(posterior.state.degradation, 2.36)
        self.assertAlmostEqual(posterior.variance_pct2, 0.009)
        self.assertEqual(posterior.correction_count, 1)
        self.assertEqual(posterior.last_observation_source, "synthetic-unit-test")
        self.assertFalse(audit.monotonic_projection_applied)

        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.observer.correct(
                prediction,
                DegradationObservation(2.4, 0.01, 11.0, "misaligned", True),
                monotonic_lower_bound_pct=1.8,
            )

    def test_correction_cannot_reverse_cumulative_degradation(self):
        prediction = HealthBelief(
            GammaHealthState(degradation=1.2, elapsed_s=1.0),
            variance_pct2=1.0,
        )
        observation = DegradationObservation(
            degradation_pct=0.0,
            variance_pct2=0.01,
            elapsed_s=1.0,
            source="synthetic-low-reading",
            synthetic=True,
        )
        posterior, audit = self.observer.correct(
            prediction,
            observation,
            monotonic_lower_bound_pct=1.0,
        )
        self.assertEqual(posterior.state.degradation, 1.0)
        self.assertTrue(audit.monotonic_projection_applied)

    def test_repeated_predict_has_no_hidden_memory(self):
        prior = self.observer.initialize(GammaHealthState())
        predicted = GammaHealthState(degradation=0.1, elapsed_s=1.0)
        first = self.observer.predict(
            prior, predicted, expected_gamma_increment_pct=0.02
        )
        second = self.observer.predict(
            prior, predicted, expected_gamma_increment_pct=0.02
        )
        self.assertEqual(first, second)


class ObservedHealthExecutionLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = WorldModelConfig(
            min_online_stacks=2,
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=5.5,
        )
        cls.model = load_lzw_multistack_world_model(
            ROOT, n_stacks=3, config=config
        )
        observers = tuple(
            GaussianDegradationObserver(
                gamma_scale_pct=health.params.gamma_scale,
                initial_variance_pct2=0.01,
            )
            for health in cls.model.health_models
        )
        cls.loop = ObservedHealthExecutionLoop(cls.model, observers)

    def test_observation_is_applied_after_executed_prediction(self):
        initial_model_state = self.model.initial_state()
        state = self.loop.initialize(initial_model_state)
        action = MultiStackAction.from_currents([90.0, 195.0, 0.0])
        unobserved = self.model.step(initial_model_state, action, 36.0)
        predicted_damage = unobserved.next_state.stacks[0].health.degradation
        observation = DegradationObservation(
            degradation_pct=predicted_damage + 0.2,
            variance_pct2=1e-4,
            elapsed_s=1.0,
            source="synthetic-stack-0",
            synthetic=True,
        )
        executed = self.loop.execute(
            state,
            action,
            36.0,
            observations=(observation, None, None),
        )

        self.assertEqual(executed.prediction, unobserved)
        self.assertEqual(
            executed.prediction.next_state.stacks[0].health.degradation,
            predicted_damage,
        )
        self.assertGreater(
            executed.next_state.model_state.stacks[0].health.degradation,
            predicted_damage,
        )
        self.assertEqual(
            executed.next_state.model_state.stacks[1].health,
            executed.prediction.next_state.stacks[1].health,
        )
        self.assertEqual(executed.next_state.beliefs[0].correction_count, 1)
        self.assertEqual(executed.next_state.beliefs[1].correction_count, 0)

    def test_no_observation_matches_model_prediction(self):
        initial = self.model.initial_state()
        state = self.loop.initialize(initial)
        action = MultiStackAction.from_currents([90.0, 195.0, 0.0])
        executed = self.loop.execute(state, action, 36.0)
        self.assertEqual(executed.next_state.model_state, executed.prediction.next_state)
        self.assertTrue(all(item.update.correction is None for item in executed.observers))

    def test_candidate_rollout_does_not_consume_observation(self):
        initial = self.model.initial_state()
        action = MultiStackAction.from_currents([90.0, 195.0, 0.0])
        before = self.model.step(initial, action, 36.0)
        state = self.loop.initialize(initial)
        self.loop.execute(
            state,
            action,
            36.0,
            observations=(
                DegradationObservation(0.1, 0.01, 1.0, "synthetic", True),
                None,
                None,
            ),
        )
        after = self.model.step(initial, action, 36.0)
        self.assertEqual(before, after)

    def test_invalid_observation_count_is_rejected(self):
        state = self.loop.initialize(self.model.initial_state())
        action = MultiStackAction.from_currents([90.0, 195.0, 0.0])
        with self.assertRaisesRegex(ValueError, "stack count"):
            self.loop.execute(state, action, 36.0, observations=(None,))


if __name__ == "__main__":
    unittest.main()
