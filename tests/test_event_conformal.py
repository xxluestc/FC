import unittest

import numpy as np

from fc_power.prediction.event_conformal import EventConditionedResidualConformal


class EventConformalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        n, horizon = 800, 3
        self.base = rng.normal(0, 5, size=(n, horizon))
        self.brake = (np.arange(n) % 4 == 0).astype(float)
        self.high = (np.arange(n) % 7 == 0).astype(float)
        bias = 3 * self.brake[:, None] - 2 * self.high[:, None]
        self.actual = self.base + bias + rng.normal(0, 0.3, size=(n, horizon))

    def test_event_correction_reduces_systematic_bias(self):
        model = EventConditionedResidualConformal(minimum_group_samples=20)
        model.fit(
            self.base[:600],
            self.actual[:600],
            self.brake[:600],
            self.high[:600],
        )
        forecast = model.predict(self.base[600:], self.brake[600:], self.high[600:])
        before = np.mean(np.abs(self.actual[600:] - self.base[600:]))
        after = np.mean(np.abs(self.actual[600:] - forecast.center))
        self.assertLess(after, before)

    def test_interval_shapes_and_order_are_valid(self):
        model = EventConditionedResidualConformal(minimum_group_samples=20).fit(
            self.base[:600],
            self.actual[:600],
            self.brake[:600],
            self.high[:600],
        )
        forecast = model.predict(self.base[600:], self.brake[600:], self.high[600:])
        self.assertEqual(forecast.center.shape, (200, 3))
        self.assertTrue(np.all(forecast.lower <= forecast.center))
        self.assertTrue(np.all(forecast.center <= forecast.upper))
        self.assertEqual(forecast.event_code.shape, (200,))

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            EventConditionedResidualConformal(minimum_group_samples=20).fit(
                self.base[:600],
                self.actual[:600],
                np.full(600, 1.2),
                self.high[:600],
            )

    def test_adaptive_intervals_do_not_use_unobserved_future(self):
        model = EventConditionedResidualConformal(minimum_group_samples=20).fit(
            self.base[:600],
            self.actual[:600],
            self.brake[:600],
            self.high[:600],
        )
        actual_a = self.actual[600:].copy()
        actual_b = actual_a.copy()
        actual_b[100:] += 1000.0
        forecast_a = model.predict_adaptive(
            self.base[600:],
            actual_a,
            self.brake[600:],
            self.high[600:],
            delay_steps=3,
            rolling_window=100,
        )
        forecast_b = model.predict_adaptive(
            self.base[600:],
            actual_b,
            self.brake[600:],
            self.high[600:],
            delay_steps=3,
            rolling_window=100,
        )
        np.testing.assert_allclose(forecast_a.lower[:103], forecast_b.lower[:103])
        np.testing.assert_allclose(forecast_a.upper[:103], forecast_b.upper[:103])


if __name__ == "__main__":
    unittest.main()
