import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.health.dynamic_proxy import (
    DynamicPerformanceLossProxy,
    LzwIvConditions,
)
from fc_power.health.lzw_gamma_calibration import (
    cumulative_damage_components,
    fit_theta_power_law,
)


ROOT = Path(__file__).resolve().parents[1]


class DynamicHealthProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = ROOT / "data/upstream_lzw"
        events = pd.read_csv(upstream / "canonical_event_table_6104.csv")
        theta = pd.read_csv(upstream / "theta_event_trajectory_6104.csv")
        components = cumulative_damage_components(events)
        mapping, _, _ = fit_theta_power_law(components.total_damage_pct, theta)
        conditions = LzwIvConditions.from_upstream_dict(
            json.loads((upstream / "current_point_cost_conditions.json").read_text())
        )
        table = pd.read_csv(upstream / "current_point_degradation_cost_table.csv")
        reference = table.equivalent_stack_power_loss_clipped_W.max()
        cls.table = table
        cls.mapping = mapping
        cls.proxy = DynamicPerformanceLossProxy(mapping, conditions, reference)

    def test_healthy_state_has_zero_proxy(self):
        result = self.proxy.evaluate(0.0, [0, 25, 90, 195, 270, 370])
        np.testing.assert_allclose(result["normalized_proxy"], 0.0, atol=1e-12)

    def test_late_state_reproduces_upstream_table(self):
        expected = self.table[self.table.health_state.eq("late")]
        result = self.proxy.evaluate(
            self.mapping.damage_reference_pct, expected.current_A.to_numpy()
        )
        np.testing.assert_allclose(
            result["current_cell_voltage_v"],
            expected.V_aged_cell_V.to_numpy(),
            atol=3e-4,
        )
        np.testing.assert_allclose(
            result["normalized_proxy"],
            expected.normalized_energy_cost_0_1.to_numpy(),
            atol=8e-3,
        )

    def test_proxy_increases_with_health_damage(self):
        damages = [0.0, 0.25, 0.5, 0.75, 1.0]
        values = [
            self.proxy.evaluate(
                fraction * self.mapping.damage_reference_pct, [195.0]
            )["normalized_proxy"][0]
            for fraction in damages
        ]
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        self.assertGreater(values[-1], values[1])

    def test_high_current_amplifies_same_health_loss(self):
        result = self.proxy.evaluate(
            0.5 * self.mapping.damage_reference_pct, [90.0, 195.0, 370.0]
        )
        self.assertTrue(np.all(np.diff(result["normalized_proxy"]) > 0))


if __name__ == "__main__":
    unittest.main()
