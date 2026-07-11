import unittest
from pathlib import Path

from fc_power.power_allocation.multistack_baselines import (
    choose_average,
    choose_rotating,
)
from fc_power.world_model import load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


class MultiStackBaselinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_lzw_multistack_world_model(ROOT, n_stacks=2)

    def test_average_uses_equal_currents(self):
        state = self.model.initial_state()
        result = choose_average(self.model, state, demand_power_kw=40.0)
        self.assertTrue(result.step.constraints.feasible)
        self.assertEqual(result.action.current_a[0], result.action.current_a[1])

    def test_rotation_changes_preferred_lead(self):
        state = self.model.initial_state()
        first = choose_rotating(self.model, state, 24.0, lead_stack=0)
        second = choose_rotating(self.model, state, 24.0, lead_stack=1)
        self.assertGreaterEqual(first.action.current_a[0], first.action.current_a[1])
        self.assertGreaterEqual(second.action.current_a[1], second.action.current_a[0])


if __name__ == "__main__":
    unittest.main()
