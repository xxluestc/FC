import unittest
from pathlib import Path

from fc_power.power_allocation.multistack_baselines import (
    choose_average,
    choose_daisy_chain_average,
    choose_rotating,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


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

    def test_three_stack_fc_only_average_equalizes_online_stacks(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state(soc=0.62)
        result = choose_average(model, state, demand_power_kw=40.0)
        online_currents = [
            current
            for current, is_on in zip(
                result.action.current_a, result.action.is_on
            )
            if is_on
        ]
        self.assertTrue(result.step.constraints.feasible)
        self.assertEqual(len(online_currents), 2)
        self.assertEqual(len(set(online_currents)), 1)
        self.assertEqual(result.step.constraints.battery_power_kw, 0.0)
        self.assertEqual(result.step.next_state.soc, state.soc)

    def test_three_stack_fc_only_rotating_prefers_lead(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state(soc=0.62)
        result = choose_rotating(model, state, 40.0, lead_stack=1)
        self.assertTrue(result.step.constraints.feasible)
        self.assertGreaterEqual(
            result.action.current_a[1],
            max(result.action.current_a[0], result.action.current_a[2]),
        )
        self.assertEqual(result.step.constraints.battery_power_kw, 0.0)
        self.assertEqual(result.step.next_state.soc, state.soc)

    def test_daisy_chain_uses_prescribed_pair_with_equal_currents(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                min_online_stacks=2,
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state(soc=0.62)
        result = choose_daisy_chain_average(
            model, state, demand_power_kw=40.0, online_assignment=(1, 2)
        )
        self.assertTrue(result.step.constraints.feasible)
        self.assertEqual(result.action.is_on, (False, True, True))
        self.assertEqual(result.action.current_a[1], result.action.current_a[2])
        self.assertEqual(result.action.current_a[0], 0.0)


if __name__ == "__main__":
    unittest.main()
