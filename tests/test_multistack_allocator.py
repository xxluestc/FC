import unittest
from pathlib import Path

from fc_power.power_allocation.multistack_allocator import (
    choose_beam,
    choose_instant,
    enumerate_actions,
    project_to_feasible,
)
from fc_power.world_model import MultiStackAction, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


class MultiStackAllocatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_lzw_multistack_world_model(ROOT, n_stacks=2)

    def test_action_grid_distinguishes_idle_and_off(self):
        state = self.model.initial_state()
        actions = list(enumerate_actions(self.model, state))
        per_stack = len(self.model.config.allowed_currents_a) + 1
        self.assertEqual(len(actions), per_stack**2)
        self.assertIn(MultiStackAction((0.0, 0.0), (False, True)), actions)

    def test_instant_controller_returns_feasible_action(self):
        state = self.model.initial_state()
        result = choose_instant(self.model, state, demand_power_kw=55.0)
        self.assertTrue(result.step.constraints.feasible)
        self.assertGreater(result.feasible_candidates, 0)
        self.assertAlmostEqual(
            result.step.constraints.stack_power_kw
            + result.step.constraints.battery_power_kw,
            55.0,
        )

    def test_health_aware_cost_prefers_healthier_stack(self):
        reference = self.model.performance_proxies[0].mapping.damage_reference_pct
        state = self.model.initial_state(degradation_pct=[0.0, reference])
        result = choose_instant(self.model, state, demand_power_kw=24.0)
        self.assertGreaterEqual(result.action.current_a[0], result.action.current_a[1])

    def test_beam_planner_rolls_health_and_soc_forward(self):
        state = self.model.initial_state()
        result = choose_beam(
            self.model,
            state,
            demand_preview_kw=[45.0, 60.0, 35.0],
            beam_width=8,
        )
        self.assertTrue(result.step.constraints.feasible)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertLessEqual(result.feasible_candidates, result.expanded_nodes)
        self.assertEqual(result.step.next_state.elapsed_s, 1.0)

    def test_safety_projection_snaps_to_feasible_grid(self):
        state = self.model.initial_state()
        requested = MultiStackAction.from_currents([200.0, 200.0])
        result = project_to_feasible(
            self.model, state, requested, demand_power_kw=50.0
        )
        self.assertTrue(result.step.constraints.feasible)
        self.assertIn(result.action.current_a[0], self.model.config.allowed_currents_a)
        self.assertIn(result.action.current_a[1], self.model.config.allowed_currents_a)


if __name__ == "__main__":
    unittest.main()
