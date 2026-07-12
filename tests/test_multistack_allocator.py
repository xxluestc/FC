import unittest
from pathlib import Path

from fc_power.power_allocation.multistack_allocator import (
    choose_beam,
    choose_instant,
    enumerate_actions,
    project_to_feasible,
    choose_terminal_soc_recovery,
)
from fc_power.world_model import (
    MultiStackAction,
    WorldModelConfig,
    load_lzw_multistack_world_model,
)


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

    def test_three_stack_lzw_model_keeps_one_stack_off(self):
        model = load_lzw_multistack_world_model(ROOT, n_stacks=3)
        state = model.initial_state()
        actions = list(enumerate_actions(model, state))
        self.assertEqual(model.config.max_online_stacks, 2)
        self.assertTrue(actions)
        self.assertTrue(all(sum(action.is_on) <= 2 for action in actions))

        invalid = MultiStackAction.from_currents([90.0, 90.0, 90.0])
        step = model.step(state, invalid, demand_power_kw=60.0)
        self.assertFalse(step.constraints.feasible)
        self.assertIn("system:max_online_stacks", step.constraints.violations)

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

    def test_fc_only_instant_tracks_without_changing_soc(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state(soc=0.63)
        result = choose_instant(model, state, demand_power_kw=40.0)
        self.assertTrue(result.step.constraints.feasible)
        self.assertLessEqual(
            abs(result.step.constraints.power_balance_error_kw), 5.5
        )
        self.assertEqual(result.step.constraints.battery_power_kw, 0.0)
        self.assertEqual(result.step.next_state.soc, state.soc)
        self.assertLessEqual(sum(result.action.is_on), 2)

    def test_fc_only_beam_ignores_terminal_soc_and_tracks_power(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state(soc=0.55)
        result = choose_beam(
            model,
            state,
            demand_preview_kw=[40.0, 40.0],
            beam_width=4,
            terminal_soc_weight=10_000.0,
        )
        self.assertTrue(result.step.constraints.feasible)
        self.assertLessEqual(
            abs(result.step.constraints.power_balance_error_kw), 5.5
        )
        self.assertEqual(result.step.next_state.soc, state.soc)
        self.assertEqual(result.step.constraints.battery_power_kw, 0.0)

    def test_fc_only_safety_projection_respects_tracking(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        state = model.initial_state()
        requested = MultiStackAction.from_currents([0.0, 0.0, 0.0])
        result = project_to_feasible(model, state, requested, demand_power_kw=40.0)
        self.assertTrue(result.step.constraints.feasible)
        self.assertLessEqual(
            abs(result.step.constraints.power_balance_error_kw), 5.5
        )
        self.assertLessEqual(sum(result.action.is_on), 2)

    def test_fc_only_terminal_soc_recovery_is_rejected(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        with self.assertRaisesRegex(ValueError, "fc_only"):
            choose_terminal_soc_recovery(
                model, model.initial_state(), demand_power_kw=40.0
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

    def test_instant_uses_audited_dwell_override_for_battery_safety(self):
        from fc_power.health.gamma_process import GammaHealthState
        from fc_power.world_model import MultiStackState, StackControlState

        stack = StackControlState(
            GammaHealthState(current_a=90.0, is_on=True), dwell_s=14.0
        )
        state = MultiStackState(soc=0.70, stacks=(stack, stack))
        result = choose_instant(self.model, state, demand_power_kw=-73.0)
        self.assertTrue(result.step.constraints.feasible)
        self.assertTrue(result.step.constraints.safety_overrides)

    def test_beam_uses_override_only_when_hard_safety_needs_it(self):
        from fc_power.health.gamma_process import GammaHealthState
        from fc_power.world_model import MultiStackState, StackControlState

        stack = StackControlState(
            GammaHealthState(current_a=90.0, is_on=True), dwell_s=14.0
        )
        state = MultiStackState(soc=0.70, stacks=(stack, stack))
        result = choose_beam(self.model, state, [-73.0], beam_width=4)
        self.assertTrue(result.step.constraints.feasible)
        self.assertTrue(result.step.constraints.safety_overrides)

    def test_terminal_recovery_moves_soc_toward_reference(self):
        for soc in (0.68, 0.72):
            state = self.model.initial_state(soc=soc)
            result = choose_terminal_soc_recovery(
                self.model, state, demand_power_kw=30.0
            )
            self.assertLess(
                abs(result.step.next_state.soc - 0.70), abs(state.soc - 0.70)
            )


if __name__ == "__main__":
    unittest.main()
