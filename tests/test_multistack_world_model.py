import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import extract_action_exposure
from fc_power.health.gamma_process import GammaHealthState
from fc_power.world_model import (
    MultiStackAction,
    MultiStackState,
    StackControlState,
    WorldModelConfig,
    load_lzw_multistack_world_model,
)
from fc_power.health.lzw_gamma_calibration import GhaderiPeiCoefficients


ROOT = Path(__file__).resolve().parents[1]


class MultiStackWorldModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_lzw_multistack_world_model(ROOT, n_stacks=2)

    def test_step_closes_power_balance_and_updates_soc(self):
        state = self.model.initial_state(soc=0.70)
        result = self.model.step(
            state,
            MultiStackAction.from_currents([195, 195]),
            demand_power_kw=50.0,
        )
        self.assertTrue(result.constraints.feasible)
        self.assertAlmostEqual(result.constraints.power_balance_error_kw, 0.0)
        self.assertAlmostEqual(
            result.constraints.stack_power_kw
            + result.constraints.battery_power_kw,
            50.0,
        )
        self.assertLess(result.next_state.soc, state.soc)
        self.assertTrue(all(item.degradation_increment_pct > 0 for item in result.stacks))

    def test_fc_only_config_requires_valid_interface_and_tolerance(self):
        with self.assertRaisesRegex(ValueError, "power_interface"):
            WorldModelConfig(power_interface="unknown")
        with self.assertRaisesRegex(ValueError, "explicit tracking tolerance"):
            WorldModelConfig(power_interface="fc_only")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            WorldModelConfig(
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=-1.0,
            )

    def test_high_load_has_larger_health_increment(self):
        state = self.model.initial_state()
        result = self.model.step(
            state,
            MultiStackAction.from_currents([195, 370]),
            demand_power_kw=60.0,
        )
        self.assertGreater(
            result.stacks[1].degradation_increment_pct,
            result.stacks[0].degradation_increment_pct,
        )

    def test_degradation_cost_uses_one_step_not_lifetime_reference(self):
        state = self.model.initial_state()
        step = self.model.step(
            state,
            MultiStackAction.from_currents([195.0, 195.0]),
            demand_power_kw=50.0,
        )
        self.assertGreater(step.cost.degradation_increment, 0.5)
        self.assertLessEqual(step.cost.degradation_increment, 1.0)
        self.assertLess(step.cost.raw_degradation_reference_pct, 0.01)

    def test_zero_current_distinguishes_energized_idle_from_off(self):
        state = self.model.initial_state()
        result = self.model.step(
            state,
            MultiStackAction.from_currents([0, 0], is_on=[True, False]),
            demand_power_kw=0.0,
        )
        self.assertGreater(result.stacks[0].degradation_increment_pct, 0.0)
        self.assertEqual(result.stacks[1].degradation_increment_pct, 0.0)
        self.assertTrue(result.next_state.stacks[0].health.is_on)
        self.assertFalse(result.next_state.stacks[1].health.is_on)

    def test_three_stack_rotation_updates_only_executed_online_stacks(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            heterogeneity_factors=[1.0, 1.05, 1.10],
        )
        state = model.initial_state()
        first_pair = MultiStackAction.from_currents([90.0, 195.0, 0.0])
        action_rows = []
        for _ in range(15):
            step = model.step(state, first_pair, demand_power_kw=36.0)
            self.assertTrue(step.constraints.feasible)
            self.assertGreater(step.stacks[0].degradation_increment_pct, 0.0)
            self.assertGreater(step.stacks[1].degradation_increment_pct, 0.0)
            self.assertEqual(step.stacks[2].degradation_increment_pct, 0.0)
            self.assertEqual(step.stacks[2].power_kw, 0.0)
            self.assertEqual(step.next_state.stacks[2].health.degradation, 0.0)
            for index, stack_step in enumerate(step.stacks):
                expected_theta = model.performance_proxies[
                    index
                ].mapping.theta_reported(stack_step.degradation_after_pct)
                np.testing.assert_allclose(stack_step.theta_reported, expected_theta)
            action_rows.append(first_pair)
            state = step.next_state

        damage_before_rotation = tuple(
            stack.health.degradation for stack in state.stacks
        )
        rotated_pair = MultiStackAction.from_currents([0.0, 270.0, 90.0])
        rotated = model.step(state, rotated_pair, demand_power_kw=45.0)
        self.assertTrue(rotated.constraints.feasible)
        self.assertEqual(rotated.stacks[0].degradation_increment_pct, 0.0)
        self.assertEqual(
            rotated.next_state.stacks[0].health.degradation,
            damage_before_rotation[0],
        )
        self.assertGreater(rotated.stacks[1].degradation_increment_pct, 0.0)
        self.assertGreater(rotated.stacks[2].degradation_increment_pct, 0.0)
        self.assertEqual(sum(item.is_on for item in rotated.stacks), 2)
        self.assertFalse(rotated.stacks[0].shifted_load)
        self.assertTrue(rotated.stacks[1].shifted_load)
        self.assertFalse(rotated.stacks[2].shifted_load)

        action_rows.append(rotated_pair)
        trajectory = pd.DataFrame(
            {
                "step": np.arange(len(action_rows)),
                **{
                    f"stack_{index}_current_a": [
                        action.current_a[index] for action in action_rows
                    ]
                    for index in range(3)
                },
                **{
                    f"stack_{index}_on": [
                        action.is_on[index] for action in action_rows
                    ]
                    for index in range(3)
                },
            }
        )
        exposure = extract_action_exposure(
            trajectory,
            n_stacks=3,
            heterogeneity_factors=(1.0, 1.05, 1.10),
        )
        self.assertEqual(exposure.start_count, (1, 1, 1))
        self.assertEqual(exposure.load_shift_count, (0, 1, 0))

    def test_aged_stack_produces_less_power_at_same_current(self):
        reference = self.model.performance_proxies[0].mapping.damage_reference_pct
        state = self.model.initial_state(degradation_pct=[0.0, reference])
        result = self.model.step(
            state,
            MultiStackAction.from_currents([195, 195]),
            demand_power_kw=50.0,
        )
        self.assertGreater(result.stacks[0].power_kw, result.stacks[1].power_kw)

    def test_infeasible_battery_residual_is_reported(self):
        state = self.model.initial_state()
        result = self.model.step(
            state,
            MultiStackAction.from_currents([0, 0]),
            demand_power_kw=400.0,
        )
        self.assertFalse(result.constraints.feasible)
        self.assertIn("battery:discharge_power_limit", result.constraints.violations)

    def test_fc_only_interface_exposes_tracking_without_battery(self):
        config = WorldModelConfig(
            max_online_stacks=2,
            power_interface="fc_only",
            fc_power_tracking_tolerance_kw=5.5,
        )
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=config,
        )
        state = model.initial_state(soc=0.61)
        result = model.step(
            state,
            MultiStackAction.from_currents([90.0, 195.0, 0.0]),
            demand_power_kw=36.0,
        )
        self.assertTrue(result.constraints.feasible)
        self.assertEqual(result.constraints.power_interface, "fc_only")
        self.assertEqual(result.constraints.battery_power_kw, 0.0)
        self.assertEqual(result.next_state.soc, state.soc)
        self.assertEqual(result.cost.battery_use, 0.0)
        self.assertEqual(result.cost.soc, 0.0)
        self.assertGreater(result.cost.power_tracking, 0.0)
        self.assertAlmostEqual(
            result.cost.raw_power_tracking_error_kw,
            result.constraints.power_balance_error_kw,
        )

        infeasible = model.step(
            state,
            MultiStackAction.from_currents([0.0, 0.0, 0.0]),
            demand_power_kw=50.0,
        )
        self.assertFalse(infeasible.constraints.feasible)
        self.assertIn("system:fc_power_tracking", infeasible.constraints.violations)
        self.assertFalse(
            any(item.startswith("battery:") for item in infeasible.constraints.violations)
        )

    def test_minimum_dwell_violation_is_reported(self):
        stack = StackControlState(
            GammaHealthState(current_a=195.0, is_on=True), dwell_s=1.0
        )
        state = MultiStackState(soc=0.70, stacks=(stack, stack))
        result = self.model.step(
            state,
            MultiStackAction.from_currents([270, 195]),
            demand_power_kw=50.0,
        )
        self.assertFalse(result.constraints.feasible)
        self.assertIn("stack_0:minimum_dwell", result.constraints.violations)

    def test_safety_override_relaxes_dwell_but_is_audited(self):
        stack = StackControlState(
            GammaHealthState(current_a=90.0, is_on=True), dwell_s=14.0
        )
        state = MultiStackState(soc=0.70, stacks=(stack, stack))
        result = self.model.step(
            state,
            MultiStackAction.from_currents([0, 0]),
            demand_power_kw=-73.0,
            allow_dwell_override=True,
        )
        self.assertTrue(result.constraints.feasible)
        self.assertIn("stack_0:minimum_dwell", result.constraints.safety_overrides)

    def test_soc_feedback_prefers_discharge_when_soc_is_high(self):
        high_soc = self.model.initial_state(soc=0.72)
        reference_soc = self.model.initial_state(soc=0.70)
        action = MultiStackAction.from_currents([25, 25])
        demand = 20.0
        high = self.model.step(high_soc, action, demand)
        reference = self.model.step(reference_soc, action, demand)
        self.assertLess(high.cost.battery_use, reference.cost.battery_use)

    def test_factory_exposes_gamma_cv_sensitivity(self):
        low = load_lzw_multistack_world_model(
            ROOT, n_stacks=2, gamma_terminal_cv=0.05
        )
        high = load_lzw_multistack_world_model(
            ROOT, n_stacks=2, gamma_terminal_cv=0.20
        )
        self.assertGreater(
            high.health_models[0].params.gamma_scale,
            low.health_models[0].params.gamma_scale,
        )

    def test_factory_exposes_literature_coefficient_sensitivity(self):
        baseline = load_lzw_multistack_world_model(ROOT, n_stacks=2)
        doubled = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=2,
            health_coefficients=GhaderiPeiCoefficients(
                start_stop_pct_per_cycle=2 * 0.00196
            ),
        )
        self.assertEqual(
            doubled.health_models[0].params.start_increment,
            2 * baseline.health_models[0].params.start_increment,
        )


if __name__ == "__main__":
    unittest.main()
