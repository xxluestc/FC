import unittest
from pathlib import Path

from fc_power.health.gamma_process import GammaHealthState
from fc_power.world_model import (
    MultiStackAction,
    MultiStackState,
    StackControlState,
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
