import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.evaluation import (
    TestScenario,
    clip_profile_to_feasible_envelope,
    paired_strategy_comparison,
    run_policy,
)
from fc_power.world_model import WorldModelConfig, load_lzw_multistack_world_model


ROOT = Path(__file__).resolve().parents[1]


class MultiStackTestbedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_lzw_multistack_world_model(
            ROOT, n_stacks=2, heterogeneity_factors=[1.0, 1.1]
        )

    @staticmethod
    def demand(length=45, value=60.0):
        return pd.DataFrame(
            {"demand_power_kw": np.full(length, value), "event": "cruise"}
        )

    def test_health_is_updated_and_carried_after_every_action(self):
        scenario = TestScenario(
            "online_health",
            self.demand(),
            (0.1, 0.7),
            stochastic_health=False,
        )
        run = run_policy(self.model, scenario, "average")
        for index in range(2):
            before = run.trajectory[f"stack_{index}_damage_before_pct"].to_numpy()
            after = run.trajectory[f"stack_{index}_damage_after_pct"].to_numpy()
            np.testing.assert_allclose(before[1:], after[:-1])
            self.assertTrue(np.all(after >= before))
        self.assertGreater(run.metrics["health_changed_steps"], 0)
        self.assertAlmostEqual(
            run.metrics["main_expected_damage_increment_pct"],
            run.metrics["main_expected_continuous_damage_pct"]
            + run.metrics["main_discrete_damage_pct"],
        )

    def test_same_health_seed_reproduces_stochastic_run(self):
        scenario = TestScenario("stochastic", self.demand(25), (0.1, 0.7), health_seed=9)
        first = run_policy(self.model, scenario, "average")
        second = run_policy(self.model, scenario, "average")
        np.testing.assert_allclose(
            first.trajectory.sampled_damage_increment_pct,
            second.trajectory.sampled_damage_increment_pct,
        )

    def test_health_aware_beam_reduces_aged_stack_current(self):
        scenario = TestScenario(
            "heterogeneous",
            self.demand(60, 55.0),
            (0.05, 0.90),
            stochastic_health=False,
        )
        run = run_policy(
            self.model, scenario, "beam_health", beam_horizon=16, beam_width=4
        )
        self.assertLess(
            run.metrics["stack_1_current_a_step"],
            run.metrics["stack_0_current_a_step"],
        )
        self.assertEqual(run.metrics["constraint_violation_steps"], 0)

    def test_global_envelope_clipping_is_health_dependent_and_audited(self):
        demand = pd.DataFrame(
            {"demand_power_kw": [-100.0, 0.0, 500.0], "event": "cruise"}
        )
        clipped = clip_profile_to_feasible_envelope(
            self.model, demand, (0.1, 0.7)
        )
        self.assertEqual(int(clipped.was_clipped.sum()), 2)
        self.assertEqual(clipped.demand_power_kw.iloc[0], -75.0)
        self.assertLess(clipped.demand_power_kw.iloc[-1], 500.0)
        self.assertIn("feasible_upper_kw", clipped.attrs)
        self.assertEqual(clipped.attrs["stack_capacity_reserve_fraction"], 0.01)

    def test_capacity_reserve_survives_online_health_drift_at_upper_bound(self):
        reference = self.model.performance_proxies[0].mapping.damage_reference_pct
        initial_damage = (0.1, 0.7)
        initial_stack_capacity = sum(
            proxy.evaluate(fraction * reference, [370.0])["stack_power_kw"][0]
            for proxy, fraction in zip(
                self.model.performance_proxies, initial_damage
            )
        )
        raw_upper = (
            initial_stack_capacity
            + self.model.config.battery.discharge_power_limit_kw
        )
        demand = pd.DataFrame(
            {
                "demand_power_kw": np.full(45, raw_upper + 10.0),
                "event": "high",
            }
        )
        clipped = clip_profile_to_feasible_envelope(
            self.model, demand, initial_damage
        )
        scenario = TestScenario(
            "reserved_upper_bound",
            clipped,
            initial_damage,
            stochastic_health=False,
        )
        run = run_policy(self.model, scenario, "average")
        self.assertEqual(run.metrics["constraint_violation_steps"], 0)
        self.assertTrue(clipped.was_clipped.all())

    def test_three_stack_online_runner_is_not_hard_coded_to_two(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            heterogeneity_factors=[1.0, 1.05, 1.1],
        )
        scenario = TestScenario(
            "three_stack",
            self.demand(10, 70.0),
            (0.1, 0.4, 0.8),
            stochastic_health=False,
        )
        run = run_policy(model, scenario, "instant_health")
        self.assertIn("stack_2_final_damage_pct", run.metrics)
        self.assertEqual(run.metrics["constraint_violation_steps"], 0)

    def test_fc_only_runner_reports_tracking_and_no_battery_metrics(self):
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
        scenario = TestScenario(
            "fc_only",
            self.demand(20, 40.0),
            (0.1, 0.4, 0.8),
            initial_soc=0.61,
            stochastic_health=False,
        )
        run = run_policy(model, scenario, "instant_health")
        self.assertTrue((run.trajectory.battery_power_kw == 0.0).all())
        self.assertTrue((run.trajectory.soc == 0.61).all())
        self.assertEqual(run.metrics["power_interface"], "fc_only")
        self.assertEqual(run.metrics["constraint_violation_steps"], 0)
        self.assertLessEqual(run.metrics["fc_tracking_max_abs_kw"], 5.5)
        self.assertEqual(run.metrics["fc_tracking_within_tolerance_share"], 1.0)
        self.assertLessEqual(run.metrics["online_stack_count_max"], 2)
        self.assertEqual(run.metrics["online_stack_count_mean"], 2.0)
        self.assertGreater(run.metrics["fc_energy_kwh"], 0.0)
        self.assertGreater(run.metrics["hydrogen_g_per_fc_kwh"], 0.0)
        self.assertFalse(run.metrics["soc_metrics_applicable"])
        self.assertTrue(np.isnan(run.metrics["hydrogen_soc_corrected_g"]))
        self.assertTrue(np.isnan(run.metrics["battery_throughput_kwh"]))

    def test_fc_only_runner_rejects_soc_recovery_tail(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        demand = self.demand(2, 40.0)
        demand["is_soc_recovery"] = [False, True]
        scenario = TestScenario("invalid_recovery", demand, (0.1, 0.4, 0.8))
        with self.assertRaisesRegex(ValueError, "SOC recovery"):
            run_policy(model, scenario, "instant_health")

    def test_fc_only_beam_runner_tracks_and_freezes_soc(self):
        model = load_lzw_multistack_world_model(
            ROOT,
            n_stacks=3,
            config=WorldModelConfig(
                max_online_stacks=2,
                power_interface="fc_only",
                fc_power_tracking_tolerance_kw=5.5,
            ),
        )
        scenario = TestScenario(
            "fc_only_beam",
            self.demand(4, 40.0),
            (0.1, 0.4, 0.8),
            initial_soc=0.58,
            stochastic_health=False,
        )
        run = run_policy(
            model,
            scenario,
            "beam_health",
            beam_horizon=2,
            beam_width=2,
        )
        self.assertEqual(run.metrics["constraint_violation_steps"], 0)
        self.assertEqual(run.metrics["fc_tracking_within_tolerance_share"], 1.0)
        self.assertTrue((run.trajectory.soc == 0.58).all())
        self.assertTrue((run.trajectory.battery_power_kw == 0.0).all())
        self.assertGreater(run.metrics["planning_expanded_nodes"], 0)
        self.assertGreaterEqual(run.metrics["planning_runtime_s"], 0.0)

    def test_beam_preview_uses_only_current_demand(self):
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
        low_future = pd.DataFrame(
            {"demand_power_kw": [40.0, 20.0], "event": "audit"}
        )
        high_future = pd.DataFrame(
            {"demand_power_kw": [40.0, 80.0], "event": "audit"}
        )
        common = {
            "initial_damage_fraction": (0.1, 0.4, 0.8),
            "stochastic_health": False,
        }
        low = run_policy(
            model,
            TestScenario("low_future", low_future, **common),
            "beam_health",
            beam_horizon=2,
            beam_width=2,
        )
        high = run_policy(
            model,
            TestScenario("high_future", high_future, **common),
            "beam_health",
            beam_horizon=2,
            beam_width=2,
        )
        low_first = [low.trajectory[f"stack_{i}_current_a"].iloc[0] for i in range(3)]
        high_first = [
            high.trajectory[f"stack_{i}_current_a"].iloc[0] for i in range(3)
        ]
        self.assertEqual(low_first, high_first)

    def test_strategy_statistics_are_paired_by_load_and_health_seed(self):
        rows = []
        for seed, reference, candidate in ((1, 10.0, 9.0), (2, 30.0, 27.0)):
            for strategy, value in (("average", reference), ("beam_health", candidate)):
                rows.append(
                    {
                        "load_source": "synthetic",
                        "load_seed": seed,
                        "health_seed": 100 + seed,
                        "strategy": strategy,
                        "hydrogen_soc_corrected_g": value,
                        "main_expected_damage_increment_pct": value,
                        "main_performance_loss_sum": value,
                        "battery_throughput_kwh": value,
                        "main_aged_stack_current_share": value,
                    }
                )
        paired = paired_strategy_comparison(pd.DataFrame(rows))
        hydrogen = paired[paired.metric == "hydrogen_soc_corrected_g"].iloc[0]
        self.assertEqual(hydrogen.n_pairs, 2)
        self.assertAlmostEqual(hydrogen.mean_delta, -2.0)
        self.assertEqual(hydrogen.lower_is_better_win_share, 1.0)


if __name__ == "__main__":
    unittest.main()
