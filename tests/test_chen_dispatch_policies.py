import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.power_allocation.chen_dispatch import ChenDispatchModel
from fc_power.power_allocation.chen_dispatch_policies import run_chen_policy


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data/processed/chen_efficiency_curves_audited.csv"


class ChenDispatchPoliciesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ChenDispatchModel(pd.read_csv(CURVES))
        cls.demand = np.asarray([25.0, 25.0, 95.0, 95.0, 25.0, 25.0])

    def test_every_policy_closes_power_balance(self):
        for strategy in (
            "average",
            "daisy_chain",
            "instantaneous",
            "sticky",
            "one_step_greedy",
            "break_even_hysteresis",
            "offline_dp",
        ):
            run = run_chen_policy(
                self.model,
                self.demand,
                strategy,
                switch_penalty_g_per_change=0.2,
            )
            self.assertLess(run.metrics["power_balance_max_abs_kw"], 1e-9)
            self.assertEqual(len(run.trajectory), len(self.demand))
            np.testing.assert_allclose(
                run.trajectory["demand_net_power_kw"],
                self.demand,
            )

    def test_sticky_trades_hydrogen_for_fewer_state_changes(self):
        instantaneous = run_chen_policy(
            self.model,
            self.demand,
            "instantaneous",
            switch_penalty_g_per_change=0.2,
        )
        sticky = run_chen_policy(
            self.model,
            self.demand,
            "sticky",
            switch_penalty_g_per_change=0.2,
        )
        self.assertGreater(
            sticky.metrics["total_hydrogen_g"],
            instantaneous.metrics["total_hydrogen_g"],
        )
        self.assertLess(
            sticky.metrics["total_stack_state_changes"],
            instantaneous.metrics["total_stack_state_changes"],
        )

    def test_offline_dp_is_lower_bound_for_the_same_evaluated_objective(self):
        penalty = 0.2
        offline = run_chen_policy(
            self.model,
            self.demand,
            "offline_dp",
            switch_penalty_g_per_change=penalty,
        )
        for strategy in (
            "average",
            "daisy_chain",
            "instantaneous",
            "sticky",
            "one_step_greedy",
            "break_even_hysteresis",
        ):
            online = run_chen_policy(
                self.model,
                self.demand,
                strategy,
                switch_penalty_g_per_change=penalty,
            )
            self.assertLessEqual(
                offline.metrics["total_evaluated_objective_g"],
                online.metrics["total_evaluated_objective_g"] + 1e-12,
            )

    def test_break_even_hysteresis_is_causal(self):
        prefix = np.asarray([25.0] * 20 + [50.0] * 20 + [95.0] * 10)
        first = run_chen_policy(
            self.model,
            np.r_[prefix, [25.0] * 30],
            "break_even_hysteresis",
            switch_penalty_g_per_change=0.1,
        )
        second = run_chen_policy(
            self.model,
            np.r_[prefix, [110.0] * 30],
            "break_even_hysteresis",
            switch_penalty_g_per_change=0.1,
        )
        self.assertEqual(
            first.trajectory.loc[: len(prefix) - 1, "mode"].tolist(),
            second.trajectory.loc[: len(prefix) - 1, "mode"].tolist(),
        )

    def test_average_and_daisy_have_distinct_power_splits(self):
        average = run_chen_policy(self.model, [50.0], "average")
        daisy = run_chen_policy(self.model, [50.0], "daisy_chain")
        self.assertEqual(average.trajectory.loc[0, "active_stack_count"], 2)
        self.assertEqual(daisy.trajectory.loc[0, "active_stack_count"], 1)
        self.assertNotEqual(
            average.trajectory.loc[0, "mode"],
            daisy.trajectory.loc[0, "mode"],
        )


if __name__ == "__main__":
    unittest.main()
