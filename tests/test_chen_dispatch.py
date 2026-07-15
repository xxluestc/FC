import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.power_allocation.chen_dispatch import (
    ChenDispatchModel,
    changed_stack_states,
)


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data/processed/chen_efficiency_curves_audited.csv"


class ChenDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ChenDispatchModel(pd.read_csv(CURVES))

    def test_off_and_single_stack_dispatch_close_power_balance(self):
        off = self.model.solve_mode(0.0, ())
        self.assertIsNotNone(off)
        self.assertEqual(off.mode, ())
        self.assertEqual(off.hydrogen_g_per_s, 0.0)

        solution = self.model.solve_mode(25.0, ("stack_3",))
        self.assertIsNotNone(solution)
        self.assertAlmostEqual(solution.power_balance_error_kw, 0.0)
        self.assertAlmostEqual(sum(solution.stack_net_power_kw), 25.0)
        self.assertGreater(solution.stack_gross_power_kw[2], 25.0)
        self.assertGreater(solution.system_efficiency_lhv_pct, 49.0)

    def test_active_stack_domain_and_n_plus_one_limit_are_explicit(self):
        self.assertIsNone(self.model.solve_mode(5.0, ("stack_1",)))
        self.assertIsNone(self.model.solve_mode(120.0, ("stack_1", "stack_2")))
        with self.assertRaisesRegex(ValueError, "at most two"):
            self.model.solve_mode(
                100.0,
                ("stack_1", "stack_2", "stack_3"),
            )

    def test_two_stack_breakpoint_search_matches_dense_global_search(self):
        demand = 75.0
        solution = self.model.solve_mode(demand, ("stack_2", "stack_3"))
        self.assertIsNotNone(solution)
        first = self.model.curves["stack_2"]
        second = self.model.curves["stack_3"]
        lower = max(
            first.minimum_net_power_kw,
            demand - second.maximum_net_power_kw,
        )
        upper = min(
            first.maximum_net_power_kw,
            demand - second.minimum_net_power_kw,
        )
        grid = np.linspace(lower, upper, 200_001)
        dense_cost = np.interp(
            grid,
            first.net_power_kw,
            first.chemical_input_lhv_kw,
        ) + np.interp(
            demand - grid,
            second.net_power_kw,
            second.chemical_input_lhv_kw,
        )
        self.assertLessEqual(
            solution.total_chemical_input_lhv_kw,
            float(dense_cost.min()) + 1e-9,
        )
        self.assertAlmostEqual(solution.power_balance_error_kw, 0.0)

    def test_instantaneous_solution_uses_efficiency_heterogeneity(self):
        low = self.model.solve_instantaneous(25.0)
        medium = self.model.solve_instantaneous(50.0)
        self.assertEqual(low.mode, ("stack_3",))
        self.assertEqual(medium.mode, ("stack_2", "stack_3"))
        self.assertAlmostEqual(medium.stack_net_power_kw[1], 25.180252486426433)
        self.assertAlmostEqual(medium.stack_net_power_kw[2], 24.819747513573567)

    def test_changed_stack_states_counts_starts_and_stops(self):
        self.assertEqual(changed_stack_states((), ("stack_3",)), 1)
        self.assertEqual(
            changed_stack_states(("stack_3",), ("stack_2", "stack_3")),
            1,
        )
        self.assertEqual(
            changed_stack_states(("stack_1", "stack_2"), ("stack_2", "stack_3")),
            2,
        )


if __name__ == "__main__":
    unittest.main()
