import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fc_power.power_allocation.chen_efficiency_curves import (
    HHV_KJ_PER_MOL,
    LHV_KJ_PER_MOL,
    audit_chen_efficiency_curves,
    summarize_chen_efficiency_curves,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/upstream_chen/chen_efficiency_curves_origin_sheet5.csv"


class ChenEfficiencyCurvesTest(unittest.TestCase):
    def setUp(self):
        self.source = pd.read_csv(SOURCE)
        self.audited = audit_chen_efficiency_curves(self.source)

    def test_origin_snapshot_has_three_complete_curves(self):
        self.assertEqual(len(self.source), 69)
        self.assertEqual(
            self.source.groupby("stack_id").size().to_dict(),
            {"stack_1": 23, "stack_2": 23, "stack_3": 23},
        )
        self.assertEqual(
            self.source.groupby("stack_id")["cell_count"].first().to_dict(),
            {"stack_1": 270, "stack_2": 300, "stack_3": 330},
        )

    def test_net_power_reconstruction_matches_chen_definition(self):
        first = self.audited.iloc[0]
        self.assertAlmostEqual(first["net_system_power_kw"], 5.663981334683803)
        self.assertGreater(first["gross_stack_power_kw"], first["net_system_power_kw"])
        self.assertTrue(
            np.all(
                self.audited["gross_stack_power_kw"]
                > self.audited["net_system_power_kw"]
            )
        )
        np.testing.assert_allclose(
            self.audited["efficiency_hhv_pct"],
            self.audited["efficiency_lhv_pct"]
            * LHV_KJ_PER_MOL
            / HHV_KJ_PER_MOL,
        )

    def test_curve_domains_and_peaks_are_traceable(self):
        summaries = {
            item["stack_id"]: item
            for item in summarize_chen_efficiency_curves(self.audited)
        }
        expected = {
            "stack_1": (8.74272975786213, 62.0845703704172, 54.26316959281429, 48.0744215565514),
            "stack_2": (9.71414417540237, 69.9589562898091, 60.07886747049697, 48.6350697365138),
            "stack_3": (10.6855585929426, 78.4564989822725, 65.80652950667101, 49.1578989353604),
        }
        for stack_id, values in expected.items():
            item = summaries[stack_id]
            self.assertAlmostEqual(item["gross_power_min_kw"], values[0])
            self.assertAlmostEqual(item["gross_power_max_kw"], values[1])
            self.assertAlmostEqual(item["net_power_max_kw"], values[2])
            self.assertAlmostEqual(item["peak_efficiency_lhv_pct"], values[3])

    def test_duplicate_stack_sample_is_rejected(self):
        duplicate = pd.concat([self.source, self.source.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            audit_chen_efficiency_curves(duplicate)


if __name__ == "__main__":
    unittest.main()
