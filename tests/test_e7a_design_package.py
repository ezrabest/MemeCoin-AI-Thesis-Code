"""Lightweight tests for E7A design package builder."""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.build_e7a_design_package import (
    discover_e3_datasets,
    map_policy_status,
    pair_generalization_matrix,
    robustness_gate_spec,
    target_family_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
E3_ROOT = ROOT / "data/training/manual_verified_datasets_direct_target_v1"


class E7ADesignPackageTests(unittest.TestCase):
    def test_target_family_matrix_has_drawdown_deferred(self):
        matrix = target_family_matrix()
        drawdown = next(row for row in matrix if row["target_name"] == "drawdown_adjusted_return")
        self.assertFalse(drawdown["feasible_from_current_E3"])
        self.assertFalse(drawdown["allowed_in_E7B"])
        self.assertIn("drawdown", drawdown["deferred_reason"].lower())

    def test_target_family_matrix_includes_identity_requirements(self):
        matrix = target_family_matrix()
        for row in matrix:
            if row["feasible_from_current_E3"]:
                self.assertIn("target_row_id", row["identity_preservation_requirement"])

    def test_robustness_gate_required_fail_maps_unusable(self):
        gates = robustness_gate_spec()
        required = [g for g in gates if g["required_for_USABLE_OFFLINE"]]
        self.assertGreaterEqual(len(required), 10)
        self.assertTrue(all(g["fail_status"] == "UNUSABLE_OFFLINE" for g in required))

    def test_pair_generalization_groupkfold_not_primary(self):
        matrix = pair_generalization_matrix()
        groupkfold = next(row for row in matrix if row["strategy_id"] == "GroupKFold_diagnostic_only")
        self.assertFalse(groupkfold["allowed_in_E7B"])

    def test_map_policy_status_gate_soft_false(self):
        row = pd.Series(
            {
                "e6r7_gate_soft": False,
                "validation_robustness_status": "PASS",
                "test_robustness_status": "PASS",
            }
        )
        self.assertEqual(map_policy_status(row), "UNUSABLE_OFFLINE")

    def test_map_policy_status_best_pair_dominated(self):
        row = pd.Series(
            {
                "e6r7_gate_soft": True,
                "validation_robustness_status": "BEST_PAIR_DOMINATED_FAIL",
                "test_robustness_status": "PASS",
                "validation_remove_best_trade_pass": True,
                "test_remove_best_trade_pass": True,
                "validation_remove_best_pair_pass": True,
                "test_remove_best_pair_pass": True,
                "validation_positive_return": True,
                "test_positive_return": True,
                "positive_both_after_pair_cap": True,
            }
        )
        self.assertEqual(map_policy_status(row), "UNUSABLE_OFFLINE")

    def test_e3_dataset_count_when_present(self):
        if not E3_ROOT.exists():
            self.skipTest("E3 root missing")
        datasets = discover_e3_datasets(E3_ROOT)
        self.assertEqual(len(datasets), 40)


if __name__ == "__main__":
    unittest.main()
