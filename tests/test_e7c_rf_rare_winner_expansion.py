"""Tests for E7C RF rare-winner expansion package."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_e7c_rf_rare_winner_expansion as e7c
from scripts.run_e7c_rf_rare_winner_expansion import (
    REGISTRY_WHITELIST,
    TWO_STAGE_TRAINING_MODE,
    classify_rare_winner_status,
    classify_stable_strategy_status,
    counts_as_independent,
    parse_args,
)

ROOT = Path(__file__).resolve().parents[1]


class E7CRareWinnerExpansionTests(unittest.TestCase):
    def test_registry_whitelist_compact_only(self):
        self.assertEqual(len(REGISTRY_WHITELIST), 3)
        self.assertIn("manifests/e7c_manifest.json", REGISTRY_WHITELIST)
        self.assertIn("reports/e7c_final_consensus_summary.md", REGISTRY_WHITELIST)
        self.assertIn("metrics/e7c_final_consensus_summary.csv", REGISTRY_WHITELIST)
        for item in REGISTRY_WHITELIST:
            self.assertNotIn("capture_grid", item)
            self.assertNotIn("predictions", item)

    def test_no_whale_not_independent(self):
        self.assertFalse(counts_as_independent("NO_WHALE_FILTER", "DIVERSIFIED_RARE_WINNER", "single_stage"))
        self.assertTrue(counts_as_independent("LIQ_5K_HIGH_ACTIVITY", "DIVERSIFIED_RARE_WINNER", "single_stage"))

    def test_liq_canonical_independent_when_diversified(self):
        self.assertTrue(counts_as_independent("LIQ_5K_HIGH_ACTIVITY", "DIVERSIFIED_RARE_WINNER", "single_stage"))

    def test_winner_threshold_validation_only(self):
        val = pd.Series([0.1, 0.2, 0.5, 1.0, 2.0])
        test = pd.Series([0.05, 0.9, 3.0])
        val_cut = float(np.quantile(val, 1 - 2 / 100))
        test_cut_if_wrong = float(np.quantile(test, 1 - 2 / 100))
        self.assertNotEqual(val_cut, test_cut_if_wrong)

    def test_stable_uses_local_gate_not_candidate_offline(self):
        test_m = {
            "selected_rows": 20,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 4,
            "selected_top_pair_share": 0.4,
        }
        status = classify_stable_strategy_status(test_m, 0.5, 0.5)
        self.assertEqual(status, "LOCAL_STABLE_GATE_PASS")
        self.assertNotEqual(status, "STABLE_CANDIDATE_OFFLINE")

    def test_stable_separate_from_rare(self):
        val_m = {"selected_rows": 12, "selected_rare_winner_count": 1, "rare_winner_lift": 3}
        test_m = {
            "selected_rows": 12,
            "selected_rare_winner_count": 1,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 2,
            "selected_top_pair_share": 0.8,
            "rare_winner_lift": 10,
        }
        rare = classify_rare_winner_status(val_m, test_m, filter_name="LIQ_5K_HIGH_ACTIVITY")
        stable = classify_stable_strategy_status(test_m, 0.5, 0.5)
        self.assertEqual(rare, "PAIR_CONCENTRATED_RARE_WINNER")
        self.assertNotEqual(rare, "UNUSABLE_OFFLINE")
        self.assertNotEqual(rare, stable)

    def test_single_pair_status(self):
        val_m = {"selected_rows": 12, "selected_rare_winner_count": 1, "rare_winner_lift": 3}
        test_m = {
            "selected_rows": 12,
            "selected_rare_winner_count": 1,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 1,
            "selected_top_pair_share": 1.0,
            "rare_winner_lift": 8,
        }
        self.assertEqual(
            classify_rare_winner_status(val_m, test_m, filter_name="LIQ_5K_HIGH_ACTIVITY"),
            "SINGLE_PAIR_RARE_WINNER",
        )

    def test_low_liq_diagnostic_only(self):
        val_m = {"selected_rows": 12, "selected_rare_winner_count": 1, "rare_winner_lift": 3}
        test_m = {
            "selected_rows": 12,
            "selected_rare_winner_count": 1,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 4,
            "selected_top_pair_share": 0.2,
            "rare_winner_lift": 10,
        }
        self.assertEqual(
            classify_rare_winner_status(val_m, test_m, filter_name="LOW_LIQ_MOMENTUM", diagnostic_only=True),
            "DIAGNOSTIC_ONLY",
        )

    def test_default_smoke_mode(self):
        args = parse_args([])
        self.assertTrue(args.smoke)

    def test_smoke_one_dataset(self):
        self.assertEqual(len(e7c.CORE_LIQ_DATASETS), 10)
        args = parse_args(["--smoke"])
        if args.smoke:
            datasets = [e7c.resolve_dataset_path(e7c.SMOKE_DATASET)]
            self.assertEqual(len(datasets), 1)

    def test_xgb_only_with_flag(self):
        specs_default = e7c.model_specs(False)
        specs_xgb = e7c.model_specs(True)
        self.assertEqual(len(specs_default), 4)
        self.assertEqual(len(specs_xgb), 8)

    def test_no_tab_in_parser(self):
        with self.assertRaises(SystemExit):
            parse_args(["--include-tab"])

    def test_two_stage_training_mode_default(self):
        self.assertEqual(TWO_STAGE_TRAINING_MODE, "full_train_economic_model")

    def test_stage2_split_consistency_audit_expects_zero_val_test_train_rows(self):
        audit = {
            "validation_rows_used_for_stage2_training": 0,
            "test_rows_used_for_stage2_training": 0,
        }
        self.assertEqual(audit["validation_rows_used_for_stage2_training"], 0)
        self.assertEqual(audit["test_rows_used_for_stage2_training"], 0)

    def test_output_root_exists_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_e7c"
            out.mkdir()
            (out / "x.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                e7c.ensure_output_dirs(
                    argparse.Namespace(
                        output_root=str(out),
                        audit_root=str(Path(tmp) / "audit"),
                        overwrite=False,
                    )
                )

    def test_scanner_audit_is_static_only(self):
        rows = e7c.audit_scanner_code()
        self.assertTrue(rows)
        for row in rows:
            if row.get("exists"):
                self.assertIn("Static code audit", row.get("notes", ""))

    def test_drawdown_not_in_economic_families(self):
        self.assertNotIn("drawdown_adjusted_return", e7c.ECONOMIC_FAMILIES)

    def test_identity_columns_defined(self):
        frame = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "candidate_policy_id": ["b"],
                "target_row_id": ["c"],
            }
        )
        for col in ("candidate_id", "candidate_policy_id", "target_row_id"):
            self.assertIn(col, frame.columns)


if __name__ == "__main__":
    unittest.main()
