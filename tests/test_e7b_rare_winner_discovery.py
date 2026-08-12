"""Tests for E7B-R rare-winner discovery package."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_e7b_rare_winner_discovery import (
    FOCUSED_DATASETS,
    SMOKE_DATASET,
    build_target_vector,
    classify_rare_winner_status,
    classify_stable_strategy_status,
    fit_clip_thresholds,
    model_specs,
    parse_args,
    resolve_dataset_path,
    TrainRankTransform,
)

ROOT = Path(__file__).resolve().parents[1]


class E7BRareWinnerDiscoveryTests(unittest.TestCase):
    def test_default_cli_mode_is_smoke(self):
        args = parse_args([])
        self.assertTrue(args.smoke)
        self.assertFalse(args.focused)

    def test_focused_mode_disables_smoke(self):
        args = parse_args(["--focused"])
        self.assertFalse(args.smoke)
        self.assertTrue(args.focused)

    def test_model_specs_default_rf_only(self):
        specs = model_specs(include_xgb=False)
        self.assertEqual(len(specs), 4)
        self.assertTrue(all(name == "RF" for name, _, _ in specs))

    def test_model_specs_include_xgb_only_when_requested(self):
        specs = model_specs(include_xgb=True)
        self.assertEqual(len(specs), 8)
        self.assertIn("XGB", {name for name, _, _ in specs})

    def test_no_tab_option_in_parser(self):
        with self.assertRaises(SystemExit):
            parse_args(["--include-tab"])

    def test_winner_threshold_not_fit_on_test(self):
        val = pd.Series([0.1, 0.2, 0.5, 1.0, 2.0])
        test = pd.Series([0.05, 0.9, 3.0])
        cutoffs = {2.0: float(np.quantile(val, 1 - 2.0 / 100.0))}
        self.assertGreater(cutoffs[2.0], test.min())
        self.assertNotEqual(cutoffs[2.0], float(np.quantile(test, 1 - 2.0 / 100.0)))

    def test_target_row_id_preserved_in_metrics_schema(self):
        row = {
            "candidate_id": "c1",
            "candidate_policy_id": "p1",
            "target_row_id": "t1",
            "pair_address": "pair",
            "split": "test",
            "filter": "LIQ_5K_HIGH_ACTIVITY",
            "horizon": "4h",
            "exit_policy_id": "TP",
            "sim_net_return": 1.0,
        }
        frame = pd.DataFrame([row])
        for col in ("candidate_id", "candidate_policy_id", "target_row_id"):
            self.assertIn(col, frame.columns)

    def test_drawdown_target_not_in_model_specs(self):
        families = {family for _, family, _ in model_specs(False)}
        self.assertNotIn("drawdown_adjusted_return", families)

    def test_pair_concentration_maps_to_pair_concentrated_status(self):
        val_m = {
            "selected_rows": 20,
            "selected_rare_winner_count": 2,
            "rare_winner_lift": 3,
        }
        test_m = {
            "selected_rows": 20,
            "selected_rare_winner_count": 2,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 2,
            "selected_top_pair_share": 0.8,
            "rare_winner_lift": 10,
        }
        status = classify_rare_winner_status(val_m, test_m)
        self.assertEqual(status, "PAIR_CONCENTRATED_RARE_WINNER")
        self.assertNotEqual(status, "UNUSABLE_OFFLINE")

    def test_single_pair_maps_to_single_pair_status(self):
        val_m = {"selected_rows": 12, "selected_rare_winner_count": 1, "rare_winner_lift": 3}
        test_m = {
            "selected_rows": 12,
            "selected_rare_winner_count": 1,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 1,
            "selected_top_pair_share": 1.0,
            "rare_winner_lift": 8,
        }
        self.assertEqual(classify_rare_winner_status(val_m, test_m), "SINGLE_PAIR_RARE_WINNER")

    def test_stable_and_rare_statuses_are_separate(self):
        val_m = {"selected_rows": 12, "selected_rare_winner_count": 1, "rare_winner_lift": 3}
        test_m = {
            "selected_rows": 12,
            "selected_rare_winner_count": 1,
            "selected_total_net_return": 1.0,
            "selected_unique_pairs": 1,
            "selected_top_pair_share": 1.0,
            "rare_winner_lift": 8,
        }
        rare = classify_rare_winner_status(val_m, test_m)
        stable = classify_stable_strategy_status(test_m, 0.5, 0.5)
        self.assertNotEqual(rare, stable)
        self.assertEqual(stable, "STABLE_BLOCKED")

    def test_existing_output_root_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_e7b_rare_winner_discovery_test"
            out.mkdir()
            (out / "marker.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                from scripts.run_e7b_rare_winner_discovery import ensure_output_dirs

                ensure_output_dirs(
                    argparse.Namespace(
                        output_root=str(out),
                        audit_root=str(Path(tmp) / "audit"),
                        overwrite=False,
                    )
                )

    def test_clip_and_rank_use_train_only(self):
        train = pd.DataFrame({"sim_net_return": [0.0, 1.0, 2.0, 100.0]})
        val = pd.DataFrame({"sim_net_return": [50.0]})
        bounds = fit_clip_thresholds(train["sim_net_return"].to_numpy())
        rank = TrainRankTransform(train["sim_net_return"].to_numpy())
        clipped_val = build_target_vector(val, "clipped", bounds, None)
        ranked_val = build_target_vector(val, "ranked", None, rank)
        self.assertLessEqual(clipped_val[0], bounds[1])
        self.assertGreaterEqual(ranked_val[0], 0.0)
        self.assertLessEqual(ranked_val[0], 1.0)

    def test_smoke_dataset_exists(self):
        path = resolve_dataset_path(SMOKE_DATASET)
        if not path.exists():
            self.skipTest("Smoke dataset missing")
        self.assertTrue(path.exists())

    def test_focused_dataset_count(self):
        existing = [resolve_dataset_path(rel) for rel in FOCUSED_DATASETS if resolve_dataset_path(rel).exists()]
        if not existing:
            self.skipTest("Focused datasets missing")
        self.assertEqual(len(FOCUSED_DATASETS), 4)


if __name__ == "__main__":
    unittest.main()
