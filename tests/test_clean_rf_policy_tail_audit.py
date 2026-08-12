"""Tests for Phase E8C clean RF policy tail audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.clean_rf_policy_tail_audit import (  # noqa: E402
    TOP_PCT_PERCENT_VALUES,
    TARGET_COL,
    RETURN_COL,
    SCORE_COL,
    apply_hard_robustness_gate,
    classify_dataset,
    discover_prediction_datasets,
    is_rare_winner_eligible,
    is_robust_strategy_eligible,
    load_predictions,
    normalize_id_column,
    run_audit,
    selected_count_from_rows,
    select_validation_policy,
    top_fraction_from_percent,
    validate_top_pct_percent,
    AuditConfig,
    PredictionDataset,
    compute_tail_metrics_row,
    compute_robustness_on_selected,
    select_top_rows,
)


def _pred_df(n: int, *, split: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = 1 if i < max(1, n // 20) else 0
        score = float(label) + rng.random() * 0.1 + (n - i) * 1e-6
        ret = 0.2 if label else -0.03
        rows.append(
            {
                "pair_address": f"pair_{i % 5}",
                TARGET_COL: label,
                RETURN_COL: ret,
                SCORE_COL: score,
                "split": split,
                "filter": "RAW_ALL_VERIFIED",
                "horizon": "1h",
                "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                "candidate_id": f" cand_{i} ",
                "target_row_id": f" tr_{i} ",
            }
        )
    return pd.DataFrame(rows)


class ThresholdTests(unittest.TestCase):
    def test_top_pct_percent_values_positive(self) -> None:
        for pct in TOP_PCT_PERCENT_VALUES:
            self.assertGreater(pct, 0)
            self.assertLessEqual(pct, 100)

    def test_fraction_interpretation(self) -> None:
        self.assertAlmostEqual(top_fraction_from_percent(0.02), 0.0002)
        self.assertAlmostEqual(top_fraction_from_percent(1.00), 0.01)

    def test_selected_count_formula(self) -> None:
        self.assertEqual(selected_count_from_rows(1000, 0.02), 1)
        self.assertEqual(selected_count_from_rows(1000, 1.00), 10)
        self.assertEqual(selected_count_from_rows(1000, 10.00), 100)

    def test_rejects_non_positive_threshold(self) -> None:
        with self.assertRaises(ValueError):
            validate_top_pct_percent(0)
        with self.assertRaises(ValueError):
            validate_top_pct_percent(-1.0)
        with self.assertRaises(ValueError):
            validate_top_pct_percent(101)


class CentralizedOutputTests(unittest.TestCase):
    def test_single_validation_and_test_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "e8b_run"
            pred_dir = run_dir / "predictions"
            pred_dir.mkdir(parents=True)
            ds = "RAW_ALL_VERIFIED_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1"
            _pred_df(200, split="validation", seed=1).to_csv(
                pred_dir / f"{ds}_validation_predictions.csv", index=False
            )
            _pred_df(200, split="test", seed=2).to_csv(
                pred_dir / f"{ds}_test_predictions.csv", index=False
            )
            result = run_audit(AuditConfig(run_dir=run_dir))
            out = Path(result["output_dir"]) / "reports"
            self.assertTrue((out / "e8c_tail_metrics_validation.parquet").exists())
            self.assertTrue((out / "e8c_tail_metrics_test.parquet").exists())
            val = pd.read_csv(out / "e8c_tail_metrics_validation.csv")
            test = pd.read_csv(out / "e8c_tail_metrics_test.csv")
            self.assertEqual(len(val["top_pct_percent"].unique()), len(TOP_PCT_PERCENT_VALUES))
            self.assertEqual(len(test["top_pct_percent"].unique()), len(TOP_PCT_PERCENT_VALUES))
            extra = list(out.glob("*top_pct*"))
            self.assertEqual(extra, [])


class ValidationSelectionTests(unittest.TestCase):
    def test_validation_only_selection_applied_to_test(self) -> None:
        val_metrics = [
            {
                "dataset_name": "ds",
                "top_pct_percent": 1.0,
                "positive_rate": 0.01,
                "selected_count": 60,
                "selected_unique_pairs": 6,
                "selected_average_sim_net_return": 0.05,
                "selected_total_sim_net_return": 3.0,
                "precision_at_top_pct": 0.2,
                "selected_top_pair_share": 0.3,
                "economic_metrics_available": True,
            },
            {
                "dataset_name": "ds",
                "top_pct_percent": 5.0,
                "positive_rate": 0.01,
                "selected_count": 60,
                "selected_unique_pairs": 6,
                "selected_average_sim_net_return": 0.01,
                "selected_total_sim_net_return": 0.6,
                "precision_at_top_pct": 0.1,
                "selected_top_pair_share": 0.4,
                "economic_metrics_available": True,
            },
        ]
        robust = {
            ("ds", 1.0): {
                "remove_best_selected_trade_total_sim_net_return": 2.0,
                "remove_top_selected_pair_total_sim_net_return": 1.5,
            },
            ("ds", 5.0): {
                "remove_best_selected_trade_total_sim_net_return": 0.5,
                "remove_top_selected_pair_total_sim_net_return": 0.4,
            },
        }
        candidates, selected = select_validation_policy(val_metrics, robust)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(float(selected["top_pct_percent"]), 1.0)
        self.assertEqual(selected["selection_source"], "validation_only")


class HardRobustnessGateTests(unittest.TestCase):
    def test_remove_best_trade_lottery_or_unusable(self) -> None:
        self.assertEqual(
            apply_hard_robustness_gate(baseline_total=0, remove_best_total=1, remove_top_pair_total=1),
            "UNUSABLE_OFFLINE",
        )
        self.assertEqual(
            apply_hard_robustness_gate(baseline_total=5, remove_best_total=0, remove_top_pair_total=3),
            "LOTTERY_ARTIFACT",
        )

    def test_remove_top_pair_blocks_robust(self) -> None:
        self.assertEqual(
            apply_hard_robustness_gate(baseline_total=5, remove_best_total=4, remove_top_pair_total=0),
            "RARE_WINNER_DETECTOR",
        )

    def test_positive_removals_allow_robust_gate(self) -> None:
        self.assertEqual(
            apply_hard_robustness_gate(baseline_total=5, remove_best_total=4, remove_top_pair_total=2),
            "ROBUST_STRATEGY_CANDIDATE",
        )


class ClassificationTests(unittest.TestCase):
    def _dataset(self) -> PredictionDataset:
        return PredictionDataset(
            dataset_name="ds",
            validation_path=Path("v.csv"),
            test_path=Path("t.csv"),
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id="POL",
        )

    def test_robust_strategy_classification(self) -> None:
        val_row = {
            "top_pct_percent": 1.0,
            "selected_total_sim_net_return": 5.0,
            "precision_at_top_pct": 0.3,
            "candidate_type": "ROBUST_STRATEGY_CANDIDATE",
        }
        test_row = {
            "selected_total_sim_net_return": 4.0,
            "precision_at_top_pct": 0.25,
            "selected_count": 60,
            "selected_unique_pairs": 6,
            "selected_average_sim_net_return": 0.05,
            "selected_top_pair_share": 0.3,
            "positive_rate": 0.05,
            "economic_metrics_available": True,
        }
        val_robust = {
            "hard_robustness_gate_status": "ROBUST_STRATEGY_CANDIDATE",
            "remove_best_selected_trade_total_sim_net_return": 3.0,
            "remove_top_selected_pair_total_sim_net_return": 2.0,
        }
        test_robust = dict(val_robust)
        out = classify_dataset(
            dataset=self._dataset(),
            val_selected=val_row,
            test_row=test_row,
            val_robust=val_robust,
            test_robust=test_robust,
            val_df=pd.DataFrame({RETURN_COL: [0.1]}),
            test_df=pd.DataFrame({RETURN_COL: [0.1]}),
        )
        self.assertEqual(out["final_classification"], "ROBUST_STRATEGY_CANDIDATE")

    def test_lottery_artifact_classification(self) -> None:
        out = classify_dataset(
            dataset=self._dataset(),
            val_selected={"selected_total_sim_net_return": 2.0, "candidate_type": "RARE_WINNER_DETECTOR"},
            test_row={"selected_total_sim_net_return": 1.0, "precision_at_top_pct": 0.1},
            val_robust={"hard_robustness_gate_status": "LOTTERY_ARTIFACT"},
            test_robust={"hard_robustness_gate_status": "ROBUST_STRATEGY_CANDIDATE"},
            val_df=pd.DataFrame({RETURN_COL: [0.1]}),
            test_df=pd.DataFrame({RETURN_COL: [0.1]}),
        )
        self.assertEqual(out["final_classification"], "LOTTERY_ARTIFACT")

    def test_validation_only_artifact(self) -> None:
        out = classify_dataset(
            dataset=self._dataset(),
            val_selected={"selected_total_sim_net_return": 2.0, "candidate_type": "RARE_WINNER_DETECTOR"},
            test_row={"selected_total_sim_net_return": -1.0, "precision_at_top_pct": 0.0},
            val_robust={"hard_robustness_gate_status": "RARE_WINNER_DETECTOR"},
            test_robust={"hard_robustness_gate_status": "UNUSABLE_OFFLINE"},
            val_df=pd.DataFrame({RETURN_COL: [0.1]}),
            test_df=pd.DataFrame({RETURN_COL: [0.1]}),
        )
        self.assertEqual(out["final_classification"], "VALIDATION_ONLY_ARTIFACT")

    def test_no_usable_signal(self) -> None:
        out = classify_dataset(
            dataset=self._dataset(),
            val_selected=None,
            test_row=None,
            val_robust={},
            test_robust={},
            val_df=pd.DataFrame({RETURN_COL: [-0.1]}),
            test_df=pd.DataFrame({RETURN_COL: [-0.1]}),
        )
        self.assertEqual(out["final_classification"], "NO_USABLE_SIGNAL")


class IdentitySanityTests(unittest.TestCase):
    def test_strip_normalization(self) -> None:
        s = pd.Series([" abc ", "def"])
        out = normalize_id_column(s)
        self.assertEqual(out.iloc[0], "abc")

    def test_prediction_only_analysis_without_ids(self) -> None:
        df = _pred_df(20, split="validation").drop(columns=["candidate_id", "target_row_id"])
        loaded = load_predictions.__wrapped__ if hasattr(load_predictions, "__wrapped__") else None
        frame = df.reset_index(drop=True)
        frame["_row_order"] = np.arange(len(frame))
        row = compute_tail_metrics_row(
            frame,
            dataset=PredictionDataset("ds", Path("v"), Path("t")),
            split="validation",
            top_pct_percent=10.0,
        )
        self.assertGreater(row["selected_count"], 0)


class SmallDataTests(unittest.TestCase):
    def test_tiny_prediction_set(self) -> None:
        df = _pred_df(3, split="validation")
        row = compute_tail_metrics_row(
            df,
            dataset=PredictionDataset("ds", Path("v"), Path("t")),
            split="validation",
            top_pct_percent=1.0,
        )
        self.assertEqual(row["selected_count"], 1)

    def test_missing_pair_address(self) -> None:
        df = _pred_df(10, split="validation").drop(columns=["pair_address"])
        selected, _, _ = select_top_rows(df, 10.0)
        stats = compute_robustness_on_selected(selected, baseline_precision=0.5, baseline_total=1.0)
        self.assertIn("hard_robustness_gate_status", stats)

    def test_missing_sim_net_return(self) -> None:
        df = _pred_df(10, split="validation").drop(columns=[RETURN_COL])
        row = compute_tail_metrics_row(
            df,
            dataset=PredictionDataset("ds", Path("v"), Path("t")),
            split="validation",
            top_pct_percent=1.0,
        )
        self.assertFalse(row["economic_metrics_available"])
        self.assertIsNone(row["selected_total_sim_net_return"])


class DiscoveryTests(unittest.TestCase):
    def test_discover_prediction_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = "RAW_ALL_VERIFIED_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1"
            (root / f"{ds}_validation_predictions.csv").write_text("x", encoding="utf-8")
            (root / f"{ds}_test_predictions.csv").write_text("x", encoding="utf-8")
            found = discover_prediction_datasets(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].dataset_name, ds)


if __name__ == "__main__":
    unittest.main()
