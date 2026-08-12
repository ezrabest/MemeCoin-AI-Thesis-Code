"""RF + TabICLv2 matrix evaluation tests (no GPU, no full matrix run)."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.training.rf_tab_matrix import (
    COMBINATION_METHODS,
    compute_score_diagnostics,
    compute_stability_flag,
    discover_tab_suffixes,
    join_rf_tab_predictions,
    load_tab_metadata,
    prepare_prediction_frame,
    run_rf_tab_matrix,
    tab_report_path,
    write_rf_tab_matrix_outputs,
)
from scripts.evaluate_rf_tab_matrix import build_parser


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _prediction_rows(
    *,
    split: str,
    model_name: str,
    target_name: str,
    count: int,
    score_offset: float = 0.0,
    start_index: int = 0,
) -> list[dict]:
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i in range(count):
        idx = start_index + i
        ts = t0 + timedelta(minutes=idx)
        prob = max(0.0, min(1.0, 1.0 - (i / max(count, 1)) + score_offset))
        rows.append({
            "event_timestamp": _iso(ts),
            "symbol": f"COIN{idx % 5}",
            "pair_address": f"pair_{idx % 5}",
            "target_name": target_name,
            "y_true": int(i % 11 == 0),
            "predicted_probability": prob,
            "model_name": model_name,
            "split": split,
            "target_return_4h": 0.08 if i % 11 == 0 else -0.02,
            "future_return_4h": 0.08 if i % 11 == 0 else -0.02,
        })
    return rows


class RfTabJoinTests(unittest.TestCase):
    def test_key_merge_on_timestamp_and_pair(self) -> None:
        rf = pd.DataFrame(_prediction_rows(
            split="validation",
            model_name="random_forest",
            target_name="label_profitable_after_fees_4h",
            count=20,
        ))
        tab = pd.DataFrame(_prediction_rows(
            split="validation",
            model_name="tabicl_v2",
            target_name="target_profitable_4h",
            count=20,
            score_offset=0.05,
        ))
        rf_frame = prepare_prediction_frame(
            rf, model_name="random_forest",
            target_aliases=("label_profitable_after_fees_4h",),
            split="validation", score_col="rf_score",
        )
        tab_frame = prepare_prediction_frame(
            tab, model_name="tabicl_v2",
            target_aliases=("target_profitable_4h",),
            split="validation", score_col="tab_score",
        )
        joined, meta = join_rf_tab_predictions(rf_frame, tab_frame)
        self.assertEqual(len(joined), 20)
        self.assertEqual(meta["join_strategy_used"], "key_merge")
        self.assertIn("pair_address", meta["join_keys"])

    def test_row_order_fallback_requires_alignment(self) -> None:
        rf_rows = _prediction_rows(
            split="validation",
            model_name="random_forest",
            target_name="label_profitable_after_fees_4h",
            count=10,
        )
        tab_rows = _prediction_rows(
            split="validation",
            model_name="tabicl_v2",
            target_name="target_profitable_4h",
            count=10,
        )
        tab_rows[3]["event_timestamp"] = _iso(datetime(2099, 1, 1, tzinfo=timezone.utc))
        rf = pd.DataFrame(rf_rows)
        tab = pd.DataFrame(tab_rows)
        rf_frame = prepare_prediction_frame(
            rf, model_name="random_forest",
            target_aliases=("label_profitable_after_fees_4h",),
            split="validation", score_col="rf_score",
        )
        tab_frame = prepare_prediction_frame(
            tab, model_name="tabicl_v2",
            target_aliases=("target_profitable_4h",),
            split="validation", score_col="tab_score",
        )
        with self.assertRaises(ValueError):
            join_rf_tab_predictions(rf_frame, tab_frame, mismatch_threshold=0.0)

    def test_row_order_fallback_allowed_when_aligned(self) -> None:
        t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        rf_rows = []
        tab_rows = []
        for i in range(4):
            ts = t0 + timedelta(minutes=i // 2)
            rf_rows.append({
                "event_timestamp": _iso(ts),
                "symbol": f"COIN{i}",
                "pair_address": f"pair_{i}",
                "target_name": "label_profitable_after_fees_4h",
                "y_true": int(i % 2 == 0),
                "predicted_probability": 0.9 - i * 0.1,
                "model_name": "random_forest",
                "split": "validation",
                "target_return_4h": 0.05,
            })
            tab_rows.append({
                "event_timestamp": _iso(ts),
                "target_name": "target_profitable_4h",
                "y_true": int(i % 2 == 0),
                "predicted_probability": 0.8 - i * 0.1,
                "model_name": "tabicl_v2",
                "split": "validation",
                "target_return_4h": 0.05,
            })
        rf = pd.DataFrame(rf_rows)
        tab = pd.DataFrame(tab_rows)
        rf_frame = prepare_prediction_frame(
            rf, model_name="random_forest",
            target_aliases=("label_profitable_after_fees_4h",),
            split="validation", score_col="rf_score",
        )
        tab_frame = prepare_prediction_frame(
            tab, model_name="tabicl_v2",
            target_aliases=("target_profitable_4h",),
            split="validation", score_col="tab_score",
        )
        joined, meta = join_rf_tab_predictions(rf_frame, tab_frame, mismatch_threshold=0.0)
        self.assertEqual(len(joined), 4)
        self.assertEqual(meta["join_strategy_used"], "row_order_fallback")
        self.assertEqual(meta["timestamp_mismatch_rate"], 0.0)


class RfTabDiagnosticsTests(unittest.TestCase):
    def test_constant_tab_score_flag(self) -> None:
        diag = compute_score_diagnostics(
            np.array([0.2, 0.5, 0.8]),
            np.array([0.42, 0.42, 0.42]),
        )
        self.assertTrue(diag["tab_constant_score_flag"])
        self.assertEqual(diag["tab_unique_score_count"], 1)

    def test_stability_flags(self) -> None:
        self.assertEqual(
            compute_stability_flag(
                {"total_return_4h": 1.0, "selected_trade_count": 10},
                {"total_return_4h": 0.5, "selected_trade_count": 8},
            ),
            "stable_positive",
        )
        self.assertEqual(
            compute_stability_flag(
                {"total_return_4h": 1.0, "selected_trade_count": 10},
                {"total_return_4h": -0.1, "selected_trade_count": 8},
            ),
            "validation_only",
        )
        self.assertEqual(
            compute_stability_flag(
                {"total_return_4h": 1.0, "selected_trade_count": 2},
                {"total_return_4h": 1.0, "selected_trade_count": 8},
            ),
            "unstable",
        )


class RfTabMetadataTests(unittest.TestCase):
    def test_load_tab_metadata_extracts_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backtest_dir = Path(tmp)
            report_path = tab_report_path(backtest_dir, "positive_enriched_4096")
            payload = {
                "context_strategy": "positive_enriched",
                "context_size_used": 4096,
                "batch_size_used": 256,
                "max_train_context_rows": 4096,
                "max_features": None,
                "feature_count": 48,
                "knn_rolling_days_used": 14,
                "knn_time_decay_alpha": 0.0,
                "positive_context_ratio": 0.5,
                "full_evaluation": True,
                "output_label": "positive_enriched_4096",
            }
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            meta = load_tab_metadata(report_path)
            self.assertTrue(meta["report_available"])
            self.assertEqual(meta["context_strategy"], "positive_enriched")
            self.assertEqual(meta["output_label"], "positive_enriched_4096")


class RfTabMatrixSmokeTests(unittest.TestCase):
    def test_synthetic_matrix_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            backtest_dir = root / "policy_backtests"
            models_dir.mkdir(parents=True)
            backtest_dir.mkdir(parents=True)

            suffix = "smoke_suffix"
            val_rf = pd.DataFrame(_prediction_rows(
                split="validation",
                model_name="random_forest",
                target_name="label_profitable_after_fees_4h",
                count=120,
            ))
            test_rf = pd.DataFrame(_prediction_rows(
                split="test",
                model_name="random_forest",
                target_name="label_profitable_after_fees_4h",
                count=60,
                start_index=120,
            ))
            val_tab = pd.DataFrame(_prediction_rows(
                split="validation",
                model_name="tabicl_v2",
                target_name="target_profitable_4h",
                count=120,
                score_offset=0.03,
            ))
            test_tab = pd.DataFrame(_prediction_rows(
                split="test",
                model_name="tabicl_v2",
                target_name="target_profitable_4h",
                count=60,
                score_offset=0.03,
                start_index=120,
            ))
            val_rf.to_parquet(models_dir / "predictions_validation.parquet", index=False)
            test_rf.to_parquet(models_dir / "predictions_test.parquet", index=False)
            val_tab.to_parquet(
                models_dir / f"tabicl_v2_predictions_validation_{suffix}.parquet",
                index=False,
            )
            test_tab.to_parquet(
                models_dir / f"tabicl_v2_predictions_test_{suffix}.parquet",
                index=False,
            )
            with open(backtest_dir / f"tabicl_v2_report_{suffix}.json", "w", encoding="utf-8") as handle:
                json.dump({
                    "context_strategy": "stratified_recent",
                    "context_size_used": 1024,
                    "batch_size_used": 64,
                    "max_train_context_rows": 1024,
                    "feature_count": 10,
                    "full_evaluation": False,
                    "output_label": suffix,
                }, handle)

            discovered = discover_tab_suffixes(models_dir)
            self.assertEqual(discovered, [suffix])

            report = run_rf_tab_matrix(
                models_dir=models_dir,
                backtest_dir=backtest_dir,
                tab_suffixes=[suffix],
                combination_methods=["average", "rf_only"],
            )
            self.assertEqual(report["candidates_evaluated"], 2)
            candidate = report["candidates"][0]
            self.assertIn("rf_score_mean", candidate["diagnostics"])
            self.assertIn("tab_constant_score_flag", candidate["diagnostics"])
            self.assertIn("best_policy", candidate)
            best = candidate["best_policy"]
            self.assertIn("test_total_return_4h", best)
            self.assertIn("stability_flag", best)
            for policy in candidate["policies"]:
                self.assertIn("selected_trade_count", policy["validation"])
                self.assertIn("selected_trade_count", policy["test"])

            outputs = write_rf_tab_matrix_outputs(
                report,
                report_path=backtest_dir / "rf_tab_matrix_report.json",
                grid_path=backtest_dir / "rf_tab_matrix_grid.parquet",
            )
            self.assertTrue(Path(outputs["report_json"]).is_file())
            self.assertTrue(Path(outputs["grid_parquet"]).is_file())

    def test_cli_help(self) -> None:
        parser = build_parser()
        self.assertIn("--list-suffixes", parser.format_help())
        self.assertIn("--join-mismatch-threshold", parser.format_help())


if __name__ == "__main__":
    unittest.main()
