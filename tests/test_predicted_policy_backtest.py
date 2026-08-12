"""Predictive policy backtest tests — validation-only threshold selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.training.baseline_model import train_baseline_models
import scripts.backtest_predicted_policy as bpp


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _synthetic_dataset(n: int = 240) -> pd.DataFrame:
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = t0 + timedelta(minutes=i)
        rows.append({
            "event_timestamp": _iso(ts),
            "symbol": f"COIN{i % 5}",
            "pair_address": f"pair_{i % 5}",
            "coin_id": i % 5,
            "whale_wave_score": 0.4 + (i % 10) * 0.02,
            "whale_wave_direction": ["UP", "DOWN", "MIXED", "UNKNOWN"][i % 4],
            "volume_spike_ratio_15m_vs_1h": 1.0 + (i % 7) * 0.1,
            "volume_zscore_1h": float(i % 3),
            "buy_sell_ratio": 1.0 + (i % 5) * 0.05,
            "buy_pressure": 0.5,
            "sell_pressure": 0.5,
            "buy_sell_imbalance": 0.1 * (i % 3),
            "txn_velocity_15m": 10 + i % 4,
            "price_return_5m": 0.01 * (i % 3),
            "price_return_15m": 0.02 * (i % 3),
            "price_return_1h": 0.03 * (i % 3),
            "price_velocity_5m": 0.001,
            "price_velocity_15m": 0.002,
            "price_acceleration_5m_to_15m": 0.0001,
            "liquidity_usd": 10000 + i,
            "liquidity_change_15m": 0.01,
            "liquidity_change_1h": 0.02,
            "liquidity_shock_score": 0.1,
            "liquidity_to_volume_ratio": 0.5,
            "whale_score": 0.3,
            "target_profitable_1h": int(i % 17 == 0),
            "target_profitable_4h": int(i % 23 == 0),
            "big_pump_1h": int(i % 41 == 0),
            "big_pump_4h": int(i % 47 == 0),
            "optimal_trade_class_1h": "AGGRESSIVE_WHALE_TRADE" if i % 59 == 0 else "NO_TRADE",
            "optimal_trade_class_4h": "NO_TRADE",
            "target_return_1h": 0.05 if i % 17 == 0 else -0.01,
            "target_return_4h": 0.08 if i % 23 == 0 else -0.02,
            "future_return_1h": 0.05 if i % 17 == 0 else -0.01,
            "max_future_return_1h": 0.2,
            "big_pump_1h_leak": 0,
            "reasoning": "should be excluded",
            "action": "BUY",
        })
    return pd.DataFrame(rows)


def _synthetic_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for split, bucket, count in (("validation", val_rows, 200), ("test", test_rows, 100)):
        for i in range(count):
            prob = 1.0 - (i / count)
            for target_name, return_col in (
                (bpp.PRIMARY_TARGET, "target_return_4h"),
                (bpp.SECONDARY_TARGET, "target_return_1h"),
            ):
                bucket.append({
                    "event_timestamp": _iso(t0 + timedelta(minutes=i)),
                    "symbol": f"COIN{i % 5}",
                    "pair_address": f"pair_{i % 5}",
                    "target_name": target_name,
                    "y_true": int(prob > 0.9),
                    "predicted_probability": prob,
                    "model_name": "logistic_regression",
                    "split": split,
                    return_col: 0.10 if prob > 0.95 else -0.02,
                    "whale_wave_score": 0.5 + prob * 0.1,
                })
    return pd.DataFrame(val_rows), pd.DataFrame(test_rows)


class PolicyUtilityTests(unittest.TestCase):
    def test_top_percent_cutoff_selects_expected_validation_rows(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cutoff = bpp.top_percent_probability_cutoff(probs, 10.0)
        self.assertEqual(cutoff, 1.0)
        selected = (probs >= cutoff).sum()
        self.assertEqual(selected, 1)

        cutoff_20 = bpp.top_percent_probability_cutoff(probs, 20.0)
        self.assertEqual(cutoff_20, 0.9)
        self.assertEqual((probs >= cutoff_20).sum(), 2)

    def test_validation_cutoff_carried_to_test(self) -> None:
        val_probs = np.linspace(0.01, 1.0, 100)
        test_probs = np.linspace(0.01, 0.5, 50)
        val_cutoff = bpp.top_percent_probability_cutoff(val_probs, 5.0)
        val_selected = (val_probs >= val_cutoff).sum()
        test_selected = (test_probs >= val_cutoff).sum()
        self.assertEqual(val_selected, 5)
        self.assertLess(test_selected, val_selected)

    def test_test_set_not_used_for_rank_cutoff(self) -> None:
        val_probs = np.array([0.2, 0.4, 0.6, 0.8])
        test_probs = np.array([0.9, 0.95, 0.99, 1.0])
        val_cutoff = bpp.top_percent_probability_cutoff(val_probs, 25.0)
        self.assertEqual(val_cutoff, 0.8)
        self.assertEqual((test_probs >= val_cutoff).sum(), 4)

    def test_zero_trade_policy_does_not_crash(self) -> None:
        frame = pd.DataFrame({
            "event_timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["A", "B", "C"],
            "target_return_4h": [0.1, -0.1, 0.2],
            "y_true": [1, 0, 1],
            bpp.PRIMARY_TARGET: [0.1, 0.2, 0.3],
        })
        mask = pd.Series([False, False, False], index=frame.index)
        result = bpp.evaluate_policy_trades(
            frame,
            mask,
            return_col="target_return_4h",
            label_col="y_true",
            prob_col=bpp.PRIMARY_TARGET,
            fee_pct=0.03,
        )
        self.assertEqual(result["trade_count"], 0)
        self.assertTrue(result["zero_trade_policy"])
        self.assertEqual(result["total_return_after_fees"], 0.0)
        self.assertEqual(result["max_drawdown"], 0.0)
        self.assertIsNone(result["profit_factor"])

    def test_profit_factor_zero_losses_safe(self) -> None:
        self.assertIsNone(bpp.profit_factor_from_net(np.array([])))
        self.assertIsNone(bpp.profit_factor_from_net(np.array([0.5, 0.3])))
        self.assertEqual(bpp.profit_factor_from_net(np.array([-0.5, -0.2])), 0.0)

    def test_max_drawdown_empty_and_single_trade(self) -> None:
        self.assertEqual(bpp.max_drawdown_from_net(np.array([])), 0.0)
        self.assertEqual(bpp.max_drawdown_from_net(np.array([0.5])), 0.0)
        self.assertEqual(bpp.max_drawdown_from_net(np.array([-0.4])), -0.4)

    def test_select_best_validation_policy_prefers_return(self) -> None:
        rows = [
            {"policy_name": "a", "trade_count": 10, "total_return_after_fees": 1.0, "max_drawdown": -2.0, "profit_factor": 1.1, "win_rate": 0.5},
            {"policy_name": "b", "trade_count": 5, "total_return_after_fees": 3.0, "max_drawdown": -1.0, "profit_factor": 1.0, "win_rate": 0.4},
        ]
        best = bpp.select_best_validation_policy(rows)
        self.assertEqual(best["policy_name"], "b")


class PredictedPolicyBacktestIntegrationTests(unittest.TestCase):
    def test_report_marks_non_oracle_and_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir(parents=True)
            data_path = Path(tmp) / "data.parquet"
            frame = _synthetic_dataset(240)
            frame.to_parquet(data_path, index=False)
            train_baseline_models(dataset_path=data_path, models_dir=models_dir)

            metrics = json.loads((models_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
            with patch.object(bpp, "load_baseline_metrics", return_value=metrics):
                with patch.object(bpp, "VALIDATION_PREDICTIONS_PATH", models_dir / "predictions_validation.parquet"):
                    with patch.object(bpp, "TEST_PREDICTIONS_PATH", models_dir / "predictions_test.parquet"):
                        report = bpp.run_backtest()

            self.assertFalse(report["is_oracle_backtest"])
            self.assertEqual(report["selection_method"], "validation_only")
            self.assertTrue(report["test_set_never_used_for_threshold_selection"])
            self.assertTrue(report["rank_cutoffs_derived_from_validation"])
            self.assertIn("selected_policy", report)
            self.assertIn("comparison_to_legacy_threshold_0_20", report)

    def test_no_gemini_or_ollama_calls(self) -> None:
        source = Path(__file__).resolve().parents[1] / "scripts" / "backtest_predicted_policy.py"
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("gemini", text)
        self.assertNotIn("ollama", text)

    def test_main_sqlite_not_modified(self) -> None:
        db_candidates = list(Path(__file__).resolve().parents[1].glob("*.db"))
        if not db_candidates:
            self.skipTest("No main sqlite database present.")
        db_path = db_candidates[0]
        before = db_path.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir(parents=True)
            val_preds, test_preds = _synthetic_predictions()
            val_preds.to_parquet(models_dir / "predictions_validation.parquet", index=False)
            test_preds.to_parquet(models_dir / "predictions_test.parquet", index=False)
            metrics = {
                "best_model_by_target": {
                    bpp.PRIMARY_TARGET: {"model_name": "logistic_regression"},
                    bpp.SECONDARY_TARGET: {"model_name": "logistic_regression"},
                }
            }
            with patch.object(bpp, "load_baseline_metrics", return_value=metrics):
                with patch.object(bpp, "VALIDATION_PREDICTIONS_PATH", models_dir / "predictions_validation.parquet"):
                    with patch.object(bpp, "TEST_PREDICTIONS_PATH", models_dir / "predictions_test.parquet"):
                        bpp.run_backtest()
        after = db_path.read_bytes()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
