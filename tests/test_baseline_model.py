"""Baseline ML training and predictive policy backtest tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.training.baseline_model import (
    build_preprocessor,
    chronological_split,
    precision_at_top_k,
    resolve_target_column,
    select_feature_columns,
    train_baseline_models,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _synthetic_dataset(n: int = 240) -> pd.DataFrame:
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = t0 + timedelta(minutes=i)
        price = 0.001 * (1 + 0.001 * i)
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


class BaselineModelUtilityTests(unittest.TestCase):
    def test_train_script_exists(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_model.py"
        self.assertTrue(path.is_file())

    def test_chronological_split_preserves_order(self) -> None:
        frame = _synthetic_dataset(100)
        train, val, test = chronological_split(frame)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        combined = pd.concat([train, val, test], ignore_index=True)
        self.assertTrue(combined["event_timestamp"].is_monotonic_increasing)

    def test_future_outcome_columns_excluded(self) -> None:
        frame = _synthetic_dataset(20)
        numeric, categorical, excluded = select_feature_columns(frame)
        self.assertNotIn("target_return_1h", numeric + categorical)
        self.assertNotIn("max_future_return_1h", numeric + categorical)
        self.assertNotIn("reasoning", numeric + categorical)
        self.assertIn("whale_wave_score", numeric)

    def test_simple_imputer_handles_nan(self) -> None:
        frame = _synthetic_dataset(30)
        numeric, categorical, _ = select_feature_columns(frame)
        frame.loc[0, "whale_wave_score"] = np.nan
        preprocessor = build_preprocessor(numeric, categorical)
        out = preprocessor.fit_transform(frame[numeric + categorical])
        self.assertFalse(np.isnan(out).any())

    def test_whale_wave_direction_encoded_or_excluded(self) -> None:
        frame = _synthetic_dataset(30)
        _, categorical, excluded = select_feature_columns(frame)
        self.assertIn("whale_wave_direction", categorical + excluded)

    def test_one_class_target_skipped(self) -> None:
        frame = _synthetic_dataset(80)
        frame["target_profitable_4h"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            models_dir = Path(tmp) / "models"
            frame.to_parquet(data_path, index=False)
            report = train_baseline_models(dataset_path=data_path, models_dir=models_dir)
        skipped = {s["target"]: s["reason"] for s in report["targets_skipped"]}
        self.assertEqual(skipped.get("label_profitable_after_fees_4h"), "fewer_than_two_classes")

    def test_precision_at_top_k(self) -> None:
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        top1 = precision_at_top_k(y_true, y_score, 10.0)
        self.assertEqual(top1, 1.0)
        top2 = precision_at_top_k(y_true, y_score, 20.0)
        self.assertEqual(top2, 0.5)

    def test_resolve_target_alias(self) -> None:
        frame = _synthetic_dataset(5)
        spec = {
            "name": "label_profitable_after_fees_1h",
            "aliases": ["target_profitable_1h"],
        }
        name, series = resolve_target_column(frame, spec)
        self.assertEqual(name, "label_profitable_after_fees_1h")
        self.assertIsNotNone(series)


class BaselineModelTrainingTests(unittest.TestCase):
    def test_training_writes_outputs_and_selects_best_by_pr_auc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            models_dir = Path(tmp) / "models"
            frame = _synthetic_dataset(240)
            frame.to_parquet(data_path, index=False)
            report = train_baseline_models(dataset_path=data_path, models_dir=models_dir)

            metrics_path = models_dir / "baseline_metrics.json"
            preds_test_path = models_dir / "predictions_test.parquet"
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(preds_test_path.is_file())
            self.assertGreater(len(report["targets_trained"]), 0)
            self.assertIn("best_model_by_target", report)

            for target, info in report["best_model_by_target"].items():
                self.assertIn("model_name", info)
                self.assertIn("best_validation_pr_auc", info)
                models = report["models_by_target"][target]["models"]
                best_name = info["model_name"]
                best_pr = info["best_validation_pr_auc"]
                val_prs = [
                    m["validation"]["pr_auc"]
                    for m in models.values()
                    if m["validation"]["pr_auc"] is not None
                ]
                if val_prs:
                    self.assertAlmostEqual(best_pr, max(val_prs), places=5)

    def test_no_sqlite_modification(self) -> None:
        db_path = Path(__file__).resolve().parents[1] / "data" / "trader.db"
        if not db_path.is_file():
            self.skipTest("trader.db not present")
        mtime_before = db_path.stat().st_mtime
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            models_dir = Path(tmp) / "models"
            _synthetic_dataset(120).to_parquet(data_path, index=False)
            train_baseline_models(dataset_path=data_path, models_dir=models_dir)
        self.assertEqual(db_path.stat().st_mtime, mtime_before)

    def test_no_gemini_or_ollama_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            models_dir = Path(tmp) / "models"
            _synthetic_dataset(120).to_parquet(data_path, index=False)
            with patch("google.generativeai.GenerativeModel") as gemini_mock:
                with patch("openai.OpenAI") as openai_mock:
                    train_baseline_models(dataset_path=data_path, models_dir=models_dir)
            gemini_mock.assert_not_called()
            openai_mock.assert_not_called()


class BaselineApiTests(unittest.TestCase):
    def test_api_returns_metrics_when_present(self) -> None:
        import importlib
        import main
        from fastapi.testclient import TestClient

        fake_metrics = {"targets_trained": ["label_profitable_after_fees_1h"]}
        importlib.reload(main)
        client = TestClient(main.app)
        with patch("app.training.baseline_model.load_baseline_metrics", return_value=fake_metrics):
            response = client.get("/api/debug/training-dataset/baseline-metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_api_not_trained_status(self) -> None:
        import importlib
        import main
        from fastapi.testclient import TestClient

        importlib.reload(main)
        client = TestClient(main.app)
        with patch("app.training.baseline_model.load_baseline_metrics", return_value=None):
            response = client.get("/api/debug/training-dataset/baseline-metrics")
        data = response.json()
        self.assertEqual(data["status"], "not_trained_yet")
        self.assertIn("train_baseline_model.py", data["manual_command"])


class PredictedPolicyBacktestTests(unittest.TestCase):
    def test_predicted_backtest_script_exists(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "backtest_predicted_policy.py"
        self.assertTrue(path.is_file())

    def test_predicted_backtest_marked_non_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir(parents=True)
            data_path = Path(tmp) / "data.parquet"
            frame = _synthetic_dataset(240)
            frame.to_parquet(data_path, index=False)
            train_baseline_models(dataset_path=data_path, models_dir=models_dir)

            import scripts.backtest_predicted_policy as bpp

            metrics = json.loads((models_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
            with patch.object(bpp, "load_baseline_metrics", return_value=metrics):
                with patch.object(bpp, "VALIDATION_PREDICTIONS_PATH", models_dir / "predictions_validation.parquet"):
                    with patch.object(bpp, "TEST_PREDICTIONS_PATH", models_dir / "predictions_test.parquet"):
                        report = bpp.run_backtest()

            self.assertFalse(report["is_oracle_backtest"])
            self.assertEqual(report["selection_method"], "validation_only")
            for policy in report["policies"].values():
                self.assertFalse(policy["validation"]["is_oracle_backtest"])
                self.assertFalse(policy["test"]["is_oracle_backtest"])


if __name__ == "__main__":
    unittest.main()
