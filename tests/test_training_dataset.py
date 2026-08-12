"""Training dataset builder, outcome labeling, and API tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.training.dataset_builder import (
    TRAINING_DIR,
    build_model_ready_dataset,
    build_training_datasets,
    load_training_summary,
)
from app.training.outcome_labeler import label_outcomes, parse_timestamp


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


class OutcomeLabelerTests(unittest.TestCase):
    def test_future_return_calculation(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        snapshots = [
            {"timestamp": _iso(t0), "price": 100.0},
            {"timestamp": _iso(t0 + timedelta(minutes=15)), "price": 110.0},
            {"timestamp": _iso(t0 + timedelta(minutes=60)), "price": 105.0},
            {"timestamp": _iso(t0 + timedelta(minutes=240)), "price": 120.0},
        ]
        out = label_outcomes(snapshots, t0, fee_pct=0.03)
        self.assertAlmostEqual(out["future_return_15m"], 0.10, places=4)
        self.assertAlmostEqual(out["future_return_1h"], 0.05, places=4)
        self.assertAlmostEqual(out["future_return_4h"], 0.20, places=4)
        self.assertTrue(out["label_up_15m"])
        self.assertTrue(out["label_profitable_after_fees_15m"])
        self.assertFalse(out["pending_outcome"])

    def test_pending_when_future_snapshots_missing(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        snapshots = [{"timestamp": _iso(t0), "price": 50.0}]
        out = label_outcomes(snapshots, t0)
        self.assertTrue(out["pending_outcome"])
        self.assertIsNone(out["future_return_15m"])
        self.assertIsNone(out["future_return_1h"])
        self.assertIsNone(out["future_return_4h"])

    def test_model_ready_excludes_pending(self) -> None:
        ready = {"signal_id": 1, "pending_outcome": False, "future_return_15m": 0.01,
                 "future_return_1h": 0.02, "future_return_4h": 0.03,
                 "label_profitable_after_fees_15m": False,
                 "label_profitable_after_fees_1h": False,
                 "label_profitable_after_fees_4h": False}
        pending = {"signal_id": 2, "pending_outcome": True}
        model = build_model_ready_dataset([ready, pending], [])
        self.assertEqual(len(model), 1)
        self.assertEqual(model[0]["source_id"], 1)


class TrainingDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._training_dir = Path(self._tmpdir.name) / "training"
        self._db_path = Path(self._tmpdir.name) / "test.db"
        os.environ["TRADER_DB_PATH"] = str(self._db_path)

        import importlib
        import app.database as database

        importlib.reload(database)
        database.init_db()
        self.db = database

        t0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        coin = self.db.upsert_coin({
            "symbol": "TRAIN/SOL",
            "pair_address": "0xtrain",
            "chain": "solana",
            "price_usd": 1.0,
        })
        self.coin_id = coin["id"]
        for minutes, price in ((0, 1.0), (15, 1.1), (60, 1.05), (240, 1.2)):
            ts = _iso(t0 + timedelta(minutes=minutes))
            self.db.insert_market_snapshot({
                "coin_id": self.coin_id,
                "timestamp": ts,
                "price": price,
                "liquidity": 50000,
                "volume_24h": 100000,
                "txns_buys": 100,
                "txns_sells": 80,
                "buy_ratio": 0.55,
                "whale_score": 0.6,
                "price_change_h1": 2.0,
                "price_change_h24": 5.0,
                "chain": "solana",
                "pair_address": "0xtrain",
            })

        event_ts = _iso(t0)
        self.db.insert_signal({
            "coin_id": self.coin_id,
            "symbol": "TRAIN/SOL",
            "timestamp": event_ts,
            "signal_type": "BUY",
            "score": 0.7,
            "confidence": 0.7,
            "reason": "test signal",
            "features_json": {"sentiment_score": 0.2},
        })
        self.db.insert_gemini_decision({
            "coin_id": self.coin_id,
            "symbol": "TRAIN/SOL",
            "timestamp": event_ts,
            "action": "BUY",
            "confidence": 0.8,
            "rationale": "test llm",
            "strategy_type": "MOMENTUM",
            "risk_score": 50,
            "trigger_type": "test",
            "provider": "ollama",
            "model_source": "ollama_qwen3_8b",
            "input_context_json": {"symbol": "TRAIN/SOL"},
            "gemini_response_json": {"action": "BUY"},
        })

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_builder_runs_without_error(self) -> None:
        with patch("app.training.dataset_builder.TRAINING_DIR", self._training_dir):
            report = build_training_datasets()
        self.assertEqual(report["signal_rows_processed"], 1)
        self.assertEqual(report["llm_rows_processed"], 1)
        self.assertGreaterEqual(report["ready_rows"], 1)
        summary_path = self._training_dir / "training_dataset_summary.json"
        self.assertTrue(summary_path.is_file())

    def test_api_returns_summary_when_present(self) -> None:
        with patch("app.training.dataset_builder.TRAINING_DIR", self._training_dir):
            build_training_datasets()
        summary = load_training_summary(self._training_dir / "training_dataset_summary.json")
        self.assertIsNotNone(summary)
        self.assertIn("rows_model_ready", summary)

        import importlib
        import main
        from fastapi.testclient import TestClient

        importlib.reload(main)
        client = TestClient(main.app)
        with patch("app.training.dataset_builder.load_training_summary", return_value=summary):
            response = client.get("/api/debug/training-dataset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["rows_signal_total"], 1)

    def test_api_not_built_status(self) -> None:
        import importlib
        import main
        from fastapi.testclient import TestClient

        importlib.reload(main)
        client = TestClient(main.app)
        with patch("app.training.dataset_builder.load_training_summary", return_value=None):
            response = client.get("/api/debug/training-dataset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "not_built_yet")


if __name__ == "__main__":
    unittest.main()
