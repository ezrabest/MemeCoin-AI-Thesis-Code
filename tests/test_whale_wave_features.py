"""Whale-wave features, labels, sizing, and vectorized builder tests."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.training.config import TRADE_CLASS_MULTIPLIERS
from app.training.snapshot_features import compute_snapshot_historical_features, sanitize
from app.training.vectorized_builder import build_training_datasets_vectorized
from app.training.wave_engine import (
    add_position_sizing_labels,
    add_whale_wave_score,
    attach_historical_features,
    compute_future_labels_for_coin,
    detect_pump_then_dump,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _snap_rows(coin_id: int, t0: datetime, prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    rows = []
    for i, price in enumerate(prices):
        vol = (volumes[i] if volumes else 1000.0 + i * 100)
        rows.append({
            "coin_id": coin_id,
            "timestamp": _iso(t0 + timedelta(minutes=5 * i)),
            "price": price,
            "liquidity": 50000.0,
            "volume_24h": vol,
            "txns_buys": 60,
            "txns_sells": 40,
            "buy_ratio": 0.6,
            "whale_score": 0.5,
            "price_change_h1": 1.0,
            "price_change_h24": 2.0,
            "chain": "solana",
            "pair_address": "0xwave",
        })
    return pd.DataFrame(rows)


class WhaleWaveFeatureTests(unittest.TestCase):
    def test_safe_division_zero_sells(self) -> None:
        out = sanitize(pd.Series([1.0]) / pd.Series([0.0]))
        self.assertTrue(pd.isna(out.iloc[0]) or np.isfinite(out.iloc[0]))

    def test_whale_wave_columns_exist(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        snaps = _snap_rows(1, t0, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        warnings: list[str] = []
        featured = compute_snapshot_historical_features(snaps, warnings)
        self.assertIn("volume_spike_ratio_15m_vs_1h", featured.columns)
        self.assertIn("buy_sell_ratio", featured.columns)

    def test_whale_wave_score_uses_historical_only(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        flat = [1.0] * 8
        pump_later = [1.0] * 8 + [5.0] * 8
        snaps_flat = _snap_rows(1, t0, flat)
        snaps_pump = _snap_rows(1, t0, pump_later)
        warnings: list[str] = []
        feat_flat = compute_snapshot_historical_features(snaps_flat, warnings)
        feat_pump = compute_snapshot_historical_features(snaps_pump, warnings)

        event = pd.DataFrame({
            "coin_id": [1],
            "pair_address": ["0xwave"],
            "event_timestamp": [_iso(t0)],
        })
        merged_flat = attach_historical_features(event, feat_flat)
        merged_pump = attach_historical_features(event, feat_pump)
        scored_flat = add_whale_wave_score(merged_flat)
        scored_pump = add_whale_wave_score(merged_pump)
        self.assertAlmostEqual(
            float(scored_flat["whale_wave_score"].iloc[0]),
            float(scored_pump["whale_wave_score"].iloc[0]),
            places=2,
        )


class AttachHistoricalFeaturesTests(unittest.TestCase):
    def test_multi_pair_non_monotonic_global_timestamps(self) -> None:
        t_a0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        t_b0 = datetime(2026, 6, 1, 13, 0, tzinfo=timezone.utc)
        snaps_a = _snap_rows(1, t_a0, [1.0, 1.1, 1.2])
        snaps_b = _snap_rows(2, t_b0, [2.0, 2.2, 2.4])
        snaps_a["pair_address"] = "pair_a"
        snaps_b["pair_address"] = "pair_b"
        warnings: list[str] = []
        featured = compute_snapshot_historical_features(
            pd.concat([snaps_a, snaps_b], ignore_index=True),
            warnings,
        )
        featured["ts"] = pd.to_datetime(featured["timestamp"], utc=True)

        events = pd.DataFrame({
            "coin_id": [1, 2, 1],
            "pair_address": ["pair_a", "pair_b", "pair_a"],
            "event_timestamp": [
                _iso(t_a0 + timedelta(minutes=10)),
                _iso(t_b0 + timedelta(minutes=5)),
                _iso(t_a0 + timedelta(minutes=5)),
            ],
        })
        merged = attach_historical_features(events, featured)
        self.assertEqual(len(merged), len(events))
        self.assertIn("volume_spike_ratio_15m_vs_1h", merged.columns)

        row_a_late = merged.iloc[0]
        row_b = merged.iloc[1]
        row_a_early = merged.iloc[2]
        self.assertEqual(float(row_a_late["price_usd"]), 1.2)
        self.assertEqual(float(row_a_early["price_usd"]), 1.1)
        self.assertEqual(float(row_b["price_usd"]), 2.2)

    def test_no_future_snapshot_attached(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        snaps = _snap_rows(1, t0, [1.0, 2.0, 3.0])
        snaps["pair_address"] = "pair_x"
        featured = compute_snapshot_historical_features(snaps, [])
        featured["ts"] = pd.to_datetime(featured["timestamp"], utc=True)
        event_time = t0 + timedelta(minutes=7)
        events = pd.DataFrame({
            "coin_id": [1],
            "pair_address": ["pair_x"],
            "event_timestamp": [_iso(event_time)],
        })
        merged = attach_historical_features(events, featured)
        self.assertEqual(float(merged["price_usd"].iloc[0]), 2.0)
        self.assertNotEqual(float(merged["price_usd"].iloc[0]), 3.0)

    def test_unmatched_events_preserved(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        snaps = _snap_rows(1, t0, [1.0, 1.1])
        snaps["pair_address"] = "pair_known"
        featured = compute_snapshot_historical_features(snaps, [])
        featured["ts"] = pd.to_datetime(featured["timestamp"], utc=True)
        events = pd.DataFrame({
            "coin_id": [1, 2],
            "pair_address": ["pair_known", "pair_missing"],
            "event_timestamp": [_iso(t0), _iso(t0)],
        })
        merged = attach_historical_features(events, featured)
        self.assertEqual(len(merged), 2)
        self.assertTrue(pd.isna(merged.loc[merged["pair_address"] == "pair_missing", "price_usd"]).iloc[0])


class WhaleWaveLabelTests(unittest.TestCase):
    def test_big_pump_and_dump_labels(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        prices = [100.0, 100.0, 130.0, 80.0, 60.0]
        snaps = _snap_rows(1, t0, prices)
        snaps["ts"] = pd.to_datetime(snaps["timestamp"], utc=True)
        events = pd.DataFrame({
            "coin_id": [1],
            "event_timestamp": [_iso(t0)],
            "ts": pd.to_datetime([_iso(t0)], utc=True),
            "price_usd": [100.0],
        })
        out = compute_future_labels_for_coin(events, snaps, fee_pct=0.03)
        self.assertTrue(bool(out["big_pump_15m"].iloc[0]))
        self.assertTrue(bool(out["big_dump_1h"].iloc[0]))

    def test_pump_then_dump_label(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        prices = [100.0, 100.0, 160.0, 120.0, 90.0]
        snaps = _snap_rows(1, t0, prices)
        snaps["ts"] = pd.to_datetime(snaps["timestamp"], utc=True)
        events = pd.DataFrame({
            "coin_id": [1],
            "event_timestamp": [_iso(t0)],
            "ts": pd.to_datetime([_iso(t0)], utc=True),
            "price_usd": [100.0],
        })
        out = compute_future_labels_for_coin(events, snaps, fee_pct=0.03)
        self.assertTrue(bool(out["pump_then_dump_1h"].iloc[0]))
        self.assertGreaterEqual(float(out["max_future_return_1h"].iloc[0]), 0.50)

    def test_negative_only_not_pump_then_dump(self) -> None:
        prices = np.array([100.0, 90.0, 80.0, 70.0])
        self.assertFalse(detect_pump_then_dump(prices, 100.0, pump_threshold=0.50, drop_threshold=-0.30))

    def test_flat_not_pump_then_dump(self) -> None:
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        self.assertFalse(detect_pump_then_dump(prices, 100.0, pump_threshold=0.50, drop_threshold=-0.30))

    def test_dump_before_pump_not_pump_then_dump(self) -> None:
        prices = np.array([100.0, 70.0, 90.0, 160.0])
        self.assertFalse(detect_pump_then_dump(prices, 100.0, pump_threshold=0.50, drop_threshold=-0.30))

    def test_pump_then_dump_implies_pump_threshold(self) -> None:
        t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        prices = [100.0, 100.0, 160.0, 100.0]
        snaps = _snap_rows(1, t0, prices)
        snaps["ts"] = pd.to_datetime(snaps["timestamp"], utc=True)
        events = pd.DataFrame({
            "coin_id": [1],
            "event_timestamp": [_iso(t0)],
            "ts": pd.to_datetime([_iso(t0)], utc=True),
            "price_usd": [100.0],
        })
        out = compute_future_labels_for_coin(events, snaps, fee_pct=0.03)
        if bool(out["pump_then_dump_1h"].iloc[0]):
            self.assertGreaterEqual(float(out["max_future_return_1h"].iloc[0]), 0.50)
        else:
            self.fail("expected pump_then_dump on pump-then-dump path")

    def test_position_sizing_multipliers(self) -> None:
        frame = pd.DataFrame({
            "whale_wave_score": [0.8, 0.5, 0.2, 0.9],
            "label_profitable_after_fees_1h": [True, True, True, False],
            "big_pump_1h": [False, False, False, True],
            "min_future_return_1h": [0.1, 0.1, 0.1, -0.5],
            "label_profitable_after_fees_4h": [True, True, True, False],
            "big_pump_4h": [False, False, False, True],
            "min_future_return_4h": [0.1, 0.1, 0.1, -0.5],
        })
        out = add_position_sizing_labels(frame)
        self.assertEqual(out["optimal_trade_class_1h"].iloc[0], "AGGRESSIVE_WHALE_TRADE")
        self.assertEqual(out["position_size_multiplier_1h"].iloc[0], TRADE_CLASS_MULTIPLIERS["AGGRESSIVE_WHALE_TRADE"])
        self.assertEqual(out["optimal_trade_class_1h"].iloc[2], "SMALL_PROBE")
        self.assertEqual(out["optimal_trade_class_1h"].iloc[3], "AVOID_DUMP")

    def test_non_profitable_not_aggressive_without_big_pump(self) -> None:
        frame = pd.DataFrame({
            "whale_wave_score": [0.95],
            "label_profitable_after_fees_1h": [False],
            "big_pump_1h": [False],
            "min_future_return_1h": [0.05],
            "label_profitable_after_fees_4h": [False],
            "big_pump_4h": [False],
            "min_future_return_4h": [0.05],
        })
        out = add_position_sizing_labels(frame)
        self.assertEqual(out["optimal_trade_class_1h"].iloc[0], "NO_TRADE")


class VectorizedBuilderSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._training_dir = Path(self._tmpdir.name) / "training"
        os.environ["TRADER_DB_PATH"] = str(self._db_path)

        import importlib
        import app.database as database

        importlib.reload(database)
        database.init_db()
        self.db = database

        t0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        coin = self.db.upsert_coin({
            "symbol": "WAVE/SOL",
            "pair_address": "0xwave",
            "chain": "solana",
            "price_usd": 1.0,
        })
        self.coin_id = coin["id"]
        for i, price in enumerate([1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.05, 1.0]):
            self.db.insert_market_snapshot({
                "coin_id": self.coin_id,
                "timestamp": _iso(t0 + timedelta(minutes=5 * i)),
                "price": price,
                "liquidity": 50000,
                "volume_24h": 10000 + i * 500,
                "txns_buys": 100,
                "txns_sells": 50,
                "buy_ratio": 0.67,
                "whale_score": 0.7,
                "price_change_h1": 2.0,
                "price_change_h24": 5.0,
                "chain": "solana",
                "pair_address": "0xwave",
            })
        self.db.insert_signal({
            "coin_id": self.coin_id,
            "symbol": "WAVE/SOL",
            "timestamp": _iso(t0),
            "signal_type": "BUY",
            "score": 0.7,
            "confidence": 0.7,
            "features_json": {"sentiment_score": 0.2},
        })

        before = self._table_counts()

        with patch("app.training.vectorized_builder.TRAINING_DIR", self._training_dir), patch(
            "app.training.dataset_builder.TRAINING_DIR", self._training_dir
        ), patch("app.models.predictor.generate_decision") as mock_ollama, patch(
            "app.models.predictor._gemini_json"
        ) as mock_gemini:
            report = build_training_datasets_vectorized()

        after = self._table_counts()
        self.assertEqual(before, after)
        mock_ollama.assert_not_called()
        mock_gemini.assert_not_called()

        summary = report["summary"]
        self.assertIn("build_stage_timings", summary)
        self.assertIn("big_pump_rate_1h", summary)
        self.assertIn("whale_wave_score", pd.read_parquet(self._training_dir / "signal_outcomes.parquet").columns)

        src = Path("app/training/vectorized_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("iterrows()", src)
        self.assertIn("merge_asof", src)

    def _table_counts(self) -> dict[str, int]:
        conn = sqlite3.connect(self._db_path)
        tables = ["market_snapshots", "signals", "gemini_decisions", "raw_provider_payloads"]
        counts = {}
        for table in tables:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return counts

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)


if __name__ == "__main__":
    unittest.main()
