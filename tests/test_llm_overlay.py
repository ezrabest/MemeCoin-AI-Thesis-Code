"""Offline stored-LLM overlay evaluation tests."""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.training.llm_overlay import (
    JOIN_TOLERANCE_SECONDS,
    OUTCOME_HORIZON_HOURS,
    build_overlay_mask,
    enrich_decision_row,
    is_qwen_ollama_decision,
    join_llm_decisions_to_candidates,
    load_stored_llm_decisions,
    run_llm_overlay_evaluation,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _create_test_db(path: Path, decisions: list[dict], coins: list[dict] | None = None) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE coins (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            pair_address TEXT UNIQUE
        );
        CREATE TABLE gemini_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            coin_id INTEGER,
            symbol TEXT,
            action TEXT,
            confidence REAL,
            risk_score INTEGER,
            provider TEXT,
            model_source TEXT,
            gemini_response_json TEXT,
            input_context_json TEXT
        );
        """
    )
    for coin in coins or []:
        conn.execute(
            "INSERT INTO coins (id, symbol, pair_address) VALUES (?, ?, ?)",
            (coin["id"], coin["symbol"], coin["pair_address"]),
        )
    for dec in decisions:
        conn.execute(
            """
            INSERT INTO gemini_decisions
            (timestamp, coin_id, symbol, action, confidence, risk_score, provider, model_source, gemini_response_json, input_context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dec.get("timestamp"),
                dec.get("coin_id"),
                dec.get("symbol"),
                dec.get("action"),
                dec.get("confidence"),
                dec.get("risk_score"),
                dec.get("provider"),
                dec.get("model_source"),
                json.dumps(dec.get("gemini_response_json")) if isinstance(dec.get("gemini_response_json"), dict) else dec.get("gemini_response_json"),
                json.dumps(dec.get("input_context_json")) if isinstance(dec.get("input_context_json"), dict) else dec.get("input_context_json"),
            ),
        )
    conn.commit()
    conn.close()


class LlmOverlaySchemaTests(unittest.TestCase):
    def test_qwen_ollama_identified_from_provider(self) -> None:
        row = {"provider": "ollama", "model_source": "qwen3:8b"}
        self.assertTrue(is_qwen_ollama_decision(row))

    def test_missing_optional_columns_do_not_crash(self) -> None:
        row = enrich_decision_row({"action": "BUY"}, ["action"])
        self.assertEqual(row["parsed_action"], "BUY")
        self.assertIsNone(row["parsed_confidence"])
        self.assertIsNone(row["parsed_risk_score"])

    def test_action_parsed_from_response_json_fallback(self) -> None:
        row = enrich_decision_row(
            {"gemini_response_json": json.dumps({"decision": "SELL", "confidence": 0.8})},
            ["gemini_response_json"],
        )
        self.assertEqual(row["parsed_action"], "SELL")
        self.assertEqual(row["parsed_confidence"], 0.8)
        self.assertIn("action_from_gemini_response_json.decision", row["_schema_fallbacks"])


class LlmOverlayJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    def _candidates(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_direct_decision_id_join_preferred(self) -> None:
        candidates = self._candidates([
            {"event_timestamp": self.t0, "pair_address": "pair_a", "decision_id": 7},
        ])
        decisions = pd.DataFrame([
            {
                "decision_id": 7,
                "decision_timestamp": self.t0 + timedelta(seconds=30),
                "pair_address": "pair_a",
                "parsed_action": "BUY",
                "is_qwen_ollama": True,
                "llm_matched": True,
            }
        ])
        with patch(
            "app.training.llm_overlay._try_direct_id_join",
            return_value=(candidates.merge(decisions, on="decision_id", how="left"), "direct_decision_id"),
        ):
            joined, meta = join_llm_decisions_to_candidates(candidates, decisions)
        self.assertEqual(meta["join_strategy_used"], "direct_decision_id")

    def test_pair_time_join_respects_tolerance(self) -> None:
        candidates = self._candidates([
            {"event_timestamp": self.t0, "pair_address": "pair_a", "coin_id": 1},
        ])
        decisions = pd.DataFrame([
            {
                "decision_id": 1,
                "decision_timestamp": self.t0 + timedelta(seconds=90),
                "pair_address": "pair_a",
                "coin_id": 1,
                "parsed_action": "BUY",
                "is_qwen_ollama": True,
            }
        ])
        joined, meta = join_llm_decisions_to_candidates(candidates, decisions)
        self.assertEqual(meta["join_strategy_used"], "pair_time_merge_asof_forward")
        self.assertTrue(bool(joined["llm_matched"].iloc[0]))

    def test_pair_time_join_rejects_beyond_tolerance(self) -> None:
        candidates = self._candidates([
            {"event_timestamp": self.t0, "pair_address": "pair_a", "coin_id": 1},
        ])
        decisions = pd.DataFrame([
            {
                "decision_id": 1,
                "decision_timestamp": self.t0 + timedelta(minutes=5),
                "pair_address": "pair_a",
                "coin_id": 1,
                "parsed_action": "BUY",
                "is_qwen_ollama": True,
            }
        ])
        joined, _meta = join_llm_decisions_to_candidates(candidates, decisions)
        self.assertFalse(bool(joined["llm_matched"].iloc[0]))

    def test_pair_time_join_never_matches_different_pair(self) -> None:
        candidates = self._candidates([
            {"event_timestamp": self.t0, "pair_address": "pair_a", "coin_id": 1},
        ])
        decisions = pd.DataFrame([
            {
                "decision_id": 1,
                "decision_timestamp": self.t0 + timedelta(seconds=30),
                "pair_address": "pair_b",
                "coin_id": 2,
                "parsed_action": "BUY",
                "is_qwen_ollama": True,
            }
        ])
        joined, _meta = join_llm_decisions_to_candidates(candidates, decisions)
        self.assertFalse(bool(joined["llm_matched"].iloc[0]))

    def test_decision_after_outcome_horizon_not_attached(self) -> None:
        candidates = self._candidates([
            {"event_timestamp": self.t0, "pair_address": "pair_a", "coin_id": 1},
        ])
        decisions = pd.DataFrame([
            {
                "decision_id": 1,
                "decision_timestamp": self.t0 + timedelta(hours=OUTCOME_HORIZON_HOURS + 1),
                "pair_address": "pair_a",
                "coin_id": 1,
                "parsed_action": "BUY",
                "is_qwen_ollama": True,
            }
        ])
        joined, _meta = join_llm_decisions_to_candidates(
            candidates,
            decisions,
            tolerance_seconds=JOIN_TOLERANCE_SECONDS * 60,
        )
        self.assertFalse(bool(joined["llm_matched"].iloc[0]))


class LlmOverlayEvaluationTests(unittest.TestCase):
    def test_missing_predictions_stop_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                run_llm_overlay_evaluation(
                    validation_predictions_path=tmp_path / "missing_val.parquet",
                    test_predictions_path=tmp_path / "missing_test.parquet",
                    db_path=tmp_path / "db.sqlite",
                )
            self.assertIn("automatic retraining is disabled", str(ctx.exception))

    def test_no_gemini_or_ollama_calls_in_scripts(self) -> None:
        for rel in ("scripts/evaluate_llm_overlay.py", "app/training/llm_overlay.py"):
            text = (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8").lower()
            self.assertNotIn("generate_decision", text)
            self.assertNotIn("call_gemini", text)
            self.assertNotIn("ollama.generate", text)

    def test_report_flags_non_oracle_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models_dir = tmp_path / "models"
            models_dir.mkdir()
            db_path = tmp_path / "trader.db"
            _create_test_db(
                db_path,
                decisions=[{
                    "timestamp": _iso(datetime(2026, 6, 1, 12, 0, 30, tzinfo=timezone.utc)),
                    "coin_id": 1,
                    "symbol": "AAA",
                    "action": "BUY",
                    "confidence": 0.9,
                    "risk_score": 20,
                    "provider": "ollama",
                    "model_source": "qwen3:8b",
                }],
                coins=[{"id": 1, "symbol": "AAA", "pair_address": "pair_a"}],
            )

            t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
            rows = []
            for i in range(120):
                rows.append({
                    "event_timestamp": _iso(t0 + timedelta(minutes=i)),
                    "symbol": "AAA",
                    "pair_address": "pair_a",
                    "target_name": "label_profitable_after_fees_4h",
                    "y_true": 1,
                    "predicted_probability": 0.5 + (i / 200),
                    "model_name": "random_forest",
                    "split": "validation",
                    "target_return_4h": 0.05,
                })
            val_preds = pd.DataFrame(rows)
            test_preds = val_preds.copy()
            test_preds["split"] = "test"
            val_preds.to_parquet(models_dir / "predictions_validation.parquet", index=False)
            test_preds.to_parquet(models_dir / "predictions_test.parquet", index=False)
            metrics = {
                "best_model_by_target": {
                    "label_profitable_after_fees_4h": {"model_name": "random_forest"},
                }
            }
            (models_dir / "baseline_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

            report = run_llm_overlay_evaluation(
                validation_predictions_path=models_dir / "predictions_validation.parquet",
                test_predictions_path=models_dir / "predictions_test.parquet",
                db_path=db_path,
                models_dir=models_dir,
            )
            self.assertFalse(report["is_oracle_backtest"])
            self.assertTrue(report["uses_stored_llm_decisions_only"])
            self.assertFalse(report["uses_new_llm_calls"])
            self.assertTrue(report["llm_overlay_is_offline"])
            self.assertTrue(report["no_automatic_retraining"])
            self.assertEqual(report["stored_llm_metadata"]["stored_llm_table_used"], "gemini_decisions")

    def test_skipped_variant_when_confidence_missing(self) -> None:
        joined = pd.DataFrame({
            "parsed_confidence": [None],
            "parsed_risk_score": [10],
            "llm_matched": [True],
            "parsed_action": ["BUY"],
            "is_qwen_ollama": [True],
        })
        rf_mask = pd.Series([True])
        variant = next(v for v in __import__("app.training.llm_overlay", fromlist=["OVERLAY_VARIANTS"]).OVERLAY_VARIANTS if v["name"] == "rf_plus_confidence_gte_0_60")
        _mask, reason = build_overlay_mask(joined, rf_mask, variant=variant, llm_subset="all_stored")
        self.assertEqual(reason, "confidence_unavailable")

    def test_main_sqlite_not_modified(self) -> None:
        db_candidates = list(Path(__file__).resolve().parents[1].glob("**/*.db"))
        db_candidates = [p for p in db_candidates if ".venv" not in str(p)]
        if not db_candidates:
            self.skipTest("No sqlite database present.")
        db_path = db_candidates[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_copy = tmp_path / "main_copy.db"
            shutil.copy2(db_path, db_copy)
            before = db_copy.read_bytes()
            models_dir = tmp_path / "models"
            models_dir.mkdir()
            _create_test_db(tmp_path / "empty.db", decisions=[])
            t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
            row = {
                "event_timestamp": _iso(t0),
                "symbol": "AAA",
                "pair_address": "pair_a",
                "target_name": "label_profitable_after_fees_4h",
                "y_true": 1,
                "predicted_probability": 0.95,
                "model_name": "random_forest",
                "split": "validation",
                "target_return_4h": 0.05,
            }
            pd.DataFrame([row]).to_parquet(models_dir / "predictions_validation.parquet", index=False)
            pd.DataFrame([{**row, "split": "test"}]).to_parquet(models_dir / "predictions_test.parquet", index=False)
            (models_dir / "baseline_metrics.json").write_text(
                json.dumps({"best_model_by_target": {"label_profitable_after_fees_4h": {"model_name": "random_forest"}}}),
                encoding="utf-8",
            )
            load_stored_llm_decisions(db_copy)
            after = db_copy.read_bytes()
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
