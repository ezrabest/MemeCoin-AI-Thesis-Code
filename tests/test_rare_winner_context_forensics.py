"""Tests for Phase E8E rare-winner context forensics audit."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.rare_winner_context_forensics import (  # noqa: E402
    CheckpointManager,
    asof_snapshot,
    build_feature_candidate_map,
    classify_e8e_final,
    discover_reservoir_files,
    identity_key,
    is_leaky_feature_name,
    limit_rows_for_smoke,
    normalize_id,
    normalize_pair,
    parse_utc_timestamp,
    recommended_status_for_feature,
    run_joinability_audit,
    inventory_sqlite_tables,
    load_pair_snapshots,
    ForensicsConfig,
    make_output_dir,
    run_forensics,
)


class NormalizationTests(unittest.TestCase):
    def test_strip_and_lower(self) -> None:
        self.assertEqual(normalize_id(" abc "), "abc")
        self.assertEqual(normalize_pair(" 0xABC "), "0xabc")

    def test_identity_key_prefers_target_row_id(self) -> None:
        key = identity_key(
            {
                "target_row_id": " tr1 ",
                "candidate_id": "c1",
                "pair_address": "0xabc",
                "event_timestamp": "2026-06-01T00:00:00Z",
            }
        )
        self.assertTrue(key.startswith("target_row_id:"))


class LeakageTests(unittest.TestCase):
    def test_is_leaky_tokens(self) -> None:
        for name in (
            "post_entry_liquidity",
            "after_event_volume",
            "future_price",
            "target_net_profitable",
            "label_valid",
            "sim_net_return",
            "exit_timestamp",
            "gap_detected",
            "max_future_ratio",
            "realized_return",
            "outcome_flag",
        ):
            leaky, _ = is_leaky_feature_name(name)
            self.assertTrue(leaky, msg=name)

    def test_feature_map_rejects_post_entry(self) -> None:
        status = recommended_status_for_feature(
            feature_name="liquidity_delta_5m_post_diag",
            pre_entry_legal=False,
            missingness_rate=0.0,
            pair_identity_risk="low",
        )
        self.assertEqual(status, "REJECT_LEAKAGE")
        leaky_status = recommended_status_for_feature(
            feature_name="future_price_asof",
            pre_entry_legal=False,
            missingness_rate=0.0,
            pair_identity_risk="low",
        )
        self.assertEqual(leaky_status, "REJECT_LEAKAGE")


class AsOfTests(unittest.TestCase):
    def test_asof_uses_lte_event_time(self) -> None:
        snaps = pd.DataFrame(
            {
                "pair_address": ["0xabc", "0xabc", "0xabc"],
                "ts": pd.to_datetime(
                    ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z", "2026-06-01T02:00:00Z"],
                    utc=True,
                ),
                "price": [1.0, 2.0, 3.0],
                "liquidity": [100.0, 200.0, 300.0],
                "volume_24h": [10.0, 20.0, 30.0],
                "fdv": [1, 1, 1],
                "txns_buys": [1, 1, 1],
                "txns_sells": [1, 1, 1],
                "txns_total": [2, 2, 2],
                "price_change_m5": [0, 0, 0],
                "price_change_h1": [0, 0, 0],
                "price_change_h6": [0, 0, 0],
                "price_change_h24": [0, 0, 0],
                "whale_score": [0.1, 0.2, 0.3],
                "buy_ratio": [0.5, 0.5, 0.5],
                "id": [1, 2, 3],
                "timestamp": ["a", "b", "c"],
            }
        )
        event = parse_utc_timestamp("2026-06-01T01:30:00Z")
        assert event is not None
        row = asof_snapshot(snaps, "0xABC", event)
        assert row is not None
        self.assertEqual(float(row["price"]), 2.0)


class JoinabilityTests(unittest.TestCase):
    def test_load_pair_snapshots_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE market_snapshots (id INTEGER, pair_address TEXT, timestamp TEXT, price REAL)"
            )
            conn.execute(
                "INSERT INTO market_snapshots VALUES (1, '0xAbC', '2026-06-01T00:00:00Z', 1.5)"
            )
            conn.commit()
            conn.close()
            snaps = load_pair_snapshots(db, ["0xabc"])
            self.assertEqual(len(snaps), 1)
            self.assertEqual(snaps.iloc[0]["pair_address"], "0xabc")

    def test_zero_match_join_reported(self) -> None:
        sample = [{"pair_address": "missing_pair", "event_timestamp": "2026-06-01T00:00:00Z"}]
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE market_snapshots (id INTEGER, pair_address TEXT, timestamp TEXT, price REAL)"
            )
            conn.commit()
            conn.close()
            rows = run_joinability_audit(sample, db_path=db, reservoir_files=[])
            sqlite_row = rows[0]
            self.assertEqual(sqlite_row["sqlite_match_count"], 0)
            self.assertFalse(sqlite_row["prediction_to_sqlite_join_possible"])

    def test_optional_sqlite_unavailable_continues(self) -> None:
        sample = [{"pair_address": "x", "event_timestamp": "2026-06-01T00:00:00Z"}]
        rows = run_joinability_audit(sample, db_path=Path("/nonexistent/db.db"), reservoir_files=[])
        self.assertEqual(rows[0]["failure_reason"], "sqlite unavailable")


class ClassificationTests(unittest.TestCase):
    def test_pair_identity_artifact(self) -> None:
        out = classify_e8e_final(
            patterns=[],
            feature_map=[],
            market_available=True,
            joinability=[{"prediction_to_sqlite_join_possible": True}],
            candidate_rows=[{"pair_address": "0x1"}, {"pair_address": "0x1"}],
        )
        self.assertEqual(out["final_classification"], "PAIR_IDENTITY_ARTIFACT")

    def test_insufficient_context_data(self) -> None:
        out = classify_e8e_final(
            patterns=[],
            feature_map=[],
            market_available=False,
            joinability=[{"prediction_to_sqlite_join_possible": False}],
            candidate_rows=[{"pair_address": "0x1"}, {"pair_address": "0x2"}],
        )
        self.assertEqual(out["final_classification"], "INSUFFICIENT_CONTEXT_DATA")


class CheckpointTests(unittest.TestCase):
    def test_stage_and_candidate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp) / "audit", force=False)
            mgr.mark_stage("sqlite_inventory", status="completed")
            mgr.mark_candidate("target_row_id:abc")
            mgr2 = CheckpointManager(Path(tmp) / "audit", force=False)
            self.assertTrue(mgr2.stage_complete("sqlite_inventory"))
            self.assertTrue(mgr2.candidate_complete("target_row_id:abc"))


class SqliteInventoryTests(unittest.TestCase):
    def test_inventory_on_tiny_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tiny.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE market_snapshots (id INTEGER, pair_address TEXT, timestamp TEXT)")
            conn.execute("CREATE TABLE whale_alerts (id INTEGER, pair_address TEXT, timestamp TEXT)")
            conn.commit()
            conn.close()
            rows = inventory_sqlite_tables(db)
            names = {r["table_name"] for r in rows}
            self.assertIn("market_snapshots", names)
            self.assertIn("whale_alerts", names)


class SmokeSchemaTests(unittest.TestCase):
    def _write_e8c(self, root: Path) -> None:
        reports = root / "reports"
        reports.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "dataset_name": "RAW_ALL_VERIFIED_8h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1",
                    "filter": "RAW_ALL_VERIFIED",
                    "horizon": "8h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "final_classification": "RARE_WINNER_DETECTOR",
                }
            ]
        ).to_csv(reports / "e8c_final_classification.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset_name": "RAW_ALL_VERIFIED_8h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1",
                    "top_pct_percent": 0.05,
                }
            ]
        ).to_csv(reports / "e8c_validation_selected_policies.csv", index=False)

    def test_smoke_produces_expected_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e8b = root / "e8b"
            pred = e8b / "predictions"
            pred.mkdir(parents=True)
            ds = "RAW_ALL_VERIFIED_8h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1"
            df = pd.DataFrame(
                {
                    "pair_address": [f"0xpair{i}" for i in range(30)],
                    "event_timestamp": pd.date_range("2026-06-10", periods=30, freq="h", tz="UTC"),
                    "target_net_profitable_after_exit": [1 if i < 3 else 0 for i in range(30)],
                    "sim_net_return": [0.1 if i < 3 else -0.03 for i in range(30)],
                    "predicted_probability": np.linspace(0.9, 0.1, 30),
                    "split": ["validation"] * 30,
                    "filter": ["RAW_ALL_VERIFIED"] * 30,
                    "horizon": ["8h"] * 30,
                    "exit_policy_id": ["TP20308_SL080_FEE0308_TIME_BY_HORIZON"] * 30,
                }
            )
            df.to_csv(pred / f"{ds}_validation_predictions.csv", index=False)
            df.assign(split="test").to_csv(pred / f"{ds}_test_predictions.csv", index=False)
            e8c = root / "e8c"
            self._write_e8c(e8c)
            db = root / "db.db"
            conn = sqlite3.connect(db)
            conn.execute(
                """
                CREATE TABLE market_snapshots (
                    id INTEGER, pair_address TEXT, timestamp TEXT,
                    price REAL, liquidity REAL, volume_24h REAL, fdv REAL,
                    txns_buys INTEGER, txns_sells INTEGER, txns_total INTEGER,
                    price_change_m5 REAL, price_change_h1 REAL, price_change_h6 REAL, price_change_h24 REAL,
                    whale_score REAL, buy_ratio REAL
                )
                """
            )
            for i in range(30):
                conn.execute(
                    "INSERT INTO market_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        i,
                        f"0xpair{i % 5}",
                        f"2026-06-10T{i:02d}:00:00Z",
                        1.0,
                        1000.0,
                        100.0,
                        1.0,
                        1,
                        1,
                        2,
                        0,
                        0,
                        0,
                        0,
                        0.1,
                        0.5,
                    ),
                )
            conn.commit()
            conn.close()
            out = make_output_dir(root / "results")
            config = ForensicsConfig(
                e8b_run_dir=e8b,
                e8c_dir=e8c,
                output_dir=out,
                sqlite_db=db,
                smoke=True,
                max_candidates=20,
            )
            result = run_forensics(config, project_root=root)
            reports = out / "reports"
            expected = [
                "e8e_run_manifest.json",
                "e8e_decision_summary.txt",
                "e8e_joinability_audit.csv",
                "e8e_sqlite_table_inventory.csv",
                "e8e_market_context_by_candidate.csv",
                "e8e_final_classification.csv",
            ]
            for name in expected:
                self.assertTrue((reports / name).exists(), msg=name)
            self.assertTrue((out / "audit" / "e8e_progress_checkpoints.jsonl").exists())
            self.assertLessEqual(result["candidate_rows"], 20)


class ReservoirDiscoveryTests(unittest.TestCase):
    def test_missing_reservoir_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            found = discover_reservoir_files(Path(tmp))
            self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
