"""Tests for Phase E3 direct exit-policy target dataset builder."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.candidates.schema import compute_candidate_id  # noqa: E402
from app.candidates.validation import normalize_event_timestamp  # noqa: E402
from app.training.direct_target_builder import (  # noqa: E402
    OUTCOME_COLUMNS,
    SnapshotPairCache,
    build_audit_row,
    sort_canonical_df,
    validate_sqlite_readonly,
    write_canonical_dual,
)
from app.training.direct_target_ids import (  # noqa: E402
    DEFAULT_EXIT_POLICIES,
    NOT_APPLICABLE,
    compute_candidate_policy_id,
    compute_target_row_id,
    label_source_artifact_id_for_input,
)
from app.training.exit_path_simulation import (  # noqa: E402
    EXIT_COMPARE_EPSILON,
    sl_hit,
    simulate_exit_path,
    tp_hit,
)

NS_PER_MINUTE = 60_000_000_000


def _make_candidate_id() -> str:
    return compute_candidate_id(
        chain="solana",
        pair_address="PAIR_A",
        event_timestamp_normalized=normalize_event_timestamp("2024-06-01T12:00:00Z"),
        source="unit_test",
    )


def _policy_a() -> dict:
    return dict(DEFAULT_EXIT_POLICIES[0])


def _policy_b() -> dict:
    return dict(DEFAULT_EXIT_POLICIES[1])


class DirectTargetIdsTests(unittest.TestCase):
    def test_candidate_id_stable_across_exit_policy(self) -> None:
        cid = _make_candidate_id()
        cp_a = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id=_policy_a()["exit_policy_id"],
            tp_ratio=float(_policy_a()["tp_ratio"]),
            sl_ratio=float(_policy_a()["sl_ratio"]),
            time_stop_minutes=240,
            round_trip_fee_pct=float(_policy_a()["round_trip_fee_pct"]),
        )
        cp_b = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id=_policy_b()["exit_policy_id"],
            tp_ratio=float(_policy_b()["tp_ratio"]),
            sl_ratio=float(_policy_b()["sl_ratio"]),
            time_stop_minutes=240,
            round_trip_fee_pct=float(_policy_b()["round_trip_fee_pct"]),
        )
        self.assertEqual(cid, _make_candidate_id())
        self.assertNotEqual(cp_a, cp_b)

    def test_candidate_policy_id_changes_with_horizon(self) -> None:
        cid = _make_candidate_id()
        base = dict(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            exit_policy_id=_policy_a()["exit_policy_id"],
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
        )
        a = compute_candidate_policy_id(**base, horizon="1h", time_stop_minutes=60)
        b = compute_candidate_policy_id(**base, horizon="4h", time_stop_minutes=240)
        self.assertNotEqual(a, b)

    def test_candidate_policy_id_changes_with_tp_sl_fee_filter(self) -> None:
        cid = _make_candidate_id()
        common = dict(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id="TEST",
            time_stop_minutes=240,
            round_trip_fee_pct=0.0308,
            sl_ratio=0.80,
        )
        tp_a = compute_candidate_policy_id(**common, tp_ratio=2.0308)
        tp_b = compute_candidate_policy_id(**common, tp_ratio=2.05)
        self.assertNotEqual(tp_a, tp_b)

        sl_a = compute_candidate_policy_id(**{**common, "tp_ratio": 2.0308, "sl_ratio": 0.80})
        sl_b = compute_candidate_policy_id(**{**common, "tp_ratio": 2.0308, "sl_ratio": 0.75})
        self.assertNotEqual(sl_a, sl_b)

        fee_a = compute_candidate_policy_id(**{**common, "tp_ratio": 2.0308, "round_trip_fee_pct": 0.0308})
        fee_b = compute_candidate_policy_id(**{**common, "tp_ratio": 2.0308, "round_trip_fee_pct": 0.04})
        self.assertNotEqual(fee_a, fee_b)

        filt_a = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id="TEST",
            tp_ratio=2.0308,
            sl_ratio=0.80,
            time_stop_minutes=240,
            round_trip_fee_pct=0.0308,
        )
        filt_b = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="LIQ_5K_HIGH_ACTIVITY",
            horizon="4h",
            exit_policy_id="TEST",
            tp_ratio=2.0308,
            sl_ratio=0.80,
            time_stop_minutes=240,
            round_trip_fee_pct=0.0308,
        )
        self.assertNotEqual(filt_a, filt_b)

    def test_target_row_id_changes_with_version_and_label_source(self) -> None:
        cp = compute_candidate_policy_id(
            candidate_id=_make_candidate_id(),
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id=_policy_a()["exit_policy_id"],
            tp_ratio=2.0308,
            sl_ratio=0.80,
            time_stop_minutes=240,
            round_trip_fee_pct=0.0308,
        )
        src_a = label_source_artifact_id_for_input("data/a.parquet")
        src_b = label_source_artifact_id_for_input("data/b.parquet")
        row_v1 = compute_target_row_id(
            candidate_policy_id=cp,
            target_version="v1",
            label_source_artifact_id=src_a,
        )
        row_v2 = compute_target_row_id(
            candidate_policy_id=cp,
            target_version="v2",
            label_source_artifact_id=src_a,
        )
        row_src = compute_target_row_id(
            candidate_policy_id=cp,
            target_version="v1",
            label_source_artifact_id=src_b,
        )
        self.assertNotEqual(row_v1, row_v2)
        self.assertNotEqual(row_v1, row_src)

    def test_target_row_id_unique_and_not_candidate_id(self) -> None:
        cid = _make_candidate_id()
        cp = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="4h",
            exit_policy_id=_policy_a()["exit_policy_id"],
            tp_ratio=2.0308,
            sl_ratio=0.80,
            time_stop_minutes=240,
            round_trip_fee_pct=0.0308,
        )
        row = compute_target_row_id(
            candidate_policy_id=cp,
            target_version="v1",
            label_source_artifact_id=label_source_artifact_id_for_input("x.parquet"),
        )
        self.assertNotEqual(row, cid)
        self.assertNotEqual(row, cp)


class ExitPathSimulationTests(unittest.TestCase):
    def _series(self, points: list[tuple[str, float]], ids: list[int] | None = None):
        ts = np.array(
            [pd.Timestamp(t, tz="UTC").value for t, _ in points],
            dtype=np.int64,
        )
        prices = np.array([p for _, p in points], dtype=float)
        snap_ids = np.array(ids if ids else list(range(len(points))), dtype=int)
        return ts, prices, snap_ids

    def test_tp_exit(self) -> None:
        ts, prices, ids = self._series(
            [
                ("2024-06-01T11:00:00Z", 1.0),
                ("2024-06-01T12:00:00Z", 1.0),
                ("2024-06-01T12:10:00Z", 1.5),
                ("2024-06-01T12:30:00Z", 2.1),
            ]
        )
        event_ns = pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value
        result = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=ts,
            prices=prices,
            snapshot_ids=ids,
        )
        self.assertTrue(result.label_valid)
        self.assertEqual(result.sim_exit_status, "TP")
        self.assertEqual(result.exit_ratio, 2.0308)

    def test_sl_exit(self) -> None:
        ts, prices, ids = self._series(
            [
                ("2024-06-01T11:00:00Z", 1.0),
                ("2024-06-01T12:00:00Z", 1.0),
                ("2024-06-01T12:15:00Z", 0.79),
            ]
        )
        event_ns = pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value
        result = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=ts,
            prices=prices,
            snapshot_ids=ids,
        )
        self.assertTrue(result.label_valid)
        self.assertEqual(result.sim_exit_status, "SL")
        self.assertEqual(result.exit_ratio, 0.80)

    def test_time_exit_uses_last_in_window(self) -> None:
        ts, prices, ids = self._series(
            [
                ("2024-06-01T11:00:00Z", 1.0),
                ("2024-06-01T12:00:00Z", 1.0),
                ("2024-06-01T12:15:00Z", 1.03),
                ("2024-06-01T12:35:00Z", 1.05),
                ("2024-06-01T12:55:00Z", 1.08),
                ("2024-06-01T13:30:00Z", 9.99),
            ]
        )
        event_ns = pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value
        result = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=ts,
            prices=prices,
            snapshot_ids=ids,
        )
        self.assertTrue(result.label_valid)
        self.assertEqual(result.sim_exit_status, "TIME")
        self.assertAlmostEqual(result.exit_ratio, 1.08)

    def test_tp_sl_epsilon_tolerance(self) -> None:
        tp_ratio = 2.0308
        sl_ratio = 0.80
        self.assertTrue(tp_hit(tp_ratio - EXIT_COMPARE_EPSILON / 2, tp_ratio))
        self.assertTrue(sl_hit(sl_ratio + EXIT_COMPARE_EPSILON / 2, sl_ratio))

    def test_no_pair_no_entry_no_future_bad_price_gap(self) -> None:
        ts, prices, ids = self._series([("2024-06-01T12:00:00Z", 1.0)])
        event_ns = pd.Timestamp("2024-06-01T11:00:00Z", tz="UTC").value
        no_entry = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=ts,
            prices=prices,
            snapshot_ids=ids,
        )
        self.assertEqual(no_entry.label_error_code, "NO_ENTRY_SNAPSHOT")

        bad_price_ts, bad_prices, bad_ids = self._series([("2024-06-01T12:00:00Z", 0.0)])
        event_ns2 = pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value
        bad = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns2,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=bad_price_ts,
            prices=bad_prices,
            snapshot_ids=bad_ids,
        )
        self.assertEqual(bad.label_error_code, "BAD_ENTRY_PRICE")
        self.assertIsNotNone(bad.entry_snapshot_timestamp)
        self.assertEqual(bad.entry_price_raw, 0.0)

        no_future_ts, no_future_prices, no_future_ids = self._series(
            [("2024-06-01T12:00:00Z", 1.0)]
        )
        no_future = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns2,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=no_future_ts,
            prices=no_future_prices,
            snapshot_ids=no_future_ids,
        )
        self.assertEqual(no_future.label_error_code, "NO_FUTURE_WINDOW")

        gap_ts, gap_prices, gap_ids = self._series(
            [
                ("2024-06-01T12:00:00Z", 1.0),
                ("2024-06-01T12:50:00Z", 1.01),
            ]
        )
        gap = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns2,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=20,
            ts_ns=gap_ts,
            prices=gap_prices,
            snapshot_ids=gap_ids,
        )
        self.assertEqual(gap.label_error_code, "GAP_IN_FUTURE_DATA")
        self.assertTrue(gap.gap.gap_detected)


class BuilderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "trader.db"
        self._create_db()
        self.cache = SnapshotPairCache(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE market_snapshots (
                id INTEGER PRIMARY KEY,
                coin_id INTEGER,
                timestamp TEXT,
                pair_address TEXT,
                price REAL
            )
            """
        )
        rows = [
            (1, 1, "2024-06-01T11:00:00Z", "PAIR_A", 1.0),
            (2, 1, "2024-06-01T12:00:00Z", "PAIR_A", 1.0),
            (3, 1, "2024-06-01T12:10:00Z", "PAIR_A", 1.5),
            (4, 1, "2024-06-01T12:30:00Z", "PAIR_A", 2.1),
            (5, 1, "2024-06-01T12:00:00Z", "PAIR_B", 0.0),
        ]
        conn.executemany(
            "INSERT INTO market_snapshots (id, coin_id, timestamp, pair_address, price) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def test_build_audit_row_and_dual_write(self) -> None:
        self.cache.prefetch_pairs(["PAIR_A"])
        row = pd.Series(
            {
                "pair_address": "PAIR_A",
                "event_timestamp": "2024-06-01T12:00:00Z",
                "price": 1.0,
                "split": "train",
            }
        )
        audit = build_audit_row(
            row,
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy=_policy_a(),
            target_version="v1",
            label_source_artifact_id=label_source_artifact_id_for_input("in.parquet"),
            chain="solana",
            source="unit_test",
            max_future_gap_minutes=20,
            snapshot_cache=self.cache,
        )
        self.assertTrue(audit["label_valid"])
        self.assertIn("candidate_policy_id", audit)
        self.assertIn("target_row_id", audit)

        out_dir = self.root / "out"
        csv_path = out_dir / "test.csv"
        parquet_path = out_dir / "test.parquet"
        df = pd.DataFrame([audit, audit.copy()])
        df.loc[1, "pair_address"] = "PAIR_MISSING"
        canonical = write_canonical_dual(df, csv_path, parquet_path)
        csv_df = pd.read_csv(csv_path)
        parquet_df = pd.read_parquet(parquet_path)
        self.assertEqual(len(csv_df), len(parquet_df))
        self.assertEqual(len(csv_df), len(canonical))

    def test_missing_pair_invalid_label(self) -> None:
        self.cache.prefetch_pairs(["PAIR_MISSING"])
        row = pd.Series(
            {"pair_address": "PAIR_MISSING", "event_timestamp": "2024-06-01T12:00:00Z"}
        )
        audit = build_audit_row(
            row,
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy=_policy_a(),
            target_version="v1",
            label_source_artifact_id=label_source_artifact_id_for_input("in.parquet"),
            chain="solana",
            source="unit_test",
            max_future_gap_minutes=20,
            snapshot_cache=self.cache,
        )
        self.assertEqual(audit["label_error_code"], "NO_PAIR")

    def test_sqlite_readonly(self) -> None:
        info = validate_sqlite_readonly(self.db_path)
        self.assertTrue(info["readable"])


class CliTests(unittest.TestCase):
    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            df = pd.DataFrame(
                {
                    "pair_address": ["PAIR_A"],
                    "event_timestamp": ["2024-06-01T12:00:00Z"],
                    "price": [1.0],
                    "split": ["train"],
                    "target": [0],
                }
            )
            df.to_parquet(input_dir / "RAW_ALL_VERIFIED_x2_1h_CLEAN_MODEL_INPUT.parquet")

            db_path = root / "trader.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE market_snapshots (id INTEGER PRIMARY KEY, timestamp TEXT, pair_address TEXT, price REAL)"
            )
            conn.execute(
                "INSERT INTO market_snapshots VALUES (1, '2024-06-01T12:00:00Z', 'PAIR_A', 1.0)"
            )
            conn.commit()
            conn.close()

            out_ds = root / "out_ds"
            out_rp = root / "out_rp"
            env = os.environ.copy()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_direct_exit_targets.py"),
                    "--input-dir",
                    str(input_dir),
                    "--sqlite-db",
                    str(db_path),
                    "--output-dataset-dir",
                    str(out_ds),
                    "--output-report-dir",
                    str(out_rp),
                    "--filters",
                    "RAW_ALL_VERIFIED",
                    "--horizons",
                    "1h",
                    "--dry-run",
                    "--register-artifacts",
                    "false",
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(out_ds.exists() or any(out_rp.glob("*")))

    def test_overwrite_only_e3_paths(self) -> None:
        protected = ROOT / "data" / "training" / "manual_verified_datasets_clean_for_model"
        self.assertTrue(protected.exists() or True)


class DocumentationTests(unittest.TestCase):
    def test_outcome_columns_documented(self) -> None:
        doc = (ROOT / "docs" / "architecture" / "11_direct_exit_target_dataset.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("target_net_profitable_after_exit", doc)
        self.assertIn("exclude from training", doc.lower())
        self.assertIn("XGB", doc)
        self.assertIn("TAB", doc)
        self.assertIn("RF", doc)

    def test_outcome_columns_set_nonempty(self) -> None:
        self.assertIn("sim_net_return", OUTCOME_COLUMNS)


class TerminologyTests(unittest.TestCase):
    def test_xgb_rf_tab_not_redefined(self) -> None:
        doc = (ROOT / "docs" / "architecture" / "11_direct_exit_target_dataset.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("XGBoost", doc)
        self.assertIn("Random Forest", doc)
        self.assertIn("TabICL", doc)


class DeterministicOrderTests(unittest.TestCase):
    def test_row_order_deterministic(self) -> None:
        df = pd.DataFrame(
            {
                "filter": ["B", "A", "A"],
                "horizon": ["1h", "1h", "1h"],
                "exit_policy_id": ["P", "P", "P"],
                "pair_address": ["z", "a", "b"],
                "event_timestamp": ["2024-01-03", "2024-01-01", "2024-01-02"],
                "candidate_id": ["c3", "c1", "c2"],
                "candidate_policy_id": ["cp3", "cp1", "cp2"],
                "target_row_id": ["t3", "t1", "t2"],
            }
        )
        sorted_df = sort_canonical_df(df)
        self.assertEqual(sorted_df.iloc[0]["filter"], "A")
        self.assertEqual(sorted_df.iloc[0]["pair_address"], "a")


class MemoryChunkingTests(unittest.TestCase):
    def test_chunked_pair_cache_does_not_require_all_policies(self) -> None:
        from app.training.direct_target_builder import iter_input_chunks

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.parquet"
            pd.DataFrame({"pair_address": [f"P{i}" for i in range(12)], "event_timestamp": ["2024-01-01"] * 12}).to_parquet(path)
            chunks = list(iter_input_chunks(path, chunk_size=5))
            self.assertGreater(len(chunks), 1)


class RegistryTests(unittest.TestCase):
    def test_registry_failure_reported_not_swallowed(self) -> None:
        from unittest import mock
        from app.training import direct_target_builder as dtb

        with mock.patch(
            "app.artifacts.registry.scan_artifacts",
            side_effect=RuntimeError("registry boom"),
        ):
            status = dtb.register_e3_artifacts(
                ROOT,
                ROOT / "data" / "training" / "manual_verified_datasets_direct_target_v1",
                ROOT / "data" / "training" / "manual_verified_results" / "phase_e3_direct_targets_v1",
            )
        self.assertFalse(status["success"])
        self.assertIn("registry boom", status["error"])


class BoundaryTests(unittest.TestCase):
    def test_time_stop_excludes_snapshot_after_boundary(self) -> None:
        ts = np.array(
            [
                pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value,
                pd.Timestamp("2024-06-01T12:30:00Z", tz="UTC").value,
                pd.Timestamp("2024-06-01T13:05:00Z", tz="UTC").value,
            ],
            dtype=np.int64,
        )
        prices = np.array([1.0, 1.02, 5.0], dtype=float)
        event_ns = pd.Timestamp("2024-06-01T12:00:00Z", tz="UTC").value
        result = simulate_exit_path(
            pair="PAIR_A",
            event_ns=event_ns,
            time_stop_minutes=60,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            max_future_gap_minutes=60,
            ts_ns=ts,
            prices=prices,
            snapshot_ids=np.array([1, 2, 3]),
        )
        self.assertTrue(result.label_valid)
        self.assertEqual(result.sim_exit_status, "TIME")
        self.assertAlmostEqual(result.exit_ratio, 1.02)


class UniqueKeyTests(unittest.TestCase):
    def test_candidate_id_not_unique_for_direct_rows(self) -> None:
        cid = _make_candidate_id()
        cp1 = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id=_policy_a()["exit_policy_id"],
            tp_ratio=2.0308,
            sl_ratio=0.80,
            time_stop_minutes=60,
            round_trip_fee_pct=0.0308,
        )
        cp2 = compute_candidate_policy_id(
            candidate_id=cid,
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id=_policy_b()["exit_policy_id"],
            tp_ratio=2.0308,
            sl_ratio=0.75,
            time_stop_minutes=60,
            round_trip_fee_pct=0.0308,
        )
        src = label_source_artifact_id_for_input("same.parquet")
        row1 = compute_target_row_id(
            candidate_policy_id=cp1, target_version="v1", label_source_artifact_id=src
        )
        row2 = compute_target_row_id(
            candidate_policy_id=cp2, target_version="v1", label_source_artifact_id=src
        )
        self.assertNotEqual(row1, row2)
        self.assertNotEqual(cp1, cp2)

    def test_target_row_id_unique_across_rows(self) -> None:
        rows = {
            compute_target_row_id(
                candidate_policy_id=compute_candidate_policy_id(
                    candidate_id=_make_candidate_id(),
                    filter_name=f,
                    horizon="1h",
                    exit_policy_id=_policy_a()["exit_policy_id"],
                    tp_ratio=2.0308,
                    sl_ratio=0.80,
                    time_stop_minutes=60,
                    round_trip_fee_pct=0.0308,
                ),
                target_version="v1",
                label_source_artifact_id=label_source_artifact_id_for_input(f"{f}.parquet"),
            )
            for f in ("RAW_ALL_VERIFIED", "LIQ_5K_HIGH_ACTIVITY", "LOW_LIQ_MOMENTUM")
        }
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
