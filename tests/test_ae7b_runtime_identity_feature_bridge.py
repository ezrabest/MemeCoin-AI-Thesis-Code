"""Tests for AE7B runtime identity + feature matrix bridge."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.bridge_persistence import RuntimeBridgeJsonlWriter, read_bridge_jsonl_safe  # noqa: E402
from app.decision.feature_parity import (  # noqa: E402
    FeatureParityStatus,
    run_feature_parity_check,
)
from app.decision.feature_schema import (  # noqa: E402
    build_feature_values,
    build_runtime_feature_schema,
    is_forbidden_feature_name,
)
from app.decision.runtime_feature_bridge import (  # noqa: E402
    build_model_schema_compatibility_matrix,
    build_runtime_bridge_record,
)
from app.decision.runtime_identity import (  # noqa: E402
    build_bridge_lineage,
    build_identity_payload,
    compute_lineage_confidence_score,
    default_scoring_policy,
    generate_as_of_feature_row_id,
    generate_candidate_id,
    generate_scoring_policy_id,
    normalize_address,
)
from app.decision.types import LineageResolutionMethod  # noqa: E402


class AE7BRuntimeIdentityTests(unittest.TestCase):
    def test_candidate_id_deterministic(self) -> None:
        payload = build_identity_payload(
            chain="solana",
            pair_address="0xABC",
            event_timestamp="2026-07-10T09:00:00+00:00",
            source_table="signals",
            source_row_id=1,
        )
        id1, status1, _ = generate_candidate_id(payload)
        id2, status2, _ = generate_candidate_id(payload)
        self.assertEqual(id1, id2)
        self.assertEqual(status1.value, "OK")

    def test_candidate_id_changes_with_identity(self) -> None:
        p1 = build_identity_payload(pair_address="0xaaa", event_timestamp="t1")
        p2 = build_identity_payload(pair_address="0xbbb", event_timestamp="t1")
        id1, _, _ = generate_candidate_id(p1)
        id2, _, _ = generate_candidate_id(p2)
        self.assertNotEqual(id1, id2)

    def test_pair_address_normalized_lowercase(self) -> None:
        self.assertEqual(normalize_address("0xAbC"), "0xabc")

    def test_missing_stable_identity_fails_closed(self) -> None:
        cid, status, _ = generate_candidate_id({})
        self.assertIsNone(cid)
        self.assertEqual(status.value, "BLOCKED_MISSING_STABLE_IDENTITY")

    def test_scoring_policy_id_deterministic(self) -> None:
        self.assertEqual(generate_scoring_policy_id(), generate_scoring_policy_id())

    def test_as_of_feature_row_id_deterministic(self) -> None:
        kwargs = dict(
            candidate_id="c1",
            scoring_policy_id="p1",
            feature_schema_id="s1",
            as_of_timestamp="2026-07-10T09:00:00+00:00",
            source_snapshot_id=10,
            source_signal_id=20,
        )
        self.assertEqual(
            generate_as_of_feature_row_id(**kwargs),
            generate_as_of_feature_row_id(**kwargs),
        )


class AE7BFeatureSchemaTests(unittest.TestCase):
    def test_feature_schema_id_deterministic(self) -> None:
        s1 = build_runtime_feature_schema()
        s2 = build_runtime_feature_schema()
        self.assertEqual(s1.feature_schema_id, s2.feature_schema_id)

    def test_forbidden_feature_names_rejected(self) -> None:
        self.assertTrue(is_forbidden_feature_name("target_net_profitable"))
        self.assertTrue(is_forbidden_feature_name("future_return"))

    def test_missing_required_feature_blocks(self) -> None:
        schema = build_runtime_feature_schema()
        result = build_feature_values(
            snapshot_row=None,
            signal_row=None,
            sentiment_agg=None,
            schema=schema,
        )
        self.assertIn("price_usd", result.missing_required)
        self.assertEqual(result.feature_status, "MISSING_REQUIRED_FEATURE")

    def test_optional_missing_becomes_null_with_missingness(self) -> None:
        schema = build_runtime_feature_schema()
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 1000.0},
            signal_row={"score": 0.5},
            sentiment_agg=None,
            schema=schema,
        )
        self.assertNotIn("price_usd", result.missing_required)
        self.assertIn("sentiment_score", result.feature_missingness)
        self.assertIsNone(result.feature_values.get("sentiment_score"))

    def test_whale_score_asof_research_only_metadata(self) -> None:
        schema = build_runtime_feature_schema()
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 1000.0, "whale_score": 0.3},
            signal_row=None,
            sentiment_agg=None,
            schema=schema,
        )
        self.assertIsNotNone(result.whale_score_metadata)
        self.assertTrue(result.whale_score_metadata["not_rule"])
        self.assertTrue(result.whale_score_metadata["not_runtime_approved_as_standalone_signal"])


class AE7BLineageTests(unittest.TestCase):
    def test_weak_implicit_lineage_when_best_effort(self) -> None:
        lineage = build_bridge_lineage(
            signal_id=1,
            snapshot_id=2,
            raw_payload_id=3,
            signal_method=LineageResolutionMethod.EXPLICIT_COLUMN,
            snapshot_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            raw_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        )
        self.assertEqual(lineage.lineage_mode, "BEST_EFFORT_IMPLICIT_LINKAGE")
        self.assertEqual(lineage.lineage_strength, "WEAK_IMPLICIT_TIME_PAIR_LINKS")
        self.assertFalse(lineage.exact_id_match)

    def test_lineage_confidence_below_half_when_not_exact(self) -> None:
        lineage = build_bridge_lineage(
            signal_id=1,
            snapshot_id=2,
            raw_payload_id=3,
            snapshot_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            raw_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        )
        self.assertLess(lineage.lineage_confidence_score, 0.5)

    def test_lineage_confidence_not_model_confidence(self) -> None:
        score = compute_lineage_confidence_score(
            signal_method=LineageResolutionMethod.EXPLICIT_COLUMN,
            snapshot_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            raw_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        )
        self.assertEqual(score, 0.35)


class AE7BBridgeRecordTests(unittest.TestCase):
    def _bundle(self) -> dict:
        return {
            "signal_row": {
                "id": 99,
                "timestamp": "2026-07-10T09:00:00+00:00",
                "symbol": "TEST/SOL",
                "score": 0.6,
            },
            "snapshot_row": {
                "id": 10,
                "timestamp": "2026-07-10T09:00:00+00:00",
                "price": 0.001,
                "liquidity": 5000.0,
                "volume_24h": 10000.0,
                "whale_score": 0.2,
                "provider": "dexscreener",
            },
            "raw_payload_row": {"id": 5, "provider": "dexscreener"},
            "coin_row": {
                "pair_address": "0xabc",
                "chain": "solana",
                "symbol": "TEST/SOL",
            },
            "sentiment_agg": None,
        }

    def test_runtime_bridge_record_has_no_target_row_id(self) -> None:
        schema = build_runtime_feature_schema()
        rec = build_runtime_bridge_record(self._bundle(), schema=schema)
        self.assertTrue(rec["target_row_id_not_required"])
        self.assertIsNone(rec["target_row_id"])
        self.assertIsNotNone(rec["candidate_id"])
        self.assertIsNotNone(rec["as_of_feature_row_id"])

    def test_no_trade_and_no_llm_authority(self) -> None:
        schema = build_runtime_feature_schema()
        rec = build_runtime_bridge_record(self._bundle(), schema=schema)
        self.assertTrue(rec["no_trade_authority"])
        self.assertFalse(rec["llm_decision_authority"])


class AE7BParityTests(unittest.TestCase):
    def test_parity_blocked_no_overlap(self) -> None:
        records = [{"candidate_id": "runtime-only", "feature_values": {"price_usd": 1.0}}]
        result = run_feature_parity_check(
            runtime_bridge_records=records,
            offline_rows_by_exact_id=None,
        )
        self.assertEqual(result.feature_parity_status, FeatureParityStatus.BLOCKED_NO_OVERLAP.value)

    def test_parity_fail_mismatch_on_exact_alignment(self) -> None:
        records = [{"candidate_id": "shared-id", "feature_values": {"price_usd": 1.0}}]
        offline = {"shared-id": {"price_usd": 9.0}}
        result = run_feature_parity_check(
            runtime_bridge_records=records,
            offline_rows_by_exact_id=offline,
            offline_feature_names=["price_usd"],
        )
        self.assertEqual(result.feature_parity_status, FeatureParityStatus.FAIL_MISMATCH.value)
        self.assertGreater(result.mismatch_count, 0)

    def test_parity_never_uses_fuzzy_matching(self) -> None:
        records = [{"candidate_id": "runtime-a", "feature_values": {}}]
        offline = {"different-id": {"price_usd": 1.0}}
        result = run_feature_parity_check(
            runtime_bridge_records=records,
            offline_rows_by_exact_id=offline,
        )
        self.assertEqual(result.overlap_rows, 0)
        self.assertEqual(result.feature_parity_status, FeatureParityStatus.BLOCKED_NO_OVERLAP.value)


class AE7BSchemaCompatibilityTests(unittest.TestCase):
    def test_compatibility_matrix_without_full_parquet_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "models" / "test_clean_rf_schema.json"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text(
                json.dumps({"feature_columns": ["price", "liquidity", "whale_score"]}),
                encoding="utf-8",
            )
            runtime_schema = build_runtime_feature_schema()
            rows = build_model_schema_compatibility_matrix(
                runtime_schema=runtime_schema,
                schema_candidate_paths=[Path("models/test_clean_rf_schema.json")],
                project_root=root,
                max_schemas=5,
            )
            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["compatibility_status"], {"COMPATIBLE", "PARTIAL_MISSING_FEATURES"})


class AE7BPersistenceTests(unittest.TestCase):
    def test_jsonl_flush_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bridge.jsonl"
            record = {"phase": "AE7B", "candidate_id": "x"}
            with mock.patch("app.decision.bridge_persistence.os.fsync") as fsync_mock:
                with RuntimeBridgeJsonlWriter(path=path) as writer:
                    writer.append_record(record)
                fsync_mock.assert_called()
            records, _ = read_bridge_jsonl_safe(path)
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
