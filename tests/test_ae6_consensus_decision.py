"""Tests for AE6 consensus decision layer."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from app.decision.builder import (  # noqa: E402
    build_decision_record,
    build_lineage_metadata,
    determine_decision_status,
    unavailable_model_scores,
)
from app.decision.consensus import compute_consensus  # noqa: E402
from app.decision.persistence import (  # noqa: E402
    DecisionJsonlWriter,
    read_jsonl_records_safe,
    serialize_decision_record,
)
from app.decision.types import (  # noqa: E402
    AE6_PHASE,
    CandidateIdentityBlock,
    ConsensusFamily,
    DecisionRecord,
    DecisionStatusAE6,
    LineageMetadata,
    LineageMode,
    LineageResolutionMethod,
    LineageStrength,
    LineageValidationError,
    ModelScoreSlot,
    ModelScoresBlock,
    ResearchContextBlock,
)


def _explicit_lineage() -> LineageMetadata:
    return build_lineage_metadata(
        provider="dexscreener",
        pair_address="pair123",
        symbol="TEST",
        snapshot_timestamp="2026-07-09T10:00:00+00:00",
        signal_timestamp="2026-07-09T10:00:01+00:00",
        raw_payload_id=1,
        snapshot_id=2,
        signal_id=3,
        raw_payload_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
        snapshot_id_resolution_method=LineageResolutionMethod.FOREIGN_KEY,
        signal_id_resolution_method=LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
    )


def _implicit_lineage() -> LineageMetadata:
    return build_lineage_metadata(
        provider="dexscreener",
        pair_address="pair123",
        symbol="TEST",
        snapshot_timestamp="2026-07-09T10:00:00+00:00",
        signal_timestamp="2026-07-09T10:00:01+00:00",
        raw_payload_id=1,
        snapshot_id=2,
        signal_id=3,
        raw_payload_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        snapshot_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
        signal_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
    )


def _minimal_signal_row() -> dict:
    return {
        "id": 99,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coin_id": 1,
        "symbol": "TEST",
        "signal_type": "WATCH",
        "score": 0.4,
        "confidence": 0.5,
        "reason": "unit_test",
        "model_source": "engine.generate_signal",
        "features_json": None,
    }


def _snapshot_row() -> dict:
    return {
        "id": 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair_address": "pair123",
        "chain": "solana",
        "provider": "dexscreener",
        "liquidity": 10000.0,
        "whale_score": 0.35,
        "price": 0.001,
    }


class AE6DecisionTypesTests(unittest.TestCase):
    def test_decision_record_schema_required_fields(self) -> None:
        record = build_decision_record(
            signal_row=_minimal_signal_row(),
            snapshot_row=_snapshot_row(),
            lineage=_implicit_lineage(),
        )
        self.assertEqual(record.phase, AE6_PHASE)
        self.assertTrue(record.no_trade_authority)
        self.assertIsInstance(record.lineage, LineageMetadata)
        self.assertIn("signal_type", record.signal_context)

    def test_lineage_metadata_mandatory_on_decision_record(self) -> None:
        with self.assertRaises(Exception):
            DecisionRecord.model_validate(
                {
                    "lineage": None,
                }
            )

    def test_missing_lineage_metadata_fails_closed(self) -> None:
        with self.assertRaises((LineageValidationError, ValidationError)):
            LineageMetadata.model_validate(
                {
                    "lineage_mode": "BEST_EFFORT_IMPLICIT_LINKAGE",
                    "lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS",
                    "raw_payload_id": 1,
                    "snapshot_id": 2,
                    "signal_id": 3,
                    "raw_payload_id_resolution_method": "BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH",
                    "snapshot_id_resolution_method": "BEST_EFFORT_PAIR_TIME_MATCH",
                    "signal_id_resolution_method": "EXPLICIT_COLUMN",
                }
            )

    def test_best_effort_requires_fallback_reason(self) -> None:
        with self.assertRaises((LineageValidationError, ValidationError)):
            LineageMetadata(
                lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE,
                lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS,
                raw_payload_id=1,
                snapshot_id=2,
                signal_id=3,
                raw_payload_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
                snapshot_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
                signal_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
                fallback_reason="",
                lineage_warning="warn",
            )

    def test_best_effort_requires_lineage_warning(self) -> None:
        with self.assertRaises((LineageValidationError, ValidationError)):
            LineageMetadata(
                lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE,
                lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS,
                raw_payload_id=1,
                snapshot_id=2,
                signal_id=3,
                raw_payload_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
                snapshot_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
                signal_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
                fallback_reason="fallback",
                lineage_warning="",
            )

    def test_explicit_lineage_mode_when_both_ids_present(self) -> None:
        lineage = _explicit_lineage()
        self.assertEqual(lineage.lineage_mode, LineageMode.EXPLICIT_LINKAGE)
        self.assertEqual(lineage.lineage_strength, LineageStrength.STRONG_EXPLICIT_LINKS)

    def test_implicit_lineage_adds_caveat_fields(self) -> None:
        lineage = _implicit_lineage()
        self.assertEqual(lineage.lineage_mode, LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE)
        self.assertTrue(lineage.fallback_reason)
        self.assertTrue(lineage.lineage_warning)

    def test_best_effort_matched_raw_payload_id_is_not_strong_explicit(self) -> None:
        lineage = _implicit_lineage()
        self.assertEqual(
            lineage.raw_payload_id_resolution_method,
            LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        )
        self.assertEqual(lineage.lineage_strength, LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS)

    def test_best_effort_matched_snapshot_id_is_not_strong_explicit(self) -> None:
        lineage = _implicit_lineage()
        self.assertEqual(
            lineage.snapshot_id_resolution_method,
            LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
        )
        self.assertEqual(lineage.lineage_mode, LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE)

    def test_resolved_best_effort_ids_still_require_fallback_and_warning(self) -> None:
        with self.assertRaises((LineageValidationError, ValidationError)):
            LineageMetadata(
                lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE,
                lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS,
                raw_payload_id=1,
                snapshot_id=2,
                signal_id=3,
                raw_payload_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
                snapshot_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
                signal_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
                fallback_reason="",
                lineage_warning="",
            )

    def test_explicit_linkage_requires_explicit_structural_methods(self) -> None:
        with self.assertRaises((LineageValidationError, ValidationError)):
            LineageMetadata(
                lineage_mode=LineageMode.EXPLICIT_LINKAGE,
                lineage_strength=LineageStrength.STRONG_EXPLICIT_LINKS,
                raw_payload_id=1,
                snapshot_id=2,
                signal_id=3,
                raw_payload_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
                snapshot_id_resolution_method=LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
                signal_id_resolution_method=LineageResolutionMethod.EXPLICIT_COLUMN,
            )


class AE6ConsensusTests(unittest.TestCase):
    def test_missing_model_scores_do_not_crash(self) -> None:
        consensus = compute_consensus(unavailable_model_scores())
        self.assertEqual(
            consensus.consensus_family,
            ConsensusFamily.NO_MODEL_CONSENSUS_AVAILABLE,
        )

    def test_no_model_consensus_is_valid_not_exception(self) -> None:
        try:
            result = compute_consensus(ModelScoresBlock())
        except Exception as exc:  # pragma: no cover - must not raise
            self.fail(f"compute_consensus raised unexpectedly: {exc}")
        self.assertEqual(result.consensus_strength, "UNAVAILABLE")
        self.assertIsNotNone(result.consensus_caveat)

    def test_consensus_family_all_three(self) -> None:
        scores = ModelScoresBlock(
            RF=ModelScoreSlot(available=True, score=0.7),
            XGB=ModelScoreSlot(available=True, score=0.8),
            TAB=ModelScoreSlot(available=True, score=0.75),
        )
        consensus = compute_consensus(scores)
        self.assertEqual(consensus.consensus_family, ConsensusFamily.TAB_XGB_RF_ALL3)
        self.assertEqual(consensus.available_model_count, 3)

    def test_single_model_only_family(self) -> None:
        scores = ModelScoresBlock(
            RF=ModelScoreSlot(available=True, score=0.6),
        )
        consensus = compute_consensus(scores)
        self.assertEqual(consensus.consensus_family, ConsensusFamily.SINGLE_MODEL_ONLY)


class AE6SafetyTests(unittest.TestCase):
    def test_no_llm_execution_placeholders(self) -> None:
        record = build_decision_record(
            signal_row=_minimal_signal_row(),
            lineage=_implicit_lineage(),
        )
        self.assertFalse(record.llm_context.llm_execution_allowed)
        self.assertFalse(record.llm_context.llm_decision_authority)
        self.assertEqual(record.llm_context.llm_missing_reason, "AE9_NOT_IMPLEMENTED_YET")

    def test_no_trade_authority_always_true(self) -> None:
        record = build_decision_record(
            signal_row=_minimal_signal_row(),
            lineage=_implicit_lineage(),
        )
        self.assertTrue(record.no_trade_authority)

    def test_whale_score_asof_is_not_hard_rule(self) -> None:
        research = ResearchContextBlock()
        self.assertTrue(research.whale_score_asof_not_rule)
        self.assertTrue(research.whale_score_asof_not_runtime_approved)

        # Freeze evaluation time to match _implicit_lineage snapshot — avoids
        # non-deterministic BLOCK from snapshot staleness as calendar time advances.
        frozen_now = datetime(2026, 7, 9, 10, 0, 2, tzinfo=timezone.utc)
        status, reasons, caveats, _, _ = determine_decision_status(
            identity=CandidateIdentityBlock(pair_address="pair123"),
            lineage=_implicit_lineage(),
            signal_context={"score": 0.9, "confidence": 0.9, "signal_type": "BUY"},
            market_context={"whale_score": 0.01, "liquidity": 50000},
            consensus_family="NO_MODEL_CONSENSUS_AVAILABLE",
            missingness=[],
            now=frozen_now,
        )
        self.assertNotEqual(status, DecisionStatusAE6.BLOCK)
        self.assertTrue(any("whale_score_asof" in c for c in caveats))


class AE6PersistenceTests(unittest.TestCase):
    def test_jsonl_writes_one_line_per_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ae6_test.jsonl"
            record = build_decision_record(
                signal_row=_minimal_signal_row(),
                lineage=_implicit_lineage(),
            )
            writer = DecisionJsonlWriter(path=path)
            writer.append_record(record)
            writer.close()
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["phase"], AE6_PHASE)

    def test_jsonl_writer_calls_flush_and_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ae6_fsync.jsonl"
            record = build_decision_record(
                signal_row=_minimal_signal_row(),
                lineage=_implicit_lineage(),
            )
            with mock.patch("app.decision.persistence.os.fsync") as fsync_mock:
                with DecisionJsonlWriter(path=path) as writer:
                    with mock.patch.object(writer._ensure_open(), "flush") as flush_mock:
                        writer.append_record(record)
                        flush_mock.assert_called_once()
                fsync_mock.assert_called()

    def test_jsonl_reader_handles_incomplete_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ae6_incomplete.jsonl"
            good = serialize_decision_record(
                build_decision_record(
                    signal_row=_minimal_signal_row(),
                    lineage=_implicit_lineage(),
                )
            )
            path.write_text(good + "\n" + '{"incomplete":', encoding="utf-8")
            records, diagnostics = read_jsonl_records_safe(path)
            self.assertEqual(len(records), 1)
            self.assertIsNotNone(diagnostics.get("incomplete_line"))


if __name__ == "__main__":
    unittest.main()
