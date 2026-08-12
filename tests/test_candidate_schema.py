"""Tests for Phase E2 unified candidate schema."""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.candidates.examples import (  # noqa: E402
    make_all3_candidate_example,
    make_minimal_candidate_example,
    make_research_rejected_candidate_example,
    make_tab_rf_candidate_example,
)
from app.candidates.schema import (  # noqa: E402
    CandidateIdentity,
    ConsensusTier,
    DecisionStatus,
    EnrichmentStatus,
    LLMReviewStatus,
    ModelScores,
    UnifiedCandidate,
    compute_candidate_id,
    infer_consensus_tier,
)
from app.candidates.serialization import (  # noqa: E402
    candidate_from_dict,
    candidate_from_flat_dict,
    candidate_from_json,
    candidate_to_dict,
    candidate_to_flat_dict,
    candidate_to_json,
    normalize_flat_value_for_export,
    normalize_flat_value_for_import,
)
from app.candidates.validation import (  # noqa: E402
    normalize_event_timestamp,
    validate_finite_numeric,
)


class CandidateSchemaTests(unittest.TestCase):
  def test_minimal_candidate_can_be_created(self) -> None:
      candidate = make_minimal_candidate_example()
      self.assertEqual(candidate.schema_version, "candidate_schema_v1")
      self.assertIsNotNone(candidate.identity.candidate_id)

  def test_deterministic_candidate_id_is_stable(self) -> None:
      identity_a = CandidateIdentity(
          pair_address="0xAbC123",
          chain="ethereum",
          event_timestamp="2024-01-01T00:00:00Z",
          source="unit_test",
          source_row_id="42",
      )
      identity_b = CandidateIdentity(
          pair_address="0xabc123",
          chain="ethereum",
          event_timestamp="2024-01-01T00:00:00Z",
          source="unit_test",
          source_row_id="42",
      )
      self.assertEqual(identity_a.candidate_id, identity_b.candidate_id)

  def test_equivalent_timestamp_inputs_normalize_same(self) -> None:
      values = [
          "2024-06-15T18:30:00Z",
          "2024-06-15T18:30:00+00:00",
          datetime(2024, 6, 15, 18, 30, 0, tzinfo=timezone.utc),
          1718476200,
          1718476200000,
      ]
      normalized = [normalize_event_timestamp(value) for value in values]
      self.assertEqual(len(set(normalized)), 1)
      self.assertEqual(normalized[0], "2024-06-15T18:30:00Z")

  def test_equivalent_timestamps_produce_same_candidate_id(self) -> None:
      base_kwargs = dict(
          pair_address="pair-1",
          chain="solana",
          source="snapshot",
      )
      ids = []
      for ts in ("2024-06-15T18:30:00Z", 1718476200, 1718476200000):
          identity = CandidateIdentity(event_timestamp=ts, **base_kwargs)
          ids.append(identity.candidate_id)
      self.assertEqual(len(set(ids)), 1)

  def test_candidate_id_changes_when_pair_timestamp_source_changes(self) -> None:
      base = dict(
          pair_address="pair-1",
          chain="solana",
          event_timestamp="2024-06-15T18:30:00Z",
          source="snapshot",
      )
      id_base = CandidateIdentity(**base).candidate_id
      id_pair = CandidateIdentity(**{**base, "pair_address": "pair-2"}).candidate_id
      id_ts = CandidateIdentity(
          **{**base, "event_timestamp": "2024-06-15T19:30:00Z"}
      ).candidate_id
      id_source = CandidateIdentity(**{**base, "source": "other"}).candidate_id
      self.assertNotEqual(id_base, id_pair)
      self.assertNotEqual(id_base, id_ts)
      self.assertNotEqual(id_base, id_source)

  def test_event_timestamp_normalized_always_present(self) -> None:
      candidate = make_minimal_candidate_example()
      self.assertIsNotNone(candidate.identity.event_timestamp_normalized)
      self.assertTrue(candidate.identity.event_timestamp_normalized.endswith("Z"))

  def test_score_validation_accepts_unit_interval(self) -> None:
      scores = ModelScores(score_xgb=0.0, score_tab=1.0, score_rf=0.5)
      self.assertEqual(scores.score_xgb, 0.0)
      self.assertEqual(scores.score_tab, 1.0)

  def test_score_validation_rejects_outside_unit_interval(self) -> None:
      with self.assertRaises(Exception):
          ModelScores(score_xgb=1.01)
      with self.assertRaises(Exception):
          ModelScores(score_rf=-0.01)

  def test_score_validation_rejects_non_finite_unless_research_helper(self) -> None:
      with self.assertRaises(ValueError):
          validate_finite_numeric(float("nan"))
      with self.assertRaises(ValueError):
          ModelScores(score_xgb=float("nan"))
      self.assertTrue(math.isnan(validate_finite_numeric(float("nan"), allow_nan_for_research=True)))

  def test_round_trip_fee_pct_decimal_convention(self) -> None:
      candidate = make_all3_candidate_example()
      self.assertAlmostEqual(candidate.exit_policy.round_trip_fee_pct or 0.0, 0.0308)

  def test_infer_consensus_tier_all3(self) -> None:
      self.assertEqual(
          infer_consensus_tier(True, True, True),
          ConsensusTier.TAB_XGB_RF_ALL3,
      )

  def test_infer_consensus_tier_tab_rf(self) -> None:
      self.assertEqual(
          infer_consensus_tier(True, False, True),
          ConsensusTier.TAB_RF_ONLY,
      )

  def test_infer_consensus_tier_tab_xgb(self) -> None:
      self.assertEqual(
          infer_consensus_tier(True, True, False),
          ConsensusTier.TAB_XGB_ONLY,
      )

  def test_infer_consensus_tier_xgb_rf(self) -> None:
      self.assertEqual(
          infer_consensus_tier(False, True, True),
          ConsensusTier.XGB_RF_ONLY,
      )

  def test_infer_consensus_tier_none(self) -> None:
      self.assertEqual(
          infer_consensus_tier(False, False, False),
          ConsensusTier.NONE,
      )

  def test_infer_consensus_tier_unknown_strict(self) -> None:
      self.assertEqual(
          infer_consensus_tier(True, None, True),
          ConsensusTier.UNKNOWN,
      )

  def test_infer_consensus_tier_non_strict_treats_none_as_false(self) -> None:
      self.assertEqual(
          infer_consensus_tier(True, None, True, strict=False),
          ConsensusTier.TAB_RF_ONLY,
      )

  def test_vote_count_computed(self) -> None:
      scores = ModelScores(in_xgb=True, in_tab=True, in_rf=False)
      self.assertEqual(scores.vote_count, 2)

  def test_dict_round_trip(self) -> None:
      original = make_all3_candidate_example()
      restored = candidate_from_dict(candidate_to_dict(original))
      self.assertEqual(restored.identity.candidate_id, original.identity.candidate_id)
      self.assertEqual(restored.consensus_tier, original.consensus_tier)

  def test_json_round_trip(self) -> None:
      original = make_tab_rf_candidate_example()
      restored = candidate_from_json(candidate_to_json(original))
      self.assertEqual(restored.model_dump(), original.model_dump())

  def test_flat_dict_parquet_round_trip(self) -> None:
      original = make_all3_candidate_example()
      flat = candidate_to_flat_dict(original, target_format="parquet")
      restored = candidate_from_flat_dict(flat, source_format="parquet")
      self.assertEqual(restored.identity.candidate_id, original.identity.candidate_id)
      self.assertEqual(restored.consensus_tier, original.consensus_tier)

  def test_flat_dict_csv_round_trip(self) -> None:
      original = make_tab_rf_candidate_example()
      flat = candidate_to_flat_dict(original, target_format="csv")
      restored = candidate_from_flat_dict(flat, source_format="csv")
      self.assertEqual(restored.identity.candidate_id, original.identity.candidate_id)
      self.assertEqual(restored.model_scores.in_tab, True)
      self.assertIsNone(restored.market.volume_24h)

  def test_none_nan_normalized_on_flat_import_export(self) -> None:
      self.assertIsNone(normalize_flat_value_for_import(float("nan")))
      self.assertEqual(normalize_flat_value_for_export(None, target_format="csv"), "")
      self.assertIsNone(normalize_flat_value_for_export(None, target_format="parquet"))
      self.assertIsNone(normalize_flat_value_for_import("", source_format="csv"))

  def test_repeated_flat_serialization_is_deterministic(self) -> None:
      candidate = make_all3_candidate_example()
      flat_a = candidate_to_flat_dict(candidate, target_format="parquet")
      flat_b = candidate_to_flat_dict(candidate, target_format="parquet")
      self.assertEqual(flat_a, flat_b)

  def test_strenum_serializes_cleanly_to_json(self) -> None:
      payload = {"tier": ConsensusTier.TAB_XGB_RF_ALL3}
      encoded = json.dumps(payload)
      self.assertEqual(json.loads(encoded)["tier"], "TAB_XGB_RF_ALL3")

  def test_enrichment_statuses_validate(self) -> None:
      candidate = make_minimal_candidate_example()
      candidate.enrichment.solana_enrichment_status = EnrichmentStatus.PENDING
      self.assertEqual(candidate.enrichment.solana_enrichment_status.value, "PENDING")

  def test_llm_review_statuses_validate(self) -> None:
      candidate = make_minimal_candidate_example()
      candidate.llm_review.qwen_review_status = LLMReviewStatus.SKIPPED
      self.assertEqual(candidate.llm_review.qwen_review_status.value, "SKIPPED")

  def test_decision_statuses_validate(self) -> None:
      candidate = make_research_rejected_candidate_example()
      self.assertEqual(candidate.decision.decision, DecisionStatus.REJECTED_RESEARCH_ONLY)

  def test_unknown_extra_fields_rejected(self) -> None:
      with self.assertRaises(Exception):
          UnifiedCandidate.model_validate(
              {
                  "identity": {
                      "pair_address": "pair",
                      "chain": "solana",
                      "event_timestamp": "2024-01-01T00:00:00Z",
                      "source": "test",
                  },
                  "unexpected_field": True,
              }
          )

  def test_examples_produce_valid_candidates(self) -> None:
      for factory in (
          make_minimal_candidate_example,
          make_all3_candidate_example,
          make_tab_rf_candidate_example,
          make_research_rejected_candidate_example,
      ):
          candidate = factory()
          self.assertIsInstance(candidate, UnifiedCandidate)
          self.assertTrue(candidate.identity.candidate_id)

  def test_compute_candidate_id_matches_identity(self) -> None:
      identity = CandidateIdentity(
          pair_address="pair",
          chain="solana",
          event_timestamp="2024-01-01T00:00:00Z",
          source="snapshot",
      )
      expected = compute_candidate_id(
          chain="solana",
          pair_address="pair",
          event_timestamp_normalized=identity.event_timestamp_normalized or "",
          source="snapshot",
      )
      self.assertEqual(identity.candidate_id, expected)

  @mock.patch("socket.socket")
  def test_no_network_calls(self, _mock_socket: mock.MagicMock) -> None:
      make_all3_candidate_example()
      candidate_to_json(make_tab_rf_candidate_example())
      if _mock_socket.called:
          self.fail("Network socket usage detected during schema-only tests")

  def test_no_sqlite_writes(self) -> None:
      db_path = ROOT / "trader.db"
      before_mtime = db_path.stat().st_mtime if db_path.exists() else None
      make_all3_candidate_example()
      candidate_from_flat_dict(
          candidate_to_flat_dict(make_minimal_candidate_example()),
          source_format="parquet",
      )
      if before_mtime is not None:
          after_mtime = db_path.stat().st_mtime
          self.assertEqual(before_mtime, after_mtime)

  def test_no_artifact_files_modified(self) -> None:
      registry_dir = ROOT / "data" / "training" / "artifact_registry"
      if not registry_dir.exists():
          return
      mtimes = {
          path: path.stat().st_mtime
          for path in registry_dir.glob("*")
          if path.is_file()
      }
      make_research_rejected_candidate_example()
      candidate_to_flat_dict(make_all3_candidate_example(), target_format="csv")
      for path, mtime in mtimes.items():
          self.assertEqual(path.stat().st_mtime, mtime, msg=f"Modified artifact file: {path}")


if __name__ == "__main__":
    unittest.main()
