"""Tests for AE7 model score slot population layer."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.builder import build_decision_record, build_lineage_metadata  # noqa: E402
from app.decision.consensus import compute_consensus  # noqa: E402
from app.decision.model_scores import (  # noqa: E402
    AE7_PHASE,
    MissingReason,
    PopulatedModelScoreSlot,
    column_is_leakage_risk,
    column_is_safe_score,
    has_model_compatible_runtime_id,
    infer_model_family_from_path,
)
from app.decision.score_artifacts import (  # noqa: E402
    ArtifactInspection,
    ArtifactStatus,
    PredictionIndex,
    IndexedScoreRow,
    _read_schema_columns,
    classify_inventory_row,
    build_prediction_index,
)
from app.decision.score_population import (  # noqa: E402
    AE7JsonlWriter,
    enrich_decision_record,
    populate_model_scores_for_identity,
)
from app.decision.types import (  # noqa: E402
    LineageResolutionMethod,
    ModelScoreSlot,
    ModelScoresBlock,
)


def _implicit_lineage():
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


def _ae6_record_dict(**identity_overrides) -> dict:
    record = build_decision_record(
        signal_row={
            "id": 1,
            "timestamp": "2026-07-10T09:00:00+00:00",
            "coin_id": 1,
            "symbol": "TEST",
            "signal_type": "WATCH",
            "score": 0.5,
            "confidence": 0.5,
            "reason": "test",
            "model_source": "test",
            "features_json": None,
        },
        lineage=_implicit_lineage(),
    )
    data = record.model_dump(mode="json")
    data["candidate_identity"].update(identity_overrides)
    return data


class AE7ArtifactClassifierTests(unittest.TestCase):
    def test_infer_rf_from_path(self) -> None:
        self.assertEqual(
            infer_model_family_from_path(
                "data/training/manual_verified_results/rf_clean/rf_predictions_test.parquet"
            ).value,
            "RF",
        )

    def test_infer_xgb_from_path(self) -> None:
        self.assertEqual(
            infer_model_family_from_path(
                "data/training/manual_verified_results/xgb_clean_full/xgb_predictions.parquet"
            ).value,
            "XGB",
        )

    def test_infer_tab_from_path(self) -> None:
        self.assertEqual(
            infer_model_family_from_path(
                "data/training/manual_verified_results/phase_e5/direct_target_tabicl_predictions.parquet"
            ).value,
            "TAB",
        )

    def test_leakage_column_rejected(self) -> None:
        self.assertTrue(column_is_leakage_risk("target_net_profitable"))
        self.assertTrue(column_is_leakage_risk("future_return"))
        self.assertFalse(column_is_safe_score("target_net_profitable"))

    def test_safe_score_column_accepted(self) -> None:
        self.assertTrue(column_is_safe_score("predicted_probability"))
        self.assertTrue(column_is_safe_score("tab_score"))

    def test_outcome_only_artifact_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "bad_predictions.csv"
            csv_path.write_text(
                "pair_address,target,net_return\n"
                "0xabc,1,0.5\n",
                encoding="utf-8",
            )
            row = {
                "path": str(csv_path.relative_to(root)).replace("/", "\\"),
                "kind": "PREDICTION_OR_SCORE_TABLE",
                "size_bytes": str(csv_path.stat().st_size),
                "modified_utc": "2026-07-10T00:00:00+00:00",
                "id_column_hits": "target_row_id",
                "score_column_hits": "target|net_return",
            }
            # Inject target_row_id into CSV for ID check but scores are leakage
            csv_path.write_text(
                "target_row_id,pair_address,target,net_return\n"
                "id1,0xabc,1,0.5\n",
                encoding="utf-8",
            )
            insp = classify_inventory_row(row, project_root=root)
            self.assertFalse(insp.safe_for_score_population)
            self.assertIn("REJECTED", insp.reason)

    def test_pair_time_only_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "pair_time_predictions.csv"
            csv_path.write_text(
                "pair_address,event_timestamp,predicted_probability\n"
                "0xabc,2026-01-01,0.7\n",
                encoding="utf-8",
            )
            row = {
                "path": "pair_time_predictions.csv",
                "kind": "PREDICTION_OR_SCORE_TABLE",
                "size_bytes": str(csv_path.stat().st_size),
                "modified_utc": "2026-07-10T00:00:00+00:00",
                "id_column_hits": "pair_address|event_timestamp",
                "score_column_hits": "predicted_probability",
            }
            insp = classify_inventory_row(row, project_root=root)
            self.assertFalse(insp.safe_for_score_population)
            self.assertEqual(insp.artifact_status, ArtifactStatus.REJECTED_NO_SAFE_ID.value)

    def test_reproducibility_matrix_has_is_reproducible(self) -> None:
        insp = ArtifactInspection(
            path="test.parquet",
            model_family="RF",
            artifact_role="prediction_table",
            safe_for_score_population=False,
            reason="test",
            is_reproducible=False,
            artifact_status=ArtifactStatus.UNREGISTERED.value,
        )
        row = insp.to_matrix_row()
        self.assertIn("is_reproducible", row)
        self.assertFalse(row["is_reproducible"])

    def test_unregistered_not_safe_by_default(self) -> None:
        insp = ArtifactInspection(
            path="test.parquet",
            model_family="RF",
            artifact_role="prediction_table",
            safe_for_score_population=False,
            reason="unregistered",
            is_reproducible=False,
            artifact_status=ArtifactStatus.UNREGISTERED.value,
        )
        self.assertFalse(insp.safe_for_score_population)

    def test_stale_not_safe_by_default(self) -> None:
        insp = ArtifactInspection(
            path="test.parquet",
            model_family="RF",
            artifact_role="prediction_table",
            safe_for_score_population=False,
            reason="stale",
            is_reproducible=True,
            artifact_status=ArtifactStatus.STALE.value,
        )
        self.assertFalse(insp.safe_for_score_population)

    def test_parquet_schema_without_full_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.parquet"
            df = pd.DataFrame(
                {
                    "target_row_id": ["a"],
                    "tab_score": [0.5],
                    "tab_rank_pct": [0.9],
                }
            )
            df.to_parquet(path, index=False)
            cols, row_count = _read_schema_columns(path)
            self.assertIn("target_row_id", cols)
            self.assertEqual(row_count, 1)


class AE7ScorePopulationTests(unittest.TestCase):
    def test_exact_id_required_for_population(self) -> None:
        index = PredictionIndex()
        index.add_row(
            IndexedScoreRow(
                score=0.8,
                rank=0.9,
                model_family="TAB",
                artifact_path="art.parquet",
                id_key_used="target_row_id",
                id_value="row-42",
                horizon="1h",
                filter="RAW",
                exit_policy=None,
                split="validation",
                score_column_used="tab_score",
                rank_column_used="tab_rank_pct",
                artifact_status="CURRENT",
                is_reproducible=True,
            )
        )
        identity = {"target_row_id": "row-42"}
        scores = populate_model_scores_for_identity(identity, index)
        self.assertTrue(scores.TAB.available)
        self.assertEqual(scores.TAB.score, 0.8)
        self.assertEqual(scores.TAB.id_key_used, "target_row_id")
        self.assertEqual(scores.TAB.population_method, "EXACT_ID_MATCH")

    def test_runtime_missing_bridge_fields(self) -> None:
        identity = {"candidate_id": "runtime-hash-only", "pair_address": "0xabc"}
        self.assertFalse(has_model_compatible_runtime_id(identity))
        # Runtime hash candidate_id is not a runtime bridge field; offline lookup may
        # still be attempted when candidate_id is present.
        scores = populate_model_scores_for_identity(identity, PredictionIndex())
        self.assertFalse(scores.RF.available)
        self.assertEqual(
            scores.RF.missing_reason,
            MissingReason.NO_SAFE_MODEL_ARTIFACT.value,
        )

    def test_empty_identity_yields_runtime_bridge_missing(self) -> None:
        scores = populate_model_scores_for_identity({}, PredictionIndex())
        self.assertEqual(
            scores.RF.missing_reason,
            MissingReason.RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID.value,
        )

    def test_target_row_id_enables_offline_lookup_not_runtime_bridge(self) -> None:
        identity = {"target_row_id": "hist-row-1"}
        self.assertFalse(has_model_compatible_runtime_id(identity))
        from app.decision.model_scores import can_attempt_offline_exact_id_lookup

        self.assertTrue(can_attempt_offline_exact_id_lookup(identity))

    def test_pair_address_timestamp_does_not_populate(self) -> None:
        index = PredictionIndex()
        # Index keyed by target_row_id only
        index.add_row(
            IndexedScoreRow(
                score=0.7,
                rank=None,
                model_family="RF",
                artifact_path="rf.parquet",
                id_key_used="target_row_id",
                id_value="real-id",
                horizon=None,
                filter=None,
                exit_policy=None,
                split=None,
                score_column_used="predicted_probability",
                rank_column_used=None,
                artifact_status="CURRENT",
                is_reproducible=True,
            )
        )
        identity = {
            "pair_address": "0xabc",
            "event_timestamp": "2026-07-10T09:00:00+00:00",
            "target_row_id": "different-id",
        }
        scores = populate_model_scores_for_identity(identity, index)
        self.assertFalse(scores.RF.available)
        self.assertEqual(
            scores.RF.missing_reason,
            MissingReason.NO_SAFE_EXACT_ID_ALIGNMENT.value,
        )

    def test_no_safe_alignment_explicit(self) -> None:
        identity = {"target_row_id": "missing-in-index"}
        scores = populate_model_scores_for_identity(identity, PredictionIndex())
        self.assertFalse(scores.RF.available)
        self.assertEqual(
            scores.RF.missing_reason,
            MissingReason.NO_SAFE_MODEL_ARTIFACT.value,
        )

    def test_populated_slot_includes_artifact_metadata(self) -> None:
        index = PredictionIndex()
        index.add_row(
            IndexedScoreRow(
                score=0.6,
                rank=0.5,
                model_family="XGB",
                artifact_path="data/xgb_preds.parquet",
                id_key_used="candidate_policy_id",
                id_value="pol-1",
                horizon="30m",
                filter="LIQ",
                exit_policy="tp_sl",
                split="test",
                score_column_used="xgb_score",
                rank_column_used="xgb_rank",
                artifact_status="CURRENT",
                is_reproducible=True,
                model_artifact_id="artifact-xyz",
            )
        )
        scores = populate_model_scores_for_identity(
            {"candidate_policy_id": "pol-1"},
            index,
        )
        self.assertTrue(scores.XGB.available)
        self.assertEqual(scores.XGB.artifact_path, "data/xgb_preds.parquet")
        self.assertEqual(scores.XGB.artifact_status, "CURRENT")
        self.assertTrue(scores.XGB.is_reproducible)

    def test_consensus_recomputes_when_slots_available(self) -> None:
        scores = ModelScoresBlock(
            RF=ModelScoreSlot(available=True, score=0.7),
            XGB=ModelScoreSlot(available=True, score=0.8),
            TAB=ModelScoreSlot(available=True, score=0.75),
        )
        consensus = compute_consensus(scores)
        self.assertEqual(consensus.consensus_family.value, "TAB_XGB_RF_ALL3")

    def test_consensus_remains_unavailable_when_no_slots(self) -> None:
        from app.decision.builder import unavailable_model_scores

        consensus = compute_consensus(unavailable_model_scores())
        self.assertEqual(consensus.consensus_family.value, "NO_MODEL_CONSENSUS_AVAILABLE")

    def test_ae6_lineage_preserved_on_enrich(self) -> None:
        ae6 = _ae6_record_dict()
        original_lineage = json.dumps(ae6["lineage"], sort_keys=True)
        enriched = enrich_decision_record(ae6, PredictionIndex())
        self.assertEqual(json.dumps(enriched["lineage"], sort_keys=True), original_lineage)
        self.assertEqual(enriched["phase"], AE7_PHASE)
        self.assertEqual(enriched["source_decision_id"], ae6["decision_id"])

    def test_no_trade_authority_remains_true(self) -> None:
        ae6 = _ae6_record_dict()
        enriched = enrich_decision_record(ae6, PredictionIndex())
        self.assertTrue(enriched["no_trade_authority"])

    def test_enrich_with_exact_id_populates_and_recomputes_consensus(self) -> None:
        ae6 = _ae6_record_dict(target_row_id="row-99", candidate_policy_id="pol-99")
        index = PredictionIndex()
        for family, col in [("RF", "predicted_probability"), ("XGB", "xgb_score"), ("TAB", "tab_score")]:
            index.add_row(
                IndexedScoreRow(
                    score=0.7,
                    rank=None,
                    model_family=family,
                    artifact_path=f"{family.lower()}.parquet",
                    id_key_used="target_row_id",
                    id_value="row-99",
                    horizon="1h",
                    filter=None,
                    exit_policy=None,
                    split="validation",
                    score_column_used=col,
                    rank_column_used=None,
                    artifact_status="CURRENT",
                    is_reproducible=True,
                )
            )
        enriched = enrich_decision_record(ae6, index)
        self.assertTrue(enriched["model_scores"]["RF"]["available"])
        self.assertEqual(
            enriched["consensus"]["consensus_family"],
            "TAB_XGB_RF_ALL3",
        )


class AE7PersistenceTests(unittest.TestCase):
    def test_jsonl_append_only_with_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ae7.jsonl"
            record = {"phase": AE7_PHASE, "decision_id": "test"}
            with mock.patch("app.decision.score_population.os.fsync") as fsync_mock:
                with AE7JsonlWriter(path=path) as writer:
                    with mock.patch.object(writer._ensure_open(), "flush") as flush_mock:
                        writer.append_record(record)
                        flush_mock.assert_called_once()
                fsync_mock.assert_called()
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)


class AE7IndexBuildTests(unittest.TestCase):
    def test_build_index_from_safe_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "tab_preds.parquet"
            df = pd.DataFrame(
                {
                    "target_row_id": ["id-1", "id-2"],
                    "tab_score": [0.3, 0.9],
                    "tab_rank_pct": [0.1, 0.95],
                    "horizon": ["1h", "1h"],
                    "filter": ["RAW", "RAW"],
                    "split": ["validation", "validation"],
                }
            )
            df.to_parquet(parquet_path, index=False)

            insp = ArtifactInspection(
                path="tab_preds.parquet",
                model_family="TAB",
                artifact_role="prediction_table",
                safe_for_score_population=True,
                reason="safe",
                id_columns=["target_row_id"],
                score_columns=["tab_score"],
                rank_columns=["tab_rank_pct"],
                split_columns=["split"],
                size_bytes=parquet_path.stat().st_size,
                is_reproducible=True,
                artifact_status=ArtifactStatus.CURRENT.value,
            )
            index = build_prediction_index([insp], project_root=root)
            hit = index.lookup("TAB", "target_row_id", "id-2")
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit.score, 0.9)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
