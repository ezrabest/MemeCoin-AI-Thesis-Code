"""AE7 decision record enrichment and score-slot population."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.decision.consensus import compute_consensus
from app.decision.model_scores import (
    AE6_SOURCE_PHASE,
    AE7_PHASE,
    BRIDGE_PHASE_GOAL,
    BRIDGE_PHASE_NAME,
    MODEL_FAMILIES,
    RUNTIME_BRIDGE_FIELDS,
    SCORE_POPULATION_MODE_HISTORICAL_OFFLINE,
    SCORE_POPULATION_MODE_RUNTIME_LIVE,
    MissingReason,
    PopulatedModelScoreSlot,
    PopulatedModelScoresBlock,
    ScorePopulationDecision,
    can_attempt_offline_exact_id_lookup,
    has_model_compatible_runtime_id,
    populated_slot_to_ae6_dict,
    runtime_bridge_fields_present,
    runtime_id_keys_present,
    unavailable_ae7_slot,
)
from app.decision.persistence import read_jsonl_records_safe
from app.decision.score_artifacts import (
    ArtifactInspection,
    PredictionIndex,
    IndexedScoreRow,
    build_prediction_index,
    classify_inventory,
    load_inventory_csv,
    write_reproducibility_matrix,
)
from app.decision.types import LineageMetadata

POPULATION_METHOD_EXACT = "EXACT_ID_MATCH"


def ae7_enriched_jsonl_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    day = dt.strftime("%Y%m%d")
    return Path(__file__).parent.parent.parent / "data" / "decision_records" / f"ae7_model_score_enriched_{day}.jsonl"


class AE7JsonlWriter:
    """Append-only AE7 enriched JSONL writer with flush + fsync per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append_record(self, record: dict[str, Any]) -> Path:
        serialized = json.dumps(record, default=str, separators=(",", ":"))
        f = self._ensure_open()
        f.write(serialized + "\n")
        f.flush()
        os.fsync(f.fileno())
        return self.path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> AE7JsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _indexed_row_to_slot(row: IndexedScoreRow) -> PopulatedModelScoreSlot:
    return PopulatedModelScoreSlot(
        available=True,
        score=row.score,
        rank=row.rank,
        model_artifact_id=row.model_artifact_id,
        prediction_artifact_id=row.artifact_path,
        horizon=row.horizon,
        filter=row.filter,
        exit_policy=row.exit_policy,
        missing_reason=None,
        model_family=row.model_family,
        id_key_used=row.id_key_used,
        id_value=row.id_value,
        split=row.split,
        score_column_used=row.score_column_used,
        rank_column_used=row.rank_column_used,
        population_method=POPULATION_METHOD_EXACT,
        artifact_status=row.artifact_status,
        is_reproducible=row.is_reproducible,
        artifact_path=row.artifact_path,
    )


def populate_model_scores_for_identity(
    identity: dict[str, Any],
    index: PredictionIndex,
    *,
    inspections_by_family: dict[str, list[ArtifactInspection]] | None = None,
) -> PopulatedModelScoresBlock:
    """Populate RF/XGB/TAB slots via exact ID match only."""
    inspections_by_family = inspections_by_family or {}
    slots: dict[str, PopulatedModelScoreSlot] = {}

    can_offline_lookup = can_attempt_offline_exact_id_lookup(identity)
    has_runtime_bridge = has_model_compatible_runtime_id(identity)

    for family in MODEL_FAMILIES:
        if not can_offline_lookup and not has_runtime_bridge:
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID.value,
                model_family=family,
            )
            continue

        if not can_offline_lookup:
            # Runtime bridge present but AE7 only implements historical/offline lookup.
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT.value,
                model_family=family,
            )
            continue

        hit = index.lookup_identity(family, identity)
        if hit is not None:
            slots[family] = _indexed_row_to_slot(hit)
            continue

        # No exact match — check if family has safe artifacts at all
        family_inspections = inspections_by_family.get(family, [])
        has_safe_artifact = any(i.safe_for_score_population for i in family_inspections)
        has_reproducible = any(i.is_reproducible for i in family_inspections)
        family_index_has_entries = bool(index._by_family.get(family))

        if family_index_has_entries or has_safe_artifact:
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.NO_SAFE_EXACT_ID_ALIGNMENT.value,
                model_family=family,
            )
        elif not family_inspections:
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.NO_SAFE_MODEL_ARTIFACT.value,
                model_family=family,
            )
        elif not has_reproducible:
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.ARTIFACT_NOT_REPRODUCIBLE_OR_STALE.value,
                model_family=family,
            )
        else:
            slots[family] = unavailable_ae7_slot(
                missing_reason=MissingReason.NO_SAFE_SCORE_COLUMN.value,
                model_family=family,
            )

    meta_slot = unavailable_ae7_slot(
        missing_reason=MissingReason.NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT.value,
        model_family="META",
    )
    return PopulatedModelScoresBlock(
        RF=slots["RF"],
        XGB=slots["XGB"],
        TAB=slots["TAB"],
        META=meta_slot,
    )


def _populated_to_consensus_block(scores: PopulatedModelScoresBlock):
    """Bridge populated slots to AE6 consensus input."""
    from app.decision.types import ModelScoreSlot, ModelScoresBlock

    def _to_ae6(slot: PopulatedModelScoreSlot) -> ModelScoreSlot:
        return ModelScoreSlot(
            available=slot.available,
            score=slot.score,
            rank=slot.rank,
            model_artifact_id=slot.model_artifact_id,
            prediction_artifact_id=slot.prediction_artifact_id,
            horizon=slot.horizon,
            filter=slot.filter,
            exit_policy=slot.exit_policy,
            missing_reason=slot.missing_reason,
        )

    return compute_consensus(
        ModelScoresBlock(
            RF=_to_ae6(scores.RF),
            XGB=_to_ae6(scores.XGB),
            TAB=_to_ae6(scores.TAB),
            META=_to_ae6(scores.META),
        )
    )


def _update_missingness(
    original: list[str],
    scores: PopulatedModelScoresBlock,
) -> list[str]:
    missing = list(original)
    for family in MODEL_FAMILIES:
        slot = getattr(scores, family)
        key = f"model_score_{family.lower()}"
        if slot.available:
            if key in missing:
                missing.remove(key)
        elif key not in missing:
            missing.append(key)
    return missing


def enrich_decision_record(
    ae6_record: dict[str, Any],
    index: PredictionIndex,
    *,
    inspections_by_family: dict[str, list[ArtifactInspection]] | None = None,
) -> dict[str, Any]:
    """Return AE7-enriched record; preserve AE6 lineage and no_trade_authority."""
    enriched = copy.deepcopy(ae6_record)
    source_decision_id = ae6_record.get("decision_id", "")

    identity = enriched.get("candidate_identity") or {}
    lineage_raw = enriched.get("lineage") or {}
    # Lineage preserved exactly — do not mutate
    lineage = LineageMetadata.model_validate(lineage_raw)

    populated_scores = populate_model_scores_for_identity(
        identity,
        index,
        inspections_by_family=inspections_by_family,
    )
    consensus = _populated_to_consensus_block(populated_scores)

    caveats = list(enriched.get("caveats") or [])
    score_caveats: list[str] = []

    if not has_model_compatible_runtime_id(identity):
        score_caveats.append(
            "Runtime record lacks candidate_policy_id / scoring_policy_id and "
            "as_of_feature_row_id + feature_schema_id bridge fields; "
            "runtime feature-matrix inference path not yet available."
        )
    elif not can_attempt_offline_exact_id_lookup(identity):
        score_caveats.append(
            "Runtime bridge fields present but AE7 only audits historical/offline "
            "exact-ID artifact lookup — live inference path is a future phase."
        )
    elif not any(getattr(populated_scores, f).available for f in MODEL_FAMILIES):
        score_caveats.append(
            "No exact-ID model score alignment found in offline prediction artifacts."
        )

    if consensus.consensus_caveat and consensus.consensus_caveat not in caveats:
        caveats.append(consensus.consensus_caveat)

    missingness = _update_missingness(enriched.get("missingness") or [], populated_scores)

    # Update model_consensus component only — preserve AE6 decision_status (audit-time
    # re-evaluation would incorrectly BLOCK records due to snapshot age).
    confidence_components = dict(enriched.get("confidence_components") or {})
    if consensus.available_model_count > 0:
        confidence_components["model_consensus"] = 1.0
    else:
        confidence_components["model_consensus"] = 0.0

    reasons = list(enriched.get("reasons") or [])
    if consensus.available_model_count == 0 and "no_model_consensus_available" not in reasons:
        reasons.append("no_model_consensus_available")

    any_score = any(getattr(populated_scores, f).available for f in MODEL_FAMILIES)
    if any_score:
        score_population_status = "POPULATED"
    elif not has_model_compatible_runtime_id(identity):
        score_population_status = "RUNTIME_IDENTITY_GAP"
    else:
        score_population_status = "NO_SAFE_ALIGNMENT"

    enriched.update(
        {
            "decision_id": str(uuid4()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": AE7_PHASE,
            "source_phase": AE6_SOURCE_PHASE,
            "source_decision_id": source_decision_id,
            "decision_status": enriched.get("decision_status"),
            "decision_confidence": enriched.get("decision_confidence"),
            "confidence_components": confidence_components,
            "lineage": lineage.model_dump(mode="json"),
            "model_scores": {
                f: populated_slot_to_ae6_dict(getattr(populated_scores, f))
                for f in ("RF", "XGB", "TAB", "META")
            },
            "consensus": consensus.model_dump(mode="json"),
            "missingness": missingness,
            "reasons": reasons,
            "caveats": caveats,
            "score_population_status": score_population_status,
            "score_population_caveats": score_caveats,
            "no_trade_authority": True,
        }
    )
    return enriched


def determine_score_population_decision(
    *,
    enriched_records: list[dict[str, Any]],
    inspections: list[ArtifactInspection],
) -> ScorePopulationDecision:
    if not enriched_records:
        return ScorePopulationDecision.NO_SAFE_RUNTIME_ALIGNMENT_YET

    with_scores = sum(
        1
        for r in enriched_records
        if any(
            (r.get("model_scores") or {}).get(f, {}).get("available")
            for f in MODEL_FAMILIES
        )
    )
    missing_compatible = sum(
        1
        for r in enriched_records
        if not has_model_compatible_runtime_id(r.get("candidate_identity") or {})
    )

    safe_artifacts = [i for i in inspections if i.safe_for_score_population]
    has_exact_id_artifacts = any(i.id_columns for i in inspections)
    only_unsafe = inspections and not has_exact_id_artifacts and not safe_artifacts

    if only_unsafe:
        return ScorePopulationDecision.BLOCKED_UNSAFE_ALIGNMENT_ONLY

    if missing_compatible == len(enriched_records):
        return ScorePopulationDecision.RUNTIME_IDENTITY_BRIDGE_REQUIRED

    if with_scores == 0 and missing_compatible > len(enriched_records) // 2:
        return ScorePopulationDecision.NO_SAFE_RUNTIME_ALIGNMENT_YET

    if with_scores == len(enriched_records) and with_scores > 0:
        return ScorePopulationDecision.MODEL_SCORE_SLOTS_POPULATED

    if with_scores > 0:
        return ScorePopulationDecision.PARTIAL_MODEL_SCORE_SLOT_POPULATION

    return ScorePopulationDecision.RUNTIME_IDENTITY_BRIDGE_REQUIRED


def build_ae7_audit_summary(
    *,
    ae6_records: list[dict[str, Any]],
    enriched_records: list[dict[str, Any]],
    inspections: list[ArtifactInspection],
    score_population_decision: ScorePopulationDecision,
) -> dict[str, Any]:
    ae6_id_counts = {
        "target_row_id": 0,
        "candidate_policy_id": 0,
        "candidate_id": 0,
    }
    for r in ae6_records:
        keys = runtime_id_keys_present(r.get("candidate_identity") or {})
        for k, present in keys.items():
            if present:
                ae6_id_counts[k] += 1

    safe_preds = [i for i in inspections if i.safe_for_score_population]
    reproducible_preds = [i for i in inspections if i.is_reproducible]

    runtime_with_score = sum(
        1
        for r in enriched_records
        if any(
            (r.get("model_scores") or {}).get(f, {}).get("available")
            for f in MODEL_FAMILIES
        )
    )
    runtime_without_alignment = sum(
        1
        for r in enriched_records
        if not any(
            (r.get("model_scores") or {}).get(f, {}).get("available")
            for f in MODEL_FAMILIES
        )
        and has_model_compatible_runtime_id(r.get("candidate_identity") or {})
    )
    runtime_missing_id = sum(
        1
        for r in enriched_records
        if not has_model_compatible_runtime_id(r.get("candidate_identity") or {})
    )

    rejected = [i for i in inspections if not i.safe_for_score_population]
    stale_deprecated = [
        i
        for i in inspections
        if i.artifact_status in {"STALE", "DEPRECATED"}
    ]
    unregistered = [
        i
        for i in inspections
        if i.artifact_status in {"UNREGISTERED", "UNKNOWN"}
    ]
    too_large = [i for i in inspections if i.artifact_status == "TOO_LARGE_SKIPPED"]

    consensus_dist = Counter(
        (r.get("consensus") or {}).get("consensus_family") for r in enriched_records
    )
    status_dist = Counter(r.get("decision_status") for r in enriched_records)

    bridge_required = score_population_decision in {
        ScorePopulationDecision.RUNTIME_IDENTITY_BRIDGE_REQUIRED,
        ScorePopulationDecision.NO_SAFE_RUNTIME_ALIGNMENT_YET,
    }

    return {
        "phase": AE7_PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ae6_record_count": len(ae6_records),
        "ae6_records_with_target_row_id": ae6_id_counts["target_row_id"],
        "ae6_records_with_candidate_policy_id": ae6_id_counts["candidate_policy_id"],
        "ae6_records_with_candidate_id": ae6_id_counts["candidate_id"],
        "target_row_id_not_expected_at_runtime": True,
        "score_population_modes": {
            "implemented_in_ae7": SCORE_POPULATION_MODE_HISTORICAL_OFFLINE,
            "not_implemented_in_ae7": SCORE_POPULATION_MODE_RUNTIME_LIVE,
            "historical_offline_description": (
                "DecisionRecord -> exact ID lookup in existing offline prediction artifacts"
            ),
            "runtime_live_description": (
                "DecisionRecord -> as-of feature row -> model artifact inference -> model score slot"
            ),
        },
        "safe_prediction_artifact_count": len(safe_preds),
        "reproducible_prediction_artifact_count": len(reproducible_preds),
        "safe_rf_artifact_count": sum(1 for i in safe_preds if i.model_family == "RF"),
        "safe_xgb_artifact_count": sum(1 for i in safe_preds if i.model_family == "XGB"),
        "safe_tab_artifact_count": sum(1 for i in safe_preds if i.model_family == "TAB"),
        "total_artifacts_inspected": len(inspections),
        "rejected_artifact_count": len(rejected),
        "stale_deprecated_artifact_count": len(stale_deprecated),
        "unregistered_unknown_artifact_count": len(unregistered),
        "too_large_skipped_count": len(too_large),
        "records_enriched": len(enriched_records),
        "runtime_records_with_any_model_score": runtime_with_score,
        "runtime_records_without_safe_alignment": runtime_without_alignment,
        "runtime_records_missing_model_compatible_id": runtime_missing_id,
        "records_with_rf_score": sum(
            1 for r in enriched_records if (r.get("model_scores") or {}).get("RF", {}).get("available")
        ),
        "records_with_xgb_score": sum(
            1 for r in enriched_records if (r.get("model_scores") or {}).get("XGB", {}).get("available")
        ),
        "records_with_tab_score": sum(
            1 for r in enriched_records if (r.get("model_scores") or {}).get("TAB", {}).get("available")
        ),
        "score_population_decision": score_population_decision.value,
        "next_required_bridge_phase": (
            {
                "name": BRIDGE_PHASE_NAME,
                "goal": BRIDGE_PHASE_GOAL,
                "required_runtime_bridge_fields": list(RUNTIME_BRIDGE_FIELDS),
                "note": (
                    "target_row_id is a historical labeled-row key for offline artifact "
                    "lookup only — not a normal live collection field."
                ),
            }
            if bridge_required
            else None
        ),
        "consensus_family_distribution": dict(consensus_dist),
        "decision_status_distribution": dict(status_dist),
        "lineage_preservation": "AE6 lineage copied unchanged into enriched records",
        "jsonl_persistence": "append-only with flush+fsync per record",
        "safety": {
            "no_model_training": True,
            "no_llm_calls": True,
            "no_external_api_calls": True,
            "no_live_trading": True,
            "no_paper_trade_execution": True,
            "no_fuzzy_pair_time_joins": True,
        },
    }


def run_ae7_score_population(
    *,
    ae6_jsonl: Path,
    inventory_csv: Path,
    project_root: Path,
    max_records: int = 50,
    max_artifacts: int = 50,
    output_root: Path | None = None,
    audit_only: bool = False,
    timestamp_slug: str | None = None,
) -> dict[str, Any]:
    """Full AE7 pipeline: classify artifacts, build index, enrich records, write audit."""
    from scripts.diagnostics._common import timestamp_slug as default_slug

    slug = timestamp_slug or default_slug()
    output_root = output_root or (project_root / "data" / "audits")
    audit_dir = output_root / f"ae7_model_score_slot_population_{slug}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    ae6_records, _ = read_jsonl_records_safe(ae6_jsonl)
    ae6_records = ae6_records[:max_records]

    inventory_rows = load_inventory_csv(inventory_csv)
    inspections = classify_inventory(
        inventory_rows,
        project_root=project_root,
        max_artifacts=max_artifacts,
    )

    matrix_path = audit_dir / "ae7_artifact_reproducibility_matrix.csv"
    write_reproducibility_matrix(inspections, matrix_path)

    index = build_prediction_index(inspections, project_root=project_root)

    inspections_by_family: dict[str, list[ArtifactInspection]] = {}
    for insp in inspections:
        inspections_by_family.setdefault(insp.model_family, []).append(insp)

    enriched_records: list[dict[str, Any]] = []
    jsonl_path: Path | None = None

    if audit_only:
        for rec in ae6_records:
            enriched_records.append(
                enrich_decision_record(
                    rec,
                    index,
                    inspections_by_family=inspections_by_family,
                )
            )
    else:
        jsonl_path = ae7_enriched_jsonl_path_for_date()
        with AE7JsonlWriter(path=jsonl_path) as writer:
            for rec in ae6_records:
                enriched = enrich_decision_record(
                    rec,
                    index,
                    inspections_by_family=inspections_by_family,
                )
                writer.append_record(enriched)
                enriched_records.append(enriched)

    decision = determine_score_population_decision(
        enriched_records=enriched_records,
        inspections=inspections,
    )
    summary = build_ae7_audit_summary(
        ae6_records=ae6_records,
        enriched_records=enriched_records,
        inspections=inspections,
        score_population_decision=decision,
    )
    summary["audit_dir"] = str(audit_dir)
    summary["reproducibility_matrix_path"] = str(matrix_path)
    summary["ae6_jsonl_path"] = str(ae6_jsonl)
    summary["inventory_csv_path"] = str(inventory_csv)
    if jsonl_path:
        summary["ae7_jsonl_path"] = str(jsonl_path)
    summary["audit_only"] = audit_only

    # Offline alignment audit CSV
    audit_csv_path = audit_dir / "ae7_offline_alignment_audit.csv"
    import csv

    audit_rows = [i.to_audit_row() for i in inspections]
    if audit_rows:
        with open(audit_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
    summary["offline_alignment_audit_path"] = str(audit_csv_path)

    summary_path = audit_dir / "ae7_model_score_slot_population_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
