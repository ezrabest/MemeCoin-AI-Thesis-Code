"""AE8 context feature record builder and orchestration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.context_intelligence.bounded_queries import QueryStats, assess_memory_safety, fetch_context_seed_rows
from app.context_intelligence.context_audits import write_ae8_audits
from app.context_intelligence.context_persistence import ContextJsonlWriter, context_jsonl_path_for_date
from app.context_intelligence.context_schema import ContextSchema, build_context_schema
from app.context_intelligence.freshness import default_threshold_for_family
from app.context_intelligence.liquidity_activity_context import build_liquidity_activity_context
from app.context_intelligence.onchain_context import build_onchain_context
from app.context_intelligence.reputation_context import build_reputation_context
from app.context_intelligence.rss_context import build_rss_context
from app.context_intelligence.types import (
    AE8_LINEAGE_FALLBACK_REASON,
    AE8_LINEAGE_WARNING,
    AE8_PHASE,
    ContextFeatureRecord,
    ContextLineageBlock,
    FreshnessMode,
    LineageValidationStatus,
)
from app.context_intelligence.whale_context import build_whale_context
from app.decision.runtime_identity import (
    build_identity_payload,
    generate_candidate_id,
)
from app.decision.types import LineageMode, LineageResolutionMethod, LineageStrength


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_context_record_id(candidate_id: str, as_of_timestamp: str, schema_id: str) -> str:
    return _sha256_hex(f"AE8_CONTEXT_RECORD|{candidate_id}|{as_of_timestamp}|{schema_id}|{uuid.uuid4()}")


def build_context_lineage(
    *,
    signal_row: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
    raw_payload_row: dict[str, Any] | None,
    freshness_blocks: dict[str, dict[str, Any]],
) -> ContextLineageBlock:
    signal_id = (signal_row or {}).get("id")
    snapshot_id = (snapshot_row or {}).get("id")
    raw_id = (raw_payload_row or {}).get("id")

    has_future = any(
        b.get("freshness_status") == "INVALID_FUTURE_TIMESTAMP" for b in freshness_blocks.values()
    )
    if has_future:
        return ContextLineageBlock(
            lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE.value,
            lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS.value,
            lineage_confidence_score=0.0,
            exact_id_match=False,
            lineage_validation_status=LineageValidationStatus.BLOCKED_FUTURE_TIMESTAMP.value,
            source_tables=["signals", "market_snapshots", "raw_provider_payloads"],
            source_signal_ids=[signal_id] if signal_id else [],
            source_snapshot_ids=[snapshot_id] if snapshot_id else [],
            source_payload_ids=[raw_id] if raw_id else [],
            resolution_methods=[
                LineageResolutionMethod.EXPLICIT_COLUMN.value if signal_id else LineageResolutionMethod.MISSING.value,
                LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH.value
                if snapshot_id
                else LineageResolutionMethod.MISSING.value,
                LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH.value
                if raw_id
                else LineageResolutionMethod.MISSING.value,
            ],
            lineage_warning=AE8_LINEAGE_WARNING,
            fallback_reason=AE8_LINEAGE_FALLBACK_REASON,
        )

    signal_method = (
        LineageResolutionMethod.EXPLICIT_COLUMN if signal_id else LineageResolutionMethod.MISSING
    )
    snapshot_method = (
        LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH
        if snapshot_id
        else LineageResolutionMethod.MISSING
    )
    raw_method = (
        LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH
        if raw_id
        else LineageResolutionMethod.MISSING
    )
    explicit_methods = {
        LineageResolutionMethod.EXPLICIT_COLUMN,
        LineageResolutionMethod.FOREIGN_KEY,
        LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
    }
    all_explicit = all(
        m in explicit_methods
        for m in (signal_method, snapshot_method, raw_method)
        if m != LineageResolutionMethod.MISSING
    ) and signal_id is not None

    if all_explicit and snapshot_id and raw_id:
        return ContextLineageBlock(
            lineage_mode=LineageMode.EXPLICIT_LINKAGE.value,
            lineage_strength=LineageStrength.STRONG_EXPLICIT_LINKS.value,
            lineage_confidence_score=1.0,
            exact_id_match=True,
            lineage_validation_status=LineageValidationStatus.PASS_EXPLICIT_ID_LINKAGE.value,
            source_tables=["signals", "market_snapshots", "raw_provider_payloads"],
            source_signal_ids=[signal_id],
            source_snapshot_ids=[snapshot_id],
            source_payload_ids=[raw_id],
            resolution_methods=[
                signal_method.value,
                snapshot_method.value,
                raw_method.value,
            ],
        )

    confidence = 0.35
    return ContextLineageBlock(
        lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE.value,
        lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS.value,
        lineage_confidence_score=confidence,
        exact_id_match=False,
        lineage_validation_status=LineageValidationStatus.PASS_WEAK_BEST_EFFORT_WITH_WARNING.value,
        source_tables=["signals", "market_snapshots", "raw_provider_payloads"],
        source_signal_ids=[signal_id] if signal_id else [],
        source_snapshot_ids=[snapshot_id] if snapshot_id else [],
        source_payload_ids=[raw_id] if raw_id else [],
        resolution_methods=[
            signal_method.value,
            snapshot_method.value,
            raw_method.value,
        ],
        lineage_warning=AE8_LINEAGE_WARNING,
        fallback_reason=AE8_LINEAGE_FALLBACK_REASON,
    )


def _collect_missingness(
    *,
    rss: dict[str, Any],
    onchain: dict[str, Any],
    whale: dict[str, Any],
    reputation: dict[str, Any],
    liquidity: dict[str, Any],
    source_statuses: dict[str, str],
) -> dict[str, Any]:
    families = {
        "rss": rss.get("rss_missingness_flag", True),
        "onchain": onchain.get("onchain_missingness_flag", True),
        "whale": whale.get("whale_score_missingness", True),
        "reputation": reputation.get("reputation_missingness_flag", True),
        "liquidity_activity": liquidity.get("liquidity_activity_missingness_flag", True),
    }
    missing_families = [k for k, v in families.items() if v]
    return {
        "family_missingness_flags": families,
        "missing_families": missing_families,
        "source_statuses": source_statuses,
    }


def build_context_feature_record(
    bundle: dict[str, Any],
    *,
    schema: ContextSchema,
    run_started_at_utc: str,
    freshness_mode: FreshnessMode | str,
    freshness_reference_timestamp: str,
    conn: sqlite3.Connection | None,
    allow_external_fetch: bool,
    stats: QueryStats,
) -> ContextFeatureRecord | None:
    signal_row = bundle.get("signal_row") or {}
    snapshot_row = bundle.get("snapshot_row")
    raw_row = bundle.get("raw_payload_row")
    coin_row = bundle.get("coin_row") or {}

    as_of_ts = signal_row.get("timestamp") or (snapshot_row or {}).get("timestamp") or run_started_at_utc
    symbol = signal_row.get("symbol") or coin_row.get("symbol")
    chain = coin_row.get("chain") or (snapshot_row or {}).get("chain")
    pair_address = coin_row.get("pair_address") or (snapshot_row or {}).get("pair_address")

    identity_payload = build_identity_payload(
        chain=chain,
        pair_address=pair_address,
        base_token_address=coin_row.get("token_address"),
        symbol=symbol,
        event_timestamp=as_of_ts,
        source_table="signals",
        source_row_id=signal_row.get("id"),
        provider=(snapshot_row or {}).get("provider") or (raw_row or {}).get("provider"),
    )
    candidate_id, identity_status, _ = generate_candidate_id(identity_payload)
    if candidate_id is None:
        return None

    thresholds = schema.freshness_thresholds

    rss_features, rss_fresh, rss_status, rss_warnings = build_rss_context(
        conn,
        symbol=symbol,
        as_of_timestamp=as_of_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=thresholds.get("rss", default_threshold_for_family("rss")),
        allow_external_fetch=allow_external_fetch,
        stats=stats,
    )

    onchain_features, onchain_fresh, onchain_status, onchain_warnings = build_onchain_context(
        raw_payload_row=raw_row,
        as_of_timestamp=as_of_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=thresholds.get("onchain", default_threshold_for_family("onchain")),
        allow_external_fetch=allow_external_fetch,
    )

    whale_features, whale_fresh, whale_status, whale_warnings, _ = build_whale_context(
        conn,
        coin_id=signal_row.get("coin_id"),
        pair_address=pair_address,
        snapshot_row=snapshot_row,
        as_of_timestamp=as_of_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=thresholds.get("whale", default_threshold_for_family("whale")),
        stats=stats,
    )

    reputation_features, rep_fresh, rep_status, rep_warnings = build_reputation_context(
        raw_payload_row=raw_row,
        coin_row=coin_row,
        as_of_timestamp=as_of_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=thresholds.get("reputation", default_threshold_for_family("reputation")),
        allow_external_fetch=allow_external_fetch,
    )

    liq_features, liq_fresh, liq_status, liq_warnings = build_liquidity_activity_context(
        snapshot_row=snapshot_row,
        prior_snapshot_row=bundle.get("prior_snapshot_row"),
        prior_6h_snapshot_row=bundle.get("prior_6h_snapshot_row"),
        signal_row=signal_row,
        as_of_timestamp=as_of_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=thresholds.get(
            "liquidity_activity", default_threshold_for_family("liquidity_activity")
        ),
    )

    freshness_blocks = {
        "rss": rss_fresh,
        "onchain": onchain_fresh,
        "whale": whale_fresh,
        "reputation": rep_fresh,
        "liquidity_activity": liq_fresh,
    }

    lineage = build_context_lineage(
        signal_row=signal_row,
        snapshot_row=snapshot_row,
        raw_payload_row=raw_row,
        freshness_blocks=freshness_blocks,
    )

    if lineage.lineage_validation_status == LineageValidationStatus.PASS_WEAK_BEST_EFFORT_WITH_WARNING.value:
        if not lineage.lineage_warning or not lineage.fallback_reason:
            raise ValueError("best_effort lineage requires lineage_warning and fallback_reason")

    source_statuses = {
        "rss": rss_status,
        "onchain": onchain_status,
        "whale": whale_status,
        "reputation": rep_status,
        "liquidity_activity": liq_status,
    }
    source_warnings = rss_warnings + onchain_warnings + whale_warnings + rep_warnings + liq_warnings

    missingness = _collect_missingness(
        rss=rss_features,
        onchain=onchain_features,
        whale=whale_features,
        reputation=reputation_features,
        liquidity=liq_features,
        source_statuses=source_statuses,
    )

    mode_str = (
        freshness_mode.value
        if isinstance(freshness_mode, FreshnessMode)
        else str(freshness_mode)
    )

    record = ContextFeatureRecord(
        context_record_id=_generate_context_record_id(candidate_id, as_of_ts, schema.context_schema_id),
        candidate_id=candidate_id,
        pair_address=pair_address,
        symbol=symbol,
        chain=chain,
        as_of_timestamp=as_of_ts,
        context_schema_id=schema.context_schema_id,
        context_schema_version=schema.context_schema_version,
        freshness_mode=mode_str,
        run_started_at_utc=run_started_at_utc,
        rss_context=rss_features,
        onchain_context=onchain_features,
        whale_context=whale_features,
        reputation_context=reputation_features,
        liquidity_activity_context=liq_features,
        context_missingness=missingness,
        context_freshness=freshness_blocks,
        lineage=lineage.to_dict(),
        source_statuses=source_statuses,
        source_warnings=source_warnings,
    )
    return record


def run_ae8_context_intelligence(
    *,
    project_root: Path,
    conn: sqlite3.Connection,
    max_records: int = 50,
    lookback_hours: float = 24.0,
    output_root: Path | None = None,
    audit_only: bool = True,
    no_db_write: bool = True,
    allow_external_fetch: bool = False,
    freshness_mode: str = "live",
) -> dict[str, Any]:
    del audit_only, no_db_write  # AE8 is always audit-only JSONL; no SQLite writes

    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    mode = (
        FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT
        if freshness_mode in {"historical-replay", "HISTORICAL_REPLAY_OR_AUDIT"}
        else FreshnessMode.LIVE_OR_CURRENT_RUNTIME
    )

    schema = build_context_schema()
    stats = QueryStats()
    bundles = fetch_context_seed_rows(
        conn,
        limit=max_records,
        lookback_hours=lookback_hours,
        stats=stats,
    )

    records: list[ContextFeatureRecord] = []
    for bundle in bundles:
        signal_row = bundle.get("signal_row") or {}
        as_of_ts = signal_row.get("timestamp") or run_started_at_utc
        ref_ts = as_of_ts if mode == FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT else run_started_at_utc

        record = build_context_feature_record(
            bundle,
            schema=schema,
            run_started_at_utc=run_started_at_utc,
            freshness_mode=mode,
            freshness_reference_timestamp=ref_ts,
            conn=conn,
            allow_external_fetch=allow_external_fetch,
            stats=stats,
        )
        if record is not None:
            records.append(record)

    memory_status = assess_memory_safety(stats, stats.queries_executed)

    out_root = output_root or (project_root / "data")
    audit_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = out_root / "audits" / f"ae8_context_intelligence_{audit_ts}"

    jsonl_path = context_jsonl_path_for_date(output_root=out_root / "context_intelligence")
    with ContextJsonlWriter(jsonl_path) as writer:
        for rec in records:
            writer.append_record(rec.to_dict())

    summary = write_ae8_audits(
        project_root=project_root,
        output_root=out_root,
        audit_dir=audit_dir,
        records=records,
        schema=schema,
        stats=stats,
        memory_safety_status=memory_status,
        allow_external_fetch=allow_external_fetch,
        freshness_mode=mode,
        run_started_at_utc=run_started_at_utc,
    )

    summary.update(
        {
            "phase": AE8_PHASE,
            "context_records_created": len(records),
            "context_schema_id": schema.context_schema_id,
            "jsonl_path": str(jsonl_path),
            "audit_dir": str(audit_dir),
        }
    )
    return summary
