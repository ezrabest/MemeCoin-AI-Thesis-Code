"""AE7B runtime feature matrix bridge orchestration."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.decision.bridge_persistence import RuntimeBridgeJsonlWriter, runtime_bridge_jsonl_path_for_date
from app.decision.feature_parity import FeatureParityStatus, run_feature_parity_check, write_feature_parity_csv
from app.decision.feature_schema import (
    BridgeReadinessDecision,
    RuntimeFeatureSchema,
    build_feature_values,
    build_runtime_feature_schema,
    infer_model_family_from_schema_path,
)
from app.decision.runtime_identity import (
    CandidateIdentityStatus,
    build_bridge_lineage,
    build_identity_payload,
    default_scoring_policy,
    generate_as_of_feature_row_id,
    generate_runtime_inference_id_placeholder,
    generate_candidate_id,
)
from app.decision.types import LineageResolutionMethod

AE7B_PHASE = "AE7B_RUNTIME_IDENTITY_FEATURE_MATRIX_BRIDGE"

MAX_SCHEMA_FILE_BYTES = 5 * 1024 * 1024
DEFAULT_PREFLIGHT_SCHEMA_CSV = (
    "data/audits/ae7b_0_runtime_identity_feature_bridge_preflight_20260710_132927"
    "/ae7b_0_model_schema_candidate_files.csv"
)


def fetch_recent_signal_bundles(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    lookback_hours: float = 24.0,
) -> list[dict[str, Any]]:
    """Read-only fetch of recent signals with best-effort snapshot/raw/coin joins."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    rows = conn.execute(
        """
        SELECT s.id AS signal_id, s.timestamp AS signal_timestamp, s.coin_id,
               s.symbol, s.signal_type, s.score, s.confidence, s.reason,
               s.model_source, s.features_json,
               c.pair_address, c.chain, c.token_address, c.symbol AS coin_symbol,
               c.quote_symbol
        FROM signals s
        LEFT JOIN coins c ON c.id = s.coin_id
        WHERE s.timestamp >= ?
        ORDER BY s.id DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()

    bundles: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        coin_id = row_dict.get("coin_id")
        pair_address = row_dict.get("pair_address")
        signal_ts = row_dict.get("signal_timestamp")
        signal_id = row_dict.get("signal_id")

        snapshot_row = None
        if coin_id is not None and signal_ts:
            snap = conn.execute(
                """
                SELECT *
                FROM market_snapshots
                WHERE coin_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (coin_id, signal_ts),
            ).fetchone()
            if snap:
                snapshot_row = dict(snap)

        raw_row = None
        if pair_address and signal_ts:
            raw = conn.execute(
                """
                SELECT id, timestamp, provider, source_type, query, chain,
                       pair_address, symbol, payload_hash
                FROM raw_provider_payloads
                WHERE pair_address = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (pair_address, signal_ts),
            ).fetchone()
            if raw:
                raw_row = dict(raw)

        sentiment_agg = None
        symbol = row_dict.get("symbol") or row_dict.get("coin_symbol")
        if symbol and signal_ts:
            sent_rows = conn.execute(
                """
                SELECT sentiment_score, source
                FROM sentiment_records
                WHERE symbols_json LIKE ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                (f"%{symbol}%", signal_ts),
            ).fetchall()
            if sent_rows:
                scores = [float(r[0]) for r in sent_rows if r[0] is not None]
                sources = {str(r[1]) for r in sent_rows if r[1]}
                sentiment_agg = {
                    "sentiment_score": round(sum(scores) / len(scores), 6) if scores else None,
                    "source_count": len(sources),
                }

        bundles.append(
            {
                "signal_row": {
                    "id": signal_id,
                    "timestamp": signal_ts,
                    "coin_id": coin_id,
                    "symbol": symbol,
                    "signal_type": row_dict.get("signal_type"),
                    "score": row_dict.get("score"),
                    "confidence": row_dict.get("confidence"),
                    "reason": row_dict.get("reason"),
                    "model_source": row_dict.get("model_source"),
                },
                "snapshot_row": snapshot_row,
                "raw_payload_row": raw_row,
                "coin_row": {
                    "pair_address": pair_address,
                    "chain": row_dict.get("chain"),
                    "token_address": row_dict.get("token_address"),
                    "quote_symbol": row_dict.get("quote_symbol"),
                    "symbol": symbol,
                },
                "sentiment_agg": sentiment_agg,
            }
        )
    return bundles


def build_runtime_bridge_record(
    bundle: dict[str, Any],
    *,
    schema: RuntimeFeatureSchema,
    scoring_policy: Any | None = None,
    policy_context: dict[str, Any] | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Build one runtime bridge record from a signal bundle."""
    scoring_policy = scoring_policy or default_scoring_policy()
    signal_row = bundle.get("signal_row") or {}
    snapshot_row = bundle.get("snapshot_row")
    raw_row = bundle.get("raw_payload_row")
    coin_row = bundle.get("coin_row") or {}

    as_of_ts = (
        signal_row.get("timestamp")
        or (snapshot_row or {}).get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    identity_payload = build_identity_payload(
        chain=coin_row.get("chain") or (snapshot_row or {}).get("chain"),
        pair_address=coin_row.get("pair_address") or (snapshot_row or {}).get("pair_address"),
        base_token_address=coin_row.get("token_address"),
        quote_token_address=None,
        symbol=signal_row.get("symbol") or coin_row.get("symbol"),
        event_timestamp=as_of_ts,
        source_table="signals",
        source_row_id=signal_row.get("id"),
        provider=(snapshot_row or {}).get("provider") or (raw_row or {}).get("provider"),
    )

    candidate_id, identity_status, identity_caveats = generate_candidate_id(identity_payload)

    lineage = build_bridge_lineage(
        signal_id=signal_row.get("id"),
        snapshot_id=(snapshot_row or {}).get("id"),
        raw_payload_id=(raw_row or {}).get("id"),
        signal_method=(
            LineageResolutionMethod.EXPLICIT_COLUMN
            if signal_row.get("id")
            else LineageResolutionMethod.MISSING
        ),
        snapshot_method=(
            LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH
            if snapshot_row
            else LineageResolutionMethod.MISSING
        ),
        raw_method=(
            LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH
            if raw_row
            else LineageResolutionMethod.MISSING
        ),
    )

    feature_result = build_feature_values(
        snapshot_row=snapshot_row,
        signal_row=signal_row,
        sentiment_agg=bundle.get("sentiment_agg"),
        schema=schema,
        policy_context=policy_context,
    )

    as_of_feature_row_id = None
    runtime_inference_id = None
    bridge_status = BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_PARTIAL.value

    if candidate_id and identity_status == CandidateIdentityStatus.OK:
        as_of_feature_row_id = generate_as_of_feature_row_id(
            candidate_id=candidate_id,
            scoring_policy_id=scoring_policy.scoring_policy_id,
            feature_schema_id=schema.feature_schema_id,
            as_of_timestamp=as_of_ts,
            source_snapshot_id=(snapshot_row or {}).get("id"),
            source_signal_id=signal_row.get("id"),
        )
        runtime_inference_id = generate_runtime_inference_id_placeholder(
            candidate_id, as_of_feature_row_id
        )
        if feature_result.has_schema_gap:
            bridge_status = BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP.value
        elif feature_result.feature_status == "OK":
            bridge_status = BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_CREATED.value
        else:
            bridge_status = BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_PARTIAL.value
    else:
        bridge_status = BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP.value

    return {
        "phase": phase or AE7B_PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_status": bridge_status,
        "candidate_id": candidate_id,
        "candidate_identity_status": identity_status.value,
        "candidate_identity_caveats": identity_caveats,
        "scoring_policy_id": scoring_policy.scoring_policy_id,
        "scoring_policy_version": scoring_policy.scoring_policy_version,
        "scoring_policy_status": scoring_policy.scoring_policy_status,
        "model_family_targets": scoring_policy.model_family_targets,
        "horizon_candidates": scoring_policy.horizon_candidates,
        "exit_policy_candidates": scoring_policy.exit_policy_candidates,
        "policy_source": scoring_policy.policy_source,
        "as_of_feature_row_id": as_of_feature_row_id,
        "feature_schema_id": schema.feature_schema_id,
        "feature_schema_version": schema.feature_schema_version,
        "feature_values": feature_result.feature_values,
        "feature_missingness": feature_result.feature_missingness,
        "feature_missingness_reasons": feature_result.feature_missingness_reasons,
        "feature_source_columns": feature_result.feature_source_columns,
        "feature_source_tables": feature_result.feature_source_tables,
        "feature_status": feature_result.feature_status,
        "missing_required_features": feature_result.missing_required,
        "rejected_features": feature_result.rejected_features,
        "whale_score_metadata": feature_result.whale_score_metadata,
        "policy_feature_metadata": feature_result.policy_feature_metadata,
        "feature_timestamp": as_of_ts,
        "as_of_timestamp": as_of_ts,
        "model_artifact_id": None,
        "runtime_inference_id": runtime_inference_id,
        "lineage": lineage.to_dict(),
        "no_trade_authority": True,
        "llm_decision_authority": False,
        "target_row_id_not_required": True,
        "target_row_id": None,
        "used_for_inference": False,
    }


def load_model_schema_from_json(path: Path) -> tuple[list[str], str]:
    """Load feature column names from small schema JSON files."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "feature_columns" in data:
        return list(data["feature_columns"]), "model_schema"
    if "safe_core_features" in data:
        return list(data["safe_core_features"]), "report_schema"
    if "features" in data and isinstance(data["features"], list):
        return list(data["features"]), "generic_schema"
    return [], "unknown_schema"


def build_model_schema_compatibility_matrix(
    *,
    runtime_schema: RuntimeFeatureSchema,
    schema_candidate_paths: list[Path],
    project_root: Path,
    max_schemas: int = 30,
    parity_status: str | None = None,
    weak_lineage: bool = True,
) -> list[dict[str, Any]]:
    """Compare runtime feature schema to inspectable model schema JSON files."""
    from app.decision.feature_parity import FeatureParityStatus

    runtime_features = set(runtime_schema.feature_names)
    rows: list[dict[str, Any]] = []

    inspected = 0
    for rel_path in schema_candidate_paths:
        if inspected >= max_schemas:
            break
        path = project_root / str(rel_path).replace("\\", "/")
        if not path.is_file():
            continue
        if path.suffix.lower() != ".json":
            continue
        if "schema" not in path.name.lower() and "feature" not in path.name.lower():
            continue
        if path.stat().st_size > MAX_SCHEMA_FILE_BYTES:
            rows.append(
                {
                    "model_family": infer_model_family_from_schema_path(str(path)),
                    "model_artifact_path": "",
                    "schema_source_path": str(rel_path),
                    "schema_status": "TOO_LARGE",
                    "required_feature_count": 0,
                    "runtime_available_feature_count": len(runtime_features),
                    "missing_feature_count": 0,
                    "extra_runtime_feature_count": 0,
                    "missing_features_sample": "",
                    "extra_features_sample": "",
                    "compatibility_status": "BLOCKED_UNSUPPORTED_ARTIFACT",
                    "compatibility_reason": "schema_file_too_large_for_safe_inspection",
                    "safe_for_future_inference": False,
                }
            )
            inspected += 1
            continue

        try:
            model_features, schema_kind = load_model_schema_from_json(path)
        except (json.JSONDecodeError, OSError):
            rows.append(
                {
                    "model_family": infer_model_family_from_schema_path(str(path)),
                    "model_artifact_path": "",
                    "schema_source_path": str(rel_path),
                    "schema_status": "UNREADABLE",
                    "required_feature_count": 0,
                    "runtime_available_feature_count": len(runtime_features),
                    "missing_feature_count": 0,
                    "extra_runtime_feature_count": 0,
                    "missing_features_sample": "",
                    "extra_features_sample": "",
                    "compatibility_status": "UNKNOWN",
                    "compatibility_reason": "schema_parse_failed",
                    "safe_for_future_inference": False,
                }
            )
            inspected += 1
            continue

        if not model_features:
            rows.append(
                {
                    "model_family": infer_model_family_from_schema_path(str(path)),
                    "model_artifact_path": "",
                    "schema_source_path": str(rel_path),
                    "schema_status": schema_kind,
                    "required_feature_count": 0,
                    "runtime_available_feature_count": len(runtime_features),
                    "missing_feature_count": 0,
                    "extra_runtime_feature_count": len(runtime_features),
                    "missing_features_sample": "",
                    "extra_features_sample": "|".join(sorted(runtime_features)[:10]),
                    "compatibility_status": "BLOCKED_MISSING_SCHEMA",
                    "compatibility_reason": "no_feature_columns_in_schema",
                    "safe_for_future_inference": False,
                }
            )
            inspected += 1
            continue

        model_set = set(model_features)
        runtime_alias = {
            "liquidity_usd": "liquidity",
            "price_usd": "price",
            "volume_h24": "volume_24h",
            "buy_sell_ratio_h24": "buy_ratio",
            "whale_score_asof": "whale_score",
            "txns_h24_buys": "txns_buys",
            "txns_h24_sells": "txns_sells",
            "txns_h24_total": "txns_total",
        }
        runtime_mapped = set()
        for rf in runtime_features:
            runtime_mapped.add(rf)
            if rf in runtime_alias:
                runtime_mapped.add(runtime_alias[rf])

        missing = sorted(model_set - runtime_mapped)
        extra = sorted(runtime_mapped - model_set)
        overlap = model_set & runtime_mapped

        if not missing:
            compat = "COMPATIBLE"
            reason = "all_model_features_available_in_runtime_schema"
        elif overlap:
            compat = "PARTIAL_MISSING_FEATURES"
            reason = f"missing_{len(missing)}_model_features_in_runtime_schema"
        else:
            compat = "PARTIAL_MISSING_FEATURES"
            reason = "no_feature_name_overlap"

        schema_safe = compat == "COMPATIBLE" and len(missing) == 0
        if parity_status and parity_status != FeatureParityStatus.PASS.value:
            schema_safe = False
        if weak_lineage:
            schema_safe = False

        rows.append(
            {
                "model_family": infer_model_family_from_schema_path(str(path)),
                "model_artifact_path": "",
                "schema_source_path": str(rel_path),
                "schema_status": schema_kind,
                "required_feature_count": len(model_set),
                "runtime_available_feature_count": len(overlap),
                "missing_feature_count": len(missing),
                "extra_runtime_feature_count": len(extra),
                "missing_features_sample": "|".join(missing[:15]),
                "extra_features_sample": "|".join(extra[:15]),
                "compatibility_status": compat,
                "compatibility_reason": reason,
                "safe_for_future_inference": schema_safe,
            }
        )
        inspected += 1

    return rows


def write_compatibility_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"compatibility_status": "UNKNOWN", "compatibility_reason": "no_schemas_inspected"}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def determine_bridge_readiness(
    records: list[dict[str, Any]],
    parity_status: str,
) -> str:
    if not records:
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP.value

    statuses = Counter(r.get("bridge_status") for r in records)
    blocked_schema = statuses.get(
        BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP.value, 0
    )
    blocked_lineage = statuses.get(
        BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP.value, 0
    )
    created = statuses.get(BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_CREATED.value, 0)
    partial = statuses.get(BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_PARTIAL.value, 0)

    if parity_status == FeatureParityStatus.FAIL_MISMATCH.value:
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_PARITY_GAP.value
    if blocked_lineage == len(records):
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP.value
    if blocked_schema == len(records):
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP.value
    if created == len(records):
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_CREATED.value
    if created > 0 or partial > 0:
        return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_PARTIAL.value
    return BridgeReadinessDecision.RUNTIME_FEATURE_BRIDGE_PARTIAL.value


def build_ae7b_audit_summary(
    *,
    records: list[dict[str, Any]],
    bundles_count: int,
    parity_result: Any,
    compatibility_rows: list[dict[str, Any]],
    bridge_readiness: str,
    audit_dir: str,
    jsonl_path: str | None,
    audit_only: bool,
    sqlite_writes: bool,
) -> dict[str, Any]:
    lineage_modes = Counter((r.get("lineage") or {}).get("lineage_mode") for r in records)
    lineage_strengths = Counter((r.get("lineage") or {}).get("lineage_strength") for r in records)
    confidence_scores = [
        (r.get("lineage") or {}).get("lineage_confidence_score") for r in records
    ]
    exact_matches = Counter((r.get("lineage") or {}).get("exact_id_match") for r in records)

    schema = build_runtime_feature_schema()
    feature_counts = {
        "feature_count": len(schema.feature_names),
        "required_feature_count": len(schema.required_features),
        "optional_feature_count": len(schema.optional_features),
        "rejected_feature_count": len(schema.rejected_features),
    }

    missing_req_total = sum(len(r.get("missing_required_features") or []) for r in records)
    missing_opt_total = sum(len(r.get("feature_missingness") or []) for r in records)

    compat_status = Counter(r.get("compatibility_status") for r in compatibility_rows)

    return {
        "phase": AE7B_PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signals_bundles_fetched": bundles_count,
        "bridge_records_created": len(records),
        "candidate_id_generated": sum(1 for r in records if r.get("candidate_id")),
        "scoring_policy_id_generated": sum(1 for r in records if r.get("scoring_policy_id")),
        "as_of_feature_row_id_generated": sum(1 for r in records if r.get("as_of_feature_row_id")),
        "feature_schema_id": schema.feature_schema_id,
        "records_blocked_missing_stable_identity": sum(
            1
            for r in records
            if r.get("candidate_identity_status")
            == CandidateIdentityStatus.BLOCKED_MISSING_STABLE_IDENTITY.value
        ),
        **feature_counts,
        "missing_required_feature_count": missing_req_total,
        "missing_optional_feature_count": missing_opt_total,
        "whale_score_asof_handling": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE with not_rule=true",
        "feature_parity": parity_result.to_summary_dict(),
        "lineage_mode_distribution": dict(lineage_modes),
        "lineage_strength_distribution": dict(lineage_strengths),
        "lineage_confidence_score_min": min(confidence_scores) if confidence_scores else None,
        "lineage_confidence_score_max": max(confidence_scores) if confidence_scores else None,
        "lineage_confidence_score_mean": (
            round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else None
        ),
        "exact_id_match_distribution": dict(exact_matches),
        "lineage_fallback_count": sum(
            1
            for r in records
            if (r.get("lineage") or {}).get("lineage_mode") == "BEST_EFFORT_IMPLICIT_LINKAGE"
        ),
        "model_schema_candidates_inspected": len(compatibility_rows),
        "compatibility_status_distribution": dict(compat_status),
        "compatible_schema_count": compat_status.get("COMPATIBLE", 0),
        "partial_missing_features_count": compat_status.get("PARTIAL_MISSING_FEATURES", 0),
        "blocked_unknown_schema_count": (
            compat_status.get("BLOCKED_MISSING_SCHEMA", 0)
            + compat_status.get("UNKNOWN", 0)
            + compat_status.get("BLOCKED_UNSUPPORTED_ARTIFACT", 0)
        ),
        "bridge_readiness_decision": bridge_readiness,
        "target_row_id_not_required_at_runtime": True,
        "audit_dir": audit_dir,
        "jsonl_path": jsonl_path,
        "audit_only": audit_only,
        "sqlite_runtime_feature_rows_written": sqlite_writes,
        "safety": {
            "no_model_training": True,
            "no_model_inference": True,
            "no_llm_calls": True,
            "no_external_api_calls": True,
            "no_live_trading": True,
            "no_paper_trade_execution": True,
            "no_sqlite_runtime_feature_row_writes": not sqlite_writes,
        },
    }


def run_ae7b_bridge(
    *,
    project_root: Path,
    conn: sqlite3.Connection,
    max_records: int = 50,
    lookback_hours: float = 24.0,
    output_root: Path | None = None,
    audit_only: bool = False,
    parity_check: bool = True,
    schema_candidates_csv: Path | None = None,
    timestamp_slug: str | None = None,
) -> dict[str, Any]:
    from scripts.diagnostics._common import timestamp_slug as default_slug

    slug = timestamp_slug or default_slug()
    output_root = output_root or (project_root / "data" / "audits")
    audit_dir = output_root / f"ae7b_runtime_identity_feature_bridge_{slug}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    schema = build_runtime_feature_schema()
    policy = default_scoring_policy()
    bundles = fetch_recent_signal_bundles(
        conn, limit=max_records, lookback_hours=lookback_hours
    )

    records: list[dict[str, Any]] = []
    jsonl_path: str | None = None

    if audit_only:
        for bundle in bundles:
            records.append(build_runtime_bridge_record(bundle, schema=schema, scoring_policy=policy))
    else:
        path = runtime_bridge_jsonl_path_for_date()
        jsonl_path = str(path)
        with RuntimeBridgeJsonlWriter(path=path) as writer:
            for bundle in bundles:
                rec = build_runtime_bridge_record(bundle, schema=schema, scoring_policy=policy)
                writer.append_record(rec)
                records.append(rec)

    parity_result = (
        run_feature_parity_check(runtime_bridge_records=records)
        if parity_check
        else run_feature_parity_check(runtime_bridge_records=[])
    )
    write_feature_parity_csv(parity_result, audit_dir / "ae7b_feature_parity_check.csv")

    schema_csv = schema_candidates_csv or (project_root / DEFAULT_PREFLIGHT_SCHEMA_CSV)
    candidate_paths: list[Path] = []
    if schema_csv.is_file():
        with open(schema_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("path", "")
                if p and p.endswith(".json") and ("schema" in p.lower() or "feature" in p.lower()):
                    candidate_paths.append(Path(p))
                if len(candidate_paths) >= 30:
                    break

    compatibility_rows = build_model_schema_compatibility_matrix(
        runtime_schema=schema,
        schema_candidate_paths=candidate_paths,
        project_root=project_root,
        max_schemas=30,
    )
    compat_path = audit_dir / "ae7b_model_schema_compatibility_matrix.csv"
    write_compatibility_matrix(compatibility_rows, compat_path)

    bridge_readiness = determine_bridge_readiness(records, parity_result.feature_parity_status)

    summary = build_ae7b_audit_summary(
        records=records,
        bundles_count=len(bundles),
        parity_result=parity_result,
        compatibility_rows=compatibility_rows,
        bridge_readiness=bridge_readiness,
        audit_dir=str(audit_dir),
        jsonl_path=jsonl_path,
        audit_only=audit_only,
        sqlite_writes=False,
    )
    summary["compatibility_matrix_path"] = str(compat_path)
    summary["feature_parity_path"] = str(audit_dir / "ae7b_feature_parity_check.csv")

    summary_path = audit_dir / "ae7b_runtime_identity_feature_bridge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
