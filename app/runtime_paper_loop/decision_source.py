"""Incremental AE6 decision source loader with durable cursor and backlog drain."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.decision.persistence import read_jsonl_records_safe
from app.execution.execution_orchestrator import _index_by_key, _match_ae9
from app.llm_audit.audit_runner import discover_latest_glob
from app.runtime_paper_loop.idempotency import AE11StateDb
from app.runtime_paper_loop.types import utc_now_iso

SOURCE_NAME = "ae6_decisions_jsonl"
DEFAULT_POLICY_VERSION = "AE6_CONSENSUS_DECISION_LAYER"


def build_source_event_key(decision: dict[str, Any]) -> str:
    """Event-level idempotency key — not pair-only."""
    identity = decision.get("candidate_identity") or {}
    market = decision.get("market_context") or {}
    lineage = decision.get("lineage") or {}
    parts = [
        str(identity.get("chain") or market.get("chain") or ""),
        str(identity.get("pair_address") or ""),
        str(
            identity.get("event_timestamp")
            or market.get("event_timestamp")
            or market.get("snapshot_timestamp")
            or ""
        ),
        str(
            market.get("source_snapshot_id")
            or lineage.get("source_snapshot_id")
            or identity.get("source_snapshot_id")
            or ""
        ),
        str(
            identity.get("source_signal_id")
            or lineage.get("source_signal_id")
            or identity.get("signal_id")
            or ""
        ),
        str(decision.get("scoring_policy_id") or decision.get("decision_policy_version") or DEFAULT_POLICY_VERSION),
        str(decision.get("decision_id") or ""),
    ]
    return "|".join(parts)


def _iter_jsonl_from_offset(path: Path, byte_offset: int = 0) -> Iterator[tuple[int, dict[str, Any] | None, str]]:
    """Yield (byte_offset_after_line, record_or_none, raw_line) from byte_offset."""
    with open(path, "rb") as f:
        if byte_offset:
            f.seek(byte_offset)
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            end = f.tell()
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                yield end, None, text
                continue
            try:
                yield end, json.loads(text), text
            except json.JSONDecodeError:
                yield end, None, text


@dataclass
class DecisionSourceLoadResult:
    candidates: list[tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]] = field(
        default_factory=list
    )
    ae6_source_file: str | None = None
    ae6_file_length: int = 0
    ae6_file_last_write_time: str | None = None
    records_scanned: int = 0
    records_seen: int = 0
    records_skipped_processed: int = 0
    records_skipped_invalid: int = 0
    records_selected_for_processing: int = 0
    eof_reached: bool = False
    cursor_before: str | None = None
    cursor_after: str | None = None
    latest_record_created_at_utc: str | None = None
    fresh_generation_attempted: bool = False
    fresh_generation_result: str | None = None
    no_new_decisions_available: bool = False
    decision_source_status: str = "OK"


def discover_ae6_path(project_root: Path) -> Path | None:
    path = discover_latest_glob(project_root, "data/decision_records/ae6_decisions_*.jsonl")
    return path if path and path.is_file() else None


def load_ae8_ae9_indexes(project_root: Path) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    ae8_path = discover_latest_glob(project_root, "data/context_intelligence/ae8_context_features_*.jsonl")
    ae9_path = discover_latest_glob(project_root, "data/llm_audit/ae9_llm_audit_records_*.jsonl")
    ae8_records: list[dict[str, Any]] = []
    ae9_records: list[dict[str, Any]] = []
    if ae8_path and ae8_path.is_file():
        ae8_records, _ = read_jsonl_records_safe(ae8_path)
    if ae9_path and ae9_path.is_file():
        ae9_records, _ = read_jsonl_records_safe(ae9_path)
    from app.execution.execution_orchestrator import _index_ae9_records

    ae8_index = _index_by_key(ae8_records, "candidate_id", "pair_address", "context_record_id")
    ae9_by_key, ae9_by_decision_id = _index_ae9_records(ae9_records)
    return ae8_index, ae9_by_key, ae9_by_decision_id


def _match_candidate(
    decision: dict[str, Any],
    ae8_index: dict[str, dict],
    ae9_by_key: dict[str, dict],
    ae9_by_decision_id: dict[str, dict],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    identity = decision.get("candidate_identity") or {}
    ctx = None
    for key in (identity.get("candidate_id"), identity.get("pair_address"), decision.get("decision_id")):
        if key and str(key) in ae8_index:
            ctx = ae8_index[str(key)]
            break
    aud = _match_ae9(decision, ae9_by_key, ae9_by_decision_id)
    return ctx, aud


def load_unprocessed_batch(
    *,
    project_root: Path,
    state_db: AE11StateDb,
    batch_size: int,
    max_scan_records: int,
    source_path: Path | None = None,
) -> DecisionSourceLoadResult:
    """Scan AE6 JSONL from durable cursor; return unprocessed batch."""
    result = DecisionSourceLoadResult()
    path = source_path or discover_ae6_path(project_root)
    if not path:
        result.decision_source_status = "NO_AE6_SOURCE_FILE"
        result.no_new_decisions_available = True
        return result

    result.ae6_source_file = str(path.resolve())
    stat = path.stat()
    result.ae6_file_length = stat.st_size
    result.ae6_file_last_write_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    cursor = state_db.get_source_cursor(SOURCE_NAME, str(path.resolve()))
    result.cursor_before = cursor.get("cursor_value") or "0"

    byte_offset = int(cursor.get("cursor_value") or 0)
    ae8_index, ae9_by_key, ae9_by_decision_id = load_ae8_ae9_indexes(project_root)

    for end_offset, record, _raw in _iter_jsonl_from_offset(path, byte_offset):
        result.records_scanned += 1
        if record is None:
            result.records_skipped_invalid += 1
            if result.records_scanned >= max_scan_records:
                result.cursor_after = str(end_offset)
                state_db.update_source_cursor(
                    SOURCE_NAME,
                    str(path.resolve()),
                    cursor_type="byte_offset",
                    cursor_value=str(end_offset),
                    eof_reached=False,
                )
                return result
            continue

        result.records_seen += 1
        decision_id = record.get("decision_id")
        event_key = build_source_event_key(record)
        created_at = record.get("created_at_utc")
        if created_at and (
            not result.latest_record_created_at_utc or created_at > result.latest_record_created_at_utc
        ):
            result.latest_record_created_at_utc = created_at

        if not decision_id:
            result.records_skipped_invalid += 1
        elif state_db.is_decision_processed(str(decision_id), source_event_key=event_key):
            result.records_skipped_processed += 1
        else:
            ctx, aud = _match_candidate(record, ae8_index, ae9_by_key, ae9_by_decision_id)
            result.candidates.append((record, ctx, aud))
            result.records_selected_for_processing += 1
            if len(result.candidates) >= batch_size:
                result.cursor_after = str(end_offset)
                state_db.update_source_cursor(
                    SOURCE_NAME,
                    str(path.resolve()),
                    cursor_type="byte_offset",
                    cursor_value=str(end_offset),
                    last_seen_decision_id=str(decision_id) if decision_id else None,
                    last_seen_created_at_utc=created_at,
                    eof_reached=False,
                )
                result.decision_source_status = "BATCH_READY"
                return result

        if result.records_scanned >= max_scan_records:
            result.cursor_after = str(end_offset)
            state_db.update_source_cursor(
                SOURCE_NAME,
                str(path.resolve()),
                cursor_type="byte_offset",
                cursor_value=str(end_offset),
                last_seen_decision_id=str(decision_id) if decision_id else cursor.get("last_seen_decision_id"),
                last_seen_created_at_utc=created_at or cursor.get("last_seen_created_at_utc"),
                eof_reached=False,
            )
            if not result.candidates:
                result.decision_source_status = "SCAN_LIMIT_NO_BATCH"
            return result

    result.eof_reached = True
    result.cursor_after = str(stat.st_size)
    state_db.update_source_cursor(
        SOURCE_NAME,
        str(path.resolve()),
        cursor_type="byte_offset",
        cursor_value=str(stat.st_size),
        last_seen_decision_id=cursor.get("last_seen_decision_id"),
        last_seen_created_at_utc=result.latest_record_created_at_utc or cursor.get("last_seen_created_at_utc"),
        eof_reached=True,
    )
    if not result.candidates:
        result.no_new_decisions_available = True
        result.decision_source_status = "NO_NEW_DECISIONS_AVAILABLE"
    else:
        result.decision_source_status = "BATCH_READY_EOF"
    return result


def generate_fresh_ae6_decisions(
    project_root: Path,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    """Call existing AE6 generation path when backlog EOF reached."""
    status: dict[str, Any] = {
        "fresh_generation_attempted": True,
        "fresh_generation_result": "NOT_ATTEMPTED",
        "decisions_generated": 0,
        "generated_at_utc": utc_now_iso(),
    }
    try:
        from scripts.diagnostics._common import open_db_readonly
        from app.decision.builder import build_decision_record, fetch_recent_signal_candidates
        from app.decision.persistence import DecisionJsonlWriter, decision_records_path_for_date

        conn = open_db_readonly()
        try:
            bundles = fetch_recent_signal_candidates(conn, limit=limit)
            if not bundles:
                status["fresh_generation_result"] = "NO_SIGNAL_CANDIDATES"
                return status
            writer = DecisionJsonlWriter(decision_records_path_for_date())
            for bundle in bundles:
                record = build_decision_record(conn=conn, **bundle)
                writer.append_record(record)
                status["decisions_generated"] += 1
            writer.close()
            status["fresh_generation_result"] = "SUCCESS"
            status["output_path"] = str(decision_records_path_for_date())
        finally:
            conn.close()
    except FileNotFoundError:
        status["fresh_generation_result"] = "DB_UNAVAILABLE"
    except Exception as exc:
        status["fresh_generation_result"] = f"ERROR:{exc}"
    return status


def reset_cursor_for_new_file(state_db: AE11StateDb, source_file: str) -> None:
    """Reset cursor when daily AE6 file changes."""
    cursor = state_db.get_source_cursor(SOURCE_NAME, source_file)
    if cursor.get("source_file") != source_file:
        state_db.update_source_cursor(
            SOURCE_NAME,
            source_file,
            cursor_type="byte_offset",
            cursor_value="0",
            eof_reached=False,
        )
