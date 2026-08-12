"""Typed mismatch events for state reconstruction audit — AE11C."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.types import utc_now_iso

MISMATCH_EVENT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "checkpoint_loaded",
    "explicit_resume_requested",
    "reconstruction_status",
    "mismatch_detected",
    "mismatch_type",
    "component",
    "field_path",
    "field",
    "checkpoint_value",
    "sqlite_value",
    "jsonl_reconstructed_value",
    "in_memory_before",
    "in_memory_after",
    "diff_value",
    "source_of_truth",
    "repair_action",
    "repair_applied",
    "repair_target",
    "repair_reason",
    "severity",
    "notes",
]


@dataclass
class MismatchEvent:
    audit_timestamp_utc: str = field(default_factory=utc_now_iso)
    loop_run_id: str = ""
    invocation_id: str = ""
    checkpoint_loaded: bool = False
    explicit_resume_requested: bool = False
    reconstruction_status: str = "OK"
    mismatch_detected: bool = False
    mismatch_type: str = "NO_MISMATCH"
    component: str = ""
    field_path: str = ""
    field: str = ""
    checkpoint_value: str | None = None
    sqlite_value: str | None = None
    jsonl_reconstructed_value: str | None = None
    in_memory_before: str | None = None
    in_memory_after: str | None = None
    diff_value: str | None = None
    source_of_truth: str = "sqlite"
    repair_action: str = "NO_REPAIR_NEEDED"
    repair_applied: bool = False
    repair_target: str = ""
    repair_reason: str = ""
    severity: str = "INFO"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(val: Any) -> str | None:
    if val is None:
        return None
    return str(val)


def build_numeric_mismatch(
    *,
    loop_run_id: str,
    invocation_id: str,
    checkpoint_loaded: bool,
    explicit_resume_requested: bool,
    mismatch_type: str,
    field_path: str,
    field: str,
    checkpoint_value: Any,
    sqlite_value: Any,
    jsonl_value: Any | None = None,
    in_memory_before: Any = None,
    in_memory_after: Any = None,
    source_of_truth: str = "sqlite",
    repair_action: str = "UPDATED_IN_MEMORY_FROM_SQLITE",
    repair_applied: bool = True,
    repair_target: str = "in_memory",
    severity: str = "WARNING",
    notes: str = "",
) -> MismatchEvent:
    try:
        diff = float(sqlite_value or 0) - float(checkpoint_value or 0)
        diff_str = f"{diff:+.4f}" if diff else "0"
    except (TypeError, ValueError):
        diff_str = None
    return MismatchEvent(
        loop_run_id=loop_run_id,
        invocation_id=invocation_id,
        checkpoint_loaded=checkpoint_loaded,
        explicit_resume_requested=explicit_resume_requested,
        reconstruction_status="MISMATCH_REPAIRED" if repair_applied else "MISMATCH",
        mismatch_detected=True,
        mismatch_type=mismatch_type,
        component=field_path.split(".")[0] if "." in field_path else field_path,
        field_path=field_path,
        field=field,
        checkpoint_value=_fmt(checkpoint_value),
        sqlite_value=_fmt(sqlite_value),
        jsonl_reconstructed_value=_fmt(jsonl_value),
        in_memory_before=_fmt(in_memory_before),
        in_memory_after=_fmt(in_memory_after),
        diff_value=diff_str,
        source_of_truth=source_of_truth,
        repair_action=repair_action,
        repair_applied=repair_applied,
        repair_target=repair_target,
        repair_reason=f"{field_path} reconciled from {source_of_truth}",
        severity=severity,
        notes=notes,
    )


def detect_checkpoint_mismatches(
    *,
    loop_run_id: str,
    invocation_id: str,
    checkpoint: dict[str, Any] | None,
    reconstructed: Any,
    state_db: Any,
    explicit_resume_requested: bool,
    run_context: Any | None = None,
) -> list[MismatchEvent]:
    """Compare checkpoint vs SQLite/reconstructed and emit MismatchEvent rows."""
    events: list[MismatchEvent] = []
    if not checkpoint:
        events.append(
            MismatchEvent(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=False,
                explicit_resume_requested=explicit_resume_requested,
                reconstruction_status="OK",
                mismatch_detected=False,
                mismatch_type="NO_MISMATCH",
                field_path="startup",
                field="checkpoint",
                repair_action="NO_REPAIR_NEEDED",
                notes="No checkpoint file present",
            )
        )
        return events

    ck_loaded = True
    ps = checkpoint.get("persistent_state") or checkpoint
    db_status = state_db.status_summary()

    ck_cash = float(ps.get("cash_balance", 0))
    recon_cash = float(reconstructed.cash_balance_usd)
    if abs(ck_cash - recon_cash) > 0.01:
        events.append(
            build_numeric_mismatch(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=ck_loaded,
                explicit_resume_requested=explicit_resume_requested,
                mismatch_type="CASH_BALANCE_MISMATCH",
                field_path="ledger.cash_balance",
                field="cash_balance",
                checkpoint_value=ck_cash,
                sqlite_value=recon_cash,
                jsonl_value=recon_cash,
                source_of_truth="sqlite_and_jsonl_reconstruction",
                repair_target="in_memory_ledger",
                notes="Prefer reconstructed ledger from SQLite + JSONL",
            )
        )

    ck_pos = set(ps.get("active_position_ids") or [])
    if hasattr(state_db, "get_open_positions"):
        positions = state_db.get_open_positions()
    else:
        positions = state_db.load_active_positions()
    db_pos = {p["position_id"] for p in positions}
    recon_pos = {p.get("position_id") for p in reconstructed.open_positions}
    if ck_pos != db_pos or ck_pos != recon_pos:
        events.append(
            MismatchEvent(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=ck_loaded,
                explicit_resume_requested=explicit_resume_requested,
                reconstruction_status="MISMATCH_REPAIRED",
                mismatch_detected=True,
                mismatch_type="OPEN_POSITION_COUNT_MISMATCH",
                component="positions",
                field_path="positions.open_count",
                field="open_position_count",
                checkpoint_value=_fmt(len(ck_pos)),
                sqlite_value=_fmt(len(db_pos)),
                jsonl_reconstructed_value=_fmt(len(recon_pos)),
                source_of_truth="sqlite",
                repair_action="PRESERVED_SQLITE_AS_SOURCE_OF_TRUTH",
                repair_applied=True,
                repair_target="in_memory",
                repair_reason="Active positions reconciled from SQLite",
                severity="WARNING",
            )
        )

    ck_processed = int(
        ps.get("processed_registry_count")
        or (ps.get("idempotency_index_status") or {}).get("processed_decisions_count")
        or 0
    )
    db_processed = int(db_status.get("processed_decisions_count", 0))
    if ck_processed != db_processed:
        events.append(
            build_numeric_mismatch(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=ck_loaded,
                explicit_resume_requested=explicit_resume_requested,
                mismatch_type="PROCESSED_DECISION_COUNT_MISMATCH",
                field_path="processed_decisions.count",
                field="processed_decisions_count",
                checkpoint_value=ck_processed,
                sqlite_value=db_processed,
                source_of_truth="sqlite",
                repair_action="PRESERVED_SQLITE_AS_SOURCE_OF_TRUTH",
                repair_target="checkpoint_hint_only",
                notes="Checkpoint count is hint; SQLite is authoritative",
            )
        )

    if run_context and not explicit_resume_requested:
        ck_loop = checkpoint.get("current_run", {}).get("loop_run_id") or checkpoint.get("loop_run_id")
        if ck_loop and ck_loop == run_context.loop_run_id:
            events.append(
                MismatchEvent(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    checkpoint_loaded=ck_loaded,
                    explicit_resume_requested=False,
                    reconstruction_status="MISMATCH_REPAIRED",
                    mismatch_detected=True,
                    mismatch_type="CHECKPOINT_RUN_ID_REUSED",
                    component="run_context",
                    field_path="checkpoint.loop_run_id",
                    field="loop_run_id",
                    checkpoint_value=_fmt(ck_loop),
                    sqlite_value=_fmt(run_context.loop_run_id),
                    repair_action="GENERATED_NEW_LOOP_RUN_ID",
                    repair_applied=True,
                    repair_target="run_context",
                    repair_reason="New invocation must not reuse checkpoint loop_run_id",
                    severity="INFO",
                )
            )

    ck_locks = len(ps.get("active_pair_keys") or [])
    db_locks = int(db_status.get("active_pair_locks", 0))
    if ck_locks != db_locks:
        events.append(
            build_numeric_mismatch(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=ck_loaded,
                explicit_resume_requested=explicit_resume_requested,
                mismatch_type="ACTIVE_LOCK_COUNT_MISMATCH",
                field_path="active_pair_locks.count",
                field="active_pair_locks",
                checkpoint_value=ck_locks,
                sqlite_value=db_locks,
                source_of_truth="sqlite",
                repair_action="PRESERVED_SQLITE_AS_SOURCE_OF_TRUTH",
            )
        )

    if not events:
        events.append(
            MismatchEvent(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                checkpoint_loaded=ck_loaded,
                explicit_resume_requested=explicit_resume_requested,
                reconstruction_status="OK",
                mismatch_detected=False,
                mismatch_type="NO_MISMATCH",
                field_path="reconstruction",
                repair_action="NO_REPAIR_NEEDED",
                notes="Checkpoint, SQLite, and reconstruction aligned",
            )
        )

    return events


def write_mismatch_audit(
    csv_path: Path,
    events: list[MismatchEvent],
    *,
    jsonl_path: Path | None = None,
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [e.to_dict() for e in events]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MISMATCH_EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
    return csv_path
