"""Atomic checkpointing for AE11 runtime loop — AE11C persistent/current_run split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.persistence import atomic_write_json, checkpoint_path, state_db_path
from app.runtime_paper_loop.types import utc_now_iso


def build_checkpoint_payload(
    *,
    loop_run_id: str,
    invocation_id: str,
    audit_root: str,
    last_completed_iteration: int,
    session_iterations_completed: int,
    active_position_ids: list[str],
    active_pair_keys: list[str],
    cash_balance: float,
    reserved_cash: float,
    latest_processed_decision_cursor: str | None,
    latest_processed_decision_timestamp: str | None,
    idempotency_index_status: dict[str, Any],
    output_file_paths_for_current_day: dict[str, str],
    ae6_source_file: str | None = None,
    ae6_source_cursor_type: str | None = None,
    ae6_source_cursor_value: str | None = None,
    ae6_last_seen_decision_id: str | None = None,
    ae6_last_seen_created_at_utc: str | None = None,
    ae6_eof_reached_at_utc: str | None = None,
    last_state_reconstruction_at_utc: str | None = None,
    explicit_resume_requested: bool = False,
    run_started_at_utc: str | None = None,
) -> dict[str, Any]:
    persistent_state = {
        "cash_balance": cash_balance,
        "reserved_cash": reserved_cash,
        "active_position_ids": active_position_ids,
        "active_pair_keys": active_pair_keys,
        "latest_processed_decision_cursor": latest_processed_decision_cursor,
        "latest_processed_decision_timestamp": latest_processed_decision_timestamp,
        "ae6_source_file": ae6_source_file,
        "ae6_source_cursor_type": ae6_source_cursor_type,
        "ae6_source_cursor_value": ae6_source_cursor_value,
        "ae6_last_seen_decision_id": ae6_last_seen_decision_id,
        "ae6_last_seen_created_at_utc": ae6_last_seen_created_at_utc,
        "last_state_reconstruction_at_utc": last_state_reconstruction_at_utc,
        "processed_registry_path": str(state_db_path()),
        "processed_registry_backend": "sqlite",
        "processed_registry_count": idempotency_index_status.get("processed_decisions_count", 0),
        "idempotency_index_status": idempotency_index_status,
        "output_file_paths_for_current_day": output_file_paths_for_current_day,
    }
    current_run = {
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "audit_root": audit_root,
        "run_started_at_utc": run_started_at_utc or utc_now_iso(),
        "explicit_resume_requested": explicit_resume_requested,
        "session_iterations_completed": session_iterations_completed,
        "last_completed_iteration": last_completed_iteration,
        "last_completed_at_utc": utc_now_iso(),
    }
    return {
        "schema_version": "AE11C_CHECKPOINT_V1",
        "persistent_state": persistent_state,
        "current_run": current_run,
        # backward-compatible top-level mirrors for readers expecting flat checkpoint
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "audit_root": audit_root,
        "last_completed_iteration": session_iterations_completed,
        "last_completed_at_utc": current_run["last_completed_at_utc"],
        "cash_balance": cash_balance,
        "reserved_cash": reserved_cash,
        "active_position_ids": active_position_ids,
        "active_pair_keys": active_pair_keys,
        "latest_processed_decision_cursor": latest_processed_decision_cursor,
        "latest_processed_decision_timestamp": latest_processed_decision_timestamp,
        "idempotency_index_status": idempotency_index_status,
        "output_file_paths_for_current_day": output_file_paths_for_current_day,
        "ae6_source_file": ae6_source_file,
        "ae6_source_cursor_type": ae6_source_cursor_type,
        "ae6_source_cursor_value": ae6_source_cursor_value,
        "ae6_last_seen_decision_id": ae6_last_seen_decision_id,
        "ae6_last_seen_created_at_utc": ae6_last_seen_created_at_utc,
        "ae6_eof_reached_at_utc": ae6_eof_reached_at_utc,
        "processed_registry_path": str(state_db_path()),
        "processed_registry_backend": "sqlite",
        "processed_registry_count": idempotency_index_status.get("processed_decisions_count", 0),
        "last_state_reconstruction_at_utc": last_state_reconstruction_at_utc,
    }


def write_checkpoint(payload: dict[str, Any], path: Path | None = None) -> Path:
    return atomic_write_json(path or checkpoint_path(), payload)


def load_checkpoint(path: Path | None = None) -> dict[str, Any] | None:
    target = path or checkpoint_path()
    if not target.is_file():
        return None
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)
