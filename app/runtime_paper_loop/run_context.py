"""Run identity vs persistent state — RunContextFactory for AE11C."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.types import utc_now_iso


def _timestamp_slug() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class PersistentStateSnapshot:
    """Loaded from checkpoint/SQLite — continues across invocations."""

    cash_balance: float | None = None
    reserved_cash: float | None = None
    active_position_ids: list[str] = field(default_factory=list)
    active_pair_keys: list[str] = field(default_factory=list)
    latest_processed_decision_cursor: str | None = None
    latest_processed_decision_timestamp: str | None = None
    ae6_source_file: str | None = None
    ae6_source_cursor_type: str | None = None
    ae6_source_cursor_value: str | None = None
    ae6_last_seen_decision_id: str | None = None
    ae6_last_seen_created_at_utc: str | None = None
    last_state_reconstruction_at_utc: str | None = None
    processed_registry_count: int = 0


@dataclass
class RunContext:
    """Identity for the current CLI invocation — fresh by default."""

    loop_run_id: str
    invocation_id: str
    run_started_at_utc: str
    audit_root: Path
    explicit_resume_requested: bool = False
    resume_loop_run_id: str | None = None
    checkpoint_loaded: bool = False
    checkpoint_loop_run_id_if_any: str | None = None
    checkpoint_audit_root_if_any: str | None = None
    loop_run_id_reused_from_checkpoint: bool = False
    audit_root_reused_from_checkpoint: bool = False
    persistent_state_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_run_id": self.loop_run_id,
            "invocation_id": self.invocation_id,
            "run_started_at_utc": self.run_started_at_utc,
            "audit_root": str(self.audit_root),
            "explicit_resume_requested": self.explicit_resume_requested,
            "resume_loop_run_id": self.resume_loop_run_id,
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_loop_run_id_if_any": self.checkpoint_loop_run_id_if_any,
            "checkpoint_audit_root_if_any": self.checkpoint_audit_root_if_any,
            "loop_run_id_reused_from_checkpoint": self.loop_run_id_reused_from_checkpoint,
            "audit_root_reused_from_checkpoint": self.audit_root_reused_from_checkpoint,
            "persistent_state_loaded": self.persistent_state_loaded,
            "current_run_loop_run_id": self.loop_run_id,
            "current_run_audit_root": str(self.audit_root),
        }


def extract_persistent_state(checkpoint: dict[str, Any] | None) -> PersistentStateSnapshot:
    """Extract persistent fields from checkpoint — backward compatible with AE11B."""
    if not checkpoint:
        return PersistentStateSnapshot()

    ps = checkpoint.get("persistent_state") or checkpoint
    idem = ps.get("idempotency_index_status") or checkpoint.get("idempotency_index_status") or {}
    return PersistentStateSnapshot(
        cash_balance=ps.get("cash_balance"),
        reserved_cash=ps.get("reserved_cash"),
        active_position_ids=list(ps.get("active_position_ids") or []),
        active_pair_keys=list(ps.get("active_pair_keys") or []),
        latest_processed_decision_cursor=ps.get("latest_processed_decision_cursor"),
        latest_processed_decision_timestamp=ps.get("latest_processed_decision_timestamp"),
        ae6_source_file=ps.get("ae6_source_file"),
        ae6_source_cursor_type=ps.get("ae6_source_cursor_type"),
        ae6_source_cursor_value=ps.get("ae6_source_cursor_value"),
        ae6_last_seen_decision_id=ps.get("ae6_last_seen_decision_id"),
        ae6_last_seen_created_at_utc=ps.get("ae6_last_seen_created_at_utc"),
        last_state_reconstruction_at_utc=ps.get("last_state_reconstruction_at_utc"),
        processed_registry_count=int(
            ps.get("processed_registry_count")
            or idem.get("processed_decisions_count")
            or 0
        ),
    )


def extract_checkpoint_run_metadata(checkpoint: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return (loop_run_id, audit_root) from checkpoint for historical reference."""
    if not checkpoint:
        return None, None
    current_run = checkpoint.get("current_run") or {}
    loop_run_id = current_run.get("loop_run_id") or checkpoint.get("loop_run_id")
    audit_root = current_run.get("audit_root") or checkpoint.get("audit_root")
    return loop_run_id, audit_root


class RunContextFactory:
    """Separate persistent state loading from current invocation identity."""

    @staticmethod
    def create(
        *,
        project_root: Path,
        checkpoint: dict[str, Any] | None,
        explicit_resume_requested: bool = False,
        resume_loop_run_id: str | None = None,
        resume_audit_root: Path | None = None,
    ) -> tuple[RunContext, PersistentStateSnapshot]:
        run_started_at = utc_now_iso()
        persistent = extract_persistent_state(checkpoint)
        ck_loop_id, ck_audit_root = extract_checkpoint_run_metadata(checkpoint)

        invocation_id = str(uuid4())
        fresh_loop_run_id = str(uuid4())
        slug = _timestamp_slug()
        fresh_audit_root = (
            project_root / "data" / "audits" / f"ae11_runtime_paper_loop_{slug}_{invocation_id[:8]}"
        )

        ctx = RunContext(
            loop_run_id=fresh_loop_run_id,
            invocation_id=invocation_id,
            run_started_at_utc=run_started_at,
            audit_root=fresh_audit_root,
            explicit_resume_requested=explicit_resume_requested,
            resume_loop_run_id=resume_loop_run_id,
            checkpoint_loaded=checkpoint is not None,
            checkpoint_loop_run_id_if_any=ck_loop_id,
            checkpoint_audit_root_if_any=ck_audit_root,
            persistent_state_loaded=checkpoint is not None,
        )

        if explicit_resume_requested:
            if resume_loop_run_id:
                ctx.loop_run_id = resume_loop_run_id
                ctx.loop_run_id_reused_from_checkpoint = resume_loop_run_id == ck_loop_id
            elif ck_loop_id:
                ctx.loop_run_id = ck_loop_id
                ctx.loop_run_id_reused_from_checkpoint = True

            if resume_audit_root and resume_audit_root.is_dir():
                ctx.audit_root = resume_audit_root
                ctx.audit_root_reused_from_checkpoint = str(resume_audit_root) == ck_audit_root
            elif ck_audit_root:
                candidate = Path(ck_audit_root)
                if candidate.is_dir():
                    ctx.audit_root = candidate
                    ctx.audit_root_reused_from_checkpoint = True

        ctx.audit_root.mkdir(parents=True, exist_ok=True)
        return ctx, persistent
