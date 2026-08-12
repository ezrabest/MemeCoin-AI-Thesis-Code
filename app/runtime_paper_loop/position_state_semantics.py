"""
AE11G position-state semantics audit.

Source-of-truth contract:
- active_positions is a position *registry* (may contain OPEN and CLOSED rows).
- active_positions.status is authoritative for lifecycle status (OPEN / CLOSED).
- OPEN positions := active_positions WHERE status='OPEN' (via get_open_positions /
  count_open_positions only for business logic).
- closed_positions is the canonical economic closed-trade store.
- closed_positions must not push status back into active_positions except diagnostics.
- A position_id in both tables is valid only when registry status='CLOSED'.
- active_pair_locks may only reference OPEN positions (else Ghost Lock).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.types import utc_now_iso

CAPACITY_COUNT_SOURCE = "status_OPEN"

SEMANTICS_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "total_position_registry_rows",
    "open_positions_count",
    "closed_rows_in_active_positions_count",
    "closed_positions_count",
    "active_pair_locks_count",
    "locks_pointing_to_closed_count",
    "locks_without_position_count",
    "cooldowns_count",
    "max_open_positions",
    "capacity_count_source",
    "open_snapshot_rows",
    "cumulative_metrics_open_positions",
    "ghost_lock_count",
    "ghost_lock_repair_count",
    "open_also_in_closed_positions_count",
    "status",
    "mismatch_type",
    "repair_action",
    "notes",
]


@dataclass
class PositionStateSemanticsResult:
    total_position_registry_rows: int = 0
    open_positions_count: int = 0
    closed_rows_in_active_positions_count: int = 0
    closed_positions_count: int = 0
    active_pair_locks_count: int = 0
    locks_pointing_to_closed_count: int = 0
    locks_without_position_count: int = 0
    ghost_lock_count: int = 0
    ghost_lock_repair_count: int = 0
    cooldowns_count: int = 0
    max_open_positions: int | None = None
    capacity_count_source: str = CAPACITY_COUNT_SOURCE
    open_snapshot_rows: int | None = None
    cumulative_metrics_open_positions: int | None = None
    open_also_in_closed_positions_count: int = 0
    position_state_semantics_status: str = "PASS"
    mismatch_type: str | None = None
    repair_action: str = "NO_REPAIR_NEEDED"
    notes: str = ""

    def to_meta(self) -> dict[str, Any]:
        return {
            "position_state_semantics_status": self.position_state_semantics_status,
            "total_position_registry_rows": self.total_position_registry_rows,
            "open_positions_count": self.open_positions_count,
            "closed_rows_in_active_positions_count": self.closed_rows_in_active_positions_count,
            "closed_positions_count": self.closed_positions_count,
            "active_pair_locks_count": self.active_pair_locks_count,
            "locks_pointing_to_closed_count": self.locks_pointing_to_closed_count,
            "locks_without_position_count": self.locks_without_position_count,
            "ghost_lock_count": self.ghost_lock_count,
            "ghost_lock_repair_count": self.ghost_lock_repair_count,
            "cooldowns_count": self.cooldowns_count,
            "capacity_count_source": self.capacity_count_source,
            "open_positions_snapshot_rows": self.open_snapshot_rows,
            "open_also_in_closed_positions_count": self.open_also_in_closed_positions_count,
        }


def audit_position_state_semantics(
    state_db: Any,
    *,
    loop_run_id: str,
    invocation_id: str,
    max_open_positions: int | None = None,
    open_snapshot_rows: int | None = None,
    cumulative_metrics_open_positions: int | None = None,
    repair_ghost_locks: bool = True,
    project_root: Path | None = None,
) -> PositionStateSemanticsResult:
    """
    Count OPEN via repository API; detect ghost locks; optionally repair with audit.
    """
    result = PositionStateSemanticsResult(
        total_position_registry_rows=state_db.count_position_registry_rows(),
        open_positions_count=state_db.count_open_positions(),
        closed_rows_in_active_positions_count=state_db.count_closed_rows_in_registry(),
        closed_positions_count=state_db.count_closed_positions_economic(),
        max_open_positions=max_open_positions,
        capacity_count_source=CAPACITY_COUNT_SOURCE,
        open_snapshot_rows=open_snapshot_rows,
        cumulative_metrics_open_positions=cumulative_metrics_open_positions,
        open_also_in_closed_positions_count=state_db.count_open_also_in_closed_positions(),
    )

    locks_row = state_db._conn.execute(
        "SELECT COUNT(*) AS cnt FROM active_pair_locks"
    ).fetchone()
    result.active_pair_locks_count = int(locks_row["cnt"]) if locks_row else 0
    result.cooldowns_count = len(state_db.load_cooldowns())

    closed_lock_row = state_db._conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM active_pair_locks l
        JOIN active_positions a ON a.position_id = l.position_id
        WHERE a.status != 'OPEN'
        """
    ).fetchone()
    result.locks_pointing_to_closed_count = int(closed_lock_row["cnt"]) if closed_lock_row else 0

    orphan_row = state_db._conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM active_pair_locks l
        LEFT JOIN active_positions a ON a.position_id = l.position_id
        WHERE a.position_id IS NULL
        """
    ).fetchone()
    result.locks_without_position_count = int(orphan_row["cnt"]) if orphan_row else 0
    result.ghost_lock_count = (
        result.locks_pointing_to_closed_count + result.locks_without_position_count
    )

    notes: list[str] = [
        "active_positions is a registry; OPEN = status='OPEN'; "
        "closed_positions is economic close SoT"
    ]
    if result.cooldowns_count > result.open_positions_count:
        notes.append(
            f"extra_cooldowns_allowed={result.cooldowns_count - result.open_positions_count}"
        )

    mismatches: list[str] = []
    repair_action = "NO_REPAIR_NEEDED"

    if result.capacity_count_source != CAPACITY_COUNT_SOURCE:
        mismatches.append("CAPACITY_COUNT_SOURCE_INVALID")

    if (
        cumulative_metrics_open_positions is not None
        and cumulative_metrics_open_positions != result.open_positions_count
    ):
        mismatches.append("CUMULATIVE_OPEN_COUNT_MISMATCH")

    if open_snapshot_rows is not None and open_snapshot_rows != result.open_positions_count:
        mismatches.append("OPEN_SNAPSHOT_COUNT_MISMATCH")

    if result.open_also_in_closed_positions_count > 0:
        mismatches.append("OPEN_AND_CLOSED_POSITIONS_OVERLAP")

    if result.ghost_lock_count > 0:
        if repair_ghost_locks and hasattr(state_db, "repair_ghost_locks"):
            repair = state_db.repair_ghost_locks()
            result.ghost_lock_repair_count = int(repair.get("ghost_lock_repair_count", 0))
            repair_action = "GHOST_LOCKS_DELETED"
            # Re-count after repair
            closed_lock_row = state_db._conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM active_pair_locks l
                JOIN active_positions a ON a.position_id = l.position_id
                WHERE a.status != 'OPEN'
                """
            ).fetchone()
            orphan_row = state_db._conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM active_pair_locks l
                LEFT JOIN active_positions a ON a.position_id = l.position_id
                WHERE a.position_id IS NULL
                """
            ).fetchone()
            result.locks_pointing_to_closed_count = (
                int(closed_lock_row["cnt"]) if closed_lock_row else 0
            )
            result.locks_without_position_count = int(orphan_row["cnt"]) if orphan_row else 0
            result.ghost_lock_count = (
                result.locks_pointing_to_closed_count + result.locks_without_position_count
            )
            locks_row = state_db._conn.execute(
                "SELECT COUNT(*) AS cnt FROM active_pair_locks"
            ).fetchone()
            result.active_pair_locks_count = int(locks_row["cnt"]) if locks_row else 0
            notes.append(f"ghost_locks_repaired={result.ghost_lock_repair_count}")
            if result.ghost_lock_count > 0:
                mismatches.append("GHOST_LOCKS_REMAIN_AFTER_REPAIR")
        else:
            mismatches.append("GHOST_LOCKS_PRESENT")

    if mismatches:
        result.position_state_semantics_status = "FAIL"
        result.mismatch_type = ",".join(mismatches)
    elif result.closed_rows_in_active_positions_count > 0:
        result.position_state_semantics_status = "PASS_WITH_REGISTRY_SEMANTICS"
        result.mismatch_type = None
    else:
        result.position_state_semantics_status = "PASS"
        result.mismatch_type = None

    result.repair_action = repair_action
    result.notes = "; ".join(notes)

    root = project_root
    if root is not None:
        write_position_state_semantics_audit(
            root / "audits" / "ae11_position_state_semantics_audit.csv",
            loop_run_id=loop_run_id,
            invocation_id=invocation_id,
            result=result,
        )

    return result


def write_position_state_semantics_audit(
    path: Path,
    *,
    loop_run_id: str,
    invocation_id: str,
    result: PositionStateSemanticsResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audit_timestamp_utc": utc_now_iso(),
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "total_position_registry_rows": result.total_position_registry_rows,
        "open_positions_count": result.open_positions_count,
        "closed_rows_in_active_positions_count": result.closed_rows_in_active_positions_count,
        "closed_positions_count": result.closed_positions_count,
        "active_pair_locks_count": result.active_pair_locks_count,
        "locks_pointing_to_closed_count": result.locks_pointing_to_closed_count,
        "locks_without_position_count": result.locks_without_position_count,
        "cooldowns_count": result.cooldowns_count,
        "max_open_positions": result.max_open_positions,
        "capacity_count_source": result.capacity_count_source,
        "open_snapshot_rows": result.open_snapshot_rows,
        "cumulative_metrics_open_positions": result.cumulative_metrics_open_positions,
        "ghost_lock_count": result.ghost_lock_count,
        "ghost_lock_repair_count": result.ghost_lock_repair_count,
        "open_also_in_closed_positions_count": result.open_also_in_closed_positions_count,
        "status": result.position_state_semantics_status,
        "mismatch_type": result.mismatch_type,
        "repair_action": result.repair_action,
        "notes": result.notes,
    }
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEMANTICS_AUDIT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
    return path
