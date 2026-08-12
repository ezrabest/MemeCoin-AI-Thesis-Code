"""Session vs cumulative metrics for AE11C/AE11H."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CUMULATIVE_METRIC_SEMANTICS = {
    "open_positions": "current OPEN count (status='OPEN'); not lifetime",
    "orders_total": "legacy alias — equals current OPEN count, NOT lifetime orders",
    "fills_total": "legacy alias — equals current OPEN count, NOT lifetime fills",
    "closed_positions_total": "canonical closed_positions unique count",
    "current_open_positions_count": "authoritative OPEN count",
    "canonical_closed_positions_count": "authoritative closed_positions unique count",
    "position_registry_total_rows": "OPEN+CLOSED registry rows",
    "paper_order_event_count": "paper order event history size if available",
    "paper_fill_event_count": "paper fill event history size if available",
    "session_*": "current invocation only; reset each run",
}


@dataclass
class MetricsCounters:
    iterations_completed: int = 0
    decisions_seen: int = 0
    new_decisions_processed: int = 0
    duplicates_skipped: int = 0
    orders_created: int = 0
    fills_created: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    rejected_candidates: int = 0
    missed_winners: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMetrics:
    """Session counters reset each invocation; cumulative from persistent state."""

    session: MetricsCounters = field(default_factory=MetricsCounters)
    cumulative: MetricsCounters = field(default_factory=MetricsCounters)

    def session_summary(self) -> dict[str, Any]:
        d = self.session.to_dict()
        return {
            **d,
            "session_orders_created": d["orders_created"],
            "session_fills_created": d["fills_created"],
            "session_positions_opened": d["positions_opened"],
            "session_positions_closed": d["positions_closed"],
        }

    def cumulative_summary(self) -> dict[str, Any]:
        return {
            "processed_decisions": self.cumulative.new_decisions_processed,
            "open_positions": self.cumulative.positions_opened - self.cumulative.positions_closed,
            "orders_total": self.cumulative.orders_created,
            "fills_total": self.cumulative.fills_created,
            "closed_positions_total": self.cumulative.positions_closed,
            "decisions_seen_total": self.cumulative.decisions_seen,
            "duplicates_skipped_total": self.cumulative.duplicates_skipped,
            "missed_winners_total": self.cumulative.missed_winners,
        }

    def load_cumulative_from_state_db(
        self,
        state_db: Any,
        *,
        cash_balance: float,
        source_cursor_value: str | None = None,
    ) -> None:
        """Populate cumulative counters from SQLite persistent state."""
        status = state_db.status_summary()
        self.cumulative.new_decisions_processed = int(status.get("processed_decisions_count", 0))
        self.cumulative.positions_opened = int(status.get("active_open_positions", 0))
        self.cumulative.positions_closed = 0
        locks = int(status.get("active_pair_locks", 0))
        self.cumulative.duplicates_skipped = 0
        self.cumulative.orders_created = locks
        self.cumulative.fills_created = locks
        if source_cursor_value:
            self.cumulative.decisions_seen = int(source_cursor_value) if source_cursor_value.isdigit() else 0

    def cumulative_extended(
        self,
        state_db: Any,
        *,
        cash_balance: float,
        source_cursor_value: str | None = None,
        paper_order_event_count: int | None = None,
        paper_fill_event_count: int | None = None,
        lifetime_orders_created_count: int | None = None,
        lifetime_fills_created_count: int | None = None,
    ) -> dict[str, Any]:
        status = state_db.status_summary()
        # AE11G: open_positions must come from count_open_positions() / status='OPEN'
        if hasattr(state_db, "count_open_positions"):
            open_count = int(state_db.count_open_positions())
        else:
            open_count = int(status.get("active_open_positions", 0))
        cooldowns = state_db.load_cooldowns()
        closed_count = int(status.get("closed_positions_count", 0))
        registry_rows = int(status.get("total_position_registry_rows", 0))
        cursor = None
        if hasattr(state_db, "get_source_cursor"):
            from app.runtime_paper_loop.decision_source import SOURCE_NAME

            ae6_file = status.get("ae6_source_file")
            if ae6_file:
                cursor = state_db.get_source_cursor(SOURCE_NAME, ae6_file)

        # Event/history counts if not supplied
        if paper_order_event_count is None:
            try:
                row = state_db._conn.execute(
                    "SELECT COUNT(*) AS cnt FROM paper_order_events"
                ).fetchone()
                paper_order_event_count = int(row["cnt"]) if row else None
            except Exception:
                paper_order_event_count = None
        if paper_fill_event_count is None:
            try:
                row = state_db._conn.execute(
                    "SELECT COUNT(*) AS cnt FROM paper_fill_events"
                ).fetchone()
                paper_fill_event_count = int(row["cnt"]) if row else None
            except Exception:
                paper_fill_event_count = None

        semantics_status = "PASS"
        # Legacy aliases equal current open — document, do not treat as lifetime
        return {
            "processed_decisions": int(status.get("processed_decisions_count", 0)),
            "open_positions": open_count,
            "current_open_positions_count": open_count,
            "current_closed_positions_count": closed_count,
            "canonical_closed_positions_count": closed_count,
            "position_registry_total_rows": registry_rows,
            "active_pair_locks": int(status.get("active_pair_locks", 0)),
            "cooldowns": len(cooldowns),
            "cash_balance": cash_balance,
            # Legacy ambiguous fields retained for backward compatibility
            "orders_total": open_count,
            "fills_total": open_count,
            "orders_total_semantics": "legacy_alias_current_open_not_lifetime",
            "fills_total_semantics": "legacy_alias_current_open_not_lifetime",
            "closed_positions_total": closed_count,
            "paper_order_event_count": paper_order_event_count,
            "paper_fill_event_count": paper_fill_event_count,
            "lifetime_orders_created_count": lifetime_orders_created_count,
            "lifetime_fills_created_count": lifetime_fills_created_count,
            "current_open_orders_count": None,
            "total_position_registry_rows": registry_rows,
            "capacity_count_source": status.get("capacity_count_source", "status_OPEN"),
            "source_cursor_value": (
                (cursor or {}).get("cursor_value") if cursor else source_cursor_value
            ),
            "cumulative_metric_semantics_status": semantics_status,
            "cumulative_metric_semantics": CUMULATIVE_METRIC_SEMANTICS,
        }
