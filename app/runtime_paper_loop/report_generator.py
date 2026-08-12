"""Authoritative AE11 report state — SQLite-first open position snapshots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.decision.persistence import read_jsonl_records_safe
from app.paper_trading.persistence import (
    paper_orders_path_for_date,
    paper_positions_path_for_date,
    paper_trades_path_for_date,
)
from app.paper_trading.types import PaperPosition
from app.runtime_paper_loop.types import utc_now_iso

OPEN_POSITION_SNAPSHOT_FIELDS = [
    "loop_run_id",
    "invocation_id",
    "snapshot_generated_at_utc",
    "position_id",
    "paper_order_id",
    "source_decision_id",
    "candidate_id",
    "pair_address",
    "chain",
    "symbol",
    "opened_at_utc",
    "entry_price",
    "current_price",
    "last_price",
    "last_price_timestamp_utc",
    "price_timestamp_utc",
    "price_age_seconds",
    "notional_usd",
    "quantity",
    "cost_basis_usd",
    "cash_debited_usd",
    "entry_fee_usd",
    "entry_slippage_usd",
    "open_market_value_usd",
    "valuation_source",
    "status",
    "unrealized_pnl_usd",
    "price_unrealized_pnl_usd",
    "total_unrealized_after_cost_pnl_usd",
    "open_entry_cost_drag_usd",
    "unrealized_return_pct",
    "tp_price",
    "sl_price",
    "time_stop_at_utc",
    "cooldown_until_utc",
    "active_pair_lock",
    "trade_authority",
    "not_model_approved",
    "not_live_approved",
    "override_type",
    "economic_enrichment_status",
    "economic_enrichment_missing_fields",
    "enrichment_available",
    "enrichment_missing_fields",
]

CLOSED_TRADE_SNAPSHOT_FIELDS = [
    "close_event_id",
    "position_id",
    "economic_close_key",
    "paper_order_id",
    "source_decision_id",
    "pair_address",
    "opened_at_utc",
    "closed_at_utc",
    "exit_reason",
    "entry_price",
    "exit_price",
    "quantity",
    "notional_usd",
    "cost_basis_usd",
    "entry_fee_usd",
    "exit_fee_usd",
    "total_fees_usd",
    "gross_pnl_usd",
    "net_pnl_usd",
    "net_return_pct",
    "cash_debited_usd",
    "cash_credited_usd",
    "wallet_configured",
    "real_transaction_attempted",
    "event_quality",
]

REPORT_CONSISTENCY_FIELDS = [
    "audit_timestamp_utc",
    "reconciliation_timestamp_utc",
    "reconciliation_started_at_utc",
    "reconciliation_completed_at_utc",
    "reconciliation_duration_ms",
    "loop_run_id",
    "invocation_id",
    "report_name",
    "sqlite_count",
    "jsonl_count",
    "csv_row_count",
    "summary_count",
    "source_of_truth",
    "status",
    "mismatch_type",
    "repair_action",
    "notes",
    "equity_bridge_summary",
    "open_snapshot_economic_fields",
    "cumulative_metric_semantics",
    "open_position_economic_completeness",
]

ENRICHMENT_OPTIONAL_FIELDS = [
    "candidate_id",
    "chain",
    "symbol",
    "entry_price",
    "current_price",
    "price_timestamp_utc",
    "price_age_seconds",
    "notional_usd",
    "quantity",
    "unrealized_pnl_usd",
    "unrealized_return_pct",
    "tp_price",
    "sl_price",
    "time_stop_at_utc",
    "trade_authority",
    "not_model_approved",
    "not_live_approved",
    "override_type",
]


@dataclass
class AuthoritativeState:
    """Fetched directly from SQLite (+ optional JSONL enrichment)."""

    active_positions: list[dict[str, Any]] = field(default_factory=list)
    active_pair_locks: dict[str, str] = field(default_factory=dict)
    cooldowns: dict[str, str] = field(default_factory=dict)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    closed_trade_hygiene: dict[str, Any] = field(default_factory=dict)
    processed_decisions_count: int = 0
    cash_balance: float | None = None
    source_of_truth_open_positions: str = "sqlite_status_OPEN"
    source_of_truth_closed_trades: str = "sqlite_closed_positions"

    @property
    def open_position_count(self) -> int:
        return len(self.active_positions)

    @property
    def closed_trade_count(self) -> int:
        return len(self.closed_trades)


@dataclass
class ReportConsistencyRow:
    audit_timestamp_utc: str
    loop_run_id: str
    invocation_id: str
    report_name: str
    sqlite_count: int
    jsonl_count: int | None
    csv_row_count: int
    summary_count: int | None
    source_of_truth: str
    status: str
    mismatch_type: str | None = None
    repair_action: str = "NO_REPAIR_NEEDED"
    notes: str = ""
    reconciliation_timestamp_utc: str | None = None
    reconciliation_started_at_utc: str | None = None
    reconciliation_completed_at_utc: str | None = None
    reconciliation_duration_ms: float | None = None
    equity_bridge_summary: str | None = None
    open_snapshot_economic_fields: str | None = None
    cumulative_metric_semantics: str | None = None
    open_position_economic_completeness: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("reconciliation_timestamp_utc") is None:
            d["reconciliation_timestamp_utc"] = d.get("audit_timestamp_utc")
        return d


def _read_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records, _ = read_jsonl_records_safe(path)
    return [r for r in records if isinstance(r, dict)]


def _index_enrichment(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index enrichment by position_id, paper_order_id, source_decision_id, pair_address."""
    by_key: dict[str, dict[str, Any]] = {}
    for rec in records:
        for key_name in ("position_id", "paper_order_id", "source_decision_id", "pair_address"):
            val = rec.get(key_name)
            if val:
                by_key[f"{key_name}:{val}"] = rec
    return by_key


def _lookup_enrichment(
    pos: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key_name in ("position_id", "paper_order_id", "source_decision_id", "pair_address"):
        val = pos.get(key_name)
        if val and f"{key_name}:{val}" in index:
            return index[f"{key_name}:{val}"]
    return None


def build_open_position_snapshot_rows(
    *,
    sqlite_positions: list[dict[str, Any]],
    cooldowns: dict[str, str],
    active_pair_locks: dict[str, str],
    enrichment_records: list[dict[str, Any]],
    loop_run_id: str,
    invocation_id: str,
    snapshot_generated_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    """SQLite active_positions as base; left-join JSONL enrichment. Never drop SQLite rows."""
    from app.runtime_paper_loop.decimal_money import (
        decimal_to_str,
        quantize_usd,
        to_decimal,
    )

    generated_at = snapshot_generated_at_utc or utc_now_iso()
    enrichment_index = _index_enrichment(enrichment_records)
    rows: list[dict[str, Any]] = []

    for pos in sqlite_positions:
        pair = pos.get("pair_address") or ""
        enrich = _lookup_enrichment(pos, enrichment_index) or {}
        # Prefer durable SQLite economics (AE11E), then JSONL enrichment
        entry_price = (
            pos.get("entry_price")
            or enrich.get("entry_price_usd")
            or enrich.get("entry_price")
            or enrich.get("filled_price_usd")
        )
        quantity = pos.get("quantity") or enrich.get("quantity")
        notional = pos.get("notional_usd") or enrich.get("notional_usd")
        current_price = (
            pos.get("last_price")
            or enrich.get("current_price")
            or enrich.get("mark_price_usd")
        )
        last_price = pos.get("last_price") or current_price
        last_price_ts = (
            pos.get("last_price_timestamp_utc")
            or pos.get("entry_price_timestamp_utc")
            or enrich.get("price_timestamp_utc")
            or enrich.get("price_timestamp")
            or enrich.get("price_timestamp_used")
        )
        cost_basis = pos.get("cost_basis_usd")
        entry_fee = pos.get("entry_fee_usd")
        entry_slip = pos.get("entry_slippage_usd")
        cash_debited = pos.get("cash_debited_usd")
        open_mv_raw = pos.get("open_market_value_usd")
        valuation_source = "sqlite_open_market_value"
        if open_mv_raw is None or str(open_mv_raw).strip() == "":
            if cost_basis not in (None, ""):
                open_mv_raw = cost_basis
                valuation_source = "cost_basis_fallback"
            elif notional not in (None, ""):
                open_mv_raw = notional
                valuation_source = "notional_fallback"
            else:
                valuation_source = "unavailable"

        price_upnl = None
        after_cost = None
        cost_drag = None
        if open_mv_raw not in (None, "") and notional not in (None, ""):
            price_upnl = decimal_to_str(
                quantize_usd(to_decimal(open_mv_raw) - to_decimal(notional))
            )
        if open_mv_raw not in (None, "") and cost_basis not in (None, ""):
            after_cost = decimal_to_str(
                quantize_usd(to_decimal(open_mv_raw) - to_decimal(cost_basis))
            )
        if notional not in (None, "") and cost_basis not in (None, ""):
            cost_drag = decimal_to_str(
                quantize_usd(to_decimal(notional) - to_decimal(cost_basis))
            )

        # Legacy unrealized_pnl_usd = price-only when computable
        legacy_upnl = pos.get("unrealized_pnl_usd") or enrich.get("unrealized_pnl_usd")
        if price_upnl is not None:
            legacy_upnl = price_upnl

        econ_status = pos.get("economic_enrichment_status") or (
            "FULL" if entry_price and quantity and notional and cash_debited else "MISSING"
        )
        econ_missing = pos.get("economic_enrichment_missing_fields") or ""
        missing: list[str] = []

        row = {
            "loop_run_id": loop_run_id,
            "invocation_id": invocation_id,
            "snapshot_generated_at_utc": generated_at,
            "position_id": pos.get("position_id"),
            "paper_order_id": pos.get("paper_order_id") or enrich.get("paper_order_id"),
            "source_decision_id": pos.get("source_decision_id") or enrich.get("source_decision_id"),
            "candidate_id": pos.get("candidate_id") or enrich.get("candidate_id"),
            "pair_address": pair,
            "chain": pos.get("chain") or enrich.get("chain"),
            "symbol": pos.get("symbol") or enrich.get("symbol"),
            "opened_at_utc": pos.get("opened_at_utc") or enrich.get("opened_at_utc"),
            "entry_price": entry_price,
            "current_price": current_price,
            "last_price": last_price,
            "last_price_timestamp_utc": last_price_ts,
            "price_timestamp_utc": last_price_ts,
            "price_age_seconds": enrich.get("price_age_seconds"),
            "notional_usd": notional,
            "quantity": quantity,
            "cost_basis_usd": cost_basis,
            "cash_debited_usd": cash_debited,
            "entry_fee_usd": entry_fee,
            "entry_slippage_usd": entry_slip,
            "open_market_value_usd": open_mv_raw,
            "valuation_source": valuation_source,
            "status": pos.get("status") or "OPEN",
            "unrealized_pnl_usd": legacy_upnl,
            "price_unrealized_pnl_usd": price_upnl,
            "total_unrealized_after_cost_pnl_usd": after_cost,
            "open_entry_cost_drag_usd": cost_drag,
            "unrealized_return_pct": pos.get("unrealized_return_pct")
            or enrich.get("unrealized_return_pct"),
            "tp_price": pos.get("tp_price") or enrich.get("tp_price"),
            "sl_price": pos.get("sl_price") or enrich.get("sl_price"),
            "time_stop_at_utc": pos.get("time_stop_at_utc") or enrich.get("time_stop_at_utc"),
            "cooldown_until_utc": cooldowns.get(pair),
            "active_pair_lock": pair in active_pair_locks or bool(active_pair_locks.get(pair)),
            "trade_authority": pos.get("trade_authority") or enrich.get("trade_authority"),
            "not_model_approved": pos.get("not_model_approved")
            if pos.get("not_model_approved") is not None
            else enrich.get("not_model_approved", True if enrich else None),
            "not_live_approved": pos.get("not_live_approved")
            if pos.get("not_live_approved") is not None
            else enrich.get("not_live_approved", True if enrich else None),
            "override_type": pos.get("override_type") or enrich.get("override_type"),
            "economic_enrichment_status": econ_status,
            "economic_enrichment_missing_fields": econ_missing,
            "enrichment_available": bool(enrich),
            "enrichment_missing_fields": econ_missing,
        }
        for field_name in ENRICHMENT_OPTIONAL_FIELDS:
            if row.get(field_name) in (None, ""):
                missing.append(field_name)
        if not row.get("enrichment_missing_fields"):
            row["enrichment_missing_fields"] = ",".join(missing) if missing else ""
        if not row.get("economic_enrichment_missing_fields") and missing:
            row["economic_enrichment_missing_fields"] = ",".join(missing)
        rows.append(row)

    return rows


def write_csv_with_headers(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> Path:
    """Always write headers; never leave a zero-byte report CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        f.flush()
    return path


def sqlite_positions_to_paper_positions(sqlite_rows: list[dict[str, Any]], enrichment: list[dict[str, Any]]) -> list[PaperPosition]:
    """Rebuild in-memory PaperPosition list from SQLite (+ optional enrichment)."""
    index = _index_enrichment(enrichment)
    result: list[PaperPosition] = []
    for pos in sqlite_rows:
        enrich = _lookup_enrichment(pos, index) or {}
        entry = (
            pos.get("entry_price")
            or enrich.get("entry_price_usd")
            or enrich.get("entry_price")
            or enrich.get("filled_price_usd")
            or 0.0
        )
        result.append(
            PaperPosition(
                position_id=str(pos.get("position_id") or ""),
                paper_order_id=str(pos.get("paper_order_id") or enrich.get("paper_order_id") or ""),
                candidate_id=str(pos.get("candidate_id") or enrich.get("candidate_id") or ""),
                symbol=str(pos.get("symbol") or enrich.get("symbol") or ""),
                pair_address=str(pos.get("pair_address") or ""),
                entry_price_usd=float(entry or 0.0),
                quantity=float(pos.get("quantity") or enrich.get("quantity") or 0.0),
                notional_usd=float(pos.get("notional_usd") or enrich.get("notional_usd") or 0.0),
                status=str(pos.get("status") or "OPEN"),
                opened_at_utc=str(pos.get("opened_at_utc") or utc_now_iso()),
                source_decision_id=pos.get("source_decision_id") or enrich.get("source_decision_id"),
            )
        )
    return result


class ReportGenerator:
    """Fetches authoritative state from SQLite; does not trust checkpoint/in-memory lists."""

    def __init__(
        self,
        *,
        state_db: Any,
        project_root: Path,
        loop_run_id: str,
        invocation_id: str,
        cash_balance: float | None = None,
    ) -> None:
        self.state_db = state_db
        self.project_root = project_root
        self.loop_run_id = loop_run_id
        self.invocation_id = invocation_id
        self.cash_balance = cash_balance
        self._last_canonical = None

    def fetch_authoritative_state(self) -> AuthoritativeState:
        """Query SQLite directly — ignore RunContext/checkpoint/in-memory position lists."""
        from app.runtime_paper_loop.closed_trade_canonical import get_canonical_closed_trades

        sqlite_positions = self.state_db.get_open_positions() if hasattr(
            self.state_db, "get_open_positions"
        ) else self.state_db.load_active_positions()
        cooldowns = self.state_db.load_cooldowns()
        locks_rows = self.state_db._conn.execute(
            "SELECT pair_address, position_id FROM active_pair_locks"
        ).fetchall()
        active_pair_locks = {r["pair_address"]: r["position_id"] for r in locks_rows}

        status = self.state_db.status_summary()
        canonical = get_canonical_closed_trades(
            self.state_db,
            project_root=self.project_root,
            loop_run_id=self.loop_run_id,
            invocation_id=self.invocation_id,
        )
        self._last_canonical = canonical

        return AuthoritativeState(
            active_positions=sqlite_positions,
            active_pair_locks=active_pair_locks,
            cooldowns=cooldowns,
            closed_trades=list(canonical.canonical_rows),
            closed_trade_hygiene=canonical.to_meta(),
            processed_decisions_count=int(status.get("processed_decisions_count", 0)),
            cash_balance=self.cash_balance,
            source_of_truth_open_positions="sqlite_status_OPEN",
            source_of_truth_closed_trades="sqlite_closed_positions",
        )

    def _load_enrichment_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        paper_dir = self.project_root / "data" / "paper_trading"
        if not paper_dir.is_dir():
            return records
        for path in sorted(paper_dir.glob("paper_orders_*.jsonl")):
            records.extend(_read_jsonl_safe(path))
        for path in sorted(paper_dir.glob("paper_positions_*.jsonl")):
            records.extend(_read_jsonl_safe(path))
        return records

    def build_open_positions_snapshot(self, state: AuthoritativeState | None = None) -> list[dict[str, Any]]:
        auth = state or self.fetch_authoritative_state()
        enrichment = self._load_enrichment_records()
        return build_open_position_snapshot_rows(
            sqlite_positions=auth.active_positions,
            cooldowns=auth.cooldowns,
            active_pair_locks=auth.active_pair_locks,
            enrichment_records=enrichment,
            loop_run_id=self.loop_run_id,
            invocation_id=self.invocation_id,
        )

    def build_closed_trades_snapshot(self, state: AuthoritativeState | None = None) -> list[dict[str, Any]]:
        auth = state or self.fetch_authoritative_state()
        return list(auth.closed_trades)

    def write_position_snapshots(
        self,
        *,
        open_path: Path,
        closed_path: Path,
        state: AuthoritativeState | None = None,
        event_history_path: Path | None = None,
        hygiene_audit_path: Path | None = None,
    ) -> dict[str, Any]:
        from app.runtime_paper_loop.closed_trade_canonical import (
            get_canonical_closed_trades,
            write_event_history,
            write_hygiene_audit,
        )

        auth = state or self.fetch_authoritative_state()
        open_rows = self.build_open_positions_snapshot(auth)
        closed_rows = self.build_closed_trades_snapshot(auth)
        write_csv_with_headers(open_path, open_rows, OPEN_POSITION_SNAPSHOT_FIELDS)
        write_csv_with_headers(closed_path, closed_rows, CLOSED_TRADE_SNAPSHOT_FIELDS)

        canonical = getattr(self, "_last_canonical", None)
        if canonical is None:
            canonical = get_canonical_closed_trades(
                self.state_db,
                project_root=self.project_root,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
            )
            self._last_canonical = canonical

        hist_path = event_history_path or (
            closed_path.parent / "ae11_closed_trade_events_history.csv"
        )
        hyg_path = hygiene_audit_path or (
            self.project_root / "audits" / "ae11_closed_trade_hygiene_audit.csv"
        )
        write_event_history(hist_path, canonical.event_history_rows)
        write_hygiene_audit(hyg_path, canonical.hygiene_rows)

        return {
            "open_positions_snapshot_rows": len(open_rows),
            "closed_trades_snapshot_rows": len(closed_rows),
            "open_path": str(open_path),
            "closed_path": str(closed_path),
            "event_history_path": str(hist_path),
            "hygiene_audit_path": str(hyg_path),
            "authoritative_open_count": auth.open_position_count,
            "authoritative_closed_count": auth.closed_trade_count,
            **canonical.to_meta(),
        }

    def build_consistency_audit(
        self,
        *,
        state: AuthoritativeState,
        open_csv_rows: int,
        closed_csv_rows: int,
        summary_open: int | None = None,
        summary_closed: int | None = None,
        hygiene: dict[str, Any] | None = None,
        open_snapshot_rows: list[dict[str, Any]] | None = None,
        equity_bridge: dict[str, Any] | None = None,
        cumulative_metrics: dict[str, Any] | None = None,
    ) -> tuple[list[ReportConsistencyRow], str, int]:
        """Return (rows, overall_status, mismatch_count)."""
        import time

        from app.runtime_paper_loop.decimal_money import (
            decimal_almost_equal,
            to_decimal,
            quantize_usd,
        )

        started = utc_now_iso()
        t0 = time.perf_counter()
        now = started
        rows: list[ReportConsistencyRow] = []
        mismatches = 0
        hyg = hygiene or state.closed_trade_hygiene or {}
        bridge = equity_bridge or {}
        cum = cumulative_metrics or {}
        snap_rows = open_snapshot_rows or []

        def _row(**kwargs: Any) -> ReportConsistencyRow:
            completed = utc_now_iso()
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return ReportConsistencyRow(
                reconciliation_timestamp_utc=completed,
                reconciliation_started_at_utc=started,
                reconciliation_completed_at_utc=completed,
                reconciliation_duration_ms=duration_ms,
                equity_bridge_summary=str(bridge.get("bridge_status") or ""),
                open_snapshot_economic_fields=(
                    "notional,cost_basis,cash_debited,entry_fee,entry_slippage,"
                    "open_market_value,price_unrealized,after_cost,cost_drag"
                ),
                cumulative_metric_semantics=str(
                    cum.get("cumulative_metric_semantics_status") or ""
                ),
                open_position_economic_completeness=str(
                    bridge.get("open_position_economic_completeness_status") or ""
                ),
                **kwargs,
            )

        open_sqlite = state.open_position_count
        open_ok = open_sqlite == open_csv_rows and (
            summary_open is None or summary_open == open_csv_rows
        )
        if not open_ok:
            mismatches += 1
        rows.append(
            _row(
                audit_timestamp_utc=now,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
                report_name="open_positions_snapshot",
                sqlite_count=open_sqlite,
                jsonl_count=None,
                csv_row_count=open_csv_rows,
                summary_count=summary_open,
                source_of_truth="sqlite",
                status="PASS" if open_ok else "FAIL",
                mismatch_type=None if open_ok else "OPEN_SNAPSHOT_COUNT_MISMATCH",
                repair_action="NO_REPAIR_NEEDED" if open_ok else "WARNING_ONLY",
                notes="SQLite active_positions WHERE status='OPEN' via get_open_positions()",
            )
        )

        closed_ok = closed_csv_rows == state.closed_trade_count and (
            summary_closed is None or summary_closed == closed_csv_rows
        )
        if not closed_ok:
            mismatches += 1
        rows.append(
            _row(
                audit_timestamp_utc=now,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
                report_name="closed_trades_snapshot",
                sqlite_count=state.closed_trade_count,
                jsonl_count=hyg.get("closed_trade_event_history_rows"),
                csv_row_count=closed_csv_rows,
                summary_count=summary_closed,
                source_of_truth="sqlite_closed_positions",
                status="PASS" if closed_ok else "FAIL",
                mismatch_type=None if closed_ok else "CLOSED_SNAPSHOT_COUNT_MISMATCH",
                notes="Canonical closed snapshot: one row per position_id from SQLite",
            )
        )

        # Uniqueness / blank close_event_id checks
        pids = [r.get("position_id") for r in state.closed_trades]
        unique_ok = len(pids) == len(set(pids))
        blank_events = sum(
            1 for r in state.closed_trades if not str(r.get("close_event_id") or "").strip()
        )
        eids = [str(r.get("close_event_id") or "") for r in state.closed_trades]
        eid_unique = len(eids) == len(set(eids))
        hygiene_ok = unique_ok and blank_events == 0 and eid_unique
        if not hygiene_ok:
            mismatches += 1
        rows.append(
            _row(
                audit_timestamp_utc=now,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
                report_name="closed_trade_hygiene",
                sqlite_count=state.closed_trade_count,
                jsonl_count=hyg.get("invalid_closed_trade_rows"),
                csv_row_count=closed_csv_rows,
                summary_count=hyg.get("canonical_closed_position_unique_count"),
                source_of_truth="sqlite_closed_positions",
                status="PASS" if hygiene_ok else "FAIL",
                mismatch_type=None
                if hygiene_ok
                else (
                    "BLANK_CLOSE_EVENT_ID"
                    if blank_events
                    else "DUPLICATE_CLOSED_POSITION_ID"
                    if not unique_ok
                    else "DUPLICATE_CLOSE_EVENT_ID"
                ),
                notes=(
                    f"unique_position_id={unique_ok}; blank_close_event_id={blank_events}; "
                    f"hygiene_status={hyg.get('closed_trade_hygiene_status')}; "
                    f"invalid_excluded={hyg.get('invalid_closed_trade_rows', 0)}"
                ),
            )
        )

        locks = len(state.active_pair_locks)
        locks_ok = locks == open_sqlite or (locks == 0 and open_sqlite == 0)
        if not locks_ok:
            mismatches += 1
        rows.append(
            _row(
                audit_timestamp_utc=now,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
                report_name="active_pair_locks",
                sqlite_count=locks,
                jsonl_count=None,
                csv_row_count=open_csv_rows,
                summary_count=open_sqlite,
                source_of_truth="sqlite",
                status="PASS" if locks_ok else "WARNING",
                mismatch_type=None if locks_ok else "ACTIVE_LOCK_COUNT_MISMATCH",
                notes="Locks should typically match open positions 1:1",
            )
        )

        cooldown_count = len(state.cooldowns)
        cooldown_ok = open_sqlite <= cooldown_count or open_sqlite == 0
        if not cooldown_ok:
            mismatches += 1
        rows.append(
            _row(
                audit_timestamp_utc=now,
                loop_run_id=self.loop_run_id,
                invocation_id=self.invocation_id,
                report_name="cooldowns",
                sqlite_count=cooldown_count,
                jsonl_count=None,
                csv_row_count=open_csv_rows,
                summary_count=open_sqlite,
                source_of_truth="sqlite",
                status="PASS" if cooldown_ok else "WARNING",
                mismatch_type=None if cooldown_ok else "COOLDOWN_COUNT_MISMATCH",
                notes="Open positions should have cooldowns; extra cooldowns from closed pairs are OK",
            )
        )

        # AE11H equity bridge / open snapshot economic sum checks
        def _sum_field(field: str) -> Any:
            total = to_decimal(0)
            for r in snap_rows:
                val = r.get(field)
                if val not in (None, ""):
                    total += to_decimal(val)
            return quantize_usd(total)

        if snap_rows and bridge:
            sum_checks = [
                ("open_notional", "notional_usd", "open_notional_usd"),
                ("open_cost_basis", "cost_basis_usd", "open_cost_basis_usd"),
                ("open_entry_fee", "entry_fee_usd", "open_entry_fee_usd"),
                ("open_entry_slippage", "entry_slippage_usd", "open_entry_slippage_usd"),
                ("open_cash_debited", "cash_debited_usd", "open_cash_debited_usd"),
                ("open_market_value", "open_market_value_usd", "open_market_value_usd"),
            ]
            for name, snap_f, bridge_f in sum_checks:
                snap_sum = _sum_field(snap_f)
                bridge_val = to_decimal(bridge.get(bridge_f) or 0)
                ok = decimal_almost_equal(snap_sum, bridge_val)
                if not ok:
                    mismatches += 1
                rows.append(
                    _row(
                        audit_timestamp_utc=now,
                        loop_run_id=self.loop_run_id,
                        invocation_id=self.invocation_id,
                        report_name=f"equity_bridge_{name}",
                        sqlite_count=open_sqlite,
                        jsonl_count=None,
                        csv_row_count=open_csv_rows,
                        summary_count=None,
                        source_of_truth="sqlite_open_plus_equity_bridge",
                        status="PASS" if ok else "FAIL",
                        mismatch_type=None if ok else f"{name.upper()}_SUM_MISMATCH",
                        notes=f"snapshot_sum={snap_sum}; bridge={bridge_val}",
                    )
                )

            bridge_status = str(
                bridge.get("bridge_status")
                or bridge.get("equity_bridge_status")
                or "UNKNOWN"
            )
            missing_econ = int(bridge.get("missing_open_economics_count") or 0)
            bridge_ok = bridge_status == "PASS" and missing_econ == 0
            if not bridge_ok:
                mismatches += 1
            rows.append(
                _row(
                    audit_timestamp_utc=now,
                    loop_run_id=self.loop_run_id,
                    invocation_id=self.invocation_id,
                    report_name="equity_bridge_summary",
                    sqlite_count=open_sqlite,
                    jsonl_count=None,
                    csv_row_count=open_csv_rows,
                    summary_count=missing_econ,
                    source_of_truth="equity_bridge",
                    status="PASS" if bridge_ok else "FAIL",
                    mismatch_type=None if bridge_ok else "EQUITY_BRIDGE_NOT_PASS",
                    notes=(
                        f"bridge_status={bridge_status}; "
                        f"missing_open_economics={missing_econ}; "
                        f"completeness={bridge.get('open_position_economic_completeness_status')}"
                    ),
                )
            )

        # Cumulative metric semantics
        current_open = cum.get("current_open_positions_count")
        if current_open is not None:
            sem_ok = int(current_open) == open_sqlite
            if not sem_ok:
                mismatches += 1
            rows.append(
                _row(
                    audit_timestamp_utc=now,
                    loop_run_id=self.loop_run_id,
                    invocation_id=self.invocation_id,
                    report_name="cumulative_metric_semantics",
                    sqlite_count=open_sqlite,
                    jsonl_count=None,
                    csv_row_count=open_csv_rows,
                    summary_count=int(current_open),
                    source_of_truth="sqlite_status_OPEN",
                    status="PASS" if sem_ok else "FAIL",
                    mismatch_type=None if sem_ok else "CURRENT_OPEN_COUNT_MISMATCH",
                    notes=(
                        "current_open_positions_count must equal OPEN count; "
                        "orders_total/fills_total are ambiguous legacy aliases"
                    ),
                )
            )

        hard_fails = sum(1 for r in rows if r.status == "FAIL")
        overall = "PASS" if hard_fails == 0 else "FAIL"
        if hard_fails == 0 and any(r.status == "WARNING" for r in rows):
            overall = "WARNING"
        return rows, overall, mismatches

    def write_consistency_audit(self, path: Path, rows: list[ReportConsistencyRow]) -> Path:
        write_csv_with_headers(path, [r.to_dict() for r in rows], REPORT_CONSISTENCY_FIELDS)
        return path
