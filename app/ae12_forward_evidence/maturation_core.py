"""Horizon maturity and no-lookahead forward return computation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.ae12_forward_evidence.idempotency import make_horizon_row_id
from app.ae12_forward_evidence.loaders import MarketSnapshotStore
from app.ae12_forward_evidence.types import HORIZON_SECONDS, HorizonOutcome, parse_ts, utc_now_iso


def compute_horizon_outcomes(
    *,
    evidence_row_id: str,
    pair_address: str | None,
    first_seen_timestamp: str | None,
    entry_price: float | None,
    horizons: list[str],
    store: MarketSnapshotStore,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any], list[HorizonOutcome], list[dict[str, Any]]]:
    """Return (evidence_horizon_fields, matured_outcome_rows, no_lookahead_audit_rows)."""

    fields: dict[str, Any] = {}
    outcomes: list[HorizonOutcome] = []
    audits: list[dict[str, Any]] = []
    first_dt = parse_ts(first_seen_timestamp)
    now_dt = now_utc or datetime.now(timezone.utc)
    computed_at = utc_now_iso()

    try:
        entry = float(entry_price) if entry_price is not None else None
    except (TypeError, ValueError):
        entry = None
    if entry is not None and entry <= 0:
        entry = None

    latest_pair = store.latest_for_pair(pair_address) if store.available else None

    # Prefetch widest window once
    max_seconds = max((HORIZON_SECONDS[h] for h in horizons if h in HORIZON_SECONDS), default=0)
    window_snaps: list[tuple[datetime, float]] = []
    if first_dt is not None and store.available and pair_address and max_seconds > 0:
        widest_end = first_dt + timedelta(seconds=max_seconds)
        # Only fetch if at least something might be available
        fetch_end = widest_end
        if latest_pair is not None and latest_pair < fetch_end:
            fetch_end = latest_pair
        if fetch_end > first_dt:
            window_snaps = store.snapshots_in_window(pair_address, first_dt, fetch_end)

    for h in horizons:
        seconds = HORIZON_SECONDS.get(h)
        h_id = make_horizon_row_id(evidence_row_id=evidence_row_id, horizon=h)
        prefix = f"horizon_{h}"

        if seconds is None or first_dt is None:
            outcome = HorizonOutcome(
                horizon=h,
                horizon_row_id=h_id,
                evidence_row_id=evidence_row_id,
                matured=False,
                no_lookahead_status="NOT_MATURED",
            )
            _apply_not_matured(fields, prefix)
            outcomes.append(outcome)
            audits.append(_audit_row(evidence_row_id, h, outcome, "MISSING_FIRST_SEEN_OR_HORIZON"))
            continue

        deadline = first_dt + timedelta(seconds=seconds)
        deadline_iso = deadline.isoformat()

        # Maturity: need local snapshot at or after deadline for this pair
        matured = False
        if store.available and latest_pair is not None and latest_pair >= deadline:
            matured = True
        elif not store.available:
            # Without snapshots we cannot claim maturity by wall clock alone for outcomes,
            # but wall-clock can still mark NOT_MATURED vs MATURED_BUT_NO_LOCAL_SNAPSHOTS.
            if now_dt >= deadline:
                matured = True
            else:
                matured = False

        if not matured:
            outcome = HorizonOutcome(
                horizon=h,
                horizon_row_id=h_id,
                evidence_row_id=evidence_row_id,
                matured=False,
                snapshot_count=0,
                computed_at=computed_at,
                price_source=None,
                no_lookahead_status="NOT_MATURED",
                maturity_deadline_utc=deadline_iso,
                latest_snapshot_utc=latest_pair.isoformat() if latest_pair else None,
            )
            _apply_not_matured(fields, prefix)
            fields[f"{prefix}_computed_at"] = computed_at
            fields[f"{prefix}_no_lookahead_status"] = "NOT_MATURED"
            outcomes.append(outcome)
            audits.append(_audit_row(evidence_row_id, h, outcome, "HORIZON_NOT_MATURED"))
            continue

        # Matured: collect snaps in (first_seen, deadline]
        snaps = [(ts, px) for ts, px in window_snaps if first_dt < ts <= deadline]
        # Safety: reject any snapshot after deadline (lookahead)
        lookahead_hits = [(ts, px) for ts, px in window_snaps if ts > deadline]
        if lookahead_hits:
            audits.append(
                {
                    "evidence_row_id": evidence_row_id,
                    "horizon": h,
                    "status": "LOOKAHEAD_VIOLATION_FLAGGED",
                    "lookahead_snapshot_count": len(lookahead_hits),
                    "message": "Snapshots after horizon deadline excluded from return computation",
                }
            )

        if not store.available:
            outcome = HorizonOutcome(
                horizon=h,
                horizon_row_id=h_id,
                evidence_row_id=evidence_row_id,
                matured=True,
                snapshot_count=0,
                computed_at=computed_at,
                price_source=None,
                no_lookahead_status="MATURED_BUT_NO_LOCAL_SNAPSHOTS",
                maturity_deadline_utc=deadline_iso,
                latest_snapshot_utc=None,
            )
            fields[f"{prefix}_matured"] = True
            fields[f"{prefix}_max_return"] = None
            fields[f"{prefix}_min_return"] = None
            fields[f"{prefix}_last_return"] = None
            fields[f"{prefix}_snapshot_count"] = 0
            fields[f"{prefix}_computed_at"] = computed_at
            fields[f"{prefix}_price_source"] = None
            fields[f"{prefix}_no_lookahead_status"] = "MATURED_BUT_NO_LOCAL_SNAPSHOTS"
            outcomes.append(outcome)
            audits.append(_audit_row(evidence_row_id, h, outcome, "MATURED_BUT_NO_LOCAL_SNAPSHOTS"))
            continue

        if not snaps or entry is None:
            status = "MATURED_BUT_NO_LOCAL_SNAPSHOTS"
            outcome = HorizonOutcome(
                horizon=h,
                horizon_row_id=h_id,
                evidence_row_id=evidence_row_id,
                matured=True,
                snapshot_count=0,
                computed_at=computed_at,
                price_source="local_market_snapshots" if store.available else None,
                no_lookahead_status=status,
                maturity_deadline_utc=deadline_iso,
                latest_snapshot_utc=latest_pair.isoformat() if latest_pair else None,
            )
            fields[f"{prefix}_matured"] = True
            fields[f"{prefix}_max_return"] = None
            fields[f"{prefix}_min_return"] = None
            fields[f"{prefix}_last_return"] = None
            fields[f"{prefix}_snapshot_count"] = 0
            fields[f"{prefix}_computed_at"] = computed_at
            fields[f"{prefix}_price_source"] = outcome.price_source
            fields[f"{prefix}_no_lookahead_status"] = status
            outcomes.append(outcome)
            audits.append(_audit_row(evidence_row_id, h, outcome, status))
            continue

        returns = [(px - entry) / entry for _, px in snaps]
        max_ret = max(returns)
        min_ret = min(returns)
        last_ret = returns[-1]
        outcome = HorizonOutcome(
            horizon=h,
            horizon_row_id=h_id,
            evidence_row_id=evidence_row_id,
            matured=True,
            max_return=max_ret,
            min_return=min_ret,
            last_return=last_ret,
            snapshot_count=len(snaps),
            computed_at=computed_at,
            price_source="local_market_snapshots",
            no_lookahead_status="NO_LOOKAHEAD_OK",
            maturity_deadline_utc=deadline_iso,
            latest_snapshot_utc=latest_pair.isoformat() if latest_pair else None,
        )
        fields[f"{prefix}_matured"] = True
        fields[f"{prefix}_max_return"] = max_ret
        fields[f"{prefix}_min_return"] = min_ret
        fields[f"{prefix}_last_return"] = last_ret
        fields[f"{prefix}_snapshot_count"] = len(snaps)
        fields[f"{prefix}_computed_at"] = computed_at
        fields[f"{prefix}_price_source"] = "local_market_snapshots"
        fields[f"{prefix}_no_lookahead_status"] = "NO_LOOKAHEAD_OK"
        outcomes.append(outcome)
        audits.append(_audit_row(evidence_row_id, h, outcome, "NO_LOOKAHEAD_OK"))

    return fields, outcomes, audits


def _apply_not_matured(fields: dict[str, Any], prefix: str) -> None:
    # Intentionally NULL (not 0.0) for not-matured returns
    fields[f"{prefix}_matured"] = False
    fields[f"{prefix}_max_return"] = None
    fields[f"{prefix}_min_return"] = None
    fields[f"{prefix}_last_return"] = None
    fields[f"{prefix}_snapshot_count"] = 0
    fields[f"{prefix}_price_source"] = None
    fields[f"{prefix}_no_lookahead_status"] = "NOT_MATURED"


def _audit_row(evidence_row_id: str, horizon: str, outcome: HorizonOutcome, code: str) -> dict[str, Any]:
    return {
        "evidence_row_id": evidence_row_id,
        "horizon": horizon,
        "horizon_row_id": outcome.horizon_row_id,
        "matured": outcome.matured,
        "max_return": outcome.max_return,
        "min_return": outcome.min_return,
        "last_return": outcome.last_return,
        "snapshot_count": outcome.snapshot_count,
        "no_lookahead_status": outcome.no_lookahead_status,
        "audit_code": code,
        "maturity_deadline_utc": outcome.maturity_deadline_utc,
        "latest_snapshot_utc": outcome.latest_snapshot_utc,
    }


def outcome_to_dict(o: HorizonOutcome) -> dict[str, Any]:
    return {
        "horizon_row_id": o.horizon_row_id,
        "evidence_row_id": o.evidence_row_id,
        "horizon": o.horizon,
        "horizon_matured": o.matured,
        "horizon_max_return": o.max_return,
        "horizon_min_return": o.min_return,
        "horizon_last_return": o.last_return,
        "horizon_snapshot_count": o.snapshot_count,
        "horizon_computed_at": o.computed_at,
        "horizon_price_source": o.price_source,
        "horizon_no_lookahead_status": o.no_lookahead_status,
        "maturity_deadline_utc": o.maturity_deadline_utc,
        "latest_snapshot_utc": o.latest_snapshot_utc,
    }
