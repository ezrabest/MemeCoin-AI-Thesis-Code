"""Market row provenance enrichment — never invent timestamps."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None and row[key] != "":
            return row[key]
    return None


def _age_seconds(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds())


def enrich_market_provenance(row: dict[str, Any]) -> dict[str, Any]:
    """Return provenance fields derived only from existing row data."""
    src = dict(row or {})
    now = _utc_now()

    source_provider = _first_present(
        src,
        "source_provider",
        "price_source",
        "latest_price_source",
        "market_data_provider",
        "provider",
        "resolution_source",
    )
    price_updated_at = _first_present(
        src,
        "price_updated_at",
        "latest_price_timestamp",
        "last_price_timestamp",
        "price_timestamp",
        "matched_price_ts",
        # Single-timestamp market rows (this codebase's `coins` table has one
        # last_seen_at covering the whole snapshot row) fall back here.
        "last_seen_at",
    )
    liquidity_updated_at = _first_present(
        src,
        "liquidity_updated_at",
        "latest_liquidity_timestamp",
        "liquidity_timestamp",
        "last_liquidity_timestamp",
        "last_seen_at",
    )
    provider_last_seen_at = _first_present(
        src,
        "provider_last_seen_at",
        "last_seen_in_market",
        "provider_seen_at",
        "market_last_seen_at",
    )

    price_ts = _parse_ts(price_updated_at)
    liq_ts = _parse_ts(liquidity_updated_at)
    seen_ts = _parse_ts(provider_last_seen_at)

    price_age_seconds = _age_seconds(price_ts, now)
    liquidity_age_seconds = _age_seconds(liq_ts, now)
    provider_seen_age_seconds = _age_seconds(seen_ts, now)

    missing: list[str] = []
    if not source_provider:
        missing.append("source_provider")
    if price_updated_at is None:
        missing.append("price_updated_at")
    if liquidity_updated_at is None:
        missing.append("liquidity_updated_at")

    if missing:
        provenance_status = "incomplete"
    elif price_ts is None or liq_ts is None:
        provenance_status = "partial_unparseable_timestamps"
    else:
        provenance_status = "ok"

    return {
        "source_provider": str(source_provider).strip() if source_provider else None,
        "price_updated_at": price_updated_at if price_updated_at is not None else None,
        "liquidity_updated_at": liquidity_updated_at if liquidity_updated_at is not None else None,
        "provider_last_seen_at": provider_last_seen_at if provider_last_seen_at is not None else None,
        "price_age_seconds": price_age_seconds,
        "liquidity_age_seconds": liquidity_age_seconds,
        "provider_seen_age_seconds": provider_seen_age_seconds,
        "provenance_status": provenance_status,
        "provenance_missing_fields": missing,
        "provenance_checked_at_utc": now.isoformat(),
        "provenance_never_invented_timestamps": True,
    }