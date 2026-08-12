"""Demo price oracle — local snapshots only, no look-ahead, runtime age checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.paper_trading.types import PriceStatus

DEFAULT_MAX_PROVIDER_TIME_SKEW_SECONDS = 5.0


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


@dataclass
class PriceLookupResult:
    price: float | None = None
    price_source: str = "local_market_snapshots"
    price_snapshot_id: int | str | None = None
    price_timestamp: str | None = None
    price_timestamp_used: str | None = None
    order_timestamp: str = ""
    order_created_at_utc: str = ""
    decision_created_at_utc: str | None = None
    system_now_utc: str = ""
    snapshot_provider_timestamp: str | None = None
    snapshot_ingested_at_utc: str | None = None
    price_age_seconds: float | None = None
    max_price_age_seconds: float = 30.0
    time_skew_seconds: float | None = None
    max_provider_time_skew_seconds: float = DEFAULT_MAX_PROVIDER_TIME_SKEW_SECONDS
    lookahead_detected: bool = False
    provider_time_skew_detected: bool = False
    price_status: str = PriceStatus.PRICE_MISSING.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "price_source": self.price_source,
            "price_snapshot_id": self.price_snapshot_id,
            "price_timestamp": self.price_timestamp,
            "price_timestamp_used": self.price_timestamp_used,
            "order_timestamp": self.order_timestamp,
            "order_created_at_utc": self.order_created_at_utc,
            "decision_created_at_utc": self.decision_created_at_utc,
            "system_now_utc": self.system_now_utc,
            "snapshot_provider_timestamp": self.snapshot_provider_timestamp,
            "snapshot_ingested_at_utc": self.snapshot_ingested_at_utc,
            "price_age_seconds": self.price_age_seconds,
            "max_price_age_seconds": self.max_price_age_seconds,
            "time_skew_seconds": self.time_skew_seconds,
            "max_provider_time_skew_seconds": self.max_provider_time_skew_seconds,
            "lookahead_detected": self.lookahead_detected,
            "provider_time_skew_detected": self.provider_time_skew_detected,
            "price_status": self.price_status,
        }


@dataclass
class DemoPriceOracle:
    """Local price oracle using market_snapshots or in-memory snapshots."""

    max_price_age_seconds: float = 30.0
    max_provider_time_skew_seconds: float = DEFAULT_MAX_PROVIDER_TIME_SKEW_SECONDS
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def lookup_price(
        self,
        *,
        coin_id: int | None = None,
        pair_address: str | None = None,
        order_created_at_utc: str | None = None,
        decision_created_at_utc: str | None = None,
        conn: sqlite3.Connection | None = None,
        # Backward-compatible alias
        order_timestamp: str | None = None,
    ) -> PriceLookupResult:
        system_now = datetime.now(timezone.utc)
        system_now_utc = system_now.isoformat()
        order_ts = order_created_at_utc or order_timestamp or system_now_utc
        order_dt = parse_ts(order_ts)
        if order_dt is None:
            result = PriceLookupResult(
                order_timestamp=order_ts,
                order_created_at_utc=order_ts,
                decision_created_at_utc=decision_created_at_utc,
                system_now_utc=system_now_utc,
                max_price_age_seconds=self.max_price_age_seconds,
                max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
                price_status=PriceStatus.PRICE_MISSING.value,
            )
            self._audit(result)
            return result

        candidates: list[dict[str, Any]] = []
        if self.snapshots:
            candidates = self._filter_snapshots(self.snapshots, coin_id, pair_address)
        elif conn is not None and coin_id is not None:
            candidates = self._query_db(conn, coin_id, order_dt)

        if not candidates:
            result = PriceLookupResult(
                order_timestamp=order_ts,
                order_created_at_utc=order_ts,
                decision_created_at_utc=decision_created_at_utc,
                system_now_utc=system_now_utc,
                max_price_age_seconds=self.max_price_age_seconds,
                max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
                price_status=PriceStatus.PRICE_MISSING.value,
            )
            self._audit(result)
            return result

        best: dict[str, Any] | None = None
        best_ts: datetime | None = None
        lookahead = False
        provider_skew = False
        skew_seconds: float | None = None

        for snap in candidates:
            snap_ts = parse_ts(snap.get("timestamp") or snap.get("snapshot_timestamp"))
            if snap_ts is None:
                continue

            provider_skew_val = (snap_ts - system_now).total_seconds()
            if provider_skew_val > self.max_provider_time_skew_seconds:
                provider_skew = True
                if skew_seconds is None or provider_skew_val > skew_seconds:
                    skew_seconds = provider_skew_val
                continue

            if snap_ts > order_dt:
                lookahead = True
                continue

            if best_ts is None or snap_ts > best_ts:
                best = snap
                best_ts = snap_ts

        if best is None or best_ts is None:
            status = PriceStatus.PRICE_MISSING.value
            if provider_skew and not lookahead:
                status = PriceStatus.PRICE_PROVIDER_TIME_SKEW_REJECTED.value
            elif lookahead:
                status = PriceStatus.PRICE_LOOKAHEAD_REJECTED.value
            result = PriceLookupResult(
                order_timestamp=order_ts,
                order_created_at_utc=order_ts,
                decision_created_at_utc=decision_created_at_utc,
                system_now_utc=system_now_utc,
                max_price_age_seconds=self.max_price_age_seconds,
                max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
                lookahead_detected=lookahead,
                provider_time_skew_detected=provider_skew,
                time_skew_seconds=skew_seconds,
                price_status=status,
            )
            self._audit(result)
            return result

        age_seconds = (order_dt - best_ts).total_seconds()
        price = best.get("price")
        try:
            price_f = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_f = None

        snap_provider_ts = best_ts.isoformat()
        snap_ingested = best.get("ingested_at_utc") or best.get("snapshot_ingested_at_utc")

        if price_f is not None and price_f <= 0:
            result = PriceLookupResult(
                price=price_f,
                price_snapshot_id=best.get("id") or best.get("snapshot_id"),
                price_timestamp=snap_provider_ts,
                price_timestamp_used=snap_provider_ts,
                order_timestamp=order_ts,
                order_created_at_utc=order_ts,
                decision_created_at_utc=decision_created_at_utc,
                system_now_utc=system_now_utc,
                snapshot_provider_timestamp=snap_provider_ts,
                snapshot_ingested_at_utc=snap_ingested,
                price_age_seconds=age_seconds,
                max_price_age_seconds=self.max_price_age_seconds,
                max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
                time_skew_seconds=(best_ts - system_now).total_seconds(),
                price_status=PriceStatus.PRICE_INVALID_ZERO_OR_NEGATIVE.value,
            )
            self._audit(result)
            return result

        if age_seconds > self.max_price_age_seconds:
            result = PriceLookupResult(
                price=price_f,
                price_snapshot_id=best.get("id") or best.get("snapshot_id"),
                price_timestamp=snap_provider_ts,
                price_timestamp_used=snap_provider_ts,
                order_timestamp=order_ts,
                order_created_at_utc=order_ts,
                decision_created_at_utc=decision_created_at_utc,
                system_now_utc=system_now_utc,
                snapshot_provider_timestamp=snap_provider_ts,
                snapshot_ingested_at_utc=snap_ingested,
                price_age_seconds=age_seconds,
                max_price_age_seconds=self.max_price_age_seconds,
                max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
                time_skew_seconds=(best_ts - system_now).total_seconds(),
                price_status=PriceStatus.PRICE_STALE.value,
            )
            self._audit(result)
            return result

        result = PriceLookupResult(
            price=price_f,
            price_snapshot_id=best.get("id") or best.get("snapshot_id"),
            price_timestamp=snap_provider_ts,
            price_timestamp_used=snap_provider_ts,
            order_timestamp=order_ts,
            order_created_at_utc=order_ts,
            decision_created_at_utc=decision_created_at_utc,
            system_now_utc=system_now_utc,
            snapshot_provider_timestamp=snap_provider_ts,
            snapshot_ingested_at_utc=snap_ingested,
            price_age_seconds=age_seconds,
            max_price_age_seconds=self.max_price_age_seconds,
            max_provider_time_skew_seconds=self.max_provider_time_skew_seconds,
            time_skew_seconds=(best_ts - system_now).total_seconds(),
            lookahead_detected=False,
            provider_time_skew_detected=False,
            price_status=PriceStatus.PRICE_OK.value if price_f is not None else PriceStatus.PRICE_MISSING.value,
        )
        self._audit(result)
        return result

    def _filter_snapshots(
        self,
        snapshots: list[dict[str, Any]],
        coin_id: int | None,
        pair_address: str | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for snap in snapshots:
            if coin_id is not None and snap.get("coin_id") != coin_id:
                continue
            if pair_address and snap.get("pair_address") and snap.get("pair_address") != pair_address:
                continue
            out.append(snap)
        return out

    def _query_db(
        self,
        conn: sqlite3.Connection,
        coin_id: int,
        order_dt: datetime,
    ) -> list[dict[str, Any]]:
        order_iso = order_dt.isoformat()
        rows = conn.execute(
            """
            SELECT id, coin_id, timestamp, price, liquidity, volume_24h
            FROM market_snapshots
            WHERE coin_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 50
            """,
            (coin_id, order_iso),
        ).fetchall()
        return [
            {
                "id": row[0],
                "coin_id": row[1],
                "timestamp": row[2],
                "price": row[3],
                "liquidity": row[4],
                "volume_24h": row[5],
            }
            for row in rows
        ]

    def _audit(self, result: PriceLookupResult) -> None:
        row = result.to_dict()
        row["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.audit_log.append(row)
