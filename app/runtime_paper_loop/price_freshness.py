"""Price freshness evaluation using snapshot timestamps — not loop runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.paper_trading.price_oracle import parse_ts
from app.runtime_paper_loop.types import utc_now_iso


@dataclass
class PriceFreshnessResult:
    price: float | None = None
    price_timestamp_utc: str | None = None
    loop_observed_at_utc: str = ""
    price_age_seconds: float | None = None
    price_timestamp_source: str | None = None
    source_snapshot_id: str | int | None = None
    exploration_max_price_age_seconds: float = 900.0
    strict_max_price_age_seconds: float = 30.0
    strict_price_fresh: bool = False
    exploration_price_fresh: bool = False
    price_timestamp_missing: bool = False
    price_missing: bool = False
    strict_price_status: str = "PRICE_UNKNOWN"
    exploration_price_status: str = "PRICE_UNKNOWN"
    rejection_reason_if_stale: str | None = None

    def to_audit_dict(
        self,
        *,
        loop_run_id: str,
        iteration: int,
        source_decision_id: str | None,
        source_event_key: str | None,
        pair_address: str | None,
    ) -> dict[str, Any]:
        return {
            "loop_run_id": loop_run_id,
            "iteration": iteration,
            "source_decision_id": source_decision_id,
            "source_event_key": source_event_key,
            "pair_address": pair_address,
            "source_snapshot_id": self.source_snapshot_id,
            "price": self.price,
            "price_timestamp_utc": self.price_timestamp_utc,
            "loop_observed_at_utc": self.loop_observed_at_utc,
            "price_age_seconds": self.price_age_seconds,
            "exploration_max_price_age_seconds": self.exploration_max_price_age_seconds,
            "strict_max_price_age_seconds": self.strict_max_price_age_seconds,
            "strict_price_fresh": self.strict_price_fresh,
            "exploration_price_fresh": self.exploration_price_fresh,
            "price_timestamp_source": self.price_timestamp_source,
            "rejection_reason_if_stale": self.rejection_reason_if_stale,
            "strict_price_freshness_decision": "FRESH" if self.strict_price_fresh else "STALE_OR_MISSING",
            "exploration_price_freshness_decision": "FRESH" if self.exploration_price_fresh else "STALE_OR_MISSING",
        }


def extract_snapshot_price_timestamp(
    price_result: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, str | int | None]:
    """Pick provider/snapshot timestamp — never loop runtime or decision created_at."""
    market = (decision or {}).get("market_context") or {}
    lineage = (decision or {}).get("lineage") or {}
    identity = (decision or {}).get("candidate_identity") or {}

    candidates: list[tuple[str, str | None]] = [
        ("snapshot_provider_timestamp", price_result.get("snapshot_provider_timestamp")),
        ("price_timestamp_used", price_result.get("price_timestamp_used")),
        ("price_timestamp", price_result.get("price_timestamp")),
        ("snapshot_ingested_at_utc", price_result.get("snapshot_ingested_at_utc")),
        ("market_context.snapshot_timestamp", market.get("snapshot_timestamp")),
        ("market_context.price_timestamp", market.get("price_timestamp")),
        ("lineage.source_snapshot_timestamp", lineage.get("source_snapshot_timestamp")),
        ("candidate_identity.event_timestamp", identity.get("event_timestamp")),
    ]
    for source, ts in candidates:
        if ts:
            return str(ts), source, (
                price_result.get("price_snapshot_id")
                or market.get("source_snapshot_id")
                or lineage.get("source_snapshot_id")
            )
    return None, None, price_result.get("price_snapshot_id")


def evaluate_price_freshness(
    price_result: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    exploration_max_price_age_seconds: float,
    strict_max_price_age_seconds: float,
    loop_observed_at_utc: str | None = None,
) -> PriceFreshnessResult:
    """Compute strict vs exploration freshness from snapshot timestamp."""
    observed = loop_observed_at_utc or utc_now_iso()
    result = PriceFreshnessResult(
        price=price_result.get("price"),
        loop_observed_at_utc=observed,
        exploration_max_price_age_seconds=exploration_max_price_age_seconds,
        strict_max_price_age_seconds=strict_max_price_age_seconds,
    )

    ts, source, snap_id = extract_snapshot_price_timestamp(price_result, decision)
    result.price_timestamp_utc = ts
    result.price_timestamp_source = source
    result.source_snapshot_id = snap_id

    if result.price is None or float(result.price) <= 0:
        result.price_missing = True
        result.strict_price_status = "PRICE_MISSING"
        result.exploration_price_status = "PRICE_MISSING"
        result.rejection_reason_if_stale = "PRICE_MISSING"
        return result

    if not ts:
        result.price_timestamp_missing = True
        result.strict_price_status = "PRICE_TIMESTAMP_MISSING"
        result.exploration_price_status = "PRICE_TIMESTAMP_MISSING"
        result.rejection_reason_if_stale = "PRICE_TIMESTAMP_MISSING"
        return result

    ts_dt = parse_ts(ts)
    obs_dt = parse_ts(observed)
    if ts_dt is None or obs_dt is None:
        result.price_timestamp_missing = True
        result.rejection_reason_if_stale = "PRICE_TIMESTAMP_UNPARSEABLE"
        return result

    age = (obs_dt - ts_dt).total_seconds()
    if age < 0:
        age = 0.0
    result.price_age_seconds = age

    result.strict_price_fresh = age <= strict_max_price_age_seconds
    result.exploration_price_fresh = age <= exploration_max_price_age_seconds
    result.strict_price_status = "PRICE_OK" if result.strict_price_fresh else "PRICE_STALE"
    result.exploration_price_status = "PRICE_OK" if result.exploration_price_fresh else "PRICE_STALE"

    if not result.exploration_price_fresh:
        result.rejection_reason_if_stale = "PRICE_STALE_EXPLORATION"
    elif not result.strict_price_fresh:
        result.rejection_reason_if_stale = "PRICE_STALE_STRICT_ONLY"

    return result
