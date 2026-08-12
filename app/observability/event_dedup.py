"""Event-level deduplication for audit counts (does not alter raw data)."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def time_bucket(ts: str | datetime | None, *, window_minutes: int = 5) -> str:
    """Bucket timestamps into fixed windows for event grouping."""
    if ts is None:
        return "unknown"
    if isinstance(ts, str):
        dt = _parse_ts(ts)
        if dt is None:
            return "unknown"
    else:
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    minute_slot = (dt.minute // window_minutes) * window_minutes
    return dt.strftime(f"%Y-%m-%dT%H:{minute_slot:02d}")


def event_group_key(
    *,
    pair_address: str,
    chain: str,
    event_type: str,
    timestamp: str | datetime | None,
    window_minutes: int = 5,
) -> str:
    bucket = time_bucket(timestamp, window_minutes=window_minutes)
    return f"{chain}|{pair_address}|{event_type}|{bucket}"


def deduplicate_events(
    events: list[dict[str, Any]],
    *,
    pair_key: str = "pair_address",
    chain_key: str = "chain",
    type_key: str = "event_type",
    ts_key: str = "timestamp",
    window_minutes: int = 5,
) -> dict[str, Any]:
    """
    Group repeated snapshots into event-level counts.
    Returns grouped counts and raw event count for comparison.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        key = event_group_key(
            pair_address=str(ev.get(pair_key) or ""),
            chain=str(ev.get(chain_key) or "unknown"),
            event_type=str(ev.get(type_key) or ev.get("alert_type") or ev.get("signal_action") or "unknown"),
            timestamp=ev.get(ts_key),
            window_minutes=window_minutes,
        )
        groups[key].append(ev)

    event_level = [
        {
            "group_key": key,
            "pair_address": items[0].get(pair_key),
            "chain": items[0].get(chain_key),
            "event_type": items[0].get(type_key) or items[0].get("alert_type"),
            "time_bucket": time_bucket(items[0].get(ts_key), window_minutes=window_minutes),
            "snapshot_count": len(items),
            "first_timestamp": items[0].get(ts_key),
            "last_timestamp": items[-1].get(ts_key),
        }
        for key, items in groups.items()
    ]
    return {
        "raw_event_count": len(events),
        "event_level_count": len(event_level),
        "dedup_ratio": round(len(event_level) / max(len(events), 1), 4),
        "groups": event_level,
    }
