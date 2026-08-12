"""AE18 cached news/sentiment panel — local reads only, never live RSS on GET.

Reads previously archived sentiment records and raw RSS payloads. If nothing is
cached, returns an explicit status instead of a blank panel.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

NEWS_SENTIMENT_CACHE_EMPTY = "NEWS_SENTIMENT_CACHE_EMPTY"
NEWS_SENTIMENT_CACHE_STALE = "NEWS_SENTIMENT_CACHE_STALE"
NEWS_SENTIMENT_CACHE_UNAVAILABLE = "NEWS_SENTIMENT_CACHE_UNAVAILABLE"
NEWS_SENTIMENT_CACHE_READY = "NEWS_SENTIMENT_CACHE_READY"

STALE_AFTER_SECONDS = 6 * 3600.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(ts: Any) -> float | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _label(score: float) -> str:
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def _load_sentiment_records(limit: int) -> tuple[list[dict[str, Any]], str]:
    try:
        from app import database as db

        return list(db.get_sentiment_records(limit=limit) or []), ""
    except Exception as exc:  # noqa: BLE001 - cache unavailable is an explicit status
        return [], f"{type(exc).__name__}: {exc}"[:200]


def _count_cached_rss_payloads() -> tuple[int, str]:
    """Count archived raw RSS payloads without triggering any network call."""
    try:
        from app import database as db

        with db.get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c, MAX(timestamp) AS last_ts FROM raw_provider_payloads "
                "WHERE source_type = 'rss_feed' OR provider LIKE 'rss_%'"
            ).fetchone()
        if row is None:
            return 0, ""
        data = dict(row)
        return int(data.get("c") or 0), str(data.get("last_ts") or "")
    except Exception:  # noqa: BLE001 - table may not exist yet
        return 0, ""


def build_cached_news_sentiment(*, limit: int = 15) -> dict[str, Any]:
    """Build the RSS / News Sentiment panel payload from local cache only."""
    records, read_error = _load_sentiment_records(max(limit, 1))
    rss_items_count, rss_last_ts = _count_cached_rss_payloads()

    items: list[dict[str, Any]] = []
    scores: list[float] = []
    latest_ts = ""
    for rec in records:
        try:
            score = float(rec.get("sentiment_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        ts = str(rec.get("timestamp") or "")
        if ts > latest_ts:
            latest_ts = ts
        symbols = rec.get("symbols_json")
        if isinstance(symbols, str) and symbols.strip().startswith("["):
            try:
                symbols = json.loads(symbols)
            except ValueError:
                symbols = []
        items.append(
            {
                "headline": rec.get("title") or rec.get("text_excerpt") or "",
                "source": rec.get("source") or "cached_rss",
                "timestamp": ts,
                "sentiment_score": score,
                "sentiment_label": _label(score),
                "related_coin_pair": None,
                "symbols": symbols if isinstance(symbols, list) else [],
                "relevance_note": "Cached headline lexicon score — not SOCIAL_CONFIRMED",
                "cached_only": True,
            }
        )

    aggregate = round(sum(scores) / len(scores), 4) if scores else None
    last_fetch_at = latest_ts or rss_last_ts
    age = _age_seconds(last_fetch_at)

    if read_error:
        status = NEWS_SENTIMENT_CACHE_UNAVAILABLE
        missing_reason = f"sentiment cache read failed: {read_error}"
    elif not items and not rss_items_count:
        status = NEWS_SENTIMENT_CACHE_EMPTY
        missing_reason = (
            "No cached sentiment records or archived RSS payloads exist yet. "
            "Run an explicit sentiment refresh to populate the cache."
        )
    elif not items:
        status = NEWS_SENTIMENT_CACHE_EMPTY
        missing_reason = (
            f"{rss_items_count} raw RSS payloads archived but no scored sentiment records exist."
        )
    elif age is not None and age > STALE_AFTER_SECONDS:
        status = NEWS_SENTIMENT_CACHE_STALE
        missing_reason = f"Newest cached record is {int(age // 3600)}h old."
    else:
        status = NEWS_SENTIMENT_CACHE_READY
        missing_reason = ""

    if status == NEWS_SENTIMENT_CACHE_READY:
        user_message = ""
    else:
        user_message = f"{status}: {missing_reason}" if missing_reason else status

    return {
        "ok": True,
        "status": "ready" if status == NEWS_SENTIMENT_CACHE_READY else "cached_status",
        "rss_news_sentiment_status": status,
        "rss_cached_items_count": rss_items_count,
        "rss_last_fetch_at": last_fetch_at or None,
        "cached_sentiment_records_count": len(items),
        "news_sentiment_summary_cached": {
            "aggregate_sentiment_score": aggregate,
            "records": len(items),
            "positive": sum(1 for s in scores if s > 0.05),
            "negative": sum(1 for s in scores if s < -0.05),
            "neutral": sum(1 for s in scores if -0.05 <= s <= 0.05),
        },
        "sentiment_source_status": "cached_local_only",
        "sentiment_cache_missing_reason": missing_reason,
        "user_message": user_message,
        "available": bool(items),
        "unavailable_reason": missing_reason or None,
        "aggregate_sentiment_score": aggregate,
        "latest_rss_update": last_fetch_at or None,
        "source": "local_sentiment_cache",
        "feed_url": None,
        "count": len(items),
        "items": items[:limit],
        "rss_status": status,
        "panel_blank": False,
        "get_path_fetches_rss_live": False,
        "external_network_on_load": False,
        "built_at_utc": _utc_now(),
        "llm_calls_used": 0,
    }
