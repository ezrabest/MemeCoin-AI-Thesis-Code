"""AE8 RSS / news context extraction from local sentiment_records."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.context_intelligence.freshness import apply_stale_nulling, compute_freshness, parse_timestamp
from app.context_intelligence.types import FreshnessMode, SourceStatus


def _window_cutoff(as_of: datetime, hours: float) -> datetime:
    return as_of - timedelta(hours=hours)


def build_rss_context(
    conn: sqlite3.Connection | None,
    *,
    symbol: str | None,
    as_of_timestamp: str,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
    allow_external_fetch: bool = False,
    stats: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    """Return (features, freshness_block, source_status, warnings)."""
    warnings: list[str] = []
    empty_features: dict[str, Any] = {
        "rss_article_count_1h": None,
        "rss_article_count_6h": None,
        "rss_article_count_24h": None,
        "rss_positive_count_24h": None,
        "rss_negative_count_24h": None,
        "rss_neutral_count_24h": None,
        "rss_sentiment_mean_24h": None,
        "rss_sentiment_min_24h": None,
        "rss_sentiment_max_24h": None,
        "rss_keyword_hit_count_24h": None,
        "rss_source_count_24h": None,
        "rss_freshness_minutes": None,
        "rss_missingness_flag": True,
    }

    if allow_external_fetch:
        warnings.append("external_rss_fetch_disabled_by_default_even_with_flag")
        return (
            empty_features,
            compute_freshness(
                source_timestamp=None,
                freshness_reference_timestamp=freshness_reference_timestamp,
                freshness_mode=freshness_mode,
                threshold_minutes=threshold_minutes,
                family_key="rss",
            ),
            SourceStatus.SOURCE_DISABLED_BY_DEFAULT.value,
            warnings,
        )

    if conn is None or not symbol:
        status = SourceStatus.SOURCE_NOT_AVAILABLE.value
        warnings.append("RSS_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="rss",
        )
        return empty_features, freshness, status, warnings

    as_of_dt = parse_timestamp(as_of_timestamp)
    if as_of_dt is None:
        as_of_dt = parse_timestamp(freshness_reference_timestamp)

    cutoff_24h = _window_cutoff(as_of_dt, 24).isoformat() if as_of_dt else as_of_timestamp
    sql = """
        SELECT timestamp, sentiment_score, source, title, symbols_json
        FROM sentiment_records
        WHERE symbols_json LIKE ? AND timestamp <= ? AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT 200
    """
    if stats is not None:
        stats.queries_executed.append(sql)
    try:
        rows = conn.execute(sql, (f"%{symbol}%", as_of_timestamp, cutoff_24h)).fetchall()
        if stats is not None:
            stats.record_scan("sentiment_records", len(rows))
            stats.record_load("sentiment_records", len(rows))
    except sqlite3.Error as exc:
        warnings.append(f"rss_query_error:{exc}")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="rss",
        )
        return empty_features, freshness, SourceStatus.SOURCE_ERROR.value, warnings

    if not rows:
        warnings.append("RSS_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="rss",
        )
        return empty_features, freshness, SourceStatus.SOURCE_EMPTY.value, warnings

    latest_ts = str(rows[0][0])
    counts = {"1h": 0, "6h": 0, "24h": 0}
    pos = neg = neutral = 0
    scores_24h: list[float] = []
    sources: set[str] = set()
    keyword_hits = 0
    sym_upper = symbol.upper()

    for ts, score, source, title, symbols_json in rows:
        row_dt = parse_timestamp(str(ts))
        if row_dt is None or as_of_dt is None:
            continue
        age_h = (as_of_dt - row_dt).total_seconds() / 3600.0
        if age_h <= 24:
            counts["24h"] += 1
            if score is not None:
                s = float(score)
                scores_24h.append(s)
                if s > 0.1:
                    pos += 1
                elif s < -0.1:
                    neg += 1
                else:
                    neutral += 1
            if source:
                sources.add(str(source))
            title_l = (title or "").lower()
            if sym_upper.lower() in title_l:
                keyword_hits += 1
            try:
                syms = json.loads(symbols_json or "[]")
                if isinstance(syms, list) and sym_upper in {str(x).upper() for x in syms}:
                    keyword_hits += 1
            except (json.JSONDecodeError, TypeError):
                pass
        if age_h <= 6:
            counts["6h"] += 1
        if age_h <= 1:
            counts["1h"] += 1

    features: dict[str, Any] = {
        "rss_article_count_1h": counts["1h"],
        "rss_article_count_6h": counts["6h"],
        "rss_article_count_24h": counts["24h"],
        "rss_positive_count_24h": pos,
        "rss_negative_count_24h": neg,
        "rss_neutral_count_24h": neutral,
        "rss_sentiment_mean_24h": round(sum(scores_24h) / len(scores_24h), 6) if scores_24h else None,
        "rss_sentiment_min_24h": min(scores_24h) if scores_24h else None,
        "rss_sentiment_max_24h": max(scores_24h) if scores_24h else None,
        "rss_keyword_hit_count_24h": keyword_hits,
        "rss_source_count_24h": len(sources),
        "rss_freshness_minutes": None,
        "rss_missingness_flag": False,
    }

    freshness = compute_freshness(
        source_timestamp=latest_ts,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=threshold_minutes,
        family_key="rss",
    )
    features["rss_freshness_minutes"] = freshness.get("freshness_minutes")

    if freshness.get("freshness_status") == "STALE":
        source_status = SourceStatus.SOURCE_STALE.value
    else:
        source_status = SourceStatus.SOURCE_OK.value

    features = apply_stale_nulling(features, freshness, missingness_flag_key="rss_missingness_flag")
    return features, freshness, source_status, warnings
