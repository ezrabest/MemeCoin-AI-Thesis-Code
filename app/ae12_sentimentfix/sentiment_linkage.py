"""Read-only sentiment linkage audit against trader.db (no writes)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def sqlite_readonly_sentiment_stats(db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "MISSING",
        "path": str(db_path),
        "sentiment_records_count": None,
        "sentiment_records_latest": None,
        "sentiment_social_marker_rows": None,
        "sentiment_news_marker_rows": None,
        "semantic_linkage_gap": False,
        "semantic_linkage_status": "UNKNOWN",
    }
    if not db_path.is_file():
        return out
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            count = int(conn.execute("SELECT COUNT(*) AS c FROM sentiment_records").fetchone()["c"])
            latest = conn.execute(
                "SELECT timestamp AS t FROM sentiment_records ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(sentiment_records)").fetchall()}
            text_cols = [c for c in ("title", "text", "body", "content", "summary", "source") if c in cols]
            social = None
            news = None
            if text_cols:
                expr = " OR ".join(
                    f"lower(COALESCE({c}, '')) LIKE '%social%' OR lower(COALESCE({c}, '')) LIKE '%twitter%' "
                    f"OR lower(COALESCE({c}, '')) LIKE '%telegram%' OR lower(COALESCE({c}, '')) LIKE '%reddit%' "
                    f"OR lower(COALESCE({c}, '')) LIKE '%community%'"
                    for c in text_cols[:2]
                )
                social = int(conn.execute(f"SELECT COUNT(*) AS c FROM sentiment_records WHERE {expr}").fetchone()["c"])
                news_expr = " OR ".join(
                    f"lower(COALESCE({c}, '')) LIKE '%news%' OR lower(COALESCE({c}, '')) LIKE '%rss%' "
                    f"OR lower(COALESCE({c}, '')) LIKE '%headline%'"
                    for c in text_cols[:2]
                )
                news = int(
                    conn.execute(f"SELECT COUNT(*) AS c FROM sentiment_records WHERE {news_expr}").fetchone()["c"]
                )
            out.update(
                {
                    "status": "OK",
                    "sentiment_records_count": count,
                    "sentiment_records_latest": str(latest["t"]) if latest and latest["t"] is not None else None,
                    "sentiment_social_marker_rows": social,
                    "sentiment_news_marker_rows": news,
                    "text_columns_used": text_cols,
                }
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        out["status"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def assess_semantic_linkage_gap(
    *,
    sentiment_stats: dict[str, Any],
    derived_semantic_unknown_share: float,
    derived_social_count: int,
) -> dict[str, Any]:
    sent_count = int(sentiment_stats.get("sentiment_records_count") or 0)
    social_markers = int(sentiment_stats.get("sentiment_social_marker_rows") or 0)
    gap = bool(sent_count > 1000 and derived_semantic_unknown_share > 0.5 and derived_social_count < max(10, social_markers // 100))
    return {
        "semantic_linkage_gap_found": gap,
        "semantic_linkage_status": "SEMANTIC_LINKAGE_GAP" if gap else "LINKAGE_LIMITED_OR_OK",
        "note": (
            "Sentiment/social marker rows exist in SQLite but derived dual-axis semantic SOCIAL is sparse; "
            "historical decision/opportunity rows lack linked semantic fields."
            if gap
            else "Linkage audited read-only; perfect join not required."
        ),
        "sentiment_records_count": sent_count,
        "sentiment_social_marker_rows": social_markers,
        "derived_social_count": derived_social_count,
        "derived_semantic_unknown_share": derived_semantic_unknown_share,
    }
