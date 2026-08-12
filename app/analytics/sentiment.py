"""
RSS headline sentiment — lightweight lexicon score in [-1, 1].
Archives raw feed items to SQLite for Gemini historical context.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

log = logging.getLogger("sentiment")

_POSITIVE = frozenset(
    {
        "surge", "rally", "bullish", "gain", "growth", "adoption",
        "breakout", "record", "partnership", "launch", "upgrade",
    }
)
_NEGATIVE = frozenset(
    {
        "crash", "dump", "bearish", "hack", "scam", "rug", "fraud",
        "decline", "loss", "ban", "lawsuit", "collapse", "warning",
    }
)

_DEFAULT_FEEDS = [
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
]


def _lexicon_score(text: str) -> float:
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _parse_items(xml_text: str, limit: int = 20) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        if not title:
            continue
        items.append({
            "title": title,
            "url": link_el.text.strip() if link_el is not None and link_el.text else "",
            "description": desc_el.text.strip() if desc_el is not None and desc_el.text else "",
        })
        if len(items) >= limit:
            break
    if not items:
        for node in root.iter():
            if node.tag.endswith("title") and node.text and node.text.strip():
                items.append({"title": node.text.strip(), "url": "", "description": ""})
            if len(items) >= limit:
                break
    return items


def _feed_list() -> list[tuple[str, str]]:
    override = os.getenv("RSS_FEED_URL")
    if override:
        return [("custom", override)]
    return list(_DEFAULT_FEEDS)


async def _fetch_feed(source: str, url: str, limit: int = 20) -> tuple[str, str, list[dict[str, str]], str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return source, url, _parse_items(resp.text, limit=limit), resp.text
    except Exception as exc:
        log.debug("RSS fetch failed (%s %s): %s", source, url, exc)
        return source, url, [], ""


async def archive_rss_sentiment(limit: int = 15) -> float:
    """Fetch Cointelegraph + Decrypt RSS, archive raw + sentiment records, return aggregate score."""
    from .. import database as db

    feeds = _feed_list()
    tasks = [_fetch_feed(src, url, limit) for src, url in feeds]
    results = await asyncio.gather(*tasks)

    all_scores: list[float] = []
    for source, url, items, raw_xml in results:
        if raw_xml:
            raw_id = db.insert_raw_payload(
                provider=f"rss_{source}",
                payload=raw_xml,
                source_type="rss_feed",
                query=url,
            )
            log.info("raw payload archived: provider=rss_%s items=%d", source, len(items))
        else:
            raw_id = None

        for item in items:
            score = _lexicon_score(item["title"])
            all_scores.append(score)
            try:
                db.insert_sentiment_record({
                    "source": source,
                    "title": item["title"],
                    "url": item.get("url"),
                    "text_excerpt": item.get("description", "")[:500],
                    "sentiment_score": score,
                    "relevance_score": 0.5,
                    "raw_ref_id": raw_id,
                })
            except Exception as exc:
                log.warning("Sentiment record insert failed (%s): %s", source, exc)

    if not all_scores:
        return 0.0
    return round(sum(all_scores) / len(all_scores), 4)


async def fetch_rss_sentiment(feed_url: str | None = None) -> float:
    if feed_url:
        _, _, items, _ = await _fetch_feed("custom", feed_url, 20)
        if not items:
            return 0.0
        scores = [_lexicon_score(i["title"]) for i in items]
        return round(sum(scores) / len(scores), 4)
    return await archive_rss_sentiment()


async def fetch_rss_sentiment_matrix(
    feed_url: str | None = None,
    limit: int = 15,
) -> dict[str, object]:
    """RSS headlines with per-line lexicon scores for the dashboard widget."""
    if feed_url:
        source, url, items, _ = await _fetch_feed("custom", feed_url, limit)
    else:
        source, url, items, _ = await _fetch_feed("cointelegraph", _DEFAULT_FEEDS[0][1], limit)

    scored = [
        {"headline": i["title"], "score": round(_lexicon_score(i["title"]), 4), "source": source}
        for i in items
    ]
    aggregate = round(sum(i["score"] for i in scored) / len(scored), 4) if scored else 0.0
    return {
        "feed_url": url,
        "source": f"{source} RSS",
        "aggregate_score": aggregate,
        "items": scored,
        "count": len(scored),
    }


def fetch_rss_sentiment_sync(feed_url: str | None = None) -> float:
    return asyncio.get_event_loop().run_until_complete(fetch_rss_sentiment(feed_url))
