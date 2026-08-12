"""RSS article reconstruction from raw_provider_payloads (no network, no LLM).

Every raw payload must produce at least one normalization-trace row.
PARSE_FAILED / NO_ITEMS_EXTRACTED are first-class statuses — never silent drops.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any
from urllib.parse import urlparse


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _child_text(item: ET.Element, names: set[str]) -> str:
    for child in list(item):
        if _local_tag(child.tag).lower() in names:
            return _cell(child.text)
    return ""


def parse_rss_payload_items(payload_text: str) -> tuple[list[dict[str, str]], str, str | None]:
    """Parse RSS/Atom XML into article dicts.

    Returns (items, parse_method, parse_error).
    Raises nothing — parse errors returned as parse_error string.
    """
    raw = payload_text if isinstance(payload_text, str) else str(payload_text or "")
    if not raw.strip():
        return [], "empty_payload", "empty_payload"

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return [], "xml_et_parse", f"{type(exc).__name__}:{exc}"
    except Exception as exc:  # noqa: BLE001 — must never silently drop
        return [], "xml_et_parse", f"{type(exc).__name__}:{exc}"

    items: list[dict[str, str]] = []
    # RSS <item>
    for item in root.iter():
        if _local_tag(item.tag).lower() != "item":
            continue
        title = _child_text(item, {"title"})
        if not title:
            continue
        url = _child_text(item, {"link", "guid"})
        published = _child_text(item, {"pubdate", "published", "date", "updated"})
        summary = _strip_html(_child_text(item, {"description", "summary", "content"}))
        items.append(
            {
                "title": title,
                "url": url,
                "published_at": published,
                "text_or_summary": summary,
            }
        )

    if items:
        return items, "xml_rss_item", None

    # Atom <entry>
    for entry in root.iter():
        if _local_tag(entry.tag).lower() != "entry":
            continue
        title = _child_text(entry, {"title"})
        if not title:
            continue
        url = ""
        for child in list(entry):
            if _local_tag(child.tag).lower() == "link":
                href = child.attrib.get("href") or _cell(child.text)
                if href:
                    url = href
                    break
        published = _child_text(entry, {"published", "updated"})
        summary = _strip_html(_child_text(entry, {"summary", "content"}))
        items.append(
            {
                "title": title,
                "url": url,
                "published_at": published,
                "text_or_summary": summary,
            }
        )

    if items:
        return items, "xml_atom_entry", None

    return [], "xml_no_items", None


def source_domain_from_url(url: str, fallback_query: str = "") -> str:
    for candidate in (url, fallback_query):
        text = _cell(candidate)
        if not text:
            continue
        try:
            host = urlparse(text).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                return host
        except Exception:  # noqa: BLE001
            continue
    return ""


def article_hash_for(
    *,
    title: str,
    url: str,
    published_at: str,
    text_or_summary: str,
) -> str:
    material = "|".join(
        [
            _cell(title).lower(),
            _cell(url).lower(),
            _cell(published_at).lower(),
            _cell(text_or_summary)[:500].lower(),
        ]
    )
    return _sha256_text(material)


def normalize_raw_rss_payload(
    *,
    raw_payload_id: Any,
    provider: str,
    source_type: str,
    query: str,
    fetched_at: str,
    payload_hash: str,
    payload_text: str,
    payload_size: int | None = None,
) -> dict[str, Any]:
    """Normalize one raw RSS payload into T0/T1/trace structures.

    Guarantees at least one trace row for the payload.
    """
    rid = _cell(raw_payload_id)
    phash = _cell(payload_hash) or _sha256_text(payload_text or "")
    size = payload_size if payload_size is not None else len(payload_text or "")

    t0 = {
        "raw_payload_id": rid,
        "provider": _cell(provider),
        "source_type": _cell(source_type) or "rss_feed",
        "query": _cell(query),
        "fetched_at": _cell(fetched_at),
        "payload_hash": phash,
        "payload_size": size,
        "parse_attempted": "true",
        "parse_status": "",
    }

    items, parse_method, parse_error = parse_rss_payload_items(payload_text or "")
    articles: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    if parse_error:
        t0["parse_status"] = "PARSE_FAILED"
        traces.append(
            {
                "raw_payload_id": rid,
                "payload_hash": phash,
                "provider": _cell(provider),
                "fetched_at": _cell(fetched_at),
                "article_id": "",
                "article_hash": "",
                "normalization_status": "PARSE_FAILED",
                "parse_method": parse_method,
                "parse_error": parse_error,
                "items_extracted": 0,
            }
        )
        return {"t0": t0, "articles": articles, "traces": traces}

    if not items:
        t0["parse_status"] = "NO_ITEMS_EXTRACTED"
        traces.append(
            {
                "raw_payload_id": rid,
                "payload_hash": phash,
                "provider": _cell(provider),
                "fetched_at": _cell(fetched_at),
                "article_id": "",
                "article_hash": "",
                "normalization_status": "NO_ITEMS_EXTRACTED",
                "parse_method": parse_method,
                "parse_error": "",
                "items_extracted": 0,
            }
        )
        return {"t0": t0, "articles": articles, "traces": traces}

    t0["parse_status"] = "PARSED"
    for idx, item in enumerate(items):
        ahash = article_hash_for(
            title=item["title"],
            url=item["url"],
            published_at=item["published_at"],
            text_or_summary=item["text_or_summary"],
        )
        article_id = f"art_{phash[:12]}_{idx}_{ahash[:12]}"
        domain = source_domain_from_url(item["url"], query)
        text = item["text_or_summary"]
        quality = "OK" if item["title"] and (item["url"] or text) else "THIN"
        llm_eligible = bool(item["title"] and (text or item["url"]))
        article = {
            "article_id": article_id,
            "article_hash": ahash,
            "raw_payload_id": rid,
            "payload_hash": phash,
            "provider": _cell(provider),
            "source_domain": domain,
            "title": item["title"],
            "url": item["url"],
            "published_at": item["published_at"],
            "fetched_at": _cell(fetched_at),
            "text_or_summary": text,
            "parse_method": parse_method,
            "quality_status": quality,
            "llm_corpus_eligible": "true" if llm_eligible else "false",
        }
        articles.append(article)
        traces.append(
            {
                "raw_payload_id": rid,
                "payload_hash": phash,
                "provider": _cell(provider),
                "fetched_at": _cell(fetched_at),
                "article_id": article_id,
                "article_hash": ahash,
                "normalization_status": "PARSED",
                "parse_method": parse_method,
                "parse_error": "",
                "items_extracted": len(items),
            }
        )

    return {"t0": t0, "articles": articles, "traces": traces}


_SYMBOL_RE = re.compile(r"\b\$([A-Z]{2,10})\b")
_TOKEN_NAME_HINTS = (
    "bitcoin",
    "ethereum",
    "solana",
    "dogecoin",
    "shiba",
    "pepe",
    "meme",
    "stablecoin",
    "defi",
    "nft",
)
_SECTOR_HINTS = (
    "etf",
    "regulation",
    "sec",
    "hack",
    "airdrop",
    "layer-2",
    "l2",
    "rwa",
    "ai",
)
_PERSON_HINTS = (
    "trump",
    "musk",
    "saylor",
    "powell",
)


def deterministic_entity_link_candidates(
    article: dict[str, Any],
    symbol_to_price_source_keys: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Light deterministic entity extraction — no LLM."""
    symbol_map = symbol_to_price_source_keys or {}
    title = _cell(article.get("title"))
    text = _cell(article.get("text_or_summary"))
    blob = f"{title} {text}"
    blob_lower = blob.lower()
    out: list[dict[str, Any]] = []

    seen: set[tuple[str, str]] = set()

    def add(
        entity_type: str,
        entity_value: str,
        *,
        matched_keys: list[str] | None = None,
        method: str,
        confidence: str,
        link_status: str,
    ) -> None:
        key = (entity_type, entity_value.lower())
        if key in seen:
            return
        seen.add(key)
        keys = matched_keys or []
        out.append(
            {
                "article_id": article.get("article_id") or "",
                "article_hash": article.get("article_hash") or "",
                "entity_type": entity_type,
                "entity_value": entity_value,
                "matched_price_source_key": keys[0] if keys else "",
                "match_method": method,
                "match_confidence": confidence,
                "link_status": link_status,
            }
        )

    for match in _SYMBOL_RE.findall(blob.upper()):
        sym = match.upper()
        keys = symbol_map.get(sym, []) or symbol_map.get(sym.lower(), [])
        if keys:
            add(
                "symbol",
                sym,
                matched_keys=keys,
                method="dollar_ticker_to_selected_symbol",
                confidence="medium",
                link_status="LINKED",
            )
        else:
            add(
                "symbol",
                sym,
                method="dollar_ticker",
                confidence="low",
                link_status="UNLINKED_CONTEXT_ONLY",
            )

    for name in _TOKEN_NAME_HINTS:
        if name in blob_lower:
            add(
                "token_name",
                name,
                method="token_name_keyword",
                confidence="low",
                link_status="UNLINKED_CONTEXT_ONLY",
            )
    for sector in _SECTOR_HINTS:
        if re.search(rf"\b{re.escape(sector)}\b", blob_lower):
            add(
                "sector",
                sector,
                method="sector_keyword",
                confidence="low",
                link_status="UNLINKED_CONTEXT_ONLY",
            )
    for person in _PERSON_HINTS:
        if person in blob_lower:
            add(
                "person",
                person,
                method="person_keyword",
                confidence="low",
                link_status="UNLINKED_CONTEXT_ONLY",
            )

    if not out:
        add(
            "unknown",
            "",
            method="no_deterministic_match",
            confidence="none",
            link_status="UNLINKED_CONTEXT_ONLY",
        )
    return out
