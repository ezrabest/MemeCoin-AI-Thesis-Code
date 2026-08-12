"""Web evidence helpers for AE12-SentimentFix Gemini adjudication."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s\]\)\"']+", text)
    out: list[str] = []
    for u in urls:
        u = u.rstrip(".,;)")
        if u not in out:
            out.append(u)
    return out


def is_valid_public_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def normalize_source_urls(urls: list[str]) -> list[str]:
    out: list[str] = []
    for u in urls or []:
        u = str(u).strip()
        if is_valid_public_url(u) and u not in out:
            out.append(u)
    return out


def web_grounding_available() -> bool:
    """Best-effort check; actual grounding depends on Gemini tool support at runtime."""
    try:
        import google.generativeai as genai  # noqa: F401

        return True
    except ImportError:
        return False


def extract_grounding_metadata(response: Any) -> dict[str, Any]:
    """Extract URLs/titles from Gemini response grounding metadata when present."""
    urls: list[str] = []
    titles: list[str] = []
    snippets: list[str] = []
    grounded = False
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            grounded = True
            chunks = getattr(gm, "grounding_chunks", None) or []
            for ch in chunks:
                web = getattr(ch, "web", None)
                if web:
                    uri = getattr(web, "uri", None) or getattr(web, "url", None)
                    title = getattr(web, "title", None)
                    if uri:
                        urls.append(str(uri))
                    if title:
                        titles.append(str(title))
            supports = getattr(gm, "grounding_supports", None) or []
            for sp in supports:
                seg = getattr(sp, "segment", None)
                if seg and getattr(seg, "text", None):
                    snippets.append(str(seg.text))
    except Exception:
        pass
    if not urls:
        urls = extract_urls_from_text(getattr(response, "text", "") or "")
    return {
        "web_grounding_used": grounded and bool(urls),
        "source_urls": normalize_source_urls(urls),
        "source_titles": titles,
        "source_snippets": snippets,
    }
