"""
DexScreener API client — async httpx (+ sync helpers for audit scripts)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger("dexscreener")

BASE_URL = "https://api.dexscreener.com/latest/dex"
PAIR_PAGE_URL = "https://dexscreener.com/{chain}/{pair}"
QUERIES = ["meme", "pepe", "doge", "wif", "bonk", "shib"]
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_TIMEOUT = httpx.Timeout(12.0)


def pair_page_url(chain_id: str | None, pair_address: str | None) -> str:
    if not chain_id or not pair_address:
        return ""
    return PAIR_PAGE_URL.format(chain=str(chain_id).lower(), pair=pair_address)


async def _get(client: httpx.AsyncClient, path: str) -> Any | None:
    try:
        from app.runtime.shutdown import should_skip_network
        from app.runtime.ui_get_network_guard import is_ui_get_path_active, record_network_attempt

        if is_ui_get_path_active():
            record_network_attempt("dexscreener")
            log.info("provider fetch skipped due to shutdown (ui_get_path)")
            return None
        if should_skip_network(context=f"dexscreener_async:{path}"):
            return None
    except Exception:
        pass
    try:
        r = await client.get(f"{BASE_URL}{path}", headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("DexScreener %s failed: %s", path, exc)
        return None


def _get_sync(path: str) -> Any | None:
    try:
        from app.runtime.shutdown import should_skip_network
        from app.runtime.ui_get_network_guard import is_ui_get_path_active, record_network_attempt

        if is_ui_get_path_active():
            record_network_attempt("dexscreener")
            return None
        if should_skip_network(context=f"dexscreener_sync:{path}"):
            return None
    except Exception:
        pass
    try:
        with httpx.Client(timeout=_TIMEOUT) as c:
            r = c.get(f"{BASE_URL}{path}", headers=_HEADERS)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        log.warning("DexScreener sync %s failed: %s", path, exc)
        return None


def _extract_pair(data: Any) -> dict | None:
    """Normalize DexScreener pair lookup / search payloads to a single pair dict."""
    if not data:
        return None
    if isinstance(data, dict):
        pair = data.get("pair")
        if isinstance(pair, dict) and pair.get("pairAddress"):
            return pair
        pairs = data.get("pairs")
        if isinstance(pairs, list):
            for p in pairs:
                if isinstance(p, dict) and p.get("pairAddress"):
                    return p
        if data.get("pairAddress"):
            return data
    if isinstance(data, list):
        for p in data:
            if isinstance(p, dict) and p.get("pairAddress"):
                return p
    return None


async def search_pairs(query: str) -> list[dict]:
    async with httpx.AsyncClient() as c:
        data = await _get(c, f"/search/?q={query}")
        return (data or {}).get("pairs") or []


def search_pairs_sync(query: str) -> list[dict]:
    data = _get_sync(f"/search/?q={query}")
    return (data or {}).get("pairs") or []


async def get_pair(chain_id: str, pair_id: str) -> dict | None:
    """GET /latest/dex/pairs/{chainId}/{pairId} — returns verified pair or None."""
    chain = str(chain_id or "").strip()
    pair = str(pair_id or "").strip()
    if not chain or not pair:
        return None
    async with httpx.AsyncClient() as c:
        data = await _get(c, f"/pairs/{chain}/{pair}")
    return _extract_pair(data)


def get_pair_sync(chain_id: str, pair_id: str) -> dict | None:
    """Sync provider pair lookup for audits / clean-feed builder."""
    chain = str(chain_id or "").strip()
    pair = str(pair_id or "").strip()
    if not chain or not pair:
        return None
    return _extract_pair(_get_sync(f"/pairs/{chain}/{pair}"))


async def get_trending_pairs(max_pairs: int = 100) -> list[dict]:
    """Fan-out 6 search queries concurrently, deduplicate by pairAddress."""
    async with httpx.AsyncClient() as c:
        tasks = [_get(c, f"/search/?q={q}") for q in QUERIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    seen: set[str] = set()
    pairs: list[dict] = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        for p in (r.get("pairs") or []):
            addr = p.get("pairAddress", "")
            if addr and addr not in seen and p.get("priceUsd"):
                seen.add(addr)
                pairs.append(p)

    return pairs[:max_pairs]


def get_trending_pairs_sync(max_pairs: int = 100, queries: list[str] | None = None) -> list[dict]:
    """Sync fan-out search for clean forward feed (no DB / no old snapshots).

    Interleaves results across queries so one meme ticker cannot dominate the
    candidate set before diversity controls run.
    """
    qs = queries or QUERIES
    per_query: list[list[dict]] = []
    for q in qs:
        per_query.append(list(search_pairs_sync(q) or []))

    seen: set[str] = set()
    pairs: list[dict] = []
    idx = 0
    while len(pairs) < max_pairs:
        progressed = False
        for bucket in per_query:
            if idx >= len(bucket):
                continue
            p = bucket[idx]
            progressed = True
            addr = str(p.get("pairAddress") or "")
            if addr and addr not in seen and p.get("priceUsd"):
                seen.add(addr)
                pairs.append(p)
                if len(pairs) >= max_pairs:
                    break
        if not progressed:
            break
        idx += 1
    return pairs[:max_pairs]
