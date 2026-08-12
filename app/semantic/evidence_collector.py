"""Bounded evidence collection for social/opportunistic semantic classification."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLUSTER_REGISTRY_PATH = DATA_DIR / "cluster_registry.json"
SEED_TARGETS_PATH = DATA_DIR / "SeedTargets" / "dexscreener_seed_targets_v1.json"

MAX_QUERIES_PER_TOKEN = 5
MAX_EVIDENCE_ITEMS = 10
MAX_SECONDS_PER_TOKEN = 60

PROMPT_EVIDENCE_NOTE = (
    "Evidence is bounded and local-first. Do not invent facts beyond these items."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_item(
    *,
    evidence_id: str,
    source_type: str,
    title: str,
    url: str = "",
    snippet: str = "",
    timestamp: str = "",
    relevance: str = "LOW",
    supports: str = "UNKNOWN",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_type": source_type,
        "title": title[:240],
        "url": url or "",
        "snippet": (snippet or "")[:800],
        "timestamp": timestamp or "",
        "retrieved_at_utc": _utc_now(),
        "relevance": relevance,
        "supports": supports,
        "notes": notes[:400],
    }


def _infer_support_from_text(text: str) -> tuple[str, str]:
    """Heuristic support tag only — never a final verdict."""
    t = (text or "").lower()
    social_keys = (
        "charity",
        "donation",
        "dao",
        "governance",
        "refi",
        "regenerative",
        "public good",
        "public-goods",
        "community treasury",
        "fan community",
        "creator economy",
        "socialfi",
        "social network",
        "ecosystem",
    )
    opp_keys = (
        "meme",
        "parody",
        "pump",
        "moon",
        "100x",
        "speculation",
        "hype",
        "copycat",
        "get rich",
        "no utility",
        "celebrity",
    )
    social_hit = any(k in t for k in social_keys)
    opp_hit = any(k in t for k in opp_keys)
    if social_hit and not opp_hit:
        return "SOCIAL", "MEDIUM"
    if opp_hit and not social_hit:
        return "OPPORTUNISTIC", "MEDIUM"
    if social_hit and opp_hit:
        return "UNKNOWN", "LOW"
    return "UNKNOWN", "LOW"


def _load_cluster_registry_entry(token_address: str, pair_address: str) -> dict[str, Any] | None:
    if not CLUSTER_REGISTRY_PATH.is_file():
        return None
    try:
        data = json.loads(CLUSTER_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in (token_address, pair_address):
        k = str(key or "").strip()
        if k and k in data and isinstance(data[k], dict):
            return data[k]
    # case-insensitive fallback
    lower_map = {str(k).lower(): v for k, v in data.items()}
    for key in (token_address, pair_address):
        k = str(key or "").strip().lower()
        if k and k in lower_map and isinstance(lower_map[k], dict):
            return lower_map[k]
    return None


def _load_user_seed(
    *,
    chain: str,
    pair_address: str,
    token_address: str,
) -> dict[str, Any] | None:
    if not SEED_TARGETS_PATH.is_file():
        return None
    try:
        payload = json.loads(SEED_TARGETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    chain_l = str(chain or "").strip().lower()
    pair_l = str(pair_address or "").strip().lower()
    token_l = str(token_address or "").strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        r_chain = str(row.get("chain") or "").strip().lower()
        r_pair = str(row.get("user_supplied_pair_address") or "").strip().lower()
        r_tok = str(row.get("user_supplied_token_address") or "").strip().lower()
        if chain_l and r_chain and chain_l != r_chain:
            continue
        if pair_l and r_pair and pair_l == r_pair:
            return row
        if token_l and r_tok and token_l == r_tok:
            return row
    return None


def _query_db_evidence(
    *,
    symbol: str,
    pair_address: str,
    chain: str,
    queries_used: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if len(queries_used) >= MAX_QUERIES_PER_TOKEN:
        return items
    try:
        from app import database as db
    except Exception:
        return items

    # coins metadata
    if len(queries_used) < MAX_QUERIES_PER_TOKEN:
        queries_used.append("db:coins")
        try:
            with db.get_db() as conn:
                row = None
                if pair_address:
                    row = conn.execute(
                        "SELECT * FROM coins WHERE lower(pair_address)=lower(?) LIMIT 1",
                        (pair_address,),
                    ).fetchone()
                if row is None and symbol:
                    row = conn.execute(
                        "SELECT * FROM coins WHERE upper(symbol)=upper(?) LIMIT 1",
                        (symbol,),
                    ).fetchone()
                if row is not None:
                    d = dict(row)
                    snippet = (
                        f"symbol={d.get('symbol')} name={d.get('name')} chain={d.get('chain')} "
                        f"provider={d.get('provider')} url={d.get('provider_url')}"
                    )
                    supports, rel = _infer_support_from_text(
                        f"{d.get('symbol')} {d.get('name')}"
                    )
                    # Name/symbol alone must stay low relevance
                    items.append(
                        _evidence_item(
                            evidence_id=f"dexmeta-{d.get('id')}",
                            source_type="DEXSCREENER",
                            title=f"Local DexScreener coin row: {d.get('symbol')}",
                            url=str(d.get("provider_url") or ""),
                            snippet=snippet,
                            timestamp=str(d.get("last_seen_at") or ""),
                            relevance="LOW",
                            supports="UNKNOWN",
                            notes="Local cache only; symbol/name alone are not sufficient for confirmation.",
                        )
                    )
                    # Official links if present in provider_url host only
                    if d.get("provider_url"):
                        items.append(
                            _evidence_item(
                                evidence_id=f"provider-url-{d.get('id')}",
                                source_type="OFFICIAL_WEBSITE",
                                title="Provider URL from local coins cache",
                                url=str(d.get("provider_url")),
                                snippet=str(d.get("provider_url")),
                                relevance="LOW",
                                supports=supports,
                                notes="Cached provider link; not crawled.",
                            )
                        )
        except Exception:
            pass

    # raw_provider_payloads
    if len(queries_used) < MAX_QUERIES_PER_TOKEN and len(items) < MAX_EVIDENCE_ITEMS:
        queries_used.append("db:raw_provider_payloads")
        try:
            with db.get_db() as conn:
                rows = []
                if pair_address:
                    rows = conn.execute(
                        """
                        SELECT id, timestamp, provider, source_type, query, symbol, payload_json_or_text
                        FROM raw_provider_payloads
                        WHERE lower(pair_address)=lower(?)
                        ORDER BY id DESC LIMIT 3
                        """,
                        (pair_address,),
                    ).fetchall()
                if not rows and symbol:
                    rows = conn.execute(
                        """
                        SELECT id, timestamp, provider, source_type, query, symbol, payload_json_or_text
                        FROM raw_provider_payloads
                        WHERE upper(symbol)=upper(?)
                        ORDER BY id DESC LIMIT 2
                        """,
                        (symbol,),
                    ).fetchall()
                for r in rows:
                    d = dict(r)
                    payload = str(d.get("payload_json_or_text") or "")[:600]
                    supports, rel = _infer_support_from_text(payload)
                    items.append(
                        _evidence_item(
                            evidence_id=f"raw-{d.get('id')}",
                            source_type="RAW_PROVIDER_PAYLOAD",
                            title=f"Raw provider payload ({d.get('provider')})",
                            snippet=payload,
                            timestamp=str(d.get("timestamp") or ""),
                            relevance=rel if supports != "UNKNOWN" else "LOW",
                            supports=supports,
                            notes="Local raw_provider_payloads cache.",
                        )
                    )
                    if len(items) >= MAX_EVIDENCE_ITEMS:
                        break
        except Exception:
            pass

    # sentiment_records (headline-level; weak coin linkage)
    if len(queries_used) < MAX_QUERIES_PER_TOKEN and len(items) < MAX_EVIDENCE_ITEMS:
        queries_used.append("db:sentiment_records")
        try:
            with db.get_db() as conn:
                cols = {c[1] for c in conn.execute("PRAGMA table_info(sentiment_records)").fetchall()}
                if "headline" in cols or "text" in cols or "title" in cols:
                    # Pull recent headlines that mention symbol (bounded)
                    sym = (symbol or "").strip()
                    if sym and len(sym) >= 3:
                        like = f"%{sym}%"
                        # Discover text-ish columns
                        text_col = "headline" if "headline" in cols else (
                            "title" if "title" in cols else ("text" if "text" in cols else None)
                        )
                        if text_col:
                            rows = conn.execute(
                                f"""
                                SELECT * FROM sentiment_records
                                WHERE {text_col} LIKE ?
                                ORDER BY id DESC LIMIT 2
                                """,
                                (like,),
                            ).fetchall()
                            for r in rows:
                                d = dict(r)
                                snippet = str(d.get(text_col) or "")
                                supports, rel = _infer_support_from_text(snippet)
                                items.append(
                                    _evidence_item(
                                        evidence_id=f"rss-{d.get('id')}",
                                        source_type="RSS",
                                        title="Local sentiment/RSS record",
                                        snippet=snippet,
                                        timestamp=str(d.get("timestamp") or d.get("created_at") or ""),
                                        relevance="LOW",
                                        supports=supports,
                                        notes="Headline-level; weak coin identity linkage.",
                                    )
                                )
        except Exception:
            pass

    return items[:MAX_EVIDENCE_ITEMS]


def collect_evidence_bundle(
    *,
    chain: str = "",
    pair_address: str = "",
    token_address: str = "",
    symbol: str = "",
    name: str = "",
    provider_url: str = "",
    allow_web_search: bool = False,
) -> dict[str, Any]:
    """Collect bounded local evidence. Optional web search is off by default."""
    started = time.monotonic()
    queries_used: list[str] = []
    evidence: list[dict[str, Any]] = []
    counter_evidence: list[dict[str, Any]] = []

    seed = _load_user_seed(chain=chain, pair_address=pair_address, token_address=token_address)
    user_seed_collection = ""
    user_seed_label = ""
    user_hypothesis = ""
    if seed:
        queries_used.append("seed_targets")
        user_seed_collection = str(seed.get("seed_collection") or "")
        user_seed_label = user_seed_collection
        # Provenance only — never final truth
        user_hypothesis = (
            f"User seed hypothesized category={user_seed_collection} "
            f"(provenance only; not final classification)."
        )
        evidence.append(
            _evidence_item(
                evidence_id=f"seed-{seed.get('target_id') or 'x'}",
                source_type="USER_SEED",
                title="User seed list category (provenance only)",
                url=str(seed.get("provider_pair_url") or ""),
                snippet=user_hypothesis,
                relevance="LOW",
                supports="UNKNOWN",
                notes="USER_SEED must not auto-confirm SOCIAL or OPPORTUNISTIC.",
            )
        )

    reg = _load_cluster_registry_entry(token_address, pair_address)
    if reg:
        queries_used.append("cluster_registry")
        reasoning = str(reg.get("reasoning") or "")
        label = str(reg.get("cluster_label") or "")
        supports, rel = _infer_support_from_text(f"{label} {reasoning}")
        item = _evidence_item(
            evidence_id="legacy-cluster-registry",
            source_type="LOCAL_CACHE",
            title=f"Legacy cluster_registry label={label}",
            snippet=reasoning or label,
            timestamp=str(reg.get("assigned_at") or ""),
            relevance="MEDIUM" if reasoning else "LOW",
            supports=supports,
            notes="Legacy sticky cluster — diagnostic evidence only, not semantic authority.",
        )
        evidence.append(item)

    if time.monotonic() - started < MAX_SECONDS_PER_TOKEN:
        evidence.extend(
            _query_db_evidence(
                symbol=symbol,
                pair_address=pair_address,
                chain=chain,
                queries_used=queries_used,
            )
        )

    # Deduplicate by evidence_id and cap
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in evidence:
        eid = str(item.get("evidence_id") or "")
        if eid in seen:
            continue
        seen.add(eid)
        deduped.append(item)
        if len(deduped) >= MAX_EVIDENCE_ITEMS:
            break

    # Split weak opposing signals into counter_evidence when mixed supports present
    supports_set = {str(i.get("supports") or "UNKNOWN") for i in deduped}
    if "SOCIAL" in supports_set and "OPPORTUNISTIC" in supports_set:
        primary = []
        for i in deduped:
            if i.get("supports") == "OPPORTUNISTIC":
                counter_evidence.append(i)
            else:
                primary.append(i)
        deduped = primary

    elapsed = time.monotonic() - started
    quality = _score_evidence_quality(deduped)
    return {
        "evidence_items": deduped,
        "counter_evidence": counter_evidence[:MAX_EVIDENCE_ITEMS],
        "queries_used": queries_used[:MAX_QUERIES_PER_TOKEN],
        "query_count": len(queries_used),
        "elapsed_seconds": round(elapsed, 3),
        "evidence_quality": quality,
        "web_search_used": False,
        "allow_web_search": bool(allow_web_search),
        "user_seed_collection": user_seed_collection,
        "user_seed_label": user_seed_label,
        "user_hypothesis": user_hypothesis,
        "bounded": {
            "max_queries": MAX_QUERIES_PER_TOKEN,
            "max_evidence_items": MAX_EVIDENCE_ITEMS,
            "max_seconds": MAX_SECONDS_PER_TOKEN,
        },
        "note": PROMPT_EVIDENCE_NOTE,
    }


def _score_evidence_quality(items: list[dict[str, Any]]) -> str:
    if not items:
        return "NONE"
    # Ignore USER_SEED-only bundles as NONE/LOW for confirmation purposes
    non_seed = [i for i in items if str(i.get("source_type")) != "USER_SEED"]
    if not non_seed:
        return "NONE"
    high = sum(1 for i in non_seed if i.get("relevance") == "HIGH")
    med = sum(1 for i in non_seed if i.get("relevance") == "MEDIUM")
    substantive = [
        i
        for i in non_seed
        if str(i.get("source_type")) in ("RAW_PROVIDER_PAYLOAD", "RSS", "PUBLIC_WEB", "OFFICIAL_WEBSITE")
        and len(str(i.get("snippet") or "")) > 40
    ]
    if high >= 1 and substantive:
        return "HIGH"
    if med >= 1 or substantive:
        return "MEDIUM"
    return "LOW"


def has_sufficient_evidence_for_llm(bundle: dict[str, Any]) -> bool:
    quality = str(bundle.get("evidence_quality") or "NONE")
    items = bundle.get("evidence_items") or []
    non_seed = [i for i in items if str(i.get("source_type")) != "USER_SEED"]
    return quality in ("HIGH", "MEDIUM", "LOW") and len(non_seed) >= 1
