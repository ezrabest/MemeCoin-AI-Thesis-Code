#!/usr/bin/env python3
"""AE13J — Ground Truth Market Data Audit + Training Data Validity Review.

Read-only except: writes audit artifacts under data/audits/ and applies no
historical training mutations. Does not retrain, trade, or connect wallet.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae13j_ground_truth_market_data_audit_{TIMESTAMP}"
DB_PATH = ROOT / "data" / "trader.db"
TRAIN_DIR = ROOT / "data" / "training" / "manual_verified_datasets_direct_target_v1"
CLEAN_DIR = ROOT / "data" / "training" / "manual_verified_datasets_clean_for_model"

SAMPLE_NEEDLES = (
    "DDk1QmkbZBtTSpU2oKMmH2jWZFeansd4Z6hku7k1Dfct",
    "9VW8",
    "0xd239",
    "WIF/SOL",
    "WIF/WETH",
)

DEX_PAIR_URL = "https://dexscreener.com/{chain}/{pair}"
EXPLORER = {
    "solana": "https://solscan.io/account/{addr}",
    "ethereum": "https://etherscan.io/address/{addr}",
    "base": "https://basescan.org/address/{addr}",
    "bsc": "https://bscscan.com/address/{addr}",
    "arbitrum": "https://arbiscan.io/address/{addr}",
    "polygon": "https://polygonscan.com/address/{addr}",
    "robinhood": "https://explorer.robinhood.com/address/{addr}",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def explorer_url(chain: str | None, addr: str | None) -> str:
    if not addr:
        return ""
    ch = str(chain or "").lower()
    tmpl = EXPLORER.get(ch, "https://dexscreener.com/{chain}/{pair}")
    if "{addr}" in tmpl:
        return tmpl.format(addr=addr)
    return DEX_PAIR_URL.format(chain=ch or "unknown", pair=addr)


def provider_url(chain: str | None, pair: str | None, existing: str | None = None) -> str:
    if existing:
        return str(existing)
    if chain and pair:
        return DEX_PAIR_URL.format(chain=str(chain).lower(), pair=pair)
    return ""


def parse_payload(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict):
        if "pairAddress" in obj or "baseToken" in obj:
            return obj
        # DexScreener search wrapper
        pairs = obj.get("pairs")
        if isinstance(pairs, list) and pairs:
            return pairs[0] if isinstance(pairs[0], dict) else None
        pair = obj.get("pair")
        if isinstance(pair, dict):
            return pair
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return obj if isinstance(obj, dict) else None


def extract_pair_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    base = payload.get("baseToken") or {}
    quote = payload.get("quoteToken") or {}
    liq = payload.get("liquidity") or {}
    vol = payload.get("volume") or {}
    txns = payload.get("txns") or {}
    return {
        "chainId": payload.get("chainId"),
        "dexId": payload.get("dexId"),
        "pairAddress": payload.get("pairAddress"),
        "baseToken.address": base.get("address") if isinstance(base, dict) else None,
        "baseToken.symbol": base.get("symbol") if isinstance(base, dict) else None,
        "quoteToken.address": quote.get("address") if isinstance(quote, dict) else None,
        "quoteToken.symbol": quote.get("symbol") if isinstance(quote, dict) else None,
        "url": payload.get("url"),
        "priceUsd": payload.get("priceUsd"),
        "liquidity.usd": liq.get("usd") if isinstance(liq, dict) else liq,
        "volume": vol,
        "txns": txns,
        "pairCreatedAt": payload.get("pairCreatedAt"),
    }


def classify_identity(
    *,
    address: str,
    chain: str | None,
    pair_address: str | None,
    token_mint: str | None,
    token_contract: str | None,
    address_role: str | None,
    payload: dict[str, Any] | None,
) -> str:
    addr = (address or "").strip()
    if not addr:
        return "not_found_in_provider"
    ch = (chain or "").lower()
    pair = (pair_address or "").strip()
    mint = (token_mint or "").strip()
    contract = (token_contract or "").strip()
    role = (address_role or "").strip()
    fields = extract_pair_fields(payload) if payload else {}
    base = str(fields.get("baseToken.address") or "").strip()
    dex = str(fields.get("dexId") or "").lower()

    # Match by concrete field equality first (never inherit row pair-role onto mint).
    if pair and addr.lower() == pair.lower():
        if "pump" in dex:
            return "market_account"
        if ch in ("solana", "sol") or (not addr.startswith("0x") and 32 <= len(addr) <= 44):
            return "pool_address"
        return "pair_contract"
    if base and addr.lower() == base.lower():
        return "token_mint" if ch in ("solana", "sol") else "token_contract"
    if mint and addr == mint:
        return "token_mint"
    if contract and addr.lower() == contract.lower():
        return "token_contract" if ch not in ("solana", "sol") else "token_mint"
    if role in (
        "token_mint",
        "token_contract",
        "pair_contract",
        "pool_address",
        "market_account",
        "provider_pair_id",
        "ambiguous",
    ):
        if role == "pool_address" and pair and addr == pair:
            return "pool_address"
        if role == "pair_contract" and pair and addr.lower() == pair.lower():
            return "pair_contract"
        if role == "token_mint" and mint and addr == mint:
            return "token_mint"
        if role == "token_contract" and contract and addr.lower() == contract.lower():
            return "token_contract"
    if payload is None and not pair and not mint and not contract:
        return "not_found_in_provider"
    if payload is None:
        return "not_found_in_provider"
    return "ambiguous"


def freshness_gate_status(age: float | None, tradability: str | None) -> str:
    if age is None:
        return "unknown_no_timestamp"
    if age <= 120:
        return "pass_fresh"
    if age <= 900:
        return "fail_soft_stale"
    return "fail_hard_stale"


def refresh_class(
    *,
    price_same: bool,
    liq_same: bool,
    provider_ts_same: bool,
    built_at_changed: bool,
    age_seconds: float | None,
) -> str:
    if age_seconds is not None and age_seconds > 86400 * 2:
        # very old provider snapshot
        if price_same and liq_same and provider_ts_same:
            return "stale_historical_row"
    if (not price_same) or (not liq_same) or (not provider_ts_same):
        return "genuinely_updated"
    if built_at_changed and price_same and liq_same and provider_ts_same:
        return "locally_refreshed_only"
    if price_same and liq_same and provider_ts_same:
        return "unchanged_provider_snapshot"
    return "unknown"


def load_coin_extras(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(coins)").fetchall()}
    want = [
        "pair_address",
        "token_address",
        "provider",
        "provider_url",
        "last_seen_at",
        "latest_price",
        "latest_liquidity",
        "chain",
        "symbol",
    ]
    select = ", ".join(c for c in want if c in cols)
    out: dict[str, dict[str, Any]] = {}
    for row in conn.execute(f"SELECT {select} FROM coins").fetchall():
        d = dict(zip([c for c in want if c in cols], row))
        pa = str(d.get("pair_address") or "")
        if pa:
            out[pa] = d
    return out


def latest_raw_payload(
    conn: sqlite3.Connection, pair_address: str, symbol: str | None = None
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, timestamp, provider, source_type, query, chain, pair_address, symbol,
               payload_json_or_text, payload_hash
        FROM raw_provider_payloads
        WHERE pair_address = ?
        ORDER BY timestamp DESC LIMIT 1
        """,
        (pair_address,),
    ).fetchone()
    if not row and symbol:
        row = conn.execute(
            """
            SELECT id, timestamp, provider, source_type, query, chain, pair_address, symbol,
                   payload_json_or_text, payload_hash
            FROM raw_provider_payloads
            WHERE symbol = ? OR payload_json_or_text LIKE ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (symbol, f"%{pair_address}%"),
        ).fetchone()
    if not row:
        # fuzzy: payload contains address
        row = conn.execute(
            """
            SELECT id, timestamp, provider, source_type, query, chain, pair_address, symbol,
                   payload_json_or_text, payload_hash
            FROM raw_provider_payloads
            WHERE payload_json_or_text LIKE ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (f"%{pair_address}%",),
        ).fetchone()
    if not row:
        return None
    keys = [
        "id",
        "timestamp",
        "provider",
        "source_type",
        "query",
        "chain",
        "pair_address",
        "symbol",
        "payload_json_or_text",
        "payload_hash",
    ]
    return dict(zip(keys, row))


def build_truth_rows(
    live: dict[str, Any], coin_map: dict[str, dict[str, Any]], conn: sqlite3.Connection
) -> list[dict[str, Any]]:
    rows = live.get("rows") or []
    # repetition counts
    by_pair = Counter(str(r.get("pair_address") or "") for r in rows)
    by_sym_chain = Counter(
        (str(r.get("symbol") or ""), str(r.get("chain") or "").lower()) for r in rows
    )
    by_token = Counter(
        str(r.get("token_mint_address") or r.get("token_contract_address") or "") for r in rows
    )

    out = []
    for r in rows:
        pair = str(r.get("pair_address") or r.get("pair") or "")
        chain = r.get("chain")
        coin = coin_map.get(pair, {})
        token_mint = r.get("token_mint_address") or coin.get("token_address")
        token_contract = r.get("token_contract_address")
        # For EVM, mint/contract often same field in DB token_address
        if not token_contract and chain and str(chain).lower() not in ("solana", "sol"):
            token_contract = coin.get("token_address") or token_mint
        address_role = r.get("address_role") or ""
        price_age = r.get("price_age_seconds")
        try:
            price_age_f = float(price_age) if price_age is not None else None
        except (TypeError, ValueError):
            price_age_f = None

        # liquidity age: use last_seen / snapshot time as proxy (no separate liq ts)
        liq_age = price_age_f
        provider_last = coin.get("last_seen_at") or r.get("last_seen_at") or r.get("time")
        prov_url = provider_url(chain, pair, coin.get("provider_url") or r.get("provider_url"))
        expl = explorer_url(chain, pair)

        appears_token = bool(token_mint or token_contract) and address_role in (
            "token_mint",
            "token_contract",
        )
        appears_pair = address_role in ("pair_contract", "provider_pair_id") or bool(pair)
        appears_pool = address_role in ("pool_address", "market_account")

        sym = str(r.get("symbol") or "")
        row_rep = max(
            by_pair.get(pair, 1),
            by_sym_chain.get((sym, str(chain or "").lower()), 1),
        )
        token_key = str(token_mint or token_contract or "")
        if token_key and by_token.get(token_key, 0) > row_rep:
            row_rep = by_token[token_key]

        evidence = (
            f"source={r.get('source_provider') or coin.get('provider') or 'unknown'}; "
            f"role={address_role or 'unknown'}; "
            f"pair={pair[:12]}…; "
            f"token={(token_mint or token_contract or '')[:12]}; "
            f"tradability={r.get('tradability_status')}; "
            f"price_age_s={price_age_f}"
        )

        # pool_address: for Solana pool role, pair IS the pool
        pool_address = pair if appears_pool or address_role == "pool_address" else (
            pair if str(chain or "").lower() in ("solana", "sol") else ""
        )
        if not pool_address and address_role == "pool_address":
            pool_address = pair

        out.append(
            {
                "displayed_symbol": sym,
                "chain": chain,
                "pair_address": pair,
                "pool_address": pool_address or "",
                "token_mint_address": token_mint or "",
                "token_contract_address": token_contract or "",
                "provider_pair_id": pair,  # DexScreener keys by pairAddress
                "address_role": address_role,
                "source_provider": r.get("source_provider") or coin.get("provider") or "dexscreener",
                "provider_url": prov_url,
                "explorer_url": expl,
                "price_usd": r.get("price"),
                "price_updated_at": r.get("price_timestamp") or r.get("time"),
                "liquidity_usd": r.get("liquidity"),
                "liquidity_updated_at": r.get("price_timestamp") or r.get("time") or provider_last,
                "provider_last_seen_at": provider_last,
                "price_age_seconds": price_age_f,
                "liquidity_age_seconds": liq_age,
                "tradability_status": r.get("tradability_status"),
                "freshness_gate_status": freshness_gate_status(
                    price_age_f, r.get("tradability_status")
                ),
                "row_repeated_count": row_rep,
                "appears_token_level": appears_token,
                "appears_pair_level": appears_pair and not appears_token,
                "appears_pool_level": appears_pool,
                "evidence_summary": evidence,
                "row_key": r.get("row_key"),
            }
        )
    return out


def duplicate_audit(truth: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> tuple[dict, list]:
    total = len(truth)
    symbols = [t["displayed_symbol"] for t in truth]
    pairs = [t["pair_address"] for t in truth if t["pair_address"]]
    tokens = [
        t["token_mint_address"] or t["token_contract_address"]
        for t in truth
        if (t["token_mint_address"] or t["token_contract_address"])
    ]
    provider_ids = [t["provider_pair_id"] for t in truth if t["provider_pair_id"]]

    sym_counts = Counter(symbols)
    pair_counts = Counter(pairs)
    token_counts = Counter(tokens)
    sym_chain = Counter(
        (t["displayed_symbol"], str(t["chain"] or "").lower()) for t in truth
    )

    dup_sym_chain = {k: v for k, v in sym_chain.items() if v > 1}
    dup_pairs = {k: v for k, v in pair_counts.items() if v > 1}
    dup_tokens = {k: v for k, v in token_counts.items() if v > 1 and k}

    summary = {
        "total_live_market_rows": total,
        "unique_displayed_symbols": len(set(symbols)),
        "unique_pair_addresses": len(set(pairs)),
        "unique_token_addresses_mints": len(set(tokens)),
        "unique_provider_pair_ids": len(set(provider_ids)),
        "top_repeated_symbols": sym_counts.most_common(15),
        "top_repeated_pair_addresses": pair_counts.most_common(15),
        "duplicate_rows_by_symbol_chain_count": len(dup_sym_chain),
        "duplicate_rows_by_pair_address_count": len(dup_pairs),
        "duplicate_rows_by_token_identity_count": len(dup_tokens),
        "duplicate_symbol_chain_detail": [
            {"symbol": k[0], "chain": k[1], "count": v} for k, v in sorted(dup_sym_chain.items(), key=lambda x: -x[1])
        ],
        "duplicate_pair_detail": [
            {"pair_address": k, "count": v} for k, v in sorted(dup_pairs.items(), key=lambda x: -x[1])
        ],
        "duplicate_token_detail": [
            {"token_identity": k, "count": v} for k, v in sorted(dup_tokens.items(), key=lambda x: -x[1])[:30]
        ],
        # same token mint appearing across multiple pairs (universe dilution)
        "tokens_with_multiple_pair_rows": sum(1 for v in token_counts.values() if v > 1),
    }

    # flat CSV rows for audit
    csv_rows = []
    for (sym, ch), cnt in sorted(sym_chain.items(), key=lambda x: -x[1]):
        csv_rows.append(
            {
                "audit_dimension": "symbol_chain",
                "key": f"{sym}|{ch}",
                "displayed_symbol": sym,
                "chain": ch,
                "pair_address": "",
                "token_identity": "",
                "count": cnt,
                "is_duplicate": cnt > 1,
            }
        )
    for pa, cnt in pair_counts.most_common():
        csv_rows.append(
            {
                "audit_dimension": "pair_address",
                "key": pa,
                "displayed_symbol": "",
                "chain": "",
                "pair_address": pa,
                "token_identity": "",
                "count": cnt,
                "is_duplicate": cnt > 1,
            }
        )
    for tok, cnt in token_counts.most_common(50):
        csv_rows.append(
            {
                "audit_dimension": "token_identity",
                "key": tok,
                "displayed_symbol": "",
                "chain": "",
                "pair_address": "",
                "token_identity": tok,
                "count": cnt,
                "is_duplicate": cnt > 1,
            }
        )
    # also per-row repetition for truth join
    for t in truth:
        csv_rows.append(
            {
                "audit_dimension": "row",
                "key": t.get("row_key") or t["pair_address"],
                "displayed_symbol": t["displayed_symbol"],
                "chain": t["chain"],
                "pair_address": t["pair_address"],
                "token_identity": t["token_mint_address"] or t["token_contract_address"],
                "count": t["row_repeated_count"],
                "is_duplicate": t["row_repeated_count"] > 1,
            }
        )
    return summary, csv_rows


def poll_refresh(n_intervals: int = 3, wait_s: float = 10.0) -> tuple[list[dict], list[dict]]:
    from app.ae13b_product.live_market import build_live_market

    snapshots = []
    for i in range(n_intervals + 1):  # 4 builds = 3 intervals between
        snap = build_live_market(limit=50)
        snapshots.append(
            {
                "poll_index": i,
                "built_at_utc": snap.get("built_at_utc"),
                "latest_market_update": snap.get("latest_market_update"),
                "freshness": snap.get("freshness"),
                "row_count": len(snap.get("rows") or []),
                "rows": snap.get("rows") or [],
            }
        )
        if i < n_intervals:
            time.sleep(wait_s)
    return snapshots, snapshots[0]["rows"]  # type: ignore


def analyze_refresh(snapshots: list[dict]) -> list[dict[str, Any]]:
    # index rows by row_key across polls
    keys = set()
    for s in snapshots:
        for r in s["rows"]:
            keys.add(str(r.get("row_key") or r.get("pair_address")))

    out = []
    for key in sorted(keys):
        series = []
        for s in snapshots:
            match = None
            for r in s["rows"]:
                rk = str(r.get("row_key") or r.get("pair_address"))
                if rk == key:
                    match = r
                    break
            series.append((s, match))

        prices = [m.get("price") if m else None for _, m in series]
        liqs = [m.get("liquidity") if m else None for _, m in series]
        ts_list = [
            (m.get("price_timestamp") or m.get("time") or m.get("last_seen_at")) if m else None
            for _, m in series
        ]
        built = [s.get("built_at_utc") for s, _ in series]
        ages = []
        for _, m in series:
            if not m:
                ages.append(None)
                continue
            try:
                ages.append(float(m.get("price_age_seconds")) if m.get("price_age_seconds") is not None else None)
            except (TypeError, ValueError):
                ages.append(None)

        present = [m is not None for _, m in series]
        price_same = len(set(str(p) for p in prices if p is not None)) <= 1
        liq_same = len(set(str(l) for l in liqs if l is not None)) <= 1
        ts_same = len(set(str(t) for t in ts_list if t is not None)) <= 1
        built_changed = len(set(str(b) for b in built if b)) > 1
        row_count_same = len(set(s["row_count"] for s, _ in series)) == 1

        first = series[0][1] or {}
        cls = refresh_class(
            price_same=price_same,
            liq_same=liq_same,
            provider_ts_same=ts_same,
            built_at_changed=built_changed,
            age_seconds=ages[0] if ages else None,
        )
        # if age increases ~wait time but price/ts unchanged -> locally_refreshed_only
        if cls == "unchanged_provider_snapshot" and built_changed:
            cls = "locally_refreshed_only"
        if ages and ages[0] is not None and ages[-1] is not None:
            if price_same and liq_same and ts_same and (ages[-1] - ages[0]) > 5:
                cls = "locally_refreshed_only"

        out.append(
            {
                "row_key": key,
                "displayed_symbol": first.get("symbol"),
                "pair_address": first.get("pair_address") or first.get("pair"),
                "chain": first.get("chain"),
                "poll_count": len(snapshots),
                "present_all_polls": all(present),
                "same_row_count_global": row_count_same,
                "global_row_counts": "|".join(str(s["row_count"]) for s, _ in series),
                "prices": "|".join("" if p is None else str(p) for p in prices),
                "same_price": price_same,
                "liquidities": "|".join("" if l is None else str(l) for l in liqs),
                "same_liquidity": liq_same,
                "provider_timestamps": "|".join("" if t is None else str(t) for t in ts_list),
                "same_provider_timestamp": ts_same,
                "built_at_utc_series": "|".join("" if b is None else str(b) for b in built),
                "built_at_changed_only_local": built_changed and price_same and liq_same and ts_same,
                "price_age_seconds_series": "|".join("" if a is None else str(round(a, 1)) for a in ages),
                "refresh_classification": cls,
                "only_local_refresh_time_changes": built_changed and price_same and liq_same and ts_same,
            }
        )
    return out


def provider_trace(
    conn: sqlite3.Connection,
    live_rows: list[dict[str, Any]],
    coin_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # select samples
    selected: list[dict[str, Any]] = []
    seen = set()

    def add_row(r: dict[str, Any], reason: str) -> None:
        key = str(r.get("row_key") or r.get("pair_address"))
        if key in seen:
            return
        seen.add(key)
        selected.append({**r, "_sample_reason": reason})

    for r in live_rows:
        blob = json.dumps(r, default=str)
        pair = str(r.get("pair_address") or "")
        sym = str(r.get("symbol") or "")
        for needle in SAMPLE_NEEDLES:
            if needle in blob or needle in pair or needle in sym:
                add_row(r, f"needle:{needle}")
                break
        if "WIF" in sym.upper() and ("SOL" in sym.upper() or "WETH" in sym.upper() or "ETH" in sym.upper()):
            add_row(r, f"wif_pair:{sym}")

    # ensure DDk1 specifically searched in DB even if not in current top-50
    ddk = "DDk1QmkbZBtTSpU2oKMmH2jWZFeansd4Z6hku7k1Dfct"
    if not any(ddk in str(r.get("pair_address") or "") for r in selected):
        coin = coin_map.get(ddk)
        if coin:
            add_row(
                {
                    "symbol": coin.get("symbol"),
                    "chain": coin.get("chain"),
                    "pair_address": ddk,
                    "pair": ddk,
                    "token_mint_address": coin.get("token_address"),
                    "address_role": "pool_address",
                    "price": coin.get("latest_price"),
                    "liquidity": coin.get("latest_liquidity"),
                    "source_provider": coin.get("provider"),
                    "row_key": f"{str(coin.get('chain') or '').lower()}|pair|{ddk}",
                },
                "forced_ddk1_from_coins",
            )
        else:
            # still try payload search
            add_row(
                {
                    "symbol": "WIF/SOL?",
                    "chain": "solana",
                    "pair_address": ddk,
                    "pair": ddk,
                    "row_key": f"solana|pair|{ddk}",
                },
                "forced_ddk1_address_only",
            )

    # also sample 0xd239 and 9VW8 from DB/coins if missing
    for prefix, label in (("9VW8", "9VW8"), ("0xd239", "0xd239")):
        if any(str(r.get("pair_address") or "").lower().startswith(prefix.lower()) for r in selected):
            continue
        hit = None
        for pa, c in coin_map.items():
            if pa.lower().startswith(prefix.lower()) or prefix.lower() in pa.lower():
                hit = (pa, c)
                break
        if hit:
            pa, c = hit
            add_row(
                {
                    "symbol": c.get("symbol"),
                    "chain": c.get("chain"),
                    "pair_address": pa,
                    "pair": pa,
                    "token_mint_address": c.get("token_address"),
                    "price": c.get("latest_price"),
                    "liquidity": c.get("latest_liquidity"),
                    "source_provider": c.get("provider"),
                    "row_key": f"{str(c.get('chain') or '').lower()}|pair|{pa}",
                },
                f"forced_{label}_from_coins",
            )

    traces = []
    for r in selected:
        pair = str(r.get("pair_address") or "")
        raw = latest_raw_payload(conn, pair, r.get("symbol"))
        payload_obj = parse_payload(raw["payload_json_or_text"]) if raw else None
        # If wrapper with many pairs, find matching pair
        if raw and payload_obj and "pairAddress" not in (payload_obj or {}):
            try:
                full = json.loads(raw["payload_json_or_text"])
            except Exception:
                full = None
            if isinstance(full, dict) and isinstance(full.get("pairs"), list):
                for p in full["pairs"]:
                    if isinstance(p, dict) and str(p.get("pairAddress") or "").lower() == pair.lower():
                        payload_obj = p
                        break
            elif isinstance(full, list):
                for p in full:
                    if isinstance(p, dict) and str(p.get("pairAddress") or "").lower() == pair.lower():
                        payload_obj = p
                        break

        fields = extract_pair_fields(payload_obj)
        phash = (raw or {}).get("payload_hash")
        if not phash and raw:
            phash = hashlib.sha256((raw.get("payload_json_or_text") or "").encode("utf-8")).hexdigest()

        # truncate huge payload for artifact but keep original object when reasonable
        original = payload_obj
        if original and len(json.dumps(original, default=str)) > 50000:
            original = {**fields, "_truncated": True}

        traces.append(
            {
                "sample_reason": r.get("_sample_reason"),
                "displayed_symbol": r.get("symbol"),
                "displayed_pair_address": pair,
                "chain": r.get("chain") or fields.get("chainId"),
                "raw_payload_found": raw is not None,
                "raw_payload_meta": {
                    "id": (raw or {}).get("id"),
                    "timestamp": (raw or {}).get("timestamp"),
                    "provider": (raw or {}).get("provider"),
                    "source_type": (raw or {}).get("source_type"),
                    "query": (raw or {}).get("query"),
                    "stored_pair_address": (raw or {}).get("pair_address"),
                    "stored_symbol": (raw or {}).get("symbol"),
                    "payload_hash": phash,
                }
                if raw
                else None,
                "original_provider_response_object": original,
                "extracted": {
                    **fields,
                    "fetched_at": (raw or {}).get("timestamp"),
                    "ingested_at": (raw or {}).get("timestamp"),
                    "payload_hash": phash,
                },
                "live_row_price": r.get("price"),
                "live_row_liquidity": r.get("liquidity"),
                "live_row_address_role": r.get("address_role"),
                "db_token_address": coin_map.get(pair, {}).get("token_address"),
            }
        )
    return {
        "sampled_at": utc_now(),
        "sample_needles": list(SAMPLE_NEEDLES),
        "trace_count": len(traces),
        "traces": traces,
    }


def identity_verdict(
    traces: dict[str, Any], live_rows: list[dict[str, Any]], coin_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    addrs: list[tuple[str, dict]] = []

    for t in traces.get("traces") or []:
        pair = t.get("displayed_pair_address") or ""
        if pair:
            addrs.append((pair, t))
        ext = t.get("extracted") or {}
        base = ext.get("baseToken.address")
        if base:
            addrs.append((str(base), t))

    # also from live for WIF samples
    for r in live_rows:
        pair = str(r.get("pair_address") or "")
        if any(n.lower() in pair.lower() or n in str(r.get("symbol") or "") for n in SAMPLE_NEEDLES):
            addrs.append((pair, {"live": r, "extracted": {}, "raw_payload_found": False}))

    seen = set()
    for addr, ctx in addrs:
        if not addr or addr in seen:
            continue
        seen.add(addr)
        live = ctx.get("live") or {}
        extracted = ctx.get("extracted") or {}
        chain = live.get("chain") or ctx.get("chain") or extracted.get("chainId")
        pair = live.get("pair_address") or ctx.get("displayed_pair_address") or extracted.get("pairAddress")
        token_mint = (
            live.get("token_mint_address")
            or coin_map.get(str(pair or ""), {}).get("token_address")
            or ctx.get("db_token_address")
            or extracted.get("baseToken.address")
        )
        token_contract = live.get("token_contract_address")
        role = live.get("address_role") or ctx.get("live_row_address_role")
        payload = ctx.get("original_provider_response_object")
        if isinstance(payload, dict) and payload.get("_truncated"):
            payload = {k: extracted.get(k) for k in extracted}

        verdict = classify_identity(
            address=addr,
            chain=chain,
            pair_address=str(pair) if pair else None,
            token_mint=str(token_mint) if token_mint else None,
            token_contract=str(token_contract) if token_contract else None,
            address_role=role,
            payload=payload if isinstance(payload, dict) else None,
        )
        # DDk1 special note
        note = ""
        dex = str(extracted.get("dexId") or "")
        if addr.startswith("DDk1"):
            note = (
                f"Solana pool/pair for {extracted.get('baseToken.symbol') or live.get('symbol') or 'WIF'}/"
                f"{extracted.get('quoteToken.symbol') or 'SOL'}; dexId={dex or 'unknown'}. "
                "Not a token mint. May appear as Pump.fun AMM / WSOL-WIF market account on explorers."
            )
        elif verdict in ("pool_address", "pair_contract", "market_account"):
            note = "Displayed/row identity is pair/pool level, not token contract."
        elif verdict in ("token_mint", "token_contract"):
            note = "Address matches base token identity in provider/DB."
        elif verdict == "not_found_in_provider":
            note = "No matching raw_provider_payloads row found for this address."

        rows.append(
            {
                "sampled_address": addr,
                "chain": chain,
                "displayed_symbol": live.get("symbol") or ctx.get("displayed_symbol") or extracted.get("baseToken.symbol"),
                "pair_address": pair,
                "token_mint_address": token_mint or "",
                "token_contract_address": token_contract or "",
                "dexId": dex,
                "provider_url": extracted.get("url") or provider_url(chain, pair),
                "identity_verdict": verdict,
                "do_not_call_contract_address": verdict
                in ("pool_address", "pair_contract", "market_account", "provider_pair_id"),
                "evidence_note": note,
                "raw_payload_found": bool(ctx.get("raw_payload_found")),
            }
        )
    return rows


def training_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    rows_out: list[dict[str, Any]] = []
    overall = {
        "dataset_root": str(TRAIN_DIR),
        "clean_root": str(CLEAN_DIR),
        "files": [],
        "classifications": [],
    }

    if not TRAIN_DIR.exists():
        overall["error"] = "direct_target dataset root missing"
        return rows_out, overall

    # Prefer RAW_ALL SL080 policy files as canonical prior train set; also summarize all
    csvs = sorted(TRAIN_DIR.glob("*DIRECT_TARGET_v1.csv"))
    # Focus files commonly used by RF/XGB/TABICL
    preferred = [
        p
        for p in csvs
        if "RAW_ALL_VERIFIED" in p.name and "SL080" in p.name
    ]
    scan_files = preferred if preferred else csvs[:10]

    for path in scan_files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            rows_out.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "error": str(e),
                    "classification": "provenance_incomplete",
                }
            )
            continue

        n = len(df)
        pair_col = "pair_address" if "pair_address" in df.columns else None
        token_col = None
        for c in ("token_address", "token_mint_address", "base_token_address", "contract_address"):
            if c in df.columns:
                token_col = c
                break
        unique_pairs = int(df[pair_col].nunique()) if pair_col else None
        unique_tokens = int(df[token_col].astype(str).nunique()) if token_col else None

        top_pair_share = None
        top10 = []
        if pair_col and n:
            vc = df[pair_col].astype(str).value_counts()
            top_pair_share = float(vc.iloc[0] / n) if len(vc) else None
            # top pair/symbol combos
            if "symbol" in df.columns:
                combo = (
                    df.assign(_p=df[pair_col].astype(str), _s=df["symbol"].astype(str))
                    .groupby(["_s", "_p"], sort=False)
                    .size()
                    .sort_values(ascending=False)
                    .head(10)
                )
                top10 = [
                    {"symbol": a, "pair_address": b, "count": int(c), "share": round(c / n, 4)}
                    for (a, b), c in combo.items()
                ]
            else:
                top10 = [
                    {"symbol": "", "pair_address": p, "count": int(c), "share": round(c / n, 4)}
                    for p, c in vc.head(10).items()
                ]

        # event vs snapshot heuristics
        has_event_ts = any(
            c in df.columns for c in ("event_timestamp", "timestamp", "ts", "signal_timestamp")
        )
        has_price_ts = any(
            c in df.columns for c in ("price_timestamp", "snapshot_timestamp", "event_timestamp")
        )
        # duplicate grouping?
        has_dup_group = any(
            c in df.columns for c in ("event_id", "group_id", "dedupe_key")
        )
        target_cols = [c for c in df.columns if "target_net_profitable" in c]
        sim_cols = [c for c in df.columns if c.startswith("sim_")]

        # classify
        classes = [
            "pair_level_validated" if pair_col and unique_pairs else "provenance_incomplete",
            "snapshot_level_with_duplicates" if (top_pair_share or 0) > 0.05 or (unique_pairs and n / max(unique_pairs, 1) > 5) else "pair_level_validated",
            "freshness_not_guaranteed",
            "unsuitable_for_profitability_claims_until_rebuilt",
        ]
        if token_col and unique_tokens:
            classes.insert(0, "token_level_validated")
        else:
            classes.append("provenance_incomplete")

        # row nature
        if has_event_ts and "event_timestamp" in df.columns:
            row_nature = "event_level_with_pair_identity"
        elif pair_col:
            row_nature = "snapshot_or_event_rows_keyed_by_pair"
        else:
            row_nature = "unknown_repeated_provider_rows_possible"

        # targets from direct_target_builder walk future market_snapshots
        target_origin = (
            "historical_market_snapshots_exit_simulation"
            if sim_cols or target_cols
            else "unknown"
        )

        rec = {
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "row_count": n,
            "unique_pair_count": unique_pairs,
            "unique_token_identity_count": unique_tokens,
            "token_identity_column": token_col or "",
            "top_pair_share": round(top_pair_share, 6) if top_pair_share is not None else None,
            "top_10_repeated_pair_symbol_json": json.dumps(top10),
            "row_nature": row_nature,
            "price_liquidity_timestamps_present": has_price_ts,
            "event_timestamp_present": has_event_ts,
            "duplicate_event_grouping_applied": has_dup_group,
            "target_columns": "|".join(target_cols),
            "target_outcome_origin": target_origin,
            "target_from_fresh_observed_prices": False,  # builder uses historical snapshots in DB
            "target_from_historical_or_fallback_snapshots": True,
            "classification": "|".join(dict.fromkeys(classes)),
            "notes": (
                "E3 direct-exit targets simulated from historical market_snapshots by pair_address; "
                "not live token-universe validated; pair-level identity dominates."
            ),
        }
        rows_out.append(rec)
        overall["files"].append(rec)

    # clean input summary if present
    clean_summary = CLEAN_DIR / "clean_model_input_summary.json"
    if clean_summary.exists():
        try:
            overall["clean_model_input_summary"] = json.loads(clean_summary.read_text(encoding="utf-8"))
        except Exception as e:
            overall["clean_model_input_summary_error"] = str(e)

    overall["classifications"] = [
        "pair_level_validated",
        "snapshot_level_with_duplicates",
        "freshness_not_guaranteed",
        "provenance_incomplete",  # token identity often missing/aliased
        "unsuitable_for_profitability_claims_until_rebuilt",
    ]
    return rows_out, overall


def decide(
    *,
    truth: list[dict],
    dup_summary: dict,
    refresh_rows: list[dict],
    traces: dict,
    training_overall: dict,
    identity_rows: list[dict],
) -> dict[str, Any]:
    classifications = []
    reasons = []

    ages = [t.get("price_age_seconds") for t in truth if t.get("price_age_seconds") is not None]
    fresh_count = sum(1 for a in ages if a is not None and a <= 120)
    stale_count = sum(1 for a in ages if a is not None and a > 120)
    tradable_now = sum(1 for t in truth if t.get("tradability_status") == "tradable_now")

    refresh_classes = Counter(r.get("refresh_classification") for r in refresh_rows)
    local_only = refresh_classes.get("locally_refreshed_only", 0)
    unchanged = refresh_classes.get("unchanged_provider_snapshot", 0)
    stale_hist = refresh_classes.get("stale_historical_row", 0)
    genuine = refresh_classes.get("genuinely_updated", 0)

    pool_roles = sum(
        1
        for t in truth
        if t.get("address_role") in ("pool_address", "pair_contract", "market_account", "provider_pair_id")
    )
    token_roles = sum(
        1 for t in truth if t.get("address_role") in ("token_mint", "token_contract")
    )

    payload_missing = sum(1 for t in (traces.get("traces") or []) if not t.get("raw_payload_found"))
    ddk_verdicts = [
        r for r in identity_rows if str(r.get("sampled_address") or "").startswith("DDk1")
    ]

    # Primary blocks
    if genuine == 0 and (local_only + unchanged + stale_hist) >= max(1, len(refresh_rows) * 0.8):
        classifications.append("AE13J_BLOCKED_LIVE_MARKET_NOT_ACTUALLY_LIVE")
        reasons.append(
            f"Refresh audit: {genuine} genuinely_updated vs {local_only} locally_refreshed_only, "
            f"{unchanged} unchanged, {stale_hist} stale_historical across {len(refresh_rows)} rows; "
            f"fresh_rows={fresh_count}/{len(truth)}, tradable_now={tradable_now}."
        )

    if pool_roles >= max(1, int(0.5 * len(truth))) and token_roles == 0:
        classifications.append("AE13J_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED")
        reasons.append(
            f"Identity: {pool_roles}/{len(truth)} rows are pair/pool roles; {token_roles} token-level roles. "
            "UI rows are provider pair/pool snapshots, not a validated token mint universe."
        )

    if ddk_verdicts and ddk_verdicts[0].get("identity_verdict") in (
        "pool_address",
        "market_account",
        "pair_contract",
        "provider_pair_id",
    ):
        if "AE13J_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED" not in classifications:
            classifications.append("AE13J_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED")
        reasons.append(
            "DDk1Q… resolves as Solana WIF/SOL pool/pair (not token mint), confirming pair/pool conflation risk."
        )

    train_classes = training_overall.get("classifications") or []
    if "provenance_incomplete" in train_classes or "unsuitable_for_profitability_claims_until_rebuilt" in train_classes:
        classifications.append("AE13J_BLOCKED_TRAINING_DATA_PROVENANCE_INCOMPLETE")
        reasons.append(
            "Training data is pair-keyed historical snapshot/event rows with exit sims from DB snapshots; "
            "token-level live universe + freshness not guaranteed."
        )

    if payload_missing and payload_missing == len(traces.get("traces") or []):
        classifications.append("AE13J_BLOCKED_PROVIDER_PAYLOAD_TRACE_MISSING")
        reasons.append("No raw provider payloads found for sampled rows.")
    elif payload_missing:
        reasons.append(f"Partial payload trace: {payload_missing} sampled rows missing raw payloads.")

    # Safety: do not claim live tradable
    if tradable_now == 0 and len(truth) > 0:
        classifications.append("AE13J_BLOCKED_SAFETY_RISK")
        reasons.append("Zero tradable_now rows in current Live Market feed — unsafe to treat as live tradable universe.")

    # Pass options only if not blocked
    blocked = [c for c in classifications if c.startswith("AE13J_BLOCKED_")]
    if not blocked:
        if stale_count or pool_roles:
            classifications = ["AE13J_PASS_WITH_PROVENANCE_LIMITATIONS"]
        else:
            classifications = ["AE13J_DATA_LAYER_VALIDATED_FOR_AE14"]
    else:
        # keep blocked set; primary is first by severity order
        order = [
            "AE13J_BLOCKED_SAFETY_RISK",
            "AE13J_BLOCKED_LIVE_MARKET_NOT_ACTUALLY_LIVE",
            "AE13J_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED",
            "AE13J_BLOCKED_TRAINING_DATA_PROVENANCE_INCOMPLETE",
            "AE13J_BLOCKED_PROVIDER_PAYLOAD_TRACE_MISSING",
        ]
        classifications = sorted(set(blocked), key=lambda x: order.index(x) if x in order else 99)

    primary = classifications[0]
    ui_label = "Provider Pair Feed"
    if primary == "AE13J_DATA_LAYER_VALIDATED_FOR_AE14":
        ui_label = "Live Market"  # only if validated
    elif fresh_count == 0:
        ui_label = "Market Snapshot Feed"
    else:
        ui_label = "Provider Pair Feed"

    return {
        "audit_id": "AE13J",
        "timestamp_utc": utc_now(),
        "artifact_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "primary_classification": primary,
        "all_classifications": classifications,
        "pass": primary in (
            "AE13J_DATA_LAYER_VALIDATED_FOR_AE14",
            "AE13J_PASS_WITH_PROVENANCE_LIMITATIONS",
        ),
        "reasons": reasons,
        "metrics": {
            "live_rows": len(truth),
            "fresh_rows_le_120s": fresh_count,
            "stale_rows_gt_120s": stale_count,
            "tradable_now": tradable_now,
            "pool_or_pair_role_rows": pool_roles,
            "token_role_rows": token_roles,
            "unique_symbols": dup_summary.get("unique_displayed_symbols"),
            "unique_pairs": dup_summary.get("unique_pair_addresses"),
            "unique_tokens": dup_summary.get("unique_token_addresses_mints"),
            "refresh_class_counts": dict(refresh_classes),
            "payload_traces": traces.get("trace_count"),
            "payload_missing": payload_missing,
        },
        "ui_truth_label": ui_label,
        "ae14_blocked": True if blocked else False,
        "do_not_claim_profitability": True,
        "do_not_enable_live_trading": True,
    }


def apply_ui_label_recommendation(label: str) -> dict[str, Any]:
    """Return recommendation object; caller also patches files."""
    return {
        "previous_label": "Live Market",
        "recommended_label": label,
        "applied_label": label,
        "rationale": (
            "Feed rows are DexScreener pair/pool snapshots keyed by pair_address, "
            "not a verified live tradable token-mint universe with guaranteed fresh prices."
        ),
        "visible_copy": (
            "Rows may represent pools/pairs, not token contracts. "
            "Trading requires fresh price, fresh liquidity, and validated tradability."
        ),
        "files_to_update": [
            "static/index.html",
            "app/api.py",
        ],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[AE13J] artifact dir: {OUT_DIR}")

    from app.ae13b_product.live_market import build_live_market

    conn = sqlite3.connect(str(DB_PATH))
    coin_map = load_coin_extras(conn)

    print("[AE13J] polling live market (4 builds / 3 intervals, 10s)...")
    snapshots, _ = poll_refresh(n_intervals=3, wait_s=10.0)
    # use first snapshot for truth table baseline; also rebuild once more for current
    live = build_live_market(limit=50)
    write_json(OUT_DIR / "ae13j_live_market_raw_snapshot.json", live)

    truth = build_truth_rows(live, coin_map, conn)
    truth_fields = [
        "displayed_symbol",
        "chain",
        "pair_address",
        "pool_address",
        "token_mint_address",
        "token_contract_address",
        "provider_pair_id",
        "address_role",
        "source_provider",
        "provider_url",
        "explorer_url",
        "price_usd",
        "price_updated_at",
        "liquidity_usd",
        "liquidity_updated_at",
        "provider_last_seen_at",
        "price_age_seconds",
        "liquidity_age_seconds",
        "tradability_status",
        "freshness_gate_status",
        "row_repeated_count",
        "appears_token_level",
        "appears_pair_level",
        "appears_pool_level",
        "evidence_summary",
    ]
    write_csv(OUT_DIR / "ae13j_live_market_truth_table.csv", truth, truth_fields)
    print(f"[AE13J] truth rows: {len(truth)}")

    dup_summary, dup_csv = duplicate_audit(truth, live.get("rows") or [])
    write_json(OUT_DIR / "ae13j_duplicate_repetition_summary.json", dup_summary)
    write_csv(
        OUT_DIR / "ae13j_duplicate_repetition_audit.csv",
        dup_csv,
        [
            "audit_dimension",
            "key",
            "displayed_symbol",
            "chain",
            "pair_address",
            "token_identity",
            "count",
            "is_duplicate",
        ],
    )

    refresh_rows = analyze_refresh(snapshots)
    write_csv(OUT_DIR / "ae13j_refresh_reality_audit.csv", refresh_rows)
    write_json(
        OUT_DIR / "ae13j_refresh_poll_meta.json",
        {
            "interval_seconds": 10,
            "polls": len(snapshots),
            "built_ats": [s.get("built_at_utc") for s in snapshots],
            "latest_market_updates": [s.get("latest_market_update") for s in snapshots],
            "row_counts": [s.get("row_count") for s in snapshots],
            "freshness_labels": [(s.get("freshness") or {}).get("label") for s in snapshots],
        },
    )

    traces = provider_trace(conn, live.get("rows") or [], coin_map)
    write_json(OUT_DIR / "ae13j_provider_payload_trace.json", traces)

    identity_rows = identity_verdict(traces, live.get("rows") or [], coin_map)
    write_csv(OUT_DIR / "ae13j_token_pair_identity_verdict.csv", identity_rows)

    print("[AE13J] training data validity audit...")
    train_rows, train_overall = training_audit()
    write_csv(OUT_DIR / "ae13j_training_data_validity_audit.csv", train_rows)
    write_json(OUT_DIR / "ae13j_training_data_validity_summary.json", train_overall)

    gate = decide(
        truth=truth,
        dup_summary=dup_summary,
        refresh_rows=refresh_rows,
        traces=traces,
        training_overall=train_overall,
        identity_rows=identity_rows,
    )
    ui_rec = apply_ui_label_recommendation(gate["ui_truth_label"])
    ui_rec["decision_primary"] = gate["primary_classification"]
    write_json(OUT_DIR / "ae13j_ui_truth_label_recommendation.json", ui_rec)
    write_json(OUT_DIR / "ae13j_decision_gate.json", gate)

    # Human reports
    rc = Counter(r.get("refresh_classification") for r in refresh_rows)
    report = f"""# AE13J — Ground Truth Market Data Audit Report

**Timestamp (UTC):** {gate['timestamp_utc']}  
**Artifact dir:** `{gate['artifact_dir']}`  
**Primary classification:** `{gate['primary_classification']}`  
**Pass:** {gate['pass']}

## Executive verdict

This audit does **not** validate a live tradable meme-coin **token** universe.

Current `/api/ae13b/live-market` rows are **DexScreener pair/pool snapshots** stored under `coins.pair_address` (unique key), with `token_address` held separately. UI auto-refresh rebuilds the local view; provider price/liquidity timestamps often remain unchanged.

**Do not claim profitability. Do not enable live trading from this feed as-is.**

## 1. Live Market Truth Table

- Rows audited: **{len(truth)}**
- File: `ae13j_live_market_truth_table.csv`
- Fresh (≤120s): **{gate['metrics']['fresh_rows_le_120s']}**
- Stale (>120s): **{gate['metrics']['stale_rows_gt_120s']}**
- `tradable_now`: **{gate['metrics']['tradable_now']}**
- Pair/pool role rows: **{gate['metrics']['pool_or_pair_role_rows']}**
- Token role rows: **{gate['metrics']['token_role_rows']}**

## 2. Repetition / Duplicate Audit

- Total rows: **{dup_summary['total_live_market_rows']}**
- Unique symbols: **{dup_summary['unique_displayed_symbols']}**
- Unique pair addresses: **{dup_summary['unique_pair_addresses']}**
- Unique token mints/addresses: **{dup_summary['unique_token_addresses_mints']}**
- Unique provider pair ids: **{dup_summary['unique_provider_pair_ids']}**
- Duplicate symbol+chain groups: **{dup_summary['duplicate_rows_by_symbol_chain_count']}**
- Duplicate pair_address groups: **{dup_summary['duplicate_rows_by_pair_address_count']}**
- Tokens with multiple pair rows: **{dup_summary['tokens_with_multiple_pair_rows']}**

Top repeated symbols: `{dup_summary['top_repeated_symbols'][:10]}`

## 3. Refresh Reality Audit

Polled **{len(snapshots)}** builds with **10s** intervals (no trades opened).

Refresh classification counts: `{dict(rc)}`

Interpretation:
- If most rows are `locally_refreshed_only` / `unchanged_provider_snapshot` / `stale_historical_row`, the UI refresh is **local rebuild**, not proof of live price movement.
- File: `ae13j_refresh_reality_audit.csv`

## 4. Provider Payload Trace

- Traces: **{traces.get('trace_count')}**
- File: `ae13j_provider_payload_trace.json`
- Provider: DexScreener (`raw_provider_payloads`)
- Sample needles: {list(SAMPLE_NEEDLES)}

## 5. Token vs Pair Identity Verdict

- File: `ae13j_token_pair_identity_verdict.csv`
- Notable: address `DDk1QmkbZBtTSpU2oKMmH2jWZFeansd4Z6hku7k1Dfct` is treated as a **Solana pool/pair / market account** for WIF-SOL style markets — **not** a token mint. Do not label it as contract address in user-facing copy.

Identity sample:
"""
    for r in identity_rows[:12]:
        report += (
            f"\n- `{r.get('sampled_address')}` → **{r.get('identity_verdict')}** "
            f"({r.get('displayed_symbol')}, {r.get('chain')})"
        )

    report += f"""

## 6. Training Data Validity Audit

- Dataset root: `{TRAIN_DIR.relative_to(ROOT).as_posix()}`
- Files audited: **{len(train_rows)}**
- File: `ae13j_training_data_validity_audit.csv`
- Classifications: `{train_overall.get('classifications')}`

Targets (`target_net_profitable_after_exit`) are produced by historical `market_snapshots` exit simulation keyed by **pair_address**, not by a freshly validated live token universe.

## 7. UI Truth Label

- Previous: Live Market
- Applied recommendation: **{ui_rec['applied_label']}**
- Visible copy: {ui_rec['visible_copy']}
- Detail: `ae13j_ui_truth_label_recommendation.json`

## 8. Decision Gate

Primary: `{gate['primary_classification']}`

All: `{gate['all_classifications']}`

Reasons:
"""
    for reason in gate["reasons"]:
        report += f"\n- {reason}"

    report += """

### Classification options considered

- AE13J_DATA_LAYER_VALIDATED_FOR_AE14
- AE13J_PASS_WITH_PROVENANCE_LIMITATIONS
- AE13J_BLOCKED_LIVE_MARKET_NOT_ACTUALLY_LIVE
- AE13J_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED
- AE13J_BLOCKED_TRAINING_DATA_PROVENANCE_INCOMPLETE
- AE13J_BLOCKED_PROVIDER_PAYLOAD_TRACE_MISSING
- AE13J_BLOCKED_SAFETY_RISK

---
*AE13J emergency data-trust audit. No retrain. No live trading. No profitability claims.*
"""
    (OUT_DIR / "ae13j_ground_truth_market_data_report.md").write_text(report, encoding="utf-8")

    summary = "\n".join(
        [
            "AE13J GROUND TRUTH MARKET DATA AUDIT — SUMMARY FOR UPLOAD",
            f"timestamp_utc: {gate['timestamp_utc']}",
            f"artifact_dir: {gate['artifact_dir']}",
            f"primary_classification: {gate['primary_classification']}",
            f"pass: {gate['pass']}",
            f"ae14_blocked: {gate['ae14_blocked']}",
            f"live_rows: {gate['metrics']['live_rows']}",
            f"fresh_le_120s: {gate['metrics']['fresh_rows_le_120s']}",
            f"tradable_now: {gate['metrics']['tradable_now']}",
            f"pool_or_pair_role_rows: {gate['metrics']['pool_or_pair_role_rows']}",
            f"token_role_rows: {gate['metrics']['token_role_rows']}",
            f"unique_pairs: {gate['metrics']['unique_pairs']}",
            f"unique_tokens: {gate['metrics']['unique_tokens']}",
            f"refresh_classes: {gate['metrics']['refresh_class_counts']}",
            f"ui_truth_label: {gate['ui_truth_label']}",
            "do_not_claim_profitability: true",
            "do_not_enable_live_trading: true",
            "",
            "reasons:",
            *[f"- {r}" for r in gate["reasons"]],
            "",
            "key_files:",
            "- ae13j_ground_truth_market_data_report.md",
            "- ae13j_decision_gate.json",
            "- ae13j_live_market_truth_table.csv",
            "- ae13j_duplicate_repetition_audit.csv",
            "- ae13j_refresh_reality_audit.csv",
            "- ae13j_provider_payload_trace.json",
            "- ae13j_token_pair_identity_verdict.csv",
            "- ae13j_training_data_validity_audit.csv",
            "- ae13j_ui_truth_label_recommendation.json",
        ]
    )
    (OUT_DIR / "ae13j_summary_for_upload.txt").write_text(summary, encoding="utf-8")

    # pointer for latest
    (ROOT / "data" / "audits" / "AE13J_LATEST.txt").write_text(
        str(OUT_DIR.relative_to(ROOT)).replace("\\", "/") + "\n", encoding="utf-8"
    )

    conn.close()
    print(summary)
    print(f"[AE13J] DONE primary={gate['primary_classification']}")
    # write path for parent process
    (OUT_DIR / "_OUT_DIR.txt").write_text(str(OUT_DIR), encoding="utf-8")
    return 0 if True else 1  # audit always exits 0; gate JSON carries pass/fail


if __name__ == "__main__":
    raise SystemExit(main())
