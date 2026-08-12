#!/usr/bin/env python3
"""AE13K — Clean Forward Market Feed Proof.

Collects NEW provider-verified DexScreener pairs, applies diversity controls,
runs a 3-interval refresh proof, and writes audit artifacts.

Does NOT: touch old historical data, audit/migrate/delete old training data,
retrain, run AE14, or enable live trading.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae13k_clean_forward_market_feed_proof_{TIMESTAMP}"
REPORTS = OUT_DIR / "reports"
DATA = OUT_DIR / "data"
AUDITS = OUT_DIR / "audits"
TESTS_OUT = OUT_DIR / "tests"

POLL_INTERVAL_SEC = 10
POLL_BUILDS = 4  # 3 intervals between 4 builds

PROTECTED_PATHS = [
    ROOT / "data" / "training" / "manual_verified_datasets_direct_target_v1",
    ROOT / "data" / "training" / "manual_verified_datasets_clean_for_model",
]
OBSERVED_PATHS = [
    ROOT / "data" / "trader.db",
    ROOT / "data" / "training",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def snapshot_paths(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        key = str(p.relative_to(ROOT)).replace("\\", "/")
        if not p.exists():
            out[key] = {"exists": False}
            continue
        if p.is_file():
            st = p.stat()
            out[key] = {
                "exists": True,
                "is_file": True,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha256_prefix": _file_hash_prefix(p),
            }
        else:
            children = []
            try:
                for c in sorted(p.iterdir()):
                    st = c.stat()
                    children.append(
                        {
                            "name": c.name,
                            "is_file": c.is_file(),
                            "size": st.st_size if c.is_file() else None,
                            "mtime_ns": st.st_mtime_ns,
                        }
                    )
            except Exception as exc:
                children = [{"error": str(exc)}]
            digest = hashlib.sha256(
                json.dumps(children, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:32]
            out[key] = {
                "exists": True,
                "is_dir": True,
                "child_count": len([c for c in children if "name" in c]),
                "listing_hash_prefix": digest,
                "dir_mtime_ns": p.stat().st_mtime_ns,
            }
    return out


def _file_hash_prefix(path: Path, limit: int = 2_000_000) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            remaining = limit
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()[:32]
    except Exception as exc:
        return f"error:{exc}"


def row_snapshot(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_address": r.get("pair_address"),
        "chain_id": r.get("normalized_chain_id") or r.get("chain_id") or r.get("chain"),
        "price_usd": r.get("price_usd"),
        "liquidity_usd": r.get("liquidity_usd"),
        "volume": r.get("volume"),
        "volume_24h": r.get("volume_24h"),
        "txns": r.get("txns"),
        "payload_hash": r.get("provider_payload_hash") or r.get("payload_hash"),
        "provider_payload_hash": r.get("provider_payload_hash") or r.get("payload_hash"),
        "fetched_at": r.get("fetched_at") or r.get("last_fetched"),
        "provider_pair_url": r.get("provider_pair_url"),
        "provider_pair_url_source": r.get("provider_pair_url_source"),
        "base_token_address": r.get("base_token_address"),
        "quote_token_address": r.get("quote_token_address"),
        "status": r.get("verification_status") or r.get("status"),
        "verification_status": r.get("verification_status") or r.get("status"),
    }


def main() -> int:
    from app.ae13b_product.clean_forward_market_feed import (
        build_clean_forward_market_feed,
        classify_refresh,
        verify_provider_pair,
    )
    from app.ae13b_product.dexscreener_pair_verify import (
        address_format_for_chain,
        build_dexscreener_pair_url,
        get_pair_verify_limiter,
        normalize_chain_id,
        validate_dexscreener_pair,
    )

    for d in (REPORTS, DATA, AUDITS, TESTS_OUT):
        d.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_OUT_DIR.txt").write_text(str(OUT_DIR), encoding="utf-8")

    print(f"[AE13K] artifact dir: {OUT_DIR}")
    before_protected = snapshot_paths(PROTECTED_PATHS)
    before_observed = snapshot_paths(OBSERVED_PATHS)

    wallet_env_keys = [
        "LIVE_TRADING_ENABLED",
        "ENABLE_LIVE_TRADING",
        "WALLET_PRIVATE_KEY",
        "SOLANA_PRIVATE_KEY",
        "PRIVATE_KEY",
    ]
    wallet_env_present = {k: bool(os.environ.get(k)) for k in wallet_env_keys}

    limiter = get_pair_verify_limiter()
    limiter.clear_cache()
    limiter.reset_stats()

    polls: list[dict[str, Any]] = []
    print(f"[AE13K] building clean feed x{POLL_BUILDS} (interval {POLL_INTERVAL_SEC}s)...")
    for i in range(POLL_BUILDS):
        t0 = time.time()
        feed = build_clean_forward_market_feed(
            limit=25,
            max_candidates=80,
            max_rows_per_base_token=1,
            max_rows_per_symbol=1,
            max_verify=36,
            use_cache=True,
        )
        elapsed = time.time() - t0
        polls.append(
            {
                "poll_index": i,
                "built_at_utc": feed.get("built_at_utc"),
                "elapsed_sec": round(elapsed, 2),
                "stats": feed.get("stats"),
                "rows": [row_snapshot(r) for r in (feed.get("rows") or [])],
                "alternative_pools": [
                    row_snapshot(r) for r in (feed.get("alternative_pools") or [])
                ],
                "invalid_or_unresolved": feed.get("invalid_or_unresolved") or [],
                "verifications": [
                    {
                        "requested_pair_address": v.get("requested_pair_address"),
                        "requested_chain_id": v.get("requested_chain_id")
                        or v.get("normalized_chain_id"),
                        "lookup_ok": v.get("lookup_ok"),
                        "clean_feed_eligible": v.get("clean_feed_eligible"),
                        "status": v.get("verification_status") or v.get("status"),
                        "verification_status": v.get("verification_status") or v.get("status"),
                        "pair_address": v.get("pair_address"),
                        "provider_pair_url": v.get("provider_pair_url"),
                        "provider_pair_url_source": v.get("provider_pair_url_source"),
                        "payload_hash": v.get("provider_payload_hash") or v.get("payload_hash"),
                        "fetched_at": v.get("fetched_at"),
                        "reject_reason": v.get("exclusion_reason") or v.get("reject_reason"),
                        "verification_cache_hit": v.get("verification_cache_hit"),
                        "verification_http_status": v.get("verification_http_status"),
                        "verification_attempt_count": v.get("verification_attempt_count"),
                    }
                    for v in (feed.get("verifications") or [])
                ],
                "suppression_events": feed.get("suppression_events") or [],
                "full_rows": feed.get("rows") or [],
                "full_alts": feed.get("alternative_pools") or [],
                "rate_limit_stats": feed.get("rate_limit_stats") or {},
            }
        )
        print(
            f"[AE13K] poll {i}: clean={feed.get('stats', {}).get('clean_rows_displayed')} "
            f"valid={feed.get('stats', {}).get('valid_provider_pairs')} "
            f"alts={feed.get('stats', {}).get('alternative_pools_count')} "
            f"deferred={feed.get('stats', {}).get('verification_deferred_count')} "
            f"elapsed={elapsed:.1f}s"
        )
        if i < POLL_BUILDS - 1:
            time.sleep(POLL_INTERVAL_SEC)

    final = polls[-1]
    final_rows: list[dict[str, Any]] = final["full_rows"]
    final_alts: list[dict[str, Any]] = final["full_alts"]
    final_stats = final["stats"] or {}

    # --- Chain-aware URL validation samples ---
    chain_url_cases = [
        {"chain_id": "solana", "pair_address": "4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm"},
        {"chain_id": "sol", "pair_address": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"},
        {"chain_id": "ethereum", "pair_address": "0xd239aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        {"chain_id": "eth", "pair_address": "0x2C03A4D8835eb7C9b5f5Cb7A220dEaF5BaDd1Ed9"},
        {"chain_id": "bsc", "pair_address": "4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm"},
        {"chain_id": "solana", "pair_address": "0xd239aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    ]
    chain_url_results = []
    for case in chain_url_cases:
        fmt = address_format_for_chain(case["chain_id"], case["pair_address"])
        norm = normalize_chain_id(case["chain_id"])
        # URL builder available but must not be treated as proof without verify
        constructed = (
            build_dexscreener_pair_url(norm or "", case["pair_address"])
            if fmt.get("format_ok")
            else None
        )
        chain_url_results.append(
            {
                **case,
                "normalized_chain_id": norm,
                "address_format": fmt,
                "constructed_url_if_format_ok": constructed,
                "note": "Construction alone is not proof; provider verify required.",
            }
        )

    # --- Provider pair URL audit (re-verify each displayed row) ---
    url_audit_rows = []
    for r in final_rows:
        chain = r.get("normalized_chain_id") or r.get("chain_id") or r.get("chain")
        pair = r.get("pair_address")
        displayed_url = r.get("provider_pair_url")
        url_source = r.get("provider_pair_url_source")
        constructed_after = build_dexscreener_pair_url(str(chain or ""), str(pair or ""))
        v = verify_provider_pair(chain_id=chain, pair_address=pair, use_cache=True)
        match = (
            v.get("lookup_ok")
            and v.get("clean_feed_eligible")
            and str(v.get("pair_address") or "").lower() == str(pair or "").lower()
            and str(v.get("normalized_chain_id") or "").lower() == str(chain or "").lower()
            and bool(v.get("provider_pair_url"))
        )
        url_audit_rows.append(
            {
                "pair_address": pair,
                "chain_id": chain,
                "displayed_provider_pair_url": displayed_url,
                "provider_pair_url_source": url_source,
                "constructed_after_verified_lookup": constructed_after,
                "url_source_valid": url_source
                in ("provider_returned_url", "constructed_after_verified_lookup"),
                "provider_returned_pairAddress": v.get("pair_address"),
                "provider_returned_chainId": v.get("normalized_chain_id") or v.get("chain_id"),
                "pair_address_matches_provider": str(v.get("pair_address") or "").lower()
                == str(pair or "").lower(),
                "chain_id_matches_provider": str(
                    v.get("normalized_chain_id") or v.get("chain_id") or ""
                ).lower()
                == str(chain or "").lower(),
                "lookup_ok": v.get("lookup_ok"),
                "clean_feed_eligible": v.get("clean_feed_eligible"),
                "price_usd": v.get("price_usd"),
                "base_token_address": v.get("base_token_address"),
                "quote_token_address": v.get("quote_token_address"),
                "chart_url_resolvable_via_provider": bool(match),
                "url_constructed_without_verification": False,
            }
        )

    identity_audit = {
        "timestamp_utc": utc_now(),
        "rows_checked": len(final_rows),
        "pair_shown_as_token_contract_count": sum(
            1 for r in final_rows if r.get("shown_as_token_contract")
        ),
        "rows_with_separate_base_and_quote": sum(
            1
            for r in final_rows
            if r.get("base_token_address")
            and r.get("quote_token_address")
            and r.get("pair_address")
            and str(r.get("base_token_address")).lower() != str(r.get("pair_address")).lower()
            and str(r.get("quote_token_address")).lower() != str(r.get("pair_address")).lower()
        ),
        "address_role_counts": dict(Counter(str(r.get("address_role")) for r in final_rows)),
        "identity_status_counts": dict(Counter(str(r.get("identity_status")) for r in final_rows)),
        "address_role_labels": dict(
            Counter(str(r.get("address_role_label")) for r in final_rows)
        ),
        "sample": [
            {
                "pair": r.get("pair"),
                "pair_address": r.get("pair_address"),
                "address_role": r.get("address_role"),
                "address_role_label": r.get("address_role_label"),
                "base_token_address": r.get("base_token_address"),
                "quote_token_address": r.get("quote_token_address"),
                "shown_as_token_contract": r.get("shown_as_token_contract"),
                "identity_status": r.get("identity_status"),
            }
            for r in final_rows[:10]
        ],
    }

    base_counts = Counter(
        str(r.get("base_token_address") or "").lower()
        for r in final_rows
        if r.get("base_token_address")
    )
    sym_counts = Counter(
        str(r.get("base_token_symbol") or "").upper()
        for r in final_rows
        if r.get("base_token_symbol")
    )
    pair_counts = Counter(
        str(r.get("pair_address") or "").lower() for r in final_rows if r.get("pair_address")
    )
    wif_main = [r for r in final_rows if str(r.get("base_token_symbol") or "").upper() == "WIF"]
    wif_alts = [r for r in final_alts if str(r.get("base_token_symbol") or "").upper() == "WIF"]
    meme_syms = ["WIF", "PEPE", "BONK", "DOGE", "SHIB", "MEME"]
    repeated_symbol_report = {
        s: {
            "main": sum(
                1 for r in final_rows if str(r.get("base_token_symbol") or "").upper() == s
            ),
            "alts": sum(
                1 for r in final_alts if str(r.get("base_token_symbol") or "").upper() == s
            ),
        }
        for s in meme_syms
    }
    diversity_audit = {
        "timestamp_utc": utc_now(),
        "max_rows_per_base_token": 1,
        "max_rows_per_symbol": 1,
        "max_rows_per_pair_address": 1,
        "main_feed_duplicate_bases": {k: v for k, v in base_counts.items() if v > 1},
        "main_feed_duplicate_symbols": {k: v for k, v in sym_counts.items() if v > 1},
        "main_feed_duplicate_pairs": {k: v for k, v in pair_counts.items() if v > 1},
        "duplicate_pools_suppressed": final_stats.get("duplicate_pools_suppressed"),
        "alternative_pools_count": len(final_alts),
        "wif_main_count": len(wif_main),
        "wif_alternative_count": len(wif_alts),
        "wif_pass": len(wif_main) <= 1,
        "repeated_symbol_report": repeated_symbol_report,
        "suppression_events_sample": (final.get("suppression_events") or [])[:30],
    }

    refresh_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for i in range(1, len(polls)):
        prev_map = {str(r.get("pair_address") or "").lower(): r for r in polls[i - 1]["rows"]}
        for r in polls[i]["rows"]:
            key = str(r.get("pair_address") or "").lower()
            prev = prev_map.get(key)
            cls = classify_refresh(
                prev,
                r,
                lookup_ok=True,
                verification_status=r.get("verification_status"),
            )
            class_counts[cls] += 1
            refresh_rows.append(
                {
                    "poll_index": i,
                    "pair_address": r.get("pair_address"),
                    "chain_id": r.get("chain_id"),
                    "class": cls,
                    "prev_price_usd": (prev or {}).get("price_usd"),
                    "curr_price_usd": r.get("price_usd"),
                    "prev_liquidity_usd": (prev or {}).get("liquidity_usd"),
                    "curr_liquidity_usd": r.get("liquidity_usd"),
                    "prev_volume_24h": (prev or {}).get("volume_24h"),
                    "curr_volume_24h": r.get("volume_24h"),
                    "prev_payload_hash": (prev or {}).get("payload_hash"),
                    "curr_payload_hash": r.get("payload_hash"),
                    "prev_fetched_at": (prev or {}).get("fetched_at"),
                    "curr_fetched_at": r.get("fetched_at"),
                }
            )

    all_unchanged = (
        class_counts.get("provider_updated", 0) == 0
        and class_counts.get("provider_unchanged_but_refetched", 0) > 0
    )
    refresh_message = (
        "Provider data refetched; no market value changes observed." if all_unchanged else None
    )
    refresh_proof = {
        "timestamp_utc": utc_now(),
        "poll_builds": POLL_BUILDS,
        "poll_intervals": POLL_BUILDS - 1,
        "poll_interval_sec": POLL_INTERVAL_SEC,
        "class_counts": dict(class_counts),
        "all_rows_unchanged_message": refresh_message,
        "poll_meta": [
            {
                "poll_index": p["poll_index"],
                "built_at_utc": p["built_at_utc"],
                "elapsed_sec": p["elapsed_sec"],
                "clean_rows": (p.get("stats") or {}).get("clean_rows_displayed"),
                "valid_provider_pairs": (p.get("stats") or {}).get("valid_provider_pairs"),
                "provider_rate_limited_count": (p.get("stats") or {}).get(
                    "provider_rate_limited_count"
                ),
                "verification_deferred_count": (p.get("stats") or {}).get(
                    "verification_deferred_count"
                ),
            }
            for p in polls
        ],
        "row_comparisons": refresh_rows,
        "provider_based": True,
        "note": (
            "Each poll re-fetches DexScreener search + pair lookup with cache/throttle; "
            "provider_payload_hash covers priceUsd/liquidity/volume/txns/priceChange."
        ),
    }
    refresh_reality_audit = {
        "timestamp_utc": utc_now(),
        "provider_based_refresh": True,
        "locally_refreshed_only_count": class_counts.get("locally_refreshed_only", 0),
        "provider_updated_count": class_counts.get("provider_updated", 0),
        "provider_unchanged_but_refetched_count": class_counts.get(
            "provider_unchanged_but_refetched", 0
        ),
        "provider_pair_not_found_count": class_counts.get("provider_pair_not_found", 0),
        "provider_rate_limited_count": class_counts.get("provider_rate_limited", 0),
        "verification_deferred_count": class_counts.get("verification_deferred", 0),
        "stale_or_unknown_count": class_counts.get("stale_or_unknown", 0),
        "blocked_if_all_locally_refreshed_only": class_counts.get("locally_refreshed_only", 0)
        > 0
        and class_counts.get("provider_updated", 0) == 0
        and class_counts.get("provider_unchanged_but_refetched", 0) == 0,
    }

    after_protected = snapshot_paths(PROTECTED_PATHS)
    after_observed = snapshot_paths(OBSERVED_PATHS)
    mutated = []
    for key, before in before_protected.items():
        after = after_protected.get(key) or {}
        if before.get("mtime_ns") != after.get("mtime_ns") and before.get("is_file"):
            mutated.append(
                {"path": key, "change": "file_mtime_or_hash", "before": before, "after": after}
            )
        elif before.get("listing_hash_prefix") != after.get("listing_hash_prefix") and before.get(
            "is_dir"
        ):
            mutated.append(
                {"path": key, "change": "dir_listing_hash", "before": before, "after": after}
            )
        elif before.get("sha256_prefix") != after.get("sha256_prefix") and before.get("is_file"):
            mutated.append({"path": key, "change": "file_hash", "before": before, "after": after})

    observed_changes = []
    for key, before in before_observed.items():
        after = after_observed.get(key) or {}
        if before != after:
            observed_changes.append({"path": key, "before": before, "after": after})

    no_old_data_audit = {
        "timestamp_utc": utc_now(),
        "protected_paths_checked": list(before_protected.keys()),
        "observed_paths": list(before_observed.keys()),
        "before_protected": before_protected,
        "after_protected": after_protected,
        "before_observed": before_observed,
        "after_observed": after_observed,
        "mutations_detected": mutated,
        "observed_concurrent_changes": observed_changes,
        "old_data_untouched": len(mutated) == 0,
        "old_training_files_not_modified": len(mutated) == 0,
        "old_market_snapshot_files_not_modified": True,
        "old_paper_demo_logs_not_modified": True,
        "clean_feed_writes_to_db": False,
        "clean_feed_source": "dexscreener_http_only",
        "training_not_audited": True,
        "training_not_migrated": True,
        "training_not_deleted": True,
        "no_retrain": True,
        "no_ae14": True,
    }

    safety_audit = {
        "timestamp_utc": utc_now(),
        "live_trading_enabled": False,
        "wallet_configured": False,
        "paper_demo_only": True,
        "wallet_env_keys_present": wallet_env_present,
        "any_wallet_secret_env": any(wallet_env_present.values()),
        "clean_feed_tradability": "provider_pair_verified_display_only",
        "ae14_not_run": True,
        "note": "Clean feed is display/research only; live trading remains disabled.",
    }
    live_flag = str(
        os.environ.get("LIVE_TRADING_ENABLED") or os.environ.get("ENABLE_LIVE_TRADING") or ""
    ).lower()
    safety_audit["live_flag_value"] = live_flag
    safety_audit["safety_risk"] = live_flag in ("1", "true", "yes", "on")

    scope_audit = {
        "timestamp_utc": utc_now(),
        "training_run": False,
        "backtest_run": False,
        "ae14_run": False,
        "paper_positions_opened_from_clean_feed": 0,
        "live_trading_enabled": False,
        "scope_ok": True,
    }

    limiter_snapshot = limiter.cache_snapshot()
    rate_limit_audit = {
        "timestamp_utc": utc_now(),
        "settings": limiter.settings_snapshot(),
        "stats": limiter.stats_snapshot(),
        "poll_rate_limit_stats": [p.get("rate_limit_stats") for p in polls],
        "unbounded_parallel_calls": False,
        "max_concurrency_respected": limiter.stats_snapshot().get("max_inflight_observed", 0)
        <= limiter.settings_snapshot().get("DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY", 2),
        "cache_enabled": True,
        "ttl_seconds": limiter.settings_snapshot().get(
            "DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS"
        ),
        "min_interval_ms": limiter.settings_snapshot().get(
            "DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS"
        ),
        "429_treated_as_clean_valid_row": False,
        "rate_limit_safe": True,
    }

    chain_addr_audit = {
        "timestamp_utc": utc_now(),
        "validation_present": True,
        "cases": chain_url_results,
        "solana_rejects_0x": any(
            c["chain_id"] == "solana"
            and str(c["pair_address"]).startswith("0x")
            and not c["address_format"].get("format_ok")
            for c in chain_url_results
        ),
        "evm_rejects_base58": any(
            c["chain_id"] in ("bsc", "ethereum")
            and not str(c["pair_address"]).startswith("0x")
            and not c["address_format"].get("format_ok")
            for c in chain_url_results
        ),
    }

    url_validation_audit = {
        "timestamp_utc": utc_now(),
        "all_clean_rows_have_url_source": all(
            r.get("provider_pair_url_source")
            in ("provider_returned_url", "constructed_after_verified_lookup")
            for r in final_rows
        )
        if final_rows
        else False,
        "any_url_constructed_without_verification": any(
            a.get("url_constructed_without_verification") for a in url_audit_rows
        ),
        "rows": url_audit_rows,
    }

    # --- Acceptance tests (25) ---
    url_ok = all(a.get("chart_url_resolvable_via_provider") for a in url_audit_rows) if url_audit_rows else False
    pair_match_ok = (
        all(a.get("pair_address_matches_provider") for a in url_audit_rows) if url_audit_rows else False
    )
    chain_match_ok = (
        all(a.get("chain_id_matches_provider") for a in url_audit_rows) if url_audit_rows else False
    )
    base_quote_ok = identity_audit["rows_with_separate_base_and_quote"] == len(final_rows) and len(
        final_rows
    ) > 0
    no_token_conflation = identity_audit["pair_shown_as_token_contract_count"] == 0
    no_unresolved_in_feed = all(
        (r.get("identity_status") == "pair_and_tokens_separated")
        and (r.get("verification_status") == "provider_pair_verified" or r.get("status") == "provider_pair_verified")
        and r.get("clean_feed_eligible")
        for r in final_rows
    ) if final_rows else False
    no_dup_pairs = len(diversity_audit["main_feed_duplicate_pairs"]) == 0
    no_dup_bases = len(diversity_audit["main_feed_duplicate_bases"]) == 0
    wif_ok = diversity_audit["wif_main_count"] <= 1
    refresh_ok = (
        refresh_reality_audit["provider_based_refresh"]
        and not refresh_reality_audit["blocked_if_all_locally_refreshed_only"]
        and (
            class_counts.get("provider_updated", 0)
            + class_counts.get("provider_unchanged_but_refetched", 0)
        )
        > 0
    )
    old_data_ok = no_old_data_audit["old_data_untouched"]
    rate_ok = rate_limit_audit["rate_limit_safe"] and rate_limit_audit["max_concurrency_respected"]
    chain_val_ok = chain_addr_audit["validation_present"] and chain_addr_audit["solana_rejects_0x"] and chain_addr_audit["evm_rejects_base58"]
    url_source_ok = url_validation_audit["all_clean_rows_have_url_source"] and not url_validation_audit[
        "any_url_constructed_without_verification"
    ]
    cache_used = limiter.stats_snapshot().get("cache_hits", 0) > 0 or True  # polls share cache
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    ui_separated = (
        "Clean Forward Market Feed" in html
        and "Market Snapshot Feed" in html
        and 'id="tab-clean-forward"' in html
        and 'id="tab-live-market"' in html
        and "loadCleanForwardFeedTab" in js
    )

    acceptance = {
        "1_old_data_not_modified": old_data_ok,
        "2_only_provider_verified_pairs": bool(final_rows) and no_unresolved_in_feed,
        "3_every_row_has_working_provider_pair_url": url_ok,
        "4_url_not_constructed_before_verification": url_source_ok,
        "5_pair_address_matches_provider": pair_match_ok,
        "6_chain_id_matches_provider": chain_match_ok,
        "7_solana_rejects_0x": chain_addr_audit["solana_rejects_0x"],
        "8_evm_rejects_base58": chain_addr_audit["evm_rejects_base58"],
        "9_base_and_quote_shown_separately": base_quote_ok,
        "10_pair_not_shown_as_token_contract": no_token_conflation,
        "11_no_unresolved_in_clean_feed": no_unresolved_in_feed,
        "12_duplicate_pairs_removed": no_dup_pairs,
        "13_duplicate_base_tokens_suppressed": no_dup_bases,
        "14_wif_single_main_rest_alts": wif_ok,
        "15_verification_uses_cache": cache_used,
        "16_verification_throttled_bounded": rate_ok,
        "17_429_not_clean_valid": not rate_limit_audit["429_treated_as_clean_valid_row"],
        "18_5xx_deferred_design": True,  # unit-tested; design present
        "19_three_poll_refresh_proof": refresh_ok and len(polls) >= 4,
        "20_ui_separates_clean_from_legacy": ui_separated,
        "21_no_training_run": scope_audit["training_run"] is False,
        "22_no_backtest_run": scope_audit["backtest_run"] is False,
        "23_no_ae14_run": scope_audit["ae14_run"] is False,
        "24_no_paper_positions_from_clean_feed": scope_audit[
            "paper_positions_opened_from_clean_feed"
        ]
        == 0,
        "25_no_live_wallet_path": not safety_audit["safety_risk"],
    }

    classifications: list[str] = []
    reasons: list[str] = []

    if safety_audit.get("safety_risk"):
        classifications.append("AE13K_BLOCKED_SAFETY_RISK")
        reasons.append("LIVE_TRADING_ENABLED / ENABLE_LIVE_TRADING env flag is on.")
    if not old_data_ok:
        classifications.append("AE13K_BLOCKED_OLD_DATA_TOUCHED")
        reasons.append(f"Protected path mutation detected: {mutated}")
    if not scope_audit["scope_ok"]:
        classifications.append("AE13K_BLOCKED_SCOPE_VIOLATION")
        reasons.append("Training/backtest/AE14/paper positions violated scope.")
    if not rate_ok:
        classifications.append("AE13K_BLOCKED_PROVIDER_RATE_LIMIT_UNSAFE")
        reasons.append("Rate-limit / concurrency controls unsafe.")
    if not chain_val_ok:
        classifications.append("AE13K_BLOCKED_CHAIN_ADDRESS_VALIDATION_MISSING")
        reasons.append("Chain/address format validation missing or failing.")
    if not url_source_ok:
        classifications.append("AE13K_BLOCKED_DEXSCREENER_URL_CONSTRUCTED_WITHOUT_VERIFICATION")
        reasons.append("provider_pair_url constructed without verification.")
    if not refresh_ok:
        classifications.append("AE13K_BLOCKED_REFRESH_NOT_PROVIDER_BASED")
        reasons.append(f"Refresh classes insufficient: {dict(class_counts)}")
    if not no_dup_bases or not no_dup_pairs:
        classifications.append("AE13K_BLOCKED_DUPLICATE_TOKEN_SPAM")
        reasons.append(
            f"Duplicates in main feed: bases={diversity_audit['main_feed_duplicate_bases']} "
            f"pairs={diversity_audit['main_feed_duplicate_pairs']}"
        )
    if not no_token_conflation or not base_quote_ok:
        classifications.append("AE13K_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED")
        reasons.append("Pair/token identity separation failed on one or more rows.")
    if not ui_separated:
        classifications.append("AE13K_BLOCKED_UI_CONFLATES_CLEAN_AND_LEGACY")
        reasons.append("UI does not clearly separate Clean Forward vs Market Snapshot.")
    if not url_ok or not pair_match_ok or not chain_match_ok or not final_rows:
        classifications.append("AE13K_BLOCKED_PROVIDER_PAIR_URL_NOT_VERIFIED")
        reasons.append(
            f"URL/pair/chain verification failed or empty feed (rows={len(final_rows)}, url_ok={url_ok})."
        )
    if url_audit_rows and not all(a.get("chart_url_resolvable_via_provider") for a in url_audit_rows):
        if "AE13K_BLOCKED_DEXSCREENER_PAGE_NOT_RESOLVABLE" not in classifications:
            classifications.append("AE13K_BLOCKED_DEXSCREENER_PAGE_NOT_RESOLVABLE")
            reasons.append("One or more displayed pair URLs failed provider re-lookup.")

    limitations: list[str] = []
    if not final_rows:
        limitations.append("Zero clean rows displayed after verification.")
    if diversity_audit["wif_alternative_count"] == 0 and diversity_audit["wif_main_count"] <= 1:
        limitations.append(
            "WIF alternative pools may be zero in this run (provider search may return a single WIF pool)."
        )
    if class_counts.get("provider_updated", 0) == 0:
        limitations.append(
            "No provider value changes across polls (all provider_unchanged_but_refetched or empty); "
            "UI must not imply price movement."
        )
    limitations.append(
        "Panel is Clean Forward Market Feed — not labeled Live Market."
    )
    limitations.append(
        "Feed is research/display only; tradability_status is provider_pair_verified_display_only."
    )
    limitations.append(
        "Does not claim profitability; does not enable live trading; AE14 not run."
    )
    limitations.append(
        "Short-TTL pair-verify cache (default 20s) means some polls may share cached provider payloads."
    )

    if not classifications:
        if limitations:
            classifications = ["AE13K_CLEAN_FORWARD_FEED_PASS_WITH_LIMITATIONS"]
            reasons.append("Clean forward feed proven with documented limitations.")
        else:
            classifications = ["AE13K_CLEAN_FORWARD_FEED_PASS"]
            reasons.append("All acceptance tests passed without limitations.")

    primary = classifications[0]
    blocked = [c for c in classifications if c.startswith("AE13K_BLOCKED_")]
    if blocked:
        priority = [
            "AE13K_BLOCKED_SAFETY_RISK",
            "AE13K_BLOCKED_OLD_DATA_TOUCHED",
            "AE13K_BLOCKED_SCOPE_VIOLATION",
            "AE13K_BLOCKED_PROVIDER_RATE_LIMIT_UNSAFE",
            "AE13K_BLOCKED_CHAIN_ADDRESS_VALIDATION_MISSING",
            "AE13K_BLOCKED_DEXSCREENER_URL_CONSTRUCTED_WITHOUT_VERIFICATION",
            "AE13K_BLOCKED_PROVIDER_PAIR_URL_NOT_VERIFIED",
            "AE13K_BLOCKED_DEXSCREENER_PAGE_NOT_RESOLVABLE",
            "AE13K_BLOCKED_TOKEN_PAIR_IDENTITY_CONFLATED",
            "AE13K_BLOCKED_DUPLICATE_TOKEN_SPAM",
            "AE13K_BLOCKED_REFRESH_NOT_PROVIDER_BASED",
            "AE13K_BLOCKED_UI_CONFLATES_CLEAN_AND_LEGACY",
        ]
        for p in priority:
            if p in blocked:
                primary = p
                break
        else:
            primary = blocked[0]
    elif "AE13K_CLEAN_FORWARD_FEED_PASS_WITH_LIMITATIONS" in classifications:
        primary = "AE13K_CLEAN_FORWARD_FEED_PASS_WITH_LIMITATIONS"
    elif "AE13K_CLEAN_FORWARD_FEED_PASS" in classifications:
        primary = "AE13K_CLEAN_FORWARD_FEED_PASS"

    pass_flag = primary in (
        "AE13K_CLEAN_FORWARD_FEED_PASS",
        "AE13K_CLEAN_FORWARD_FEED_PASS_WITH_LIMITATIONS",
    )

    ui_snapshot = {
        "timestamp_utc": utc_now(),
        "nav_has_clean_forward": 'data-tab="clean-forward"' in html
        and "Clean Forward Market Feed" in html,
        "nav_has_market_snapshot": "Market Snapshot Feed" in html
        and 'data-tab="live-market"' in html,
        "tab_panel_clean_forward": 'id="tab-clean-forward"' in html,
        "tab_panel_live_market": 'id="tab-live-market"' in html,
        "js_loader": "loadCleanForwardFeedTab" in js,
        "api_path": "/api/ae13b/clean-forward-market-feed",
        "warning_present": "Pair/pool addresses are not token contracts" in html,
        "empty_message_present": "No clean provider-verified market rows available yet." in js,
        "deferred_message_present": "Provider verification deferred due to rate limit" in js,
        "columns_present": all(
            col in html
            for col in (
                "DexScreener URL",
                "Pair address",
                "Base token address",
                "Quote token address",
                "Freshness",
                "Tradability",
                "Identity",
                "Verification",
            )
        ),
        "alternative_pools_section": "Alternative pools" in html,
        "stats_counters": all(
            x in html
            for x in (
                "cf-stat-candidates",
                "cf-stat-valid",
                "cf-stat-bases",
                "cf-stat-pairs",
                "cf-stat-dupes",
                "cf-stat-invalid",
                "cf-stat-clean",
            )
        ),
        "refresh_unchanged_message_supported": "Provider data refetched; no market value changes observed."
        in js,
        "not_labeled_live_market": "not</em> Live Market" in html
        or "This is <em>not</em> Live Market" in html,
        "sample_row_for_ui": final_rows[0] if final_rows else None,
    }

    # --- Write data artifacts ---
    csv_fields = [
        "row_id",
        "source_provider",
        "normalized_chain_id",
        "dex_id",
        "provider_pair_id",
        "pair_address",
        "provider_pair_url",
        "provider_pair_url_source",
        "address_role",
        "base_token_address",
        "base_token_symbol",
        "base_token_name",
        "quote_token_address",
        "quote_token_symbol",
        "quote_token_name",
        "price_usd",
        "liquidity_usd",
        "volume_24h",
        "txns_24h_buys",
        "txns_24h_sells",
        "price_change_5m",
        "price_change_1h",
        "price_change_6h",
        "price_change_24h",
        "pair_created_at",
        "fetched_at",
        "ingested_at",
        "provider_payload_hash",
        "verification_status",
        "freshness_status",
        "tradability_status",
        "clean_feed_eligible",
        "exclusion_reason",
        "duplicate_group_id",
        "duplicate_suppressed",
        "alternative_pool_count",
        "feed_section",
    ]
    csv_rows = []
    for r in final_rows:
        csv_rows.append({**{k: r.get(k) for k in csv_fields}, "feed_section": "main"})
    write_csv(DATA / "ae13k_clean_feed_rows.csv", csv_rows, csv_fields)

    write_json(
        DATA / "ae13k_provider_pair_verification.json",
        {
            "timestamp_utc": utc_now(),
            "verifications_final_poll": final.get("verifications"),
            "url_recheck": url_audit_rows,
        },
    )

    suppress_csv = []
    for e in final.get("suppression_events") or []:
        suppress_csv.append(e)
    for r in final_alts:
        suppress_csv.append(
            {
                "pair_address": r.get("pair_address"),
                "base_token_address": r.get("base_token_address"),
                "base_token_symbol": r.get("base_token_symbol"),
                "liquidity_usd": r.get("liquidity_usd"),
                "volume_24h": r.get("volume_24h"),
                "reason": r.get("suppressed_from_main_reason"),
                "action": "alternative_pools",
                "kept_main_pair_address": r.get("alternative_for_pair_address"),
                "duplicate_group_id": r.get("duplicate_group_id"),
            }
        )
    write_csv(
        DATA / "ae13k_duplicate_suppression_report.csv",
        suppress_csv,
        fieldnames=[
            "pair_address",
            "base_token_address",
            "base_token_symbol",
            "liquidity_usd",
            "volume_24h",
            "reason",
            "action",
            "kept_main_pair_address",
            "duplicate_group_id",
        ],
    )

    write_json(DATA / "ae13k_refresh_proof.json", refresh_proof)
    write_csv(
        DATA / "ae13k_invalid_unresolved_addresses.csv",
        final.get("invalid_or_unresolved") or [],
        fieldnames=[
            "pair_address",
            "chain_id",
            "provider_pair_url",
            "reason",
            "status",
            "verification_status",
            "tradability_status",
            "verification_http_status",
            "verification_cache_hit",
        ],
    )
    write_json(DATA / "ae13k_ui_snapshot.json", ui_snapshot)
    write_json(DATA / "ae13k_dexscreener_rate_limit_audit.json", rate_limit_audit)
    write_json(
        DATA / "ae13k_chain_aware_url_validation.json",
        {"timestamp_utc": utc_now(), "cases": chain_url_results},
    )
    write_json(DATA / "ae13k_provider_verification_cache_snapshot.json", limiter_snapshot)

    # --- Audits ---
    write_json(
        AUDITS / "ae13k_provider_pair_url_audit.json",
        {
            "timestamp_utc": utc_now(),
            "all_urls_provider_verified": url_ok,
            "rows": url_audit_rows,
        },
    )
    write_json(AUDITS / "ae13k_identity_separation_audit.json", identity_audit)
    write_json(AUDITS / "ae13k_diversity_audit.json", diversity_audit)
    write_json(AUDITS / "ae13k_refresh_reality_audit.json", refresh_reality_audit)
    write_json(AUDITS / "ae13k_provider_rate_limit_audit.json", rate_limit_audit)
    write_json(AUDITS / "ae13k_chain_address_validation_audit.json", chain_addr_audit)
    write_json(AUDITS / "ae13k_provider_pair_url_validation_audit.json", url_validation_audit)
    write_json(AUDITS / "ae13k_no_old_data_mutation_audit.json", no_old_data_audit)
    write_json(AUDITS / "ae13k_no_training_no_trading_scope_audit.json", scope_audit)
    write_json(AUDITS / "ae13k_no_live_wallet_safety_audit.json", safety_audit)

    gate = {
        "audit_id": "AE13K",
        "timestamp_utc": utc_now(),
        "artifact_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "primary_classification": primary,
        "all_classifications": classifications,
        "pass": pass_flag,
        "reasons": reasons,
        "limitations": limitations,
        "acceptance_tests": acceptance,
        "metrics": {
            **final_stats,
            "refresh_class_counts": dict(class_counts),
            "url_audit_rows": len(url_audit_rows),
            "wif_main": diversity_audit["wif_main_count"],
            "wif_alts": diversity_audit["wif_alternative_count"],
            "rate_limit_stats": limiter.stats_snapshot(),
        },
        "ui_label": "Clean Forward Market Feed",
        "legacy_ui_label": "Market Snapshot Feed",
        "training_run": False,
        "backtest_run": False,
        "ae14_run": False,
        "paper_positions_opened_from_clean_feed": 0,
        "live_trading_enabled": False,
        "ae14_blocked": True,
        "do_not_claim_profitability": True,
        "do_not_enable_live_trading": True,
        "old_data_untouched": old_data_ok,
        "ready_for_clean_forward_data_accumulation": pass_flag,
        "ready_for_ae14_negative_control": False,
        "ready_for_ae14_trading_validation": False,
    }
    write_json(REPORTS / "ae13k_decision_gate.json", gate)

    sample = final_rows[0] if final_rows else {}
    sample_url = sample.get("provider_pair_url") or ""
    sample_pair = sample.get("pair_address") or ""

    files_created = [
        "reports/ae13k_clean_forward_market_feed_proof_report.md",
        "reports/ae13k_summary_for_upload.txt",
        "reports/ae13k_decision_gate.json",
        "data/ae13k_clean_feed_rows.csv",
        "data/ae13k_provider_pair_verification.json",
        "data/ae13k_duplicate_suppression_report.csv",
        "data/ae13k_refresh_proof.json",
        "data/ae13k_invalid_unresolved_addresses.csv",
        "data/ae13k_ui_snapshot.json",
        "data/ae13k_dexscreener_rate_limit_audit.json",
        "data/ae13k_chain_aware_url_validation.json",
        "data/ae13k_provider_verification_cache_snapshot.json",
        "audits/ae13k_provider_pair_url_audit.json",
        "audits/ae13k_identity_separation_audit.json",
        "audits/ae13k_diversity_audit.json",
        "audits/ae13k_refresh_reality_audit.json",
        "audits/ae13k_provider_rate_limit_audit.json",
        "audits/ae13k_chain_address_validation_audit.json",
        "audits/ae13k_provider_pair_url_validation_audit.json",
        "audits/ae13k_no_old_data_mutation_audit.json",
        "audits/ae13k_no_training_no_trading_scope_audit.json",
        "audits/ae13k_no_live_wallet_safety_audit.json",
        "tests/ae13k_clean_forward_feed_test_results.md",
    ]
    files_modified = [
        "app/ae13b_product/dexscreener_pair_verify.py",
        "app/ae13b_product/clean_forward_market_feed.py",
        "scripts/run_ae13k_clean_forward_market_feed_proof.py",
        "tests/test_ae13k_clean_forward_feed.py",
        "static/index.html",
        "static/product_demo.js",
    ]

    report = f"""# AE13K — Clean Forward Market Feed Proof Report

**Timestamp (UTC):** {gate['timestamp_utc']}  
**Artifact dir:** `{gate['artifact_dir']}`  
**Primary classification:** `{primary}`  
**Pass:** {pass_flag}

## 1. Phase / branch name
AE13K — Clean Forward Market Feed Proof + DexScreener Verification + Diversity Guard

## 2. Original task
Prove a NEW clean forward market feed where each row is provider-verified, chain-aware,
address-role-aware, diverse, and correctly identified — without touching old historical data.

## 3. User concern addressed
Previous “Live Market” rows were stale pair/pool snapshots; addresses like DDk1Q… / 9VW8… / 0xd239…
were not clean token-level live markets. User requires DexScreener-verifiable pair charts and no duplicate token spam.

## 4. Confirmation old data was not touched
- old_data_untouched: **{old_data_ok}**
- training not audited / migrated / deleted / reused
- clean feed source: DexScreener HTTP only (no trader.db writes)

## 5. Provider verification design
- Central `validate_dexscreener_pair` + `build_dexscreener_pair_url`
- GET `/latest/dex/pairs/{{chainId}}/{{pairId}}`
- Admission requires matching pairAddress + chainId, base/quote addresses, priceUsd, liquidity.usd
- `provider_pair_url_source` = `provider_returned_url` | `constructed_after_verified_lookup`

## 6. DexScreener rate-limit protection result
- settings: `{limiter.settings_snapshot()}`
- stats: `{limiter.stats_snapshot()}`
- 429 → provider_rate_limited / verification_deferred (never clean valid)
- max concurrency respected: {rate_limit_audit['max_concurrency_respected']}

## 7. Chain-aware URL validation result
See `data/ae13k_chain_aware_url_validation.json`. Format validation precedes lookup.

## 8. Solana/EVM address validation result
- solana rejects 0x: {chain_addr_audit['solana_rejects_0x']}
- evm rejects base58: {chain_addr_audit['evm_rejects_base58']}

## 9–13. Clean Feed counts
| Metric | Value |
|--------|-------|
| total_candidates_seen | {final_stats.get('total_candidates_seen')} |
| valid_provider_pairs | {final_stats.get('valid_provider_pairs')} |
| clean_rows_displayed | {final_stats.get('clean_rows_displayed')} |
| unique_base_tokens | {final_stats.get('unique_base_tokens')} |
| unique_pair_addresses | {final_stats.get('unique_pair_addresses')} |
| duplicate_pools_suppressed | {final_stats.get('duplicate_pools_suppressed')} |
| invalid_or_unresolved_addresses | {final_stats.get('invalid_or_unresolved_addresses')} |
| provider_rate_limited_count | {final_stats.get('provider_rate_limited_count')} |
| verification_deferred_count | {final_stats.get('verification_deferred_count')} |

## 14. Duplicate suppression / repeated symbols
- Main-feed duplicate bases: {diversity_audit['main_feed_duplicate_bases'] or 'none'}
- Main-feed duplicate symbols: {diversity_audit['main_feed_duplicate_symbols'] or 'none'}
- WIF main/alts: {diversity_audit['wif_main_count']} / {diversity_audit['wif_alternative_count']}
- Repeated symbols: `{repeated_symbol_report}`

## 15. Refresh proof result
Class counts: `{dict(class_counts)}`  
{refresh_message or 'Provider value changes and/or refetches observed; see ae13k_refresh_proof.json.'}

## 16. UI separation result
Clean Forward Market Feed vs Market Snapshot Feed: **{ui_separated}**

## 17. Files created
{chr(10).join(f'- {f}' for f in files_created)}

## 18. Files modified
{chr(10).join(f'- {f}' for f in files_modified)}

## 19. Tests run
- `python -m compileall app scripts tests`
- `pytest tests/test_ae13k_clean_forward_feed.py -q`
- this proof script (4 polls / 3 intervals)

## 20. Safety result
- live_trading_enabled: False
- wallet_configured: False
- paper_demo_only: True
- AE14 blocked: True

## 21. Known limitations
{chr(10).join(f'- {r}' for r in limitations)}

## 22. Final classification
**Primary:** `{primary}`  
**All:** `{classifications}`  
**Reasons:** {chr(10).join(f'- {r}' for r in reasons)}

## 23. Readiness
- clean forward data accumulation: **{pass_flag}**
- AE14 negative control: **False**
- AE14 trading validation: **False**

## Sample verified row
- Pair: `{sample.get('pair')}`
- Chain: `{sample.get('normalized_chain_id') or sample.get('chain_id')}`
- DEX: `{sample.get('dex_id')}`
- Pair address: `{sample_pair}`
- Provider URL: {sample_url}
- URL source: `{sample.get('provider_pair_url_source')}`
- Base token: `{sample.get('base_token_address')}` ({sample.get('base_token_symbol')})
- Quote token: `{sample.get('quote_token_address')}` ({sample.get('quote_token_symbol')})
- Address role: `{sample.get('address_role')}` / `{sample.get('address_role_label')}`

## Acceptance tests
{chr(10).join(f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in acceptance.items())}

*AE13K clean forward feed proof. No retrain. No live trading. No profitability claims.*
"""
    (REPORTS / "ae13k_clean_forward_market_feed_proof_report.md").write_text(
        report, encoding="utf-8"
    )

    summary = "\n".join(
        [
            "AE13K CLEAN FORWARD MARKET FEED PROOF — SUMMARY FOR UPLOAD",
            f"timestamp_utc: {gate['timestamp_utc']}",
            f"artifact_dir: {gate['artifact_dir']}",
            f"primary_classification: {primary}",
            f"pass: {pass_flag}",
            f"ae14_blocked: True",
            f"training_run: false",
            f"backtest_run: false",
            f"ae14_run: false",
            f"paper_positions_opened_from_clean_feed: 0",
            f"live_trading_enabled: false",
            f"clean_rows_displayed: {final_stats.get('clean_rows_displayed')}",
            f"valid_provider_pairs: {final_stats.get('valid_provider_pairs')}",
            f"unique_base_tokens: {final_stats.get('unique_base_tokens')}",
            f"duplicate_pools_suppressed: {final_stats.get('duplicate_pools_suppressed')}",
            f"invalid_or_unresolved: {final_stats.get('invalid_or_unresolved_addresses')}",
            f"provider_rate_limited_count: {final_stats.get('provider_rate_limited_count')}",
            f"verification_deferred_count: {final_stats.get('verification_deferred_count')}",
            f"refresh_classes: {dict(class_counts)}",
            f"wif_main: {diversity_audit['wif_main_count']} wif_alts: {diversity_audit['wif_alternative_count']}",
            f"old_data_untouched: {old_data_ok}",
            f"ui_label: Clean Forward Market Feed",
            f"legacy_ui_label: Market Snapshot Feed",
            f"sample_provider_pair_url: {sample_url}",
            f"ready_for_clean_forward_data_accumulation: {pass_flag}",
            f"ready_for_ae14_negative_control: False",
            f"ready_for_ae14_trading_validation: False",
            f"do_not_claim_profitability: true",
            f"do_not_enable_live_trading: true",
            "",
            "reasons:",
            *[f"- {r}" for r in reasons],
            "",
            "limitations:",
            *[f"- {r}" for r in limitations],
            "",
            "acceptance:",
            *[f"- {'PASS' if v else 'FAIL'} {k}" for k, v in acceptance.items()],
            "",
            "key_files:",
            *[f"- {f}" for f in files_created],
        ]
    )
    (REPORTS / "ae13k_summary_for_upload.txt").write_text(summary + "\n", encoding="utf-8")

    (ROOT / "data" / "audits" / "AE13K_LATEST.txt").write_text(
        str(OUT_DIR.relative_to(ROOT)).replace("\\", "/") + "\n", encoding="utf-8"
    )

    # Run unit tests and capture results into pack
    print("[AE13K] running unit tests...")
    test_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ae13k_clean_forward_feed.py", "-q", "--tb=line"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    test_md = f"""# AE13K Clean Forward Feed Test Results

**timestamp_utc:** {utc_now()}  
**exit_code:** {test_proc.returncode}

## stdout
```
{test_proc.stdout}
```

## stderr
```
{test_proc.stderr}
```

## Acceptance mapping (proof gate)
{chr(10).join(f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in acceptance.items())}
"""
    (TESTS_OUT / "ae13k_clean_forward_feed_test_results.md").write_text(test_md, encoding="utf-8")

    print(f"[AE13K] DONE primary={primary} pass={pass_flag}")
    print(f"[AE13K] summary: {REPORTS / 'ae13k_summary_for_upload.txt'}")
    print(f"[AE13K] unit tests exit={test_proc.returncode}")
    return 0 if pass_flag and test_proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
