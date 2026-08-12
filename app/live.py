"""
Background whale-watcher loop — Gemini 2.5 Flash evaluation and autonomous paper execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analytics.features import build_feature_row, resolve_cluster_label
from .analytics.scan_persist import archive_dexscreener_search, new_scan_id, persist_pair_pipeline
from . import database as db
from .analytics.sentiment import fetch_rss_sentiment
from .dexscreener import get_trending_pairs, search_pairs
from .engine import compute_whale_score
from .execution.paper import get_paper_trader


def _run_selected_clean_collection_cycle() -> dict[str, Any]:
    """Exact-pair Selected/Clean + open-position fetches ahead of trending.

    Controlled by RUNTIME_SELECTED_COLLECTION_ENABLED (default true).
    Uses additive DB writers only; never treats ae16b_* as pair address.
    """
    from .clean_forward.runtime_selected_collection import (
        DEFAULT_FETCH_STATE_PATH,
        DEFAULT_PAPER_STATE,
        DEFAULT_POLICY,
        DEFAULT_SELECTED_PATH,
        build_runtime_priority_queue,
        load_fetch_state,
        load_open_positions,
        load_selected_csv,
        run_priority_fetch_cycle,
        save_fetch_state,
        selected_collection_enabled,
    )

    if not selected_collection_enabled():
        return {"enabled": False, "attempts": 0, "success": 0, "cooldown_skips": 0}

    selected_path = Path(
        os.getenv("CLEAN_FORWARD_CURATED_TARGETS_PATH", "") or str(DEFAULT_SELECTED_PATH)
    )
    if not selected_path.exists():
        selected_path = DEFAULT_SELECTED_PATH
    paper_path = Path(os.getenv("PAPER_STATE_PATH", "") or str(DEFAULT_PAPER_STATE))
    state_path = Path(
        os.getenv("RUNTIME_SELECTED_FETCH_STATE_PATH", "") or str(DEFAULT_FETCH_STATE_PATH)
    )

    selected_rows = load_selected_csv(selected_path) if selected_path.exists() else []
    open_positions = load_open_positions(paper_path)
    fetch_state = load_fetch_state(state_path)
    queue = build_runtime_priority_queue(
        selected_rows=selected_rows,
        open_positions=open_positions,
        fetch_state=fetch_state,
        include_discovery=False,
    )
    cycle = run_priority_fetch_cycle(
        queue,
        policy=DEFAULT_POLICY,
        fetch_state=fetch_state,
        mode="write-db",
        respect_cooldown=True,
        selected_only=True,
        include_open_positions=True,
        include_discovery=False,
    )
    save_fetch_state(state_path, cycle["fetch_state"])
    attempts = cycle["attempts"]
    return {
        "enabled": True,
        "attempts": len(attempts),
        "success": sum(1 for a in attempts if a.target_fetch_status == "SUCCESS"),
        "cooldown_skips": sum(
            1 for a in attempts if a.target_fetch_status == "SKIPPED_COOLDOWN_ACTIVE"
        ),
    }
from .models import (
    MIN_LIQUIDITY_USD,
    MarketState,
    Network,
    TokenMetadata,
    TokenRegistry,
    TradeType,
    WhaleActivity,
    append_whale_activity,
)
from .llm_config import (
    get_llm_provider,
    get_scan_gemini_decisions_stored,
    is_headless_data_collection,
    is_llm_audit_only_provider,
    is_ollama_provider_active,
    reset_ollama_scan_budget,
    reset_scan_llm_decision_counters,
)
from .models.predictor import (
    analyze_market_state,
    analyze_open_position,
    execute_trade_decision,
    normalize_execution_settings,
)
from .observability.audit_reasons import AuditReason
from .observability.llm_gate import evaluate_llm_short_circuit
from .observability.pipeline_audit import safe_record_pipeline_decision
from .observability.effective_settings import get_effective_settings
from .observability.actionability import build_candidate_from_pair, evaluate_and_execute_candidate

log = logging.getLogger("live")

_RAW_POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
try:
    from app.runtime.shutdown import clamp_scan_interval_seconds

    POLL_INTERVAL: float = clamp_scan_interval_seconds(_RAW_POLL_INTERVAL)
except Exception:
    POLL_INTERVAL = float(max(5, _RAW_POLL_INTERVAL if _RAW_POLL_INTERVAL > 0 else 5))
MIN_WHALE_SCORE: float = float(os.getenv("MIN_WHALE_SCORE", "0.30"))

DATA_DIR = Path(__file__).parent.parent / "data"
TRANSPARENCY_LOG_PATH = DATA_DIR / "token_transparency_log.json"

_CHAIN_MAP: dict[str, Network] = {
    "solana": Network.SOLANA,
    "ethereum": Network.ETHEREUM,
    "bsc": Network.BSC,
    "base": Network.BASE,
    "arbitrum": Network.ARBITRUM,
    "polygon": Network.POLYGON,
}

# Academic transparency — last scan passed vs dropped token arrays
_passed_tokens_log: list[dict[str, Any]] = []
_dropped_tokens_log: list[dict[str, Any]] = []
_last_scan_at: str | None = None


def get_token_transparency_logs() -> dict[str, Any]:
    return {
        "scan_at": _last_scan_at,
        "passed": list(_passed_tokens_log),
        "dropped": list(_dropped_tokens_log),
        "passed_count": len(_passed_tokens_log),
        "dropped_count": len(_dropped_tokens_log),
    }


def _persist_transparency_logs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = get_token_transparency_logs()
    with open(TRANSPARENCY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _shutdown_requested() -> bool:
    try:
        from app.runtime.shutdown import is_shutting_down

        return is_shutting_down()
    except Exception:
        return False


def _pair_row(
    pair: dict,
    *,
    whale_score: float | None = None,
    cluster_label: str = "",
    filter_status: str = "passed",
    drop_reason: str = "",
) -> dict[str, Any]:
    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    txns = (pair.get("txns") or {}).get("h24") or {}
    buys = int(txns.get("buys") or 0)
    sells = int(txns.get("sells") or 0)
    liq = float((pair.get("liquidity") or {}).get("usd") or 0)
    pc1h = float((pair.get("priceChange") or {}).get("h1") or 0)
    pc24 = float((pair.get("priceChange") or {}).get("h24") or 0)
    vol = float((pair.get("volume") or {}).get("h24") or 0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract_address": pair.get("pairAddress") or "",
        "symbol": f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}",
        "base_symbol": base.get("symbol", "?"),
        "network": (pair.get("chainId") or "unknown").lower(),
        "price_usd": float(pair.get("priceUsd") or 0),
        "liquidity_usd": liq,
        "volume_24h": vol,
        "price_change_1h": pc1h,
        "price_change_24h": pc24,
        "buy_ratio": round(buys / max(buys + sells, 1), 4),
        "whale_score": round(whale_score, 4) if whale_score is not None else None,
        "cluster_label": cluster_label,
        "filter_status": filter_status,
        "drop_reason": drop_reason,
    }


def _evaluate_drop_reason(pair: dict, min_liq: float, min_score: float) -> str | None:
    addr = (pair.get("pairAddress") or "").strip()
    if not addr:
        return "Missing pair address"
    if not pair.get("priceUsd"):
        return "Missing price data"
    liq = float((pair.get("liquidity") or {}).get("usd") or 0)
    if liq < min_liq:
        return f"Liquidity under ${min_liq:,.0f} (actual ${liq:,.0f})"
    whale_score = compute_whale_score(pair)
    if whale_score < min_score:
        return f"Low whale score {whale_score:.3f} below minimum {min_score:.3f}"
    return None


def _parse_pair(
    pair: dict,
    min_liquidity: float,
    *,
    bypass_liquidity: bool = False,
) -> tuple[TokenMetadata, MarketState] | None:
    addr = (pair.get("pairAddress") or "").strip()
    base = pair.get("baseToken") or {}
    chain = (pair.get("chainId") or "unknown").lower()
    liq = float((pair.get("liquidity") or {}).get("usd") or 0)
    txns = (pair.get("txns") or {}).get("h24") or {}

    if not addr:
        return None
    if not bypass_liquidity and liq < min_liquidity:
        return None

    token = TokenMetadata(
        contract_address=addr,
        symbol=base.get("symbol", "?"),
        name=base.get("name", base.get("symbol", "?")),
        network=_CHAIN_MAP.get(chain, Network.UNKNOWN),
    )
    try:
        state = MarketState(
            contract_address=addr,
            price_usd=float(pair.get("priceUsd") or 0),
            liquidity_usd=liq,
            volume_24h=float((pair.get("volume") or {}).get("h24") or 0),
            price_change_24h=float((pair.get("priceChange") or {}).get("h24") or 0),
            price_change_1h=float((pair.get("priceChange") or {}).get("h1") or 0),
            txns_buys_24h=int(txns.get("buys") or 0),
            txns_sells_24h=int(txns.get("sells") or 0),
        )
    except ValueError as exc:
        log.debug("Dropped %s: %s", token.symbol, exc)
        return None

    return token, state


async def _evaluate_whale_event(
    pair: dict,
    state: MarketState,
    token: TokenMetadata,
    symbol: str,
    chain: str,
    whale_score: float,
    cluster_label: str,
    sentiment_score: float,
    settings: dict[str, Any],
    *,
    coin_id: int,
    alert_type: str,
) -> dict[str, Any] | None:
    """Gemini inference ONLY for high-conviction whale-like market flow events."""
    settings = normalize_execution_settings(settings)
    threshold = float(settings.get("llm_score_threshold", 0.50))
    if whale_score < threshold:
        return None

    from .analytics.features import ClusterLabel

    try:
        cluster_enum = ClusterLabel(cluster_label)
    except ValueError:
        cluster_enum = await resolve_cluster_label(token, pair)
        cluster_label = cluster_enum.value

    metrics = build_feature_row(pair, state, cluster_enum, whale_score)
    open_positions = get_paper_trader().get_positions("OPEN")
    decision, decision_id = await analyze_market_state(
        metrics,
        cluster_label,
        sentiment_score,
        open_positions=open_positions,
        coin_id=coin_id,
        trigger_type=f"whale_like_event:{alert_type}",
    )

    log.info(
        "LLM %s — %s | %s strategy=%s risk=%d conf=%.2f (whale-like event: %s)",
        token.symbol,
        decision.decision,
        cluster_label,
        decision.strategy_type,
        decision.risk_score,
        decision.confidence,
        alert_type,
    )

    coin = {
        "symbol": symbol,
        "chain": chain,
        "price_usd": state.price_usd,
        "pair_address": token.contract_address,
        "coin_id": coin_id,
        "decision_ref_id": decision_id,
    }
    provider = get_llm_provider()
    # AE19 primary guard: LLM outputs are audit/shadow only — never execute.
    if is_llm_audit_only_provider(provider):
        log.info(
            "LLM audit-only decision stored; execution not attempted provider=%s action=%s",
            provider,
            decision.decision,
        )
        return {
            "ok": True,
            "decision": decision.decision,
            "symbol": symbol,
            "execution_attempted": False,
            "audit_only": True,
            "reason": "LLM_PROVIDER_AUDIT_ONLY",
            "provider": provider,
            "decision_ref_id": decision_id,
        }

    trader = get_paper_trader()
    if decision.decision == "BUY":
        alloc_usd = trader.compute_strategy_notional(decision.strategy_type)
        log.info(
            "ALLOC %s strategy=%s equity=$%.2f → size=$%.2f",
            symbol,
            decision.strategy_type,
            trader.get_wallet_summary().get("total_equity_usd", 0),
            alloc_usd,
        )
    exec_result = execute_trade_decision(
        decision,
        coin,
        cluster_label,
        settings,
        cur_price=state.price_usd,
        decision_ref_id=decision_id,
        coin_id=coin_id,
        provider=provider,
    )
    if exec_result.get("audit_only"):
        return exec_result
    if decision.decision == "BUY" and not exec_result.get("ok"):
        log.error(
            "AGENT_BUY failed %s strategy=%s: %s",
            symbol,
            decision.strategy_type,
            exec_result.get("error"),
        )
    elif decision.decision == "BUY" and exec_result.get("ok"):
        pos = exec_result.get("position") or {}
        log.info(
            "AGENT_BUY opened #%s %s strategy=%s size=$%.2f @ $%.6f",
            pos.get("id"),
            symbol,
            exec_result.get("strategy_type"),
            exec_result.get("size_usd") or pos.get("size_usd", 0),
            state.price_usd,
        )
        log.info("paper trades inserted: 1")
    elif decision.decision == "SELL" and exec_result.get("ok"):
        log.info("AGENT_SELL closed position %s @ $%.6f", symbol, state.price_usd)
        log.info("paper trades inserted: 1")
    return exec_result


async def _manage_open_positions(
    market_entries: list[dict[str, Any]],
    cluster_by_pair: dict[str, str],
    sentiment_score: float,
    settings: dict[str, Any],
) -> None:
    """Re-evaluate open positions each scan so Gemini can issue SELL exits."""
    settings = normalize_execution_settings(settings)
    trader = get_paper_trader()
    for pos in trader.get_positions("OPEN"):
        symbol = pos.get("symbol", "")
        pair_address = str(pos.get("pair_address") or "").strip()
        coin_id = pos.get("coin_id")
        market_entry = next(
            (
                entry for entry in market_entries
                if str(entry.get("pair_address") or "").strip() == pair_address and pair_address
            ),
            None,
        )
        if market_entry is None and coin_id is not None:
            market_entry = next(
                (entry for entry in market_entries if entry.get("coin_id") == coin_id),
                None,
            )
        cur_price = float(market_entry["price_usd"]) if market_entry else None
        cluster = cluster_by_pair.get(
            pair_address,
            pos.get("cluster_label", "OPPORTUNISTIC_SPECULATIVE"),
        )
        if coin_id is None and pair_address:
            coin_rec = db.get_coin_by_pair_address(pair_address)
            coin_id = coin_rec["id"] if coin_rec else None
        decision, decision_id = await analyze_open_position(
            pos, cur_price or float(pos.get("entry_price", 0)), cluster, sentiment_score, coin_id=coin_id
        )
        if decision.decision != "SELL":
            continue
        provider = get_llm_provider()
        # AE19 primary guard: LLM SELL is analysis only — never close paper positions.
        if is_llm_audit_only_provider(provider):
            log.info(
                "LLM audit-only decision stored; execution not attempted provider=%s action=%s",
                provider,
                decision.decision,
            )
            continue
        coin = {
            "symbol": symbol,
            "chain": pos.get("chain", "solana"),
            "price_usd": cur_price,
            "pair_address": pair_address,
            "coin_id": coin_id,
            "decision_ref_id": decision_id,
        }
        exec_result = execute_trade_decision(
            decision,
            coin,
            cluster,
            settings,
            cur_price=cur_price,
            decision_ref_id=decision_id,
            coin_id=coin_id,
            pair_address=pair_address,
            position_id=pos.get("id"),
            provider=provider,
        )
        if exec_result.get("audit_only"):
            continue
        if exec_result.get("ok"):
            log.info(
                "AGENT_SELL exit #%s %s net_roi=%.2f%%",
                pos["id"],
                symbol,
                (exec_result.get("position") or {}).get("net_roi_pct", 0) * 100,
            )


async def analyze_contract_address(
    contract_address: str,
    chain: str | None = None,
    *,
    force_llm: bool = True,
    bypass_filters: bool = False,
) -> dict[str, Any]:
    """Force AI analysis for a user-supplied contract (watchlist)."""
    from . import database as db

    settings = normalize_execution_settings(db.get_settings())
    min_liq = float(settings.get("min_liquidity_usd", MIN_LIQUIDITY_USD))
    sentiment_score = await fetch_rss_sentiment()
    addr = contract_address.strip()

    pairs = await search_pairs(addr)
    if chain:
        filtered = [p for p in pairs if (p.get("chainId") or "").lower() == chain.lower()]
        if filtered:
            pairs = filtered
    if not pairs:
        return {"ok": False, "error": f"No DexScreener pair for {addr}", "contract_address": addr}

    pair = pairs[0]
    filter_note = ""
    if bypass_filters:
        result = _parse_pair(pair, min_liq, bypass_liquidity=True)
        filter_note = "bypassed_standard_filters"
    else:
        result = _parse_pair(pair, min_liq)
        if result is None:
            drop = _evaluate_drop_reason(pair, min_liq, float(settings.get("min_whale_score", MIN_WHALE_SCORE)))
            return {
                "ok": False,
                "error": drop or "Pair below minimum liquidity threshold",
                "contract_address": addr,
            }

    if result is None:
        return {"ok": False, "error": "Invalid market state for pair", "contract_address": addr}

    token, state = result
    cluster = await resolve_cluster_label(token, pair)
    whale_score = compute_whale_score(pair)
    symbol = (
        f"{(pair.get('baseToken') or {}).get('symbol', '?')}/"
        f"{(pair.get('quoteToken') or {}).get('symbol', '?')}"
    )
    chain_id = (pair.get("chainId") or "unknown").lower()

    db.upsert_coin({
        "symbol": symbol,
        "base_symbol": (pair.get("baseToken") or {}).get("symbol", "?"),
        "quote_symbol": (pair.get("quoteToken") or {}).get("symbol", "?"),
        "chain": chain_id,
        "pair_address": token.contract_address,
        "price_usd": state.price_usd,
        "volume_24h": state.volume_24h,
        "liquidity_usd": state.liquidity_usd,
        "price_change_24h": state.price_change_24h,
        "price_change_6h": float((pair.get("priceChange") or {}).get("h6") or 0),
        "price_change_1h": state.price_change_1h,
        "fdv": float(pair.get("fdv") or 0) or None,
        "txns_24h": state.txns_buys_24h + state.txns_sells_24h,
        "whale_score": whale_score,
        "boosted": bool((pair.get("boosts") or {}).get("active") or 0),
        "dex_url": pair.get("url"),
        "cluster_label": cluster.value,
    })

    decision = None
    execution: dict[str, Any] | None = None
    if force_llm:
        metrics = build_feature_row(pair, state, cluster, whale_score)
        coin_rec = db.get_coin_by_pair_address(token.contract_address)
        coin_id = coin_rec["id"] if coin_rec else None
        open_positions = get_paper_trader().get_positions("OPEN")
        decision, decision_id = await analyze_market_state(
            metrics,
            cluster.value,
            sentiment_score,
            open_positions=open_positions,
            coin_id=coin_id,
            trigger_type="watchlist_manual",
        )
        get_paper_trader().set_market_prices([
            {
                "pair_address": token.contract_address,
                "coin_id": coin_id,
                "price_usd": state.price_usd,
            }
        ])
        coin = {
            "symbol": symbol,
            "chain": chain_id,
            "price_usd": state.price_usd,
            "pair_address": token.contract_address,
            "coin_id": coin_id,
            "decision_ref_id": decision_id,
        }
        provider = get_llm_provider()
        if is_llm_audit_only_provider(provider):
            log.info(
                "LLM audit-only decision stored; execution not attempted provider=%s action=%s",
                provider,
                decision.decision,
            )
            execution = {
                "ok": True,
                "decision": decision.decision,
                "symbol": symbol,
                "execution_attempted": False,
                "audit_only": True,
                "reason": "LLM_PROVIDER_AUDIT_ONLY",
                "provider": provider,
            }
        else:
            execution = execute_trade_decision(
                decision,
                coin,
                cluster.value,
                settings,
                cur_price=state.price_usd,
                decision_ref_id=decision_id,
                coin_id=coin_id,
                provider=provider,
            )

    return {
        "ok": True,
        "contract_address": addr,
        "symbol": symbol,
        "cluster_label": cluster.value,
        "whale_score": whale_score,
        "filter_note": filter_note,
        "decision": decision.model_dump() if decision else None,
        "execution": execution,
        "wallet": get_paper_trader().get_wallet_summary(),
    }


async def scan_once(local_registry: TokenRegistry) -> int:
    global _passed_tokens_log, _dropped_tokens_log, _last_scan_at

    if _shutdown_requested():
        log.info("scan_once skipped because shutdown is active")
        return 0

    settings = normalize_execution_settings(db.get_settings())
    min_liq = float(settings.get("min_liquidity_usd", MIN_LIQUIDITY_USD))
    min_score = float(settings.get("min_whale_score", MIN_WHALE_SCORE))
    reset_scan_llm_decision_counters()
    scan_id = new_scan_id()
    reset_ollama_scan_budget()

    from .analytics.sentiment import archive_rss_sentiment

    try:
        sentiment_score = await archive_rss_sentiment()
    except Exception as exc:
        log.error("RSS archival failed: %s", exc, exc_info=True)
        sentiment_score = 0.0

    if _shutdown_requested():
        log.info("scan_once stopped after RSS due to shutdown")
        return 0

    # Priority 0A/0B exact-pair Selected/Clean + open-position mark prices BEFORE trending.
    try:
        selected_stats = await asyncio.to_thread(_run_selected_clean_collection_cycle)
        log.info(
            "Selected/Clean exact-pair cycle: attempts=%s success=%s cooldown_skips=%s",
            selected_stats.get("attempts"),
            selected_stats.get("success"),
            selected_stats.get("cooldown_skips"),
        )
    except Exception as exc:
        log.error("Selected/Clean collection cycle failed: %s", exc, exc_info=True)

    if _shutdown_requested():
        log.info("scan_once stopped before trending due to shutdown")
        return 0

    pairs = await get_trending_pairs()

    if _shutdown_requested():
        log.info("scan_once stopped after trending fetch due to shutdown")
        return 0

    try:
        archive_dexscreener_search(
            "trending_batch",
            {"pair_count": len(pairs), "scan_id": scan_id},
            source_type="trending_summary",
        )
    except Exception as exc:
        log.error("Dexscreener batch archival failed: %s", exc, exc_info=True)

    log.info(
        "Scan %s — %d pairs | RSS=%.3f | LLM on whale-like events only",
        scan_id,
        len(pairs),
        sentiment_score,
    )

    passed_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    csv_events: list[WhaleActivity] = []
    coins_upserted = 0
    snapshots_inserted = 0
    scan_market_entries: list[dict[str, Any]] = []
    cluster_by_pair: dict[str, str] = {}

    for pair in pairs:
        if _shutdown_requested():
            log.info("scan_once pair loop stopped due to shutdown")
            break
        base = pair.get("baseToken") or {}
        quote = pair.get("quoteToken") or {}
        symbol = f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}"
        try:
            drop_reason = _evaluate_drop_reason(pair, min_liq, min_score)
            whale_score = compute_whale_score(pair)

            if drop_reason:
                row = _pair_row(
                    pair,
                    whale_score=whale_score,
                    filter_status="dropped",
                    drop_reason=drop_reason,
                )
                dropped_rows.append(row)
                drop_audit = []
                if "Liquidity" in drop_reason:
                    drop_audit.append(AuditReason.BELOW_LIQUIDITY_THRESHOLD.value)
                if "whale score" in drop_reason.lower():
                    drop_audit.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
                if "Missing" in drop_reason:
                    drop_audit.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
                safe_record_pipeline_decision(
                    pair_address=(pair.get("pairAddress") or ""),
                    chain=(pair.get("chainId") or "unknown").lower(),
                    symbol=symbol,
                    audit_reasons=drop_audit or [AuditReason.SETTINGS_BLOCKED.value],
                    scan_id=scan_id,
                    whale_score=whale_score,
                    stage="filter_dropped",
                )
                await asyncio.to_thread(
                    persist_pair_pipeline,
                    pair,
                    scan_id=scan_id,
                    source_query="trending",
                    filter_status="dropped",
                    drop_reason=drop_reason,
                    sentiment_score=sentiment_score,
                )
                continue

            result = _parse_pair(pair, min_liq)
            if result is None:
                dropped_rows.append(
                    _pair_row(
                        pair,
                        whale_score=whale_score,
                        filter_status="dropped",
                        drop_reason="Invalid market state (parse error)",
                    )
                )
                await asyncio.to_thread(
                    persist_pair_pipeline,
                    pair,
                    scan_id=scan_id,
                    source_query="trending",
                    filter_status="dropped",
                    drop_reason="Invalid market state (parse error)",
                    sentiment_score=sentiment_score,
                )
                continue

            token, state = result
            local_registry.register(token)
            cluster = await resolve_cluster_label(token, pair)
            chain = (pair.get("chainId") or "unknown").lower()
            cluster_by_pair[token.contract_address] = cluster.value

            persisted = await asyncio.to_thread(
                persist_pair_pipeline,
                pair,
                scan_id=scan_id,
                source_query="trending",
                filter_status="passed",
                cluster_label=cluster.value,
                sentiment_score=sentiment_score,
            )
            if persisted and not persisted.get("dropped"):
                coins_upserted += 1
                snapshots_inserted += 1

            passed_rows.append(
                _pair_row(
                    pair,
                    whale_score=whale_score,
                    cluster_label=cluster.value,
                    filter_status="passed",
                )
            )

            coin_id = persisted.get("coin_id") if persisted else None
            alert = persisted.get("alert") if persisted else None
            scan_market_entries.append({
                "pair_address": token.contract_address,
                "coin_id": coin_id,
                "price_usd": state.price_usd,
                "symbol": symbol,
            })

            llm_threshold = float(settings.get("llm_score_threshold", 0.50))
            signal_action = persisted.get("signal_action") if persisted else None
            signal_score = float(persisted.get("whale_score", whale_score)) if persisted else whale_score
            open_positions = get_paper_trader().get_positions("OPEN")
            has_open = any(
                str(p.get("pair_address") or "").strip() == token.contract_address
                for p in open_positions
            )

            eff = get_effective_settings(settings)
            economic_enabled = bool(settings.get("economic_gate_enabled", False))

            if economic_enabled:
                cluster_is_default = cluster.value == "OPPORTUNISTIC_SPECULATIVE"
                candidate = build_candidate_from_pair(
                    pair,
                    whale_score=whale_score,
                    signal_action=signal_action or "WATCH",
                    signal_score=signal_score,
                    coin_id=coin_id,
                    chain=chain,
                    symbol=symbol,
                    price=state.price_usd,
                    liquidity_usd=state.liquidity_usd,
                    alert_type=alert["alert_type"] if alert else None,
                    cluster_label=cluster.value,
                    cluster_is_default=cluster_is_default,
                    sentiment_score=sentiment_score,
                    sentiment_available=sentiment_score != 0.0,
                    has_open_position=has_open,
                    scan_id=scan_id,
                    settings_hash=eff.settings_hash,
                )
                try:
                    action_result = await evaluate_and_execute_candidate(
                        candidate,
                        pair=pair,
                        settings=settings,
                        alert=alert,
                    )
                    if action_result.get("action") not in ("PAPER_BUY_EXECUTED", "SKIPPED"):
                        safe_record_pipeline_decision(
                            pair_address=token.contract_address,
                            coin_id=coin_id,
                            chain=chain,
                            symbol=symbol,
                            audit_reasons=action_result.get("reasons", []),
                            scan_id=scan_id,
                            signal_action=signal_action,
                            alert_type=alert["alert_type"] if alert else None,
                            whale_score=whale_score,
                            current_execution_price=state.price_usd,
                            stage=f"economic_{action_result.get('action', 'unknown').lower()}",
                            decision_trace_id=candidate.decision_trace_id,
                            settings_hash=eff.settings_hash,
                        )
                except Exception as exc:
                    log.error("Economic gate eval failed for %s: %s", symbol, exc, exc_info=True)
            else:
                should_llm, llm_audit_reasons = evaluate_llm_short_circuit(
                    alert=alert,
                    coin_id=coin_id,
                    whale_score=whale_score,
                    llm_threshold=llm_threshold,
                    signal_action=signal_action,
                    alert_type=alert["alert_type"] if alert else None,
                    has_open_position=has_open,
                    price_usd=state.price_usd,
                    pair_address=token.contract_address,
                    trading_mode=str(settings.get("trading_mode", "DEMO")),
                    auto_execution_enabled=bool(settings.get("auto_execution_enabled", True)),
                    enforce_risk_gate=bool(settings.get("enforce_risk_gate", False)),
                )
                if should_llm:
                    try:
                        get_paper_trader().set_market_prices(
                            scan_market_entries,
                            price_timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        await _evaluate_whale_event(
                            pair,
                            state,
                            token,
                            symbol,
                            chain,
                            whale_score,
                            cluster.value,
                            sentiment_score,
                            settings,
                            coin_id=coin_id,
                            alert_type=alert["alert_type"],
                        )
                    except Exception as exc:
                        log.error("Gemini/trade eval failed for %s: %s", symbol, exc, exc_info=True)
                else:
                    safe_record_pipeline_decision(
                        pair_address=token.contract_address,
                        coin_id=coin_id,
                        chain=chain,
                        symbol=symbol,
                        audit_reasons=llm_audit_reasons or [AuditReason.LLM_THRESHOLD_NOT_MET.value],
                        scan_id=scan_id,
                        signal_action=signal_action,
                        alert_type=alert["alert_type"] if alert else None,
                        whale_score=whale_score,
                        current_execution_price=state.price_usd,
                        stage="llm_skipped",
                    )

            if alert:
                trade_type = (
                    TradeType.BUY
                    if alert["alert_type"] in ("LARGE_BUY", "ACCUMULATION", "PUMP_SIGNAL")
                    else TradeType.SELL
                )
                csv_events.append(
                    WhaleActivity(
                        token_contract_address=token.contract_address,
                        symbol=token.symbol,
                        network=token.network,
                        wallet_address="pool",
                        trade_type=trade_type,
                        transaction_size_usd=state.volume_24h,
                        price_usd_at_trade=state.price_usd,
                        liquidity_usd_at_trade=state.liquidity_usd,
                        volume_24h_at_trade=state.volume_24h,
                        price_change_24h=state.price_change_24h,
                        buy_ratio=state.buy_ratio,
                        whale_score=whale_score,
                        alert_type=alert["alert_type"],
                        cluster_label=cluster.value,
                    )
                )
        except Exception as exc:
            log.error("Pair scan failed for %s: %s", symbol, exc, exc_info=True)
            continue

    if _shutdown_requested():
        log.info("scan_once stopped before position management due to shutdown")
        return len(csv_events)

    _last_scan_at = datetime.now(timezone.utc).isoformat()
    trader = get_paper_trader()
    trader.set_market_prices(scan_market_entries, price_timestamp=_last_scan_at)

    try:
        await _manage_open_positions(scan_market_entries, cluster_by_pair, sentiment_score, settings)
    except Exception as exc:
        log.error("Open position management failed: %s", exc, exc_info=True)

    _passed_tokens_log = passed_rows
    _dropped_tokens_log = dropped_rows
    _persist_transparency_logs()

    log.info(
        "Scan done — coins upserted: %d | snapshots inserted: %d | passed: %d | dropped: %d | "
        "gemini decisions stored: %d | whale CSV events: %d",
        coins_upserted,
        snapshots_inserted,
        len(passed_rows),
        len(dropped_rows),
        get_scan_gemini_decisions_stored(),
        len(csv_events),
    )
    if csv_events:
        try:
            await append_whale_activity(csv_events)
            log.info("Logged %d aggregate whale-like events → whale_trades_log.csv", len(csv_events))
        except Exception as exc:
            log.error("Whale CSV append failed: %s", exc, exc_info=True)

    storage_stats = db.get_storage_stats()
    log.info(
        "Storage row counts — coins=%d snapshots=%d raw=%d signals=%d whale_alerts=%d "
        "paper_trades=%d gemini=%d sentiment=%d pipeline_audit=%d",
        storage_stats.get("coins", {}).get("rows", 0),
        storage_stats.get("market_snapshots", {}).get("rows", 0),
        storage_stats.get("raw_provider_payloads", {}).get("rows", 0),
        storage_stats.get("signals", {}).get("rows", 0),
        storage_stats.get("whale_alerts", {}).get("rows", 0),
        storage_stats.get("paper_trades", {}).get("rows", 0),
        storage_stats.get("gemini_decisions", {}).get("rows", 0),
        storage_stats.get("sentiment_records", {}).get("rows", 0),
        storage_stats.get("pipeline_audit", {}).get("rows", 0),
    )

    return len(csv_events)


async def watcher_loop() -> None:
    import time

    from app.runtime.shutdown import (
        clamp_scan_interval_seconds,
        is_shutting_down,
        should_skip_network,
    )

    local_registry = TokenRegistry()
    poll_seconds = clamp_scan_interval_seconds(POLL_INTERVAL)
    if is_headless_data_collection():
        llm_mode = "HEADLESS (no LLM)"
        log.info("Headless Data Collection mode active: Gemini API calls disabled.")
    elif is_ollama_provider_active():
        llm_mode = f"Ollama ON ({get_llm_provider()})"
    elif get_llm_provider() == "none":
        llm_mode = "LLM_PROVIDER=none"
    else:
        llm_mode = "Gemini agent ON"
    log.info(
        "Whale watcher | poll=%.0fs | min_liq=$%.0f | %s",
        poll_seconds, MIN_LIQUIDITY_USD, llm_mode,
    )
    while True:
        if is_shutting_down() or should_skip_network(context="watcher_loop_start"):
            log.info("background scanner stopped")
            break
        t0 = time.monotonic()
        try:
            await scan_once(local_registry)
        except asyncio.CancelledError:
            log.info("background scanner stopped")
            raise
        except Exception as exc:
            log.error("Scan failed: %s", exc, exc_info=True)
        if is_shutting_down():
            log.info("background scanner stopped")
            break
        remaining = poll_seconds - (time.monotonic() - t0)
        # Forbid "Next scan in 0s" — clamp non-positive remaining to full poll interval
        sleep_for = clamp_scan_interval_seconds(remaining if remaining > 0 else poll_seconds)
        log.info("Next scan in %.0fs | registry=%d", sleep_for, len(local_registry))
        try:
            await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            log.info("background scanner stopped")
            raise
