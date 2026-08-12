"""
Build compact Gemini context from SQLite historical memory for a single coin.
Only invoked for high-conviction whale-like events — not for every scanned token.
"""
from __future__ import annotations

import json
from typing import Any

from . import database as db
from .execution.paper import get_paper_trader


def build_gemini_context(coin_id: int) -> dict[str, Any]:
    """
    Aggregate structured context for one coin from durable storage:
    snapshots, whale-like alerts, sentiment, paper trades, prior Gemini decisions.
    """
    coin = db.get_coin_by_id(coin_id)
    if not coin:
        return {"error": f"coin_id {coin_id} not found"}

    snapshots = db.get_market_snapshots(coin_id, limit=30)
    latest = snapshots[-1] if snapshots else None

    prior_decisions = db.get_gemini_decisions(limit=15, coin_id=coin_id)
    paper_trades = db.get_trades(limit=15, coin_id=coin_id)
    whale_alerts = db.get_whale_alerts(limit=10, coin_id=coin_id)
    signals = db.get_signals(limit=5, coin_id=coin_id)
    sentiment = db.get_sentiment_records(limit=5)

    open_positions = [
        p for p in get_paper_trader().get_positions("OPEN")
        if p.get("symbol") == coin.get("symbol")
        or p.get("pair_address") == coin.get("pair_address")
    ]

    churn_guard = _build_churn_guard(prior_decisions, paper_trades)

    compact_snapshots = [
        {
            "ts": s.get("timestamp"),
            "price": s.get("price"),
            "liquidity": s.get("liquidity"),
            "volume_24h": s.get("volume_24h"),
            "whale_score": s.get("whale_score"),
            "buy_ratio": s.get("buy_ratio"),
            "pc_h1": s.get("price_change_h1"),
            "pc_h24": s.get("price_change_h24"),
        }
        for s in snapshots[-10:]
    ]

    compact_decisions = [
        {
            "ts": d.get("timestamp"),
            "action": d.get("action"),
            "strategy": d.get("strategy_type"),
            "confidence": d.get("confidence"),
            "rationale": (d.get("rationale") or "")[:200],
            "outcome_pnl": d.get("outcome_pnl"),
            "outcome_status": d.get("outcome_status"),
        }
        for d in prior_decisions
    ]

    compact_trades = [
        {
            "ts": t.get("timestamp"),
            "side": t.get("side"),
            "value": t.get("value"),
            "pnl": t.get("pnl"),
            "net_roi_pct": t.get("net_roi_pct"),
            "reason": t.get("reason"),
        }
        for t in paper_trades
    ]

    compact_alerts = [
        {
            "ts": a.get("timestamp"),
            "type": a.get("alert_type"),
            "whale_score": a.get("whale_score"),
            "is_wallet_level": a.get("is_real_wallet_level"),
            "terminology": a.get("terminology"),
            "description": (a.get("description") or "")[:160],
        }
        for a in whale_alerts
    ]

    return {
        "coin_id": coin_id,
        "symbol": coin.get("symbol"),
        "pair_address": coin.get("pair_address"),
        "chain": coin.get("chain"),
        "latest_snapshot": latest,
        "snapshot_series": compact_snapshots,
        "whale_like_alerts": compact_alerts,
        "recent_signals": [
            {"type": s.get("signal_type"), "score": s.get("score"), "reason": s.get("reason")}
            for s in signals
        ],
        "global_sentiment_sample": [
            {"source": s.get("source"), "title": s.get("title"), "score": s.get("sentiment_score")}
            for s in sentiment[:3]
        ],
        "prior_gemini_decisions": compact_decisions,
        "app_paper_trades": compact_trades,
        "open_paper_positions": open_positions,
        "churn_guard": churn_guard,
        "wallet_summary": get_paper_trader().get_wallet_summary(),
        "data_notes": {
            "whale_alerts": "aggregate pool-level flow unless is_wallet_level=true",
            "paper_trades": "app-generated simulated trades only",
        },
    }


def _build_churn_guard(
    prior_decisions: list[dict[str, Any]],
    paper_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize recent buy→sell→rebuy patterns to inject anti-churn guidance."""
    if not prior_decisions and not paper_trades:
        return {"status": "no_history", "guidance": "No prior decisions for this token."}

    last_decisions = prior_decisions[:5]
    last_trades = paper_trades[:5]

    recent_sells = [d for d in last_decisions if d.get("action") == "SELL"]
    recent_buys = [d for d in last_decisions if d.get("action") == "BUY"]
    losing_sells = [t for t in last_trades if t.get("side") == "sell" and (t.get("pnl") or 0) < 0]

    guidance_parts: list[str] = []
    if recent_sells and recent_buys:
        guidance_parts.append(
            "Recent BUY and SELL decisions exist for this token. "
            "Avoid panic-selling on fee-noise (<3% moves) and do NOT re-BUY immediately "
            "after a loss-exit unless whale-like flow shows a materially new setup."
        )
    if losing_sells:
        guidance_parts.append(
            f"Last {len(losing_sells)} closed paper trade(s) were net losers — "
            "require stronger conviction before another BUY."
        )
    if recent_sells and not guidance_parts:
        guidance_parts.append("Recent SELL recorded — justify any new BUY with new flow evidence.")

    return {
        "status": "active",
        "recent_sell_count": len(recent_sells),
        "recent_buy_count": len(recent_buys),
        "recent_losing_exits": len(losing_sells),
        "guidance": " ".join(guidance_parts) or "Review prior decisions before acting.",
        "last_actions": [{"action": d.get("action"), "ts": d.get("timestamp")} for d in last_decisions[:3]],
    }


def context_prompt_summary(context: dict[str, Any]) -> str:
    """One-line summary for gemini_decisions.prompt_summary."""
    sym = context.get("symbol", "?")
    n_snaps = len(context.get("snapshot_series") or [])
    n_prior = len(context.get("prior_gemini_decisions") or [])
    return f"{sym} | {n_snaps} snapshots | {n_prior} prior decisions | churn={context.get('churn_guard', {}).get('status')}"
