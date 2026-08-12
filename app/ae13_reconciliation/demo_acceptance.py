"""AE13 DEMO_ACCEPTANCE_MODE — bounded paper/demo trade for UI/ledger wiring proof.

Fail-closed: rejected unless DEMO/PAPER mode, live disabled, no wallet,
DEMO_ACCEPTANCE_MODE explicitly enabled, and required evidence flags set.
Never routes to live execution.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("ae13.demo_acceptance")

DEMO_ACCEPTANCE_NOTIONAL_USD = 25.0
DEMO_ACCEPTANCE_SYMBOL = "AE13DEMO/USD"
DEMO_ACCEPTANCE_PAIR = "ae13_demo_acceptance_pair"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_demo_acceptance_guard(
    *,
    trading_mode: str,
    live_trading_enabled: bool,
    wallet_configured: bool,
    demo_acceptance_mode_enabled: bool,
    order_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed gate for demo acceptance orders."""
    flags = order_flags or {}
    mode = str(trading_mode or "").upper()
    reasons: list[str] = []

    if mode not in ("DEMO", "PAPER"):
        reasons.append(f"application_mode_not_demo_or_paper:{mode or 'MISSING'}")
    if live_trading_enabled:
        reasons.append("live_trading_enabled_true")
    if wallet_configured:
        reasons.append("wallet_configured_true")
    if not demo_acceptance_mode_enabled:
        reasons.append("demo_acceptance_mode_not_enabled")

    required_true = (
        "demo_acceptance_only",
        "not_live_approved",
        "not_profitability_evidence",
        "not_strategy_evidence",
    )
    for key in required_true:
        if flags and flags.get(key) is not True:
            reasons.append(f"order_flag_missing_or_false:{key}")

    allowed = len(reasons) == 0
    return {
        "allowed": allowed,
        "rejected": not allowed,
        "reasons": reasons,
        "checked_at_utc": _utc_now(),
        "trading_mode": mode,
        "live_trading_enabled": bool(live_trading_enabled),
        "wallet_configured": bool(wallet_configured),
        "demo_acceptance_mode_enabled": bool(demo_acceptance_mode_enabled),
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
    }


def _pick_demo_coin() -> dict[str, Any]:
    """Prefer a real DB coin so PaperTrader fill-price guards accept the order."""
    try:
        from app import database as db

        coins = db.get_coins(limit=20, sort_by="last_seen")
        for coin in coins:
            price = coin.get("latest_price") or coin.get("price_usd")
            pair = coin.get("pair_address")
            cid = coin.get("id") or coin.get("coin_id")
            if pair and cid is not None and price and float(price) > 0:
                return {
                    "id": int(cid),
                    "coin_id": int(cid),
                    "symbol": coin.get("symbol") or DEMO_ACCEPTANCE_SYMBOL,
                    "chain": coin.get("chain") or "solana",
                    "pair_address": str(pair),
                    "token_address": coin.get("token_address"),
                    "price_usd": float(price),
                    "latest_price": float(price),
                    "name": coin.get("name") or "AE13 Demo Acceptance",
                }
    except Exception as exc:  # noqa: BLE001
        log.warning("demo acceptance coin lookup failed: %s", exc)

    # Last-resort synthetic — may still fail paper fill guards without coin_id
    return {
        "id": -913,
        "coin_id": -913,
        "symbol": DEMO_ACCEPTANCE_SYMBOL,
        "chain": "solana",
        "pair_address": DEMO_ACCEPTANCE_PAIR,
        "token_address": "ae13_demo_acceptance_token",
        "price_usd": 1.0,
        "latest_price": 1.0,
        "name": "AE13 Demo Acceptance",
    }


def create_demo_acceptance_order(
    *,
    trading_mode: str,
    live_trading_enabled: bool,
    wallet_configured: bool,
    demo_acceptance_mode_enabled: bool,
    settings: dict[str, Any] | None = None,
    notional_usd: float = DEMO_ACCEPTANCE_NOTIONAL_USD,
    execute: bool = True,
) -> dict[str, Any]:
    """Create one bounded paper/demo acceptance trade via PaperTrader when allowed."""
    order_flags = {
        "demo_acceptance_only": True,
        "not_strategy_evidence": True,
        "not_profitability_evidence": True,
        "not_live_approved": True,
        "paper_demo_only": True,
    }
    guard = evaluate_demo_acceptance_guard(
        trading_mode=trading_mode,
        live_trading_enabled=live_trading_enabled,
        wallet_configured=wallet_configured,
        demo_acceptance_mode_enabled=demo_acceptance_mode_enabled,
        order_flags=order_flags,
    )
    audit: dict[str, Any] = {
        "phase": "AE13_DEMO_ACCEPTANCE",
        "guard": guard,
        "order_flags": order_flags,
        "created_at_utc": _utc_now(),
        "execute_requested": bool(execute),
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
    }

    if not guard["allowed"]:
        audit["status"] = "REJECTED"
        audit["rejection_reasons"] = guard["reasons"]
        log.warning("DEMO_ACCEPTANCE_MODE order rejected: %s", guard["reasons"])
        return audit

    if not execute:
        audit["status"] = "GUARD_PASS_NO_EXECUTE"
        return audit

    from app.execution.paper import get_paper_trader

    trader = get_paper_trader()
    wallet_before = trader.get_wallet_summary()
    coin = _pick_demo_coin()
    trader.set_market_prices(
        [
            {
                "pair_address": coin.get("pair_address"),
                "coin_id": coin.get("coin_id") or coin.get("id"),
                "price_usd": coin.get("price_usd") or coin.get("latest_price") or 1.0,
            }
        ],
        price_timestamp=_utc_now(),
    )
    pos = trader.open_position(
        coin,
        size_usd=float(notional_usd),
        cluster_label="UNKNOWN_UNRESOLVED",
        settings=settings or {},
        reason_code="DEMO_ACCEPTANCE_MODE",
        allow_coin_price_fallback=True,
    )
    if not pos:
        audit["status"] = "OPEN_FAILED"
        audit["wallet_before"] = wallet_before
        audit["wallet_after"] = trader.get_wallet_summary()
        audit["coin_used"] = {
            "symbol": coin.get("symbol"),
            "coin_id": coin.get("coin_id") or coin.get("id"),
            "pair_address": coin.get("pair_address"),
        }
        audit["rejection_reasons"] = ["paper_trader_open_position_returned_none"]
        return audit

    # Stamp acceptance flags onto open position state (non-destructive metadata)
    positions = trader.get_positions(status="OPEN")
    for p in positions:
        if int(p.get("id", -1)) == int(pos.get("id", -2)):
            p["demo_acceptance_only"] = True
            p["not_strategy_evidence"] = True
            p["not_profitability_evidence"] = True
            p["not_live_approved"] = True
            p["paper_demo_only"] = True
            p["trade_authority"] = "DEMO_ACCEPTANCE_ONLY"
            break
    trader._save_state()  # noqa: SLF001 — intentional metadata stamp for UI/audit

    wallet_after = trader.get_wallet_summary()
    order_id = hashlib.sha256(
        f"ae13_demo_acceptance|{pos.get('id')}|{pos.get('opened_at') or _utc_now()}".encode()
    ).hexdigest()[:16]

    audit.update(
        {
            "status": "CREATED",
            "paper_order_id": f"ae13_demo_{order_id}",
            "position": pos,
            "coin_used": {
                "symbol": coin.get("symbol"),
                "coin_id": coin.get("coin_id") or coin.get("id"),
                "pair_address": coin.get("pair_address"),
            },
            "wallet_before": wallet_before,
            "wallet_after": wallet_after,
            "notional_usd": float(notional_usd),
            "symbol": coin.get("symbol") or DEMO_ACCEPTANCE_SYMBOL,
            "message": (
                "Bounded DEMO_ACCEPTANCE_MODE paper order created to prove UI/ledger wiring. "
                "Not strategy evidence. Not profitability evidence. Not live approved."
            ),
        }
    )
    log.info(
        "DEMO_ACCEPTANCE_MODE paper order created position_id=%s notional=%s",
        pos.get("id"),
        notional_usd,
    )
    return audit


def maybe_close_demo_acceptance_position(
    *,
    trading_mode: str,
    live_trading_enabled: bool,
    wallet_configured: bool,
    demo_acceptance_mode_enabled: bool,
    position_id: int | None = None,
) -> dict[str, Any]:
    """Optional close path for acceptance lifecycle visibility (paper only)."""
    guard = evaluate_demo_acceptance_guard(
        trading_mode=trading_mode,
        live_trading_enabled=live_trading_enabled,
        wallet_configured=wallet_configured,
        demo_acceptance_mode_enabled=demo_acceptance_mode_enabled,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    if not guard["allowed"]:
        return {"status": "REJECTED", "guard": guard}

    from app.execution.paper import get_paper_trader

    trader = get_paper_trader()
    opens = trader.get_positions(status="OPEN")
    target = None
    if position_id is not None:
        target = next((p for p in opens if int(p.get("id", -1)) == int(position_id)), None)
    else:
        target = next(
            (
                p
                for p in opens
                if p.get("demo_acceptance_only")
                or p.get("trade_authority") == "DEMO_ACCEPTANCE_ONLY"
                or p.get("symbol") == DEMO_ACCEPTANCE_SYMBOL
            ),
            None,
        )
    if not target:
        return {"status": "NO_ACCEPTANCE_POSITION", "guard": guard}

    closed = trader.close_position(
        int(target["id"]),
        float(target.get("entry_price") or 1.0),
        reason_code="DEMO_ACCEPTANCE_CLOSE",
    )
    return {
        "status": "CLOSED" if closed else "CLOSE_FAILED",
        "guard": guard,
        "closed": closed,
        "wallet": trader.get_wallet_summary(),
        "demo_acceptance_only": True,
        "not_strategy_evidence": True,
        "not_profitability_evidence": True,
        "not_live_approved": True,
    }
