"""Backend portfolio risk guard for paper/demo order paths.

Runs server-side before paper order creation. UI / Demo Queue cannot bypass it.
Paper/demo only — never implies live trading or profitability.

AE13G: returns structured RiskGuardResult with rejection_reasons[] and blocking_guards[].
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "demo_equity": 10_000.0,
    "max_notional_per_trade": 100.0,
    "max_position_pct": 0.05,
    "max_open_positions": 6,
    "max_trades_per_hour": 30,
    "max_trade_notional_per_hour": 1500.0,
    "max_symbol_exposure_pct": 0.10,
    "max_chain_exposure_pct": 0.50,
    "daily_demo_drawdown_limit": 0.25,
    "cooldown_seconds": 30,
    "pair_lock_enabled": True,
    "duplicate_pair_guard": True,
    "min_liquidity": 0.0,
    "price_freshness_limit_seconds": 900,
}

# Human reason fragment -> machine guard id
_REASON_TO_GUARD: list[tuple[str, str, str]] = [
    ("requested notional is zero", "missing_notional", "REQUESTED_NOTIONAL_MISSING"),
    ("position size exceeds", "max_position_pct", "MAX_POSITION_PCT"),
    ("max_notional_per_trade", "max_notional_per_trade", "MAX_NOTIONAL_PER_TRADE"),
    ("notional exceeds", "max_notional_per_trade", "MAX_NOTIONAL_PER_TRADE"),
    ("max open positions", "max_open_positions", "MAX_OPEN_POSITIONS"),
    ("duplicate pair", "duplicate_pair_guard", "DUPLICATE_PAIR_ALREADY_OPEN"),
    ("same-pair", "same_pair_duplicate_guard", "SAME_PAIR_DUPLICATE"),
    ("max trades per hour", "max_trades_per_hour", "MAX_TRADES_PER_HOUR"),
    ("max trade notional per hour", "max_trade_notional_per_hour", "MAX_TRADE_NOTIONAL_PER_HOUR"),
    ("pair cooldown", "cooldown", "PAIR_COOLDOWN_ACTIVE"),
    ("pair lock", "pair_lock", "PAIR_LOCK_ACTIVE"),
    ("missing price", "missing_price", "MISSING_PRICE"),
    ("invalid price", "invalid_price", "INVALID_PRICE"),
    ("stale price", "stale_price", "STALE_PRICE"),
    ("liquidity below", "liquidity", "LIQUIDITY_TOO_LOW"),
    ("liquidity missing", "liquidity", "LIQUIDITY_MISSING"),
    ("max symbol exposure", "max_symbol_exposure", "MAX_SYMBOL_EXPOSURE"),
    ("max chain exposure", "max_chain_exposure", "MAX_CHAIN_EXPOSURE"),
    ("unsupported chain", "unsupported_chain", "UNSUPPORTED_CHAIN"),
    ("provider unavailable", "provider_unavailable", "PROVIDER_UNAVAILABLE"),
    ("semantic", "semantic", "SEMANTIC_BLOCK"),
    # AE13I GateKeeper reason -> guard mappings, so risk-guard-adjacent copy
    # (aggregate_rejection_counts / format_top_rejection_summary) recognizes
    # gate-originated blockers with the same guard ids the gate itself emits.
    ("manual reentry", "manual_reentry_block", "MANUAL_REENTRY_BLOCK_ACTIVE"),
    ("manual_reentry_block_active", "manual_reentry_block", "MANUAL_REENTRY_BLOCK_ACTIVE"),
    ("manual re-entry", "manual_reentry_block", "MANUAL_REENTRY_BLOCK_ACTIVE"),
    ("stagnant", "stagnant_price_guard", "PRICE_STAGNANT_NO_RECENT_MOMENTUM"),
    ("price_stagnant", "stagnant_price_guard", "PRICE_STAGNANT_NO_RECENT_MOMENTUM"),
    ("freshness", "freshness_gate", "NOT_OPENED_STALE_MARKET_DATA"),
    ("not_opened_stale", "freshness_gate", "NOT_OPENED_STALE_MARKET_DATA"),
    ("no new signal", "system_reentry_no_new_signal", "REENTRY_BLOCK_NO_NEW_SIGNAL"),
    ("reentry_block_no_new_signal", "system_reentry_no_new_signal", "REENTRY_BLOCK_NO_NEW_SIGNAL"),
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _classify_reason(reason: str) -> tuple[str, str]:
    low = str(reason or "").lower()
    for needle, guard, code in _REASON_TO_GUARD:
        if needle in low:
            return guard, code
    return "unknown", "UNKNOWN_RISK_BLOCK"


def resolve_risk_settings(
    *,
    settings: dict[str, Any] | None = None,
    bot_state: dict[str, Any] | None = None,
    strategy_lane: str | None = None,
    risk_mode: str | None = None,
    preset_id: str | None = None,
) -> dict[str, Any]:
    """Merge defaults with settings / demo-bot state / explicit risk_mode preset.

    Active bot preset (via bot_state or risk_mode/preset_id) is the source of truth
    when provided. Lotto lane still caps notional at $25 as a hard ceiling.
    """
    out = dict(DEFAULTS)
    s = settings or {}
    b = bot_state or {}

    # Explicit risk_mode / preset_id inherits preset limits (Demo Queue / Watchlist)
    mode_key = str(risk_mode or preset_id or b.get("preset_id") or b.get("risk_mode") or "").lower().strip()
    if mode_key:
        try:
            from app.ae13b_product.presets import get_preset

            preset = get_preset(mode_key)
            out["max_open_positions"] = int(preset.get("max_open_positions", out["max_open_positions"]))
            out["max_trades_per_hour"] = int(preset.get("max_trades_per_hour", out["max_trades_per_hour"]))
            out["max_notional_per_trade"] = float(preset.get("max_notional_usd", out["max_notional_per_trade"]))
            out["cooldown_seconds"] = int(preset.get("cooldown_seconds", out["cooldown_seconds"]))
            out["preset_id"] = preset.get("id")
            out["risk_mode"] = preset.get("id")
        except Exception:
            out["preset_id"] = mode_key or None
            out["risk_mode"] = mode_key or None
    else:
        out["preset_id"] = None
        out["risk_mode"] = None

    if s.get("starting_capital") is not None:
        try:
            out["demo_equity"] = float(s["starting_capital"])
        except (TypeError, ValueError):
            pass
    if b.get("demo_equity") is not None:
        try:
            out["demo_equity"] = float(b["demo_equity"])
        except (TypeError, ValueError):
            pass

    pct = s.get("max_position_size_pct", out["max_position_pct"])
    try:
        pct_f = float(pct)
        if pct_f > 1:
            pct_f = pct_f / 100.0
        out["max_position_pct"] = max(0.0001, min(1.0, pct_f))
    except (TypeError, ValueError):
        pass

    # bot_state overrides (active demo bot is authoritative when present)
    if b.get("max_notional_usd") is not None:
        try:
            out["max_notional_per_trade"] = float(b["max_notional_usd"])
        except (TypeError, ValueError):
            pass
    elif s.get("max_position_size_usd") is not None:
        try:
            out["max_notional_per_trade"] = float(s["max_position_size_usd"])
        except (TypeError, ValueError):
            pass

    if b.get("max_open_positions") is not None:
        try:
            out["max_open_positions"] = int(b["max_open_positions"])
        except (TypeError, ValueError):
            pass
    if b.get("max_trades_per_hour") is not None:
        try:
            out["max_trades_per_hour"] = int(b["max_trades_per_hour"])
        except (TypeError, ValueError):
            pass
    if b.get("cooldown_seconds") is not None:
        try:
            out["cooldown_seconds"] = int(b["cooldown_seconds"])
        except (TypeError, ValueError):
            pass

    if s.get("min_liquidity_usd") is not None:
        try:
            out["min_liquidity"] = float(s["min_liquidity_usd"])
        except (TypeError, ValueError):
            pass

    lane = str(strategy_lane or "").lower()
    if "lotto" in lane or mode_key == "lotto":
        out["max_notional_per_trade"] = min(float(out["max_notional_per_trade"]), 25.0)
        if not out.get("preset_id"):
            out["preset_id"] = "lotto"
            out["risk_mode"] = "lotto"

    return out


def evaluate_demo_risk_guard(
    *,
    requested_notional: float,
    demo_equity: float | None = None,
    open_positions: list[dict[str, Any]] | None = None,
    recent_trades: list[dict[str, Any]] | None = None,
    pair_address: str | None = None,
    symbol: str | None = None,
    chain: str | None = None,
    price: float | None = None,
    price_timestamp: str | None = None,
    liquidity: float | None = None,
    strategy_lane: str | None = None,
    settings: dict[str, Any] | None = None,
    bot_state: dict[str, Any] | None = None,
    pair_cooldowns: dict[str, Any] | None = None,
    risk_mode: str | None = None,
    preset_id: str | None = None,
    token_contract_address: str | None = None,
    gate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate portfolio risk limits. Collects all inexpensive blockers.

    This guard intentionally stays focused on portfolio-level limits (position
    size, exposure, rate limits, cooldowns). Market-data freshness/provenance/
    address-role/reentry enforcement is owned upstream by MarketDataGateKeeper
    (AE13I). If a `gate_result` is supplied, its blockers are merged in as an
    additional hard block so downstream aggregate/summary helpers see a single
    unified rejection list, but the gate itself remains the primary enforcer.
    """
    cfg = resolve_risk_settings(
        settings=settings,
        bot_state=bot_state,
        strategy_lane=strategy_lane,
        risk_mode=risk_mode,
        preset_id=preset_id,
    )
    equity = float(demo_equity if demo_equity is not None else cfg["demo_equity"])
    if equity <= 0:
        equity = float(cfg["demo_equity"])

    max_by_pct = equity * float(cfg["max_position_pct"])
    max_allowed = min(float(cfg["max_notional_per_trade"]), max_by_pct)
    req = float(requested_notional or 0)
    original_requested_notional = req
    demo_notional_clamped = False
    demo_notional_policy = ""
    opens = list(open_positions or [])
    trades = list(recent_trades or [])
    pair = str(pair_address or "").strip()
    token = str(token_contract_address or "").strip()
    reasons: list[str] = []
    blocking_guards: list[str] = []
    rejection_codes: list[str] = []
    warnings: list[str] = []
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    demo_buy_notional = max(1.0, _env_float("DEMO_BUY_NOTIONAL_USD", 25.0))
    demo_buy_max_notional = max(1.0, _env_float("DEMO_BUY_MAX_NOTIONAL_USD", 50.0))
    demo_buy_notional = min(demo_buy_notional, demo_buy_max_notional)

    if req > 0 and (req > demo_buy_max_notional + 1e-9 or req > max_allowed + 1e-9):
        warnings.append(
            f"DEMO_ACTION_ALLOWED_WITH_NOTIONAL_CLAMP: requested ${req:.2f}; "
            f"using small demo notional ${demo_buy_notional:.2f}."
        )
        demo_notional_policy = (
            f"Requested notional ${req:.2f} exceeded demo policy/risk cap; "
            f"effective demo notional set to ${demo_buy_notional:.2f}."
        )
        req = demo_buy_notional
        demo_notional_clamped = True


    def _block(msg: str, guard: str, code: str) -> None:
        reasons.append(msg)
        if guard not in blocking_guards:
            blocking_guards.append(guard)
        if code not in rejection_codes:
            rejection_codes.append(code)

    if gate_result and not gate_result.get("passed", True):
        for gate_reason in gate_result.get("rejection_reasons") or []:
            reasons.append(str(gate_reason))
        for gate_guard in gate_result.get("blocking_guards") or []:
            if gate_guard not in blocking_guards:
                blocking_guards.append(gate_guard)
        gate_code = gate_result.get("rejection_code")
        if gate_code and gate_code not in rejection_codes:
            rejection_codes.append(gate_code)

    if req <= 0:
        _block(
            "Blocked: requested notional is zero or missing",
            "missing_notional",
            "REQUESTED_NOTIONAL_MISSING",
        )

    if req > max_allowed + 1e-9:
        if max_by_pct <= float(cfg["max_notional_per_trade"]) + 1e-9:
            pct_disp = float(cfg["max_position_pct"]) * 100
            _block(
                f"Blocked: position size exceeds {pct_disp:g}% demo portfolio limit",
                "max_position_pct",
                "MAX_POSITION_PCT",
            )
        else:
            _block(
                f"Blocked: notional exceeds max_notional_per_trade (${max_allowed:.2f})",
                "max_notional_per_trade",
                "MAX_NOTIONAL_PER_TRADE",
            )

    if len(opens) >= int(cfg["max_open_positions"]):
        _block(
            f"Blocked: max open positions reached ({cfg['max_open_positions']})",
            "max_open_positions",
            "MAX_OPEN_POSITIONS",
        )

    if cfg.get("duplicate_pair_guard") and pair:
        for p in opens:
            if str(p.get("pair_address") or "").strip() == pair:
                _block(
                    "Blocked: duplicate pair already open",
                    "duplicate_pair_guard",
                    "DUPLICATE_PAIR_ALREADY_OPEN",
                )
                break
        else:
            # Same token contract across different pools counts as same-asset duplicate
            if token:
                for p in opens:
                    p_tok = str(
                        p.get("token_contract_address")
                        or p.get("contract_address")
                        or p.get("token_address")
                        or ""
                    ).strip()
                    if p_tok and p_tok.lower() == token.lower():
                        _block(
                            "Blocked: same-pair / same-token duplicate already open",
                            "same_pair_duplicate_guard",
                            "SAME_PAIR_DUPLICATE",
                        )
                        break

    now = _utc_now()
    hour_ago = now - timedelta(hours=1)
    hour_trades = []
    hour_notional = 0.0
    for t in trades:
        side = str(t.get("side") or "").lower()
        reason = str(t.get("reason_code") or t.get("event_type") or "").upper()
        # Count successful opens only for rate limits (not RISK_GUARD_BLOCK attempts)
        if reason in ("RISK_GUARD_BLOCK", "REJECTED"):
            continue
        if side == "sell":
            continue
        if side not in ("buy", "open", ""):
            continue
        ts = _parse_ts(t.get("timestamp") or t.get("created_at"))
        if ts is None or ts < hour_ago:
            continue
        hour_trades.append(t)
        try:
            hour_notional += float(t.get("notional_usd") or t.get("size_usd") or 0)
        except (TypeError, ValueError):
            pass

    if len(hour_trades) >= int(cfg["max_trades_per_hour"]):
        _block(
            "Blocked: max trades per hour reached",
            "max_trades_per_hour",
            "MAX_TRADES_PER_HOUR",
        )

    if hour_notional + max(0.0, min(req, max_allowed)) > float(
        cfg["max_trade_notional_per_hour"]
    ):
        _block(
            "Blocked: max trade notional per hour reached",
            "max_trade_notional_per_hour",
            "MAX_TRADE_NOTIONAL_PER_HOUR",
        )

    if cfg.get("pair_lock_enabled") and pair and pair_cooldowns:
        until = _parse_ts(pair_cooldowns.get(pair))
        if until and until > now:
            _block(
                "Blocked: pair cooldown active",
                "cooldown",
                "PAIR_COOLDOWN_ACTIVE",
            )

    locked_pairs = set()
    if bot_state and bot_state.get("locked_pairs"):
        try:
            locked_pairs = {str(p).strip() for p in (bot_state.get("locked_pairs") or [])}
        except TypeError:
            locked_pairs = set()
    if pair and pair in locked_pairs:
        _block(
            "Blocked: pair lock active",
            "pair_lock",
            "PAIR_LOCK_ACTIVE",
        )

    # Price freshness
    limit_s = float(cfg["price_freshness_limit_seconds"])
    if price is None:
        _block(
            "Blocked: missing price",
            "missing_price",
            "MISSING_PRICE",
        )
    elif float(price or 0) <= 0:
        _block(
            "Blocked: invalid price (must be > 0)",
            "invalid_price",
            "INVALID_PRICE",
        )
    else:
        pts = _parse_ts(price_timestamp)
        if pts is None:
            warnings.append("Price timestamp missing; freshness not verified")
        else:
            age = (now - pts).total_seconds()
            if age > limit_s:
                _block("Blocked: stale price", "stale_price", "STALE_PRICE")

    min_liq = float(cfg.get("min_liquidity") or 0)
    if min_liq > 0:
        if liquidity is None:
            warnings.append("Liquidity missing; min_liquidity configured but not enforced as hard block")
        else:
            try:
                if float(liquidity) < min_liq:
                    _block(
                        f"Blocked: liquidity below minimum (${min_liq:.0f})",
                        "liquidity",
                        "LIQUIDITY_TOO_LOW",
                    )
            except (TypeError, ValueError):
                _block(
                    "Blocked: liquidity missing or invalid",
                    "liquidity",
                    "LIQUIDITY_MISSING",
                )

    # Symbol / chain exposure
    sym = str(symbol or "").strip().upper()
    if "/" in sym:
        sym_base = sym.split("/")[0].strip()
    else:
        sym_base = sym
    if sym_base:
        sym_exp = 0.0
        for p in opens:
            p_sym = str(p.get("symbol") or "").strip().upper()
            p_base = p_sym.split("/")[0].strip() if "/" in p_sym else p_sym
            if p_base == sym_base or p_sym == sym:
                try:
                    sym_exp += float(p.get("size_usd") or p.get("notional_usd") or 0)
                except (TypeError, ValueError):
                    pass
        if (sym_exp + min(req, max_allowed)) / equity > float(
            cfg["max_symbol_exposure_pct"]
        ):
            _block(
                f"Blocked: max symbol exposure exceeded for {sym_base}",
                "max_symbol_exposure",
                "MAX_SYMBOL_EXPOSURE",
            )

    ch = str(chain or "").strip().lower()
    if ch:
        ch_exp = 0.0
        for p in opens:
            if str(p.get("chain") or "").strip().lower() == ch:
                try:
                    ch_exp += float(p.get("size_usd") or p.get("notional_usd") or 0)
                except (TypeError, ValueError):
                    pass
        if (ch_exp + min(req, max_allowed)) / equity > float(
            cfg["max_chain_exposure_pct"]
        ):
            _block(
                f"Blocked: max chain exposure exceeded for {ch}",
                "max_chain_exposure",
                "MAX_CHAIN_EXPOSURE",
            )

    approved = 0.0 if reasons else min(req, max_allowed)
    passed = len(reasons) == 0
    primary_blocker = blocking_guards[0] if blocking_guards else None
    rejection_code = rejection_codes[0] if rejection_codes else None
    reason = reasons[0] if reasons else "risk_guard_passed"

    candidate_context = {
        "symbol": symbol,
        "pair_address": pair or None,
        "token_contract_address": token or None,
        "chain": chain,
        "strategy_lane": strategy_lane or "",
        "price": price,
        "price_timestamp": price_timestamp,
        "liquidity": liquidity,
        "open_positions_count": len(opens),
        "max_open_positions": int(cfg["max_open_positions"]),
        "available_slots": max(0, int(cfg["max_open_positions"]) - len(opens)),
        "max_open_blocking": "max_open_positions" in blocking_guards,
    }

    settings_snapshot = {
        k: cfg[k]
        for k in (
            "max_notional_per_trade",
            "max_position_pct",
            "max_open_positions",
            "max_trades_per_hour",
            "max_trade_notional_per_hour",
            "max_symbol_exposure_pct",
            "max_chain_exposure_pct",
            "daily_demo_drawdown_limit",
            "cooldown_seconds",
            "pair_lock_enabled",
            "duplicate_pair_guard",
            "min_liquidity",
            "price_freshness_limit_seconds",
        )
    }
    settings_snapshot["preset_id"] = cfg.get("preset_id")
    settings_snapshot["risk_mode"] = cfg.get("risk_mode")

    return {
        # Legacy fields (compat)
        "risk_guard_passed": passed,
        "risk_guard_reason": reason,
        "risk_guard_reasons": list(reasons),
        # AE13G structured RiskGuardResult
        "passed": passed,
        "primary_blocker": primary_blocker,
        "rejection_code": rejection_code,
        "rejection_reason": None if passed else reason,
        "rejection_reasons": list(reasons),
        "blocking_guards": list(blocking_guards),
        "rejection_codes": list(rejection_codes),
        "warnings": list(warnings),
        "candidate_context": candidate_context,
        "settings_snapshot": settings_snapshot,
        "checked_at_utc": now.isoformat(),
        "requested_notional": original_requested_notional,
        "effective_requested_notional": req,
        "approved_notional": approved,
        "executed_notional_usd": approved,
        "max_allowed_notional": max_allowed,
        "demo_notional_clamped": demo_notional_clamped,
        "demo_notional_policy": demo_notional_policy,
        "demo_equity": equity,
        "max_position_pct": float(cfg["max_position_pct"]),
        "max_notional_per_trade": float(cfg["max_notional_per_trade"]),
        "max_open_positions": int(cfg["max_open_positions"]),
        "max_trades_per_hour": int(cfg["max_trades_per_hour"]),
        "strategy_lane": strategy_lane or "",
        "preset_id": cfg.get("preset_id"),
        "risk_mode": cfg.get("risk_mode") or cfg.get("preset_id"),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
    }


def assert_demo_risk_allowed(**kwargs: Any) -> dict[str, Any]:
    result = evaluate_demo_risk_guard(**kwargs)
    if not result["risk_guard_passed"]:
        raise DemoRiskGuardError(result["risk_guard_reason"], detail=result)
    return result


def aggregate_rejection_counts(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate top blockers across rejected open attempts for cycle UI copy."""
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for att in attempts or []:
        guards = att.get("blocking_guards") or []
        reasons = att.get("rejection_reasons") or []
        if not guards and att.get("primary_blocker"):
            guards = [att["primary_blocker"]]
        if not guards and att.get("rejection_code"):
            guards = [str(att["rejection_code"]).lower()]
        if not guards:
            guards = ["unknown"]
        for i, g in enumerate(guards):
            key = str(g)
            counts[key] = counts.get(key, 0) + 1
            if key not in labels:
                if i < len(reasons):
                    labels[key] = str(reasons[i])
                else:
                    labels[key] = key.replace("_", " ")
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {
            "guard": g,
            "count": c,
            "label": labels.get(g, g.replace("_", " ")),
        }
        for g, c in ranked
    ]


def format_top_rejection_summary(
    attempts: list[dict[str, Any]],
    *,
    candidates_selected: int | None = None,
) -> str:
    """ASCII-safe cycle summary for UI / API."""
    dist = aggregate_rejection_counts(attempts)
    n = len(attempts) if candidates_selected is None else int(candidates_selected)
    if not attempts and n == 0:
        return "No trade attempts this cycle."
    if not dist:
        return f"No new trade: {n} candidates rejected (no structured reason recorded)."
    lines = [f"No new trade: {n} candidates rejected.", "Top reasons:"]
    for row in dist[:8]:
        lines.append(f"- {row['label']}: {row['count']}")
    return "\n".join(lines)


class DemoRiskGuardError(Exception):
    def __init__(self, reason: str, *, detail: dict[str, Any] | None = None) -> None:
        self.reason = reason
        self.detail = detail or {}
        super().__init__(reason)
