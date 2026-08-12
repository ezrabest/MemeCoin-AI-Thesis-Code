from pathlib import Path
import py_compile

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

# ------------------------------------------------------------------
# 1) New orthogonal market activity evaluator
# ------------------------------------------------------------------
ma_path = Path("app/clean_forward/market_activity.py")
ma_path.parent.mkdir(parents=True, exist_ok=True)
if ma_path.exists():
    ma_path.with_suffix(".py.bak_ae18_market_activity_stage4b").write_text(
        ma_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

ma_path.write_text(r'''"""AE18 market activity / tradability axis.

This module is deliberately orthogonal to display rendering:
- Missing symbols are display metadata only.
- Missing/stale/zero transaction activity is market activity / tradability.
- Liquidity and market cap alone are never treated as proof of tradable activity.
"""
from __future__ import annotations

from typing import Any

ACTIVE_PROVIDER_TXNS = "ACTIVE_PROVIDER_TXNS"
NO_RECENT_PROVIDER_TXNS = "NO_RECENT_PROVIDER_TXNS"
ACTIVITY_STAGNANT = "ACTIVITY_STAGNANT"
ACTIVITY_UNKNOWN = "ACTIVITY_UNKNOWN"

WATCH_ONLY_NO_RECENT_PROVIDER_TXNS = "WATCH_ONLY_NO_RECENT_PROVIDER_TXNS"
WATCH_ONLY_ACTIVITY_STAGNANT = "WATCH_ONLY_ACTIVITY_STAGNANT"
WATCH_ONLY_ACTIVITY_UNKNOWN = "WATCH_ONLY_ACTIVITY_UNKNOWN"

TXN_FIELDS = (
    "txns_m5_buys",
    "txns_m5_sells",
    "txns_h1_buys",
    "txns_h1_sells",
    "txns_h6_buys",
    "txns_h6_sells",
    "txns_h24_buys",
    "txns_h24_sells",
)

VOLUME_FIELDS = (
    "volume_m5",
    "volume_h1",
    "volume_h6",
    "volume_h24",
    "volume_24h",
)

DELTA_FIELDS = (
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "price_change_5m",
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
)

def _num(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def _values(row: dict[str, Any], fields: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    for f in fields:
        n = _num(row.get(f))
        if n is not None:
            out.append(n)
    return out

def _any_positive(vals: list[float]) -> bool:
    return any(v > 0 for v in vals)

def _any_nonzero(vals: list[float]) -> bool:
    return any(abs(v) > 1e-12 for v in vals)

def evaluate_market_activity(row: dict[str, Any]) -> dict[str, Any]:
    """Classify provider market activity without using symbols as evidence.

    ACTIVE requires provider transaction flow and non-zero volume.
    Liquidity/market_cap/FDV alone never upgrades a row to active.
    """
    tx_vals = _values(row, TXN_FIELDS)
    vol_vals = _values(row, VOLUME_FIELDS)
    delta_vals = _values(row, DELTA_FIELDS)

    tx_total = sum(max(0.0, v) for v in tx_vals)
    volume_total = sum(max(0.0, v) for v in vol_vals)
    has_txns = tx_total > 0
    has_volume = volume_total > 0
    has_delta = _any_nonzero(delta_vals)

    liquidity = _num(row.get("liquidity_usd") or row.get("liquidity"))
    market_cap = _num(row.get("market_cap"))
    fdv = _num(row.get("fdv"))
    price = _num(row.get("price_usd") or row.get("price"))

    has_static_market_metadata = any(
        v is not None and v > 0 for v in (liquidity, market_cap, fdv, price)
    )
    observed_any_activity_field = bool(tx_vals or vol_vals or delta_vals)

    if has_txns and has_volume:
        status = ACTIVE_PROVIDER_TXNS
        readiness = "PAPER_ELIGIBLE"
        block_reason = ""
    elif tx_vals and tx_total <= 0:
        status = NO_RECENT_PROVIDER_TXNS
        readiness = WATCH_ONLY_NO_RECENT_PROVIDER_TXNS
        block_reason = (
            "NO_RECENT_PROVIDER_TXNS — provider reports zero transaction flow; "
            "liquidity/market-cap alone is not treated as tradable activity."
        )
    elif has_static_market_metadata and not has_txns and not has_volume and not has_delta:
        status = ACTIVITY_STAGNANT
        readiness = WATCH_ONLY_ACTIVITY_STAGNANT
        block_reason = (
            "ACTIVITY_STAGNANT — static market metadata exists, but provider "
            "transactions, volume, and deltas do not show active flow."
        )
    elif observed_any_activity_field:
        status = ACTIVITY_STAGNANT
        readiness = WATCH_ONLY_ACTIVITY_STAGNANT
        block_reason = (
            "ACTIVITY_STAGNANT — provider activity fields are present but do not "
            "meet active transaction + non-zero volume criteria."
        )
    else:
        status = ACTIVITY_UNKNOWN
        readiness = WATCH_ONLY_ACTIVITY_UNKNOWN
        block_reason = (
            "ACTIVITY_UNKNOWN — insufficient provider transaction/volume/delta "
            "metadata to classify this market as active."
        )

    return {
        "market_activity_status": status,
        "activity_trade_readiness_status": readiness,
        "activity_trade_block_reason": block_reason,
        "market_activity_blocks_demo_entry": status != ACTIVE_PROVIDER_TXNS,
        "provider_txns_observed_field_count": len(tx_vals),
        "provider_txns_recent_total": tx_total,
        "provider_volume_observed_field_count": len(vol_vals),
        "provider_volume_recent_total": volume_total,
        "provider_price_delta_observed_field_count": len(delta_vals),
        "provider_price_delta_any_nonzero": has_delta,
        "market_activity_provenance": "ae18_market_activity_axis_from_provider_runtime_fields",
        "activity_uses_symbol_display": False,
        "activity_uses_liquidity_or_market_cap_as_activity_proxy": False,
    }
''', encoding="utf-8")

# ------------------------------------------------------------------
# 2) Add fields to runtime index schema/order
# ------------------------------------------------------------------
idx_path = Path("app/clean_forward/runtime_identity_index.py")
idx = idx_path.read_text(encoding="utf-8")
idx_path.with_suffix(".py.bak_ae18_market_activity_stage4b").write_text(idx, encoding="utf-8")

if '"market_activity_status",' not in idx:
    idx = replace_once(
        idx,
        '''    "trade_readiness_status",
    "trade_block_reason",
''',
        '''    "trade_readiness_status",
    "trade_block_reason",
    "display_status",
    "market_activity_status",
    "activity_trade_readiness_status",
    "activity_trade_block_reason",
    "market_activity_blocks_demo_entry",
    "provider_txns_observed_field_count",
    "provider_txns_recent_total",
    "provider_volume_observed_field_count",
    "provider_volume_recent_total",
    "provider_price_delta_observed_field_count",
    "provider_price_delta_any_nonzero",
    "market_activity_provenance",
    "activity_uses_symbol_display",
    "activity_uses_liquidity_or_market_cap_as_activity_proxy",
''',
        "runtime index activity fields",
    )

idx_path.write_text(idx, encoding="utf-8")

# ------------------------------------------------------------------
# 3) Evaluate activity when building canonical runtime index rows
# ------------------------------------------------------------------
canon_path = Path("app/clean_forward/canonical_market_identity.py")
canon = canon_path.read_text(encoding="utf-8")
canon_path.with_suffix(".py.bak_ae18_market_activity_stage4b").write_text(canon, encoding="utf-8")

if "from app.clean_forward.market_activity import ACTIVE_PROVIDER_TXNS, evaluate_market_activity" not in canon:
    canon = replace_once(
        canon,
        "from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key\n",
        "from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key\n"
        "from app.clean_forward.market_activity import ACTIVE_PROVIDER_TXNS, evaluate_market_activity\n",
        "canonical activity import",
    )

canon = replace_once(
    canon,
    '''    return apply_display_resilience(out, allow_cache_lookup=True)
''',
    '''    out = apply_display_resilience(out, allow_cache_lookup=True)
    out["display_status"] = (
        out.get("display_metadata_status")
        or out.get("symbol_resolution_status")
        or out.get("symbol_pair_display_status")
        or ""
    )

    activity = evaluate_market_activity(out)
    out.update(activity)

    # Missing symbols alone do not block trading. Market inactivity does.
    if activity.get("market_activity_status") != ACTIVE_PROVIDER_TXNS:
        if str(out.get("trade_readiness_status") or "") == "PAPER_ELIGIBLE":
            out["trade_readiness_status"] = activity.get("activity_trade_readiness_status")
            out["trade_block_reason"] = activity.get("activity_trade_block_reason")

    return out
''',
    "canonical apply market activity before return",
)

canon_path.write_text(canon, encoding="utf-8")

# ------------------------------------------------------------------
# 4) Propagate activity fields through Clean Forward / Live Market / Opportunities
# ------------------------------------------------------------------
feed_path = Path("app/ae13b_product/runtime_market_feed.py")
feed = feed_path.read_text(encoding="utf-8")
feed_path.with_suffix(".py.bak_ae18_market_activity_stage4b").write_text(feed, encoding="utf-8")

if "ACTIVITY_FIELDS = (" not in feed:
    feed = replace_once(
        feed,
        '''def _utc_now() -> str:
''',
        '''ACTIVITY_FIELDS = (
    "display_status",
    "market_activity_status",
    "activity_trade_readiness_status",
    "activity_trade_block_reason",
    "market_activity_blocks_demo_entry",
    "provider_txns_observed_field_count",
    "provider_txns_recent_total",
    "provider_volume_observed_field_count",
    "provider_volume_recent_total",
    "provider_price_delta_observed_field_count",
    "provider_price_delta_any_nonzero",
    "market_activity_provenance",
    "activity_uses_symbol_display",
    "activity_uses_liquidity_or_market_cap_as_activity_proxy",
)

def _activity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {f: row.get(f) for f in ACTIVITY_FIELDS if row.get(f) not in (None, "")}


def _utc_now() -> str:
''',
        "runtime feed activity constants",
    )

feed = replace_once(
    feed,
    '''        "external_network_on_load": False,
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
''',
    '''        "external_network_on_load": False,
        **_activity_fields(row),
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
''',
    "clean forward activity propagation",
)

feed = replace_once(
    feed,
    '''        "_stale": row.get("freshness_status") not in (None, "", "fresh"),
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
''',
    '''        "_stale": row.get("freshness_status") not in (None, "", "fresh"),
        **_activity_fields(row),
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
''',
    "live market activity propagation",
)

if "for fld in ACTIVITY_FIELDS:" not in feed:
    feed = replace_once(
        feed,
        '''            for fld in RESILIENCE_FIELDS:
                if disp.get(fld) not in (None, ""):
                    row[fld] = disp[fld]
            row.update(_social_fields(ir))
''',
        '''            for fld in RESILIENCE_FIELDS:
                if disp.get(fld) not in (None, ""):
                    row[fld] = disp[fld]
            for fld in ACTIVITY_FIELDS:
                if ir.get(fld) not in (None, ""):
                    row[fld] = ir.get(fld)
            row.update(_social_fields(ir))
''',
        "opportunities activity propagation",
    )

feed_path.write_text(feed, encoding="utf-8")

# ------------------------------------------------------------------
# 5) Enforce activity axis in Market Opportunities and BUY DEMO action
# ------------------------------------------------------------------
api_path = Path("app/api.py")
api = api_path.read_text(encoding="utf-8")
api_path.with_suffix(".py.bak_ae18_market_activity_stage4b").write_text(api, encoding="utf-8")

api = replace_once(
    api,
    '''        price = row.get("price_usd")
''',
    '''        activity_status = str(row.get("market_activity_status") or "ACTIVITY_UNKNOWN").strip() or "ACTIVITY_UNKNOWN"
        if activity_status != "ACTIVE_PROVIDER_TXNS":
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_MARKET_ACTIVITY",
                    row.get("activity_trade_block_reason")
                    or f"{activity_status} — market is not classified as active provider flow.",
                    canonical=canonical,
                    extra={
                        "market_activity_status": activity_status,
                        "activity_trade_readiness_status": row.get("activity_trade_readiness_status"),
                        "activity_trade_block_reason": row.get("activity_trade_block_reason"),
                        "provider_txns_recent_total": row.get("provider_txns_recent_total"),
                        "provider_volume_recent_total": row.get("provider_volume_recent_total"),
                        "provider_price_delta_any_nonzero": row.get("provider_price_delta_any_nonzero"),
                        "display_status": row.get("display_status"),
                        "symbol_pair_display": row.get("symbol_pair_display"),
                    },
                )
            )

        price = row.get("price_usd")
''',
    "buy demo activity guard",
)

api = replace_once(
    api,
    '''            if not r.get("canonical_market_identity"):
                r["action"] = "Blocked"
                r["reason"] = (
                    "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED — no canonical market URL in runtime index."
                )
            elif trade_status == "PAPER_ELIGIBLE":
                r["action"] = "Demo Buy candidate"
                r["reason"] = "Eligible for bounded paper/demo exploration."
            else:
''',
    '''            activity_status = str(r.get("market_activity_status") or "ACTIVITY_UNKNOWN")
            if not r.get("canonical_market_identity"):
                r["action"] = "Blocked"
                r["reason"] = (
                    "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED — no canonical market URL in runtime index."
                )
            elif activity_status != "ACTIVE_PROVIDER_TXNS":
                r["action"] = "Watch"
                r["reason"] = (
                    r.get("activity_trade_block_reason")
                    or f"{activity_status} — market is not classified as active provider flow."
                )
            elif trade_status == "PAPER_ELIGIBLE":
                r["action"] = "Demo Buy candidate"
                r["reason"] = "Eligible for bounded paper/demo exploration."
            else:
''',
    "market opportunities activity classification",
)

api_path.write_text(api, encoding="utf-8")

# ------------------------------------------------------------------
# 6) Focused tests for activity/display separation
# ------------------------------------------------------------------
test_path = Path("tests/test_ae18_market_activity_trade_axis.py")
test_path.write_text(r'''from app.clean_forward.market_activity import (
    ACTIVE_PROVIDER_TXNS,
    ACTIVITY_STAGNANT,
    ACTIVITY_UNKNOWN,
    NO_RECENT_PROVIDER_TXNS,
    evaluate_market_activity,
)

def test_missing_symbols_do_not_affect_active_market_activity():
    row = {
        "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
        "price_usd": 1.0,
        "liquidity_usd": 10000,
        "txns_h24_buys": 10,
        "txns_h24_sells": 5,
        "volume_h24": 1234,
        "price_change_h24": 2.5,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == ACTIVE_PROVIDER_TXNS
    assert out["activity_uses_symbol_display"] is False
    assert out["market_activity_blocks_demo_entry"] is False

def test_zero_txns_are_not_actionable_even_with_liquidity():
    row = {
        "symbol_pair_display": "PRETTY/USDC",
        "price_usd": 1.0,
        "liquidity_usd": 999999,
        "market_cap": 5000000,
        "txns_h24_buys": 0,
        "txns_h24_sells": 0,
        "volume_h24": 0,
        "price_change_m5": 0,
        "price_change_h1": 0,
        "price_change_h6": 0,
        "price_change_h24": 0,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == NO_RECENT_PROVIDER_TXNS
    assert out["market_activity_blocks_demo_entry"] is True
    assert out["activity_uses_liquidity_or_market_cap_as_activity_proxy"] is False

def test_static_metadata_without_activity_is_stagnant():
    row = {
        "price_usd": 0.01,
        "liquidity_usd": 10000,
        "volume_h24": 0,
        "price_change_h24": 0,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == ACTIVITY_STAGNANT
    assert out["market_activity_blocks_demo_entry"] is True

def test_missing_activity_metadata_is_unknown_not_active():
    out = evaluate_market_activity({"price_usd": 1.0})
    assert out["market_activity_status"] == ACTIVITY_UNKNOWN
    assert out["market_activity_blocks_demo_entry"] is True
''', encoding="utf-8")

# ------------------------------------------------------------------
# Syntax checks
# ------------------------------------------------------------------
for p in [
    ma_path,
    idx_path,
    canon_path,
    feed_path,
    api_path,
    test_path,
]:
    py_compile.compile(str(p), doraise=True)
    print(f"OK syntax: {p}")

print("AE18 stage4B market activity axis patch applied.")
