from pathlib import Path
import py_compile

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

# ------------------------------------------------------------
# 1) market_activity.py — price-only is UNKNOWN, not STAGNANT
# ------------------------------------------------------------
ma_path = Path("app/clean_forward/market_activity.py")
ma = ma_path.read_text(encoding="utf-8")
ma_path.with_suffix(".py.bak_ae18_market_activity_stage4c").write_text(ma, encoding="utf-8")

ma = replace_once(
    ma,
    '''    has_static_market_metadata = any(
        v is not None and v > 0 for v in (liquidity, market_cap, fdv, price)
    )
''',
    '''    # Price alone is not enough to classify a market as stagnant.
    # Stagnant means provider reports static market context such as liquidity,
    # market cap, or FDV, but no txns/volume/deltas.
    has_static_market_metadata = any(
        v is not None and v > 0 for v in (liquidity, market_cap, fdv)
    )
''',
    "price-only should not imply stagnant",
)

ma_path.write_text(ma, encoding="utf-8")

# ------------------------------------------------------------
# 2) canonical_market_identity.py — expose fresh provider fetch timestamps
#    and force trade_readiness_status to activity gate result.
# ------------------------------------------------------------
canon_path = Path("app/clean_forward/canonical_market_identity.py")
canon = canon_path.read_text(encoding="utf-8")
canon_path.with_suffix(".py.bak_ae18_market_activity_stage4c").write_text(canon, encoding="utf-8")

canon = replace_once(
    canon,
    '''    activity = evaluate_market_activity(out)
    out.update(activity)

    # Missing symbols alone do not block trading. Market inactivity does.
    if activity.get("market_activity_status") != ACTIVE_PROVIDER_TXNS:
        if str(out.get("trade_readiness_status") or "") == "PAPER_ELIGIBLE":
            out["trade_readiness_status"] = activity.get("activity_trade_readiness_status")
            out["trade_block_reason"] = activity.get("activity_trade_block_reason")

    return out
''',
    '''    # Normalize timestamp aliases for all UI/API/bot consumers.
    # fetched_at / last_fetched means provider data was fetched/refreshed.
    # This does not assert that the numeric price changed.
    fetch_ts = (
        out.get("fetched_at")
        or out.get("last_fetched")
        or out.get("provider_fetch_at")
        or out.get("loaded_at")
        or out.get("last_market_update_at")
        or out.get("price_updated_at")
    )
    if fetch_ts:
        out["provider_fetch_at"] = fetch_ts
        out["market_data_refreshed_at"] = fetch_ts
        out["last_market_update_at"] = fetch_ts
        out["price_updated_at"] = fetch_ts

    activity = evaluate_market_activity(out)
    out.update(activity)

    # Missing symbols alone do not block trading. Market inactivity does.
    # Keep the functional readiness axis synchronized everywhere.
    if activity.get("market_activity_status") != ACTIVE_PROVIDER_TXNS:
        out["trade_readiness_status"] = activity.get("activity_trade_readiness_status")
        out["trade_block_reason"] = activity.get("activity_trade_block_reason")
    elif not out.get("trade_readiness_status"):
        out["trade_readiness_status"] = "PAPER_ELIGIBLE"

    return out
''',
    "timestamp aliases and hard activity readiness sync",
)

canon_path.write_text(canon, encoding="utf-8")

# ------------------------------------------------------------
# 3) runtime_identity_index.py — include timestamp aliases in index order
# ------------------------------------------------------------
idx_path = Path("app/clean_forward/runtime_identity_index.py")
idx = idx_path.read_text(encoding="utf-8")
idx_path.with_suffix(".py.bak_ae18_market_activity_stage4c").write_text(idx, encoding="utf-8")

if '"provider_fetch_at",' not in idx:
    idx = replace_once(
        idx,
        '''    "trade_readiness_status",
    "trade_block_reason",
''',
        '''    "trade_readiness_status",
    "trade_block_reason",
    "provider_fetch_at",
    "market_data_refreshed_at",
    "last_market_update_at",
    "price_updated_at",
''',
        "runtime index timestamp aliases",
    )

idx_path.write_text(idx, encoding="utf-8")

# ------------------------------------------------------------
# 4) runtime_market_feed.py — propagate timestamp aliases through all views
# ------------------------------------------------------------
feed_path = Path("app/ae13b_product/runtime_market_feed.py")
feed = feed_path.read_text(encoding="utf-8")
feed_path.with_suffix(".py.bak_ae18_market_activity_stage4c").write_text(feed, encoding="utf-8")

if '"provider_fetch_at",' not in feed:
    feed = replace_once(
        feed,
        '''ACTIVITY_FIELDS = (
''',
        '''ACTIVITY_FIELDS = (
    "provider_fetch_at",
    "market_data_refreshed_at",
    "last_market_update_at",
    "price_updated_at",
''',
        "activity/timestamp feed propagation",
    )

feed_path.write_text(feed, encoding="utf-8")

# ------------------------------------------------------------
# 5) api.py — hard gate demo buy by market_activity_status stays above price
#    Already added in stage4B; no code change unless absent.
# ------------------------------------------------------------

for p in [ma_path, canon_path, idx_path, feed_path]:
    py_compile.compile(str(p), doraise=True)
    print(f"OK syntax: {p}")

print("AE18 stage4C unknown/timestamp/activity-readiness patch applied.")
