"""AE13E targeted tests — keyed refresh helpers, watchlist coalescing, demo queue, risk guard, resolver, filters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae13b_product.contract_resolver import (
    addresses_equal,
    classify_contract_format,
    normalize_address,
    normalize_chain,
    resolve_identity,
)
from app.ae13b_product.demo_queue import add_to_demo_queue, evaluate_queue_item, list_demo_queue
from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
from app.ae13b_product.live_market import _SEMANTIC_FILTERS, build_live_market
from app.analytics.watchlist import (
    display_coalesce,
    list_watchlist,
    pin_watchlist_item,
    remove_watchlist_item,
    run_watchlist_semantic_check,
    set_watchlist_evidence,
    upsert_watchlist_item,
)


@pytest.fixture()
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.analytics.watchlist.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.analytics.watchlist.WATCHLIST_PATH", tmp_path / "watchlist.json")
    monkeypatch.setattr("app.ae13b_product.demo_queue.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.ae13b_product.demo_queue.QUEUE_PATH", tmp_path / "demo_trade_queue.json")
    return tmp_path


def test_display_coalesce_user_contract_not_blank():
    entry = {
        "user_entered_contract_or_pair_address": "B6h248NJkAcBAkaCnji889a26tCiGXGN8cxhEJ4dX391",
        "user_entered_chain": "solana",
        "user_entered_symbol": None,
        "market_match_status": "waiting_for_market_match",
    }
    d = display_coalesce(entry)
    assert d["display_symbol"] != "—"
    assert "B6h248" in d["display_name"] or "Contract" in d["display_name"]
    assert d["display_id"].startswith("B6h248")
    assert d["market_match_explanation"]


def test_display_coalesce_user_symbol_primary():
    entry = {
        "user_entered_symbol": "MYCOIN",
        "user_entered_name": "My Coin",
        "user_entered_contract_or_pair_address": "0xabc",
        "market_symbol": "OTHER",
        "market_name": "Other Name",
    }
    d = display_coalesce(entry)
    assert d["display_symbol"] == "MYCOIN"
    assert d["display_name"] == "My Coin"


def test_watchlist_identity_and_actions(isolated_data, monkeypatch):
    monkeypatch.setattr(
        "app.analytics.watchlist._feed_registry",
        lambda *a, **k: None,
    )
    entry = upsert_watchlist_item(
        contract_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        chain="bsc",
        expected_category="user thinks social",
    )
    assert entry["display_symbol"] != "—"
    assert entry["display_name"].startswith("Contract") or "0x20d6" in entry["display_name"]
    assert entry["market_match_status"] == "waiting_for_market_match"
    assert entry["identity_resolution_status"] in ("user_entered_only", "user_entered_identity")
    assert entry["user_expected_category"] == "user thinks social"

    pinned = pin_watchlist_item(entry["id"], True)
    assert pinned and pinned["pinned"] is True

    ev = set_watchlist_evidence(
        entry["id"],
        user_evidence_url="https://example.com/mission",
        user_evidence_note="claimed charity",
    )
    assert ev["semantic_status"] == "evidence_provided_pending_check"
    assert ev.get("semantic_classification") != "SOCIAL_CONFIRMED" or ev.get("user_evidence_url")

    items = list_watchlist()
    assert any(i["id"] == entry["id"] for i in items)
    assert remove_watchlist_item(entry["id"]) is True


def test_demo_queue_add_evaluate(isolated_data, monkeypatch):
    monkeypatch.setattr(
        "app.ae13b_product.contract_resolver.resolve_identity",
        lambda **kwargs: {
            "resolution_status": "unresolved",
            "reason": "not in feed",
            "matched_price": None,
            "checked_at": "2026-07-18T00:00:00+00:00",
        },
    )
    entry = add_to_demo_queue(
        watchlist_id="wl-test-1",
        symbol="TEST",
        chain="solana",
        contract_or_pair_address="SoLTestAddress111111111111111111111111111",
        source="watchlist_manual",
    )
    assert entry["paper_demo_only"] is True
    assert entry["not_live_approved"] is True
    assert entry["strategy_lane"] == "Manual Watchlist Scout"
    assert any(i["queue_id"] == entry["queue_id"] for i in list_demo_queue())

    result = evaluate_queue_item(entry["queue_id"])
    assert result["ok"] is True
    assert result["paper_demo_only"] is True
    assert result["live_trading_implied"] is False
    assert result["risk_guard"]["paper_demo_only"] is True


def test_demo_risk_guard_position_pct():
    risk = evaluate_demo_risk_guard(
        requested_notional=1000,
        demo_equity=10_000,
        settings={"max_position_size_pct": 0.05},
        bot_state={"max_notional_usd": 500},
        price=1.0,
        price_timestamp="2099-01-01T00:00:00+00:00",
        pair_address="pair-a",
    )
    assert risk["risk_guard_passed"] is False
    assert "5%" in risk["risk_guard_reason"] or "portfolio" in risk["risk_guard_reason"].lower()
    assert risk["max_allowed_notional"] == 500.0  # min(500, 5% of 10k)


def test_demo_risk_guard_max_trades_and_duplicate():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    trades = [{"side": "buy", "timestamp": now, "notional_usd": 10} for _ in range(5)]
    risk = evaluate_demo_risk_guard(
        requested_notional=50,
        demo_equity=10_000,
        bot_state={"max_trades_per_hour": 4, "max_notional_usd": 100},
        recent_trades=trades,
        price=1.0,
        price_timestamp="2099-01-01T00:00:00+00:00",
        pair_address="p1",
    )
    assert risk["risk_guard_passed"] is False
    assert "trades per hour" in risk["risk_guard_reason"].lower()

    risk2 = evaluate_demo_risk_guard(
        requested_notional=50,
        demo_equity=10_000,
        open_positions=[{"pair_address": "dup", "size_usd": 10}],
        price=1.0,
        price_timestamp="2099-01-01T00:00:00+00:00",
        pair_address="dup",
    )
    assert risk2["risk_guard_passed"] is False
    assert "duplicate" in risk2["risk_guard_reason"].lower()


def test_demo_risk_guard_stale_price():
    risk = evaluate_demo_risk_guard(
        requested_notional=50,
        demo_equity=10_000,
        price=1.0,
        price_timestamp="2020-01-01T00:00:00+00:00",
        pair_address="p2",
    )
    assert risk["risk_guard_passed"] is False
    assert "stale" in risk["risk_guard_reason"].lower()


def test_contract_resolver_normalization():
    assert normalize_chain("ETH") == "ethereum"
    assert normalize_chain("bnb") == "bsc"
    assert normalize_address("0xAbCDef0123456789abcdef0123456789ABCDEF01", chain="ethereum") == (
        "0xabcdef0123456789abcdef0123456789abcdef01"
    )
    sol = "B6h248NJkAcBAkaCnji889a26tCiGXGN8cxhEJ4dX391"
    assert normalize_address(sol, chain="solana") == sol
    assert addresses_equal(
        "0xABCDEF0123456789ABCDEF0123456789ABCDEF01",
        "0xabcdef0123456789abcdef0123456789abcdef01",
        chain="ethereum",
    )
    assert classify_contract_format(sol, chain="solana") == "contract_format_valid"
    assert classify_contract_format("0x123", chain="ethereum") == "contract_format_invalid"


def test_contract_resolver_opaque_reason(monkeypatch):
    monkeypatch.setattr(
        "app.database.get_coins",
        lambda **kwargs: [],
    )
    result = resolve_identity(
        chain="solana",
        contract_or_pair_address="B6h248NJkAcBAkaCnji889a26tCiGXGN8cxhEJ4dX391",
        allow_external=False,
    )
    assert result["resolution_status"] in (
        "unresolved",
        "user_entered_only",
        "user_entered_identity",
        "unresolved_local_only",
    )
    assert result["reason"]
    assert "External resolver not called" in result["reason"] or "not include" in result["reason"].lower() or "user input" in result["reason"].lower() or "External" in result["reason"]
    assert result["external_resolver_attempted"] is False


def test_backend_semantic_filter_separation():
    assert _SEMANTIC_FILTERS["social"] == frozenset({"SOCIAL_CONFIRMED"})
    assert "OPPORTUNISTIC_SUSPECTED" in _SEMANTIC_FILTERS["opportunistic"]
    assert "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED" in _SEMANTIC_FILTERS["opportunistic"]
    assert "SOCIAL_CONFIRMED" not in _SEMANTIC_FILTERS["opportunistic"]
    assert "NEEDS_REVIEW" in _SEMANTIC_FILTERS["unknown"]
    assert "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED" in _SEMANTIC_FILTERS["infrastructure"]


def test_live_market_rows_have_stable_keys(monkeypatch):
    monkeypatch.setattr(
        "app.database.get_coins",
        lambda **kwargs: [
            {
                "id": 1,
                "symbol": "AAA",
                "name": "Aaa",
                "chain": "solana",
                "pair_address": "PairAAA111",
                "latest_price": 1.2,
                "latest_liquidity": 5000,
                "latest_volume_24h": 1000,
                "latest_whale_score": 0.5,
                "last_seen_at": "2099-01-01T00:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(
        "app.ae13b_product.live_market._latest_snapshot_map",
        lambda ids: {1: {"timestamp": "2099-01-01T00:00:00+00:00"}},
    )

    class FakeReg:
        def observe_candidate(self, cand):
            return {
                "semantic_signal_family": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
                "semantic_label_human": "Opportunistic (confirmed)",
                "semantic_status": "Classified",
                "trading_opportunity_state": "WATCH",
                "coin_identity": "coin:1",
                "first_seen_at": "2099-01-01T00:00:00+00:00",
                "last_seen_at": "2099-01-01T00:00:00+00:00",
                "seen_count": 1,
            }

    monkeypatch.setattr(
        "app.ae13_semantic.runtime_registry.get_semantic_registry",
        lambda: FakeReg(),
    )
    monkeypatch.setattr(
        "app.live.get_token_transparency_logs",
        lambda: {"passed_count": 1, "dropped_count": 0, "scan_at": None},
    )
    data = build_live_market(limit=10, status_filter="all")
    assert data["rows"]
    assert data["rows"][0]["row_key"] == "solana|pair|PairAAA111"
    social = build_live_market(limit=10, status_filter="social")
    assert social["status_filter_applied"] == "social"
    assert all(
        r.get("semantic_signal_family") == "SOCIAL_CONFIRMED" for r in social["rows"]
    )


def test_no_live_wallet_safety_flags():
    from app.ae13b_product.execution_guard import evaluate_paper_demo_execution_guard

    g = evaluate_paper_demo_execution_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )
    assert g["allowed"] is True
    assert g["wallet_configured"] is False
    assert g["private_key_accessed"] is False
    assert g["live_submission_status"] == "NOT_SUBMITTED_NO_WALLET"
    assert g["live_trading_ready"] is False
    assert g["live_trading_approval"] == "NO"
    assert g["profitability_proven"] is False
