"""AE13F targeted tests — identity model, tracking, filters, stale price, provider, safety."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ae13b_product.contract_resolver import (
    STATUS_USER_ENTERED,
    normalize_resolution_status,
    resolve_identity,
)
from app.ae13b_product.demo_queue import add_to_demo_queue, evaluate_queue_item
from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
from app.ae13b_product.external_resolver import (
    MODE_LOCAL,
    attempt_external_lookup,
    get_external_resolver_status,
    set_external_resolver_mode,
)
from app.ae13b_product.identity_model import (
    apply_resolved_only,
    attach_identity_objects,
    build_user_entered_identity,
)
from app.ae13b_product.live_market import DEFAULT_FILTER_MODE, _SEMANTIC_FILTERS, build_live_market
from app.ae13b_product.provider_status import build_provider_status
from app.ae13b_product.stale_price_status import build_stale_price_status, row_price_freshness
from app.analytics.watchlist import (
    display_coalesce,
    enable_external_lookup_for_watchlist_item,
    list_watchlist,
    resolve_watchlist_identity,
    run_watchlist_semantic_check,
    set_tracking_enabled,
    set_watchlist_evidence,
    update_watchlist_identity,
    upsert_watchlist_item,
)


@pytest.fixture()
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.analytics.watchlist.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.analytics.watchlist.WATCHLIST_PATH", tmp_path / "watchlist.json")
    monkeypatch.setattr("app.ae13b_product.demo_queue.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.ae13b_product.demo_queue.QUEUE_PATH", tmp_path / "demo_trade_queue.json")
    monkeypatch.setattr("app.ae13b_product.external_resolver.DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "app.ae13b_product.external_resolver.CONFIG_PATH", tmp_path / "external_resolver_config.json"
    )
    monkeypatch.setattr(
        "app.ae13b_product.external_resolver.CACHE_PATH", tmp_path / "external_resolver_cache.json"
    )
    return tmp_path


def test_non_destructive_identity_model(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    monkeypatch.setattr("app.database.get_coins", lambda **kwargs: [])
    entry = upsert_watchlist_item(
        name="Giggle Fund",
        symbol="GIGGLE",
        contract_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        chain="bsc",
        expected_category="user thinks social",
    )
    assert entry["user_entered_name"] == "Giggle Fund"
    assert entry["user_entered_symbol"] == "GIGGLE"
    assert entry["display_name"] == "Giggle Fund"
    assert entry["display_symbol"] == "GIGGLE"
    assert entry["display_symbol"] != "—"

    user_before = build_user_entered_identity(entry)
    result = resolve_watchlist_identity(entry["id"])
    assert result["ok"] is True
    assert result["identity_preserved"] is True
    item = result["item"]
    assert item["user_entered_name"] == "Giggle Fund"
    assert item["user_entered_symbol"] == "GIGGLE"
    assert item["user_entered_contract_or_pair_address"].lower().startswith("0x20d601")
    # Resolver must not erase user identity
    for k, v in user_before.items():
        if k in ("created_at", "updated_at"):
            continue
        if v:
            assert item.get(k) == v
    assert item.get("resolved_identity")
    assert item.get("user_entered_identity")
    assert item["identity_resolution_status"] in (
        STATUS_USER_ENTERED,
        "user_entered_identity",
        "unresolved_local_only",
    )
    reason = (result.get("resolution") or {}).get("reason") or ""
    assert reason
    assert "user input" in reason.lower() or "not found" in reason.lower() or "external" in reason.lower()


def test_apply_resolved_only_never_overwrites_user():
    entry = {
        "user_entered_symbol": "GIGGLE",
        "user_entered_name": "Giggle Fund",
        "user_entered_chain": "bsc",
        "user_entered_contract_or_pair_address": "0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
    }
    apply_resolved_only(
        entry,
        {
            "resolution_status": "local_match",
            "matched_symbol": "OTHER",
            "matched_name": "Other Name",
            "matched_chain": "ethereum",
            "matched_contract_address": "0xdead",
            "reason": "test",
            "confidence": 0.9,
        },
    )
    assert entry["user_entered_symbol"] == "GIGGLE"
    assert entry["user_entered_name"] == "Giggle Fund"
    assert entry["resolved_symbol"] == "OTHER"
    attach_identity_objects(entry)
    assert entry["user_entered_identity"]["user_entered_symbol"] == "GIGGLE"
    assert entry["resolved_identity"]["resolved_symbol"] == "OTHER"


def test_manual_identity_enrichment(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    entry = upsert_watchlist_item(
        contract_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        chain="bsc",
    )
    updated = update_watchlist_identity(
        entry["id"],
        name="Giggle Fund",
        symbol="GIGGLE",
        chain="bsc",
        contract_or_pair_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        display_label="Giggle Fund",
    )
    assert updated["user_entered_name"] == "Giggle Fund"
    assert updated["display_name"] == "Giggle Fund"
    ev = set_watchlist_evidence(
        entry["id"],
        user_evidence_note="charitable/educational narrative",
        user_claimed_social_mission="educational charity fund",
        user_expected_category="user thinks social",
    )
    assert ev["semantic_classification"] == "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
    assert ev["semantic_classification"] != "SOCIAL_CONFIRMED"


def test_watchlist_tracking(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    entry = upsert_watchlist_item(symbol="TEST", chain="solana", contract_address="SoLTest111")
    tracked = set_tracking_enabled(entry["id"], True)
    assert tracked["tracking_enabled"] is True
    assert tracked["collection_status"]
    stopped = set_tracking_enabled(entry["id"], False)
    assert stopped["tracking_enabled"] is False
    # Stop tracking does not remove
    assert any(i["id"] == entry["id"] for i in list_watchlist())


def test_semantic_without_market_match(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    monkeypatch.setattr("app.database.get_coins", lambda **kwargs: [])
    entry = upsert_watchlist_item(
        name="Giggle Fund",
        symbol="GIGGLE",
        contract_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        chain="bsc",
        expected_category="user thinks social",
    )
    set_watchlist_evidence(
        entry["id"],
        user_evidence_note="charitable educational mission",
        user_claimed_social_mission="charity",
    )
    sem = run_watchlist_semantic_check(entry["id"])
    assert sem["ok"] is True
    assert sem["requires_market_match"] is False
    fam = sem["semantic_signal_family"]
    assert fam != "SOCIAL_CONFIRMED"
    assert fam in (
        "SOCIAL_CANDIDATE_NEEDS_VERIFICATION",
        "NEEDS_REVIEW",
        "UNKNOWN_INSUFFICIENT_EVIDENCE",
    )


def test_demo_queue_missing_and_stale_price(isolated_data, monkeypatch):
    monkeypatch.setattr(
        "app.ae13b_product.contract_resolver.resolve_identity",
        lambda **kwargs: {
            "resolution_status": "user_entered_identity",
            "reason": "no price",
            "matched_price": None,
            "checked_at": "2026-07-18T00:00:00+00:00",
        },
    )
    entry = add_to_demo_queue(
        watchlist_id="wl-giggle",
        symbol="GIGGLE",
        chain="bsc",
        contract_or_pair_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
    )
    result = evaluate_queue_item(entry["queue_id"])
    assert result["ok"] is True
    assert result["decision"] == "NOT_ENOUGH_DATA"
    assert "price" in (result["reason"] or "").lower() or "market" in (result["reason"] or "").lower()
    assert result["paper_demo_only"] is True

    monkeypatch.setattr(
        "app.ae13b_product.contract_resolver.resolve_identity",
        lambda **kwargs: {
            "resolution_status": "local_match",
            "reason": "matched",
            "matched_price": 1.0,
            "matched_price_ts": "2020-01-01T00:00:00+00:00",
            "checked_at": "2026-07-18T00:00:00+00:00",
        },
    )
    result2 = evaluate_queue_item(entry["queue_id"])
    assert result2["decision"] in ("BLOCKED", "NOT_ENOUGH_DATA")
    assert result2.get("stale_price_status") or "stale" in (result2.get("reason") or "").lower()


def test_filters_hide_not_dim_default(monkeypatch):
    assert DEFAULT_FILTER_MODE == "hide"
    monkeypatch.setattr(
        "app.database.get_coins",
        lambda **kwargs: [
            {
                "id": 1,
                "symbol": "SOC",
                "name": "Social",
                "chain": "solana",
                "pair_address": "PairSoc",
                "latest_price": 1.0,
                "latest_liquidity": 5000,
                "latest_volume_24h": 1000,
                "latest_whale_score": 0.5,
                "last_seen_at": "2099-01-01T00:00:00+00:00",
            },
            {
                "id": 2,
                "symbol": "OPP",
                "name": "Opp",
                "chain": "solana",
                "pair_address": "PairOpp",
                "latest_price": 1.0,
                "latest_liquidity": 5000,
                "latest_volume_24h": 1000,
                "latest_whale_score": 0.4,
                "last_seen_at": "2099-01-01T00:00:00+00:00",
            },
        ],
    )
    monkeypatch.setattr(
        "app.ae13b_product.live_market._latest_snapshot_map",
        lambda ids: {
            1: {"timestamp": "2099-01-01T00:00:00+00:00"},
            2: {"timestamp": "2099-01-01T00:00:00+00:00"},
        },
    )

    class FakeReg:
        def observe_candidate(self, cand):
            sym = cand.get("symbol")
            if sym == "SOC":
                fam = "SOCIAL_CONFIRMED"
            else:
                fam = "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"
            return {
                "semantic_signal_family": fam,
                "semantic_label_human": fam,
                "trading_opportunity_state": "DEMO_CANDIDATE",
                "coin_identity": f"coin:{sym}",
                "registry_key": f"coin:{sym}",
            }

    monkeypatch.setattr(
        "app.ae13_semantic.runtime_registry.get_semantic_registry",
        lambda: FakeReg(),
    )
    data = build_live_market(limit=50, status_filter="social", filter_mode="hide")
    assert data["filter_hides_non_matching"] is True
    assert data["filter_backend_authoritative"] is True
    assert data["filter_mode"] == "hide"
    assert all(r["semantic_signal_family"] == "SOCIAL_CONFIRMED" for r in data["rows"])
    assert data["total_before_filter"] == 2
    assert data["count"] == 1
    assert "Showing" in data["filter_result_label"]

    highlight = build_live_market(limit=50, status_filter="social", filter_mode="highlight")
    assert highlight["filter_mode"] == "highlight"
    assert len(highlight["rows"]) == 2


def test_backend_filter_strict_equality():
    assert _SEMANTIC_FILTERS["social"] == frozenset({"SOCIAL_CONFIRMED"})
    assert "OPPORTUNISTIC_SUSPECTED" in _SEMANTIC_FILTERS["opportunistic"]
    assert "SOCIAL_CONFIRMED" not in _SEMANTIC_FILTERS["opportunistic"]
    assert "NEEDS_REVIEW" in _SEMANTIC_FILTERS["unknown"]


def test_keyed_live_market_reconciliation(monkeypatch):
    monkeypatch.setattr(
        "app.database.get_coins",
        lambda **kwargs: [
            {
                "id": 1,
                "symbol": "AAA",
                "name": "Aaa",
                "chain": "solana",
                "pair_address": "PairAAA111",
                "contract_address": "TokAAA",
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
                "semantic_signal_family": "NEEDS_REVIEW",
                "semantic_label_human": "Needs review",
                "trading_opportunity_state": "WATCH",
                "coin_identity": "coin:aaa",
                "registry_key": "coin:aaa",
            }

    monkeypatch.setattr(
        "app.ae13_semantic.runtime_registry.get_semantic_registry",
        lambda: FakeReg(),
    )
    data = build_live_market(limit=10)
    assert data["reconciliation"]["strategy"] == "keyed_reconciliation"
    assert data["reconciliation"]["array_index_as_key"] is False
    row = data["rows"][0]
    assert row["row_key"] == "solana|pair|PairAAA111"
    assert row.get("price_freshness")
    assert row.get("stale_price_applies_to") == "selected_candidate"


def test_stale_price_status_scoped():
    st = build_stale_price_status(
        applies_to="selected_candidate",
        last_price_timestamp="2020-01-01T00:00:00+00:00",
        affected_symbol="WIF/SOL",
        source="demo_queue",
        blocks_demo_trade=True,
        market_feed_active=True,
    )
    assert st["stale_price_applies_to"] == "selected_candidate"
    assert st["blocks_demo_trade"] is True
    assert "WIF" in st["label"] or "stale" in st["label"].lower() or "blocked" in st["label"].lower()
    assert st["price_age_seconds"] is not None
    assert st["freshness_limit_seconds"]

    missing = row_price_freshness(price=None, timestamp=None, symbol="GIGGLE")
    assert missing["price_missing"] is True
    assert "No current price" in missing["label"]


def test_provider_status_fail_soft(monkeypatch):
    status = build_provider_status()
    assert "provider_selected" in status or "llm_provider_selected" in status
    assert status["local_rules_active"] is True
    assert status["demo_trading_blocked_by_provider"] is False
    assert status.get("provider_status_explanation")
    assert status.get("fail_soft") is True
    assert status["provider_health"] in (
        "active",
        "unavailable_metrics_helper",
        "inactive",
    )


def test_external_resolver_not_silent(isolated_data):
    status = get_external_resolver_status()
    assert status["external_resolver_enabled"] is False
    assert status["silent_calls_forbidden"] is True
    set_external_resolver_mode(MODE_LOCAL)
    result = attempt_external_lookup(
        chain="bsc",
        contract_or_pair_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        user_confirmed=False,
    )
    assert result["external_resolver_attempted"] is False
    assert "not enabled" in result["reason"].lower() or "local" in result["reason"].lower()


def test_risk_guard_preserved():
    risk = evaluate_demo_risk_guard(
        requested_notional=50,
        demo_equity=10_000,
        price=1.0,
        price_timestamp="2020-01-01T00:00:00+00:00",
        pair_address="p2",
    )
    assert risk["risk_guard_passed"] is False
    assert "stale" in risk["risk_guard_reason"].lower()
    assert risk["paper_demo_only"] is True


def test_safety_flags_in_watchlist(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    entry = upsert_watchlist_item(symbol="SAFE", chain="solana")
    assert entry["paper_demo_only"] is True
    assert entry["not_live_approved"] is True
    assert entry["live_trading_implied"] is False


def test_resolution_status_normalization():
    assert normalize_resolution_status("matched_live_market") == "local_match"
    assert normalize_resolution_status("user_entered_only") == "user_entered_identity"
    assert normalize_resolution_status("unresolved") == "unresolved_local_only"


def test_giggle_like_display_and_resolve(isolated_data, monkeypatch):
    monkeypatch.setattr("app.analytics.watchlist._feed_registry", lambda *a, **k: None)
    monkeypatch.setattr("app.database.get_coins", lambda **kwargs: [])
    entry = upsert_watchlist_item(
        name="Giggle Fund",
        symbol="GIGGLE",
        contract_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        chain="bsc",
        expected_category="user thinks social",
    )
    d = display_coalesce(entry)
    assert d["display_name"] == "Giggle Fund"
    assert d["display_symbol"] == "GIGGLE"
    assert "0x20d601" in d["display_id"].lower()
    res = resolve_identity(
        chain="bsc",
        contract_or_pair_address="0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
        symbol="GIGGLE",
        allow_external=False,
    )
    assert res["resolution_status"] in (
        "user_entered_identity",
        "unresolved_local_only",
        "local_match",
    )
    assert res["reason"]
    assert res["external_resolver_attempted"] is False
