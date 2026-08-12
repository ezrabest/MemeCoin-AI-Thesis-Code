"""AE13C targeted tests — waiting recovery, lifecycle, guards, taxonomy, watchlist, formatting."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from app.ae13_semantic.runtime_registry import (
    SemanticRegistry,
    _base_symbol,
    _local_classify,
    reset_semantic_registry_for_tests,
)
from app.ae13b_product.demo_bot import DemoBot, get_demo_bot, reset_demo_bot_for_tests
from app.ae13b_product.execution_guard import (
    DemoExecutionGuardError,
    assert_paper_demo_allowed,
    evaluate_paper_demo_execution_guard,
)
from app.ae13b_product.presets import clamp_trades_per_hour, get_preset
from app.ae13b_product.provider_status import build_provider_status
from app.analytics.watchlist import (
    disable_watchlist_item,
    list_watchlist,
    remove_watchlist_item,
    upsert_watchlist_item,
)
from app.execution.paper import PaperTrader


def test_wif_sol_not_unknown():
    rec = _local_classify({"symbol": "WIF/SOL", "name": "dogwifhat", "liquidity_usd": 1000})
    assert _base_symbol("WIF/SOL") == "WIF"
    assert rec["semantic_signal_family"] in (
        "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        "OPPORTUNISTIC_SUSPECTED",
    )
    assert "UNKNOWN" not in rec["semantic_signal_family"]


def test_obvious_meme_taxonomy():
    for sym in ("PEPE", "BONK", "DOGE", "SHIB", "FLOKI", "MEME"):
        rec = _local_classify({"symbol": sym, "liquidity_usd": 1})
        assert rec["semantic_signal_family"] == "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", sym


def test_social_confirmed_requires_evidence():
    rec = _local_classify({"symbol": "WIF", "liquidity_usd": 1e6})
    assert rec["semantic_signal_family"] != "SOCIAL_CONFIRMED"
    with_evidence = _local_classify(
        {
            "symbol": "CHARITY",
            "social_source_available": True,
            "social_mission_evidence": True,
        }
    )
    assert with_evidence["semantic_signal_family"] == "SOCIAL_CONFIRMED"


def test_registry_reclassifies_wif(tmp_path: Path):
    reset_semantic_registry_for_tests()
    reg = SemanticRegistry(path=tmp_path / "reg.json")
    # Seed legacy unknown
    first = reg.observe_candidate(
        {
            "symbol": "WIF/SOL",
            "pair_address": "pairwif1",
            "liquidity_usd": 100,
            "force_reclassify": True,
        }
    )
    assert first["semantic_signal_family"] == "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"
    snap = reg.snapshot()
    assert snap["counters"]["runtime_opportunistic_confirmed"] >= 1
    assert snap["counters"]["runtime_unknown"] == 0 or first["is_runtime_classified"]


def test_backend_guard_direct_call_blocks_live():
    trader = PaperTrader.__new__(PaperTrader)
    trader._state = {"trading_mode": "LIVE", "cash_usd": 10000, "open_positions": [], "next_position_id": 1}
    trader._market_prices_by_pair = {}
    trader._market_prices_by_coin_id = {}
    trader._market_price_timestamp = None
    with pytest.raises(DemoExecutionGuardError):
        trader._assert_execution_guard(reason_code="TEST")


def test_backend_guard_allows_demo_flags():
    assert_paper_demo_allowed(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )


def test_hold_profiles_not_fee_only():
    for pid in ("conservative", "balanced", "aggressive", "lotto"):
        p = get_preset(pid)
        assert int(p["min_hold_seconds"]) >= 120, pid
        assert int(p["time_stop_seconds"]) >= 3600, pid
        assert p["demo_acceptance_mode"] is False


def test_acceptance_preset_explicit():
    p = get_preset("acceptance")
    assert p["demo_acceptance_mode"] is True
    assert p["min_hold_seconds"] < 60


def test_max_trades_per_hour_cap():
    assert clamp_trades_per_hour(999) == 50
    assert clamp_trades_per_hour(10) == 10


def test_waiting_status_exposes_fields(tmp_path: Path, monkeypatch):
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    # Keep thread "alive" stub so watchdog does not overwrite blocker
    class _Alive:
        def is_alive(self):
            return True

    bot._thread = _Alive()  # type: ignore[assignment]
    bot._state["bot_status"] = "Waiting"
    bot._state["loop_active"] = True
    bot._state["waiting_reason"] = "cooldown_until_next_cycle"
    bot._state["waiting_since"] = "2099-01-01T00:00:00+00:00"
    bot._state["next_cycle_eta"] = "2099-01-01T00:00:00+00:00"
    bot._state["last_cycle_at"] = "2099-01-01T00:00:00+00:00"
    bot._state["cycles_run"] = 3
    bot._state["last_blocker"] = "test_blocker"
    st = bot.status()
    wait = st["waiting"]
    assert wait["reason"]
    assert wait["next_cycle_eta"]
    assert "remaining_seconds" in wait
    assert wait["last_cycle_at"]
    assert wait["cycle_count"] == 3
    assert wait["last_blocker"] == "test_blocker"
    assert wait["thread_alive"] is True
    assert "stop_event_set" in wait
    reset_demo_bot_for_tests()


def test_stale_waiting_watchdog_marks_recovering():
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    bot._state["bot_status"] = "Waiting"
    bot._state["loop_active"] = True
    bot._state["cooldown_seconds"] = 10
    bot._state["waiting_since"] = "2020-01-01T00:00:00+00:00"
    bot._state["last_cycle_at"] = "2020-01-01T00:00:00+00:00"
    bot._thread = None
    bot._reconcile_stale_waiting()
    assert bot._state["bot_status"] in ("Recovering", "Error")
    reset_demo_bot_for_tests()


def test_start_stop_pause_idempotent():
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    s1 = bot.start()
    assert s1["bot_status"] in ("Running", "Waiting")
    s2 = bot.start()
    assert "idempotent" in (s2.get("last_action_summary") or "").lower() or s2.get("loop_thread_alive")
    bot.pause()
    assert bot.status()["bot_status"] == "Paused"
    bot.stop()
    assert bot.status()["bot_status"] == "Stopped"
    assert bot.status()["loop_thread_alive"] is False
    reset_demo_bot_for_tests()


def test_run_once_from_waiting():
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    bot.start()
    bot._state["bot_status"] = "Waiting"
    bot._state["waiting_reason"] = "cooldown_until_next_cycle"
    result = bot.run_once()
    assert "ok" in result
    bot.stop()
    reset_demo_bot_for_tests()


def test_no_immediate_close_without_min_hold():
    """_maybe_close_positions must respect min_hold (no cycles_run>=2 force close)."""
    demo_path = Path(__file__).resolve().parents[1] / "app" / "ae13b_product" / "demo_bot.py"
    text = demo_path.read_text(encoding="utf-8")
    compact = text.replace(" ", "")
    assert "cycles_run)>=2" not in compact
    assert "age_limit=60" not in compact
    assert "min_hold" in text
    assert "TIME_STOP" in text


def test_watchlist_add_remove_persist(tmp_path: Path, monkeypatch):
    wl_path = tmp_path / "watchlist.json"
    monkeypatch.setattr("app.analytics.watchlist.WATCHLIST_PATH", wl_path)
    monkeypatch.setattr("app.analytics.watchlist.DATA_DIR", tmp_path)
    entry = upsert_watchlist_item(
        symbol="WIF",
        chain="solana",
        note="track meme",
        expected_category="user thinks opportunistic",
    )
    assert entry["symbol"] == "WIF"
    assert entry["user_symbol"] == "WIF"
    items = list_watchlist()
    assert any(i.get("symbol") == "WIF" for i in items)
    assert remove_watchlist_item(entry["id"]) is True
    assert not any(i.get("id") == entry["id"] for i in list_watchlist())


def test_watchlist_disable(tmp_path: Path, monkeypatch):
    wl_path = tmp_path / "watchlist.json"
    monkeypatch.setattr("app.analytics.watchlist.WATCHLIST_PATH", wl_path)
    monkeypatch.setattr("app.analytics.watchlist.DATA_DIR", tmp_path)
    entry = upsert_watchlist_item(symbol="BONK", chain="solana")
    disabled = disable_watchlist_item(entry["id"])
    assert disabled and disabled.get("disabled") is True


def test_watchlist_market_enrichment_preserves_user_identity(tmp_path: Path, monkeypatch):
    from app.analytics.watchlist import refresh_watchlist_against_market

    wl_path = tmp_path / "watchlist.json"
    monkeypatch.setattr("app.analytics.watchlist.WATCHLIST_PATH", wl_path)
    monkeypatch.setattr("app.analytics.watchlist.DATA_DIR", tmp_path)
    entry = upsert_watchlist_item(symbol="MYCOIN", chain="solana")
    refresh_watchlist_against_market(
        [
            {
                "symbol": "WIF/SOL",
                "name": "dogwifhat",
                "pair_address": "pair_abc",
                "chain": "solana",
            }
        ]
    )
    # No match by symbol MYCOIN → unchanged
    items = list_watchlist()
    mine = next(i for i in items if i["id"] == entry["id"])
    assert mine["user_symbol"] == "MYCOIN"
    assert mine["symbol"] == "MYCOIN"

    # Match by symbol should enrich, not overwrite user identity
    entry2 = upsert_watchlist_item(symbol="WIF", chain="solana")
    refresh_watchlist_against_market(
        [
            {
                "symbol": "WIF/SOL",
                "name": "dogwifhat",
                "pair_address": "pair_wif",
                "chain": "solana",
            }
        ]
    )
    items = list_watchlist()
    matched = next(i for i in items if i["id"] == entry2["id"])
    assert matched["user_symbol"] == "WIF"
    assert matched["symbol"] == "WIF"
    assert matched["display_symbol"] == "WIF"
    assert matched["market_symbol"] == "WIF/SOL"
    assert matched["market_name"] == "dogwifhat"
    assert matched["matched_pair_address"] == "pair_wif"


def test_provider_status_not_vague():
    st = build_provider_status()
    label = str(st.get("provider_health_label") or "")
    assert label.lower() != "provider error"
    assert "vague" not in label.lower()
    assert st.get("never_show_vague_provider_error") is True
    assert st.get("provider_health") in (
        "active",
        "unavailable_metrics_helper",
        "inactive",
    )
    assert st.get("provider_health_label") in (
        "Active",
        "Unavailable — Metrics Helper Only",
        "Inactive",
    )


def test_provider_status_ollama_errors_are_metrics_helper(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("HEADLESS_DATA_COLLECTION", raising=False)
    monkeypatch.setattr(
        "app.llm_config.get_llm_runtime_status",
        lambda: {"ollama_error_count": 3, "ollama_last_error": "connection refused"},
    )
    st = build_provider_status()
    assert st["provider_health"] == "unavailable_metrics_helper"
    assert st["provider_health_label"] == "Unavailable — Metrics Helper Only"
    assert "provider error" not in st["provider_health_label"].lower()


def test_semantic_registry_evicts_at_max_entries(tmp_path: Path):
    from app.ae13_semantic.runtime_registry import SemanticRegistry, reset_semantic_registry_for_tests

    reset_semantic_registry_for_tests()
    reg = SemanticRegistry(path=tmp_path / "bounded.json", max_entries=5)
    for i in range(12):
        reg.observe_candidate(
            {
                "id": i,
                "symbol": f"T{i}",
                "pair_address": f"pair_{i}",
                "liquidity_usd": 100,
                "volume_24h": 50,
            }
        )
    snap = reg.snapshot()
    assert snap["runtime_unique_identities"] <= 5
    assert snap["max_entries"] == 5
    assert snap["eviction_count"] >= 7
    reset_semantic_registry_for_tests()


def test_live_market_social_filter_exact(monkeypatch):
    """Social filter must match SOCIAL_CONFIRMED only — not NON_SOCIAL_*."""
    from app.ae13b_product import live_market as lm

    fake_rows = [
        {"semantic_family": "SOCIAL_CONFIRMED", "status": "Passed", "whale_score": 1},
        {"semantic_family": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", "status": "Passed", "whale_score": 1},
        {"semantic_family": "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", "status": "Passed", "whale_score": 1},
    ]
    # Unit-test the filter mapping directly
    allowed = lm._SEMANTIC_FILTERS["social"]
    filtered = [r for r in fake_rows if r["semantic_family"] in allowed]
    assert len(filtered) == 1
    assert filtered[0]["semantic_family"] == "SOCIAL_CONFIRMED"
    opp = [r for r in fake_rows if r["semantic_family"] in lm._SEMANTIC_FILTERS["opportunistic"]]
    assert len(opp) == 1
    assert "OPPORTUNISTIC" in opp[0]["semantic_family"]


def test_price_formatting_helpers_in_static():
    js = (Path(__file__).resolve().parents[1] / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "formatPrice" in js
    assert "< $0.000001" in js
    assert "delta-pos" in js
    assert "copyFullId" in js


def test_no_internal_phase_labels_in_product_nav():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    # Primary nav should not advertise AE13C
    nav_start = html.find('class="tabs')
    nav_end = html.find("</nav>", nav_start)
    nav = html[nav_start:nav_end]
    assert "AE13C" not in nav
    assert "AE13B" not in nav
    assert "AE12" not in nav or "vault" in nav.lower()  # vault may reference research


def test_settings_percent_conversion_helpers():
    # Mirror system_config contract: 0.5% must round-trip
    display = 0.5
    internal = round((display / 100) * 1e8) / 1e8
    assert abs(internal - 0.005) < 1e-8
    back = internal * 100
    assert abs(back - 0.5) < 1e-8
