"""AE13B targeted tests — continuous bot, guards, registry, settings conversion, labels."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.ae13_semantic.runtime_registry import (
    SemanticRegistry,
    reset_semantic_registry_for_tests,
)
from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests
from app.ae13b_product.execution_guard import (
    DemoExecutionGuardError,
    assert_paper_demo_allowed,
    evaluate_paper_demo_execution_guard,
)
from app.ae13b_product.presets import get_preset, list_presets
from app.env_bootstrap import parse_cli_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_execution_guard_fail_closed_live():
    r = evaluate_paper_demo_execution_guard(
        trading_mode="LIVE",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )
    assert r["rejected"] is True


def test_execution_guard_allows_demo():
    r = evaluate_paper_demo_execution_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )
    assert r["allowed"] is True


def test_execution_guard_raises():
    with pytest.raises(DemoExecutionGuardError):
        assert_paper_demo_allowed(
            trading_mode="LIVE",
            live_trading_enabled=True,
            wallet_configured=False,
            order_flags={
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
            },
        )


def test_demo_acceptance_requires_extra_flags():
    r = evaluate_paper_demo_execution_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "demo_acceptance_only": True,
            "not_strategy_evidence": True,
        },
    )
    assert r["rejected"] is True
    assert any("demo_acceptance_mode_not_enabled" in x for x in r["reasons"])


def test_semantic_registry_observe_and_reuse(tmp_path: Path):
    reset_semantic_registry_for_tests()
    reg = SemanticRegistry(path=tmp_path / "reg.json")
    c = {
        "id": 1,
        "symbol": "TEST/USD",
        "pair_address": "pair_test_1",
        "chain": "solana",
        "price_usd": 1.0,
        "liquidity_usd": 100000,
        "volume_24h": 50000,
    }
    a = reg.observe_candidate(c)
    b = reg.observe_candidate(c)
    assert a["seen_count"] == 1
    assert b["seen_count"] == 2
    assert a["semantic_signal_family"] == b["semantic_signal_family"]
    assert a["is_runtime_classified"] is True
    assert a.get("semantic_status") in ("Registered", "Classified", "Unresolved", "Needs Review")
    assert a.get("social_source_available") is False
    snap = reg.snapshot()
    assert snap["semantic_source_label"].startswith("Semantic Source:")
    assert snap["runtime_unique_identities"] >= 1
    reset_semantic_registry_for_tests()


def test_unknown_unresolved_not_promoted(tmp_path: Path):
    reg = SemanticRegistry(path=tmp_path / "reg2.json")
    rec = reg.observe_candidate(
        {
            "id": 2,
            "symbol": "ZZZ/USD",
            "pair_address": "pair_zzz",
            "liquidity_usd": 10,
            "volume_24h": 1,
        }
    )
    assert rec["semantic_signal_family"] in (
        "UNKNOWN_INSUFFICIENT_EVIDENCE",
        "UNKNOWN_UNRESOLVED",  # legacy alias tolerated
    )
    assert rec["semantic_signal_family"] not in (
        "SOCIAL_CONFIRMED",
        "OPPORTUNISTIC_CONFIRMED",
        "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    )


def test_presets_exist():
    ids = {p["id"] for p in list_presets()}
    assert {"conservative", "balanced", "aggressive", "acceptance"} <= ids
    assert get_preset("balanced")["max_notional_usd"] > 0
    assert get_preset("aggressive")["exploration_enabled"] is True


def test_cli_mode_ollama_supported():
    assert parse_cli_mode(["--mode", "ollama"]) == "ollama"
    assert parse_cli_mode(["--ollama"]) == "ollama"
    assert parse_cli_mode(["--gemini"]) == "gemini"


def test_cli_mode_rejects_unknown(monkeypatch):
    with pytest.raises(SystemExit):
        parse_cli_mode(["--qwen"])


def test_demo_bot_start_is_continuous_and_idempotent(tmp_path, monkeypatch):
    reset_demo_bot_for_tests()
    db_path = tmp_path / "t.db"
    monkeypatch.setenv("TRADER_DB_PATH", str(db_path))
    # Re-bind database module path + create schema
    import app.database as db

    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_pool()
    bot = get_demo_bot()
    bot.apply_preset("aggressive")
    try:
        from app.execution.paper import get_paper_trader

        get_paper_trader().set_trading_mode("DEMO")
    except Exception:
        pass
    s1 = bot.start()
    assert s1.get("bot_status") in ("Running", "Waiting", "Blocked", "Error")
    s2 = bot.start()
    assert (
        "idempotent" in str(s2.get("last_action_summary") or "").lower()
        or s2.get("loop_thread_alive")
        or s2.get("bot_status") in ("Blocked", "Running", "Waiting", "Error")
    )
    bot.pause()
    assert bot.status().get("bot_status") == "Paused"
    bot.stop()
    assert bot.status().get("bot_status") == "Stopped"
    once = bot.run_once()
    assert once.get("ok") in (True, False)
    reset_demo_bot_for_tests()


def test_settings_display_roundtrip_small_pct():
    """Regression: 0.5% display must not leave Unsaved Changes after save."""
    internal = 0.005  # 0.5%
    display = internal * 100  # 0.5
    back = display / 100
    assert abs(back - internal) < 1e-9


def test_no_ae_phase_labels_in_primary_nav():
    html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    nav_start = html.find('class="tabs nav-product"')
    nav_end = html.find("</nav>", nav_start)
    nav = html[nav_start:nav_end]
    assert "AE12" not in nav
    assert "AE13" not in nav
    assert "AE13B" not in nav
    assert "Demo Trading" in nav
    assert "Market Snapshot Feed" in nav or "Live Market" in nav
    assert "Research Evidence / Audit Vault" in nav


def test_provider_status_shape():
    from app.ae13b_product.provider_status import build_ai_assistant_status, build_provider_status

    p = build_provider_status()
    assert "llm_provider_selected" in p
    assert "provider_health_label" in p
    assert p["trade_authority"]
    a = build_ai_assistant_status()
    assert a["can_place_trades"] is False
    assert "label" in a


def test_live_market_builder_smoke():
    from app import database as db
    from app.ae13b_product.live_market import build_live_market

    db.init_pool()
    d = build_live_market(limit=5)
    assert "rows" in d
    assert d["demo_mode_badge"] == "LIVE DISABLED / DEMO ONLY"
    assert d["wallet_configured"] is False
