"""AE14 focused tests — Clean Forward → GateKeeper/demo candidate bridge."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_cf_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "row_id": "base|pair|0xABC123",
        "source_provider": "dexscreener",
        "normalized_chain_id": "base",
        "chain": "base",
        "chain_id": "base",
        "provider_pair_id": "0xABC123",
        "pair_address": "0xABC123",
        "base_token_address": "0xBASETOKEN",
        "base_token_symbol": "SOL",
        "quote_token_address": "0xQUOTE",
        "quote_token_symbol": "cbBTC",
        "pair": "SOL/cbBTC",
        "pair_label": "SOL/cbBTC",
        "price": "77.77",
        "price_usd": "77.77",
        "liquidity": 558118.71,
        "liquidity_usd": 558118.71,
        "observed_at": _fresh_ts(),
        "fetched_at": _fresh_ts(),
        "last_fetched": _fresh_ts(),
        "ingested_at": _fresh_ts(),
        "verification_status": "provider_pair_verified",
        "freshness_status": "fresh",
        "identity_status": "pair_and_tokens_separated",
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "address_role": "pair_contract",
        "price_change_1h": 0.5,
        "txns_24h_buys": 10,
        "txns_24h_sells": 8,
    }
    row.update(overrides)
    return row


def test_01_valid_row_passes_bridge_normalization():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(_valid_cf_row())
    assert result["ok"] is True
    assert result["legacy_market_snapshots_used"] is False
    assert result["source"] == "clean_forward_market_feed"
    cand = result["candidate"]
    assert cand is not None
    assert cand["latest_price"] == pytest.approx(77.77)
    assert cand["latest_liquidity"] == pytest.approx(558118.71)
    assert cand["price_updated_at"]
    assert cand["liquidity_updated_at"] == cand["price_updated_at"]
    assert cand["candidate_source"] == "clean_forward_market_feed"
    assert cand["paper_demo_only"] is True
    assert cand["live_trading_ready"] is False
    assert cand["legacy_market_snapshots_used"] is False
    assert cand["coin_id"] is None
    assert cand["id"] is None


def test_02_missing_price_returns_ok_false_no_throw():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(price_usd=None, price=None)
    )
    assert result["ok"] is False
    assert result["candidate"] is None
    assert any("price" in r for r in result["block_reasons"])


def test_03_invalid_numeric_string_returns_ok_false_no_throw():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(price_usd="not-a-number", price="not-a-number")
    )
    assert result["ok"] is False
    assert any("price" in r for r in result["block_reasons"])


def test_04_missing_liquidity_returns_ok_false_no_throw():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(liquidity_usd=None, liquidity=None)
    )
    assert result["ok"] is False
    assert any("liquidity" in r for r in result["block_reasons"])


def test_05_missing_timestamp_returns_ok_false_no_throw():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(
            observed_at=None,
            fetched_at=None,
            last_fetched=None,
            ingested_at=None,
        )
    )
    assert result["ok"] is False
    assert any("timestamp" in r for r in result["block_reasons"])


def test_06_live_trading_ready_true_rejected():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(live_trading_ready=True)
    )
    assert result["ok"] is False
    assert any("live_trading_ready" in r for r in result["block_reasons"])


def test_07_shown_as_token_contract_true_rejected():
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    result = build_clean_forward_gatekeeper_candidate(
        _valid_cf_row(shown_as_token_contract=True)
    )
    assert result["ok"] is False
    assert any("shown_as_token_contract" in r for r in result["block_reasons"])


def test_08_demo_queue_evaluate_uses_clean_forward_price(tmp_path, monkeypatch):
    from app.ae13b_product import demo_queue
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows

    monkeypatch.setattr(demo_queue, "DATA_DIR", tmp_path)
    monkeypatch.setattr(demo_queue, "QUEUE_PATH", tmp_path / "demo_trade_queue.json")

    row = _valid_cf_row(
        pair_address="0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        provider_pair_id="0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        chain="base",
        price_usd="12.34",
        price="12.34",
        liquidity_usd=100000.0,
        liquidity=100000.0,
    )
    set_cached_clean_forward_rows([row])

    entry = demo_queue.add_to_demo_queue(
        symbol="SOL",
        pair="SOL/cbBTC",
        chain="base",
        contract_or_pair_address="0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        source="ae14_clean_forward_inline_queue_eval_smoke",
        market_match_status="provider_pair_verified",
        risk_mode="paper_demo_only_clean_forward",
        max_notional=100.0,
        user_hypothesis="AE14 smoke",
    )

    # Identity resolver still returns no matched_price (the old blocker path).
    with patch(
        "app.ae13b_product.contract_resolver.resolve_identity",
        return_value={
            "resolution_status": "user_entered_identity",
            "resolution_source": "watchlist_user_input",
            "matched_symbol": "SOL",
            "matched_chain": "base",
            "matched_pair_address": None,
            "matched_price": None,
            "matched_price_ts": None,
            "matched_liquidity": None,
            "paper_demo_only": True,
        },
    ):
        result = demo_queue.evaluate_queue_item(entry["queue_id"])

    assert result["ok"] is True
    assert result.get("rejection_code") != "NOT_OPENED_MISSING_PRICE"
    assert result.get("clean_forward_bridge_used") is True
    assert result.get("legacy_market_snapshots_used") is False
    gate = result.get("gate_result") or {}
    # May still block on other gates (e.g. stagnant), but not missing price.
    assert gate.get("rejection_code") != "NOT_OPENED_MISSING_PRICE"
    assert (result.get("queue_item") or {}).get("clean_forward_bridge_used") is True


def test_09_demo_bot_ae14_path_records_audit_flags(tmp_path, monkeypatch):
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows
    from app.ae13b_product.demo_bot import reset_demo_bot_for_tests, get_demo_bot

    set_cached_clean_forward_rows([_valid_cf_row()])
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    # Keep demo execution allowed
    monkeypatch.setattr(
        "app.ae13b_product.demo_bot.resolve_runtime_guard_context",
        lambda: {
            "trading_mode": "DEMO",
            "live_trading_enabled": False,
            "settings": {"starting_capital": 10000},
        },
    )
    out = bot.run_once()
    assert out.get("ok") is True
    assert out.get("clean_forward_bridge_used") is True
    assert out.get("legacy_market_snapshots_used") is False
    assert out.get("candidate_source") == "clean_forward_market_feed"
    record = (out.get("status") or {}).get("last_cycle_record") or {}
    assert record.get("clean_forward_bridge_used") is True
    assert record.get("legacy_market_snapshots_used") is False
    assert record.get("paper_demo_only") is True
    assert record.get("not_profitability_evidence") is True
    assert record.get("live_trading_ready") is False
    reset_demo_bot_for_tests()
    set_cached_clean_forward_rows(None)


def test_10_non_clean_forward_watchlist_behavior_unchanged(tmp_path, monkeypatch):
    from app.ae13b_product import demo_queue

    monkeypatch.setattr(demo_queue, "DATA_DIR", tmp_path)
    monkeypatch.setattr(demo_queue, "QUEUE_PATH", tmp_path / "demo_trade_queue.json")

    entry = demo_queue.add_to_demo_queue(
        symbol="TEST",
        chain="solana",
        contract_or_pair_address="SoLTestAddress111111111111111111111111111",
        source="watchlist_manual",
    )
    with patch(
        "app.ae13b_product.contract_resolver.resolve_identity",
        return_value={
            "resolution_status": "unresolved",
            "reason": "not in feed",
            "matched_price": None,
            "matched_price_ts": None,
            "matched_liquidity": None,
            "checked_at": _fresh_ts(),
            "paper_demo_only": True,
        },
    ):
        result = demo_queue.evaluate_queue_item(entry["queue_id"])

    assert result["ok"] is True
    assert result.get("clean_forward_bridge_used") is False
    assert result.get("legacy_market_snapshots_used") is False
    # Legacy path still surfaces missing price via GateKeeper when no match.
    assert result.get("rejection_code") == "NOT_OPENED_MISSING_PRICE" or result.get(
        "decision"
    ) in ("BLOCKED", "NOT_ENOUGH_DATA")


def test_bridge_module_has_no_heavy_imports():
    import ast

    path = ROOT / "app" / "ae13b_product" / "clean_forward_bridge.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imported.add(mod.split(".")[0] if mod else "")
            if mod.startswith("app."):
                imported.add(mod)
    # Stdlib / typing only — no app.* runtime deps.
    assert "app" not in imported
    assert not any(m.startswith("app.") for m in imported)
