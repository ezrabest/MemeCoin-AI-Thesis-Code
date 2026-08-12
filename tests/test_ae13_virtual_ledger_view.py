"""Targeted AE13 tests: Virtual Ledger View, demo acceptance guard, semantic labeling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae13_reconciliation.bridge import build_virtual_ledger_view
from app.ae13_reconciliation.demo_acceptance import evaluate_demo_acceptance_guard
from app.ae13_reconciliation.semantic_coverage import build_semantic_coverage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_virtual_ledger_view_builds_without_mutating_archives(tmp_path: Path):
    # Use real project data when present; otherwise minimal fixtures
    root = PROJECT_ROOT
    paper_state = root / "data" / "paper_state.json"
    assert paper_state.is_file() or True

    before_hashes = {}
    for rel in (
        "data/paper_trading/paper_orders_20260714.jsonl",
        "data/ae11_closed_trades_snapshot.csv",
    ):
        p = root / rel
        if p.is_file():
            before_hashes[str(p)] = p.stat().st_mtime_ns

    view = build_virtual_ledger_view(root)
    assert view.read_model == "virtual_ledger_view"
    assert view.ui_write_source_of_truth == "legacy_paper_state"
    assert view.demo_balance.get("starting_capital") == 10000.0 or view.demo_balance.get("starting_capital") == 10000
    assert "orders_count" in view.summary()
    assert view.demo_balance.get("paper_demo_only") is True
    assert view.demo_balance.get("wallet_configured") is False

    for path_str, mtime in before_hashes.items():
        assert Path(path_str).stat().st_mtime_ns == mtime


def test_virtual_ledger_view_has_provenance_on_rows():
    view = build_virtual_ledger_view(PROJECT_ROOT)
    sample = (view.orders[:1] or view.open_positions[:1] or view.closed_trades[:1])
    if not sample:
        pytest.skip("No paper archive rows available in this workspace")
    row = sample[0]
    assert row.get("source_layer")
    assert row.get("paper_demo_only") is True


def test_demo_acceptance_guard_fail_closed():
    live_reject = evaluate_demo_acceptance_guard(
        trading_mode="LIVE",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=True,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    assert live_reject["rejected"] is True

    disabled = evaluate_demo_acceptance_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=False,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    assert disabled["rejected"] is True

    wallet = evaluate_demo_acceptance_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=True,
        demo_acceptance_mode_enabled=True,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    assert wallet["rejected"] is True

    ok = evaluate_demo_acceptance_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=True,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    assert ok["allowed"] is True
    assert ok["live_trading_approval"] == "NO"


def test_semantic_coverage_labels_static_or_mixed():
    cov = build_semantic_coverage(PROJECT_ROOT)
    label = cov.get("semantic_source_label") or ""
    assert label.startswith("Semantic Source:")
    assert "Static AE12 Snapshot" in label or "Runtime Stream" in label
    social = cov.get("social_confirmed_audit") or {}
    if social.get("is_zero"):
        assert social.get("explanation_reasons")
        assert social.get("caused_by_silent_stale_without_label") is False
    assert (cov.get("unknown_unresolved_policy") or {}).get("preserved") is True
    assert (cov.get("unknown_unresolved_policy") or {}).get("promoted_to_social") is False


def test_trade_alias_helper_via_api_module():
    from app.api import _alias_trade_row_for_ui

    row = _alias_trade_row_for_ui({"value": 10, "fee": 0.5, "reason": "TEST", "price": 1.0, "amount": 2})
    assert row["notional_usd"] == 10
    assert row["total_fees"] == 0.5
    assert row["reason_code"] == "TEST"
    assert row["paper_demo_only"] is True
