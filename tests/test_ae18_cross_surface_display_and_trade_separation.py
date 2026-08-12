"""AE18 final stabilization tests.

Covers URL-key join integrity, Market Opportunities cross-surface display
parity, strict GET isolation, and symbol/trade-readiness separation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae13b_product import runtime_market_feed as rmf
from app.clean_forward.provider_resilience_statuses import (
    DISPLAY_READY,
    ENTRY_BLOCKED_IDENTITY_UNRESOLVED,
    ENTRY_BLOCKED_MARKET_DATA_MISSING,
    ENTRY_BLOCKED_MARKET_DATA_STALE,
    ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE,
    ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE,
    FORBIDDEN_BLOCK_REASON,
    IDENTITY_READY,
    IDENTITY_UNRESOLVED,
    IDENTITY_UNVERIFIABLE,
    MARKET_DATA_MISSING,
    MARKET_DATA_READY,
    MARKET_DATA_STALE,
    MARKET_DATA_UNVERIFIABLE,
    PAPER_ELIGIBLE,
    SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
    assert_block_reason_not_symbol_only,
    block_reason_for,
    classify_display_metadata_status,
    classify_identity_readiness,
    classify_trade_readiness,
    is_proper_symbol_pair_display,
)
from app.clean_forward.provider_url_key import (
    normalize_provider_pair_url_key,
    try_normalize_provider_pair_url_key,
)
from app.clean_forward.runtime_identity_index import load_runtime_identity_index
from app.runtime.ui_get_network_guard import (
    WRITE_COUNTER_KEYS,
    reset_counters_for_tests,
    snapshot_counters,
    ui_get_network_guard,
)

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "data" / "audits"

EVM_URL = "https://dexscreener.com/base/0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4"
SOL_URL = "https://dexscreener.com/solana/26hq83ubjaxGmcMTbGQqrWMUrtwg92EeJgFUYd7zKvUH"


# ---------------------------------------------------------------------------
# PART A — URL key normalization integrity (tests 1-6)
# ---------------------------------------------------------------------------
def test_url_key_trims_whitespace():
    assert normalize_provider_pair_url_key(f"  {EVM_URL}  ") == EVM_URL


def test_url_key_removes_trailing_slash():
    assert normalize_provider_pair_url_key(EVM_URL + "/") == EVM_URL


def test_url_key_preserves_final_segment_case():
    key = normalize_provider_pair_url_key(EVM_URL)
    assert key.endswith("0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4")


def test_url_key_lowercasing_is_forbidden():
    key = normalize_provider_pair_url_key(EVM_URL)
    assert key != EVM_URL.lower()
    assert any(ch.isupper() for ch in key)


@pytest.mark.parametrize("bad", ["", "   ", "not a url", "ftp://dexscreener.com/base/x"])
def test_url_key_rejects_malformed(bad):
    key, reason = try_normalize_provider_pair_url_key(bad, require_dexscreener=True)
    assert key is None
    assert reason


def test_url_key_whitespace_and_slash_map_to_same_key():
    assert normalize_provider_pair_url_key(f" {EVM_URL}/ ") == normalize_provider_pair_url_key(
        EVM_URL
    )


def test_url_key_preserves_solana_and_evm_case_sensitive_segments():
    """Test 7 — mixed-case Solana base58 and EVM checksum must not be mutated."""
    for url in (SOL_URL, EVM_URL):
        segment = url.rsplit("/", 1)[-1]
        assert normalize_provider_pair_url_key(url).rsplit("/", 1)[-1] == segment


def test_normalized_key_used_for_index_join():
    rows = load_runtime_identity_index()["rows"]
    join_map = rmf.build_index_join_map(rows)
    assert len(join_map) == len([r for r in rows if r.get("provider_pair_url_exact")])
    for key in join_map:
        assert key == normalize_provider_pair_url_key(key)


# ---------------------------------------------------------------------------
# PART C — Cross-surface display (tests 7-13)
# ---------------------------------------------------------------------------
def _surface_displays():
    rows = load_runtime_identity_index()["rows"]
    clean = {
        r["provider_pair_url_exact"]: r["symbol_pair_display"]
        for r in (rmf._index_row_to_clean_forward(x) for x in rows)
    }
    snap = {
        r["provider_pair_url_exact"]: r["symbol_pair_display"]
        for r in (rmf._index_row_to_live_market(x) for x in rows)
    }
    opp_rows = rmf.build_opportunities_from_index(limit=1000)["rows"]
    opp_rows = rmf.enrich_opportunity_rows(opp_rows)["rows"]
    opp = {r["provider_pair_url_exact"]: r["symbol_pair_display"] for r in opp_rows}
    return clean, snap, opp


def test_market_opportunities_uses_same_display_as_runtime_index():
    clean, snap, opp = _surface_displays()
    assert opp, "market opportunities must not be empty"
    for url, display in clean.items():
        assert snap[url] == display
        assert opp[url] == display


def test_market_opportunities_does_not_overwrite_valid_symbol_display():
    rows = rmf.build_opportunities_from_index(limit=1000)["rows"]
    out = rmf.enrich_opportunity_rows(rows)
    assert out["opportunities_rows_where_valid_symbol_was_overwritten"] == 0


def test_market_opportunities_joins_by_normalized_url_key():
    rows = rmf.build_opportunities_from_index(limit=1000)["rows"]
    out = rmf.enrich_opportunity_rows(rows)
    assert out["joined_by"] == "normalized_provider_pair_url_key"
    assert out["opportunities_missing_runtime_index_join_count"] == 0
    assert out["opportunities_joined_runtime_index_count"] == len(rows)


def test_market_opportunities_does_not_join_by_pair_address():
    """A row carrying only a pair_address must not resolve to an index row."""
    index_rows = load_runtime_identity_index()["rows"]
    seeded = next(r for r in index_rows if r.get("pair_address_derived"))
    row = {"symbol": "X", "pair_address": seeded["pair_address_derived"], "chain": "base"}
    out = rmf.enrich_opportunity_rows([row])
    assert out["opportunities_joined_runtime_index_count"] == 0
    assert out["rows"][0].get("canonical_market_identity") in ("", None)
    assert not is_proper_symbol_pair_display(out["rows"][0]["symbol_pair_display"])


def test_market_opportunities_unresolved_count_matches_other_surfaces():
    clean, snap, opp = _surface_displays()
    def unresolved(mapping):
        return sorted(u for u, d in mapping.items() if not is_proper_symbol_pair_display(d))

    assert unresolved(opp) == unresolved(clean) == unresolved(snap)


def test_market_opportunities_not_all_rows_unresolved_when_index_has_symbols():
    """Regression guard for the 45/45 unresolved defect."""
    rows = rmf.build_opportunities_from_index(limit=1000)["rows"]
    rows = rmf.enrich_opportunity_rows(rows)["rows"]
    proper = sum(1 for r in rows if is_proper_symbol_pair_display(r["symbol_pair_display"]))
    index_proper = sum(
        1
        for r in load_runtime_identity_index()["rows"]
        if is_proper_symbol_pair_display(r.get("symbol_pair_display"))
    )
    assert proper == index_proper
    assert proper > 0
    assert proper == len(rows) - sum(
        1 for r in rows if not is_proper_symbol_pair_display(r["symbol_pair_display"])
    )


def test_market_opportunities_dto_preserves_required_fields():
    rows = rmf.build_opportunities_from_index(limit=1000)["rows"]
    rows = rmf.enrich_opportunity_rows(rows)["rows"]
    required = (
        "provider_pair_url_exact",
        "normalized_provider_pair_url_key",
        "canonical_market_identity",
        "symbol_pair_display",
        "display_metadata_status",
        "provider_resolution_status",
        "symbol_resolution_status",
        "market_data_status",
        "identity_readiness_status",
        "trade_readiness_status",
    )
    for row in rows:
        for field in required:
            assert row.get(field) not in (None, ""), (field, row.get("provider_pair_url_exact"))
        if not is_proper_symbol_pair_display(row["symbol_pair_display"]):
            assert row.get("unresolved_reason")


def test_frontend_renders_server_symbol_display_without_recomputing():
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "symbol_pair_display" in js
    # No client-side reconstruction of a pair from raw token addresses.
    assert "base_token_address +" not in js
    assert "baseAddress + '/'" not in js


# ---------------------------------------------------------------------------
# PART D — Strict GET isolation (tests 14-19)
# ---------------------------------------------------------------------------
def _assert_get_clean(snap):
    assert snap["external_network_calls_on_get"] == 0
    assert snap["dexscreener_calls_on_get"] == 0
    assert snap["helius_calls_on_get"] == 0
    assert snap["rss_calls_on_get"] == 0
    assert snap["recursive_audit_scan_on_get"] == 0
    for key in WRITE_COUNTER_KEYS:
        assert snap[key] == 0, key


def test_clean_forward_get_has_no_side_effects():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/clean-forward"):
        rmf.build_clean_forward_from_index(limit=50)
    _assert_get_clean(snapshot_counters())


def test_market_snapshot_get_has_no_side_effects():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/live-market"):
        rmf.build_live_market_from_index(limit=50)
    _assert_get_clean(snapshot_counters())


def test_market_opportunities_get_has_no_side_effects():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/opportunities"):
        from app.api import _build_opportunities

        _build_opportunities(50)
    _assert_get_clean(snapshot_counters())


def test_portfolio_get_has_no_side_effects():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/portfolio"):
        from app.api import get_paper_trader

        get_paper_trader().get_positions(status="OPEN")
    _assert_get_clean(snapshot_counters())


def test_get_does_not_write_last_good_cache():
    from app.clean_forward.last_good_display_cache import upsert_last_good_display
    from app.runtime.ui_get_network_guard import UiGetWriteForbidden

    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/clean-forward"):
        with pytest.raises(UiGetWriteForbidden):
            upsert_last_good_display(
                {"provider_pair_url_exact": EVM_URL, "symbol_pair_display": "A/B"}
            )
    assert snapshot_counters()["cache_write_on_get"] >= 1


def test_get_does_not_write_audits():
    from app.clean_forward.manual_display_overrides import _write_audit
    from app.runtime.ui_get_network_guard import UiGetWriteForbidden

    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/opportunities"):
        with pytest.raises(UiGetWriteForbidden):
            _write_audit(AUDITS / "should_never_exist.json", {"passed": False})
    assert snapshot_counters()["audit_write_on_get"] >= 1
    assert not (AUDITS / "should_never_exist.json").exists()


# ---------------------------------------------------------------------------
# PART B — Symbol / trade readiness separation (tests 20-25)
# ---------------------------------------------------------------------------
def _row(**kw):
    base = {
        "provider_pair_url_exact": EVM_URL,
        "provider_pair_url_final_segment_exact": EVM_URL.rsplit("/", 1)[-1],
        "chain": "base",
        "provider_base_token_address": "0xaaa",
        "provider_quote_token_address": "0xbbb",
        "price_usd": 1.25,
        "freshness_status": "fresh",
    }
    base.update(kw)
    return base


def test_missing_symbol_alone_does_not_block_paper_eligibility():
    row = _row(symbol_pair_display="SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING")
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_READY,
        identity_readiness_status=classify_identity_readiness(row),
    )
    assert status == PAPER_ELIGIBLE
    assert not str(status).startswith("ENTRY_BLOCKED")


def test_missing_symbol_with_valid_market_data_stays_eligible_or_watch_only():
    row = _row()
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_READY,
        identity_readiness_status=classify_identity_readiness(row),
    )
    assert status in {PAPER_ELIGIBLE, "WATCH_ONLY"}
    assert classify_display_metadata_status(
        has_proper_display=False
    ) == SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE


def test_missing_symbol_plus_stale_price_blocks_for_stale_not_symbol():
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_STALE,
        identity_readiness_status=IDENTITY_READY,
    )
    assert status == ENTRY_BLOCKED_MARKET_DATA_STALE
    assert block_reason_for(status) == "MARKET_DATA_STALE"


def test_missing_symbol_plus_missing_price_blocks_for_missing_price():
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_MISSING,
        identity_readiness_status=IDENTITY_READY,
    )
    assert status == ENTRY_BLOCKED_MARKET_DATA_MISSING
    assert block_reason_for(status) == "MARKET_DATA_MISSING"


def test_missing_symbol_plus_unverifiable_identity_blocks_for_identity():
    for identity in (IDENTITY_UNRESOLVED, IDENTITY_UNVERIFIABLE):
        status = classify_trade_readiness(
            market_data_status=MARKET_DATA_READY,
            identity_readiness_status=identity,
        )
        assert status == ENTRY_BLOCKED_IDENTITY_UNRESOLVED


def test_unverifiable_market_data_blocks_with_explicit_reason():
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_UNVERIFIABLE,
        identity_readiness_status=IDENTITY_READY,
    )
    assert status == ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE


def test_unsafe_position_continuity_blocks_entry():
    status = classify_trade_readiness(
        market_data_status=MARKET_DATA_READY,
        identity_readiness_status=IDENTITY_READY,
        position_continuity_safe=False,
    )
    assert status == ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE


def test_block_reason_is_never_symbol_missing_only():
    for status in (
        ENTRY_BLOCKED_MARKET_DATA_MISSING,
        ENTRY_BLOCKED_MARKET_DATA_STALE,
        ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE,
        ENTRY_BLOCKED_IDENTITY_UNRESOLVED,
        ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE,
    ):
        reason = block_reason_for(status)
        assert reason
        assert reason != FORBIDDEN_BLOCK_REASON
        assert assert_block_reason_not_symbol_only(reason)


def test_no_runtime_row_is_blocked_solely_because_symbol_missing():
    rows = load_runtime_identity_index()["rows"]
    for row in rows:
        display_ok = is_proper_symbol_pair_display(row.get("symbol_pair_display"))
        status = str(row.get("trade_readiness_status") or "")
        if display_ok or not status.startswith("ENTRY_BLOCKED"):
            continue
        # Blocked rows must have a non-display cause.
        assert row.get("market_data_status") != MARKET_DATA_READY or row.get(
            "identity_readiness_status"
        ) in {IDENTITY_UNRESOLVED, IDENTITY_UNVERIFIABLE}


def test_display_metadata_status_present_for_every_unresolved_row():
    rows = load_runtime_identity_index()["rows"]
    for row in rows:
        if is_proper_symbol_pair_display(row.get("symbol_pair_display")):
            continue
        assert row.get("display_metadata_status") in {
            SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
            DISPLAY_READY,
        }
        assert row.get("unresolved_reason")


# ---------------------------------------------------------------------------
# Audits produced by this task must pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "ae18_market_opportunities_symbol_regression_audit.json",
        "ae18_unresolved_symbol_trade_readiness_audit.json",
        "ae18_get_isolation_strict_audit.json",
    ],
)
def test_new_ae18_audits_pass(name):
    path = AUDITS / name
    assert path.exists(), f"missing audit {name}"
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["passed"] is True, audit
    assert audit["fail_closed"] is False


def test_market_opportunities_regression_audit_counts_are_consistent():
    audit = json.loads(
        (AUDITS / "ae18_market_opportunities_symbol_regression_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["market_opportunities_proper_symbol_pair_count"] == audit[
        "clean_forward_proper_symbol_pair_count"
    ]
    assert audit["market_opportunities_unresolved_symbol_count"] == audit[
        "clean_forward_unresolved_symbol_count"
    ]
    assert audit["opportunities_rows_where_valid_symbol_was_overwritten"] == 0
    assert audit["raw_address_primary_display_count"] == 0
    assert audit["base_only_primary_display_count"] == 0
    assert audit["empty_primary_display_count"] == 0
    assert audit["market_opportunities_proper_symbol_pair_count"] > 0


def test_unresolved_symbol_trade_readiness_audit_has_no_symbol_only_blocks():
    audit = json.loads(
        (AUDITS / "ae18_unresolved_symbol_trade_readiness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["rows_blocked_due_to_symbol_only_count"] == 0
    for url, reason in (audit.get("block_reasons_by_url") or {}).items():
        assert reason != FORBIDDEN_BLOCK_REASON, url


def test_strict_get_isolation_audit_counters_are_zero():
    audit = json.loads(
        (AUDITS / "ae18_get_isolation_strict_audit.json").read_text(encoding="utf-8")
    )
    for key in (
        "get_network_calls_count",
        "get_cache_write_count",
        "audit_write_on_get_count",
        "dexscreener_calls_on_get",
        "helius_calls_on_get",
        "rss_calls_on_get",
        "index_rebuild_on_get",
        "symbol_rehydration_on_get",
        "provider_refresh_on_get",
        "recursive_audit_scan_on_get",
    ):
        assert audit[key] == 0, key
