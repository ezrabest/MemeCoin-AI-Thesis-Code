"""AE18 symbol cache rehydration — cold/manual rehydration, one display contract,
atomic index update, GET isolation.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from app.ae13b_product import manual_refresh_runtime_index as mr
from app.ae13b_product import runtime_market_feed as rmf
from app.clean_forward import symbol_rehydration as sr
from app.clean_forward.display_identity import (
    PARTIAL_PROVIDER_SYMBOLS_MISSING,
    SYMBOL_PAIR_UNAVAILABLE,
    SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
    derive_symbol_pair_display,
    is_symbol_pair_available,
)
from app.clean_forward.runtime_identity_index import (
    RuntimeIndexValidationError,
    load_runtime_identity_index,
    validate_index_rows,
    write_runtime_index_validated,
)

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "data" / "audits"

SOLANA_URL = "https://dexscreener.com/solana/9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm"
BASE_URL = "https://dexscreener.com/base/0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A"


class _FakeResult:
    def __init__(self, payload: dict | None, **extra):
        self._payload = payload
        self._extra = extra

    def to_dict(self, *, include_raw: bool = False):
        data = {
            "verification_status": "verified" if self._payload else "provider_pair_not_found",
            "verification_http_status": 200,
            **self._extra,
        }
        if include_raw:
            data["raw_pair"] = self._payload
        return data


def _pair_payload(base="PUMP", quote="USDC"):
    return {
        "chainId": "solana",
        "dexId": "raydium",
        "url": SOLANA_URL,
        "pairAddress": "9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm",
        "baseToken": {"address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn", "name": "Pump", "symbol": base},
        "quoteToken": {"address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "name": "USD Coin", "symbol": quote},
        "priceUsd": "0.0125",
        "liquidity": {"usd": 250000.0},
        "volume": {"m5": 10.0, "h1": 100.0, "h6": 600.0, "h24": 2400.0},
        "txns": {
            "m5": {"buys": 1, "sells": 2},
            "h1": {"buys": 10, "sells": 20},
            "h6": {"buys": 60, "sells": 30},
            "h24": {"buys": 240, "sells": 120},
        },
        "priceChange": {"m5": 0.5, "h1": -1.5, "h6": 3.0, "h24": -7.25},
        "pairCreatedAt": 1700000000000,
    }


def _validator(payload):
    calls: list[tuple[str, str]] = []

    def _fn(chain, pair, *, use_cache=True):
        calls.append((chain, pair))
        return _FakeResult(payload)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


def _row_missing_symbols():
    return {
        "provider_pair_url_exact": SOLANA_URL,
        "canonical_market_identity": SOLANA_URL,
        "chain": "solana",
        "provider_base_token_address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn",
        "provider_quote_token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    }


# 1 / 2 / 3 — rehydration populates provider symbols and repairs display
def test_row_with_url_and_missing_symbols_triggers_rehydration():
    row = _row_missing_symbols()
    assert sr.row_needs_symbol_rehydration(row) is True
    validator = _validator(_pair_payload())
    out = sr.rehydrate_row_symbols(row, validator=validator)
    assert out["attempted"] is True
    assert out["success"] is True
    assert validator.calls == [("solana", "9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm")]


def test_dexscreener_symbols_populate_provider_fields():
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(_pair_payload()))
    row = out["row"]
    assert row["provider_base_token_symbol"] == "PUMP"
    assert row["provider_quote_token_symbol"] == "USDC"
    assert row["provider_base_token_name"] == "Pump"
    assert row["provider_dex_id"] == "raydium"
    assert row["price_usd"] == "0.0125"
    assert row["liquidity_usd"] == 250000.0
    assert row["volume_m5"] == 10.0 and row["volume_h24"] == 2400.0
    assert row["txns_h6_buys"] == 60 and row["txns_m5_sells"] == 2
    assert row["price_change_h24"] == -7.25
    assert row["pair_created_at"] == 1700000000000


def test_symbol_pair_display_becomes_base_quote_after_rehydration():
    before = derive_symbol_pair_display(_row_missing_symbols())
    assert before["symbol_pair_display"] == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(_pair_payload()))
    after = derive_symbol_pair_display(out["row"])
    assert after["symbol_pair_display"] == "PUMP/USDC"
    assert after["symbol_pair_display_status"] == "FULL_PAIR"


# 4 — no unavailable status when provider returned symbols
def test_unavailable_status_not_emitted_when_provider_has_symbols():
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(_pair_payload()))
    display = derive_symbol_pair_display(out["row"])["symbol_pair_display"]
    assert display not in {SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING, PARTIAL_PROVIDER_SYMBOLS_MISSING}


def test_failed_rehydration_records_exact_reason():
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(None))
    assert out["attempted"] is True
    assert out["success"] is False
    assert out["failure_code"]
    assert "verification_status" in out["failure_reason"]


# 5 / 6 — forbidden primary displays
def test_raw_address_pair_never_primary_symbol_pair():
    display = derive_symbol_pair_display(_row_missing_symbols())
    assert display["symbol_pair_display"] == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
    assert "pumpCm" in display["symbol_pair_address_fallback"]


def test_base_only_display_is_forbidden():
    row = dict(_row_missing_symbols(), provider_base_token_symbol="PUMP")
    display = derive_symbol_pair_display(row)
    assert display["symbol_pair_display"] == PARTIAL_PROVIDER_SYMBOLS_MISSING
    assert display["symbol_pair_known_side_symbol"] == "PUMP"


def test_no_symbols_and_no_addresses_is_unavailable():
    display = derive_symbol_pair_display({"provider_pair_url_exact": SOLANA_URL})
    assert display["symbol_pair_display"] == SYMBOL_PAIR_UNAVAILABLE


# 7 / 8 — one central display function, consistent across surfaces
def test_single_central_symbol_display_function():
    for fn in (rmf._index_row_to_clean_forward, rmf._index_row_to_live_market):
        src = inspect.getsource(fn)
        assert "_display_fields(" in src or "resolve_display_fields(" in src
        assert "provider_base_token_symbol" not in src, f"{fn.__name__} re-implements symbol logic"
    assert rmf._display_fields is rmf.resolve_display_fields


def test_same_canonical_identity_same_display_across_surfaces():
    rows = load_runtime_identity_index()["rows"]
    assert rows, "runtime index must exist"
    clean = {r["canonical_market_identity"]: r["symbol_pair_display"] for r in (rmf._index_row_to_clean_forward(x) for x in rows)}
    snap = {r["canonical_market_identity"]: r["symbol_pair_display"] for r in (rmf._index_row_to_live_market(x) for x in rows)}
    opp_rows = rmf.enrich_opportunity_rows(
        [{"canonical_market_identity": r.get("canonical_market_identity"), "chain": r.get("chain")} for r in rows]
    )["rows"]
    opp = {r["canonical_market_identity"]: r["symbol_pair_display"] for r in opp_rows}
    for canonical, display in clean.items():
        assert snap[canonical] == display
        assert opp[canonical] == display


def test_runtime_index_has_no_forbidden_primary_displays():
    rows = load_runtime_identity_index()["rows"]
    for r in rows:
        display = str(r.get("symbol_pair_display") or "")
        assert display not in ("", "-")
        assert display != r.get("canonical_market_identity")
        assert display != r.get("pair_address_derived")
        if is_symbol_pair_available(display):
            assert "/" in display
            for side in display.split("/"):
                assert not side.startswith("0x")
                assert len(side) < 24


def test_materially_repaired_symbol_coverage():
    rows = load_runtime_identity_index()["rows"]
    proper = sum(1 for r in rows if is_symbol_pair_available(r.get("symbol_pair_display")))
    assert proper > 14, f"expected material repair, got {proper}/{len(rows)}"


# 9 / 10 — GET isolation
def test_get_path_does_not_call_dexscreener_or_rehydrate():
    from app.runtime.ui_get_network_guard import snapshot_counters, ui_get_network_guard

    validator = _validator(_pair_payload())
    with ui_get_network_guard("/api/ae13b/clean-forward-market-feed"):
        rows = load_runtime_identity_index()["rows"]
        [rmf._index_row_to_clean_forward(r) for r in rows]
        out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=validator)
    assert out["attempted"] is False
    assert out["failure_code"] == "PROVIDER_REFRESH_DISABLED"
    assert validator.calls == []
    snap = snapshot_counters()
    assert snap.get("dexscreener_calls_on_get", 0) == 0


def test_get_isolation_audit_passes():
    audit = json.loads((AUDITS / "ae18_ui_get_network_isolation_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    for key in (
        "external_network_calls_on_get",
        "dexscreener_calls_on_get",
        "helius_calls_on_get",
        "rss_calls_on_get",
        "provider_refresh_on_get",
        "symbol_rehydration_on_get",
        "index_rebuild_on_get",
    ):
        assert audit[key] == 0, key


# 11 / 12 — explicit paths may call the provider
def test_rebuild_flag_enables_rehydration():
    import scripts.rebuild_canonical_market_identity_index as rebuild

    src = inspect.getsource(rebuild.build_index_from_sources)
    assert "enabled=allow_dexscreener" in src
    parser_src = inspect.getsource(rebuild.main)
    assert "--allow-dexscreener-rehydration" in inspect.getsource(rebuild)
    assert "write_runtime_index_validated" in parser_src


def test_rehydrate_rows_disabled_makes_no_calls():
    validator = _validator(_pair_payload())
    result = sr.rehydrate_rows([_row_missing_symbols()], enabled=False, validator=validator)
    assert result["dex_rehydration_attempted_count"] == 0
    assert result["rows_rehydration_needed"] == 1
    assert validator.calls == []


def test_manual_refresh_performs_conditional_rehydration():
    src = inspect.getsource(mr.manual_refresh_runtime_index)
    assert "row_needs_symbol_rehydration" in src
    assert "rehydrate_row_symbols" in src
    for field in (
        "rows_checked",
        "rows_rehydration_needed",
        "dex_rehydration_attempted_count",
        "dex_rehydration_success_count",
        "dex_rehydration_failed_count",
        "runtime_index_update_status",
        "failed_rehydration_urls",
        "failed_rehydration_reasons",
    ):
        assert field in src, field


# 13 / 14 — atomic index update
def test_runtime_index_write_is_atomic(tmp_path):
    rows = [
        {
            "canonical_market_identity": SOLANA_URL,
            "provider_pair_url_exact": SOLANA_URL,
            "symbol_pair_display": "PUMP/USDC",
        }
    ]
    jsonl = tmp_path / "idx.jsonl"
    csv_path = tmp_path / "idx.csv"
    report = write_runtime_index_validated(rows, jsonl_path=jsonl, csv_path=csv_path)
    assert report["temp_jsonl_written"] and report["temp_csv_written"]
    assert report["temp_validation_passed"] is True
    assert report["final_jsonl_replaced"] and report["final_csv_replaced"]
    assert report["final_index_row_count"] == 1
    assert not list(tmp_path.glob(".idx_*"))


@pytest.mark.parametrize(
    "bad_row",
    [
        {"canonical_market_identity": "", "provider_pair_url_exact": SOLANA_URL, "symbol_pair_display": "A/B"},
        {"canonical_market_identity": SOLANA_URL, "provider_pair_url_exact": "", "symbol_pair_display": "A/B"},
        {"canonical_market_identity": SOLANA_URL, "provider_pair_url_exact": SOLANA_URL, "symbol_pair_display": "-"},
        {"canonical_market_identity": SOLANA_URL, "provider_pair_url_exact": SOLANA_URL, "symbol_pair_display": ""},
    ],
)
def test_failed_validation_does_not_replace_existing_index(tmp_path, bad_row):
    jsonl = tmp_path / "idx.jsonl"
    csv_path = tmp_path / "idx.csv"
    good = [{"canonical_market_identity": SOLANA_URL, "provider_pair_url_exact": SOLANA_URL, "symbol_pair_display": "PUMP/USDC"}]
    write_runtime_index_validated(good, jsonl_path=jsonl, csv_path=csv_path)
    original = jsonl.read_text(encoding="utf-8")

    with pytest.raises(RuntimeIndexValidationError):
        write_runtime_index_validated([bad_row], jsonl_path=jsonl, csv_path=csv_path)

    assert jsonl.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".idx_*"))


def test_duplicate_canonical_identity_rejected(tmp_path):
    row = {"canonical_market_identity": SOLANA_URL, "provider_pair_url_exact": SOLANA_URL, "symbol_pair_display": "A/B"}
    with pytest.raises(RuntimeIndexValidationError):
        write_runtime_index_validated([row, dict(row)], jsonl_path=tmp_path / "a.jsonl", csv_path=tmp_path / "a.csv")


def test_validate_index_rows_flags_raw_address_display():
    report = validate_index_rows(
        [
            {
                "canonical_market_identity": SOLANA_URL,
                "provider_pair_url_exact": SOLANA_URL,
                "symbol_pair_display": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            }
        ]
    )
    assert report["invalid_symbol_pair_display_count"] == 1
    assert report["passed"] is False


def test_atomic_update_audit_passes():
    audit = json.loads((AUDITS / "ae18_runtime_index_atomic_update_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["final_index_row_count"] == len(load_runtime_identity_index()["rows"])
    assert audit["duplicate_canonical_identity_count"] == 0
    assert audit["invalid_symbol_pair_display_count"] == 0


# 15 / 16 — identity rules
def test_pair_address_remains_derived_helper_only():
    rows = load_runtime_identity_index()["rows"]
    for r in rows:
        assert r.get("canonical_market_identity_type") == "PROVIDER_URL"
        assert str(r["canonical_market_identity"]).startswith("http")
        assert r["canonical_market_identity"] != r.get("pair_address_derived")
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(_pair_payload()))
    assert out["row"]["canonical_market_identity"] == SOLANA_URL
    assert out["row"]["pair_address_derived"] == "9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm"


def test_exact_url_final_segment_casing_preserved():
    row = {
        "provider_pair_url_exact": BASE_URL,
        "canonical_market_identity": BASE_URL,
        "chain": "base",
    }
    out = sr.rehydrate_row_symbols(row, validator=_validator(_pair_payload()))
    assert out["row"]["provider_pair_url"].endswith("0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A")
    rows = load_runtime_identity_index()["rows"]
    for r in rows:
        segment = str(r.get("provider_pair_url_final_segment_exact") or "")
        if segment:
            assert str(r["provider_pair_url_exact"]).endswith(segment)


def test_provider_url_not_replaced_by_untrusted_provider_url():
    payload = _pair_payload()
    payload["url"] = "https://dexscreener.com/solana/SOMEOTHERPAIR"
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=_validator(payload))
    assert out["row"]["provider_pair_url"] == SOLANA_URL
    assert out["row"]["provider_pair_url_trusted_equivalent"] is False


# 17 — shutdown
def test_shutdown_skips_rehydration(monkeypatch):
    import app.runtime.shutdown as shutdown

    validator = _validator(_pair_payload())
    monkeypatch.setattr(shutdown, "is_shutting_down", lambda: True)
    out = sr.rehydrate_row_symbols(_row_missing_symbols(), validator=validator)
    assert out["attempted"] is False
    assert out["failure_code"] == "CONTROLLED_SHUTDOWN_SKIP"
    assert validator.calls == []


def test_shutdown_lifecycle_audit_still_passes():
    audit = json.loads((AUDITS / "ae18_shutdown_lifecycle_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True


# Audit presence
@pytest.mark.parametrize(
    "name",
    [
        "ae18_symbol_cache_regression_audit.json",
        "ae18_symbol_rehydration_result_audit.json",
        "ae18_cross_surface_symbol_display_audit.json",
        "ae18_runtime_index_atomic_update_audit.json",
        "ae18_ui_get_network_isolation_audit.json",
        "ae18_manual_refresh_url_first_audit.json",
    ],
)
def test_new_audits_pass(name):
    audit = json.loads((AUDITS / name).read_text(encoding="utf-8"))
    assert audit["passed"] is True, name
    assert audit["fail_closed"] is True


def test_rehydration_audit_reports_every_unresolved_row():
    audit = json.loads((AUDITS / "ae18_symbol_rehydration_result_audit.json").read_text(encoding="utf-8"))
    assert audit["raw_address_symbol_pair_after_count"] == 0
    assert audit["base_only_symbol_pair_after_count"] == 0
    assert audit["proper_symbol_pair_after_count"] > 14
    for row in audit["row_diagnostics"]:
        if row["symbol_pair_display_status"] != "FULL_PAIR":
            assert row["rehydration_failure_reason"] or row["symbol_pair_missing_reason"]


def test_regression_audit_documents_root_cause():
    audit = json.loads((AUDITS / "ae18_symbol_cache_regression_audit.json").read_text(encoding="utf-8"))
    assert audit["rows_checked"] > 0
    assert audit["rows_eligible_for_dexscreener_symbol_rehydration"] >= 0
    assert len(audit["root_cause_summary"]) > 40
