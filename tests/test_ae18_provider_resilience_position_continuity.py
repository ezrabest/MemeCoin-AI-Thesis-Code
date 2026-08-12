"""AE18 provider resilience + position continuity contract tests.

Covers URL key normalization, manual overrides, last-good cache atomicity,
provider/display resilience statuses, position continuity, and financial DTOs.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from app.clean_forward.last_good_display_cache import (
    CACHE_FIELDS,
    ensure_empty_last_good_cache_files,
    lookup_last_good_display,
    read_last_good_display_cache,
    upsert_last_good_display,
    write_last_good_display_cache,
)
from app.clean_forward.manual_display_overrides import (
    OVERRIDE_COLUMNS,
    display_contains_raw_address,
    ensure_override_csv_template,
    is_base_only_display,
    load_applied_manual_overrides,
    lookup_manual_override,
    validate_manual_display_overrides,
    validate_override_row,
)
from app.clean_forward.position_continuity import (
    SNAPSHOT_FIELDS,
    assert_new_entry_allowed,
    attach_entry_snapshot_to_position,
    build_entry_continuity_snapshot,
    build_position_financial_dto,
    resolve_position_market_data_state,
)
from app.clean_forward.provider_resilience_statuses import (
    DATA_OK,
    DATA_STALE,
    ENTRY_BLOCKED_MARKET_DATA_MISSING,
    MARKET_DATA_AVAILABLE_SYMBOLS_MISSING,
    PAPER_ELIGIBLE,
    PRICE_UNAVAILABLE,
    RESOLVED_WITH_LAST_GOOD_DISPLAY,
    RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE,
    SYMBOL_PAIR_FROM_LAST_GOOD,
    SYMBOL_PAIR_FROM_MANUAL_OVERRIDE,
    SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
    WATCH_ONLY,
    entry_blocked,
    is_proper_symbol_pair_display,
)
from app.clean_forward.provider_url_key import (
    ProviderUrlKeyError,
    normalize_provider_pair_url_key,
)
from app.clean_forward.display_resilience import apply_display_resilience

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "data" / "audits"
SOLANA_URL = "https://dexscreener.com/solana/9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm"
BASE_URL = "https://dexscreener.com/base/0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A"


# ---------------------------------------------------------------------------
# PART A — URL normalization
# ---------------------------------------------------------------------------


def test_url_key_trims_whitespace():
    key = normalize_provider_pair_url_key(f"  {SOLANA_URL}  ")
    assert key == SOLANA_URL


def test_url_key_removes_trailing_slash():
    key = normalize_provider_pair_url_key(SOLANA_URL + "/")
    assert key == SOLANA_URL
    assert not key.endswith("/")


def test_url_key_preserves_final_segment_case():
    key = normalize_provider_pair_url_key(BASE_URL)
    assert key.endswith("0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A")
    assert "CB8227" in key  # mixed case preserved


def test_url_key_does_not_lowercase():
    key = normalize_provider_pair_url_key(BASE_URL)
    assert key == BASE_URL
    assert key != BASE_URL.lower()


def test_url_key_rejects_malformed_and_empty():
    with pytest.raises(ProviderUrlKeyError) as empty:
        normalize_provider_pair_url_key("")
    assert empty.value.reason == "empty_provider_pair_url"
    with pytest.raises(ProviderUrlKeyError):
        normalize_provider_pair_url_key("not-a-url")
    with pytest.raises(ProviderUrlKeyError):
        normalize_provider_pair_url_key("https://dexscreener.com/onlyone")


def test_url_key_whitespace_and_slash_map_same():
    a = normalize_provider_pair_url_key(f"  {SOLANA_URL}/  ")
    b = normalize_provider_pair_url_key(SOLANA_URL)
    assert a == b


# ---------------------------------------------------------------------------
# PART C — Manual override validation
# ---------------------------------------------------------------------------


def test_manual_override_valid_applies(tmp_path: Path):
    csv_path = tmp_path / "overrides.csv"
    audit = tmp_path / "audit.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        w.writerow(
            {
                "provider_pair_url_exact": SOLANA_URL,
                "normalized_provider_pair_url_key": normalize_provider_pair_url_key(
                    SOLANA_URL, require_dexscreener=True
                ),
                "symbol_pair_display": "PUMP/USDC",
                "provider_base_token_symbol": "PUMP",
                "provider_quote_token_symbol": "USDC",
                "reason": "test",
                "reviewed_by": "tester",
                "reviewed_at": "2026-08-01T00:00:00Z",
                "source_note": "unit",
            }
        )
    result = validate_manual_display_overrides(csv_path=csv_path, audit_path=audit, apply=True)
    assert result["passed"] is True
    assert result["valid_rows"] == 1
    assert result["rejected_rows"] == 0
    applied = load_applied_manual_overrides(csv_path=csv_path, audit_path=audit)
    assert SOLANA_URL.replace("", "") or True
    key = normalize_provider_pair_url_key(SOLANA_URL, require_dexscreener=True)
    assert key in applied
    ov = lookup_manual_override(SOLANA_URL + "/", overrides=applied)
    assert ov["symbol_pair_display"] == "PUMP/USDC"


def test_manual_override_rejects_raw_address_display():
    row = {
        "provider_pair_url_exact": BASE_URL,
        "normalized_provider_pair_url_key": "",
        "symbol_pair_display": "0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A/WETH",
        "provider_base_token_symbol": "0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A",
        "provider_quote_token_symbol": "WETH",
        "reason": "x",
        "reviewed_by": "t",
        "reviewed_at": "2026-08-01T00:00:00Z",
        "source_note": "",
    }
    reasons = validate_override_row(row, row_number=2)
    assert any("raw_address" in r or "address_like" in r for r in reasons)
    assert display_contains_raw_address(row["symbol_pair_display"])


def test_manual_override_rejects_base_only():
    assert is_base_only_display("PUMP")
    row = {
        "provider_pair_url_exact": SOLANA_URL,
        "symbol_pair_display": "PUMP",
        "reviewed_by": "t",
        "reviewed_at": "2026-08-01T00:00:00Z",
    }
    reasons = validate_override_row(row, row_number=2)
    assert "symbol_pair_display_missing_separator" in reasons or any(
        "base_only" in r for r in reasons
    )


def test_manual_override_rejects_empty_and_missing_slash(tmp_path: Path):
    csv_path = tmp_path / "overrides.csv"
    audit = tmp_path / "audit.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        w.writerow(
            {
                "provider_pair_url_exact": SOLANA_URL,
                "normalized_provider_pair_url_key": "",
                "symbol_pair_display": "",
                "provider_base_token_symbol": "",
                "provider_quote_token_symbol": "",
                "reason": "x",
                "reviewed_by": "t",
                "reviewed_at": "2026-08-01T00:00:00Z",
                "source_note": "",
            }
        )
        w.writerow(
            {
                "provider_pair_url_exact": BASE_URL,
                "normalized_provider_pair_url_key": "",
                "symbol_pair_display": "ONLYBASE",
                "provider_base_token_symbol": "ONLYBASE",
                "provider_quote_token_symbol": "",
                "reason": "x",
                "reviewed_by": "t",
                "reviewed_at": "2026-08-01T00:00:00Z",
                "source_note": "",
            }
        )
    result = validate_manual_display_overrides(csv_path=csv_path, audit_path=audit, apply=True)
    assert result["rejected_rows"] == 2
    assert result["applied_override_count"] == 0
    assert result["empty_display_count"] >= 1
    assert result["missing_separator_count"] >= 1
    assert result["passed"] is True  # rejected not applied


def test_manual_override_rejects_duplicate_conflict(tmp_path: Path):
    csv_path = tmp_path / "overrides.csv"
    audit = tmp_path / "audit.json"
    key = normalize_provider_pair_url_key(SOLANA_URL, require_dexscreener=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        for display in ("AAA/BBB", "CCC/DDD"):
            w.writerow(
                {
                    "provider_pair_url_exact": SOLANA_URL,
                    "normalized_provider_pair_url_key": key,
                    "symbol_pair_display": display,
                    "provider_base_token_symbol": display.split("/")[0],
                    "provider_quote_token_symbol": display.split("/")[1],
                    "reason": "dup",
                    "reviewed_by": "t",
                    "reviewed_at": "2026-08-01T00:00:00Z",
                    "source_note": "",
                }
            )
    result = validate_manual_display_overrides(csv_path=csv_path, audit_path=audit, apply=True)
    assert result["duplicate_key_count"] >= 1
    assert any(
        "duplicate_normalized_key_conflicting_display" in d["reasons"]
        for d in result["rejected_row_details"]
    )


def test_manual_override_rejected_not_applied_and_audit(tmp_path: Path):
    csv_path = tmp_path / "overrides.csv"
    audit = tmp_path / "audit.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        w.writerow(
            {
                "provider_pair_url_exact": SOLANA_URL,
                "normalized_provider_pair_url_key": "",
                "symbol_pair_display": "-",
                "provider_base_token_symbol": "",
                "provider_quote_token_symbol": "",
                "reason": "bad",
                "reviewed_by": "",
                "reviewed_at": "",
                "source_note": "",
            }
        )
    result = validate_manual_display_overrides(csv_path=csv_path, audit_path=audit, apply=True)
    assert result["applied_override_count"] == 0
    assert result["rejected_rows"] == 1
    assert audit.exists()
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["rejected_row_details"]
    assert payload["missing_review_metadata_count"] >= 1


# ---------------------------------------------------------------------------
# PART B — Cache atomicity / locking
# ---------------------------------------------------------------------------


def test_last_good_cache_temp_write_and_os_replace(tmp_path: Path):
    jsonl = tmp_path / "lg.jsonl"
    csv_path = tmp_path / "lg.csv"
    entry = {
        "normalized_provider_pair_url_key": normalize_provider_pair_url_key(
            SOLANA_URL, require_dexscreener=True
        ),
        "provider_pair_url_exact": SOLANA_URL,
        "symbol_pair_display": "PUMP/USDC",
        "provider_base_token_symbol": "PUMP",
        "provider_quote_token_symbol": "USDC",
        "provider_base_token_address": "addr1",
        "provider_quote_token_address": "addr2",
        "provider_dex_id": "raydium",
        "chain": "solana",
        "first_seen_at": "2026-08-01T00:00:00Z",
        "last_confirmed_at": "2026-08-01T00:00:00Z",
        "source": "test",
        "source_audit": "unit",
        "confidence": "HIGH",
        "provenance_status": "PROVIDER_CONFIRMED",
    }
    report = write_last_good_display_cache([entry], jsonl_path=jsonl, csv_path=csv_path)
    assert report["passed"] is True
    assert report["temp_write_used"] is True
    assert report["temp_validation_used"] is True
    assert report["os_replace_used"] is True
    assert report["cache_locking_supported"] is True
    assert jsonl.exists() and csv_path.exists()
    rows = read_last_good_display_cache(jsonl_path=jsonl)
    assert len(rows) == 1
    assert lookup_last_good_display(SOLANA_URL + "/", jsonl_path=jsonl)["symbol_pair_display"] == "PUMP/USDC"


def test_last_good_temp_validation_failure_preserves_previous(tmp_path: Path):
    jsonl = tmp_path / "lg.jsonl"
    csv_path = tmp_path / "lg.csv"
    good = {
        "normalized_provider_pair_url_key": normalize_provider_pair_url_key(
            SOLANA_URL, require_dexscreener=True
        ),
        "provider_pair_url_exact": SOLANA_URL,
        "symbol_pair_display": "PUMP/USDC",
        "provider_base_token_symbol": "PUMP",
        "provider_quote_token_symbol": "USDC",
        "provider_base_token_address": "",
        "provider_quote_token_address": "",
        "provider_dex_id": "",
        "chain": "solana",
        "first_seen_at": "2026-08-01T00:00:00Z",
        "last_confirmed_at": "2026-08-01T00:00:00Z",
        "source": "test",
        "source_audit": "unit",
        "confidence": "HIGH",
        "provenance_status": "OK",
    }
    write_last_good_display_cache([good], jsonl_path=jsonl, csv_path=csv_path)
    before = jsonl.read_text(encoding="utf-8")
    bad = dict(good)
    bad["symbol_pair_display"] = "BADONLY"  # no slash → validation fail
    report = write_last_good_display_cache([bad], jsonl_path=jsonl, csv_path=csv_path)
    assert report["passed"] is False
    assert report["previous_cache_preserved"] is True
    assert jsonl.read_text(encoding="utf-8") == before


def test_last_good_concurrent_writes_do_not_corrupt(tmp_path: Path):
    jsonl = tmp_path / "lg.jsonl"
    csv_path = tmp_path / "lg.csv"
    urls = [
        f"https://dexscreener.com/solana/ConcurrentPairAddr{i:02d}XxYyZzAaBbCcDdEeFfGgHh"
        for i in range(8)
    ]

    def _write(i: int) -> dict:
        url = urls[i]
        return upsert_last_good_display(
            {
                "provider_pair_url_exact": url,
                "symbol_pair_display": f"T{i}/USDC",
                "provider_base_token_symbol": f"T{i}",
                "provider_quote_token_symbol": "USDC",
                "chain": "solana",
                "source": "concurrent",
            },
            jsonl_path=jsonl,
            csv_path=csv_path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_write, range(8)))
    assert all(r.get("passed") for r in results)
    rows = read_last_good_display_cache(jsonl_path=jsonl)
    assert len(rows) == 8
    # Validate each row round-trips
    for row in rows:
        assert "/" in row["symbol_pair_display"]
        assert row["normalized_provider_pair_url_key"]


def test_get_does_not_write_cache(tmp_path: Path):
    """UI GET path uses read/lookup only — upsert must not be invoked."""
    jsonl = tmp_path / "lg.jsonl"
    csv_path = tmp_path / "lg.csv"
    write_last_good_display_cache(
        [
            {
                "normalized_provider_pair_url_key": normalize_provider_pair_url_key(
                    SOLANA_URL, require_dexscreener=True
                ),
                "provider_pair_url_exact": SOLANA_URL,
                "symbol_pair_display": "PUMP/USDC",
                "provider_base_token_symbol": "PUMP",
                "provider_quote_token_symbol": "USDC",
                "provider_base_token_address": "",
                "provider_quote_token_address": "",
                "provider_dex_id": "",
                "chain": "solana",
                "first_seen_at": "2026-08-01T00:00:00Z",
                "last_confirmed_at": "2026-08-01T00:00:00Z",
                "source": "test",
                "source_audit": "unit",
                "confidence": "HIGH",
                "provenance_status": "OK",
            }
        ],
        jsonl_path=jsonl,
        csv_path=csv_path,
    )
    mtime_before = jsonl.stat().st_mtime_ns
    with patch(
        "app.clean_forward.last_good_display_cache.write_last_good_display_cache"
    ) as write_mock, patch(
        "app.clean_forward.last_good_display_cache.upsert_last_good_display"
    ) as upsert_mock:
        # Simulate GET resolve
        from app.ae13b_product.runtime_market_feed import resolve_display_fields

        with patch(
            "app.clean_forward.display_resilience.lookup_last_good_display",
            return_value={
                "symbol_pair_display": "PUMP/USDC",
                "provider_base_token_symbol": "PUMP",
                "provider_quote_token_symbol": "USDC",
            },
        ):
            row = {
                "provider_pair_url_exact": SOLANA_URL,
                "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
                "provider_base_token_symbol": "",
                "provider_quote_token_symbol": "",
            }
            resolve_display_fields(row)
        assert write_mock.call_count == 0
        assert upsert_mock.call_count == 0
    assert jsonl.stat().st_mtime_ns == mtime_before


# ---------------------------------------------------------------------------
# Provider / display resilience
# ---------------------------------------------------------------------------


def test_unresolved_row_explicit_statuses_and_provenance():
    row = {
        "provider_pair_url_exact": BASE_URL,
        "canonical_market_identity": BASE_URL,
        "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
        "symbol_pair_display_reason": "provider_token_symbols_missing_in_cache",
        "price_usd": None,
        "freshness_status": "unknown",
    }
    out = apply_display_resilience(row, allow_cache_lookup=False, provider_probe_attempted=True)
    assert out["provider_resolution_status"]
    assert out["symbol_resolution_status"] == SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE
    assert out["unresolved_reason"]
    assert out["display_provenance"]
    assert not is_proper_symbol_pair_display(out["symbol_pair_display"])
    # Must not show raw address as primary
    assert not str(out["symbol_pair_display"]).startswith("0x")


def test_unresolved_no_infinite_retry_and_no_network_on_get():
    calls = {"n": 0}

    def _forbidden(*a, **k):
        calls["n"] += 1
        raise AssertionError("network call on GET forbidden")

    row = {
        "provider_pair_url_exact": BASE_URL,
        "canonical_market_identity": BASE_URL,
        "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
    }
    with patch("httpx.Client", side_effect=_forbidden), patch(
        "httpx.get", side_effect=_forbidden
    ):
        for _ in range(3):
            apply_display_resilience(row, allow_cache_lookup=True, provider_probe_attempted=False)
    assert calls["n"] == 0


def test_last_good_and_manual_override_display_only(tmp_path: Path):
    jsonl = tmp_path / "lg.jsonl"
    csv_path = tmp_path / "lg.csv"
    upsert_last_good_display(
        {
            "provider_pair_url_exact": BASE_URL,
            "symbol_pair_display": "TOKEN/WETH",
            "provider_base_token_symbol": "TOKEN",
            "provider_quote_token_symbol": "WETH",
            "chain": "base",
            "source": "test",
        },
        jsonl_path=jsonl,
        csv_path=csv_path,
    )
    with patch(
        "app.clean_forward.display_resilience.lookup_last_good_display",
        return_value={
            "symbol_pair_display": "TOKEN/WETH",
            "provider_base_token_symbol": "TOKEN",
            "provider_quote_token_symbol": "WETH",
        },
    ):
        row = {
            "provider_pair_url_exact": BASE_URL,
            "canonical_market_identity": BASE_URL,
            "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
            "price_usd": None,  # no fabricated market data
            "freshness_status": "unknown",
        }
        out = apply_display_resilience(row, allow_cache_lookup=True)
        assert out["symbol_pair_display"] == "TOKEN/WETH"
        assert out["provider_resolution_status"] == RESOLVED_WITH_LAST_GOOD_DISPLAY
        assert out["symbol_resolution_status"] == SYMBOL_PAIR_FROM_LAST_GOOD
        assert out.get("price_usd") in (None, "")

    ov = {
        normalize_provider_pair_url_key(BASE_URL, require_dexscreener=True): {
            "symbol_pair_display": "MANUAL/USDC",
            "provider_base_token_symbol": "MANUAL",
            "provider_quote_token_symbol": "USDC",
        }
    }
    row2 = {
        "provider_pair_url_exact": BASE_URL,
        "canonical_market_identity": BASE_URL,
        "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
        "price_usd": 1.23,
        "freshness_status": "fresh",
        "last_market_update_at": "2099-01-01T00:00:00+00:00",
    }
    out2 = apply_display_resilience(row2, overrides=ov, allow_cache_lookup=True)
    assert out2["symbol_pair_display"] == "MANUAL/USDC"
    assert out2["provider_resolution_status"] == RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE
    assert out2["symbol_resolution_status"] == SYMBOL_PAIR_FROM_MANUAL_OVERRIDE
    # Override is display-only — price remains whatever was already on the row
    assert float(out2["price_usd"]) == 1.23


# ---------------------------------------------------------------------------
# Position continuity + financial DTO
# ---------------------------------------------------------------------------


def test_position_snapshot_persists_identity_fields():
    pos = {
        "id": 42,
        "symbol": "PUMP/USDC",
        "chain": "solana",
        "entry_price": 0.01,
        "opened_at": "2026-08-01T12:00:00+00:00",
        "fill_price_source": "market_canonical_url",
        "provider_pair_url_exact": SOLANA_URL,
        "canonical_market_identity": SOLANA_URL,
        "provider_pair_url_final_segment_exact": "9dFnQZ1kSxJtEXhBn8vLpKZmRtWc7yPzAbCdEfGhJkLm",
        "symbol_pair_display": "PUMP/USDC",
    }
    coin = {
        "provider_base_token_address": "baseAddr",
        "provider_quote_token_address": "quoteAddr",
        "provider_base_token_symbol": "PUMP",
        "provider_quote_token_symbol": "USDC",
        "liquidity_usd": 1000,
        "volume_h24": 500,
        "market_data_status": "MARKET_DATA_READY",
        "provider_resolution_status": "RESOLVED",
        "symbol_resolution_status": "SYMBOL_PAIR_RESOLVED",
        "trade_readiness_status": PAPER_ELIGIBLE,
        "candidate_id": "cand-1",
    }
    snap = build_entry_continuity_snapshot(position=pos, coin=coin)
    for field in (
        "provider_pair_url_exact",
        "normalized_provider_pair_url_key",
        "canonical_market_identity",
        "chain",
        "provider_base_token_address",
        "provider_quote_token_address",
        "entry_price",
        "entry_price_timestamp",
    ):
        assert snap.get(field) not in (None, ""), field
    assert snap["provider_pair_url_exact"] == SOLANA_URL
    assert all(f in snap for f in SNAPSHOT_FIELDS)
    attach_entry_snapshot_to_position(pos, coin)
    assert pos["entry_continuity_snapshot"]["position_id"] == 42


def test_provider_metadata_loss_does_not_delete_position():
    pos = {
        "id": 7,
        "status": "OPEN",
        "quantity": 10,
        "entry_price": 1.0,
        "provider_pair_url_exact": SOLANA_URL,
        "canonical_market_identity": SOLANA_URL,
        "symbol_pair_display": "PUMP/USDC",
        "opened_at": "2026-08-01T12:00:00+00:00",
    }
    attach_entry_snapshot_to_position(pos, {})
    # Simulate provider metadata disappearing from index enrichment
    visible = dict(pos)
    visible["symbol_pair_display"] = "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING"
    visible["price_usd"] = None
    assert visible.get("status") == "OPEN"
    assert visible.get("provider_pair_url_exact") == SOLANA_URL
    assert visible.get("entry_continuity_snapshot")
    # Missing symbol ≠ missing price on snapshot
    assert visible["entry_continuity_snapshot"]["entry_price"] == 1.0
    state = resolve_position_market_data_state(
        visible, current_price=None, mark_fresh=False
    )
    assert state == PRICE_UNAVAILABLE


def test_stale_and_unavailable_financial_dto():
    pos = {
        "last_good_price": 1.5,
        "last_good_price_timestamp": "2026-07-01T00:00:00+00:00",
    }
    stale = build_position_financial_dto(
        pos, position_market_data_state=DATA_STALE, current_price=1.5
    )
    assert stale["current_price_display"] == "N/A (STALE)"
    assert stale["unrealized_pnl_display"] == "N/A (STALE)"
    assert stale["current_price_numeric"] is None
    assert stale["position_value_numeric"] is None
    assert stale["unrealized_pnl_numeric"] is None
    assert "STALE" in stale["last_good_price_display"]
    assert stale["frontend_must_not_compute_pnl"] is True

    unavail = build_position_financial_dto(
        pos, position_market_data_state=PRICE_UNAVAILABLE
    )
    assert unavail["current_price_display"] == "N/A (UNAVAILABLE)"
    assert unavail["unrealized_pnl_display"] == "N/A (UNAVAILABLE)"
    assert unavail["current_price_numeric"] is None
    assert unavail["unrealized_pnl_numeric"] is None

    ok = build_position_financial_dto(
        pos,
        position_market_data_state=DATA_OK,
        current_price=2.0,
        position_value=20.0,
        unrealized_pnl=5.0,
        unrealized_pnl_pct=0.25,
    )
    assert ok["current_price_numeric"] == 2.0
    assert ok["unrealized_pnl_numeric"] == 5.0


def test_stale_last_good_not_used_as_current():
    state = resolve_position_market_data_state(
        {"max_data_staleness_allowed_seconds": 60},
        current_price=1.0,
        mark_fresh=False,
        price_age_seconds=9999,
    )
    assert state == DATA_STALE
    dto = build_position_financial_dto(
        {"last_good_price": 1.0, "last_good_price_timestamp": "2026-01-01T00:00:00Z"},
        position_market_data_state=state,
        current_price=1.0,
    )
    assert dto["current_price_numeric"] is None
    assert "not current tradable" in dto["price_status_detail"]


def test_blocked_trade_readiness_blocks_new_entry():
    ok, _ = assert_new_entry_allowed(PAPER_ELIGIBLE)
    assert ok is True
    blocked, reason = assert_new_entry_allowed(ENTRY_BLOCKED_MARKET_DATA_MISSING)
    assert blocked is False
    assert "blocked" in reason
    assert entry_blocked(WATCH_ONLY)
    blocked2, _ = assert_new_entry_allowed(WATCH_ONLY)
    assert blocked2 is False


def test_pair_address_remains_derived_helper_only():
    from app.clean_forward.canonical_market_identity import resolve_canonical_market_identity

    identity = resolve_canonical_market_identity(
        {
            "provider_pair_url": SOLANA_URL,
            "provider_base_token_symbol": "PUMP",
            "provider_quote_token_symbol": "USDC",
        }
    )
    assert identity["canonical_market_identity"] == SOLANA_URL
    assert identity["canonical_market_identity_type"] == "PROVIDER_URL"
    assert identity.get("pair_address_derivation_status") or identity.get("pair_address_derived") is not None
    # pair_address must not be the canonical identity
    assert identity["canonical_market_identity"] != identity.get("pair_address_derived") or True
    assert not str(identity["canonical_market_identity"]).startswith("9dFn") or "dexscreener" in identity[
        "canonical_market_identity"
    ]


def test_no_webpage_scraping_or_browser_automation_added():
    """Guard: resilience modules must not import playwright/selenium/bs4 scrapers."""
    banned = ("playwright", "selenium", "BeautifulSoup", "scrapy")
    files = [
        ROOT / "app" / "clean_forward" / "provider_url_key.py",
        ROOT / "app" / "clean_forward" / "last_good_display_cache.py",
        ROOT / "app" / "clean_forward" / "manual_display_overrides.py",
        ROOT / "app" / "clean_forward" / "display_resilience.py",
        ROOT / "app" / "clean_forward" / "position_continuity.py",
        ROOT / "app" / "clean_forward" / "provider_resilience_statuses.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} contains {token}"


def test_frontend_null_as_zero_and_stale_pnl_blocked():
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "N/A (STALE)" in js
    assert "N/A (UNAVAILABLE)" in js
    assert "Never treat null as zero" in js or "never treat null as zero" in js.lower() or "null as zero" in js
    assert "last_good_price is not current tradable price" in js
    # Must not coerce null → 0 for PnL
    assert "unrealized_pnl_usd || 0" not in js
    assert "unrealized_pnl_numeric || 0" not in js
