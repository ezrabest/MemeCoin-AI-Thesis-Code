"""AE18 follow-up: display repair, social classification, sentiment panel,
BUY DEMO CANDIDATE action, and structured provider refresh failures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae13b_product import provider_refresh_errors as refresh_errors
from app.ae13b_product.news_sentiment_cache import (
    NEWS_SENTIMENT_CACHE_EMPTY,
    NEWS_SENTIMENT_CACHE_READY,
    NEWS_SENTIMENT_CACHE_STALE,
    NEWS_SENTIMENT_CACHE_UNAVAILABLE,
    build_cached_news_sentiment,
)
from app.ae13b_product.runtime_market_feed import (
    _index_row_to_clean_forward,
    _index_row_to_live_market,
    build_clean_forward_from_index,
    build_live_market_from_index,
    enrich_opportunity_rows,
    repair_legacy_position_identity,
)
from app.clean_forward.canonical_market_identity import build_index_row
from app.clean_forward.display_identity import (
    PARTIAL_PROVIDER_SYMBOLS_MISSING,
    SYMBOL_PAIR_UNAVAILABLE,
    SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
    classify_social_candidate,
    derive_symbol_pair_display,
    is_symbol_pair_available,
)
from app.clean_forward.runtime_identity_index import write_runtime_index
from app.runtime.shutdown import reset_shutdown_for_tests
from app.runtime.ui_get_network_guard import (
    reset_counters_for_tests,
    snapshot_counters,
    ui_get_network_guard,
)

ROOT = Path(__file__).resolve().parents[1]

STALE_URLS = [
    "https://dexscreener.com/robinhood/0xb3F901859ACbEF2288E187993AA50911A5404762",
    "https://dexscreener.com/base/0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4",
    "https://dexscreener.com/robinhood/0xEA63b938967e65B2D71d99Bc8cFD9c4cB3c7c105",
    "https://dexscreener.com/base/0x02a26e25e8d1932f07ab89c8014d53730fd9ffe63ab9ca920a7a0d2a74376789",
]


@pytest.fixture(autouse=True)
def _reset_runtime_flags():
    reset_shutdown_for_tests()
    reset_counters_for_tests()
    yield
    reset_shutdown_for_tests()
    reset_counters_for_tests()


def _looks_like_address(text: str) -> bool:
    for part in str(text or "").split("/"):
        cleaned = part.strip().replace("\u2026", "").replace("...", "")
        if cleaned.lower().startswith("0x") and len(cleaned) >= 8:
            return True
        if len(cleaned) >= 24 and cleaned.isalnum() and not cleaned.isupper():
            return True
    return False


# ---------------------------------------------------------------- PART A / B


def test_stale_row_does_not_show_raw_address_as_symbol_pair():
    row = {
        "provider_pair_url": STALE_URLS[0],
        "chain": "robinhood",
        "provider_base_token_address": "0x702285df8D54905f00037A82f1EF3E51e4659b03",
        "provider_quote_token_address": "0x0Bd7Cf1E4C4C0eD0eA1F0f7B7d3F0a1f1c5cAD73",
    }
    built = build_index_row(row, last_identity_rebuild_at="2026-07-31T00:00:00+00:00")
    assert built["symbol_pair_display"] == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
    assert not _looks_like_address(built["symbol_pair_display"])
    # Address pair is retained for the details column only.
    assert "0x7022" in built["symbol_pair_address_fallback"]


def test_stale_rows_have_explicit_unresolved_reason():
    from app.clean_forward.provider_resilience_statuses import (
        RESOLVED_WITH_LAST_GOOD_DISPLAY,
        RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE,
        is_proper_symbol_pair_display,
    )

    for url in STALE_URLS:
        built = build_index_row(
            {"provider_pair_url": url, "provider_base_token_address": "0xabc123abc123abc123"},
            last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
        )
        if is_proper_symbol_pair_display(built.get("symbol_pair_display")):
            # Last-good / manual override may recover display without fabricating market data.
            assert built.get("provider_resolution_status") in {
                RESOLVED_WITH_LAST_GOOD_DISPLAY,
                RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE,
                "RESOLVED",
            }
            assert built.get("symbol_pair_display_reason") in {
                "last_good_display_cache",
                "manual_display_override",
                "",
            }
        else:
            assert not is_symbol_pair_available(built["symbol_pair_display"])
            assert built["symbol_pair_display_reason"] or built.get("unresolved_reason"), (
                f"missing reason for {url}"
            )


def test_symbol_pair_never_base_only_when_quote_symbol_exists():
    out = derive_symbol_pair_display(
        {"provider_base_token_symbol": "PUMP", "provider_quote_token_symbol": "SOL"}
    )
    assert out["symbol_pair_display"] == "PUMP/SOL"


def test_quote_symbol_missing_yields_partial_status_not_half_address():
    out = derive_symbol_pair_display(
        {
            "provider_base_token_symbol": "PUMP",
            "provider_quote_token_address": "So11111111111111111111111111111111111111112",
        }
    )
    assert out["symbol_pair_display"] == PARTIAL_PROVIDER_SYMBOLS_MISSING
    assert out["symbol_pair_known_side_symbol"] == "PUMP"
    assert "So1111" in out["symbol_pair_address_fallback"]


def test_base_only_symbol_without_quote_evidence_is_partial():
    out = derive_symbol_pair_display({"provider_base_token_symbol": "PUMP"})
    assert out["symbol_pair_display"] == PARTIAL_PROVIDER_SYMBOLS_MISSING
    assert out["symbol_pair_display_reason"] == "only_base_token_symbol_available_from_provider"


def test_market_snapshot_shows_full_pair_not_base_only():
    index_row = build_index_row(
        {
            "provider_pair_url": "https://dexscreener.com/solana/AbCdEf",
            "provider_base_token_symbol": "PUMP",
            "provider_quote_token_symbol": "USDC",
        },
        last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
    )
    mapped = _index_row_to_live_market(index_row)
    assert mapped["symbol"] == "PUMP/USDC"
    assert mapped["symbol_pair_display"] == "PUMP/USDC"


def test_clean_feed_shows_full_pair_or_explicit_status():
    unresolved = build_index_row(
        {"provider_pair_url": STALE_URLS[1], "provider_base_token_address": "0xdeadbeefdeadbeef"},
        last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
    )
    mapped = _index_row_to_clean_forward(unresolved)
    assert mapped["symbol_pair_display"] in (
        SYMBOL_PAIR_UNAVAILABLE,
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
    )
    assert mapped["symbol_pair_available"] is False
    assert not _looks_like_address(mapped["symbol_pair_display"])


def test_live_index_has_no_raw_address_symbol_pairs():
    data = build_live_market_from_index(limit=500)
    if not data.get("ok"):
        pytest.skip("runtime index not built")
    for row in data["rows"]:
        assert not _looks_like_address(row["symbol_pair_display"]), row["canonical_market_identity"]


def test_clean_feed_index_has_no_base_only_symbol_pairs():
    data = build_clean_forward_from_index(limit=500)
    if not data.get("ok"):
        pytest.skip("runtime index not built")
    for row in data["rows"]:
        display = row["symbol_pair_display"]
        if is_symbol_pair_available(display):
            assert "/" in display, display


def test_market_opportunities_shows_full_pair_or_status():
    rows = [{"symbol": "PUMP", "pair_address": "0xdoesnotexist", "chain": "base"}]
    out = enrich_opportunity_rows(rows)
    row = out["rows"][0]
    assert not _looks_like_address(row["symbol_pair_display"])
    if is_symbol_pair_available(row["symbol_pair_display"]):
        assert "/" in row["symbol_pair_display"]
    else:
        assert row["symbol_pair_display_reason"]
    assert row["pair_address_is_canonical"] is False


def test_portfolio_shows_full_pair_or_status():
    index_rows = [
        build_index_row(
            {
                "provider_pair_url": "https://dexscreener.com/solana/AbCdEf",
                "provider_base_token_symbol": "PUMP",
                "provider_quote_token_symbol": "SOL",
            },
            last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
        )
    ]
    repaired = repair_legacy_position_identity(
        {"canonical_market_identity": "https://dexscreener.com/solana/AbCdEf"}, index_rows
    )
    assert repaired["symbol_pair_display"] == "PUMP/SOL"

    orphan = repair_legacy_position_identity(
        {"canonical_market_identity": "https://dexscreener.com/base/0xUnknown"}, index_rows
    )
    assert orphan["symbol_pair_display"] == SYMBOL_PAIR_UNAVAILABLE
    assert orphan["symbol_pair_display_reason"]


def test_manual_refresh_repairs_display_atomically(tmp_path, monkeypatch):
    from app.ae13b_product import manual_refresh_runtime_index as mr

    jsonl = tmp_path / "index.jsonl"
    csv_path = tmp_path / "index.csv"
    seed = tmp_path / "seed.csv"
    seed.write_text(
        "provider_pair_url,chain,provider_base_token_address\n"
        "https://dexscreener.com/base/0xAbCdEf1234,base,0xbase111\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mr, "INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr(mr, "INDEX_CSV_PATH", csv_path)

    def fake_verify(src, *, use_cache, overwrite=False):
        out = dict(src)
        out["provider_base_token_symbol"] = "REPAIRED"
        out["provider_quote_token_symbol"] = "WETH"
        return out

    monkeypatch.setattr(mr, "_verify_one_row", fake_verify)
    meta = mr.manual_refresh_runtime_index(force=True, seed_csv=seed, allow_dexscreener=True)

    assert meta["runtime_index_update_status"] == "ATOMIC_OK"
    assert meta["runtime_index_updated"] is True
    written = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert written[0]["symbol_pair_display"] == "REPAIRED/WETH"
    assert meta["display_repaired_rows"] >= 1


# -------------------------------------------------------------------- PART C


def test_social_classification_from_seed_collection():
    out = classify_social_candidate({"seed_collection": "USER_SEED_SOCIALFI"})
    assert out["is_social_candidate"] is True
    assert out["social_classification"] == "SOCIAL_CANDIDATE_UNCONFIRMED"
    assert "seed_collection" in out["social_source"]
    assert out["semantic_status"] == "SOCIAL_CANDIDATE_UNCONFIRMED"


def test_social_classification_not_invented_without_evidence():
    out = classify_social_candidate({"seed_collection": "EXISTING_CLEAN_FORWARD"})
    assert out["is_social_candidate"] is False
    assert out["social_classification"] == "NON_SOCIAL_OR_UNCLASSIFIED"
    assert out["social_reason"] == "no_social_source_evidence"


def test_social_fields_propagate_into_runtime_index():
    built = build_index_row(
        {
            "provider_pair_url": "https://dexscreener.com/base/0xAbC",
            "seed_collection": "USER_SEED_SOCIAL_INFRA",
            "target_source": "USER_DEXSCREENER_SEED",
            "provider_base_token_symbol": "SOC",
            "provider_quote_token_symbol": "WETH",
        },
        last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
    )
    assert built["is_social_candidate"] is True
    assert built["seed_collection"] == "USER_SEED_SOCIAL_INFRA"
    assert built["manual_curation_status"] == "USER_MANUAL_SEED"


def test_social_rows_visible_through_social_filter():
    data = build_live_market_from_index(limit=500, status_filter="social", filter_mode="hide")
    if not data.get("ok"):
        pytest.skip("runtime index not built")
    assert data["social_rows_count"] > 0
    assert len(data["rows"]) == data["passed_filter"]
    for row in data["rows"]:
        assert row["is_social_candidate"] or row["is_social_confirmed"]


def test_other_semantic_filters_still_work():
    for key in ("all", "opportunistic", "unknown", "unresolved", "infrastructure"):
        data = build_live_market_from_index(limit=50, status_filter=key)
        if not data.get("ok"):
            pytest.skip("runtime index not built")
        assert data["status_filter"] == key


# -------------------------------------------------------------------- PART D


def test_rss_panel_never_blank():
    panel = build_cached_news_sentiment(limit=5)
    assert panel["panel_blank"] is False
    assert panel["rss_news_sentiment_status"] in (
        NEWS_SENTIMENT_CACHE_READY,
        NEWS_SENTIMENT_CACHE_EMPTY,
        NEWS_SENTIMENT_CACHE_STALE,
        NEWS_SENTIMENT_CACHE_UNAVAILABLE,
    )


def test_rss_panel_reads_cache_only_on_get():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/news-sentiment-cache"):
        panel = build_cached_news_sentiment(limit=5)
    snap = snapshot_counters()
    assert snap["rss_calls_on_get"] == 0
    assert snap["external_network_calls_on_get"] == 0
    assert panel["get_path_fetches_rss_live"] is False


def test_rss_panel_reports_empty_status(monkeypatch):
    import app.ae13b_product.news_sentiment_cache as cache

    monkeypatch.setattr(cache, "_load_sentiment_records", lambda limit: ([], ""))
    monkeypatch.setattr(cache, "_count_cached_rss_payloads", lambda: (0, ""))
    panel = cache.build_cached_news_sentiment(limit=5)
    assert panel["rss_news_sentiment_status"] == NEWS_SENTIMENT_CACHE_EMPTY
    assert panel["sentiment_cache_missing_reason"]
    assert panel["panel_blank"] is False


def test_rss_panel_reports_stale_status(monkeypatch):
    import app.ae13b_product.news_sentiment_cache as cache

    monkeypatch.setattr(
        cache,
        "_load_sentiment_records",
        lambda limit: (
            [{"title": "old news", "sentiment_score": 0.5, "timestamp": "2020-01-01T00:00:00+00:00"}],
            "",
        ),
    )
    monkeypatch.setattr(cache, "_count_cached_rss_payloads", lambda: (1, ""))
    panel = cache.build_cached_news_sentiment(limit=5)
    assert panel["rss_news_sentiment_status"] == NEWS_SENTIMENT_CACHE_STALE


def test_rss_panel_reports_unavailable_status(monkeypatch):
    import app.ae13b_product.news_sentiment_cache as cache

    monkeypatch.setattr(cache, "_load_sentiment_records", lambda limit: ([], "OperationalError: no table"))
    panel = cache.build_cached_news_sentiment(limit=5)
    assert panel["rss_news_sentiment_status"] == NEWS_SENTIMENT_CACHE_UNAVAILABLE


# -------------------------------------------------------------------- PART E


def _js() -> str:
    return (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8", errors="ignore")


def _api_src() -> str:
    return (ROOT / "app" / "api.py").read_text(encoding="utf-8", errors="ignore")


def test_buy_demo_candidate_button_has_handler():
    js = _js()
    assert "BUY DEMO CANDIDATE" in js
    assert "mktBuyDemoCandidate" in js
    assert 'onclick="mktBuyDemoCandidate(this)"' in js
    assert "/api/ae13b/demo/buy-candidate" in js


def test_demo_endpoint_is_demo_only_and_url_first():
    src = _api_src()
    assert "ae13b_demo_buy_candidate" in src
    assert "DEMO_ACTION_BLOCKED_MODE_DISABLED" in src
    assert "_enforce_paper_demo_execution_guard" in src
    for code in (
        "DEMO_ACTION_BLOCKED_RISK_GATE",
        "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED",
        "DEMO_ACTION_BLOCKED_PRICE_UNAVAILABLE",
        "DEMO_ACTION_BLOCKED_CANDIDATE_NOT_FOUND",
        "DEMO_ACTION_FAILED_INTERNAL_ERROR",
    ):
        assert code in src


def test_demo_action_requires_canonical_url_not_pair_address():
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    res = client.post("/api/ae13b/demo/buy-candidate", json={})
    body = res.json()
    assert body["ok"] is False
    assert body["demo_action_status"] == "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED"
    assert body["blocked_reason_explicit"] is True
    assert body["pair_address_required_as_canonical"] is False

    # A bare pair address is not accepted as identity.
    res2 = client.post(
        "/api/ae13b/demo/buy-candidate",
        json={"canonical_market_identity": "0xb3F901859ACbEF2288E187993AA50911A5404762"},
    )
    body2 = res2.json()
    assert body2["demo_action_status"] == "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED"


def test_demo_action_unknown_candidate_blocked_with_reason():
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    res = client.post(
        "/api/ae13b/demo/buy-candidate",
        json={"canonical_market_identity": "https://dexscreener.com/base/0xNotInIndexAtAll"},
    )
    body = res.json()
    assert body["ok"] is False
    assert body["demo_action_status"] in (
        "DEMO_ACTION_BLOCKED_CANDIDATE_NOT_FOUND",
        "DEMO_ACTION_BLOCKED_MODE_DISABLED",
    )
    assert body["demo_action_blocked_reason"]


def test_no_live_trading_or_wallet_path_in_demo_action():
    src = _api_src()
    start = src.index("def ae13b_demo_buy_candidate")
    end = src.index("@app.post(\"/api/demo/sell\")")
    body = src[start:end]
    # Strip the docstring so prose ("no wallet, signer, ...") is not matched as code.
    code = body.split('"""', 2)[-1]
    for forbidden in (
        "private_key",
        "signer(",
        "sign_transaction",
        "send_transaction",
        "wallet_keypair",
        "live_execute",
        "Keypair",
    ):
        assert forbidden not in code
    assert '"live_trading_enabled": False' in body
    assert '"real_signing_enabled": False' in body


# -------------------------------------------------------------------- PART F


def test_refresh_failures_are_structured():
    for code in refresh_errors.ERROR_CODES:
        failure = refresh_errors.build_refresh_failure(error_code=code, provider_url="https://x/y")
        assert failure["refresh_error_code"] == code
        assert failure["refresh_error_reason"]
        assert failure["recovery_instruction"]
        assert "retryable" in failure
        assert failure["refresh_status"] == "FAILED"


def test_generic_aborted_without_reason_is_classified():
    class AbortError(Exception):
        pass

    exc = AbortError("signal is aborted without reason")
    code = refresh_errors.classify_refresh_exception(exc)
    assert code == refresh_errors.NETWORK_TIMEOUT
    failure = refresh_errors.build_refresh_failure(error_code=code, exception=exc)
    assert "without reason" not in failure["user_message"].lower()
    assert failure["recovery_instruction"]

    summary = refresh_errors.summarize_failures([failure])
    assert summary["generic_abort_without_reason_count"] == 0
    assert summary["recovery_instruction_present_count"] == 1


def test_http_status_classification():
    assert refresh_errors.classify_refresh_exception(None, http_status=404) == refresh_errors.PROVIDER_HTTP_404
    assert refresh_errors.classify_refresh_exception(None, http_status=429) == refresh_errors.PROVIDER_HTTP_429
    assert refresh_errors.classify_refresh_exception(None, http_status=503) == refresh_errors.PROVIDER_HTTP_5XX


def test_ui_no_longer_prints_generic_refresh_failure():
    js = _js()
    assert "Provider pair verification failed or refresh unavailable" not in js
    assert "refreshErrorMessage" in js
    assert "NETWORK_TIMEOUT" in js


def test_manual_refresh_reports_shutdown_structured(monkeypatch):
    from app.ae13b_product import manual_refresh_runtime_index as mr

    monkeypatch.setattr(mr, "is_shutting_down", lambda: True)
    meta = mr.manual_refresh_runtime_index()
    assert meta["refresh_error_code"] == refresh_errors.CONTROLLED_SHUTDOWN_SKIP
    assert meta["recovery_instruction"]
    assert meta["failure_summary"]["controlled_shutdown_skip_count"] == 1


# ----------------------------------------------------------------- PART G/H/I


def test_get_paths_remain_network_isolated():
    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/clean-forward-market-feed"):
        build_clean_forward_from_index(limit=10)
    with ui_get_network_guard("/api/ae13b/live-market"):
        build_live_market_from_index(limit=10)
    with ui_get_network_guard("/api/ae13b/opportunities"):
        enrich_opportunity_rows([{"symbol": "X", "pair_address": "0xabc"}])
    with ui_get_network_guard("/api/ae13b/news-sentiment-cache"):
        build_cached_news_sentiment(limit=5)
    snap = snapshot_counters()
    assert snap["external_network_calls_on_get"] == 0
    assert snap["dexscreener_calls_on_get"] == 0
    assert snap["helius_calls_on_get"] == 0
    assert snap["rss_calls_on_get"] == 0
    assert snap["provider_refresh_on_get"] == 0


def test_shutdown_lifecycle_audit_still_passes():
    audit = json.loads(
        (ROOT / "data" / "audits" / "ae18_shutdown_lifecycle_audit.json").read_text(encoding="utf-8")
    )
    assert audit["passed"] is True
    assert audit["min_scan_interval_seconds"] >= 5.0
    assert audit["next_scan_zero_prevented"] is True


def test_pair_address_remains_derived_helper_only():
    built = build_index_row(
        {
            "provider_pair_url": "https://dexscreener.com/base/0xAbCdEf",
            "provider_base_token_symbol": "A",
            "provider_quote_token_symbol": "B",
        },
        last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
    )
    assert built["canonical_market_identity_type"] == "PROVIDER_URL"
    assert built["canonical_market_identity"].startswith("https://")
    mapped = _index_row_to_clean_forward(built)
    assert mapped["pair_address_label"] == "DERIVED HELPER ID"
    assert mapped["data_row_key"] == built["canonical_market_identity"]


def test_new_audits_pass():
    audits_dir = ROOT / "data" / "audits"
    names = [
        "ae18_stale_degraded_display_repair_audit.json",
        "ae18_symbol_pair_display_audit.json",
        "ae18_social_classification_display_audit.json",
        "ae18_rss_news_sentiment_panel_audit.json",
        "ae18_buy_demo_candidate_action_audit.json",
        "ae18_provider_refresh_failure_reason_audit.json",
    ]
    for name in names:
        path = audits_dir / name
        assert path.exists(), name
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["passed"] is True, name
        assert data["fail_closed"] is True, name


def test_stale_display_audit_has_zero_raw_address_after():
    data = json.loads(
        (ROOT / "data" / "audits" / "ae18_stale_degraded_display_repair_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["raw_address_symbol_pair_after_count"] == 0
    assert data["unresolved_rows_without_reason_count"] == 0
    for entry in data["affected_urls_checked"]:
        assert entry["raw_address_as_primary_symbol"] is False


def test_index_write_round_trip_keeps_social_and_display_fields(tmp_path):
    row = build_index_row(
        {
            "provider_pair_url": "https://dexscreener.com/base/0xAbCdEf",
            "seed_collection": "USER_SEED_SOCIALFI",
            "provider_base_token_symbol": "SOC",
            "provider_quote_token_symbol": "WETH",
        },
        last_identity_rebuild_at="2026-07-31T00:00:00+00:00",
    )
    jsonl = tmp_path / "i.jsonl"
    csv_path = tmp_path / "i.csv"
    write_runtime_index([row], jsonl_path=jsonl, csv_path=csv_path, atomic=True)
    loaded = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["symbol_pair_display"] == "SOC/WETH"
    assert loaded["symbol_pair_display_status"] == "FULL_PAIR"
    assert loaded["is_social_candidate"] is True
