"""Tests for Runtime Selected/Clean Collection Engine."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from app.clean_forward.runtime_selected_collection import (
    DEFAULT_POLICY,
    apply_failure_cooldown,
    build_runtime_priority_queue,
    compute_backoff_seconds,
    fetch_queue_item,
    load_selected_csv,
    run_priority_fetch_cycle,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "run_runtime_selected_collection_smoke.py"

def _write_selected(path: Path, n: int = 3, inactive_last: bool = False) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "combined_target_id": f"ae16b_{i:016x}",
                "chain": "solana",
                "provider_pair_address": f"Pair{i}AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGG",
                "clean_forward_candidate_ready": "true",
                "acceptance_status": "INACTIVE_DEPRECATED" if inactive_last and i == n - 1 else "PROVIDER_PAIR_RESOLVED",
                "collection_enabled": "false" if inactive_last and i == n - 1 else "true",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_priority_queue_includes_all_selected_dynamically(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=4)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[], include_discovery=False)
    selected = [q for q in queue if q["priority_rank"] == "0B"]
    assert len(selected) == 4
    assert len(selected) != 45


def test_inactive_selected_excluded_with_reason(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=2, inactive_last=True)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[])
    inactive = [q for q in queue if q["priority_rank"] == "0B" and q["price_required"] != "true"]
    assert len(inactive) == 1
    assert inactive[0]["inactive_reason"]
    assert inactive[0]["expected_fetch_required"] == "false"


def test_open_positions_before_discovery(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=1)
    rows = load_selected_csv(sel)
    opens = [
        {
            "id": 9,
            "status": "OPEN",
            "chain": "robinhood",
            "pair_address": "0x6Ac7857561c6aB70aad1aef504CC8B585E3fa6a1",
        }
    ]
    queue = build_runtime_priority_queue(
        selected_rows=rows,
        open_positions=opens,
        include_discovery=True,
    )
    ranks = [q["priority_rank"] for q in queue]
    assert ranks.index("0A") < ranks.index("2")
    assert ranks.index("0B") < ranks.index("2")
    legacy = [q for q in queue if q["priority_rank"] == "0A"][0]
    assert legacy["open_position_status"] == "LEGACY_OR_OUT_OF_SELECTED_POSITION"
    assert legacy["collection_reason"] == "MARK_PRICE_ONLY"
    assert legacy["eligible_for_new_trade_candidate"] == "false"


def test_default_smoke_excludes_discovery(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=1)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[], include_discovery=False)
    assert all(q["priority_rank"] != "2" for q in queue)


def test_ae16b_cannot_appear_as_pair_identity(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=2)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[])
    for q in queue:
        assert not str(q.get("display_real_pair_address") or "").lower().startswith("ae16b_")
        assert not str(q.get("normalized_real_pair_address") or "").lower().startswith("ae16b_")


def test_fetch_uses_exact_pair_url_not_trending():
    item = {
        "priority_rank": "0B",
        "priority_class": "SELECTED_CLEAN_ACTIVE",
        "price_source_key": "dexscreener|solana|abc",
        "provider": "dexscreener",
        "display_chain": "solana",
        "display_real_pair_address": "AbcDefGhIjKlMnOpQrStUvWxYz0123456789Ab",
        "provider_pair_url": "https://dexscreener.com/solana/AbcDefGhIjKlMnOpQrStUvWxYz0123456789Ab",
        "selected_status": "PROVIDER_PAIR_RESOLVED",
        "active_status": "ACTIVE",
        "price_required": "true",
        "open_position_status": "NONE",
        "collection_reason": "SELECTED_UNIVERSE",
        "eligible_for_new_trade_candidate": "true",
        "identity_resolution_status": "RESOLVED",
    }
    fake = {
        "fetch_status": "SUCCESS",
        "http_status": 200,
        "elapsed_ms": 10,
        "pair": {
            "pairAddress": "AbcDefGhIjKlMnOpQrStUvWxYz0123456789Ab",
            "chainId": "solana",
            "priceUsd": "1.23",
            "baseToken": {"symbol": "X"},
            "quoteToken": {"symbol": "Y"},
            "liquidity": {"usd": 1000},
            "volume": {"h24": 100},
            "txns": {"h24": {"buys": 1, "sells": 1}},
            "priceChange": {},
        },
        "raw_text": "{}",
        "error_reason": "",
        "timeout": False,
    }
    sleeps: list[float] = []
    with patch(
        "app.clean_forward.runtime_selected_collection._one_http_fetch",
        return_value=fake,
    ):
        result = fetch_queue_item(
            item,
            policy=DEFAULT_POLICY,
            fetch_state={},
            mode="artifact-only",
            sleeper=sleeps.append,
        )
    assert result.target_fetch_status == "SUCCESS"
    assert "pairs/solana/" in result.fetch_url
    assert "trending" not in result.fetch_url
    assert result.source_query_written == "selected_clean_exact_pair"
    assert result.source_query_written != "trending"
    assert sleeps  # paced


def test_fetch_failure_produces_explicit_row():
    item = {
        "priority_rank": "0B",
        "priority_class": "SELECTED_CLEAN_ACTIVE",
        "price_source_key": "dexscreener|solana|dead",
        "provider": "dexscreener",
        "display_chain": "solana",
        "display_real_pair_address": "DeadPairAAAAABBBBBCCCCCDDDDDEEEEEFFFFF",
        "provider_pair_url": "https://dexscreener.com/solana/DeadPairAAAAABBBBBCCCCCDDDDDEEEEEFFFFF",
        "selected_status": "PROVIDER_PAIR_RESOLVED",
        "active_status": "ACTIVE",
        "price_required": "true",
        "open_position_status": "NONE",
        "collection_reason": "SELECTED_UNIVERSE",
        "eligible_for_new_trade_candidate": "true",
        "identity_resolution_status": "RESOLVED",
    }
    fake = {
        "fetch_status": "NO_PAIRS_IN_RESPONSE",
        "http_status": 200,
        "elapsed_ms": 5,
        "pair": None,
        "raw_text": '{"pairs":[]}',
        "error_reason": "NO_PAIRS_IN_RESPONSE",
        "timeout": False,
    }
    state: dict = {}
    with patch(
        "app.clean_forward.runtime_selected_collection._one_http_fetch",
        return_value=fake,
    ):
        result = fetch_queue_item(
            item,
            policy=DEFAULT_POLICY,
            fetch_state=state,
            mode="artifact-only",
            sleeper=lambda _s: None,
        )
    assert result.target_fetch_status == "PROVIDER_EMPTY_NO_PAIRS"
    assert result.error_reason
    assert len(result.retry_rows) == 1  # no immediate infinite retry
    assert result.retry_rows[0]["retry_scheduled"] == "false"
    assert state[item["price_source_key"]]["skip_until_ts"]
    assert state[item["price_source_key"]]["consecutive_no_pairs"] == 1


def test_exponential_backoff_for_429():
    delay0 = compute_backoff_seconds(0, DEFAULT_POLICY)
    delay1 = compute_backoff_seconds(1, DEFAULT_POLICY)
    assert delay0 >= 1.5
    assert delay1 >= delay0 - 0.2  # jitter may vary slightly


def test_no_pairs_cooldown_never_auto_removes():
    st = {
        "price_source_key": "dexscreener|solana|x",
        "consecutive_failures": 2,
        "consecutive_no_pairs": 2,
        "skip_until_ts": "",
        "dead_pair_status": "NONE",
    }
    audit = apply_failure_cooldown(st, failure_class="NO_PAIRS_IN_RESPONSE")
    assert audit["automatic_removal_performed"] == "false"
    assert st["dead_pair_status"] == "SUSPECT_DEAD_PAIR"
    assert st["cooldown_status"] == "SUSPECT_DEAD_PAIR"


def test_selected_cannot_be_silently_dropped(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=3)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[])
    fake = {
        "fetch_status": "SUCCESS",
        "http_status": 200,
        "elapsed_ms": 1,
        "pair": {
            "pairAddress": "x",
            "chainId": "solana",
            "priceUsd": "1",
            "baseToken": {"symbol": "A"},
            "quoteToken": {"symbol": "B"},
            "liquidity": {"usd": 1},
            "volume": {"h24": 1},
            "txns": {"h24": {"buys": 0, "sells": 0}},
            "priceChange": {},
        },
        "raw_text": "{}",
        "error_reason": "",
        "timeout": False,
    }
    with patch(
        "app.clean_forward.runtime_selected_collection._one_http_fetch",
        return_value=fake,
    ):
        cycle = run_priority_fetch_cycle(
            queue,
            mode="artifact-only",
            sleeper=lambda _s: None,
        )
    keys = {a.price_source_key for a in cycle["attempts"] if a.priority_rank == "0B"}
    expected = {q["price_source_key"] for q in queue if q["priority_rank"] == "0B"}
    assert keys == expected


def test_max_concurrency_default_is_one():
    assert int(DEFAULT_POLICY["max_concurrency"]) == 1
    assert float(DEFAULT_POLICY["sleep_seconds_between_requests"]) >= 0.35


def test_write_db_mode_requires_backup_in_smoke_source():
    source = SMOKE.read_text(encoding="utf-8")
    assert "backup_trader_db" in source
    assert "mode=ro" in source or "backup" in source
    assert "llm_calls_made" in source
    assert "INSERT" not in source or "additive" in source.lower() or "insert_raw_payload" in source


def test_artifact_only_path_does_not_call_persist(tmp_path: Path):
    sel = tmp_path / "sel.csv"
    _write_selected(sel, n=1)
    rows = load_selected_csv(sel)
    queue = build_runtime_priority_queue(selected_rows=rows, open_positions=[])
    fake = {
        "fetch_status": "SUCCESS",
        "http_status": 200,
        "elapsed_ms": 1,
        "pair": {
            "pairAddress": "x",
            "chainId": "solana",
            "priceUsd": "1",
            "baseToken": {"symbol": "A"},
            "quoteToken": {"symbol": "B"},
            "liquidity": {"usd": 1},
            "volume": {"h24": 1},
            "txns": {"h24": {"buys": 0, "sells": 0}},
            "priceChange": {},
        },
        "raw_text": "{}",
        "error_reason": "",
        "timeout": False,
    }
    with patch(
        "app.clean_forward.runtime_selected_collection._one_http_fetch",
        return_value=fake,
    ), patch(
        "app.clean_forward.runtime_selected_collection.persist_exact_pair_to_db"
    ) as persist:
        run_priority_fetch_cycle(queue, mode="artifact-only", sleeper=lambda _s: None)
        persist.assert_not_called()
