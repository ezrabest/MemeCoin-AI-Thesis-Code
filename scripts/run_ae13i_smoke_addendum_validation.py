"""AE13I Smoke Addendum validation + audit pack.

Covers: manual close freshness hard guard (Part A), Demo Queue GateKeeper
re-evaluation freshness (Part B), address alias cleanup (Part C), global
text sanitizer (Part D), AE14 readiness (Part E), and gatekeeper/reentry/
stagnant regression (Part F/G).

Paper/demo only. Never starts a live server or wallet. Mirrors the
structure of scripts/run_ae13i_validation.py.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGETED_TEST_FILES = ["tests/test_ae13i_smoke_addendum.py"]
REGRESSION_TEST_FILES = ["tests/test_ae13i_data_trust_gatekeeper.py"]
COMPILEALL_TARGETS = ["app", "scripts", "tests"]

CLASSIFICATION_PASS_WITH_LIMITATIONS = "AE13I_SMOKE_ADDENDUM_PASS_WITH_LIMITATIONS"
CLASSIFICATION_BLOCKED = "AE13I_SMOKE_ADDENDUM_BLOCKED"


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_minutes_ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ---------------------------------------------------------------------------
# Isolated PaperTrader / demo_queue / reentry_blocks environment (never
# touches real data/, never starts a server or listens on a socket).
# ---------------------------------------------------------------------------


def _isolated_trader():
    import os

    tmp = tempfile.TemporaryDirectory()
    data_dir = Path(tmp.name) / "data"
    data_dir.mkdir(parents=True)
    os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

    import app.database as database
    import app.execution.paper as paper

    importlib.reload(paper)
    importlib.reload(database)
    paper.DATA_DIR = data_dir
    paper.STATE_PATH = data_dir / "paper_state.json"
    paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
    database.DATA_DIR = data_dir
    database.DB_PATH = data_dir / "test.db"
    database.init_db()
    trader = paper.PaperTrader()
    return tmp, trader, paper


def _isolated_reentry_blocks():
    import app.ae13b_product.reentry_blocks as reentry_blocks

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name) / "reentry_blocks.json"
    orig = reentry_blocks.blocks_file_path
    reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]
    return tmp, reentry_blocks, orig


def _isolated_demo_queue():
    import app.ae13b_product.demo_queue as demo_queue

    tmp = tempfile.TemporaryDirectory()
    importlib.reload(demo_queue)
    demo_queue.QUEUE_PATH = Path(tmp.name) / "demo_trade_queue.json"
    return tmp, demo_queue


def _fresh_row(**overrides) -> dict:
    row = {
        "chain": "solana",
        "symbol": "SMOKE/SOL",
        "pair_address": "pool_smoke_111",
        "latest_price": 1.0,
        "price_updated_at": _utc_now_iso(),
        "latest_liquidity": 50000.0,
        "liquidity_updated_at": _utc_now_iso(),
        "source_provider": "dexscreener",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Part A audits (1-6): manual close freshness hard guard
# ---------------------------------------------------------------------------


def _audit_1_proposed_price_no_timestamp_never_fresh() -> dict:
    from app.ae13b_product.close_freshness import classify_manual_close_freshness

    result = classify_manual_close_freshness(
        close_price=1.23, price_timestamp="", close_price_source="proposed_price",
        close_price_age_seconds=None,
    )
    ok = (
        result["close_freshness_status"] == "unknown_or_fallback"
        and result["close_used_fallback_price"] is True
        and result["manual_close_warning_shown"] is True
        and result["reason_code"] == "MANUAL_CLOSE_WITH_STALE_OR_FALLBACK_PRICE"
    )
    return {
        "pass": ok,
        "description": (
            "Reproduces the exact smoke-test bug: close_freshness_status=fresh "
            "with proposed_price and empty timestamp MUST be impossible."
        ),
        "result": result,
        "blocker_if_fail": "AE13I_BLOCKED_CLOSE_FRESHNESS_BUG_NOT_FIXED",
    }


def _audit_2_all_fallback_sources_never_fresh() -> dict:
    from app.ae13b_product.close_freshness import (
        FALLBACK_SOURCES,
        classify_manual_close_freshness,
    )

    offenders = []
    for source in FALLBACK_SOURCES:
        result = classify_manual_close_freshness(
            close_price=1.0, price_timestamp=_utc_now_iso(), close_price_source=source,
            close_price_age_seconds=1.0,
        )
        if result["close_freshness_status"] == "fresh":
            offenders.append(source)
    return {
        "pass": len(offenders) == 0,
        "offenders": offenders,
        "fallback_sources_checked": sorted(FALLBACK_SOURCES),
        "blocker_if_fail": "AE13I_BLOCKED_FALLBACK_SOURCE_CLASSIFIED_FRESH",
    }


def _audit_3_genuine_fresh_price_is_fresh() -> dict:
    from app.ae13b_product.close_freshness import classify_manual_close_freshness

    result = classify_manual_close_freshness(
        close_price=1.0, price_timestamp=_utc_now_iso(), close_price_source="provider",
        close_price_age_seconds=5.0,
    )
    ok = result["close_freshness_status"] == "fresh" and result["close_used_fallback_price"] is False
    return {
        "pass": ok,
        "result": result,
        "blocker_if_fail": "AE13I_BLOCKED_FRESH_PRICE_MISCLASSIFIED",
    }


def _audit_4_caller_cannot_spoof_freshness() -> dict:
    tmp, trader, _paper = _isolated_trader()
    try:
        trader.set_market_prices(
            [{"pair_address": "pool_spoof_1", "coin_id": 950, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )
        pos = trader.open_position(
            _fresh_row(symbol="SPOOF/SOL", pair_address="pool_spoof_1", coin_id=950, activity_delta_1h_pct=5.0),
            size_usd=10.0, settings={}, reason_code="AE13I_SMOKE_AUDIT",
        )
        opened_ok = pos is not None
        closed = None
        if opened_ok:
            trader._market_prices_by_pair = {}
            trader._market_prices_by_coin_id = {}
            trader._market_price_timestamp = None
            closed = trader.close_position(
                int(pos["id"]), 1.10, reason_code="MANUAL_SELL",
                proposed_pair_address="pool_spoof_1", proposed_coin_id=950,
                closed_by="user_manual", close_price_source="proposed_price",
                close_freshness_status="fresh", close_used_fallback_price=False,
            )
        ok = bool(
            opened_ok
            and closed is not None
            and closed.get("close_freshness_status") == "unknown_or_fallback"
            and closed.get("close_used_fallback_price") is True
        )
        return {
            "pass": ok,
            "position_opened": opened_ok,
            "closed_freshness_status": closed.get("close_freshness_status") if closed else None,
            "closed_used_fallback_price": closed.get("close_used_fallback_price") if closed else None,
            "blocker_if_fail": "AE13I_BLOCKED_CLOSE_FRESHNESS_HARD_GUARD_BYPASSABLE",
        }
    finally:
        tmp.cleanup()


def _audit_5_paper_close_persists_freshness_fields() -> dict:
    tmp, trader, _paper = _isolated_trader()
    try:
        trader.set_market_prices(
            [{"pair_address": "pool_persist_close_1", "coin_id": 951, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )
        pos = trader.open_position(
            _fresh_row(
                symbol="PERSIST/SOL", pair_address="pool_persist_close_1", coin_id=951,
                activity_delta_1h_pct=5.0,
            ),
            size_usd=10.0, settings={}, reason_code="AE13I_SMOKE_AUDIT",
        )
        opened_ok = pos is not None
        closed = None
        rows = []
        if opened_ok:
            closed = trader.close_position(
                int(pos["id"]), 1.05, reason_code="MANUAL_SELL", closed_by="user_manual",
            )
            rows = trader.get_trades_from_log(limit=10)
        required_fields = (
            "manual_close", "close_price_age_seconds", "close_freshness_status",
            "close_used_fallback_price", "manual_close_warning_shown", "close_price_source",
        )
        closed_has_all = bool(closed) and all(f in closed for f in required_fields)
        sell_rows = [r for r in rows if r.get("side") == "sell"]
        csv_has_source = bool(sell_rows) and bool(sell_rows[-1].get("close_price_source"))
        ok = bool(opened_ok and closed_has_all and csv_has_source)
        return {
            "pass": ok,
            "position_opened": opened_ok,
            "closed_dict_has_all_freshness_fields": closed_has_all,
            "csv_row_has_close_price_source": csv_has_source,
            "blocker_if_fail": "AE13I_BLOCKED_CLOSE_FRESHNESS_NOT_PERSISTED",
        }
    finally:
        tmp.cleanup()


def _audit_6_api_manual_close_response_and_price_source() -> dict:
    from app.ae13b_product.close_freshness import MANUAL_CLOSE_FALLBACK_WARNING
    from app.api import _manual_close_response, _resolve_manual_close_price_source
    from app.execution.paper import get_paper_trader

    price_info = _resolve_manual_close_price_source(
        {"pair_address": "pool_api_smoke_1", "coin_id": 1, "entry_price": 1.0},
        get_paper_trader(),
        2.5,
    )
    explicit_price_labeled_proposed = price_info["close_price_source"] == "proposed_price"

    closed = {
        "id": 999, "closed_by": "user_manual", "close_used_fallback_price": True,
        "close_freshness_status": "unknown_or_fallback",
    }
    payload = _manual_close_response(closed)
    warning_matches = payload.get("warning") == MANUAL_CLOSE_FALLBACK_WARNING

    ok = explicit_price_labeled_proposed and warning_matches
    return {
        "pass": ok,
        "explicit_close_price_labeled_proposed_price": explicit_price_labeled_proposed,
        "warning_text_matches_spec": warning_matches,
        "warning_text": payload.get("warning"),
        "blocker_if_fail": "AE13I_BLOCKED_MANUAL_CLOSE_API_WIRING_INCOMPLETE",
    }


# ---------------------------------------------------------------------------
# Part B audits (7-12): Demo Queue GateKeeper re-evaluation
# ---------------------------------------------------------------------------


def _audit_7_demo_queue_list_attaches_freshness_fields() -> dict:
    tmp, demo_queue = _isolated_demo_queue()
    try:
        demo_queue.add_to_demo_queue(symbol="LISTQ", pair="pool_listq_1", chain="solana")
        items = demo_queue.list_demo_queue()
        required = (
            "last_gatekeeper_evaluated_at", "gatekeeper_status", "tradability_status",
            "freshness_gate_status", "provenance_status", "address_role",
            "market_data_status", "evaluation_stale", "evaluation_stale_reason",
            "gatekeeper_evaluated",
        )
        missing = [f for f in required if not items or f not in items[0]]
        return {
            "pass": bool(items) and not missing,
            "items_count": len(items),
            "missing_fields": missing,
            "blocker_if_fail": "AE13I_BLOCKED_DEMO_QUEUE_FRESHNESS_FIELDS_MISSING",
        }
    finally:
        tmp.cleanup()


def _audit_8_demo_queue_stale_evaluation_flagged() -> dict:
    tmp, demo_queue = _isolated_demo_queue()
    try:
        never = demo_queue.add_to_demo_queue(symbol="NEVERQ", pair="pool_neverq_1", chain="solana")
        never_item = demo_queue.get_queue_item(never["queue_id"])

        old = demo_queue.add_to_demo_queue(symbol="OLDQ", pair="pool_oldq_1", chain="solana")
        demo_queue.update_queue_evaluation(
            old["queue_id"], last_decision="BLOCKED",
            extra={
                "gate_result": {"passed": False},
                "last_gatekeeper_evaluated_at": _iso_minutes_ago(30),
                "gatekeeper_status": "fail", "gatekeeper_evaluated": True,
            },
        )
        old_item = demo_queue.get_queue_item(old["queue_id"])

        fresh = demo_queue.add_to_demo_queue(symbol="FRESHQ", pair="pool_freshq_1", chain="solana")
        demo_queue.update_queue_evaluation(
            fresh["queue_id"], last_decision="BLOCKED",
            extra={
                "gate_result": {"passed": False},
                "last_gatekeeper_evaluated_at": _utc_now_iso(),
                "gatekeeper_status": "fail", "gatekeeper_evaluated": True,
            },
        )
        fresh_item = demo_queue.get_queue_item(fresh["queue_id"])

        ok = bool(
            never_item and never_item.get("evaluation_stale") is True
            and old_item and old_item.get("evaluation_stale") is True
            and fresh_item and fresh_item.get("evaluation_stale") is False
        )
        return {
            "pass": ok,
            "never_evaluated_is_stale": bool(never_item and never_item.get("evaluation_stale")),
            "old_gate_evaluation_is_stale": bool(old_item and old_item.get("evaluation_stale")),
            "recent_gate_evaluation_is_not_stale": bool(
                fresh_item and fresh_item.get("evaluation_stale") is False
            ),
            "blocker_if_fail": "AE13I_BLOCKED_STALE_EVALUATION_NOT_DETECTED",
        }
    finally:
        tmp.cleanup()


def _audit_9_manual_cooldown_precheck_runs_first() -> dict:
    tmp_q, demo_queue = _isolated_demo_queue()
    tmp_r, reentry_blocks, orig_path = _isolated_reentry_blocks()
    try:
        reentry_blocks.add_manual_close_block(
            {"id": 1, "pair_address": "pool_cooldown_smoke_1", "chain": "solana", "symbol": "COOLSMOKE"},
            "manual_take_profit", duration_seconds=3600,
        )
        entry = demo_queue.add_to_demo_queue(symbol="COOLSMOKE", pair="pool_cooldown_smoke_1", chain="solana")
        result = demo_queue.evaluate_queue_item(entry["queue_id"])
        ok = bool(
            result.get("decision") == "BLOCKED_MANUAL_REENTRY_COOLDOWN"
            and result.get("manual_cooldown_active") is True
            and "gate_result" not in result
        )
        return {
            "pass": ok,
            "decision": result.get("decision"),
            "manual_cooldown_active": result.get("manual_cooldown_active"),
            "gate_result_absent_confirming_short_circuit": "gate_result" not in result,
            "blocker_if_fail": "AE13I_BLOCKED_COOLDOWN_PRECHECK_NOT_FIRST",
        }
    finally:
        reentry_blocks.blocks_file_path = orig_path
        tmp_r.cleanup()
        tmp_q.cleanup()


def _audit_10_gate_runs_before_risk_guard() -> dict:
    src = (ROOT / "app" / "ae13b_product" / "demo_queue.py").read_text(encoding="utf-8")
    try:
        gate_idx = src.index("validate_market_data_gate(gate_row")
        risk_idx = src.index("evaluate_demo_risk_guard(")
        ok = gate_idx < risk_idx
    except ValueError:
        ok = False
        gate_idx = risk_idx = -1
    return {
        "pass": ok,
        "gate_call_index": gate_idx,
        "risk_guard_call_index": risk_idx,
        "blocker_if_fail": "AE13I_BLOCKED_RISK_GUARD_RUNS_BEFORE_GATE",
    }


def _audit_11_risk_mode_refreshed_from_active_preset() -> dict:
    from unittest.mock import patch

    tmp_q, demo_queue = _isolated_demo_queue()
    try:
        entry = demo_queue.add_to_demo_queue(symbol="RISKSMOKE", pair="pool_risksmoke_1", chain="solana")
        demo_queue.update_queue_evaluation(
            entry["queue_id"], last_decision="WATCH", extra={"risk_mode": "stale_balanced_snapshot"},
        )
        mock_resolution = {
            "matched_chain": "solana", "matched_symbol": "RISKSMOKE",
            "matched_pair_address": "pool_risksmoke_1", "matched_price": 1.0,
            "matched_price_ts": _utc_now_iso(), "matched_liquidity": 50000.0,
            "resolution_source": "dexscreener", "resolution_status": "matched_live_market",
            "matched_name": "RISKSMOKE", "matched_token_contract_address": None,
            "matched_token_mint_address": None,
        }
        with patch(
            "app.ae13b_product.contract_resolver.resolve_identity", return_value=mock_resolution,
        ):
            result = demo_queue.evaluate_queue_item(entry["queue_id"])
        active_profile = demo_queue.get_active_demo_risk_profile()
        ok = bool(
            "risk_mode" in result
            and result["risk_mode"] == active_profile["risk_mode"]
            and result["risk_mode"] != "stale_balanced_snapshot"
        )
        return {
            "pass": ok,
            "result_risk_mode": result.get("risk_mode"),
            "active_preset_risk_mode": active_profile["risk_mode"],
            "blocker_if_fail": "AE13I_BLOCKED_RISK_MODE_NOT_REFRESHED",
        }
    finally:
        tmp_q.cleanup()


def _audit_12_explicit_risk_mode_not_overridden() -> dict:
    from unittest.mock import patch

    tmp_q, demo_queue = _isolated_demo_queue()
    try:
        entry = demo_queue.add_to_demo_queue(
            symbol="EXPLICITSMOKE", pair="pool_explicitsmoke_1", chain="solana", risk_mode="aggressive",
        )
        inherits_flag_ok = entry.get("inherits_active_bot_preset") is False
        mock_resolution = {
            "matched_chain": "solana", "matched_symbol": "EXPLICITSMOKE",
            "matched_pair_address": "pool_explicitsmoke_1", "matched_price": 1.0,
            "matched_price_ts": _utc_now_iso(), "matched_liquidity": 50000.0,
            "resolution_source": "dexscreener", "resolution_status": "matched_live_market",
            "matched_name": "EXPLICITSMOKE", "matched_token_contract_address": None,
            "matched_token_mint_address": None,
        }
        with patch(
            "app.ae13b_product.contract_resolver.resolve_identity", return_value=mock_resolution,
        ):
            result = demo_queue.evaluate_queue_item(entry["queue_id"])
        ok = bool(inherits_flag_ok and result.get("risk_mode") == "aggressive")
        return {
            "pass": ok,
            "inherits_active_bot_preset_false": inherits_flag_ok,
            "result_risk_mode": result.get("risk_mode"),
            "blocker_if_fail": "AE13I_BLOCKED_EXPLICIT_RISK_MODE_OVERRIDDEN",
        }
    finally:
        tmp_q.cleanup()


# ---------------------------------------------------------------------------
# Part C audits (13-14): address alias cleanup
# ---------------------------------------------------------------------------


def _audit_13_pool_pair_address_disclosure() -> dict:
    from app.ae13b_product.live_market import compute_contract_address_disclosure

    pool = compute_contract_address_disclosure(
        raw_contract_address=None, address_role="pool_address", token_contract_address=None,
        token_mint_address=None, pair_address="poolABC123",
    )
    pair = compute_contract_address_disclosure(
        raw_contract_address=None, address_role="pair_contract", token_contract_address=None,
        token_mint_address=None, pair_address="0xd2391dB4D7B9841b989521088c3Bf8C4cFe404d8",
    )
    ok = (
        pool["contract_address_deprecated"] is True
        and pool["contract_address_role"] == "pool_address_alias"
        and pool["address_display_label"] == "Pool / Pair address"
        and pair["contract_address_role"] == "pair_address_alias"
        and pair["address_display_label"] == "Pool / Pair address"
        and "pool/pair address" in (pool["contract_address_warning"] or "")
    )
    return {
        "pass": ok,
        "pool_address_disclosure": pool,
        "pair_contract_disclosure": pair,
        "blocker_if_fail": "AE13I_BLOCKED_CONTRACT_ADDRESS_ALIAS_UNDISCLOSED",
    }


def _audit_14_actual_token_contract_has_no_warning() -> dict:
    from app.ae13b_product.live_market import compute_contract_address_disclosure

    result = compute_contract_address_disclosure(
        raw_contract_address=None, address_role="token_contract",
        token_contract_address="0xTOKENCONTRACT", token_mint_address=None, pair_address="0xPAIRADDR",
    )
    ok = (
        result["contract_address_role"] is None
        and result["contract_address_warning"] is None
        and result["address_display_label"] == "Contract address"
    )
    return {
        "pass": ok,
        "result": result,
        "blocker_if_fail": "AE13I_BLOCKED_TOKEN_CONTRACT_FALSELY_FLAGGED",
    }


# ---------------------------------------------------------------------------
# Part D audits (15-16): global text sanitizer
# ---------------------------------------------------------------------------


def _audit_15_text_sanitizer_ascii_safe() -> dict:
    from app.ae13b_product.text_sanitizer import sanitize_payload, sanitize_text

    dash_ok = sanitize_text("a \u2014 b \u2013 c") == "a - b - c"
    ellipsis_ok = sanitize_text("loading\u2026") == "loading..."
    payload = sanitize_payload({"note": "paper \u2014 demo", "count": 5, "flag": True, "empty": None})
    payload_ok = payload["note"] == "paper - demo" and payload["count"] == 5 and payload["flag"] is True
    static_files_ok = True
    offenders = []
    for rel_path in ("static/index.html", "static/product_demo.js"):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for bad_char in ("\u2014", "\u2013", "\u2026"):
            if bad_char in text:
                offenders.append(f"{rel_path}:{bad_char!r}")
                static_files_ok = False
    ok = dash_ok and ellipsis_ok and payload_ok and static_files_ok
    return {
        "pass": ok,
        "dash_normalization_ok": dash_ok,
        "ellipsis_normalization_ok": ellipsis_ok,
        "recursive_payload_sanitization_ok": payload_ok,
        "static_ui_files_ascii_safe": static_files_ok,
        "static_file_offenders": offenders,
        "blocker_if_fail": "AE13I_BLOCKED_TEXT_SANITIZER_INCOMPLETE",
    }


def _audit_16_api_responses_sanitized() -> dict:
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    endpoints = [
        "/api/demo-queue", "/api/watchlist", "/api/trades", "/api/ae13b/demo-bot/status",
        "/api/ae13b/live-market", "/api/ae13b/provider-status", "/api/ae14/readiness",
    ]
    offenders = []
    statuses = {}
    for ep in endpoints:
        try:
            resp = client.get(ep)
            statuses[ep] = resp.status_code
            for bad_char in ("\u2014", "\u2013", "\u2026"):
                if bad_char in resp.text:
                    offenders.append(f"{ep}:{bad_char!r}")
        except Exception as exc:  # pragma: no cover - defensive
            statuses[ep] = f"error: {exc!r}"
    ok = len(offenders) == 0
    return {
        "pass": ok,
        "endpoint_statuses": statuses,
        "offenders": offenders,
        "blocker_if_fail": "AE13I_BLOCKED_API_RESPONSES_NOT_SANITIZED",
    }


# ---------------------------------------------------------------------------
# Part E audits (17-18): AE14 readiness
# ---------------------------------------------------------------------------


def _audit_17_ae14_readiness_negative_control_vs_trading_validation() -> dict:
    from app.ae13b_product.ae14_readiness import (
        NEGATIVE_CONTROL_REASON,
        compute_ae14_readiness,
    )

    no_data = compute_ae14_readiness(market_rows=[])
    all_stale = compute_ae14_readiness(
        market_rows=[{"tradability_status": "stale_market_data"}] * 5,
    )
    enough_tradable = compute_ae14_readiness(
        market_rows=[{"tradability_status": "tradable_now"} for _ in range(12)],
        min_tradable_rows_for_ae14=10,
    )
    below_threshold = compute_ae14_readiness(
        market_rows=[{"tradability_status": "tradable_now"} for _ in range(3)],
        min_tradable_rows_for_ae14=10,
    )
    ok = (
        no_data["ready_for_negative_control"] is True
        and no_data["ready_for_trading_validation"] is False
        and all_stale["reason"] == NEGATIVE_CONTROL_REASON
        and enough_tradable["ready_for_trading_validation"] is True
        and below_threshold["ready_for_trading_validation"] is False
        and below_threshold["ready_for_negative_control"] is True
    )
    return {
        "pass": ok,
        "no_data_result": no_data,
        "all_stale_result": all_stale,
        "enough_tradable_result": enough_tradable,
        "below_threshold_result": below_threshold,
        "blocker_if_fail": "AE13I_BLOCKED_AE14_READINESS_LOGIC_WRONG",
    }


def _audit_18_ae14_endpoints_wired() -> dict:
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    readiness_resp = client.get("/api/ae14/readiness")
    status_resp = client.get("/api/ae13b/demo-bot/status")
    readiness_ok = readiness_resp.status_code == 200 and "ready_for_negative_control" in readiness_resp.json()
    status_has_readiness = status_resp.status_code == 200 and "ae14_readiness" in status_resp.json()
    ok = readiness_ok and status_has_readiness
    return {
        "pass": ok,
        "ae14_readiness_endpoint_ok": readiness_ok,
        "demo_bot_status_includes_ae14_readiness": status_has_readiness,
        "blocker_if_fail": "AE13I_BLOCKED_AE14_ENDPOINTS_NOT_WIRED",
    }


# ---------------------------------------------------------------------------
# Part F/G audits (19-21): regression - gatekeeper / reentry / stagnant
# ---------------------------------------------------------------------------


def _audit_19_gatekeeper_still_blocks_stale_and_missing() -> dict:
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

    stale = validate_market_data_gate(
        _fresh_row(price_updated_at=_iso_minutes_ago(60), price_age_seconds=3600.0), for_open=True,
    )
    missing = validate_market_data_gate(_fresh_row(latest_price=None), for_open=True)
    healthy = validate_market_data_gate(
        _fresh_row(activity_delta_1h_pct=5.0, activity_delta_4h_pct=8.0), for_open=True,
    )
    ok = (not stale["passed"]) and (not missing["passed"]) and healthy["passed"]
    return {
        "pass": ok,
        "stale_price_blocked": not stale["passed"],
        "missing_price_blocked": not missing["passed"],
        "healthy_row_passes": healthy["passed"],
        "blocker_if_fail": "AE13I_BLOCKED_GATEKEEPER_WEAKENED",
    }


def _audit_20_reentry_block_still_blocks() -> dict:
    tmp, reentry_blocks, orig_path = _isolated_reentry_blocks()
    try:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        reentry_blocks.add_manual_close_block(
            {"id": 1, "pair_address": "pool_regsmoke_cooldown", "chain": "solana", "symbol": "REGSMOKE"},
            "manual_take_profit", duration_seconds=3600,
        )
        gate = validate_market_data_gate(
            _fresh_row(pair_address="pool_regsmoke_cooldown", symbol="REGSMOKE"), for_open=True,
        )
        ok = (not gate["passed"]) and gate["rejection_code"] == "MANUAL_REENTRY_BLOCK_ACTIVE"
        return {
            "pass": ok,
            "gate_passed": gate["passed"],
            "rejection_code": gate.get("rejection_code"),
            "blocker_if_fail": "AE13I_BLOCKED_REENTRY_GUARD_WEAKENED",
        }
    finally:
        reentry_blocks.blocks_file_path = orig_path
        tmp.cleanup()


def _audit_21_stagnant_guard_still_blocks() -> dict:
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

    stagnant = validate_market_data_gate(_fresh_row(activity_delta_4h_pct=0.05), for_open=True)
    ok = (not stagnant["passed"]) and stagnant.get("rejection_code") == "PRICE_STAGNANT_NO_RECENT_MOMENTUM"
    return {
        "pass": ok,
        "gate_passed": stagnant["passed"],
        "rejection_code": stagnant.get("rejection_code"),
        "blocker_if_fail": "AE13I_BLOCKED_STAGNANT_GUARD_WEAKENED",
    }


def _audit_no_live_wallet_safety(compile_ok: bool) -> dict:
    import re

    offenders = []
    for rel in (
        "app/ae13b_product/close_freshness.py",
        "app/ae13b_product/demo_queue.py",
        "app/ae13b_product/live_market.py",
        "app/ae13b_product/text_sanitizer.py",
        "app/ae13b_product/ae14_readiness.py",
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        if re.search(r"signTransaction|private_key|sendRawTransaction", src, re.I):
            offenders.append(rel)
    ok = compile_ok and not offenders
    return {
        "pass": ok,
        "offenders": offenders,
        "blocker_if_fail": "AE13I_BLOCKED_SAFETY_RISK",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / "audits" / f"ae13i_smoke_addendum_close_queue_address_readiness_{ts}"
    reports = out / "reports"
    data_dir = out / "data"
    audits = out / "audits"
    tests_dir = out / "tests"
    scripts_dir = out / "scripts"
    for d in (reports, data_dir, audits, tests_dir, scripts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. compileall ------------------------------------------------------
    compile_res = _run([sys.executable, "-m", "compileall", "-q", *COMPILEALL_TARGETS])
    _write_text(data_dir / "compileall_output.txt", compile_res["stdout"] + compile_res["stderr"])
    compile_ok = compile_res["returncode"] == 0

    # --- 2. targeted pytest (new smoke-addendum suite) -----------------------
    pytest_res = _run([sys.executable, "-m", "pytest", *TARGETED_TEST_FILES, "-v"])
    _write_text(data_dir / "pytest_smoke_addendum_output.txt", pytest_res["stdout"] + "\n" + pytest_res["stderr"])
    tests_ok = pytest_res["returncode"] == 0

    # --- 3. regression pytest (existing gatekeeper/reentry/stagnant suite) --
    regression_res = _run([sys.executable, "-m", "pytest", *REGRESSION_TEST_FILES, "-v"])
    _write_text(
        data_dir / "pytest_regression_output.txt", regression_res["stdout"] + "\n" + regression_res["stderr"]
    )
    regression_ok = regression_res["returncode"] == 0

    # --- 4. functional audits -------------------------------------------------
    audit_fns = {
        "01_close_freshness_proposed_price_no_timestamp_never_fresh.json": (
            _audit_1_proposed_price_no_timestamp_never_fresh
        ),
        "02_close_freshness_all_fallback_sources_never_fresh.json": _audit_2_all_fallback_sources_never_fresh,
        "03_close_freshness_genuine_fresh_price_is_fresh.json": _audit_3_genuine_fresh_price_is_fresh,
        "04_close_freshness_caller_cannot_spoof.json": _audit_4_caller_cannot_spoof_freshness,
        "05_paper_close_persists_freshness_fields.json": _audit_5_paper_close_persists_freshness_fields,
        "06_api_manual_close_response_and_price_source.json": _audit_6_api_manual_close_response_and_price_source,
        "07_demo_queue_list_attaches_freshness_fields.json": _audit_7_demo_queue_list_attaches_freshness_fields,
        "08_demo_queue_stale_evaluation_flagged.json": _audit_8_demo_queue_stale_evaluation_flagged,
        "09_manual_cooldown_precheck_runs_first.json": _audit_9_manual_cooldown_precheck_runs_first,
        "10_gate_runs_before_risk_guard.json": _audit_10_gate_runs_before_risk_guard,
        "11_risk_mode_refreshed_from_active_preset.json": _audit_11_risk_mode_refreshed_from_active_preset,
        "12_explicit_risk_mode_not_overridden.json": _audit_12_explicit_risk_mode_not_overridden,
        "13_pool_pair_address_disclosure.json": _audit_13_pool_pair_address_disclosure,
        "14_actual_token_contract_has_no_warning.json": _audit_14_actual_token_contract_has_no_warning,
        "15_text_sanitizer_ascii_safe.json": _audit_15_text_sanitizer_ascii_safe,
        "16_api_responses_sanitized.json": _audit_16_api_responses_sanitized,
        "17_ae14_readiness_negative_control_vs_trading_validation.json": (
            _audit_17_ae14_readiness_negative_control_vs_trading_validation
        ),
        "18_ae14_endpoints_wired.json": _audit_18_ae14_endpoints_wired,
        "19_gatekeeper_still_blocks_stale_and_missing.json": _audit_19_gatekeeper_still_blocks_stale_and_missing,
        "20_reentry_block_still_blocks.json": _audit_20_reentry_block_still_blocks,
        "21_stagnant_guard_still_blocks.json": _audit_21_stagnant_guard_still_blocks,
    }
    audit_results: dict[str, dict] = {}
    for filename, fn in audit_fns.items():
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover - defensive
            result = {"pass": False, "error": repr(exc)}
        audit_results[filename] = result
        _write_json(audits / filename, result)

    safety_audit = _audit_no_live_wallet_safety(compile_ok)
    _write_json(audits / "22_no_live_wallet_safety.json", safety_audit)
    audit_results["22_no_live_wallet_safety.json"] = safety_audit

    all_pass = all(bool(r.get("pass")) for r in audit_results.values())

    # --- 5. reference copy of key scripts/tests for this audit pack ---------
    import shutil

    for rel in ("scripts/run_ae13i_smoke_addendum_validation.py",):
        src_path = ROOT / rel
        if src_path.exists():
            shutil.copy2(src_path, scripts_dir / src_path.name)
    for rel in TARGETED_TEST_FILES + REGRESSION_TEST_FILES:
        src_path = ROOT / rel
        if src_path.exists():
            shutil.copy2(src_path, tests_dir / Path(rel).name)

    limitations = [
        "AE14 trading-validation readiness depends on live provider data at "
        "the moment /api/ae14/readiness is called; if the market snapshot "
        "has fewer than min_tradable_rows_for_ae14 (default 10) fresh rows "
        "when this validation runs, ready_for_trading_validation will "
        "legitimately be False even though the code path is correct -- this "
        "reflects real data availability, not a defect.",
        "Demo Queue risk_mode refresh from the active bot preset only "
        "happens once GateKeeper passes (i.e. once the evaluation reaches "
        "the RiskGuard preview stage); a queue item whose pair never "
        "resolves market data will stay gate-BLOCKED and will not have its "
        "risk_mode refreshed until a real market match is found. This is "
        "intentional (GateKeeper-first ordering, per Part B) but means the "
        "risk_mode-refresh audits above use a mocked identity resolution to "
        "reach that code path deterministically without live market data.",
        "The text sanitizer normalizes a fixed, observed set of mojibake "
        "sequences and Unicode punctuation; a not-yet-observed encoding "
        "corruption or unicode symbol outside the tables in text_sanitizer.py "
        "would not be caught until characterized and added.",
        "Address-role disclosure (contract_address_deprecated / "
        "contract_address_role / contract_address_warning) is only as "
        "accurate as the underlying AddressRoleClassifier's chain/format "
        "heuristics; a chain or address format not covered by "
        "SOLANA_CHAINS/EVM_CHAINS in app/ae13b_product/address_role.py may "
        "still resolve to 'unknown_or_provider_pair' rather than a precise "
        "pool/pair role.",
        "This validation run does not start a live HTTP server; API "
        "endpoint checks use FastAPI's in-process TestClient only, so "
        "genuine network-layer behavior (CORS, real concurrency, uvicorn "
        "startup) is not exercised here.",
    ]

    if not compile_ok or not tests_ok or not regression_ok or not all_pass:
        classification = CLASSIFICATION_BLOCKED
        failing = [name for name, r in audit_results.items() if not r.get("pass")]
        if not compile_ok:
            classification = "AE13I_SMOKE_ADDENDUM_BLOCKED_COMPILE_ERROR"
        elif not tests_ok:
            classification = "AE13I_SMOKE_ADDENDUM_BLOCKED_TARGETED_TESTS_FAILING"
        elif not regression_ok:
            classification = "AE13I_SMOKE_ADDENDUM_BLOCKED_REGRESSION_TESTS_FAILING"
        elif failing:
            classification = audit_results[failing[0]].get("blocker_if_fail", CLASSIFICATION_BLOCKED)
    else:
        classification = CLASSIFICATION_PASS_WITH_LIMITATIONS

    decision = {
        "classification": classification,
        "compileall_ok": compile_ok,
        "targeted_tests_ok": tests_ok,
        "targeted_tests_count": pytest_res["stdout"].count(" PASSED") + pytest_res["stdout"].count(" FAILED"),
        "regression_tests_ok": regression_ok,
        "regression_tests_count": (
            regression_res["stdout"].count(" PASSED") + regression_res["stdout"].count(" FAILED")
        ),
        "audits_all_pass": all_pass,
        "audit_results_summary": {name: r.get("pass") for name, r in audit_results.items()},
        "limitations": limitations,
        "timestamp_utc": _utc_now_iso(),
        "output_root": str(out),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
        "no_wallet": True,
        "no_live_trading": True,
        "no_rebuild_or_retrain": True,
    }
    _write_json(reports / "ae13i_smoke_addendum_decision_gate.json", decision)

    # --- Full markdown report -------------------------------------------
    def _status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    sections = []
    sections.append("# AE13I Smoke Addendum -- Validation Report\n")
    sections.append(f"**Classification:** `{classification}`\n")
    sections.append(
        "**Scope:** Paper/demo only. No wallet. No live trading. No rebuild/retrain. "
        "This report never claims profitability and is not evidence of strategy performance.\n"
    )
    sections.append("## 1. Executive Summary\n")
    sections.append(
        "AE13I Smoke Addendum adds a hard freshness guard for manual position "
        "closes (Part A) so a close can never be labeled 'fresh' without a "
        "genuinely fresh, provider-backed price and timestamp; attaches "
        "GateKeeper evaluation-freshness fields to the Demo Trade Queue and "
        "enforces manual-cooldown-then-GateKeeper-then-RiskGuard ordering "
        "(Part B); stops silently aliasing pool/pair addresses as token "
        "contract addresses (Part C); adds a global ASCII-safe text "
        "sanitizer applied to key API responses (Part D); and adds an AE14 "
        "negative-control / trading-validation readiness diagnostic wired "
        "into the API and dashboard (Part E), while leaving the existing "
        "GateKeeper/reentry/stagnant guards intact (Part F/G).\n"
    )
    sections.append("## 2. compileall\n")
    sections.append(f"Result: **{_status(compile_ok)}** (returncode={compile_res['returncode']})\n")
    sections.append("## 3. Targeted pytest (tests/test_ae13i_smoke_addendum.py)\n")
    sections.append(f"Result: **{_status(tests_ok)}** (returncode={pytest_res['returncode']})\n")
    sections.append("## 4. Regression pytest (tests/test_ae13i_data_trust_gatekeeper.py)\n")
    sections.append(f"Result: **{_status(regression_ok)}** (returncode={regression_res['returncode']})\n")
    section_no = 5
    for filename, result in audit_results.items():
        label = filename.replace(".json", "").replace("_", " ")
        sections.append(f"## {section_no}. {label}\n")
        sections.append(f"Result: **{_status(bool(result.get('pass')))}**\n")
        sections.append("```json\n" + json.dumps(result, indent=2, default=str) + "\n```\n")
        section_no += 1
    sections.append(f"## {section_no}. Known Limitations\n")
    for lim in limitations:
        sections.append(f"- {lim}\n")
    section_no += 1
    sections.append(f"## {section_no}. Safety Disclosures\n")
    sections.append(
        "Paper/demo only. No wallet is configured or accessed. No live "
        "trading path exists in any file touched by this pass. No model was "
        "rebuilt or retrained. This report is not evidence of trading "
        "profitability.\n"
    )
    section_no += 1
    sections.append(f"## {section_no}. Output Manifest\n")
    sections.append(f"Output root: `{out}`\n")
    for sub in ("reports", "data", "audits", "tests", "scripts"):
        subdir = out / sub
        files = sorted(p.name for p in subdir.glob("*")) if subdir.exists() else []
        sections.append(f"- `{sub}/`: {', '.join(files) if files else '(empty)'}\n")

    report_md = "\n".join(sections)
    _write_text(reports / "ae13i_smoke_addendum_report.md", report_md)

    _write_text(
        reports / "ae13i_smoke_addendum_summary_for_upload.txt",
        f"AE13I Smoke Addendum {classification}\n"
        "Manual close freshness hard guard + Demo Queue GateKeeper "
        "re-evaluation freshness + address alias cleanup + global text "
        "sanitizer + AE14 readiness. Gatekeeper/reentry/stagnant guards "
        "not weakened.\n"
        f"compileall={'ok' if compile_ok else 'fail'} "
        f"smoke_addendum_tests={'ok' if tests_ok else 'fail'} "
        f"regression_tests={'ok' if regression_ok else 'fail'} "
        f"audits_all_pass={all_pass}\n"
        "Paper/demo only. No wallet. No live trading. Not profitability evidence.\n"
        f"output={out}\n",
    )
    _write_text(
        tests_dir / "ae13i_smoke_addendum_test_results.md",
        "# AE13I Smoke Addendum test results\n\n"
        f"- compileall rc={compile_res['returncode']}\n"
        f"- smoke addendum pytest rc={pytest_res['returncode']}\n"
        f"- regression pytest rc={regression_res['returncode']}\n\n"
        f"## smoke addendum pytest stdout (tail)\n\n```\n{pytest_res['stdout'][-8000:]}\n```\n\n"
        f"## regression pytest stdout (tail)\n\n```\n{regression_res['stdout'][-4000:]}\n```\n",
    )

    print(json.dumps(decision, indent=2, default=str))
    return 0 if classification == CLASSIFICATION_PASS_WITH_LIMITATIONS else 1


if __name__ == "__main__":
    raise SystemExit(main())
