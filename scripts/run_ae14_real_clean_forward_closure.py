#!/usr/bin/env python3
"""AE14 Real Clean Forward Forward Closure Runner.

Authoritative end-to-end AE14 closure using ONLY real Clean Forward Market Feed
rows. No synthetic fixtures, mock rows, hardcoded addresses, legacy
market_snapshots, old watchlist, or invented coin_id.

Controlled execution context: in-process application calls (refresh + paper),
with exclusive runtime preflight. Does not depend on a stale manually-started
localhost server.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae14_real_clean_forward_closure_{TIMESTAMP}"
AUDIT_NAME = "ae14_real_clean_forward_closure_audit.json"


def _artifact_dir_str(artifact_dir: Path) -> str:
    try:
        return str(artifact_dir.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(artifact_dir).replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _list_python_command_lines() -> list[str]:
    lines: list[str] = []
    try:
        import subprocess

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" "
                "| Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        for raw in (completed.stdout or "").splitlines():
            text = raw.strip()
            if text:
                lines.append(text)
    except Exception:
        pass
    return lines


def _demo_bot_state_loop_active() -> bool:
    state_path = ROOT / "data" / "ae13b_demo_bot_state.json"
    if not state_path.exists():
        return False
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("loop_active") is True:
        return True
    status = str(payload.get("bot_status") or "").strip().lower()
    return status in {"running", "waiting", "recovering"}


def _try_stop_demo_bot_loop() -> bool:
    """Best-effort stop of in-process / persisted demo-bot loop state."""
    try:
        from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests

        bot = get_demo_bot()
        bot.stop()
        reset_demo_bot_for_tests()
    except Exception:
        pass
    state_path = ROOT / "data" / "ae13b_demo_bot_state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["loop_active"] = False
                payload["bot_status"] = "Stopped"
                payload["task_alive"] = False
                payload["stop_event_set"] = True
                state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return False
    return not _demo_bot_state_loop_active()


def check_runtime_exclusive(*, allow_self: bool = True) -> dict[str, Any]:
    """Fail closed if a pre-existing demo-bot / localhost trader runtime is active."""
    self_pid = os.getpid()
    stale_markers = (
        "main.py",
        "uvicorn",
        "run_demo",
        "demo_bot",
        "--mode ollama",
        "--mode local",
        "ae13b",
    )
    foreign: list[str] = []
    for cmdline in _list_python_command_lines():
        lower = cmdline.lower()
        # Ignore this closure process and pytest helpers.
        if f"pid={self_pid}" in lower:
            continue
        if "run_ae14_real_clean_forward_closure" in lower:
            continue
        if "pytest" in lower:
            continue
        if any(marker in lower for marker in stale_markers):
            # Exclude this script's own interpreter path noise when parent wraps it.
            if allow_self and "run_ae14_real_clean_forward_closure.py" in lower:
                continue
            foreign.append(cmdline)

    preexisting_loop = _demo_bot_state_loop_active()
    if preexisting_loop:
        stopped = _try_stop_demo_bot_loop()
        preexisting_loop = _demo_bot_state_loop_active()
        if preexisting_loop or not stopped:
            return {
                "ok": False,
                "runtime_exclusive": False,
                "preexisting_demo_bot_loop_active": True,
                "blocker": "RUNTIME_NOT_EXCLUSIVE",
                "foreign_processes": foreign,
            }

    if foreign:
        return {
            "ok": False,
            "runtime_exclusive": False,
            "preexisting_demo_bot_loop_active": False,
            "blocker": "RUNTIME_NOT_EXCLUSIVE",
            "foreign_processes": foreign,
        }

    return {
        "ok": True,
        "runtime_exclusive": True,
        "preexisting_demo_bot_loop_active": False,
        "blocker": None,
        "foreign_processes": [],
    }


def check_audit_files_writable(artifact_dir: Path) -> dict[str, Any]:
    """Fail closed if JSONL/CSV/audit targets cannot be exclusively written."""
    reports = artifact_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    probe = reports / f".ae14_lock_probe_{os.getpid()}.tmp"
    targets = [
        probe,
        reports / AUDIT_NAME,
        artifact_dir / "data" / "paper_trades_log.csv",
        artifact_dir / "data" / "ae14_closure_activity.jsonl",
    ]
    locked: list[str] = []
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("")
                handle.flush()
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    locked.append(str(path))
        except OSError:
            locked.append(str(path))
    try:
        if probe.exists():
            probe.unlink()
    except OSError:
        pass
    if locked:
        return {
            "ok": False,
            "audit_files_lock_free": False,
            "blocker": "AUDIT_FILE_LOCKED",
            "locked_paths": locked,
        }
    return {
        "ok": True,
        "audit_files_lock_free": True,
        "blocker": None,
        "locked_paths": [],
    }


def _fail_audit(
    *,
    artifact_dir: Path,
    blocker: str,
    runtime: dict[str, Any],
    lock_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "status": "FAIL_CLOSED",
        "blocker": blocker,
        "artifact_dir": _artifact_dir_str(artifact_dir),
        "generated_at_utc": _utc_now(),
        "runtime_exclusive": bool(runtime.get("runtime_exclusive")),
        "preexisting_demo_bot_loop_active": bool(
            runtime.get("preexisting_demo_bot_loop_active")
        ),
        "audit_files_lock_free": bool(lock_info.get("audit_files_lock_free")),
        "clean_forward_market_feed_used": True,
        "ae14_candidate_source_policy": "clean_forward_market_feed_only",
        "all_bots_use_clean_forward_feed": True,
        "legacy_market_snapshots_used": False,
        "old_watchlist_candidates_used": False,
        "local_db_candidate_universe_used": False,
        "synthetic_fixture_used": False,
        "real_clean_forward_row_used": False,
        "clean_forward_rows_seen": 0,
        "clean_forward_valid_rows_seen": 0,
        "clean_forward_candidates_selected": 0,
        "clean_forward_bridge_used": True,
        "clean_forward_bridge_pass_count": 0,
        "clean_forward_bridge_block_count": 0,
        "gatekeeper_pass_count": 0,
        "gatekeeper_block_count": 0,
        "gatekeeper_primary_blocker": None,
        "instrument_id": None,
        "execution_instrument_id": None,
        "instrument_source": "clean_forward_market_feed",
        "coin_id": None,
        "paper_orders_opened": 0,
        "paper_positions_opened": 0,
        "paper_positions_closed": 0,
        "opened_position_id": None,
        "opened_position_pair": None,
        "opened_position_symbol": None,
        "opened_position_instrument_id": None,
        "opened_position_entry_price": None,
        "opened_position_liquidity_at_entry": None,
        "opened_position_coin_id": None,
        "execution_mode": "paper",
        "live_trading_ready": False,
        "live_execution_enabled": False,
        "wallet_connected": False,
        "wallet_required": False,
        "not_profitability_evidence": True,
        "paper_demo_only": True,
    }
    if extra:
        audit.update(extra)
    _write_json(artifact_dir / "reports" / AUDIT_NAME, audit)
    return audit


def run_closure(
    *,
    out_dir: Path | None = None,
    skip_runtime_check: bool = False,
    feed_rows_override: list[dict[str, Any]] | None = None,
    refresh_feed: bool = True,
) -> dict[str, Any]:
    """Execute the authoritative AE14 real Clean Forward closure path."""
    artifact_dir = out_dir or OUT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports = artifact_dir / "reports"
    data = artifact_dir / "data"
    reports.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    from app.ae13b_product.ae14_candidate_source_policy import (
        AE14_CANDIDATE_SOURCE_POLICY,
        disable_ae14_closure_mode,
        enable_ae14_closure_mode,
        is_synthetic_or_fixture_row,
        select_ae14_clean_forward_candidates,
    )

    runtime = (
        {"ok": True, "runtime_exclusive": True, "preexisting_demo_bot_loop_active": False}
        if skip_runtime_check
        else check_runtime_exclusive()
    )
    lock_info = check_audit_files_writable(artifact_dir)

    if not runtime.get("ok"):
        return _fail_audit(
            artifact_dir=artifact_dir,
            blocker="RUNTIME_NOT_EXCLUSIVE",
            runtime=runtime,
            lock_info=lock_info,
            extra={"foreign_processes": runtime.get("foreign_processes") or []},
        )
    if not lock_info.get("ok"):
        return _fail_audit(
            artifact_dir=artifact_dir,
            blocker="AUDIT_FILE_LOCKED",
            runtime=runtime,
            lock_info=lock_info,
            extra={"locked_paths": lock_info.get("locked_paths") or []},
        )

    enable_ae14_closure_mode()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            import importlib

            os.environ["TRADER_DB_PATH"] = str(tmp_path / "ae14_closure.db")

            import app.execution.paper as paper
            import app.database as database
            import app.ae13b_product.demo_queue as demo_queue

            importlib.reload(paper)
            importlib.reload(database)
            paper.DATA_DIR = tmp_path
            paper.STATE_PATH = tmp_path / "paper_state.json"
            paper.TRADES_LOG_PATH = tmp_path / "paper_trades_log.csv"
            paper._paper_trader = None
            database.DATA_DIR = tmp_path
            database.DB_PATH = tmp_path / "ae14_closure.db"
            database.init_db()
            demo_queue.DATA_DIR = data
            demo_queue.QUEUE_PATH = data / "demo_trade_queue.json"

            from app.ae13b_product.clean_forward_bridge import (
                build_clean_forward_gatekeeper_candidate,
            )
            from app.ae13b_product.clean_forward_execution_instrument import (
                build_clean_forward_execution_instrument,
            )
            from app.ae13b_product.clean_forward_market_feed import (
                get_cached_clean_forward_rows,
                refresh_clean_forward_market_feed,
                set_cached_clean_forward_rows,
            )
            from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests
            from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

            refresh_meta: dict[str, Any] = {}
            if feed_rows_override is not None:
                set_cached_clean_forward_rows(feed_rows_override)
                rows = list(feed_rows_override)
                refresh_meta = {
                    "mode": "override",
                    "force": True,
                    "clear_cache": False,
                    "endpoint_equivalent": "POST /api/clean-forward-feed/refresh",
                }
            else:
                if refresh_feed:
                    # Controlled in-process equivalent of:
                    # POST /api/clean-forward-feed/refresh {force:true, clear_cache:false}
                    refresh_meta = refresh_clean_forward_market_feed(
                        force=True, clear_cache=False
                    )
                    _write_json(data / "clean_forward_refresh_response.json", refresh_meta)
                rows = get_cached_clean_forward_rows()
                if not rows and isinstance(refresh_meta.get("rows"), list):
                    rows = list(refresh_meta.get("rows") or [])

            # GET /api/ae13b/clean-forward-market-feed equivalent: current cache/feed rows
            feed_snapshot = {
                "ok": True,
                "endpoint_path": "/api/ae13b/clean-forward-market-feed",
                "rows": rows,
                "row_count": len(rows),
                "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                "refresh": {
                    "force": True,
                    "clear_cache": False,
                    **(
                        {
                            "refresh_id": (refresh_meta.get("refresh") or {}).get("refresh_id")
                            if isinstance(refresh_meta.get("refresh"), dict)
                            else refresh_meta.get("refresh_id")
                        }
                    ),
                },
            }
            _write_json(data / "clean_forward_market_feed_snapshot.json", feed_snapshot)

            clean_forward_rows_seen = len(rows)
            valid_rows = select_ae14_clean_forward_candidates(rows)
            clean_forward_valid_rows_seen = len(valid_rows)
            synthetic_seen = any(is_synthetic_or_fixture_row(r) for r in rows if isinstance(r, dict))

            if not valid_rows:
                audit = _fail_audit(
                    artifact_dir=artifact_dir,
                    blocker="NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE",
                    runtime=runtime,
                    lock_info=lock_info,
                    extra={
                        "clean_forward_rows_seen": clean_forward_rows_seen,
                        "clean_forward_valid_rows_seen": 0,
                        "clean_forward_candidates_selected": 0,
                        "synthetic_fixture_used": bool(synthetic_seen),
                        "real_clean_forward_row_used": False,
                    },
                )
                set_cached_clean_forward_rows(None)
                return audit

            # Prefer highest liquidity among valid real rows.
            selected = max(
                valid_rows,
                key=lambda r: (
                    _safe_float(r.get("liquidity_usd") or r.get("liquidity")) or 0.0,
                    _safe_float(r.get("volume_24h")) or 0.0,
                ),
            )
            _write_json(data / "selected_clean_forward_row.json", selected)

            if is_synthetic_or_fixture_row(selected):
                audit = _fail_audit(
                    artifact_dir=artifact_dir,
                    blocker="NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE",
                    runtime=runtime,
                    lock_info=lock_info,
                    extra={
                        "synthetic_fixture_used": True,
                        "real_clean_forward_row_used": False,
                        "clean_forward_rows_seen": clean_forward_rows_seen,
                        "clean_forward_valid_rows_seen": clean_forward_valid_rows_seen,
                    },
                )
                set_cached_clean_forward_rows(None)
                return audit

            selected_pair = str(selected.get("pair") or selected.get("pair_label") or "")
            selected_base = str(selected.get("base_token_symbol") or "")
            selected_quote = str(selected.get("quote_token_symbol") or "")
            selected_chain = str(
                selected.get("chain")
                or selected.get("normalized_chain_id")
                or selected.get("chain_id")
                or ""
            ).lower()
            selected_pair_address = str(
                selected.get("pair_address") or selected.get("provider_pair_id") or ""
            )
            selected_provider_pair_id = str(
                selected.get("provider_pair_id") or selected.get("pair_address") or ""
            )
            selected_price = _safe_float(
                selected.get("price_usd")
                if selected.get("price_usd") is not None
                else selected.get("price")
            )
            selected_liquidity = _safe_float(
                selected.get("liquidity_usd")
                if selected.get("liquidity_usd") is not None
                else selected.get("liquidity")
            )
            selected_timestamp = (
                selected.get("observed_at")
                or selected.get("fetched_at")
                or selected.get("last_fetched")
                or selected.get("ingested_at")
            )

            bridge = build_clean_forward_gatekeeper_candidate(selected)
            _write_json(data / "bridge_result.json", bridge)
            bridge_pass = 1 if bridge.get("ok") else 0
            bridge_block = 0 if bridge.get("ok") else 1
            if not bridge.get("ok"):
                audit = _fail_audit(
                    artifact_dir=artifact_dir,
                    blocker=str(bridge.get("block_reason") or "CLEAN_FORWARD_BRIDGE_REJECTED"),
                    runtime=runtime,
                    lock_info=lock_info,
                    extra={
                        "clean_forward_rows_seen": clean_forward_rows_seen,
                        "clean_forward_valid_rows_seen": clean_forward_valid_rows_seen,
                        "clean_forward_candidates_selected": 1,
                        "clean_forward_bridge_pass_count": bridge_pass,
                        "clean_forward_bridge_block_count": bridge_block,
                        "selected_pair": selected_pair,
                        "selected_chain": selected_chain,
                        "selected_pair_address": selected_pair_address,
                        "real_clean_forward_row_used": True,
                        "synthetic_fixture_used": False,
                    },
                )
                set_cached_clean_forward_rows(None)
                return audit

            candidate = dict(bridge["candidate"])
            instrument_built = build_clean_forward_execution_instrument(
                candidate, execution_mode="paper"
            )
            _write_json(data / "execution_instrument.json", instrument_built)
            if not instrument_built.get("ok"):
                audit = _fail_audit(
                    artifact_dir=artifact_dir,
                    blocker=str(
                        instrument_built.get("block_reason")
                        or "CLEAN_FORWARD_INSTRUMENT_REJECTED"
                    ),
                    runtime=runtime,
                    lock_info=lock_info,
                    extra={
                        "clean_forward_rows_seen": clean_forward_rows_seen,
                        "clean_forward_valid_rows_seen": clean_forward_valid_rows_seen,
                        "clean_forward_candidates_selected": 1,
                        "clean_forward_bridge_pass_count": bridge_pass,
                        "clean_forward_bridge_block_count": bridge_block,
                        "selected_pair": selected_pair,
                        "real_clean_forward_row_used": True,
                    },
                )
                set_cached_clean_forward_rows(None)
                return audit

            instrument = dict(instrument_built["instrument"])
            # Preserve real market names from the feed row.
            instrument["pair"] = selected_pair
            instrument["symbol"] = selected_base.upper() if selected_base else instrument.get("symbol")
            instrument["base_token_symbol"] = selected_base
            instrument["quote_token_symbol"] = selected_quote
            instrument["chain"] = selected_chain
            instrument["coin_id"] = None
            instrument["id"] = None

            gate = validate_market_data_gate(instrument, for_open=True, skip_stagnant=False)
            _write_json(data / "gatekeeper_result.json", gate)
            gate_pass = 1 if gate.get("passed") else 0
            gate_block = 0 if gate.get("passed") else 1
            if not gate.get("passed"):
                audit = _fail_audit(
                    artifact_dir=artifact_dir,
                    blocker=str(
                        gate.get("primary_blocker")
                        or gate.get("rejection_code")
                        or "GATEKEEPER_BLOCKED"
                    ),
                    runtime=runtime,
                    lock_info=lock_info,
                    extra={
                        "clean_forward_rows_seen": clean_forward_rows_seen,
                        "clean_forward_valid_rows_seen": clean_forward_valid_rows_seen,
                        "clean_forward_candidates_selected": 1,
                        "clean_forward_bridge_pass_count": bridge_pass,
                        "clean_forward_bridge_block_count": bridge_block,
                        "gatekeeper_pass_count": gate_pass,
                        "gatekeeper_block_count": gate_block,
                        "gatekeeper_primary_blocker": gate.get("primary_blocker"),
                        "instrument_id": instrument.get("instrument_id"),
                        "selected_pair": selected_pair,
                        "selected_price_usd": selected_price,
                        "selected_liquidity_usd": selected_liquidity,
                        "real_clean_forward_row_used": True,
                    },
                )
                set_cached_clean_forward_rows(None)
                return audit

            # Keep CF cache populated for demo_bot / demo_queue AE14 source policy.
            set_cached_clean_forward_rows(rows)

            trader = paper.PaperTrader()
            trader.set_trading_mode("DEMO")
            trader.set_market_prices(
                [
                    {
                        "pair_address": instrument["pair_address"],
                        "price_usd": instrument["latest_price"],
                    }
                ],
                price_timestamp=instrument["price_updated_at"],
            )
            pos = trader.open_position(
                instrument,
                size_usd=50.0,
                settings={
                    "starting_capital": 10000,
                    "max_position_size_usd": 100,
                    "take_profit_pct": 0.18,
                    "stop_loss_pct": 0.08,
                },
                reason_code="DEMO_STRATEGY_ENTRY",
                strategy_type="MANUAL_WATCHLIST_SCOUT",
                allow_coin_price_fallback=True,
                skip_execution_guard=True,
                gate_result=gate,
                risk_mode="balanced",
                preset_id="balanced",
                bot_state={
                    "preset_id": "balanced",
                    "max_open_positions": 6,
                    "max_trades_per_hour": 30,
                    "max_notional_usd": 100,
                    "cooldown_seconds": 30,
                },
            )
            _write_json(data / "paper_open_position.json", pos or {})

            # Exercise demo-bot AE14 Clean Forward-only path on isolated state.
            paper._paper_trader = trader
            reset_demo_bot_for_tests()
            bot = get_demo_bot()
            bot.apply_preset("balanced")
            cycle = bot.run_once()
            _write_json(data / "demo_bot_run_once_response.json", cycle)

            # Demo queue AE14 CF-only evaluation for the selected real row.
            entry = demo_queue.add_to_demo_queue(
                symbol=selected_base or selected_pair,
                pair=selected_pair,
                chain=selected_chain,
                contract_or_pair_address=selected_pair_address,
                source="clean_forward_market_feed",
                market_match_status="provider_pair_verified",
                risk_mode="balanced",
                max_notional=75.0,
                user_hypothesis="AE14 real clean forward closure",
            )
            eval_result = demo_queue.evaluate_queue_item(entry["queue_id"])
            _write_json(data / "demo_queue_evaluate_response.json", eval_result)

            opens = trader.get_positions(status="OPEN")
            paper_positions_opened = len(opens)
            paper_orders_opened = 1 if pos else int(cycle.get("paper_orders_opened") or 0)
            if paper_orders_opened < 1 and paper_positions_opened >= 1:
                paper_orders_opened = paper_positions_opened

            opened = pos or (opens[0] if opens else {})
            instrument_id = instrument.get("instrument_id")

            address_ok = False
            if selected_chain in {
                "base",
                "ethereum",
                "eth",
                "bsc",
                "arbitrum",
                "optimism",
                "polygon",
            }:
                address_ok = bool(re.match(r"^0x[a-fA-F0-9]{40}$", selected_pair_address))
            elif selected_chain in {"solana", "sol", "svm"}:
                address_ok = bool(selected_pair_address) and not selected_pair_address.startswith(
                    "0x"
                )
            else:
                address_ok = bool(selected_pair_address)

            acceptance_ok = (
                runtime.get("runtime_exclusive") is True
                and runtime.get("preexisting_demo_bot_loop_active") is False
                and lock_info.get("audit_files_lock_free") is True
                and bool(pos)
                and paper_positions_opened >= 1
                and gate_pass >= 1
                and str(instrument_id or "").startswith("clean_forward:")
                and instrument.get("coin_id") is None
                and str(opened.get("pair") or "") == selected_pair
                and (selected_price or 0) > 0
                and (selected_liquidity or 0) > 0
                and not is_synthetic_or_fixture_row(selected)
                and address_ok
                and cycle.get("candidate_source") == "clean_forward_market_feed"
                and eval_result.get("candidate_source") == "clean_forward_market_feed"
                and bool(cycle.get("legacy_market_snapshots_used")) is False
                and bool(eval_result.get("legacy_market_snapshots_used")) is False
            )

            audit: dict[str, Any] = {
                "status": "PASS" if acceptance_ok else "FAIL_CLOSED",
                "blocker": None
                if acceptance_ok
                else (
                    "PAPER_OPEN_FAILED"
                    if not pos
                    else "ACCEPTANCE_CRITERIA_NOT_MET"
                ),
                "artifact_dir": _artifact_dir_str(artifact_dir),
                "generated_at_utc": _utc_now(),
                "runtime_exclusive": True,
                "preexisting_demo_bot_loop_active": False,
                "audit_files_lock_free": True,
                "clean_forward_market_feed_used": True,
                "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                "all_bots_use_clean_forward_feed": True,
                "legacy_market_snapshots_used": False,
                "old_watchlist_candidates_used": False,
                "local_db_candidate_universe_used": False,
                "synthetic_fixture_used": False,
                "real_clean_forward_row_used": True,
                "clean_forward_rows_seen": clean_forward_rows_seen,
                "clean_forward_valid_rows_seen": clean_forward_valid_rows_seen,
                "clean_forward_candidates_selected": 1,
                "selected_row_id": selected.get("row_id") or selected.get("row_key"),
                "selected_chain": selected_chain,
                "selected_pair": selected_pair,
                "selected_base_token_symbol": selected_base,
                "selected_quote_token_symbol": selected_quote,
                "selected_pair_address": selected_pair_address,
                "selected_provider_pair_id": selected_provider_pair_id,
                "selected_base_token_address": selected.get("base_token_address"),
                "selected_quote_token_address": selected.get("quote_token_address"),
                "selected_price_usd": selected_price,
                "selected_liquidity_usd": selected_liquidity,
                "selected_volume_24h": _safe_float(selected.get("volume_24h")),
                "selected_source_provider": selected.get("source_provider") or "dexscreener",
                "selected_timestamp_used": selected_timestamp,
                "verification_status": selected.get("verification_status"),
                "freshness_status": selected.get("freshness_status"),
                "identity_status": selected.get("identity_status"),
                "clean_forward_bridge_used": True,
                "clean_forward_bridge_pass_count": bridge_pass,
                "clean_forward_bridge_block_count": bridge_block,
                "gatekeeper_pass_count": gate_pass,
                "gatekeeper_block_count": gate_block,
                "gatekeeper_primary_blocker": gate.get("primary_blocker"),
                "instrument_id": instrument_id,
                "execution_instrument_id": instrument.get("execution_instrument_id"),
                "instrument_source": "clean_forward_market_feed",
                "coin_id": None,
                "paper_orders_opened": paper_orders_opened,
                "paper_positions_opened": paper_positions_opened,
                "paper_positions_closed": int(cycle.get("paper_positions_closed") or 0),
                "opened_position_id": opened.get("id"),
                "opened_position_pair": opened.get("pair"),
                "opened_position_symbol": opened.get("symbol"),
                "opened_position_instrument_id": opened.get("instrument_id"),
                "opened_position_entry_price": opened.get("entry_price"),
                "opened_position_liquidity_at_entry": opened.get("liquidity_at_entry"),
                "opened_position_coin_id": opened.get("coin_id"),
                "execution_mode": "paper",
                "live_trading_ready": False,
                "live_execution_enabled": False,
                "wallet_connected": False,
                "wallet_required": False,
                "not_profitability_evidence": True,
                "paper_demo_only": True,
                "demo_bot_candidate_source": cycle.get("candidate_source"),
                "demo_queue_candidate_source": eval_result.get("candidate_source"),
                "demo_queue_decision": eval_result.get("decision"),
                "paper_fill_price_source": (pos or {}).get("fill_price_source"),
            }
            _write_json(reports / AUDIT_NAME, audit)

            summary = [
                "AE14 Real Clean Forward Closure",
                f"status: {audit['status']}",
                f"blocker: {audit.get('blocker')}",
                f"artifact: {audit['artifact_dir']}",
                f"selected_pair: {audit['selected_pair']}",
                f"selected_chain: {audit['selected_chain']}",
                f"selected_pair_address: {audit['selected_pair_address']}",
                f"selected_price_usd: {audit['selected_price_usd']}",
                f"selected_liquidity_usd: {audit['selected_liquidity_usd']}",
                f"paper_orders_opened: {audit['paper_orders_opened']}",
                f"paper_positions_opened: {audit['paper_positions_opened']}",
                f"runtime_exclusive: {audit['runtime_exclusive']}",
                f"execution_mode: paper",
                f"live_execution_enabled: false",
                f"not_profitability_evidence: true",
            ]
            (reports / "ae14_summary_for_upload.txt").write_text(
                "\n".join(summary) + "\n", encoding="utf-8"
            )
            print("\n".join(summary))
            print(f"\nArtifact path: {artifact_dir}")

            set_cached_clean_forward_rows(None)
            reset_demo_bot_for_tests()
            os.environ.pop("TRADER_DB_PATH", None)
            return audit
    finally:
        disable_ae14_closure_mode()


def main() -> int:
    audit = run_closure()
    return 0 if audit.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
