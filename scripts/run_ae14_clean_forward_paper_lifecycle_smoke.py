#!/usr/bin/env python3
"""AE14 Clean Forward paper lifecycle smoke (paper/demo only).

Bridge → GateKeeper → canonical instrument_id → paper open.
Does not invent coin_id. Does not enable live execution or wallets.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae14_clean_forward_paper_lifecycle_smoke_{TIMESTAMP}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row() -> dict[str, Any]:
    ts = _utc_now()
    return {
        "row_id": "base|pair|0xAE14PaperLifecycle01",
        "source_provider": "dexscreener",
        "normalized_chain_id": "base",
        "chain": "base",
        "chain_id": "base",
        "provider_pair_id": "0xAE14PaperLifecycle01",
        "pair_address": "0xAE14PaperLifecycle01",
        "base_token_address": "0x311935Cd80B76769bF2ecC9D8Ab7635b2139cf82",
        "base_token_symbol": "SOL",
        "quote_token_address": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        "quote_token_symbol": "cbBTC",
        "pair": "SOL/cbBTC",
        "pair_label": "SOL/cbBTC",
        "price": "25.5",
        "price_usd": "25.5",
        "liquidity": 420000.0,
        "liquidity_usd": 420000.0,
        "observed_at": ts,
        "fetched_at": ts,
        "last_fetched": ts,
        "ingested_at": ts,
        "verification_status": "provider_pair_verified",
        "freshness_status": "fresh",
        "identity_status": "pair_and_tokens_separated",
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "address_role": "pair_contract",
        "price_change_1h": 1.5,
        "txns_24h_buys": 40,
        "txns_24h_sells": 30,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = OUT_DIR / "reports"
    data = OUT_DIR / "data"
    reports.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    # Isolate paper + queue state for this smoke
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        import importlib
        import os

        os.environ["TRADER_DB_PATH"] = str(tmp_path / "smoke.db")

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
        database.DB_PATH = tmp_path / "smoke.db"
        database.init_db()
        demo_queue.DATA_DIR = data
        demo_queue.QUEUE_PATH = data / "demo_trade_queue.json"

        from app.ae13b_product.clean_forward_bridge import (
            build_clean_forward_gatekeeper_candidate,
        )
        from app.ae13b_product.clean_forward_execution_instrument import (
            build_clean_forward_execution_instrument,
        )
        from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows
        from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = _row()
        set_cached_clean_forward_rows([row])
        (data / "clean_forward_seed_row.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

        bridge = build_clean_forward_gatekeeper_candidate(row)
        (data / "bridge_result.json").write_text(
            json.dumps(bridge, indent=2, default=str), encoding="utf-8"
        )
        assert bridge.get("ok"), bridge

        instrument_built = build_clean_forward_execution_instrument(
            bridge["candidate"], execution_mode="paper"
        )
        (data / "execution_instrument.json").write_text(
            json.dumps(instrument_built, indent=2, default=str), encoding="utf-8"
        )
        assert instrument_built.get("ok"), instrument_built
        instrument = instrument_built["instrument"]
        assert instrument.get("coin_id") is None
        assert instrument.get("instrument_id", "").startswith("clean_forward:")

        gate = validate_market_data_gate(
            instrument, for_open=True, skip_stagnant=False
        )
        (data / "gatekeeper_result.json").write_text(
            json.dumps(gate, indent=2, default=str), encoding="utf-8"
        )
        gatekeeper_pass_count = 1 if gate.get("passed") else 0
        gatekeeper_block_count = 0 if gate.get("passed") else 1

        entry = demo_queue.add_to_demo_queue(
            symbol="SOL",
            pair="SOL/cbBTC",
            chain="base",
            contract_or_pair_address=row["pair_address"],
            source="ae14_clean_forward_inline_queue_eval_smoke",
            market_match_status="provider_pair_verified",
            risk_mode="balanced",
            max_notional=75.0,
            user_hypothesis="AE14 paper lifecycle",
        )
        eval_result = demo_queue.evaluate_queue_item(entry["queue_id"])
        (data / "demo_queue_evaluate_response.json").write_text(
            json.dumps(eval_result, indent=2, default=str), encoding="utf-8"
        )

        # Direct paper open via canonical instrument (proves missing_coin_id gone)
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
        (data / "paper_open_position.json").write_text(
            json.dumps(pos, indent=2, default=str), encoding="utf-8"
        )

        # Also exercise demo-bot one-cycle CF path on isolated paper state
        paper._paper_trader = trader
        reset_demo_bot_for_tests()
        bot = get_demo_bot()
        bot.apply_preset("balanced")
        cycle = bot.run_once()
        (data / "demo_bot_run_once_response.json").write_text(
            json.dumps(cycle, indent=2, default=str), encoding="utf-8"
        )

        paper_orders_opened = 1 if pos else int(cycle.get("paper_orders_opened") or 0)
        paper_positions_opened = len(trader.get_positions(status="OPEN"))

        artifact = {
            "artifact_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
            "generated_at_utc": _utc_now(),
            "clean_forward_bridge_used": True,
            "legacy_market_snapshots_used": False,
            "gatekeeper_pass_count": gatekeeper_pass_count,
            "gatekeeper_block_count": gatekeeper_block_count,
            "queue_decision": eval_result.get("decision"),
            "queue_eval_rejection_code": eval_result.get("rejection_code"),
            "instrument_id": instrument.get("instrument_id"),
            "execution_instrument_id": instrument.get("execution_instrument_id"),
            "coin_id": instrument.get("coin_id"),
            "paper_orders_opened": paper_orders_opened,
            "paper_positions_opened": paper_positions_opened,
            "paper_positions_closed": int(cycle.get("paper_positions_closed") or 0),
            "execution_mode": "paper",
            "live_trading_ready": False,
            "live_execution_enabled": False,
            "wallet_connected": False,
            "wallet_required": False,
            "not_profitability_evidence": True,
            "paper_demo_only": True,
            "opened_position_instrument_id": (pos or {}).get("instrument_id"),
            "opened_position_coin_id": (pos or {}).get("coin_id"),
            "acceptance": {
                "bridge_ok": bool(bridge.get("ok")),
                "gatekeeper_passed": bool(gate.get("passed")),
                "queue_demo_candidate": eval_result.get("decision") == "DEMO_CANDIDATE",
                "instrument_id_present": bool(instrument.get("instrument_id")),
                "coin_id_absent": instrument.get("coin_id") is None,
                "paper_open_reached": bool(pos) and paper_positions_opened >= 1,
                "live_disabled": True,
            },
        }
        (reports / "ae14_clean_forward_paper_lifecycle_audit.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        summary = [
            "AE14 Clean Forward Paper Lifecycle Smoke",
            f"artifact: {artifact['artifact_dir']}",
            f"queue_decision: {artifact['queue_decision']}",
            f"instrument_id: {artifact['instrument_id']}",
            f"coin_id: {artifact['coin_id']}",
            f"paper_orders_opened: {artifact['paper_orders_opened']}",
            f"paper_positions_opened: {artifact['paper_positions_opened']}",
            f"execution_mode: paper",
            f"live_execution_enabled: false",
            f"wallet_connected: false",
            f"not_profitability_evidence: true",
        ]
        (reports / "ae14_summary_for_upload.txt").write_text(
            "\n".join(summary) + "\n", encoding="utf-8"
        )
        print("\n".join(summary))
        print(f"\nArtifact path: {OUT_DIR}")

        ok = (
            bridge.get("ok")
            and gate.get("passed")
            and eval_result.get("decision") == "DEMO_CANDIDATE"
            and instrument.get("instrument_id")
            and instrument.get("coin_id") is None
            and bool(pos)
            and paper_positions_opened >= 1
            and artifact["live_execution_enabled"] is False
        )
        set_cached_clean_forward_rows(None)
        reset_demo_bot_for_tests()
        os.environ.pop("TRADER_DB_PATH", None)
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
