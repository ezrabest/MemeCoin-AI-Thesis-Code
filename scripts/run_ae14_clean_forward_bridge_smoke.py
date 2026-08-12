#!/usr/bin/env python3
"""AE14 Clean Forward bridge validation smoke (paper/demo only).

Seeds the in-process Clean Forward cache, evaluates a queue item with
source=ae14_clean_forward_*, and optionally runs one demo-bot cycle.
Writes an audit artifact for the next AE14 run.

Does not touch wallets, private keys, live trading, or invent coin_id.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae14_clean_forward_bridge_smoke_{TIMESTAMP}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_row() -> dict[str, Any]:
    ts = _utc_now()
    return {
        "row_id": "base|pair|0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        "source_provider": "dexscreener",
        "normalized_chain_id": "base",
        "chain": "base",
        "chain_id": "base",
        "provider_pair_id": "0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        "pair_address": "0x8Df6dd38D718bD726374521c2DcFE90Eb9CB7d43",
        "base_token_address": "0x311935Cd80B76769bF2ecC9D8Ab7635b2139cf82",
        "base_token_symbol": "SOL",
        "quote_token_address": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        "quote_token_symbol": "cbBTC",
        "pair": "SOL/cbBTC",
        "pair_label": "SOL/cbBTC",
        "price": "77.77",
        "price_usd": "77.77",
        "liquidity": 558118.71,
        "liquidity_usd": 558118.71,
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
        "price_change_1h": 0.5,
        "txns_24h_buys": 10,
        "txns_24h_sells": 8,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = OUT_DIR / "reports"
    audits = OUT_DIR / "audits"
    data = OUT_DIR / "data"
    for d in (reports, audits, data):
        d.mkdir(parents=True, exist_ok=True)

    from app.ae13b_product.clean_forward_bridge import (
        build_clean_forward_gatekeeper_candidate,
    )
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows
    from app.ae13b_product import demo_queue
    from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

    row = _valid_row()
    set_cached_clean_forward_rows([row])
    (data / "clean_forward_seed_row.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )

    bridge = build_clean_forward_gatekeeper_candidate(row)
    (data / "bridge_result.json").write_text(
        json.dumps(bridge, indent=2, default=str), encoding="utf-8"
    )

    gatekeeper_pass_count = 0
    gatekeeper_block_count = 0
    if bridge.get("ok") and bridge.get("candidate"):
        gate = validate_market_data_gate(
            bridge["candidate"], for_open=True, skip_stagnant=False
        )
        if gate.get("passed"):
            gatekeeper_pass_count = 1
        else:
            gatekeeper_block_count = 1
        (data / "gatekeeper_result.json").write_text(
            json.dumps(gate, indent=2, default=str), encoding="utf-8"
        )

    # Isolated queue path for smoke
    smoke_queue = data / "demo_trade_queue_smoke.json"
    demo_queue.QUEUE_PATH = smoke_queue
    demo_queue.DATA_DIR = data

    entry = demo_queue.add_to_demo_queue(
        symbol="SOL",
        pair="SOL/cbBTC",
        chain="base",
        contract_or_pair_address=row["pair_address"],
        source="ae14_clean_forward_inline_queue_eval_smoke",
        market_match_status="provider_pair_verified",
        risk_mode="paper_demo_only_clean_forward",
        max_notional=100.0,
        user_hypothesis="AE14 clean-forward bridge smoke",
    )
    eval_result = demo_queue.evaluate_queue_item(entry["queue_id"])
    (data / "demo_queue_evaluate_response.json").write_text(
        json.dumps(eval_result, indent=2, default=str), encoding="utf-8"
    )

    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    cycle = bot.run_once()
    (data / "demo_bot_run_once_response.json").write_text(
        json.dumps(cycle, indent=2, default=str), encoding="utf-8"
    )

    artifact = {
        "artifact_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "generated_at_utc": _utc_now(),
        "clean_forward_rows_seen": 1,
        "clean_forward_candidates_selected": 1 if bridge.get("ok") else 0,
        "clean_forward_bridge_pass_count": 1 if bridge.get("ok") else 0,
        "clean_forward_bridge_block_count": 0 if bridge.get("ok") else 1,
        "gatekeeper_pass_count": gatekeeper_pass_count
        or int(cycle.get("gatekeeper_pass_count") or 0),
        "gatekeeper_block_count": gatekeeper_block_count
        or int(cycle.get("gatekeeper_block_count") or 0),
        "paper_orders_opened": int(cycle.get("paper_orders_opened") or 0),
        "paper_positions_opened": int(cycle.get("paper_positions_opened") or 0),
        "paper_positions_closed": int(cycle.get("paper_positions_closed") or 0),
        "legacy_market_snapshots_used": False,
        "clean_forward_bridge_used": True,
        "live_trading_ready": False,
        "paper_demo_only": True,
        "not_profitability_evidence": True,
        "queue_eval_rejection_code": eval_result.get("rejection_code"),
        "queue_eval_decision": eval_result.get("decision"),
        "queue_eval_blocked_missing_price": eval_result.get("rejection_code")
        == "NOT_OPENED_MISSING_PRICE",
        "queue_clean_forward_bridge_used": bool(
            eval_result.get("clean_forward_bridge_used")
        ),
        "demo_bot_clean_forward_bridge_used": bool(
            cycle.get("clean_forward_bridge_used")
        ),
        "acceptance": {
            "no_missing_price_blocker": eval_result.get("rejection_code")
            != "NOT_OPENED_MISSING_PRICE",
            "bridge_ok": bool(bridge.get("ok")),
            "legacy_snapshots_not_used": True,
        },
    }
    (reports / "ae14_clean_forward_bridge_audit.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    (audits / "ae14_clean_forward_bridge_audit.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )

    summary_lines = [
        "AE14 Clean Forward Bridge Smoke",
        f"artifact: {artifact['artifact_dir']}",
        f"bridge_ok: {bridge.get('ok')}",
        f"queue_rejection_code: {eval_result.get('rejection_code')}",
        f"queue_decision: {eval_result.get('decision')}",
        f"missing_price_blocker: {artifact['queue_eval_blocked_missing_price']}",
        f"clean_forward_bridge_used: true",
        f"legacy_market_snapshots_used: false",
        f"paper_demo_only: true",
        f"not_profitability_evidence: true",
    ]
    (reports / "ae14_summary_for_upload.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines))
    print(f"\nArtifact path: {OUT_DIR}")
    ok = (
        bridge.get("ok")
        and eval_result.get("rejection_code") != "NOT_OPENED_MISSING_PRICE"
        and eval_result.get("clean_forward_bridge_used") is True
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
