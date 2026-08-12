#!/usr/bin/env python3
"""Inspect a coin by symbol from SQLite storage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import (
    get_coin_detail,
    get_coins,
    get_gemini_decisions,
    get_market_snapshots,
    get_signals,
    get_trades,
    get_whale_alerts,
    init_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored coin data")
    parser.add_argument("--symbol", required=True, help="Coin symbol e.g. PEPE/WETH")
    args = parser.parse_args()

    init_pool()
    sym = args.symbol.upper()
    coins = get_coins(limit=500, sort_by="symbol")
    match = next((c for c in coins if (c.get("symbol") or "").upper() == sym), None)
    if not match:
        # partial match
        match = next((c for c in coins if sym in (c.get("symbol") or "").upper()), None)
    if not match:
        print(f"No coin found for symbol {args.symbol}")
        sys.exit(1)

    coin_id = match["id"]
    detail = get_coin_detail(coin_id)
    print("=== COIN ===")
    print(json.dumps(match, indent=2))
    print("\n=== LATEST SNAPSHOT ===")
    snaps = get_market_snapshots(coin_id, limit=10)
    print(json.dumps(snaps[-1] if snaps else {}, indent=2))
    print("\n=== LAST 10 SNAPSHOTS ===")
    print(json.dumps(snaps, indent=2))
    print("\n=== WHALE-LIKE ALERTS ===")
    print(json.dumps(get_whale_alerts(limit=10, coin_id=coin_id), indent=2))
    print("\n=== SIGNALS ===")
    print(json.dumps(get_signals(limit=10, coin_id=coin_id), indent=2))
    print("\n=== GEMINI DECISIONS ===")
    print(json.dumps(get_gemini_decisions(limit=10, coin_id=coin_id), indent=2))
    print("\n=== APP PAPER TRADES ===")
    print(json.dumps(get_trades(limit=10, coin_id=coin_id), indent=2))
    print("\n=== RAW PAYLOAD COUNT ===")
    print(detail.get("raw_payload_count") if detail else 0)


if __name__ == "__main__":
    main()
