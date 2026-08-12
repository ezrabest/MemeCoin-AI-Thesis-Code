#!/usr/bin/env python3
"""Run one watcher scan in headless mode and print collection debug status."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HEADLESS_DATA_COLLECTION", "true")
os.environ.setdefault("ENABLE_GEMINI", "false")

from app import database as db
from app.database import get_collection_debug_status
from app.live import scan_once
from app.llm_config import get_llm_runtime_status
from app.models import TokenRegistry


async def main() -> None:
    db.init_pool()
    before = get_collection_debug_status()
    await scan_once(TokenRegistry())
    after = get_collection_debug_status()
    print(json.dumps({
        "before": before,
        "after": after,
        "deltas": {
            "market_snapshots": after["market_snapshot_count"] - before["market_snapshot_count"],
            "raw_payloads": after["raw_payload_count"] - before["raw_payload_count"],
            "signals": after["signal_count"] - before["signal_count"],
            "whale_alerts": after["whale_alert_count"] - before["whale_alert_count"],
            "llm_skipped_db": after["llm_skipped_count_db"] - before["llm_skipped_count_db"],
        },
        "llm_runtime": get_llm_runtime_status(),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
