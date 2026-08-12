#!/usr/bin/env python3
"""Run one watcher scan with Ollama LLM provider (non-headless) and print collection counters."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["HEADLESS_DATA_COLLECTION"] = "false"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["ENABLE_GEMINI"] = "false"

from app import database as db
from app.database import get_collection_debug_status
from app.live import scan_once
from app.models import TokenRegistry


def _report(status: dict) -> dict:
    runtime = status.get("llm_runtime", {})
    return {
        "market_snapshots": status["market_snapshot_count"],
        "raw_provider_payloads": status["raw_payload_count"],
        "signals": status["signal_count"],
        "whale_alerts": status["whale_alert_count"],
        "gemini_decisions": status["gemini_decision_count_db"],
        "llm_skipped_db": status["llm_skipped_count_db"],
        "gemini_call_count": runtime.get("gemini_call_count", 0),
        "ollama_call_count": runtime.get("ollama_call_count", 0),
        "ollama_error_count": runtime.get("ollama_error_count", 0),
    }


async def main() -> None:
    db.init_pool()
    before = _report(get_collection_debug_status())
    await scan_once(TokenRegistry())
    after = _report(get_collection_debug_status())
    print(json.dumps({"before": before, "after": after}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
