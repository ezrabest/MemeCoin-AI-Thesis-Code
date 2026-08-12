#!/usr/bin/env python3
"""AE18 — Context Intelligence Layer (real Helius/Solana continuation).

Does NOT: start AE19/AE20, train, backtest, mutate trader.db, connect wallet,
load private keys, sign/submit transactions, enable live trading, or claim profitability.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    """Load .env into process env if present. Never print secret values."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, _, val = raw.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in __import__("os").environ:
                __import__("os").environ[key] = val
    except OSError:
        pass


_load_dotenv()

from app.ae18 import PHASE  # noqa: E402
from app.ae18.pipeline import run_ae18_context_intelligence_layer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE18 Context Intelligence Layer")
    p.add_argument("--ae17-root", type=str, default=None)
    p.add_argument("--ae16-root", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--allow-external-fetch", action="store_true", default=False)
    p.add_argument(
        "--allow-public-rpc",
        action="store_true",
        default=False,
        help="Allow public Solana RPC fallback when Helius is not configured",
    )
    p.add_argument("--max-candidates", type=int, default=15)
    p.add_argument("--rpc-min-delay-ms", type=int, default=250)
    p.add_argument("--rpc-max-calls", type=int, default=250)
    p.add_argument("--signatures-per-pair", type=int, default=10)
    p.add_argument("--transactions-per-pair", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ae18_context_intelligence_layer(
            ROOT,
            ae17_root=args.ae17_root,
            ae16_root=args.ae16_root,
            output_root=args.output_root,
            allow_external_fetch=args.allow_external_fetch,
            allow_public_rpc=args.allow_public_rpc,
            max_candidates=args.max_candidates,
            rpc_min_delay_ms=args.rpc_min_delay_ms,
            rpc_max_calls=args.rpc_max_calls,
            signatures_per_pair=args.signatures_per_pair,
            transactions_per_pair=args.transactions_per_pair,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{PHASE}] unexpected error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    classification = result.get("classification")
    print(f"[{PHASE}] classification: {classification}")
    print(f"[{PHASE}] output_root: {result.get('output_root')}")
    print(f"[{PHASE}] selected: {result.get('selected_candidate_count')}")
    print(f"[{PHASE}] context_records: {result.get('context_record_count')}")
    print(f"[{PHASE}] rpc_attempted: {(result.get('rpc_stats') or {}).get('rpc_calls_attempted')}")
    print(f"[{PHASE}] real_context_extracted: {result.get('context_extracted_count')}")
    print(f"[{PHASE}] ae18_status: OPEN")
    print(f"[{PHASE}] ae19_status: BLOCKED")
    print(f"[{PHASE}] ae20_status: BLOCKED")

    if str(classification).startswith("AE18_BLOCKED_"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
