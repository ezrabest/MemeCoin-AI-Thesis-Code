#!/usr/bin/env python3
"""AE17 — Original E7 Repair: Meta-Model / Stacking Layer (deterministic shadow).

Read-only consumer of AE16 RF/XGB/TAB/consensus outputs.
Does NOT: train meta-models, call fit(), import sklearn/xgboost/tabicl for training,
start AE18/AE19, call LLMs, call Helius/Solana, mutate trader.db, backtest,
connect wallet, enable live trading, or claim profitability.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.meta import CLASSIFICATION_MISSING_AE16, PHASE  # noqa: E402
from app.meta.discovery import discover_ae16_artifacts  # noqa: E402
from app.meta.pipeline import run_ae17_meta_stacking_layer, write_blocked_missing_inputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE17 Meta-Model / Stacking Layer (rule-based shadow combinator)"
    )
    p.add_argument(
        "--ae16-root",
        type=str,
        default=None,
        help="Optional AE16 audit root to inspect first",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root (default: data/audits/ae17_meta_stacking_layer_<timestamp>)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ae17_meta_stacking_layer(
            ROOT,
            ae16_root=args.ae16_root,
            output_root=args.output_root,
        )
    except FileNotFoundError as exc:
        # Controlled failure — never surface raw stack as the only artifact.
        discovery = discover_ae16_artifacts(ROOT, ae16_root=args.ae16_root)
        blocked = write_blocked_missing_inputs(ROOT, discovery)
        print(f"[{PHASE}] controlled blocker: {CLASSIFICATION_MISSING_AE16}")
        print(f"detail: {exc}")
        print(f"output_root: {blocked.get('output_root')}")
        return 2
    except Exception as exc:  # noqa: BLE001 — final controlled boundary
        # Still attempt discovery audit for operators.
        try:
            discovery = discover_ae16_artifacts(ROOT, ae16_root=args.ae16_root)
            write_blocked_missing_inputs(ROOT, discovery)
        except Exception:  # noqa: BLE001
            pass
        print(f"[{PHASE}] unexpected error (see traceback): {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    classification = result.get("classification")
    print(f"[{PHASE}] classification: {classification}")
    print(f"[{PHASE}] output_root: {result.get('output_root')}")
    print(f"[{PHASE}] feature_rows: {result.get('feature_row_count')}")
    print(f"[{PHASE}] shadow_rows: {result.get('shadow_row_count')}")
    print(f"[{PHASE}] ae18_status: BLOCKED")
    print(f"[{PHASE}] ae19_status: BLOCKED")
    if classification == CLASSIFICATION_MISSING_AE16:
        return 2
    if str(classification).startswith("AE17_BLOCKED_"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
