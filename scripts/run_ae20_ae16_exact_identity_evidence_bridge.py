#!/usr/bin/env python3
"""AE20↔AE16 exact-identity evidence bridge generator (derived artifact only).

Does NOT: start AE21, run 24–48h, train, backtest, mutate trader.db, mutate raw files,
connect wallet, enable live trading, grant trade authority, override risk gates,
claim profitability, claim live readiness, use symbol-only joins, invent identity links,
or use lowercase/casefold/case-insensitive joins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE20↔AE16 exact identity evidence bridge")
    p.add_argument("--paper-demo-only", type=_str2bool, default=True)
    p.add_argument("--clean-forward-only", type=_str2bool, default=True)
    p.add_argument("--no-lowercase-joins", type=_str2bool, default=True)
    p.add_argument("--ae16-real-evidence-path", type=str, default=None)
    p.add_argument("--canonical-index-path", type=str, default=None)
    p.add_argument("--trader-db-path", type=str, default=None)
    p.add_argument(
        "--smoke-root",
        action="append",
        default=None,
        help="Optional AE20 smoke root(s) for AE20 identity inputs. Repeatable.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from app.ae20.exact_identity_evidence_bridge import (
        run_ae20_ae16_exact_identity_evidence_bridge,
    )

    result = run_ae20_ae16_exact_identity_evidence_bridge(
        ROOT,
        paper_demo_only=args.paper_demo_only,
        clean_forward_only=args.clean_forward_only,
        no_lowercase_joins=args.no_lowercase_joins,
        ae16_real_evidence_path=args.ae16_real_evidence_path,
        canonical_index_path=args.canonical_index_path,
        trader_db_path=args.trader_db_path,
        smoke_roots=args.smoke_root,
    )
    print(
        json.dumps(
            {
                "classification": result.get("classification"),
                "output_root": result.get("output_root"),
                "bridge_csv": result.get("bridge_csv"),
                "matched_rows": result.get("ae20_ae16_derived_bridge_matched_rows"),
                "unmatched_rows": result.get("ae20_ae16_unmatched_rows"),
                "profitability_claim": False,
            },
            indent=2,
        )
    )
    print(f"\noutput_root:\n{result.get('output_root')}")
    print(f"classification:\n{result.get('classification')}")
    classification = str(result.get("classification") or "")
    return 2 if "BLOCKED" in classification else 0


if __name__ == "__main__":
    raise SystemExit(main())
