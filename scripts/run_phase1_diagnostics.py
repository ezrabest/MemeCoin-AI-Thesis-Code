#!/usr/bin/env python3
"""Run Phase 1 diagnostic audit reports (15–30 min session helper)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 observability diagnostic session")
    parser.add_argument("--all", action="store_true", help="Run all Phase 1 audit reports")
    parser.add_argument("--settings", action="store_true")
    parser.add_argument("--storage", action="store_true")
    parser.add_argument("--whale-wave", action="store_true")
    parser.add_argument("--sentiment", action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any([args.settings, args.storage, args.whale_wave, args.sentiment])

    paths: list[str] = []

    if run_all or args.settings:
        from app.observability.effective_settings import get_effective_settings

        p = get_effective_settings().write_audit_report()
        paths.append(str(p))
        print(f"Settings effective: {p}")

    if run_all or args.storage:
        from scripts.reconcile_storage import run_check

        r = run_check()
        paths.append(r["output_path"])
        print(f"Storage reconcile: {r['output_path']}")

    if run_all or args.whale_wave:
        from app.observability.whale_wave_audit import run_whale_wave_audit

        r = run_whale_wave_audit()
        paths.append(r["output_path"])
        print(f"Whale-wave audit: {r['output_path']}")

    if run_all or args.sentiment:
        from app.observability.sentiment_cluster_audit import run_sentiment_cluster_audit

        r = run_sentiment_cluster_audit()
        paths.append(r["output_path"])
        print(f"Sentiment/cluster audit: {r['output_path']}")

    print(f"Phase 1 diagnostics complete — {len(paths)} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
