#!/usr/bin/env python3
"""AE7C-1 scoring policy binding + parity harness + inference gate smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.ae7c1_binding_parity_gate import run_ae7c1_binding_parity_gate  # noqa: E402
from scripts.diagnostics._common import open_db_readonly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE7C-1 policy binding parity gate smoke")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--audit-only", action="store_true", default=True)
    parser.add_argument("--no-db-write", action="store_true", default=True)
    parser.add_argument("--parity-mode", default="auto", choices=["auto", "exact-only", "synthetic-only", "off"])
    parser.add_argument("--allow-local-inference-if-gates-pass", action="store_true", default=False)
    args = parser.parse_args()
    del args.no_db_write

    conn = open_db_readonly()
    try:
        summary = run_ae7c1_binding_parity_gate(
            project_root=ROOT,
            conn=conn,
            max_records=args.max_records,
            lookback_hours=args.lookback_hours,
            audit_only=args.audit_only,
            parity_mode=args.parity_mode,
            allow_local_inference_if_gates_pass=args.allow_local_inference_if_gates_pass,
        )
        gate = summary.get("inference_readiness_gate") or {}
        parity = summary.get("feature_parity") or {}
        compact = {
            "phase": summary.get("phase"),
            "dominant_scoring_policy_binding_status": summary.get(
                "dominant_scoring_policy_binding_status"
            ),
            "feature_parity_status": parity.get("feature_parity_status"),
            "inference_gate_status": gate.get("inference_gate_status"),
            "inference_allowed": gate.get("inference_allowed"),
            "inference_executed": (summary.get("inference_attempt") or {}).get(
                "inference_executed"
            ),
            "audit_dir": summary.get("audit_dir"),
        }
        print(json.dumps(compact, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
