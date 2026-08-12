#!/usr/bin/env python3
"""Read-only audit of social/opportunistic semantic layer.

Default: no classification, no DB mutation.
Optional:
  --classify-sample   classify at most 10 tokens (no trading)
  --persist           persist sample verdicts to data/semantic_verdicts.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _try_api_counts(base_url: str) -> dict | None:
    url = base_url.rstrip("/") + "/api/semantic/counts"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit social/opportunistic semantic layer")
    parser.add_argument("--classify-sample", action="store_true")
    parser.add_argument("--persist", action="store_true", help="Persist sample verdicts (requires --classify-sample)")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--max-sample", type=int, default=10)
    args = parser.parse_args()

    from app.semantic.llm_semantic_client import resolve_semantic_llm_provider
    from app.semantic.semantic_registry import load_semantic_verdicts
    from app.semantic.social_opportunistic_classifier import (
        classify_token_social_opportunistic,
        get_authoritative_semantic_counts,
        legacy_cluster_label_counts,
    )

    out_dir = ROOT / "data" / "audits" / f"social_opportunistic_semantic_{_utc()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = get_authoritative_semantic_counts(project_root=ROOT)
    legacy = legacy_cluster_label_counts(project_root=ROOT)
    api_counts = _try_api_counts(args.api_base)

    # Before snapshot
    before_path = out_dir / "before_counts.json"
    before_path.write_text(json.dumps(before, indent=2), encoding="utf-8")

    sample_verdicts: list[dict] = []
    if args.classify_sample:
        # Sample from seed targets + a few registry keys — no trading
        seeds_path = ROOT / "data" / "SeedTargets" / "dexscreener_seed_targets_v1.json"
        tokens: list[dict] = []
        if seeds_path.is_file():
            payload = json.loads(seeds_path.read_text(encoding="utf-8"))
            for row in (payload.get("rows") or [])[: args.max_sample]:
                tokens.append(
                    {
                        "chain": row.get("chain") or "",
                        "pair_address": row.get("user_supplied_pair_address") or "",
                        "token_address": row.get("user_supplied_token_address") or "",
                        "symbol": "",
                        "name": "",
                        "provider_url": row.get("provider_pair_url") or "",
                    }
                )
        for t in tokens[: args.max_sample]:
            v = classify_token_social_opportunistic(
                chain=t.get("chain") or "",
                pair_address=t.get("pair_address") or "",
                token_address=t.get("token_address") or "",
                symbol=t.get("symbol") or "",
                name=t.get("name") or "",
                provider_url=t.get("provider_url") or "",
                persist=bool(args.persist),
            )
            sample_verdicts.append(v)
        # Always write audit-folder copies even without --persist
        (out_dir / "sample_verdicts.json").write_text(
            json.dumps(sample_verdicts, indent=2, default=str),
            encoding="utf-8",
        )
        with open(out_dir / "semantic_verdict_sample.csv", "w", encoding="utf-8", newline="") as f:
            fields = [
                "identity_key",
                "symbol",
                "chain",
                "pair_address",
                "user_seed_label",
                "user_hypothesis",
                "semantic_status",
                "cluster_label",
                "confidence",
                "evidence_quality",
                "provider",
                "model",
                "reasoning",
                "no_trade_authority",
            ]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in sample_verdicts:
                w.writerow(row)

    after = get_authoritative_semantic_counts(project_root=ROOT)
    (out_dir / "after_counts.json").write_text(json.dumps(after, indent=2), encoding="utf-8")

    verdicts = load_semantic_verdicts()
    insufficient = [v for v in verdicts if v.get("semantic_status") == "INSUFFICIENT_EVIDENCE"]
    samples = verdicts[:10] if verdicts else sample_verdicts[:10]

    mismatches = []
    if api_counts:
        for key in (
            "social_confirmed_count",
            "opportunistic_confirmed_count",
            "insufficient_evidence_count",
            "classification_failed_count",
            "legacy_socially_motivated_count",
            "legacy_opportunistic_speculative_count",
        ):
            if int(api_counts.get(key) or 0) != int(after.get(key) or 0):
                mismatches.append(
                    {
                        "key": key,
                        "api": api_counts.get(key),
                        "authoritative": after.get(key),
                    }
                )
    else:
        mismatches.append({"note": "API not reachable; skipped API mismatch check"})

    # Whale-log vs authoritative (documents the fixed bug)
    try:
        from app.models.predictor import count_by_cluster

        whale_or_auth = count_by_cluster()
    except Exception as exc:
        whale_or_auth = {"error": str(exc)}

    report = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_counts": legacy.get("registry"),
        "db_counts": {
            "legacy_socially_motivated_count": legacy.get("legacy_socially_motivated_count"),
            "legacy_opportunistic_speculative_count": legacy.get("legacy_opportunistic_speculative_count"),
            "by_table": legacy.get("db", {}).get("by_table"),
            "tables_scanned": legacy.get("db", {}).get("tables_scanned"),
            "coins_table_has_cluster_label": legacy.get("coins_table_has_cluster_label"),
        },
        "semantic_verdict_counts": {
            "social_confirmed_count": after.get("social_confirmed_count"),
            "opportunistic_confirmed_count": after.get("opportunistic_confirmed_count"),
            "insufficient_evidence_count": after.get("insufficient_evidence_count"),
            "classification_failed_count": after.get("classification_failed_count"),
            "total_semantic_verdicts": after.get("total_semantic_verdicts"),
        },
        "api_counts": api_counts,
        "count_by_cluster_now": whale_or_auth,
        "mismatch_report": mismatches,
        "before_counts": before,
        "after_counts": after,
        "sample_verdicts": samples,
        "insufficient_evidence_cases": insufficient[:50],
        "llm_provider_resolved": resolve_semantic_llm_provider(),
        "web_evidence_collected": False,
        "classify_sample_ran": bool(args.classify_sample),
        "persist_enabled": bool(args.persist),
        "safety_confirmation": {
            "no_trade_authority": True,
            "live_trading_enabled_changed": False,
            "wallet_connected": False,
            "paper_demo_orders_opened": False,
            "risk_gates_overridden": False,
            "historical_trading_results_mutated": False,
            "cluster_registry_mutated": False,
            "gemini_code_removed": False,
            "qwen_ollama_code_removed": False,
            "cluster_label_enum_broken": False,
        },
        "output_dir": str(out_dir),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Human-readable before/after
    lines = [
        "Social/Opportunistic Semantic Layer Audit",
        f"output: {out_dir}",
        "",
        "BEFORE:",
        f"  system social_confirmed={before.get('social_confirmed_count')}",
        f"  system opportunistic_confirmed={before.get('opportunistic_confirmed_count')}",
        f"  insufficient={before.get('insufficient_evidence_count')}",
        f"  failed={before.get('classification_failed_count')}",
        f"  legacy_social(DB)={before.get('legacy_socially_motivated_count')}",
        f"  legacy_opp(DB)={before.get('legacy_opportunistic_speculative_count')}",
        f"  registry_social={before.get('legacy_registry_socially_motivated_count')}",
        f"  registry_opp={before.get('legacy_registry_opportunistic_speculative_count')}",
        "",
        "AFTER:",
        f"  system social_confirmed={after.get('social_confirmed_count')}",
        f"  system opportunistic_confirmed={after.get('opportunistic_confirmed_count')}",
        f"  insufficient={after.get('insufficient_evidence_count')}",
        f"  failed={after.get('classification_failed_count')}",
        f"  legacy_social(DB)={after.get('legacy_socially_motivated_count')}",
        f"  legacy_opp(DB)={after.get('legacy_opportunistic_speculative_count')}",
        "",
        f"API reachable: {api_counts is not None}",
        f"mismatches: {mismatches}",
        f"LLM provider: {report['llm_provider_resolved']}",
        f"sample classified: {len(sample_verdicts)}",
        "safety: no trade authority; no wallet; no live; no paper orders opened by audit.",
    ]
    (out_dir / "before_after_counter_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
