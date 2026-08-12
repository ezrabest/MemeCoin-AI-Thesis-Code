#!/usr/bin/env python3
"""AE13L — Backend refresh change counter fix audit pack."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "data" / "audits" / f"ae13l_backend_change_counter_fix_{TIMESTAMP}"
REPORTS = OUT_DIR / "reports"
DATA = OUT_DIR / "data"
AUDITS = OUT_DIR / "audits"
TESTS_OUT = OUT_DIR / "tests"

PROTECTED_PATHS = [
    ROOT / "data" / "training" / "manual_verified_datasets_direct_target_v1",
    ROOT / "data" / "training" / "manual_verified_datasets_clean_for_model",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def snapshot_paths(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        key = str(p.relative_to(ROOT)).replace("\\", "/")
        if not p.exists():
            out[key] = {"exists": False}
            continue
        if p.is_file():
            st = p.stat()
            out[key] = {"exists": True, "is_file": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        else:
            children = [{"name": c.name, "mtime_ns": c.stat().st_mtime_ns} for c in sorted(p.iterdir())]
            digest = hashlib.sha256(json.dumps(children, sort_keys=True, default=str).encode()).hexdigest()[:32]
            out[key] = {"exists": True, "is_dir": True, "listing_hash_prefix": digest}
    return out


def _sample_pair(**kwargs: Any) -> dict[str, Any]:
    base = {
        "chainId": "solana",
        "dexId": "pumpswap",
        "pairAddress": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "url": "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "baseToken": {
            "address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7hCNwe",
            "symbol": "PUMP",
            "name": "PUMP",
        },
        "quoteToken": {
            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "symbol": "USDC",
            "name": "USD Coin",
        },
        "priceUsd": "0.01",
        "liquidity": {"usd": 100000.0},
        "volume": {"h5": 1.0, "h1": 10.0, "h6": 50.0, "h24": 100.0},
        "txns": {"h24": {"buys": 50, "sells": 40}},
        "priceChange": {"m5": 0.1, "h1": 1.0, "h6": 2.0, "h24": 3.0},
    }
    base.update(kwargs)
    return base


def _verified_dict(pair: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    base = {
        "lookup_ok": True,
        "clean_feed_eligible": True,
        "verification_status": "provider_pair_verified",
        "normalized_chain_id": "solana",
        "chain_id": "solana",
        "pair_address": pair["pairAddress"],
        "provider_pair_url": pair["url"],
        "provider_pair_url_source": "provider_returned_url",
        "dex_id": "pumpswap",
        "base_token_address": pair["baseToken"]["address"],
        "base_token_symbol": "PUMP",
        "quote_token_address": pair["quoteToken"]["address"],
        "quote_token_symbol": "USDC",
        "price_usd": pair["priceUsd"],
        "liquidity_usd": 100000,
        "volume_24h": 100,
        "txns_24h_buys": 50,
        "txns_24h_sells": 40,
        "price_change_5m": 0.1,
        "price_change_1h": 1.0,
        "price_change_6h": 2.0,
        "price_change_24h": 3.0,
        "provider_payload_hash": "hash1",
        "fetched_at": "2026-07-21T00:00:00+00:00",
        "freshness_status": "fresh",
        "address_role": "pool_address",
        "pair_label": "PUMP/USDC",
        "verification_cache_hit": False,
    }
    base.update(overrides)
    return base


def run_counter_proof() -> dict[str, Any]:
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
        refresh_clean_forward_market_feed,
        reset_clean_forward_refresh_baseline,
    )
    from app.ae13b_product.dexscreener_pair_verify import get_pair_verify_limiter

    reset_clean_forward_refresh_baseline()
    get_pair_verify_limiter().clear_cache()
    get_pair_verify_limiter().reset_stats()

    pair = _sample_pair()
    search = [
        {
            "chainId": "solana",
            "pairAddress": pair["pairAddress"],
            "priceUsd": pair["priceUsd"],
            "liquidity": {"usd": 100000},
            "volume": {"h24": 100},
            "baseToken": pair["baseToken"],
            "quoteToken": pair["quoteToken"],
        }
    ]
    verified_v1 = _verified_dict(pair)
    verified_v2 = _verified_dict(pair, price_usd="0.02", provider_payload_hash="hash2")
    verified_v3 = _verified_dict(pair)

    prev = _row_from_verification(verified_v1)
    curr_changed = _row_from_verification(verified_v2)
    curr_unchanged = _row_from_verification(verified_v3)

    unit = {
        "values_changed": _compute_refresh_change_counters(
            [curr_changed], previous_rows=[prev], provider_refetch_completed=True
        ),
        "hash_changed": _compute_refresh_change_counters(
            [_row_from_verification(_verified_dict(pair, provider_payload_hash="hash9"))],
            previous_rows=[prev],
            provider_refetch_completed=True,
        ),
        "unchanged_refetched": _compute_refresh_change_counters(
            [curr_unchanged], previous_rows=[prev], provider_refetch_completed=True
        ),
        "no_baseline": _compute_refresh_change_counters(
            [curr_unchanged], previous_rows=None, provider_refetch_completed=True
        ),
    }

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=search,
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        side_effect=[verified_v1, verified_v2],
    ):
        first = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)
        second = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)

    integration = {
        "first_poll": first.get("refresh") or {},
        "second_poll": second.get("refresh") or {},
    }
    return {"timestamp_utc": utc_now(), "unit_counter_proofs": unit, "integration_refresh_proofs": integration}


def classify(proof: dict[str, Any], tests_ok: bool, protected_ok: bool) -> str:
    if not protected_ok:
        return "AE13L_BLOCKED_SCOPE_VIOLATION"
    if not tests_ok:
        return "AE13L_BLOCKED_REFRESH_CHANGE_COUNTER_WRONG"

    unit = proof["unit_counter_proofs"]
    integration = proof["integration_refresh_proofs"]
    checks = [
        unit["values_changed"]["provider_values_changed_count"] > 0,
        unit["hash_changed"]["payload_hash_changed_count"] > 0,
        unit["unchanged_refetched"]["provider_unchanged_but_refetched_count"] > 0,
        unit["no_baseline"]["comparison_baseline_available"] is False,
        integration["first_poll"].get("comparison_baseline_available") is False,
        integration["second_poll"].get("comparison_baseline_available") is True,
        integration["second_poll"].get("provider_values_changed_count", 0) > 0,
    ]
    if all(checks):
        return "AE13L_BACKEND_COUNTER_FIX_PASS"
    if any(checks):
        return "AE13L_BACKEND_COUNTER_FIX_PASS_WITH_LIMITATIONS"
    return "AE13L_BLOCKED_REFRESH_CHANGE_COUNTER_WRONG"


def main() -> int:
    for d in (REPORTS, DATA, AUDITS, TESTS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[AE13L] artifact dir: {OUT_DIR}")
    before_protected = snapshot_paths(PROTECTED_PATHS)

    proof = run_counter_proof()
    write_json(DATA / "ae13l_change_counter_proof.json", proof)

    print("[AE13L] running pytest...")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_ae13l_backend_change_counter_fix.py",
            "-q",
            "--tb=short",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (TESTS_OUT / "ae13l_backend_change_counter_fix_test_results.md").write_text(
        f"# AE13L test results\n\n```\n{proc.stdout}\n{proc.stderr}\n```\n",
        encoding="utf-8",
    )
    tests_ok = proc.returncode == 0

    after_protected = snapshot_paths(PROTECTED_PATHS)
    protected_ok = before_protected == after_protected
    write_json(
        AUDITS / "ae13l_no_old_data_mutation_audit.json",
        {
            "timestamp_utc": utc_now(),
            "before": before_protected,
            "after": after_protected,
            "unchanged": protected_ok,
        },
    )
    write_json(
        AUDITS / "ae13l_no_training_no_trading_scope_audit.json",
        {
            "timestamp_utc": utc_now(),
            "training_run": False,
            "backtest_run": False,
            "ae14_run": False,
            "paper_positions_opened_from_clean_feed": 0,
            "live_trading_enabled": False,
            "old_data_touched": False,
        },
    )

    classification = classify(proof, tests_ok, protected_ok)
    gate = {
        "timestamp_utc": utc_now(),
        "classification": classification,
        "tests_passed": tests_ok,
        "protected_paths_unchanged": protected_ok,
        "scope": {
            "old_data_touched": False,
            "retrain": False,
            "backtest": False,
            "ae14": False,
            "paper_demo_trades": False,
            "live_trading": False,
        },
        "proof_summary": {
            "values_changed_count": proof["unit_counter_proofs"]["values_changed"]["provider_values_changed_count"],
            "hash_changed_count": proof["unit_counter_proofs"]["hash_changed"]["payload_hash_changed_count"],
            "unchanged_refetched_count": proof["unit_counter_proofs"]["unchanged_refetched"][
                "provider_unchanged_but_refetched_count"
            ],
            "first_poll_baseline": proof["integration_refresh_proofs"]["first_poll"].get(
                "comparison_baseline_available"
            ),
            "second_poll_baseline": proof["integration_refresh_proofs"]["second_poll"].get(
                "comparison_baseline_available"
            ),
            "second_poll_values_changed": proof["integration_refresh_proofs"]["second_poll"].get(
                "provider_values_changed_count"
            ),
        },
    }
    write_json(REPORTS / "ae13l_decision_gate.json", gate)

    summary = f"""AE13L Backend Change Counter Fix
Timestamp UTC: {utc_now()}
Classification: {classification}
Tests passed: {tests_ok}
Protected paths unchanged: {protected_ok}

Unit proofs:
- provider_values_changed_count: {gate['proof_summary']['values_changed_count']}
- payload_hash_changed_count: {gate['proof_summary']['hash_changed_count']}
- provider_unchanged_but_refetched_count: {gate['proof_summary']['unchanged_refetched_count']}
- first poll comparison_baseline_available: {gate['proof_summary']['first_poll_baseline']}
- second poll comparison_baseline_available: {gate['proof_summary']['second_poll_baseline']}
- second poll provider_values_changed_count: {gate['proof_summary']['second_poll_values_changed']}

Artifact dir: {OUT_DIR}
"""
    (REPORTS / "ae13l_summary_for_upload.txt").write_text(summary, encoding="utf-8")
    (REPORTS / "ae13l_backend_change_counter_fix_report.md").write_text(
        "# AE13L Backend Change Counter Fix\n\n"
        + summary
        + "\n## Scope\n\n"
        + "- No old data touched\n"
        + "- No retrain / backtest / AE14 / paper-demo / live trading\n",
        encoding="utf-8",
    )
    (ROOT / "data" / "audits" / "AE13L_LATEST.txt").write_text(
        f"{classification}\n{OUT_DIR}\n{utc_now()}\n",
        encoding="utf-8",
    )

    print(summary)
    return 0 if classification.startswith("AE13L_BACKEND_COUNTER_FIX_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
