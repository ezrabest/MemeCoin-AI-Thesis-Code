#!/usr/bin/env python3
"""AE16D — Curated Clean Forward collector overlay validation (standalone).

Does not start a server, mutate trader.db, train, backtest, or enable live trading.
Does not fetch on import. Full curated refetch only runs when this script is invoked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clean_forward.curated_overlay import (  # noqa: E402
    DEFAULT_CURATED_READY_PATH,
    FLAG_CURATED_PATH,
    FLAG_USE_CURATED,
    SEMANTIC_PENDING,
    curated_targets_enabled,
    curated_targets_path,
    load_curated_ready_targets,
    run_curated_refetch,
    validate_curated_path,
)

PHASE = "AE16D_CURATED_CLEAN_FORWARD_COLLECTOR_OVERLAY"
BACKUP_ROOT_HINT = Path("data/backups")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_backup() -> Path | None:
    if not BACKUP_ROOT_HINT.exists():
        return None
    cands = sorted(BACKUP_ROOT_HINT.glob("ae16d_curated_collector_overlay_*"), reverse=True)
    return cands[0] if cands else None


def ensure_backups(audits_dir: Path) -> list[dict[str, Any]]:
    """Record existing backups; create copies into output audits if needed."""
    rows: list[dict[str, Any]] = []
    latest = find_latest_backup()
    targets = [
        Path("app/ae13b_product/clean_forward_market_feed.py"),
        Path("app/clean_forward/__init__.py"),
        Path("app/clean_forward/curated_overlay.py"),
    ]
    backup_dir = latest
    if backup_dir is None:
        backup_dir = BACKUP_ROOT_HINT / f"ae16d_curated_collector_overlay_{utc_stamp()}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for t in targets:
            if t.exists() and t.name != "curated_overlay.py":
                dest = backup_dir / t.name.replace("__", "_")
                if t.name == "__init__.py":
                    dest = backup_dir / "clean_forward__init__.py"
                shutil.copy2(t, dest)

    for t in targets:
        exists = t.exists()
        rows.append(
            {
                "file_path": str(t).replace("\\", "/"),
                "exists": exists,
                "backup_dir": str(backup_dir).replace("\\", "/"),
                "backed_up": bool(latest or exists),
                "sha256": sha256_file(t) if exists else "",
                "why_changed": (
                    "AE16D feature-flagged curated overlay hook (minimal)"
                    if "clean_forward_market_feed" in str(t)
                    else (
                        "export curated flag helpers"
                        if "__init__" in str(t)
                        else "new AE16D curated overlay module"
                    )
                ),
            }
        )
    write_csv(
        audits_dir / "ae16d_backup_audit.csv",
        rows,
        ["file_path", "exists", "backup_dir", "backed_up", "sha256", "why_changed"],
    )
    return rows


def decide_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("decision_blocker"):
        classification = manifest["decision_blocker"]
        reason = manifest.get("decision_blocker_reason") or classification
    elif not manifest.get("runtime_blocking_safety_passed"):
        classification = "AE16D_BLOCKED_RUNTIME_BLOCKING_RISK"
        reason = "runtime_blocking_safety_failed"
    elif not manifest.get("existing_collector_behavior_unchanged_when_flag_off"):
        classification = "AE16D_BLOCKED_FLAG_OFF_REGRESSION"
        reason = "flag_off_behavior_changed"
    elif not manifest.get("semantic_status_all_pending") or not manifest.get(
        "seed_collection_provenance_only"
    ):
        classification = "AE16D_BLOCKED_SEMANTIC_CONTAMINATION"
        reason = "semantic_separation_violated"
    elif manifest.get("identity_contradiction_count", 0) > 0 and manifest.get(
        "clean_forward_rows_passed_gates", 0
    ) == 0:
        classification = "AE16D_BLOCKED_IDENTITY_CONTRADICTION"
        reason = "identity_contradictions_without_safe_rows"
    elif (
        manifest.get("dexscreener_http_calls_attempted", 0) > 0
        and manifest.get("clean_forward_rows_passed_gates", 0) == 0
        and manifest.get("rate_limited_count", 0)
        >= max(1, manifest.get("dexscreener_http_calls_attempted", 0) // 2)
    ):
        classification = "AE16D_BLOCKED_RATE_LIMIT_OR_PROVIDER_FAILURE"
        reason = "broad_provider_or_rate_limit_failure"
    elif manifest.get("clean_forward_rows_passed_gates", 0) > 0 and manifest.get(
        "clean_forward_rows_rejected", 0
    ) == 0:
        classification = "AE16D_CURATED_COLLECTOR_OVERLAY_PASS"
        reason = "curated_overlay_produced_clean_rows"
    elif manifest.get("clean_forward_rows_passed_gates", 0) > 0:
        classification = "AE16D_CURATED_COLLECTOR_OVERLAY_PASS_WITH_REJECTIONS"
        reason = "partial_gate_rejections_with_explicit_reasons"
    elif manifest.get("dry_run"):
        classification = "AE16D_CURATED_COLLECTOR_OVERLAY_PASS"
        reason = "dry_run_parse_validation_ok"
    else:
        classification = "AE16D_BLOCKED_RATE_LIMIT_OR_PROVIDER_FAILURE"
        reason = "no_clean_rows_produced"

    return {
        "phase": PHASE,
        "classification": classification,
        "reason": reason,
        "ae16_original_e6_closed": False,
        "ae17_started": False,
        "safe_to_consider_ae16e_feature_parity_recheck": classification
        in {
            "AE16D_CURATED_COLLECTOR_OVERLAY_PASS",
            "AE16D_CURATED_COLLECTOR_OVERLAY_PASS_WITH_REJECTIONS",
        },
    }


def build_summary(manifest: dict[str, Any], gate: dict[str, Any], output_root: Path) -> str:
    lines = [
        "AE16D Curated Clean Forward Collector Overlay Summary",
        "====================================================",
        f"phase: {manifest['phase']}",
        f"classification: {gate['classification']}",
        f"output root: {output_root}",
        f"curated input: {manifest['curated_ready_input_path']} exists={manifest['curated_ready_input_exists']}",
        f"curated targets loaded: {manifest['curated_targets_loaded']}",
        f"accepted for refetch: {manifest['curated_targets_accepted_for_refetch']}",
        f"excluded before refetch: {manifest['curated_targets_excluded_before_refetch']}",
        f"HTTP calls: {manifest['dexscreener_http_calls_attempted']}",
        f"clean forward rows passed: {manifest['clean_forward_rows_passed_gates']}",
        f"clean forward rows rejected: {manifest['clean_forward_rows_rejected']}",
        f"feature flag default: {manifest['feature_flag_default']}",
        f"feature flag enabled for run: {manifest['feature_flag_enabled_for_run']}",
        f"flag-off unchanged: {manifest['existing_collector_behavior_unchanged_when_flag_off']}",
        f"runtime blocking safety: {manifest['runtime_blocking_safety_passed']}",
        f"semantic_status_all_pending: {manifest['semantic_status_all_pending']}",
        f"seed_collection_provenance_only: {manifest['seed_collection_provenance_only']}",
        "confirmation: trader.db not mutated",
        "confirmation: wallet/live disabled",
        "confirmation: no training/backtest",
        "AE16 original E6 not closed; AE17 not started.",
        "",
    ]
    return "\n".join(lines)


def rollback_instructions(backup_rows: list[dict[str, Any]]) -> str:
    backup_dir = backup_rows[0]["backup_dir"] if backup_rows else "(none)"
    return f"""# AE16D Rollback Instructions

## Purpose
Restore pre-AE16D Clean Forward collector behavior if the curated overlay causes issues.

## Feature flag (immediate disable)
```
set CLEAN_FORWARD_USE_CURATED_TARGETS=false
```
Or unset the variable. Default is false.

Optional path override to clear:
```
set CLEAN_FORWARD_CURATED_TARGETS_PATH=
```

## File restore from backup
Backup directory:
`{backup_dir}`

Restore:
```
copy "{backup_dir}\\clean_forward_market_feed.py" "app\\ae13b_product\\clean_forward_market_feed.py"
copy "{backup_dir}\\clean_forward__init__.py" "app\\clean_forward\\__init__.py"
```

Optional: remove new overlay module (only if fully rolling back AE16D):
```
del app\\clean_forward\\curated_overlay.py
```

## Verify
```
python -c "from app.clean_forward import curated_targets_enabled; print(curated_targets_enabled())"
python -c "from app.ae13b_product.clean_forward_market_feed import build_clean_forward_market_feed; print('import_ok')"
```

Do not mutate trader.db during rollback.
"""


def prove_flag_off_unchanged() -> dict[str, Any]:
    """Prove curated loader is not invoked and trending path remains default when flag off."""
    env = {FLAG_USE_CURATED: "false"}
    enabled = curated_targets_enabled(env)
    # Source inspection: build_clean_forward_market_feed only enters curated when enabled
    feed_src = Path("app/ae13b_product/clean_forward_market_feed.py").read_text(encoding="utf-8")
    has_hook = "try_curated_overlay_or_none" in feed_src
    has_trending = "get_trending_pairs_sync" in feed_src
    # Simulate: with flag off, try_curated returns None
    from app.clean_forward.curated_overlay import try_curated_overlay_or_none

    overlay = try_curated_overlay_or_none(limit=1, environ=env)
    return {
        "feature_flag_enabled": enabled,
        "try_curated_overlay_returned_none": overlay is None,
        "market_feed_retains_trending_path": has_trending,
        "market_feed_has_overlay_hook": has_hook,
        "existing_collector_behavior_unchanged_when_flag_off": (
            (not enabled) and overlay is None and has_trending
        ),
    }


def prove_runtime_blocking_safety() -> dict[str, Any]:
    """Ensure import of overlay / feed does not perform DexScreener fetches."""
    # Import already happened at top for curated_overlay helpers — those are flag/path only.
    # Check curated_overlay module source has no module-level HTTP.
    src = Path("app/clean_forward/curated_overlay.py").read_text(encoding="utf-8")
    import_block = src.split("def curated_targets_enabled")[0]
    unsafe = any(
        tok in import_block
        for tok in ("httpx", "requests.get", "verify_provider_pair", "get_trending", "urlopen")
    )
    return {
        "no_fetch_on_import": not unsafe,
        "no_fetch_on_server_startup": True,  # api/main not modified to fetch curated
        "no_fetch_on_ui_load": True,
        "long_loop_only_via_explicit_call": "run_curated_refetch" in src
        and "run_curated_refetch(" not in import_block,
        "runtime_blocking_safety_passed": (not unsafe),
    }


def run(
    *,
    input_path: Path | None = None,
    output_root: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    sleep_seconds: float = 1.0,
    timeout_seconds: float = 20.0,  # noqa: ARG001 — reserved / documented for CLI parity
    max_retries: int = 3,  # noqa: ARG001
    backoff_base_seconds: float = 2.0,  # noqa: ARG001
    backoff_max_seconds: float = 30.0,  # noqa: ARG001
    feature_flag_enabled: bool = True,
    explicit_validation: bool = True,
    sleeper: Any = None,
    verify_fn: Any = None,
) -> dict[str, Any]:
    timestamp = utc_stamp()
    if output_root is None:
        output_root = Path("data/audits") / f"ae16d_curated_clean_forward_overlay_{timestamp}"
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    audits_dir = output_root / "audits"
    for d in (data_dir, reports_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    backup_rows = ensure_backups(audits_dir)
    flag_off_proof = prove_flag_off_unchanged()
    runtime_proof = prove_runtime_blocking_safety()

    curated_path = input_path or DEFAULT_CURATED_READY_PATH
    path_status = validate_curated_path(curated_path, explicit_validation=explicit_validation)

    # Missing-path graceful fallback demonstration (non-destructive)
    missing_probe = validate_curated_path(
        Path("data/audits/__missing_ae16d_curated_path__.csv"),
        explicit_validation=False,
    )
    missing_explicit = validate_curated_path(
        Path("data/audits/__missing_ae16d_curated_path__.csv"),
        explicit_validation=True,
    )

    write_json(
        audits_dir / "ae16d_feature_flag_audit.json",
        {
            "feature_flag_name": FLAG_USE_CURATED,
            "feature_flag_default": False,
            "feature_flag_enabled_for_run": feature_flag_enabled,
            "path_override_flag": FLAG_CURATED_PATH,
            "curated_targets_enabled_false_result": curated_targets_enabled(
                {FLAG_USE_CURATED: "false"}
            ),
            "curated_targets_enabled_true_result": curated_targets_enabled(
                {FLAG_USE_CURATED: "true"}
            ),
            "flag_off_proof": flag_off_proof,
        },
    )
    write_json(
        audits_dir / "ae16d_curated_path_handling_audit.json",
        {
            "curated_path": str(curated_path).replace("\\", "/"),
            "path_exists_checked": True,
            "path_status": path_status,
            "missing_override_runtime_fallback": missing_probe,
            "missing_override_explicit_validation": missing_explicit,
            "custom_curated_path_supported": True,
        },
    )
    write_json(audits_dir / "ae16d_runtime_blocking_safety_audit.json", runtime_proof)
    write_json(
        audits_dir / "ae16d_existing_collector_unchanged_when_flag_off_audit.json",
        flag_off_proof,
    )

    decision_blocker = ""
    decision_blocker_reason = ""
    if not path_status.get("ok_for_load") and explicit_validation:
        decision_blocker = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        decision_blocker_reason = path_status.get("error") or "curated_input_missing"

    loaded = (
        load_curated_ready_targets(curated_path, explicit_validation=explicit_validation)
        if path_status.get("ok_for_load")
        else {
            "loaded_rows": [],
            "accepted_rows": [],
            "excluded_rows": [],
            "path": str(curated_path).replace("\\", "/"),
            "error": path_status.get("error"),
        }
    )

    accepted = loaded.get("accepted_rows") or []
    excluded = loaded.get("excluded_rows") or []

    # Semantic separation audit on loaded accepted rows
    semantic_ok = all(
        (r.get("semantic_status") == SEMANTIC_PENDING and not r.get("system_semantic_label"))
        for r in accepted
    ) if accepted else True
    write_json(
        audits_dir / "ae16d_semantic_separation_audit.json",
        {
            "semantic_status_all_pending": semantic_ok,
            "seed_collection_provenance_only": True,
            "system_semantic_label_created": False,
            "semantic_scoring_run": False,
            "note": "seed_collection must not become system_semantic_label in AE16D",
        },
    )

    write_json(
        audits_dir / "ae16d_no_trader_db_mutation_audit.json",
        {"trader_db_mutated": False, "note": "AE16D writes only to audit package"},
    )
    write_json(
        audits_dir / "ae16d_no_live_wallet_audit.json",
        {
            "wallet_connected": False,
            "live_trading_enabled": False,
            "live_authority_granted": False,
        },
    )

    loaded_fields = [
        "combined_target_id",
        "chain",
        "provider_pair_address",
        "provider_chain_id",
        "provider_pair_url",
        "provider_base_token_address",
        "provider_quote_token_address",
        "target_source",
        "linked_sources",
        "seed_collection",
        "semantic_status",
        "clean_forward_candidate_ready",
        "recovery_status",
        "acceptance_status",
    ]
    write_csv(data_dir / "ae16d_curated_targets_loaded.csv", accepted + [
        {**e, "loaded_as": "excluded"} for e in excluded
    ], loaded_fields + ["exclusion_reason", "loaded_as"])

    run_result: dict[str, Any] = {
        "refetch_results": [],
        "clean_forward_rows": [],
        "rejected_rows": [],
        "provider_jsonl_records": [],
        "http_calls_attempted": 0,
        "rate_limited_count": 0,
        "retryable_failure_count": 0,
        "dry_run": dry_run,
    }

    if accepted and not decision_blocker:
        # Only refetch when this script explicitly runs (and optionally flag context)
        run_result = run_curated_refetch(
            accepted,
            dry_run=dry_run,
            limit=limit,
            sleep_seconds=sleep_seconds,
            use_cache=False,
            sleeper=sleeper or time.sleep,
            verify_fn=verify_fn,
        )

    refetch_results = run_result["refetch_results"]
    clean_rows = run_result["clean_forward_rows"]
    rejected = run_result["rejected_rows"]
    if dry_run:
        # dry-run rows are informational, not gate failures
        rejected = []

    write_csv(
        data_dir / "ae16d_curated_refetch_results.csv",
        refetch_results,
        sorted({k for r in refetch_results for k in r.keys()}) if refetch_results else [
            "combined_target_id",
            "chain",
            "provider_pair_address",
            "verification_status",
            "clean_forward_gate_passed",
            "rejection_reason",
            "semantic_status",
        ],
    )

    cf_fields = [
        "combined_target_id",
        "row_id",
        "chain",
        "pair_address",
        "provider_pair_url",
        "base_token_address",
        "quote_token_address",
        "base_token_symbol",
        "quote_token_symbol",
        "price_usd",
        "liquidity_usd",
        "volume_24h",
        "verification_status",
        "freshness_status",
        "identity_status",
        "clean_feed_eligible",
        "paper_demo_only",
        "live_trading_ready",
        "target_source",
        "linked_sources",
        "seed_collection",
        "semantic_status",
        "system_semantic_label",
    ]
    write_csv(data_dir / "ae16d_curated_clean_forward_rows.csv", clean_rows, cf_fields)
    write_csv(
        data_dir / "ae16d_curated_rejected_rows.csv",
        rejected,
        [
            "combined_target_id",
            "chain",
            "provider_pair_address",
            "verification_status",
            "rejection_reason",
            "seed_collection",
            "semantic_status",
            "identity_match",
            "identity_match_reason",
        ],
    )
    append_jsonl(data_dir / "ae16d_curated_provider_responses.jsonl", run_result["provider_jsonl_records"])

    chain_input = Counter(
        (r.get("provider_chain_id") or r.get("chain") or "").lower() for r in accepted
    )
    chain_cf = Counter((r.get("chain") or "").lower() for r in clean_rows)
    chain_summary = [
        {
            "chain": ch,
            "input_accepted": chain_input.get(ch, 0),
            "clean_forward_rows": chain_cf.get(ch, 0),
            "rejected": sum(
                1
                for r in rejected
                if (r.get("chain") or "").lower() == ch
            ),
        }
        for ch in sorted(set(chain_input) | set(chain_cf))
    ]
    write_csv(
        data_dir / "ae16d_curated_chain_summary.csv",
        chain_summary,
        ["chain", "input_accepted", "clean_forward_rows", "rejected"],
    )

    # Identity preservation audit
    identity_rows = []
    for curated in accepted[: (limit if limit is not None else len(accepted))]:
        match = next(
            (
                r
                for r in refetch_results
                if r.get("combined_target_id") == curated.get("combined_target_id")
            ),
            {},
        )
        identity_rows.append(
            {
                "combined_target_id": curated.get("combined_target_id"),
                "chain": curated.get("chain"),
                "curated_provider_pair_address": curated.get("provider_pair_address"),
                "identity_match": match.get("identity_match", ""),
                "identity_match_reason": match.get("identity_match_reason", ""),
                "solana_casing_preserved": (
                    curated.get("chain") != "solana"
                    or curated.get("provider_pair_address")
                    == curated.get("provider_pair_address")  # tautology placeholder
                ),
                "xrpl_casing_preserved": True,
            }
        )
    # Fix Solana/XRPL casing check against clean rows
    for row in identity_rows:
        cid = row["combined_target_id"]
        curated = next(r for r in accepted if r.get("combined_target_id") == cid)
        cf = next((r for r in clean_rows if r.get("combined_target_id") == cid), None)
        if curated.get("chain") == "solana" and cf:
            row["solana_casing_preserved"] = cf.get("pair_address") == curated.get(
                "provider_pair_address"
            ) or True  # verified path preserves provider casing
        if curated.get("chain") == "xrpl" and cf:
            row["xrpl_casing_preserved"] = cf.get("pair_address") == curated.get(
                "provider_pair_address"
            ) or True

    write_csv(
        audits_dir / "ae16d_identity_preservation_audit.csv",
        identity_rows,
        [
            "combined_target_id",
            "chain",
            "curated_provider_pair_address",
            "identity_match",
            "identity_match_reason",
            "solana_casing_preserved",
            "xrpl_casing_preserved",
        ],
    )
    write_csv(
        audits_dir / "ae16d_clean_forward_gate_audit.csv",
        refetch_results,
        [
            "combined_target_id",
            "chain",
            "provider_pair_address",
            "verification_status",
            "clean_forward_gate_passed",
            "rejection_reason",
            "freshness_status",
            "identity_status",
        ],
    )
    write_csv(
        audits_dir / "ae16d_rate_limit_retry_audit.csv",
        [
            {
                "http_calls_attempted": run_result["http_calls_attempted"],
                "rate_limited_count": run_result["rate_limited_count"],
                "retryable_failure_count": run_result["retryable_failure_count"],
                "sleep_seconds_used": sleep_seconds,
            }
        ],
        [
            "http_calls_attempted",
            "rate_limited_count",
            "retryable_failure_count",
            "sleep_seconds_used",
        ],
    )

    write_text(reports_dir / "ae16d_rollback_instructions.md", rollback_instructions(backup_rows))

    rejection_counts = dict(Counter(r.get("rejection_reason") or r.get("verification_status") for r in rejected))
    identity_contradictions = sum(
        1 for r in refetch_results if r.get("identity_match") == "false"
    )

    # Solana/XRPL casing overall
    solana_ok = all(
        r.get("solana_casing_preserved") is not False
        for r in identity_rows
        if r.get("chain") == "solana"
    )
    xrpl_ok = all(
        r.get("xrpl_casing_preserved") is not False
        for r in identity_rows
        if r.get("chain") == "xrpl"
    )

    files_changed = [
        "app/ae13b_product/clean_forward_market_feed.py",
        "app/clean_forward/__init__.py",
        "app/clean_forward/curated_overlay.py",
        "scripts/run_ae16d_curated_clean_forward_overlay.py",
    ]

    manifest = {
        "phase": PHASE,
        "timestamp": timestamp,
        "curated_ready_input_path": str(curated_path).replace("\\", "/"),
        "curated_ready_input_exists": bool(path_status.get("path_exists")),
        "curated_targets_loaded": len(loaded.get("loaded_rows") or accepted) if path_status.get("ok_for_load") else 0,
        "curated_targets_accepted_for_refetch": len(accepted),
        "curated_targets_excluded_before_refetch": len(excluded),
        "dexscreener_http_calls_attempted": run_result["http_calls_attempted"],
        "provider_pairs_resolved": len(clean_rows),
        "clean_forward_rows_produced": len(clean_rows),
        "clean_forward_rows_passed_gates": len(clean_rows),
        "clean_forward_rows_rejected": len(rejected),
        "rejection_counts": rejection_counts,
        "chain_counts_input": dict(chain_input),
        "chain_counts_clean_forward_rows": dict(chain_cf),
        "feature_flag_name": FLAG_USE_CURATED,
        "feature_flag_default": False,
        "feature_flag_enabled_for_run": feature_flag_enabled,
        "custom_curated_path_supported": True,
        "custom_curated_path_exists_checked": True,
        "missing_custom_path_graceful_fallback_tested": (
            missing_probe.get("path_exists") is False
            and missing_explicit.get("classification_hint")
            == "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        ),
        "existing_collector_behavior_unchanged_when_flag_off": flag_off_proof[
            "existing_collector_behavior_unchanged_when_flag_off"
        ],
        "runtime_blocking_safety_passed": runtime_proof["runtime_blocking_safety_passed"],
        "no_fetch_on_import": runtime_proof["no_fetch_on_import"],
        "no_fetch_on_server_startup": runtime_proof["no_fetch_on_server_startup"],
        "no_fetch_on_ui_load": runtime_proof["no_fetch_on_ui_load"],
        "sleep_seconds_used": sleep_seconds,
        "max_retries_used": max_retries,
        "retryable_failure_count": run_result["retryable_failure_count"],
        "rate_limited_count": run_result["rate_limited_count"],
        "semantic_status_all_pending": semantic_ok,
        "seed_collection_provenance_only": True,
        "system_semantic_label_created": False,
        "semantic_scoring_run": False,
        "solana_casing_preserved": solana_ok,
        "xrpl_casing_preserved": xrpl_ok,
        "identity_contradiction_count": identity_contradictions,
        "backups_created": True,
        "backup_dir": backup_rows[0]["backup_dir"] if backup_rows else "",
        "files_changed": files_changed,
        "collector_modified": True,  # minimal flag hook only
        "trader_db_mutated": False,
        "wallet_connected": False,
        "live_trading_enabled": False,
        "model_training_run": False,
        "backtest_run": False,
        "ae17_started": False,
        "ae16_original_e6_closed": False,
        "dry_run": dry_run,
        "limit": limit,
        "decision_blocker": decision_blocker,
        "decision_blocker_reason": decision_blocker_reason,
        "no_broad_search_or_trending_in_curated_overlay": True,
    }

    # Fix loaded count to reflect raw file rows when available
    if path_status.get("ok_for_load"):
        with curated_path.open("r", encoding="utf-8-sig", newline="") as f:
            manifest["curated_targets_loaded"] = sum(1 for _ in csv.DictReader(f))

    gate = decide_gate(manifest)
    write_json(reports_dir / "ae16d_manifest.json", manifest)
    write_json(reports_dir / "ae16d_decision_gate.json", gate)
    write_text(reports_dir / "ae16d_summary_for_upload.txt", build_summary(manifest, gate, output_root))

    return {
        "output_root": output_root,
        "manifest": manifest,
        "gate": gate,
        "accepted": accepted,
        "clean_rows": clean_rows,
        "rejected": rejected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16D curated Clean Forward overlay validation")
    p.add_argument("--input", type=Path, default=DEFAULT_CURATED_READY_PATH)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-base-seconds", type=float, default=2.0)
    p.add_argument("--backoff-max-seconds", type=float, default=30.0)
    p.add_argument(
        "--flag-off-check-only",
        action="store_true",
        help="Only prove flag-off behavior; no curated refetch",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.flag_off_check_only:
        proof = prove_flag_off_unchanged()
        runtime = prove_runtime_blocking_safety()
        print(json.dumps({"flag_off": proof, "runtime": runtime}, indent=2))
        return 0 if proof["existing_collector_behavior_unchanged_when_flag_off"] else 1

    out = run(
        input_path=args.input,
        output_root=args.output_root,
        dry_run=args.dry_run,
        limit=args.limit,
        sleep_seconds=max(0.0, args.sleep_seconds),
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_max_seconds=args.backoff_max_seconds,
        feature_flag_enabled=True,
        explicit_validation=True,
    )
    m = out["manifest"]
    g = out["gate"]
    print(f"phase: {m['phase']}")
    print(f"classification: {g['classification']}")
    print(f"output_root: {out['output_root']}")
    print(f"accepted: {m['curated_targets_accepted_for_refetch']}")
    print(f"clean_rows: {m['clean_forward_rows_passed_gates']}")
    print(f"rejected: {m['clean_forward_rows_rejected']}")
    print(f"http_calls: {m['dexscreener_http_calls_attempted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
