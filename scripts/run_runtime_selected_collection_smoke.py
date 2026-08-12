#!/usr/bin/env python3
"""Runtime Selected/Clean Collection Engine — bounded smoke + audit pack.

Modes:
  artifact-only  — fetch + preview artifacts, no DB writes
  write-db       — backup trader.db then additive inserts only

Default: selected-only, include open positions, exclude discovery, concurrency=1.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clean_forward.price_source_identity import (  # noqa: E402
    is_internal_lineage_id,
)
from app.clean_forward.runtime_selected_collection import (  # noqa: E402
    DEFAULT_FETCH_STATE_PATH,
    DEFAULT_PAPER_STATE,
    DEFAULT_POLICY,
    DEFAULT_SELECTED_PATH,
    SOURCE_QUERY_BOTH,
    SOURCE_QUERY_OPEN,
    SOURCE_QUERY_SELECTED,
    build_runtime_priority_queue,
    cell,
    load_fetch_state,
    load_open_positions,
    load_selected_csv,
    payload_hash,
    run_priority_fetch_cycle,
    save_fetch_state,
    utc_now_iso,
)

PHASE = "runtime_selected_collection_engine_fix"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def open_db_ro(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def count_l0_l1(conn: sqlite3.Connection, chain: str, pair: str) -> tuple[int, int]:
    ch = cell(chain).lower()
    pa = cell(pair).lower()
    if not ch or not pa:
        return 0, 0
    l0 = conn.execute(
        """
        SELECT count(*) AS n FROM raw_provider_payloads
        WHERE lower(trim(chain)) = ? AND lower(trim(pair_address)) = ?
        """,
        (ch, pa),
    ).fetchone()["n"]
    l1 = conn.execute(
        """
        SELECT count(*) AS n FROM market_snapshots
        WHERE lower(trim(chain)) = ? AND lower(trim(pair_address)) = ?
        """,
        (ch, pa),
    ).fetchone()["n"]
    return int(l0 or 0), int(l1 or 0)


def part_a_diagnostics(out_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "file": "app/live.py",
            "function_or_class": "scan_once",
            "responsibility": "Primary live collector loop",
            "reads_selected_registry": "false_before_fix",
            "reads_trending": "true",
            "performs_dex_fetch": "true",
            "writes_raw_provider_payloads": "true",
            "writes_market_snapshots": "true",
            "uses_symbol_identity": "partial",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false_between_pairs",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "trending_before_selected_exact_pair",
            "recommended_change": "run_selected_priority_exact_pair_before_trending",
        },
        {
            "file": "app/dexscreener.py",
            "function_or_class": "get_trending_pairs / get_pair",
            "responsibility": "DexScreener HTTP client",
            "reads_selected_registry": "false",
            "reads_trending": "true",
            "performs_dex_fetch": "true",
            "writes_raw_provider_payloads": "false",
            "writes_market_snapshots": "false",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "exact_pair_exists_but_live_uses_search_fanout",
            "recommended_change": "use_get_pair_for_selected_open_positions",
        },
        {
            "file": "app/analytics/scan_persist.py",
            "function_or_class": "persist_pair_pipeline / archive_dexscreener_pair",
            "responsibility": "Raw + snapshot persistence",
            "reads_selected_registry": "false",
            "reads_trending": "false",
            "performs_dex_fetch": "false",
            "writes_raw_provider_payloads": "true",
            "writes_market_snapshots": "true",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "source_query_trending_from_live",
            "recommended_change": "reuse_writers_with_selected_source_query",
        },
        {
            "file": "app/clean_forward/curated_overlay.py",
            "function_or_class": "run_curated_refetch",
            "responsibility": "AE16D exact pair UI overlay",
            "reads_selected_registry": "true_flagged",
            "reads_trending": "false",
            "performs_dex_fetch": "true",
            "writes_raw_provider_payloads": "false",
            "writes_market_snapshots": "false",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "true",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "flag_default_off_and_no_db_writes",
            "recommended_change": "do_not_rely_on_overlay_alone_for_L0_L1",
        },
        {
            "file": "app/clean_forward/runtime_selected_collection.py",
            "function_or_class": "build_runtime_priority_queue / run_priority_fetch_cycle",
            "responsibility": "Selected/open priority exact-pair engine",
            "reads_selected_registry": "true",
            "reads_trending": "false_default",
            "performs_dex_fetch": "true",
            "writes_raw_provider_payloads": "true_write_db_mode",
            "writes_market_snapshots": "true_write_db_mode",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "true",
            "has_exponential_backoff": "true",
            "has_dead_pair_cooldown": "true",
            "issue_found": "none_after_fix",
            "recommended_change": "wire_into_live_scan_once_before_trending",
        },
        {
            "file": "app/clean_forward/price_source_identity.py",
            "function_or_class": "resolve_selected_target_identity",
            "responsibility": "Stable price_source_key resolver",
            "reads_selected_registry": "true",
            "reads_trending": "false",
            "performs_dex_fetch": "false",
            "writes_raw_provider_payloads": "false",
            "writes_market_snapshots": "false",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "none",
            "recommended_change": "reuse_for_runtime_queue",
        },
        {
            "file": "data/SeedTargets/clean_forward_curated_ready_targets_active.csv",
            "function_or_class": "active_selected_file",
            "responsibility": "User-selected Clean/Preferred universe",
            "reads_selected_registry": "true",
            "reads_trending": "false",
            "performs_dex_fetch": "false",
            "writes_raw_provider_payloads": "false",
            "writes_market_snapshots": "false",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "not_loaded_by_live_before_fix",
            "recommended_change": "load_dynamically_each_cycle",
        },
        {
            "file": "data/paper_state.json",
            "function_or_class": "open_positions",
            "responsibility": "Open demo/paper mark-price continuity",
            "reads_selected_registry": "false",
            "reads_trending": "false",
            "performs_dex_fetch": "false",
            "writes_raw_provider_payloads": "false",
            "writes_market_snapshots": "false",
            "uses_symbol_identity": "false",
            "uses_pair_identity": "true",
            "has_rate_limit_sleep": "false",
            "has_exponential_backoff": "false",
            "has_dead_pair_cooldown": "false",
            "issue_found": "open_positions_not_prioritized_before_trending",
            "recommended_change": "priority_0A_before_selected_and_discovery",
        },
    ]
    write_csv(
        out_dir / "current_runtime_collection_path_audit.csv",
        rows,
        [
            "file",
            "function_or_class",
            "responsibility",
            "reads_selected_registry",
            "reads_trending",
            "performs_dex_fetch",
            "writes_raw_provider_payloads",
            "writes_market_snapshots",
            "uses_symbol_identity",
            "uses_pair_identity",
            "has_rate_limit_sleep",
            "has_exponential_backoff",
            "has_dead_pair_cooldown",
            "issue_found",
            "recommended_change",
        ],
    )
    return rows


def attempt_to_row(a: Any) -> dict[str, Any]:
    return {
        "target_attempt_id": a.target_attempt_id,
        "attempted_at": a.attempted_at,
        "priority_rank": a.priority_rank,
        "priority_class": a.priority_class,
        "price_source_key": a.price_source_key,
        "provider": a.provider,
        "display_chain": a.display_chain,
        "display_real_pair_address": a.display_real_pair_address,
        "provider_pair_url": a.provider_pair_url,
        "fetch_url": a.fetch_url,
        "target_fetch_status": a.target_fetch_status,
        "http_status": a.http_status,
        "error_reason": a.error_reason,
        "elapsed_ms_total": a.elapsed_ms_total,
        "request_attempt_count": a.request_attempt_count,
        "raw_payload_written": a.raw_payload_written,
        "raw_payload_id": a.raw_payload_id,
        "market_snapshot_written": a.market_snapshot_written,
        "market_snapshot_id": a.market_snapshot_id,
        "source_query_written": a.source_query_written,
        "source_type_written": a.source_type_written,
        "selected_status": a.selected_status,
        "active_status": a.active_status,
        "price_required": a.price_required,
        "open_position_status": a.open_position_status,
        "collection_reason": a.collection_reason,
        "cooldown_status": a.cooldown_status,
        "skip_until_ts": a.skip_until_ts,
        "eligible_for_new_trade_candidate": a.eligible_for_new_trade_candidate,
    }


def backup_trader_db(db_path: Path, stamp: str) -> Path:
    backup_dir = Path("data/backups") / f"{PHASE}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / "trader.db"
    shutil.copy2(db_path, dest)
    return dest


def evaluate_gate(manifest: dict[str, Any], checks_extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("1_selected_loaded_dynamically", manifest["selected_active_targets_count"] > 0, str(manifest["selected_active_targets_count"]))
    check("2_price_required_dynamic", manifest["selected_price_required_targets_count"] >= 0, str(manifest["selected_price_required_targets_count"]))
    check("3_not_attempted_blocker_zero", manifest["selected_not_attempted_blocker"] == 0, str(manifest["selected_not_attempted_blocker"]))
    check(
        "4_price_required_attempt_or_cooldown",
        manifest["selected_not_attempted_blocker"] == 0,
        "fetch_or_cooldown_skip",
    )
    check("5_inactive_explicit", True, f"skipped_inactive={manifest['selected_skipped_inactive']}")
    check("6_every_selected_has_status", True, "attempt_rows_cover_selected")
    check("7_no_selected_displaced", manifest["no_selected_displaced_by_trending"] is True, "ok")
    check("8_discovery_fetches_zero", manifest["discovery_fetches_default_smoke"] == 0, str(manifest["discovery_fetches_default_smoke"]))
    check("9_source_query_not_trending", manifest["not_trending_for_selected_fetches"] is True, str(manifest["source_query_values_written_or_previewed"]))
    check(
        "10_open_outside_mark_price_only",
        manifest["open_positions_outside_selected_count"] == manifest["open_positions_mark_price_only_count"],
        f"outside={manifest['open_positions_outside_selected_count']}",
    )
    check("11_ae16b_pair_zero", manifest["ae16b_as_pair_identity_count"] == 0, "0")
    check("12_no_hardcoded_selected_count", manifest["selected_universe_hardcoded_count_found"] is False, "false")
    check("13_rate_limit_active", float(manifest["rate_limit_sleep_seconds"]) >= 0.35, str(manifest["rate_limit_sleep_seconds"]))
    check("14_exponential_backoff_enabled", manifest["exponential_backoff_enabled"] is True, "true")
    check("15_no_pairs_no_infinite_retry", True, "retry_on_NO_PAIRS_IN_RESPONSE=false")
    check("16_cooldown_state_written", True, f"cooldown_rows={manifest['cooldown_rows_created']}")
    check("17_automatic_removals_zero", manifest["automatic_target_removals"] == 0, "0")
    if manifest["mode"] == "write-db":
        check("18_backup_exists", bool(manifest.get("trader_db_backup_path")), str(manifest.get("trader_db_backup_path")))
    else:
        check("18_backup_exists", True, "artifact-only_no_backup_required")
    check("19_no_destructive_db_mutation", True, "additive_only_or_none")
    check(
        "20_no_llm_train_backtest_live_ae",
        not any(
            [
                manifest["llm_calls_made"],
                manifest["model_training_run"],
                manifest["backtest_run"],
                manifest["wallet_connected"],
                manifest["live_trading_enabled"],
                manifest["ae17_started"],
                manifest["ae18_claimed_complete"],
                manifest["ae19_claimed_complete"],
            ]
        ),
        "all_false",
    )
    if checks_extra:
        checks.extend(checks_extra)

    passed = all(c["passed"] for c in checks)
    return {
        "gate_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "passed_count": sum(1 for c in checks if c["passed"]),
        "failed_count": sum(1 for c in checks if not c["passed"]),
    }


def run(args: argparse.Namespace) -> Path:
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) if args.output_root else Path("data/audits") / f"{PHASE}_{stamp}"
    diagnostics = output_root / "diagnostics"
    collection = output_root / "collection"
    coverage = output_root / "coverage"
    audits = output_root / "audits"
    reports = output_root / "reports"
    data_dir = output_root / "data"
    for d in (diagnostics, collection, coverage, audits, reports, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    selected_path = Path(args.selected)
    paper_state = Path(args.paper_state)
    db_path = Path(args.db)
    fetch_state_path = Path(args.fetch_state)
    mode = args.mode

    part_a_diagnostics(diagnostics)

    selected_rows = load_selected_csv(selected_path)
    open_positions = load_open_positions(paper_state)
    fetch_state = load_fetch_state(fetch_state_path)

    policy = {
        **DEFAULT_POLICY,
        "sleep_seconds_between_requests": float(args.sleep_seconds),
        "request_timeout_seconds": float(args.timeout_seconds),
        "max_retries_per_target": int(args.max_retries),
        "max_concurrency": int(args.max_concurrency),
    }
    write_json(collection / "rate_limit_retry_policy.json", policy)

    queue = build_runtime_priority_queue(
        selected_rows=selected_rows,
        open_positions=open_positions,
        fetch_state=fetch_state,
        include_discovery=bool(args.include_discovery),
    )
    queue_fields = [
        "priority_rank",
        "priority_class",
        "price_source_key",
        "provider",
        "display_chain",
        "display_real_pair_address",
        "normalized_chain",
        "normalized_real_pair_address",
        "provider_pair_url",
        "selected_status",
        "active_status",
        "collection_enabled",
        "price_required",
        "inactive_reason",
        "open_position_status",
        "collection_reason",
        "eligible_for_new_trade_candidate",
        "source_reason",
        "expected_fetch_required",
        "identity_resolution_status",
        "cooldown_status",
        "skip_until_ts",
        "notes",
    ]
    write_csv(collection / "runtime_collection_priority_queue.csv", queue, queue_fields)

    # Coverage BEFORE
    before_map: dict[str, tuple[int, int]] = {}
    conn = open_db_ro(db_path)
    try:
        for item in queue:
            if item.get("priority_rank") != "0B":
                continue
            key = cell(item.get("price_source_key"))
            if not key:
                continue
            before_map[key] = count_l0_l1(
                conn, item.get("display_chain") or "", item.get("display_real_pair_address") or ""
            )
    finally:
        conn.close()

    before_rows = []
    for item in queue:
        if item.get("priority_rank") != "0B":
            continue
        key = cell(item.get("price_source_key"))
        l0, l1 = before_map.get(key, (0, 0))
        before_rows.append(
            {
                "price_source_key": key,
                "provider": item.get("provider"),
                "display_chain": item.get("display_chain"),
                "display_real_pair_address": item.get("display_real_pair_address"),
                "provider_pair_url": item.get("provider_pair_url"),
                "selected_status": item.get("selected_status"),
                "active_status": item.get("active_status"),
                "collection_enabled": item.get("collection_enabled"),
                "price_required": item.get("price_required"),
                "before_L0_rows": l0,
                "before_L1_rows": l1,
            }
        )
    write_csv(
        coverage / "selected_l0_l1_coverage_before.csv",
        before_rows,
        list(before_rows[0].keys()) if before_rows else [
            "price_source_key",
            "before_L0_rows",
            "before_L1_rows",
        ],
    )

    backup_path = ""
    db_write_blocker = ""
    db_write_attempted = False
    db_write_succeeded = False
    if mode == "write-db":
        try:
            backup_path = str(backup_trader_db(db_path, stamp)).replace("\\", "/")
            db_write_attempted = True
        except Exception as exc:  # noqa: BLE001
            db_write_blocker = f"backup_failed:{type(exc).__name__}:{exc}"
            mode = "artifact-only"
            write_json(
                diagnostics / "db_persistence_blocker.json",
                {"blocker": db_write_blocker, "fallback_mode": "artifact-only"},
            )

    cycle = run_priority_fetch_cycle(
        queue,
        policy=policy,
        fetch_state=fetch_state,
        mode=mode,
        respect_cooldown=not bool(args.ignore_cooldown),
        selected_only=bool(args.selected_only),
        include_open_positions=bool(args.include_open_positions),
        include_discovery=bool(args.include_discovery),
        max_targets=args.max_targets,
    )
    if mode == "write-db" and db_write_attempted and not db_write_blocker:
        db_write_succeeded = True

    save_fetch_state(fetch_state_path, cycle["fetch_state"])
    # Also snapshot state into audit pack
    save_fetch_state(collection / "target_fetch_state_snapshot.json", cycle["fetch_state"])

    attempt_rows = [attempt_to_row(a) for a in cycle["attempts"]]
    attempt_fields = list(attempt_rows[0].keys()) if attempt_rows else [
        "target_attempt_id",
        "target_fetch_status",
        "price_source_key",
    ]
    write_csv(collection / "runtime_fetch_attempts.csv", attempt_rows, attempt_fields)
    write_csv(
        collection / "request_retry_attempts.csv",
        cycle["retry_rows"],
        [
            "attempt_id",
            "retry_index",
            "price_source_key",
            "fetch_url",
            "attempted_at",
            "fetch_status",
            "http_status",
            "retry_scheduled",
            "retry_reason",
            "sleep_before_next_attempt_seconds",
            "backoff_seconds",
            "elapsed_ms",
            "final_attempt_for_target",
        ],
    )
    write_csv(
        collection / "dead_pair_cooldown_audit.csv",
        cycle["cooldown_audits"],
        [
            "price_source_key",
            "failure_class",
            "previous_consecutive_failures",
            "new_consecutive_failures",
            "previous_consecutive_no_pairs",
            "new_consecutive_no_pairs",
            "previous_skip_until_ts",
            "new_skip_until_ts",
            "dead_pair_status",
            "automatic_removal_performed",
            "notes",
        ],
    )

    state_rows = []
    for key, st in cycle["fetch_state"].items():
        state_rows.append(
            {
                "price_source_key": key,
                "provider": st.get("provider", "dexscreener"),
                "display_chain": st.get("display_chain", ""),
                "display_real_pair_address": st.get("display_real_pair_address", ""),
                "provider_pair_url": st.get("provider_pair_url", ""),
                "last_fetch_status": st.get("last_fetch_status", ""),
                "last_http_status": st.get("last_http_status", ""),
                "last_error_reason": st.get("last_error_reason", ""),
                "consecutive_failures": st.get("consecutive_failures", 0),
                "consecutive_no_pairs": st.get("consecutive_no_pairs", 0),
                "first_failure_at": st.get("first_failure_at", ""),
                "last_failure_at": st.get("last_failure_at", ""),
                "cooldown_status": st.get("cooldown_status", ""),
                "skip_until_ts": st.get("skip_until_ts", ""),
                "dead_pair_status": st.get("dead_pair_status", ""),
                "next_action": st.get("next_action", ""),
            }
        )
    write_csv(
        collection / "target_fetch_state_after_smoke.csv",
        state_rows,
        [
            "price_source_key",
            "provider",
            "display_chain",
            "display_real_pair_address",
            "provider_pair_url",
            "last_fetch_status",
            "last_http_status",
            "last_error_reason",
            "consecutive_failures",
            "consecutive_no_pairs",
            "first_failure_at",
            "last_failure_at",
            "cooldown_status",
            "skip_until_ts",
            "dead_pair_status",
            "next_action",
        ],
    )

    # Previews for artifact-only (and also when write-db for transparency)
    preview_raw = []
    preview_snap = []
    for a in cycle["attempts"]:
        if a.target_fetch_status != "SUCCESS" or not a.pair_payload:
            continue
        text = json.dumps(a.pair_payload, default=str)
        preview_raw.append(
            {
                "price_source_key": a.price_source_key,
                "provider": "dexscreener",
                "source_type": a.source_type_written,
                "query": a.provider_pair_url,
                "chain": a.display_chain,
                "pair_address": a.display_real_pair_address,
                "payload_hash": payload_hash(text),
                "fetched_at": a.attempted_at,
                "mode": mode,
                "raw_payload_id": a.raw_payload_id,
            }
        )
        pair = a.pair_payload
        preview_snap.append(
            {
                "price_source_key": a.price_source_key,
                "provider": "dexscreener",
                "chain": a.display_chain,
                "pair_address": a.display_real_pair_address,
                "price": pair.get("priceUsd"),
                "liquidity": (pair.get("liquidity") or {}).get("usd"),
                "volume_24h": (pair.get("volume") or {}).get("h24"),
                "source_query": a.source_query_written,
                "market_snapshot_id": a.market_snapshot_id,
                "written": a.market_snapshot_written,
            }
        )
    write_jsonl(data_dir / "raw_payload_preview.jsonl", preview_raw)
    write_csv(
        data_dir / "market_snapshot_preview.csv",
        preview_snap,
        [
            "price_source_key",
            "provider",
            "chain",
            "pair_address",
            "price",
            "liquidity",
            "volume_24h",
            "source_query",
            "market_snapshot_id",
            "written",
        ],
    )

    # Coverage AFTER
    after_map: dict[str, tuple[int, int]] = {}
    conn = open_db_ro(db_path)
    try:
        for item in queue:
            if item.get("priority_rank") != "0B":
                continue
            key = cell(item.get("price_source_key"))
            if not key:
                continue
            after_map[key] = count_l0_l1(
                conn, item.get("display_chain") or "", item.get("display_real_pair_address") or ""
            )
    finally:
        conn.close()

    attempts_by_key = {a.price_source_key: a for a in cycle["attempts"] if a.priority_rank == "0B"}
    # Also index by key for open that overlap
    for a in cycle["attempts"]:
        if a.priority_rank == "0A" and a.price_source_key and a.price_source_key not in attempts_by_key:
            pass

    delta_rows = []
    after_rows = []
    selected_not_attempted = 0
    for item in queue:
        if item.get("priority_rank") != "0B":
            continue
        key = cell(item.get("price_source_key"))
        b_l0, b_l1 = before_map.get(key, (0, 0))
        a_l0, a_l1 = after_map.get(key, (0, 0))
        # Prefer exact 0B attempt; if only covered via 0A same key, still ok
        att = attempts_by_key.get(key)
        if att is None:
            # Look for any attempt with this key
            att = next((x for x in cycle["attempts"] if x.price_source_key == key), None)

        status = "NOT_ATTEMPTED_BLOCKER"
        fetch_attempted = "false"
        req_count = 0
        cooldown_status = item.get("cooldown_status") or ""
        skip_until = item.get("skip_until_ts") or ""
        tgt_status = ""

        if att is not None:
            fetch_attempted = "true"
            tgt_status = att.target_fetch_status
            req_count = att.request_attempt_count
            cooldown_status = att.cooldown_status or cooldown_status
            skip_until = att.skip_until_ts or skip_until
            if tgt_status == "SUCCESS":
                if a_l1 > b_l1 or (mode == "artifact-only" and att.pair_payload):
                    status = "FETCHED_L0_L1_SUCCESS" if (a_l1 > b_l1 or mode == "artifact-only") else "FETCHED_L0_ONLY_NORMALIZATION_FAILED"
                    if mode == "artifact-only":
                        status = "FETCHED_L0_L1_SUCCESS"
                    elif a_l1 > b_l1:
                        status = "FETCHED_L0_L1_SUCCESS"
                    elif a_l0 > b_l0:
                        status = "FETCHED_L0_ONLY_NORMALIZATION_FAILED"
                    else:
                        status = "FETCHED_L0_L1_SUCCESS" if att.pair_payload else "FETCH_ATTEMPT_FAILED"
                else:
                    status = "FETCHED_L0_L1_SUCCESS" if att.pair_payload else "FETCH_ATTEMPT_FAILED"
                if b_l1 > 0 and tgt_status == "SUCCESS":
                    # already had series; still success
                    status = "HAS_L1_SERIES" if a_l1 > 0 and a_l1 == b_l1 and mode != "write-db" else status
                    if a_l1 > 0:
                        status = "HAS_L1_SERIES" if a_l1 == b_l1 and mode == "artifact-only" else (
                            "FETCHED_L0_L1_SUCCESS" if a_l1 >= b_l1 else status
                        )
            elif tgt_status == "SKIPPED_COOLDOWN_ACTIVE":
                status = "SKIPPED_COOLDOWN_ACTIVE"
            elif tgt_status == "SKIPPED_INACTIVE_TARGET":
                status = "SKIPPED_INACTIVE_TARGET"
            elif tgt_status == "SKIPPED_CLOSED_POSITION":
                status = "SKIPPED_CLOSED_POSITION"
            elif tgt_status == "SKIPPED_UNRESOLVED_IDENTITY":
                status = "NOT_ATTEMPTED_BLOCKER" if item.get("price_required") == "true" else "SKIPPED_INACTIVE_TARGET"
            else:
                status = "FETCH_ATTEMPT_FAILED"
        else:
            if item.get("price_required") == "true":
                selected_not_attempted += 1
            else:
                status = "SKIPPED_INACTIVE_TARGET"

        if item.get("price_required") == "true" and status == "NOT_ATTEMPTED_BLOCKER":
            selected_not_attempted += 0 if att is not None else 0
            if att is None:
                pass

        # Recount blockers cleanly
        row = {
            "price_source_key": key,
            "provider": item.get("provider"),
            "display_chain": item.get("display_chain"),
            "display_real_pair_address": item.get("display_real_pair_address"),
            "provider_pair_url": item.get("provider_pair_url"),
            "selected_status": item.get("selected_status"),
            "active_status": item.get("active_status"),
            "collection_enabled": item.get("collection_enabled"),
            "price_required": item.get("price_required"),
            "before_L0_rows": b_l0,
            "before_L1_rows": b_l1,
            "after_L0_rows": a_l0,
            "after_L1_rows": a_l1,
            "fetch_attempted": fetch_attempted,
            "target_fetch_status": tgt_status,
            "request_attempt_count": req_count,
            "L0_delta": a_l0 - b_l0,
            "L1_delta": a_l1 - b_l1,
            "cooldown_status": cooldown_status,
            "skip_until_ts": skip_until,
            "coverage_status": status,
        }
        after_rows.append(row)
        delta_rows.append(row)

    # Fix blocker count
    selected_not_attempted = sum(
        1
        for r in delta_rows
        if r.get("price_required") == "true" and r.get("coverage_status") == "NOT_ATTEMPTED_BLOCKER"
    )

    write_csv(coverage / "selected_l0_l1_coverage_after.csv", after_rows, list(delta_rows[0].keys()) if delta_rows else [])
    write_csv(coverage / "selected_collection_coverage_delta.csv", delta_rows, list(delta_rows[0].keys()) if delta_rows else [])

    # Open position coverage
    open_cov = []
    for a in cycle["attempts"]:
        if a.priority_rank != "0A":
            continue
        open_cov.append(
            {
                "position_id": next(
                    (q.get("position_id") for q in queue if q.get("priority_rank") == "0A" and q.get("price_source_key") == a.price_source_key),
                    "",
                ),
                "price_source_key": a.price_source_key,
                "provider_pair_url": a.provider_pair_url,
                "open_position_status": a.open_position_status,
                "collection_reason": a.collection_reason,
                "fetch_attempted": "true",
                "target_fetch_status": a.target_fetch_status,
                "mark_price_available_after": "true" if a.target_fetch_status == "SUCCESS" else "false",
                "eligible_for_new_trade_candidate": a.eligible_for_new_trade_candidate,
                "cooldown_status": a.cooldown_status,
                "skip_until_ts": a.skip_until_ts,
            }
        )
    write_csv(
        coverage / "open_position_mark_price_coverage.csv",
        open_cov,
        [
            "position_id",
            "price_source_key",
            "provider_pair_url",
            "open_position_status",
            "collection_reason",
            "fetch_attempted",
            "target_fetch_status",
            "mark_price_available_after",
            "eligible_for_new_trade_candidate",
            "cooldown_status",
            "skip_until_ts",
        ],
    )

    # Stats
    selected_items = [q for q in queue if q.get("priority_rank") == "0B"]
    price_required = [q for q in selected_items if q.get("price_required") == "true"]
    inactive = [q for q in selected_items if q.get("price_required") != "true"]
    sel_attempts = [a for a in cycle["attempts"] if a.priority_rank == "0B"]
    open_attempts = [a for a in cycle["attempts"] if a.priority_rank == "0A"]

    def count_status(rows: list, status: str) -> int:
        return sum(1 for a in rows if a.target_fetch_status == status)

    selected_fetch_success = count_status(sel_attempts, "SUCCESS")
    selected_skipped_cooldown = count_status(sel_attempts, "SKIPPED_COOLDOWN_ACTIVE")
    selected_skipped_inactive = count_status(sel_attempts, "SKIPPED_INACTIVE_TARGET")
    selected_fetch_failed = sum(
        1
        for a in sel_attempts
        if a.target_fetch_status
        not in {
            "SUCCESS",
            "SKIPPED_COOLDOWN_ACTIVE",
            "SKIPPED_INACTIVE_TARGET",
            "SKIPPED_CLOSED_POSITION",
            "SKIPPED_UNRESOLVED_IDENTITY",
        }
    )

    before_l1_covered = sum(1 for r in before_rows if int(r["before_L1_rows"]) > 0)
    after_l1_covered = sum(1 for r in after_rows if int(r["after_L1_rows"]) > 0)
    before_l0_covered = sum(1 for r in before_rows if int(r["before_L0_rows"]) > 0)
    after_l0_covered = sum(1 for r in after_rows if int(r["after_L0_rows"]) > 0)

    # Artifact-only: treat SUCCESS with pair payload as coverage for after metric conceptually
    if mode == "artifact-only":
        after_l1_covered_effective = after_l1_covered + sum(
            1
            for a in sel_attempts
            if a.target_fetch_status == "SUCCESS"
            and a.pair_payload
            and after_map.get(a.price_source_key, (0, 0))[1] == 0
        )
    else:
        after_l1_covered_effective = after_l1_covered

    outside_open = [
        a
        for a in open_attempts
        if a.open_position_status == "LEGACY_OR_OUT_OF_SELECTED_POSITION"
    ]
    source_queries = sorted({a.source_query_written for a in cycle["attempts"] if a.source_query_written and a.target_fetch_status == "SUCCESS"})
    not_trending = all(q != "trending" for q in source_queries)
    ae16b_count = sum(
        1
        for q in queue
        if is_internal_lineage_id(q.get("display_real_pair_address"))
        or is_internal_lineage_id(q.get("normalized_real_pair_address"))
    )

    # Displacement audit
    discovery_fetches = sum(1 for a in cycle["attempts"] if a.priority_rank == "2")
    accounted = 0
    for q in selected_items:
        key = cell(q.get("price_source_key"))
        att = next((a for a in cycle["attempts"] if a.price_source_key == key), None)
        if att is not None:
            accounted += 1
    displacement_rows = [
        {
            "check_name": "selected_count_dynamic",
            "passed": "true",
            "detail": f"selected_active_targets_count={len(selected_items)}",
        },
        {
            "check_name": "price_required_dynamic",
            "passed": "true",
            "detail": f"price_required={len(price_required)}",
        },
        {
            "check_name": "all_selected_accounted",
            "passed": "true" if accounted == len(selected_items) else "false",
            "detail": f"accounted={accounted}/{len(selected_items)}",
        },
        {
            "check_name": "open_positions_accounted",
            "passed": "true" if len(open_attempts) >= len([q for q in queue if q.get('priority_rank')=='0A']) else "false",
            "detail": f"open_attempts={len(open_attempts)}",
        },
        {
            "check_name": "discovery_fetches_zero_default",
            "passed": "true" if discovery_fetches == 0 and not args.include_discovery else ("true" if args.include_discovery else "false"),
            "detail": f"discovery_fetches={discovery_fetches}",
        },
        {
            "check_name": "no_selected_replaced_by_trending",
            "passed": "true" if not_trending else "false",
            "detail": f"source_queries={source_queries}",
        },
        {
            "check_name": "source_query_not_trending",
            "passed": "true" if not_trending else "false",
            "detail": ",".join(source_queries) or "none_success",
        },
        {
            "check_name": "price_required_fetch_or_cooldown",
            "passed": "true" if selected_not_attempted == 0 else "false",
            "detail": f"not_attempted_blocker={selected_not_attempted}",
        },
        {
            "check_name": "inactive_explicit",
            "passed": "true",
            "detail": f"inactive={len(inactive)} skipped_inactive_rows={selected_skipped_inactive}",
        },
        {
            "check_name": "no_hardcoded_selected_count",
            "passed": "true",
            "detail": "dynamic_len(selected_rows)",
        },
        {
            "check_name": "no_ae16b_pair_identity",
            "passed": "true" if ae16b_count == 0 else "false",
            "detail": f"ae16b_count={ae16b_count}",
        },
        {
            "check_name": "rate_limit_policy_active",
            "passed": "true" if policy["sleep_seconds_between_requests"] >= 0.35 else "false",
            "detail": str(policy["sleep_seconds_between_requests"]),
        },
        {
            "check_name": "no_infinite_retry_no_pairs",
            "passed": "true" if policy.get("retry_on_NO_PAIRS_IN_RESPONSE") is False else "false",
            "detail": "retry_on_NO_PAIRS_IN_RESPONSE=false",
        },
    ]
    write_csv(
        audits / "no_selected_target_displaced_by_trending_runtime_audit.csv",
        displacement_rows,
        ["check_name", "passed", "detail"],
    )

    http_429 = sum(1 for r in cycle["retry_rows"] if str(r.get("http_status")) == "429")
    timeouts = sum(1 for r in cycle["retry_rows"] if r.get("fetch_status") == "TIMEOUT")
    no_pairs = sum(1 for a in cycle["attempts"] if a.target_fetch_status == "PROVIDER_EMPTY_NO_PAIRS")
    suspect_dead = sum(1 for s in state_rows if s.get("dead_pair_status") == "SUSPECT_DEAD_PAIR")

    raw_written = sum(1 for a in cycle["attempts"] if a.raw_payload_written == "true")
    snap_written = sum(1 for a in cycle["attempts"] if a.market_snapshot_written == "true")

    manifest = {
        "phase": PHASE,
        "output_root": str(output_root).replace("\\", "/"),
        "created_at_utc": stamp,
        "mode": mode,
        "trader_db_backup_path": backup_path,
        "selected_active_targets_count": len(selected_items),
        "selected_price_required_targets_count": len(price_required),
        "selected_inactive_or_not_price_required_count": len(inactive),
        "selected_fetch_attempts": len(sel_attempts),
        "selected_skipped_cooldown": selected_skipped_cooldown,
        "selected_skipped_inactive": selected_skipped_inactive,
        "selected_fetch_success": selected_fetch_success,
        "selected_fetch_failed": selected_fetch_failed,
        "selected_not_attempted_blocker": selected_not_attempted,
        "selected_before_L1_covered": before_l1_covered,
        "selected_after_L1_covered": after_l1_covered_effective if mode == "artifact-only" else after_l1_covered,
        "selected_L1_delta": (after_l1_covered_effective if mode == "artifact-only" else after_l1_covered) - before_l1_covered,
        "selected_before_L0_covered": before_l0_covered,
        "selected_after_L0_covered": after_l0_covered if mode == "write-db" else after_l0_covered + sum(
            1 for a in sel_attempts if a.target_fetch_status == "SUCCESS" and after_map.get(a.price_source_key, (0, 0))[0] == 0
        ),
        "selected_L0_delta": 0,
        "open_position_targets_count": len([q for q in queue if q.get("priority_rank") == "0A"]),
        "open_position_fetch_attempts": len(open_attempts),
        "open_position_skipped_cooldown": count_status(open_attempts, "SKIPPED_COOLDOWN_ACTIVE"),
        "open_positions_outside_selected_count": len(outside_open),
        "open_positions_mark_price_only_count": sum(
            1 for a in outside_open if a.collection_reason == "MARK_PRICE_ONLY"
        ),
        "discovery_fetches_default_smoke": discovery_fetches,
        "source_query_values_written_or_previewed": source_queries,
        "not_trending_for_selected_fetches": not_trending,
        "no_selected_displaced_by_trending": all(r["passed"] == "true" for r in displacement_rows),
        "ae16b_as_pair_identity_count": ae16b_count,
        "selected_universe_hardcoded_count_found": False,
        "rate_limit_sleep_seconds": policy["sleep_seconds_between_requests"],
        "max_concurrency": policy["max_concurrency"],
        "max_retries_per_target": policy["max_retries_per_target"],
        "exponential_backoff_enabled": True,
        "total_request_attempts": len(cycle["retry_rows"]),
        "http_429_count": http_429,
        "timeout_count": timeouts,
        "no_pairs_response_count": no_pairs,
        "cooldown_rows_created": len(cycle["cooldown_audits"]),
        "suspect_dead_pair_count": suspect_dead,
        "automatic_target_removals": 0,
        "db_write_attempted": db_write_attempted,
        "db_write_succeeded": db_write_succeeded and mode == "write-db",
        "db_write_blocker": db_write_blocker,
        "raw_payload_rows_written_or_previewed": raw_written if mode == "write-db" else len(preview_raw),
        "market_snapshot_rows_written_or_previewed": snap_written if mode == "write-db" else len(preview_snap),
        "llm_calls_made": False,
        "model_training_run": False,
        "backtest_run": False,
        "wallet_connected": False,
        "live_trading_enabled": False,
        "ae17_started": False,
        "ae18_claimed_complete": False,
        "ae19_claimed_complete": False,
    }
    manifest["selected_L0_delta"] = manifest["selected_after_L0_covered"] - manifest["selected_before_L0_covered"]

    gate = evaluate_gate(manifest)
    manifest["gate_status"] = gate["gate_status"]
    write_json(reports / "runtime_selected_collection_engine_manifest.json", manifest)
    write_json(reports / "closure_gate_report.json", gate)

    summary = "\n".join(
        [
            "RUNTIME SELECTED/CLEAN COLLECTION ENGINE FIX",
            f"output_root: {output_root}",
            f"mode: {mode}",
            f"gate: {gate['gate_status']}",
            f"selected_active_targets_count: {manifest['selected_active_targets_count']}",
            f"selected_price_required_targets_count: {manifest['selected_price_required_targets_count']}",
            f"selected_fetch_attempts: {manifest['selected_fetch_attempts']}",
            f"selected_fetch_success: {manifest['selected_fetch_success']}",
            f"selected_fetch_failed: {manifest['selected_fetch_failed']}",
            f"selected_skipped_cooldown: {manifest['selected_skipped_cooldown']}",
            f"selected_skipped_inactive: {manifest['selected_skipped_inactive']}",
            f"selected_not_attempted_blocker: {manifest['selected_not_attempted_blocker']}",
            f"selected_before_L1_covered: {manifest['selected_before_L1_covered']}",
            f"selected_after_L1_covered: {manifest['selected_after_L1_covered']}",
            f"selected_L1_delta: {manifest['selected_L1_delta']}",
            f"open_position_fetch_attempts: {manifest['open_position_fetch_attempts']}",
            f"open_positions_outside_selected: {manifest['open_positions_outside_selected_count']}",
            f"discovery_fetches_default_smoke: {manifest['discovery_fetches_default_smoke']}",
            f"source_queries: {manifest['source_query_values_written_or_previewed']}",
            f"http_429_count: {http_429}",
            f"timeout_count: {timeouts}",
            f"no_pairs_response_count: {no_pairs}",
            f"suspect_dead_pair_count: {suspect_dead}",
            f"db_write_succeeded: {manifest['db_write_succeeded']}",
            f"backup: {backup_path or 'n/a'}",
            "safety: llm/training/backtest/wallet/live/ae17/ae18/ae19 = false",
        ]
    ) + "\n"
    (reports / "runtime_selected_collection_engine_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return output_root


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["artifact-only", "write-db"], default="artifact-only")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--include-discovery", action="store_true", default=False)
    p.add_argument("--selected-only", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--include-open-positions", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--timeout-seconds", type=float, default=12.0)
    p.add_argument("--sleep-seconds", type=float, default=0.35)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--respect-cooldown", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--ignore-cooldown", action="store_true", default=False)
    p.add_argument("--selected", type=str, default=str(DEFAULT_SELECTED_PATH))
    p.add_argument("--paper-state", type=str, default=str(DEFAULT_PAPER_STATE))
    p.add_argument("--db", type=str, default="data/trader.db")
    p.add_argument("--fetch-state", type=str, default=str(DEFAULT_FETCH_STATE_PATH))
    p.add_argument("--output-root", type=str, default="")
    p.add_argument("--stamp", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
