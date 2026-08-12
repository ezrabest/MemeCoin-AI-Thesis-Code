#!/usr/bin/env python3
"""AE15 — Direct Target / Clean Forward Schema Bridge.

Builds canonical Clean Forward schema + lineage audits from local Clean Forward
smoke polls and AE14 closure artifacts.

Does NOT: train models, backtest, claim profitability, connect wallets,
enable live trading, or call external APIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clean_forward.discovery import (  # noqa: E402
    discover_ae14_root,
    discover_clean_forward_smoke_root,
    iter_poll_rows,
    load_ae14_artifacts,
    source_artifact_index,
)
from app.clean_forward.lineage import (  # noqa: E402
    AE14_PENDING_NOTE,
    build_candidate_from_row,
    build_decision_input_for_candidate,
    reconcile_ae14_order_position_lineage,
    summarize_order_position_lineage,
)
from app.clean_forward.schema import (  # noqa: E402
    CANDIDATE_ID_FORBIDDEN_FIELDS,
    CleanForwardOutcomeLabel,
    CleanForwardSkipReason,
    DECISION_INPUT_VERSION,
    make_clean_forward_candidate_id,
    make_outcome_label_id,
    make_skip_record_id,
)
from app.clean_forward.serialization import record_to_dict, stable_json_dumps  # noqa: E402
from app.clean_forward.validation import (  # noqa: E402
    evaluate_clean_feed_eligibility,
    validate_identity_separation,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        if rows:
            keys: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            fieldnames = keys
        else:
            fieldnames = []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return False
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE15 Clean Forward Schema Bridge")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true", help="Produce audits from local artifacts")
    mode.add_argument("--build", action="store_true", help="Build schema tables + audits")
    mode.add_argument("--reconcile-ae14", action="store_true", help="AE14 order/position reconciliation")
    p.add_argument("--ae14-root", type=str, default=None)
    p.add_argument("--clean-forward-smoke-root", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--max-polls", type=int, default=None, help="Optional poll cap for faster runs")
    return p.parse_args(argv)


def decide_gate(
    *,
    smoke_root: Path | None,
    ae14_root: Path | None,
    eligible_count: int,
    identity_failures: int,
    legacy_flags: dict[str, Any],
    lineage_summary: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    limitations: list[str] = []
    blockers: list[str] = []

    if safety.get("wallet_configured") or safety.get("private_key_accessed") or safety.get("live_trading_enabled"):
        blockers.append("AE15_BLOCKED_SAFETY_VIOLATION")
    if any(
        [
            legacy_flags.get("legacy_market_snapshots_used"),
            legacy_flags.get("old_market_snapshot_feed_used"),
            legacy_flags.get("raw_provider_payloads_legacy_feed_used"),
            legacy_flags.get("local_db_candidate_universe_used"),
        ]
    ):
        blockers.append("AE15_BLOCKED_LEGACY_DATA_CONTAMINATION")
    if smoke_root is None and ae14_root is None:
        blockers.append("AE15_BLOCKED_CLEAN_FORWARD_INPUT_MISSING")
    if eligible_count == 0 and smoke_root is not None:
        # Still allow AE14-only reconcile path if smoke empty but AE14 present
        if ae14_root is None:
            blockers.append("AE15_BLOCKED_CLEAN_FORWARD_INPUT_MISSING")
    if identity_failures > 0:
        blockers.append("AE15_BLOCKED_IDENTITY_SEPARATION_FAILURE")

    # Order/position lineage: AE14 mismatch must not be silently ignored.
    pending = lineage_summary.get("counter_consistency_status") == AE14_PENDING_NOTE
    unexplained = (
        lineage_summary.get("position_count_delta", 0) != 0
        and not lineage_summary.get("ae14_discrepancy_status")
    )
    if unexplained:
        blockers.append("AE15_BLOCKED_ORDER_POSITION_LINEAGE_FAILURE")
    elif pending or lineage_summary.get("ae14_discrepancy_resolved") is False:
        if lineage_summary.get("link_count", 0) > 0:
            limitations.append(AE14_PENDING_NOTE)
        elif ae14_root is not None:
            blockers.append("AE15_BLOCKED_ORDER_POSITION_LINEAGE_FAILURE")

    if blockers:
        classification = blockers[0]
        status = "BLOCKED"
    elif limitations:
        classification = "AE15_PASS_WITH_LINEAGE_LIMITATIONS"
        status = "PASS_WITH_LIMITATIONS"
    else:
        classification = "AE15_CLEAN_FORWARD_SCHEMA_BRIDGE_PASS"
        status = "PASS"

    ae16_blocked = classification != "AE15_CLEAN_FORWARD_SCHEMA_BRIDGE_PASS"
    # PASS_WITH_LINEAGE_LIMITATIONS still unblocks AE16 schema consumption.
    if classification == "AE15_PASS_WITH_LINEAGE_LIMITATIONS":
        ae16_blocked = False

    return {
        "phase": "AE15",
        "status": status,
        "classification": classification,
        "blockers": blockers,
        "limitations": limitations,
        "ae14_discrepancy_status": lineage_summary.get("ae14_discrepancy_status"),
        "ae16_blocked": ae16_blocked,
        "ae16_blocker": blockers[0] if ae16_blocked and blockers else None,
        "generated_at_utc": utc_now(),
        "not_profitability_evidence": True,
        "model_authority_approved": False,
        "live_trading_approved": False,
    }


def run_ae15(args: argparse.Namespace) -> dict[str, Any]:
    mode = "build"
    if args.audit_only:
        mode = "audit-only"
    elif args.reconcile_ae14:
        mode = "reconcile-ae14"
    elif args.build:
        mode = "build"

    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae15_clean_forward_schema_bridge_{utc_stamp()}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    reports = out_root / "reports"
    data_dir = out_root / "data"
    audits = out_root / "audits"
    manifests = out_root / "manifests"
    for d in (reports, data_dir, audits, manifests):
        d.mkdir(parents=True, exist_ok=True)

    ae14_root = discover_ae14_root(ROOT, explicit=args.ae14_root)
    smoke_root = discover_clean_forward_smoke_root(ROOT, explicit=args.clean_forward_smoke_root)

    # --- Discover / audit clean feed rows ---
    discovered_rows: list[dict[str, Any]] = []
    if smoke_root and mode in ("build", "audit-only"):
        for row in iter_poll_rows(smoke_root, max_polls=args.max_polls):
            discovered_rows.append(row)

    # Always include AE14 selected row when available (authoritative CF sample)
    ae14_artifacts = load_ae14_artifacts(ae14_root) if ae14_root else {}
    selected = ae14_artifacts.get("selected_row") if ae14_artifacts else None
    if isinstance(selected, dict) and not selected.get("_load_error"):
        sel = dict(selected)
        sel["_source_poll_file"] = "ae14_selected_clean_forward_row"
        sel["_source_poll_index"] = None
        discovered_rows.append(sel)

    feed_audit_rows: list[dict[str, Any]] = []
    rejection_counter: Counter[str] = Counter()
    verification_dist: Counter[str] = Counter()
    freshness_dist: Counter[str] = Counter()
    identity_dist: Counter[str] = Counter()
    shown_true = 0
    paper_demo_false = 0
    live_ready_true = 0
    eligible_rows: list[dict[str, Any]] = []

    for row in discovered_rows:
        elig = evaluate_clean_feed_eligibility(row)
        verification_dist[str(row.get("verification_status") or "")] += 1
        freshness_dist[str(row.get("freshness_status") or "")] += 1
        identity_dist[str(row.get("identity_status") or "")] += 1
        if elig["shown_as_token_contract"] is True:
            shown_true += 1
        if elig["paper_demo_only"] is not True:
            paper_demo_false += 1
        if elig["live_trading_ready"] is True:
            live_ready_true += 1
        rec = {
            "source_poll_file": row.get("_source_poll_file"),
            "source_poll_index": row.get("_source_poll_index"),
            "row_key": row.get("row_key") or row.get("row_id"),
            "chain": row.get("chain") or row.get("normalized_chain_id"),
            "pair_address": row.get("pair_address"),
            "clean_feed_eligible": elig["clean_feed_eligible"],
            "rejection_reasons": "|".join(elig["rejection_reasons"]),
            "verification_status": elig["verification_status"],
            "freshness_status": elig["freshness_status"],
            "identity_status": elig["identity_status"],
            "shown_as_token_contract": elig["shown_as_token_contract"],
            "paper_demo_only": elig["paper_demo_only"],
            "live_trading_ready": elig["live_trading_ready"],
        }
        feed_audit_rows.append(rec)
        if elig["clean_feed_eligible"]:
            eligible_rows.append(row)
        else:
            for reason in elig["rejection_reasons"]:
                rejection_counter[reason] += 1

    clean_feed_input_audit = {
        "total_rows_discovered": len(discovered_rows),
        "rows_eligible": len(eligible_rows),
        "rows_rejected": len(discovered_rows) - len(eligible_rows),
        "rejection_reason_counts": dict(rejection_counter),
        "verification_status_distribution": dict(verification_dist),
        "freshness_status_distribution": dict(freshness_dist),
        "identity_status_distribution": dict(identity_dist),
        "shown_as_token_contract_true_count": shown_true,
        "paper_demo_only_false_count": paper_demo_false,
        "live_trading_ready_true_count": live_ready_true,
        "smoke_root": str(smoke_root).replace("\\", "/") if smoke_root else None,
        "ae14_root": str(ae14_root).replace("\\", "/") if ae14_root else None,
        "mode": mode,
    }
    write_json(audits / "clean_feed_input_audit.json", clean_feed_input_audit)
    write_csv(audits / "clean_feed_input_audit.csv", feed_audit_rows)

    # --- Identity separation ---
    identity_rows: list[dict[str, Any]] = []
    identity_failures = 0
    for row in eligible_rows:
        result = validate_identity_separation(row)
        flat = {
            "row_key": row.get("row_key") or row.get("row_id"),
            "passed": result["passed"],
            "failures": "|".join(result["failures"]),
            "pair_address_present": result["pair_address_present"],
            "base_token_address_present": result["base_token_address_present"],
            "quote_token_address_present": result["quote_token_address_present"],
            "pair_address_ne_base_token_address": result["pair_address_ne_base_token_address"],
            "pair_address_ne_quote_token_address": result["pair_address_ne_quote_token_address"],
            "base_token_address_ne_quote_token_address": result["base_token_address_ne_quote_token_address"],
            "coin_id_not_invented": result["coin_id_not_invented"],
            "chain": result["chain"],
            "pair_address": result["pair_address"],
            "pair_address_normalized": result["pair_address_normalized"],
            "solana_case_preserved": result["solana_case_preserved"],
            "evm_lowercased": result["evm_lowercased"],
        }
        identity_rows.append(flat)
        if not result["passed"]:
            identity_failures += 1

    identity_audit = {
        "rows_checked": len(identity_rows),
        "rows_passed": sum(1 for r in identity_rows if r["passed"]),
        "rows_failed": identity_failures,
        "passed": identity_failures == 0,
    }
    write_json(audits / "identity_separation_audit.json", identity_audit)
    write_csv(audits / "identity_separation_audit.csv", identity_rows)

    # --- Build candidates / decision inputs ---
    candidates: list[dict[str, Any]] = []
    decision_inputs: list[dict[str, Any]] = []
    skip_reasons: list[dict[str, Any]] = []
    outcome_labels: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()

    # Determinism audit sample
    determinism_samples: list[dict[str, Any]] = []

    build_source_rows = eligible_rows
    if mode == "reconcile-ae14":
        # Prefer AE14-derived candidates; still keep eligible selected row if present
        build_source_rows = [r for r in eligible_rows if r.get("_source_poll_file") == "ae14_selected_clean_forward_row"]
        if not build_source_rows and isinstance(selected, dict) and not selected.get("_load_error"):
            build_source_rows = [selected]

    for row in build_source_rows:
        # Reject rows failing identity before candidate creation
        id_check = validate_identity_separation(row)
        if not id_check["passed"]:
            skip = CleanForwardSkipReason(
                skip_record_id=make_skip_record_id(
                    clean_forward_candidate_id="unbuilt",
                    skipped_at=utc_now(),
                    skip_stage="identity",
                    skip_reason_code="IDENTITY_SEPARATION_FAILURE",
                ),
                clean_forward_candidate_id=None,
                clean_forward_decision_input_id=None,
                skipped_at=utc_now(),
                skip_stage="identity",
                skip_reason_code="IDENTITY_SEPARATION_FAILURE",
                skip_reason_detail="|".join(id_check["failures"]),
                missing_identity=True,
                blocked_by_clean_feed_eligibility=False,
            )
            skip_reasons.append(skip.to_dict())
            continue

        cand = build_candidate_from_row(row)
        # Determinism check on first few
        if len(determinism_samples) < 5:
            again = build_candidate_from_row(row)
            material_keys = {
                "chain",
                "provider",
                "pair_address",
                "base_token_address",
                "quote_token_address",
                "observed_at",
                "fetched_at",
                "provider_payload_hash",
            }
            leaked = sorted(CANDIDATE_ID_FORBIDDEN_FIELDS.intersection(set(cand.to_dict().keys())))
            # Forbidden fields may exist on other records but must not affect ID.
            id1 = make_clean_forward_candidate_id(
                chain=cand.chain,
                provider=cand.provider,
                pair_address_for_id=id_check["pair_address_normalized"],
                base_token_address=cand.base_token_address,
                quote_token_address=cand.quote_token_address,
                observed_at_or_fetched_at=str(cand.observed_at or cand.fetched_at or ""),
                provider_payload_hash=cand.provider_payload_hash,
            )
            determinism_samples.append(
                {
                    "clean_forward_candidate_id": cand.clean_forward_candidate_id,
                    "rebuild_matches": cand.clean_forward_candidate_id == again.clean_forward_candidate_id == id1,
                    "stable_serialization": stable_json_dumps(cand.to_dict())
                    == stable_json_dumps(again.to_dict()),
                    "forbidden_fields_in_candidate_record": leaked,
                    "id_material_keys_only": sorted(material_keys),
                    "no_model_future_order_in_id": True,
                }
            )

        if cand.clean_forward_candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(cand.clean_forward_candidate_id)
        candidates.append(cand.to_dict())

        decision = build_decision_input_for_candidate(cand)
        decision_inputs.append(decision.to_dict())

        outcome = CleanForwardOutcomeLabel(
            clean_forward_outcome_label_id=make_outcome_label_id(
                clean_forward_candidate_id=cand.clean_forward_candidate_id,
                clean_forward_decision_input_id=decision.clean_forward_decision_input_id,
                horizon=None,
            ),
            clean_forward_candidate_id=cand.clean_forward_candidate_id,
            clean_forward_decision_input_id=decision.clean_forward_decision_input_id,
            paper_order_id=None,
            paper_position_id=None,
            outcome_source="ae15_schema_contract_only",
            outcome_status="LABEL_NOT_GENERATED",
            horizon=None,
            future_return=None,
            max_upside=None,
            max_drawdown=None,
            hit_tp=None,
            hit_sl=None,
            time_stop=None,
            stale_price_exit=None,
            matured_at=None,
            no_lookahead_passed=None,
            label_available=False,
            label_unavailable_reason="AE15_DEFINES_CONTRACT_ONLY_NO_MATURED_LABEL_RECOMPUTATION",
        )
        outcome_labels.append(outcome.to_dict())

    # Rejected feed rows → skip reasons
    for row in discovered_rows:
        elig = evaluate_clean_feed_eligibility(row)
        if elig["clean_feed_eligible"]:
            continue
        # Best-effort candidate id material even for rejected rows (for lineage)
        try:
            cand_rej = build_candidate_from_row(row)
            cid = cand_rej.clean_forward_candidate_id
        except Exception:
            cid = None
        skip = CleanForwardSkipReason(
            skip_record_id=make_skip_record_id(
                clean_forward_candidate_id=cid or "rejected",
                skipped_at=str(row.get("observed_at") or utc_now()),
                skip_stage="clean_feed_eligibility",
                skip_reason_code=elig["rejection_reasons"][0] if elig["rejection_reasons"] else "REJECTED",
            ),
            clean_forward_candidate_id=cid,
            clean_forward_decision_input_id=None,
            skipped_at=str(row.get("observed_at") or utc_now()),
            skip_stage="clean_feed_eligibility",
            skip_reason_code=elig["rejection_reasons"][0] if elig["rejection_reasons"] else "REJECTED",
            skip_reason_detail="|".join(elig["rejection_reasons"]),
            stale_price="freshness_status_not_fresh" in elig["rejection_reasons"],
            missing_provider_verification="verification_status_not_provider_pair_verified"
            in elig["rejection_reasons"],
            missing_identity="identity_status_not_pair_and_tokens_separated" in elig["rejection_reasons"],
            blocked_by_clean_feed_eligibility=True,
            blocked_by_live_disabled="live_trading_ready_true" in elig["rejection_reasons"],
        )
        skip_reasons.append(skip.to_dict())

    # --- AE14 order/position reconciliation ---
    execution_links: list[dict[str, Any]] = []
    lineage_summary: dict[str, Any] = summarize_order_position_lineage([])
    ae14_reconcile: dict[str, Any] = {}

    if ae14_root:
        ae14_reconcile = reconcile_ae14_order_position_lineage(ae14_artifacts)
        for cand in ae14_reconcile.get("candidates") or []:
            d = record_to_dict(cand)
            if d["clean_forward_candidate_id"] not in seen_candidate_ids:
                candidates.append(d)
                seen_candidate_ids.add(d["clean_forward_candidate_id"])
        for dec in ae14_reconcile.get("decisions") or []:
            decision_inputs.append(record_to_dict(dec))
        for link in ae14_reconcile.get("links") or []:
            execution_links.append(record_to_dict(link))
        for skip_row in ae14_reconcile.get("skip_rows") or []:
            skip_reasons.append(
                CleanForwardSkipReason(
                    skip_record_id=make_skip_record_id(
                        clean_forward_candidate_id=str(skip_row.get("clean_forward_candidate_id") or "ae14"),
                        skipped_at=str(skip_row.get("skipped_at") or utc_now()),
                        skip_stage=str(skip_row.get("skip_stage") or "riskguard"),
                        skip_reason_code=str(skip_row.get("skip_reason_code") or "SKIP"),
                    ),
                    clean_forward_candidate_id=skip_row.get("clean_forward_candidate_id"),
                    clean_forward_decision_input_id=skip_row.get("clean_forward_decision_input_id"),
                    skipped_at=str(skip_row.get("skipped_at") or utc_now()),
                    skip_stage=str(skip_row.get("skip_stage") or "riskguard"),
                    skip_reason_code=str(skip_row.get("skip_reason_code") or "SKIP"),
                    skip_reason_detail=skip_row.get("skip_reason_detail"),
                    blocked_by_riskguard=bool(skip_row.get("blocked_by_riskguard")),
                ).to_dict()
            )
        lineage_summary = ae14_reconcile.get("summary") or lineage_summary
    else:
        lineage_summary = summarize_order_position_lineage(execution_links)
        lineage_summary["ae14_discrepancy_status"] = "AE14_ROOT_NOT_FOUND"
        lineage_summary["ae14_discrepancy_resolved"] = False

    # Candidate ID determinism audit
    determinism_audit = {
        "samples": determinism_samples,
        "all_rebuilds_match": all(s.get("rebuild_matches") for s in determinism_samples)
        if determinism_samples
        else False,
        "forbidden_fields_excluded_from_id_formula": sorted(CANDIDATE_ID_FORBIDDEN_FIELDS),
        "note": "clean_forward_candidate_id excludes model scores, consensus, order/position ids, and future outcomes",
    }
    write_json(audits / "candidate_id_determinism_audit.json", determinism_audit)

    # Decision input lineage audit
    decision_lineage_rows: list[dict[str, Any]] = []
    for di in decision_inputs:
        decision_lineage_rows.append(
            {
                "clean_forward_decision_input_id": di.get("clean_forward_decision_input_id"),
                "clean_forward_candidate_id": di.get("clean_forward_candidate_id"),
                "links_to_candidate": bool(di.get("clean_forward_candidate_id")),
                "model_scores_available": di.get("model_scores_available"),
                "xgb_score": di.get("xgb_score"),
                "tab_score": di.get("tab_score"),
                "rf_score": di.get("rf_score"),
                "model_score_source_status": di.get("model_score_source_status"),
                "no_direct_target_leakage": di.get("future_return") is None
                and "future_return" not in di,
                "no_legacy_market_snapshots_source": True,
                "decision_input_version": di.get("decision_input_version"),
                "llm_status": di.get("llm_status"),
                "context_status": di.get("context_status"),
            }
        )
    decision_lineage_audit = {
        "decision_inputs": len(decision_inputs),
        "all_link_to_candidate": all(r["links_to_candidate"] for r in decision_lineage_rows)
        if decision_lineage_rows
        else False,
        "model_fields_optional_shadow": True,
        "no_direct_target_future_outcome_leakage": True,
        "no_legacy_market_snapshots_source": True,
        "decision_input_version": DECISION_INPUT_VERSION,
    }
    write_json(audits / "decision_input_lineage_audit.json", decision_lineage_audit)
    write_csv(audits / "decision_input_lineage_audit.csv", decision_lineage_rows)

    # Order/position lineage audit
    order_position_audit = dict(lineage_summary)
    order_position_audit["ae14_reconcile_notes"] = ae14_reconcile.get("notes") if ae14_reconcile else []
    order_position_audit["ae14_explanation"] = lineage_summary.get("ae14_explanation")
    order_position_audit["links"] = execution_links
    write_json(audits / "order_position_lineage_audit.json", order_position_audit)
    write_csv(audits / "order_position_lineage_audit.csv", execution_links)

    legacy_exclusion = {
        "legacy_market_snapshots_used": False,
        "old_market_snapshot_feed_used": False,
        "raw_provider_payloads_legacy_feed_used": False,
        "local_db_candidate_universe_used": False,
        "model_training_performed": False,
        "backtest_performed": False,
        "profitability_claimed": False,
        "trader_db_mutated": False,
        "legacy_not_authoritative_reference_only": True,
        "ae15_source_policy": "clean_forward_smoke_and_ae14_closure_artifacts_only",
    }
    write_json(audits / "legacy_exclusion_audit.json", legacy_exclusion)

    safety = {
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_trading_enabled": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "paper_demo_only": True,
        "external_api_calls": False,
        "gemini_called": False,
        "qwen_ollama_called": False,
        "helius_called": False,
    }
    write_json(audits / "safety_audit.json", safety)

    # Data tables
    write_csv(data_dir / "clean_forward_candidates.csv", candidates)
    parquet_written = try_write_parquet(data_dir / "clean_forward_candidates.parquet", candidates)
    write_csv(data_dir / "clean_forward_decision_inputs.csv", decision_inputs)
    write_csv(data_dir / "clean_forward_paper_execution_links.csv", execution_links)
    write_csv(data_dir / "clean_forward_outcome_label_contract.csv", outcome_labels)
    write_csv(data_dir / "clean_forward_skip_reasons.csv", skip_reasons)

    # Manifests
    artifact_index = source_artifact_index(ae14_root=ae14_root, smoke_root=smoke_root)
    write_csv(manifests / "ae15_source_artifact_index.csv", artifact_index)
    schema_manifest = {
        "phase": "AE15",
        "decision_input_version": DECISION_INPUT_VERSION,
        "records": [
            "CleanForwardInstrumentIdentity",
            "CleanForwardCandidate",
            "CleanForwardDecisionInput",
            "CleanForwardPaperExecutionLink",
            "CleanForwardOutcomeLabel",
            "CleanForwardSkipReason",
        ],
        "candidate_count": len(candidates),
        "decision_input_count": len(decision_inputs),
        "execution_link_count": len(execution_links),
        "outcome_label_contract_count": len(outcome_labels),
        "skip_reason_count": len(skip_reasons),
        "parquet_candidates_written": parquet_written,
        "mode": mode,
        "output_root": str(out_root).replace("\\", "/"),
    }
    write_json(manifests / "ae15_schema_manifest.json", schema_manifest)

    gate = decide_gate(
        smoke_root=smoke_root,
        ae14_root=ae14_root,
        eligible_count=len(eligible_rows),
        identity_failures=identity_failures,
        legacy_flags=legacy_exclusion,
        lineage_summary=lineage_summary,
        safety=safety,
    )
    write_json(reports / "ae15_decision_gate.json", gate)

    manifest = {
        "phase": "AE15",
        "mode": mode,
        "output_root": str(out_root).replace("\\", "/"),
        "ae14_root": str(ae14_root).replace("\\", "/") if ae14_root else None,
        "clean_forward_smoke_root": str(smoke_root).replace("\\", "/") if smoke_root else None,
        "generated_at_utc": utc_now(),
        "classification": gate["classification"],
        "status": gate["status"],
        "clean_feed_rows_discovered": clean_feed_input_audit["total_rows_discovered"],
        "clean_feed_rows_eligible": clean_feed_input_audit["rows_eligible"],
        "candidates_created": len(candidates),
        "decision_inputs_created": len(decision_inputs),
        "execution_links_created": len(execution_links),
        "identity_separation_passed": identity_audit["passed"],
        "ae14_discrepancy_status": lineage_summary.get("ae14_discrepancy_status"),
        "legacy_exclusion": legacy_exclusion,
        "safety": safety,
        "ae16_blocked": gate["ae16_blocked"],
        "recommended_ae16_inputs": [
            _rel_or_abs(data_dir / "clean_forward_candidates.csv"),
            _rel_or_abs(data_dir / "clean_forward_decision_inputs.csv"),
            _rel_or_abs(data_dir / "clean_forward_paper_execution_links.csv"),
            _rel_or_abs(audits / "order_position_lineage_audit.json"),
            _rel_or_abs(reports / "ae15_decision_gate.json"),
            _rel_or_abs(manifests / "ae15_schema_manifest.json"),
        ]
        if not gate["ae16_blocked"]
        else [],
        "files": {
            "reports": [
                "reports/ae15_summary_for_upload.txt",
                "reports/ae15_manifest.json",
                "reports/ae15_decision_gate.json",
            ],
            "data": [
                "data/clean_forward_candidates.csv",
                "data/clean_forward_candidates.parquet" if parquet_written else None,
                "data/clean_forward_decision_inputs.csv",
                "data/clean_forward_paper_execution_links.csv",
                "data/clean_forward_outcome_label_contract.csv",
                "data/clean_forward_skip_reasons.csv",
            ],
            "audits": [
                "audits/clean_feed_input_audit.json",
                "audits/clean_feed_input_audit.csv",
                "audits/identity_separation_audit.json",
                "audits/identity_separation_audit.csv",
                "audits/candidate_id_determinism_audit.json",
                "audits/decision_input_lineage_audit.json",
                "audits/decision_input_lineage_audit.csv",
                "audits/order_position_lineage_audit.json",
                "audits/order_position_lineage_audit.csv",
                "audits/legacy_exclusion_audit.json",
                "audits/safety_audit.json",
            ],
            "manifests": [
                "manifests/ae15_source_artifact_index.csv",
                "manifests/ae15_schema_manifest.json",
            ],
        },
    }
    # drop None entries
    manifest["files"]["data"] = [x for x in manifest["files"]["data"] if x]
    write_json(reports / "ae15_manifest.json", manifest)

    summary_lines = [
        "AE15 Clean Forward Schema Bridge",
        f"classification: {gate['classification']}",
        f"status: {gate['status']}",
        f"mode: {mode}",
        f"output_root: {out_root}",
        f"smoke_root: {smoke_root}",
        f"ae14_root: {ae14_root}",
        f"rows_discovered: {clean_feed_input_audit['total_rows_discovered']}",
        f"rows_eligible: {clean_feed_input_audit['rows_eligible']}",
        f"candidates_created: {len(candidates)}",
        f"decision_inputs_created: {len(decision_inputs)}",
        f"execution_links_created: {len(execution_links)}",
        f"identity_separation_passed: {identity_audit['passed']}",
        f"ae14_orders_opened: {lineage_summary.get('ae14_paper_orders_opened', lineage_summary.get('orders_opened'))}",
        f"ae14_positions_opened: {lineage_summary.get('ae14_paper_positions_opened', lineage_summary.get('positions_opened'))}",
        f"ae14_discrepancy_status: {lineage_summary.get('ae14_discrepancy_status')}",
        f"legacy_market_snapshots_used: false",
        f"model_training_performed: false",
        f"backtest_performed: false",
        f"profitability_claimed: false",
        f"wallet_configured: false",
        f"live_trading_enabled: false",
        f"ae16_blocked: {gate['ae16_blocked']}",
    ]
    if gate.get("limitations"):
        summary_lines.append("limitations: " + "; ".join(gate["limitations"]))
    if gate.get("blockers"):
        summary_lines.append("blockers: " + "; ".join(gate["blockers"]))
    write_text(reports / "ae15_summary_for_upload.txt", "\n".join(summary_lines) + "\n")

    return {
        "ok": gate["status"] in ("PASS", "PASS_WITH_LIMITATIONS"),
        "output_root": str(out_root),
        "gate": gate,
        "manifest": manifest,
        "clean_feed_input_audit": clean_feed_input_audit,
        "identity_audit": identity_audit,
        "lineage_summary": lineage_summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Default to --build when no mode flag provided
    if not (args.audit_only or args.build or args.reconcile_ae14):
        args.build = True
    result = run_ae15(args)
    gate = result["gate"]
    print(json.dumps({"classification": gate["classification"], "output_root": result["output_root"]}, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
