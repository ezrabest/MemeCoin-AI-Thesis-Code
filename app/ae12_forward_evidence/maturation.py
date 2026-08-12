"""AE12 forward-evidence maturation orchestrator (derived audit layer only)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.ae12_forward_evidence.idempotency import (
    IdempotencyGuard,
    append_processed_keys,
    load_processed_keys,
    load_run_state,
    make_evidence_row_id,
    write_run_state,
)
from app.ae12_forward_evidence.loaders import (
    MarketSnapshotStore,
    build_paper_linkage_index,
    discover_sources,
    index_by_keys,
    load_opportunity_rows,
)
from app.ae12_forward_evidence.maturation_core import compute_horizon_outcomes, outcome_to_dict
from app.ae12_forward_evidence.missed_winners import detect_missed_winners, missed_winner_counts_by_horizon
from app.ae12_forward_evidence.opportunity_analysis import (
    build_rejection_reason_summary,
    build_strict_vs_exploration_comparison,
    build_trade_vs_no_trade_comparison,
)
from app.ae12_forward_evidence.qwen_linkage import classify_qwen_ollama_linkage, summarize_linkage
from app.ae12_forward_evidence.reason_recovery import recover_reason
from app.ae12_forward_evidence.reports import (
    append_csv,
    count_csv_rows,
    ensure_dirs,
    write_csv,
    write_json,
    write_upload_report,
)
from app.ae12_forward_evidence.safety import audit_append_only, audit_no_trader_db_writes, audit_wallet_safety
from app.ae12_forward_evidence.types import (
    AE12_PHASE,
    AE12_SCHEMA_VERSION,
    Ae12RunConfig,
    DEFAULT_MISSED_WINNER_THRESHOLDS,
    HORIZON_SECONDS,
    ReasonRecoveryStatus,
    TRADED_ACTIONS,
    parse_ts,
    utc_now_iso,
)


class OutputRootExistsError(RuntimeError):
    pass


def _resolve_output_root(project_root: Path, output_root: Path | None, resume: bool) -> Path:
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return project_root / "data" / "audits" / f"ae12_forward_evidence_maturation_{stamp}"
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    return root


def _price_freshness_status(opp: dict[str, Any]) -> str:
    if opp.get("stale_price") is True:
        return "STALE_AT_ENTRY"
    # trade decision may carry age; opportunity has boolean
    if opp.get("stale_price") is False:
        return "FRESH_OR_ACCEPTABLE"
    return "UNKNOWN"


def _is_traded(opp: dict[str, Any], paper: dict[str, Any] | None) -> bool:
    if paper and paper.get("was_traded"):
        return True
    if opp.get("paper_order_id") or opp.get("position_id"):
        action = str(opp.get("paper_action_taken") or "").upper()
        if action in TRADED_ACTIONS or action in {"FILLED", "OPENED"}:
            return True
        if opp.get("paper_order_id"):
            return True
    action = str(opp.get("paper_action_taken") or "").upper()
    return action in TRADED_ACTIONS


def run_forward_evidence_maturation(config: Ae12RunConfig) -> dict[str, Any]:
    project_root = Path(config.project_root)
    output_root = _resolve_output_root(project_root, Path(config.output_root) if config.output_root else None, config.resume)

    if output_root.exists() and any(output_root.iterdir()) and not config.resume:
        if config.fail_if_output_exists:
            raise OutputRootExistsError(
                f"Output root already exists: {output_root}. "
                "Pass --resume to continue without duplication, or choose a new --output-root."
            )

    dirs = ensure_dirs(output_root)
    sources = discover_sources(project_root, Path(config.db_path) if config.db_path else None)
    horizons = [h for h in config.horizons if h in HORIZON_SECONDS]
    if not horizons:
        horizons = list(HORIZON_SECONDS.keys())

    state_dir = dirs["state"]
    processed_evidence_path = state_dir / "processed_evidence_row_keys.jsonl"
    processed_horizon_path = state_dir / "processed_horizon_row_keys.jsonl"
    run_state_path = state_dir / "ae12_run_state.json"

    prior_evidence: set[str] = set()
    prior_horizons: set[str] = set()
    if config.resume:
        prior_evidence = load_processed_keys(processed_evidence_path)
        prior_horizons = load_processed_keys(processed_horizon_path)

    guard = IdempotencyGuard(processed_evidence=prior_evidence, processed_horizons=prior_horizons)

    # Load primary + secondary sources
    opportunity_rows = load_opportunity_rows(
        sources.opportunity_files, project_root=project_root, max_rows=config.max_rows
    )
    print(f"[ae12] loaded opportunity_rows={len(opportunity_rows)}", flush=True)
    needed_decision_ids = {
        str(r.get("source_decision_id") or r.get("decision_id"))
        for r in opportunity_rows
        if r.get("source_decision_id") or r.get("decision_id")
    }
    needed_candidate_ids = {
        str(r.get("candidate_id")) for r in opportunity_rows if r.get("candidate_id")
    }
    needed_ae9_ids = {
        str(r.get("source_llm_audit_record_id"))
        for r in opportunity_rows
        if r.get("source_llm_audit_record_id")
    }
    trade_index = index_by_keys(
        sources.trade_decision_files,
        project_root=project_root,
        key_fields=("source_decision_id", "decision_id", "candidate_id"),
        only_keys=needed_decision_ids | needed_candidate_ids,
    )
    ae6_index = index_by_keys(
        sources.ae6_files,
        project_root=project_root,
        key_fields=("decision_id",),
        only_keys=needed_decision_ids,
    )
    ae9_by_id = index_by_keys(
        sources.ae9_audit_files,
        project_root=project_root,
        key_fields=("audit_record_id", "source_decision_id", "candidate_id"),
        only_keys=needed_ae9_ids | needed_decision_ids | needed_candidate_ids,
    )
    paper_index = build_paper_linkage_index(
        sources.paper_order_files,
        sources.paper_position_files,
        sources.paper_trade_files,
        project_root=project_root,
    )
    # Runtime events rarely have decision_id; keep empty-capable index
    runtime_events_by_key = index_by_keys(
        sources.runtime_event_files,
        project_root=project_root,
        key_fields=("source_decision_id", "decision_id", "candidate_id"),
    )

    store = MarketSnapshotStore(db_path=sources.db_path)
    store.open()

    # Bulk prefetch market snapshots (pair_address is unindexed — avoid per-row scans)
    pairs = {str(r.get("pair_address")) for r in opportunity_rows if r.get("pair_address")}
    first_seen_dts = [
        parse_ts(r.get("first_seen_timestamp") or r.get("created_at_utc")) for r in opportunity_rows
    ]
    first_seen_dts = [d for d in first_seen_dts if d is not None]
    max_h_sec = max((HORIZON_SECONDS[h] for h in horizons), default=0)
    if first_seen_dts and store.available:
        t_min = min(first_seen_dts) - timedelta(seconds=1)
        t_max_candidates = max(first_seen_dts) + timedelta(seconds=max_h_sec)
        t_max = store.global_latest_ts or t_max_candidates
        if store.global_latest_ts and t_max_candidates < store.global_latest_ts:
            # Keep enough history for maturity: up to global latest helps long horizons
            t_max = store.global_latest_ts
        print(
            f"[ae12] prefetching market snapshots for {len(pairs)} pairs "
            f"window=({t_min.isoformat()} .. {t_max.isoformat()})",
            flush=True,
        )
        store.prefetch_pairs(pairs, t_min=t_min, t_max=t_max)

    evidence_rows: list[dict[str, Any]] = []
    matured_outcomes: list[dict[str, Any]] = []
    opportunity_full: list[dict[str, Any]] = []
    reason_recovery_audit: list[dict[str, Any]] = []
    missing_warnings: list[dict[str, Any]] = []
    qwen_rows: list[dict[str, Any]] = []
    paper_linkage_rows: list[dict[str, Any]] = []
    no_lookahead_audits: list[dict[str, Any]] = []
    horizon_maturity_audits: list[dict[str, Any]] = []
    linkage_quality_audits: list[dict[str, Any]] = []

    run_id = f"ae12_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    for opp in opportunity_rows:
        source_file = str(opp.get("_source_file") or "")
        source_line_no = int(opp.get("_source_line_no") or 0)
        candidate_id = opp.get("candidate_id")
        decision_id = opp.get("source_decision_id") or opp.get("decision_id")
        pair_address = opp.get("pair_address")
        first_seen = opp.get("first_seen_timestamp") or opp.get("created_at_utc")

        evidence_row_id = make_evidence_row_id(
            source_file=source_file,
            source_line_no=source_line_no,
            candidate_id=str(candidate_id) if candidate_id else None,
            decision_id=str(decision_id) if decision_id else None,
            pair_address=str(pair_address) if pair_address else None,
            first_seen_timestamp=str(first_seen) if first_seen else None,
        )
        if not guard.accept_evidence(evidence_row_id):
            continue

        td = None
        if decision_id and str(decision_id) in trade_index:
            td = trade_index[str(decision_id)]
        elif candidate_id and str(candidate_id) in trade_index:
            td = trade_index[str(candidate_id)]

        ae6 = ae6_index.get(str(decision_id)) if decision_id else None
        ae9 = None
        ae9_id = opp.get("source_llm_audit_record_id")
        if ae9_id and str(ae9_id) in ae9_by_id:
            ae9 = ae9_by_id[str(ae9_id)]
        elif decision_id and str(decision_id) in ae9_by_id:
            ae9 = ae9_by_id[str(decision_id)]
        elif candidate_id and str(candidate_id) in ae9_by_id:
            ae9 = ae9_by_id[str(candidate_id)]

        paper = None
        if candidate_id and str(candidate_id) in paper_index:
            paper = paper_index[str(candidate_id)]
        elif decision_id and str(decision_id) in paper_index:
            paper = paper_index[str(decision_id)]

        was_traded = _is_traded(opp, paper)
        recovered = recover_reason(
            opportunity=opp,
            trade_decision=td,
            ae6=ae6,
            runtime_events_by_key=runtime_events_by_key,
            was_traded=was_traded,
            paper_action_taken=opp.get("paper_action_taken"),
        )

        qwen = classify_qwen_ollama_linkage(opportunity=opp, ae6=ae6, ae9=ae9)
        qwen_rows.append(qwen)

        identity = (ae6 or {}).get("candidate_identity") if isinstance((ae6 or {}).get("candidate_identity"), dict) else {}
        chain = identity.get("chain")
        symbol = identity.get("symbol")

        strict_decision = opp.get("strict_shadow_decision")
        exploration_decision = opp.get("exploration_decision")
        is_strict_approved = str(strict_decision or "").upper() in {"TRADE", "APPROVED", "PASS"}
        is_exploration_only = (not is_strict_approved) and str(exploration_decision or "").upper() in {
            "TRADE",
            "TRADE_EXPLORATION_OVERRIDE",
            "PAPER_BUY",
            "NO_TRADE",
        }

        trade_authority = None
        not_model_approved = None
        not_live_approved = None
        if td and isinstance(td.get("exploration_mode"), dict):
            trade_authority = td["exploration_mode"].get("trade_authority")
            not_model_approved = td["exploration_mode"].get("not_model_approved")
            not_live_approved = td["exploration_mode"].get("not_live_approved")
        if paper and paper.get("order"):
            trade_authority = trade_authority or paper["order"].get("trade_authority")
            not_model_approved = (
                not_model_approved
                if not_model_approved is not None
                else paper["order"].get("not_model_approved")
            )
            not_live_approved = (
                not_live_approved
                if not_live_approved is not None
                else paper["order"].get("not_live_approved")
            )

        evidence: dict[str, Any] = {
            "evidence_row_id": evidence_row_id,
            "run_id": run_id,
            "source_file": source_file,
            "source_line_no": source_line_no,
            "candidate_id": candidate_id,
            "decision_id": decision_id,
            "source_decision_id": decision_id,
            "pair_address": pair_address,
            "chain": chain,
            "symbol": symbol,
            "first_seen_timestamp": first_seen,
            "decision_timestamp": (ae6 or {}).get("created_at_utc") or opp.get("created_at_utc"),
            "price_at_first_seen": opp.get("price_at_first_seen"),
            "liquidity_at_first_seen": opp.get("liquidity_at_first_seen"),
            "volume_at_first_seen": opp.get("volume_at_first_seen"),
            "whale_score_asof": opp.get("whale_score_at_first_seen"),
            "ae6_decision_action": opp.get("ae6_decision_status") or (ae6 or {}).get("decision_status"),
            "AE6_qwen_text_or_qwen_marker_present": qwen.get("ae6_qwen_marker_present"),
            "ae8_context_status": opp.get("ae8_context_status"),
            "ae9_audit_status": opp.get("ae9_audit_verdict"),
            "ae9_audit_blockers": opp.get("ae9_audit_blockers") or [],
            "strict_shadow_decision": strict_decision,
            "exploration_decision": exploration_decision,
            "trade_authority": trade_authority,
            "not_model_approved": not_model_approved,
            "not_live_approved": not_live_approved,
            "paper_order_id": (paper or {}).get("paper_order_id") or opp.get("paper_order_id"),
            "paper_position_id": (paper or {}).get("paper_position_id") or opp.get("position_id"),
            "paper_trade_id": (paper or {}).get("paper_trade_id"),
            "paper_action_taken": opp.get("paper_action_taken"),
            "was_traded": was_traded,
            "reason_not_traded": recovered.get("reason_not_traded"),
            "rejection_reason": recovered.get("rejection_reason"),
            "reason_source": recovered.get("reason_source"),
            "reason_recovery_status": recovered.get("reason_recovery_status"),
            "is_strict_approved": is_strict_approved,
            "is_exploration_only": is_exploration_only and not was_traded,
            "price_freshness_status": _price_freshness_status(opp),
            "source_quality_flags": [],
            "blocked_by_ae9": opp.get("blocked_by_ae9"),
            "stale_price": opp.get("stale_price"),
            "missing_context": opp.get("missing_context"),
            "max_open_positions_hit": opp.get("max_open_positions_hit"),
            "cooldown_active": opp.get("cooldown_active"),
            "duplicate_active_pair": opp.get("duplicate_active_pair"),
            "duplicate_reason": opp.get("duplicate_reason"),
            "qwen_linkage_status": qwen.get("qwen_linkage_status"),
            "ollama_linkage_status": qwen.get("ollama_linkage_status"),
        }

        # Missing field warnings
        for field, code in [
            ("candidate_id", "MISSING_CANDIDATE_ID"),
            ("decision_id", "MISSING_DECISION_ID"),
            ("pair_address", "MISSING_PAIR_ADDRESS"),
            ("first_seen_timestamp", "MISSING_FIRST_SEEN_TIMESTAMP"),
            ("price_at_first_seen", "MISSING_PRICE_AT_FIRST_SEEN"),
        ]:
            if not evidence.get(field):
                missing_warnings.append(
                    {
                        "evidence_row_id": evidence_row_id,
                        "candidate_id": candidate_id,
                        "decision_id": decision_id,
                        "pair_address": pair_address,
                        "first_seen_timestamp": first_seen,
                        "missing_field": field,
                        "warning_code": code,
                        "warning_message": f"Required field {field} missing in normalized evidence row",
                        "source_file": source_file,
                        "source_line_no": source_line_no,
                    }
                )
                evidence["source_quality_flags"].append(code)

        if recovered.get("reason_recovery_status") == ReasonRecoveryStatus.MISSING_IN_SOURCE.value:
            missing_warnings.append(
                {
                    "evidence_row_id": evidence_row_id,
                    "candidate_id": candidate_id,
                    "decision_id": decision_id,
                    "pair_address": pair_address,
                    "first_seen_timestamp": first_seen,
                    "missing_field": "reason_not_traded",
                    "warning_code": "MISSING_REJECTION_REASON",
                    "warning_message": "Rejection/no-trade reason could not be recovered from any source",
                    "source_file": source_file,
                    "source_line_no": source_line_no,
                }
            )

        if qwen.get("qwen_linkage_status") in {"ABSENT", "MENTION_ONLY", "LOG_ONLY_NOT_ROW_LINKED"}:
            missing_warnings.append(
                {
                    "evidence_row_id": evidence_row_id,
                    "candidate_id": candidate_id,
                    "decision_id": decision_id,
                    "pair_address": pair_address,
                    "first_seen_timestamp": first_seen,
                    "missing_field": "qwen_row_linkage",
                    "warning_code": "MISSING_QWEN_ROW_LINKAGE",
                    "warning_message": f"Qwen linkage status={qwen.get('qwen_linkage_status')}",
                    "source_file": source_file,
                    "source_line_no": source_line_no,
                }
            )

        h_fields, h_outcomes, h_audits = compute_horizon_outcomes(
            evidence_row_id=evidence_row_id,
            pair_address=str(pair_address) if pair_address else None,
            first_seen_timestamp=str(first_seen) if first_seen else None,
            entry_price=opp.get("price_at_first_seen"),
            horizons=horizons,
            store=store,
        )
        evidence.update(h_fields)
        no_lookahead_audits.extend(h_audits)
        for ho in h_outcomes:
            if not guard.accept_horizon(ho.horizon_row_id):
                continue
            matured_outcomes.append(outcome_to_dict(ho))
            horizon_maturity_audits.append(
                {
                    "evidence_row_id": evidence_row_id,
                    "horizon": ho.horizon,
                    "matured": ho.matured,
                    "no_lookahead_status": ho.no_lookahead_status,
                    "snapshot_count": ho.snapshot_count,
                    "max_return": ho.max_return,
                }
            )
            if ho.matured and ho.no_lookahead_status == "MATURED_BUT_NO_LOCAL_SNAPSHOTS":
                missing_warnings.append(
                    {
                        "evidence_row_id": evidence_row_id,
                        "candidate_id": candidate_id,
                        "decision_id": decision_id,
                        "pair_address": pair_address,
                        "first_seen_timestamp": first_seen,
                        "missing_field": f"market_snapshots_{ho.horizon}",
                        "warning_code": "MISSING_MARKET_SNAPSHOTS_FOR_HORIZON",
                        "warning_message": f"Horizon {ho.horizon} matured but no usable local snapshots",
                        "source_file": source_file,
                        "source_line_no": source_line_no,
                    }
                )

        reason_recovery_audit.append(
            {
                "evidence_row_id": evidence_row_id,
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "reason_not_traded": recovered.get("reason_not_traded"),
                "rejection_reason": recovered.get("rejection_reason"),
                "reason_source": recovered.get("reason_source"),
                "reason_recovery_status": recovered.get("reason_recovery_status"),
                "was_traded": was_traded,
            }
        )

        paper_linkage_rows.append(
            {
                "evidence_row_id": evidence_row_id,
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "pair_address": pair_address,
                "paper_order_id": evidence.get("paper_order_id"),
                "paper_position_id": evidence.get("paper_position_id"),
                "paper_trade_id": evidence.get("paper_trade_id"),
                "was_traded": was_traded,
                "paper_action_taken": opp.get("paper_action_taken"),
                "linkage_source": "jsonl_paper_trading",
            }
        )

        linkage_quality_audits.append(
            {
                "evidence_row_id": evidence_row_id,
                "trade_decision_linked": td is not None,
                "ae6_linked": ae6 is not None,
                "ae9_linked": ae9 is not None or bool(ae9_id),
                "paper_linked": paper is not None,
                "qwen_linkage_status": qwen.get("qwen_linkage_status"),
            }
        )

        opportunity_full.append({**opp, **{k: evidence[k] for k in (
            "evidence_row_id",
            "was_traded",
            "reason_not_traded",
            "rejection_reason",
            "reason_source",
            "reason_recovery_status",
            "qwen_linkage_status",
        ) if k in evidence}, **h_fields})

        evidence_rows.append(evidence)

    store.close()

    # Missed winners
    thresholds = dict(DEFAULT_MISSED_WINNER_THRESHOLDS)
    thresholds.update(config.missed_winner_thresholds or {})
    missed = detect_missed_winners(evidence_rows, thresholds=thresholds, horizons=horizons)
    missed_by_h = missed_winner_counts_by_horizon(missed)
    missed_keys = {(m.get("evidence_row_id"), m.get("horizon")) for m in missed}
    for row in evidence_rows:
        hs_hit = [h for h in horizons if (row.get("evidence_row_id"), h) in missed_keys]
        row["missed_winner_horizons"] = hs_hit
        for h in hs_hit:
            row[f"is_missed_winner_{h}"] = True

    trade_vs = build_trade_vs_no_trade_comparison(evidence_rows, horizons=horizons)
    # Fix missed counts from actual missed list
    for row in trade_vs:
        row["missed_winner_count"] = missed_by_h.get(str(row.get("horizon")), 0)

    strict_vs = build_strict_vs_exploration_comparison(evidence_rows, horizons=horizons)
    rejection_summary = build_rejection_reason_summary(evidence_rows)

    reason_recovery_counts = Counter(r.get("reason_recovery_status") for r in reason_recovery_audit)
    missing_reason_count = reason_recovery_counts.get(ReasonRecoveryStatus.MISSING_IN_SOURCE.value, 0)

    # Horizon maturity aggregates
    horizon_maturity: dict[str, Any] = {}
    for h in horizons:
        matured_c = sum(1 for r in evidence_rows if r.get(f"horizon_{h}_matured") is True)
        not_matured_c = sum(1 for r in evidence_rows if r.get(f"horizon_{h}_matured") is not True)
        ok_c = sum(
            1
            for r in evidence_rows
            if r.get(f"horizon_{h}_no_lookahead_status") == "NO_LOOKAHEAD_OK"
        )
        no_snap = sum(
            1
            for r in evidence_rows
            if r.get(f"horizon_{h}_no_lookahead_status") == "MATURED_BUT_NO_LOCAL_SNAPSHOTS"
        )
        horizon_maturity[h] = {
            "matured_count": matured_c,
            "not_matured_count": not_matured_c,
            "no_lookahead_ok_count": ok_c,
            "matured_but_no_snapshots_count": no_snap,
        }

    reason_coverage = [
        {
            "metric": "total_evidence_rows",
            "value": len(evidence_rows),
        },
        {
            "metric": "reason_recovered",
            "value": sum(
                1
                for r in reason_recovery_audit
                if r.get("reason_recovery_status")
                not in {
                    ReasonRecoveryStatus.MISSING_IN_SOURCE.value,
                    ReasonRecoveryStatus.TRADED_NO_REJECTION.value,
                }
            ),
        },
        {
            "metric": "missing_in_source",
            "value": missing_reason_count,
        },
        {
            "metric": "traded_no_rejection",
            "value": reason_recovery_counts.get(ReasonRecoveryStatus.TRADED_NO_REJECTION.value, 0),
        },
    ]
    for k, v in reason_recovery_counts.items():
        reason_coverage.append({"metric": f"status::{k}", "value": v})

    qwen_counts = summarize_linkage(qwen_rows)
    sanity = [
        r
        for r in qwen_rows
        if r.get("qwen_linkage_status")
        in {"ROW_LINKED_AE9_RECORD", "ROW_LINKED_AE6_DECISION"}
        and (r.get("candidate_id") or r.get("decision_id"))
    ][:10]

    wallet = audit_wallet_safety(
        project_root=project_root,
        live_dry_run_files=sources.live_dry_run_files,
        trade_decision_sample=list(trade_index.values()),
        no_real_wallet=config.no_real_wallet,
    )
    append_audit = audit_append_only(
        project_root=project_root,
        output_root=output_root,
        mutated_paths=[dirs["reports"], dirs["data"], dirs["audits"], dirs["state"]],
    )
    db_write_audit = audit_no_trader_db_writes(output_root, sources.db_path)
    idempo = guard.to_audit_row()

    known_limitations = [
        "Forward returns are labels only; not profitability proof.",
        "SQLite paper_trades may be stale; JSONL paper artifacts preferred.",
        "AE11 opportunity capture often writes horizon_matured=false at capture time; AE12 recomputes.",
        "AE12.2 field scan looked for reason_not_traded; source field is often reason_for_no_trade.",
        "Qwen markers in AE6 llm_context do not imply operational trade authority.",
        "Market snapshot coverage may be uneven across pairs/horizons.",
        "Runtime collector may still be writing; this pass is derived/read-only against inputs.",
    ]
    if not store.available:
        known_limitations.append(f"Market snapshots unavailable: {store.unavailable_reason}")

    readiness = {
        "gate_name": "ae12_final_system_readiness_gate",
        "status": "FORWARD_EVIDENCE_READY_FOR_REPORTING",
        "live_trading_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
        "wallet_safety_status": wallet.get("audit_status"),
        "evidence_row_count": len(evidence_rows),
        "can_proceed_to_ui_final_report": True,
        "needs_persistence_fix": missing_reason_count > 0
        and all(
            r.get("reason_recovery_status") == ReasonRecoveryStatus.MISSING_IN_SOURCE.value
            for r in reason_recovery_audit
            if not r.get("was_traded")
        )
        and len(evidence_rows) > 0
        and missing_reason_count == sum(1 for r in reason_recovery_audit if not r.get("was_traded")),
        "notes": [
            "Safe for Final MSc reporting as forward-evidence audit, not live readiness.",
            "Do not claim profitability or live trading authorization.",
        ],
        "created_at_utc": utc_now_iso(),
    }
    # needs_persistence_fix: only if almost all non-traded lack reasons
    nont = [r for r in reason_recovery_audit if not r.get("was_traded")]
    if nont:
        miss_frac = sum(
            1
            for r in nont
            if r.get("reason_recovery_status") == ReasonRecoveryStatus.MISSING_IN_SOURCE.value
        ) / len(nont)
        readiness["needs_persistence_fix"] = miss_frac > 0.95
    else:
        readiness["needs_persistence_fix"] = False

    if readiness["needs_persistence_fix"]:
        readiness["status"] = "NEEDS_PERSISTENCE_FIX_FOR_REJECTION_REASONS"
        readiness["can_proceed_to_ui_final_report"] = True  # still reportable with explicit gap

    summary = {
        "phase": AE12_PHASE,
        "schema_version": AE12_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "run_id": run_id,
        "candidate_evidence_row_count": len(evidence_rows),
        "candidate_evidence_row_count_this_run": len(evidence_rows),
        "matured_outcome_row_count": len(matured_outcomes),
        "missed_winner_count": len(missed),
        "missed_winners_by_horizon": missed_by_h,
        "missing_reason_count": missing_reason_count,
        "missing_data_warning_count": len(missing_warnings),
        "reason_recovery_counts": dict(reason_recovery_counts),
        "horizon_maturity": horizon_maturity,
        "trade_vs_no_trade": trade_vs,
        "trade_vs_no_trade_interpretations": {
            r["horizon"]: r["interpretation_status"] for r in trade_vs
        },
        "strict_vs_exploration": strict_vs,
        "qwen_linkage_counts": qwen_counts,
        "qwen_linkage_sanity_sample": sanity,
        "wallet_safety": wallet,
        "idempotency": idempo,
        "readiness_gate": readiness,
        "known_limitations": known_limitations,
        "market_snapshots_available": store.available,
        "market_snapshots_reason": store.unavailable_reason,
        "horizons": horizons,
        "missed_winner_thresholds": thresholds,
        "no_external_apis": config.no_external_apis,
        "no_real_wallet": config.no_real_wallet,
        "max_rows": config.max_rows,
        "resume": config.resume,
        "source_file_counts": {
            "opportunity_capture": len(sources.opportunity_files),
            "trade_decisions": len(sources.trade_decision_files),
            "ae6": len(sources.ae6_files),
            "paper_orders": len(sources.paper_order_files),
            "paper_positions": len(sources.paper_position_files),
            "paper_trades": len(sources.paper_trade_files),
            "ae9": len(sources.ae9_audit_files),
        },
    }

    # Writes — resume appends new rows; zero-new resume preserves existing data CSVs
    resume_append = bool(config.resume and prior_evidence)
    data_specs = [
        ("ae12_candidate_evidence_rows.csv", evidence_rows),
        ("ae12_matured_outcomes.csv", matured_outcomes),
        ("ae12_opportunity_capture_full.csv", opportunity_full),
        ("ae12_missed_winners_full.csv", missed),
        ("ae12_paper_trade_linkage.csv", paper_linkage_rows),
        ("ae12_qwen_ollama_linkage.csv", qwen_rows),
        ("ae12_reason_recovery_audit.csv", reason_recovery_audit),
    ]
    for name, rows in data_specs:
        path = dirs["data"] / name
        if resume_append:
            if rows:
                append_csv(path, rows)
            # else preserve existing
        else:
            write_csv(path, rows)

    strict_flat = [
        {
            "total_candidates": strict_vs["total_candidates"],
            "strict_approved": strict_vs["strict_approved"],
            "strict_blocked": strict_vs["strict_blocked"],
            "exploration_traded": strict_vs["exploration_traded"],
            "exploration_only_trades": strict_vs["exploration_only_trades"],
            "strict_approved_trades": strict_vs["strict_approved_trades"],
            "strict_blocked_but_exploration_traded": strict_vs[
                "strict_blocked_but_exploration_traded"
            ],
        }
    ]
    for rr in strict_vs.get("return_comparison_by_horizon") or []:
        strict_flat.append(rr)

    # Comparison/summary CSVs rewritten from this run's batch (full recompute on fresh run)
    if not resume_append or evidence_rows:
        write_csv(dirs["data"] / "ae12_trade_vs_no_trade_comparison.csv", trade_vs)
        write_csv(dirs["data"] / "ae12_strict_vs_exploration_comparison.csv", strict_flat)
        write_csv(dirs["data"] / "ae12_rejection_reason_summary.csv", rejection_summary)

    audit_append_specs = [
        ("ae12_no_lookahead_audit.csv", no_lookahead_audits),
        ("ae12_linkage_quality_audit.csv", linkage_quality_audits),
        ("ae12_horizon_maturity_audit.csv", horizon_maturity_audits),
        ("ae12_missing_data_warning_audit.csv", missing_warnings),
    ]
    for name, rows in audit_append_specs:
        path = dirs["audits"] / name
        if resume_append:
            if rows:
                append_csv(path, rows)
        else:
            write_csv(path, rows)

    write_csv(dirs["audits"] / "ae12_append_only_audit.csv", append_audit)
    write_csv(dirs["audits"] / "ae12_idempotency_audit.csv", [idempo])
    write_json(dirs["audits"] / "ae12_wallet_safety_audit.json", wallet)
    if not resume_append or evidence_rows:
        write_csv(dirs["audits"] / "ae12_reason_coverage_audit.csv", reason_coverage)
        write_csv(
            dirs["audits"] / "ae12_qwen_linkage_audit.csv",
            [{"status": k, "count": v} for k, v in qwen_counts.items()],
        )
    write_json(dirs["audits"] / "ae12_no_trader_db_write_audit.json", db_write_audit)

    # Cumulative evidence count for resume reporting
    evidence_csv = dirs["data"] / "ae12_candidate_evidence_rows.csv"
    cumulative_evidence = count_csv_rows(evidence_csv)
    if cumulative_evidence <= 0:
        cumulative_evidence = len(prior_evidence) + len(evidence_rows)
    summary["candidate_evidence_row_count"] = cumulative_evidence
    summary["candidate_evidence_row_count_this_run"] = len(evidence_rows)
    summary["idempotency"] = idempo
    readiness["evidence_row_count"] = cumulative_evidence
    summary["readiness_gate"] = readiness

    manifest = {
        "phase": AE12_PHASE,
        "schema_version": AE12_SCHEMA_VERSION,
        "output_root": str(output_root),
        "created_at_utc": summary["created_at_utc"],
        "files": {
            "reports": [
                "ae12_forward_evidence_manifest.json",
                "ae12_forward_evidence_summary.json",
                "ae12_forward_evidence_for_upload.txt",
                "ae12_final_system_readiness_gate.json",
            ],
            "data": [
                "ae12_candidate_evidence_rows.csv",
                "ae12_matured_outcomes.csv",
                "ae12_opportunity_capture_full.csv",
                "ae12_missed_winners_full.csv",
                "ae12_trade_vs_no_trade_comparison.csv",
                "ae12_strict_vs_exploration_comparison.csv",
                "ae12_rejection_reason_summary.csv",
                "ae12_paper_trade_linkage.csv",
                "ae12_qwen_ollama_linkage.csv",
                "ae12_reason_recovery_audit.csv",
            ],
            "audits": [
                "ae12_no_lookahead_audit.csv",
                "ae12_append_only_audit.csv",
                "ae12_idempotency_audit.csv",
                "ae12_linkage_quality_audit.csv",
                "ae12_horizon_maturity_audit.csv",
                "ae12_wallet_safety_audit.json",
                "ae12_reason_coverage_audit.csv",
                "ae12_qwen_linkage_audit.csv",
                "ae12_missing_data_warning_audit.csv",
            ],
            "state": [
                "processed_evidence_row_keys.jsonl",
                "processed_horizon_row_keys.jsonl",
                "ae12_run_state.json",
            ],
        },
    }
    write_json(dirs["reports"] / "ae12_forward_evidence_manifest.json", manifest)
    write_json(dirs["reports"] / "ae12_forward_evidence_summary.json", summary)
    write_json(dirs["reports"] / "ae12_final_system_readiness_gate.json", readiness)
    write_upload_report(dirs["reports"] / "ae12_forward_evidence_for_upload.txt", summary)

    append_processed_keys(processed_evidence_path, guard.new_evidence_keys, key_field="evidence_row_id")
    append_processed_keys(processed_horizon_path, guard.new_horizon_keys, key_field="horizon_row_id")
    prior_state = load_run_state(run_state_path) if config.resume else {}
    write_run_state(
        run_state_path,
        {
            **prior_state,
            "last_run_id": run_id,
            "last_run_at_utc": utc_now_iso(),
            "total_evidence_keys": len(prior_evidence) + len(guard.new_evidence_keys),
            "total_horizon_keys": len(prior_horizons) + len(guard.new_horizon_keys),
            "last_summary_counts": {
                "evidence_rows": len(evidence_rows),
                "matured_outcomes": len(matured_outcomes),
                "missed_winners": len(missed),
            },
        },
    )

    summary["output_root"] = str(output_root)
    return summary
