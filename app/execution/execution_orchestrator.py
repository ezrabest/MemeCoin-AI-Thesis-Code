"""AE10 trading orchestration — paper/demo execution and live dry-run wiring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.decision.persistence import read_jsonl_records_safe
from app.execution.adapters import PaperExecutionAdapter
from app.execution.live_no_wallet_adapter import LiveNoWalletDryRunAdapter
from app.execution.types import OrderIntent
from app.llm_audit.audit_runner import discover_latest_glob
from app.paper_trading.ledger import DemoLedger, reset_demo_account
from app.paper_trading.order_simulator import PaperOrderSimulator
from app.paper_trading.persistence import (
    JsonlWriter,
    load_demo_account_state,
    live_dry_run_orders_path_for_date,
    paper_orders_path_for_date,
    paper_positions_path_for_date,
    paper_trades_path_for_date,
    save_demo_account_state,
)
from app.paper_trading.price_oracle import DemoPriceOracle
from app.paper_trading.reports import write_ae10_audits
from app.paper_trading.types import (
    AE10_PHASE,
    Ae10FinalStatus,
    DemoAccount,
    PaperOrderStatus,
    PriceStatus,
    RUNTIME_INFERENCE_STATUS,
    TRADING_AUTHORIZATION_STATUS,
    TraceabilityRecord,
)


def _index_ae9_records(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_decision_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("source_decision_id")
        if sid and str(sid) not in by_decision_id:
            by_decision_id[str(sid)] = rec
        for key in (rec.get("candidate_id"), rec.get("pair_address"), rec.get("audit_record_id")):
            if key and str(key) not in by_key:
                by_key[str(key)] = rec
    return by_key, by_decision_id


def _match_ae9(
    decision: dict[str, Any] | None,
    ae9_by_key: dict[str, dict[str, Any]],
    ae9_by_decision_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not decision:
        return None
    decision_id = decision.get("decision_id")
    if decision_id and str(decision_id) in ae9_by_decision_id:
        return ae9_by_decision_id[str(decision_id)]
    identity = decision.get("candidate_identity") or {}
    for key in (identity.get("candidate_id"), identity.get("pair_address"), decision_id):
        if key and str(key) in ae9_by_key:
            return ae9_by_key[str(key)]
    return None


def _opportunity_capture_observations(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Forward-research observation only — no model claims, no future-return justification."""
    observations: list[dict[str, Any]] = []
    for decision in decisions:
        market = decision.get("market_context") or {}
        identity = decision.get("candidate_identity") or {}
        liquidity = float(market.get("liquidity") or 0)
        whale = float(market.get("whale_score") or 0)
        h24 = market.get("price_change_h24")
        filter_status = market.get("filter_status")
        flags: list[str] = []
        if liquidity >= 100_000:
            flags.append("high_liquidity")
        if whale >= 0.5:
            flags.append("high_whale_score")
        if h24 is not None and abs(float(h24)) >= 20:
            flags.append("strong_24h_move_observed_at_snapshot")
        if filter_status == "passed":
            flags.append("passed_clean_filter")
        if flags:
            observations.append(
                {
                    "candidate_id": identity.get("candidate_id"),
                    "symbol": identity.get("symbol"),
                    "pair_address": identity.get("pair_address"),
                    "observation_flags": flags,
                    "note": "Opportunity Capture Audit — observation only; 24h move is snapshot metadata not entry-time approval",
                }
            )
    return observations


def _index_by_key(records: list[dict[str, Any]], *keys: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        for key in keys:
            val = rec.get(key)
            if val and str(val) not in index:
                index[str(val)] = rec
            identity = rec.get("candidate_identity") or {}
            for ik in ("candidate_id", "pair_address"):
                iv = identity.get(ik)
                if iv and str(iv) not in index:
                    index[str(iv)] = rec
    return index


def build_traceability_record(
    decision: dict[str, Any] | None,
    context: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    *,
    execution_mode: str = "PAPER_DEMO",
) -> TraceabilityRecord:
    identity = (decision or {}).get("candidate_identity") or {}
    candidate_id = (
        identity.get("candidate_id")
        or (context or {}).get("candidate_id")
        or (audit or {}).get("candidate_id")
        or ""
    )
    missing: list[str] = []
    if not candidate_id:
        missing.append("candidate_id")

    source_decision_id = (decision or {}).get("decision_id")
    source_context_id = (context or {}).get("context_record_id")
    source_audit_id = (audit or {}).get("audit_record_id")

    trace_status = "COMPLETE"
    if missing:
        trace_status = "INCOMPLETE"
    elif not source_decision_id:
        trace_status = "PARTIAL_NO_DECISION"

    return TraceabilityRecord(
        candidate_id=candidate_id,
        source_decision_id=source_decision_id,
        source_context_record_id=source_context_id,
        source_llm_audit_record_id=source_audit_id,
        decision_status=(decision or {}).get("decision_status"),
        consensus_family=((decision or {}).get("consensus") or {}).get("consensus_family"),
        context_schema_id=(context or {}).get("context_schema_id"),
        audit_verdict=(audit or {}).get("llm_verdict"),
        audit_blockers=list((audit or {}).get("audit_blockers") or []),
        audit_warnings=list((audit or {}).get("audit_warnings") or []),
        scoring_policy_id=(decision or {}).get("scoring_policy_id")
        or identity.get("candidate_policy_id"),
        execution_mode=execution_mode,
        no_wallet_dry_run=execution_mode == "LIVE_NO_WALLET_DRY_RUN",
        no_live_submission=True,
        traceability_status=trace_status,
    )


def _llm_provider_status(
    provider: str,
    *,
    allow_local_qwen: bool,
    allow_ollama: bool,
    allow_gemini: bool,
    external_calls_made: int,
    local_llm_calls_made: int,
) -> dict[str, Any]:
    statuses: list[str] = ["MOCK_DEFAULT", "DISABLED_BY_DEFAULT"]
    if provider == "mock":
        statuses = ["MOCK_DEFAULT", "DISABLED_BY_DEFAULT"]
    elif provider == "qwen":
        statuses = ["LOCAL_QWEN_AVAILABLE" if allow_local_qwen else "CONFIG_MISSING"]
    elif provider == "ollama":
        statuses = ["OLLAMA_AVAILABLE" if allow_ollama else "CONFIG_MISSING"]
    elif provider == "gemini":
        statuses = ["GEMINI_CONFIGURED" if allow_gemini else "CONFIG_MISSING"]

    if provider != "mock" and not (allow_local_qwen or allow_ollama or allow_gemini):
        statuses.append("NOT_RUN")

    return {
        "provider_selected": provider,
        "allow_local_qwen": allow_local_qwen,
        "allow_ollama": allow_ollama,
        "allow_gemini": allow_gemini,
        "external_calls_made": external_calls_made,
        "local_llm_calls_made": local_llm_calls_made,
        "audit_only_status": True,
        "llm_trade_authority": False,
        "statuses": statuses,
    }


def _determine_final_status(
    *,
    traceability_records: list[TraceabilityRecord],
    paper_orders_created: int,
    live_dry_run_created: int,
    enable_paper: bool,
    enable_live_dry_run: bool,
    price_lookahead: int,
    price_missing: int,
    blocked_reasons: list[str],
) -> str:
    if not traceability_records and blocked_reasons:
        return Ae10FinalStatus.AE10_BLOCKED_NO_INPUT_ARTIFACTS.value
    if price_lookahead > 0:
        return Ae10FinalStatus.AE10_BLOCKED_PRICE_LOOKAHEAD.value
    if blocked_reasons:
        return Ae10FinalStatus.AE10_BLOCKED_WITH_EXACT_REASONS.value
    if enable_live_dry_run and live_dry_run_created > 0:
        return Ae10FinalStatus.AE10_LIVE_DRY_RUN_WIRED_NO_WALLET.value
    if enable_paper and paper_orders_created > 0:
        return Ae10FinalStatus.AE10_PAPER_DEMO_READY.value
    if enable_paper and paper_orders_created == 0:
        return Ae10FinalStatus.AE10_PAPER_DEMO_PARTIAL_NO_ORDERS.value
    if traceability_records:
        return Ae10FinalStatus.AE10_TRACEABILITY_READY.value
    return Ae10FinalStatus.AE10_BLOCKED_NO_INPUT_ARTIFACTS.value


def run_ae10_trading_orchestration(
    *,
    project_root: Path,
    max_records: int = 50,
    audit_only: bool = True,
    no_db_write: bool = True,
    enable_paper_demo_orders: bool = False,
    allow_paper_trades_with_audit_blockers: bool = False,
    enable_live_dry_run_orders: bool = False,
    reset_demo_account_flag: bool = False,
    clear_history: bool = False,
    starting_balance_usd: float = 10_000.0,
    max_price_age_seconds: float = 30.0,
    provider: str = "mock",
    allow_local_qwen: bool = False,
    allow_ollama: bool = False,
    allow_gemini: bool = False,
    output_root: Path | None = None,
    ae6_jsonl: Path | None = None,
    ae8_context_jsonl: Path | None = None,
    ae9_audit_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Main AE10 orchestration."""
    from datetime import datetime, timezone

    from scripts.diagnostics._common import open_db_readonly, timestamp_slug

    ts_slug = timestamp_slug()
    audit_dir = output_root or (
        project_root / "data" / "audits" / f"ae10_trading_orchestration_{ts_slug}"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    ae6_path = ae6_jsonl or discover_latest_glob(
        project_root, "data/decision_records/ae6_decisions_*.jsonl"
    )
    ae8_path = ae8_context_jsonl or discover_latest_glob(
        project_root, "data/context_intelligence/ae8_context_features_*.jsonl"
    )
    ae9_path = ae9_audit_jsonl or discover_latest_glob(
        project_root, "data/llm_audit/ae9_llm_audit_records_*.jsonl"
    )

    source_paths = {
        "ae6_decisions_jsonl": str(ae6_path.resolve()) if ae6_path and ae6_path.is_file() else None,
        "ae8_context_features_jsonl": str(ae8_path.resolve()) if ae8_path and ae8_path.is_file() else None,
        "ae9_llm_audit_jsonl": str(ae9_path.resolve()) if ae9_path and ae9_path.is_file() else None,
    }

    ae6_records: list[dict[str, Any]] = []
    ae8_records: list[dict[str, Any]] = []
    ae9_records: list[dict[str, Any]] = []

    if ae6_path and ae6_path.is_file():
        ae6_records, _ = read_jsonl_records_safe(ae6_path)
    if ae8_path and ae8_path.is_file():
        ae8_records, _ = read_jsonl_records_safe(ae8_path)
    if ae9_path and ae9_path.is_file():
        ae9_records, _ = read_jsonl_records_safe(ae9_path)

    ae8_index = _index_by_key(ae8_records, "candidate_id", "pair_address", "context_record_id")
    ae9_by_key, ae9_by_decision_id = _index_ae9_records(ae9_records)

    candidates: list[tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]] = []
    if ae6_records:
        for decision in ae6_records[:max_records]:
            identity = decision.get("candidate_identity") or {}
            ctx = None
            for key in (identity.get("candidate_id"), identity.get("pair_address"), decision.get("decision_id")):
                if key and str(key) in ae8_index:
                    ctx = ae8_index[str(key)]
                    break
            aud = _match_ae9(decision, ae9_by_key, ae9_by_decision_id)
            candidates.append((decision, ctx, aud))
    elif ae8_records:
        for ctx in ae8_records[:max_records]:
            aud = ae9_by_key.get(str(ctx.get("candidate_id")))
            candidates.append((None, ctx, aud))
    elif ae9_records:
        for aud in ae9_records[:max_records]:
            candidates.append((None, None, aud))

    ae9_linkage_diagnostics: dict[str, Any] = {
        "ae9_records_loaded": len(ae9_records),
        "ae9_unique_source_decision_ids": len(ae9_by_decision_id),
        "missing_link_reasons": [],
    }
    if not ae9_records:
        ae9_linkage_diagnostics["missing_link_reasons"].append("missing_ae9_artifact")
    elif not ae9_by_decision_id:
        ae9_linkage_diagnostics["missing_link_reasons"].append("ae9_records_lack_source_decision_id")

    opportunity_observations = _opportunity_capture_observations(
        [d for d, _, _ in candidates if d]
    )

    saved_state = load_demo_account_state()
    if saved_state and not no_db_write:
        ledger = DemoLedger(account=DemoAccount.from_dict(saved_state))
    else:
        ledger = DemoLedger(
            account=DemoAccount(starting_balance_usd=starting_balance_usd, cash_balance_usd=starting_balance_usd)
        )

    reset_audit: dict[str, Any] | None = None
    if reset_demo_account_flag:
        reset_audit = reset_demo_account(
            ledger,
            starting_balance_usd=starting_balance_usd,
            clear_history=clear_history,
        )

    price_oracle = DemoPriceOracle(max_price_age_seconds=max_price_age_seconds)
    simulator = PaperOrderSimulator(price_oracle=price_oracle)
    paper_adapter = PaperExecutionAdapter(ledger, simulator)
    live_adapter = LiveNoWalletDryRunAdapter()

    conn = None
    if not no_db_write:
        try:
            conn = open_db_readonly()
        except FileNotFoundError:
            conn = None
    elif ae6_records:
        try:
            conn = open_db_readonly()
        except FileNotFoundError:
            conn = None

    traceability_records: list[TraceabilityRecord] = []
    traceability_csv_rows: list[dict[str, Any]] = []
    live_dry_run_orders: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    price_lookahead = 0
    price_missing = 0
    price_stale = 0
    price_skew = 0
    external_calls_made = 0
    local_llm_calls_made = 0

    paper_orders_writer: JsonlWriter | None = None
    paper_positions_writer: JsonlWriter | None = None
    paper_trades_writer: JsonlWriter | None = None
    live_writer: JsonlWriter | None = None

    if not audit_only and not no_db_write:
        paper_orders_writer = JsonlWriter(paper_orders_path_for_date())
        paper_positions_writer = JsonlWriter(paper_positions_path_for_date())
        paper_trades_writer = JsonlWriter(paper_trades_path_for_date())
        live_writer = JsonlWriter(live_dry_run_orders_path_for_date())

    for decision, context, audit in candidates:
        trace = build_traceability_record(decision, context, audit)
        traceability_records.append(trace)
        traceability_csv_rows.append(
            {
                "traceability_id": trace.traceability_id,
                "candidate_id": trace.candidate_id,
                "source_decision_id": trace.source_decision_id or "",
                "source_context_record_id": trace.source_context_record_id or "",
                "source_llm_audit_record_id": trace.source_llm_audit_record_id or "",
                "traceability_status": trace.traceability_status,
            }
        )

        if not trace.candidate_id:
            blocked_reasons.append("missing_candidate_id")
            continue

        identity = (decision or {}).get("candidate_identity") or {}
        decision_created_at_utc = (decision or {}).get("created_at_utc")
        order_created_at_utc = datetime.now(timezone.utc).isoformat()
        coin_id = identity.get("coin_id")

        price_result = price_oracle.lookup_price(
            coin_id=coin_id,
            pair_address=identity.get("pair_address") or trace.candidate_id,
            order_created_at_utc=order_created_at_utc,
            decision_created_at_utc=decision_created_at_utc,
            conn=conn,
        )
        price_dict = price_result.to_dict()

        if price_result.price_status == PriceStatus.PRICE_LOOKAHEAD_REJECTED.value:
            price_lookahead += 1
        if price_result.price_status == PriceStatus.PRICE_STALE.value:
            price_stale += 1
        if price_result.price_status == PriceStatus.PRICE_PROVIDER_TIME_SKEW_REJECTED.value:
            price_skew += 1
        if price_result.price_status == PriceStatus.PRICE_MISSING.value:
            price_missing += 1

        trace_dict = trace.to_dict()
        intent = OrderIntent(
            candidate_id=trace.candidate_id,
            symbol=identity.get("symbol") or (context or {}).get("symbol") or "",
            pair_address=identity.get("pair_address") or (context or {}).get("pair_address") or "",
            notional_usd=100.0,
            requested_price_usd=price_result.price,
            source_decision_id=trace.source_decision_id,
            source_context_record_id=trace.source_context_record_id,
            source_llm_audit_record_id=trace.source_llm_audit_record_id,
            decision_status=trace.decision_status,
            consensus_family=trace.consensus_family,
            context_schema_id=trace.context_schema_id,
            audit_verdict=trace.audit_verdict,
            audit_blockers=trace.audit_blockers,
            audit_warnings=trace.audit_warnings,
            scoring_policy_id=trace.scoring_policy_id,
            decision_created_at_utc=decision_created_at_utc,
            order_created_at_utc=order_created_at_utc,
            coin_id=coin_id,
        )

        if enable_paper_demo_orders:
            result = paper_adapter.execute(
                intent,
                traceability=trace_dict,
                price_result=price_dict,
                allow_audit_blockers=allow_paper_trades_with_audit_blockers,
            )
            if paper_orders_writer and result.record:
                paper_orders_writer.append_dict(result.record)
                if result.success:
                    pos = ledger.positions[-1] if ledger.positions else None
                    if pos and paper_positions_writer:
                        paper_positions_writer.append_dict(pos.to_dict())

        if enable_live_dry_run_orders:
            live_result = live_adapter.execute(intent)
            live_dry_run_orders.append(live_result.record)
            if live_writer:
                live_writer.append_dict(live_result.record)

    if conn:
        conn.close()

    for w in (paper_orders_writer, paper_positions_writer, paper_trades_writer, live_writer):
        if w:
            w.close()

    demo_state_path: Path | None = None
    if not audit_only and not no_db_write:
        demo_state_path = save_demo_account_state(ledger.account.to_dict())

    paper_orders = [o.to_dict() for o in ledger.orders]
    paper_positions = [p.to_dict() for p in ledger.positions]
    paper_trades = [t.to_dict() for t in ledger.trades]

    filled_orders = [o for o in ledger.orders if o.status == PaperOrderStatus.PAPER_FILLED.value]
    rejected_orders = [o for o in ledger.orders if o.status == PaperOrderStatus.PAPER_REJECTED.value]
    rejection_reasons: dict[str, int] = {}
    for o in rejected_orders:
        reason = o.paper_trade_reason or "unknown"
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    latency_values = [
        o.execution_latency_ms
        for o in ledger.orders
        if o.execution_latency_ms is not None
    ]
    latency_min = min(latency_values) if latency_values else None
    latency_max = max(latency_values) if latency_values else None
    latency_mean = sum(latency_values) / len(latency_values) if latency_values else None

    missing_decision_ts = sum(
        1 for o in ledger.orders if o.execution_latency_status == "MISSING_DECISION_TIMESTAMP"
    )
    not_filled = sum(1 for o in ledger.orders if o.execution_latency_status == "NOT_FILLED")

    price_status_counts: dict[str, int] = {}
    for row in price_oracle.audit_log:
        ps = row.get("price_status", "UNKNOWN")
        price_status_counts[ps] = price_status_counts.get(ps, 0) + 1

    valid_transitions = sum(
        1 for r in simulator.state_machine.audit_log if r.get("transition_allowed")
    )
    rejected_transitions = sum(
        1 for r in simulator.state_machine.audit_log if not r.get("transition_allowed")
    )

    llm_audit = _llm_provider_status(
        provider,
        allow_local_qwen=allow_local_qwen,
        allow_ollama=allow_ollama,
        allow_gemini=allow_gemini,
        external_calls_made=external_calls_made,
        local_llm_calls_made=local_llm_calls_made,
    )

    final_status = _determine_final_status(
        traceability_records=traceability_records,
        paper_orders_created=len(filled_orders) + len(rejected_orders) if enable_paper_demo_orders else 0,
        live_dry_run_created=len(live_dry_run_orders),
        enable_paper=enable_paper_demo_orders,
        enable_live_dry_run=enable_live_dry_run_orders,
        price_lookahead=price_lookahead,
        price_missing=price_missing,
        blocked_reasons=blocked_reasons,
    )

    with_source_decision = sum(1 for t in traceability_records if t.source_decision_id)
    with_source_context = sum(1 for t in traceability_records if t.source_context_record_id)
    with_source_audit = sum(1 for t in traceability_records if t.source_llm_audit_record_id)
    without_source_audit = len(traceability_records) - with_source_audit
    missing_trace = sum(1 for t in traceability_records if t.traceability_status != "COMPLETE")

    if without_source_audit > 0:
        if not ae9_records:
            ae9_linkage_diagnostics["missing_link_reasons"].append("missing_ae9_artifact")
        elif ae9_by_decision_id and with_source_decision > 0:
            ae9_linkage_diagnostics["missing_link_reasons"].append("source_decision_id_mismatch_between_ae6_and_ae9")
        if with_source_decision == 0:
            ae9_linkage_diagnostics["missing_link_reasons"].append("missing_source_decision_id_in_traceability")
    ae9_linkage_diagnostics["records_with_source_llm_audit_record_id"] = with_source_audit
    ae9_linkage_diagnostics["records_without_source_llm_audit_record_id"] = without_source_audit

    summary = {
        "phase": AE10_PHASE,
        "final_status": final_status,
        "traceability_records_created": len(traceability_records),
        "records_with_source_decision_id": with_source_decision,
        "records_with_source_context_record_id": with_source_context,
        "records_with_source_llm_audit_record_id": with_source_audit,
        "records_missing_traceability": missing_trace,
        "paper_orders_created": len(ledger.orders) if enable_paper_demo_orders else 0,
        "paper_orders_filled": len(filled_orders),
        "paper_orders_rejected": len(rejected_orders),
        "rejection_reasons": rejection_reasons,
        "paper_positions_opened": len([p for p in ledger.positions if p.status == "OPEN"]),
        "paper_trades_closed": len(ledger.trades),
        "live_dry_run_orders_created": len(live_dry_run_orders),
        "demo_account_balance": ledger.account.cash_balance_usd,
        "demo_account_equity": ledger.account.equity_usd,
        "reset_demo_account_executed": reset_demo_account_flag,
        "wallet_configured": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "source_paths": source_paths,
        "output_root": str(audit_dir.resolve()),
        "demo_account_state_path": str(demo_state_path.resolve()) if demo_state_path else None,
    }

    decision_gate = {
        "final_status": final_status,
        "traceability_records_created": len(traceability_records),
        "paper_orders_created": len(ledger.orders) if enable_paper_demo_orders else 0,
        "paper_positions_opened": len([p for p in ledger.positions if p.status == "OPEN"]),
        "paper_trades_closed": len(ledger.trades),
        "live_dry_run_orders_created": len(live_dry_run_orders),
        "demo_account_balance": ledger.account.cash_balance_usd,
        "reset_demo_account_executed": reset_demo_account_flag,
        "wallet_configured": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "llm_provider_status": llm_audit,
        "runtime_inference_status": RUNTIME_INFERENCE_STATUS,
        "trading_authorization_status": TRADING_AUTHORIZATION_STATUS,
        "recommended_next_phase": "AE11_LIVE_WALLET_PROVIDER_WIRING",
        "blocked_reasons": blocked_reasons,
        "enable_paper_demo_orders": enable_paper_demo_orders,
        "enable_live_dry_run_orders": enable_live_dry_run_orders,
    }

    output_paths = write_ae10_audits(
        audit_dir,
        summary=summary,
        decision_gate=decision_gate,
        traceability_records=[t.to_dict() for t in traceability_records],
        paper_orders=paper_orders,
        paper_positions=paper_positions,
        live_dry_run_orders=live_dry_run_orders,
        no_wallet_audit=live_adapter.audit_summary(),
        reset_audit=reset_audit,
        traceability_csv_rows=traceability_csv_rows,
        llm_provider_audit=llm_audit,
        state_machine_rows=simulator.state_machine.audit_log,
        price_oracle_rows=price_oracle.audit_log,
        latency_rows=simulator.latency_audit,
    )
    output_paths = {k: str(Path(v).resolve()) for k, v in output_paths.items()}

    return {
        **summary,
        "decision_gate": decision_gate,
        "output_paths": output_paths,
        "paper_demo_summary": {
            "demo_account_initialized": True,
            "reset_executed": reset_demo_account_flag,
            "starting_balance": starting_balance_usd,
            "ending_balance": ledger.account.cash_balance_usd,
            "paper_orders_created": len(ledger.orders) if enable_paper_demo_orders else 0,
            "paper_positions_opened": len([p for p in ledger.positions if p.status == "OPEN"]),
            "paper_trades_closed": len(ledger.trades),
            "rejected_paper_orders": len(rejected_orders),
            "rejection_reasons": rejection_reasons,
        },
        "state_machine_summary": {
            "valid_transitions": valid_transitions,
            "rejected_invalid_transitions": rejected_transitions,
            "state_machine_audit_path": output_paths.get("ae10_order_state_machine_audit"),
        },
        "price_oracle_summary": {
            "max_price_age_seconds": max_price_age_seconds,
            "price_lookups_attempted": len(price_oracle.audit_log),
            "PRICE_OK": price_status_counts.get("PRICE_OK", 0),
            "PRICE_MISSING": price_status_counts.get("PRICE_MISSING", 0),
            "PRICE_STALE": price_status_counts.get("PRICE_STALE", 0),
            "PRICE_LOOKAHEAD_REJECTED": price_status_counts.get("PRICE_LOOKAHEAD_REJECTED", 0),
            "PRICE_PROVIDER_TIME_SKEW_REJECTED": price_status_counts.get(
                "PRICE_PROVIDER_TIME_SKEW_REJECTED", 0
            ),
            "PRICE_INVALID_ZERO_OR_NEGATIVE": price_status_counts.get(
                "PRICE_INVALID_ZERO_OR_NEGATIVE", 0
            ),
            "future_snapshots_rejected": price_lookahead,
            "stale_snapshots_rejected": price_stale,
            "provider_time_skew_rejected": price_skew,
            "rejected_for_price": price_missing + price_stale + price_lookahead + price_skew,
        },
        "execution_latency_summary": {
            "records_with_execution_latency_ms": len(latency_values),
            "missing_decision_timestamp": missing_decision_ts,
            "not_filled": not_filled,
            "latency_min_ms": latency_min,
            "latency_max_ms": latency_max,
            "latency_mean_ms": latency_mean,
        },
        "live_adapter_summary": {
            "live_dry_run_orders_created": len(live_dry_run_orders),
            "wallet_configured": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "no_wallet_adapter_status": live_adapter.audit_summary(),
        },
        "llm_provider_summary": llm_audit,
        "ae9_linkage_diagnostics": ae9_linkage_diagnostics,
        "opportunity_capture_observations": opportunity_observations,
        "safety_confirmation": {
            "no_model_training": True,
            "no_real_llm_calls_by_default": provider == "mock",
            "external_llms_optional": True,
            "no_external_api_calls_by_default": external_calls_made == 0,
            "no_private_key_access": not live_adapter.private_key_accessed,
            "no_real_wallet_calls": True,
            "no_real_transaction_signing": True,
            "no_real_transaction_submission": True,
            "paper_demo_reset_only": True,
            "live_adapter_no_wallet_dry_run": True,
            "no_destructive_db_operations": True,
        },
    }
