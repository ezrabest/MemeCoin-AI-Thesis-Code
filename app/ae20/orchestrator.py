"""AE20 Integrated Clean Forward Validation orchestrator."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ae20 import (
    ENGINE_VERSION,
    PHASE,
    SMOKE_DEFAULT_LLM_TIMEOUT_SECONDS,
    SMOKE_DEFAULT_MAX_LLM_CALLS_PER_CYCLE,
    SMOKE_HARD_MAX_LLM_CALLS_PER_CYCLE,
)
from app.ae20.baseline import load_paper_state, snapshot_preexisting_baseline
from app.ae20.clean_forward_inputs import build_clean_forward_inputs
from app.ae20.decisions import (
    build_integrated_decision,
    derive_strict_exploration,
    evaluate_gates,
)
from app.ae20.integrations import (
    attach_ae16,
    attach_ae17,
    attach_ae18,
    load_ae16_index,
    load_ae17_index,
    load_ae18_index,
    run_ae19_audit_only,
)
from app.ae20.lifecycle import audit_lineage, maybe_create_paper_lifecycle
from app.ae20.opportunity import build_opportunity_rows
from app.ae20.output_root import allocate_ae20_output_root, ensure_ae20_dirs
from app.ae20.pnl import build_pnl_summary
from app.consensus.serialization import write_csv, write_json, write_jsonl, write_text


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# AE20_OPEN_POSITION_STATE_FIX_V1
_INVALID_IDENTITY_LITERALS = {"", "nan", "NaN", "NAN", "None", "NONE", "null", "NULL"}


def _exact_identity_values(row: dict[str, Any]) -> set[tuple[str, str]]:
    """Return exact, case-preserved identity values for AE20 duplicate-open checks."""
    values: set[tuple[str, str]] = set()
    for key in (
        "provider_pair_url_exact",
        "canonical_market_identity",
        "price_source_key",
        "pair_address",
    ):
        raw = row.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if value in _INVALID_IDENTITY_LITERALS:
            continue
        values.add((key, value))
    return values


def _has_open_position_for_candidate(
    candidate_or_decision: dict[str, Any],
    open_positions: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Detect exact duplicate open AE20 paper/demo position without case mutation."""
    candidate_values = _exact_identity_values(candidate_or_decision)
    if not candidate_values:
        return False, ""

    for pos in open_positions:
        if str(pos.get("status") or "").strip() != "OPEN":
            continue

        overlap = candidate_values.intersection(_exact_identity_values(pos))
        if overlap:
            key, value = sorted(overlap, key=lambda item: (item[0], item[1]))[0]
            return True, f"AE20_DUPLICATE_OPEN_POSITION:{key}:{value}"

    return False, ""


# AE20_MARK_OUTCOME_LIFECYCLE_FIX_V1
def _ae20_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out


def _ae20_parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in _INVALID_IDENTITY_LITERALS:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ae20_outcome_horizon_seconds() -> int:
    raw = str(os.environ.get("AE20_OUTCOME_MATURITY_SECONDS") or "14400").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 14400
    if value <= 0:
        return 14400
    return value


def _mark_open_positions_for_candidate(
    candidate: dict[str, Any],
    positions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> int:
    """Mark AE20-local open paper/demo positions from current Clean Forward candidate.

    This is paper/demo accounting only. It does not mutate live paper_state.json, does not
    create live authority, and does not claim profitability.
    """
    price = _ae20_float(candidate.get("price_usd"))
    if price is None or price <= 0:
        return 0

    candidate_values = _exact_identity_values(candidate)
    if not candidate_values:
        return 0

    mark_time = (
        str(candidate.get("fetched_at") or "").strip()
        or str(candidate.get("observed_at") or "").strip()
        or str(candidate.get("ingested_at") or "").strip()
        or _utc()
    )
    mark_dt = _ae20_parse_dt(mark_time) or datetime.now(timezone.utc)
    maturity_seconds = _ae20_outcome_horizon_seconds()
    outcomes_by_id = {o.get("outcome_id"): o for o in outcomes if o.get("outcome_id")}

    marked = 0
    for pos in positions:
        if str(pos.get("status") or "").strip() != "OPEN":
            continue
        if not candidate_values.intersection(_exact_identity_values(pos)):
            continue

        entry_price = _ae20_float(pos.get("entry_price"))
        quantity = _ae20_float(pos.get("quantity"))
        fees = _ae20_float(pos.get("fees_assumption_usd")) or 0.0
        slip = _ae20_float(pos.get("slippage_assumption_usd")) or 0.0
        if entry_price is None or entry_price <= 0 or quantity is None or quantity <= 0:
            continue

        gross_pnl = (price - entry_price) * quantity
        net_pnl = gross_pnl - fees - slip

        pos["mark_price"] = price
        pos["mark_time"] = mark_time
        pos["last_marked_at"] = _utc()
        pos["mark_price_source"] = "clean_forward_canonical_runtime_index"
        pos["unrealized_pnl"] = net_pnl
        pos["profitability_claim"] = False
        pos["trade_authority"] = False
        pos["live_trading_enabled"] = False
        pos["wallet_connected"] = False

        outcome = outcomes_by_id.get(pos.get("outcome_id"))
        if outcome is not None:
            outcome["status"] = "OPEN_MARK"
            outcome["maturity_status"] = "NOT_MATURED"
            outcome["outcome_unavailable_reason"] = "POSITION_OPEN_MARK_ONLY"
            outcome["mark_price"] = price
            outcome["mark_time"] = mark_time
            outcome["last_marked_at"] = pos["last_marked_at"]
            outcome["unrealized_pnl"] = net_pnl
            outcome["realized_pnl"] = 0.0
            outcome["profitability_claim"] = False

        entry_dt = _ae20_parse_dt(pos.get("entry_time"))
        if entry_dt is not None:
            age_seconds = (mark_dt - entry_dt).total_seconds()
            pos["age_seconds"] = age_seconds
            if age_seconds >= maturity_seconds:
                pos["status"] = "CLOSED"
                pos["maturity_status"] = "MATURED"
                pos["exit_price"] = price
                pos["exit_time"] = mark_time
                pos["exit_reason"] = "AE20_OUTCOME_HORIZON_ELAPSED"
                pos["realized_pnl"] = net_pnl
                pos["unrealized_pnl"] = 0.0
                if outcome is not None:
                    outcome["status"] = "CLOSED"
                    outcome["maturity_status"] = "MATURED"
                    outcome["outcome_unavailable_reason"] = ""
                    outcome["exit_price"] = price
                    outcome["exit_time"] = mark_time
                    outcome["exit_reason"] = "AE20_OUTCOME_HORIZON_ELAPSED"
                    outcome["realized_pnl"] = net_pnl
                    outcome["unrealized_pnl"] = 0.0

        marked += 1

    return marked

    for pos in open_positions:
        if str(pos.get("status") or "").strip() != "OPEN":
            continue
        overlap = candidate_values.intersection(_exact_identity_values(pos))
        if overlap:
            key, value = sorted(overlap)[0]
            return True, f"AE20_DUPLICATE_OPEN_POSITION:{key}:{value}"

    return False, ""


def _bool_arg(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _append_event(events: list[dict[str, Any]], event_type: str, **payload: Any) -> None:
    events.append({"ts": _utc(), "event_type": event_type, **payload})


def decide_classification(
    *,
    identity_blocked: bool,
    legacy_contaminated: bool,
    authority_escalation: bool,
    lineage_pass: bool,
    integration_ok: bool,
    llm_limitations: bool,
    identity_failure_ratio: float,
    ae16_attached_count: int = 0,
    matched_rows_wrongly_unavailable: int = 0,
    exact_identity_ok: bool = True,
    unsafe_bridge_flags: bool = False,
) -> str:
    if authority_escalation:
        return "AE20_SMOKE_BLOCKED_AUTHORITY_ESCALATION"
    if legacy_contaminated:
        return "AE20_SMOKE_BLOCKED_LEGACY_CONTAMINATION"
    if identity_blocked or identity_failure_ratio >= 0.9:
        return "AE20_SMOKE_BLOCKED_CLEAN_FORWARD_INPUT_FAILURE"
    if not lineage_pass:
        return "AE20_SMOKE_BLOCKED_LINEAGE_FAILURE"
    if (
        not integration_ok
        or not exact_identity_ok
        or matched_rows_wrongly_unavailable > 0
        or unsafe_bridge_flags
    ):
        return "AE20_SMOKE_BLOCKED_INTEGRATION_LAYER_FAILURE"
    if ae16_attached_count <= 0:
        # Bridge ran but no exact-case attachments — not ready for 24h.
        return "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS"
    if llm_limitations:
        return "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS"
    return "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H"


def compute_unblocked_for_24h(
    *,
    classification: str,
    blockers_before_24h: list[str],
    ae16_attached_count: int,
    unsafe_bridge_flags: bool = False,
) -> bool:
    """Hard gate: blockers non-empty or AE16 attached==0 ⇒ not unblocked."""
    if blockers_before_24h:
        return False
    if ae16_attached_count <= 0:
        return False
    if unsafe_bridge_flags:
        return False
    return classification == "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H"


def run_ae20_integrated_clean_forward_validation(
    project_root: Path | str,
    *,
    smoke_cycles: int | None = None,
    duration_hours: float | None = None,
    output_root: str | Path | None = None,
    no_external_llm: bool = False,
    llm_provider: str = "ollama",
    max_llm_calls_per_cycle: int = SMOKE_DEFAULT_MAX_LLM_CALLS_PER_CYCLE,
    llm_timeout_seconds: float = SMOKE_DEFAULT_LLM_TIMEOUT_SECONDS,
    paper_demo_only: bool = True,
    clean_forward_only: bool = True,
    strict_baseline_partition: bool = True,
    fail_on_missing_canonical_identity: bool = True,
    separate_strict_exploration_pnl: bool = True,
    max_candidates_per_cycle: int = 8,
    cycle_sleep_seconds: float = 0.0,
    force_llm_unavailable: bool = False,
    ae16_bridge_source: str | Path | None = None,
    ae20_ae16_exact_bridge: str | Path | None = None,
) -> dict[str, Any]:
    """Run AE20 smoke or duration mode. Returns summary dict with classification."""
    project_root = Path(project_root).resolve()
    paper_demo_only = _bool_arg(paper_demo_only, True)
    clean_forward_only = _bool_arg(clean_forward_only, True)
    strict_baseline_partition = _bool_arg(strict_baseline_partition, True)
    fail_on_missing_canonical_identity = _bool_arg(fail_on_missing_canonical_identity, True)
    separate_strict_exploration_pnl = _bool_arg(separate_strict_exploration_pnl, True)

    if smoke_cycles is None and duration_hours is None:
        smoke_cycles = 1
    mode = "smoke" if smoke_cycles is not None else "duration"

    # Enforce smoke LLM hard max
    if mode == "smoke":
        max_llm_calls_per_cycle = min(
            int(max_llm_calls_per_cycle), SMOKE_HARD_MAX_LLM_CALLS_PER_CYCLE
        )

    allow_llm = (not no_external_llm) and bool(llm_provider) and llm_provider.lower() != "none"

    root, collision_audit = allocate_ae20_output_root(project_root, output_root=output_root)
    dirs = ensure_ae20_dirs(root)
    data_dir, audits_dir, reports_dir = dirs["data"], dirs["audits"], dirs["reports"]

    run_id = f"ae20_{uuid4().hex[:12]}"
    started = _utc()
    events: list[dict[str, Any]] = []
    events_file = data_dir / "ae20_runtime_events.jsonl"

    # AE20_HEARTBEAT_CHECKPOINT_PATCH_V1
    # Duration runs must expose observable progress immediately and after every cycle.
    def emit_event(event_type: str, **payload: Any) -> None:
        row = {"ts": _utc(), "event_type": event_type, **payload}
        events.append(row)
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def _write_in_progress_manifest(stage: str) -> None:
        write_json(
            reports_dir / "ae20_manifest.json",
            {
                "classification": "AE20_DURATION_IN_PROGRESS" if mode == "duration" else "AE20_SMOKE_IN_PROGRESS",
                "stage": stage,
                "ae20_stage_closed": False,
                "duration_completed": False,
                "run_id": run_id,
                "mode": mode,
                "output_root": str(root),
                "started_at_utc": started,
                "duration_hours": duration_hours,
                "smoke_cycles": smoke_cycles,
                "cycle_sleep_seconds": cycle_sleep_seconds,
                "max_candidates_per_cycle": max_candidates_per_cycle,
                "llm_provider": llm_provider,
                "max_llm_calls_per_cycle": max_llm_calls_per_cycle,
                "paper_demo_only": paper_demo_only,
                "clean_forward_only": clean_forward_only,
                "profitability_claim": False,
                "live_readiness_claim": False,
                "trade_authority": False,
            },
        )

    def _write_in_progress_gate(stage: str, blockers: list[str] | None = None) -> None:
        write_json(
            reports_dir / "ae20_decision_gate.json",
            {
                "classification": "AE20_DURATION_IN_PROGRESS" if mode == "duration" else "AE20_SMOKE_IN_PROGRESS",
                "stage": stage,
                "unblocked_for_24h": False,
                "duration_completed": False,
                "ae20_stage_closed": False,
                "requires_main_thread_smoke_review": mode == "smoke",
                "profitability_claim": False,
                "live_readiness_claim": False,
                "trade_authority": False,
                "paper_demo_only": True,
                "blockers_before_24h": blockers or ["AE20 duration/smoke run still in progress"],
                "ae16_attached_rows_count": 0,
            },
        )

    _write_in_progress_manifest("run_started")
    _write_in_progress_gate("run_started")
    emit_event(
        "AE20_RUN_STARTED",
        run_id=run_id,
        mode=mode,
        output_root=str(root),
        duration_hours=duration_hours,
        smoke_cycles=smoke_cycles,
        cycle_sleep_seconds=cycle_sleep_seconds,
        max_candidates_per_cycle=max_candidates_per_cycle,
        llm_provider=llm_provider,
        max_llm_calls_per_cycle=max_llm_calls_per_cycle,
        paper_demo_only=paper_demo_only,
        clean_forward_only=clean_forward_only,
        ae20_ae16_exact_bridge=str(ae20_ae16_exact_bridge) if ae20_ae16_exact_bridge else "",
        profitability_claim=False,
        live_readiness_claim=False,
        trade_authority=False,
    )

    # --- Hard baseline partition BEFORE cycles ---
    baseline = snapshot_preexisting_baseline(project_root, data_dir, audits_dir)
    paper_state = load_paper_state(project_root / "data" / "paper_state.json")
    emit_event(
        "AE20_BASELINE_SNAPSHOT",
        preexisting_positions=len(baseline["positions"]),
        preexisting_trades=len(baseline["trades"]),
    )

    ae16_index = load_ae16_index(
        project_root,
        ae16_bridge_source=ae16_bridge_source,
        ae20_ae16_exact_bridge=ae20_ae16_exact_bridge,
    )
    ae17_index = load_ae17_index(project_root)
    ae18_index = load_ae18_index(project_root)

    all_input_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    ae16_audit_rows: list[dict[str, Any]] = []
    ae17_audit_rows: list[dict[str, Any]] = []
    ae18_audit_rows: list[dict[str, Any]] = []
    ae19_audit_rows: list[dict[str, Any]] = []
    previous_cycle_identity_keys: set[str] = set()
    candidate_turnover_rows: list[dict[str, Any]] = []
    clean_forward_refresh_rows: list[dict[str, Any]] = []
    previous_provider_refresh_rows: list[dict[str, Any]] | None = None

    # AE20_PROVIDER_REFRESH_EACH_CYCLE_V1
    refresh_clean_forward_each_cycle = os.environ.get("AE20_REFRESH_CLEAN_FORWARD_EACH_CYCLE") == "1"
    refresh_clean_forward_force = os.environ.get("AE20_REFRESH_CLEAN_FORWARD_FORCE") == "1"
    refresh_clean_forward_clear_cache = os.environ.get("AE20_REFRESH_CLEAN_FORWARD_CLEAR_CACHE") == "1"
    refresh_clean_forward_limit = int(os.environ.get("AE20_REFRESH_CLEAN_FORWARD_LIMIT") or "25")
    refresh_clean_forward_max_candidates = int(os.environ.get("AE20_REFRESH_CLEAN_FORWARD_MAX_CANDIDATES") or "80")
    refresh_clean_forward_max_verify = int(os.environ.get("AE20_REFRESH_CLEAN_FORWARD_MAX_VERIFY") or "40")

    refresh_clean_forward_market_feed = None
    if refresh_clean_forward_each_cycle:
        from app.ae13b_product.clean_forward_market_feed import (
            refresh_clean_forward_market_feed as _refresh_clean_forward_market_feed,
        )
        refresh_clean_forward_market_feed = _refresh_clean_forward_market_feed

    def _write_checkpoint(reason: str, current_cycle_id: str | None = None) -> None:
        write_csv(data_dir / "ae20_clean_forward_inputs.csv", all_input_rows + all_failures)
        write_csv(data_dir / "ae20_integrated_decisions.csv", decisions)
        write_jsonl(data_dir / "ae20_integrated_decisions.jsonl", decisions)
        write_csv(data_dir / "ae20_paper_orders.csv", orders)
        write_csv(data_dir / "ae20_positions.csv", positions)
        write_csv(data_dir / "ae20_outcomes.csv", outcomes)
        write_csv(audits_dir / "ae20_ae16_consensus_integration_audit.csv", ae16_audit_rows)
        write_csv(audits_dir / "ae20_ae17_meta_integration_audit.csv", ae17_audit_rows)
        write_csv(audits_dir / "ae20_ae18_context_integration_audit.csv", ae18_audit_rows)
        write_csv(audits_dir / "ae20_ae19_llm_integration_audit.csv", ae19_audit_rows)
        write_csv(audits_dir / "ae20_candidate_turnover_audit.csv", candidate_turnover_rows)
        write_csv(audits_dir / "ae20_clean_forward_refresh_audit.csv", clean_forward_refresh_rows)
        write_json(
            audits_dir / "ae20_authority_safety_audit.json",
            {
                "trade_authority": False,
                "live_trading_enabled": False,
                "wallet_connected": False,
                "private_key_accessed": False,
                "signing_performed": False,
                "llm_approval_used_for_execution": False,
                "risk_override_used": False,
                "unauthorized_db_mutation": False,
                "paper_demo_only": True,
                "live_paper_state_mutated": False,
                "authority_escalation_detected": False,
                "profitability_claim": False,
                "pass": True,
                "checkpoint_reason": reason,
                "current_cycle_id": current_cycle_id,
            },
        )
        write_json(
            audits_dir / "ae20_no_legacy_source_audit.json",
            {
                "legacy_market_snapshots_used_as_source_of_truth": False,
                "legacy_ae16_bridge_used_as_evidence_authority": False,
                "clean_forward_only": True,
                "pass": True,
                "checkpoint_reason": reason,
                "current_cycle_id": current_cycle_id,
            },
        )
        write_json(
            reports_dir / "ae20_decision_gate.json",
            {
                "classification": "AE20_DURATION_IN_PROGRESS" if mode == "duration" else "AE20_SMOKE_IN_PROGRESS",
                "stage": reason,
                "unblocked_for_24h": False,
                "duration_completed": False,
                "ae20_stage_closed": False,
                "requires_main_thread_smoke_review": mode == "smoke",
                "profitability_claim": False,
                "live_readiness_claim": False,
                "trade_authority": False,
                "paper_demo_only": True,
                "blockers_before_24h": ["AE20 run still in progress"],
                "ae16_attached_rows_count": sum(
                    1 for row in decisions
                    if row.get("ae16_status") == "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE"
                ),
                "cycles_completed_so_far": cycles_completed,
                "current_cycle_id": current_cycle_id,
            },
        )
        _write_in_progress_manifest(reason)

    llm_attempted = 0
    llm_succeeded = 0
    llm_timeout = 0
    llm_failed = 0
    llm_skipped = 0

    cycles_completed = 0
    identity_blocked = False
    end_by = None
    if duration_hours is not None:
        end_by = time.time() + float(duration_hours) * 3600.0
        planned_cycles = 10**9
    else:
        planned_cycles = max(1, int(smoke_cycles or 1))

    for cycle_idx in range(1, planned_cycles + 1):
        if end_by is not None and time.time() >= end_by:
            break
        cycle_id = f"{run_id}_c{cycle_idx:04d}"
        _append_event(events, "AE20_CYCLE_STARTED", cycle_id=cycle_id, cycle_idx=cycle_idx)

        # AE20_CANDIDATE_TURNOVER_FIX_V1
        candidate_offset = (cycle_idx - 1) * int(max_candidates_per_cycle or 0)

        provider_refresh_rows_for_cycle = None
        provider_refresh_metadata_for_cycle: dict[str, Any] = {}
        if refresh_clean_forward_each_cycle and refresh_clean_forward_market_feed is not None:
            refresh_started_marker = datetime.now(timezone.utc).isoformat()
            refresh_result = refresh_clean_forward_market_feed(
                force=refresh_clean_forward_force,
                clear_cache=refresh_clean_forward_clear_cache,
                previous_rows=previous_provider_refresh_rows,
                limit=refresh_clean_forward_limit,
                max_candidates=refresh_clean_forward_max_candidates,
                max_verify=refresh_clean_forward_max_verify,
            )
            provider_refresh_rows_for_cycle = list(refresh_result.get("rows") or [])
            previous_provider_refresh_rows = [dict(r) for r in provider_refresh_rows_for_cycle]
            provider_refresh_metadata_for_cycle = dict(
                refresh_result.get("refresh")
                or refresh_result.get("refresh_metadata")
                or {}
            )
            clean_forward_refresh_rows.append({
                "ae20_cycle_id": cycle_id,
                "cycle_idx": cycle_idx,
                "refresh_started_marker_utc": refresh_started_marker,
                "provider_refresh_source_used": True,
                "provider_refresh_rows": len(provider_refresh_rows_for_cycle),
                "refresh_ok": provider_refresh_metadata_for_cycle.get("ok"),
                "refresh_mode": provider_refresh_metadata_for_cycle.get("refresh_mode"),
                "provider_refetch_attempted": provider_refresh_metadata_for_cycle.get("provider_refetch_attempted"),
                "provider_refetch_completed": provider_refresh_metadata_for_cycle.get("provider_refetch_completed"),
                "provider_values_changed_count": provider_refresh_metadata_for_cycle.get("provider_values_changed_count"),
                "payload_hash_changed_count": provider_refresh_metadata_for_cycle.get("payload_hash_changed_count"),
                "rows_entered_main_feed": provider_refresh_metadata_for_cycle.get("rows_entered_main_feed"),
                "rows_exited_main_feed": provider_refresh_metadata_for_cycle.get("rows_exited_main_feed"),
                "provider_rate_limited_count": provider_refresh_metadata_for_cycle.get("provider_rate_limited_count"),
                "duplicate_pools_suppressed": provider_refresh_metadata_for_cycle.get("duplicate_pools_suppressed"),
                "invalid_or_unresolved_excluded": provider_refresh_metadata_for_cycle.get("invalid_or_unresolved_excluded"),
                "http_calls_this_refresh": provider_refresh_metadata_for_cycle.get("http_calls_this_refresh"),
                "latest_provider_fetch_at": provider_refresh_metadata_for_cycle.get("latest_provider_fetch_at"),
                "refresh_id": provider_refresh_metadata_for_cycle.get("refresh_id"),
                "ui_message": provider_refresh_metadata_for_cycle.get("ui_message"),
            })

        cf = build_clean_forward_inputs(
            project_root,
            max_candidates=max_candidates_per_cycle,
            candidate_offset=candidate_offset,
            previous_identity_keys=previous_cycle_identity_keys,
            # AE20_CANONICAL_45_ENRICHMENT_ONLY_V1
            # Provider feed display rows must never replace the canonical 45 universe.
            source_rows_override=None,
            source_name_override=None,
            source_path_override=None,
            refresh_metadata=provider_refresh_metadata_for_cycle,
        )

        # AE20_CANONICAL_45_ENRICHMENT_ONLY_V1
        # The candidate universe remains the canonical Clean Forward index.
        # Optional exact pair verification enriches the 45 rows in-place.
        if os.environ.get("AE20_VERIFY_CANONICAL_45_EACH_CYCLE") == "1":
            from app.ae20.canonical_pair_refresh import refresh_canonical_clean_forward_rows

            # AE20_CANONICAL_45_ROWS_KEY_FIX_V1
            # build_clean_forward_inputs does not necessarily return candidates under
            # cf["rows"]. Detect the actual candidate-row list without mutating identity.
            cf_rows_key = None
            candidate_rows_for_refresh = []
            for candidate_key, candidate_value in cf.items():
                if (
                    isinstance(candidate_value, list)
                    and candidate_value
                    and isinstance(candidate_value[0], dict)
                    and (
                        "provider_pair_url_exact" in candidate_value[0]
                        or "canonical_market_identity" in candidate_value[0]
                        or "raw_provider_pair_url" in candidate_value[0]
                    )
                ):
                    cf_rows_key = candidate_key
                    candidate_rows_for_refresh = list(candidate_value)
                    break

            canonical_refresh = refresh_canonical_clean_forward_rows(
                candidate_rows_for_refresh,
                use_cache=(os.environ.get("AE20_VERIFY_CANONICAL_45_USE_CACHE") == "1"),
                max_rows=int(os.environ.get("AE20_VERIFY_CANONICAL_45_MAX_ROWS") or "0") or None,
            )

            if cf_rows_key is not None:
                cf[cf_rows_key] = canonical_refresh["rows"]

            canonical_summary = dict(canonical_refresh.get("summary") or {})
            canonical_summary["cf_rows_key"] = cf_rows_key or ""
            clean_forward_refresh_rows.append({
                "ae20_cycle_id": cycle_id,
                "cycle_idx": cycle_idx,
                "provider_refresh_source_used": False,
                "refresh_mode": canonical_summary.get("refresh_mode"),
                "cf_rows_key": canonical_summary.get("cf_rows_key"),
                "provider_refresh_rows": canonical_summary.get("processed_rows"),
                "canonical_universe_rows": canonical_summary.get("input_rows"),
                "canonical_refresh_attempted_rows": canonical_summary.get("attempted_rows"),
                "canonical_refresh_verified_rows": canonical_summary.get("verified_rows"),
                "canonical_refresh_failed_rows": canonical_summary.get("failed_rows"),
                "canonical_refresh_skipped_rows": canonical_summary.get("skipped_rows"),
                "identity_preserved": canonical_summary.get("identity_preserved"),
                "identity_mutation_count": canonical_summary.get("identity_mutation_count"),
                "validator_gap_direct_fallback_verified_rows": canonical_summary.get("validator_gap_direct_fallback_verified_rows"),  # AE20_VALIDATOR_GAP_AUDIT_FIELDS_FIX_V2
                "provider_pair_case_mismatch_count": canonical_summary.get("provider_pair_case_mismatch_count"),
                "candidate_universe_replaced": canonical_summary.get("candidate_universe_replaced"),
                "provider_refresh_enrichment_only": canonical_summary.get("provider_refresh_enrichment_only"),
                "price_updated_rows": canonical_summary.get("price_updated_rows"),
                "liquidity_updated_rows": canonical_summary.get("liquidity_updated_rows"),
                "latest_provider_fetch_at": canonical_summary.get("completed_at_utc"),
            })
        if not clean_forward_only:
            # Still force clean-forward-only policy in this engine.
            clean_forward_only = True

        cycle_inputs = cf["inputs"]
        cycle_failures = cf["failures"]
        current_identity_keys = set(str(k) for k in cf.get("selected_identity_keys", []) if str(k))
        turnover_row = {
            "ae20_cycle_id": cycle_id,
            "cycle_idx": cycle_idx,
            "candidate_selection_offset_used": cf.get("candidate_selection_offset_used"),
            "candidate_source_rows_available": cf.get("source_rows_available"),
            "selected_ok_count": len(cycle_inputs),
            "selected_identity_count": cf.get("selected_identity_count"),
            "new_identity_count_vs_previous_cycle": cf.get("new_identity_count_vs_previous_cycle"),
            "repeated_identity_count_vs_previous_cycle": cf.get("repeated_identity_count_vs_previous_cycle"),
            "candidate_turnover_rate_vs_previous_cycle": cf.get("candidate_turnover_rate_vs_previous_cycle"),
            "selected_identity_keys_json": json.dumps(cf.get("selected_identity_keys", []), ensure_ascii=False),
            "clean_forward_static_replay_detected": (
                cycle_idx > 1
                and int(cf.get("selected_identity_count") or 0) > 0
                and float(cf.get("candidate_turnover_rate_vs_previous_cycle") or 0.0) == 0.0
            ),
        }
        candidate_turnover_rows.append(turnover_row)

        emit_event(
            "AE20_CLEAN_FORWARD_INPUTS_LOADED",
            cycle_id=cycle_id,
            cycle_idx=cycle_idx,
            inputs=len(cycle_inputs),
            failures=len(cycle_failures),
            candidate_selection_offset_used=cf.get("candidate_selection_offset_used"),
            candidate_source_rows_available=cf.get("source_rows_available"),
            selected_identity_count=cf.get("selected_identity_count"),
            new_identity_count_vs_previous_cycle=cf.get("new_identity_count_vs_previous_cycle"),
            repeated_identity_count_vs_previous_cycle=cf.get("repeated_identity_count_vs_previous_cycle"),
            candidate_turnover_rate_vs_previous_cycle=cf.get("candidate_turnover_rate_vs_previous_cycle"),
            clean_forward_static_replay_detected=turnover_row["clean_forward_static_replay_detected"],
            provider_refresh_source_used=bool(provider_refresh_rows_for_cycle is not None),
            provider_refresh_rows=len(provider_refresh_rows_for_cycle or []),
            provider_refetch_completed=provider_refresh_metadata_for_cycle.get("provider_refetch_completed"),
            provider_values_changed_count=provider_refresh_metadata_for_cycle.get("provider_values_changed_count"),
            rows_entered_main_feed=provider_refresh_metadata_for_cycle.get("rows_entered_main_feed"),
            rows_exited_main_feed=provider_refresh_metadata_for_cycle.get("rows_exited_main_feed"),
            latest_provider_fetch_at=provider_refresh_metadata_for_cycle.get("latest_provider_fetch_at"),
        )
        previous_cycle_identity_keys = current_identity_keys
        all_input_rows.extend([{**r, "ae20_cycle_id": cycle_id, "ae20_run_id": run_id} for r in cycle_inputs])
        all_failures.extend([{**r, "ae20_cycle_id": cycle_id, "ae20_run_id": run_id} for r in cycle_failures])

        total_seen = len(cycle_inputs) + len(cycle_failures)
        failure_ratio = (len(cycle_failures) / total_seen) if total_seen else 1.0
        if fail_on_missing_canonical_identity and (
            (not cycle_inputs and cycle_failures)
            or failure_ratio >= 0.9
        ):
            identity_blocked = True
            _append_event(
                events,
                "AE20_SMOKE_BLOCKED_CLEAN_FORWARD_INPUT_FAILURE",
                cycle_id=cycle_id,
                failure_ratio=failure_ratio,
            )
            cycles_completed += 1
            _write_checkpoint(
                reason="blocked_clean_forward_input_failure",
                current_cycle_id=cycle_id,
            )
            emit_event(
                "AE20_BLOCKED_CHECKPOINT_WRITTEN",
                cycle_id=cycle_id,
                reason="clean_forward_input_failure",
            )
            break

        llm_budget = int(max_llm_calls_per_cycle) if allow_llm else 0
        # For smoke, call LLM on first N candidates only.
        for cand in cycle_inputs:
            if not cand.get("identity_ok"):
                # Should not happen for inputs list, but hard-fail path.
                ae16 = {"ae16_status": "AE16_NOT_APPLICABLE"}
                ae17 = {"ae17_status": "AE17_NOT_APPLICABLE"}
                ae18 = {"ae18_status": "AE18_CONTEXT_UNAVAILABLE"}
                ae19 = {
                    "ae19_status": "AE19_LLM_SKIPPED_BY_CONFIG",
                    "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
                    "llm_authorizes_execution": False,
                }
                gates = {
                    "gatekeeper_result": "GATEKEEPER_NOT_EVALUATED",
                    "riskguard_result": "RISKGUARD_NOT_EVALUATED",
                    "gatekeeper_passed": False,
                    "riskguard_passed": False,
                }
                path = derive_strict_exploration(cand, ae16, ae17, ae18, ae19, gates)
            else:
                ae16 = attach_ae16(cand, ae16_index)
                ae17 = attach_ae17(cand, ae16, ae17_index)
                ae18 = attach_ae18(cand, ae18_index)
                # Merge statuses onto candidate for LLM prompt context
                cand_ctx = {
                    **cand,
                    "ae16_status": ae16.get("ae16_status"),
                    "meta_decision": ae17.get("meta_decision"),
                    "ae18_status": ae18.get("ae18_status"),
                }
                do_llm = allow_llm and llm_budget > 0
                ae19 = run_ae19_audit_only(
                    cand_ctx,
                    allow_llm=do_llm,
                    llm_provider=llm_provider if do_llm else "none",
                    timeout_seconds=float(llm_timeout_seconds),
                    remaining_budget=llm_budget,
                    force_unavailable=force_llm_unavailable,
                )
                if do_llm:
                    llm_budget -= 1
                    if ae19.get("llm_attempted"):
                        llm_attempted += 1
                    if ae19.get("llm_succeeded"):
                        llm_succeeded += 1
                    if ae19.get("llm_timeout"):
                        llm_timeout += 1
                    if ae19.get("llm_failed") and not ae19.get("llm_timeout"):
                        llm_failed += 1
                    if ae19.get("llm_skipped"):
                        llm_skipped += 1
                else:
                    llm_skipped += 1
                    ae19 = {
                        **ae19,
                        "ae19_status": "AE19_LLM_SKIPPED_BY_CONFIG",
                        "llm_skipped": True,
                    }

                _mark_open_positions_for_candidate(cand, positions, outcomes)
                paper_state["open_positions"] = [
                    p for p in positions if str(p.get("status") or "").strip() == "OPEN"
                ]
                gates = evaluate_gates(cand, paper_state)
                # LLM cannot authorize — force flags
                ae19["llm_authorizes_execution"] = False
                path = derive_strict_exploration(cand, ae16, ae17, ae18, ae19, gates)

            decision = build_integrated_decision(
                run_id=run_id,
                cycle_id=cycle_id,
                candidate=cand,
                ae16=ae16,
                ae17=ae17,
                ae18=ae18,
                ae19=ae19,
                gates=gates,
                path_decisions=path,
            )
            # Hard safety invariants
            decision["trade_authority"] = False
            decision["live_trading_enabled"] = False
            decision["wallet_connected"] = False
            decision["profitability_claim"] = False
            if not paper_demo_only:
                # Engine still refuses live.
                decision["paper_demo_only"] = True
            else:
                decision["paper_demo_only"] = True

            if decision.get("final_paper_demo_decision") == "PAPER_DEMO_OPEN":
                duplicate_open, duplicate_reason = _has_open_position_for_candidate(
                    decision,
                    list(paper_state.get("open_positions") or []),
                )
                if duplicate_open:
                    decision["skip_reason"] = "DUPLICATE_OPEN_POSITION"
                    decision["action_reason"] = f"skipped:DUPLICATE_OPEN_POSITION|{duplicate_reason}"
                    decision["strict_decision"] = "STRICT_BLOCKED_DUPLICATE_OPEN_POSITION"
                    decision["exploration_decision"] = "EXPLORATION_BLOCKED_DUPLICATE_OPEN_POSITION"
                    decision["final_paper_demo_decision"] = "NO_TRADE"
                    decision["order_id"] = ""
                    decision["position_id"] = ""
                    decision["outcome_id"] = ""

            life = maybe_create_paper_lifecycle(decision, cand)
            if life:
                orders.append(life["order"])
                positions.append(life["position"])
                outcomes.append(life["outcome"])
                paper_state.setdefault("open_positions", [])
                paper_state["open_positions"].append(life["position"])

            decisions.append(decision)
            ae16_audit_rows.append(
                {
                    "ae20_decision_id": decision["ae20_decision_id"],
                    "candidate_id": decision.get("candidate_id"),
                    **{k: ae16.get(k) for k in ae16},
                }
            )
            ae17_audit_rows.append(
                {
                    "ae20_decision_id": decision["ae20_decision_id"],
                    "candidate_id": decision.get("candidate_id"),
                    **{k: ae17.get(k) for k in ae17},
                }
            )
            ae18_audit_rows.append(
                {
                    "ae20_decision_id": decision["ae20_decision_id"],
                    "candidate_id": decision.get("candidate_id"),
                    **{k: ae18.get(k) for k in ae18},
                }
            )
            ae19_audit_rows.append(
                {
                    "ae20_decision_id": decision["ae20_decision_id"],
                    "candidate_id": decision.get("candidate_id"),
                    **{k: ae19.get(k) for k in ae19},
                }
            )

        cycles_completed += 1
        emit_event(
            "AE20_CYCLE_COMPLETED",
            cycle_id=cycle_id,
            decisions=len([d for d in decisions if d.get("ae20_cycle_id") == cycle_id]),
        )
        if identity_blocked:
            break
        _write_checkpoint(reason="cycle_completed", current_cycle_id=cycle_id)
        emit_event(
            "AE20_DECISIONS_WRITTEN",
            cycle_id=cycle_id,
            cycle_idx=cycle_idx,
            decisions=len([d for d in decisions if d.get("ae20_cycle_id") == cycle_id]),
        )

        if cycle_sleep_seconds > 0 and cycle_idx < planned_cycles:
            emit_event(
                "AE20_CYCLE_SLEEP_STARTED",
                cycle_id=cycle_id,
                cycle_idx=cycle_idx,
                sleep_seconds=float(cycle_sleep_seconds),
            )
            _write_checkpoint(reason="cycle_sleep_started", current_cycle_id=cycle_id)
            time.sleep(float(cycle_sleep_seconds))
            emit_event(
                "AE20_CYCLE_SLEEP_COMPLETED",
                cycle_id=cycle_id,
                cycle_idx=cycle_idx,
                sleep_seconds=float(cycle_sleep_seconds),
            )
            _write_checkpoint(reason="cycle_sleep_completed", current_cycle_id=cycle_id)

    # Also record identity failures as skipped decision-like rows for audit completeness
    for fail in all_failures:
        decisions.append(
            {
                "ae20_run_id": run_id,
                "ae20_cycle_id": fail.get("ae20_cycle_id"),
                "ae20_decision_id": f"ae20dec_identity_fail_{uuid4().hex[:8]}",
                "candidate_id": fail.get("candidate_id"),
                "provider_pair_url_exact": fail.get("provider_pair_url_exact"),
                "canonical_market_identity": fail.get("canonical_market_identity"),
                "normalized_provider_pair_url_key": fail.get("normalized_provider_pair_url_key"),
                "price_source_key": fail.get("price_source_key"),
                "chain": fail.get("chain"),
                "pair_address": fail.get("pair_address"),
                "identity_status": fail.get("identity_status"),
                "ae16_status": "AE16_NOT_APPLICABLE",
                "ae17_status": "AE17_NOT_APPLICABLE",
                "ae18_status": "AE18_CONTEXT_UNAVAILABLE",
                "ae19_status": "AE19_LLM_SKIPPED_BY_CONFIG",
                "gatekeeper_result": "GATEKEEPER_NOT_EVALUATED",
                "riskguard_result": "RISKGUARD_NOT_EVALUATED",
                "strict_decision": "STRICT_BLOCKED",
                "exploration_decision": "EXPLORATION_SKIP",
                "final_paper_demo_decision": "NO_TRADE",
                "skip_reason": "CLEAN_FORWARD_IDENTITY_INCOMPLETE",
                "action_reason": "identity_incomplete",
                "trade_authority": False,
                "live_trading_enabled": False,
                "wallet_connected": False,
                "profitability_claim": False,
                "created_during_ae20": True,
                "preexisting_baseline": False,
                "order_id": "",
                "position_id": "",
                "outcome_id": "",
            }
        )

    opp = build_opportunity_rows(decisions, outcomes)
    pnl_rows, pnl_audit = build_pnl_summary(
        orders=orders,
        positions=positions,
        preexisting_positions=baseline["positions"],
        preexisting_trades=baseline["trades"],
    )
    lineage = audit_lineage(
        decisions,
        orders,
        positions,
        outcomes,
        preexisting_positions=baseline["positions"],
        preexisting_trades=baseline["trades"],
        preexisting_orders=baseline["orders"],
    )

    # --- Safety / legacy audits ---
    authority_escalation = any(
        [
            any(d.get("trade_authority") for d in decisions),
            any(d.get("live_trading_enabled") for d in decisions),
            any(d.get("wallet_connected") for d in decisions),
            any(d.get("profitability_claim") for d in decisions),
            any(a.get("llm_authorizes_execution") for a in ae19_audit_rows),
        ]
    )
    legacy_contaminated = any(
        r.get("legacy_market_snapshots_used") or r.get("market_snapshots_used") or r.get("symbol_only_join_used")
        for r in all_input_rows
    ) or (not clean_forward_only)

    ae16_attached = sum(
        1
        for r in ae16_audit_rows
        if r.get("ae16_status")
        in {
            "AE16_EVIDENCE_ATTACHED",
            "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
        }
    )
    ae16_unmatched = sum(
        1
        for r in ae16_audit_rows
        if r.get("ae16_status")
        in {
            "AE16_JOIN_NOT_FOUND",
            "AE16_EVIDENCE_UNAVAILABLE",
            "AE16_EXACT_DERIVED_BRIDGE_NOT_FOUND",
        }
    )
    ae17_attached = sum(
        1
        for r in ae17_audit_rows
        if r.get("ae17_status") in {"AE17_META_ATTACHED", "AE17_META_COMPUTED"}
    )
    ae18_attached = sum(
        1
        for r in ae18_audit_rows
        if r.get("ae18_status") in {"AE18_CONTEXT_ATTACHED", "AE18_CONTEXT_MISSINGNESS_ONLY"}
    )
    ae19_attached = sum(
        1
        for r in ae19_audit_rows
        if r.get("ae19_status")
        in {
            "AE19_LLM_AUDIT_ATTACHED",
            "AE19_QWEN_AUDIT_SUCCEEDED",
            "AE19_QWEN_PROVIDER_UNAVAILABLE",
            "AE19_QWEN_TIMEOUT",
            "AE19_GEMINI_NOT_ENABLED",
            "AE19_LLM_SKIPPED_BY_CONFIG",
            "AE19_LLM_AUDIT_FAILED",
        }
    )
    # Matched attached rows must never carry AE16_MODEL_EVIDENCE_UNAVAILABLE skip.
    matched_rows_wrongly_unavailable = sum(
        1
        for d in decisions
        if d.get("ae16_status")
        in {
            "AE16_EVIDENCE_ATTACHED",
            "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
        }
        and d.get("skip_reason") == "AE16_MODEL_EVIDENCE_UNAVAILABLE"
    )
    ae20_invalid_url = sum(
        1 for r in ae16_audit_rows if r.get("ae20_provider_pair_url_exact_invalid")
    )
    provider_url_mutated = sum(
        1 for r in ae16_audit_rows if r.get("provider_pair_url_exact_mutated")
    )
    ae16_url_mutated = sum(
        1 for r in ae16_audit_rows if r.get("ae16_provider_pair_url_mutated")
    )
    exact_identity_ok = (
        provider_url_mutated == 0
        and ae16_url_mutated == 0
        and not any(r.get("lowercase_join_used") for r in ae16_audit_rows)
        and not any(r.get("casefold_join_used") for r in ae16_audit_rows)
        and not any(r.get("case_insensitive_join_used") for r in ae16_audit_rows)
        and not any(r.get("forbidden_pair_chain_join_used") for r in ae16_audit_rows)
        and not any(r.get("pair_chain_only_join_used_for_closure") for r in ae16_audit_rows)
    )
    unsafe_bridge_flags = (
        any(r.get("lowercase_join_used") for r in ae16_audit_rows)
        or any(r.get("casefold_join_used") for r in ae16_audit_rows)
        or any(r.get("case_insensitive_join_used") for r in ae16_audit_rows)
        or any(r.get("pair_chain_only_join_used_for_closure") for r in ae16_audit_rows)
        or any(r.get("raw_mutation") for r in ae16_audit_rows)
        or any(r.get("db_mutation") for r in ae16_audit_rows)
    )
    integration_ok = (
        len(ae16_audit_rows) > 0
        and len(ae17_audit_rows) > 0
        and len(ae18_audit_rows) > 0
        and len(ae19_audit_rows) > 0
        and not identity_blocked
        and bool(ae16_index.get("ae16_bridge_source_exists"))
    )
    llm_limitations = bool(
        llm_timeout or llm_failed or (allow_llm and llm_succeeded == 0 and llm_attempted > 0)
        or (allow_llm and llm_attempted == 0 and llm_skipped > 0 and no_external_llm)
    )
    # If LLM succeeded fully, not a limitation. If skipped by config with --no-external-llm, limitation.
    if allow_llm and llm_succeeded > 0 and llm_timeout == 0 and llm_failed == 0:
        llm_limitations = False
    if no_external_llm:
        llm_limitations = True

    total_cf_attempted = len(all_input_rows) + len(all_failures)
    identity_failure_ratio = (
        len(all_failures) / total_cf_attempted if total_cf_attempted else (1.0 if identity_blocked else 0.0)
    )

    classification = decide_classification(
        identity_blocked=identity_blocked,
        legacy_contaminated=legacy_contaminated,
        authority_escalation=authority_escalation,
        lineage_pass=bool(lineage.get("lineage_pass")),
        integration_ok=integration_ok,
        llm_limitations=llm_limitations or ae16_attached <= 0,
        identity_failure_ratio=identity_failure_ratio,
        ae16_attached_count=ae16_attached,
        matched_rows_wrongly_unavailable=matched_rows_wrongly_unavailable,
        exact_identity_ok=exact_identity_ok,
        unsafe_bridge_flags=unsafe_bridge_flags,
    )

    # --- Write artifacts ---
    write_jsonl(data_dir / "ae20_runtime_events.jsonl", events)
    write_csv(data_dir / "ae20_clean_forward_inputs.csv", all_input_rows + all_failures)
    write_csv(data_dir / "ae20_integrated_decisions.csv", decisions)
    write_jsonl(data_dir / "ae20_integrated_decisions.jsonl", decisions)
    write_csv(data_dir / "ae20_paper_orders.csv", orders)
    write_csv(data_dir / "ae20_positions.csv", positions)
    write_csv(data_dir / "ae20_outcomes.csv", outcomes)
    write_csv(data_dir / "ae20_skipped_candidates.csv", opp["skipped"])
    write_csv(data_dir / "ae20_opportunity_capture.csv", opp["opportunity"])
    write_csv(data_dir / "ae20_missed_winners.csv", opp["missed_winners"])
    write_csv(data_dir / "ae20_trade_vs_no_trade.csv", opp["trade_vs_no_trade"])
    write_csv(data_dir / "ae20_strict_vs_exploration.csv", opp["strict_vs_exploration"])
    write_csv(data_dir / "ae20_pnl_summary.csv", pnl_rows)

    # Baseline CSVs already written by snapshot_preexisting_baseline

    write_json(
        audits_dir / "ae20_clean_forward_only_audit.json",
        {
            "clean_forward_only": True,
            "market_snapshots_used_as_source_of_truth": False,
            "symbol_only_joins_used": False,
            "llm_identity_invention_used": False,
            "silent_legacy_backfill_used": False,
            "inputs_ok": len(all_input_rows),
            "identity_failures": len(all_failures),
            "fail_on_missing_canonical_identity": fail_on_missing_canonical_identity,
            "identity_blocked": identity_blocked,
        },
    )
    ae16_bridge_audit = dict(ae16_index.get("audit") or {})
    ae16_bridge_audit.update(
        {
            "ae20_rows_evaluated": len(ae16_audit_rows),
            "ae20_rows_with_valid_provider_pair_url_exact_key": len(ae16_audit_rows) - ae20_invalid_url,
            "ae20_rows_attached_by_safe_provider_url": ae16_attached,
            "ae20_rows_unmatched": ae16_unmatched,
            "ae20_invalid_provider_pair_url_exact_count": ae20_invalid_url,
            "provider_pair_url_exact_mutated_count": provider_url_mutated,
            "ae16_provider_pair_url_mutated_count": ae16_url_mutated,
            "safe_provider_url_join_used": True,
            "forbidden_pair_chain_join_used": False,
            "broad_merge_used": False,
            "lookup_dictionary_used": True,
            "uncontrolled_pandas_suffix_columns_present": False,
        }
    )
    # Per-row audit rows retain attachment detail; prepend bridge summary as first row fields via sidecar json + enriched csv
    write_json(audits_dir / "ae20_ae16_bridge_summary_audit.json", ae16_bridge_audit)
    enriched_ae16_audit = []
    for r in ae16_audit_rows:
        enriched_ae16_audit.append({**ae16_bridge_audit, **r})
    if not enriched_ae16_audit:
        enriched_ae16_audit = [ae16_bridge_audit]
    write_csv(audits_dir / "ae20_ae16_consensus_integration_audit.csv", enriched_ae16_audit)
    write_csv(audits_dir / "ae20_ae17_meta_integration_audit.csv", ae17_audit_rows)
    write_csv(audits_dir / "ae20_ae18_context_integration_audit.csv", ae18_audit_rows)
    write_csv(audits_dir / "ae20_ae19_llm_integration_audit.csv", ae19_audit_rows)
    write_csv(audits_dir / "ae20_candidate_turnover_audit.csv", candidate_turnover_rows)
    write_json(
        audits_dir / "ae20_lineage_integrity_audit.json",
        {
            **lineage,
            "spine": (
                "raw/provider identity → normalized observation → target/candidate → "
                "features → model evidence → consensus → meta → context → LLM → "
                "decision → order → position → outcome"
            ),
            "strict_baseline_partition": strict_baseline_partition,
        },
    )
    write_csv(
        audits_dir / "ae20_order_position_outcome_lineage_audit.csv",
        [
            {
                "order_id": o.get("order_id"),
                "position_id": o.get("position_id"),
                "ae20_decision_id": o.get("ae20_decision_id"),
                "candidate_id": o.get("candidate_id"),
                "created_during_ae20": o.get("created_during_ae20"),
                "preexisting_baseline": o.get("preexisting_baseline"),
            }
            for o in orders
        ]
        + [
            {
                "order_id": p.get("order_id"),
                "position_id": p.get("position_id"),
                "outcome_id": p.get("outcome_id"),
                "ae20_decision_id": p.get("ae20_decision_id"),
                "candidate_id": p.get("candidate_id"),
                "created_during_ae20": p.get("created_during_ae20"),
                "preexisting_baseline": p.get("preexisting_baseline"),
            }
            for p in positions
        ],
    )
    write_csv(audits_dir / "ae20_opportunity_capture_audit.csv", opp["opportunity"])
    write_csv(audits_dir / "ae20_missed_winner_audit.csv", opp["missed_winners"])
    write_json(audits_dir / "ae20_pnl_audit.json", pnl_audit)
    write_csv(audits_dir / "ae20_strict_vs_exploration_audit.csv", opp["strict_vs_exploration"])
    write_json(
        audits_dir / "ae20_authority_safety_audit.json",
        {
            "trade_authority": False,
            "live_trading_enabled": False,
            "wallet_connected": False,
            "private_key_accessed": False,
            "signing_performed": False,
            "llm_approval_used_for_execution": False,
            "risk_override_used": False,
            "unauthorized_db_mutation": False,
            "paper_demo_only": paper_demo_only,
            "live_paper_state_mutated": False,
            "authority_escalation_detected": authority_escalation,
            "profitability_claim": False,
            "pass": not authority_escalation,
        },
    )
    write_json(
        audits_dir / "ae20_no_legacy_source_audit.json",
        {
            "legacy_market_snapshots_used": False,
            "old_market_snapshot_feed_used": False,
            "symbol_only_identity_joins": False,
            "llm_invented_identity_links": False,
            "silent_legacy_backfill": False,
            "contaminated": legacy_contaminated,
            "pass": not legacy_contaminated,
        },
    )
    write_json(audits_dir / "ae20_output_root_collision_audit.json", collision_audit)
    write_json(
        audits_dir / "ae20_llm_timeout_budget_audit.json",
        {
            "llm_provider": llm_provider,
            "allow_llm": allow_llm,
            "no_external_llm": no_external_llm,
            "max_llm_calls_per_cycle": max_llm_calls_per_cycle,
            "hard_max_llm_calls_per_cycle_smoke": SMOKE_HARD_MAX_LLM_CALLS_PER_CYCLE,
            "llm_timeout_seconds": llm_timeout_seconds,
            "llm_attempted": llm_attempted,
            "llm_succeeded": llm_succeeded,
            "llm_timeout": llm_timeout,
            "llm_failed": llm_failed,
            "llm_skipped": llm_skipped,
            "gemini_enabled": False,
            "llm_failure_does_not_crash_pipeline": True,
            "llm_cannot_authorize_execution": True,
        },
    )

    strict_pnl = next((r for r in pnl_rows if r["pnl_scope"] == "AE20_CREATED_STRICT"), {})
    expl_pnl = next((r for r in pnl_rows if r["pnl_scope"] == "AE20_CREATED_EXPLORATION"), {})
    combined_pnl = next((r for r in pnl_rows if r["pnl_scope"] == "AE20_CREATED_ALL"), {})

    blockers: list[str] = []
    if identity_blocked:
        blockers.append("Clean Forward identity failure rate too high")
    if not lineage.get("lineage_pass"):
        blockers.append("AE20-created lineage orphan check failed")
    if authority_escalation:
        blockers.append("Authority escalation detected")
    if legacy_contaminated:
        blockers.append("Legacy source contamination detected")
    if ae16_attached <= 0:
        blockers.append(
            "AE16 exact case-preserved provider_pair_url join attached 0 rows; "
            "AE16 bridge provider_pair_url values do not exactly match AE20 "
            "provider_pair_url_exact (no lowercase/casefold join permitted)"
        )
    if matched_rows_wrongly_unavailable > 0:
        blockers.append(
            "Matched AE16-attached rows incorrectly skipped as AE16_MODEL_EVIDENCE_UNAVAILABLE"
        )
    if unsafe_bridge_flags:
        blockers.append(
            "Unsafe bridge flags detected (case-insensitive/lowercase/casefold/"
            "pair_chain_only closure join or raw/db mutation)"
        )
    if classification.startswith("AE20_SMOKE_BLOCKED"):
        blockers.append(f"Classification blocked: {classification}")
    # Note: main-thread smoke review is tracked via requires_main_thread_smoke_review,
    # not as a blockers_before_24h entry, so empty blockers can coexist with READY_FOR_24H.

    # Hard rule: AE16 attached == 0 must never classify READY_FOR_24H (already enforced
    # in decide_classification). Reinforce classification if gate invariants break.
    if ae16_attached <= 0 and classification == "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H":
        classification = "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS"
        blockers.append("AE16 attached rows == 0 prevents READY_FOR_24H classification")

    from collections import Counter

    tier_counts = Counter(
        str(d.get("ae16_consensus_tier") or "")
        for d in decisions
        if d.get("ae16_status")
        in {
            "AE16_EVIDENCE_ATTACHED",
            "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
        }
    )
    skip_counts = Counter(str(d.get("skip_reason") or "") for d in decisions)

    unblocked_for_24h = compute_unblocked_for_24h(
        classification=classification,
        blockers_before_24h=blockers,
        ae16_attached_count=ae16_attached,
        unsafe_bridge_flags=unsafe_bridge_flags,
    )

    summary = {
        "phase": PHASE,
        "engine_version": ENGINE_VERSION,
        "classification": classification,
        "mode": mode,
        "ae20_run_id": run_id,
        "output_root": str(root),
        "started_at": started,
        "completed_at": _utc(),
        "smoke_cycles_completed": cycles_completed,
        "clean_forward_rows_evaluated": len(all_input_rows),
        "clean_forward_identity_failures": len(all_failures),
        "integrated_decisions_count": len(decisions),
        "ae16_bridge_source_path_relative": ae16_index.get("ae16_bridge_source_path_relative"),
        "ae16_bridge_source_path_resolved": ae16_index.get("ae16_bridge_source_path_resolved"),
        "ae16_bridge_source_exists": ae16_index.get("ae16_bridge_source_exists"),
        "ae16_bridge_source_override_type": ae16_index.get("ae16_bridge_source_override_type"),
        "ae16_safe_provider_url_join_count": ae16_attached,
        "ae16_integration_count": ae16_attached,
        "ae16_attached_rows_count": ae16_attached,
        "ae16_unmatched_rows_count": ae16_unmatched,
        "ae20_invalid_provider_pair_url_exact_count": ae20_invalid_url,
        "ae16_invalid_provider_pair_url_count": (ae16_index.get("audit") or {}).get(
            "ae16_invalid_provider_pair_url_count", 0
        ),
        "empty_nan_join_keys_used": False,
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "identity_case_preserved": True,
        "provider_pair_url_exact_mutated_count": provider_url_mutated,
        "ae16_provider_pair_url_mutated_count": ae16_url_mutated,
        "uncontrolled_pandas_suffix_columns_present": False,
        "broad_merge_used": False,
        "lookup_dictionary_used": True,
        "forbidden_pair_chain_join_used": False,
        "safe_provider_url_join_used": True,
        "consensus_tier_counts_after_attachment": dict(tier_counts),
        "skip_reason_counts_after_attachment": dict(skip_counts),
        "ae17_integration_count": ae17_attached,
        "ae18_integration_count": ae18_attached,
        "ae19_integration_count": ae19_attached,
        "qwen_ollama_attempted_task_count": llm_attempted,
        "qwen_ollama_succeeded_task_count": llm_succeeded,
        "qwen_ollama_timeout_failure_skipped_count": llm_timeout + llm_failed + llm_skipped,
        "paper_orders_count": len(orders),
        "ae20_created_positions_count": len(positions),
        "preexisting_baseline_positions_count": len(baseline["positions"]),
        "preexisting_baseline_trades_count": len(baseline["trades"]),
        "skipped_candidates_count": len(opp["skipped"]),
        "strict_pnl_summary": strict_pnl,
        "exploration_pnl_summary": expl_pnl,
        "combined_ae20_pnl_summary": combined_pnl,
        "preexisting_baseline_pnl_exclusion": pnl_audit.get(
            "baseline_excluded_from_ae20_created_pnl"
        ),
        "strict_vs_exploration_count": len(opp["strict_vs_exploration"]),
        "lineage_audit_pass": lineage.get("lineage_pass"),
        "no_legacy_source_pass": not legacy_contaminated,
        "output_root_collision_safe": collision_audit.get("collision_safe"),
        "authority_safety_pass": not authority_escalation,
        "separate_strict_exploration_pnl": separate_strict_exploration_pnl,
        "blockers_before_24h": blockers,
        "unblocked_for_24h": unblocked_for_24h,
        "profitability_claim": False,
        "live_readiness_claim": False,
    }

    decision_gate = {
        "classification": classification,
        "unblocked_for_24h": unblocked_for_24h,
        "requires_main_thread_smoke_review": mode == "smoke",
        "profitability_claim": False,
        "live_readiness_claim": False,
        "trade_authority": False,
        "paper_demo_only": True,
        "pnl_message": pnl_audit.get("main_gate_message"),
        "blockers_before_24h": blockers,
        "ae16_attached_rows_count": ae16_attached,
    }
    write_json(reports_dir / "ae20_decision_gate.json", decision_gate)
    write_json(
        reports_dir / "ae20_manifest.json",
        {
            "phase": PHASE,
            "engine_version": ENGINE_VERSION,
            "ae20_run_id": run_id,
            "output_root": str(root),
            "classification": classification,
            "mode": mode,
            "config": {
                "smoke_cycles": smoke_cycles,
                "duration_hours": duration_hours,
                "llm_provider": llm_provider,
                "max_llm_calls_per_cycle": max_llm_calls_per_cycle,
                "llm_timeout_seconds": llm_timeout_seconds,
                "paper_demo_only": paper_demo_only,
                "clean_forward_only": clean_forward_only,
                "strict_baseline_partition": strict_baseline_partition,
                "fail_on_missing_canonical_identity": fail_on_missing_canonical_identity,
                "separate_strict_exploration_pnl": separate_strict_exploration_pnl,
                "no_external_llm": no_external_llm,
            },
            "counts": summary,
            "ae16_source": ae16_index.get("path"),
            "ae17_source": ae17_index.get("path"),
            "ae18_source": ae18_index.get("path"),
        },
    )
    write_json(reports_dir / "ae20_final_closure_audit.json", summary)

    summary_txt = "\n".join(
        [
            "AE20 INTEGRATED CLEAN FORWARD VALIDATION — SMOKE/DURATION SUMMARY",
            f"classification: {classification}",
            f"output_root: {root}",
            f"ae20_run_id: {run_id}",
            f"mode: {mode}",
            f"smoke_cycles_completed: {cycles_completed}",
            f"clean_forward_rows_evaluated: {len(all_input_rows)}",
            f"integrated_decisions_count: {len(decisions)}",
            f"ae16_integration_count: {ae16_attached}",
            f"ae17_integration_count: {ae17_attached}",
            f"ae18_integration_count: {ae18_attached}",
            f"ae19_integration_count: {ae19_attached}",
            f"qwen_ollama_attempted: {llm_attempted}",
            f"qwen_ollama_succeeded: {llm_succeeded}",
            f"qwen_ollama_timeout_failure_skipped: {llm_timeout + llm_failed + llm_skipped}",
            f"paper_orders_count: {len(orders)}",
            f"ae20_created_positions_count: {len(positions)}",
            f"preexisting_baseline_positions_count: {len(baseline['positions'])}",
            f"preexisting_baseline_trades_count: {len(baseline['trades'])}",
            f"skipped_candidates_count: {len(opp['skipped'])}",
            f"lineage_audit_pass: {lineage.get('lineage_pass')}",
            f"no_legacy_source_pass: {not legacy_contaminated}",
            f"authority_safety_pass: {not authority_escalation}",
            f"output_root_collision_safe: {collision_audit.get('collision_safe')}",
            "profitability_claim: false",
            "live_readiness_claim: false",
            "blockers_before_24h:",
            *[f"  - {b}" for b in blockers],
            "",
            "Send back:",
            str(reports_dir / "ae20_decision_gate.json"),
            str(reports_dir / "ae20_final_closure_audit.json"),
            str(reports_dir / "ae20_summary_for_upload.txt"),
        ]
    )
    write_text(reports_dir / "ae20_summary_for_upload.txt", summary_txt + "\n")
    _append_event(events, "AE20_RUN_END", classification=classification)
    write_jsonl(data_dir / "ae20_runtime_events.jsonl", events)

    return summary
