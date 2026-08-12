"""Integrated decision construction, GateKeeper/RiskGuard, strict vs exploration."""

from __future__ import annotations

from typing import Any
from uuid import uuid4


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def evaluate_gates(candidate: dict[str, Any], paper_state: dict[str, Any]) -> dict[str, Any]:
    """Run GateKeeper then RiskGuard. LLM labels never influence pass/fail."""
    gate_passed = False
    gate_blocker = ""
    gate_status = "GATEKEEPER_NOT_EVALUATED"
    risk_passed = False
    risk_blocker = ""
    risk_status = "RISKGUARD_NOT_EVALUATED"

    try:
        from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        # AE20_GATEKEEPER_BRIDGE_FIX_V1
        # Build a GateKeeper-compatible Clean Forward row. Do not invent market values:
        # map already-observed AE20/canonical refresh fields into the field names required
        # by the existing GateKeeper contract.
        ts = (
            candidate.get("fetched_at")
            or candidate.get("observed_at")
            or candidate.get("ingested_at")
            or candidate.get("ae20_canonical_pair_refresh_at")
        )
        refresh_ok = _cell(candidate.get("ae20_canonical_pair_refresh_lookup_ok")) in {"true", "True", "TRUE", "1"}
        refresh_status = _cell(candidate.get("ae20_canonical_pair_refresh_status"))
        provider_verified = refresh_ok or refresh_status in {
            "provider_pair_verified",
            "provider_pair_verified_direct_fallback_validator_gap",
        }

        gk_source_row = {
            "provider_pair_url": candidate.get("provider_pair_url_exact"),
            "provider_pair_url_exact": candidate.get("provider_pair_url_exact"),
            "canonical_market_identity": candidate.get("canonical_market_identity"),
            "chain": candidate.get("chain"),
            "pair_address": candidate.get("pair_address"),
            "pool_address": candidate.get("pair_address"),
            "provider_pair_id": candidate.get("pair_address"),
            "price_usd": candidate.get("price_usd"),
            "latest_price": candidate.get("price_usd"),
            "liquidity_usd": candidate.get("liquidity_usd"),
            "latest_liquidity": candidate.get("liquidity_usd"),
            "price_source_key": candidate.get("price_source_key"),
            "source_provider": "dexscreener",
            "observed_at": candidate.get("observed_at"),
            "fetched_at": candidate.get("fetched_at"),
            "ingested_at": candidate.get("ingested_at"),
            "price_updated_at": ts,
            "liquidity_updated_at": ts,
            "provider_last_seen_at": ts,
            "symbol": candidate.get("symbol") or candidate.get("token_symbol"),
            "token_symbol": candidate.get("token_symbol") or candidate.get("symbol"),
            "verification_status": "provider_pair_verified" if provider_verified else refresh_status,
            "freshness_status": "fresh" if ts else "",
            "identity_status": "pair_and_tokens_separated",
            "shown_as_token_contract": False,
            "paper_demo_only": True,
            "live_trading_ready": False,
            "legacy_market_snapshots_used": False,
            "market_snapshots_used": False,
        }

        gk_candidate = build_clean_forward_gatekeeper_candidate(gk_source_row)
        if isinstance(gk_candidate, dict) and gk_candidate.get("candidate"):
            gate = validate_market_data_gate(gk_candidate["candidate"])
        else:
            gate = validate_market_data_gate(gk_source_row)
        gate_dict = gate.to_dict() if hasattr(gate, "to_dict") else dict(gate)
        gate_passed = bool(gate_dict.get("passed"))
        gate_blocker = _cell(gate_dict.get("primary_blocker") or gate_dict.get("rejection_code"))
        gate_status = "GATEKEEPER_PASS" if gate_passed else "GATEKEEPER_BLOCK"
    except Exception as exc:  # noqa: BLE001
        gate_status = "GATEKEEPER_EVAL_ERROR"
        gate_blocker = f"{type(exc).__name__}:{exc}"
        gate_dict = {"passed": False, "error": gate_blocker}

    try:
        from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

        price = None
        try:
            price = float(candidate.get("price_usd")) if candidate.get("price_usd") not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        risk = evaluate_demo_risk_guard(
            requested_notional=25.0,
            demo_equity=float(paper_state.get("cash_usd") or paper_state.get("starting_capital") or 10000.0),
            open_positions=list(paper_state.get("open_positions") or []),
            pair_address=_cell(candidate.get("pair_address")) or None,
            symbol=_cell(candidate.get("symbol")) or None,
            chain=_cell(candidate.get("chain")) or None,
            price=price,
            liquidity=(
                float(candidate["liquidity_usd"])
                if candidate.get("liquidity_usd") not in (None, "")
                else None
            ),
            gate_result=gate_dict if isinstance(gate_dict, dict) else None,
        )
        risk_passed = bool(risk.get("passed") or risk.get("risk_guard_passed"))
        risk_blocker = _cell(
            risk.get("primary_blocker")
            or risk.get("rejection_code")
            or risk.get("risk_guard_reason")
        )
        risk_status = "RISKGUARD_PASS" if risk_passed else "RISKGUARD_BLOCK"
    except Exception as exc:  # noqa: BLE001
        risk_status = "RISKGUARD_EVAL_ERROR"
        risk_blocker = f"{type(exc).__name__}:{exc}"
        risk = {"passed": False, "error": risk_blocker}

    return {
        "gatekeeper_result": gate_status,
        "gatekeeper_passed": gate_passed,
        "gatekeeper_blocker": gate_blocker,
        "riskguard_result": risk_status,
        "riskguard_passed": risk_passed,
        "riskguard_blocker": risk_blocker,
        "gate_raw": gate_dict if isinstance(gate_dict, dict) else {},
        "risk_raw": risk if isinstance(risk, dict) else {},
    }


def derive_strict_exploration(
    candidate: dict[str, Any],
    ae16: dict[str, Any],
    ae17: dict[str, Any],
    ae18: dict[str, Any],
    ae19: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Derive strict vs exploration decisions. LLM never authorizes execution."""
    skip_reason = ""
    action_reason = ""
    ae16_status = _cell(ae16.get("ae16_status"))
    tier = _cell(ae16.get("ae16_consensus_tier") or ae16.get("consensus_tier"))

    if not candidate.get("identity_ok"):
        skip_reason = "CLEAN_FORWARD_IDENTITY_INCOMPLETE"
        strict = "STRICT_BLOCKED"
        exploration = "EXPLORATION_SKIP"
        final = "NO_TRADE"
    elif ae16_status not in {
        "AE16_EVIDENCE_ATTACHED",
        "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
    }:
        # Unmatched / unavailable — not attached evidence.
        skip_reason = "AE16_MODEL_EVIDENCE_UNAVAILABLE"
        if ae16_status in {
            "AE16_EVIDENCE_UNAVAILABLE",
            "AE16_JOIN_NOT_FOUND",
            "AE16_EXACT_DERIVED_BRIDGE_NOT_FOUND",
            "",
        }:
            skip_reason = "AE16_MODEL_EVIDENCE_UNAVAILABLE"
        strict = "STRICT_BLOCKED"
        exploration = "EXPLORATION_WATCH"
        final = "NO_TRADE"
        action_reason = "shadow_watch_without_strict_authority"
    elif tier in {"REJECT", "AE16_REJECT"}:
        skip_reason = "AE16_REJECT"
        strict = "STRICT_BLOCKED"
        exploration = "EXPLORATION_RESEARCH"
        final = "NO_TRADE"
        action_reason = "ae16_consensus_reject"
    else:
        # AE16_EVIDENCE_ATTACHED (including attached packages whose tier is
        # MODEL_EVIDENCE_UNAVAILABLE). Do NOT use AE16_MODEL_EVIDENCE_UNAVAILABLE
        # as skip reason for matched rows — continue to meta/gate/risk.
        recognized_tiers = {
            "TAB_XGB_RF_ALL3",
            "TAB_RF_ONLY",
            "TAB_XGB_ONLY",
            "RF_XGB_ONLY",
            "XGB_RF_ONLY",
            "SINGLE_MODEL_ONLY",
            "MODEL_EVIDENCE_UNAVAILABLE",
        }
        if _cell(ae17.get("meta_decision")) in {
            "META_REJECT",
            "META_LOW_CONFIDENCE",
            "META_UNAVAILABLE",
            "",
        }:
            if _cell(ae17.get("meta_decision")) in {
                "META_LOW_CONFIDENCE",
                "META_UNAVAILABLE",
                "",
            }:
                skip_reason = "AE17_META_LOW_CONFIDENCE"
            else:
                skip_reason = "AE16_REJECT"
            strict = "STRICT_BLOCKED"
            exploration = "EXPLORATION_RESEARCH"
            final = "NO_TRADE"
            action_reason = (
                f"ae16_attached_tier={tier or 'UNKNOWN'}; meta_blocked"
                if tier in recognized_tiers or tier
                else "meta_blocked"
            )
        elif ae18.get("ae18_status") in {
            "AE18_CONTEXT_UNAVAILABLE",
            "AE18_RESOLVER_UNRESOLVED",
            "AE18_RESOLVER_AMBIGUOUS",
        }:
            skip_reason = "AE18_CONTEXT_UNAVAILABLE"
            strict = "STRICT_BLOCKED"
            exploration = "EXPLORATION_WATCH"
            final = "NO_TRADE"
        elif not gates.get("gatekeeper_passed"):
            blocker = _cell(gates.get("gatekeeper_blocker")).upper()
            if "STALE" in blocker:
                skip_reason = "STALE_PRICE"
            elif "MISSING_PRICE" in blocker or (
                "PRICE" in blocker and "MISSING" in blocker
            ):
                skip_reason = "MISSING_PRICE"
            elif "ACTIVITY" in blocker:
                skip_reason = "MARKET_ACTIVITY_BLOCK"
            else:
                skip_reason = "GATEKEEPER_BLOCK"
            strict = "STRICT_BLOCKED"
            exploration = "EXPLORATION_WATCH"
            final = "NO_TRADE"
        elif not gates.get("riskguard_passed"):
            blocker = _cell(gates.get("riskguard_blocker")).upper()
            if "COOLDOWN" in blocker:
                skip_reason = "COOLDOWN"
            elif "DUPLICATE_PAIR_GUARD" in blocker or "DUPLICATE_OPEN_POSITION" in blocker or "PAIR_ALREADY_OPEN" in blocker or "PAIR_LOCK" in blocker:
                skip_reason = "DUPLICATE_OPEN_POSITION"
            elif "MAX_OPEN" in blocker:
                skip_reason = "MAX_OPEN_POSITIONS"
            elif "EXPOSURE" in blocker:
                skip_reason = "MAX_EXPOSURE"
            else:
                skip_reason = "RISKGUARD_BLOCK"
            strict = "STRICT_BLOCKED"
            exploration = "EXPLORATION_WATCH"
            final = "NO_TRADE"
        else:
            # AE20_STRICT_EXPLORATION_SEMANTICS_FIX_V1
            # Strict paper/demo approval is reserved for stronger AE16 tiers.
            # Research-only tiers may be explored in paper/demo, but must not be labeled strict.
            meta = _cell(ae17.get("meta_decision"))
            strict_eligible_tiers = {"TAB_XGB_RF_ALL3", "TAB_RF_ONLY"}
            exploration_paper_demo_tiers = {"TAB_XGB_ONLY"}

            if meta in {"META_STRONG_WATCH", "META_SECONDARY_WATCH"} and tier in strict_eligible_tiers:
                strict = "STRICT_APPROVED_PAPER_DEMO"
                exploration = "EXPLORATION_PAPER_DEMO"
                final = "PAPER_DEMO_OPEN"
                action_reason = f"strict_and_exploration_paper_demo_only:tier={tier};meta={meta}"
                skip_reason = ""
            elif meta == "META_STRONG_WATCH" and tier in exploration_paper_demo_tiers:
                strict = "STRICT_BLOCKED_RESEARCH_TIER"
                exploration = "EXPLORATION_PAPER_DEMO"
                final = "PAPER_DEMO_OPEN"
                action_reason = f"exploration_paper_demo_only:tier={tier};meta={meta}"
                skip_reason = ""
            elif tier == "SINGLE_MODEL_ONLY":
                skip_reason = "SINGLE_MODEL_RESEARCH_ONLY"
                strict = "STRICT_BLOCKED_RESEARCH_TIER"
                exploration = "EXPLORATION_WATCH"
                final = "NO_TRADE"
                action_reason = f"single_model_research_watch_only:tier={tier};meta={meta}"
            else:
                skip_reason = "EXPLORATION_ONLY"
                strict = "STRICT_BLOCKED"
                exploration = "EXPLORATION_WATCH"
                final = "NO_TRADE"
                action_reason = f"exploration_watch_only:tier={tier};meta={meta}"

    # LLM BUY/SELL cannot authorize or unblock.
    llm_label = _cell(ae19.get("llm_action_label")).upper()
    if llm_label in {"BUY", "SELL"} and final == "NO_TRADE":
        action_reason = (action_reason + "|llm_label_ignored_no_authority").strip("|")
    if ae19.get("llm_authorizes_execution"):
        ae19["llm_authorizes_execution"] = False

    return {
        "strict_decision": strict,
        "exploration_decision": exploration,
        "final_paper_demo_decision": final,
        "skip_reason": skip_reason,
        "action_reason": action_reason
        or ("skipped:" + skip_reason if skip_reason else "evaluated"),
        "trade_authority": False,
        "live_trading_enabled": False,
        "wallet_connected": False,
        "profitability_claim": False,
        "created_during_ae20": True,
        "preexisting_baseline": False,
    }


def build_integrated_decision(
    *,
    run_id: str,
    cycle_id: str,
    candidate: dict[str, Any],
    ae16: dict[str, Any],
    ae17: dict[str, Any],
    ae18: dict[str, Any],
    ae19: dict[str, Any],
    gates: dict[str, Any],
    path_decisions: dict[str, Any],
) -> dict[str, Any]:
    decision_id = _new_id("ae20dec")
    return {
        "ae20_run_id": run_id,
        "ae20_cycle_id": cycle_id,
        "ae20_decision_id": decision_id,
        "candidate_id": candidate.get("candidate_id"),
        "clean_forward_candidate_id": candidate.get("clean_forward_candidate_id"),
        "provider_pair_url_exact": candidate.get("provider_pair_url_exact"),
        "canonical_market_identity": candidate.get("canonical_market_identity"),
        "normalized_provider_pair_url_key": candidate.get("normalized_provider_pair_url_key"),
        "price_source_key": candidate.get("price_source_key"),
        "chain": candidate.get("chain"),
        "pair_address": candidate.get("pair_address"),
        "observed_at": candidate.get("observed_at"),
        "fetched_at": candidate.get("fetched_at"),
        "ingested_at": candidate.get("ingested_at"),
        "identity_status": candidate.get("identity_status"),
        "ae16_status": ae16.get("ae16_status"),
        "ae16_provider_pair_url_original": ae16.get("ae16_provider_pair_url_original"),
        "ae16_consensus_tier": ae16.get("ae16_consensus_tier") or ae16.get("consensus_tier"),
        "ae16_model_vote_count": ae16.get("ae16_model_vote_count") or ae16.get("model_vote_count"),
        "ae16_attached_model_count": ae16.get("attached_model_count"),
        "ae16_rf_evidence_status": ae16.get("ae16_rf_evidence_status") or ae16.get("rf_evidence_status"),
        "ae16_xgb_evidence_status": ae16.get("ae16_xgb_evidence_status") or ae16.get("xgb_evidence_status"),
        "ae16_tab_evidence_status": ae16.get("ae16_tab_evidence_status") or ae16.get("tab_evidence_status"),
        "ae16_rf_score": ae16.get("ae16_rf_score") if ae16.get("ae16_rf_score") is not None else ae16.get("rf_score"),
        "ae16_xgb_score": ae16.get("ae16_xgb_score") if ae16.get("ae16_xgb_score") is not None else ae16.get("xgb_score"),
        "ae16_tab_score": ae16.get("ae16_tab_score") if ae16.get("ae16_tab_score") is not None else ae16.get("tab_score"),
        "ae16_rf_vote": ae16.get("ae16_rf_vote") if "ae16_rf_vote" in ae16 else ae16.get("rf_vote"),
        "ae16_xgb_vote": ae16.get("ae16_xgb_vote") if "ae16_xgb_vote" in ae16 else ae16.get("xgb_vote"),
        "ae16_tab_vote": ae16.get("ae16_tab_vote") if "ae16_tab_vote" in ae16 else ae16.get("tab_vote"),
        "ae16_consensus_reason": ae16.get("ae16_consensus_reason") or ae16.get("consensus_reason"),
        "ae16_consensus_engine_version": ae16.get("ae16_consensus_engine_version")
        or ae16.get("consensus_engine_version"),
        "exact_identity_join_used": ae16.get("exact_identity_join_used", True),
        "case_insensitive_join_used": ae16.get("case_insensitive_join_used", False),
        "lowercase_join_used": ae16.get("lowercase_join_used", False),
        "casefold_join_used": ae16.get("casefold_join_used", False),
        "forbidden_pair_chain_join_used": ae16.get("forbidden_pair_chain_join_used", False),
        "ae17_status": ae17.get("ae17_status"),
        "ae17_meta_score": ae17.get("meta_score"),
        "ae17_meta_decision": ae17.get("meta_decision"),
        "ae17_meta_formula_version": ae17.get("meta_formula_version"),
        "ae17_score_integrity_status": ae17.get("score_integrity_status"),
        "ae17_context_missingness_component": ae17.get("context_missingness_component"),
        "ae18_status": ae18.get("ae18_status"),
        "ae18_context_summary": ae18.get("context_summary"),
        "ae18_missingness_flags": ae18.get("missingness_flags"),
        "ae18_resolver_status": ae18.get("resolver_status"),
        "ae19_status": ae19.get("ae19_status"),
        "ae19_llm_task_ref": ae19.get("llm_task_ref"),
        "ae19_authority_status": ae19.get("authority_status"),
        "ae19_llm_action_label": ae19.get("llm_action_label"),
        "gatekeeper_result": gates.get("gatekeeper_result"),
        "gatekeeper_blocker": gates.get("gatekeeper_blocker"),  # AE20_GATEKEEPER_BLOCKER_PERSISTENCE_FIX_V1
        "riskguard_result": gates.get("riskguard_result"),
        "riskguard_blocker": gates.get("riskguard_blocker"),
        **path_decisions,
        "order_id": "",
        "position_id": "",
        "outcome_id": "",
    }
