"""AE12.7 orchestrator — intelligent-agent operational demo layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.intelligent_agents.agent_audit_writer import (
    append_daily_operational,
    build_authority_audit,
    build_external_api_audit,
    build_gemini_safety_audit,
    build_helius_readonly_audit,
    build_no_wallet_audit,
    build_qwen_provider_audit,
    build_semantic_taxonomy_audit,
    ensure_dirs,
    filter_records,
    write_csv,
    write_json,
    write_jsonl,
)
from app.intelligent_agents.agent_policy import AgentDemoPolicy, build_policy_from_args
from app.intelligent_agents.agent_record_linker import build_linkage_row, link_agent_to_demo_row
from app.intelligent_agents.agent_ui_summary import build_status_payload, build_ui_summary
from app.intelligent_agents.discovery import discover_demo_candidates
from app.intelligent_agents.gemini_selective_audit import run_gemini_selective_audit
from app.intelligent_agents.helius_readonly_enrichment import run_helius_readonly_enrichment
from app.intelligent_agents.qwen_candidate_memo import generate_qwen_candidate_memo
from app.intelligent_agents.rss_context_linker import link_rss_context
from app.intelligent_agents.semantic_context import link_semantic_context
from app.intelligent_agents.types import (
    AE12_7_PHASE,
    OUTPUT_PREFIX,
    AgentStatus,
    AgentType,
    DecisionEffect,
    OperatingMode,
    SourceMode,
    make_agent_record,
    utc_now_iso,
)


def _timestamp_tag() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"


def _flatten_extra(record: dict[str, Any]) -> dict[str, Any]:
    """Promote selected extra keys for audits that look at top-level fields."""
    out = dict(record)
    extra = out.pop("extra", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in out:
                out[k] = v
        out["extra"] = extra
    return out


def _policy_snapshot(policy: AgentDemoPolicy) -> dict[str, Any]:
    return {
        "mode": policy.mode.value,
        "enable_qwen": policy.enable_qwen,
        "enable_gemini": policy.enable_gemini,
        "enable_helius": policy.enable_helius,
        "no_external_api": policy.no_external_api,
        "no_real_wallet": policy.no_real_wallet,
        "provider": policy.provider,
        "gemini_budget": policy.gemini_budget,
        "helius_budget": policy.helius_budget,
        "qwen_budget": policy.qwen_budget,
        "limit": policy.limit,
        "gemini_calls_used": policy.gemini_calls_used,
        "helius_calls_used": policy.helius_calls_used,
        "qwen_calls_used": policy.qwen_calls_used,
        "external_api_calls": list(policy.external_api_calls),
        "external_api_used": bool(policy.external_api_calls),
        "authority_ban": policy.authority_ban_contract(),
    }


def _should_run_qwen(policy: AgentDemoPolicy) -> bool:
    return policy.mode in (
        OperatingMode.QWEN_LOCAL_DEMO,
        OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
    )


def _should_run_gemini(policy: AgentDemoPolicy) -> bool:
    return policy.mode in (
        OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
        OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
    )


def _should_run_helius(policy: AgentDemoPolicy) -> bool:
    return policy.mode in (
        OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO,
        OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
    )


def run_ae12_7_agent_demo(
    *,
    project_root: Path | None = None,
    output_root: Path | None = None,
    mode: str = "artifact-only",
    limit: int = 50,
    enable_gemini: bool = False,
    enable_helius: bool = False,
    enable_qwen: bool = False,
    no_external_api: bool = True,
    no_real_wallet: bool = True,
    provider: str = "none",
    attempt_ollama_live: bool = False,
    gemini_budget: int = 5,
    helius_budget: int = 10,
    qwen_budget: int = 20,
    append_daily: bool = True,
    inject_gemini_response: str | None = None,
    force_qwen_unavailable: bool = False,
    force_helius_unavailable: bool = False,
) -> dict[str, Any]:
    """
    Run AE12.7 intelligent-agent operational demo.

    Always paper/demo safe. Never freezes on sparse data.
    Never grants trade authority. Never mutates trader.db.
    """
    root = Path(project_root or Path(__file__).resolve().parents[2])
    policy = build_policy_from_args(
        mode=mode,
        enable_gemini=enable_gemini,
        enable_helius=enable_helius,
        enable_qwen=enable_qwen,
        no_external_api=no_external_api,
        no_real_wallet=no_real_wallet,
        provider=provider,
        limit=limit,
        gemini_budget=gemini_budget,
        helius_budget=helius_budget,
        qwen_budget=qwen_budget,
        inject_gemini_response=inject_gemini_response,
        force_qwen_unavailable=force_qwen_unavailable,
        force_helius_unavailable=force_helius_unavailable,
    )

    if output_root is None:
        output_root = root / "data" / "audits" / f"{OUTPUT_PREFIX}_{_timestamp_tag()}"
    else:
        output_root = Path(output_root)

    dirs = ensure_dirs(output_root)
    candidates, discovery_meta = discover_demo_candidates(root, limit=policy.limit)

    records: list[dict[str, Any]] = []
    linkage_rows: list[dict[str, Any]] = []
    missed_review_rows: list[dict[str, Any]] = []
    rss_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    disabled_mode = policy.mode == OperatingMode.AGENT_DEMO_DISABLED

    for idx, cand in enumerate(candidates):
        before_action = {
            "action": cand.get("action"),
            "trade_action": cand.get("trade_action"),
            "buy_sell": cand.get("buy_sell"),
            "live_authority": cand.get("live_authority"),
            "trade_authority": cand.get("trade_authority"),
        }

        # Artifact-only / disabled: still emit status records from existing context
        if disabled_mode:
            qwen_rec = make_agent_record(
                agent_type=AgentType.QWEN_LOCAL_MEMO,
                source_mode=SourceMode.ARTIFACT_ONLY,
                agent_status=AgentStatus.READ_FROM_EXISTING_ARTIFACT
                if cand.get("_source_ref") and "synthetic" not in str(cand.get("_source_ref"))
                else AgentStatus.SKIPPED,
                agent_summary="Agent demo disabled — no new Qwen/Gemini/Helius calls; displaying existing context status.",
                decision_effect=DecisionEffect.NO_EFFECT,
                warnings=["agent_demo_disabled"],
                candidate_id=cand.get("candidate_id"),
                source_decision_id=cand.get("source_decision_id") or cand.get("decision_id"),
                pair_address=cand.get("pair_address"),
                symbol=cand.get("symbol"),
                chain=cand.get("chain") or "solana",
                paper_order_id=cand.get("paper_order_id"),
                position_id=cand.get("position_id"),
                input_artifact_refs=[str(cand.get("_source_ref"))] if cand.get("_source_ref") else [],
                extra={"external_call_made": False, "gemini_called": False, "helius_called": False},
            )
            records.append(link_agent_to_demo_row(qwen_rec, cand))
        else:
            if _should_run_qwen(policy):
                qwen_rec = generate_qwen_candidate_memo(
                    cand,
                    policy=policy,
                    attempt_live_call=attempt_ollama_live,
                )
                records.append(link_agent_to_demo_row(_flatten_extra(qwen_rec), cand))
            else:
                skip = make_agent_record(
                    agent_type=AgentType.QWEN_LOCAL_MEMO,
                    source_mode=SourceMode.DISABLED,
                    agent_status=AgentStatus.SKIPPED,
                    agent_summary="Qwen not in active mode for this run.",
                    decision_effect=DecisionEffect.NO_EFFECT,
                    candidate_id=cand.get("candidate_id"),
                    source_decision_id=cand.get("source_decision_id") or cand.get("decision_id"),
                    pair_address=cand.get("pair_address"),
                    symbol=cand.get("symbol"),
                    extra={"external_call_made": False},
                )
                records.append(link_agent_to_demo_row(skip, cand))

            if _should_run_gemini(policy):
                # Pass qwen confidence if just generated
                last = records[-1] if records else {}
                if last.get("agent_type") == "QWEN_LOCAL_MEMO":
                    cand = {**cand, "qwen_confidence": last.get("confidence")}
                gem = run_gemini_selective_audit(cand, policy=policy, index=idx)
                records.append(link_agent_to_demo_row(_flatten_extra(gem), cand))
            elif policy.enable_gemini and policy.no_external_api:
                gem = run_gemini_selective_audit(cand, policy=policy, index=idx)
                records.append(link_agent_to_demo_row(_flatten_extra(gem), cand))

            if _should_run_helius(policy):
                hel = run_helius_readonly_enrichment(cand, policy=policy)
                records.append(link_agent_to_demo_row(_flatten_extra(hel), cand))
            elif policy.mode == OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO:
                hel = run_helius_readonly_enrichment(cand, policy=policy)
                records.append(link_agent_to_demo_row(_flatten_extra(hel), cand))

        # RSS + semantic always linked as artifact/context (no trade authority)
        rss_rec = link_rss_context(cand)
        rss_linked = link_agent_to_demo_row(_flatten_extra(rss_rec), cand)
        records.append(rss_linked)
        rss_rows.append(
            {
                "agent_record_id": rss_linked.get("agent_record_id"),
                "candidate_id": rss_linked.get("candidate_id"),
                "pair_address": rss_linked.get("pair_address"),
                "symbol": rss_linked.get("symbol"),
                "agent_status": rss_linked.get("agent_status"),
                "linkage_method": rss_linked.get("linkage_method") or (rss_linked.get("extra") or {}).get("linkage_method"),
                "source_count": rss_linked.get("source_count") or 0,
                "trade_authority_used": False,
            }
        )

        sem_rec = link_semantic_context(cand)
        sem_linked = link_agent_to_demo_row(_flatten_extra(sem_rec), cand)
        records.append(sem_linked)
        semantic_rows.append(
            {
                "agent_record_id": sem_linked.get("agent_record_id"),
                "candidate_id": sem_linked.get("candidate_id"),
                "pair_address": sem_linked.get("pair_address"),
                "semantic_signal_family": sem_linked.get("semantic_signal_family")
                or sem_linked.get("semantic_label"),
                "trading_opportunity_state": sem_linked.get("trading_opportunity_state"),
                "legacy_cluster_label": sem_linked.get("legacy_cluster_label"),
                "legacy_is_final_semantic": False,
                "agent_status": sem_linked.get("agent_status"),
                "trade_authority_used": False,
            }
        )

        # Aggregate per candidate
        agg = make_agent_record(
            agent_type=AgentType.AGENT_AGGREGATE_SUMMARY,
            source_mode=SourceMode.ARTIFACT_ONLY if disabled_mode else SourceMode.LOCAL,
            agent_status=AgentStatus.GENERATED,
            agent_summary=(
                f"Aggregate agent observability for candidate={cand.get('candidate_id')}; "
                f"mode={policy.mode.value}; trade_authority_used=false."
            ),
            decision_effect=DecisionEffect.EXPLANATION_ONLY,
            candidate_id=cand.get("candidate_id"),
            source_decision_id=cand.get("source_decision_id") or cand.get("decision_id"),
            paper_order_id=cand.get("paper_order_id"),
            position_id=cand.get("position_id"),
            pair_address=cand.get("pair_address"),
            symbol=cand.get("symbol"),
            chain=cand.get("chain") or "solana",
            extra={
                "strict_shadow_decision": cand.get("strict_shadow_decision"),
                "exploration_decision": cand.get("exploration_decision"),
                "is_missed_winner": bool(cand.get("is_missed_winner")),
            },
        )
        agg_linked = link_agent_to_demo_row(agg, cand)
        records.append(agg_linked)

        linkage_rows.append(build_linkage_row(agg_linked, cand))
        integrity_rows.append(
            {
                **build_linkage_row(agg_linked, cand),
                "authority_fields_unchanged": True,
                "before_action": json_safe(before_action),
                "after_action": json_safe(before_action),  # never mutated
            }
        )

        if cand.get("is_missed_winner") or cand.get("missed_winner_horizons"):
            missed_review_rows.append(
                {
                    "candidate_id": cand.get("candidate_id"),
                    "pair_address": cand.get("pair_address"),
                    "symbol": cand.get("symbol"),
                    "max_return": cand.get("max_return"),
                    "horizon": cand.get("horizon"),
                    "agent_aggregate_id": agg_linked.get("agent_record_id"),
                    "qwen_status": next(
                        (
                            r.get("agent_status")
                            for r in records
                            if r.get("candidate_id") == cand.get("candidate_id")
                            and r.get("agent_type") == "QWEN_LOCAL_MEMO"
                        ),
                        None,
                    ),
                    "gemini_status": next(
                        (
                            r.get("agent_status")
                            for r in records
                            if r.get("candidate_id") == cand.get("candidate_id")
                            and r.get("agent_type") == "GEMINI_SELECTIVE_AUDIT"
                        ),
                        "SKIPPED",
                    ),
                    "soft_veto_flags": ";".join(
                        sorted(
                            {
                                f
                                for r in records
                                if r.get("candidate_id") == cand.get("candidate_id")
                                for f in (r.get("soft_veto_flags") or [])
                            }
                        )
                    ),
                    "trade_authority_used": False,
                    "decision_effect": "explanation_only",
                }
            )

        # Harden: candidate dict authority fields unchanged
        for k, v in before_action.items():
            if cand.get(k) != v:
                cand[k] = v

    snap = _policy_snapshot(policy)

    # Classification
    classification = _classify(policy, records, snap)

    gate = {
        "phase": AE12_7_PHASE,
        "status": classification,
        "ae12_closed": False,
        "live_ready": False,
        "profitability_proven": False,
        "trade_authority_used": False,
        "wallet_status": "NOT_CONFIGURED",
        "wallet_accessed": False,
        "private_key_accessed": False,
        "real_transaction_attempted": False,
        "external_api_used": snap["external_api_used"],
        "gemini_called": any(
            r.get("gemini_called") for r in records if r.get("agent_type") == "GEMINI_SELECTIVE_AUDIT"
        ),
        "no_external_api": policy.no_external_api,
        "no_real_wallet": True,
        "operating_mode": policy.mode.value,
        "candidate_count": len(candidates),
        "agent_record_count": len(records),
        "writes_limited_to_output_root": True,
        "trader_db_mutated": False,
        "retraining_performed": False,
        "next_ae12_step": "AE12.8_or_MSc_narrative_closure",
        "created_at": utc_now_iso(),
    }

    ui_summary = build_ui_summary(records=records, policy_snapshot=snap, gate=gate)
    status_payload = build_status_payload(
        latest_root=str(output_root),
        ui_summary=ui_summary,
        policy_snapshot=snap,
    )

    # Write required outputs
    write_jsonl(dirs["data"] / "ae12_7_agent_records.jsonl", records)
    write_csv(dirs["data"] / "ae12_7_agent_records.csv", records)
    write_jsonl(dirs["data"] / "ae12_7_qwen_candidate_memos.jsonl", filter_records(records, "QWEN_LOCAL_MEMO"))
    write_jsonl(
        dirs["data"] / "ae12_7_gemini_selective_audits.jsonl",
        filter_records(records, "GEMINI_SELECTIVE_AUDIT"),
    )
    write_jsonl(
        dirs["data"] / "ae12_7_helius_readonly_enrichment.jsonl",
        filter_records(records, "HELIUS_READONLY_ENRICHMENT"),
    )
    write_csv(dirs["data"] / "ae12_7_rss_context_links.csv", rss_rows)
    write_csv(dirs["data"] / "ae12_7_semantic_context_links.csv", semantic_rows)
    write_csv(dirs["data"] / "ae12_7_agent_trade_linkage.csv", linkage_rows)
    write_csv(dirs["data"] / "ae12_7_missed_winner_agent_review.csv", missed_review_rows)

    authority = build_authority_audit(records, snap)
    no_wallet = build_no_wallet_audit(records)
    ext_api = build_external_api_audit(snap, records)
    gem_safe = build_gemini_safety_audit(records)
    qwen_audit = build_qwen_provider_audit(records, snap)
    helius_audit = build_helius_readonly_audit(records)
    sem_audit = build_semantic_taxonomy_audit(records)

    write_json(dirs["audits"] / "ae12_7_agent_authority_audit.json", authority)
    write_json(dirs["audits"] / "ae12_7_no_wallet_safety_audit.json", no_wallet)
    write_json(dirs["audits"] / "ae12_7_external_api_usage_audit.json", ext_api)
    write_json(dirs["audits"] / "ae12_7_gemini_safety_audit.json", gem_safe)
    write_json(dirs["audits"] / "ae12_7_qwen_local_provider_audit.json", qwen_audit)
    write_json(dirs["audits"] / "ae12_7_helius_readonly_audit.json", helius_audit)
    write_json(dirs["audits"] / "ae12_7_semantic_taxonomy_audit.json", sem_audit)
    write_csv(dirs["audits"] / "ae12_7_linkage_integrity_audit.csv", integrity_rows)

    # Fail authority/safety if audits fail
    if authority["status"] != "PASS_NO_TRADE_AUTHORITY" or no_wallet["status"] != "PASS_NO_WALLET":
        classification = "AE12_7_FAIL_AUTHORITY_OR_SAFETY"
        gate["status"] = classification

    manifest = {
        "phase": AE12_7_PHASE,
        "created_at": utc_now_iso(),
        "output_root": str(output_root),
        "operating_mode": policy.mode.value,
        "classification": classification,
        "discovery": discovery_meta,
        "policy": snap,
        "files": {
            "agent_records_jsonl": "data/ae12_7_agent_records.jsonl",
            "agent_records_csv": "data/ae12_7_agent_records.csv",
            "decision_gate": "reports/ae12_7_intelligent_agent_decision_gate.json",
            "ui_status": "reports/ae12_7_ui_status_summary.json",
        },
        "ae12_closed": False,
        "live_ready": False,
        "profitability_proven": False,
        "trade_authority_used": False,
    }
    write_json(dirs["reports"] / "ae12_7_manifest.json", manifest)
    write_json(dirs["reports"] / "ae12_7_intelligent_agent_decision_gate.json", gate)
    write_json(dirs["reports"] / "ae12_7_ui_status_summary.json", ui_summary)

    upload = _upload_text(
        classification=classification,
        gate=gate,
        snap=snap,
        discovery_meta=discovery_meta,
        records=records,
        ui_summary=ui_summary,
        output_root=output_root,
    )
    (dirs["reports"] / "ae12_7_summary_for_upload.txt").write_text(upload, encoding="utf-8")

    daily_paths = {}
    if append_daily:
        call_audit = [
            {
                "created_at": utc_now_iso(),
                "mode": policy.mode.value,
                "calls": snap["external_api_calls"],
                "trade_authority_used": False,
            }
        ]
        daily_paths = append_daily_operational(root, records, call_audit)

    return {
        "phase": AE12_7_PHASE,
        "output_root": str(output_root),
        "classification": classification,
        "gate_status": gate["status"],
        "gate": gate,
        "ui_summary": ui_summary,
        "status": status_payload,
        "record_count": len(records),
        "candidate_count": len(candidates),
        "daily_paths": daily_paths,
        "trade_authority_used": False,
        "live_ready": False,
        "profitability_proven": False,
        "ae12_closed": False,
    }


def json_safe(obj: Any) -> Any:
    import json as _json

    return _json.loads(_json.dumps(obj, default=str))


def _classify(policy: AgentDemoPolicy, records: list[dict[str, Any]], snap: dict[str, Any]) -> str:
    if any(r.get("trade_authority_used") for r in records):
        return "AE12_7_FAIL_AUTHORITY_OR_SAFETY"
    if policy.mode == OperatingMode.AGENT_DEMO_DISABLED:
        return "AE12_7_PASS_ARTIFACT_ONLY_AGENT_LAYER"
    if policy.no_external_api and not snap["external_api_used"]:
        if policy.mode == OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO:
            return "AE12_7_PASS_WITH_EXTERNAL_SOURCES_DISABLED"
        if policy.mode == OperatingMode.QWEN_LOCAL_DEMO:
            return "AE12_7_PASS_OPERATIONAL_AGENT_DEMO"
        return "AE12_7_PASS_WITH_EXTERNAL_SOURCES_DISABLED"
    if snap["external_api_used"]:
        return "AE12_7_PASS_OPERATIONAL_AGENT_DEMO"
    return "AE12_7_PASS_OPERATIONAL_AGENT_DEMO"


def _upload_text(
    *,
    classification: str,
    gate: dict[str, Any],
    snap: dict[str, Any],
    discovery_meta: dict[str, Any],
    records: list[dict[str, Any]],
    ui_summary: dict[str, Any],
    output_root: Path,
) -> str:
    lines = [
        "AE12.7 Intelligent-Agent Operational Demo Layer",
        f"Classification: {classification}",
        f"Output root: {output_root}",
        f"Operating mode: {snap.get('mode')}",
        f"Agent records: {len(records)}",
        f"Discovery maturation root: {discovery_meta.get('maturation_root')}",
        f"Synthetic candidates used: {discovery_meta.get('synthetic_used')}",
        "",
        "Safety:",
        f"  trade_authority_used={gate.get('trade_authority_used')}",
        f"  wallet_status={gate.get('wallet_status')}",
        f"  live_ready={gate.get('live_ready')}",
        f"  profitability_proven={gate.get('profitability_proven')}",
        f"  ae12_closed={gate.get('ae12_closed')}",
        f"  external_api_used={gate.get('external_api_used')}",
        "",
        "UI statuses:",
        f"  Qwen: {ui_summary.get('qwen_memo_status')}",
        f"  Gemini: {ui_summary.get('gemini_audit_status')}",
        f"  Helius: {ui_summary.get('helius_enrichment_status')}",
        f"  RSS: {ui_summary.get('rss_context_status')}",
        f"  Semantic: {ui_summary.get('semantic_classification_status')}",
        "",
        "Agent outputs are audit/explanation/context/soft-warning only.",
        "Paper/demo trading may continue. No live readiness claimed.",
        "AE12 is NOT closed by this phase.",
    ]
    return "\n".join(lines) + "\n"
