"""AE18 Context Intelligence Layer pipeline (real Helius/Solana continuation)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ae18 import CONTEXT_ENGINE_VERSION, PHASE, SAFETY_BOUNDARY
from app.ae18.audits import (
    audit_authority_safety,
    audit_context_value_presence,
    audit_helius_solana_readonly,
    audit_input_lineage,
    audit_missingness_provenance,
    audit_no_symbol_only_join,
    audit_real_helius_fetch,
    audit_reputation_scam,
    audit_rpc_cache_throttle,
    audit_rss_news,
    audit_semantic_context,
    audit_wallet_whale_evidence,
    audit_whale_score_separation,
    build_resolver_audit_csv,
    decide_classification,
)
from app.ae18.collectors import (
    collect_helius_solana_readonly,
    collect_reputation_scam_context,
    collect_rss_news_context,
    collect_semantic_context,
    collect_whale_evidence_separated,
    fetch_local_onchain_payload,
    open_readonly_db,
)
from app.ae18.constants import CLASSIFICATION_BLOCKED_NO_SOURCES
from app.ae18.discovery import discover_candidate_inputs, load_candidate_targets
from app.ae18.models import AE18ContextRecord, AE18MissingnessRecord, AE18ResolverLink
from app.ae18.preflight import (
    build_authenticated_rpc_url,
    maybe_fallback_to_public_rpc,
    resolve_rpc_config,
    run_solana_preflight_safety,
)
from app.ae18.real_fetch import fetch_candidate_onchain_context
from app.ae18.readonly_rpc import AE18ReadOnlyRpcClient
from app.ae18.resolver import resolve_candidate_identity, resolve_text_to_candidate
from app.ae18.selector import (
    DEFAULT_MAX_CANDIDATES,
    attach_selection_fields_from_rows,
    enrich_candidates_with_token_identity,
    load_open_paper_pairs,
    select_interesting_solana_candidates,
)
from app.consensus.serialization import read_csv_dicts, write_csv, write_json, write_jsonl, write_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_ae18_context_intelligence_layer(
    project_root: Path,
    *,
    ae17_root: str | Path | None = None,
    ae16_root: str | Path | None = None,
    output_root: str | Path | None = None,
    allow_external_fetch: bool = False,
    allow_public_rpc: bool = False,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    rpc_min_delay_ms: int = 250,
    rpc_max_calls: int = 250,
    signatures_per_pair: int = 10,
    transactions_per_pair: int = 10,
) -> dict[str, Any]:
    root = project_root.resolve()
    stamp = utc_stamp()
    out = Path(output_root) if output_root else root / "data" / "audits" / f"ae18_context_intelligence_{stamp}"
    if not out.is_absolute():
        out = root / out
    data_dir = out / "data"
    audits_dir = out / "audits"
    reports_dir = out / "reports"
    for d in (data_dir, audits_dir, reports_dir, root / "reports"):
        d.mkdir(parents=True, exist_ok=True)

    discovery = discover_candidate_inputs(root, ae17_root=ae17_root, ae16_root=ae16_root)
    discovery_dict = discovery.to_dict()
    if discovery.status != "AE18_INPUTS_DISCOVERED" or not discovery.candidate_csv:
        return _write_blocked(root, out, discovery_dict, CLASSIFICATION_BLOCKED_NO_SOURCES)

    candidates = load_candidate_targets(root, discovery.candidate_csv)
    if not candidates:
        return _write_blocked(root, out, discovery_dict, CLASSIFICATION_BLOCKED_NO_SOURCES)

    # Attach selection fields from source CSV rows
    try:
        source_rows = read_csv_dicts(
            root / discovery.candidate_csv if not Path(discovery.candidate_csv).is_absolute() else Path(discovery.candidate_csv)
        )
    except OSError:
        source_rows = []
    candidates = attach_selection_fields_from_rows(candidates, source_rows)
    candidates = enrich_candidates_with_token_identity(candidates, root)

    open_pairs = load_open_paper_pairs(root)
    selected, selection_rows = select_interesting_solana_candidates(
        candidates,
        max_candidates=max_candidates,
        open_paper_pair_addresses=open_pairs,
    )
    write_csv(data_dir / "ae18_interesting_candidate_selection.csv", selection_rows)

    # Pre-flight safety (always run before any external fetch)
    preflight = run_solana_preflight_safety()
    write_json(audits_dir / "ae18_solana_preflight_safety_audit.json", preflight)

    rpc_config = resolve_rpc_config(allow_public_rpc=allow_public_rpc)
    rpc_probe: dict[str, Any] = {}
    rpc_client: AE18ReadOnlyRpcClient | None = None
    if allow_external_fetch and preflight.get("preflight_passed") and rpc_config.get("rpc_configured"):
        rpc_config, rpc_probe = maybe_fallback_to_public_rpc(
            rpc_config,
            allow_public_rpc=allow_public_rpc,
        )
        write_json(audits_dir / "ae18_rpc_auth_probe_audit.json", {
            "helius_configured": rpc_config.get("helius_configured"),
            "rpc_provider_used": rpc_config.get("rpc_provider_used"),
            "probe": {k: v for k, v in rpc_probe.items() if k != "error_preview" or not rpc_probe.get("invalid_api_key")},
            "fallback_reason": rpc_config.get("fallback_reason"),
        })
        # If Helius auth failed and public fallback not allowed, mark unconfigured for fetch.
        if rpc_probe.get("invalid_api_key") and not rpc_probe.get("fallback_applied"):
            rpc_config = {**rpc_config, "rpc_configured": False, "rpc_provider_used": None}
        elif rpc_config.get("rpc_configured"):
            rpc_url = build_authenticated_rpc_url(rpc_config)
            rpc_client = AE18ReadOnlyRpcClient(
                rpc_url=rpc_url,
                provider_used=str(rpc_config.get("rpc_provider_used") or "HELIUS_RPC"),
                min_delay_ms=rpc_min_delay_ms,
                max_calls=rpc_max_calls,
            )

    conn = open_readonly_db(root)
    context_records: list[AE18ContextRecord] = []
    missingness_records: list[AE18MissingnessRecord] = []
    resolver_links: list[AE18ResolverLink] = []
    helius_rows: list[dict[str, Any]] = []
    rss_rows: list[dict[str, Any]] = []
    reputation_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    wallet_whale_rows: list[dict[str, Any]] = []
    tx_summary_rows: list[dict[str, Any]] = []
    value_presence_rows: list[dict[str, Any]] = []

    process_set = selected if (allow_external_fetch and selected) else candidates
    candidates_fetched = 0

    for candidate in process_set:
        self_link = resolve_candidate_identity(
            candidate,
            context_record_id=f"self_{candidate.clean_forward_candidate_id}",
        )
        resolver_links.append(self_link)

        used_real_fetch = False
        if (
            allow_external_fetch
            and rpc_client is not None
            and preflight.get("preflight_passed")
            and (candidate.chain or "").lower() == "solana"
            and candidate in selected
        ):
            result = fetch_candidate_onchain_context(
                candidate,
                rpc_client,
                signatures_per_pair=signatures_per_pair,
                transactions_per_pair=transactions_per_pair,
            )
            used_real_fetch = True
            candidates_fetched += 1
            context_records.append(result["helius_record"])
            context_records.append(result["wallet_record"])
            context_records.append(result["pool_record"])
            if result.get("missingness"):
                missingness_records.append(result["missingness"])
            helius_rows.append(result["helius_row"])
            wallet_whale_rows.append(result["wallet_row"])
            tx_summary_rows.append(result["tx_summary_row"])
            value_presence_rows.append(
                {
                    "clean_forward_candidate_id": candidate.clean_forward_candidate_id,
                    "price_source_key": candidate.price_source_key,
                    "chain": candidate.chain,
                    "pair_address": candidate.pair_address,
                    "value_presence_class": result["value_presence_class"],
                }
            )
        else:
            raw_payload = fetch_local_onchain_payload(conn, candidate)
            hel_rec, hel_miss, hel_safety = collect_helius_solana_readonly(
                candidate,
                raw_payload_row=raw_payload,
                allow_external=False,
            )
            if allow_external_fetch and (candidate.chain or "").lower() != "solana":
                hel_rec.missingness_reason = hel_rec.missingness_reason or "PROVIDER_NOT_CALLED_IN_THIS_MODE"
                value_presence_rows.append(
                    {
                        "clean_forward_candidate_id": candidate.clean_forward_candidate_id,
                        "price_source_key": candidate.price_source_key,
                        "chain": candidate.chain,
                        "pair_address": candidate.pair_address,
                        "value_presence_class": "NOT_SOLANA",
                    }
                )
            context_records.append(hel_rec)
            if hel_miss:
                missingness_records.append(hel_miss)
            helius_rows.append({**hel_rec.to_dict(), "safety": hel_safety})

            wallet_ev = (hel_rec.evidence_payload or {}).get("wallet_level_evidence", {})
            pool_rec, wallet_rec, whale_miss = collect_whale_evidence_separated(
                candidate,
                wallet_evidence=wallet_ev,
            )
            context_records.append(pool_rec)
            context_records.append(wallet_rec)
            if whale_miss:
                missingness_records.append(whale_miss)
            wallet_whale_rows.append(wallet_rec.to_dict())

        rss_rec, rss_miss, rss_items = collect_rss_news_context(candidate, conn)
        context_records.append(rss_rec)
        if rss_miss:
            missingness_records.append(rss_miss)
        for item in rss_items:
            link = resolve_text_to_candidate(
                item,
                candidates,
                context_record_id=rss_rec.context_record_id,
                observed_at=candidate.observed_at,
            )
            resolver_links.append(link)
            rss_rows.append({**item, "resolver_status": link.resolver_status})

        rep_rec, rep_miss = collect_reputation_scam_context(
            candidate,
            raw_payload_row=fetch_local_onchain_payload(conn, candidate) if not used_real_fetch else None,
        )
        context_records.append(rep_rec)
        if rep_miss:
            missingness_records.append(rep_miss)
        reputation_rows.append(rep_rec.to_dict())

        sem_rec, sem_miss = collect_semantic_context(candidate)
        context_records.append(sem_rec)
        if sem_miss:
            missingness_records.append(sem_miss)
        semantic_rows.append(sem_rec.to_dict())

    if conn:
        conn.close()

    # Demo symbol-only rejection
    sym_link = resolve_text_to_candidate(
        {"text_item_id": "demo_symbol_only_sol", "symbol": "SOL", "title": "SOL rallies today"},
        candidates,
        context_record_id="demo_symbol_only",
    )
    resolver_links.append(sym_link)

    rpc_stats = rpc_client.stats.to_dict() if rpc_client else {
        "rpc_calls_attempted": 0,
        "rpc_calls_successful": 0,
        "rpc_calls_failed": 0,
        "rpc_calls_skipped_by_cache": 0,
        "retry_after_used_count": 0,
        "backoff_retry_count": 0,
        "rate_limit_count": 0,
        "average_delay_ms": 0.0,
        "max_retries_reached_count": 0,
        "forbidden_method_attempts": [],
    }
    raw_log = rpc_client.raw_call_log if rpc_client else []
    write_jsonl(data_dir / "ae18_helius_solana_raw_rpc_calls.jsonl", raw_log)

    summary_rows = _build_candidate_summaries(process_set, context_records, resolver_links)
    missingness_summary = _build_missingness_summary(missingness_records)

    ctx_dicts = [r.to_dict() for r in context_records]
    write_csv(data_dir / "ae18_context_records.csv", ctx_dicts)
    write_jsonl(data_dir / "ae18_context_records.jsonl", ctx_dicts)
    write_csv(data_dir / "ae18_candidate_context_summary.csv", summary_rows)
    write_csv(data_dir / "ae18_resolver_links.csv", [l.to_dict() for l in resolver_links])
    write_csv(data_dir / "ae18_missingness_summary.csv", missingness_summary)
    write_csv(data_dir / "ae18_helius_solana_context.csv", helius_rows)
    write_jsonl(data_dir / "ae18_helius_solana_context.jsonl", helius_rows)
    if rss_rows:
        write_csv(data_dir / "ae18_rss_news_context.csv", rss_rows)
    if reputation_rows:
        write_csv(data_dir / "ae18_reputation_scam_context.csv", reputation_rows)
    if semantic_rows:
        write_csv(data_dir / "ae18_semantic_context.csv", semantic_rows)
    write_csv(data_dir / "ae18_wallet_whale_context.csv", wallet_whale_rows)
    write_jsonl(data_dir / "ae18_wallet_whale_context.jsonl", wallet_whale_rows)
    if tx_summary_rows:
        write_csv(data_dir / "ae18_onchain_transaction_summary.csv", tx_summary_rows)

    no_sym = audit_no_symbol_only_join(resolver_links)
    whale = audit_whale_score_separation(context_records)
    wallet_whale = audit_wallet_whale_evidence(context_records)
    miss = audit_missingness_provenance(context_records, missingness_records)
    readonly = audit_helius_solana_readonly(root)
    authority = audit_authority_safety(context_records)
    lineage = audit_input_lineage(context_records, discovery_dict)
    rss_audit = audit_rss_news(context_records, resolver_links)
    rep_audit = audit_reputation_scam(context_records)
    sem_audit = audit_semantic_context(context_records)

    context_extracted_count = sum(
        1 for r in value_presence_rows if r.get("value_presence_class") == "REAL_ONCHAIN_CONTEXT_EXTRACTED"
    )
    wallet_available_count = sum(1 for r in context_records if r.context_family == "wallet_whale" and r.available)
    wallet_missing_count = sum(1 for r in context_records if r.context_family == "wallet_whale" and not r.available)
    flow_extracted = sum(
        1
        for r in tx_summary_rows
        if r.get("flow_pressure_direction") in {"BUY_PRESSURE", "SELL_PRESSURE", "MIXED"}
    )
    flow_unknown = sum(1 for r in tx_summary_rows if r.get("flow_pressure_direction") == "UNKNOWN")

    real_fetch_audit = audit_real_helius_fetch(
        external_fetch_enabled=allow_external_fetch,
        rpc_config=rpc_config,
        selection_count=len(selection_rows),
        solana_selected=len(selected),
        candidates_fetched=candidates_fetched,
        rpc_stats=rpc_stats,
        raw_payloads_saved=len(raw_log),
        context_extracted_count=context_extracted_count,
        wallet_evidence_available_count=wallet_available_count,
        missingness_count=len(missingness_records),
    )
    cache_audit = audit_rpc_cache_throttle(rpc_stats)
    value_audit = audit_context_value_presence(value_presence_rows)

    available_count = sum(1 for r in context_records if r.available)
    classification = decide_classification(
        no_symbol_audit=no_sym,
        whale_audit=whale,
        missingness_audit=miss,
        readonly_audit=readonly,
        authority_audit=authority,
        discovery_status=discovery.status,
        context_record_count=len(context_records),
        available_source_count=available_count,
        external_fetch_enabled=allow_external_fetch,
        preflight_passed=bool(preflight.get("preflight_passed")),
        rpc_configured=bool(rpc_config.get("rpc_configured")),
        rpc_calls_attempted=int(rpc_stats.get("rpc_calls_attempted") or 0),
        rpc_calls_successful=int(rpc_stats.get("rpc_calls_successful") or 0),
        context_extracted_count=context_extracted_count,
        raw_payloads_saved=len(raw_log),
        wallet_whale_audit=wallet_whale,
    )

    write_json(audits_dir / "ae18_input_lineage_audit.json", lineage)
    write_csv(audits_dir / "ae18_resolver_audit.csv", build_resolver_audit_csv(resolver_links))
    write_json(audits_dir / "ae18_helius_solana_readonly_audit.json", readonly)
    write_json(audits_dir / "ae18_rss_news_audit.json", rss_audit)
    write_json(audits_dir / "ae18_reputation_scam_audit.json", rep_audit)
    write_json(audits_dir / "ae18_semantic_context_audit.json", sem_audit)
    write_json(audits_dir / "ae18_whale_score_separation_audit.json", whale)
    write_json(audits_dir / "ae18_wallet_whale_evidence_audit.json", wallet_whale)
    write_json(audits_dir / "ae18_missingness_provenance_audit.json", miss)
    write_json(audits_dir / "ae18_authority_safety_audit.json", authority)
    write_json(audits_dir / "ae18_no_symbol_only_join_audit.json", no_sym)
    write_json(audits_dir / "ae18_real_helius_fetch_audit.json", real_fetch_audit)
    write_json(audits_dir / "ae18_rpc_cache_throttle_audit.json", cache_audit)
    write_json(audits_dir / "ae18_context_value_presence_audit.json", value_audit)

    account_success = sum(1 for r in tx_summary_rows if r.get("account_found"))
    sig_success = sum(1 for r in tx_summary_rows if (r.get("signatures_found") or 0) > 0)
    tx_success = sum(1 for r in tx_summary_rows if (r.get("transactions_loaded") or 0) > 0)

    gate = {
        "phase": PHASE,
        "classification": classification,
        "created_at_utc": utc_now(),
        "context_engine_version": CONTEXT_ENGINE_VERSION,
        "ae18_status": "OPEN",
        "ae19_status": "BLOCKED",
        "ae20_status": "BLOCKED",
        "previous_ae18_status": "AE18_CONTEXT_INFRASTRUCTURE_AND_MISSINGNESS_CONTRACT_PASS",
        "previous_ae18_root": "data/audits/ae18_context_intelligence_20260727T143134Z",
        **SAFETY_BOUNDARY,
        "transaction_builder_available": False,
        "training_performed": False,
        "backtest_performed": False,
        "trader_db_mutated": False,
        "external_fetch_enabled": allow_external_fetch,
        "helius_configured": bool(rpc_config.get("helius_configured")),
        "rpc_provider_used": rpc_config.get("rpc_provider_used"),
        "preflight_passed": bool(preflight.get("preflight_passed")),
        "discovery": discovery_dict,
        "counts": {
            "candidates_total": len(candidates),
            "candidates_selected": len(selected),
            "solana_candidates_selected": len(selected),
            "candidates_fetched": candidates_fetched,
            "context_records": len(context_records),
            "helius_solana": sum(1 for r in context_records if r.context_family == "helius_solana"),
            "rss_news": sum(1 for r in context_records if r.context_family == "rss_news"),
            "reputation_scam": sum(1 for r in context_records if r.context_family == "reputation_scam"),
            "semantic": sum(1 for r in context_records if r.context_family == "semantic"),
            "resolver_links": len(resolver_links),
            "unresolved_links": sum(
                1 for l in resolver_links if l.resolver_status in ("IDENTITY_UNRESOLVED", "RESOLVER_AMBIGUOUS")
            ),
            "symbol_only_rejections": no_sym.get("symbol_only_rejection_count", 0),
            "missingness_records": len(missingness_records),
            "rpc_calls_attempted": rpc_stats.get("rpc_calls_attempted", 0),
            "rpc_calls_successful": rpc_stats.get("rpc_calls_successful", 0),
            "rpc_calls_failed": rpc_stats.get("rpc_calls_failed", 0),
            "rpc_calls_skipped_by_cache": rpc_stats.get("rpc_calls_skipped_by_cache", 0),
            "raw_rpc_payloads_saved": len(raw_log),
            "account_info_success": account_success,
            "signature_fetch_success": sig_success,
            "transaction_fetch_success": tx_success,
            "real_onchain_context_extracted": context_extracted_count,
            "wallet_whale_available": wallet_available_count,
            "wallet_whale_missing": wallet_missing_count,
            "flow_pressure_extracted": flow_extracted,
            "flow_pressure_unknown": flow_unknown,
        },
        "throttle_stats": rpc_stats,
        "audit_results": {
            "no_symbol_only_join": no_sym.get("passed"),
            "whale_score_separation": whale.get("passed"),
            "wallet_whale_evidence": wallet_whale.get("passed"),
            "missingness_provenance": miss.get("passed"),
            "helius_solana_readonly": readonly.get("passed"),
            "authority_safety": authority.get("passed"),
            "preflight_safety": preflight.get("preflight_passed"),
        },
    }

    manifest = {
        "phase": PHASE,
        "output_root": str(out.relative_to(root)).replace("\\", "/") if out.is_relative_to(root) else str(out),
        "created_at_utc": utc_now(),
        "files": _list_output_files(out),
        "classification": classification,
    }
    summary = _build_summary_text(gate, classification)
    write_json(reports_dir / "ae18_decision_gate.json", gate)
    write_json(reports_dir / "ae18_manifest.json", manifest)
    write_text(reports_dir / "ae18_summary_for_upload.txt", summary)
    write_json(root / "reports" / "ae18_decision_gate.json", gate)

    return {
        "classification": classification,
        "output_root": str(out),
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "solana_candidate_count": len(selected),
        "context_record_count": len(context_records),
        "helius_solana_count": gate["counts"]["helius_solana"],
        "rss_news_count": gate["counts"]["rss_news"],
        "reputation_scam_count": gate["counts"]["reputation_scam"],
        "semantic_count": gate["counts"]["semantic"],
        "resolver_link_count": len(resolver_links),
        "unresolved_link_count": gate["counts"]["unresolved_links"],
        "symbol_only_rejection_count": no_sym.get("symbol_only_rejection_count", 0),
        "rpc_stats": rpc_stats,
        "raw_payloads_saved": len(raw_log),
        "context_extracted_count": context_extracted_count,
        "wallet_whale_available": wallet_available_count,
        "wallet_whale_missing": wallet_missing_count,
        "preflight_passed": bool(preflight.get("preflight_passed")),
        "helius_configured": bool(rpc_config.get("helius_configured")),
        "rpc_provider_used": rpc_config.get("rpc_provider_used"),
        "gate": gate,
    }


def _write_blocked(root: Path, out: Path, discovery: dict[str, Any], classification: str) -> dict[str, Any]:
    reports = out / "reports"
    audits = out / "audits"
    for d in (reports, audits):
        d.mkdir(parents=True, exist_ok=True)
    gate = {
        "phase": PHASE,
        "classification": classification,
        "created_at_utc": utc_now(),
        "ae18_status": "OPEN",
        "ae19_status": "BLOCKED",
        **SAFETY_BOUNDARY,
        "discovery": discovery,
    }
    write_json(reports / "ae18_decision_gate.json", gate)
    write_json(audits / "ae18_input_lineage_audit.json", {"passed": False, "discovery": discovery})
    return {"classification": classification, "output_root": str(out), "context_record_count": 0}


def _build_candidate_summaries(
    candidates: list,
    records: list[AE18ContextRecord],
    links: list[AE18ResolverLink],
) -> list[dict[str, Any]]:
    by_cand: dict[str, list[AE18ContextRecord]] = {}
    for r in records:
        by_cand.setdefault(r.clean_forward_candidate_id, []).append(r)
    link_by_cand = Counter(l.clean_forward_candidate_id for l in links if l.clean_forward_candidate_id)
    rows = []
    for c in candidates:
        recs = by_cand.get(c.clean_forward_candidate_id, [])
        rows.append(
            {
                "clean_forward_candidate_id": c.clean_forward_candidate_id,
                "price_source_key": c.price_source_key,
                "chain": c.chain,
                "pair_address": c.pair_address,
                "context_families_present": ",".join(sorted({r.context_family for r in recs})),
                "context_available_count": sum(1 for r in recs if r.available),
                "context_unavailable_count": sum(1 for r in recs if not r.available),
                "resolver_link_count": link_by_cand.get(c.clean_forward_candidate_id, 0),
                "no_trade_authority": True,
            }
        )
    return rows


def _build_missingness_summary(missingness: list[AE18MissingnessRecord]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for m in missingness:
        counter[m.missingness_reason] += 1
    return [{"missingness_reason": k, "count": v} for k, v in sorted(counter.items())]


def _list_output_files(out: Path) -> list[str]:
    return [str(p.relative_to(out)).replace("\\", "/") for p in sorted(out.rglob("*")) if p.is_file()]


def _build_summary_text(gate: dict[str, Any], classification: str) -> str:
    c = gate.get("counts", {})
    return "\n".join(
        [
            f"AE18 Context Intelligence Layer — {classification}",
            f"Created: {gate.get('created_at_utc')}",
            f"AE18 status: OPEN",
            f"Selected candidates: {c.get('candidates_selected', 0)}",
            f"Solana selected: {c.get('solana_candidates_selected', 0)}",
            f"Candidates fetched: {c.get('candidates_fetched', 0)}",
            f"Context records: {c.get('context_records', 0)}",
            f"RPC attempted/successful/failed: {c.get('rpc_calls_attempted')}/{c.get('rpc_calls_successful')}/{c.get('rpc_calls_failed')}",
            f"Raw RPC payloads saved: {c.get('raw_rpc_payloads_saved', 0)}",
            f"Real on-chain context extracted: {c.get('real_onchain_context_extracted', 0)}",
            f"Wallet whale available/missing: {c.get('wallet_whale_available')}/{c.get('wallet_whale_missing')}",
            f"Flow pressure extracted/unknown: {c.get('flow_pressure_extracted')}/{c.get('flow_pressure_unknown')}",
            f"Symbol-only rejections: {c.get('symbol_only_rejections', 0)}",
            "",
            "AE18 now proves:",
            "- Candidate-centered read-only Helius/Solana context fetch path",
            "- Pre-flight wallet/private-key/signer fail-closed safety",
            "- Allowlisted RPC client with throttle/cache/429 handling",
            "- Explicit whale_score POOL_FLOW_PROXY separation from wallet evidence",
            "- No symbol-only joins",
            "",
            "AE18 still does not prove:",
            "- Profitability or live readiness",
            "- Complete RSS/reputation coverage",
            "- Unambiguous flow pressure for all candidates",
            "",
            f"AE19 status: {gate.get('ae19_status', 'BLOCKED')}",
        ]
    )
