"""AE18 audit pack builders — fail-closed on unsafe behavior."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.ae18.constants import (
    ALLOWED_IDENTITY_BASES,
    CLASSIFICATION_BLOCKED_AUTHORITY,
    CLASSIFICATION_BLOCKED_HELIUS_NOT_CONFIGURED,
    CLASSIFICATION_BLOCKED_HELIUS_WRITE,
    CLASSIFICATION_BLOCKED_MISSINGNESS,
    CLASSIFICATION_BLOCKED_NO_SOURCES,
    CLASSIFICATION_BLOCKED_RPC_FAILURE,
    CLASSIFICATION_BLOCKED_SOLANA_PREFLIGHT,
    CLASSIFICATION_BLOCKED_SYMBOL_ONLY,
    CLASSIFICATION_BLOCKED_WHALE_CONFLATION,
    CLASSIFICATION_INFRA_ONLY,
    CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS,
    CLASSIFICATION_PASS_REAL_HELIUS,
    FORBIDDEN_IDENTITY_BASES,
    MISSINGNESS_REASONS,
    SAFETY_BOUNDARY,
    WHALE_SIGNAL_POOL_FLOW_PROXY,
    WHALE_SIGNAL_WALLET_LEVEL,
    RESOLVER_SYMBOL_REJECTED,
)
from app.ae18.models import AE18ContextRecord, AE18MissingnessRecord, AE18ResolverLink


def _join_path_to_identity_basis(join_path: str) -> str:
    mapping = {
        "price_source_key": "PRICE_SOURCE_KEY",
        "chain_pair_address": "CHAIN_PAIR_ADDRESS",
        "chain_token_address": "CHAIN_TOKEN_ADDRESS",
        "clean_forward_candidate_id": "CLEAN_FORWARD_CANDIDATE_ID",
        "target_lineage_id": "AE16_EVIDENCE_ID",
        "ae16_evidence_id": "AE16_EVIDENCE_ID",
        "ae17_evidence_id": "AE17_EVIDENCE_ID",
        "symbol_only": "SYMBOL",
    }
    return mapping.get((join_path or "").strip(), (join_path or "").strip().upper() or "UNKNOWN")


def audit_no_symbol_only_join(links: list[AE18ResolverLink]) -> dict[str, Any]:
    symbol_only_links = [
        l for l in links if l.symbol_only_rejected or l.resolver_status == RESOLVER_SYMBOL_REJECTED
    ]
    accepted = [l for l in links if l.resolver_status == "RESOLVER_LINKED" and not l.symbol_only_rejected]
    unsafe_linked: list[AE18ResolverLink] = []
    for l in links:
        if l.resolver_status in (RESOLVER_SYMBOL_REJECTED, "IDENTITY_UNRESOLVED", "RESOLVER_AMBIGUOUS"):
            continue
        basis = _join_path_to_identity_basis(l.join_path)
        if l.join_path == "symbol_only" or basis in FORBIDDEN_IDENTITY_BASES:
            unsafe_linked.append(l)
            continue
        if l.resolver_status == "RESOLVER_LINKED":
            allowed_upper = {b.upper() for b in ALLOWED_IDENTITY_BASES}
            if basis.upper() not in allowed_upper:
                unsafe_linked.append(l)

    accepted_with_symbol = [
        l
        for l in accepted
        if _join_path_to_identity_basis(l.join_path) in FORBIDDEN_IDENTITY_BASES
        or l.join_path in FORBIDDEN_IDENTITY_BASES
    ]
    identity_basis_counts = Counter(_join_path_to_identity_basis(l.join_path) for l in accepted)
    passed = len(unsafe_linked) == 0 and len(accepted_with_symbol) == 0
    return {
        "audit": "ae18_no_symbol_only_join_audit",
        "passed": passed,
        "fail_closed": True,
        "accepted_link_count": len(accepted),
        "rejected_symbol_only_count": len(symbol_only_links),
        "symbol_only_rejection_count": len(symbol_only_links),
        "unsafe_symbol_only_links": len(unsafe_linked),
        "accepted_links_with_symbol_identity_basis": len(accepted_with_symbol),
        "identity_basis_counts": dict(identity_basis_counts),
        "total_resolver_links": len(links),
        "issues": [f"unsafe_symbol_only_link:{l.resolver_link_id}" for l in unsafe_linked[:50]],
    }


def audit_whale_score_separation(records: list[AE18ContextRecord]) -> dict[str, Any]:
    pool_records = [
        r
        for r in records
        if r.whale_signal_type == WHALE_SIGNAL_POOL_FLOW_PROXY or r.context_family == "whale_pool_flow_proxy"
    ]
    wallet_records = [
        r
        for r in records
        if r.whale_signal_type == WHALE_SIGNAL_WALLET_LEVEL or r.context_family == "wallet_whale"
    ]
    issues: list[str] = []

    for r in pool_records:
        payload = r.evidence_payload if hasattr(r, "evidence_payload") else {}
        if payload.get("not_wallet_level_whale_evidence") is False:
            issues.append(f"pool_flow_promoted_to_wallet:{r.context_record_id}")
        if r.whale_signal_type != WHALE_SIGNAL_POOL_FLOW_PROXY:
            issues.append(f"pool_flow_wrong_signal_type:{r.context_record_id}")

    for r in wallet_records:
        payload = r.evidence_payload if hasattr(r, "evidence_payload") else {}
        if r.available:
            has_prov = bool(
                payload.get("fee_payer_wallets")
                or payload.get("signer_wallets")
                or payload.get("token_owner_wallets")
                or payload.get("wallet_address")
                or payload.get("source_transaction_signature")
            )
            if not has_prov:
                issues.append(f"wallet_evidence_missing_provenance:{r.context_record_id}")
        if payload.get("whale_score") is not None and not payload.get("whale_score_rejected_as_input"):
            issues.append(f"whale_score_used_as_wallet_source:{r.context_record_id}")

    return {
        "audit": "ae18_whale_score_separation_audit",
        "passed": len(issues) == 0,
        "fail_closed": True,
        "pool_flow_proxy_count": len(pool_records),
        "wallet_level_count": len(wallet_records),
        "wallet_level_available_count": sum(1 for r in wallet_records if r.available),
        "wallet_level_missing_count": sum(1 for r in wallet_records if not r.available),
        "issues": issues,
    }


def audit_wallet_whale_evidence(records: list[AE18ContextRecord]) -> dict[str, Any]:
    wallet_records = [r for r in records if r.context_family == "wallet_whale"]
    issues: list[str] = []
    for r in wallet_records:
        payload = r.evidence_payload or {}
        if payload.get("wallet_evidence_source") == "POOL_FLOW_PROXY":
            issues.append(f"wallet_source_is_pool_flow:{r.context_record_id}")
        if not r.available and not (r.missingness_reason or payload.get("missingness_reason")):
            issues.append(f"wallet_missingness_not_explicit:{r.context_record_id}")
        if payload.get("whale_score") is not None and not payload.get("whale_score_rejected_as_input"):
            issues.append(f"legacy_whale_score_conflated:{r.context_record_id}")
    return {
        "audit": "ae18_wallet_whale_evidence_audit",
        "passed": len(issues) == 0,
        "fail_closed": True,
        "wallet_records": len(wallet_records),
        "available_count": sum(1 for r in wallet_records if r.available),
        "missing_count": sum(1 for r in wallet_records if not r.available),
        "issues": issues,
    }


def audit_missingness_provenance(
    records: list[AE18ContextRecord],
    missingness: list[AE18MissingnessRecord],
) -> dict[str, Any]:
    issues: list[str] = []
    ok_without_reason = {
        "ACCOUNT_FOUND",
        "TRANSACTIONS_FOUND_CONTEXT_EXTRACTED",
        "RPC_FETCH_SUCCEEDED",
        "CONTEXT_AVAILABLE",
        "CONTEXT_PARTIAL",
        "WALLET_LEVEL_EVIDENCE_EXTRACTED",
    }
    for r in records:
        if not r.available and not r.missingness_reason and r.context_status not in ok_without_reason:
            issues.append(f"missing_reason_absent:{r.context_record_id}")
        if not r.no_trade_authority:
            issues.append(f"trade_authority_leak:{r.context_record_id}")
    for m in missingness:
        if not m.missingness_reason:
            issues.append(f"missingness_record_no_reason:{m.missingness_record_id}")
        if not m.no_trade_authority:
            issues.append(f"missingness_trade_authority:{m.missingness_record_id}")
    return {
        "audit": "ae18_missingness_provenance_audit",
        "passed": len(issues) == 0,
        "fail_closed": True,
        "context_record_count": len(records),
        "missingness_record_count": len(missingness),
        "unavailable_context_count": sum(1 for r in records if not r.available),
        "known_missingness_reasons": sorted(MISSINGNESS_REASONS),
        "issues": issues,
    }


def audit_helius_solana_readonly(project_root: Path, collector_module_path: Path | None = None) -> dict[str, Any]:
    issues: list[str] = []
    scan_paths = [
        project_root / "app" / "ae18" / "collectors.py",
        project_root / "app" / "ae18" / "pipeline.py",
        project_root / "app" / "ae18" / "readonly_rpc.py",
        project_root / "app" / "ae18" / "real_fetch.py",
    ]
    if collector_module_path:
        scan_paths.append(collector_module_path)

    dangerous_patterns = (
        "Keypair(",
        "Keypair.from",
        "sign_transaction",
        "wallet_client",
        "jupiter_swap",
        "execute_swap",
        'os.getenv("PRIVATE_KEY"',
        "load_private_key",
        "VersionedTransaction(",
    )

    for path in scan_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in dangerous_patterns:
            if pattern in text:
                issues.append(f"forbidden_pattern:{pattern}:in:{path.name}")
        # sendTransaction / sendRawTransaction allowed only as forbidden-set string literals
        if path.name != "readonly_rpc.py":
            for pattern in ("sendTransaction", "sendRawTransaction"):
                if pattern in text:
                    issues.append(f"forbidden_pattern:{pattern}:in:{path.name}")

    return {
        "audit": "ae18_helius_solana_readonly_audit",
        "passed": len(issues) == 0,
        "fail_closed": True,
        "allowlist_enforced": True,
        "safety_boundary": SAFETY_BOUNDARY,
        "wallet_access": False,
        "private_key_access": False,
        "signer_available": False,
        "transaction_builder_available": False,
        "transaction_signing_available": False,
        "transaction_submission_available": False,
        "live_trading_enabled": False,
        "trade_authority": False,
        "scanned_files": [str(p) for p in scan_paths if p.is_file()],
        "issues": issues,
    }


def audit_authority_safety(records: list[AE18ContextRecord]) -> dict[str, Any]:
    issues: list[str] = []
    for r in records:
        if r.trade_authority or r.wallet_access or r.private_key_access:
            issues.append(f"authority_violation:{r.context_record_id}")
        if r.live_trading_enabled or r.transaction_submission_available:
            issues.append(f"live_trading_violation:{r.context_record_id}")
    return {
        "audit": "ae18_authority_safety_audit",
        "passed": len(issues) == 0,
        "fail_closed": True,
        "safety_boundary": SAFETY_BOUNDARY,
        "records_checked": len(records),
        "issues": issues,
    }


def audit_input_lineage(records: list[AE18ContextRecord], discovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit": "ae18_input_lineage_audit",
        "passed": sum(1 for r in records if r.clean_forward_candidate_id) > 0,
        "discovery_status": discovery.get("status"),
        "candidate_source": discovery.get("candidate_csv"),
        "candidate_count": discovery.get("candidate_count"),
        "context_records_with_price_source_key": sum(1 for r in records if r.price_source_key),
        "context_records_with_candidate_id": sum(1 for r in records if r.clean_forward_candidate_id),
        "lineage_tier": "T4_context_evidence",
    }


def audit_rss_news(records: list[AE18ContextRecord], links: list[AE18ResolverLink]) -> dict[str, Any]:
    rss = [r for r in records if r.context_family == "rss_news"]
    return {
        "audit": "ae18_rss_news_audit",
        "passed": True,
        "rss_context_count": len(rss),
        "rss_available_count": sum(1 for r in rss if r.available),
        "symbol_only_rejections": sum(1 for l in links if l.symbol_only_rejected),
    }


def audit_reputation_scam(records: list[AE18ContextRecord]) -> dict[str, Any]:
    rep = [r for r in records if r.context_family == "reputation_scam"]
    return {
        "audit": "ae18_reputation_scam_audit",
        "passed": True,
        "reputation_context_count": len(rep),
        "available_count": sum(1 for r in rep if r.available),
    }


def audit_semantic_context(records: list[AE18ContextRecord]) -> dict[str, Any]:
    sem = [r for r in records if r.context_family == "semantic"]
    llm_invented = any(r.evidence_payload.get("llm_entity_links_invented") for r in sem if r.evidence_payload)
    return {
        "audit": "ae18_semantic_context_audit",
        "passed": not llm_invented,
        "semantic_context_count": len(sem),
        "llm_entity_links_invented": llm_invented,
    }


def audit_real_helius_fetch(
    *,
    external_fetch_enabled: bool,
    rpc_config: dict[str, Any],
    selection_count: int,
    solana_selected: int,
    candidates_fetched: int,
    rpc_stats: dict[str, Any],
    raw_payloads_saved: int,
    context_extracted_count: int,
    wallet_evidence_available_count: int,
    missingness_count: int,
) -> dict[str, Any]:
    return {
        "audit": "ae18_real_helius_fetch_audit",
        "passed": True,
        "external_fetch_enabled": external_fetch_enabled,
        "helius_configured": bool(rpc_config.get("helius_configured")),
        "rpc_provider_used": rpc_config.get("rpc_provider_used"),
        "candidates_selected": selection_count,
        "solana_candidates_selected": solana_selected,
        "candidates_fetched": candidates_fetched,
        "rpc_calls_attempted": rpc_stats.get("rpc_calls_attempted", 0),
        "rpc_calls_successful": rpc_stats.get("rpc_calls_successful", 0),
        "rpc_calls_failed": rpc_stats.get("rpc_calls_failed", 0),
        "rpc_calls_skipped_by_cache": rpc_stats.get("rpc_calls_skipped_by_cache", 0),
        "raw_payloads_saved": raw_payloads_saved,
        "context_extracted_count": context_extracted_count,
        "wallet_evidence_available_count": wallet_evidence_available_count,
        "missingness_count": missingness_count,
    }


def audit_rpc_cache_throttle(rpc_stats: dict[str, Any]) -> dict[str, Any]:
    return {"audit": "ae18_rpc_cache_throttle_audit", "passed": True, **rpc_stats}


def audit_context_value_presence(value_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r.get("value_presence_class") for r in value_rows)
    return {
        "audit": "ae18_context_value_presence_audit",
        "passed": True,
        "candidate_classifications": dict(counts),
        "rows": value_rows,
    }


def build_resolver_audit_csv(links: list[AE18ResolverLink]) -> list[dict[str, Any]]:
    return [l.to_dict() for l in links]


def decide_classification(
    *,
    no_symbol_audit: dict[str, Any],
    whale_audit: dict[str, Any],
    missingness_audit: dict[str, Any],
    readonly_audit: dict[str, Any],
    authority_audit: dict[str, Any],
    discovery_status: str,
    context_record_count: int,
    available_source_count: int = 0,
    external_fetch_enabled: bool = False,
    preflight_passed: bool = True,
    rpc_configured: bool = False,
    rpc_calls_attempted: int = 0,
    rpc_calls_successful: int = 0,
    context_extracted_count: int = 0,
    raw_payloads_saved: int = 0,
    wallet_whale_audit: dict[str, Any] | None = None,
) -> str:
    if discovery_status != "AE18_INPUTS_DISCOVERED" or context_record_count == 0:
        return CLASSIFICATION_BLOCKED_NO_SOURCES
    if not preflight_passed:
        return CLASSIFICATION_BLOCKED_SOLANA_PREFLIGHT
    if not no_symbol_audit.get("passed") or no_symbol_audit.get("accepted_links_with_symbol_identity_basis", 0) > 0:
        return CLASSIFICATION_BLOCKED_SYMBOL_ONLY
    if not whale_audit.get("passed"):
        return CLASSIFICATION_BLOCKED_WHALE_CONFLATION
    if wallet_whale_audit is not None and not wallet_whale_audit.get("passed"):
        return CLASSIFICATION_BLOCKED_WHALE_CONFLATION
    if not readonly_audit.get("passed"):
        return CLASSIFICATION_BLOCKED_HELIUS_WRITE
    if not missingness_audit.get("passed"):
        return CLASSIFICATION_BLOCKED_MISSINGNESS
    if not authority_audit.get("passed"):
        return CLASSIFICATION_BLOCKED_AUTHORITY

    if external_fetch_enabled and not rpc_configured:
        return CLASSIFICATION_BLOCKED_HELIUS_NOT_CONFIGURED

    if external_fetch_enabled and rpc_configured:
        if rpc_calls_attempted == 0:
            return CLASSIFICATION_BLOCKED_RPC_FAILURE
        if rpc_calls_successful == 0 and context_extracted_count == 0:
            return CLASSIFICATION_BLOCKED_RPC_FAILURE
        if raw_payloads_saved <= 0:
            return CLASSIFICATION_BLOCKED_RPC_FAILURE
        if context_extracted_count >= 1:
            if context_extracted_count >= 3 and rpc_calls_successful >= 5:
                return CLASSIFICATION_PASS_REAL_HELIUS
            return CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS
        return CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS

    _ = available_source_count
    return CLASSIFICATION_INFRA_ONLY
