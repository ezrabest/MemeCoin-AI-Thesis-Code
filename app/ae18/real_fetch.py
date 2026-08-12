"""AE18 real Helius/Solana read-only fetch orchestration for selected candidates."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from app.ae18.collectors import _base_record, _missingness
from app.ae18.constants import WHALE_SIGNAL_POOL_FLOW_PROXY, WHALE_SIGNAL_WALLET_LEVEL
from app.ae18.models import AE18CandidateTarget, AE18ContextRecord, AE18MissingnessRecord
from app.ae18.onchain_extract import (
    build_wallet_whale_evidence,
    classify_flow_pressure,
    extract_account_context,
    extract_signature_list,
    extract_wallet_behavior,
    summarize_transaction_fetch,
)
from app.ae18.readonly_rpc import AE18ReadOnlyRpcClient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_candidate_onchain_context(
    candidate: AE18CandidateTarget,
    client: AE18ReadOnlyRpcClient,
    *,
    signatures_per_pair: int = 10,
    transactions_per_pair: int = 10,
) -> dict[str, Any]:
    """Fetch account/signatures/transactions and extract context for one candidate."""
    cid = candidate.clean_forward_candidate_id
    psk = candidate.price_source_key
    pair = candidate.pair_address
    meta = {
        "clean_forward_candidate_id": cid,
        "price_source_key": psk,
        "chain": candidate.chain or "solana",
        "pair_address": pair,
    }

    value_class = "RPC_UNAVAILABLE"
    methods_used: list[str] = []
    provenance_hashes: list[str] = []
    rpc_errors = 0
    rate_limits = 0

    # 1) getAccountInfo
    acct_resp = client.call(
        "getAccountInfo",
        [pair, {"encoding": "jsonParsed"}],
        **meta,
    )
    methods_used.append("getAccountInfo")
    if acct_resp.get("response_hash"):
        provenance_hashes.append(str(acct_resp["response_hash"]))
    if acct_resp.get("rate_limit_status") == "RPC_RATE_LIMITED":
        rate_limits += 1
        value_class = "RATE_LIMITED"
    if not acct_resp.get("success"):
        rpc_errors += 1

    account_ctx = extract_account_context(
        pair_address=pair,
        account_info_result=acct_resp.get("result") if acct_resp.get("success") else None,
        response_hash=acct_resp.get("response_hash"),
        source=client.provider_used,
    )
    if acct_resp.get("success"):
        value_class = "REAL_ONCHAIN_CONTEXT_EXTRACTED" if account_ctx["account_found"] else "RPC_REACHABLE_BUT_NO_USEFUL_CONTEXT"

    # 2) getSignaturesForAddress
    sig_resp = client.call(
        "getSignaturesForAddress",
        [pair, {"limit": signatures_per_pair}],
        **meta,
    )
    methods_used.append("getSignaturesForAddress")
    if sig_resp.get("response_hash"):
        provenance_hashes.append(str(sig_resp["response_hash"]))
    if sig_resp.get("rate_limit_status") == "RPC_RATE_LIMITED":
        rate_limits += 1
        value_class = "RATE_LIMITED"
    if not sig_resp.get("success"):
        rpc_errors += 1

    signatures = extract_signature_list(sig_resp.get("result") if sig_resp.get("success") else None)

    # 3) getTransaction for each signature (bounded)
    loaded_txs: list[dict[str, Any]] = []
    tx_hashes: list[str] = []
    tx_failed = 0
    to_fetch = signatures[:transactions_per_pair]
    for sig_item in to_fetch:
        if client.budget_exhausted:
            break
        sig = sig_item["signature"]
        tx_resp = client.call(
            "getTransaction",
            [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            **meta,
        )
        methods_used.append("getTransaction")
        if tx_resp.get("response_hash"):
            tx_hashes.append(str(tx_resp["response_hash"]))
            provenance_hashes.append(str(tx_resp["response_hash"]))
        if tx_resp.get("rate_limit_status") == "RPC_RATE_LIMITED":
            rate_limits += 1
        if tx_resp.get("success") and isinstance(tx_resp.get("result"), dict):
            loaded_txs.append(tx_resp["result"])
        else:
            tx_failed += 1
            if not tx_resp.get("success"):
                rpc_errors += 1

    tx_summary = summarize_transaction_fetch(
        signatures_requested=signatures_per_pair,
        signatures=signatures,
        transactions_requested=len(to_fetch),
        loaded_txs=loaded_txs,
        failed_count=tx_failed,
        rpc_error_count=rpc_errors,
        rate_limit_count=rate_limits,
        response_hashes=tx_hashes,
    )

    wallet_behavior = extract_wallet_behavior(loaded_txs)
    flow = classify_flow_pressure(
        loaded_txs,
        base_token_address=candidate.base_token_address,
        quote_token_address=candidate.quote_token_address,
    )
    wallet_evidence = build_wallet_whale_evidence(
        wallet_behavior=wallet_behavior,
        flow=flow,
        signatures=[s["signature"] for s in signatures],
        rpc_methods=sorted(set(methods_used)),
        provenance_hashes=provenance_hashes,
        provider_used=client.provider_used,
        whale_score=candidate.whale_score,  # explicitly rejected inside builder
    )

    if rate_limits and not account_ctx["account_found"] and not loaded_txs:
        value_class = "RATE_LIMITED"
    elif rpc_errors and not account_ctx["account_found"] and not signatures and not loaded_txs:
        value_class = "ERROR"
    elif account_ctx["account_found"] or loaded_txs or signatures:
        value_class = "REAL_ONCHAIN_CONTEXT_EXTRACTED"
    elif acct_resp.get("success") or sig_resp.get("success"):
        value_class = "RPC_REACHABLE_BUT_NO_USEFUL_CONTEXT"

    # Context status labels
    if rate_limits and value_class == "RATE_LIMITED":
        context_status = "RPC_RATE_LIMITED"
        missingness_reason = "RPC_RATE_LIMITED"
    elif account_ctx["account_found"] and loaded_txs:
        context_status = "TRANSACTIONS_FOUND_CONTEXT_EXTRACTED"
        missingness_reason = ""
    elif account_ctx["account_found"]:
        context_status = "ACCOUNT_FOUND"
        missingness_reason = "TRANSACTIONS_NOT_FOUND" if not signatures else ""
    elif acct_resp.get("success"):
        context_status = "ACCOUNT_NOT_FOUND"
        missingness_reason = "SOURCE_EMPTY_RESPONSE"
    elif not acct_resp.get("success") and not sig_resp.get("success"):
        context_status = "RPC_FETCH_FAILED"
        missingness_reason = "SOURCE_UNAVAILABLE_PENDING_FETCH"
    else:
        context_status = "RPC_FETCH_ATTEMPTED"
        missingness_reason = "SOURCE_EMPTY_RESPONSE"

    helius_record = _base_record(
        candidate,
        "helius_solana",
        context_status=context_status,
        source_name="helius_solana",
        source_type="solana_rpc_readonly",
        attempted=True,
        available=bool(account_ctx["account_found"] or loaded_txs or signatures),
        missingness_reason=missingness_reason,
        provenance_status="RPC_FETCH_SUCCEEDED" if (acct_resp.get("success") or sig_resp.get("success")) else "RPC_FETCH_FAILED",
        resolver_status="RESOLVER_LINKED",
        resolver_join_path="price_source_key" if candidate.price_source_key else "chain_pair_address",
        resolver_confidence=1.0,
        evidence_payload={
            "account": account_ctx,
            "transaction_summary": tx_summary,
            "wallet_behavior": wallet_behavior,
            "flow_pressure": flow,
            "value_presence_class": value_class,
            "rpc_provider_used": client.provider_used,
            "read_only_enforced": True,
        },
    )

    miss = None
    if missingness_reason:
        miss = _missingness(
            candidate,
            source_name="helius_solana",
            source_type="solana_rpc_readonly",
            reason=missingness_reason,
            attempted=True,
        )

    # Wallet whale context record
    wallet_available = bool(wallet_evidence.get("wallet_evidence_available"))
    wallet_rec = _base_record(
        candidate,
        "wallet_whale",
        context_status="WALLET_LEVEL_EVIDENCE_EXTRACTED" if wallet_available else "WALLET_LEVEL_DATA_NOT_AVAILABLE",
        source_name="helius_solana",
        source_type="wallet_level_readonly",
        attempted=True,
        available=wallet_available,
        missingness_reason="" if wallet_available else "WALLET_LEVEL_DATA_NOT_AVAILABLE",
        provenance_status="WALLET_LEVEL_READONLY_PROVENANCE" if wallet_available else "WALLET_LEVEL_MISSINGNESS_EMITTED",
        whale_signal_type=WHALE_SIGNAL_WALLET_LEVEL,
        evidence_payload=wallet_evidence,
    )

    # Legacy whale_score remains POOL_FLOW_PROXY only
    score = candidate.whale_score
    pool_available = score is not None and str(score).strip() != ""
    pool_rec = _base_record(
        candidate,
        "whale_pool_flow_proxy",
        context_status="CONTEXT_AVAILABLE" if pool_available else "CONTEXT_UNAVAILABLE",
        source_name="legacy_whale_score",
        source_type="pool_flow_proxy",
        attempted=True,
        available=pool_available,
        whale_signal_type=WHALE_SIGNAL_POOL_FLOW_PROXY,
        missingness_reason="" if pool_available else "SOURCE_EMPTY_RESPONSE",
        provenance_status="POOL_FLOW_PROXY_NOT_WALLET_EVIDENCE",
        evidence_payload={
            "whale_score": score,
            "whale_signal_type": WHALE_SIGNAL_POOL_FLOW_PROXY,
            "not_wallet_level_whale_evidence": True,
            "not_on_chain_whale_evidence": True,
        },
    )

    tx_summary_row = {
        "clean_forward_candidate_id": cid,
        "price_source_key": psk,
        "chain": candidate.chain,
        "pair_address": pair,
        "base_token_address": candidate.base_token_address,
        "quote_token_address": candidate.quote_token_address,
        **account_ctx,
        **{k: v for k, v in tx_summary.items() if k != "transaction_response_hashes"},
        "transaction_response_hashes": "|".join(tx_summary.get("transaction_response_hashes") or []),
        "unique_fee_payers_count": len(wallet_behavior.get("unique_fee_payers") or []),
        "unique_signers_count": len(wallet_behavior.get("unique_signers") or []),
        "top_fee_payer_count": wallet_behavior.get("top_fee_payer_count"),
        "fee_payer_concentration_share": wallet_behavior.get("fee_payer_concentration_share"),
        "flow_pressure_direction": flow.get("flow_pressure_direction"),
        "flow_pressure_confidence": flow.get("flow_pressure_confidence"),
        "flow_pressure_reason": flow.get("flow_pressure_reason"),
        "buy_like_flow_count": flow.get("buy_like_flow_count"),
        "sell_like_flow_count": flow.get("sell_like_flow_count"),
        "value_presence_class": value_class,
        "wallet_evidence_available": wallet_available,
        "rpc_provider_used": client.provider_used,
    }

    helius_row = {
        **helius_record.to_dict(),
        "account_found": account_ctx["account_found"],
        "account_status": account_ctx["account_status"],
        "signatures_found": tx_summary["signatures_found"],
        "transactions_loaded": tx_summary["transactions_loaded"],
        "flow_pressure_direction": flow.get("flow_pressure_direction"),
        "value_presence_class": value_class,
        "rpc_provider_used": client.provider_used,
    }

    wallet_row = {
        **wallet_rec.to_dict(),
        **{k: wallet_evidence.get(k) for k in (
            "wallet_evidence_available",
            "wallet_evidence_source",
            "whale_like_wallet_activity_detected",
            "whale_evidence_confidence",
            "whale_evidence_reason",
            "repeated_wallet_activity_detected",
        )},
        "fee_payer_wallets_count": len(wallet_evidence.get("fee_payer_wallets") or []),
        "signer_wallets_count": len(wallet_evidence.get("signer_wallets") or []),
        "token_owner_wallets_count": len(wallet_evidence.get("token_owner_wallets") or []),
    }

    return {
        "helius_record": helius_record,
        "wallet_record": wallet_rec,
        "pool_record": pool_rec,
        "missingness": miss,
        "helius_row": helius_row,
        "wallet_row": wallet_row,
        "tx_summary_row": tx_summary_row,
        "value_presence_class": value_class,
        "wallet_evidence": wallet_evidence,
        "flow": flow,
        "rpc_attempted": True,
    }
