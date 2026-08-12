"""On-chain context extraction: accounts, transactions, flow pressure, wallet whale evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.ae18.constants import WHALE_SIGNAL_WALLET_LEVEL


def extract_account_context(
    *,
    pair_address: str,
    account_info_result: Any,
    response_hash: str | None,
    source: str,
) -> dict[str, Any]:
    found = False
    owner = None
    executable = None
    lamports = None
    data_present = False
    status = "ACCOUNT_NOT_FOUND"

    value = None
    if isinstance(account_info_result, dict):
        value = account_info_result.get("value")
    if value is not None and isinstance(value, dict):
        found = True
        owner = value.get("owner")
        executable = value.get("executable")
        lamports = value.get("lamports")
        data = value.get("data")
        data_present = data is not None and data != "" and data != []
        status = "ACCOUNT_FOUND"

    return {
        "pair_address": pair_address,
        "account_found": found,
        "account_owner_program": owner,
        "executable": executable,
        "lamports": lamports,
        "account_data_present": data_present,
        "account_status": status,
        "account_info_source": source,
        "account_info_response_hash": response_hash,
    }


def extract_signature_list(signatures_result: Any) -> list[dict[str, Any]]:
    if not isinstance(signatures_result, list):
        return []
    out: list[dict[str, Any]] = []
    for item in signatures_result:
        if not isinstance(item, dict):
            continue
        sig = item.get("signature")
        if not sig:
            continue
        out.append(
            {
                "signature": str(sig),
                "blockTime": item.get("blockTime"),
                "err": item.get("err"),
                "slot": item.get("slot"),
            }
        )
    return out


def map_account_index_to_pubkey(tx: dict[str, Any], account_index: int) -> str | None:
    try:
        message = ((tx.get("transaction") or {}).get("message") or {})
        keys = message.get("accountKeys") or []
        if account_index < 0 or account_index >= len(keys):
            return None
        key = keys[account_index]
        if isinstance(key, dict):
            return str(key.get("pubkey") or "") or None
        return str(key) if key else None
    except (TypeError, AttributeError, KeyError):
        return None


def _token_amount(entry: dict[str, Any]) -> float | None:
    ui = entry.get("uiTokenAmount") or {}
    if ui.get("uiAmount") is not None:
        try:
            return float(ui["uiAmount"])
        except (TypeError, ValueError):
            pass
    if ui.get("amount") is not None and ui.get("decimals") is not None:
        try:
            return float(ui["amount"]) / (10 ** int(ui["decimals"]))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def compute_token_balance_deltas(tx: dict[str, Any]) -> list[dict[str, Any]]:
    meta = tx.get("meta") or {}
    pre = meta.get("preTokenBalances") or []
    post = meta.get("postTokenBalances") or []
    if not isinstance(pre, list) or not isinstance(post, list):
        return []
    if not pre and not post:
        return []

    pre_map: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in pre:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("accountIndex")
        mint = entry.get("mint")
        if idx is None or not mint:
            continue
        pre_map[(int(idx), str(mint))] = entry

    post_map: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in post:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("accountIndex")
        mint = entry.get("mint")
        if idx is None or not mint:
            continue
        post_map[(int(idx), str(mint))] = entry

    keys = set(pre_map) | set(post_map)
    deltas: list[dict[str, Any]] = []
    for key in keys:
        idx, mint = key
        pre_e = pre_map.get(key)
        post_e = post_map.get(key)
        pre_amt = _token_amount(pre_e) if pre_e else 0.0
        post_amt = _token_amount(post_e) if post_e else 0.0
        if pre_amt is None or post_amt is None:
            continue
        owner = (post_e or pre_e or {}).get("owner")
        pubkey = map_account_index_to_pubkey(tx, idx)
        deltas.append(
            {
                "accountIndex": idx,
                "account_pubkey": pubkey,
                "mint": mint,
                "owner": owner,
                "pre_amount": pre_amt,
                "post_amount": post_amt,
                "delta": post_amt - pre_amt,
            }
        )
    return deltas


def classify_flow_pressure(
    transactions: list[dict[str, Any]],
    *,
    base_token_address: str,
    quote_token_address: str,
) -> dict[str, Any]:
    """Strict flow pressure — default UNKNOWN unless unambiguous."""
    default = {
        "flow_pressure_direction": "UNKNOWN",
        "flow_pressure_confidence": "LOW",
        "flow_pressure_reason": "insufficient_or_ambiguous_token_flow",
        "buy_like_flow_count": 0,
        "sell_like_flow_count": 0,
        "pool_related_token_delta_count": 0,
        "largest_token_delta_abs": None,
        "token_mints_observed": [],
        "token_owner_wallets_observed": [],
        "token_accounts_observed": [],
    }

    base = (base_token_address or "").strip()
    quote = (quote_token_address or "").strip()
    if not base or not quote:
        default["flow_pressure_reason"] = "base_or_quote_token_address_missing"
        return default

    buy = 0
    sell = 0
    pool_deltas = 0
    largest_abs = 0.0
    mints: set[str] = set()
    owners: set[str] = set()
    accounts: set[str] = set()
    ambiguous = False
    reasons: list[str] = []

    for tx in transactions:
        if not isinstance(tx, dict):
            ambiguous = True
            reasons.append("non_dict_transaction")
            continue
        meta = tx.get("meta") or {}
        if meta.get("err"):
            continue
        pre = meta.get("preTokenBalances")
        post = meta.get("postTokenBalances")
        if not isinstance(pre, list) or not isinstance(post, list) or (not pre and not post):
            ambiguous = True
            reasons.append("missing_pre_post_token_balances")
            continue

        deltas = compute_token_balance_deltas(tx)
        if not deltas:
            ambiguous = True
            reasons.append("delta_computation_failed")
            continue

        # Detect multi-mint / multi-hop ambiguity
        tx_mints = {d["mint"] for d in deltas if d.get("mint")}
        mints |= tx_mints
        for d in deltas:
            if d.get("owner"):
                owners.add(str(d["owner"]))
            if d.get("account_pubkey"):
                accounts.add(str(d["account_pubkey"]))
            if d.get("account_pubkey") is None and d.get("accountIndex") is not None:
                ambiguous = True
                reasons.append("accountIndex_pubkey_unmapped")

        relevant = [d for d in deltas if d.get("mint") in {base, quote}]
        if not relevant:
            ambiguous = True
            reasons.append("no_base_quote_mint_deltas")
            continue

        # Extra mints beyond base/quote => multi-hop ambiguity
        if tx_mints - {base, quote}:
            ambiguous = True
            reasons.append("multi_hop_or_extra_mint_detected")
            continue

        base_deltas = [d["delta"] for d in relevant if d.get("mint") == base]
        quote_deltas = [d["delta"] for d in relevant if d.get("mint") == quote]
        if not base_deltas or not quote_deltas:
            ambiguous = True
            reasons.append("missing_paired_base_quote_delta")
            continue

        # Unambiguous swap-like pattern: one side net increase in base for taker is hard;
        # use pool-relative heuristic carefully: if any owner decreases base and increases quote => sell-like
        # and increases base decreases quote => buy-like — only when exactly one clear pattern.
        patterns: set[str] = set()
        by_owner: dict[str, dict[str, float]] = {}
        for d in relevant:
            owner = str(d.get("owner") or "")
            if not owner:
                ambiguous = True
                reasons.append("token_owner_unknown")
                continue
            by_owner.setdefault(owner, {})
            by_owner[owner][d["mint"]] = by_owner[owner].get(d["mint"], 0.0) + float(d["delta"])
            pool_deltas += 1
            largest_abs = max(largest_abs, abs(float(d["delta"])))

        for owner, mint_delta in by_owner.items():
            b = mint_delta.get(base, 0.0)
            q = mint_delta.get(quote, 0.0)
            if b > 0 and q < 0:
                patterns.add("BUY")
            elif b < 0 and q > 0:
                patterns.add("SELL")
            elif b != 0 or q != 0:
                ambiguous = True
                reasons.append("non_swap_like_delta_pattern")

        if patterns == {"BUY"}:
            buy += 1
        elif patterns == {"SELL"}:
            sell += 1
        elif patterns == {"BUY", "SELL"}:
            ambiguous = True
            reasons.append("contradictory_buy_sell_in_same_tx")
        else:
            ambiguous = True
            reasons.append("no_clear_buy_sell_pattern")

    out = {
        "flow_pressure_direction": "UNKNOWN",
        "flow_pressure_confidence": "LOW",
        "flow_pressure_reason": ";".join(sorted(set(reasons))) or "no_classifiable_transactions",
        "buy_like_flow_count": buy,
        "sell_like_flow_count": sell,
        "pool_related_token_delta_count": pool_deltas,
        "largest_token_delta_abs": largest_abs if pool_deltas else None,
        "token_mints_observed": sorted(mints),
        "token_owner_wallets_observed": sorted(owners),
        "token_accounts_observed": sorted(accounts),
    }

    if ambiguous or (buy > 0 and sell > 0):
        out["flow_pressure_direction"] = "UNKNOWN" if buy == 0 or sell == 0 or buy == sell else "MIXED"
        if buy > 0 and sell > 0 and not ambiguous:
            out["flow_pressure_direction"] = "MIXED"
            out["flow_pressure_confidence"] = "LOW"
            out["flow_pressure_reason"] = "mixed_buy_and_sell_patterns"
        else:
            out["flow_pressure_direction"] = "UNKNOWN"
            out["flow_pressure_confidence"] = "LOW"
        return out

    if buy > 0 and sell == 0 and not ambiguous:
        out["flow_pressure_direction"] = "BUY_PRESSURE"
        out["flow_pressure_confidence"] = "MEDIUM" if buy >= 2 else "LOW"
        out["flow_pressure_reason"] = "unambiguous_base_increase_quote_decrease_patterns"
        return out
    if sell > 0 and buy == 0 and not ambiguous:
        out["flow_pressure_direction"] = "SELL_PRESSURE"
        out["flow_pressure_confidence"] = "MEDIUM" if sell >= 2 else "LOW"
        out["flow_pressure_reason"] = "unambiguous_base_decrease_quote_increase_patterns"
        return out

    return out


def extract_wallet_behavior(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    fee_payers: list[str] = []
    signers: list[str] = []

    for tx in transactions:
        if not isinstance(tx, dict):
            continue
        message = ((tx.get("transaction") or {}).get("message") or {})
        keys = message.get("accountKeys") or []
        if not keys:
            continue
        # fee payer is accountKeys[0]
        first = keys[0]
        fp = first.get("pubkey") if isinstance(first, dict) else first
        if fp:
            fee_payers.append(str(fp))
        for key in keys:
            if isinstance(key, dict):
                if key.get("signer") is True:
                    pk = key.get("pubkey")
                    if pk:
                        signers.append(str(pk))
            # legacy string keys: only first is known fee payer; signer flags unavailable
        # Also signatures length implies signers but not addresses without header flags

    fp_counts = Counter(fee_payers)
    sig_counts = Counter(signers)
    unique_fp = sorted(fp_counts)
    unique_sig = sorted(sig_counts) if sig_counts else sorted(set(fee_payers))
    top_fp = fp_counts.most_common(1)[0][1] if fp_counts else 0
    top_sig = sig_counts.most_common(1)[0][1] if sig_counts else (top_fp if fee_payers else 0)
    total_fp = sum(fp_counts.values()) or 1
    total_sig = sum(sig_counts.values()) or (total_fp if fee_payers else 1)

    repeated_fp = [w for w, n in fp_counts.items() if n >= 2]
    repeated_sig = [w for w, n in sig_counts.items() if n >= 2]

    return {
        "unique_fee_payers": unique_fp,
        "unique_signers": unique_sig,
        "repeated_fee_payers": repeated_fp,
        "repeated_signers": repeated_sig,
        "top_fee_payer_count": top_fp,
        "top_signer_count": top_sig,
        "signer_concentration_share": round(top_sig / total_sig, 6) if signers or fee_payers else None,
        "fee_payer_concentration_share": round(top_fp / total_fp, 6) if fee_payers else None,
    }


def build_wallet_whale_evidence(
    *,
    wallet_behavior: dict[str, Any],
    flow: dict[str, Any],
    signatures: list[str],
    rpc_methods: list[str],
    provenance_hashes: list[str],
    provider_used: str,
    whale_score: Any = None,
) -> dict[str, Any]:
    """Build wallet-level whale evidence. Rejects whale_score as input/source."""
    if whale_score is not None:
        # Explicit rejection — never use as wallet evidence source
        _ = whale_score  # acknowledged and discarded

    fee_payers = list(wallet_behavior.get("unique_fee_payers") or [])
    signers = list(wallet_behavior.get("unique_signers") or [])
    token_owners = list(flow.get("token_owner_wallets_observed") or [])
    has_wallet_data = bool(fee_payers or signers or token_owners)

    if not has_wallet_data:
        return {
            "wallet_evidence_available": False,
            "wallet_evidence_source": "NOT_AVAILABLE",
            "fee_payer_wallets": [],
            "signer_wallets": [],
            "token_owner_wallets": [],
            "repeated_wallet_activity_detected": False,
            "whale_like_wallet_activity_detected": "unknown",
            "whale_evidence_confidence": "LOW",
            "whale_evidence_reason": "WALLET_LEVEL_DATA_NOT_AVAILABLE",
            "missingness_reason": "WALLET_LEVEL_DATA_NOT_AVAILABLE",
            "source_signatures": signatures,
            "source_rpc_methods": rpc_methods,
            "provenance_hashes": provenance_hashes,
            "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
            "whale_score_rejected_as_input": True,
        }

    repeated = bool(wallet_behavior.get("repeated_fee_payers") or wallet_behavior.get("repeated_signers"))
    concentration = wallet_behavior.get("fee_payer_concentration_share") or 0.0
    whale_like: str | bool = "unknown"
    confidence = "LOW"
    reason = "wallet_addresses_observed_insufficient_for_whale_classification"
    if repeated and concentration >= 0.5:
        whale_like = True
        confidence = "MEDIUM"
        reason = "repeated_fee_payer_or_signer_with_high_concentration"
    elif fee_payers or signers:
        whale_like = False
        confidence = "LOW"
        reason = "wallets_observed_without_repeated_concentrated_activity"

    source = "HELIUS_RPC" if "HELIUS" in (provider_used or "").upper() else "SOLANA_RPC"
    return {
        "wallet_evidence_available": True,
        "wallet_evidence_source": source,
        "fee_payer_wallets": fee_payers,
        "signer_wallets": signers,
        "token_owner_wallets": token_owners,
        "repeated_wallet_activity_detected": repeated,
        "whale_like_wallet_activity_detected": whale_like,
        "whale_evidence_confidence": confidence,
        "whale_evidence_reason": reason,
        "missingness_reason": "",
        "source_signatures": signatures,
        "source_rpc_methods": rpc_methods,
        "provenance_hashes": provenance_hashes,
        "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
        "whale_score_rejected_as_input": True,
    }


def summarize_transaction_fetch(
    *,
    signatures_requested: int,
    signatures: list[dict[str, Any]],
    transactions_requested: int,
    loaded_txs: list[dict[str, Any]],
    failed_count: int,
    rpc_error_count: int,
    rate_limit_count: int,
    response_hashes: list[str],
) -> dict[str, Any]:
    times = [s.get("blockTime") for s in signatures if s.get("blockTime") is not None]
    return {
        "signatures_requested": signatures_requested,
        "signatures_found": len(signatures),
        "transactions_requested": transactions_requested,
        "transactions_loaded": len(loaded_txs),
        "transactions_failed": failed_count,
        "newest_signature_time": max(times) if times else None,
        "oldest_signature_time": min(times) if times else None,
        "rpc_error_count": rpc_error_count,
        "rate_limit_count": rate_limit_count,
        "transaction_response_hashes": response_hashes,
    }
