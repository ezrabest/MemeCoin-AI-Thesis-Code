"""
Conservative audit-only wallet behavior summary from parsed pool activity events.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from .solana_pool_activity import (
    OWNER_MATCHED_FEE_PAYER,
    OWNER_MATCHED_SIGNER,
    OWNER_UNMATCHED,
    QUOTE_USDC,
    QUOTE_WSOL,
    SIDE_BUY_BASE,
    SIDE_IGNORED_FAILED,
    SIDE_SELL_BASE,
    SIDE_UNKNOWN,
)

IDENTITY_TOKEN_OWNER_FEE_PAYER = "TOKEN_OWNER_MATCHED_FEE_PAYER"
IDENTITY_TOKEN_OWNER_SIGNER = "TOKEN_OWNER_MATCHED_SIGNER"
IDENTITY_TOKEN_OWNER_UNMATCHED = "TOKEN_OWNER_UNMATCHED_TO_SIGNER"
IDENTITY_SIGNER_FALLBACK = "SIGNER_FALLBACK"
IDENTITY_FEE_PAYER_FALLBACK = "FEE_PAYER_FALLBACK"
IDENTITY_UNKNOWN = "UNKNOWN_TRADER"

AGGREGATOR_PROGRAM_IDS = frozenset(
    {
        "routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS",
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "JUP4Fb2cqiRUcaTHhrPC5G3NMLxTfeLnZsR934gj8ZD",
    }
)

WARN_LOW_SAMPLE = "LOW_SAMPLE_SIZE"
WARN_HIGH_UNKNOWN = "HIGH_UNKNOWN_SIDE_RATE"
WARN_POOL_VAULTS = "POOL_VAULTS_UNCERTAIN"
WARN_HELIUS_MISMATCH = "HELIUS_VALIDATION_MISMATCH"
WARN_FAILED_EXCLUDED = "FAILED_TX_EXCLUDED"
WARN_NOT_SIGNAL = "BEHAVIOR_NOT_TRADING_SIGNAL"
WARN_RELAYER = "FEE_PAYER_RELAYER_SUSPECTED"
WARN_MULTI_SIGNER = "MULTIPLE_SIGNERS_COMPLEX_ROUTE"
WARN_TOKEN_OWNER_USED = "TOKEN_OWNER_IDENTITY_USED"
WARN_TOKEN_OWNER_UNMATCHED = "TOKEN_OWNER_UNMATCHED_TO_SIGNER"

WHALE_USDC_VOLUME_THRESHOLD = Decimal("1000")
WHALE_NATIVE_VOLUME_THRESHOLD = Decimal("10")
MAX_TRADE_WHALE_USDC = Decimal("500")
MAX_TRADE_WHALE_NATIVE = Decimal("5")
MM_BALANCE_RATIO_MAX = Decimal("0.35")
NOISE_USDC_THRESHOLD = Decimal("5")
NOISE_NATIVE_THRESHOLD = Decimal("0.01")
MIN_DIRECTIONAL_FOR_LABEL = 10


def _to_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _decimal_str(value: Decimal) -> str:
    return format(value, "f")


def _suspected_aggregator(parsed: dict[str, Any]) -> bool:
    program_ids = set(parsed.get("program_ids") or [])
    if program_ids & AGGREGATOR_PROGRAM_IDS:
        return True
    source = str(parsed.get("helius_source") or "").upper()
    return source in {"JUPITER", "JUPITER_V6", "ORCA", "RAYDIUM"}


def resolve_trader_identity(parsed: dict[str, Any]) -> dict[str, Any]:
    fee_payer = parsed.get("fee_payer")
    signers = list(parsed.get("signer_wallets") or [])
    token_balances = parsed.get("token_balances") or []

    token_accounts: list[str] = []
    token_owner_wallets: list[str] = []
    matched_fee_payer_owners: list[str] = []
    matched_signer_owners: list[str] = []
    unmatched_owners: list[str] = []

    for row in token_balances:
        token_account = row.get("token_account")
        owner = row.get("token_owner_wallet") or row.get("owner")
        status = row.get("owner_match_status")
        if token_account:
            token_accounts.append(token_account)
        if not owner:
            continue
        if owner not in token_owner_wallets:
            token_owner_wallets.append(owner)
        if status == OWNER_MATCHED_FEE_PAYER:
            matched_fee_payer_owners.append(owner)
        elif status == OWNER_MATCHED_SIGNER:
            matched_signer_owners.append(owner)
        elif status == OWNER_UNMATCHED:
            unmatched_owners.append(owner)

    aggregator = _suspected_aggregator(parsed)
    multiple_signers = len(signers) > 1

    trader_wallet: str | None = None
    identity_source = IDENTITY_UNKNOWN
    confidence = "low"

    if matched_fee_payer_owners:
        trader_wallet = matched_fee_payer_owners[0]
        identity_source = IDENTITY_TOKEN_OWNER_FEE_PAYER
        confidence = "high"
    elif matched_signer_owners:
        trader_wallet = matched_signer_owners[0]
        identity_source = IDENTITY_TOKEN_OWNER_SIGNER
        confidence = "high"
    elif unmatched_owners and len(set(unmatched_owners)) == 1:
        trader_wallet = unmatched_owners[0]
        identity_source = IDENTITY_TOKEN_OWNER_UNMATCHED
        confidence = "medium"
    elif signers and not aggregator:
        trader_wallet = signers[0]
        identity_source = IDENTITY_SIGNER_FALLBACK
        confidence = "medium"
    elif fee_payer and not aggregator:
        trader_wallet = fee_payer
        identity_source = IDENTITY_FEE_PAYER_FALLBACK
        confidence = "low"
    elif fee_payer:
        trader_wallet = fee_payer
        identity_source = IDENTITY_FEE_PAYER_FALLBACK
        confidence = "low"

    relayer_suspected = bool(
        aggregator
        and fee_payer
        and trader_wallet
        and trader_wallet != fee_payer
    )

    return {
        "trader_wallet": trader_wallet,
        "trader_identity_source": identity_source,
        "identity_confidence": confidence,
        "fee_payer": fee_payer,
        "signer_wallets": signers,
        "token_owner_wallets": token_owner_wallets,
        "token_accounts": token_accounts,
        "aggregator_or_router_suspected": aggregator,
        "relayer_suspected": relayer_suspected,
        "multiple_signers": multiple_signers,
    }


def build_behavior_events(parsed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for parsed in parsed_rows:
        identity = resolve_trader_identity(parsed)
        events.append(
            {
                "signature": parsed.get("signature"),
                "block_time": parsed.get("block_time"),
                "slot": parsed.get("slot"),
                "failed_transaction": parsed.get("failed_transaction"),
                "parse_status": parsed.get("parse_status"),
                "side": parsed.get("side"),
                "quote_token_type": parsed.get("quote_token_type"),
                "approx_usd_value": parsed.get("approx_usd_value"),
                "quote_amount_native": parsed.get("quote_amount_native"),
                "base_delta_pool_str": parsed.get("base_delta_pool_str"),
                "quote_delta_pool_str": parsed.get("quote_delta_pool_str"),
                **identity,
            }
        )
    return events


def _wallet_label(
    *,
    tx_count: int,
    buy_count: int,
    sell_count: int,
    gross_usdc: Decimal,
    gross_native: Decimal,
    max_usdc: Decimal,
    max_native: Decimal,
    net_base: Decimal,
    net_usdc: Decimal,
    net_wsol: Decimal,
    directional_count: int,
) -> tuple[str, str, str]:
    total_quote = gross_usdc + gross_native
    if directional_count < MIN_DIRECTIONAL_FOR_LABEL:
        if total_quote >= WHALE_USDC_VOLUME_THRESHOLD and max_usdc >= MAX_TRADE_WHALE_USDC:
            return "unknown", "low", "High volume but insufficient directional sample for labeling"
        return "unknown", "low", "Insufficient directional transactions for behavior labeling"

    buy_ratio = Decimal(buy_count) / Decimal(max(directional_count, 1))
    sell_ratio = Decimal(sell_count) / Decimal(max(directional_count, 1))
    balance_gap = abs(buy_ratio - sell_ratio)

    if total_quote < NOISE_USDC_THRESHOLD and gross_native < NOISE_NATIVE_THRESHOLD and tx_count <= 2:
        return "noise", "low", "Low value and low repeat count"

    if balance_gap <= MM_BALANCE_RATIO_MAX and buy_count >= 3 and sell_count >= 3:
        if abs(net_usdc) < WHALE_USDC_VOLUME_THRESHOLD / 10 and abs(net_wsol) < WHALE_NATIVE_VOLUME_THRESHOLD:
            return "possible_market_maker", "medium", "Balanced buy/sell with low net inventory"

    if (
        gross_usdc >= WHALE_USDC_VOLUME_THRESHOLD or gross_native >= WHALE_NATIVE_VOLUME_THRESHOLD
    ) and (max_usdc >= MAX_TRADE_WHALE_USDC or max_native >= MAX_TRADE_WHALE_NATIVE):
        if buy_count > sell_count:
            return "possible_whale_accumulator", "medium", "High quote volume with net buy bias"
        if sell_count > buy_count:
            return "possible_whale_dumper", "medium", "High quote volume with net sell bias"

    if balance_gap <= MM_BALANCE_RATIO_MAX and tx_count >= 5:
        return "possible_arbitrage_bot", "low", "Repeated balanced activity pattern"

    return "unknown", "low", "No conservative label matched"


def summarize_wallet_behavior(events: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = [WARN_NOT_SIGNAL, WARN_FAILED_EXCLUDED]

    failed_count = sum(1 for e in events if e.get("failed_transaction"))
    unknown_count = sum(
        1
        for e in events
        if not e.get("failed_transaction") and e.get("side") == SIDE_UNKNOWN
    )
    directional_events = [
        e
        for e in events
        if not e.get("failed_transaction") and e.get("side") in {SIDE_BUY_BASE, SIDE_SELL_BASE}
    ]

    if failed_count:
        warnings.append(WARN_FAILED_EXCLUDED)
    if unknown_count and len(events):
        unknown_rate = unknown_count / max(len(events) - failed_count, 1)
        if unknown_rate >= 0.5:
            warnings.append(WARN_HIGH_UNKNOWN)
    if any(e.get("parse_status") in {"PARTIAL", "UNKNOWN_FORMAT"} for e in events):
        warnings.append(WARN_POOL_VAULTS)
    if len(directional_events) < MIN_DIRECTIONAL_FOR_LABEL:
        warnings.append(WARN_LOW_SAMPLE)

    wallet_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in directional_events:
        wallet = event.get("trader_wallet")
        if not wallet:
            continue
        wallet_groups[wallet].append(event)

    wallets_out: list[dict[str, Any]] = []
    for wallet, group in wallet_groups.items():
        buy_count = sum(1 for e in group if e.get("side") == SIDE_BUY_BASE)
        sell_count = sum(1 for e in group if e.get("side") == SIDE_SELL_BASE)
        unknown_w = sum(1 for e in group if e.get("side") == SIDE_UNKNOWN)

        gross_usdc = Decimal("0")
        gross_native = Decimal("0")
        max_usdc = Decimal("0")
        max_native = Decimal("0")
        net_base = Decimal("0")
        net_usdc = Decimal("0")
        net_wsol = Decimal("0")

        fee_payers: set[str] = set()
        signers: set[str] = set()
        owners: set[str] = set()
        identity_sources: set[str] = set()
        confidences: set[str] = set()

        block_times: list[int] = []
        for e in group:
            quote_type = e.get("quote_token_type")
            usdc_amt = _to_decimal(e.get("approx_usd_value"))
            native_amt = _to_decimal(e.get("quote_amount_native"))
            base_delta = _to_decimal(e.get("base_delta_pool_str"))
            quote_delta = _to_decimal(e.get("quote_delta_pool_str"))

            if quote_type == QUOTE_USDC:
                gross_usdc += usdc_amt
                max_usdc = max(max_usdc, usdc_amt)
                net_usdc += -quote_delta
            elif quote_type in {QUOTE_WSOL, "OTHER"}:
                gross_native += native_amt
                max_native = max(max_native, native_amt)
                if quote_type == QUOTE_WSOL:
                    net_wsol += -quote_delta

            net_base += -base_delta
            if e.get("fee_payer"):
                fee_payers.add(e["fee_payer"])
            for s in e.get("signer_wallets") or []:
                signers.add(s)
            for o in e.get("token_owner_wallets") or []:
                owners.add(o)
            identity_sources.add(e.get("trader_identity_source") or IDENTITY_UNKNOWN)
            confidences.add(e.get("identity_confidence") or "low")
            if e.get("block_time") is not None:
                block_times.append(int(e["block_time"]))

        identity_source = next(iter(identity_sources)) if len(identity_sources) == 1 else IDENTITY_UNKNOWN
        identity_confidence = "high" if confidences == {"high"} else ("medium" if "medium" in confidences else "low")

        if identity_source.startswith("TOKEN_OWNER"):
            warnings.append(WARN_TOKEN_OWNER_USED)
        if identity_source == IDENTITY_TOKEN_OWNER_UNMATCHED:
            warnings.append(WARN_TOKEN_OWNER_UNMATCHED)
        if any(e.get("relayer_suspected") for e in group):
            warnings.append(WARN_RELAYER)
        if any(e.get("multiple_signers") for e in group):
            warnings.append(WARN_MULTI_SIGNER)

        label, label_conf, reason = _wallet_label(
            tx_count=len(group),
            buy_count=buy_count,
            sell_count=sell_count,
            gross_usdc=gross_usdc,
            gross_native=gross_native,
            max_usdc=max_usdc,
            max_native=max_native,
            net_base=net_base,
            net_usdc=net_usdc,
            net_wsol=net_wsol,
            directional_count=len(directional_events),
        )

        wallets_out.append(
            {
                "trader_wallet": wallet,
                "trader_identity_source": identity_source,
                "identity_confidence": identity_confidence,
                "likely_behavior": label,
                "behavior_confidence": label_conf,
                "behavior_reason": reason,
                "fee_payer_examples": sorted(fee_payers)[:3],
                "signer_wallet_examples": sorted(signers)[:3],
                "token_owner_wallet_examples": sorted(owners)[:3],
                "tx_count": len(group),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "unknown_count": unknown_w,
                "gross_usdc_volume": _decimal_str(gross_usdc),
                "gross_quote_native_volume": _decimal_str(gross_native),
                "max_usdc_trade": _decimal_str(max_usdc),
                "max_quote_native_trade": _decimal_str(max_native),
                "net_base": _decimal_str(net_base),
                "net_usdc": _decimal_str(net_usdc),
                "net_wsol": _decimal_str(net_wsol),
                "repeat_count": len(group),
                "first_seen_block_time": min(block_times) if block_times else None,
                "last_seen_block_time": max(block_times) if block_times else None,
            }
        )

    wallets_out.sort(key=lambda w: _to_decimal(w.get("gross_usdc_volume")) + _to_decimal(w.get("gross_quote_native_volume")), reverse=True)

    overall_label = "unknown"
    overall_confidence = "low"
    overall_reason = "Insufficient data for aggregate behavior label"

    if wallets_out:
        top = wallets_out[0]
        overall_label = top.get("likely_behavior", "unknown")
        overall_confidence = top.get("behavior_confidence", "low")
        overall_reason = top.get("behavior_reason", overall_reason)

    if len(directional_events) < MIN_DIRECTIONAL_FOR_LABEL:
        overall_label = "unknown"
        overall_confidence = "low"
        overall_reason = "Fewer than 10 successful directional transactions"

    return {
        "behavior_audit_enabled": True,
        "input_event_count": len(events),
        "successful_directional_event_count": len(directional_events),
        "unknown_event_count": unknown_count,
        "failed_event_count": failed_count,
        "unique_trader_wallets": len(wallet_groups),
        "wallets": wallets_out,
        "likely_behavior": overall_label,
        "confidence": overall_confidence,
        "reason": overall_reason,
        "warnings": sorted(set(warnings)),
    }
