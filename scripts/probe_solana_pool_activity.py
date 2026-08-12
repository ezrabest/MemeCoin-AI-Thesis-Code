#!/usr/bin/env python3
"""Probe recent Solana pool/pair activity via raw JSON-RPC (Phase 1/2 audit tool)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.parsers.solana_pool_activity import (  # noqa: E402
    SIDE_BUY_BASE,
    SIDE_IGNORED_FAILED,
    SIDE_SELL_BASE,
    SIDE_UNKNOWN,
    QUOTE_OTHER,
    QUOTE_UNKNOWN,
    QUOTE_USDC,
    QUOTE_WSOL,
    compact_example,
    dedupe_signatures,
    infer_pool_swap,
)
from app.parsers.solana_wallet_behavior import (  # noqa: E402
    WARN_HELIUS_MISMATCH,
    build_behavior_events,
    summarize_wallet_behavior,
)
from app.providers.helius import validate_signatures_against_raw  # noqa: E402
from app.providers.helius_budget import HeliusBudgetManager  # noqa: E402
from app.providers.solana_rpc import (  # noqa: E402
    PUBLIC_RPC_URL,
    SOLANA_RPC_OK,
    SolanaRpcClient,
    utc_now_iso,
)


def _to_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def cap_public_rpc_limit(
    limit: int,
    max_limit: int,
    rpc_url: str,
    allow_large: bool,
) -> int:
    effective = min(limit, max_limit)
    if rpc_url.rstrip("/") == PUBLIC_RPC_URL.rstrip("/") and not allow_large:
        effective = min(effective, 100)
    return effective


def run_probe(
    *,
    pool_address: str,
    limit: int = 25,
    max_limit: int = 100,
    rpc_url: str | None = None,
    audit_dir: Path | None = None,
    include_raw_examples: bool = False,
    raw_example_limit: int = 3,
    allow_large_public_rpc_run: bool = False,
    validate_with_helius: bool = False,
    helius_validation_limit: int = 5,
    helius_cache_dir: Path | None = None,
    helius_monthly_budget: int = 100_000,
) -> dict[str, Any]:
    client = SolanaRpcClient(rpc_url=rpc_url)
    effective_limit = cap_public_rpc_limit(
        limit,
        max_limit,
        client.get_rpc_url(),
        allow_large_public_rpc_run,
    )

    signature_rows = client.get_signatures_for_address(pool_address, limit=effective_limit)
    signatures_found = len(signature_rows)
    signatures = dedupe_signatures(signature_rows)
    signatures_deduped = len(signatures)

    parsed_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    raw_examples: list[dict[str, Any]] = []
    tx_failed_to_fetch_count = 0

    for signature in signatures:
        tx_response = client.get_transaction(signature)
        if tx_response.get("status") != SOLANA_RPC_OK or not tx_response.get("result"):
            tx_failed_to_fetch_count += 1
            parse_errors.append(
                {
                    "signature": signature,
                    "status": tx_response.get("status"),
                    "error": tx_response.get("error"),
                }
            )
            continue

        tx = tx_response["result"]
        parsed = infer_pool_swap(tx, pool_address)
        parsed_rows.append(parsed)

        if include_raw_examples and len(raw_examples) < raw_example_limit:
            raw_examples.append(tx)

    buy_count = 0
    sell_count = 0
    unknown_count = 0
    failed_transaction_count = 0
    failed_transaction_examples: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    quote_token_type_counts = Counter({QUOTE_USDC: 0, QUOTE_WSOL: 0, QUOTE_OTHER: 0, QUOTE_UNKNOWN: 0})

    gross_usdc_volume = Decimal("0")
    gross_quote_native_volume = Decimal("0")
    max_usdc_trade = Decimal("0")
    max_quote_native_trade = Decimal("0")
    net_base = Decimal("0")
    net_usdc = Decimal("0")
    net_wsol = Decimal("0")

    wallet_usdc: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    wallet_quote_native: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    unique_traders: set[str] = set()

    behavior_events = build_behavior_events(parsed_rows)

    for parsed, behavior_event in zip(parsed_rows, behavior_events):
        quote_type = parsed.get("quote_token_type") or QUOTE_UNKNOWN
        quote_token_type_counts[quote_type] += 1

        if parsed.get("failed_transaction"):
            failed_transaction_count += 1
            if len(failed_transaction_examples) < raw_example_limit:
                failed_transaction_examples.append(compact_example(parsed))
            continue

        side = parsed.get("side")
        if side == SIDE_BUY_BASE:
            buy_count += 1
        elif side == SIDE_SELL_BASE:
            sell_count += 1
        elif side == SIDE_UNKNOWN:
            unknown_count += 1
        elif side == SIDE_IGNORED_FAILED:
            continue

        trader = behavior_event.get("trader_wallet") or parsed.get("trader_wallet")
        if trader:
            unique_traders.add(trader)

        base_delta_pool = _to_decimal(parsed.get("base_delta_pool_str") or parsed.get("base_delta_pool"))
        quote_delta_pool = _to_decimal(parsed.get("quote_delta_pool_str") or parsed.get("quote_delta_pool"))

        if side in {SIDE_BUY_BASE, SIDE_SELL_BASE}:
            net_base += -base_delta_pool
            if quote_type == QUOTE_USDC:
                usdc_amt = _to_decimal(parsed.get("approx_usd_value"))
                gross_usdc_volume += usdc_amt
                if usdc_amt > max_usdc_trade:
                    max_usdc_trade = usdc_amt
                net_usdc += -quote_delta_pool
                if trader:
                    wallet_usdc[trader] += usdc_amt
            elif quote_type == QUOTE_WSOL:
                wsol_amt = _to_decimal(parsed.get("quote_amount_native"))
                gross_quote_native_volume += wsol_amt
                if wsol_amt > max_quote_native_trade:
                    max_quote_native_trade = wsol_amt
                net_wsol += -quote_delta_pool
                if trader:
                    wallet_quote_native[trader] += wsol_amt
            elif quote_type == QUOTE_OTHER:
                native_amt = _to_decimal(parsed.get("quote_amount_native"))
                gross_quote_native_volume += native_amt
                if native_amt > max_quote_native_trade:
                    max_quote_native_trade = native_amt
                if trader:
                    wallet_quote_native[trader] += native_amt

        if len(examples) < raw_example_limit and side in {SIDE_BUY_BASE, SIDE_SELL_BASE, SIDE_UNKNOWN}:
            examples.append(compact_example(parsed))

    def _top_wallets(wallet_map: dict[str, Decimal], limit_n: int = 5) -> list[dict[str, str]]:
        ranked = sorted(wallet_map.items(), key=lambda item: item[1], reverse=True)
        return [
            {"wallet": wallet, "volume": format(volume, "f")}
            for wallet, volume in ranked[:limit_n]
            if volume > Decimal("0")
        ]

    budget_manager = HeliusBudgetManager(monthly_budget=helius_monthly_budget)
    helius_validation = validate_signatures_against_raw(
        parsed_rows,
        enabled=validate_with_helius,
        validation_limit=helius_validation_limit,
        cache_dir=helius_cache_dir,
        budget_manager=budget_manager if validate_with_helius else None,
    )

    wallet_behavior_audit = summarize_wallet_behavior(behavior_events)
    if helius_validation.get("helius_mismatches"):
        warnings = list(wallet_behavior_audit.get("warnings") or [])
        if WARN_HELIUS_MISMATCH not in warnings:
            warnings.append(WARN_HELIUS_MISMATCH)
        wallet_behavior_audit["warnings"] = sorted(set(warnings))

    audit: dict[str, Any] = {
        "pool_address": pool_address,
        "source_provider": "solana_rpc",
        "rpc_url_used": client.get_rpc_url(),
        "tx_requested": effective_limit,
        "signatures_found": signatures_found,
        "signatures_deduped": signatures_deduped,
        "tx_fetched": len(parsed_rows),
        "tx_parsed": len(parsed_rows),
        "tx_failed_to_fetch_count": tx_failed_to_fetch_count,
        "failed_transaction_count": failed_transaction_count,
        "rpc_calls_attempted": client.stats.rpc_calls_attempted,
        "rpc_calls_succeeded": client.stats.rpc_calls_succeeded,
        "rpc_rate_limited_count": client.stats.rpc_rate_limited_count,
        "rpc_forbidden_count": client.stats.rpc_forbidden_count,
        "rpc_retry_count": client.stats.rpc_retry_count,
        "rpc_timeout_count": client.stats.rpc_timeout_count,
        "rpc_null_result_count": client.stats.rpc_null_result_count,
        "cache_hits": client.stats.cache_hits,
        "cache_misses": client.stats.cache_misses,
        "unique_trader_wallets": len(unique_traders),
        "top_wallets_by_gross_usdc_volume": _top_wallets(wallet_usdc),
        "top_wallets_by_quote_native_volume": _top_wallets(wallet_quote_native),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "unknown_count": unknown_count,
        "quote_token_type_counts": dict(quote_token_type_counts),
        "gross_usdc_volume": format(gross_usdc_volume, "f"),
        "gross_quote_native_volume": format(gross_quote_native_volume, "f"),
        "max_usdc_trade": format(max_usdc_trade, "f"),
        "max_quote_native_trade": format(max_quote_native_trade, "f"),
        "net_base": format(net_base, "f"),
        "net_usdc": format(net_usdc, "f"),
        "net_wsol": format(net_wsol, "f"),
        "examples": examples,
        "failed_transaction_examples": failed_transaction_examples,
        "parse_errors": parse_errors,
        "helius_validation": helius_validation,
        "wallet_behavior_audit": wallet_behavior_audit,
    }

    if include_raw_examples:
        audit["raw_examples"] = raw_examples

    out_dir = audit_dir or (ROOT / "data" / "audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"solana_pool_probe_{utc_now_iso()}.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["audit_path"] = str(out_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Solana pool activity via raw RPC")
    parser.add_argument("--address", required=True, help="Pool/pair address (not a trader wallet)")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-limit", type=int, default=100)
    parser.add_argument("--rpc-url", default=None)
    parser.add_argument("--audit-dir", default="data/audits")
    parser.add_argument("--include-raw-examples", action="store_true")
    parser.add_argument("--raw-example-limit", type=int, default=3)
    parser.add_argument("--allow-large-public-rpc-run", action="store_true")
    parser.add_argument("--validate-with-helius", action="store_true")
    parser.add_argument("--helius-validation-limit", type=int, default=5)
    parser.add_argument("--helius-cache-dir", default="data/cache/helius")
    parser.add_argument("--helius-monthly-budget", type=int, default=100_000)
    args = parser.parse_args()

    audit = run_probe(
        pool_address=args.address,
        limit=args.limit,
        max_limit=args.max_limit,
        rpc_url=args.rpc_url,
        audit_dir=Path(args.audit_dir),
        include_raw_examples=args.include_raw_examples,
        raw_example_limit=args.raw_example_limit,
        allow_large_public_rpc_run=args.allow_large_public_rpc_run,
        validate_with_helius=args.validate_with_helius,
        helius_validation_limit=args.helius_validation_limit,
        helius_cache_dir=Path(args.helius_cache_dir),
        helius_monthly_budget=args.helius_monthly_budget,
    )

    summary = {k: v for k, v in audit.items() if k not in {"raw_examples"}}
    print(json.dumps(summary, indent=2))
    print(f"\nAudit written: {audit.get('audit_path')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
