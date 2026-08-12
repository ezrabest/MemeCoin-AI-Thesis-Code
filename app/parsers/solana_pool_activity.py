"""
Parse Solana pool/pair transactions for signer, token-delta, and swap direction audit.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WSOL_MINT = "So11111111111111111111111111111111111111112"

QUOTE_MINTS = {USDC_MINT, WSOL_MINT}

OWNER_MATCHED_FEE_PAYER = "MATCHED_FEE_PAYER"
OWNER_MATCHED_SIGNER = "MATCHED_SIGNER"
OWNER_UNMATCHED = "UNMATCHED_TO_SIGNER"
OWNER_UNKNOWN = "UNKNOWN_OWNER"

SIDE_BUY_BASE = "BUY_BASE"
SIDE_SELL_BASE = "SELL_BASE"
SIDE_UNKNOWN = "UNKNOWN"
SIDE_IGNORED_FAILED = "IGNORED_FAILED"

PARSE_OK = "OK"
PARSE_PARTIAL = "PARTIAL"
PARSE_FAILED_TX = "FAILED_TRANSACTION"
PARSE_UNKNOWN_FORMAT = "UNKNOWN_FORMAT"
PARSE_ERROR = "ERROR"

QUOTE_USDC = "USDC"
QUOTE_WSOL = "WSOL"
QUOTE_OTHER = "OTHER"
QUOTE_UNKNOWN = "UNKNOWN"


def decimal_to_json_safe(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _ui_amount_from_token_amount(ui_token_amount: dict[str, Any] | None) -> Decimal | None:
    if not ui_token_amount:
        return None
    ui_amount_string = ui_token_amount.get("uiAmountString")
    if ui_amount_string is not None and ui_amount_string != "":
        return _to_decimal(ui_amount_string)
    ui_amount = ui_token_amount.get("uiAmount")
    if ui_amount is not None:
        return _to_decimal(ui_amount)
    raw_amount = ui_token_amount.get("amount")
    decimals = ui_token_amount.get("decimals")
    if raw_amount is not None and decimals is not None:
        try:
            return Decimal(str(raw_amount)) / (Decimal(10) ** int(decimals))
        except (InvalidOperation, ValueError, TypeError):
            return None
    return None


def extract_account_keys(tx: dict[str, Any]) -> list[dict[str, Any]]:
    transaction = tx.get("transaction") or {}
    message = transaction.get("message") or {}
    account_keys = message.get("accountKeys") or []
    loaded = (tx.get("meta") or {}).get("loadedAddresses") or {}
    writable = loaded.get("writable") or []
    readonly = loaded.get("readonly") or []

    keys: list[dict[str, Any]] = []
    for index, entry in enumerate(account_keys):
        if isinstance(entry, str):
            keys.append({"pubkey": entry, "signer": False, "writable": False, "account_index": index})
        elif isinstance(entry, dict):
            row = dict(entry)
            row["account_index"] = index
            keys.append(row)

    base_index = len(keys)
    for offset, pubkey in enumerate(writable):
        keys.append(
            {
                "pubkey": pubkey,
                "signer": False,
                "writable": True,
                "source": "loadedWritable",
                "account_index": base_index + offset,
            }
        )
    base_index = len(keys)
    for offset, pubkey in enumerate(readonly):
        keys.append(
            {
                "pubkey": pubkey,
                "signer": False,
                "writable": False,
                "source": "loadedReadonly",
                "account_index": base_index + offset,
            }
        )
    return keys


def extract_signer_wallets(tx: dict[str, Any]) -> list[str]:
    signers: list[str] = []
    for key in extract_account_keys(tx):
        if key.get("signer") and key.get("pubkey"):
            signers.append(str(key["pubkey"]))
    return signers


def extract_fee_payer(tx: dict[str, Any]) -> str | None:
    keys = extract_account_keys(tx)
    if keys:
        return keys[0].get("pubkey")
    return None


def extract_program_ids(tx: dict[str, Any]) -> list[str]:
    transaction = tx.get("transaction") or {}
    message = transaction.get("message") or {}
    program_ids: list[str] = []
    seen: set[str] = set()

    for instruction in message.get("instructions") or []:
        program_id = instruction.get("programId")
        if program_id and program_id not in seen:
            seen.add(program_id)
            program_ids.append(program_id)

    meta = tx.get("meta") or {}
    for inner in meta.get("innerInstructions") or []:
        for instruction in inner.get("instructions") or []:
            program_id = instruction.get("programId")
            if program_id and program_id not in seen:
                seen.add(program_id)
                program_ids.append(program_id)
    return program_ids


def _balance_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row.get("accountIndex", -1)),
        str(row.get("mint") or ""),
        str(row.get("owner") or ""),
    )


def build_token_balance_map(tx: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    account_keys = extract_account_keys(tx)
    pre_rows = (tx.get("meta") or {}).get("preTokenBalances") or []
    post_rows = (tx.get("meta") or {}).get("postTokenBalances") or []

    pre_map = {_balance_key(row): row for row in pre_rows}
    post_map = {_balance_key(row): row for row in post_rows}
    all_keys = set(pre_map) | set(post_map)

    balances: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key in all_keys:
        account_index, mint, owner = key
        pre_row = pre_map.get(key)
        post_row = post_map.get(key)
        pre_amount = _ui_amount_from_token_amount((pre_row or {}).get("uiTokenAmount"))
        post_amount = _ui_amount_from_token_amount((post_row or {}).get("uiTokenAmount"))
        if pre_amount is None:
            pre_amount = Decimal("0")
        if post_amount is None:
            post_amount = Decimal("0")
        delta = post_amount - pre_amount

        token_account = None
        if 0 <= account_index < len(account_keys):
            token_account = account_keys[account_index].get("pubkey")

        balances[key] = {
            "account_index": account_index,
            "token_account": token_account,
            "mint": mint,
            "owner": owner,
            "program_id": (post_row or pre_row or {}).get("programId"),
            "pre_ui_amount": pre_amount,
            "post_ui_amount": post_amount,
            "ui_amount_delta": delta,
            "pre_ui_amount_str": decimal_to_json_safe(pre_amount),
            "post_ui_amount_str": decimal_to_json_safe(post_amount),
            "ui_amount_delta_str": decimal_to_json_safe(delta),
        }
    return balances


def _owner_match_status(owner: str | None, fee_payer: str | None, signers: list[str]) -> str:
    if not owner:
        return OWNER_UNKNOWN
    if fee_payer and owner == fee_payer:
        return OWNER_MATCHED_FEE_PAYER
    if owner in signers:
        return OWNER_MATCHED_SIGNER
    return OWNER_UNMATCHED


def _resolve_trader_wallet(
    fee_payer: str | None,
    signers: list[str],
    trader_rows: list[dict[str, Any]],
) -> str | None:
    if fee_payer:
        return fee_payer
    if signers:
        return signers[0]
    for row in trader_rows:
        owner = row.get("owner")
        if owner:
            return owner
    return None


def _classify_quote_token(mint: str | None) -> str:
    if mint == USDC_MINT:
        return QUOTE_USDC
    if mint == WSOL_MINT:
        return QUOTE_WSOL
    if mint:
        return QUOTE_OTHER
    return QUOTE_UNKNOWN


def _extract_inner_transfer_checked(tx: dict[str, Any]) -> list[dict[str, Any]]:
    transfers: list[dict[str, Any]] = []
    meta = tx.get("meta") or {}
    for inner in meta.get("innerInstructions") or []:
        for instruction in inner.get("instructions") or []:
            parsed = instruction.get("parsed")
            if not isinstance(parsed, dict):
                continue
            if parsed.get("type") != "transferChecked":
                continue
            info = parsed.get("info") or {}
            transfers.append(
                {
                    "source": info.get("source"),
                    "destination": info.get("destination"),
                    "mint": info.get("mint"),
                    "authority": info.get("authority"),
                    "token_amount": info.get("tokenAmount"),
                }
            )
    return transfers


def _extract_address_table_lookup_accounts(tx: dict[str, Any]) -> list[str]:
    transaction = tx.get("transaction") or {}
    message = transaction.get("message") or {}
    lookups = message.get("addressTableLookups") or []
    accounts: list[str] = []
    for lookup in lookups:
        account_key = lookup.get("accountKey")
        if account_key:
            accounts.append(account_key)
    return accounts


def _identify_pool_rows(
    balance_rows: list[dict[str, Any]],
    pool_address: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, str | None]:
    pool_rows: list[dict[str, Any]] = []
    trader_rows: list[dict[str, Any]] = []

    for row in balance_rows:
        owner = row.get("owner") or ""
        token_account = row.get("token_account") or ""
        if owner == pool_address or token_account == pool_address:
            pool_rows.append(row)
        else:
            trader_rows.append(row)

    quote_mint: str | None = None
    base_mint: str | None = None
    mints_in_pool = {row.get("mint") for row in pool_rows if row.get("mint")}
    if USDC_MINT in mints_in_pool:
        quote_mint = USDC_MINT
    elif WSOL_MINT in mints_in_pool:
        quote_mint = WSOL_MINT
    else:
        quote_candidates = [m for m in mints_in_pool if m in QUOTE_MINTS]
        if len(quote_candidates) == 1:
            quote_mint = quote_candidates[0]

    if quote_mint:
        base_candidates = [m for m in mints_in_pool if m != quote_mint]
        if len(base_candidates) == 1:
            base_mint = base_candidates[0]

    if not base_mint or not quote_mint:
        all_mints = {row.get("mint") for row in balance_rows if row.get("mint")}
        if USDC_MINT in all_mints:
            quote_mint = quote_mint or USDC_MINT
        elif WSOL_MINT in all_mints:
            quote_mint = quote_mint or WSOL_MINT
        if quote_mint:
            base_candidates = [m for m in all_mints if m != quote_mint]
            if len(base_candidates) == 1:
                base_mint = base_candidates[0]

    return pool_rows, trader_rows, base_mint, quote_mint


def parse_transaction_token_deltas(tx: dict[str, Any], pool_address: str) -> dict[str, Any]:
    try:
        balance_map = build_token_balance_map(tx)
        balance_rows = list(balance_map.values())

        fee_payer = extract_fee_payer(tx)
        signers = extract_signer_wallets(tx)
        pool_rows, trader_rows, base_mint, quote_mint = _identify_pool_rows(
            balance_rows,
            pool_address,
        )

        for row in balance_rows:
            row["owner_match_status"] = _owner_match_status(row.get("owner"), fee_payer, signers)

        trader_wallet = _resolve_trader_wallet(fee_payer, signers, trader_rows)
        quote_token_type = _classify_quote_token(quote_mint)

        base_delta_pool = Decimal("0")
        quote_delta_pool = Decimal("0")
        base_delta_trader = Decimal("0")
        quote_delta_trader = Decimal("0")

        for row in pool_rows:
            mint = row.get("mint")
            delta = row.get("ui_amount_delta") or Decimal("0")
            if mint == base_mint:
                base_delta_pool += delta
            elif mint == quote_mint:
                quote_delta_pool += delta

        for row in trader_rows:
            mint = row.get("mint")
            delta = row.get("ui_amount_delta") or Decimal("0")
            if mint == base_mint:
                base_delta_trader += delta
            elif mint == quote_mint:
                quote_delta_trader += delta

        return {
            "token_balances": balance_rows,
            "pool_rows": pool_rows,
            "trader_rows": trader_rows,
            "base_token_mint": base_mint,
            "quote_token_mint": quote_mint,
            "quote_token_type": quote_token_type,
            "base_delta_pool": base_delta_pool,
            "quote_delta_pool": quote_delta_pool,
            "base_delta_trader": base_delta_trader,
            "quote_delta_trader": quote_delta_trader,
            "fee_payer": fee_payer,
            "signer_wallets": signers,
            "trader_wallet": trader_wallet,
        }
    except Exception as exc:
        return {"error": str(exc)}


def infer_pool_swap(tx: dict[str, Any], pool_address: str) -> dict[str, Any]:
    transaction = tx.get("transaction") or {}
    meta = tx.get("meta") or {}
    signatures = transaction.get("signatures") or []
    signature = signatures[0] if signatures else None

    base_result: dict[str, Any] = {
        "signature": signature,
        "block_time": tx.get("blockTime"),
        "slot": tx.get("slot"),
        "transaction_index": tx.get("transactionIndex"),
        "fee": meta.get("fee"),
        "meta_err": meta.get("err"),
        "failed_transaction": meta.get("err") is not None,
        "account_keys": extract_account_keys(tx),
        "signer_wallets": extract_signer_wallets(tx),
        "fee_payer": extract_fee_payer(tx),
        "program_ids": extract_program_ids(tx),
        "address_table_lookup_accounts": _extract_address_table_lookup_accounts(tx),
        "inner_transfer_checked": _extract_inner_transfer_checked(tx),
        "pre_token_balances": meta.get("preTokenBalances") or [],
        "post_token_balances": meta.get("postTokenBalances") or [],
        "pool_address": pool_address,
    }

    if meta.get("err") is not None:
        base_result.update(
            {
                "parse_status": PARSE_FAILED_TX,
                "side": SIDE_IGNORED_FAILED,
                "approx_usd_value": None,
                "quote_amount_native": None,
            }
        )
        return base_result

    delta_info = parse_transaction_token_deltas(tx, pool_address)
    if delta_info.get("error"):
        base_result.update(
            {
                "parse_status": PARSE_ERROR,
                "side": SIDE_UNKNOWN,
                "parse_error": delta_info["error"],
            }
        )
        return base_result

    base_result.update(delta_info)
    base_mint = delta_info.get("base_token_mint")
    quote_mint = delta_info.get("quote_token_mint")
    quote_token_type = delta_info.get("quote_token_type")
    base_delta_pool = delta_info.get("base_delta_pool") or Decimal("0")
    quote_delta_pool = delta_info.get("quote_delta_pool") or Decimal("0")

    pool_rows = delta_info.get("pool_rows") or []
    confident_pool = bool(pool_rows) and base_mint and quote_mint

    side = SIDE_UNKNOWN
    parse_status = PARSE_PARTIAL

    if confident_pool:
        if base_delta_pool > Decimal("0") and quote_delta_pool < Decimal("0"):
            side = SIDE_SELL_BASE
            parse_status = PARSE_OK
        elif base_delta_pool < Decimal("0") and quote_delta_pool > Decimal("0"):
            side = SIDE_BUY_BASE
            parse_status = PARSE_OK
        elif base_delta_pool == Decimal("0") and quote_delta_pool == Decimal("0"):
            parse_status = PARSE_UNKNOWN_FORMAT
        else:
            parse_status = PARSE_PARTIAL
    elif not base_mint or not quote_mint:
        parse_status = PARSE_UNKNOWN_FORMAT

    approx_usd_value: Decimal | None = None
    quote_amount_native: Decimal | None = None

    if side in {SIDE_BUY_BASE, SIDE_SELL_BASE} and quote_mint:
        quote_abs = abs(quote_delta_pool)
        if quote_token_type == QUOTE_USDC:
            approx_usd_value = quote_abs
        elif quote_token_type == QUOTE_WSOL:
            quote_amount_native = quote_abs
            approx_usd_value = None
        elif quote_token_type == QUOTE_OTHER:
            quote_amount_native = quote_abs
            approx_usd_value = None

    base_result.update(
        {
            "parse_status": parse_status,
            "side": side,
            "approx_usd_value": decimal_to_json_safe(approx_usd_value),
            "quote_amount_native": decimal_to_json_safe(quote_amount_native),
            "base_delta_pool_str": decimal_to_json_safe(base_delta_pool),
            "quote_delta_pool_str": decimal_to_json_safe(quote_delta_pool),
            "base_delta_trader_str": decimal_to_json_safe(delta_info.get("base_delta_trader")),
            "quote_delta_trader_str": decimal_to_json_safe(delta_info.get("quote_delta_trader")),
        }
    )
    return base_result


def dedupe_signatures(signature_rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in signature_rows:
        sig = row.get("signature")
        if not sig or sig in seen:
            continue
        seen.add(sig)
        ordered.append(sig)
    return ordered


def compact_example(parsed: dict[str, Any]) -> dict[str, Any]:
    token_balances = []
    for row in parsed.get("token_balances") or []:
        token_balances.append(
            {
                "token_account": row.get("token_account"),
                "token_owner_wallet": row.get("owner"),
                "owner_match_status": row.get("owner_match_status"),
                "mint": row.get("mint"),
                "ui_amount_delta_str": row.get("ui_amount_delta_str"),
            }
        )
    return {
        "signature": parsed.get("signature"),
        "block_time": parsed.get("block_time"),
        "slot": parsed.get("slot"),
        "failed_transaction": parsed.get("failed_transaction"),
        "parse_status": parsed.get("parse_status"),
        "side": parsed.get("side"),
        "trader_wallet": parsed.get("trader_wallet"),
        "fee_payer": parsed.get("fee_payer"),
        "signer_wallets": parsed.get("signer_wallets"),
        "base_token_mint": parsed.get("base_token_mint"),
        "quote_token_mint": parsed.get("quote_token_mint"),
        "quote_token_type": parsed.get("quote_token_type"),
        "base_delta_pool_str": parsed.get("base_delta_pool_str"),
        "quote_delta_pool_str": parsed.get("quote_delta_pool_str"),
        "approx_usd_value": parsed.get("approx_usd_value"),
        "quote_amount_native": parsed.get("quote_amount_native"),
        "token_balances": token_balances,
        "program_ids": parsed.get("program_ids"),
    }
