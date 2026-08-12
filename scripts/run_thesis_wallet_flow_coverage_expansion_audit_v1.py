from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_wallet_flow_coverage_expansion_audit_{STAMP}"
CACHE = OUT / "helius_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

HELIUS_API_KEY = os.environ.get("HELIUS_API_KEY", "").strip()
DATASET_CSV = os.environ.get("THESIS_CONTEXT_DATASET_CSV", "").strip()

MAX_CASES = int(os.environ.get("THESIS_WALLET_MAX_CASES", "1500"))
MAX_WINDOWS = int(os.environ.get("THESIS_WALLET_MAX_WINDOWS", "2000"))
WINDOW_BEFORE_HOURS = float(os.environ.get("THESIS_WALLET_WINDOW_BEFORE_HOURS", "24"))
WINDOW_AFTER_HOURS = float(os.environ.get("THESIS_WALLET_WINDOW_AFTER_HOURS", "4"))
MAX_PAGES_PER_WINDOW = int(os.environ.get("THESIS_HELIUS_MAX_PAGES_PER_WINDOW", "5"))
LIMIT_PER_PAGE = int(os.environ.get("THESIS_HELIUS_LIMIT_PER_PAGE", "100"))
SLEEP_SECONDS = float(os.environ.get("THESIS_HELIUS_SLEEP_SECONDS", "0.12"))
RANDOM_SEED = int(os.environ.get("THESIS_RANDOM_SEED", "42"))

LARGE_FLOW_PERCENTILE = float(os.environ.get("THESIS_WALLET_LARGE_FLOW_PERCENTILE", "0.95"))

HELIUS_BASE = "https://api-mainnet.helius-rpc.com/v0/addresses/{address}/transactions"

WSOL_MINTS = {
    "SO11111111111111111111111111111111111111111",
    "SO11111111111111111111111111111111111111112",
}

STABLE_OR_COMMON_MINTS = {
    "EPJFWDD5AUFQSSQEM2QN1XZYBAPC8G4WEGGKZWYDT1V",  # USDC
    "ES9VMFRZACERMJFRF4H2FYD4KCONKY11MCDNVMKTDZWC",  # USDT
    "SO11111111111111111111111111111111111111111",
    "SO11111111111111111111111111111111111111112",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s


def norm_upper(x: Any) -> str:
    return norm(x).upper()


def looks_like_solana_address(x: Any) -> bool:
    s = norm(x)
    if not (32 <= len(s) <= 60):
        return False
    bad = set("0OIl+/=")
    if any(ch in bad for ch in s):
        return False
    return True


def unix_seconds(ts: Any) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def iso_from_unix(sec: int | float | None) -> str | None:
    if sec is None or pd.isna(sec):
        return None
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).isoformat()


def sha_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def read_table_cols(con: sqlite3.Connection, table: str, wanted: list[str]) -> pd.DataFrame:
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    use = [c for c in wanted if c in cols]
    if not use:
        return pd.DataFrame()
    sql = ", ".join(f'"{c}"' for c in use)
    return pd.read_sql_query(f'SELECT {sql} FROM "{table}"', con)


def find_latest_dataset() -> Path:
    if DATASET_CSV:
        p = Path(DATASET_CSV)
        if not p.exists():
            raise FileNotFoundError(f"THESIS_CONTEXT_DATASET_CSV not found: {p}")
        return p

    candidates = list(
        (ROOT / "data" / "audits").glob(
            "thesis_context_event_level_dataset_build_audit_*/03_event_level_context_rebuild_dataset.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("Could not find event-level context rebuild dataset CSV.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_cases() -> tuple[pd.DataFrame, Path]:
    dataset_path = find_latest_dataset()
    events = pd.read_csv(dataset_path, low_memory=False)

    if "canonical_coin_id" not in events.columns:
        raise SystemExit("Dataset missing canonical_coin_id.")
    if "candidate_event_time_utc" not in events.columns:
        raise SystemExit("Dataset missing candidate_event_time_utc.")
    if "label_x2_sl_4h" not in events.columns:
        raise SystemExit("Dataset missing label_x2_sl_4h.")

    events["candidate_event_time_utc"] = pd.to_datetime(events["candidate_event_time_utc"], errors="coerce", utc=True)
    events = events[events["candidate_event_time_utc"].notna()].copy()
    events["canonical_coin_id"] = pd.to_numeric(events["canonical_coin_id"], errors="coerce").astype("Int64")

    con = connect_ro()
    coins = read_table_cols(con, "coins", ["id", "symbol", "name", "chain", "pair_address"])
    con.close()

    coins["canonical_coin_id"] = pd.to_numeric(coins["id"], errors="coerce").astype("Int64")
    coins["chain_norm"] = coins.get("chain", "").astype(str).str.upper().str.strip()
    coins["pair_address"] = coins.get("pair_address", "").astype(str).str.strip()

    merged = events.merge(
        coins[["canonical_coin_id", "symbol", "name", "chain", "pair_address", "chain_norm"]],
        on="canonical_coin_id",
        how="left",
        suffixes=("", "_coin"),
    )

    merged["pair_address"] = merged["pair_address"].map(norm)
    merged["is_solana_candidate"] = (
        merged["chain_norm"].str.contains("SOL", na=False)
        & merged["pair_address"].map(looks_like_solana_address)
    )

    sol = merged[merged["is_solana_candidate"]].copy()

    if sol.empty:
        raise SystemExit("No Solana candidate events with usable pair_address were found.")

    sol["label_priority"] = sol["label_x2_sl_4h"].map({"WINNER": 0, "LOSER": 1, "FLAT": 2}).fillna(3)
    sol["raw_payload_count_past_24h_num"] = pd.to_numeric(sol.get("raw_payload_count_past_24h", 0), errors="coerce").fillna(0)
    sol["pool_flow_count_past_24h_num"] = pd.to_numeric(sol.get("pool_flow_count_past_24h", 0), errors="coerce").fillna(0)

    # Prioritize rare labels and high existing context coverage.
    sol = sol.sort_values(
        ["label_priority", "raw_payload_count_past_24h_num", "pool_flow_count_past_24h_num", "candidate_event_time_utc"],
        ascending=[True, False, False, True],
    ).copy()

    if MAX_CASES > 0 and len(sol) > MAX_CASES:
        rare = sol[sol["label_x2_sl_4h"].isin(["WINNER", "LOSER"])].copy()
        flat = sol[sol["label_x2_sl_4h"].eq("FLAT")].copy()
        remaining = max(0, MAX_CASES - len(rare))
        if remaining > 0:
            flat = flat.head(remaining)
        sol = pd.concat([rare, flat], ignore_index=True).head(MAX_CASES)

    sol = sol.reset_index(drop=True)
    if "candidate_event_id" not in sol.columns:
        sol["candidate_event_id"] = [
            f"WALLET_CASE_{i:06d}" for i in range(len(sol))
        ]

    sol["window_start_utc"] = sol["candidate_event_time_utc"] - pd.to_timedelta(WINDOW_BEFORE_HOURS, unit="h")
    sol["window_end_utc"] = sol["candidate_event_time_utc"] + pd.to_timedelta(WINDOW_AFTER_HOURS, unit="h")
    sol["window_start_unix"] = sol["window_start_utc"].map(unix_seconds)
    sol["window_end_unix"] = sol["window_end_utc"].map(unix_seconds)
    sol["event_unix"] = sol["candidate_event_time_utc"].map(unix_seconds)

    return sol, dataset_path


def build_merged_windows(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for address, g in cases.groupby("pair_address"):
        gg = g.sort_values("window_start_unix").copy()
        current = None

        for _, r in gg.iterrows():
            start = int(r["window_start_unix"])
            end = int(r["window_end_unix"])

            if current is None:
                current = {
                    "query_address": address,
                    "window_start_unix": start,
                    "window_end_unix": end,
                    "case_ids": [r["candidate_event_id"]],
                    "case_count": 1,
                }
                continue

            if start <= current["window_end_unix"]:
                current["window_end_unix"] = max(current["window_end_unix"], end)
                current["case_ids"].append(r["candidate_event_id"])
                current["case_count"] += 1
            else:
                rows.append(current)
                current = {
                    "query_address": address,
                    "window_start_unix": start,
                    "window_end_unix": end,
                    "case_ids": [r["candidate_event_id"]],
                    "case_count": 1,
                }

        if current is not None:
            rows.append(current)

    windows = pd.DataFrame(rows)
    windows["window_id"] = [
        "HLW|" + sha_key(r["query_address"], r["window_start_unix"], r["window_end_unix"])
        for _, r in windows.iterrows()
    ]
    windows["window_start_utc"] = windows["window_start_unix"].map(iso_from_unix)
    windows["window_end_utc"] = windows["window_end_unix"].map(iso_from_unix)

    windows = windows.sort_values(["case_count", "window_start_unix"], ascending=[False, True]).reset_index(drop=True)

    if MAX_WINDOWS > 0 and len(windows) > MAX_WINDOWS:
        windows = windows.head(MAX_WINDOWS).copy()

    return windows


def cached_get_transactions(address: str, start_unix: int, end_unix: int, window_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safe_window_id = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(window_id)
    )
    cache_file = CACHE / f"{safe_window_id}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data.get("transactions", []), {
            "window_id": window_id,
            "query_address": address,
            "cache_status": "HIT",
            "request_pages": data.get("request_pages", 0),
            "transactions_returned": len(data.get("transactions", [])),
            "error": "",
        }

    all_txs: list[dict[str, Any]] = []
    before_sig = None
    pages = 0
    error = ""

    for page in range(MAX_PAGES_PER_WINDOW):
        pages += 1

        params = {
            "api-key": HELIUS_API_KEY,
            "gte-time": str(int(start_unix)),
            "lte-time": str(int(end_unix)),
            "token-accounts": "all",
            "sort-order": "desc",
            "commitment": "confirmed",
            "limit": str(min(max(LIMIT_PER_PAGE, 1), 100)),
        }
        if before_sig:
            params["before-signature"] = before_sig

        url = HELIUS_BASE.format(address=urllib.parse.quote(address, safe="")) + "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "memecoin-thesis-wallet-flow-audit/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )

        last_exc = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = resp.read().decode("utf-8")
                    page_data = json.loads(body)
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in {429, 500, 502, 503, 504}:
                    time.sleep((attempt + 1) * 1.5)
                    continue
                error = f"HTTPError {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:500]}"
                page_data = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep((attempt + 1) * 1.0)
                page_data = None
        else:
            error = repr(last_exc)
            page_data = None

        if page_data is None:
            break

        if isinstance(page_data, dict) and "error" in page_data:
            error = json.dumps(page_data.get("error"), ensure_ascii=False)[:1000]
            break

        if not isinstance(page_data, list):
            error = f"Unexpected response type: {type(page_data).__name__}"
            break

        if not page_data:
            break

        all_txs.extend(page_data)

        before_sig = page_data[-1].get("signature")
        if not before_sig:
            break

        if len(page_data) < LIMIT_PER_PAGE:
            break

        time.sleep(SLEEP_SECONDS)

    payload = {
        "window_id": window_id,
        "query_address": address,
        "start_unix": int(start_unix),
        "end_unix": int(end_unix),
        "request_pages": pages,
        "transactions": all_txs,
        "error": error,
        "created_at": now_iso(),
    }
    cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return all_txs, {
        "window_id": window_id,
        "query_address": address,
        "cache_status": "MISS",
        "request_pages": pages,
        "transactions_returned": len(all_txs),
        "error": error,
    }


def tx_timestamp(tx: dict[str, Any]) -> int | None:
    for k in ["timestamp", "blockTime"]:
        v = tx.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None


def parse_transactions(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fetch_rows = []
    tx_rows = []
    transfer_rows = []

    for i, r in windows.iterrows():
        address = r["query_address"]
        start = int(r["window_start_unix"])
        end = int(r["window_end_unix"])
        window_id = r["window_id"]

        txs, fetch = cached_get_transactions(address, start, end, window_id)
        fetch.update({
            "window_start_unix": start,
            "window_end_unix": end,
            "window_start_utc": iso_from_unix(start),
            "window_end_utc": iso_from_unix(end),
            "case_count": int(r["case_count"]),
        })
        fetch_rows.append(fetch)

        for tx in txs:
            ts = tx_timestamp(tx)
            if ts is None:
                continue

            sig = norm(tx.get("signature"))
            tx_type = norm(tx.get("type"))
            source = norm(tx.get("source"))
            fee_payer = norm(tx.get("feePayer"))
            fee = tx.get("fee")
            description = norm(tx.get("description"))

            native_transfers = tx.get("nativeTransfers") or []
            token_transfers = tx.get("tokenTransfers") or []
            account_data = tx.get("accountData") or []

            tx_rows.append({
                "window_id": window_id,
                "query_address": address,
                "signature": sig,
                "timestamp_unix": ts,
                "timestamp_utc": iso_from_unix(ts),
                "type": tx_type,
                "source": source,
                "fee_payer": fee_payer,
                "fee": fee,
                "description": description[:500],
                "native_transfer_count": len(native_transfers),
                "token_transfer_count": len(token_transfers),
                "account_data_count": len(account_data),
            })

            for j, tr in enumerate(token_transfers):
                from_user = norm(tr.get("fromUserAccount"))
                to_user = norm(tr.get("toUserAccount"))
                mint = norm_upper(tr.get("mint"))
                amount = tr.get("tokenAmount")
                raw_amount = tr.get("rawTokenAmount")

                try:
                    amount_float = float(amount) if amount is not None else np.nan
                except Exception:
                    amount_float = np.nan

                transfer_rows.append({
                    "window_id": window_id,
                    "query_address": address,
                    "signature": sig,
                    "timestamp_unix": ts,
                    "timestamp_utc": iso_from_unix(ts),
                    "transfer_index": j,
                    "transfer_kind": "token",
                    "mint": mint,
                    "from_user_account": from_user,
                    "to_user_account": to_user,
                    "from_token_account": norm(tr.get("fromTokenAccount")),
                    "to_token_account": norm(tr.get("toTokenAccount")),
                    "token_amount": amount_float,
                    "raw_token_amount": json.dumps(raw_amount, ensure_ascii=False) if raw_amount is not None else "",
                    "native_amount_lamports": np.nan,
                })

            for j, tr in enumerate(native_transfers):
                amount = tr.get("amount")
                try:
                    amount_float = float(amount) if amount is not None else np.nan
                except Exception:
                    amount_float = np.nan

                transfer_rows.append({
                    "window_id": window_id,
                    "query_address": address,
                    "signature": sig,
                    "timestamp_unix": ts,
                    "timestamp_utc": iso_from_unix(ts),
                    "transfer_index": j,
                    "transfer_kind": "native",
                    "mint": "NATIVE_SOL",
                    "from_user_account": norm(tr.get("fromUserAccount")),
                    "to_user_account": norm(tr.get("toUserAccount")),
                    "from_token_account": "",
                    "to_token_account": "",
                    "token_amount": np.nan,
                    "raw_token_amount": "",
                    "native_amount_lamports": amount_float,
                })

    return pd.DataFrame(fetch_rows), pd.DataFrame(tx_rows), pd.DataFrame(transfer_rows)


def assign_to_cases(cases: pd.DataFrame, tx_rows: pd.DataFrame, transfer_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tx_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    case_tx = []
    case_transfers = []

    tx_by_addr = {
        addr: g.copy()
        for addr, g in tx_rows.groupby("query_address")
    }
    transfer_by_sig = {
        sig: g.copy()
        for sig, g in transfer_rows.groupby("signature")
    } if not transfer_rows.empty else {}

    for _, case in cases.iterrows():
        addr = case["pair_address"]
        if addr not in tx_by_addr:
            continue

        start = int(case["window_start_unix"])
        end = int(case["window_end_unix"])
        event_ts = int(case["event_unix"])
        case_id = case["candidate_event_id"]

        txg = tx_by_addr[addr]
        txg = txg[(txg["timestamp_unix"] >= start) & (txg["timestamp_unix"] <= end)].copy()

        for _, tx in txg.iterrows():
            phase = "PRE_EVENT" if int(tx["timestamp_unix"]) <= event_ts else "POST_EVENT"
            row = tx.to_dict()
            row.update({
                "candidate_event_id": case_id,
                "canonical_coin_id": int(case["canonical_coin_id"]),
                "symbol": case.get("symbol", ""),
                "label_x2_sl_4h": case.get("label_x2_sl_4h", ""),
                "chronological_split": case.get("chronological_split", ""),
                "candidate_event_time_utc": str(case["candidate_event_time_utc"]),
                "event_unix": event_ts,
                "phase": phase,
                "minutes_from_event": (int(tx["timestamp_unix"]) - event_ts) / 60.0,
            })
            case_tx.append(row)

            sig = tx["signature"]
            if sig in transfer_by_sig:
                tg = transfer_by_sig[sig]
                for _, tr in tg.iterrows():
                    trrow = tr.to_dict()
                    trrow.update({
                        "candidate_event_id": case_id,
                        "canonical_coin_id": int(case["canonical_coin_id"]),
                        "symbol": case.get("symbol", ""),
                        "label_x2_sl_4h": case.get("label_x2_sl_4h", ""),
                        "chronological_split": case.get("chronological_split", ""),
                        "candidate_event_time_utc": str(case["candidate_event_time_utc"]),
                        "event_unix": event_ts,
                        "phase": phase,
                        "minutes_from_event": (int(tr["timestamp_unix"]) - event_ts) / 60.0,
                    })
                    case_transfers.append(trrow)

    return pd.DataFrame(case_tx), pd.DataFrame(case_transfers)


def compute_case_features(cases: pd.DataFrame, case_tx: pd.DataFrame, case_transfers: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, c in cases.iterrows():
        case_id = c["candidate_event_id"]
        addr = c["pair_address"]

        txg = case_tx[case_tx["candidate_event_id"] == case_id] if not case_tx.empty else pd.DataFrame()
        trg = case_transfers[case_transfers["candidate_event_id"] == case_id] if not case_transfers.empty else pd.DataFrame()

        pre_tx = txg[txg["phase"] == "PRE_EVENT"] if not txg.empty else pd.DataFrame()
        post_tx = txg[txg["phase"] == "POST_EVENT"] if not txg.empty else pd.DataFrame()
        pre_tr = trg[trg["phase"] == "PRE_EVENT"] if not trg.empty else pd.DataFrame()

        external_wallets = set()
        if not pre_tr.empty:
            for col in ["from_user_account", "to_user_account"]:
                for x in pre_tr[col].dropna().astype(str):
                    if x and x != addr and looks_like_solana_address(x):
                        external_wallets.add(x)

        token_pre = pre_tr[pre_tr["transfer_kind"] == "token"].copy() if not pre_tr.empty else pd.DataFrame()
        native_pre = pre_tr[pre_tr["transfer_kind"] == "native"].copy() if not pre_tr.empty else pd.DataFrame()

        dominant_mint = ""
        target_like_mint = ""
        if not token_pre.empty:
            mint_counts = token_pre["mint"].value_counts()
            if not mint_counts.empty:
                dominant_mint = mint_counts.index[0]
            non_common = token_pre[~token_pre["mint"].isin(STABLE_OR_COMMON_MINTS)]
            if not non_common.empty:
                target_like_mint = non_common["mint"].value_counts().index[0]

        large_transfer_count = 0
        large_wallet_link_count = 0
        if not token_pre.empty:
            tmp = token_pre.copy()
            tmp["abs_token_amount"] = pd.to_numeric(tmp["token_amount"], errors="coerce").abs()
            vals = tmp["abs_token_amount"].dropna()
            if len(vals) >= 5:
                threshold = float(vals.quantile(LARGE_FLOW_PERCENTILE))
                large = tmp[tmp["abs_token_amount"] >= threshold].copy()
                large_transfer_count = int(len(large))
                wallets = set()
                for col in ["from_user_account", "to_user_account"]:
                    for x in large[col].dropna().astype(str):
                        if x and x != addr and looks_like_solana_address(x):
                            wallets.add(x)
                large_wallet_link_count = len(wallets)

        pre_in_to_address = 0
        pre_out_from_address = 0
        if not pre_tr.empty:
            pre_in_to_address = int((pre_tr["to_user_account"] == addr).sum())
            pre_out_from_address = int((pre_tr["from_user_account"] == addr).sum())

        if pre_in_to_address > pre_out_from_address:
            crude_direction = "NET_IN_TO_QUERY_ADDRESS"
        elif pre_out_from_address > pre_in_to_address:
            crude_direction = "NET_OUT_FROM_QUERY_ADDRESS"
        elif pre_in_to_address == 0 and pre_out_from_address == 0:
            crude_direction = "NO_DIRECT_QUERY_ADDRESS_DIRECTION"
        else:
            crude_direction = "BALANCED_OR_MIXED"

        real_wallet_evidence = (
            len(pre_tx) > 0
            and len(pre_tr) > 0
            and len(external_wallets) > 0
        )

        rows.append({
            "candidate_event_id": case_id,
            "canonical_coin_id": int(c["canonical_coin_id"]),
            "symbol": c.get("symbol", ""),
            "pair_address": addr,
            "label_x2_sl_4h": c.get("label_x2_sl_4h", ""),
            "chronological_split": c.get("chronological_split", ""),
            "candidate_event_time_utc": str(c["candidate_event_time_utc"]),
            "window_start_utc": str(c["window_start_utc"]),
            "window_end_utc": str(c["window_end_utc"]),
            "pre_tx_count": int(len(pre_tx)),
            "post_tx_count": int(len(post_tx)),
            "pre_transfer_count": int(len(pre_tr)),
            "pre_token_transfer_count": int(len(token_pre)),
            "pre_native_transfer_count": int(len(native_pre)),
            "pre_external_wallet_count": int(len(external_wallets)),
            "dominant_pre_mint": dominant_mint,
            "target_like_pre_mint": target_like_mint,
            "large_pre_token_transfer_count": large_transfer_count,
            "large_pre_wallet_link_count": large_wallet_link_count,
            "pre_in_to_query_address_transfer_count": pre_in_to_address,
            "pre_out_from_query_address_transfer_count": pre_out_from_address,
            "crude_query_address_direction": crude_direction,
            "real_wallet_level_evidence": bool(real_wallet_evidence),
            "large_wallet_level_evidence": bool(large_wallet_link_count > 0),
            "evidence_class": (
                "LARGE_WALLET_LEVEL_EVIDENCE"
                if large_wallet_link_count > 0 else
                "REAL_WALLET_LEVEL_EVIDENCE"
                if real_wallet_evidence else
                "POOL_OR_TX_ACTIVITY_ONLY"
                if len(pre_tx) > 0 else
                "NO_HELIUS_PRE_EVENT_ACTIVITY"
            ),
        })

    return pd.DataFrame(rows)


def write_summary(
    dataset_path: Path,
    cases: pd.DataFrame,
    windows: pd.DataFrame,
    fetch_df: pd.DataFrame,
    case_tx: pd.DataFrame,
    case_transfers: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    by_label = (
        features.groupby(["label_x2_sl_4h", "evidence_class"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["label_x2_sl_4h", "evidence_class"])
    )

    by_split = (
        features.groupby(["chronological_split", "evidence_class"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["chronological_split", "evidence_class"])
    )

    by_label.to_csv(OUT / "05_wallet_flow_coverage_by_label.csv", index=False, encoding="utf-8-sig")
    by_split.to_csv(OUT / "06_wallet_flow_coverage_by_split.csv", index=False, encoding="utf-8-sig")

    total_cases = int(len(features))
    real_cases = int(features["real_wallet_level_evidence"].sum()) if total_cases else 0
    large_cases = int(features["large_wallet_level_evidence"].sum()) if total_cases else 0

    label_counts = features["label_x2_sl_4h"].value_counts(dropna=False).to_dict()
    evidence_counts = features["evidence_class"].value_counts(dropna=False).to_dict()

    if total_cases == 0:
        conclusion = "NO_CASES_BUILT"
    elif large_cases >= 30 and real_cases / total_cases >= 0.20:
        conclusion = "WALLET_FLOW_EXPANSION_PRODUCED_TESTABLE_COVERAGE"
    elif real_cases > 0:
        conclusion = "WALLET_FLOW_EXPANSION_PRODUCED_SPARSE_REAL_WALLET_EVIDENCE"
    else:
        conclusion = "WALLET_FLOW_EXPANSION_REMAINS_POOL_ACTIVITY_OR_MISSINGNESS"

    summary = {
        "classification": "THESIS_WALLET_FLOW_COVERAGE_EXPANSION_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "dataset_csv": str(dataset_path),
        "created_at": now_iso(),
        "safety": {
            "read_only_audit": True,
            "helius_read_only_queries": True,
            "new_model_training": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
        },
        "helius": {
            "endpoint": "Enhanced Transactions By Address",
            "token_accounts": "all",
            "commitment": "confirmed",
            "limit_per_page": LIMIT_PER_PAGE,
            "max_pages_per_window": MAX_PAGES_PER_WINDOW,
            "window_before_hours": WINDOW_BEFORE_HOURS,
            "window_after_hours": WINDOW_AFTER_HOURS,
        },
        "case_selection": {
            "max_cases": MAX_CASES,
            "cases_selected": int(len(cases)),
            "unique_pair_addresses": int(cases["pair_address"].nunique()) if not cases.empty else 0,
            "merged_query_windows": int(len(windows)),
        },
        "fetch_summary": {
            "windows_fetched": int(len(fetch_df)),
            "cache_hits": int((fetch_df["cache_status"] == "HIT").sum()) if not fetch_df.empty else 0,
            "cache_misses": int((fetch_df["cache_status"] == "MISS").sum()) if not fetch_df.empty else 0,
            "request_pages_total": int(pd.to_numeric(fetch_df.get("request_pages", 0), errors="coerce").fillna(0).sum()) if not fetch_df.empty else 0,
            "transactions_returned_total": int(pd.to_numeric(fetch_df.get("transactions_returned", 0), errors="coerce").fillna(0).sum()) if not fetch_df.empty else 0,
            "windows_with_errors": int(fetch_df["error"].fillna("").ne("").sum()) if not fetch_df.empty and "error" in fetch_df.columns else 0,
        },
        "parsed_counts": {
            "transaction_rows_assigned_to_cases": int(len(case_tx)),
            "transfer_rows_assigned_to_cases": int(len(case_transfers)),
            "case_feature_rows": total_cases,
        },
        "coverage": {
            "real_wallet_level_cases": real_cases,
            "real_wallet_level_case_rate": real_cases / total_cases if total_cases else None,
            "large_wallet_level_cases": large_cases,
            "large_wallet_level_case_rate": large_cases / total_cases if total_cases else None,
            "label_counts": label_counts,
            "evidence_class_counts": evidence_counts,
        },
        "scientific_conclusion": conclusion,
        "thesis_safe_interpretation": (
            "Wallet-flow expansion is read-only coverage/missingness evidence. "
            "Only cases with external wallet actors and transfer-level links should be described as real wallet-level evidence. "
            "Pool/pair-address activity alone remains pool-flow proxy and does not prove whale-entry or whale-exit alpha."
        ),
        "outputs": {
            "candidate_cases": str(OUT / "00_wallet_flow_candidate_cases.csv"),
            "merged_windows": str(OUT / "01_helius_merged_query_windows.csv"),
            "helius_fetch_summary": str(OUT / "02_helius_fetch_summary.csv"),
            "transaction_rows": str(OUT / "03_helius_transaction_rows_assigned.csv"),
            "transfer_rows": str(OUT / "04_helius_transfer_rows_assigned.csv"),
            "case_features": str(OUT / "05_wallet_flow_case_features.csv"),
            "coverage_by_label": str(OUT / "05_wallet_flow_coverage_by_label.csv"),
            "coverage_by_split": str(OUT / "06_wallet_flow_coverage_by_split.csv"),
        },
    }

    with open(OUT / "thesis_wallet_flow_coverage_expansion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Audit 3 — Wallet-flow Coverage Expansion")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append(f"Dataset: `{dataset_path}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only audit")
    lines.append("- Helius read-only queries only")
    lines.append("- No model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet connection")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## Helius query design")
    lines.append("- Endpoint: Enhanced Transactions By Address")
    lines.append("- Query address: Solana `pair_address` from canonical coin bridge")
    lines.append(f"- Window: {WINDOW_BEFORE_HOURS}h before to {WINDOW_AFTER_HOURS}h after candidate event")
    lines.append("- `token-accounts=all`")
    lines.append("- `commitment=confirmed`")
    lines.append(f"- limit/page: {LIMIT_PER_PAGE}")
    lines.append(f"- max pages/window: {MAX_PAGES_PER_WINDOW}")
    lines.append("")
    lines.append("## Case selection")
    lines.append(f"- cases selected: {len(cases):,}")
    lines.append(f"- unique pair addresses: {cases['pair_address'].nunique() if not cases.empty else 0:,}")
    lines.append(f"- merged Helius query windows: {len(windows):,}")
    lines.append("")
    lines.append("## Fetch summary")
    for k, v in summary["fetch_summary"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Parsed counts")
    for k, v in summary["parsed_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- real wallet-level cases: {real_cases}/{total_cases} ({real_cases / total_cases if total_cases else None})")
    lines.append(f"- large wallet-level cases: {large_cases}/{total_cases} ({large_cases / total_cases if total_cases else None})")
    lines.append(f"- label counts: {label_counts}")
    lines.append(f"- evidence class counts: {evidence_counts}")
    lines.append("")
    lines.append("## Coverage by label")
    for _, r in by_label.iterrows():
        lines.append(f"- `{r['label_x2_sl_4h']}` / `{r['evidence_class']}`: {int(r['cases'])}")
    lines.append("")
    lines.append("## Scientific conclusion")
    lines.append(f"`{conclusion}`")
    lines.append("")
    lines.append("## Thesis-safe interpretation")
    lines.append(
        "This audit expands read-only wallet-flow coverage. It must not be interpreted as proof of whale alpha unless "
        "real wallet-level coverage is sufficient and directionality is independently supported. Pool/pair-address activity "
        "alone remains pool-flow proxy."
    )
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_wallet_flow_coverage_expansion_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_wallet_flow_coverage_expansion_summary.json"),
        "summary_md": str(md),
        "scientific_conclusion": conclusion,
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


def main() -> None:
    if not HELIUS_API_KEY:
        raise SystemExit(
            "HELIUS_API_KEY is missing. Set it first, for example:\n"
            '$env:HELIUS_API_KEY = "DUMMY"'
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset_path = find_latest_dataset()
    cases, dataset_path = load_cases()
    windows = build_merged_windows(cases)

    cases.to_csv(OUT / "00_wallet_flow_candidate_cases.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUT / "01_helius_merged_query_windows.csv", index=False, encoding="utf-8-sig")

    fetch_df, tx_df, transfer_df = parse_transactions(windows)

    fetch_df.to_csv(OUT / "02_helius_fetch_summary.csv", index=False, encoding="utf-8-sig")
    tx_df.to_csv(OUT / "03_helius_transaction_rows_raw.csv", index=False, encoding="utf-8-sig")
    transfer_df.to_csv(OUT / "04_helius_transfer_rows_raw.csv", index=False, encoding="utf-8-sig")

    case_tx, case_transfers = assign_to_cases(cases, tx_df, transfer_df)

    case_tx.to_csv(OUT / "03_helius_transaction_rows_assigned.csv", index=False, encoding="utf-8-sig")
    case_transfers.to_csv(OUT / "04_helius_transfer_rows_assigned.csv", index=False, encoding="utf-8-sig")

    features = compute_case_features(cases, case_tx, case_transfers)
    features.to_csv(OUT / "05_wallet_flow_case_features.csv", index=False, encoding="utf-8-sig")

    write_summary(dataset_path, cases, windows, fetch_df, case_tx, case_transfers, features)


if __name__ == "__main__":
    main()

