from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_CSV = Path(os.environ.get(
    "THESIS_CONTEXT_DATASET_CSV",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_context_event_level_dataset_build_audit_20260810_212600\03_event_level_context_rebuild_dataset.csv"
))

OUT = ROOT / "data" / "audits" / f"thesis_missing_winner_chain_provider_feasibility_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

RAW_PROVIDER_LOOKBACK_HOURS = [24, 168]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return float(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if pd.isna(x):
        return None
    return x


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s


def norm_lower(x: Any) -> str:
    return norm(x).lower()


def norm_upper(x: Any) -> str:
    return norm(x).upper()


def looks_like_solana_address(x: Any) -> bool:
    s = norm(x)
    if not (32 <= len(s) <= 60):
        return False
    bad = set("0OIl+/=")
    if any(ch in bad for ch in s):
        return False
    if s.startswith("0x"):
        return False
    return True


def looks_like_evm_address(x: Any) -> bool:
    s = norm(x)
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", s))


def normalize_chain(chain: Any, pair_address: Any = "") -> str:
    c = norm_upper(chain)
    addr = norm(pair_address)

    if "SOL" in c:
        return "SOLANA"
    if "BSC" in c or "BNB" in c or "BINANCE" in c:
        return "BSC_EVM"
    if "BASE" in c:
        return "BASE_EVM"
    if "ARBITRUM" in c or c == "ARB":
        return "ARBITRUM_EVM"
    if "POLYGON" in c or "MATIC" in c:
        return "POLYGON_EVM"
    if "AVAX" in c or "AVALANCHE" in c:
        return "AVALANCHE_EVM"
    if "ETH" in c or "ERC" in c:
        return "ETHEREUM_EVM"
    if "EVM" in c:
        return "EVM_UNKNOWN"
    if looks_like_evm_address(addr):
        return "EVM_UNKNOWN"
    if looks_like_solana_address(addr):
        return "SOLANA_ADDRESS_NO_CHAIN"
    if c:
        return "OTHER_OR_UNKNOWN_CHAIN_" + c.replace(" ", "_")[:40]
    return "UNKNOWN_CHAIN"


def address_type(pair_address: Any) -> str:
    if looks_like_evm_address(pair_address):
        return "EVM_0X_ADDRESS"
    if looks_like_solana_address(pair_address):
        return "SOLANA_BASE58_ADDRESS"
    if norm(pair_address):
        return "OTHER_INVALID_OR_UNRECOGNIZED_ADDRESS"
    return "MISSING_PAIR_ADDRESS"


def route_class(chain_norm: str, pair_address_type: str) -> str:
    if chain_norm == "SOLANA" and pair_address_type == "SOLANA_BASE58_ADDRESS":
        return "SOLANA_HELIUS_ROUTE"
    if chain_norm == "SOLANA_ADDRESS_NO_CHAIN" and pair_address_type == "SOLANA_BASE58_ADDRESS":
        return "SOLANA_HELIUS_ROUTE_CHAIN_INFERRED_FROM_ADDRESS"
    if chain_norm.endswith("_EVM") or chain_norm == "EVM_UNKNOWN" or pair_address_type == "EVM_0X_ADDRESS":
        return "NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE"
    if pair_address_type == "MISSING_PAIR_ADDRESS":
        return "PAIR_OR_TOKEN_ADDRESS_RESOLUTION_REQUIRED"
    return "NON_SOLANA_NON_EVM_OR_UNKNOWN_PROVIDER_REQUIRED"


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchall()
    return bool(rows)


def table_cols(con: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(con, table):
        return []
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def read_table_cols(con: sqlite3.Connection, table: str, wanted: list[str]) -> pd.DataFrame:
    cols = table_cols(con, table)
    use = [c for c in wanted if c in cols]
    if not use:
        return pd.DataFrame()
    sql = ", ".join(f'"{c}"' for c in use)
    return pd.read_sql_query(f'SELECT {sql} FROM "{table}"', con)


def find_time_col(cols: list[str]) -> str | None:
    preferred = [
        "created_at",
        "timestamp",
        "snapshot_at",
        "fetched_at",
        "updated_at",
        "observed_at",
        "inserted_at",
        "time",
    ]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        cl = c.lower()
        if "time" in cl or "date" in cl or "created" in cl:
            return c
    return None


def add_event_key(df: pd.DataFrame, time_col: str = "candidate_event_time_utc") -> pd.DataFrame:
    out = df.copy()
    out["canonical_coin_id_num"] = pd.to_numeric(out["canonical_coin_id"], errors="coerce").astype("Int64")
    out["_event_time_dt"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    out["_event_time_ns"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    valid = out["_event_time_dt"].notna()
    out.loc[valid, "_event_time_ns"] = out.loc[valid, "_event_time_dt"].astype("int64").astype("Int64")

    out["event_key"] = out["canonical_coin_id_num"].astype(str) + "|" + out["_event_time_ns"].astype(str)
    out.loc[out["canonical_coin_id_num"].isna() | out["_event_time_dt"].isna(), "event_key"] = ""
    return out


def build_symbol_chain_key(symbol: Any, chain_norm_value: Any) -> str:
    s = norm_upper(symbol)
    c = norm_upper(chain_norm_value)
    if not s or not c:
        return ""
    return f"{s}|{c}"


def load_coin_bridge(con: sqlite3.Connection) -> pd.DataFrame:
    coins = read_table_cols(con, "coins", ["id", "symbol", "name", "chain", "pair_address"])
    if coins.empty:
        raise ValueError("Could not read coins identity columns.")

    coins["canonical_coin_id_num"] = pd.to_numeric(coins["id"], errors="coerce").astype("Int64")
    coins["symbol_coin"] = coins.get("symbol", "").astype(str)
    coins["name_coin"] = coins.get("name", "").astype(str)
    coins["chain_coin"] = coins.get("chain", "").astype(str)
    coins["pair_address"] = coins.get("pair_address", "").astype(str).str.strip()

    coins["chain_norm"] = [
        normalize_chain(c, a)
        for c, a in zip(coins["chain_coin"], coins["pair_address"])
    ]
    coins["pair_address_type"] = coins["pair_address"].map(address_type)
    coins["wallet_route_class"] = [
        route_class(c, a)
        for c, a in zip(coins["chain_norm"], coins["pair_address_type"])
    ]
    coins["pair_address_norm"] = coins["pair_address"].map(norm_lower)
    coins["symbol_chain_key"] = [
        build_symbol_chain_key(s, c)
        for s, c in zip(coins["symbol_coin"], coins["chain_norm"])
    ]
    return coins


def load_raw_provider(con: sqlite3.Connection, coins: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not table_exists(con, "raw_provider_payloads"):
        return pd.DataFrame(), {"raw_provider_table_exists": False}

    cols = table_cols(con, "raw_provider_payloads")
    time_col = find_time_col(cols)

    wanted = [
        "id",
        "symbol",
        "chain",
        "pair_address",
        "source",
        "provider",
        "created_at",
        "timestamp",
        "snapshot_at",
        "fetched_at",
        "updated_at",
        "observed_at",
        "inserted_at",
        "time",
    ]
    raw = read_table_cols(con, "raw_provider_payloads", wanted)
    if raw.empty:
        return pd.DataFrame(), {
            "raw_provider_table_exists": True,
            "raw_provider_rows_loaded": 0,
            "raw_provider_time_col": time_col,
        }

    raw["raw_symbol"] = raw.get("symbol", "").astype(str)
    raw["raw_chain"] = raw.get("chain", "").astype(str)
    raw["raw_pair_address"] = raw.get("pair_address", "").astype(str).str.strip()
    raw["raw_chain_norm"] = [
        normalize_chain(c, a)
        for c, a in zip(raw["raw_chain"], raw["raw_pair_address"])
    ]
    raw["raw_pair_address_norm"] = raw["raw_pair_address"].map(norm_lower)
    raw["raw_symbol_chain_key"] = [
        build_symbol_chain_key(s, c)
        for s, c in zip(raw["raw_symbol"], raw["raw_chain_norm"])
    ]

    # pair-address bridge
    pair_map = (
        coins[coins["pair_address_norm"].ne("")]
        .dropna(subset=["canonical_coin_id_num"])
        .drop_duplicates("pair_address_norm")
        .set_index("pair_address_norm")["canonical_coin_id_num"]
        .to_dict()
    )

    # symbol+chain bridge only if unique in coins
    symbol_chain_counts = coins[coins["symbol_chain_key"].ne("")]["symbol_chain_key"].value_counts()
    unique_symbol_chain = set(symbol_chain_counts[symbol_chain_counts == 1].index)
    sym_chain_map = (
        coins[coins["symbol_chain_key"].isin(unique_symbol_chain)]
        .drop_duplicates("symbol_chain_key")
        .set_index("symbol_chain_key")["canonical_coin_id_num"]
        .to_dict()
    )

    raw["canonical_coin_id_pair"] = raw["raw_pair_address_norm"].map(pair_map)
    raw["canonical_coin_id_symbol_chain"] = raw["raw_symbol_chain_key"].map(sym_chain_map)
    raw["canonical_coin_id_num"] = raw["canonical_coin_id_pair"].combine_first(raw["canonical_coin_id_symbol_chain"])
    raw["canonical_coin_id_num"] = pd.to_numeric(raw["canonical_coin_id_num"], errors="coerce").astype("Int64")

    if time_col and time_col in raw.columns:
        raw["raw_time_dt"] = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
    else:
        raw["raw_time_dt"] = pd.NaT

    meta = {
        "raw_provider_table_exists": True,
        "raw_provider_rows_loaded": int(len(raw)),
        "raw_provider_columns": cols,
        "raw_provider_time_col": time_col,
        "raw_provider_rows_resolved_to_coin": int(raw["canonical_coin_id_num"].notna().sum()),
        "raw_provider_rows_with_time": int(raw["raw_time_dt"].notna().sum()),
    }
    return raw, meta


def compute_raw_provider_coverage(winners: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if raw.empty or "canonical_coin_id_num" not in raw.columns:
        for _, w in winners.iterrows():
            rows.append({
                "event_key": w["event_key"],
                "canonical_coin_id": int(w["canonical_coin_id_num"]) if pd.notna(w["canonical_coin_id_num"]) else None,
                "label_x2_sl_4h": w.get("label_x2_sl_4h"),
                "chronological_split": w.get("chronological_split"),
                "chain_norm": w.get("chain_norm"),
                "wallet_route_class": w.get("wallet_route_class"),
                "raw_provider_any_rows_for_coin": 0,
                "raw_provider_any_timed_rows_for_coin": 0,
                "raw_provider_count_past_24h": 0,
                "raw_provider_count_past_168h": 0,
            })
        return pd.DataFrame(rows)

    raw_by_coin = {
        int(cid): g.copy()
        for cid, g in raw[raw["canonical_coin_id_num"].notna()].groupby("canonical_coin_id_num")
    }

    for _, w in winners.iterrows():
        cid = w["canonical_coin_id_num"]
        if pd.isna(cid):
            coin_raw = pd.DataFrame()
        else:
            coin_raw = raw_by_coin.get(int(cid), pd.DataFrame())

        event_t = w["_event_time_dt"]
        row = {
            "event_key": w["event_key"],
            "canonical_coin_id": int(cid) if pd.notna(cid) else None,
            "symbol": w.get("symbol_coin"),
            "pair_address": w.get("pair_address"),
            "label_x2_sl_4h": w.get("label_x2_sl_4h"),
            "chronological_split": w.get("chronological_split"),
            "candidate_event_time_utc": str(event_t) if pd.notna(event_t) else "",
            "chain_norm": w.get("chain_norm"),
            "pair_address_type": w.get("pair_address_type"),
            "wallet_route_class": w.get("wallet_route_class"),
            "raw_provider_any_rows_for_coin": int(len(coin_raw)),
            "raw_provider_any_timed_rows_for_coin": int(coin_raw["raw_time_dt"].notna().sum()) if not coin_raw.empty and "raw_time_dt" in coin_raw.columns else 0,
        }

        for h in RAW_PROVIDER_LOOKBACK_HOURS:
            col = f"raw_provider_count_past_{h}h"
            if coin_raw.empty or pd.isna(event_t) or "raw_time_dt" not in coin_raw.columns:
                row[col] = 0
                continue
            start = event_t - pd.to_timedelta(h, unit="h")
            x = coin_raw[
                (coin_raw["raw_time_dt"].notna())
                & (coin_raw["raw_time_dt"] >= start)
                & (coin_raw["raw_time_dt"] <= event_t)
            ]
            row[col] = int(len(x))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Event-level dataset not found: {DATASET_CSV}")
    if not DB.exists():
        raise FileNotFoundError(f"trader.db not found: {DB}")

    events = pd.read_csv(DATASET_CSV, low_memory=False)
    required = ["canonical_coin_id", "candidate_event_time_utc", "label_x2_sl_4h", "chronological_split"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Event dataset missing required columns: {missing}")

    events = add_event_key(events)

    con = connect_ro()
    coins = load_coin_bridge(con)
    raw_provider, raw_meta = load_raw_provider(con, coins)
    con.close()

    merged = events.merge(
        coins[[
            "canonical_coin_id_num",
            "symbol_coin",
            "name_coin",
            "chain_coin",
            "pair_address",
            "chain_norm",
            "pair_address_type",
            "wallet_route_class",
        ]],
        on="canonical_coin_id_num",
        how="left",
    )

    merged["is_winner"] = merged["label_x2_sl_4h"].eq("WINNER")
    winners = merged[merged["is_winner"]].copy()
    non_solana_winners = winners[~winners["wallet_route_class"].astype(str).str.startswith("SOLANA_HELIUS_ROUTE")].copy()
    solana_winners = winners[winners["wallet_route_class"].astype(str).str.startswith("SOLANA_HELIUS_ROUTE")].copy()

    # Row-level winner classification.
    winner_cols = [
        "event_key",
        "canonical_coin_id",
        "canonical_coin_id_num",
        "candidate_event_time_utc",
        "chronological_split",
        "label_x2_sl_4h",
        "symbol_coin",
        "name_coin",
        "chain_coin",
        "pair_address",
        "chain_norm",
        "pair_address_type",
        "wallet_route_class",
    ]
    winner_cols = [c for c in winner_cols if c in winners.columns]
    winners[winner_cols].to_csv(OUT / "01_all_winner_chain_provider_classification.csv", index=False, encoding="utf-8-sig")
    non_solana_winners[winner_cols].to_csv(OUT / "04_non_solana_missing_from_helius_winner_rows.csv", index=False, encoding="utf-8-sig")

    # Summaries.
    chain_summary = (
        winners.groupby(["chain_norm", "pair_address_type", "wallet_route_class"], dropna=False)
        .size()
        .reset_index(name="winner_rows")
        .sort_values("winner_rows", ascending=False)
    )
    chain_summary.to_csv(OUT / "02_winner_chain_provider_summary.csv", index=False, encoding="utf-8-sig")

    split_chain = (
        winners.groupby(["chronological_split", "chain_norm", "wallet_route_class"], dropna=False)
        .size()
        .reset_index(name="winner_rows")
        .sort_values(["chronological_split", "winner_rows"], ascending=[True, False])
    )
    split_chain.to_csv(OUT / "03_winner_split_by_chain_provider.csv", index=False, encoding="utf-8-sig")

    route_summary = (
        merged.groupby(["wallet_route_class", "label_x2_sl_4h"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["wallet_route_class", "label_x2_sl_4h"])
    )
    route_summary.to_csv(OUT / "05_route_class_by_label_all_events.csv", index=False, encoding="utf-8-sig")

    # Raw-provider coverage for winners.
    raw_cov = compute_raw_provider_coverage(winners, raw_provider)
    raw_cov.to_csv(OUT / "06_raw_provider_coverage_for_winners.csv", index=False, encoding="utf-8-sig")

    raw_cov_summary = (
        raw_cov.groupby(["wallet_route_class", "chronological_split"], dropna=False)
        .agg(
            winner_rows=("event_key", "count"),
            rows_with_any_raw_provider=("raw_provider_any_rows_for_coin", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            rows_with_any_timed_raw_provider=("raw_provider_any_timed_rows_for_coin", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            rows_with_raw_past_24h=("raw_provider_count_past_24h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            rows_with_raw_past_168h=("raw_provider_count_past_168h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            median_raw_past_24h=("raw_provider_count_past_24h", "median"),
            median_raw_past_168h=("raw_provider_count_past_168h", "median"),
        )
        .reset_index()
        .sort_values(["wallet_route_class", "chronological_split"])
    )
    raw_cov_summary.to_csv(OUT / "07_raw_provider_coverage_summary_for_winners.csv", index=False, encoding="utf-8-sig")

    # Feasibility / provider-route decision table.
    total_winners = int(len(winners))
    solana_winner_count = int(len(solana_winners))
    non_solana_count = int(len(non_solana_winners))
    evm_count = int((winners["wallet_route_class"] == "NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE").sum())
    resolver_required_count = int((winners["wallet_route_class"] == "PAIR_OR_TOKEN_ADDRESS_RESOLUTION_REQUIRED").sum())
    unknown_provider_count = int((winners["wallet_route_class"] == "NON_SOLANA_NON_EVM_OR_UNKNOWN_PROVIDER_REQUIRED").sum())

    evm_valtest = int(
        (
            winners["wallet_route_class"].eq("NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE")
            & winners["chronological_split"].isin(["validation", "test"])
        ).sum()
    )

    non_solana_valtest = int(
        (
            ~winners["wallet_route_class"].astype(str).str.startswith("SOLANA_HELIUS_ROUTE")
            & winners["chronological_split"].isin(["validation", "test"])
        ).sum()
    )

    raw_any_non_solana = int(
        (
            raw_cov["wallet_route_class"].ne("SOLANA_HELIUS_ROUTE")
            & (pd.to_numeric(raw_cov["raw_provider_any_rows_for_coin"], errors="coerce") > 0)
        ).sum()
    ) if not raw_cov.empty else 0

    raw_past24_non_solana = int(
        (
            raw_cov["wallet_route_class"].ne("SOLANA_HELIUS_ROUTE")
            & (pd.to_numeric(raw_cov["raw_provider_count_past_24h"], errors="coerce") > 0)
        ).sum()
    ) if not raw_cov.empty else 0

    raw_past168_non_solana = int(
        (
            raw_cov["wallet_route_class"].ne("SOLANA_HELIUS_ROUTE")
            & (pd.to_numeric(raw_cov["raw_provider_count_past_168h"], errors="coerce") > 0)
        ).sum()
    ) if not raw_cov.empty else 0

    decisions = [{
        "question": "How many WINNER rows are covered by the current Solana/Helius route?",
        "answer": f"{solana_winner_count}/{total_winners}",
        "classification": "SOLANA_HELIUS_PARTIAL_WINNER_COVERAGE",
        "evidence": f"non_solana_missing_from_helius={non_solana_count}",
    }, {
        "question": "Are many WINNER rows outside the Solana/Helius route?",
        "answer": "YES" if non_solana_count > solana_winner_count else "NO",
        "classification": "MAJORITY_WINNERS_OUTSIDE_SOLANA_HELIUS" if non_solana_count > solana_winner_count else "SOLANA_COVERS_LARGE_WINNER_SHARE",
        "evidence": f"solana={solana_winner_count}; non_solana={non_solana_count}; total={total_winners}",
    }, {
        "question": "Is an EVM-compatible non-Solana wallet-flow route potentially useful?",
        "answer": "YES" if evm_count > 0 else "NO",
        "classification": "EVM_WALLET_FLOW_ROUTE_FEASIBLE_CANDIDATE" if evm_count > 0 else "NO_EVM_WINNERS_IDENTIFIED",
        "evidence": f"evm_like_winners={evm_count}; evm_validation_test_winners={evm_valtest}",
    }, {
        "question": "Can non-Solana expansion help chronological validation/test winner coverage?",
        "answer": "YES" if non_solana_valtest > 0 else "NO",
        "classification": "NON_SOLANA_ROUTE_CAN_RESTORE_VALIDATION_TEST_WINNERS" if non_solana_valtest > 0 else "NON_SOLANA_ROUTE_DOES_NOT_RESTORE_VALIDATION_TEST_WINNERS",
        "evidence": f"non_solana_validation_test_winners={non_solana_valtest}",
    }, {
        "question": "Is there already raw-provider coverage for non-Solana WINNER rows?",
        "answer": "YES" if raw_any_non_solana > 0 else "NO",
        "classification": "RAW_PROVIDER_CAN_SUPPORT_NON_SOLANA_FEASIBILITY" if raw_any_non_solana > 0 else "RAW_PROVIDER_DOES_NOT_CURRENTLY_SUPPORT_NON_SOLANA_WINNERS",
        "evidence": f"raw_any_non_solana={raw_any_non_solana}; raw_past24_non_solana={raw_past24_non_solana}; raw_past168_non_solana={raw_past168_non_solana}",
    }, {
        "question": "How many WINNER rows still require pair/token address resolution?",
        "answer": str(resolver_required_count),
        "classification": "ADDRESS_RESOLUTION_REQUIRED_FOR_SOME_WINNERS" if resolver_required_count > 0 else "NO_ADDRESS_RESOLUTION_GAP_FOR_WINNERS",
        "evidence": f"resolver_required={resolver_required_count}; unknown_provider_required={unknown_provider_count}",
    }]

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "08_provider_route_decision_table.csv", index=False, encoding="utf-8-sig")

    expansion_plan_rows = []
    for route, g in winners.groupby("wallet_route_class", dropna=False):
        route_str = str(route)
        split_counts = {str(k): int(v) for k, v in g["chronological_split"].value_counts(dropna=False).to_dict().items()}
        chain_counts = {str(k): int(v) for k, v in g["chain_norm"].value_counts(dropna=False).to_dict().items()}

        if route_str.startswith("SOLANA_HELIUS_ROUTE"):
            proposed_action = "Already covered by Solana/Helius audits; no more Helius can create validation/test Solana winners if none exist."
        elif route_str == "NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE":
            proposed_action = "Candidate for EVM wallet-flow feasibility via chain-specific RPC/explorer/DexScreener-backed route; requires provider integration separate from Helius."
        elif route_str == "PAIR_OR_TOKEN_ADDRESS_RESOLUTION_REQUIRED":
            proposed_action = "Resolve pair/token address before any wallet-flow provider call."
        else:
            proposed_action = "Inspect chain/address manually; provider route not inferable from current metadata."

        expansion_plan_rows.append({
            "wallet_route_class": route_str,
            "winner_rows": int(len(g)),
            "chain_counts": json.dumps(chain_counts, ensure_ascii=False),
            "split_counts": json.dumps(split_counts, ensure_ascii=False),
            "proposed_action": proposed_action,
        })

    expansion_plan = pd.DataFrame(expansion_plan_rows).sort_values("winner_rows", ascending=False)
    expansion_plan.to_csv(OUT / "09_non_solana_expansion_plan_by_route.csv", index=False, encoding="utf-8-sig")

    if non_solana_count > solana_winner_count and non_solana_valtest > 0:
        final = "MAJORITY_WINNERS_OUTSIDE_HELIUS_AND_NON_SOLANA_ROUTE_CAN_RESTORE_CHRONOLOGICAL_WINNERS"
    elif non_solana_count > solana_winner_count:
        final = "MAJORITY_WINNERS_OUTSIDE_HELIUS_BUT_CHRONOLOGICAL_RESTORE_STATUS_LIMITED"
    elif non_solana_count > 0:
        final = "SOME_WINNERS_OUTSIDE_HELIUS_REQUIRE_NON_SOLANA_FEASIBILITY_ROUTE"
    else:
        final = "ALL_WINNERS_COVERED_BY_SOLANA_HELIUS_ROUTE"

    summary = {
        "classification": "THESIS_MISSING_WINNER_CHAIN_PROVIDER_FEASIBILITY_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "created_at": now_iso(),
        "safety": {
            "read_only_post_processing": True,
            "helius_queries": False,
            "external_provider_queries": False,
            "new_model_training": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
        },
        "inputs": {
            "event_level_dataset": str(DATASET_CSV),
            "trader_db": str(DB),
        },
        "raw_provider_meta": raw_meta,
        "winner_counts": {
            "total_winner_rows": total_winners,
            "solana_helius_route_winner_rows": solana_winner_count,
            "non_solana_or_non_helius_winner_rows": non_solana_count,
            "evm_compatible_winner_rows": evm_count,
            "resolver_required_winner_rows": resolver_required_count,
            "unknown_provider_required_winner_rows": unknown_provider_count,
            "non_solana_validation_test_winner_rows": non_solana_valtest,
            "evm_validation_test_winner_rows": evm_valtest,
        },
        "raw_provider_winner_coverage": {
            "non_solana_winner_rows_with_any_raw_provider": raw_any_non_solana,
            "non_solana_winner_rows_with_raw_past24h": raw_past24_non_solana,
            "non_solana_winner_rows_with_raw_past168h": raw_past168_non_solana,
        },
        "final_scientific_conclusion": final,
        "outputs": {
            "all_winner_classification": str(OUT / "01_all_winner_chain_provider_classification.csv"),
            "winner_chain_provider_summary": str(OUT / "02_winner_chain_provider_summary.csv"),
            "winner_split_by_chain_provider": str(OUT / "03_winner_split_by_chain_provider.csv"),
            "non_solana_missing_from_helius_winners": str(OUT / "04_non_solana_missing_from_helius_winner_rows.csv"),
            "route_class_by_label_all_events": str(OUT / "05_route_class_by_label_all_events.csv"),
            "raw_provider_coverage_for_winners": str(OUT / "06_raw_provider_coverage_for_winners.csv"),
            "raw_provider_coverage_summary_for_winners": str(OUT / "07_raw_provider_coverage_summary_for_winners.csv"),
            "provider_route_decision_table": str(OUT / "08_provider_route_decision_table.csv"),
            "non_solana_expansion_plan_by_route": str(OUT / "09_non_solana_expansion_plan_by_route.csv"),
            "summary_json": str(OUT / "thesis_missing_winner_chain_provider_feasibility_summary.json"),
            "summary_md": str(OUT / "thesis_missing_winner_chain_provider_feasibility_summary.md"),
        },
    }

    summary_json = OUT / "thesis_missing_winner_chain_provider_feasibility_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3E — Missing-winner Chain / Provider Feasibility")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only post-processing audit")
    lines.append("- No Helius queries")
    lines.append("- No external provider queries")
    lines.append("- No model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet connection")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## WINNER coverage by provider route")
    lines.append(f"- total WINNER rows: {total_winners}")
    lines.append(f"- Solana/Helius-route WINNER rows: {solana_winner_count}")
    lines.append(f"- non-Solana/non-Helius WINNER rows: {non_solana_count}")
    lines.append(f"- EVM-compatible WINNER rows: {evm_count}")
    lines.append(f"- address-resolution-required WINNER rows: {resolver_required_count}")
    lines.append(f"- unknown-provider-required WINNER rows: {unknown_provider_count}")
    lines.append(f"- non-Solana validation/test WINNER rows: {non_solana_valtest}")
    lines.append(f"- EVM validation/test WINNER rows: {evm_valtest}")
    lines.append("")
    lines.append("## Route summary")
    for _, r in chain_summary.iterrows():
        lines.append(
            f"- `{r['wallet_route_class']}` / `{r['chain_norm']}` / `{r['pair_address_type']}`: {int(r['winner_rows'])}"
        )
    lines.append("")
    lines.append("## Split × route summary")
    for _, r in split_chain.iterrows():
        lines.append(
            f"- `{r['chronological_split']}` / `{r['wallet_route_class']}` / `{r['chain_norm']}`: {int(r['winner_rows'])}"
        )
    lines.append("")
    lines.append("## Raw-provider coverage for WINNER rows")
    lines.append(f"- raw provider rows loaded: {raw_meta.get('raw_provider_rows_loaded')}")
    lines.append(f"- raw provider rows resolved to coin: {raw_meta.get('raw_provider_rows_resolved_to_coin')}")
    lines.append(f"- raw provider timed rows: {raw_meta.get('raw_provider_rows_with_time')}")
    lines.append(f"- non-Solana winners with any raw provider rows: {raw_any_non_solana}")
    lines.append(f"- non-Solana winners with raw provider rows in past 24h: {raw_past24_non_solana}")
    lines.append(f"- non-Solana winners with raw provider rows in past 168h: {raw_past168_non_solana}")
    lines.append("")
    lines.append("## Decision table")
    for _, r in decision_df.iterrows():
        lines.append(f"- {r['question']} `{r['classification']}` — {r['evidence']}")
    lines.append("")
    lines.append("## Expansion plan by route")
    for _, r in expansion_plan.iterrows():
        lines.append(
            f"- `{r['wallet_route_class']}`: winner_rows={int(r['winner_rows'])}; "
            f"split_counts={r['split_counts']}; action={r['proposed_action']}"
        )
    lines.append("")
    lines.append("## Final scientific conclusion")
    lines.append(f"`{final}`")
    lines.append("")
    lines.append("## Output files")
    for _, p in summary["outputs"].items():
        lines.append(f"- `{Path(p).name}`")

    summary_md = OUT / "thesis_missing_winner_chain_provider_feasibility_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": final,
        "total_winner_rows": total_winners,
        "solana_helius_route_winner_rows": solana_winner_count,
        "non_solana_or_non_helius_winner_rows": non_solana_count,
        "evm_compatible_winner_rows": evm_count,
        "non_solana_validation_test_winner_rows": non_solana_valtest,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
