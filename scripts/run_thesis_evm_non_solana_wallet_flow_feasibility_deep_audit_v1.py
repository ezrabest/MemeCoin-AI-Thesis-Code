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

THREE_E_ROOT = Path(os.environ.get(
    "THESIS_3E_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_missing_winner_chain_provider_feasibility_audit_20260810_231253"
))

OUT = ROOT / "data" / "audits" / f"thesis_evm_non_solana_wallet_flow_feasibility_deep_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"


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


def norm_upper(x: Any) -> str:
    return norm(x).upper()


def norm_lower(x: Any) -> str:
    return norm(x).lower()


def looks_like_evm_address(x: Any) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", norm(x)))


def looks_like_solana_address(x: Any) -> bool:
    s = norm(x)
    if s.startswith("0x"):
        return False
    if not (32 <= len(s) <= 60):
        return False
    return not any(ch in set("0OIl+/=") for ch in s)


def normalize_chain(chain: Any, address: Any = "") -> str:
    c = norm_upper(chain)
    a = norm(address)

    if "SOL" in c:
        return "SOLANA"
    if "BASE" in c:
        return "BASE_EVM"
    if "BSC" in c or "BNB" in c or "BINANCE" in c:
        return "BSC_EVM"
    if "ETH" in c or "ERC" in c:
        return "ETHEREUM_EVM"
    if "ARB" in c:
        return "ARBITRUM_EVM"
    if "POLYGON" in c or "MATIC" in c:
        return "POLYGON_EVM"
    if "AVAX" in c or "AVALANCHE" in c:
        return "AVALANCHE_EVM"
    if looks_like_evm_address(a):
        return "EVM_UNKNOWN"
    if looks_like_solana_address(a):
        return "SOLANA_ADDRESS_NO_CHAIN"
    if c:
        return "OTHER_OR_UNKNOWN_CHAIN_" + c.replace(" ", "_")[:40]
    return "UNKNOWN_CHAIN"


def address_type(address: Any) -> str:
    if looks_like_evm_address(address):
        return "EVM_0X_ADDRESS"
    if looks_like_solana_address(address):
        return "SOLANA_BASE58_ADDRESS"
    if norm(address):
        return "OTHER_INVALID_OR_UNRECOGNIZED_ADDRESS"
    return "MISSING_ADDRESS"


def wallet_route(chain_norm: str, addr_type: str) -> str:
    if chain_norm == "SOLANA" and addr_type == "SOLANA_BASE58_ADDRESS":
        return "SOLANA_HELIUS_ROUTE"
    if chain_norm == "SOLANA_ADDRESS_NO_CHAIN" and addr_type == "SOLANA_BASE58_ADDRESS":
        return "SOLANA_HELIUS_ROUTE_CHAIN_INFERRED_FROM_ADDRESS"
    if chain_norm.endswith("_EVM") or chain_norm == "EVM_UNKNOWN" or addr_type == "EVM_0X_ADDRESS":
        return "NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE"
    if addr_type == "MISSING_ADDRESS":
        return "PAIR_OR_TOKEN_ADDRESS_RESOLUTION_REQUIRED"
    return "NON_SOLANA_NON_EVM_OR_UNKNOWN_PROVIDER_REQUIRED"


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchall())


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
    preferred = ["created_at", "timestamp", "snapshot_at", "fetched_at", "updated_at", "observed_at", "inserted_at", "time"]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        cl = c.lower()
        if "time" in cl or "date" in cl or "created" in cl:
            return c
    return None


def build_symbol_chain_key(symbol: Any, chain_norm_value: Any) -> str:
    s = norm_upper(symbol)
    c = norm_upper(chain_norm_value)
    if not s or not c:
        return ""
    return f"{s}|{c}"


def add_event_key(df: pd.DataFrame, time_col: str = "candidate_event_time_utc") -> pd.DataFrame:
    out = df.copy()
    out["canonical_coin_id_num"] = pd.to_numeric(out["canonical_coin_id"], errors="coerce").astype("Int64")
    out["_event_time_dt"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    return out


def load_coins(con: sqlite3.Connection) -> pd.DataFrame:
    coins = read_table_cols(con, "coins", ["id", "symbol", "name", "chain", "pair_address"])
    if coins.empty:
        raise ValueError("Could not load coins table.")

    coins["canonical_coin_id_num"] = pd.to_numeric(coins["id"], errors="coerce").astype("Int64")
    coins["symbol_coin"] = coins.get("symbol", "").astype(str)
    coins["name_coin"] = coins.get("name", "").astype(str)
    coins["chain_coin"] = coins.get("chain", "").astype(str)
    coins["pair_address"] = coins.get("pair_address", "").astype(str).str.strip()
    coins["chain_norm"] = [normalize_chain(c, a) for c, a in zip(coins["chain_coin"], coins["pair_address"])]
    coins["pair_address_type"] = coins["pair_address"].map(address_type)
    coins["wallet_route_class"] = [wallet_route(c, t) for c, t in zip(coins["chain_norm"], coins["pair_address_type"])]
    coins["pair_address_norm"] = coins["pair_address"].map(norm_lower)
    coins["symbol_chain_key"] = [build_symbol_chain_key(s, c) for s, c in zip(coins["symbol_coin"], coins["chain_norm"])]
    return coins


def load_raw_provider(con: sqlite3.Connection, coins: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not table_exists(con, "raw_provider_payloads"):
        return pd.DataFrame(), {"exists": False}

    cols = table_cols(con, "raw_provider_payloads")
    time_col = find_time_col(cols)

    wanted = [
        "id", "symbol", "chain", "pair_address", "source", "provider",
        "created_at", "timestamp", "snapshot_at", "fetched_at", "updated_at",
        "observed_at", "inserted_at", "time", "payload"
    ]
    raw = read_table_cols(con, "raw_provider_payloads", wanted)

    if raw.empty:
        return raw, {"exists": True, "rows": 0, "time_col": time_col}

    raw["raw_symbol"] = raw.get("symbol", "").astype(str)
    raw["raw_chain"] = raw.get("chain", "").astype(str)
    raw["raw_pair_address"] = raw.get("pair_address", "").astype(str).str.strip()
    raw["raw_chain_norm"] = [normalize_chain(c, a) for c, a in zip(raw["raw_chain"], raw["raw_pair_address"])]
    raw["raw_pair_address_type"] = raw["raw_pair_address"].map(address_type)
    raw["raw_route_class"] = [wallet_route(c, t) for c, t in zip(raw["raw_chain_norm"], raw["raw_pair_address_type"])]
    raw["raw_pair_address_norm"] = raw["raw_pair_address"].map(norm_lower)
    raw["raw_symbol_chain_key"] = [build_symbol_chain_key(s, c) for s, c in zip(raw["raw_symbol"], raw["raw_chain_norm"])]

    pair_map = (
        coins[coins["pair_address_norm"].ne("")]
        .drop_duplicates("pair_address_norm")
        .set_index("pair_address_norm")["canonical_coin_id_num"]
        .to_dict()
    )

    key_counts = coins[coins["symbol_chain_key"].ne("")]["symbol_chain_key"].value_counts()
    unique_keys = set(key_counts[key_counts == 1].index)
    sym_chain_map = (
        coins[coins["symbol_chain_key"].isin(unique_keys)]
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

    # Detect potential useful fields in payload text, if present.
    if "payload" in raw.columns:
        txt = raw["payload"].astype(str).str.lower()
        raw["payload_mentions_txn"] = txt.str.contains("txn|txns|transaction|transactions|buys|sells|volume|liquidity|price", regex=True, na=False)
        raw["payload_mentions_wallet"] = txt.str.contains("wallet|holder|maker|buyer|seller|owner", regex=True, na=False)
    else:
        raw["payload_mentions_txn"] = False
        raw["payload_mentions_wallet"] = False

    meta = {
        "exists": True,
        "rows": int(len(raw)),
        "columns": cols,
        "time_col": time_col,
        "rows_resolved_to_coin": int(raw["canonical_coin_id_num"].notna().sum()),
        "rows_with_time": int(raw["raw_time_dt"].notna().sum()),
        "rows_evm_route": int(raw["raw_route_class"].eq("NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE").sum()),
        "rows_payload_mentions_txn": int(raw["payload_mentions_txn"].sum()),
        "rows_payload_mentions_wallet": int(raw["payload_mentions_wallet"].sum()),
    }
    return raw, meta


def coverage_for_events(events: pd.DataFrame, raw: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    rows = []
    raw_by_coin = {
        int(k): g.copy()
        for k, g in raw[raw["canonical_coin_id_num"].notna()].groupby("canonical_coin_id_num")
    } if not raw.empty else {}

    for _, e in events.iterrows():
        cid = e["canonical_coin_id_num"]
        g = raw_by_coin.get(int(cid), pd.DataFrame()) if pd.notna(cid) else pd.DataFrame()
        t = e["_event_time_dt"]

        row = {
            "canonical_coin_id": int(cid) if pd.notna(cid) else None,
            "symbol": e.get("symbol_coin"),
            "chain_norm": e.get("chain_norm"),
            "wallet_route_class": e.get("wallet_route_class"),
            "chronological_split": e.get("chronological_split"),
            "label_x2_sl_4h": e.get("label_x2_sl_4h"),
            "candidate_event_time_utc": str(t) if pd.notna(t) else "",
            "pair_address": e.get("pair_address"),
            "raw_provider_rows_for_coin": int(len(g)),
            "raw_provider_timed_rows_for_coin": int(g["raw_time_dt"].notna().sum()) if not g.empty else 0,
            "raw_provider_evm_route_rows_for_coin": int(g["raw_route_class"].eq("NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE").sum()) if not g.empty else 0,
            "raw_payload_txn_like_rows_for_coin": int(g["payload_mentions_txn"].sum()) if not g.empty else 0,
            "raw_payload_wallet_like_rows_for_coin": int(g["payload_mentions_wallet"].sum()) if not g.empty else 0,
        }

        for h in [24, 168]:
            if g.empty or pd.isna(t):
                row[f"raw_count_past_{h}h"] = 0
                row[f"raw_txn_like_count_past_{h}h"] = 0
                row[f"raw_wallet_like_count_past_{h}h"] = 0
            else:
                start = t - pd.to_timedelta(h, unit="h")
                x = g[(g["raw_time_dt"].notna()) & (g["raw_time_dt"] >= start) & (g["raw_time_dt"] <= t)]
                row[f"raw_count_past_{h}h"] = int(len(x))
                row[f"raw_txn_like_count_past_{h}h"] = int(x["payload_mentions_txn"].sum())
                row[f"raw_wallet_like_count_past_{h}h"] = int(x["payload_mentions_wallet"].sum())

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")
    if not DB.exists():
        raise FileNotFoundError(f"DB not found: {DB}")

    events = pd.read_csv(DATASET_CSV, low_memory=False)
    events = add_event_key(events)

    con = connect_ro()
    coins = load_coins(con)
    raw, raw_meta = load_raw_provider(con, coins)
    con.close()

    merged = events.merge(
        coins[[
            "canonical_coin_id_num", "symbol_coin", "name_coin", "chain_coin",
            "pair_address", "chain_norm", "pair_address_type", "wallet_route_class"
        ]],
        on="canonical_coin_id_num",
        how="left",
    )

    evm_events = merged[merged["wallet_route_class"].eq("NON_SOLANA_EVM_COMPATIBLE_ROUTE_POSSIBLE")].copy()
    evm_winners = evm_events[evm_events["label_x2_sl_4h"].eq("WINNER")].copy()
    non_solana = merged[~merged["wallet_route_class"].astype(str).str.startswith("SOLANA_HELIUS_ROUTE")].copy()
    non_solana_winners = non_solana[non_solana["label_x2_sl_4h"].eq("WINNER")].copy()

    # Core exports.
    evm_winners.to_csv(OUT / "01_evm_winner_events.csv", index=False, encoding="utf-8-sig")
    non_solana_winners.to_csv(OUT / "02_non_solana_winner_events.csv", index=False, encoding="utf-8-sig")

    route_label = (
        merged.groupby(["wallet_route_class", "label_x2_sl_4h"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["wallet_route_class", "label_x2_sl_4h"])
    )
    route_label.to_csv(OUT / "03_route_label_distribution_all_events.csv", index=False, encoding="utf-8-sig")

    evm_label_split = (
        evm_events.groupby(["chronological_split", "label_x2_sl_4h"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["chronological_split", "label_x2_sl_4h"])
    )
    evm_label_split.to_csv(OUT / "04_evm_label_split_distribution.csv", index=False, encoding="utf-8-sig")

    evm_chain_summary = (
        evm_events.groupby(["chain_norm", "pair_address_type"], dropna=False)
        .size()
        .reset_index(name="event_rows")
        .sort_values("event_rows", ascending=False)
    )
    evm_chain_summary.to_csv(OUT / "05_evm_chain_address_summary.csv", index=False, encoding="utf-8-sig")

    # Raw-provider coverage.
    evm_winner_raw_cov = coverage_for_events(evm_winners, raw)
    evm_winner_raw_cov.to_csv(OUT / "06_evm_winner_raw_provider_coverage.csv", index=False, encoding="utf-8-sig")

    evm_all_raw_cov = coverage_for_events(evm_events, raw)
    evm_all_raw_cov.to_csv(OUT / "07_evm_all_events_raw_provider_coverage.csv", index=False, encoding="utf-8-sig")

    evm_raw_by_label = (
        evm_all_raw_cov.groupby(["chronological_split", "label_x2_sl_4h"], dropna=False)
        .agg(
            rows=("label_x2_sl_4h", "count"),
            rows_with_raw_past_24h=("raw_count_past_24h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            rows_with_raw_past_168h=("raw_count_past_168h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            median_raw_past_24h=("raw_count_past_24h", "median"),
            median_raw_past_168h=("raw_count_past_168h", "median"),
            rows_with_txn_like_past_24h=("raw_txn_like_count_past_24h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            rows_with_wallet_like_past_24h=("raw_wallet_like_count_past_24h", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
        )
        .reset_index()
        .sort_values(["chronological_split", "label_x2_sl_4h"])
    )
    evm_raw_by_label.to_csv(OUT / "08_evm_raw_provider_coverage_by_label_split.csv", index=False, encoding="utf-8-sig")

    # Decision metrics.
    total_winners = int((merged["label_x2_sl_4h"] == "WINNER").sum())
    evm_winner_n = int(len(evm_winners))
    evm_valtest_winner_n = int(evm_winners["chronological_split"].isin(["validation", "test"]).sum())
    evm_test_winner_n = int((evm_winners["chronological_split"] == "test").sum())
    evm_validation_winner_n = int((evm_winners["chronological_split"] == "validation").sum())

    evm_winner_raw24 = int((pd.to_numeric(evm_winner_raw_cov["raw_count_past_24h"], errors="coerce") > 0).sum()) if not evm_winner_raw_cov.empty else 0
    evm_winner_raw168 = int((pd.to_numeric(evm_winner_raw_cov["raw_count_past_168h"], errors="coerce") > 0).sum()) if not evm_winner_raw_cov.empty else 0
    evm_winner_txn_like24 = int((pd.to_numeric(evm_winner_raw_cov["raw_txn_like_count_past_24h"], errors="coerce") > 0).sum()) if not evm_winner_raw_cov.empty else 0
    evm_winner_wallet_like24 = int((pd.to_numeric(evm_winner_raw_cov["raw_wallet_like_count_past_24h"], errors="coerce") > 0).sum()) if not evm_winner_raw_cov.empty else 0

    if evm_winner_n > 0 and evm_valtest_winner_n > 0 and evm_winner_raw24 == evm_winner_n:
        feasibility = "EVM_ROUTE_STRONGLY_FEASIBLE_FOR_NEXT_EXPANSION"
    elif evm_winner_n > 0 and evm_valtest_winner_n > 0 and evm_winner_raw168 > 0:
        feasibility = "EVM_ROUTE_FEASIBLE_BUT_FEATURE_QUALITY_REQUIRES_REVIEW"
    elif evm_winner_n > 0:
        feasibility = "EVM_ROUTE_HAS_WINNERS_BUT_LIMITED_COVERAGE_FOR_NEXT_EXPANSION"
    else:
        feasibility = "NO_EVM_WINNERS_FOR_EXPANSION"

    decisions = [{
        "question": "Does the EVM/non-Solana route recover WINNER rows missed by Helius?",
        "answer": "YES" if evm_winner_n > 0 else "NO",
        "classification": "EVM_RECOVERS_MISSING_WINNERS" if evm_winner_n > 0 else "NO_EVM_WINNER_RECOVERY",
        "evidence": f"evm_winners={evm_winner_n}; total_winners={total_winners}",
    }, {
        "question": "Does the EVM route recover validation/test WINNER rows?",
        "answer": "YES" if evm_valtest_winner_n > 0 else "NO",
        "classification": "EVM_RESTORES_CHRONOLOGICAL_WINNER_COVERAGE" if evm_valtest_winner_n > 0 else "NO_EVM_VALIDATION_TEST_WINNERS",
        "evidence": f"validation_winners={evm_validation_winner_n}; test_winners={evm_test_winner_n}",
    }, {
        "question": "Is there raw-provider support for EVM WINNER events?",
        "answer": "YES" if evm_winner_raw24 > 0 else "NO",
        "classification": "RAW_PROVIDER_SUPPORTS_EVM_EXPANSION" if evm_winner_raw24 > 0 else "RAW_PROVIDER_NOT_ENOUGH_FOR_EVM_EXPANSION",
        "evidence": f"raw24={evm_winner_raw24}/{evm_winner_n}; raw168={evm_winner_raw168}/{evm_winner_n}; txn_like24={evm_winner_txn_like24}; wallet_like24={evm_winner_wallet_like24}",
    }, {
        "question": "Can this audit alone prove EVM wallet-flow predictive signal?",
        "answer": "NO",
        "classification": "DESIGN_FEASIBILITY_ONLY",
        "evidence": "No EVM transaction-level provider calls were made; this audit only maps feasibility and raw-provider coverage.",
    }, {
        "question": "Recommended next step",
        "answer": feasibility,
        "classification": feasibility,
        "evidence": "Proceed only with read-only EVM-specific provider expansion if provider/API route is available.",
    }]
    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "09_evm_feasibility_decision_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "classification": "THESIS_EVM_NON_SOLANA_WALLET_FLOW_FEASIBILITY_DEEP_AUDIT_COMPLETED",
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
            "three_e_root": str(THREE_E_ROOT),
        },
        "raw_provider_meta": raw_meta,
        "key_counts": {
            "total_winner_rows": total_winners,
            "evm_event_rows": int(len(evm_events)),
            "evm_winner_rows": evm_winner_n,
            "evm_validation_winner_rows": evm_validation_winner_n,
            "evm_test_winner_rows": evm_test_winner_n,
            "evm_validation_test_winner_rows": evm_valtest_winner_n,
            "evm_winner_rows_with_raw_past24h": evm_winner_raw24,
            "evm_winner_rows_with_raw_past168h": evm_winner_raw168,
            "evm_winner_rows_with_txn_like_raw_past24h": evm_winner_txn_like24,
            "evm_winner_rows_with_wallet_like_raw_past24h": evm_winner_wallet_like24,
        },
        "final_scientific_conclusion": feasibility,
        "outputs": {
            "evm_winner_events": str(OUT / "01_evm_winner_events.csv"),
            "non_solana_winner_events": str(OUT / "02_non_solana_winner_events.csv"),
            "route_label_distribution_all_events": str(OUT / "03_route_label_distribution_all_events.csv"),
            "evm_label_split_distribution": str(OUT / "04_evm_label_split_distribution.csv"),
            "evm_chain_address_summary": str(OUT / "05_evm_chain_address_summary.csv"),
            "evm_winner_raw_provider_coverage": str(OUT / "06_evm_winner_raw_provider_coverage.csv"),
            "evm_all_events_raw_provider_coverage": str(OUT / "07_evm_all_events_raw_provider_coverage.csv"),
            "evm_raw_provider_coverage_by_label_split": str(OUT / "08_evm_raw_provider_coverage_by_label_split.csv"),
            "evm_feasibility_decision_table": str(OUT / "09_evm_feasibility_decision_table.csv"),
            "summary_json": str(OUT / "thesis_evm_non_solana_wallet_flow_feasibility_deep_summary.json"),
            "summary_md": str(OUT / "thesis_evm_non_solana_wallet_flow_feasibility_deep_summary.md"),
        },
    }

    summary_json = OUT / "thesis_evm_non_solana_wallet_flow_feasibility_deep_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3F — EVM / Non-Solana Wallet-flow Feasibility Deep Audit")
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
    lines.append("## Key counts")
    for k, v in summary["key_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Raw-provider metadata")
    for k, v in raw_meta.items():
        if k == "columns":
            lines.append(f"- `{k}`: {len(v)} columns")
        else:
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## EVM label/split distribution")
    for _, r in evm_label_split.iterrows():
        lines.append(f"- `{r['chronological_split']}` / `{r['label_x2_sl_4h']}`: {int(r['rows'])}")
    lines.append("")
    lines.append("## EVM raw-provider coverage by label/split")
    for _, r in evm_raw_by_label.iterrows():
        lines.append(
            f"- `{r['chronological_split']}` / `{r['label_x2_sl_4h']}`: "
            f"rows={int(r['rows'])}; raw24={int(r['rows_with_raw_past_24h'])}; "
            f"raw168={int(r['rows_with_raw_past_168h'])}; "
            f"txn_like24={int(r['rows_with_txn_like_past_24h'])}; "
            f"wallet_like24={int(r['rows_with_wallet_like_past_24h'])}"
        )
    lines.append("")
    lines.append("## Decision table")
    for _, r in decision_df.iterrows():
        lines.append(f"- {r['question']} `{r['classification']}` — {r['evidence']}")
    lines.append("")
    lines.append("## Final scientific conclusion")
    lines.append(f"`{feasibility}`")
    lines.append("")
    lines.append("## Output files")
    for _, p in summary["outputs"].items():
        lines.append(f"- `{Path(p).name}`")

    summary_md = OUT / "thesis_evm_non_solana_wallet_flow_feasibility_deep_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": feasibility,
        "evm_winner_rows": evm_winner_n,
        "evm_validation_test_winner_rows": evm_valtest_winner_n,
        "evm_winner_rows_with_raw_past24h": evm_winner_raw24,
        "evm_winner_rows_with_raw_past168h": evm_winner_raw168,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
