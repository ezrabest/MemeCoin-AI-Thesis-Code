from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
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

THREE_F_ROOT = Path(os.environ.get(
    "THESIS_3F_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_evm_non_solana_wallet_flow_feasibility_deep_audit_20260810_231702"
))

OUT = ROOT / "data" / "audits" / f"thesis_evm_chain_resolution_provider_readiness_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"
ENV_FILE = ROOT / ".env"


CHAIN_PATTERNS = {
    "BASE_EVM": [
        r'\bbase\b',
        r'"chainid"\s*:\s*"base"',
        r"'chainid'\s*:\s*'base'",
    ],
    "BSC_EVM": [
        r'\bbsc\b',
        r'\bbnb\b',
        r'\bbinance\b',
        r'"chainid"\s*:\s*"bsc"',
        r'"chainid"\s*:\s*"bnb"',
    ],
    "ETHEREUM_EVM": [
        r'\bethereum\b',
        r'\beth\b',
        r'\berc20\b',
        r'"chainid"\s*:\s*"ethereum"',
        r'"chainid"\s*:\s*"eth"',
    ],
    "ARBITRUM_EVM": [
        r'\barbitrum\b',
        r'\barb\b',
        r'"chainid"\s*:\s*"arbitrum"',
    ],
    "POLYGON_EVM": [
        r'\bpolygon\b',
        r'\bmatic\b',
        r'"chainid"\s*:\s*"polygon"',
    ],
    "AVALANCHE_EVM": [
        r'\bavalanche\b',
        r'\bavax\b',
        r'"chainid"\s*:\s*"avalanche"',
    ],
    "OPTIMISM_EVM": [
        r'\boptimism\b',
        r'\bop mainnet\b',
        r'"chainid"\s*:\s*"optimism"',
    ],
}


KEY_GROUPS = {
    "ETHERSCAN_OR_ETHEREUM_EXPLORER": [
        "ETHERSCAN_API_KEY",
        "ETH_API_KEY",
        "ETHEREUM_API_KEY",
    ],
    "BSCSCAN": [
        "BSCSCAN_API_KEY",
        "BSC_SCAN_API_KEY",
        "BNBSCAN_API_KEY",
    ],
    "BASESCAN": [
        "BASESCAN_API_KEY",
        "BASE_SCAN_API_KEY",
    ],
    "ARBISCAN": [
        "ARBISCAN_API_KEY",
        "ARBITRUMSCAN_API_KEY",
    ],
    "POLYGONSCAN": [
        "POLYGONSCAN_API_KEY",
        "POLYGON_SCAN_API_KEY",
    ],
    "SNOWTRACE_OR_AVALANCHE": [
        "SNOWTRACE_API_KEY",
        "AVALANCHE_API_KEY",
        "AVAX_API_KEY",
    ],
    "OPTIMISM_EXPLORER": [
        "OPTIMISTIC_ETHERSCAN_API_KEY",
        "OPTIMISM_API_KEY",
        "OPSCAN_API_KEY",
    ],
    "GENERAL_EVM_RPC_OR_INDEXER": [
        "ALCHEMY_API_KEY",
        "ALCHEMY_ETH_API_KEY",
        "ALCHEMY_BASE_API_KEY",
        "INFURA_API_KEY",
        "INFURA_PROJECT_ID",
        "QUICKNODE_API_KEY",
        "MORALIS_API_KEY",
        "COVALENT_API_KEY",
        "BITQUERY_API_KEY",
        "ANKR_API_KEY",
        "DRPC_API_KEY",
    ],
    "DEXSCREENER_OR_MARKET_ONLY": [
        "DEXSCREENER_API_KEY",
    ],
    "HELIUS_SOLANA_ONLY": [
        "HELIUS_API_KEY",
        "HELIUS_KEY",
        "HELIUS_RPC_API_KEY",
        "SOLANA_HELIUS_API_KEY",
    ],
}


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
    if "OPTIMISM" in c or c == "OP":
        return "OPTIMISM_EVM"
    if looks_like_evm_address(a):
        return "EVM_UNKNOWN"
    if looks_like_solana_address(a):
        return "SOLANA_ADDRESS_NO_CHAIN"
    if c:
        return "OTHER_OR_UNKNOWN_CHAIN_" + c.replace(" ", "_")[:50]
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


def query_table_by_keys(
    con: sqlite3.Connection,
    table: str,
    wanted_cols: list[str],
    coin_ids: list[int],
    pair_addresses: list[str],
    symbols: list[str],
) -> pd.DataFrame:
    cols = table_cols(con, table)
    if not cols:
        return pd.DataFrame()

    use = [c for c in wanted_cols if c in cols]
    if not use:
        return pd.DataFrame()

    where_parts = []
    params: list[Any] = []

    if "coin_id" in cols and coin_ids:
        ph = ",".join(["?"] * len(coin_ids))
        where_parts.append(f'"coin_id" IN ({ph})')
        params.extend(coin_ids)

    if "pair_address" in cols and pair_addresses:
        ph = ",".join(["?"] * len(pair_addresses))
        where_parts.append(f'LOWER("pair_address") IN ({ph})')
        params.extend([p.lower() for p in pair_addresses])

    if "symbol" in cols and symbols:
        ph = ",".join(["?"] * len(symbols))
        where_parts.append(f'UPPER("symbol") IN ({ph})')
        params.extend([s.upper() for s in symbols])

    if not where_parts:
        return pd.DataFrame()

    sql = (
        f'SELECT {", ".join(f"""\"{c}\"""" for c in use)} '
        f'FROM "{table}" WHERE ' + " OR ".join(where_parts)
    )

    try:
        df = pd.read_sql_query(sql, con, params=params)
    except Exception:
        return pd.DataFrame()

    df["_source_table"] = table
    return df


def parse_env_keys() -> dict[str, str]:
    keys = {}
    if not ENV_FILE.exists():
        return keys

    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^\s*(?:export\s+)?([^=\s]+)\s*=\s*(.*)\s*$', line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        keys[k] = v
    return keys


def extract_text_chain_hints(text: str) -> list[str]:
    s = norm_lower(text)
    if not s:
        return []

    hints = []
    for chain, patterns in CHAIN_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                hints.append(chain)
                break
    return hints


def load_coins(con: sqlite3.Connection) -> pd.DataFrame:
    cols = table_cols(con, "coins")
    wanted = ["id", "symbol", "name", "chain", "pair_address"]
    use = [c for c in wanted if c in cols]
    sql = ", ".join(f'"{c}"' for c in use)
    coins = pd.read_sql_query(f'SELECT {sql} FROM "coins"', con)

    coins["canonical_coin_id_num"] = pd.to_numeric(coins["id"], errors="coerce").astype("Int64")
    coins["symbol_coin"] = coins.get("symbol", "").astype(str)
    coins["name_coin"] = coins.get("name", "").astype(str)
    coins["chain_coin"] = coins.get("chain", "").astype(str)
    coins["pair_address"] = coins.get("pair_address", "").astype(str).str.strip()
    coins["pair_address_norm"] = coins["pair_address"].map(norm_lower)
    coins["chain_norm"] = [normalize_chain(c, a) for c, a in zip(coins["chain_coin"], coins["pair_address"])]
    coins["pair_address_type"] = coins["pair_address"].map(address_type)
    coins["wallet_route_class"] = [wallet_route(c, a) for c, a in zip(coins["chain_norm"], coins["pair_address_type"])]
    return coins


def standardize_hint_rows(df: pd.DataFrame, table: str, evm_coin_ids: set[int], pair_to_coin: dict[str, int], symbol_to_coin: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["_source_table"] = table

    if "coin_id" in out.columns:
        out["_hint_coin_id"] = pd.to_numeric(out["coin_id"], errors="coerce").astype("Int64")
    else:
        out["_hint_coin_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if "pair_address" in out.columns:
        out["_pair_norm"] = out["pair_address"].astype(str).map(norm_lower)
        out["_hint_coin_id_pair"] = out["_pair_norm"].map(pair_to_coin)
        out["_hint_coin_id"] = out["_hint_coin_id"].combine_first(pd.to_numeric(out["_hint_coin_id_pair"], errors="coerce").astype("Int64"))

    if "symbol" in out.columns:
        out["_symbol_norm"] = out["symbol"].astype(str).str.upper().str.strip()
        out["_hint_coin_id_symbol"] = out["_symbol_norm"].map(symbol_to_coin)
        out["_hint_coin_id"] = out["_hint_coin_id"].combine_first(pd.to_numeric(out["_hint_coin_id_symbol"], errors="coerce").astype("Int64"))

    out = out[out["_hint_coin_id"].notna()].copy()
    out["_hint_coin_id_int"] = out["_hint_coin_id"].astype(int)
    out = out[out["_hint_coin_id_int"].isin(evm_coin_ids)].copy()

    text_cols = [c for c in [
        "chain", "source", "provider", "source_query", "payload", "raw_payload", "url", "metadata"
    ] if c in out.columns]

    rows = []
    for _, r in out.iterrows():
        chain_value = r.get("chain", "")
        addr_value = r.get("pair_address", "")
        explicit_norm = normalize_chain(chain_value, addr_value)

        combined_text = " ".join(norm(r.get(c, ""))[:2000] for c in text_cols)
        text_hints = extract_text_chain_hints(combined_text)

        hint_values = []
        if explicit_norm not in {"UNKNOWN_CHAIN", "EVM_UNKNOWN"} and not explicit_norm.startswith("OTHER_OR_UNKNOWN"):
            hint_values.append(explicit_norm)
        hint_values.extend(text_hints)

        if not hint_values:
            hint_values = ["NO_SPECIFIC_CHAIN_HINT"]

        for h in hint_values:
            rows.append({
                "canonical_coin_id": int(r["_hint_coin_id_int"]),
                "source_table": table,
                "chain_hint": h,
                "raw_chain_value": norm(chain_value),
                "pair_address": norm(addr_value),
                "source": norm(r.get("source", "")),
                "provider": norm(r.get("provider", "")),
                "source_query": norm(r.get("source_query", ""))[:300],
            })

    return pd.DataFrame(rows)


def provider_group_inventory(env_keys: dict[str, str]) -> pd.DataFrame:
    rows = []
    for group, keys in KEY_GROUPS.items():
        present = [k for k in keys if k in env_keys and norm(env_keys.get(k))]
        rows.append({
            "provider_key_group": group,
            "present": bool(present),
            "present_key_names": ",".join(present),
            "candidate_key_names_checked": ",".join(keys),
        })
    return pd.DataFrame(rows)


def group_present(inv: pd.DataFrame, group: str) -> bool:
    x = inv[inv["provider_key_group"] == group]
    if x.empty:
        return False
    return bool(x.iloc[0]["present"])


def readiness_for_chain(chain: str, inv: pd.DataFrame) -> tuple[str, str]:
    generic = group_present(inv, "GENERAL_EVM_RPC_OR_INDEXER")

    if chain == "ETHEREUM_EVM":
        specific = group_present(inv, "ETHERSCAN_OR_ETHEREUM_EXPLORER")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Ethereum/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Ethereum/generic EVM key detected in .env."

    if chain == "BSC_EVM":
        specific = group_present(inv, "BSCSCAN")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "BSC/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No BSC/generic EVM key detected in .env."

    if chain == "BASE_EVM":
        specific = group_present(inv, "BASESCAN")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Base/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Base/generic EVM key detected in .env."

    if chain == "ARBITRUM_EVM":
        specific = group_present(inv, "ARBISCAN")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Arbitrum/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Arbitrum/generic EVM key detected in .env."

    if chain == "POLYGON_EVM":
        specific = group_present(inv, "POLYGONSCAN")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Polygon/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Polygon/generic EVM key detected in .env."

    if chain == "AVALANCHE_EVM":
        specific = group_present(inv, "SNOWTRACE_OR_AVALANCHE")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Avalanche/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Avalanche/generic EVM key detected in .env."

    if chain == "OPTIMISM_EVM":
        specific = group_present(inv, "OPTIMISM_EXPLORER")
        if specific or generic:
            return "CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT", "Optimism/EVM route appears key-ready."
        return "PROVIDER_KEY_NOT_FOUND", "No Optimism/generic EVM key detected in .env."

    if chain == "EVM_UNKNOWN_UNRESOLVED":
        return "CHAIN_RESOLUTION_REQUIRED_BEFORE_PROVIDER_CALLS", "0x address alone is not enough to choose an EVM network safely."

    if chain == "AMBIGUOUS_EVM_CHAIN":
        return "AMBIGUOUS_CHAIN_REQUIRES_MANUAL_OR_DEX_RESOLUTION", "Multiple chain hints conflict."

    return "PROVIDER_ROUTE_UNKNOWN", "Provider route not inferable."


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")
    if not DB.exists():
        raise FileNotFoundError(f"DB not found: {DB}")

    three_f_evm = THREE_F_ROOT / "01_evm_winner_events.csv"
    if not three_f_evm.exists():
        raise FileNotFoundError(f"3F EVM winner file not found: {three_f_evm}")

    evm_winners = pd.read_csv(three_f_evm, low_memory=False)
    evm_winners["canonical_coin_id_num"] = pd.to_numeric(evm_winners["canonical_coin_id_num"], errors="coerce").astype("Int64")
    evm_winners["_event_time_dt"] = pd.to_datetime(evm_winners["candidate_event_time_utc"], errors="coerce", utc=True)

    con = connect_ro()
    coins = load_coins(con)

    evm_coin_ids = set(int(x) for x in evm_winners["canonical_coin_id_num"].dropna().unique())
    evm_pairs = sorted(set(norm_lower(x) for x in evm_winners.get("pair_address", pd.Series(dtype=str)).dropna() if norm(x)))
    evm_symbols = sorted(set(norm_upper(x) for x in evm_winners.get("symbol_coin", pd.Series(dtype=str)).dropna() if norm(x)))

    pair_to_coin = {}
    for _, r in evm_winners.iterrows():
        p = norm_lower(r.get("pair_address"))
        cid = r.get("canonical_coin_id_num")
        if p and pd.notna(cid):
            pair_to_coin[p] = int(cid)

    symbol_counts = evm_winners["symbol_coin"].astype(str).str.upper().str.strip().value_counts()
    unique_symbols = set(symbol_counts[symbol_counts == 1].index)
    symbol_to_coin = {}
    for _, r in evm_winners.iterrows():
        s = norm_upper(r.get("symbol_coin"))
        cid = r.get("canonical_coin_id_num")
        if s in unique_symbols and pd.notna(cid):
            symbol_to_coin[s] = int(cid)

    table_specs = {
        "raw_provider_payloads": [
            "id", "coin_id", "symbol", "chain", "pair_address", "source",
            "provider", "source_query", "payload", "raw_payload", "metadata",
            "timestamp", "created_at", "fetched_at", "observed_at"
        ],
        "market_snapshots": [
            "id", "coin_id", "symbol", "chain", "pair_address", "source",
            "provider", "source_query", "timestamp", "created_at", "snapshot_at"
        ],
        "signals": [
            "id", "coin_id", "symbol", "chain", "pair_address", "source",
            "provider", "source_query", "timestamp", "created_at"
        ],
        "paper_trades": [
            "id", "coin_id", "symbol", "chain", "pair_address", "source",
            "provider", "source_query", "timestamp", "created_at", "opened_at", "closed_at"
        ],
        "whale_alerts": [
            "id", "coin_id", "symbol", "chain", "pair_address", "source",
            "provider", "source_query", "timestamp", "created_at"
        ],
    }

    table_inventory = []
    hint_frames = []

    for table, wanted in table_specs.items():
        cols = table_cols(con, table)
        exists = bool(cols)
        table_inventory.append({
            "table": table,
            "exists": exists,
            "columns": ",".join(cols),
            "columns_count": len(cols),
        })

        if not exists:
            continue

        df = query_table_by_keys(
            con=con,
            table=table,
            wanted_cols=wanted,
            coin_ids=sorted(evm_coin_ids),
            pair_addresses=evm_pairs,
            symbols=evm_symbols,
        )
        table_inventory[-1]["matched_rows_loaded"] = int(len(df))

        hints = standardize_hint_rows(
            df,
            table=table,
            evm_coin_ids=evm_coin_ids,
            pair_to_coin=pair_to_coin,
            symbol_to_coin=symbol_to_coin,
        )
        if not hints.empty:
            hint_frames.append(hints)

    con.close()

    table_inventory_df = pd.DataFrame(table_inventory)
    table_inventory_df.to_csv(OUT / "00_table_inventory_and_matched_rows.csv", index=False, encoding="utf-8-sig")

    all_hints = pd.concat(hint_frames, ignore_index=True) if hint_frames else pd.DataFrame(
        columns=["canonical_coin_id", "source_table", "chain_hint"]
    )
    all_hints.to_csv(OUT / "01_evm_chain_hint_rows.csv", index=False, encoding="utf-8-sig")

    # Resolve per canonical coin.
    resolution_rows = []
    for cid, g in all_hints.groupby("canonical_coin_id", dropna=False):
        hint_counts = Counter(g["chain_hint"].astype(str))
        specific = {
            k: v for k, v in hint_counts.items()
            if k.endswith("_EVM")
            and k not in {"EVM_UNKNOWN"}
            and "UNKNOWN" not in k
        }

        if specific:
            top_count = max(specific.values())
            top = sorted([k for k, v in specific.items() if v == top_count])
            if len(top) == 1:
                resolved = top[0]
                status = "RESOLVED_SPECIFIC_EVM_CHAIN"
            else:
                resolved = "AMBIGUOUS_EVM_CHAIN"
                status = "AMBIGUOUS_SPECIFIC_CHAIN_HINTS"
        else:
            resolved = "EVM_UNKNOWN_UNRESOLVED"
            status = "NO_SPECIFIC_CHAIN_HINT"

        source_table_counts = Counter(g["source_table"].astype(str))

        resolution_rows.append({
            "canonical_coin_id": int(cid),
            "resolved_chain": resolved,
            "resolution_status": status,
            "hint_counts_json": json.dumps(dict(hint_counts), ensure_ascii=False),
            "source_table_counts_json": json.dumps(dict(source_table_counts), ensure_ascii=False),
            "specific_hint_counts_json": json.dumps(specific, ensure_ascii=False),
            "hint_rows": int(len(g)),
        })

    # Ensure all EVM winner coins appear even if no hints.
    existing_cids = set(int(r["canonical_coin_id"]) for r in resolution_rows)
    for cid in sorted(evm_coin_ids - existing_cids):
        resolution_rows.append({
            "canonical_coin_id": int(cid),
            "resolved_chain": "EVM_UNKNOWN_UNRESOLVED",
            "resolution_status": "NO_HINT_ROWS_LOADED",
            "hint_counts_json": "{}",
            "source_table_counts_json": "{}",
            "specific_hint_counts_json": "{}",
            "hint_rows": 0,
        })

    resolution = pd.DataFrame(resolution_rows)

    coin_context = evm_winners[[
        "canonical_coin_id_num", "symbol_coin", "name_coin", "pair_address",
        "chronological_split", "label_x2_sl_4h", "candidate_event_time_utc"
    ]].copy()
    coin_context = coin_context.rename(columns={"canonical_coin_id_num": "canonical_coin_id"})

    winner_resolution = coin_context.merge(resolution, on="canonical_coin_id", how="left")
    winner_resolution.to_csv(OUT / "02_evm_winner_chain_resolution.csv", index=False, encoding="utf-8-sig")

    resolution_summary = (
        winner_resolution.groupby(["resolved_chain", "resolution_status", "chronological_split"], dropna=False)
        .size()
        .reset_index(name="winner_rows")
        .sort_values(["resolved_chain", "chronological_split"])
    )
    resolution_summary.to_csv(OUT / "03_resolution_summary_by_split.csv", index=False, encoding="utf-8-sig")

    # Env/provider readiness.
    env_keys = parse_env_keys()
    env_inventory = provider_group_inventory(env_keys)
    env_inventory.to_csv(OUT / "04_env_provider_key_inventory_names_only.csv", index=False, encoding="utf-8-sig")

    readiness_rows = []
    for chain, g in winner_resolution.groupby("resolved_chain", dropna=False):
        status, note = readiness_for_chain(str(chain), env_inventory)
        readiness_rows.append({
            "resolved_chain": chain,
            "winner_rows": int(len(g)),
            "validation_winner_rows": int((g["chronological_split"] == "validation").sum()),
            "test_winner_rows": int((g["chronological_split"] == "test").sum()),
            "provider_readiness": status,
            "note": note,
        })

    readiness = pd.DataFrame(readiness_rows).sort_values("winner_rows", ascending=False)
    readiness.to_csv(OUT / "05_chain_provider_readiness.csv", index=False, encoding="utf-8-sig")

    unresolved = winner_resolution[winner_resolution["resolved_chain"].isin(["EVM_UNKNOWN_UNRESOLVED", "AMBIGUOUS_EVM_CHAIN"])].copy()
    unresolved.to_csv(OUT / "06_unresolved_or_ambiguous_evm_winner_rows.csv", index=False, encoding="utf-8-sig")

    # Decision table.
    total = int(len(winner_resolution))
    resolved_specific = int(winner_resolution["resolution_status"].eq("RESOLVED_SPECIFIC_EVM_CHAIN").sum())
    unresolved_n = int(winner_resolution["resolved_chain"].eq("EVM_UNKNOWN_UNRESOLVED").sum())
    ambiguous_n = int(winner_resolution["resolved_chain"].eq("AMBIGUOUS_EVM_CHAIN").sum())
    valtest_total = int(winner_resolution["chronological_split"].isin(["validation", "test"]).sum())
    valtest_resolved = int(
        winner_resolution["chronological_split"].isin(["validation", "test"]).sum()
        and len(winner_resolution[
            winner_resolution["chronological_split"].isin(["validation", "test"])
            & winner_resolution["resolution_status"].eq("RESOLVED_SPECIFIC_EVM_CHAIN")
        ])
    )

    key_ready_winner_rows = int(readiness[
        readiness["provider_readiness"].eq("CHAIN_SPECIFIC_OR_GENERIC_PROVIDER_KEY_PRESENT")
    ]["winner_rows"].sum()) if not readiness.empty else 0

    if resolved_specific > 0 and key_ready_winner_rows > 0 and valtest_resolved > 0:
        final = "EVM_CHAIN_AND_PROVIDER_READY_FOR_READ_ONLY_EXPANSION"
    elif resolved_specific > 0 and valtest_resolved > 0:
        final = "EVM_CHAIN_RESOLVED_BUT_PROVIDER_KEY_READINESS_INCOMPLETE"
    elif unresolved_n == total:
        final = "EVM_ROUTE_FEASIBLE_BUT_CHAIN_RESOLUTION_REQUIRED_BEFORE_PROVIDER_CALLS"
    else:
        final = "EVM_ROUTE_PARTIALLY_RESOLVED_REQUIRES_TARGETED_CHAIN_RESOLUTION"

    decisions = [{
        "question": "Can EVM winner rows be assigned to a specific EVM chain from existing DB metadata?",
        "answer": "YES" if resolved_specific > 0 else "NO",
        "classification": "SPECIFIC_CHAIN_RESOLUTION_AVAILABLE" if resolved_specific > 0 else "NO_SPECIFIC_CHAIN_RESOLUTION_FROM_EXISTING_METADATA",
        "evidence": f"resolved_specific={resolved_specific}/{total}; unresolved={unresolved_n}; ambiguous={ambiguous_n}",
    }, {
        "question": "Can validation/test EVM winners be chain-resolved?",
        "answer": "YES" if valtest_resolved > 0 else "NO",
        "classification": "VALIDATION_TEST_EVM_WINNERS_CHAIN_RESOLVED" if valtest_resolved > 0 else "VALIDATION_TEST_EVM_WINNERS_NOT_CHAIN_RESOLVED",
        "evidence": f"valtest_resolved={valtest_resolved}/{valtest_total}",
    }, {
        "question": "Are provider/API keys available in .env for resolved EVM routes?",
        "answer": "YES" if key_ready_winner_rows > 0 else "NO_OR_NOT_APPLICABLE",
        "classification": "PROVIDER_KEY_READY_FOR_SOME_EVM_WINNERS" if key_ready_winner_rows > 0 else "NO_PROVIDER_KEY_READY_OR_NO_RESOLVED_CHAIN",
        "evidence": f"key_ready_winner_rows={key_ready_winner_rows}/{total}; key names only are listed in env inventory.",
    }, {
        "question": "Should EVM transaction-provider calls be made now?",
        "answer": "YES" if final == "EVM_CHAIN_AND_PROVIDER_READY_FOR_READ_ONLY_EXPANSION" else "NO",
        "classification": final,
        "evidence": "Do not make EVM provider calls until chain resolution and provider readiness are both sufficient.",
    }]
    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "07_evm_chain_resolution_decision_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "classification": "THESIS_EVM_CHAIN_RESOLUTION_PROVIDER_READINESS_AUDIT_COMPLETED",
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
            "env_values_printed": False,
        },
        "inputs": {
            "dataset": str(DATASET_CSV),
            "three_f_root": str(THREE_F_ROOT),
            "db": str(DB),
            "env_file": str(ENV_FILE),
        },
        "counts": {
            "evm_winner_rows": total,
            "resolved_specific_chain_winner_rows": resolved_specific,
            "unresolved_evm_winner_rows": unresolved_n,
            "ambiguous_evm_winner_rows": ambiguous_n,
            "validation_test_evm_winner_rows": valtest_total,
            "validation_test_chain_resolved_winner_rows": valtest_resolved,
            "provider_key_ready_winner_rows": key_ready_winner_rows,
            "hint_rows": int(len(all_hints)),
        },
        "final_scientific_conclusion": final,
        "outputs": {
            "table_inventory": str(OUT / "00_table_inventory_and_matched_rows.csv"),
            "chain_hint_rows": str(OUT / "01_evm_chain_hint_rows.csv"),
            "winner_chain_resolution": str(OUT / "02_evm_winner_chain_resolution.csv"),
            "resolution_summary_by_split": str(OUT / "03_resolution_summary_by_split.csv"),
            "env_provider_key_inventory": str(OUT / "04_env_provider_key_inventory_names_only.csv"),
            "chain_provider_readiness": str(OUT / "05_chain_provider_readiness.csv"),
            "unresolved_or_ambiguous_winners": str(OUT / "06_unresolved_or_ambiguous_evm_winner_rows.csv"),
            "decision_table": str(OUT / "07_evm_chain_resolution_decision_table.csv"),
            "summary_json": str(OUT / "thesis_evm_chain_resolution_provider_readiness_summary.json"),
            "summary_md": str(OUT / "thesis_evm_chain_resolution_provider_readiness_summary.md"),
        },
    }

    summary_json = OUT / "thesis_evm_chain_resolution_provider_readiness_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3G — EVM Chain Resolution / Provider Readiness")
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
    lines.append("- .env values were not printed")
    lines.append("")
    lines.append("## Counts")
    for k, v in summary["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Resolution summary by split")
    for _, r in resolution_summary.iterrows():
        lines.append(
            f"- `{r['resolved_chain']}` / `{r['resolution_status']}` / `{r['chronological_split']}`: {int(r['winner_rows'])}"
        )
    lines.append("")
    lines.append("## Provider key inventory")
    for _, r in env_inventory.iterrows():
        lines.append(
            f"- `{r['provider_key_group']}`: present={bool(r['present'])}; keys=`{r['present_key_names']}`"
        )
    lines.append("")
    lines.append("## Chain provider readiness")
    for _, r in readiness.iterrows():
        lines.append(
            f"- `{r['resolved_chain']}`: winners={int(r['winner_rows'])}; "
            f"validation={int(r['validation_winner_rows'])}; test={int(r['test_winner_rows'])}; "
            f"readiness=`{r['provider_readiness']}`; note={r['note']}"
        )
    lines.append("")
    lines.append("## Decision table")
    for _, r in decision_df.iterrows():
        lines.append(f"- {r['question']} `{r['classification']}` — {r['evidence']}")
    lines.append("")
    lines.append("## Final scientific conclusion")
    lines.append(f"`{final}`")
    lines.append("")
    lines.append("## Output files")
    for _, p in summary["outputs"].items():
        lines.append(f"- `{Path(p).name}`")

    summary_md = OUT / "thesis_evm_chain_resolution_provider_readiness_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": final,
        "evm_winner_rows": total,
        "resolved_specific_chain_winner_rows": resolved_specific,
        "unresolved_evm_winner_rows": unresolved_n,
        "validation_test_evm_winner_rows": valtest_total,
        "validation_test_chain_resolved_winner_rows": valtest_resolved,
        "provider_key_ready_winner_rows": key_ready_winner_rows,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
