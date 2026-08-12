from __future__ import annotations

import hashlib
import json
import math
import os
import re
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

THREE_F_ROOT = Path(os.environ.get(
    "THESIS_3F_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_evm_non_solana_wallet_flow_feasibility_deep_audit_20260810_231702"
))

THREE_G_ROOT = Path(os.environ.get(
    "THESIS_3G_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_evm_chain_resolution_provider_readiness_audit_20260810_232055"
))

OUT = ROOT / "data" / "audits" / f"thesis_dexscreener_evm_chain_resolution_audit_{STAMP}"
CACHE = OUT / "dexscreener_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

EVM_WINNERS_CSV = THREE_F_ROOT / "01_evm_winner_events.csv"
ENV_FILE = ROOT / ".env"

CHAIN_IDS = [
    x.strip()
    for x in os.environ.get(
        "THESIS_EVM_CHAIN_IDS",
        "ethereum,bsc,base,arbitrum,polygon,avalanche,optimism,linea,blast,scroll,fantom,cronos,gnosis,mantle,celo,moonbeam,opbnb"
    ).split(",")
    if x.strip()
]

SLEEP_SECONDS = float(os.environ.get("THESIS_DEXSCREENER_SLEEP_SECONDS", "0.20"))
TIMEOUT_SECONDS = int(os.environ.get("THESIS_DEXSCREENER_TIMEOUT_SECONDS", "30"))


KEY_GROUPS = {
    "ETHERSCAN_OR_ETHEREUM_EXPLORER": [
        "ETHERSCAN_API_KEY", "ETH_API_KEY", "ETHEREUM_API_KEY",
    ],
    "BSCSCAN": [
        "BSCSCAN_API_KEY", "BSC_SCAN_API_KEY", "BNBSCAN_API_KEY",
    ],
    "BASESCAN": [
        "BASESCAN_API_KEY", "BASE_SCAN_API_KEY",
    ],
    "ARBISCAN": [
        "ARBISCAN_API_KEY", "ARBITRUMSCAN_API_KEY",
    ],
    "POLYGONSCAN": [
        "POLYGONSCAN_API_KEY", "POLYGON_SCAN_API_KEY",
    ],
    "GENERAL_EVM_RPC_OR_INDEXER": [
        "ALCHEMY_API_KEY", "INFURA_API_KEY", "INFURA_PROJECT_ID",
        "QUICKNODE_API_KEY", "MORALIS_API_KEY", "COVALENT_API_KEY",
        "BITQUERY_API_KEY", "ANKR_API_KEY", "DRPC_API_KEY",
    ],
    "HELIUS_SOLANA_ONLY": [
        "HELIUS_API_KEY", "HELIUS_KEY", "HELIUS_RPC_API_KEY", "SOLANA_HELIUS_API_KEY",
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


def looks_like_evm_address(x: Any) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", norm(x)))


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


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


def get_json_cached(url: str, cache_prefix: str) -> tuple[Any, dict[str, Any]]:
    cache_file = CACHE / f"{cache_prefix}_{sha(url)}.json"

    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return payload.get("data"), {
                "url": url,
                "cache_status": "HIT",
                "http_status": payload.get("http_status"),
                "error": payload.get("error", ""),
            }
        except Exception as exc:
            return None, {
                "url": url,
                "cache_status": "BAD_CACHE",
                "http_status": None,
                "error": repr(exc),
            }

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "memecoin-thesis-dexscreener-chain-resolution/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    data = None
    status = None
    error = ""

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                status = int(resp.status)
                body = resp.read().decode("utf-8")
                data = json.loads(body) if body else None
            break
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = exc.read().decode("utf-8", errors="ignore")
            error = f"HTTPError {status}: {body[:500]}"
            if status in {429, 500, 502, 503, 504}:
                time.sleep((attempt + 1) * 1.5)
                continue
            break
        except Exception as exc:
            error = repr(exc)
            time.sleep((attempt + 1) * 1.0)

    cache_file.write_text(
        json.dumps(
            {
                "url": url,
                "http_status": status,
                "error": error,
                "data": data,
                "created_at": now_iso(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    time.sleep(SLEEP_SECONDS)

    return data, {
        "url": url,
        "cache_status": "MISS",
        "http_status": status,
        "error": error,
    }


def pairs_from_response(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("pairs"), list):
            return [x for x in data.get("pairs") if isinstance(x, dict)]
        if isinstance(data.get("pair"), dict):
            return [data.get("pair")]
    return []


def pair_row(address: str, chain_id: str, endpoint_kind: str, p: dict[str, Any]) -> dict[str, Any]:
    base = p.get("baseToken") or {}
    quote = p.get("quoteToken") or {}

    pair_address = norm_lower(p.get("pairAddress"))
    base_address = norm_lower(base.get("address"))
    quote_address = norm_lower(quote.get("address"))
    target = norm_lower(address)

    exact_pair = pair_address == target
    base_match = base_address == target
    quote_match = quote_address == target
    token_match = base_match or quote_match

    return {
        "target_address": target,
        "queried_chain_id": chain_id,
        "endpoint_kind": endpoint_kind,
        "returned_chain_id": norm_lower(p.get("chainId")) or chain_id,
        "dex_id": norm_lower(p.get("dexId")),
        "pair_address": pair_address,
        "base_token_address": base_address,
        "base_token_symbol": norm(base.get("symbol")),
        "quote_token_address": quote_address,
        "quote_token_symbol": norm(quote.get("symbol")),
        "url": norm(p.get("url")),
        "exact_pair_address_match": exact_pair,
        "token_address_match": token_match,
        "match_type": (
            "EXACT_PAIR_ADDRESS_MATCH"
            if exact_pair else
            "TOKEN_ADDRESS_MATCH"
            if token_match else
            "NO_DIRECT_ADDRESS_MATCH"
        ),
        "liquidity_usd": (p.get("liquidity") or {}).get("usd") if isinstance(p.get("liquidity"), dict) else None,
        "fdv": p.get("fdv"),
        "market_cap": p.get("marketCap"),
    }


def resolve_address(address: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = norm_lower(address)
    matches = []
    fetches = []

    for chain_id in CHAIN_IDS:
        # Pair-address lookup.
        pair_url = f"https://api.dexscreener.com/latest/dex/pairs/{urllib.parse.quote(chain_id)}/{urllib.parse.quote(target)}"
        data, meta = get_json_cached(pair_url, "pair")
        meta.update({
            "target_address": target,
            "chain_id": chain_id,
            "endpoint_kind": "pair_lookup",
        })
        fetches.append(meta)

        for p in pairs_from_response(data):
            matches.append(pair_row(target, chain_id, "pair_lookup", p))

        # Token-address lookup.
        token_url = f"https://api.dexscreener.com/token-pairs/v1/{urllib.parse.quote(chain_id)}/{urllib.parse.quote(target)}"
        data, meta = get_json_cached(token_url, "token_pairs")
        meta.update({
            "target_address": target,
            "chain_id": chain_id,
            "endpoint_kind": "token_pairs_lookup",
        })
        fetches.append(meta)

        for p in pairs_from_response(data):
            matches.append(pair_row(target, chain_id, "token_pairs_lookup", p))

    return matches, fetches


def chain_provider_group(chain_id: str) -> str:
    c = norm_lower(chain_id)

    if c == "ethereum":
        return "ETHERSCAN_OR_ETHEREUM_EXPLORER"
    if c == "bsc":
        return "BSCSCAN"
    if c == "base":
        return "BASESCAN"
    if c == "arbitrum":
        return "ARBISCAN"
    if c == "polygon":
        return "POLYGONSCAN"
    if c == "avalanche":
        return "SNOWTRACE_OR_AVALANCHE"

    return "GENERAL_EVM_RPC_OR_INDEXER"


def resolve_from_matches(address: str, matches: pd.DataFrame) -> dict[str, Any]:
    target = norm_lower(address)

    if matches.empty:
        return {
            "target_address": target,
            "resolved_chain_id": "",
            "resolution_status": "UNRESOLVED_BY_DEXSCREENER",
            "resolution_method": "",
            "exact_pair_match_chains": "",
            "token_match_chains": "",
            "match_rows": 0,
            "best_liquidity_usd": None,
        }

    g = matches[matches["target_address"].eq(target)].copy()
    if g.empty:
        return {
            "target_address": target,
            "resolved_chain_id": "",
            "resolution_status": "UNRESOLVED_BY_DEXSCREENER",
            "resolution_method": "",
            "exact_pair_match_chains": "",
            "token_match_chains": "",
            "match_rows": 0,
            "best_liquidity_usd": None,
        }

    exact = g[g["exact_pair_address_match"].astype(bool)].copy()
    token = g[g["token_address_match"].astype(bool)].copy()

    exact_chains = sorted(set(exact["returned_chain_id"].dropna().astype(str)))
    token_chains = sorted(set(token["returned_chain_id"].dropna().astype(str)))

    best_liq = None
    if "liquidity_usd" in g.columns:
        liq = pd.to_numeric(g["liquidity_usd"], errors="coerce").dropna()
        if not liq.empty:
            best_liq = float(liq.max())

    if len(exact_chains) == 1:
        return {
            "target_address": target,
            "resolved_chain_id": exact_chains[0],
            "resolution_status": "RESOLVED_BY_EXACT_PAIR_ADDRESS",
            "resolution_method": "exact_pair_address",
            "exact_pair_match_chains": ",".join(exact_chains),
            "token_match_chains": ",".join(token_chains),
            "match_rows": int(len(g)),
            "best_liquidity_usd": best_liq,
        }

    if len(exact_chains) > 1:
        return {
            "target_address": target,
            "resolved_chain_id": "",
            "resolution_status": "AMBIGUOUS_EXACT_PAIR_MULTICHAIN",
            "resolution_method": "exact_pair_address",
            "exact_pair_match_chains": ",".join(exact_chains),
            "token_match_chains": ",".join(token_chains),
            "match_rows": int(len(g)),
            "best_liquidity_usd": best_liq,
        }

    if len(token_chains) == 1:
        return {
            "target_address": target,
            "resolved_chain_id": token_chains[0],
            "resolution_status": "RESOLVED_BY_TOKEN_ADDRESS_SINGLE_CHAIN",
            "resolution_method": "token_address",
            "exact_pair_match_chains": "",
            "token_match_chains": ",".join(token_chains),
            "match_rows": int(len(g)),
            "best_liquidity_usd": best_liq,
        }

    if len(token_chains) > 1:
        return {
            "target_address": target,
            "resolved_chain_id": "",
            "resolution_status": "AMBIGUOUS_TOKEN_MULTICHAIN",
            "resolution_method": "token_address",
            "exact_pair_match_chains": "",
            "token_match_chains": ",".join(token_chains),
            "match_rows": int(len(g)),
            "best_liquidity_usd": best_liq,
        }

    return {
        "target_address": target,
        "resolved_chain_id": "",
        "resolution_status": "NO_DIRECT_ADDRESS_MATCH_IN_RETURNED_PAIRS",
        "resolution_method": "",
        "exact_pair_match_chains": ",".join(exact_chains),
        "token_match_chains": ",".join(token_chains),
        "match_rows": int(len(g)),
        "best_liquidity_usd": best_liq,
    }


def main() -> None:
    if not EVM_WINNERS_CSV.exists():
        raise FileNotFoundError(f"Missing 3F EVM winners file: {EVM_WINNERS_CSV}")

    evm = pd.read_csv(EVM_WINNERS_CSV, low_memory=False)

    if "pair_address" not in evm.columns:
        raise ValueError("EVM winners file missing pair_address")

    evm["pair_address_norm"] = evm["pair_address"].map(norm_lower)
    evm = evm[evm["pair_address_norm"].map(looks_like_evm_address)].copy()

    unique_addresses = sorted(evm["pair_address_norm"].dropna().unique())

    all_matches = []
    all_fetches = []

    for i, addr in enumerate(unique_addresses, start=1):
        matches, fetches = resolve_address(addr)
        all_matches.extend(matches)
        all_fetches.extend(fetches)

        if i % 10 == 0:
            print(f"Resolved/looked up {i}/{len(unique_addresses)} addresses...")

    matches_df = pd.DataFrame(all_matches)
    fetch_df = pd.DataFrame(all_fetches)

    if matches_df.empty:
        matches_df = pd.DataFrame(columns=[
            "target_address", "queried_chain_id", "endpoint_kind", "returned_chain_id",
            "exact_pair_address_match", "token_address_match", "match_type"
        ])

    address_resolution = pd.DataFrame([
        resolve_from_matches(addr, matches_df)
        for addr in unique_addresses
    ])

    evm_join = evm.merge(
        address_resolution,
        left_on="pair_address_norm",
        right_on="target_address",
        how="left",
    )

    env_keys = parse_env_keys()
    provider_inventory = provider_group_inventory(env_keys)

    generic_present = bool(
        provider_inventory[
            provider_inventory["provider_key_group"].eq("GENERAL_EVM_RPC_OR_INDEXER")
        ]["present"].any()
    ) if not provider_inventory.empty else False

    def provider_readiness(chain_id: Any) -> str:
        c = norm_lower(chain_id)
        if not c:
            return "CHAIN_NOT_RESOLVED"

        group = chain_provider_group(c)
        specific = bool(
            provider_inventory[
                provider_inventory["provider_key_group"].eq(group)
            ]["present"].any()
        ) if not provider_inventory.empty else False

        if specific or generic_present:
            return "CHAIN_RESOLVED_AND_PROVIDER_KEY_PRESENT"
        return "CHAIN_RESOLVED_PROVIDER_KEY_MISSING"

    evm_join["provider_group_needed"] = evm_join["resolved_chain_id"].map(chain_provider_group)
    evm_join["provider_readiness"] = evm_join["resolved_chain_id"].map(provider_readiness)

    fetch_df.to_csv(OUT / "01_dexscreener_fetch_summary.csv", index=False, encoding="utf-8-sig")
    matches_df.to_csv(OUT / "02_dexscreener_pair_matches.csv", index=False, encoding="utf-8-sig")
    address_resolution.to_csv(OUT / "03_evm_address_chain_resolution.csv", index=False, encoding="utf-8-sig")
    evm_join.to_csv(OUT / "04_evm_winner_chain_resolution_joined.csv", index=False, encoding="utf-8-sig")
    provider_inventory.to_csv(OUT / "05_env_provider_key_inventory_names_only.csv", index=False, encoding="utf-8-sig")

    resolution_by_split = (
        evm_join.groupby(["chronological_split", "resolution_status", "resolved_chain_id"], dropna=False)
        .size()
        .reset_index(name="winner_rows")
        .sort_values(["chronological_split", "resolution_status", "resolved_chain_id"])
    )
    resolution_by_split.to_csv(OUT / "06_resolution_by_split.csv", index=False, encoding="utf-8-sig")

    readiness_by_chain = (
        evm_join.groupby(["resolved_chain_id", "provider_readiness", "provider_group_needed"], dropna=False)
        .agg(
            winner_rows=("pair_address_norm", "count"),
            validation_rows=("chronological_split", lambda x: int((x == "validation").sum())),
            test_rows=("chronological_split", lambda x: int((x == "test").sum())),
            train_rows=("chronological_split", lambda x: int((x == "train").sum())),
        )
        .reset_index()
        .sort_values("winner_rows", ascending=False)
    )
    readiness_by_chain.to_csv(OUT / "07_readiness_by_chain.csv", index=False, encoding="utf-8-sig")

    total_winner_rows = int(len(evm_join))
    unique_address_count = int(len(unique_addresses))
    resolved_rows = int(evm_join["resolution_status"].isin([
        "RESOLVED_BY_EXACT_PAIR_ADDRESS",
        "RESOLVED_BY_TOKEN_ADDRESS_SINGLE_CHAIN",
    ]).sum())
    resolved_addresses = int(address_resolution["resolution_status"].isin([
        "RESOLVED_BY_EXACT_PAIR_ADDRESS",
        "RESOLVED_BY_TOKEN_ADDRESS_SINGLE_CHAIN",
    ]).sum())
    ambiguous_rows = int(evm_join["resolution_status"].astype(str).str.startswith("AMBIGUOUS").sum())
    unresolved_rows = int(evm_join["resolution_status"].isin([
        "UNRESOLVED_BY_DEXSCREENER",
        "NO_DIRECT_ADDRESS_MATCH_IN_RETURNED_PAIRS",
    ]).sum())

    valtest_rows = int(evm_join["chronological_split"].isin(["validation", "test"]).sum())
    valtest_resolved_rows = int(
        evm_join[
            evm_join["chronological_split"].isin(["validation", "test"])
            & evm_join["resolution_status"].isin([
                "RESOLVED_BY_EXACT_PAIR_ADDRESS",
                "RESOLVED_BY_TOKEN_ADDRESS_SINGLE_CHAIN",
            ])
        ].shape[0]
    )

    provider_ready_rows = int(evm_join["provider_readiness"].eq("CHAIN_RESOLVED_AND_PROVIDER_KEY_PRESENT").sum())
    provider_missing_rows = int(evm_join["provider_readiness"].eq("CHAIN_RESOLVED_PROVIDER_KEY_MISSING").sum())

    fetch_errors = int(fetch_df["error"].fillna("").ne("").sum()) if not fetch_df.empty and "error" in fetch_df.columns else 0

    if resolved_rows > 0 and valtest_resolved_rows > 0 and provider_ready_rows > 0:
        final = "EVM_CHAIN_RESOLVED_AND_PROVIDER_READY_FOR_READ_ONLY_TRANSACTION_EXPANSION"
    elif resolved_rows > 0 and valtest_resolved_rows > 0:
        final = "EVM_CHAIN_RESOLVED_PROVIDER_KEYS_REQUIRED_BEFORE_TRANSACTION_EXPANSION"
    elif resolved_rows > 0:
        final = "EVM_CHAIN_RESOLVED_BUT_NO_VALIDATION_TEST_WINNER_RESOLUTION"
    elif ambiguous_rows > 0:
        final = "EVM_CHAIN_RESOLUTION_AMBIGUOUS_REQUIRES_MANUAL_REVIEW"
    else:
        final = "EVM_CHAIN_RESOLUTION_FAILED_WITH_DEXSCREENER"

    decisions = [{
        "question": "Did DEX Screener resolve any EVM winner addresses to a specific chain?",
        "answer": "YES" if resolved_rows > 0 else "NO",
        "classification": "DEXSCREENER_RESOLVED_EVM_CHAINS" if resolved_rows > 0 else "NO_CHAIN_RESOLUTION",
        "evidence": f"resolved_rows={resolved_rows}/{total_winner_rows}; resolved_addresses={resolved_addresses}/{unique_address_count}; ambiguous_rows={ambiguous_rows}; unresolved_rows={unresolved_rows}",
    }, {
        "question": "Did chain resolution restore validation/test EVM winner coverage?",
        "answer": "YES" if valtest_resolved_rows > 0 else "NO",
        "classification": "VALIDATION_TEST_EVM_WINNERS_CHAIN_RESOLVED" if valtest_resolved_rows > 0 else "NO_VALIDATION_TEST_EVM_CHAIN_RESOLUTION",
        "evidence": f"valtest_resolved_rows={valtest_resolved_rows}/{valtest_rows}",
    }, {
        "question": "Are EVM provider keys available for resolved rows?",
        "answer": "YES" if provider_ready_rows > 0 else "NO",
        "classification": "PROVIDER_KEY_PRESENT_FOR_RESOLVED_EVM_ROWS" if provider_ready_rows > 0 else "PROVIDER_KEYS_REQUIRED",
        "evidence": f"provider_ready_rows={provider_ready_rows}; provider_missing_rows={provider_missing_rows}",
    }, {
        "question": "Should EVM transaction calls be made immediately?",
        "answer": "YES" if final == "EVM_CHAIN_RESOLVED_AND_PROVIDER_READY_FOR_READ_ONLY_TRANSACTION_EXPANSION" else "NO",
        "classification": final,
        "evidence": "Proceed only if both chain resolution and provider-key readiness are sufficient.",
    }]

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "08_chain_resolution_decision_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "classification": "THESIS_DEXSCREENER_EVM_CHAIN_RESOLUTION_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "created_at": now_iso(),
        "safety": {
            "read_only_external_metadata_queries": True,
            "dexscreener_queries": True,
            "helius_queries": False,
            "evm_transaction_provider_queries": False,
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
            "three_f_root": str(THREE_F_ROOT),
            "three_g_root": str(THREE_G_ROOT),
            "evm_winners_csv": str(EVM_WINNERS_CSV),
        },
        "query_design": {
            "chain_ids_tried": CHAIN_IDS,
            "endpoints": [
                "/latest/dex/pairs/{chainId}/{pairId}",
                "/token-pairs/v1/{chainId}/{tokenAddress}",
            ],
            "unique_addresses_queried": unique_address_count,
            "sleep_seconds": SLEEP_SECONDS,
        },
        "counts": {
            "evm_winner_rows": total_winner_rows,
            "unique_evm_addresses": unique_address_count,
            "resolved_winner_rows": resolved_rows,
            "resolved_unique_addresses": resolved_addresses,
            "ambiguous_winner_rows": ambiguous_rows,
            "unresolved_winner_rows": unresolved_rows,
            "validation_test_winner_rows": valtest_rows,
            "validation_test_resolved_winner_rows": valtest_resolved_rows,
            "provider_ready_rows": provider_ready_rows,
            "provider_missing_rows": provider_missing_rows,
            "dexscreener_fetch_rows": int(len(fetch_df)),
            "fetch_errors": fetch_errors,
        },
        "final_scientific_conclusion": final,
        "outputs": {
            "dexscreener_fetch_summary": str(OUT / "01_dexscreener_fetch_summary.csv"),
            "dexscreener_pair_matches": str(OUT / "02_dexscreener_pair_matches.csv"),
            "evm_address_chain_resolution": str(OUT / "03_evm_address_chain_resolution.csv"),
            "evm_winner_chain_resolution_joined": str(OUT / "04_evm_winner_chain_resolution_joined.csv"),
            "env_provider_key_inventory": str(OUT / "05_env_provider_key_inventory_names_only.csv"),
            "resolution_by_split": str(OUT / "06_resolution_by_split.csv"),
            "readiness_by_chain": str(OUT / "07_readiness_by_chain.csv"),
            "decision_table": str(OUT / "08_chain_resolution_decision_table.csv"),
            "summary_json": str(OUT / "thesis_dexscreener_evm_chain_resolution_summary.json"),
            "summary_md": str(OUT / "thesis_dexscreener_evm_chain_resolution_summary.md"),
        },
    }

    summary_json = OUT / "thesis_dexscreener_evm_chain_resolution_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3H — DEX Screener EVM Chain Resolution")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only external metadata queries")
    lines.append("- DEX Screener metadata only")
    lines.append("- No Helius queries")
    lines.append("- No EVM transaction-provider queries")
    lines.append("- No model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet connection")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("- .env values were not printed")
    lines.append("")
    lines.append("## Query design")
    lines.append(f"- unique EVM addresses queried: {unique_address_count}")
    lines.append(f"- chain IDs tried: {', '.join(CHAIN_IDS)}")
    lines.append("- endpoints:")
    lines.append("  - `/latest/dex/pairs/{chainId}/{pairId}`")
    lines.append("  - `/token-pairs/v1/{chainId}/{tokenAddress}`")
    lines.append("")
    lines.append("## Counts")
    for k, v in summary["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Resolution by split")
    for _, r in resolution_by_split.iterrows():
        lines.append(
            f"- `{r['chronological_split']}` / `{r['resolution_status']}` / `{r['resolved_chain_id']}`: {int(r['winner_rows'])}"
        )
    lines.append("")
    lines.append("## Readiness by chain")
    for _, r in readiness_by_chain.iterrows():
        lines.append(
            f"- `{r['resolved_chain_id']}` / `{r['provider_readiness']}` / `{r['provider_group_needed']}`: "
            f"winner_rows={int(r['winner_rows'])}; validation={int(r['validation_rows'])}; "
            f"test={int(r['test_rows'])}; train={int(r['train_rows'])}"
        )
    lines.append("")
    lines.append("## Provider key inventory")
    for _, r in provider_inventory.iterrows():
        lines.append(
            f"- `{r['provider_key_group']}`: present={bool(r['present'])}; keys=`{r['present_key_names']}`"
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

    summary_md = OUT / "thesis_dexscreener_evm_chain_resolution_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": final,
        "evm_winner_rows": total_winner_rows,
        "unique_evm_addresses": unique_address_count,
        "resolved_winner_rows": resolved_rows,
        "resolved_unique_addresses": resolved_addresses,
        "validation_test_winner_rows": valtest_rows,
        "validation_test_resolved_winner_rows": valtest_resolved_rows,
        "provider_ready_rows": provider_ready_rows,
        "provider_missing_rows": provider_missing_rows,
        "fetch_errors": fetch_errors,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
