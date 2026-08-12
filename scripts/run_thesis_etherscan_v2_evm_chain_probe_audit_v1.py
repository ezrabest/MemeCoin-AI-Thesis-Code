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

OUT = ROOT / "data" / "audits" / f"thesis_etherscan_v2_evm_chain_probe_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = ROOT / "data" / "audits" / "_cache" / "etherscan_v2_chain_probe"
CACHE.mkdir(parents=True, exist_ok=True)

EVM_WINNERS_CSV = THREE_F_ROOT / "01_evm_winner_events.csv"
ENV_FILE = ROOT / ".env"

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"

# Keep this intentionally bounded. Add more chainids later only if needed.
CHAIN_IDS = [
    ("1", "ethereum"),
    ("56", "bsc"),
    ("8453", "base"),
    ("42161", "arbitrum"),
    ("137", "polygon"),
    ("10", "optimism"),
    ("43114", "avalanche"),
    ("59144", "linea"),
    ("81457", "blast"),
]

SLEEP_SECONDS = float(os.environ.get("THESIS_ETHERSCAN_SLEEP_SECONDS", "0.25"))
TIMEOUT_SECONDS = int(os.environ.get("THESIS_ETHERSCAN_TIMEOUT_SECONDS", "30"))


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


def read_dotenv_key() -> str:
    if os.environ.get("ETHERSCAN_API_KEY"):
        return os.environ["ETHERSCAN_API_KEY"].strip()

    if not ENV_FILE.exists():
        raise FileNotFoundError(f".env not found: {ENV_FILE}")

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

        if k == "ETHERSCAN_API_KEY" and v.strip():
            return v.strip()

    raise RuntimeError("ETHERSCAN_API_KEY not found in environment or .env")


def build_urls(params: dict[str, Any], api_key: str) -> tuple[str, str]:
    safe_params = dict(params)
    safe_params["apikey"] = "REDACTED"

    request_params = dict(params)
    request_params["apikey"] = api_key

    safe_url = ETHERSCAN_BASE + "?" + urllib.parse.urlencode(safe_params)
    request_url = ETHERSCAN_BASE + "?" + urllib.parse.urlencode(request_params)
    return safe_url, request_url


def get_etherscan_json(params: dict[str, Any], api_key: str, cache_prefix: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    safe_url, request_url = build_urls(params, api_key)
    cache_file = CACHE / f"{cache_prefix}_{sha(safe_url)}.json"

    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return payload.get("data"), {
                "safe_url": safe_url,
                "cache_status": "HIT",
                "http_status": payload.get("http_status"),
                "api_status": payload.get("api_status"),
                "api_message": payload.get("api_message"),
                "error": payload.get("error", ""),
            }
        except Exception as exc:
            return None, {
                "safe_url": safe_url,
                "cache_status": "BAD_CACHE",
                "http_status": None,
                "api_status": None,
                "api_message": None,
                "error": repr(exc),
            }

    status = None
    data = None
    error = ""

    req = urllib.request.Request(
        request_url,
        headers={
            "User-Agent": "memecoin-thesis-etherscan-v2-chain-probe/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                status = int(resp.status)
                body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
            break
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = exc.read().decode("utf-8", errors="ignore")
            error = f"HTTPError {status}: {body[:300]}"
            if status in {429, 500, 502, 503, 504}:
                time.sleep((attempt + 1) * 1.5)
                continue
            break
        except Exception as exc:
            error = repr(exc)
            time.sleep((attempt + 1) * 1.0)

    payload = {
        "safe_url": safe_url,
        "http_status": status,
        "api_status": data.get("status") if isinstance(data, dict) else None,
        "api_message": data.get("message") if isinstance(data, dict) else None,
        "error": error,
        "data": data,
        "created_at": now_iso(),
    }
    cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    time.sleep(SLEEP_SECONDS)

    return data, {
        "safe_url": safe_url,
        "cache_status": "MISS",
        "http_status": status,
        "api_status": payload["api_status"],
        "api_message": payload["api_message"],
        "error": error,
    }


def result_len(data: dict[str, Any] | None) -> int:
    if not isinstance(data, dict):
        return 0
    r = data.get("result")
    if isinstance(r, list):
        return len(r)
    if isinstance(r, dict):
        return 1
    if isinstance(r, str) and r:
        return 1
    return 0


def is_contract_code(data: dict[str, Any] | None) -> tuple[bool, int, str]:
    """
    Strict validation for Etherscan proxy.eth_getCode.

    A valid contract-code response must be a hexadecimal string beginning with 0x.
    API messages, error strings, unsupported-chain messages, or other text must NOT
    be counted as deployed bytecode.
    """
    if not isinstance(data, dict):
        return False, 0, ""

    r = data.get("result")
    if not isinstance(r, str):
        return False, 0, ""

    code = r.strip()

    if not code:
        return False, 0, ""

    if not re.fullmatch(r"0x[0-9a-fA-F]*", code):
        return False, 0, code[:120]

    if code.lower() == "0x":
        return False, 0, code[:120]

    return True, max(0, (len(code) - 2) // 2), code[:120]


def first_result_has_rows(data: dict[str, Any] | None) -> bool:
    if not isinstance(data, dict):
        return False
    r = data.get("result")
    return isinstance(r, list) and len(r) > 0


def probe_address_chain(address: str, chainid: str, chain_name: str, api_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fetch_rows: list[dict[str, Any]] = []

    base_row = {
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "contract_deployed": False,
        "code_size_bytes": 0,
        "code_prefix": "",
        "normal_tx_probe_has_rows": False,
        "address_tokentx_probe_has_rows": False,
        "contract_tokentx_probe_has_rows": False,
        "activity_score": 0,
    }

    # 1) Minimal contract existence probe.
    params = {
        "chainid": chainid,
        "module": "proxy",
        "action": "eth_getCode",
        "address": address,
        "tag": "latest",
    }
    data, meta = get_etherscan_json(params, api_key, "eth_getCode")
    meta.update({
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "probe_type": "eth_getCode",
    })
    fetch_rows.append(meta)

    deployed, code_size, code_prefix = is_contract_code(data)
    base_row["contract_deployed"] = deployed
    base_row["code_size_bytes"] = code_size
    base_row["code_prefix"] = code_prefix

    if not deployed:
        return base_row, fetch_rows

    # 2) Minimal normal transaction probe, only for deployed contracts.
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "txlist",
        "address": address,
        "page": 1,
        "offset": 1,
        "sort": "desc",
    }
    data, meta = get_etherscan_json(params, api_key, "txlist")
    meta.update({
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "probe_type": "txlist_address_offset1",
    })
    fetch_rows.append(meta)
    base_row["normal_tx_probe_has_rows"] = first_result_has_rows(data)

    # 3) Minimal token-transfer-by-address probe.
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "tokentx",
        "address": address,
        "page": 1,
        "offset": 1,
        "sort": "desc",
    }
    data, meta = get_etherscan_json(params, api_key, "tokentx_address")
    meta.update({
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "probe_type": "tokentx_address_offset1",
    })
    fetch_rows.append(meta)
    base_row["address_tokentx_probe_has_rows"] = first_result_has_rows(data)

    # 4) Minimal token-transfer-by-contract probe, useful if address is a token contract.
    params = {
        "chainid": chainid,
        "module": "account",
        "action": "tokentx",
        "contractaddress": address,
        "page": 1,
        "offset": 1,
        "sort": "desc",
    }
    data, meta = get_etherscan_json(params, api_key, "tokentx_contract")
    meta.update({
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "probe_type": "tokentx_contract_offset1",
    })
    fetch_rows.append(meta)
    base_row["contract_tokentx_probe_has_rows"] = first_result_has_rows(data)

    score = 100
    if base_row["normal_tx_probe_has_rows"]:
        score += 20
    if base_row["address_tokentx_probe_has_rows"]:
        score += 10
    if base_row["contract_tokentx_probe_has_rows"]:
        score += 10
    score += min(20, int(code_size > 0) + int(code_size > 1000) + int(code_size > 10000))

    base_row["activity_score"] = score
    return base_row, fetch_rows


def resolve_address(address: str, probes: pd.DataFrame) -> dict[str, Any]:
    g = probes[probes["target_address"].eq(address)].copy()
    deployed = g[g["contract_deployed"].astype(bool)].copy()

    if deployed.empty:
        return {
            "target_address": address,
            "resolved_chainid": "",
            "resolved_chain_name": "",
            "resolution_status": "UNRESOLVED_NO_CONTRACT_CODE_ON_PROBED_CHAINS",
            "deployed_chainids": "",
            "deployed_chain_names": "",
            "best_activity_score": 0,
            "deployed_chain_count": 0,
        }

    deployed["activity_score_num"] = pd.to_numeric(deployed["activity_score"], errors="coerce").fillna(0)
    max_score = deployed["activity_score_num"].max()
    best = deployed[deployed["activity_score_num"].eq(max_score)].copy()

    chainids = sorted(set(deployed["chainid"].astype(str)))
    chain_names = sorted(set(deployed["chain_name"].astype(str)))

    if len(best) == 1:
        r = best.iloc[0]
        status = "RESOLVED_SINGLE_BEST_CHAIN_BY_CODE_AND_ACTIVITY"
        if len(deployed) == 1:
            status = "RESOLVED_SINGLE_DEPLOYED_CHAIN"
        return {
            "target_address": address,
            "resolved_chainid": str(r["chainid"]),
            "resolved_chain_name": str(r["chain_name"]),
            "resolution_status": status,
            "deployed_chainids": ",".join(chainids),
            "deployed_chain_names": ",".join(chain_names),
            "best_activity_score": int(max_score),
            "deployed_chain_count": int(len(deployed)),
        }

    return {
        "target_address": address,
        "resolved_chainid": "",
        "resolved_chain_name": "",
        "resolution_status": "AMBIGUOUS_MULTICHAIN_DEPLOYMENT_EQUAL_SCORE",
        "deployed_chainids": ",".join(chainids),
        "deployed_chain_names": ",".join(chain_names),
        "best_activity_score": int(max_score),
        "deployed_chain_count": int(len(deployed)),
    }


def main() -> None:
    api_key = read_dotenv_key()

    if not EVM_WINNERS_CSV.exists():
        raise FileNotFoundError(f"Missing EVM winners file: {EVM_WINNERS_CSV}")

    evm = pd.read_csv(EVM_WINNERS_CSV, low_memory=False)

    if "pair_address" not in evm.columns:
        raise ValueError("EVM winner file missing pair_address")

    evm["pair_address_norm"] = evm["pair_address"].map(norm_lower)
    evm = evm[evm["pair_address_norm"].map(looks_like_evm_address)].copy()

    unique_addresses = sorted(evm["pair_address_norm"].dropna().unique())

    probe_rows: list[dict[str, Any]] = []
    fetch_rows: list[dict[str, Any]] = []

    for i, addr in enumerate(unique_addresses, start=1):
        for chainid, chain_name in CHAIN_IDS:
            row, fetch = probe_address_chain(addr, chainid, chain_name, api_key)
            probe_rows.append(row)
            fetch_rows.extend(fetch)

        if i % 5 == 0:
            print(f"Probed {i}/{len(unique_addresses)} EVM addresses across {len(CHAIN_IDS)} chains...")

    probes = pd.DataFrame(probe_rows)
    fetches = pd.DataFrame(fetch_rows)

    resolutions = pd.DataFrame([
        resolve_address(addr, probes)
        for addr in unique_addresses
    ])

    joined = evm.merge(
        resolutions,
        left_on="pair_address_norm",
        right_on="target_address",
        how="left",
    )

    fetches.to_csv(OUT / "01_etherscan_v2_fetch_summary.csv", index=False, encoding="utf-8-sig")
    probes.to_csv(OUT / "02_etherscan_v2_chain_probe_by_address.csv", index=False, encoding="utf-8-sig")
    resolutions.to_csv(OUT / "03_evm_address_chain_resolution.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(OUT / "04_evm_winner_joined_chain_resolution.csv", index=False, encoding="utf-8-sig")

    resolution_by_split = (
        joined.groupby(["chronological_split", "resolution_status", "resolved_chain_name"], dropna=False)
        .size()
        .reset_index(name="winner_rows")
        .sort_values(["chronological_split", "resolution_status", "resolved_chain_name"])
    )
    resolution_by_split.to_csv(OUT / "05_resolution_by_split.csv", index=False, encoding="utf-8-sig")

    resolution_by_chain = (
        joined.groupby(["resolved_chainid", "resolved_chain_name", "resolution_status"], dropna=False)
        .agg(
            winner_rows=("pair_address_norm", "count"),
            validation_rows=("chronological_split", lambda x: int((x == "validation").sum())),
            test_rows=("chronological_split", lambda x: int((x == "test").sum())),
            train_rows=("chronological_split", lambda x: int((x == "train").sum())),
        )
        .reset_index()
        .sort_values("winner_rows", ascending=False)
    )
    resolution_by_chain.to_csv(OUT / "06_resolution_by_chain.csv", index=False, encoding="utf-8-sig")

    total_rows = int(len(joined))
    unique_n = int(len(unique_addresses))

    resolved_statuses = {
        "RESOLVED_SINGLE_DEPLOYED_CHAIN",
        "RESOLVED_SINGLE_BEST_CHAIN_BY_CODE_AND_ACTIVITY",
    }

    resolved_rows = int(joined["resolution_status"].isin(resolved_statuses).sum())
    resolved_addresses = int(resolutions["resolution_status"].isin(resolved_statuses).sum())
    ambiguous_rows = int(joined["resolution_status"].astype(str).str.startswith("AMBIGUOUS").sum())
    unresolved_rows = int(joined["resolution_status"].astype(str).str.startswith("UNRESOLVED").sum())

    valtest_rows = int(joined["chronological_split"].isin(["validation", "test"]).sum())
    valtest_resolved_rows = int(
        joined[
            joined["chronological_split"].isin(["validation", "test"])
            & joined["resolution_status"].isin(resolved_statuses)
        ].shape[0]
    )

    deployed_probe_rows = int(probes["contract_deployed"].astype(bool).sum()) if not probes.empty else 0
    fetch_errors = int(fetches["error"].fillna("").ne("").sum()) if not fetches.empty and "error" in fetches.columns else 0

    provider_ready_rows = resolved_rows  # ETHERSCAN_API_KEY is required and was used successfully for the probes.

    if resolved_rows > 0 and valtest_resolved_rows > 0:
        final = "EVM_CHAIN_PROBING_RESOLVED_VALIDATION_TEST_WINNERS_READY_FOR_READ_ONLY_TX_EXPANSION"
    elif resolved_rows > 0:
        final = "EVM_CHAIN_PROBING_RESOLVED_ONLY_TRAIN_WINNERS"
    elif ambiguous_rows > 0:
        final = "EVM_CHAIN_PROBING_AMBIGUOUS_MULTICHAIN_REQUIRES_MANUAL_REVIEW"
    else:
        final = "EVM_CHAIN_PROBING_DID_NOT_RESOLVE_EVM_WINNERS"

    decisions = [{
        "question": "Did Etherscan V2 eth_getCode resolve any EVM winner address to a deployed chain?",
        "answer": "YES" if resolved_rows > 0 else "NO",
        "classification": "ETHERSCAN_V2_CHAIN_PROBE_RESOLVED_ADDRESSES" if resolved_rows > 0 else "NO_VALID_DEPLOYED_CONTRACT_CODE_FOUND_ON_PROBED_CHAINS",
        "evidence": f"resolved_rows={resolved_rows}/{total_rows}; resolved_addresses={resolved_addresses}/{unique_n}; valid_deployed_probe_rows={deployed_probe_rows}",
    }, {
        "question": "Did chain probing restore validation/test EVM winner coverage?",
        "answer": "YES" if valtest_resolved_rows > 0 else "NO",
        "classification": "VALIDATION_TEST_EVM_WINNERS_CHAIN_RESOLVED" if valtest_resolved_rows > 0 else "NO_VALIDATION_TEST_EVM_WINNER_CHAIN_RESOLUTION",
        "evidence": f"valtest_resolved_rows={valtest_resolved_rows}/{valtest_rows}",
    }, {
        "question": "Is provider readiness satisfied for resolved rows?",
        "answer": "YES" if provider_ready_rows > 0 else "NO",
        "classification": "ETHERSCAN_V2_KEY_CONFIRMED_BY_SUCCESSFUL_PROBES" if provider_ready_rows > 0 else "NO_RESOLVED_ROWS_FOR_PROVIDER_EXPANSION",
        "evidence": f"provider_ready_rows={provider_ready_rows}; ETHERSCAN_API_KEY value was not printed.",
    }, {
        "question": "Should full EVM transaction expansion be run now?",
        "answer": "YES" if final == "EVM_CHAIN_PROBING_RESOLVED_VALIDATION_TEST_WINNERS_READY_FOR_READ_ONLY_TX_EXPANSION" else "NO",
        "classification": final,
        "evidence": "Only proceed to transaction expansion when chain probing resolves validation/test EVM winners.",
    }]

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "07_chain_probe_decision_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "classification": "THESIS_ETHERSCAN_V2_EVM_CHAIN_PROBE_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "created_at": now_iso(),
        "safety": {
            "read_only_external_metadata_queries": True,
            "etherscan_v2_queries": True,
            "minimal_chain_probe_only": True,
            "full_transaction_history_pulled": False,
            "helius_queries": False,
            "new_model_training": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
            "env_values_printed": False,
        },
        "query_design": {
            "base_url": ETHERSCAN_BASE,
            "chain_ids_tried": [{"chainid": c, "chain_name": n} for c, n in CHAIN_IDS],
            "unique_addresses": unique_n,
            "primary_probe": "proxy.eth_getCode",
            "secondary_probes_only_when_code_found": [
                "account.txlist offset=1",
                "account.tokentx address offset=1",
                "account.tokentx contractaddress offset=1",
            ],
            "cache_dir": str(CACHE),
        },
        "counts": {
            "evm_winner_rows": total_rows,
            "unique_evm_addresses": unique_n,
            "resolved_winner_rows": resolved_rows,
            "resolved_unique_addresses": resolved_addresses,
            "ambiguous_winner_rows": ambiguous_rows,
            "unresolved_winner_rows": unresolved_rows,
            "validation_test_winner_rows": valtest_rows,
            "validation_test_resolved_winner_rows": valtest_resolved_rows,
            "deployed_probe_rows": deployed_probe_rows,
            "provider_ready_rows": provider_ready_rows,
            "fetch_rows": int(len(fetches)),
            "fetch_errors": fetch_errors,
        },
        "final_scientific_conclusion": final,
        "outputs": {
            "fetch_summary": str(OUT / "01_etherscan_v2_fetch_summary.csv"),
            "chain_probe_by_address": str(OUT / "02_etherscan_v2_chain_probe_by_address.csv"),
            "address_chain_resolution": str(OUT / "03_evm_address_chain_resolution.csv"),
            "winner_joined_resolution": str(OUT / "04_evm_winner_joined_chain_resolution.csv"),
            "resolution_by_split": str(OUT / "05_resolution_by_split.csv"),
            "resolution_by_chain": str(OUT / "06_resolution_by_chain.csv"),
            "decision_table": str(OUT / "07_chain_probe_decision_table.csv"),
            "summary_json": str(OUT / "thesis_etherscan_v2_evm_chain_probe_summary.json"),
            "summary_md": str(OUT / "thesis_etherscan_v2_evm_chain_probe_summary.md"),
        },
    }

    summary_json = OUT / "thesis_etherscan_v2_evm_chain_probe_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3I — Etherscan V2 Minimal EVM Chain Probing")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only external metadata queries")
    lines.append("- Etherscan V2 minimal chain probe only")
    lines.append("- No full transaction-history pull")
    lines.append("- No Helius queries")
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
    lines.append(f"- unique EVM addresses: {unique_n}")
    lines.append(f"- chain IDs tried: {', '.join([f'{n}:{c}' for c, n in CHAIN_IDS])}")
    lines.append("- primary probe: `proxy.eth_getCode`")
    lines.append("- secondary probes, only if code found: `txlist`, `tokentx(address)`, `tokentx(contractaddress)`, each with offset=1")
    lines.append("")
    lines.append("## Counts")
    for k, v in summary["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Resolution by split")
    for _, r in resolution_by_split.iterrows():
        lines.append(
            f"- `{r['chronological_split']}` / `{r['resolution_status']}` / `{r['resolved_chain_name']}`: {int(r['winner_rows'])}"
        )
    lines.append("")
    lines.append("## Resolution by chain")
    for _, r in resolution_by_chain.iterrows():
        lines.append(
            f"- `{r['resolved_chain_name']}` / `{r['resolved_chainid']}` / `{r['resolution_status']}`: "
            f"winner_rows={int(r['winner_rows'])}; validation={int(r['validation_rows'])}; "
            f"test={int(r['test_rows'])}; train={int(r['train_rows'])}"
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

    summary_md = OUT / "thesis_etherscan_v2_evm_chain_probe_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": final,
        "evm_winner_rows": total_rows,
        "unique_evm_addresses": unique_n,
        "resolved_winner_rows": resolved_rows,
        "resolved_unique_addresses": resolved_addresses,
        "validation_test_winner_rows": valtest_rows,
        "validation_test_resolved_winner_rows": valtest_resolved_rows,
        "provider_ready_rows": provider_ready_rows,
        "fetch_errors": fetch_errors,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
