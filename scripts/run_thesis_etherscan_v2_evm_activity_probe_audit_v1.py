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

OUT = ROOT / "data" / "audits" / f"thesis_etherscan_v2_evm_activity_probe_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = ROOT / "data" / "audits" / "_cache" / "etherscan_v2_activity_probe"
CACHE.mkdir(parents=True, exist_ok=True)

EVM_WINNERS_CSV = THREE_F_ROOT / "01_evm_winner_events.csv"
ENV_FILE = ROOT / ".env"
BASE = "https://api.etherscan.io/v2/api"

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


def read_key() -> str:
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

        if k == "ETHERSCAN_API_KEY" and v:
            return v

    raise RuntimeError("ETHERSCAN_API_KEY not found")


def build_urls(params: dict[str, Any], key: str) -> tuple[str, str]:
    safe = dict(params)
    safe["apikey"] = "REDACTED"

    req = dict(params)
    req["apikey"] = key

    return (
        BASE + "?" + urllib.parse.urlencode(safe),
        BASE + "?" + urllib.parse.urlencode(req),
    )


def call(params: dict[str, Any], key: str, prefix: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    safe_url, request_url = build_urls(params, key)
    cache_file = CACHE / f"{prefix}_{sha(safe_url)}.json"

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return payload.get("data"), {
            "safe_url": safe_url,
            "cache_status": "HIT",
            "http_status": payload.get("http_status"),
            "api_status": payload.get("api_status"),
            "api_message": payload.get("api_message"),
            "error": payload.get("error", ""),
        }

    status = None
    data = None
    error = ""

    request = urllib.request.Request(
        request_url,
        headers={
            "User-Agent": "memecoin-thesis-etherscan-v2-activity-probe/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
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


def has_result_rows(data: dict[str, Any] | None) -> tuple[bool, int, str]:
    """
    Strict Etherscan activity validation.

    Count activity only when Etherscan reports status == '1'.
    Do not count NOTOK messages, unsupported endpoint messages, error strings,
    or "No transactions found" text as activity.
    """
    if not isinstance(data, dict):
        return False, 0, ""

    status = str(data.get("status", "")).strip()
    message = str(data.get("message", "")).strip()
    result = data.get("result")

    if status != "1":
        prefix = ""
        if isinstance(result, str):
            prefix = result[:120]
        else:
            prefix = str(result)[:120]
        return False, 0, prefix

    if isinstance(result, list):
        return len(result) > 0, len(result), ""

    if isinstance(result, str):
        s = result.strip()

        # This branch is mainly for account.balance.
        # It should not turn API text into activity.
        if s.isdigit():
            return int(s) > 0, 1 if int(s) > 0 else 0, s[:120]

        return False, 0, s[:120]

    return False, 0, str(result)[:120]


def probe(address: str, chainid: str, chain_name: str, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {
        "target_address": address,
        "chainid": chainid,
        "chain_name": chain_name,
        "txlist_has_rows": False,
        "txlist_rows_seen": 0,
        "tokentx_address_has_rows": False,
        "tokentx_address_rows_seen": 0,
        "internal_tx_has_rows": False,
        "internal_tx_rows_seen": 0,
        "balance_nonzero": False,
        "balance_prefix": "",
        "activity_score": 0,
    }
    fetches = []

    calls = [
        ("txlist", {
            "chainid": chainid,
            "module": "account",
            "action": "txlist",
            "address": address,
            "page": 1,
            "offset": 1,
            "sort": "desc",
        }),
        ("tokentx_address", {
            "chainid": chainid,
            "module": "account",
            "action": "tokentx",
            "address": address,
            "page": 1,
            "offset": 1,
            "sort": "desc",
        }),
        ("txlistinternal", {
            "chainid": chainid,
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "page": 1,
            "offset": 1,
            "sort": "desc",
        }),
        ("balance", {
            "chainid": chainid,
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
        }),
    ]

    for probe_type, params in calls:
        data, meta = call(params, key, probe_type)
        meta.update({
            "target_address": address,
            "chainid": chainid,
            "chain_name": chain_name,
            "probe_type": probe_type,
        })
        fetches.append(meta)

        has_rows, n, prefix = has_result_rows(data)

        if probe_type == "txlist":
            row["txlist_has_rows"] = has_rows
            row["txlist_rows_seen"] = n
        elif probe_type == "tokentx_address":
            row["tokentx_address_has_rows"] = has_rows
            row["tokentx_address_rows_seen"] = n
        elif probe_type == "txlistinternal":
            row["internal_tx_has_rows"] = has_rows
            row["internal_tx_rows_seen"] = n
        elif probe_type == "balance":
            row["balance_prefix"] = prefix
            try:
                row["balance_nonzero"] = int(prefix) > 0
            except Exception:
                row["balance_nonzero"] = False

    score = 0
    if row["txlist_has_rows"]:
        score += 50
    if row["tokentx_address_has_rows"]:
        score += 40
    if row["internal_tx_has_rows"]:
        score += 20
    # Balance is recorded but intentionally not used for chain resolution.
    # Nonzero dust balances across multiple chains are not enough to identify
    # the relevant trading chain. Resolution requires txlist/tokentx/internal activity.
    row["activity_score"] = score
    return row, fetches


def resolve_address(address: str, probes: pd.DataFrame) -> dict[str, Any]:
    g = probes[probes["target_address"].eq(address)].copy()

    active = g[pd.to_numeric(g["activity_score"], errors="coerce").fillna(0) > 0].copy()

    if active.empty:
        return {
            "target_address": address,
            "resolved_chainid": "",
            "resolved_chain_name": "",
            "resolution_status": "UNRESOLVED_NO_ACTIVITY_ON_PROBED_CHAINS",
            "active_chainids": "",
            "active_chain_names": "",
            "best_activity_score": 0,
            "active_chain_count": 0,
        }

    active["score_num"] = pd.to_numeric(active["activity_score"], errors="coerce").fillna(0)
    max_score = active["score_num"].max()
    best = active[active["score_num"].eq(max_score)].copy()

    chainids = sorted(set(active["chainid"].astype(str)))
    names = sorted(set(active["chain_name"].astype(str)))

    if len(best) == 1:
        r = best.iloc[0]
        status = "RESOLVED_SINGLE_ACTIVE_CHAIN"
        if len(active) > 1:
            status = "RESOLVED_SINGLE_BEST_ACTIVE_CHAIN"
        return {
            "target_address": address,
            "resolved_chainid": str(r["chainid"]),
            "resolved_chain_name": str(r["chain_name"]),
            "resolution_status": status,
            "active_chainids": ",".join(chainids),
            "active_chain_names": ",".join(names),
            "best_activity_score": int(max_score),
            "active_chain_count": int(len(active)),
        }

    return {
        "target_address": address,
        "resolved_chainid": "",
        "resolved_chain_name": "",
        "resolution_status": "AMBIGUOUS_MULTICHAIN_ACTIVITY_EQUAL_SCORE",
        "active_chainids": ",".join(chainids),
        "active_chain_names": ",".join(names),
        "best_activity_score": int(max_score),
        "active_chain_count": int(len(active)),
    }


def main() -> None:
    key = read_key()

    if not EVM_WINNERS_CSV.exists():
        raise FileNotFoundError(f"Missing EVM winners file: {EVM_WINNERS_CSV}")

    evm = pd.read_csv(EVM_WINNERS_CSV, low_memory=False)
    evm["pair_address_norm"] = evm["pair_address"].map(norm_lower)
    evm = evm[evm["pair_address_norm"].map(looks_like_evm_address)].copy()

    addresses = sorted(evm["pair_address_norm"].dropna().unique())

    probe_rows = []
    fetch_rows = []

    for i, address in enumerate(addresses, start=1):
        for chainid, chain_name in CHAIN_IDS:
            row, fetches = probe(address, chainid, chain_name, key)
            probe_rows.append(row)
            fetch_rows.extend(fetches)

        if i % 5 == 0:
            print(f"Activity-probed {i}/{len(addresses)} addresses across {len(CHAIN_IDS)} chains...")

    probes = pd.DataFrame(probe_rows)
    fetches = pd.DataFrame(fetch_rows)
    resolutions = pd.DataFrame([resolve_address(a, probes) for a in addresses])

    joined = evm.merge(
        resolutions,
        left_on="pair_address_norm",
        right_on="target_address",
        how="left",
    )

    fetches.to_csv(OUT / "01_etherscan_v2_activity_fetch_summary.csv", index=False, encoding="utf-8-sig")
    probes.to_csv(OUT / "02_etherscan_v2_activity_probe_by_address.csv", index=False, encoding="utf-8-sig")
    resolutions.to_csv(OUT / "03_evm_address_activity_resolution.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(OUT / "04_evm_winner_joined_activity_resolution.csv", index=False, encoding="utf-8-sig")

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

    total = int(len(joined))
    unique_n = int(len(addresses))

    resolved_statuses = {"RESOLVED_SINGLE_ACTIVE_CHAIN", "RESOLVED_SINGLE_BEST_ACTIVE_CHAIN"}

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

    fetch_errors = int(fetches["error"].fillna("").ne("").sum()) if not fetches.empty else 0

    # Some Etherscan endpoints return NOTOK + "No transactions found"; this is not a transport error.
    api_status_counts = {
        str(k): int(v)
        for k, v in fetches["api_status"].fillna("").astype(str).value_counts().to_dict().items()
    } if "api_status" in fetches.columns else {}

    api_message_counts = {
        str(k): int(v)
        for k, v in fetches["api_message"].fillna("").astype(str).value_counts().to_dict().items()
    } if "api_message" in fetches.columns else {}

    if resolved_rows > 0 and valtest_resolved_rows > 0:
        final = "EVM_ACTIVITY_PROBE_RESOLVED_VALIDATION_TEST_WINNERS_READY_FOR_TX_EXPANSION"
    elif resolved_rows > 0:
        final = "EVM_ACTIVITY_PROBE_RESOLVED_ONLY_TRAIN_WINNERS"
    elif ambiguous_rows > 0:
        final = "EVM_ACTIVITY_PROBE_AMBIGUOUS_MULTICHAIN_REQUIRES_MANUAL_REVIEW"
    else:
        final = "EVM_ACTIVITY_PROBE_DID_NOT_RESOLVE_EVM_WINNERS"

    decisions = [{
        "question": "Did minimal Etherscan activity probes resolve any EVM winner address to an active chain?",
        "answer": "YES" if resolved_rows > 0 else "NO",
        "classification": "ETHERSCAN_ACTIVITY_PROBE_RESOLVED_ADDRESSES" if resolved_rows > 0 else "NO_STRICT_TX_ACTIVITY_FOUND_ON_PROBED_CHAINS",
        "evidence": f"resolved_rows={resolved_rows}/{total}; resolved_addresses={resolved_addresses}/{unique_n}; ambiguous_rows={ambiguous_rows}; unresolved_rows={unresolved_rows}",
    }, {
        "question": "Did activity probing restore validation/test EVM winner coverage?",
        "answer": "YES" if valtest_resolved_rows > 0 else "NO",
        "classification": "VALIDATION_TEST_EVM_WINNERS_ACTIVITY_RESOLVED" if valtest_resolved_rows > 0 else "NO_VALIDATION_TEST_EVM_ACTIVITY_RESOLUTION",
        "evidence": f"valtest_resolved_rows={valtest_resolved_rows}/{valtest_rows}",
    }, {
        "question": "Should full EVM transaction expansion be run now?",
        "answer": "YES" if final == "EVM_ACTIVITY_PROBE_RESOLVED_VALIDATION_TEST_WINNERS_READY_FOR_TX_EXPANSION" else "NO",
        "classification": final,
        "evidence": "Proceed only if activity probing resolves validation/test EVM winners.",
    }]

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(OUT / "07_activity_probe_decision_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "classification": "THESIS_ETHERSCAN_V2_EVM_ACTIVITY_PROBE_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "created_at": now_iso(),
        "safety": {
            "read_only_external_metadata_queries": True,
            "etherscan_v2_queries": True,
            "minimal_activity_probe_only": True,
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
            "chain_ids_tried": [{"chainid": c, "chain_name": n} for c, n in CHAIN_IDS],
            "unique_addresses": unique_n,
            "probes": [
                "account.txlist offset=1",
                "account.tokentx address offset=1",
                "account.txlistinternal offset=1",
                "account.balance",
            ],
        },
        "counts": {
            "evm_winner_rows": total,
            "unique_evm_addresses": unique_n,
            "resolved_winner_rows": resolved_rows,
            "resolved_unique_addresses": resolved_addresses,
            "ambiguous_winner_rows": ambiguous_rows,
            "unresolved_winner_rows": unresolved_rows,
            "validation_test_winner_rows": valtest_rows,
            "validation_test_resolved_winner_rows": valtest_resolved_rows,
            "fetch_rows": int(len(fetches)),
            "fetch_errors": fetch_errors,
            "api_status_counts": api_status_counts,
            "api_message_counts": api_message_counts,
        },
        "final_scientific_conclusion": final,
        "outputs": {
            "fetch_summary": str(OUT / "01_etherscan_v2_activity_fetch_summary.csv"),
            "activity_probe_by_address": str(OUT / "02_etherscan_v2_activity_probe_by_address.csv"),
            "address_activity_resolution": str(OUT / "03_evm_address_activity_resolution.csv"),
            "winner_joined_activity_resolution": str(OUT / "04_evm_winner_joined_activity_resolution.csv"),
            "resolution_by_split": str(OUT / "05_resolution_by_split.csv"),
            "resolution_by_chain": str(OUT / "06_resolution_by_chain.csv"),
            "decision_table": str(OUT / "07_activity_probe_decision_table.csv"),
            "summary_json": str(OUT / "thesis_etherscan_v2_evm_activity_probe_summary.json"),
            "summary_md": str(OUT / "thesis_etherscan_v2_evm_activity_probe_summary.md"),
        },
    }

    summary_json = OUT / "thesis_etherscan_v2_evm_activity_probe_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3J — Etherscan V2 Minimal EVM Activity Probe")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only external metadata queries")
    lines.append("- Etherscan V2 minimal activity probe only")
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
    lines.append("- probes: `txlist offset=1`, `tokentx(address) offset=1`, `txlistinternal offset=1`, `balance`")
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

    summary_md = OUT / "thesis_etherscan_v2_evm_activity_probe_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": final,
        "evm_winner_rows": total,
        "unique_evm_addresses": unique_n,
        "resolved_winner_rows": resolved_rows,
        "resolved_unique_addresses": resolved_addresses,
        "validation_test_winner_rows": valtest_rows,
        "validation_test_resolved_winner_rows": valtest_resolved_rows,
        "fetch_errors": fetch_errors,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
