from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
ENV_FILE = ROOT / ".env"
AUDITS = ROOT / "data" / "audits"
OUT = AUDITS / f"thesis_etherscan_key_usage_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://api.etherscan.io/v2/api"


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

        if k == "ETHERSCAN_API_KEY" and v.strip():
            return v.strip()

    raise RuntimeError("ETHERSCAN_API_KEY not found")


def call(params: dict) -> dict:
    key = read_key()
    request_params = dict(params)
    request_params["apikey"] = key

    safe_params = dict(params)
    safe_params["apikey"] = "REDACTED"

    url = BASE + "?" + urllib.parse.urlencode(request_params)
    safe_url = BASE + "?" + urllib.parse.urlencode(safe_params)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "memecoin-thesis-etherscan-usage-diagnostic/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body) if body else {}

    return {
        "safe_url": safe_url,
        "http_status": 200,
        "data": data,
    }


def find_latest_3i() -> Path | None:
    roots = sorted(
        [p for p in AUDITS.glob("thesis_etherscan_v2_evm_chain_probe_audit_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return roots[0] if roots else None


def summarize_fetch_cache(latest_3i: Path | None) -> dict:
    if latest_3i is None:
        return {"latest_3i_root": None, "fetch_summary_exists": False}

    fetch_csv = latest_3i / "01_etherscan_v2_fetch_summary.csv"
    if not fetch_csv.exists():
        return {"latest_3i_root": str(latest_3i), "fetch_summary_exists": False}

    df = pd.read_csv(fetch_csv, low_memory=False)

    summary = {
        "latest_3i_root": str(latest_3i),
        "fetch_summary_exists": True,
        "fetch_rows": int(len(df)),
    }

    for col in ["cache_status", "http_status", "api_status", "api_message", "error"]:
        if col in df.columns:
            vc = df[col].fillna("").astype(str).value_counts().to_dict()
            summary[f"{col}_counts"] = {str(k): int(v) for k, v in vc.items()}

    return summary


def extract_usage(resp: dict) -> dict:
    data = resp.get("data", {})
    result = data.get("result") if isinstance(data, dict) else None

    if isinstance(result, dict):
        return {
            "status": data.get("status"),
            "message": data.get("message"),
            "creditsUsed": result.get("creditsUsed"),
            "creditsAvailable": result.get("creditsAvailable"),
            "creditLimit": result.get("creditLimit"),
            "limitInterval": result.get("limitInterval"),
            "intervalExpiryTimespan": result.get("intervalExpiryTimespan"),
        }

    return {
        "status": data.get("status") if isinstance(data, dict) else None,
        "message": data.get("message") if isinstance(data, dict) else None,
        "result": str(result)[:300],
    }


def main() -> None:
    latest_3i = find_latest_3i()
    cache_summary = summarize_fetch_cache(latest_3i)

    usage_before_resp = call({
        "module": "getapilimit",
        "action": "getapilimit",
    })
    usage_before = extract_usage(usage_before_resp)

    # Known Ethereum mainnet contract probe: WETH contract.
    # This is a single read-only proxy.eth_getCode call.
    test_resp = call({
        "chainid": "1",
        "module": "proxy",
        "action": "eth_getCode",
        "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "tag": "latest",
    })

    test_data = test_resp.get("data", {})
    test_result = test_data.get("result") if isinstance(test_data, dict) else None
    valid_hex_code = isinstance(test_result, str) and re.fullmatch(r"0x[0-9a-fA-F]+", test_result or "") is not None and test_result.lower() != "0x"

    usage_after_resp = call({
        "module": "getapilimit",
        "action": "getapilimit",
    })
    usage_after = extract_usage(usage_after_resp)

    credits_before = usage_before.get("creditsUsed")
    credits_after = usage_after.get("creditsUsed")

    credit_delta = None
    try:
        credit_delta = int(credits_after) - int(credits_before)
    except Exception:
        credit_delta = None

    summary = {
        "classification": "ETHERSCAN_KEY_USAGE_DIAGNOSTIC_COMPLETED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "read_only": True,
            "single_known_contract_probe": True,
            "full_transaction_history_pulled": False,
            "db_mutated": False,
            "wallet_connected": False,
            "key_value_printed": False,
        },
        "cache_summary_for_latest_3i": cache_summary,
        "usage_before": usage_before,
        "known_contract_probe": {
            "safe_url": test_resp["safe_url"],
            "json_keys": list(test_data.keys()) if isinstance(test_data, dict) else [],
            "result_prefix": str(test_result)[:80],
            "valid_hex_contract_code": bool(valid_hex_code),
            "result_length": len(test_result) if isinstance(test_result, str) else None,
        },
        "usage_after": usage_after,
        "credit_delta_observed": credit_delta,
    }

    summary_json = OUT / "etherscan_key_usage_diagnostic_summary.json"
    summary_md = OUT / "etherscan_key_usage_diagnostic_summary.md"

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Etherscan API Key / Usage Diagnostic")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only")
    lines.append("- One known-contract `eth_getCode` probe")
    lines.append("- No full transaction-history pull")
    lines.append("- No DB mutation")
    lines.append("- No wallet connection")
    lines.append("- API key value was not printed")
    lines.append("")
    lines.append("## Latest 3I cache/fetch summary")
    lines.append(f"- latest 3I root: `{cache_summary.get('latest_3i_root')}`")
    lines.append(f"- fetch rows: {cache_summary.get('fetch_rows')}")
    lines.append(f"- cache status counts: {cache_summary.get('cache_status_counts')}")
    lines.append(f"- API status counts: {cache_summary.get('api_status_counts')}")
    lines.append(f"- API message counts: {cache_summary.get('api_message_counts')}")
    lines.append(f"- error counts: {cache_summary.get('error_counts')}")
    lines.append("")
    lines.append("## Usage before")
    lines.append(f"- `{usage_before}`")
    lines.append("")
    lines.append("## Known contract probe")
    lines.append(f"- valid hex contract code: {valid_hex_code}")
    lines.append(f"- result prefix: `{str(test_result)[:80]}`")
    lines.append(f"- result length: {len(test_result) if isinstance(test_result, str) else None}")
    lines.append("")
    lines.append("## Usage after")
    lines.append(f"- `{usage_after}`")
    lines.append("")
    lines.append("## Observed credit delta")
    lines.append(f"`{credit_delta}`")
    lines.append("")
    lines.append("## Output files")
    lines.append(f"- `{summary_json.name}`")
    lines.append(f"- `{summary_md.name}`")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "known_contract_probe_valid_hex_code": bool(valid_hex_code),
        "credits_before": credits_before,
        "credits_after": credits_after,
        "credit_delta_observed": credit_delta,
        "latest_3i_cache_status_counts": cache_summary.get("cache_status_counts"),
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
