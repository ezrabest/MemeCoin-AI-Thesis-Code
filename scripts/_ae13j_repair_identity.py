#!/usr/bin/env python3
"""Repair AE13J identity verdict CSV after classify_identity fix (no re-poll)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ae13j_ground_truth_market_data_audit import (  # noqa: E402
    classify_identity,
    write_csv,
    write_json,
)

OUT = ROOT / "data" / "audits" / "ae13j_ground_truth_market_data_audit_20260720_175255"


def main() -> int:
    traces = json.loads((OUT / "ae13j_provider_payload_trace.json").read_text(encoding="utf-8"))
    rows = []
    seen: set[str] = set()
    for t in traces.get("traces") or []:
        pair = str(t.get("displayed_pair_address") or "")
        extracted = t.get("extracted") or {}
        payload = t.get("original_provider_response_object")
        chain = t.get("chain") or extracted.get("chainId")
        token_mint = t.get("db_token_address") or extracted.get("baseToken.address")
        for addr in (pair, extracted.get("baseToken.address")):
            if not addr or addr in seen:
                continue
            seen.add(str(addr))
            verdict = classify_identity(
                address=str(addr),
                chain=chain,
                pair_address=pair,
                token_mint=str(token_mint) if token_mint else None,
                token_contract=None,
                address_role=t.get("live_row_address_role"),
                payload=payload if isinstance(payload, dict) else None,
            )
            dex = str(extracted.get("dexId") or "")
            note = ""
            if str(addr).startswith("DDk1"):
                note = (
                    f"Solana pool/pair for {extracted.get('baseToken.symbol') or t.get('displayed_symbol')}/"
                    f"{extracted.get('quoteToken.symbol') or 'SOL'}; dexId={dex or 'unknown'}. "
                    "Not a token mint. Pump.fun AMM / WSOL-WIF style market account on explorers."
                )
            elif verdict in ("pool_address", "pair_contract", "market_account"):
                note = "Displayed/row identity is pair/pool level, not token contract."
            elif verdict in ("token_mint", "token_contract"):
                note = "Address matches base token identity in provider/DB."
            rows.append(
                {
                    "sampled_address": addr,
                    "chain": chain,
                    "displayed_symbol": t.get("displayed_symbol")
                    or extracted.get("baseToken.symbol"),
                    "pair_address": pair,
                    "token_mint_address": token_mint or "",
                    "token_contract_address": "",
                    "dexId": dex,
                    "provider_url": extracted.get("url") or "",
                    "identity_verdict": verdict,
                    "do_not_call_contract_address": verdict
                    in (
                        "pool_address",
                        "pair_contract",
                        "market_account",
                        "provider_pair_id",
                    ),
                    "evidence_note": note,
                    "raw_payload_found": bool(t.get("raw_payload_found")),
                }
            )
    write_csv(OUT / "ae13j_token_pair_identity_verdict.csv", rows)
    ui = json.loads((OUT / "ae13j_ui_truth_label_recommendation.json").read_text(encoding="utf-8"))
    ui["applied_to_code"] = True
    ui["files_updated"] = [
        "static/index.html",
        "app/api.py",
        "static/product_demo.js",
        "app/ae13b_product/live_market.py",
        "tests/test_ae13b_product_demo.py",
    ]
    write_json(OUT / "ae13j_ui_truth_label_recommendation.json", ui)

    for r in rows:
        a = str(r["sampled_address"])
        if a.startswith("DDk1") or a.startswith("5GZ") or a.startswith("0x40ec"):
            print(a[:28], r["identity_verdict"], r["displayed_symbol"])
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
