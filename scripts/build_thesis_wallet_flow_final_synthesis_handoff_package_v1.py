from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT = ROOT / "data" / "audits" / f"thesis_wallet_flow_final_synthesis_handoff_{STAMP}"
SOURCE_SUMMARIES = OUT / "source_summaries"
TABLES = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SUMMARIES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

AUDIT_ROOTS = {
    "03_full_solana_coverage": ROOT / "data" / "audits" / "thesis_wallet_flow_coverage_expansion_audit_20260810_225748",
    "03b_full_solana_label_directionality": ROOT / "data" / "audits" / "thesis_wallet_flow_label_association_directionality_audit_20260810_231521",
    "03c_full_solana_chronology": ROOT / "data" / "audits" / "thesis_wallet_flow_sample_chronology_universe_audit_20260810_231610",
    "03e_missing_winner_chain_provider": ROOT / "data" / "audits" / "thesis_missing_winner_chain_provider_feasibility_audit_20260810_231253",
    "03f_evm_feasibility": ROOT / "data" / "audits" / "thesis_evm_non_solana_wallet_flow_feasibility_deep_audit_20260810_231702",
    "03g_evm_chain_readiness": ROOT / "data" / "audits" / "thesis_evm_chain_resolution_provider_readiness_audit_20260810_232055",
    "03h_dexscreener_chain_resolution": ROOT / "data" / "audits" / "thesis_dexscreener_evm_chain_resolution_audit_20260810_232527",
    "03i_etherscan_chain_probe": ROOT / "data" / "audits" / "thesis_etherscan_v2_evm_chain_probe_audit_20260811_001641",
    "03j_etherscan_activity_probe": ROOT / "data" / "audits" / "thesis_etherscan_v2_evm_activity_probe_audit_20260811_004707",
    "etherscan_key_usage_diagnostic": ROOT / "data" / "audits" / "thesis_etherscan_key_usage_diagnostic_20260811_001358",
}

EXCLUDE_EXACT = {
    "03_helius_transaction_rows_assigned.csv",
    "04_helius_transfer_rows_assigned.csv",
}

MAX_COPY_BYTES = 25 * 1024 * 1024


KEY_NUMBERS = {
    "classification": "THESIS_WALLET_FLOW_FINAL_SYNTHESIS_HANDOFF",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "safety": {
        "read_only_audits": True,
        "no_training": True,
        "no_backtest": True,
        "no_trader_db_mutation": True,
        "no_wallet_connection": True,
        "no_live_trading": True,
        "no_trade_authority": True,
    },
    "full_solana_helius": {
        "case_rows": 7785,
        "unique_pair_addresses": 96,
        "merged_helius_windows": 334,
        "transactions_returned": 39342,
        "transaction_rows_assigned": 140969,
        "transfer_rows_assigned": 573497,
        "real_wallet_level_cases": 7552,
        "large_wallet_level_cases": 6344,
        "label_counts": {
            "FLAT": 7682,
            "LOSER": 68,
            "WINNER": 19,
            "NO_LABEL": 16,
        },
        "large_wallet_evidence_by_label": {
            "WINNER": "0/19",
            "LOSER": "9/68",
            "FLAT": "6326/7682",
        },
        "winner_large_wallet_fisher_p": 1.0926730762276547e-14,
        "winner_large_wallet_odds_ratio": 0.005748972964671601,
        "flat_large_wallet_fisher_p": 1.1886888284660341e-45,
        "flat_large_wallet_odds_ratio": 21.55450732708381,
    },
    "solana_chronology": {
        "full_dataset_winners": {
            "total": 94,
            "train": 79,
            "validation": 8,
            "test": 7,
        },
        "solana_resolvable_winners": {
            "total": 19,
            "train": 19,
            "validation": 0,
            "test": 0,
        },
        "conclusion": "NO_VALIDATION_TEST_WINNERS_IN_SOLANA_RESOLVABLE_UNIVERSE",
        "next_step": "NO_MORE_HELIUS_CAN_FIX_CHRONOLOGICAL_WINNER_GENERALIZATION_FOR_THIS_DATASET",
    },
    "missing_winner_non_solana": {
        "total_winner_rows": 94,
        "solana_helius_route_winner_rows": 19,
        "non_solana_or_non_helius_winner_rows": 75,
        "evm_compatible_winner_rows": 59,
        "non_solana_validation_test_winner_rows": 15,
        "evm_validation_test_winner_rows": 9,
        "conclusion": "MAJORITY_WINNERS_OUTSIDE_HELIUS_AND_NON_SOLANA_ROUTE_CAN_RESTORE_CHRONOLOGICAL_WINNERS",
    },
    "evm_followup": {
        "evm_winner_rows": 59,
        "unique_evm_addresses": 38,
        "dexscreener_resolved_winner_rows": 0,
        "etherscan_getcode_resolved_winner_rows": 0,
        "etherscan_activity_resolved_winner_rows": 0,
        "validation_test_resolved_winner_rows": 0,
        "conclusion": "EVM_METADATA_NOT_TRANSACTION_RESOLVABLE_FROM_CURRENT_ADDRESS_FIELDS",
    },
    "final_thesis_safe_conclusion": (
        "Wallet-flow evidence was technically recoverable at scale for Solana via Helius, "
        "but it did not support a pre-event winner/whale-entry signal. The non-Solana "
        "opportunity is substantial because most winners are outside Helius, but the current "
        "metadata does not resolve those rows into chain-specific transaction evidence. "
        "Therefore, wallet-flow should be reported as forensic/context/provenance infrastructure "
        "with clear coverage and lineage limitations, not as a validated predictive alpha source."
    ),
}


MAIN_THREAD_UPDATE = """Wallet-flow / whale-context thesis audit — final synthesis update

This is not an AE-stage reopening and not a trading/profitability claim. It is a thesis-level robustness and reporting audit for the wallet-flow/context component.

Scope and safety:
All runs were read-only. No trader.db mutation, no model training, no backtest, no wallet connection, no live trading, no new LLM trade authority, and no trade execution occurred.

1. Full Solana/Helius wallet-flow expansion
The Solana/Helius audit was expanded from a selected sample to the full Solana-resolvable universe:
- 7,785 Solana-resolvable cases
- 96 unique pair addresses
- 334 merged Helius query windows
- 39,342 transactions returned
- 140,969 transaction rows assigned to cases
- 573,497 transfer rows assigned to cases
- 7,552/7,785 real wallet-level cases
- 6,344/7,785 large wallet-level cases

This resolved the earlier FLAT selection-bias concern because the full Solana-resolvable universe was tested, not only a selected subset.

2. Solana winner signal result
The full Solana result did not support a pre-event winner / whale-entry signal:
- WINNER: 0/19 with large-wallet evidence
- LOSER: 9/68 with large-wallet evidence
- FLAT: 6,326/7,682 with large-wallet evidence

Fisher tests showed large-wallet evidence was depleted in WINNER rows and enriched in FLAT rows:
- WINNER vs non-WINNER large-wallet evidence: p=1.0926730762276547e-14, OR=0.005748972964671601
- FLAT vs non-FLAT large-wallet evidence: p=1.1886888284660341e-45, OR=21.55450732708381

The large-wallet flag is now case-level discriminative, not degenerate:
- real wallet-level cases: 7,552/7,785
- large wallet-level cases: 6,344/7,785
- real-not-large cases: 1,208

3. Solana chronological limitation
The full event-level dataset contains 94 WINNER rows:
- train: 79
- validation: 8
- test: 7

However, the full Solana-resolvable universe contains only 19 WINNER rows, all in train:
- train: 19
- validation: 0
- test: 0

Therefore, no amount of additional Helius querying can create Solana validation/test WINNER coverage. The Solana wallet-flow result is a full-universe Solana association/coverage result, but not a chronologically validated winner predictor.

4. Missing-winner / non-Solana feasibility
A missing-winner chain/provider audit showed that most WINNER rows are outside the current Helius/Solana route:
- total WINNER rows: 94
- Solana/Helius-route WINNER rows: 19
- non-Solana/non-Helius WINNER rows: 75
- EVM-compatible WINNER rows: 59
- non-Solana validation/test WINNER rows: 15
- EVM validation/test WINNER rows: 9

This means that a general “wallet-flow” claim cannot be based on Helius alone. Helius is a Solana-specific wallet-flow route, not a full-market wallet-flow route.

5. EVM / non-Solana follow-up
The EVM path was investigated because it could theoretically recover 59 additional WINNER rows, including 9 validation/test winners. However:
- existing DB metadata left all 59 EVM-compatible winners as EVM_UNKNOWN_UNRESOLVED
- DEX Screener did not resolve the 38 unique 0x addresses into chain-specific pair/token evidence
- Etherscan V2 eth_getCode found no valid deployed contract code on the probed chains
- strict Etherscan V2 activity probing found no transaction/token/internal activity sufficient to resolve those addresses to a chain

Final EVM conclusion:
The available 0x metadata is not sufficient to reconstruct transaction-resolvable EVM wallet-flow evidence from the current address fields. Therefore, EVM/non-Solana wallet-flow remains a justified future extension, not a validated result in the present thesis.

6. Thesis-safe conclusion
Wallet-flow evidence was technically recoverable at scale for Solana via Helius, but it did not support a pre-event winner/whale-entry signal. The non-Solana opportunity is substantial because most winners are outside Helius, but the current metadata does not resolve those rows into chain-specific transaction evidence. Therefore, wallet-flow should be reported as a forensic/context/provenance layer with clear coverage and lineage limitations, not as a validated predictive alpha source.
"""


THESIS_SAFE_WORDING = """# Thesis-safe wallet-flow wording

## Recommended Results wording

The wallet-flow component was evaluated as a read-only context and provenance layer rather than as an execution or trade-authority mechanism. For Solana-resolvable candidates, the study expanded Helius-based extraction to the full available Solana universe, covering 7,785 candidate cases across 96 pair addresses. This produced substantial technical coverage, including 7,552 cases with real wallet-level evidence and 6,344 cases with large wallet-level evidence.

However, this coverage did not translate into a pre-event winner signal. In the full Solana-resolvable universe, 0/19 WINNER rows exhibited large-wallet evidence, compared with 9/68 LOSER rows and 6,326/7,682 FLAT rows. Thus, in this dataset, the observed Solana wallet-flow signal was concentrated in FLAT cases rather than in future WINNER cases.

This result should not be interpreted as chronologically validated prediction because all 19 Solana-resolvable WINNER rows occurred in the training split, with no Solana WINNER rows in validation or test. Additional Helius calls cannot repair this limitation because it arises from the chronological distribution of the available Solana data rather than from incomplete Helius coverage.

A separate missing-winner feasibility audit showed that most WINNER rows were outside the Solana/Helius route. Of 94 total WINNER rows, only 19 were Solana/Helius-resolvable, while 75 were non-Solana or non-Helius rows. Fifty-nine of these appeared EVM-compatible and included nine validation/test winners. This supports the scientific importance of future non-Solana wallet-flow reconstruction. However, DEX Screener and Etherscan V2 probes showed that the currently stored 0x address fields were not sufficient to reconstruct chain-specific transaction evidence for these EVM rows.

Accordingly, wallet-flow is best interpreted as a technically feasible forensic and provenance layer with strong Solana coverage, but not as a validated alpha source in the present thesis. Non-Solana wallet-flow reconstruction remains an important future extension requiring improved chain-specific address lineage.

## Recommended Discussion wording

The wallet-flow experiments demonstrate both the value and the limitations of external on-chain context. They show that high-volume read-only extraction is technically feasible for Solana via Helius, but they also show that coverage alone is not equivalent to predictive validity. The full Solana audit failed to identify a pre-event whale-entry pattern among WINNER cases. At the same time, the absence of validation/test Solana winners prevents chronological validation of the Solana-specific result.

The EVM follow-up is equally informative. It showed that many winners are outside the Solana route and that non-Solana expansion could materially improve winner coverage. Nevertheless, the currently available metadata does not resolve those rows into chain-specific transaction evidence. This finding identifies a concrete engineering requirement for future versions: order/candidate lineage must preserve chain-specific pair or token-contract identifiers sufficient for transaction-provider reconstruction.

## One-sentence thesis conclusion

Wallet-flow was validated as read-only context and provenance infrastructure, but not as a predictive whale-entry alpha source.
"""


README = f"""# Wallet-flow Final Synthesis Handoff Package

Created: {datetime.now(timezone.utc).isoformat()}

## Purpose

This package consolidates the thesis wallet-flow / whale-context audit chain.

It is intended for:
- main-thread handoff,
- thesis Results/Discussion reporting,
- evidence packaging,
- future audit traceability.

## Safety boundary

All included audits were read-only and did not perform:
- model training,
- backtesting,
- trader.db mutation,
- wallet connection,
- live trading,
- trade execution,
- LLM trade authority.

## Included audit roots

""" + "\n".join(f"- `{k}`: `{v}`" for k, v in AUDIT_ROOTS.items()) + """

## Key conclusion

Wallet-flow evidence was technically recoverable at scale for Solana via Helius, but it did not support a pre-event winner/whale-entry signal. Non-Solana/EVM expansion is scientifically justified because most winners are outside the Helius route, but current metadata does not resolve those rows into chain-specific transaction evidence.

## Raw files excluded

The large raw Helius assignment files are intentionally excluded from the ZIP:
- `03_helius_transaction_rows_assigned.csv`
- `04_helius_transfer_rows_assigned.csv`

Their paths and sizes are recorded in `06_excluded_large_files_manifest.csv` when present.
"""


def safe_copy(src: Path, dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "source_path": str(src),
        "package_path": str(dest.relative_to(OUT)),
        "size_bytes": src.stat().st_size,
        "copied": True,
        "reason": "",
    }


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten(prefix: str, obj: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
    elif isinstance(obj, list):
        rows.append({"key": prefix, "value": json.dumps(obj, ensure_ascii=False)})
    else:
        rows.append({"key": prefix, "value": obj})


def main() -> None:
    manifest: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    write_text(OUT / "00_README_WALLET_FLOW_FINAL_HANDOFF.md", README)
    write_text(OUT / "03_main_thread_update_wallet_flow_final.txt", MAIN_THREAD_UPDATE)
    write_text(OUT / "04_thesis_safe_wallet_flow_wording.md", THESIS_SAFE_WORDING)

    key_json = OUT / "01_wallet_flow_final_key_numbers.json"
    key_json.write_text(json.dumps(KEY_NUMBERS, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    flatten("", KEY_NUMBERS, flat_rows)
    write_csv(OUT / "02_wallet_flow_final_key_numbers_flat.csv", flat_rows)

    for audit_name, root in AUDIT_ROOTS.items():
        if not root.exists():
            excluded.append({
                "audit": audit_name,
                "source_path": str(root),
                "size_bytes": "",
                "reason": "AUDIT_ROOT_NOT_FOUND",
            })
            continue

        for p in sorted(root.iterdir()):
            if not p.is_file():
                continue

            size = p.stat().st_size

            if p.name in EXCLUDE_EXACT:
                excluded.append({
                    "audit": audit_name,
                    "source_path": str(p),
                    "size_bytes": size,
                    "reason": "EXCLUDED_RAW_LARGE_ASSIGNMENT_FILE",
                })
                continue

            if size > MAX_COPY_BYTES:
                excluded.append({
                    "audit": audit_name,
                    "source_path": str(p),
                    "size_bytes": size,
                    "reason": f"EXCLUDED_OVER_{MAX_COPY_BYTES}_BYTES",
                })
                continue

            if p.suffix.lower() in {".json", ".md", ".txt"}:
                dest = SOURCE_SUMMARIES / audit_name / p.name
                manifest.append({"audit": audit_name, **safe_copy(p, dest)})
            elif p.suffix.lower() in {".csv"}:
                dest = TABLES / audit_name / p.name
                manifest.append({"audit": audit_name, **safe_copy(p, dest)})

    write_csv(OUT / "05_manifest.csv", manifest)
    write_csv(OUT / "06_excluded_large_files_manifest.csv", excluded)

    zip_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))

    summary = {
        "status": "OK",
        "output_root": str(OUT),
        "zip": str(zip_path),
        "files_copied": len(manifest),
        "files_excluded": len(excluded),
        "readme": str(OUT / "00_README_WALLET_FLOW_FINAL_HANDOFF.md"),
        "main_thread_update": str(OUT / "03_main_thread_update_wallet_flow_final.txt"),
        "thesis_safe_wording": str(OUT / "04_thesis_safe_wallet_flow_wording.md"),
        "manifest": str(OUT / "05_manifest.csv"),
        "excluded_manifest": str(OUT / "06_excluded_large_files_manifest.csv"),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
