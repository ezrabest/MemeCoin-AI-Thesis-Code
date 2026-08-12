---
generated_at: 2026-07-15T15:44:18.757250+00:00
source_ae12_output_root: data\audits\ae12_forward_evidence_maturation_20260714_235401
phase: AE12.5 Final MSc Reporting
---

> **Provenance:** Values in this document were generated from AE12 audit artifacts
> (JSON/CSV), not hard-coded constants. Re-run `scripts/run_ae12_generate_final_docs.py`
> after a new AE12 maturation pass to refresh numbers.

**Generated at (UTC):** `2026-07-15T15:44:18.757250+00:00`  
**Source AE12 output root:** `data\audits\ae12_forward_evidence_maturation_20260714_235401`

**Source files used:**
- `data\audits\ae12_forward_evidence_maturation_20260714_235401\reports\ae12_forward_evidence_summary.json`
- `data\audits\ae12_forward_evidence_maturation_20260714_235401\reports\ae12_final_system_readiness_gate.json`
- `data\audits\ae12_forward_evidence_maturation_20260714_235401\data\ae12_trade_vs_no_trade_comparison.csv`
- `data\audits\ae12_forward_evidence_maturation_20260714_235401\data\ae12_strict_vs_exploration_comparison.csv`
- `data\audits\ae12_forward_evidence_maturation_20260714_235401\audits\ae12_wallet_safety_audit.json`
- `E:\Projects\Final Project\memecoin_trader\data\audits\ae12_forward_evidence_maturation_20260714_235401\reports\ae12_forward_evidence_summary.json`

# AE12 Demo Script (Research / Paper-Demo Only)

## Before you start

1. Runtime may need a **manual restart** to pick up AE12.5 API/UI code (no hot-reload assumption).
2. Do **not** connect a real wallet.
3. Do **not** enable live trading.
4. Label everything as paper/demo/exploration, research-only, not live-approved, not profitability-proven.

## Demo steps

1. Start the server manually when approved.
2. Open the dashboard → **AE12 Forward Evidence** tab.
3. Call `GET /api/ae12/status` - expect gate `FORWARD_EVIDENCE_READY_FOR_REPORTING`, `live_ready=false`, `profitability_proven=false`.
4. Call `GET /api/ae12/forward-evidence-summary` - candidate rows `63531`.
5. Show Missed Winners panel - emphasize outcome labels only.
6. Show Trade vs No-Trade - mixed interpretations; not profitability proof.
7. Show Strict vs Exploration - strict approved `0` (zero in this evidence set).
8. Show Qwen panel - NO_TRADE_AUTHORITY; Ollama largely absent.
9. Show Wallet/Safety - wallet_configured=`False`,
   private_key_accessed=`False`,
   live_submission_status=`NOT_SUBMITTED_NO_WALLET`, live trading approval = NO.

## Talking points (safe)

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set

## Do not say

- The system is profitable
- The system is live-ready
- Qwen decides trades
- Strict policy approved trades
- Missed winners prove the strategy would have profited

## Limitations

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set
- This report is **not live-approved** and **not profitability-proven**
- future work includes strict policy calibration, runtime UI hardening, longer forward validation, and optional live-wallet gate only after separate approval
