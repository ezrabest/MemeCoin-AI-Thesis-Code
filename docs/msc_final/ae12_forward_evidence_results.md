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

# AE12 Forward Evidence Results

## Summary

- Candidate evidence rows: `63531`
- Matured outcome rows: `317655`
- Missing data warnings: `22125`
- Missed winners: `6219`
- Missed winners by horizon: `{'5m': 1152, '15m': 1170, '1h': 1213, '6h': 1415, '24h': 1269}`

## Horizon maturity

| Horizon | Matured | Not matured | No-lookahead OK | Matured but no snapshots |
|---|---:|---:|---:|---:|
| 5m | 63342 | 189 | 63113 | 229 |
| 15m | 63181 | 350 | 63167 | 14 |
| 1h | 62685 | 846 | 62677 | 8 |
| 6h | 59662 | 3869 | 59661 | 1 |
| 24h | 36906 | 26625 | 36905 | 1 |

Price freshness and horizon maturity are distinct: maturity means enough time elapsed;
freshness concerns entry-price staleness at decision time.

## Trade vs no-trade (outcome labels only)

- Traded: `221`
- Not traded: `63310`
- Interpretations: `{'5m': 'MIXED', '15m': 'MIXED', '1h': 'MIXED', '6h': 'MIXED', '24h': 'TRADED_OUTPERFORMED'}`

forward returns are outcome labels only.

## Limitations

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set
- This report is **not live-approved** and **not profitability-proven**
- future work includes strict policy calibration, runtime UI hardening, longer forward validation, and optional live-wallet gate only after separate approval
