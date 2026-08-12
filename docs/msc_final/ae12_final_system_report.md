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

# AE12 Final System Report

## 1. System objective

The MemeCoin AI Trader is an MSc research platform for multimodal memecoin market analysis,
paper/demo decision orchestration, and forward-evidence auditing. This AE12 package
summarizes derived forward-evidence results for reporting. It does **not** authorize live trading
and does **not** claim profitability.

## 2. Architecture

Layers (audit lineage): data collection → RF/XGB/TAB + meta scoring → context intelligence (AE8)
→ consensus decision (AE6/AE7) → LLM audit (AE9; no trade authority) → runtime paper loop (AE11)
→ forward evidence maturation (AE12.3-AE12.4) → observability / final reporting (AE12.5).

AE12.5 exposes cached, read-only views of existing AE12 artifacts. It does not re-run maturation.

## 3. Data collection

From AE12.1 census (when available):

- Latest DB collection timestamp: `2026-07-14T19:49:11.092618+00:00`
- Market snapshot count: `2588840`
- Sentiment/RSS count: `776595`
- Paper/demo evidence: `{'paper_trades_files': 3, 'paper_positions_files': 4, 'opportunity_capture_files': 4, 'trade_decision_files': 4}`
- AE11 loop timestamp: `2026-07-14T17:36:32.815162+00:00`
- AE11 older than DB collection: `True`

## 4. ML models / RF-XGB-TAB / meta-layer

Classic and tabular models produce research signals and meta-layer scores that feed decision
records. Models are not live trade authority. This report does not retrain RF/XGB/TAB.

## 5. Context intelligence

AE8 context freshness and family presence influence audit blockers and exploration gates.
Missing context families appear in AE9 audit blockers and AE12 linkage samples.

## 6. Qwen / Gemini intelligent-agent audit layer

Qwen/Gemini/Ollama are audit/explanation layers, not trade authority.

- ROW_LINKED_AE9_RECORD: `41659`
- MENTION_ONLY: `21872`
- Ollama status: `ABSENT`
- llm_trade_authority_status: `NO_TRADE_AUTHORITY`
- qwen_trade_authority (gate): `False`

## 7. Paper / demo runtime

Runtime paper/demo exploration produced opportunity capture and trade-decision JSONL consumed
by AE12. paper/demo exploration is not live-trading approval.

## 8. Forward evidence methodology

AE12 recomputes horizon maturity and forward returns from market snapshots with no-lookahead
guards. Horizon maturity (enough wall-clock time) is distinct from price freshness at entry.

- Candidate evidence rows: `63531`
- Matured outcome rows: `317655`
- Missing data warnings: `22125`
- Gate: `FORWARD_EVIDENCE_READY_FOR_REPORTING`
- needs_persistence_fix: `False`

### Horizon maturity

- **5m**: matured=63342, not_matured=189, no_lookahead_ok=63113
- **15m**: matured=63181, not_matured=350, no_lookahead_ok=63167
- **1h**: matured=62685, not_matured=846, no_lookahead_ok=62677
- **6h**: matured=59662, not_matured=3869, no_lookahead_ok=59661
- **24h**: matured=36906, not_matured=26625, no_lookahead_ok=36905

## 9. Opportunity capture and missed winners

Missed winners are outcome labels only - they do not prove the strategy would have profited.

- Total missed winners: `6219`
- By horizon: `{'5m': 1152, '15m': 1170, '1h': 1213, '6h': 1415, '24h': 1269}`

## 10. Trade vs no-trade

forward returns are outcome labels only.

| Horizon | Traded | Not traded | Med traded | Med not | Max traded | Max not | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| 5m | 221 | 63310 | 0.0 | 0.0 | 0.49058295964125553 | 0.939334637964775 | MIXED |
| 15m | 221 | 63310 | 0.0 | 0.0 | 0.49058295964125553 | 1.1123024830699775 | MIXED |
| 1h | 221 | 63310 | 0.0 | 0.0 | 0.49058295964125553 | 1.3341772151898734 | MIXED |
| 6h | 221 | 63310 | 0.0 | 0.0 | 1.3253796095444688 | 1.8472775564409032 | MIXED |
| 24h | 221 | 63310 | 0.00395709910209356 | 0.0008928571428571684 | 1.983163045042452 | 3.6921069797782122 | TRADED_OUTPERFORMED |

- Traded count: `221`
- Not traded count: `63310`

## 11. Strict vs exploration

strict policy approved zero candidates in this AE12 evidence set.

- Strict approved: `0`
- Strict blocked: `63531`
- Exploration-only trades: `142`

### Top blockers

- `ACTIVE_PAIR_LOCK`: 32246
- `max_open_positions`: 29461
- `COOLDOWN_ACTIVE`: 1431
- `price_stale_exploration`: 150
- `price_stale_strict`: 58
- `price_price_stale`: 54
- `exploration_flags_not_enabled`: 47
- `no_trade_authority`: 46
- `missing_context`: 38

## 12. Safety / no wallet / no real transaction

- wallet_configured: `False`
- private_key_accessed: `False`
- live_submission_status: `NOT_SUBMITTED_NO_WALLET`
- live_trading_approval: `NO`
- live_trading_ready: `False`
- profitability_proven: `False`
- real_wallet_connected: `False`

## 13. Results

- Readiness gate: `FORWARD_EVIDENCE_READY_FOR_REPORTING`
- Evidence row count: `63531`
- can_proceed_to_ui_final_report: `True`
- Known limitations (from AE12 summary): `['Forward returns are labels only; not profitability proof.', 'SQLite paper_trades may be stale; JSONL paper artifacts preferred.', 'AE11 opportunity capture often writes horizon_matured=false at capture time; AE12 recomputes.', 'AE12.2 field scan looked for reason_not_traded; source field is often reason_for_no_trade.', 'Qwen markers in AE6 llm_context do not imply operational trade authority.', 'Market snapshot coverage may be uneven across pairs/horizons.', 'Runtime collector may still be writing; this pass is derived/read-only against inputs.']`

## 14. Limitations

See dedicated limitations section below. Values above are research/audit outcomes only.

## 15. Future work

future work includes strict policy calibration, runtime UI hardening, longer forward validation,
and optional live-wallet gate only after separate approval.

## Limitations

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set
- This report is **not live-approved** and **not profitability-proven**
- future work includes strict policy calibration, runtime UI hardening, longer forward validation, and optional live-wallet gate only after separate approval
