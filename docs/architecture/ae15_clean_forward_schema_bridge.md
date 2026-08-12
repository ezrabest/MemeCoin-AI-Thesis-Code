# AE15 — Direct Target / Clean Forward Schema Bridge

## Purpose

AE15 builds the **canonical Clean Forward schema and lineage layer** that connects:

```
Clean Forward Market Feed
→ CleanForwardCandidate
→ CleanForwardDecisionInput
→ Direct Target / model evidence references (shadow/optional)
→ GateKeeper / RiskGuard decision trace
→ Paper order
→ Paper position
→ Skip / outcome label contracts
→ Audit-ready lineage
```

AE15 is a **data contract, schema, lineage, and audit** phase. It does not approve model authority, live trading, or profitability.

## Why AE15 exists

AE14 proved a real Clean Forward market row can reach paper/demo execution (`AE14_REAL_CLEAN_FORWARD_CLOSURE_PASS`), but left a non-blocking counter discrepancy:

- `paper_orders_opened = 1`
- `paper_positions_opened = 2`

Carry-forward note: `AE14_POSITION_COUNTER_RECONCILIATION_PENDING`

AE15 makes that class of discrepancy **impossible to leave unexplained**: every paper position must link to decision/candidate lineage (or be explicitly classified as preexisting / reconstructed / order-missing).

## Package

`app/clean_forward/`

| Module | Role |
|--------|------|
| `schema.py` | Record dataclasses + deterministic ID builders |
| `identity.py` | Chain-aware address normalization + instrument identity |
| `validation.py` | Clean-feed eligibility + identity separation |
| `serialization.py` | Stable JSON / payload hashing |
| `discovery.py` | Local AE14 + smoke poll discovery (no external APIs) |
| `lineage.py` | Candidate/decision builders + AE14 order↔position reconciliation |

## Eligibility rule

A row is clean-forward-valid only when **all** are true:

- `verification_status = provider_pair_verified`
- `freshness_status = fresh`
- `identity_status = pair_and_tokens_separated`
- `shown_as_token_contract = false`
- `paper_demo_only = true`
- `live_trading_ready = false`

## Identity rules

- Pair address must not be stored as token contract.
- Base and quote token addresses remain separate from pair address.
- `coin_id` must not be invented (null/absent for Clean Forward).
- Solana addresses preserve case.
- EVM addresses may be lowercased only on EVM-compatible chains.

## Deterministic IDs

### `clean_forward_candidate_id`

SHA-256 over:

`namespace | chain | provider | pair_address_for_id | base | quote | observed/fetched | provider_payload_hash`

Must **not** include model scores, consensus tier, paper order/position ids, or future outcomes.

### `clean_forward_decision_input_id`

SHA-256 over:

`candidate_id | snapshot_ts | preset | risk_mode | strict/exploration | decision_input_version`

Model score fields (`xgb_score`, `tab_score`, `rf_score`) are optional/shadow only. AE15 does not grant model authority.

## AE14 reconciliation

From AE14 artifacts:

1. **Position #1 (Bonk/MET)** — explicit `PaperTrader.open_position` with reconstructed order link.
2. **Position #2 (PUMP/MET)** — `demo_bot.run_once()` opened a second Clean Forward position; AE14 incremented `paper_positions_opened` but did **not** allocate a separate `paper_orders_opened` / durable `paper_order_id`.

AE15 records:

- `positions_without_order` includes position `2`
- `counter_consistency_status = AE14_POSITION_COUNTER_RECONCILIATION_PENDING`
- `ae14_discrepancy_resolved = false`

This is an explained limitation, not a silent pass.

## Run modes

```bash
python scripts/run_ae15_clean_forward_schema_bridge.py --audit-only
python scripts/run_ae15_clean_forward_schema_bridge.py --build
python scripts/run_ae15_clean_forward_schema_bridge.py --reconcile-ae14
```

Optional:

```bash
--ae14-root <path>
--clean-forward-smoke-root <path>
--output-root <path>
--max-polls <n>
```

Default AE14 root preference:

`data/audits/ae14_real_clean_forward_closure_20260721_210220`

Default smoke root preference:

`data/clean_forward_smoke_2h_20260721_164202`

## Outputs

`data/audits/ae15_clean_forward_schema_bridge_<timestamp>/`

- `reports/ae15_decision_gate.json`
- `reports/ae15_manifest.json`
- `reports/ae15_summary_for_upload.txt`
- `data/clean_forward_*.csv` (+ parquet when pyarrow available)
- `audits/*_audit.json|csv`
- `manifests/ae15_schema_manifest.json`
- `manifests/ae15_source_artifact_index.csv`

## Decision gate values

- `AE15_CLEAN_FORWARD_SCHEMA_BRIDGE_PASS`
- `AE15_PASS_WITH_LINEAGE_LIMITATIONS`
- `AE15_BLOCKED_CLEAN_FORWARD_INPUT_MISSING`
- `AE15_BLOCKED_IDENTITY_SEPARATION_FAILURE`
- `AE15_BLOCKED_LEGACY_DATA_CONTAMINATION`
- `AE15_BLOCKED_ORDER_POSITION_LINEAGE_FAILURE`
- `AE15_BLOCKED_SAFETY_VIOLATION`

## Explicit non-goals

- No model training / retraining (XGB, RF, TAB)
- No backtest
- No profitability claim
- No wallet / private key / live trading
- No Gemini / Qwen / Ollama / Helius / external API calls
- No `trader.db` mutation for AE15 (file-based outputs only)
- Legacy `market_snapshots` / old Market Snapshot Feed are **not** AE15 sources of truth

## Tests

```bash
python -m compileall app scripts tests
pytest tests/test_ae15_clean_forward_schema_bridge.py -q
```

## Downstream (AE16+)

When gate is `AE15_CLEAN_FORWARD_SCHEMA_BRIDGE_PASS` or `AE15_PASS_WITH_LINEAGE_LIMITATIONS`, AE16 may consume:

- `data/clean_forward_candidates.csv`
- `data/clean_forward_decision_inputs.csv`
- `data/clean_forward_paper_execution_links.csv`
- `audits/order_position_lineage_audit.json`
- `reports/ae15_decision_gate.json`
- `manifests/ae15_schema_manifest.json`

AE16 must treat `AE14_POSITION_COUNTER_RECONCILIATION_PENDING` as a lineage constraint for any new order/position counters.
