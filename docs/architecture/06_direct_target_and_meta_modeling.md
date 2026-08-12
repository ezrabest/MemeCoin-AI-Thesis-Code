# 06 — Direct Target and Meta-Modeling

## Purpose

Document the shift from x2 return targets to direct net-profit labels, Phase D1 findings, dataset construction plan, and meta-model/stacking inputs.

## Diagram

See [diagrams/direct_target_pipeline.mmd](diagrams/direct_target_pipeline.mmd).

## Why x2 Was Useful but Insufficient

The x2 (2× return) target enabled model discovery and policy sweeps but does not directly answer: **"Was this trade net profitable after fees, slippage, TP/SL, and time-stop?"** Phase D1 audited selected trades with direct net-return basis.

## Direct Target Definition

```text
net_profitable_after_exit_policy = net_return_after_TP_SL_time_fee > 0
```

Equivalently: binary label from fee-adjusted exit simulation on each row, not proxy x2 hit.

## Phase D1 Result Interpretation

- Scripts: `phase_d_replay_v5_tiers_direct_target.py`, `phase_d_v4_audit_from_v5_selected_trades.py`
- Basis: `audited_v5_net_return` on selected-trade decomposition
- Tier 1 `TAB_XGB_RF_ALL3` subset remains strongest under direct-target audit
- Outputs under `data/training/manual_verified_results/phase_d_exit_target_audit_v4_from_v5/`

## Next: Full Direct Target Dataset (E3)

Build a complete labeled dataset where every row carries:

- Features at decision time
- Exit policy applied (TP, SL, time-stop, fees)
- `net_return_after_exit`
- `net_profitable_after_exit_policy` label
- Source artifact provenance (manifest hashes)

## Recommended Training Order

1. **XGB direct target** (E4) — broad ranker first
2. **RF direct target** (E4) — confirmation model
3. **TAB direct target** (E5) — only after XGB/RF justify heavier TabICL run
4. **Tiered consensus rerun** (E6) — on direct-target predictions
5. **Meta-model / stacking** (E7)

## Meta-Model Inputs (Future)

| Input | Role |
|-------|------|
| `score_xgb`, `score_tab`, `score_rf` | Base model scores |
| Ranks per model | Relative standing |
| `consensus_tier` | Tier 1 / Tier 2 / reject |
| `filter`, `horizon` | Policy regime |
| Pair cap state | Concentration context |
| Liquidity / activity | Regime features |
| Exit policy context | TP, SL, time-stop |
| RSS/news sentiment context | Layer 3 features |
| Later enrichment features | Helius/wallet/reputation |

Meta-model output: refined eligibility probability or tier-weighted score — **not** a replacement for audit trail.

## Current State

- Phase D1 audit complete on v5 selected trades
- No full direct-target training dataset or retrained models
- x2-labeled RF artifacts still used at runtime

## Target State

- Direct-target models replace x2 proxies for production eligibility
- Meta-model combines scores + consensus + context under manifest discipline
- All training exports registered in artifact registry (E1)

## Key Inputs

- Phase B V5.1 selected trades, exit sim outputs, CLEAN_MODEL_INPUT features

## Key Outputs

- `direct_target_dataset.parquet` + manifest
- Direct-target model artifacts (XGB, RF, TAB)
- Meta-model artifact and evaluation report

## Consumers

- Runtime scoring (E10), consensus tier rerun (E6), UI model panels

## Open Questions

- Label leakage guards for forward return windows
- Whether meta-model should predict tier-conditional profitability vs global

## Non-Goals

- Building direct-target dataset or retraining in Phase E0
- Changing existing model artifacts or training scripts
