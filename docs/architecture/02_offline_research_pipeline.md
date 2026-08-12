# 02 — Offline Research Pipeline

## Purpose

Document the offline research pipeline from raw market data through model evaluation, exit simulation, consensus decomposition, and Phase D1 direct net-profit audit.

## Diagram

See [diagrams/offline_research_pipeline.mmd](diagrams/offline_research_pipeline.mmd).

```text
raw market data
→ manually verified datasets
→ CLEAN_MODEL_INPUT
→ RF / TAB / XGB predictions
→ exit simulation
→ strict policy selection
→ model comparison
→ consensus decomposition
→ Phase D1 direct net-profit audit
→ next: full direct target dataset
```

## Current State

| Stage | Implementation |
|-------|----------------|
| Dataset build | `scripts/build_training_dataset.py`, `app/training/dataset_builder.py` → `model_ready_dataset.parquet` |
| RF training | `scripts/train_baseline_model.py`, `app/training/baseline_model.py` |
| TAB evaluation | `scripts/evaluate_tabicl_v2.py`, `app/training/tabicl_v2_eval.py` |
| XGB evaluation | `scripts/run_xgb_clean_full_cuda.py`, `scripts/evaluate_xgb_manual_verified_clean.py` |
| Exit simulation | Artifacts in `exit_sim_fixed/`, `exit_sim_xgb_full/` |
| Policy selection | Strict validation-selected policies; filters, horizons, top_pct, pair_cap, TP/SL grid |
| Consensus | `phase_b_v5_audited_consensus_rerun.py`, `phase_b_two_of_three_composition_v3/v4.py` |
| Direct target audit | `phase_d_v4_audit_from_v5_selected_trades.py` |

### Best-known standalone policies (documented closure)

**XGB — broad ranker:**

```text
model: XGB
filter: RAW_ALL_VERIFIED
horizon: 24h
top_pct: 0.5%
pair_cap: 50
TP: 2.0308
SL: 0.80
```

**TAB — focused ranker:**

```text
filter: LIQ_5K_HIGH_ACTIVITY
horizon: 4h
top_pct: 2%
pair_cap: 50
TP: 2.0308
SL: 0.80
```

**RF:** Weak standalone vs XGB/TAB; valuable as conservative confirmation, especially when agreeing with TAB.

### Filters, horizons, pair caps, TP/SL

- Filters: `RAW_ALL_VERIFIED`, `LIQ_5K_HIGH_ACTIVITY`, whale variants (`NO_WHALE_FILTER` ≈ `LIQ_5K_HIGH_ACTIVITY` in latest work)
- Horizons: 4h (TAB focus), 24h (XGB focus)
- Pair cap: 50 (standard in closure policies)
- TP/SL: 2.0308 / 0.80 (fee-adjusted evaluation in exit sim)
- Fee-adjusted evaluation applied in exit simulation manifests

### Why no more heavy x2 sweeps

The x2 return target was useful for discovery but is insufficient as the final training objective. Phase D1 established direct net-profit audit on selected trades. Further heavy x2 grid sweeps are deprioritized in favor of direct-target dataset construction (E3) and direct-target model retraining (E4–E5).

## Target State

- Single canonical builder produces `CLEAN_MODEL_INPUT` and all derived prediction tables
- Every export carries manifest with `content_hash`, `schema_hash`, `git_commit_hash`
- Direct-target labeled dataset feeds XGB → RF → TAB retraining order
- Tiered consensus rerun on direct-target models (E6)

## Key Inputs

- SQLite snapshots and manually verified row sets
- Model feature JSONs per filter/horizon
- Exit policy grid parameters

## Key Outputs

- Prediction parquets, strict comparison CSVs, consensus decomposition tables
- Phase D audit reports with combo labels (`TAB_XGB_RF_ALL3`, etc.)
- Future: `direct_target_dataset.parquet` with manifest

## Consumers

- Phase E model integration (E4–E7)
- Documentation and branch upload artifacts
- Future artifact registry (E1)

## Open Questions

- Consensus generator source script missing from repo — lineage of some CSVs is data-only
- Optimal direct-target label window vs current TP/SL/time-stop definitions

## Non-Goals

- Running training, XGB sweeps, or direct-target construction in Phase E0
- Changing offline script behavior
