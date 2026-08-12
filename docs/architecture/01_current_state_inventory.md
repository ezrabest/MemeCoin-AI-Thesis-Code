# 01 — Current State Inventory

## Purpose

Inventory what exists today in the MemeCoin AI Trader codebase versus what is partial or missing, to ground Phase E implementation planning.

## Diagram

No dedicated diagram; see [00_system_overview.md](00_system_overview.md) and [diagrams/system_overview.mmd](diagrams/system_overview.mmd).

## Existing

| Component | Location / Notes |
|-----------|------------------|
| DexScreener market data ingestion | `app/dexscreener.py`, `app/live.py`, `app/analytics/scan_persist.py` |
| SQLite runtime storage | `app/database.py`, `data/trader.db` — coins, snapshots, signals, paper_trades, pipeline_audit, sentiment_records |
| CSV/Parquet research artifacts | `data/training/`, `data/training/manual_verified_results/` |
| RF offline artifacts | `data/training/models/*.joblib`, `baseline_metrics.json` |
| TAB offline evaluation | `app/training/tabicl_v2_eval.py`, `tabicl_v2_predictions_*.parquet` |
| XGB clean CUDA evaluation | `scripts/run_xgb_clean_full_cuda.py`, `data/training/manual_verified_results/xgb_clean_full/` |
| Exit simulation | `data/training/manual_verified_results/exit_sim_fixed/`, `exit_sim_xgb_full/` |
| Phase B V5.1 selected-trade decomposition | `scripts/phase_b_v5_audited_consensus_rerun.py`, v5 outputs |
| Phase D1 direct net-profit audit | `scripts/phase_d_v4_audit_from_v5_selected_trades.py`, phase_d outputs |
| RSS sentiment | `app/analytics/sentiment.py` — Cointelegraph + Decrypt; archives to SQLite |
| Paper/demo trading infrastructure | `app/execution/paper.py`, `app/observability/actionability.py`, `app/observability/economic_gate.py` |
| Solana raw parser/probe | `app/parsers/solana_pool_activity.py`, `app/providers/solana_rpc.py` |
| Partial Helius validation/enrichment | `app/providers/helius.py`, `app/parsers/solana_wallet_behavior.py` |

## Partial

| Component | Gap |
|-----------|-----|
| Helius live validation | Budget pacing exists; not invoked on every scan candidate |
| wSOL live-smoke validation | Research scripts in phase_c whale audit; not production path |
| Qwen operational layer | Ollama client exists; used mainly on whale-like events, not full candidate memo pipeline |
| Gemini selective audit | Default for whale decisions; not tiered-consensus selective audit |
| Wallet-level whale intelligence | Parsers exist; coarse `whale_score` gate showed no independent value vs `NO_WHALE_FILTER` |
| UI panels for new architecture | Dashboard has sentiment; no consensus tier, model scores, or lineage panels |
| Structured audit reason sanitization | `parse_audit_reasons_field()` in diagnostics only; API bug in `app/api.py` |

## Missing

| Component | Target Phase |
|-----------|--------------|
| Full direct target dataset | E3 |
| Direct-target RF/XGB/TAB retraining | E4, E5 |
| Meta-model / stacking | E7 |
| Artifact registry | E1 |
| Unified candidate schema | E2 |
| Production-grade reputation/scam layer | E8 |
| Full UI integration for four-layer architecture | E11 |

## Key Inputs

- Codebase under `app/`, `scripts/`, `static/`, `data/`
- Research manifests under `manual_verified_results/`

## Key Outputs

- This inventory as baseline for gap analysis in E1–E12

## Consumers

- All Phase E architecture and roadmap documents
- Implementation teams for E1+

## Open Questions

- Original consensus intersection generator script not in repo (`phase_b_v5_locate_consensus_source.py` finding).
- Whether live economic gate thresholds align with best offline policies (XGB 0.5% / TAB 2% / TP 2.0308 / SL 0.80).

## Non-Goals

- Modifying any inventoried component in Phase E0
- Re-running training or sweeps to refresh inventory
