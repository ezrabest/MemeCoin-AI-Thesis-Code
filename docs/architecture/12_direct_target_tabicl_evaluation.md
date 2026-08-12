# Phase E5 — Direct-Target TabICL Evaluation

Offline research/evaluation phase comparing TabICL / TabICLv2 ranking signal against E4A XGB/RF outputs under the E3 direct economic target (`target_net_profitable_after_exit` → `target_net_profitable`).

## Scope

- **In scope:** dependency audit, TAB inference, metrics, policy evaluation, E4A join, consensus tier reconstruction, artifact registration.
- **Out of scope:** runtime trading, UI, SQLite, LLM calls, Solana/Helius/RSS, production gates.

## Inputs

| Input | Default path |
|-------|----------------|
| E3 direct-target datasets | `data/training/manual_verified_datasets_direct_target_v1` |
| E4A XGB/RF outputs | `data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_full_20260630_195312` |

## Execution modes

| Mode | CLI | Description |
|------|-----|-------------|
| Smoke (default) | `--smoke` | Single focused dataset, small row cap, context size 75, `max_workers=1` |
| Focused | `--focused` | LIQ_5K_HIGH_ACTIVITY + NO_WHALE_FILTER, horizons 1h/4h/8h/24h |
| Full | `--full` | All 40 filter × horizon × exit-policy combinations |

Run from project root. TabICL inference requires `.venv-tabicl`; smoke/tests may use `--skip-tab-inference`.

```bash
python scripts/evaluate_direct_target_tabicl.py --smoke --skip-tab-inference
python scripts/evaluate_direct_target_tabicl.py --focused --context-sizes 512 --max-context-size 512 --device cuda
```

## Pre-run dependency gate

Before TAB evaluation, E5 writes `audit/direct_target_tabicl_dependency_audit.json` verifying:

1. E3 dataset root readable
2. E4A comparison root readable
3. Required E4A prediction parquets, metrics, policy outputs, manifest on disk
4. E4A root discoverable in artifact registry (unless `--allow-registry-warnings`)

Fail-fast if the E4A comparison chain is incomplete.

## Outputs

Under `data/training/manual_verified_results/phase_e5_direct_target_tabicl_<timestamp>/`:

- `reports/direct_target_tabicl_manifest.json`
- `audit/direct_target_tabicl_dependency_audit.json`
- `audit/direct_target_tabicl_run_audit.jsonl`
- `predictions/direct_target_tabicl_predictions_{validation|test}_*.parquet`
- `metrics/direct_target_tabicl_metrics_*.json`
- `consensus/direct_target_tab_xgb_rf_join_diagnostic.csv`
- `consensus/direct_target_consensus_tier_summary.csv`
- `reports/direct_target_tabicl_summary_for_upload.txt`

## Consensus tiers (Anchor Plan)

| Tier label | Meaning |
|------------|---------|
| `TAB_XGB_RF_ALL3` | Tier 1 candidate |
| `TAB_RF_ONLY` | Tier 2 candidate |
| `TAB_XGB_ONLY` | Rejected / research-only |
| `XGB_RF_ONLY` | Rejected / research-only |

Rank-based inclusion at top 0.5%, 1%, 2%, 5% with pair-cap policy evaluation (10, 25, 50).

## Identity hierarchy

Join priority:

1. `target_row_id`
2. `candidate_policy_id` + split + filter + horizon + exit_policy_id
3. `candidate_id` (fallback only with uniqueness proof)

## Module map

| File | Role |
|------|------|
| `app/training/direct_target_tabicl.py` | Core E5 orchestration |
| `scripts/evaluate_direct_target_tabicl.py` | CLI entry point |
| `tests/test_direct_target_tabicl.py` | Unit and smoke integration tests |

Reuses E4A dataset discovery, feature construction, and policy helpers from `app/training/direct_target_xgb_rf.py` and TabICL inference from `app/training/tabicl_v2_eval.py`.

## E6 gate

E5 provides evidence only. Proceed to E6 only after:

- Dependency audit pass with E4A prediction parquets on disk
- Focused/full TAB runs with TabICL venv
- Validation-selected consensus tier performance and concentration controls reviewed
