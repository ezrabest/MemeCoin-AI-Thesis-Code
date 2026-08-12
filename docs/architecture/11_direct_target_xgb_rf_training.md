# Phase E4A — Direct Target XGB/RF Training

## Purpose

Phase E4A implements controlled **offline** training and evaluation infrastructure for **XGB** and **RF** on Phase E3 direct net-profitable exit-policy target datasets. This phase produces evidence artifacts only; it does not change runtime trading, UI, or the Anchor Plan.

## Direct target source

- **Input root:** `data/training/manual_verified_datasets_direct_target_v1`
- **E3 manifest:** `data/training/manual_verified_results/phase_e3_direct_targets_v1/phase_e3_direct_target_manifest.json`
- **Identity hierarchy:** `candidate_id` → `candidate_policy_id` → `target_row_id` (preserved in predictions and audits)

## Canonical target normalization

Acceptable source columns:

- `target_net_profitable_after_exit_policy`
- `target_net_profitable_after_exit`

Both are renamed to the single canonical training label:

```text
target_net_profitable
```

If both aliases exist in one dataset → `AMBIGUOUS_TARGET_ALIAS` (dataset skipped).

## Target-normalization audit log

Mandatory separate CSV:

```text
data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_v1/audit/phase_e4_target_normalization_audit.csv
```

Records original alias, dtypes, null counts, positive/negative counts, and normalization status per dataset.

## Valid label handling

Train/evaluate only rows with explicit validity (`label_valid`, etc.) or, if absent, non-null binary 0/1 targets. Invalid labels are excluded with full row-count accounting.

## Row-count invariants (fail-fast)

Per dataset, invariants are asserted before model training:

```text
post_valid_filter == train + validation + test
feature_matrix_* == split row counts
prediction_* == split row counts
```

Failure → `ROW_COUNT_INVARIANT_FAILED` (`RuntimeError`), dataset marked failed, audit preserved, other datasets may continue.

## Leakage exclusions

Outcome, target, future, return, identity, and split metadata columns are excluded from the feature matrix but retained in prediction outputs. Features are numeric/boolean only.

## Model roles

| Model | E4A role |
|-------|----------|
| **XGB** | Primary broad ranker on direct target; CUDA by default (`tree_method=hist`, `device=cuda`) |
| **RF** | Confirmation ranker on same direct target; CPU |
| **TAB** | **Deferred to E5** — not trained in E4A |

## XGB CUDA policy

- Default: CUDA required; fail closed on CUDA errors unless `--allow-cpu-fallback`
- Metrics record `xgb_device_requested`, `xgb_device_used`, `cpu_fallback_used`, `xgboost_version`

## RF baseline

`RandomForestClassifier` with `class_weight=balanced_subsample`, `min_samples_leaf=5`, `n_estimators=500`, `random_state` from CLI.

## random_state determinism

CLI `--random-state` is passed to XGB, RF, and any smoke-only limiting logic. No `train_test_split`; E3 split column is used as-is without shuffle.

## Full Pipeline artifacts

Every model is a fitted sklearn `Pipeline`:

```text
SimpleImputer(strategy=median) → XGBClassifier / RandomForestClassifier
```

Saved as `.joblib` with a matching `_preprocessing.json` sidecar containing:

- `feature_columns_in_order`
- `imputer_statistics_by_feature` (exact learned medians)
- `sklearn_version` (exact version string)
- `random_state`

## Incremental audit logging

Append-only JSONL at `audit/phase_e4_run_audit.jsonl`, flushed after each event (`dataset_started`, `model_trained`, `row_count_invariant_failed`, etc.).

## Outputs

```text
data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_v1/
  models/
  predictions/
  metrics/
  policy_evaluation/
  audit/
  reports/
```

## Registry handling

After outputs are written, E4 artifacts are merged into the E1 file registry via explicit include root. Registration failures emit a repair command:

```bash
python scripts/register_existing_artifacts.py --include-root data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_v1
```

## XGB/RF agreement (diagnostic only)

`direct_target_xgb_rf_agreement_diagnostic.csv` reports overlap slices (`XGB_AND_RF`, `XGB_ONLY`, `RF_ONLY`) at top-k with pair cap 50. This does **not** change Anchor Plan consensus tiers (`TAB_XGB_RF_ALL3` requires TAB in E5+).

## Why TAB is deferred to E5

E4A establishes XGB/RF infrastructure, leakage guards, Pipeline artifacts, and policy evaluation on the direct target. TAB training/evaluation on the same E3 datasets is a separate phase to keep scope auditable and avoid mixing TabICL dependencies with XGB CUDA setup.

## CLI

```bash
# Smoke (default after implementation)
python scripts/train_direct_target_xgb_rf.py \
  --filter LIQ_5K_HIGH_ACTIVITY --horizon 1h \
  --exit-policy TP20308_SL080_FEE0308_TIME_BY_HORIZON \
  --model both --smoke --overwrite --register-artifacts true

# Full run (user-initiated only)
python scripts/train_direct_target_xgb_rf.py \
  --all --model both --overwrite --register-artifacts true --xgb-device cuda
```

## Module layout

| Path | Role |
|------|------|
| `app/training/direct_target_xgb_rf.py` | Core training, evaluation, audit |
| `scripts/train_direct_target_xgb_rf.py` | CLI entry point |
| `tests/test_direct_target_xgb_rf.py` | Unit/integration tests |
