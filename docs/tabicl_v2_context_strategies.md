# TabICLv2 context strategies (offline evaluation)

Offline-only TabICLv2 evaluation supports multiple **train-only** context strategies.
Validation and test rows are never used as context.

## Strategies

| Strategy | Description |
|----------|-------------|
| `stratified_recent` | Recent train rows with deterministic positive/negative mix (default ratio 0.25) |
| `positive_enriched` | Train rows enriched with positives (default ratio 0.50, `random_state=42`) |
| `nearest_neighbors_context` | Per-batch context from nearest train rows in preprocessed feature space |
| `whale_wave_context` | Per-batch context from whale/market microstructure feature similarity |
| `ensemble_small_contexts` | Average probabilities from multiple smaller member contexts |

If `--context-strategy` is omitted, legacy deterministic sampling is preserved.

## Memory safety

- Context is capped by `--context-size` and `--max-train-context-rows`
- Predictions run in `--batch-size` chunks
- CUDA OOM triggers batch/context reduction and cache clearing
- Nearest-neighbor index is fit **once** on the train split

## Nearest-neighbor modes

### Fixed full-train KNN (baseline)

Omit `--knn-rolling-days` to fit one NearestNeighbors index on the full train split (legacy behavior).

### Rolling time-aware KNN

Pass `--knn-rolling-days` (e.g. 14 or 30) to enable train-only rolling context:

- For each prediction batch, use train rows with `event_timestamp < batch_min_time`
- Start with a window of `batch_min_time - rolling_days` .. `batch_min_time`
- Expand up to `--knn-max-rolling-days` when `--knn-expand-window true` and slice is smaller than `--knn-min-context-rows`
- Fit NearestNeighbors on the temporal slice once per batch (cached by day bucket + window)
- Optional reranking via `--knn-time-decay-alpha`

`event_timestamp` is used for temporal slicing only and remains excluded from model features.

## Commands

Single strategy:

```powershell
.venv-tabicl\Scripts\python.exe scripts\evaluate_tabicl_v2.py --context-size 4096 --max-train-context-rows 4096 --batch-size 512 --context-strategy stratified_recent --output-suffix stratified_recent
```

Small sweep:

```powershell
.venv-tabicl\Scripts\python.exe scripts\sweep_tabicl_v2_context_strategies.py --max-rows 30000 --max-features 50 --context-size 2048 --max-train-context-rows 2048 --batch-size 256
```

Ensemble:

```powershell
.venv-tabicl\Scripts\python.exe scripts\evaluate_tabicl_v2.py --context-strategy ensemble_small_contexts --ensemble-count 4 --ensemble-context-size 2048 --batch-size 512 --output-suffix ensemble_small_contexts
```

Fixed vs rolling KNN comparison:

```powershell
.venv-tabicl\Scripts\python.exe scripts\evaluate_tabicl_v2.py --context-strategy nearest_neighbors_context --context-size 4096 --max-train-context-rows 4096 --batch-size 512 --output-suffix nearest_neighbors_fixed_4096

.venv-tabicl\Scripts\python.exe scripts\evaluate_tabicl_v2.py --context-strategy nearest_neighbors_context --context-size 4096 --max-train-context-rows 4096 --batch-size 512 --knn-rolling-days 14 --knn-min-context-rows 512 --knn-expand-window true --knn-max-rolling-days 90 --output-suffix nearest_neighbors_rolling_14d_4096
```

## Outputs

Per strategy (suffix = `--output-suffix` or strategy name):

- `data/training/models/tabicl_v2_predictions_validation_<suffix>.parquet`
- `data/training/models/tabicl_v2_predictions_test_<suffix>.parquet`
- `data/training/policy_backtests/tabicl_v2_report_<suffix>.json`

Sweep:

- `data/training/policy_backtests/tabicl_v2_context_strategy_sweep.json`
- `data/training/policy_backtests/tabicl_v2_context_strategy_sweep.csv`

Strategies are ranked by **validation** `precision_at_top_1_percent`, then validation return, then validation PR-AUC.

Existing outputs are not overwritten unless `--overwrite-outputs` is passed.
