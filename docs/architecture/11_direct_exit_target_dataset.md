# Phase E3 — Direct Net-Profitable Exit-Policy Target Dataset

## Purpose

Phase E3 builds deterministic target labels for downstream XGB / TAB / RF training based on **realized simulated net return** after take-profit, stop-loss, time-stop, and round-trip fees — not the legacy x2 ranking proxy.

This phase is **dataset/target infrastructure only**. It does not train models, run inference, or change runtime trading behavior.

## Anchor Plan alignment

| Layer | Component | E3 role |
|-------|-----------|---------|
| 1 | XGB / TAB / RF | Consumes direct-target datasets later; E3 does not train or score |
| 2 | Consensus economics | Unchanged |
| 3 | Context intelligence | Unchanged |
| 4 | Reasoning & audit | Unchanged |

Terminology **XGB**, **RF**, and **TAB** retain their existing project meanings (XGBoost, Random Forest, TabICL/TabICLv2).

## Target definition

- **Target name:** `net_profitable_after_exit_policy`
- **Target version:** `v1`
- **Label:** `target_net_profitable_after_exit = sim_net_return > 0`
- **Net return:** `sim_net_return = (exit_ratio - 1.0) - round_trip_fee_pct`

### Default exit policies

| exit_policy_id | tp_ratio | sl_ratio | round_trip_fee_pct | time_stop |
|----------------|----------|----------|--------------------|-----------|
| `TP20308_SL080_FEE0308_TIME_BY_HORIZON` | 2.0308 | 0.80 | 0.0308 | horizon minutes |
| `TP20308_SL075_FEE0308_TIME_BY_HORIZON` | 2.0308 | 0.75 | 0.0308 | horizon minutes |

## Identity rules

### candidate_id (event-level, E2)

SHA-256 over `chain | normalized_pair_address | event_timestamp_normalized | source [| source_row_id]`.

Must **not** include filter, horizon, exit policy, or target metadata.

### candidate_policy_id (E3)

SHA-256 over:

```
candidate_id | filter | horizon | top_pct | pair_cap | exit_policy_id |
tp_ratio | sl_ratio | time_stop_minutes | round_trip_fee_pct
```

For full direct-target datasets: `top_pct = not_applicable`, `pair_cap = not_applicable`.

### target_row_id (E3)

SHA-256 over:

```
candidate_policy_id | target_name | target_version | label_source_artifact_id
```

**Hard rule:** `candidate_id` is not the unique key for direct-target rows. Use `target_row_id`.

## Precision strategy

Floating-point TP/SL decisions use a single tolerance constant:

```python
EXIT_COMPARE_EPSILON = 1e-9
```

- **TP hit:** `ratio >= tp_ratio - EXIT_COMPARE_EPSILON`
- **SL hit:** `ratio <= sl_ratio + EXIT_COMPARE_EPSILON`

Critical price ratios are computed via `Decimal(str(value))` where practical. All threshold comparisons use `EXIT_COMPARE_EPSILON` consistently so IEEE-754 rounding cannot flip TP/SL outcomes.

## Simulation semantics

1. Resolve `pair_address` and `event_timestamp`.
2. Entry snapshot = latest `market_snapshots` row with same pair and `timestamp <= event_timestamp`.
3. Future window: `(entry_timestamp, entry_timestamp + time_stop_minutes]` — strict upper bound.
4. Walk future snapshots chronologically; detect TP, SL, or TIME exit.
5. Gap handling via `--max-future-gap-minutes` (default 20). Horizons shorter than 20 minutes use `min(time_stop, max_gap)`.
6. Invalid labels are retained with explicit `label_error_code` — never dropped silently.

## Outcome / audit columns (exclude from training features)

Store these columns in the dataset for audit and label provenance. **Downstream training must exclude them as features:**

- `target_net_profitable_after_exit`
- `sim_net_return`, `sim_exit_status`, `exit_ratio`
- `max_future_ratio`, `min_future_ratio`
- `label_valid`, `label_error_code`, `label_error_detail`
- `entry_snapshot_timestamp`, `entry_price_raw`, `entry_price`, `entry_snapshot_id`
- `future_window_*`, `first_future_snapshot_timestamp`, `last_future_snapshot_timestamp`
- `future_snapshot_count`
- `gap_*`, `exit_timestamp`
- `candidate_policy_id`, `target_row_id`, `label_source_artifact_id`
- Any other path-derived or outcome fields

Feature columns from `*_CLEAN_MODEL_INPUT.parquet` remain usable when the above are excluded.

## Module layout

| Module | Role |
|--------|------|
| `app/training/direct_target_ids.py` | ID formulas, defaults, naming |
| `app/training/exit_path_simulation.py` | TP/SL/TIME simulation with gap detection |
| `app/training/direct_target_builder.py` | Chunked builder, SQLite prefetch, dual-format writes |
| `scripts/build_direct_exit_targets.py` | CLI entry point |

## CLI

```bash
python scripts/build_direct_exit_targets.py --dry-run
python scripts/build_direct_exit_targets.py --overwrite --all-default-exit-policies
```

Defaults:

- `--input-dir data/training/manual_verified_datasets_clean_for_model`
- `--sqlite-db data/trader.db` (read-only URI mode)
- `--output-dataset-dir data/training/manual_verified_datasets_direct_target_v1`
- `--output-report-dir data/training/manual_verified_results/phase_e3_direct_targets_v1`
- `--chunk-size 5000`
- `--snapshot-prefetch-mode chunked_pair_cache`
- `--max-future-gap-minutes 20`

## Outputs

Per filter / horizon / exit policy:

- `data/training/manual_verified_datasets_direct_target_v1/{FILTER}_{HORIZON}_{EXIT_POLICY_ID}_DIRECT_TARGET_v1.parquet`
- Matching `.csv` from the **same canonical row stream**

Reports under `data/training/manual_verified_results/phase_e3_direct_targets_v1/`:

- Audit rows, summary, invalid-label diagnostic, gap diagnostic, manifest, upload summary

## SQLite access

- Open with `file:...?mode=ro`
- Read `market_snapshots` only
- No writes, indexes, or schema changes in `trader.db`

## Memory / performance

The builder processes input parquet in chunks (`--chunk-size`, default 5000) and prefetches snapshots only for pairs in the current chunk (`chunked_pair_cache`). It does not load all filters, horizons, or policies simultaneously.

## What E3 does not change

- Runtime / demo / paper trading
- RF / TAB / XGB artifacts or training scripts
- Production SQLite schema
- Live collection, UI, risk settings, Qwen, Gemini, Helius, Solana RPC
