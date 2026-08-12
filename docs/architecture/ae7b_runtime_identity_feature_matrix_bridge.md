# AE7B — Runtime Candidate Identity + Feature Matrix Bridge

## Purpose

AE7B implements the **runtime/live bridge** between local SQLite runtime tables and the decision layer fields required before safe RF/XGB/TAB model-score population at inference time (AE7C).

AE6 produces `DecisionRecord` objects with mandatory lineage metadata. AE7 proved that model-score slots cannot be populated safely without runtime-compatible identity and feature-row fields. AE7B fills that gap.

## What AE7B implements

1. **Runtime candidate identity** (`candidate_id`) — deterministic SHA-256 over namespace `RUNTIME_CANDIDATE_ID_V1` and a sorted JSON identity payload.
2. **Scoring policy identity** (`scoring_policy_id`) — non-trading placeholder policy for AE7B; no model inference.
3. **As-of feature rows** (`as_of_feature_row_id`, `feature_values`, missingness, source maps).
4. **Feature schema** (`feature_schema_id`, enforcement, forbidden-pattern rejection).
5. **Feature parity audit** — exact-ID comparison only; `BLOCKED_NO_OVERLAP` when no safe overlap exists.
6. **Weak implicit lineage** — truthful `lineage_confidence_score` and resolution methods.
7. **Model-schema compatibility matrix** — inspects small JSON schema files without full-loading large artifacts.
8. **JSONL persistence** — append-only bridge records with flush/fsync per line.

## What AE7B does not implement

- Model training or retraining (RF, XGB, TAB, TabICL, meta-models).
- Runtime model inference or RF/XGB/TAB score population.
- LLM calls (Qwen, Gemini, Ollama).
- External APIs (Helius, Solana RPC, provider fetches).
- Live or paper trading execution.
- SQLite writes for `runtime_feature_rows`.
- Mutation of AE6/AE7 JSONL outputs.

## Why `target_row_id` is not required at runtime

`target_row_id` belongs to **labeled historical target rows** and depends on target/label-source context. Live runtime events do not have realized outcomes or training labels. AE7B bridge records set `target_row_id_not_required: true` and `target_row_id: null`.

Historical/offline score population (AE7 mode 1) may still use `target_row_id` for exact-ID lookup in offline prediction artifacts. Runtime/live score population (AE7 mode 2) uses `candidate_id` + `as_of_feature_row_id` instead.

## Identity generation

### `candidate_id`

Input fields (priority when available):

- `chain`, `pair_address` (normalized lowercase)
- `base_token_address`, `quote_token_address`
- `symbol`, `event_timestamp`
- `source_table`, `source_row_id`, `provider`

Serialized as sorted JSON, hashed: `sha256("RUNTIME_CANDIDATE_ID_V1|" + payload)`.

If no stable identity exists: `candidate_identity_status = BLOCKED_MISSING_STABLE_IDENTITY`.

`pair_address` is used for identity/lineage only — **not** as a model feature.

### `scoring_policy_id`

Placeholder non-trading policy:

- Namespace: `AE7B_DEFAULT_NON_TRADING_SCORING_POLICY_V1`
- `scoring_policy_status = PLACEHOLDER_NO_MODEL_INFERENCE`
- `model_family_targets = ["RF", "XGB", "TAB"]`

### `as_of_feature_row_id`

Deterministic hash of:

`candidate_id | scoring_policy_id | feature_schema_id | as_of_timestamp | source_snapshot_id | source_signal_id`

### `feature_schema_id`

Derived from schema **content** (feature names, dtypes, required/optional sets), not from clock time. Hash of `AE7B_RUNTIME_V1|schema_hash`.

## Schema enforcement

- **Required features** (initial set: `price_usd`, `liquidity_usd`): missing values → `feature_status = MISSING_REQUIRED_FEATURE`; bridge may be `RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP`.
- **Optional features**: `null` with entry in `feature_missingness`.
- **Forbidden patterns** (`target`, `label`, `future`, `return`, etc.): excluded into `rejected_features`.
- **No silent zero imputation** for missing required features.

## Feature parity

Compares runtime bridge records to offline rows **only via exact `candidate_id` alignment**. No fuzzy pair/time matching.

| Status | Meaning |
|--------|---------|
| `PASS` | Exact-aligned features match within tolerance |
| `FAIL_MISMATCH` | Overlap exists but values differ |
| `BLOCKED_NO_OVERLAP` | No safe exact-ID overlap (expected pre-AE7B) |
| `BLOCKED_MISSING_SCHEMA` | No runtime records to compare |
| `BLOCKED_UNSAFE_ID_ALIGNMENT` | Unsafe alignment attempted |

When `BLOCKED_NO_OVERLAP`: `future_inference_readiness = BLOCKED_PENDING_PARITY_SET`. This does **not** fail AE7B implementation.

## As-of safe fields

From local tables only (`market_snapshots`, `signals`, `sentiment_records`, `coins`):

- Price, liquidity, volume, txn counts, price changes, fdv
- `signal_score`, sentiment aggregates
- `whale_score_asof` (research-only metadata)

Excluded: labels, targets, future returns, exit simulation, train/test columns.

## `whale_score_asof` — research only

Included as an optional feature value when present, with metadata:

```json
{
  "whale_score_status": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
  "not_rule": true,
  "not_runtime_approved_as_standalone_signal": true
}
```

Not used as a hard buy gate or runtime approval signal.

## Lineage

Preflight confirmed `has_explicit_raw_snapshot_lineage: false`. AE7B records truthful weak lineage:

- `lineage_mode = BEST_EFFORT_IMPLICIT_LINKAGE`
- `lineage_strength = WEAK_IMPLICIT_TIME_PAIR_LINKS`
- `exact_id_match = false` when resolved by pair/time/provider matching
- Warning: *Lineage fallback: best-effort match using provider, pair_address, timestamps, and scan/run context.*

### `lineage_confidence_score`

| Score | Meaning |
|-------|---------|
| 1.0 | Explicit structural lineage |
| 0.7 | Deterministic stored source row id, no FK |
| 0.35 | Best-effort provider/pair/time match |
| 0.0 | Missing/unresolved |

When `exact_id_match = false`, score is **below 0.5**.

This is **lineage confidence only** — not trading confidence, not model confidence.

## JSONL vs SQLite

AE7B writes bridge records to:

`data/runtime_bridge/ae7b_runtime_feature_rows_YYYYMMDD.jsonl`

Audits go to:

`data/audits/ae7b_runtime_identity_feature_bridge_<timestamp>/`

SQLite is opened **read-only**. No `INSERT` into `runtime_feature_rows`.

## Model-schema compatibility

Inspects candidate schema JSON files from AE7B-0 preflight inventory. Uses schema-only / small-file reads — no full Parquet/pandas loads on large artifacts.

Statuses: `COMPATIBLE`, `PARTIAL_MISSING_FEATURES`, `BLOCKED_MISSING_SCHEMA`, `BLOCKED_LEAKAGE_RISK`, `BLOCKED_UNSUPPORTED_ARTIFACT`, `UNKNOWN`.

## Bridge readiness decisions

- `RUNTIME_FEATURE_BRIDGE_CREATED`
- `RUNTIME_FEATURE_BRIDGE_PARTIAL`
- `RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP`
- `RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP`
- `RUNTIME_FEATURE_BRIDGE_BLOCKED_PARITY_GAP`

## AE7C — future runtime inference

AE7C will consume AE7B outputs:

1. Read `candidate_id`, `as_of_feature_row_id`, `feature_schema_id`.
2. Verify model-schema compatibility and feature parity when overlap exists.
3. Run RF/XGB/TAB inference against compatible artifacts.
4. Populate AE7 model-score slots on enriched decision records.

AE7B does **not** run inference — it only prepares identity, features, lineage, and compatibility audits.

## Smoke script

```bash
python scripts/run_ae7b_runtime_identity_feature_bridge_smoke.py \
  --max-records 50 \
  --lookback-hours 3 \
  --audit-only \
  --no-db-write \
  --parity-check
```

## Safety invariants

All bridge records retain:

- `no_trade_authority = true`
- `llm_decision_authority = false`
