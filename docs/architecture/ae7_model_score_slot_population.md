# AE7 — Model Score Slot Population / Meta-Model Readiness

## What AE7 implements

AE7 extends the AE6 consensus decision layer with a **safe model-score slot population** readiness/audit pipeline:

```text
AE7-0 artifact inventory
→ artifact classification + reproducibility matrix
→ lightweight exact-ID prediction index (RF / XGB / TAB)
→ AE6 DecisionRecord enrichment (append-only AE7 JSONL)
→ consensus recomputation
→ bridge-readiness audit
```

Core modules:

| Module | Role |
|--------|------|
| `app/decision/model_scores.py` | AE7 constants, populated slot types, missing-reason taxonomy |
| `app/decision/score_artifacts.py` | Inventory loader, artifact classifier, reproducibility, prediction index |
| `app/decision/score_population.py` | Record enrichment, audit summary, JSONL persistence |
| `scripts/run_ae7_model_score_slot_population_smoke.py` | Local smoke / audit runner |

## What AE7 does not implement

| Scope | Status |
|-------|--------|
| Meta-model training | **Not in AE7** |
| RF / XGB / TAB retraining | **Forbidden** |
| Runtime model inference | **Not in AE7** |
| Qwen / Gemini / Ollama calls | **Forbidden** |
| Helius / Solana / external APIs | **Forbidden** |
| Live or paper trade execution | **Forbidden** |
| UI changes | **Not in AE7** |
| Risk setting changes | **Not in AE7** |
| Fuzzy pair/time score joins | **Forbidden** |

AE7 is **readiness and safe population auditing**, not model training or live inference.

## Two distinct score-population modes

AE7 documents and implements only the **historical/offline** path. The **runtime/live** path is explicitly deferred.

### Historical / offline mode (implemented in AE7)

```text
DecisionRecord → exact ID lookup in existing offline prediction artifacts → model score slot
```

Allowed exact-ID keys in **offline prediction artifacts**:

```text
target_row_id exact match
candidate_policy_id exact match (with policy context when present)
candidate_id exact match (with event context when present)
```

`target_row_id` is valid in **labeled historical prediction tables**. It is used for offline audit/backfill alignment — not as a field live runtime collection is expected to preserve.

### Runtime / live mode (next phase — not in AE7)

```text
DecisionRecord → as-of feature row → model artifact inference → model score slot
```

Live runtime score population must **not** depend on finding existing offline prediction rows. The next phase builds a feature-matrix bridge and runs inference against registered model artifacts.

## Why this is not model training

AE7 only **reads existing offline prediction artifacts** and attempts **exact identity alignment**. It does not fit models, retrain checkpoints, run inference, or write new prediction tables. Missing scores remain explicitly unavailable.

## Why exact ID alignment is mandatory (offline mode)

Model scores affect consensus and downstream meta-model readiness. Joining by `pair_address + timestamp`, symbol, liquidity buckets, or nearest-neighbor matching can attach the wrong offline row to a candidate. AE7 offline lookup allows only the exact-ID keys listed above.

## Why pair/time matching is forbidden for model score population

AE6 may use best-effort pair/time linkage for **lineage caveats**. That is acceptable for traceability but **not** for attaching quantitative model scores. AE7 keeps those concerns separate.

## Why missing runtime bridge fields are a gap, not an AE7 bug

Current AE6 runtime records often have:

- `candidate_id` — runtime SHA256 hash (not training-compatible)
- `candidate_policy_id` / `scoring_policy_id` — absent
- `as_of_feature_row_id` / `feature_schema_id` — absent
- `target_row_id` — absent (expected — not a normal live field)

When this is true, AE7 correctly returns:

```text
score_population_decision = RUNTIME_IDENTITY_BRIDGE_REQUIRED
missing_reason = RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID
```

AE7 must not invent scores, run inference, or backfill via pair/time joins.

## How RF / XGB / TAB slots are populated (AE7 scope)

1. Load AE7-0 inventory CSV.
2. Classify prediction artifacts (model family, exact ID columns, safe score columns).
3. Assess reproducibility via artifact registry + manifest proximity.
4. Build a per-family exact-ID index from **CURRENT, reproducible** artifacts only.
5. For each AE6 record, attempt offline lookup by `target_row_id` → `candidate_policy_id` → `candidate_id`.
6. On hit: populate slot with `population_method = EXACT_ID_MATCH`.
7. Recompute consensus via `app/decision/consensus.py`.

Records with only runtime bridge fields (no offline lookup keys) remain unavailable until the runtime inference path exists.

## How missing slots are recorded

| `missing_reason` | Meaning |
|------------------|---------|
| `RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID` | No offline lookup keys and no runtime bridge fields |
| `NO_SAFE_EXACT_ID_ALIGNMENT` | Lookup keys present but no index hit |
| `NO_SAFE_SCORE_COLUMN` | Artifact family lacks safe score column |
| `NO_SAFE_MODEL_ARTIFACT` | No safe artifact for model family |
| `ARTIFACT_NOT_REPRODUCIBLE_OR_STALE` | Artifact failed reproducibility gate |
| `NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT` | Runtime bridge present but live inference not implemented |

## How artifact reproducibility is checked

For each inspected artifact AE7 writes `ae7_artifact_reproducibility_matrix.csv` with registry entry, content hash, schema hash, manifest proximity, `artifact_status`, and `is_reproducible`.

## Why stale / deprecated / unregistered artifacts are not trusted

Offline predictions without registry/manifest linkage cannot be tied to a known training run. Default is reject.

## How consensus changes when slots become available

`compute_consensus()` reads available RF/XGB/TAB slots. When multiple slots populate, consensus families advance from `SINGLE_MODEL_ONLY` through partial combinations to `TAB_XGB_RF_ALL3`.

## Why runtime may still show NO_MODEL_CONSENSUS_AVAILABLE

Even with safe offline artifacts, current AE6 runtime records lack the **runtime feature-matrix bridge** needed for live inference, and their runtime `candidate_id` values do not align with offline prediction tables. Consensus remains `NO_MODEL_CONSENSUS_AVAILABLE` until the next bridge phase.

## Next required phase: Runtime Candidate Identity + Feature Matrix Bridge

**Goal:** Enable runtime model inference — not historical prediction lookup.

Required runtime bridge fields:

```text
candidate_id
candidate_policy_id or scoring_policy_id (when a scoring policy is selected)
as_of_feature_row_id
feature_schema_id
model_artifact_id
runtime_inference_id
```

**Not required at live collection time:**

```text
target_row_id   # historical labeled-row key; depends on target/label-source context
```

## How AE7 prepares meta-model readiness (AE7B+)

AE7 delivers:

- Classified, reproducibility-scored artifact inventory
- Exact-ID offline prediction index abstraction
- Enriched decision records with explicit score missingness
- Bridge-readiness audit distinguishing offline vs runtime score paths

Downstream meta-model work can trust only `EXACT_ID_MATCH` populated slots from the offline path, and later `RUNTIME_INFERENCE` slots from the bridge phase.

## Outputs

```text
data/decision_records/ae7_model_score_enriched_YYYYMMDD.jsonl
data/audits/ae7_model_score_slot_population_<timestamp>/
  ae7_artifact_reproducibility_matrix.csv
  ae7_offline_alignment_audit.csv
  ae7_model_score_slot_population_summary.json
```

## Smoke command

```powershell
python scripts/run_ae7_model_score_slot_population_smoke.py --max-records 10 --max-artifacts 50 --audit-only
```

## Safety guarantees

- No model training or inference
- No LLM or external API calls
- No trading execution
- No AE6 JSONL overwrite
- No fuzzy pair/time score joins
- Parquet schema inspection without full-file load
- Column-limited reads for index building
