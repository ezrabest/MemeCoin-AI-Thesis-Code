# 04 — Data Lineage and Storage

## Purpose

Define storage roles for SQLite, CSV, Parquet, JSON, JSONL, and model artifacts; canonical builder policy; and future artifact registry requirements.

## Diagram

See [diagrams/data_lineage_storage.mmd](diagrams/data_lineage_storage.mmd).

```text
raw data
→ canonical builder
→ canonical dataframe/table
→ parquet/csv export
→ optional SQLite mirror
→ artifact registry
→ consumers
```

## Storage Policy (Required)

```text
Never generate SQLite and CSV/Parquet independently from different logic paths.
Generate one canonical dataframe/table.
Write all artifacts from that canonical object.
Attach manifest/hash.
Register the artifact.
```

## Runtime Source of Truth — SQLite

SQLite (`data/trader.db`, `app/database.py`) is the **runtime/system source of truth** for:

| Domain | Table / artifact |
|--------|------------------|
| Live/demo candidates | `coins`, `market_snapshots` |
| Paper/demo decisions | `paper_trades`, `gemini_decisions` |
| Model decision logs | `pipeline_audit` |
| Enrichment logs | `raw_provider_payloads` |
| LLM reviews | `gemini_decisions` |
| Settings | `data/settings.json` (file-backed) |
| Audit trail | `pipeline_audit`, `data/audits/*.jsonl` |
| Runtime status | watcher state, sentiment_records |

## Research / Export Artifacts — CSV / Parquet

CSV and Parquet are **research/export artifacts** for:

- Branch uploads and offline audits
- Model input datasets (`model_ready_dataset.parquet`)
- Comparison reports and sweeps
- Documentation and phase reports

They must **not** diverge from the canonical builder output.

## Future Artifact Registry

Every artifact must include at least one immutable identity field:

```text
git_commit_hash
content_hash
schema_hash
source_hash
```

Prefer both:

```text
git_commit_hash + content_hash
```

### Required future registry fields

```text
artifact_id
artifact_type
phase
path
sqlite_table
row_count
schema_hash
content_hash
source_hash
git_commit_hash
created_at_utc
created_by_script
source_artifacts
model_version
target_definition
notes
```

### Artifact validity rule

```text
No model prediction, selected-trade table, direct-target dataset, or policy-summary artifact should be considered valid unless it has a manifest with content hash, schema hash, and code/version provenance.
```

## Why Artifact Signing Matters

- Prevents confusion between old and new RF/TAB/XGB artifacts
- Prevents mixing predictions from different code versions
- Removes uncertainty about which artifact produced which result
- Enables deterministic lineage for model comparisons
- Allows UI/debug screens to show active dataset/model/report version

## Current State

- SQLite WAL mode via `app/sqlite_util.py`
- Multiple independent export paths in scripts (risk of divergence)
- Some manifests exist (e.g. `xgb_clean_full_cuda_manifest.json`, `consensus_manifest.json`) but no unified registry
- JSONL audit files in `data/audits/`
- Model artifacts: `.joblib`, feature JSONs, prediction parquets

## Target State

- Canonical builder module used by all export paths
- `artifact_registry` table or equivalent (Phase E1)
- UI Data Lineage Panel showing `artifact_id`, hashes, model versions

## Key Inputs

- Raw DexScreener/RSS/RPC data, training labels, model outputs

## Key Outputs

- Canonical dataframe → Parquet/CSV/SQLite mirror + manifest
- Registry entries linking consumers to provenance

## Consumers

- Training scripts, runtime inference loaders, UI lineage panel, QA validation

## Open Questions

- Registry in SQLite vs separate manifest index file
- Schema hash algorithm (column order, dtypes, nullable policy)
- Migration discipline when `unified candidate schema` (E2) lands

## Non-Goals

- Implementing registry table or canonical builder in Phase E0
- Modifying existing data files or SQLite schema
- Running DB migrations
