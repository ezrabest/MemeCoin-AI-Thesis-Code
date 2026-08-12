# 09 — Artifact Registry (Phase E1)

## Purpose

Phase E1 introduces a **file-based artifact registry** and data-lineage foundation for research and model artifacts. The registry records immutable identities (content hash, schema hash, path id, artifact id), human-readable paths, inferred metadata, and validation status without changing runtime behavior, model outputs, or existing research files.

## What Problem the Registry Solves

Research artifacts today live as CSV, Parquet, JSON, and model files under `data/training/`. Without a registry:

- The same logical artifact can be referenced by unstable absolute paths across machines.
- Schema drift (e.g. `int32` vs `int64`) is hard to detect consistently.
- Lineage from datasets → predictions → consensus → audits is implicit in folder names only.
- Future SQLite mirroring risks diverging from file exports if built from separate logic paths.

The registry makes artifacts **auditable, hash-addressable, and validation-friendly** while preserving the Phase E0 canonical-builder policy: one canonical object → write artifacts → register metadata.

## Current Phase E1 Scope

Implemented in `app/artifacts/`:

| Module | Role |
|--------|------|
| `hash_utils.py` | Content hash (streaming SHA-256), path id, artifact id, schema hash, dtype normalization |
| `manifest_schema.py` | `ArtifactRecord` dataclass schema |
| `registry.py` | Scanning, hash cache reuse, atomic registry writes, validation |

Scripts:

- `scripts/register_existing_artifacts.py` — scan known research roots and write registry
- `scripts/validate_artifact_registry.py` — validate registry and emit JSON audit report

Tests: `tests/test_artifact_registry.py`

## Explicitly Out of Scope (E1)

- SQLite registry mirroring (`trader.db` unchanged)
- Runtime integration (scan loop, API, UI)
- Model retraining, exit simulation reruns, prediction regeneration
- Modifying, moving, or deleting existing research artifacts
- Perfect automatic lineage inference for all artifact graphs
- Decision Gate / Anchor Plan changes

SQLite mirroring may be proposed in a later **E1B** phase behind an explicit Decision Gate.

## ArtifactRecord Schema

Each registry row is an `ArtifactRecord` (`app/artifacts/manifest_schema.py`) with required fields:

```text
artifact_id, artifact_type, phase, branch_name, logical_name
project_root, project_relative_path, path_id, path, path_exists
file_name, extension, size_bytes, modified_time_ns, modified_time_utc, created_at_utc
content_hash, hash_status, schema_hash
row_count, column_count, columns, raw_dtypes, normalized_dtypes
model, filter, horizon, split, target_name, policy_name, consensus_tier
source_artifact_ids, source_paths, generated_by_script, git_commit_hash
notes, warnings, metadata
```

Supported `artifact_type` values include: `clean_model_input`, `model_prediction`, `exit_policy_sweep`, `strict_policy_selection`, `consensus_selected_trades`, `direct_target_audit`, `manifest`, `summary_report`, `architecture_doc`, `onchain_audit`, `settings_snapshot`, `model_artifact`, `policy_backtest`, `unknown`.

## Path Normalization and artifact_id Stability

All stored paths are **project-relative POSIX paths** (forward slashes, no drive letters).

| Field | Definition |
|-------|------------|
| `project_relative_path` | Path relative to detected project root |
| `path_id` | `sha256("path:v1\|" + project_relative_path)` |
| `content_hash` | SHA-256 of raw file bytes |
| `logical_artifact_key` | Deterministic string from phase, type, logical_name, model, filter, horizon, split, target, policy, consensus_tier |
| `artifact_id` | `sha256("artifact:v1\|" + logical_artifact_key + "\|" + content_hash)` |

`artifact_id` does **not** include absolute paths or `project_root`, so the same file content and metadata yield the same id on Windows and Linux.

## Hashing Rules

- **content_hash**: stream file bytes in chunks; required unless `hash_status` is `failed` or `skipped_by_size_limit` with matching warnings.
- **schema_hash**: hash canonical JSON of ordered column names + **normalized** dtypes (not full row data).
- **hash_status**: `computed`, `reused_from_cache`, `failed`, `skipped_by_size_limit`.

## Hash Caching / Short-Circuit Hashing

On rescan, if an existing registry entry matches `project_relative_path`, `size_bytes`, `modified_time_ns`, and `extension`, the scanner reuses:

- `content_hash`, `schema_hash`, `row_count`, `column_count`, `columns`, `raw_dtypes`, `normalized_dtypes`

and sets `hash_status = reused_from_cache`, `metadata.cache_reused = true`.

`--force-rehash` disables cache reuse.

Uses `stat().st_mtime_ns` for change detection.

## Schema Dtype Normalization

`raw_dtypes` preserve reader-reported dtypes. `normalized_dtypes` collapse platform variants before `schema_hash`:

| Raw family | Normalized |
|------------|------------|
| int8–int64, nullable Int64 | int64 |
| uint8–uint64 | uint64 |
| float16–float64, nullable Float64 | float64 |
| bool, boolean | bool |
| datetime variants | datetime64[ns] |
| string-like | string_or_object |
| category | category |
| object | object |

## Registry Paths

| Output | Path |
|--------|------|
| JSONL registry | `data/training/artifact_registry/artifact_registry.jsonl` |
| CSV mirror (optional) | `data/training/artifact_registry/artifact_registry.csv` |
| Summary (optional) | `data/training/artifact_registry/artifact_registry_summary.json` |
| Validation reports | `data/audits/artifact_registry_validation_<timestamp>.json` |

All writes use atomic temp-file + `os.replace`.

## Validation Rules

`scripts/validate_artifact_registry.py` checks:

- Registry exists; paths exist on disk
- No duplicate `artifact_id`, `project_relative_path`, or `path_id`
- Required identity fields present
- Valid `hash_status`; `content_hash` present unless explained
- Tabular CSV/Parquet: `schema_hash`, `row_count`, dtypes unless `SCHEMA_READ_FAILED`
- Source paths exist or are warning-marked
- Missing `.git` does not fail validation

Report `status`: `ok`, `warning`, or `error`. `--fail-on-error` exits nonzero only when errors exist.

## Preventing CSV / SQLite / Parquet Divergence

Phase E1 encodes the policy from [04 — Data Lineage and Storage](04_data_lineage_and_storage.md):

```text
canonical object/dataframe
  → write artifacts (CSV/Parquet)
  → register artifact metadata/lineage (JSONL registry)
  → optional SQLite mirror (future E1B+, not E1)
```

The registry is a **read-mostly catalog of written files**. It does not generate tabular data independently. Future exporters should register each write in the same transaction flow as the file export.

## Later Phases: Safe SQLite Mirroring

A future controlled phase may:

1. Read `artifact_registry.jsonl` as source of truth for research metadata
2. Upsert into a dedicated SQLite table (not `trader.db` runtime tables) using `artifact_id` as primary key
3. Reject writes where `content_hash` or `schema_hash` disagrees with on-disk file
4. Never rebuild Parquet/CSV from SQLite alone

E1 provides the manifest layer required before that gate.

## Default Scan Roots

Registration scans only known research/documentation directories (see `DEFAULT_SCAN_ROOTS` in `app/artifacts/registry.py`). Missing roots emit warnings, not failures.

## Open Questions

1. **E1B Decision Gate**: When to mirror registry rows into SQLite and which DB file/table owns research metadata?
2. **Lineage graph**: Should `source_artifact_ids` be populated by explicit manifest files in Phase B/D scripts rather than path heuristics?
3. **Versioned duplicate paths**: E1 treats duplicate `project_relative_path` as validation errors; future versioning may allow superseded rows with explicit `supersedes_artifact_id`.
4. **Large model binaries**: Default has no size skip; operators may use `--max-file-size-mb` for very large artifacts if needed.
5. **TAB artifact naming**: TabICL outputs may need dedicated path conventions for reliable `model=TAB` inference.

## Anchor Plan

This phase does **not** change the Anchor Plan (XGB/TAB/RF ranking, tiered consensus, context intelligence, Qwen/Gemini audit). It supports auditability and data consistency for later implementation phases E2–E12.
