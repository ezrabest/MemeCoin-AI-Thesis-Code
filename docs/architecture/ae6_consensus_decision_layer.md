# AE6 — Consensus Decision Layer

## What AE6 implements

AE6 is the first real architecture execution layer after data collection. It produces **audit-ready consensus decision records** from recent runtime data:

```text
RAW / provider payloads
→ normalized market snapshots
→ signals / model scores / context
→ consensus decision record   ← AE6
→ later UI / paper-demo / LLM audit (AE7–AE9)
```

The `app/decision/` package defines:

- Structured decision types (`DecisionRecord`, `LineageMetadata`, model score slots)
- Transparent baseline consensus over RF / XGB / TAB slots
- Conservative non-trading decision status assignment
- Append-only JSONL persistence with per-record `flush` + `fsync`

Every record includes:

- Identity and lineage
- Market and signal context
- Model score slots (explicit missingness when unavailable)
- Consensus family and strength
- Research/context/LLM placeholders
- `no_trade_authority = true`

Allowed decision statuses:

| Status | Meaning |
|--------|---------|
| `WATCH` | Runtime data present; insufficient model/context consensus for promotion |
| `RESEARCH_CANDIDATE` | Active signal support; models/context not sufficient for review tier |
| `PAPER_CANDIDATE_REVIEW` | Meets documented **review-only** thresholds — **does not open paper trades** |
| `BLOCK` | Hard safety issues: missing identity, stale snapshot, invalid lineage |
| `NO_DECISION` | Default before status evaluation in some paths |

## What AE6 does not implement

| Phase | Scope |
|-------|-------|
| AE7 | Meta-model training |
| AE8 | Helius / Solana / RSS expansion (placeholders only; local RSS read optional) |
| AE9 | Qwen / Gemini execution (placeholders only) |
| — | UI changes |
| — | Live or paper trade execution |
| — | Risk setting changes |
| — | Model training or rescoring |

## RAW → derived → signal → decision lineage

AE6-0 found `lineage_strength = WEAK_IMPLICIT_TIME_PAIR_LINKS`. Tables do not always carry explicit `raw_payload_id` / `snapshot_id` foreign keys on signals.

AE6 therefore supports two lineage modes:

### `EXPLICIT_LINKAGE` / `STRONG_EXPLICIT_LINKS`

Used when **both** `raw_payload_id` and `snapshot_id` are present on the decision lineage object.

### `BEST_EFFORT_IMPLICIT_LINKAGE` / `WEAK_IMPLICIT_TIME_PAIR_LINKS`

Used when either ID is missing. The builder matches by:

- `provider`
- `pair_address`
- signal / snapshot / raw timestamps
- scan/run context where available

Required fallback text:

```text
Lineage fallback: best-effort match using provider, pair_address, timestamps, and scan/run context.
```

Required caveat:

```text
RAW/derived lineage is best-effort implicit, based on provider/pair/timestamp context.
```

## Why `LineageMetadata` is mandatory

Decision records without explained lineage cannot be audited. AE6 **fails closed** if:

- `lineage` is missing
- `lineage_mode` or `lineage_strength` is missing
- `BEST_EFFORT_IMPLICIT_LINKAGE` lacks `fallback_reason` or `lineage_warning`

Silent lineage gaps are not permitted.

## Model score slots

Slots exist for `RF`, `XGB`, `TAB`, and `META`. Each slot records:

```text
available, score, rank, model_artifact_id, prediction_artifact_id,
horizon, filter, exit_policy, missing_reason
```

When runtime aligned scores are unavailable (current default):

```text
available: false
missing_reason: "NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT"
```

AE6 does **not** invent scores, train models, or load large offline prediction artifacts unless a safe existing runtime path is added later.

## Consensus families

`app/decision/consensus.py` computes:

```text
available_model_count, vote_count, consensus_family,
consensus_strength, consensus_caveat
```

Families:

- `TAB_XGB_RF_ALL3`
- `TAB_RF_ONLY`
- `TAB_XGB_ONLY`
- `XGB_RF_ONLY`
- `SINGLE_MODEL_ONLY`
- `NO_MODEL_CONSENSUS_AVAILABLE`

### Why `NO_MODEL_CONSENSUS_AVAILABLE` is valid

Missing aligned RF/XGB/TAB runtime scores are **expected** in the current deployment. This state:

- Is **not** an exception
- Does **not** stop decision-record generation
- Sets `consensus_strength = "UNAVAILABLE"` and an explicit caveat

## Research context — `whale_score_asof`

`whale_score_asof` is recorded as:

```text
status = "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE"
not_rule = true
not_runtime_approved = true
```

AE6 does **not** implement hard rules such as `whale_score_asof <= 0.0235` as buy/block gates.

## LLM placeholders (AE9)

```text
qwen_memo_available: false
gemini_audit_available: false
llm_execution_allowed: false
llm_decision_authority: false
llm_missing_reason: "AE9_NOT_IMPLEMENTED_YET"
```

LLM output must never be treated as a trading decision in AE6.

## Decision status thresholds (non-trading review only)

Named constants in `app/decision/builder.py` (sourced from existing engine thresholds):

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `REVIEW_MIN_SIGNAL_SCORE` | `SIGNAL_BUY_PROB_THRESHOLD` (0.65) | Review-tier signal support |
| `REVIEW_MIN_WHALE_SCORE` | `SIGNAL_BUY_WHALE_THRESHOLD` (0.5) | Review-tier activity support |
| `REVIEW_MIN_LIQUIDITY_USD` | `SIGNAL_BUY_LIQUIDITY_USD` (25000) | Review-tier liquidity |
| `DEFAULT_MAX_SNAPSHOT_AGE_SECONDS` | 300 | Staleness block threshold |

`PAPER_CANDIDATE_REVIEW` means **worth logging for future paper-demo integration** — not “open a paper trade now.”

## No trade execution

- `no_trade_authority = true` on every record
- Smoke script does not call paper trading, economic gate execution, or LLM providers
- AE6 does not modify risk settings or live/demo trading behavior

## JSONL persistence

System of record:

```text
data/decision_records/ae6_decisions_YYYYMMDD.jsonl
```

Safety properties:

1. Serialize full record **before** opening/writing
2. One complete JSON object per line
3. `flush()` after each line
4. `os.fsync()` after each line
5. Append-only — no rewrite or truncate of prior lines

`read_jsonl_records_safe()` tolerates a final incomplete/corrupt line after crash.

## Smoke script

```powershell
python scripts/run_ae6_consensus_decision_smoke.py --max-records 50 --audit-only
```

Options:

- `--max-records` — cap decision records (default 50)
- `--output-root` — audit output root
- `--no-db-write` — no-op (JSONL-only; accepted for CLI compatibility)
- `--audit-only` — build records + audit summary without appending JSONL

Audit output:

```text
data/audits/ae6_consensus_decision_layer_<timestamp>/ae6_consensus_decision_summary.json
```

## Future integration

| Phase | Plugs into AE6 via |
|-------|-------------------|
| AE7 | Populating `model_scores` slots + meta consensus |
| AE8 | `context_placeholders` (Helius, Solana, reputation, scam flags) |
| AE9 | `llm_context` (Qwen memo, Gemini audit) — audit only, no authority |
| UI / paper-demo | Reads JSONL decision records + traces; no AE6 execution |

## Package layout

```text
app/decision/
  __init__.py
  types.py        — DecisionRecord, LineageMetadata, placeholders
  consensus.py    — baseline consensus function
  builder.py      — runtime row → decision record
  persistence.py  — JSONL writer/reader with fsync
```
