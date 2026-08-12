# 09 — LLM Reasoning and Audit Layer

## Purpose

Document Qwen/Ollama and Gemini roles, LLM gating rules, auditable outputs, and the **audit reason sanitization pipeline** (future implementation requirement).

## Diagram

See [diagrams/llm_reasoning_audit.mmd](diagrams/llm_reasoning_audit.mmd) and [diagrams/audit_reason_sanitization_pipeline.mmd](diagrams/audit_reason_sanitization_pipeline.mmd).

## LLM Rule

```text
XGB/TAB/RF determine quantitative eligibility.
Policy layer determines economic feasibility.
Solana/Helius/RSS/reputation provide context.
Qwen explains, summarizes, and flags.
Gemini audits selectively.
Execution remains paper/demo until explicitly upgraded.
```

## Qwen / Ollama — Local Operational Reasoning

| Role | Description |
|------|-------------|
| Local operational reasoning | Default local path via Ollama (`qwen3:8b` in `app/llm_config.py`) |
| Candidate memo generator | Summary after numeric gates pass |
| Sentiment/context summarizer | RSS + enrichment bundle |
| Risk explainer | Human-readable flags |
| Soft veto assistant | Recommend WATCH/BLOCK without bypassing gates |
| Audit memo generator | Structured reason codes + narrative |

**Not** a quantitative entry model. Run **only after** numeric/model/policy filtering — **not on every market row**.

### Target Qwen Output Fields

- Candidate summary
- Sentiment summary
- Enrichment summary
- Reason codes
- Risk flags
- Final explanation
- Soft veto recommendation

## Gemini — Selective External Audit

| Role | Description |
|------|-------------|
| Selective external audit | Not runtime default for every candidate |
| False-positive analysis | Review Tier 1 edge cases |
| Scam/reputation deep-dive | High-risk keyword or wallet flags |
| Periodic Qwen benchmark | Quality comparison |
| **Not** final decision authority | Execution remains gated by numeric layers |

## Short-Circuit and Budget

- `app/observability/llm_gate.py` — skip LLM when economic/risk gates block
- Ollama budget: e.g. 5 calls/scan (`app/llm_config.py`)
- **Max LLM calls per scan** concept: prioritize Tier 1 > Tier 2 > enrichment-flagged
- **No LLM-only BUY** — ever

## Auditable LLM Outputs

All LLM outputs must be:

- Persisted (`gemini_decisions`, JSONL traces)
- Linked to candidate ID and scan cycle
- Include provider tag (`gemini`, `ollama_qwen3_8b`)
- Subject to audit reason sanitization before UI aggregation

## Audit Reason Sanitization Pipeline

Future code must **not** infer audit reasons by iterating over raw strings.

### Required Flow

```text
raw audit payload
→ json.loads / structured parser
→ schema/type validation
→ sanitize reason list
→ normalize reason codes
→ reject/flag malformed payloads
→ write sanitized audit event
→ UI / report / decision trace
```

### Mermaid (canonical)

See [diagrams/audit_reason_sanitization_pipeline.mmd](diagrams/audit_reason_sanitization_pipeline.mmd).

### Explicit Future Coding Rule

```text
Audit reasons must be parsed as structured JSON or typed objects. A raw string must never be treated as an iterable collection of reasons, because that can create character-level false reasons.
```

### Current Bug (Document Only — Do Not Fix in E0)

**Location:** `app/api.py` — `GET /api/pipeline/audit/recent` (~lines 295–299)

```python
for reason in (r.get("audit_reasons") or r.get("audit_reasons_json") or []):
```

`audit_reasons_json` is stored as a JSON **string** in SQLite. Iterating a string yields single-character "reasons" in `reason_counts`.

**Correct pattern exists in:** `scripts/diagnostics/_common.py` → `parse_audit_reasons_field()`

**Required fix:** Phase E9/E10 — use structured parser in API and all consumers.

## Current State

- Gemini: whale-event decisions, position management, UI chat (`POST /api/chat`)
- Ollama: optional via `--mode ollama`
- `AuditReason` enum in `app/observability/audit_reasons.py` (~70 codes)
- Pipeline audit writes JSON string to SQLite; API aggregation bug present

## Target State

- Qwen memo pipeline for tier-qualified candidates
- Gemini invoked selectively (flags, Tier 1 spot-check, scam deep-dive)
- All audit reasons sanitized before API/UI
- `AUDIT_PARSE_INVALID` flag on malformed payloads

## Key Inputs

- Tier-qualified candidate bundle, sentiment context, enrichment summary
- Economic gate outcome, model scores

## Key Outputs

- Qwen memo JSON, optional Gemini audit JSON
- Sanitized reason list, decision trace entries

## Consumers

- UI LLM Audit Panel, pipeline audit API, JSONL reports

## Open Questions / Required Future Fixes

1. Implement `parse_audit_reasons_field()` (or equivalent) in `app/api.py`
2. Mixed-case reason codes (UPPER_SNAKE vs lowercase `*_DETAIL`) — normalization table needed
3. Diagnostic `inspect_audit_reason_parsing.py` uses wrong endpoint path (`/api/audit/recent` vs `/api/pipeline/audit/recent`)

## Non-Goals

- Fixing parsing code in Phase E0
- Changing LLM provider routing or budgets
- Invoking Qwen/Gemini/Ollama in this branch
