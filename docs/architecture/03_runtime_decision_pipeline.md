# 03 — Runtime Decision Pipeline

## Purpose

Document the target runtime flow from scan through tiered consensus, enrichment, LLM reasoning, audit sanitization, and paper/demo decision outputs.

## Diagram

See [diagrams/runtime_decision_pipeline.mmd](diagrams/runtime_decision_pipeline.mmd).

```text
scan
→ candidate feature build
→ XGB/TAB/RF scoring
→ consensus tier assignment
→ pair cap / concentration check
→ exit economics
→ enrichment if high-value candidate
→ RSS/news sentiment context
→ Qwen memo
→ optional Gemini audit
→ audit reason sanitization
→ PAPER_BUY / WATCH / BLOCK
→ audit trail
```

**Audit reason sanitization** (required future step): see [diagrams/audit_reason_sanitization_pipeline.mmd](diagrams/audit_reason_sanitization_pipeline.mmd) and [09_llm_reasoning_and_audit_layer.md](09_llm_reasoning_and_audit_layer.md).

## Current State

| Step | Today (`app/live.py`, observability) |
|------|--------------------------------------|
| Scan | DexScreener trending; RSS archival each cycle |
| Feature build | `app/analytics/features.py`, engine whale score |
| Scoring | RF runtime inference + Tab lookup; **XGB not in live path** |
| Consensus tier | **Not implemented** in runtime |
| Economic gate | `app/observability/economic_gate.py` — RF threshold, Tab boost, slippage/freshness |
| LLM | Gemini/Ollama on whale-like events; `llm_gate` short-circuit |
| Decision | `evaluate_and_execute_candidate()` → paper buy when gates pass |
| Audit | `pipeline_audit` SQLite + JSONL; **API reason parsing bug** |

### Runtime constraints (preserve)

- **No wallet connected** — DEMO/LIVE modes are paper/demo; not real-money execution
- **Deterministic numeric gates first** — LLM must not bypass RF/economic gates
- **LLM not on every row** — budget limits (`llm_config.py`: e.g. 5 Ollama calls/scan)
- **Audit trail required** for every candidate path

## Target State

Full pipeline above with:

- XGB/TAB/RF scores attached to each candidate
- Tier assignment: Tier 1 `TAB_XGB_RF_ALL3`, Tier 2 `TAB_RF_ONLY`, reject combos research-only
- Enrichment only for high-value candidates (Solana RPC → Helius → wallet intelligence)
- RSS sentiment as context input to Qwen memo
- Structured audit reason sanitization before UI/API exposure
- Explicit `PAPER_BUY` / `WATCH` / `BLOCK` with decision trace

## Key Inputs

- Live DexScreener pairs, settings, model artifacts
- RSS sentiment matrix, optional RPC/Helius enrichment payloads
- Consensus tier rules from [05_model_roles_and_consensus.md](05_model_roles_and_consensus.md)

## Key Outputs

- `pipeline_audit` rows with sanitized `audit_reasons`
- `gemini_decisions`, paper trade records
- JSONL `decision_trace_*.jsonl`

## Consumers

- Background watcher, FastAPI endpoints, UI panels (future)

## Open Questions / Required Future Fixes

1. **Audit reason API bug:** `GET /api/pipeline/audit/recent` in `app/api.py` iterates `audit_reasons_json` string as iterable → character-level false reasons. Must use structured parser per [09](09_llm_reasoning_and_audit_layer.md). Correct pattern exists in `scripts/diagnostics/_common.py` → `parse_audit_reasons_field()`.
2. Mapping tiered consensus to existing `economic_gate` without breaking DEMO behavior.
3. Max LLM calls per scan vs candidate volume.

## Non-Goals

- Implementing runtime pipeline changes in Phase E0
- Connecting real wallet or live DEX execution
