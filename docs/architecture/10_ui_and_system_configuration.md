# 10 — UI and System Configuration

## Purpose

Document future UI architecture for the four-layer decision system. **Do not implement UI in Phase E0** — documentation only.

## Diagram

See [diagrams/ui_system_configuration.mmd](diagrams/ui_system_configuration.mmd).

## Current State

| Component | Location |
|-----------|----------|
| SPA shell | `static/index.html` — dashboard, positions, settings, analytics, Gemini chat |
| Settings module | `static/system_config.js` — effective settings, PATCH save, inspector |
| API backend | `app/api.py` — tokens, paper, settings, sentiment, analytics |
| Existing panels | Market table, RSS sentiment sidebar, wallet/positions, training charts |

**Missing:** Model scores, consensus tier, enrichment detail, LLM audit parse status, data lineage.

## Target UI Panels

### Model Scores Panel

- XGB score, TAB score, RF score
- Per-model ranks
- Model version / artifact_id from registry

### Consensus Tier Panel

- `TAB_XGB_RF_ALL3` (Tier 1)
- `TAB_RF_ONLY` (Tier 2)
- Rejected combos (`TAB_XGB_ONLY`, `XGB_RF_ONLY`) — visible for transparency, not actionable BUY

### Candidate Monitor

- Candidate state machine
- Active filter / horizon
- Decision status: pending → gated → enriched → memo → final

### Exit Policy Panel

- TP, SL, time-stop, fee assumptions
- Simulated / expected return
- Policy ID linked to offline closure policies

### Enrichment Panel

- Solana RPC status
- Helius validation status
- Wallet intelligence summary
- RSS sentiment (candidate + market)
- Reputation/scam flags

### LLM Audit Panel

- Qwen memo (structured)
- Gemini audit status (if invoked)
- Reason codes (sanitized)
- Audit parse status (`OK` / `AUDIT_PARSE_INVALID`)

### Paper Execution Panel

- `PAPER_BUY`, `WATCH`, `BLOCK` actions and history
- Open paper positions
- Realized paper P/L
- DEMO vs LIVE mode indicator (still paper until upgraded)

### Data Lineage Panel

- `artifact_id`
- `git_commit_hash`
- `content_hash`
- `schema_hash`
- Model versions (XGB, TAB, RF, meta)
- Dataset version
- Settings hash

## System Configuration

| Area | Source |
|------|--------|
| Trading mode | `trading_mode`: DEMO (default) / LIVE |
| Economic gate | RF threshold, Tab boost, slippage, freshness |
| LLM | Provider (`none` / `gemini` / `ollama`), budgets |
| Helius | API key, credit budget |
| Risk | Auto-execution, paper trading enabled |

Settings flow: `GET /api/settings/effective` → UI form → `PATCH /api/settings` → watcher reload.

## Key Inputs

- FastAPI endpoints (existing + future candidate detail endpoint)
- SQLite + artifact registry

## Key Outputs

- Operator visibility into four-layer decision
- Config changes via settings PATCH (unchanged in E0)

## Consumers

- Human operator, demo validation (E12), QA

## Open Questions

- Single candidate detail view vs inline table expansion
- Real-time WebSocket vs poll for scan updates
- How much Gemini chat overlaps with LLM Audit Panel

## Non-Goals

- Editing `static/index.html` or `static/system_config.js` in Phase E0
- New API endpoints
- UI implementation of any panel
