# MemeCoin AI Trader — Architecture Index

> **Note:** This branch is a Single Source of Truth for architecture design only; no logic modifications are permitted. If a diagram requires a logic change to be valid, document the required change in the Open Questions / Future Implementation section instead of implementing it.

## Phase E Purpose

Phase E moves the MemeCoin AI Trader from a single-model trading classifier toward an audited **four-layer decision system**: quantitative ranking (XGB / TAB / RF), tiered consensus economics with direct net-profit targets, context intelligence (Solana / Helius / wallet-level whale / RSS sentiment), and reasoning/audit (Qwen / Gemini). This documentation package defines the current system state, the target architecture, data lineage policy, UI roadmap, and implementation phases E1–E12.

## Branch Safety Rule

Phase E0 is documentation-only. Architecture may describe future code, future schema, future UI, and future runtime behavior, but this branch must not implement those changes. Any discovered mismatch between current code and target architecture must be recorded as an open question or future implementation item.

## Four-Layer Architecture Summary

| Layer | Name | Components |
|-------|------|------------|
| **1** | Quantitative Ranking | XGB (broad ranker), TAB/TabICL (focused ranker), RF (confirmation) |
| **2** | Consensus Economics | Tier 1 `TAB_XGB_RF_ALL3`, Tier 2 `TAB_RF_ONLY`, exit simulation, TP/SL/time-stop, fees, pair cap, direct net-profit target |
| **3** | Context Intelligence | DexScreener context, Solana RPC, Helius enrichment, wallet-level whale intelligence, RSS/news sentiment, reputation/scam risk |
| **4** | Reasoning and Audit | Qwen/Ollama operational reasoning, Gemini selective audit, candidate memo, soft veto, audit reason sanitization, decision trace |

**LLM rule:**

```text
XGB/TAB/RF determine quantitative eligibility.
Policy layer determines economic feasibility.
Solana/Helius/RSS/reputation provide context.
Qwen explains, summarizes, and flags.
Gemini audits selectively.
Execution remains paper/demo until explicitly upgraded.
```

## Document Map

| Document | Description |
|----------|-------------|
| [00 — System Overview](00_system_overview.md) | End-to-end four-layer architecture |
| [01 — Current State Inventory](01_current_state_inventory.md) | What exists, partial, and missing today |
| [02 — Offline Research Pipeline](02_offline_research_pipeline.md) | Training, exit sim, consensus decomposition |
| [03 — Runtime Decision Pipeline](03_runtime_decision_pipeline.md) | Target scan-to-decision flow |
| [04 — Data Lineage and Storage](04_data_lineage_and_storage.md) | SQLite truth, artifacts, registry |
| [05 — Model Roles and Consensus](05_model_roles_and_consensus.md) | XGB/TAB/RF and tiered consensus |
| [06 — Direct Target and Meta-Modeling](06_direct_target_and_meta_modeling.md) | Net-profit target and stacking |
| [07 — Context Intelligence Layer](07_context_intelligence_layer.md) | Solana, Helius, whale intelligence |
| [08 — News Sentiment Pipeline](08_news_sentiment_and_reasoning_pipeline.md) | RSS as context intelligence |
| [09 — LLM Reasoning and Audit](09_llm_reasoning_and_audit_layer.md) | Qwen, Gemini, audit sanitization |
| [09 — Artifact Registry (E1)](09_artifact_registry.md) | File-based registry, hashes, lineage foundation |
| [10 — Unified Candidate Schema (E2)](10_unified_candidate_schema.md) | Canonical Pydantic candidate schema, serialization, validation |
| [10 — UI and System Configuration](10_ui_and_system_configuration.md) | Future UI panels and settings |
| [11 — Phase E Roadmap](11_phase_e_roadmap.md) | E0–E12 implementation phases |
| [11 — Direct Target XGB/RF Training (E4A)](11_direct_target_xgb_rf_training.md) | Offline XGB/RF on E3 direct targets |
| [11 — Direct Exit Target Dataset (E3)](11_direct_exit_target_dataset.md) | E3 direct net-profitable target datasets |

## Diagram Index

| Diagram | File |
|---------|------|
| System overview | [diagrams/system_overview.mmd](diagrams/system_overview.mmd) |
| Offline research pipeline | [diagrams/offline_research_pipeline.mmd](diagrams/offline_research_pipeline.mmd) |
| Runtime decision pipeline | [diagrams/runtime_decision_pipeline.mmd](diagrams/runtime_decision_pipeline.mmd) |
| Data lineage and storage | [diagrams/data_lineage_storage.mmd](diagrams/data_lineage_storage.mmd) |
| Model consensus tiers | [diagrams/model_consensus_tiers.mmd](diagrams/model_consensus_tiers.mmd) |
| Direct target pipeline | [diagrams/direct_target_pipeline.mmd](diagrams/direct_target_pipeline.mmd) |
| Context intelligence layer | [diagrams/context_intelligence_layer.mmd](diagrams/context_intelligence_layer.mmd) |
| News sentiment reasoning | [diagrams/news_sentiment_reasoning_pipeline.mmd](diagrams/news_sentiment_reasoning_pipeline.mmd) |
| LLM reasoning and audit | [diagrams/llm_reasoning_audit.mmd](diagrams/llm_reasoning_audit.mmd) |
| Audit reason sanitization | [diagrams/audit_reason_sanitization_pipeline.mmd](diagrams/audit_reason_sanitization_pipeline.mmd) |
| UI and system configuration | [diagrams/ui_system_configuration.mmd](diagrams/ui_system_configuration.mmd) |
| Phase E roadmap | [diagrams/phase_e_roadmap.mmd](diagrams/phase_e_roadmap.mmd) |

## Current Status

- **Runtime:** FastAPI + background watcher; DexScreener ingestion; SQLite persistence; RF runtime inference + offline Tab lookup; economic gate; paper/demo trading; RSS sentiment archival; Gemini/Ollama on whale-like events.
- **Research:** XGB clean CUDA evaluation complete; Phase B V5.1 audited consensus decomposition; Phase D1 direct net-profit audit; exit simulation artifacts under `data/training/manual_verified_results/`.
- **Not yet integrated:** XGB runtime scoring, tiered consensus in live path, wallet-level whale intelligence in runtime, structured audit reason sanitization in API.
- **Phase E1 (artifact registry):** File-based registry implemented; see [09 — Artifact Registry](09_artifact_registry.md).
- **Phase E2 (unified candidate schema):** Pydantic v2 schema in `app/candidates/`; see [10 — Unified Candidate Schema](10_unified_candidate_schema.md).

## What This Branch Does Not Change

This branch creates or edits **only** `docs/architecture/**`. It does **not** modify:

- `app/**`, `scripts/**`, `static/**`, `data/**`, `tests/**`
- Live/demo/paper trading behavior, risk settings, economic gates
- Model inference, training, Helius/Solana runtime, RSS fetching, LLM routing
- SQLite schema, model artifacts, datasets, CSV/Parquet/SQLite data files

## Open Questions / Future Implementation

See individual documents for phase-specific items. Cross-cutting open questions:

1. **Audit reason API bug:** `GET /api/pipeline/audit/recent` iterates `audit_reasons_json` as a string, producing character-level false reasons. Fix required in Phase E9/E10 using structured parsing (see [09](09_llm_reasoning_and_audit_layer.md)).
2. **Consensus generator script:** Original consensus intersection generator not found in repo; outputs exist as data artifacts only.
3. **XGB not wired to runtime:** XGB is research-only; runtime uses RF + Tab lookup.
4. **Helius/Solana:** Parsers exist but are not invoked on every scan pair in live path.
5. **Artifact registry:** Phase E1 file-based registry implemented; SQLite mirroring deferred to E1B Decision Gate (see [09_artifact_registry.md](09_artifact_registry.md)).
