# 11 — Phase E Roadmap

## Purpose

Implementation roadmap from E0 (this branch) through E12 (QA/demo validation). Each phase lists goal, inputs, outputs, likely affected files, acceptance criteria, and what not to change.

## Diagram

See [diagrams/phase_e_roadmap.mmd](diagrams/phase_e_roadmap.mmd).

---

## E0 — Architecture Docs (This Branch)

| | |
|---|---|
| **Goal** | Complete architecture package with Mermaid diagrams |
| **Inputs** | Current codebase, Phase A–D research findings |
| **Outputs** | `docs/architecture/**` |
| **Files likely affected** | `docs/architecture/**` only |
| **Acceptance** | 12+ MD files, 12+ MMD files, index with safety note |
| **Do not change** | Any application code, data, scripts, UI |

---

## E1 — Artifact Registry

| | |
|---|---|
| **Goal** | Implement `artifact_registry` with manifest hashes and provenance |
| **Inputs** | [04_data_lineage_and_storage.md](04_data_lineage_and_storage.md) spec |
| **Outputs** | Registry table/module, manifest writer, registration API |
| **Files likely affected** | `app/database.py`, new `app/artifacts/`, scripts export paths |
| **Acceptance** | Every new export registers `content_hash`, `schema_hash`, `git_commit_hash` |
| **Do not change** | Model weights, trading behavior, live scan logic |

---

## E2 — Unified Candidate Schema

| | |
|---|---|
| **Goal** | Single candidate record schema across runtime, exports, UI |
| **Inputs** | E1 registry, current SQLite tables |
| **Outputs** | Pydantic/dataclass schema, migration plan |
| **Files likely affected** | `app/models/`, `app/database.py`, export scripts |
| **Acceptance** | One schema doc + validation; candidates carry scores, tier, sentiment fields |
| **Do not change** | Consensus logic, LLM routing until E6/E9 |

---

## E3 — Direct Target Dataset Builder

| | |
|---|---|
| **Goal** | Build full `direct_target_dataset` with net-profit labels |
| **Inputs** | V5 selected trades, exit sim, CLEAN_MODEL_INPUT |
| **Outputs** | Parquet + manifest registered in E1 |
| **Files likely affected** | `scripts/` new builder, `app/training/` |
| **Acceptance** | Validity rule satisfied; row-level `net_profitable_after_exit_policy` |
| **Do not change** | Runtime inference, UI |

---

## E4 — Direct-Target XGB / RF

| | |
|---|---|
| **Goal** | Retrain XGB and RF on direct target |
| **Inputs** | E3 dataset |
| **Outputs** | New model artifacts + evaluation reports |
| **Files likely affected** | `scripts/train_*`, `data/training/models/` |
| **Acceptance** | Beats or matches x2-proxy baseline on direct-target holdout |
| **Do not change** | Live runtime until E10 |

---

## E5 — Direct-Target TAB

| | |
|---|---|
| **Goal** | Retrain TAB/TabICL on direct target (if E4 justifies cost) |
| **Inputs** | E3 dataset, E4 results |
| **Outputs** | TabICL direct-target predictions + manifest |
| **Files likely affected** | `app/training/tabicl_v2_eval.py`, scripts |
| **Acceptance** | Documented ROI vs training cost |
| **Do not change** | XGB/RF artifacts from E4 |

---

## E6 — Tiered Consensus Direct-Target Evaluation

| | |
|---|---|
| **Goal** | Rerun tiered consensus on direct-target model outputs |
| **Inputs** | E4/E5 predictions, tier rules from [05](05_model_roles_and_consensus.md) |
| **Outputs** | Tier performance tables, updated closure policies |
| **Files likely affected** | `scripts/phase_*` successors |
| **Acceptance** | Tier 1 `TAB_XGB_RF_ALL3` validated under direct labels |
| **Do not change** | Rejected `ALL3_INTERSECT` standalone strategy |

---

## E7 — Meta-Model / Stacking

| | |
|---|---|
| **Goal** | Stack scores + tier + context features |
| **Inputs** | E6 outputs, sentiment/enrichment features |
| **Outputs** | Meta-model artifact, evaluation report |
| **Files likely affected** | `app/training/`, scripts |
| **Acceptance** | Improves calibration vs best single model on direct target |
| **Do not change** | Base model artifacts without version bump |

---

## E8 — Context Intelligence Integration

| | |
|---|---|
| **Goal** | Wire Solana/Helius wallet intelligence + reputation slot |
| **Inputs** | [07](07_context_intelligence_layer.md), parsers/providers |
| **Outputs** | Enrichment pipeline for high-value candidates |
| **Files likely affected** | `app/live.py`, `app/observability/`, providers |
| **Acceptance** | Enrichment logged; no coarse whale_score BUY gate |
| **Do not change** | Layer 1–2 numeric gates |

---

## E9 — Qwen / Gemini Reasoning Layer

| | |
|---|---|
| **Goal** | Candidate memo pipeline, selective Gemini, audit sanitization |
| **Inputs** | [08](08_news_sentiment_and_reasoning_pipeline.md), [09](09_llm_reasoning_and_audit_layer.md) |
| **Outputs** | Sanitized audit API, Qwen memo schema, Gemini triggers |
| **Files likely affected** | `app/api.py`, `app/models/predictor.py`, `app/llm_*` |
| **Acceptance** | No character-level audit bug; LLM budget respected |
| **Do not change** | Numeric eligibility rules |

---

## E10 — Runtime Paper/Demo Integration

| | |
|---|---|
| **Goal** | Full four-layer pipeline in watcher |
| **Inputs** | E6–E9 components |
| **Outputs** | Tier-aware paper/demo decisions |
| **Files likely affected** | `app/live.py`, `app/observability/actionability.py` |
| **Acceptance** | PAPER_BUY/WATCH/BLOCK with full trace; still no real wallet |
| **Do not change** | Real-money execution |

---

## E11 — UI Integration

| | |
|---|---|
| **Goal** | Panels from [10](10_ui_and_system_configuration.md) |
| **Inputs** | E10 APIs, E1 lineage |
| **Outputs** | Updated `static/index.html`, `system_config.js` |
| **Files likely affected** | `static/**`, `app/api.py` endpoints |
| **Acceptance** | All documented panels visible with live data |
| **Do not change** | Backend consensus rules without E10 alignment |

---

## E12 — QA / Demo Validation

| | |
|---|---|
| **Goal** | End-to-end demo validation, regression suite |
| **Inputs** | Full E10/E11 system |
| **Outputs** | QA report, demo script, acceptance sign-off |
| **Files likely affected** | `tests/`, docs |
| **Acceptance** | Deterministic replay of sample scan; artifact lineage verified |
| **Do not change** | Production data files without backup |

---

## Suggested Next Phase

**E1 Artifact Registry** — establishes provenance discipline required before E3 dataset builder and all model retraining. Alternative parallel start: **E2 Unified Candidate Schema** if registry and schema design should be co-developed.

## Non-Goals (E0)

- Implementing any E1–E12 work in this branch
