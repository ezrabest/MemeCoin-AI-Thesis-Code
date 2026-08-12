# AE16 — Tiered Consensus Engine on Direct Target / Clean Forward Bridge

## Purpose

AE16 implements the original **E6 Tiered Consensus Engine** over the cleaned AE15 Clean Forward package.

```
AE15 cleaned Clean Forward candidates
→ model-evidence adapter (RF / XGB / TAB)
→ tiered consensus decisions (research/shadow/paper-demo only)
→ audits + decision gate
```

AE16 is a **schema / evidence attachment / consensus-engine** phase. It does not train, backtest, trade live, or grant wallet authority.

## Inputs (mandatory)

Only the cleaned AE15 package:

```
data/audits/ae15_cleaned_for_ae16_20260722_194200/data/
  ae16_clean_forward_candidates.csv
  ae16_clean_forward_decision_inputs.csv
  ae16_clean_forward_outcome_label_contract.csv
  ae16_clean_forward_paper_execution_links.csv
```

Expected counts: 961 / 961 / 961 / 1.

AE15 leaves `model_scores_available=False` with empty RF/XGB/TAB scores. That is expected. AE16 attaches evidence via an explicit adapter.

## Package

`app/consensus/`

| Module | Role |
|--------|------|
| `model_evidence.py` | Discover historical RF/XGB/TAB artifacts; fail-closed attachment |
| `tiered_engine.py` | Vote rules + consensus tier assignment |
| `audits.py` | Preflight, contract, invented-scores, authority, legacy audits |
| `serialization.py` | CSV/JSON/JSONL writers |

Runner: `scripts/run_ae16_tiered_consensus_engine.py`

## Model families

| Label | Meaning |
|-------|---------|
| **XGB** | XGBoost / XGBClassifier artifacts, predictions, ranks, scores |
| **RF** | Random Forest artifacts, predictions, ranks, scores |
| **TAB** | TabICL / TabICLv2 tabular artifacts, predictions, ranks, scores |

## Attachment rules

- Never invent scores.
- Never default missing scores to `0`.
- Never count missing scores as votes.
- Never treat AE15 placeholder columns / `consensus_tier_shadow` as final evidence.
- Exact-ID join only (`candidate_policy_id` / `target_row_id` / `candidate_id`). Pair/timestamp-only alignment is rejected.
- Expected failures become `attachment_status` rows; they do not crash the full run.

## Consensus tiers

| Tier | Meaning |
|------|---------|
| `TAB_XGB_RF_ALL3` | Tier 1 — all three voting |
| `TAB_RF_ONLY` | Tier 2 — TAB + RF |
| `TAB_XGB_ONLY` | Research-only |
| `XGB_RF_ONLY` | Research-only |
| `SINGLE_MODEL_ONLY` | One model voting |
| `MODEL_EVIDENCE_UNAVAILABLE` | No attached evidence |
| `CONSENSUS_NOT_COMPUTABLE` | Unexpected pattern |
| `RESEARCH_ONLY_WATCH` / `REJECT_OR_SKIP` | Reserved research/reject labels |

A model votes only when `evidence_attached=true`, score is numeric/non-null, and `attachment_status=MODEL_EVIDENCE_ATTACHED`.

## Authority

All AE16 outputs enforce:

- `trade_authority = false`
- `live_trading_ready = false`
- `wallet_authority = false`
- `risk_gate_override_authority = false`
- `paper_demo_only = true`
- `authority_status = RESEARCH_SHADOW_ONLY`

## Run

```bash
python scripts/run_ae16_tiered_consensus_engine.py
python scripts/run_ae16_tiered_consensus_engine.py --input-root data/audits/ae15_cleaned_for_ae16_20260722_194200/data
```

## Classifications

- `AE16_TIERED_CONSENSUS_ENGINE_PASS_WITH_MODEL_EVIDENCE`
- `AE16_TIERED_CONSENSUS_ENGINE_PASS_SCHEMA_ONLY_NO_MODEL_EVIDENCE`
- `AE16_BLOCKED_INPUT_FILES_MISSING`
- `AE16_BLOCKED_INPUT_CONTRACT_FAILED`
- `AE16_BLOCKED_MODEL_EVIDENCE_ADAPTER`
- `AE16_BLOCKED_LEGACY_CONTAMINATION`
- `AE16_BLOCKED_INVENTED_OR_DEFAULTED_SCORES`
- `AE16_BLOCKED_AUTHORITY_ESCALATION`

## AE16 continuation — model-evidence bridge completion

When the engine shell alone yields `MODEL_EVIDENCE_UNAVAILABLE` for all candidates, run:

```bash
python scripts/run_ae16_model_evidence_bridge_completion.py
```

This continuation performs read-only RF/XGB/TAB discovery, exact-ID join audit, feature-parity / inference compatibility, attachment v2, and consensus v2.

It does **not** train, backtest, or start AE17.

If neither exact-ID join nor feature-parity inference is safe, AE16 remains blocked (for example `AE16_BLOCKED_FEATURE_PARITY_GAP`) with proof under:

`data/audits/ae16_model_evidence_bridge_completion_<timestamp>/`
