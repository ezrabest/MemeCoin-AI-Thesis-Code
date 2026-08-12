# 05 — Model Roles and Consensus

## Purpose

Document XGB, TAB/TabICL, and RF roles; tiered consensus interpretation; and explicit rejection of generic TWO_OF_THREE and standalone ALL3_INTERSECT strategies.

## Diagram

See [diagrams/model_consensus_tiers.mmd](diagrams/model_consensus_tiers.mmd).

## Model Roles

### XGB — Primary Broad Ranker

XGB is currently the **strongest standalone broad-market ranking model**.

Best broad standalone policy:

```text
model: XGB
filter: RAW_ALL_VERIFIED
horizon: 24h
top_pct: 0.5%
pair_cap: 50
TP: 2.0308
SL: 0.80
```

**Role:** Primary broad candidate generator / ranker. Not yet wired into live runtime (research-only today).

### TAB / TabICL — Focused High-Quality Ranker

TAB remains valuable as a **focused high-quality ranking model** in liquidity/activity regimes.

Best focused TAB policy:

```text
filter: LIQ_5K_HIGH_ACTIVITY
horizon: 4h
top_pct: 2%
pair_cap: 50
TP: 2.0308
SL: 0.80
```

**Role:** Focused ranker for high-activity liquidity band. Do not discard TAB.

### RF — Conservative Confirmation

RF is **weak as a standalone model** relative to XGB/TAB.

**Role:** Conservative confirmation / sanity signal. Especially important when RF agrees with TAB.

Runtime today: RF threshold ~0.70 in economic gate (`app/observability/economic_gate.py`).

## Tiered Consensus (Not Generic TWO_OF_THREE)

Generic `TWO_OF_THREE` is **too coarse** — it does not distinguish which pair of models agrees. Use **tiered interpretation** within audited selected-trade decomposition.

### Tier Table

| Tier | Combo | Meaning | Current action |
|------|-------|---------|----------------|
| Tier 1 | `TAB_XGB_RF_ALL3` | TAB, XGB, and RF all agree | Primary gold signal / economic engine |
| Tier 2 | `TAB_RF_ONLY` | TAB and RF agree; XGB does not | Secondary focused confirmation |
| Reject | `TAB_XGB_ONLY` | Weak/negative in audit | Research-only — not independent production entry |
| Reject | `XGB_RF_ONLY` | Weak/negative in audit | Research-only — not independent production entry |

### Tier 1 — `TAB_XGB_RF_ALL3`

- Strongest internal consensus signal in audited decomposition
- Primary economic engine for future evaluation
- **Not** the same as old standalone `ALL3_INTERSECT` strategy

### Tier 2 — `TAB_RF_ONLY`

- Strongest pair-only confirmation rule
- Secondary focused confirmation when XGB disagrees

### Reject / Research-Only

- `TAB_XGB_ONLY`, `XGB_RF_ONLY` — do not use as independent production/demo entries unless future direct-target retraining proves otherwise

## ALL3_INTERSECT vs TAB_XGB_RF_ALL3 (Critical Distinction)

| Concept | Status |
|---------|--------|
| Old standalone `ALL3_INTERSECT` strategy | **Rejected** — over-filtered standalone strategy |
| `TAB_XGB_RF_ALL3` within audited selected-trade decomposition | **Accepted** — strongest internal Tier 1 signal |

```text
Within audited selected-trade decomposition, the subset where TAB + XGB + RF all agree is the strongest internal Tier 1 signal.
```

Do **not** revive `ALL3_INTERSECT` as a standalone entry strategy.

## Current State

- Combo labels produced in Phase B V5.1 and Phase D scripts
- `TWO_OF_THREE` and `ALL3_INTERSECT` CSVs exist in `consensus_intersections/` as historical artifacts
- Live path uses RF + Tab only; no tier assignment

## Target State

- Runtime and UI expose `consensus_tier` per candidate
- Tier 1 drives primary paper/demo consideration after economic gate
- Tier 2 secondary path with stricter enrichment/LLM review
- Reject combos logged for research only

## Key Inputs

- `score_xgb`, `score_tab`, `score_rf` (and ranks)
- Policy context: filter, horizon, top_pct, pair_cap

## Key Outputs

- `consensus_tier`, `consensus_combo` fields on candidate/decision records

## Consumers

- Economic gate, direct-target meta-model (E7), UI Consensus Tier Panel

## Open Questions

- Thresholds for "agree" (rank intersection vs score threshold vs selected-trade membership)
- Whether Tier 2 should require additional RSS/Helius clearance

## Non-Goals

- Changing consensus scripts or re-running Phase B in E0
- Implementing tier assignment in live code
