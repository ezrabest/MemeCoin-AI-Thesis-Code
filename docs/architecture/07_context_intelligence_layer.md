# 07 — Context Intelligence Layer

## Purpose

Document Solana RPC, Helius validation, wallet-level whale intelligence, reputation/scam future slot, and RSS as context — not primary entry gates.

## Diagram

See [diagrams/context_intelligence_layer.mmd](diagrams/context_intelligence_layer.mmd).

## Architectural Role

```text
Solana/Helius should initially be an enrichment and audit layer, not a primary entry gate.
```

```text
Use wallet-level intelligence to enrich or block high-value candidates.
```

**Non-goal:**

```text
Do not use coarse whale_score as a primary BUY gate.
```

## Current Whale Proxy Finding

Latest model work showed `LIQ_5K_HIGH_ACTIVITY` and `NO_WHALE_FILTER` behaved **nearly identically**. This does **not** mean whales are irrelevant — the current **coarse whale_score gate is insufficient**.

The whale thesis remains alive as **wallet-level intelligence**, not `coarse whale_score gate`.

## Solana Raw RPC

| Component | Path |
|-----------|------|
| RPC client | `app/providers/solana_rpc.py` |
| Pool activity parser | `app/parsers/solana_pool_activity.py` — swap direction, token deltas |
| Wallet behavior | `app/parsers/solana_wallet_behavior.py` |

**Role:** Parse transactions for signer, fee payer, token owner, token delta, buyer/seller inference. Cache under `data/cache/solana_rpc/`.

## Helius Validation / Enrichment

| Component | Path |
|-----------|------|
| Enhanced transactions API | `app/providers/helius.py` |
| Budget pacing | `app/providers/helius_budget.py` |
| Validation compare | `compare_raw_with_helius()` |

**Role:** Validate raw RPC parses; enrich with enhanced metadata. Research-heavy in `phase_c_whale_manual_audit/`.

## Wallet-Level Whale Intelligence (Target)

Candidate-centered analysis using:

- Signer, fee payer, token owner
- Token delta, buyer/seller inference
- Wallet behavior, repeat-wallet signal
- Suspicious wallet / router / relayer detection
- Flow pressure, buy/sell pressure over time windows

Initially: **enrich or block** high-value candidates after Layer 1–2 pass, not filter entire market.

## Reputation / Scam Risk (Future)

Production-grade reputation layer not yet implemented. Slot in Layer 3 alongside RSS negative flags and Gemini deep-dives.

## RSS / News Sentiment

Documented in detail in [08_news_sentiment_and_reasoning_pipeline.md](08_news_sentiment_and_reasoning_pipeline.md). RSS is context intelligence, not a trading model.

## Current State

- Parsers and providers implemented; limited live scan integration
- Engine `whale_score` used in signals; no independent value in latest filter comparison
- Helius credit budget exists; not per-candidate on every scan

## Target State

- Enrichment pipeline triggered for Tier 1 / high-confidence Tier 2 candidates
- Wallet intelligence features attached to candidate record
- Soft BLOCK or WATCH from suspicious flow patterns
- All enrichment logged to SQLite + decision trace

## Key Inputs

- Pair address, recent signatures, Helius enhanced tx payloads
- Tier/consensus context from Layer 2

## Key Outputs

- Enrichment summary for Qwen memo
- `wallet_risk_flags`, `flow_pressure` features (future schema E2)
- Audit events for enrichment success/failure

## Consumers

- Qwen memo ([09](09_llm_reasoning_and_audit_layer.md))
- UI Enrichment Panel ([10](10_ui_and_system_configuration.md))
- Meta-model features (E7, later)

## Open Questions

- Helius credits per scan budget vs candidate volume
- Minimum tx sample size for reliable wallet behavior classification
- Integration point: after economic gate vs after consensus tier only

## Non-Goals

- Helius API calls or Solana RPC probes in Phase E0
- Replacing `whale_score` in `app/engine.py`
- Using whale_score as primary BUY gate
