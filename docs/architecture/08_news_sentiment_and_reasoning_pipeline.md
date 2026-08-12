# 08 — News Sentiment and Reasoning Pipeline

## Purpose

Formalize RSS/news sentiment as **context intelligence** — feeding enrichment, Qwen reasoning, Gemini audit, UI explanation, and decision trace. RSS is **not** a trading model.

## Diagram

See [diagrams/news_sentiment_reasoning_pipeline.mmd](diagrams/news_sentiment_reasoning_pipeline.mmd).

```text
Cointelegraph RSS / Decrypt RSS
        ↓
raw RSS payload archive
        ↓
headline/article normalization
        ↓
sentiment scoring
        ↓
risk-keyword extraction
        ↓
token/narrative matching
        ↓
candidate sentiment context
        ↓
Qwen reasoning memo
        ↓
optional Gemini audit
        ↓
PAPER_BUY / WATCH / BLOCK explanation
```

## Core Principle

```text
RSS sentiment is not a trading model.
RSS sentiment is a context-intelligence input that feeds candidate enrichment, Qwen reasoning, Gemini audit, UI explanation, and final decision trace.
```

## News Sources

| Source | Status |
|--------|--------|
| Cointelegraph RSS | **Active** — default feed in `app/analytics/sentiment.py` |
| Decrypt RSS | **Active** — default feed |
| Future optional sources | Architecture slot; override via `RSS_FEED_URL` env |

## Current Implementation

| Step | Location |
|------|----------|
| Fetch + parse | `app/analytics/sentiment.py` |
| Lexicon score [-1, 1] | Per headline + aggregate |
| Raw archive | `raw_provider_payloads` via `archive_rss_sentiment()` |
| Persistence | `sentiment_records` table |
| Scan integration | `app/live.py` — each scan cycle |
| API | `GET /api/sentiment/matrix` |
| UI | Dashboard sentiment sidebar in `static/index.html` |

## Sentiment Outputs (Target Schema)

| Field | Description |
|-------|-------------|
| `market_sentiment_score` | Aggregate RSS score for scan cycle |
| `candidate_sentiment_score` | Token/narrative-matched score |
| `negative_news_flag` | Broad negative market tone |
| `scam_risk_language_flag` | Rug/hack/fraud language detected |
| `rug_hack_fraud_warning_flags` | Specific risk keyword hits |
| `narrative_strength` | Match confidence to candidate narrative |
| `source_reliability` | Per-source weight (future) |
| `sentiment_context_for_llm` | Structured bundle for Qwen/Gemini prompts |

## Correct Usage Examples

**Positive narrative supports Tier 1:**

```text
Tier 1 candidate + positive/neutral sentiment + no warning language
→ confidence support
```

**Warning language triggers review:**

```text
Tier 1 candidate + scam/rug/hack warning language
→ WATCH/BLOCK or Qwen/Gemini review
```

**Narrative alone cannot BUY:**

```text
XGB broad candidate + weak consensus + strong market narrative
→ WATCH, not automatic BUY
```

## Incorrect Usage

```text
RSS positive → BUY
```

Never map RSS score directly to execution.

## Target State

- Per-candidate narrative matching (not just market-wide aggregate)
- Risk-keyword extraction pipeline separate from lexicon score
- `sentiment_context_for_llm` in Qwen memo input
- UI sentiment panel shows candidate-level context
- All sentiment inputs in decision trace JSONL

## Key Inputs

- Cointelegraph/Decrypt RSS XML
- Candidate token symbol, name, narrative tags

## Key Outputs

- Sentiment context object for Layer 4
- Flags for WATCH/BLOCK escalation
- Archived raw payloads for audit

## Consumers

- Qwen memo generator, Gemini audit, UI Enrichment Panel, decision trace

## Open Questions

- Token/narrative matching algorithm (symbol vs fuzzy headline match)
- Staleness window for RSS items
- Whether to add more crypto news feeds without diluting signal

## Non-Goals

- Changing RSS fetch logic or lexicon in Phase E0
- Using sentiment as standalone model score
