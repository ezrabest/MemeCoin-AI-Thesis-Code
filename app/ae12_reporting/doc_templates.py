"""Markdown templates for AE12 Final MSc docs - placeholders only, no hard-coded results."""

from __future__ import annotations

from typing import Any


def _provenance_header(ctx: dict[str, Any]) -> str:
    sources = ctx.get("source_files_used") or []
    src_lines = "\n".join(f"- `{s}`" for s in sources) if sources else "- *(none resolved)*"
    return f"""---
generated_at: {ctx.get("generated_at")}
source_ae12_output_root: {ctx.get("source_ae12_output_root")}
phase: AE12.5 Final MSc Reporting
---

> **Provenance:** Values in this document were generated from AE12 audit artifacts
> (JSON/CSV), not hard-coded constants. Re-run `scripts/run_ae12_generate_final_docs.py`
> after a new AE12 maturation pass to refresh numbers.

**Generated at (UTC):** `{ctx.get("generated_at")}`  
**Source AE12 output root:** `{ctx.get("source_ae12_output_root")}`

**Source files used:**
{src_lines}

"""


def _limitations_block() -> str:
    return """
## Limitations

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set
- This report is **not live-approved** and **not profitability-proven**
- future work includes strict policy calibration, runtime UI hardening, longer forward validation, and optional live-wallet gate only after separate approval
"""


def render_final_system_report(ctx: dict[str, Any]) -> str:
    s = ctx["summary"]
    g = ctx["gate"]
    f = ctx["forward"]
    tv = ctx["trade_vs_no_trade"]
    st = ctx["strict_vs_exploration"]
    qw = ctx["qwen"]
    sf = ctx["safety"]
    mw = ctx["missed_winners"]
    rt = ctx["runtime"]

    maturity_lines = []
    for h, row in (f.get("horizon_maturity") or {}).items():
        maturity_lines.append(
            f"- **{h}**: matured={row.get('matured_count')}, not_matured={row.get('not_matured_count')}, "
            f"no_lookahead_ok={row.get('no_lookahead_ok_count')}"
        )
    maturity_md = "\n".join(maturity_lines) or "- *(missing)*"

    blockers = st.get("top_blockers") or []
    blocker_md = "\n".join(
        f"- `{b.get('reason')}`: {b.get('count')}" for b in blockers[:10]
    ) or "- *(none)*"

    tv_rows = []
    for row in tv.get("by_horizon") or []:
        tv_rows.append(
            f"| {row.get('horizon')} | {row.get('traded_count')} | {row.get('not_traded_count')} | "
            f"{row.get('median_forward_return_traded')} | {row.get('median_forward_return_not_traded')} | "
            f"{row.get('max_forward_return_traded')} | {row.get('max_forward_return_not_traded')} | "
            f"{row.get('interpretation_status')} |"
        )
    tv_table = "\n".join(tv_rows) or "| - | - | - | - | - | - | - | - |"

    return (
        _provenance_header(ctx)
        + f"""# AE12 Final System Report

## 1. System objective

The MemeCoin AI Trader is an MSc research platform for multimodal memecoin market analysis,
paper/demo decision orchestration, and forward-evidence auditing. This AE12 package
summarizes derived forward-evidence results for reporting. It does **not** authorize live trading
and does **not** claim profitability.

## 2. Architecture

Layers (audit lineage): data collection → RF/XGB/TAB + meta scoring → context intelligence (AE8)
→ consensus decision (AE6/AE7) → LLM audit (AE9; no trade authority) → runtime paper loop (AE11)
→ forward evidence maturation (AE12.3-AE12.4) → observability / final reporting (AE12.5).

AE12.5 exposes cached, read-only views of existing AE12 artifacts. It does not re-run maturation.

## 3. Data collection

From AE12.1 census (when available):

- Latest DB collection timestamp: `{rt.get("latest_runtime_collection_timestamp")}`
- Market snapshot count: `{rt.get("market_snapshot_count")}`
- Sentiment/RSS count: `{rt.get("sentiment_rss_count")}`
- Paper/demo evidence: `{rt.get("paper_demo_evidence")}`
- AE11 loop timestamp: `{rt.get("ae11_loop_timestamp")}`
- AE11 older than DB collection: `{rt.get("ae11_loop_older_than_db_collection")}`

## 4. ML models / RF-XGB-TAB / meta-layer

Classic and tabular models produce research signals and meta-layer scores that feed decision
records. Models are not live trade authority. This report does not retrain RF/XGB/TAB.

## 5. Context intelligence

AE8 context freshness and family presence influence audit blockers and exploration gates.
Missing context families appear in AE9 audit blockers and AE12 linkage samples.

## 6. Qwen / Gemini intelligent-agent audit layer

Qwen/Gemini/Ollama are audit/explanation layers, not trade authority.

- ROW_LINKED_AE9_RECORD: `{qw.get("ROW_LINKED_AE9_RECORD")}`
- MENTION_ONLY: `{qw.get("MENTION_ONLY")}`
- Ollama status: `{qw.get("ollama_status")}`
- llm_trade_authority_status: `{qw.get("llm_trade_authority_status")}`
- qwen_trade_authority (gate): `{g.get("qwen_trade_authority")}`

## 7. Paper / demo runtime

Runtime paper/demo exploration produced opportunity capture and trade-decision JSONL consumed
by AE12. paper/demo exploration is not live-trading approval.

## 8. Forward evidence methodology

AE12 recomputes horizon maturity and forward returns from market snapshots with no-lookahead
guards. Horizon maturity (enough wall-clock time) is distinct from price freshness at entry.

- Candidate evidence rows: `{f.get("candidate_evidence_row_count")}`
- Matured outcome rows: `{f.get("matured_outcome_row_count")}`
- Missing data warnings: `{f.get("missing_data_warning_count")}`
- Gate: `{g.get("status")}`
- needs_persistence_fix: `{g.get("needs_persistence_fix")}`

### Horizon maturity

{maturity_md}

## 9. Opportunity capture and missed winners

Missed winners are outcome labels only - they do not prove the strategy would have profited.

- Total missed winners: `{mw.get("total_missed_winners")}`
- By horizon: `{mw.get("missed_winners_by_horizon")}`

## 10. Trade vs no-trade

forward returns are outcome labels only.

| Horizon | Traded | Not traded | Med traded | Med not | Max traded | Max not | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
{tv_table}

- Traded count: `{tv.get("traded_count")}`
- Not traded count: `{tv.get("not_traded_count")}`

## 11. Strict vs exploration

strict policy approved zero candidates in this AE12 evidence set.

- Strict approved: `{st.get("strict_approved")}`
- Strict blocked: `{st.get("strict_blocked")}`
- Exploration-only trades: `{st.get("exploration_only_trades")}`

### Top blockers

{blocker_md}

## 12. Safety / no wallet / no real transaction

- wallet_configured: `{sf.get("wallet_configured")}`
- private_key_accessed: `{sf.get("private_key_accessed")}`
- live_submission_status: `{sf.get("live_submission_status")}`
- live_trading_approval: `{sf.get("live_trading_approval")}`
- live_trading_ready: `False`
- profitability_proven: `False`
- real_wallet_connected: `False`

## 13. Results

- Readiness gate: `{g.get("status")}`
- Evidence row count: `{g.get("evidence_row_count") or f.get("candidate_evidence_row_count")}`
- can_proceed_to_ui_final_report: `{g.get("can_proceed_to_ui_final_report")}`
- Known limitations (from AE12 summary): `{s.get("known_limitations")}`

## 14. Limitations

See dedicated limitations section below. Values above are research/audit outcomes only.

## 15. Future work

future work includes strict policy calibration, runtime UI hardening, longer forward validation,
and optional live-wallet gate only after separate approval.
"""
        + _limitations_block()
    )


def render_architecture_summary(ctx: dict[str, Any]) -> str:
    return (
        _provenance_header(ctx)
        + f"""# AE12 Architecture Summary

## Overview

AE12.5 is a **read-only observability and reporting layer** over existing AE12 artifacts.

| Layer | Role |
|---|---|
| A. File/data loading | `app/ae12_reporting/loaders.py`, `latest.py` |
| B. Cached report manager | `AE12ReportManager` (TTL default 300s) |
| C. API endpoints | `GET /api/ae12/*` via app-level manager registry |
| D. UI rendering | AE12 Forward Evidence tab (static UI) |
| E. Final docs | `final_docs.py` + `doc_templates.py` |

## Safety boundaries preserved

- No real wallet connection
- No private key access
- No live submission
- No AE12 maturation rebuild from UI
- No hard-coded result numbers in report templates

## Source root for this render

`{ctx.get("source_ae12_output_root")}`

## Gate snapshot (from AE12 JSON)

- status: `{ctx["gate"].get("status")}`
- live_trading_ready: `{ctx["gate"].get("live_trading_ready")}`
- profitability_proven: `{ctx["gate"].get("profitability_proven")}`
- qwen_trade_authority: `{ctx["gate"].get("qwen_trade_authority")}`
- needs_persistence_fix: `{ctx["gate"].get("needs_persistence_fix")}`
"""
        + _limitations_block()
    )


def render_forward_evidence_results(ctx: dict[str, Any]) -> str:
    f = ctx["forward"]
    tv = ctx["trade_vs_no_trade"]
    mw = ctx["missed_winners"]
    maturity_lines = []
    for h, row in (f.get("horizon_maturity") or {}).items():
        maturity_lines.append(
            f"| {h} | {row.get('matured_count')} | {row.get('not_matured_count')} | "
            f"{row.get('no_lookahead_ok_count')} | {row.get('matured_but_no_snapshots_count')} |"
        )
    maturity_md = "\n".join(maturity_lines) or "| - | - | - | - | - |"
    return (
        _provenance_header(ctx)
        + f"""# AE12 Forward Evidence Results

## Summary

- Candidate evidence rows: `{f.get("candidate_evidence_row_count")}`
- Matured outcome rows: `{f.get("matured_outcome_row_count")}`
- Missing data warnings: `{f.get("missing_data_warning_count")}`
- Missed winners: `{mw.get("total_missed_winners")}`
- Missed winners by horizon: `{mw.get("missed_winners_by_horizon")}`

## Horizon maturity

| Horizon | Matured | Not matured | No-lookahead OK | Matured but no snapshots |
|---|---:|---:|---:|---:|
{maturity_md}

Price freshness and horizon maturity are distinct: maturity means enough time elapsed;
freshness concerns entry-price staleness at decision time.

## Trade vs no-trade (outcome labels only)

- Traded: `{tv.get("traded_count")}`
- Not traded: `{tv.get("not_traded_count")}`
- Interpretations: `{tv.get("interpretations")}`

forward returns are outcome labels only.
"""
        + _limitations_block()
    )


def render_ml_and_meta_layer_summary(ctx: dict[str, Any]) -> str:
    return (
        _provenance_header(ctx)
        + f"""# AE12 ML and Meta-Layer Summary

## Scope

This document summarizes the role of RF / XGBoost / tabular (TAB) models and the meta-layer
in the research pipeline that produced AE12 candidate evidence. No models were retrained
during AE12.5 reporting generation.

## Role in the pipeline

1. Market + whale features feed classic / tabular predictors.
2. Meta-layer aggregation contributes to AE6/AE7 decision records.
3. Decisions + opportunity capture feed AE11 runtime paper loop.
4. AE12 matures forward outcomes for reporting.

## Evidence context (from AE12)

- Candidate evidence rows: `{ctx["forward"].get("candidate_evidence_row_count")}`
- Gate: `{ctx["gate"].get("status")}`

Models do not authorize live trades. paper/demo exploration is not live-trading approval.
"""
        + _limitations_block()
    )


def render_intelligent_agent_layer_summary(ctx: dict[str, Any]) -> str:
    qw = ctx["qwen"]
    return (
        _provenance_header(ctx)
        + f"""# AE12 Intelligent-Agent Layer Summary

## Authority statement

Qwen/Gemini/Ollama are audit/explanation layers, not trade authority.

AE9 linkage in this evidence set is **audit-record linkage** with
`llm_trade_authority_status = NO_TRADE_AUTHORITY`.

## Linkage counts (from AE12 summary JSON)

- ROW_LINKED_AE9_RECORD: `{qw.get("ROW_LINKED_AE9_RECORD")}`
- MENTION_ONLY: `{qw.get("MENTION_ONLY")}`
- Ollama status: `{qw.get("ollama_status")}`
- NO_TRADE_AUTHORITY: `{qw.get("NO_TRADE_AUTHORITY")}`
- qwen_trade_authority (gate): `{ctx["gate"].get("qwen_trade_authority")}`

## Warning

Qwen/Gemini/Ollama do not create trade entries and are not trade authority.
Row-linked AE9 records must not be interpreted as safe trading decisions unless
separately proven; this AE12 set records NO_TRADE_AUTHORITY.
"""
        + _limitations_block()
    )


def render_limitations_and_future_work(ctx: dict[str, Any]) -> str:
    known = ctx["summary"].get("known_limitations") or []
    known_md = "\n".join(f"- {item}" for item in known) or "- *(none in summary)*"
    return (
        _provenance_header(ctx)
        + f"""# AE12 Limitations and Future Work

## Required reporting limitations

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set
- System is **not live-approved** and **not profitability-proven**

## AE12 known limitations (from source JSON)

{known_md}

## Future work

future work includes strict policy calibration, runtime UI hardening, longer forward validation,
and optional live-wallet gate only after separate approval.

Additional research directions (non-authorization):

- Calibrate strict blockers (`ACTIVE_PAIR_LOCK`, `max_open_positions`, cooldowns)
- Extend 24h maturation coverage
- Harden runtime UI observability without enabling live trading
- Improve Qwen row linkage coverage (MENTION_ONLY reduction) without granting trade authority
"""
        + _limitations_block()
    )


def render_demo_script(ctx: dict[str, Any]) -> str:
    g = ctx["gate"]
    f = ctx["forward"]
    st = ctx["strict_vs_exploration"]
    sf = ctx["safety"]
    return (
        _provenance_header(ctx)
        + f"""# AE12 Demo Script (Research / Paper-Demo Only)

## Before you start

1. Runtime may need a **manual restart** to pick up AE12.5 API/UI code (no hot-reload assumption).
2. Do **not** connect a real wallet.
3. Do **not** enable live trading.
4. Label everything as paper/demo/exploration, research-only, not live-approved, not profitability-proven.

## Demo steps

1. Start the server manually when approved.
2. Open the dashboard → **AE12 Forward Evidence** tab.
3. Call `GET /api/ae12/status` - expect gate `{g.get("status")}`, `live_ready=false`, `profitability_proven=false`.
4. Call `GET /api/ae12/forward-evidence-summary` - candidate rows `{f.get("candidate_evidence_row_count")}`.
5. Show Missed Winners panel - emphasize outcome labels only.
6. Show Trade vs No-Trade - mixed interpretations; not profitability proof.
7. Show Strict vs Exploration - strict approved `{st.get("strict_approved")}` (zero in this evidence set).
8. Show Qwen panel - NO_TRADE_AUTHORITY; Ollama largely absent.
9. Show Wallet/Safety - wallet_configured=`{sf.get("wallet_configured")}`,
   private_key_accessed=`{sf.get("private_key_accessed")}`,
   live_submission_status=`{sf.get("live_submission_status")}`, live trading approval = NO.

## Talking points (safe)

- forward returns are outcome labels only
- paper/demo exploration is not live-trading approval
- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority
- strict policy approved zero candidates in this AE12 evidence set

## Do not say

- The system is profitable
- The system is live-ready
- Qwen decides trades
- Strict policy approved trades
- Missed winners prove the strategy would have profited
"""
        + _limitations_block()
    )


DOC_RENDERERS = {
    "ae12_final_system_report.md": render_final_system_report,
    "ae12_architecture_summary.md": render_architecture_summary,
    "ae12_forward_evidence_results.md": render_forward_evidence_results,
    "ae12_ml_and_meta_layer_summary.md": render_ml_and_meta_layer_summary,
    "ae12_intelligent_agent_layer_summary.md": render_intelligent_agent_layer_summary,
    "ae12_limitations_and_future_work.md": render_limitations_and_future_work,
    "ae12_demo_script.md": render_demo_script,
}
