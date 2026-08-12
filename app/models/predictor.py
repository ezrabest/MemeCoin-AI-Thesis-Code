"""
Multimodal LLM inference engine — Gemini 2.5 Flash (profit-max autonomous agent).
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..analytics.features import ClusterLabel
from ..execution.paper import get_paper_trader
from ..llm_client import OLLAMA_PROVIDER_TAG, generate_decision
from ..llm_config import (
    OLLAMA_FALLBACK_REASON,
    SKIP_REASON,
    SKIP_REASON_BUDGET,
    SKIP_REASON_NONE,
    build_llm_authority_boundary,
    get_llm_provider,
    is_gemini_provider_active,
    is_headless_data_collection,
    is_llm_audit_only_provider,
    is_ollama_provider_active,
    normalize_llm_provider_name,
    record_gemini_call,
    record_llm_skipped,
    record_ollama_skipped,
    record_scan_llm_decision_stored,
    try_consume_ollama_call,
)

log = logging.getLogger("predictor")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DATA_DIR = Path(__file__).parent.parent.parent / "data"
WHALE_LOG_PATH = DATA_DIR / "whale_trades_log.csv"
DECISIONS_LOG_PATH = DATA_DIR / "llm_decisions_log.csv"
SETTINGS_PATH = DATA_DIR / "settings.json"

DECISION_FIELDS = [
    "timestamp",
    "symbol",
    "cluster_label",
    "decision",
    "strategy_type",
    "risk_score",
    "confidence",
    "reasoning",
    "position_id",
]


def _ensure_decision_csv_header() -> list[str]:
    if not DECISIONS_LOG_PATH.exists() or DECISIONS_LOG_PATH.stat().st_size == 0:
        return list(DECISION_FIELDS)

    with open(DECISIONS_LOG_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [field for field in DECISION_FIELDS if field not in existing]
    if not missing:
        return list(DECISION_FIELDS)

    merged_fields = existing + missing
    with open(DECISIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return merged_fields

CLUSTER_PERSONA = """You classify token behavioral identity for an academic trading thesis."""

TRADING_AGENT_SKILL = """
## Trading Agent Framework Skill (Quantitative)

Apply these principles on every decision:

1. **EMA momentum** — Compute mental EMA(3) vs EMA(10) on the 10-row price velocity series.
   BUY bias when short EMA > long EMA and slope is positive; SELL when crossing down.

2. **Volume exhaustion** — If volume_24h rises but price_change_1h stalls or buy_ratio falls across
   the last 3 snapshots, distribution is likely — avoid new BUYs; favor SELL on open positions.

3. **Whale score momentum** — Rising whale_score across the series confirms accumulation (WHALE_RIDER).
   Spiking whale_score with flat price favors SCALPING_OPPORTUNITY entry.

4. **Fee drag math (mandatory)** — Round-trip cost ≈ 3.0% (1.5% entry + 1.5% exit on notional).
   Minimum gross move to break even: gain_pct > 3.03% / (1 - 0.015) ≈ 3.08%.
   Only BUY when expected move exceeds fee drag + 0.5% margin on tight spreads.

5. **Strategy selection** — Set strategy_type strictly:
   - SCALPING_OPPORTUNITY: quick in-and-out spikes, +1h velocity, moderate whale activity.
   - WHALE_RIDER: sustained whale_score momentum, high buy_ratio, accumulation waves.
"""

TRADING_SYSTEM_PROMPT = """You are a high-frequency, ultra-aggressive crypto speculator operating in Demo Mode.
Your ONLY goal is to maximize short-term Net ROI. Default to action, not caution.

Hard rules — follow exactly:
1. If a coin has a positive 1h price trend (price_change_1h > 0) AND confirmed whale activity
   (whale_score meets or exceeds the system threshold), you MUST issue a BUY decision immediately
   to secure short-term ROI. Do not stay flat.
2. HOLD is ONLY allowed when 1h momentum is flat/negative AND whale_score is weak AND no open
   position requires management.
3. Issue SELL to lock profits or cut losses on open positions when momentum reverses.
4. Never refuse BUY due to RSS sentiment, cluster label, or generic risk aversion alone.

Combine live metrics with your global knowledge (meme cycles, narratives, virality) but bias
heavily toward entering pumps early. Fees are acceptable; missing pumps is not."""

StrategyType = Literal["SCALPING_OPPORTUNITY", "WHALE_RIDER"]


class TradeDecision(BaseModel):
    decision: Literal["BUY", "SELL", "HOLD"]
    strategy_type: StrategyType = "SCALPING_OPPORTUNITY"
    risk_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    position_id: int | None = None
    symbol: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("strategy_type", mode="before")
    @classmethod
    def normalize_strategy(cls, v: str) -> str:
        raw = str(v or "").upper()
        if "WHALE" in raw or "RIDER" in raw:
            return "WHALE_RIDER"
        return "SCALPING_OPPORTUNITY"


def _get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _configure_genai() -> bool:
    if not is_gemini_provider_active():
        return False
    key = _get_api_key()
    if not key:
        return False
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        return True
    except Exception as exc:
        log.warning("Gemini configure failed: %s", exc)
        return False


def _default_decision(reason: str) -> TradeDecision:
    return TradeDecision(
        decision="HOLD",
        strategy_type="SCALPING_OPPORTUNITY",
        risk_score=50,
        confidence=0.0,
        reasoning=reason,
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _gemini_json(prompt: str, *, temperature: float = 0.35) -> dict[str, Any]:
    if not is_gemini_provider_active():
        raise RuntimeError("Gemini provider not active")
    import google.generativeai as genai

    record_gemini_call()
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )
    response = model.generate_content(prompt)
    return _parse_json_response(response.text or "{}")


async def classify_token_cluster(
    *,
    symbol: str,
    name: str,
    network: str,
    contract_address: str,
    description: str = "",
) -> tuple[ClusterLabel, str]:
    """
    Semantic behavioral cluster from token identity — evaluated once at discovery.
    """
    if is_headless_data_collection() or get_llm_provider() == "none":
        log.info("LLM evaluation skipped - Headless Data Collection mode active")
        record_llm_skipped()
        reason = SKIP_REASON if is_headless_data_collection() else SKIP_REASON_NONE
        return ClusterLabel.OPPORTUNISTIC_SPECULATIVE, reason

    if is_ollama_provider_active():
        return (
            ClusterLabel.OPPORTUNISTIC_SPECULATIVE,
            "Cluster default — Ollama provider skips cluster LLM",
        )

    if not _configure_genai():
        return ClusterLabel.OPPORTUNISTIC_SPECULATIVE, "GEMINI_API_KEY not configured — default speculative"

    prompt = f"""{CLUSTER_PERSONA}

Classify this token's BEHAVIORAL identity (not current volume) into exactly one label.
This label is permanent for the token's lifetime in our system.

OPPORTUNISTIC_SPECULATIVE: Tokens explicitly launched for speculative hype, memes with zero
utility, parody coins, or satirical scams (e.g. coins like 'STARTUP' mocking taking people's money).

SOCIALLY_MOTIVATED: Tokens backed by genuine community initiatives, charity fundraisers,
environmental causes, or long-term ecosystem building.

Token identity:
- symbol: {symbol}
- name: {name}
- network: {network}
- contract_address: {contract_address}
- description: {description or name}

Return ONLY JSON:
{{"cluster_label": "OPPORTUNISTIC_SPECULATIVE" or "SOCIALLY_MOTIVATED", "reasoning": "brief string"}}
"""

    def _call() -> tuple[ClusterLabel, str]:
        data = _gemini_json(prompt, temperature=0.15)
        raw = str(data.get("cluster_label", "")).upper()
        if "SOCIAL" in raw:
            label = ClusterLabel.SOCIALLY_MOTIVATED
        else:
            label = ClusterLabel.OPPORTUNISTIC_SPECULATIVE
        return label, str(data.get("reasoning", ""))

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as exc:
        log.warning("classify_token_cluster failed: %s", exc)
        return ClusterLabel.OPPORTUNISTIC_SPECULATIVE, f"Classification failed: {exc}"


def _build_analysis_context(
    metrics: dict[str, Any],
    *,
    coin_id: int | None = None,
) -> tuple[dict[str, Any], str]:
    from ..gemini_context import build_gemini_context, context_prompt_summary

    if coin_id is not None:
        historical_context = build_gemini_context(coin_id)
        prompt_summary = context_prompt_summary(historical_context)
    else:
        contract = str(metrics.get("token_contract_address") or "")
        symbol = str(metrics.get("symbol") or "")
        timeseries = get_whale_timeseries(contract, symbol=symbol, limit=10)
        historical_context = {"legacy_timeseries": timeseries, "note": "no coin_id — limited context"}
        prompt_summary = "legacy path"
    return historical_context, prompt_summary


def _trade_decision_from_ollama(parsed: dict[str, Any], metrics: dict[str, Any]) -> TradeDecision:
    action = str(parsed.get("action", "HOLD")).upper()
    if action == "SKIPPED":
        action = "HOLD"

    strategy_raw = str(parsed.get("strategy_type", "UNKNOWN")).upper()
    if strategy_raw in ("WHALE_FLOW", "RISK_OFF"):
        strategy_type: StrategyType = "WHALE_RIDER"
    else:
        strategy_type = "SCALPING_OPPORTUNITY"

    risk_level = str(parsed.get("risk_level", "MEDIUM")).upper()
    risk_score = {"LOW": 30, "MEDIUM": 50, "HIGH": 80}.get(risk_level, 50)

    return TradeDecision(
        decision=action if action in ("BUY", "SELL", "HOLD") else "HOLD",
        strategy_type=strategy_type,
        risk_score=risk_score,
        confidence=float(parsed.get("confidence", 0.0)),
        reasoning=str(parsed.get("rationale") or OLLAMA_FALLBACK_REASON),
        symbol=metrics.get("symbol"),
    )


def _build_market_prompt(
    metrics: dict[str, Any],
    historical_context: dict[str, Any],
    cluster_label: str,
    sentiment_score: float,
    trigger_type: str,
    *,
    open_positions: list[dict[str, Any]] | None = None,
) -> str:
    settings = _load_settings()
    extra = settings.get("prompt_extra", "")
    behavior = settings.get("prompt_behavior", "aggressive")
    llm_threshold = float(settings.get("llm_score_threshold", 0.30))
    positions = open_positions if open_positions is not None else get_paper_trader().get_positions("OPEN")
    wallet = get_paper_trader().get_wallet_summary()

    return f"""{TRADING_SYSTEM_PROMPT}

{TRADING_AGENT_SKILL}

Trading style setting: {behavior}.
{extra}

IMPORTANT — Anti-churn memory (read carefully):
{json.dumps(historical_context.get("churn_guard", {}), indent=2)}

Prior LLM decisions and app paper trades for this token are in historical_context.
Do NOT re-BUY immediately after a recent SELL unless aggregate whale-like flow shows
a materially new setup. Fee round-trip ≈ 3% — avoid churn on noise.

Live market metrics (current snapshot):
{json.dumps(metrics, indent=2)}

Structured historical context (SQLite memory — compact JSON):
{json.dumps(historical_context, indent=2)}

Persistent behavioral cluster: {cluster_label}
RSS sentiment (-1 bearish to +1 bullish): {sentiment_score}
Whale activity threshold: {llm_threshold}
Demo wallet equity: ${wallet.get("total_equity_usd", 0):,.2f} (SCALP=10%, WHALE_RIDER=30% allocation)
Trigger: {trigger_type} (aggregate pool-level whale-like flow — NOT wallet-level unless flagged)

Open demo positions (manage exits actively):
{json.dumps(positions, indent=2)}
"""


async def log_skipped_llm_decision(
    symbol: str,
    cluster_label: str,
    *,
    coin_id: int | None = None,
    input_context: dict[str, Any] | None = None,
    prompt_summary: str = "",
    trigger_type: str = "",
    pair_address: str = "",
    reason: str | None = None,
) -> int | None:
    """Persist an explicit SKIPPED decision — no external LLM API call."""
    from .. import database as db

    skip_reason = reason or SKIP_REASON
    record_llm_skipped()
    log.info("%s", skip_reason)

    ts = datetime.now(timezone.utc).isoformat()
    context = dict(input_context or {})
    if pair_address and "pair_address" not in context:
        context["pair_address"] = pair_address
    if coin_id is not None and "coin_id" not in context:
        context["coin_id"] = coin_id

    row = {
        "timestamp": ts,
        "symbol": symbol,
        "cluster_label": cluster_label,
        "decision": "SKIPPED",
        "strategy_type": "",
        "risk_score": 0,
        "confidence": 0.0,
        "reasoning": skip_reason,
        "position_id": "",
    }

    def _write_csv() -> None:
        try:
            fields = _ensure_decision_csv_header()
            exists = DECISIONS_LOG_PATH.exists() and DECISIONS_LOG_PATH.stat().st_size > 0
            with open(DECISIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow({field: row.get(field, "") for field in fields})
        except Exception as exc:
            log.warning("Skipped LLM decision CSV write failed: %s", exc)

    await asyncio.get_event_loop().run_in_executor(None, _write_csv)

    try:
        decision_id = db.insert_gemini_decision({
            "timestamp": ts,
            "coin_id": coin_id,
            "symbol": symbol,
            "prompt_summary": prompt_summary or f"{symbol} SKIPPED",
            "input_context_json": context,
            "gemini_response_json": {"skipped": True, "reason": skip_reason},
            "action": "SKIPPED",
            "confidence": 0.0,
            "rationale": skip_reason,
            "strategy_type": "",
            "risk_score": 0,
            "trigger_type": trigger_type,
            "provider": get_llm_provider() if get_llm_provider() != "gemini" else "none",
            "model_source": "skipped",
        })
    except Exception as exc:
        log.warning("Skipped LLM decision SQLite insert failed: %s", exc)
        return None

    log.info("llm decisions skipped stored: 1 (id=%s symbol=%s)", decision_id, symbol)
    return decision_id


async def analyze_market_state(
    metrics: dict[str, Any],
    cluster_label: str,
    sentiment_score: float,
    *,
    open_positions: list[dict[str, Any]] | None = None,
    coin_id: int | None = None,
    trigger_type: str = "whale_like_event",
) -> tuple[TradeDecision, int | None]:
    """
    Fuse live metrics + SQLite historical memory for ONE coin (via build_gemini_context).
    Only call when a high-conviction whale-like event fires — not for every scanned token.
    """
    historical_context, prompt_summary = _build_analysis_context(metrics, coin_id=coin_id)
    symbol = str(metrics.get("symbol") or "?")
    pair_address = str(metrics.get("token_contract_address") or historical_context.get("pair_address") or "")
    llm_threshold = float(_load_settings().get("llm_score_threshold", 0.30))

    if is_headless_data_collection():
        decision_id = await log_skipped_llm_decision(
            symbol,
            cluster_label,
            coin_id=coin_id,
            input_context=historical_context,
            prompt_summary=prompt_summary,
            trigger_type=trigger_type,
            pair_address=pair_address,
            reason=SKIP_REASON,
        )
        return _default_decision(SKIP_REASON), decision_id

    provider = get_llm_provider()
    if provider == "none":
        decision_id = await log_skipped_llm_decision(
            symbol,
            cluster_label,
            coin_id=coin_id,
            input_context=historical_context,
            prompt_summary=prompt_summary,
            trigger_type=trigger_type,
            pair_address=pair_address,
            reason=SKIP_REASON_NONE,
        )
        return _default_decision(SKIP_REASON_NONE), decision_id

    prompt = _build_market_prompt(
        metrics,
        historical_context,
        cluster_label,
        sentiment_score,
        trigger_type,
        open_positions=open_positions,
    )

    if provider == "ollama":
        if not try_consume_ollama_call():
            record_ollama_skipped()
            decision_id = await log_skipped_llm_decision(
                symbol,
                cluster_label,
                coin_id=coin_id,
                input_context=historical_context,
                prompt_summary=prompt_summary,
                trigger_type=trigger_type,
                pair_address=pair_address,
                reason=SKIP_REASON_BUDGET,
            )
            return _default_decision(SKIP_REASON_BUDGET), decision_id

        parsed = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_decision(prompt, historical_context),
        )
        if parsed.get("action") == "SKIPPED":
            rationale = str(parsed.get("rationale") or OLLAMA_FALLBACK_REASON)
            decision_id = await log_decision(
                symbol,
                cluster_label,
                _default_decision(rationale),
                coin_id=coin_id,
                input_context=historical_context,
                prompt_summary=prompt_summary,
                trigger_type=trigger_type,
                raw_prompt=prompt,
                provider="ollama",
                model_source=OLLAMA_PROVIDER_TAG,
                response_payload=parsed,
                stored_action="SKIPPED",
            )
            return _default_decision(rationale), decision_id

        decision = apply_aggressive_buy_override(
            metrics,
            _trade_decision_from_ollama(parsed, metrics),
            llm_threshold,
        )
        decision_id = await log_decision(
            symbol,
            cluster_label,
            decision,
            coin_id=coin_id,
            input_context=historical_context,
            prompt_summary=prompt_summary,
            trigger_type=trigger_type,
            raw_prompt=prompt,
            provider="ollama",
            model_source=OLLAMA_PROVIDER_TAG,
            response_payload=parsed,
        )
        return decision, decision_id

    if not is_gemini_provider_active() or not _configure_genai():
        return _default_decision("GEMINI_API_KEY not configured — defaulting to HOLD"), None

    gemini_prompt = prompt + """

Return ONLY valid JSON:
{"decision": "BUY" or "SELL" or "HOLD",
 "strategy_type": "SCALPING_OPPORTUNITY" or "WHALE_RIDER",
 "risk_score": 0-100 integer, "confidence": 0.0-1.0 float,
 "reasoning": "brief string", "position_id": null or integer (required for SELL when multiple open),
 "symbol": null or string (pair symbol for SELL/BUY target)}
"""

    def _call() -> TradeDecision:
        data = _gemini_json(gemini_prompt, temperature=0.55)
        decision = TradeDecision.model_validate(data)
        return apply_aggressive_buy_override(metrics, decision, llm_threshold)

    try:
        decision = await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as exc:
        log.warning("analyze_market_state failed: %s", exc)
        fallback = _default_decision(f"LLM inference failed: {exc}")
        decision = apply_aggressive_buy_override(metrics, fallback, llm_threshold)

    decision_id = await log_decision(
        symbol,
        cluster_label,
        decision,
        coin_id=coin_id,
        input_context=historical_context,
        prompt_summary=prompt_summary,
        trigger_type=trigger_type,
        raw_prompt=gemini_prompt,
        provider="gemini",
        model_source=MODEL_NAME,
        response_payload=decision.model_dump(),
    )
    return decision, decision_id


def get_whale_timeseries(
    contract_address: str,
    *,
    symbol: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Last N whale_trades_log.csv rows for a token — price velocity, buy_ratio, whale momentum.
    """
    rows = _read_whale_rows(limit=2000)
    addr = contract_address.strip()
    matched = [r for r in rows if (r.get("token_contract_address") or "").strip() == addr]
    if not matched and symbol:
        base = str(symbol).split("/")[0].upper()
        matched = [
            r for r in rows
            if (r.get("symbol") or "").upper() == base
            or (r.get("symbol") or "").upper().startswith(base)
        ]
    series: list[dict[str, Any]] = []
    for r in matched[-limit:]:
        try:
            series.append({
                "timestamp": r.get("timestamp"),
                "price_usd": round(float(r.get("price_usd_at_trade") or 0), 8),
                "buy_ratio": round(float(r.get("buy_ratio") or 0), 4),
                "whale_score": round(float(r.get("whale_score") or 0), 4),
                "price_change_24h": round(float(r.get("price_change_24h") or 0), 4),
                "volume_24h": round(float(r.get("volume_24h_at_trade") or 0), 2),
                "liquidity_usd": round(float(r.get("liquidity_usd_at_trade") or 0), 2),
            })
        except (TypeError, ValueError):
            continue
    return series


async def analyze_open_position(
    position: dict[str, Any],
    cur_price: float,
    cluster_label: str,
    sentiment_score: float,
    *,
    coin_id: int | None = None,
) -> tuple[TradeDecision, int | None]:
    """Focused exit analysis for an active paper position."""
    metrics = {
        "symbol": position.get("symbol"),
        "token_contract_address": position.get("pair_address", ""),
        "position_id": position.get("id"),
        "entry_price": position.get("entry_price"),
        "current_price": cur_price,
        "strategy_type": position.get("strategy_type", "SCALPING_OPPORTUNITY"),
        "unrealized_pct": round(
            (cur_price - float(position.get("entry_price", cur_price)))
            / max(float(position.get("entry_price", 1)), 1e-12)
            * 100,
            2,
        ),
        "cluster_label": cluster_label,
    }
    decision, decision_id = await analyze_market_state(
        metrics,
        cluster_label,
        sentiment_score,
        open_positions=[position],
        coin_id=coin_id,
        trigger_type="open_position_exit",
    )
    if decision.decision == "SELL" and decision.position_id is None:
        decision.position_id = int(position["id"])
    return decision, decision_id


def normalize_execution_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize position sizing and coerce canonical numeric settings."""
    from app.observability.settings_normalize import normalize_canonical_settings

    return normalize_canonical_settings(dict(settings))


def apply_aggressive_buy_override(
    metrics: dict[str, Any],
    decision: TradeDecision,
    llm_threshold: float,
) -> TradeDecision:
    """Force BUY when +1h trend and whale activity confirm — never stay flat."""
    pc1h = float(metrics.get("price_change_1h") or 0)
    whale_score = float(metrics.get("whale_score") or 0)
    if pc1h > 0 and whale_score >= llm_threshold and decision.decision != "SELL":
        if decision.decision != "BUY":
            strategy: StrategyType = (
                "WHALE_RIDER" if whale_score >= 0.55 else "SCALPING_OPPORTUNITY"
            )
            return TradeDecision(
                decision="BUY",
                strategy_type=strategy,
                risk_score=max(decision.risk_score, 70),
                confidence=max(decision.confidence, 0.75),
                reasoning=(
                    f"Aggressive override: +1h {pc1h:.2f}% with whale_score "
                    f"{whale_score:.3f} ≥ {llm_threshold} — mandatory BUY ({strategy})."
                ),
                symbol=decision.symbol or metrics.get("symbol"),
            )
    return decision


def execute_trade_decision(
    decision: TradeDecision,
    coin: dict[str, Any],
    cluster_label: str,
    settings: dict[str, Any],
    *,
    cur_price: float | None = None,
    decision_ref_id: int | None = None,
    coin_id: int | None = None,
    pair_address: str | None = None,
    position_id: int | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Route a trade decision into paper execution.

    AE19: gemini/qwen/ollama decisions are audit/shadow only — never execute.
    """
    symbol = str(coin.get("symbol") or decision.symbol or "?")
    resolved_provider = normalize_llm_provider_name(provider) if provider else get_llm_provider()
    if is_llm_audit_only_provider(resolved_provider):
        action = decision.decision
        log.info(
            "LLM audit-only decision stored; execution not attempted provider=%s action=%s",
            resolved_provider,
            action,
        )
        return {
            "ok": True,
            "decision": action,
            "symbol": symbol,
            "execution_attempted": False,
            "audit_only": True,
            "reason": "LLM_PROVIDER_AUDIT_ONLY",
            "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
            "provider": resolved_provider,
        }

    trader = get_paper_trader()
    exec_settings = normalize_execution_settings(settings)
    coin_payload = {**coin, "coin_id": coin_id or coin.get("coin_id"), "decision_ref_id": decision_ref_id}

    if decision.decision == "BUY":
        wallet = trader.get_wallet_summary()
        size_usd = trader.compute_strategy_notional(decision.strategy_type)
        pos = trader.try_autonomous_buy(
            coin_payload,
            cluster_label,
            exec_settings,
            risk_score=decision.risk_score,
            strategy_type=decision.strategy_type,
            size_usd=size_usd,
        )
        if pos is None and size_usd >= 10.0:
            pos = trader.open_position(
                coin_payload,
                size_usd=round(size_usd, 2),
                cluster_label=cluster_label,
                settings=exec_settings,
                reason_code=f"AGENT_BUY_{decision.strategy_type}",
                strategy_type=decision.strategy_type,
            )
        if pos is None:
            reason = (
                f"BUY blocked for {symbol}: strategy={decision.strategy_type} "
                f"size=${size_usd:,.2f} auto_execution={exec_settings.get('auto_execution_enabled')}, "
                f"mode={wallet.get('trading_mode')}, cash=${wallet.get('cash_usd', 0):,.2f}"
            )
            log.error("EXEC_FAIL %s", reason)
            return {
                "ok": False,
                "decision": "BUY",
                "strategy_type": decision.strategy_type,
                "size_usd": size_usd,
                "error": reason,
                "symbol": symbol,
            }
        log.info(
            "EXEC_OK AGENT_BUY #%s %s strategy=%s size=$%.2f",
            pos["id"],
            symbol,
            decision.strategy_type,
            pos.get("size_usd", 0),
        )
        return {
            "ok": True,
            "decision": "BUY",
            "strategy_type": decision.strategy_type,
            "size_usd": pos.get("size_usd"),
            "position": pos,
            "wallet": trader.get_wallet_summary(),
        }

    if decision.decision == "SELL":
        closed = trader.try_autonomous_sell(
            symbol=decision.symbol or symbol,
            position_id=decision.position_id or position_id,
            pair_address=pair_address or coin.get("pair_address"),
            coin_id=coin_id or coin.get("coin_id"),
            cur_price=cur_price,
            settings=exec_settings,
        )
        if closed is None:
            reason = f"SELL blocked for {symbol}: no matching open position or auto_execution off"
            log.warning("EXEC_FAIL %s", reason)
            return {"ok": False, "decision": "SELL", "error": reason, "symbol": symbol}
        log.info("EXEC_OK AGENT_SELL #%s %s", closed.get("id"), symbol)
        return {"ok": True, "decision": "SELL", "position": closed, "wallet": trader.get_wallet_summary()}

    return {"ok": True, "decision": "HOLD", "symbol": symbol}


async def log_decision(
    symbol: str,
    cluster_label: str,
    decision: TradeDecision,
    *,
    coin_id: int | None = None,
    input_context: dict[str, Any] | None = None,
    prompt_summary: str = "",
    trigger_type: str = "",
    raw_prompt: str = "",
    provider: str | None = None,
    model_source: str | None = None,
    response_payload: dict[str, Any] | None = None,
    stored_action: str | None = None,
) -> int | None:
    """Append to CSV audit log and SQLite gemini_decisions table."""
    from .. import database as db

    ts = datetime.now(timezone.utc).isoformat()
    action_label = stored_action or decision.decision
    stored_strategy = (response_payload or {}).get("strategy_type") or decision.strategy_type
    row = {
        "timestamp": ts,
        "symbol": symbol,
        "cluster_label": cluster_label,
        "decision": action_label,
        "strategy_type": stored_strategy,
        "risk_score": decision.risk_score,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "position_id": decision.position_id or "",
    }

    def _write_csv() -> None:
        try:
            fields = _ensure_decision_csv_header()
            exists = DECISIONS_LOG_PATH.exists() and DECISIONS_LOG_PATH.stat().st_size > 0
            with open(DECISIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow({field: row.get(field, "") for field in fields})
        except Exception as exc:
            log.warning("Gemini decision CSV write failed: %s", exc)

    await asyncio.get_event_loop().run_in_executor(None, _write_csv)

    response_json = response_payload if response_payload is not None else decision.model_dump()
    if not isinstance(response_json, dict):
        response_json = {"raw_response": response_json}
    else:
        response_json = dict(response_json)

    raw_provider = provider or "gemini"
    provider_normalized = normalize_llm_provider_name(raw_provider)
    input_context_payload = dict(input_context or {})
    if is_llm_audit_only_provider(provider_normalized):
        authority = build_llm_authority_boundary(execution_attempted=False)
        response_json.update(authority)
        # Preserve model output labels while marking non-executable authority.
        response_json.setdefault("model_output_action", action_label)
        response_json["execution_authority"] = False
        input_context_payload = {**input_context_payload, **authority}

    if raw_prompt:
        try:
            db.insert_raw_payload(
                provider=raw_provider,
                payload={"prompt": raw_prompt[:8000], "response": response_json},
                source_type="decision_pair",
                symbol=symbol,
            )
        except Exception as exc:
            log.warning("LLM raw payload archival failed: %s", exc)

    try:
        decision_id = db.insert_gemini_decision({
            "timestamp": ts,
            "coin_id": coin_id,
            "symbol": symbol,
            "prompt_summary": prompt_summary or f"{symbol} {decision.decision}",
            "input_context_json": input_context_payload,
            "gemini_response_json": response_json,
            "action": action_label,
            "confidence": decision.confidence,
            "rationale": decision.reasoning,
            "strategy_type": stored_strategy,
            "risk_score": decision.risk_score,
            "trigger_type": trigger_type,
            "provider": provider,
            "model_source": model_source,
        })
    except Exception as exc:
        log.warning("LLM decision SQLite insert failed: %s", exc)
        return None
    record_scan_llm_decision_stored(provider_normalized or provider)
    log.info(
        "llm decisions stored: 1 (id=%s symbol=%s action=%s provider=%s)",
        decision_id,
        symbol,
        action_label,
        provider or "gemini",
    )
    return decision_id


# ── CSV analytics for chat grounding ─────────────────────────────────────────

def _read_whale_rows(limit: int = 500) -> list[dict[str, str]]:
    if not WHALE_LOG_PATH.exists():
        return []
    with open(WHALE_LOG_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:]


def read_whale_log_rows(limit: int = 500) -> list[dict[str, str]]:
    """Public accessor for whale_trades_log.csv rows."""
    return _read_whale_rows(limit)


def avg_whale_score() -> float | None:
    rows = _read_whale_rows()
    if not rows:
        return None
    scores = [float(r["whale_score"]) for r in rows if r.get("whale_score")]
    return round(sum(scores) / len(scores), 4) if scores else None


def count_by_cluster() -> dict[str, int]:
    """Legacy cluster counts for analytics.

    Prefer authoritative DB/registry totals (paper_trades + cluster_registry) so
    UI/API are not truncated to the last 500 whale-log rows (which previously
    hid SOCIALLY_MOTIVATED and made optional coin cluster aggregations look empty).
    """
    try:
        from app.semantic.social_opportunistic_classifier import get_authoritative_semantic_counts

        auth = get_authoritative_semantic_counts()
        return {
            "SOCIALLY_MOTIVATED": int(auth.get("legacy_socially_motivated_count") or 0),
            "OPPORTUNISTIC_SPECULATIVE": int(auth.get("legacy_opportunistic_speculative_count") or 0),
        }
    except Exception:
        rows = _read_whale_rows()
        counts: dict[str, int] = {}
        for r in rows:
            label = r.get("cluster_label") or "UNKNOWN"
            counts[label] = counts.get(label, 0) + 1
        return counts


def avg_whale_score_by_cluster(cluster_label: str) -> float | None:
    rows = [
        r
        for r in _read_whale_rows()
        if (r.get("cluster_label") or "").upper() == cluster_label.upper()
    ]
    if not rows:
        return None
    scores = [float(r["whale_score"]) for r in rows if r.get("whale_score")]
    return round(sum(scores) / len(scores), 4) if scores else None


def latest_metrics(contract: str | None = None) -> dict[str, Any] | None:
    rows = _read_whale_rows()
    if not rows:
        return None
    if contract:
        rows = [r for r in rows if r.get("token_contract_address") == contract]
        if not rows:
            return None
    return dict(rows[-1])


def _load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_settings(settings: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def _build_live_context() -> str:
    trader = get_paper_trader()
    avg = avg_whale_score()
    clusters = count_by_cluster()
    paper = trader.net_roi_summary()
    wallet = trader.get_wallet_summary()
    settings = _load_settings()
    positions = trader.get_positions("OPEN")
    return json.dumps(
        {
            "avg_whale_score": avg,
            "cluster_counts": clusters,
            "paper_net_roi": paper,
            "demo_wallet": wallet,
            "open_positions": positions,
            "open_positions_count": wallet.get("open_positions_count", len(positions)),
            "settings": settings,
            "whale_log_rows": len(_read_whale_rows()),
            "model": MODEL_NAME,
        },
        indent=2,
    )


def _try_local_intent(message: str) -> tuple[str | None, list[str]]:
    """Fast path for wallet/stats without a round-trip to Gemini."""
    msg = message.lower()
    tools: list[str] = []

    if re.search(r"why\s+(no|not|didn.?t)\s+trade|blocker|recent\s+blocker|last\s+(action|blocker)|demo\s+bot\s+status|what\s+is\s+the\s+bot", msg):
        try:
            from app.ae13b_product.demo_bot import get_demo_bot

            st = get_demo_bot().status()
            tools.append("demo_bot_status")
            return (
                f"Demo bot status: {st.get('bot_status')}. "
                f"Last action: {st.get('last_action_summary') or '—'}. "
                f"Last blocker: {st.get('last_block_reason') or 'none'}. "
                f"Cycles: {st.get('cycles_run')}, attempts: {st.get('trade_attempt_count')}, "
                f"opened/closed: {st.get('trades_opened')}/{st.get('trades_closed')}. "
                f"Next cycle ETA: {st.get('next_cycle_eta') or 'n/a'}. "
                "Paper/demo only — AI has no trade authority."
            ), tools
        except Exception as exc:
            return f"Could not read demo bot status: {exc}", tools

    if re.search(r"opportunistic.*whale|whale.*opportunistic", msg):
        avg = avg_whale_score_by_cluster("OPPORTUNISTIC_SPECULATIVE")
        tools.append("avg_whale_score_by_cluster")
        if avg is None:
            return "No OPPORTUNISTIC_SPECULATIVE events in whale_trades_log.csv yet.", tools
        return f"Average whale score for OPPORTUNISTIC_SPECULATIVE coins: {avg:.4f}", tools

    if re.search(r"social.*whale|whale.*social", msg):
        avg = avg_whale_score_by_cluster("SOCIALLY_MOTIVATED")
        tools.append("avg_whale_score_by_cluster")
        if avg is None:
            return "No SOCIALLY_MOTIVATED events in whale_trades_log.csv yet.", tools
        return f"Average whale score for SOCIALLY_MOTIVATED coins: {avg:.4f}", tools

    if re.search(r"average\s+whale\s+score|avg\s+whale", msg):
        avg = avg_whale_score()
        tools.append("avg_whale_score")
        if avg is None:
            return "No whale trades logged yet in whale_trades_log.csv.", tools
        return f"Average whale score across logged events: {avg:.4f}", tools

    if re.search(r"demo\s+wallet|wallet\s+balance|current\s+balance|available\s+cash|open\s+positions?", msg):
        w = get_paper_trader().get_wallet_summary()
        tools.append("demo_wallet")
        return (
            f"Demo wallet ({w['trading_mode']}): starting ${w['starting_capital']:,.2f}, "
            f"cash ${w['cash_usd']:,.2f}, open positions value ${w['positions_value_usd']:,.2f}, "
            f"total equity ${w['total_equity_usd']:,.2f}, "
            f"{w['open_positions_count']} open position(s)."
        ), tools

    if re.search(r"total\s+fees|fees\s+paid|cumulative\s+fees", msg):
        w = get_paper_trader().get_wallet_summary()
        tools.append("cumulative_fees")
        return (
            f"Cumulative fees paid: swap/DEX ${w['cumulative_swap_fees']:,.2f}, "
            f"Solana priority ${w['cumulative_priority_fees']:,.2f}, "
            f"total ${w['cumulative_total_fees']:,.2f} (includes 1.5% slippage/DEX per leg)."
        ), tools

    if re.search(r"net\s+roi|after\s+fees|realized\s+pnl", msg):
        summary = get_paper_trader().net_roi_summary()
        w = get_paper_trader().get_wallet_summary()
        tools.append("paper_net_roi_summary")
        return (
            f"Paper trading net ROI: {summary['trade_count']} closed trades, "
            f"avg net ROI {summary['avg_net_roi_pct']:.2%}, total net PnL ${summary['total_net_pnl']:.2f}. "
            f"Total fees paid: ${w['cumulative_total_fees']:,.2f}."
        ), tools

    if re.search(r"cluster|SOCIALLY|OPPORTUNISTIC", msg):
        counts = count_by_cluster()
        tools.append("count_by_cluster")
        if not counts:
            return "No cluster labels in CSV yet.", tools
        parts = [f"{k}: {v}" for k, v in counts.items()]
        return "Cluster distribution: " + ", ".join(parts), tools

    m = re.search(
        r"(?:set|update)\s+(?:max\s+)?risk(?:\s+threshold|\s+score)?\s+(?:to\s+)?(\d+)",
        msg,
    )
    if m:
        val = int(m.group(1))
        settings = _load_settings()
        settings["max_risk_score"] = val
        _save_settings(settings)
        tools.append("update_settings")
        return f"Updated max_risk_score to {val}.", tools

    m = re.search(
        r"(?:set|update)\s+max\s+position(?:\s+size)?\s+(?:to\s+)?([\d.]+)",
        msg,
    )
    if m:
        val = float(m.group(1))
        settings = _load_settings()
        settings["max_position_size_pct"] = val if val <= 1 else val / 100
        _save_settings(settings)
        tools.append("update_settings")
        return f"Updated max_position_size_pct to {settings['max_position_size_pct']}.", tools

    return None, tools


class TradingChatService:
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def chat(self, message: str, session_id: str | None = None) -> tuple[str, str, list[str]]:
        sid = session_id or str(uuid.uuid4())
        local_reply, tools = _try_local_intent(message)
        if local_reply is not None:
            self._sessions.setdefault(sid, []).append({"role": "user", "content": message})
            self._sessions[sid].append({"role": "assistant", "content": local_reply})
            return local_reply, sid, tools

        if is_headless_data_collection() or get_llm_provider() == "none":
            reply = (
                "LLM inactive / provider not configured — Metrics Assistant mode. "
                "General chat is not enabled. I can still answer: "
                "'average whale score', 'net roi after fees', 'open positions', "
                "'why no trade', or 'demo bot status'."
            )
            return reply, sid, tools

        if is_ollama_provider_active():
            reply = await self._ollama_assistant_chat(message, sid, history_text="")
            tools.append("ollama_assistant_chat")
            self._sessions.setdefault(sid, []).append({"role": "user", "content": message})
            self._sessions[sid].append({"role": "assistant", "content": reply})
            return reply, sid, tools

        if not _configure_genai():
            reply = (
                "Gemini API is not configured. Metrics Assistant can still answer: "
                "'average whale score', 'net roi after fees', or 'open positions'."
            )
            return reply, sid, tools

        context = _build_live_context()
        history = self._sessions.get(sid, [])[-6:]
        history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)

        prompt = f"""{TRADING_SYSTEM_PROMPT}

You are the interactive trading copilot for the paper/demo workstation.
You have NO trade authority — you explain only; you cannot place trades or approve live trading.
Use live system state below AND your crypto knowledge.
When discussing positions, cite open_positions_count and individual open positions.

Live system state:
{context}

Conversation:
{history_text}
user: {message}

Answer concisely. Never claim profitability or live readiness. Never instruct real wallet actions.
"""

        def _call() -> str:
            if not is_gemini_provider_active():
                raise RuntimeError("Gemini provider not active")
            import google.generativeai as genai

            record_gemini_call()
            model = genai.GenerativeModel(MODEL_NAME)
            return model.generate_content(prompt).text or "No response."

        try:
            reply = await asyncio.get_event_loop().run_in_executor(None, _call)
            tools.append("gemini_chat")
        except Exception as exc:
            log.warning("Chat failed: %s", exc)
            reply = f"Chat error: {exc}"

        self._sessions.setdefault(sid, []).append({"role": "user", "content": message})
        self._sessions[sid].append({"role": "assistant", "content": reply})
        return reply, sid, tools

    async def _ollama_assistant_chat(self, message: str, sid: str, history_text: str = "") -> str:
        """Provider-backed explanation assistant (no trade authority)."""
        from app.llm_client import generate_assistant_reply
        from app.llm_config import get_ollama_model

        context = _build_live_context()
        # Enrich with demo bot / semantic status when available
        try:
            from app.ae13b_product.demo_bot import get_demo_bot
            from app.ae13_semantic.runtime_registry import get_semantic_registry

            bot = get_demo_bot().status()
            reg = get_semantic_registry().snapshot()
            extra = {
                "demo_bot_status": bot.get("bot_status"),
                "demo_last_blocker": bot.get("last_block_reason"),
                "demo_last_action": bot.get("last_action_summary"),
                "demo_cycles": bot.get("cycles_run"),
                "semantic_source": reg.get("semantic_source_label"),
                "runtime_unique_identities": reg.get("runtime_unique_identities"),
                "social_confirmed_explanation": reg.get("social_confirmed_explanation"),
                "trade_authority": "AI explanation only — no trade authority",
            }
            context = json.dumps({**json.loads(context), **extra}, indent=2)
        except Exception:
            pass

        history = self._sessions.get(sid, [])[-6:]
        hist = history_text or "\n".join(f"{h['role']}: {h['content']}" for h in history)

        def _call() -> str:
            return generate_assistant_reply(
                user_message=message,
                context_json_text=context,
                history_text=hist,
            )

        try:
            reply = await asyncio.get_event_loop().run_in_executor(None, _call)
            return (
                f"{reply}\n\n"
                f"— {get_ollama_model()} · AI Assistant — explanation only, no trade authority"
            )
        except Exception as exc:
            log.warning("Ollama assistant chat failed: %s", exc)
            return (
                "Ollama assistant is temporarily unavailable. "
                f"({exc}) Metrics still available: average whale score, net ROI after fees, "
                "open positions, recent blockers, why no trade."
            )


# Keep get_chat_service at module end — patch carefully if structure differs
_CHAT_SERVICE: TradingChatService | None = None


def get_chat_service() -> TradingChatService:
    global _CHAT_SERVICE
    if _CHAT_SERVICE is None:
        _CHAT_SERVICE = TradingChatService()
    return _CHAT_SERVICE
