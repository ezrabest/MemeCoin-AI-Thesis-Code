"""Build AE6 decision records from runtime signals, snapshots, and lineage context."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.decision.consensus import compute_consensus
from app.decision.types import (
    AE6_PHASE,
    AE8_MISSING_REASON,
    AE9_MISSING_REASON,
    LINEAGE_FALLBACK_REASON,
    LINEAGE_IMPLICIT_CAVEAT,
    MODEL_MISSING_REASON,
    CandidateIdentityBlock,
    ContextPlaceholdersBlock,
    DecisionRecord,
    DecisionStatusAE6,
    LineageMetadata,
    LineageMode,
    LineageResolutionMethod,
    LineageStrength,
    LLMContextBlock,
    ModelScoreSlot,
    ModelScoresBlock,
    ResearchContextBlock,
    RiskContextBlock,
)
from app.engine import (
    SIGNAL_BUY_LIQUIDITY_USD,
    SIGNAL_BUY_PROB_THRESHOLD,
    SIGNAL_BUY_WHALE_THRESHOLD,
)

# Named non-trading review thresholds (documented in docs/architecture/ae6_consensus_decision_layer.md)
REVIEW_MIN_SIGNAL_SCORE = SIGNAL_BUY_PROB_THRESHOLD
REVIEW_MIN_WHALE_SCORE = SIGNAL_BUY_WHALE_THRESHOLD
REVIEW_MIN_LIQUIDITY_USD = float(SIGNAL_BUY_LIQUIDITY_USD)
DEFAULT_MAX_SNAPSHOT_AGE_SECONDS = 300.0
LINEAGE_MATCH_WINDOW_SECONDS = 600.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def unavailable_model_scores() -> ModelScoresBlock:
    """All model slots explicitly unavailable — no invented scores."""
    slot = ModelScoreSlot(available=False, missing_reason=MODEL_MISSING_REASON)
    return ModelScoresBlock(RF=slot, XGB=slot, TAB=slot, META=slot)


def build_lineage_metadata(
    *,
    provider: str | None = None,
    source: str | None = None,
    endpoint: str | None = None,
    pair_address: str | None = None,
    symbol: str | None = None,
    snapshot_timestamp: str | None = None,
    signal_timestamp: str | None = None,
    raw_payload_id: int | str | None = None,
    snapshot_id: int | str | None = None,
    signal_id: int | str | None = None,
    raw_payload_id_resolution_method: LineageResolutionMethod = LineageResolutionMethod.MISSING,
    snapshot_id_resolution_method: LineageResolutionMethod = LineageResolutionMethod.MISSING,
    signal_id_resolution_method: LineageResolutionMethod = LineageResolutionMethod.MISSING,
    raw_payload_timestamp_window: str | None = None,
) -> LineageMetadata:
    """Construct validated lineage metadata with explicit or best-effort mode.

    IDs found via pair/provider/time matching remain BEST_EFFORT linkage.
    """
    explicit_methods = {
        LineageResolutionMethod.EXPLICIT_COLUMN,
        LineageResolutionMethod.FOREIGN_KEY,
        LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
    }
    required = (
        (raw_payload_id, raw_payload_id_resolution_method),
        (snapshot_id, snapshot_id_resolution_method),
        (signal_id, signal_id_resolution_method),
    )
    has_missing = any(value is None for value, _ in required)
    has_best_effort = any(
        value is not None
        and method
        in {
            LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        }
        for value, method in required
    )
    all_explicit = all(
        value is not None and method in explicit_methods for value, method in required
    )

    if has_missing:
        raise ValueError(
            "LineageMetadata requires raw_payload_id, snapshot_id, and signal_id for AE6 decision records"
        )

    if all_explicit:
        return LineageMetadata(
            lineage_mode=LineageMode.EXPLICIT_LINKAGE,
            lineage_strength=LineageStrength.STRONG_EXPLICIT_LINKS,
            provider=provider,
            source=source,
            endpoint=endpoint,
            pair_address=pair_address,
            symbol=symbol,
            snapshot_timestamp=snapshot_timestamp,
            signal_timestamp=signal_timestamp,
            raw_payload_id=raw_payload_id,
            snapshot_id=snapshot_id,
            signal_id=signal_id,
            raw_payload_id_resolution_method=raw_payload_id_resolution_method,
            snapshot_id_resolution_method=snapshot_id_resolution_method,
            signal_id_resolution_method=signal_id_resolution_method,
            raw_payload_timestamp_window=raw_payload_timestamp_window,
            fallback_reason=None,
            lineage_warning=None,
        )

    if not has_best_effort:
        raise ValueError(
            "Unable to classify lineage: explicit linkage requires explicit structural resolution methods"
        )

    return LineageMetadata(
        lineage_mode=LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE,
        lineage_strength=LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS,
        provider=provider,
        source=source,
        endpoint=endpoint,
        pair_address=pair_address,
        symbol=symbol,
        snapshot_timestamp=snapshot_timestamp,
        signal_timestamp=signal_timestamp,
        raw_payload_id=raw_payload_id,
        snapshot_id=snapshot_id,
        signal_id=signal_id,
        raw_payload_id_resolution_method=raw_payload_id_resolution_method,
        snapshot_id_resolution_method=snapshot_id_resolution_method,
        signal_id_resolution_method=signal_id_resolution_method,
        raw_payload_timestamp_window=raw_payload_timestamp_window,
        fallback_reason=LINEAGE_FALLBACK_REASON,
        lineage_warning=LINEAGE_FALLBACK_REASON,
    )


def _compute_candidate_id(
    *,
    chain: str | None,
    pair_address: str | None,
    event_timestamp: str | None,
    source_signal_id: int | str | None,
) -> str | None:
    if not pair_address or not event_timestamp:
        return None
    payload = "|".join(
        [
            chain or "",
            pair_address,
            event_timestamp,
            str(source_signal_id or ""),
            AE6_PHASE,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_missingness(identity: CandidateIdentityBlock) -> list[str]:
    missing: list[str] = []
    optional_fields = {
        "pair_address": identity.pair_address,
        "chain": identity.chain,
        "symbol": identity.symbol,
        "coin_id": identity.coin_id,
        "event_timestamp": identity.event_timestamp,
        "source_signal_id": identity.source_signal_id,
        "source_snapshot_id": identity.source_snapshot_id,
        "source_raw_payload_id": identity.source_raw_payload_id,
        "candidate_policy_id": identity.candidate_policy_id,
        "target_row_id": identity.target_row_id,
        "base_token_address": identity.base_token_address,
        "quote_token_address": identity.quote_token_address,
    }
    for field_name, value in optional_fields.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    return missing


def _snapshot_age_seconds(snapshot_timestamp: str | None, now: datetime) -> float | None:
    snap_ts = _parse_ts(snapshot_timestamp)
    if snap_ts is None:
        return None
    return (now - snap_ts).total_seconds()


def _build_rss_context(
    conn: sqlite3.Connection | None,
    symbol: str | None,
) -> ContextPlaceholdersBlock:
    """Use local sentiment_records when available; never call external RSS."""
    ctx = ContextPlaceholdersBlock()
    if conn is None or not symbol:
        ctx.context_missingness.append("rss_sentiment")
        ctx.rss_caveat = AE8_MISSING_REASON if conn is None else "symbol_missing_for_rss_lookup"
        return ctx

    try:
        rows = conn.execute(
            """
            SELECT sentiment_score, source
            FROM sentiment_records
            WHERE symbols_json LIKE ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (f"%{symbol}%",),
        ).fetchall()
    except sqlite3.Error:
        ctx.context_missingness.append("rss_sentiment")
        ctx.rss_caveat = "local_sentiment_query_failed"
        return ctx

    if not rows:
        ctx.context_missingness.append("rss_sentiment")
        ctx.rss_caveat = "no_local_sentiment_rows_for_symbol"
        return ctx

    scores = [float(r[0]) for r in rows if r[0] is not None]
    sources = {str(r[1]) for r in rows if r[1]}
    if scores:
        ctx.rss_sentiment_available = True
        ctx.rss_sentiment_score = round(sum(scores) / len(scores), 6)
        ctx.rss_source_count = len(sources)
    else:
        ctx.context_missingness.append("rss_sentiment_score")
        ctx.rss_caveat = "sentiment_rows_without_scores"
    return ctx


def determine_decision_status(
    *,
    identity: CandidateIdentityBlock,
    lineage: LineageMetadata,
    signal_context: dict[str, Any],
    market_context: dict[str, Any],
    consensus_family: str,
    missingness: list[str],
    max_snapshot_age_seconds: float = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    now: datetime | None = None,
) -> tuple[DecisionStatusAE6, list[str], list[str], float | None, dict[str, float | None]]:
    """Conservative non-trading decision status with explicit named thresholds."""
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    caveats: list[str] = []
    confidence_components: dict[str, float | None] = {
        "signal_score": None,
        "signal_confidence": None,
        "whale_score": None,
        "liquidity_usd": None,
        "model_consensus": None,
    }

    if not identity.pair_address:
        reasons.append("missing_pair_address")
        return DecisionStatusAE6.BLOCK, reasons, caveats, 0.0, confidence_components

    snap_age = _snapshot_age_seconds(
        lineage.snapshot_timestamp or identity.event_timestamp,
        now,
    )
    if snap_age is not None and snap_age > max_snapshot_age_seconds:
        reasons.append(
            f"snapshot_stale: age_seconds={snap_age:.1f} > threshold={max_snapshot_age_seconds}"
        )
        return DecisionStatusAE6.BLOCK, reasons, caveats, 0.0, confidence_components

    signal_score = signal_context.get("score")
    signal_confidence = signal_context.get("confidence")
    whale_score = market_context.get("whale_score")
    liquidity = market_context.get("liquidity")

    if signal_score is not None:
        confidence_components["signal_score"] = float(signal_score)
    if signal_confidence is not None:
        confidence_components["signal_confidence"] = float(signal_confidence)
    if whale_score is not None:
        confidence_components["whale_score"] = float(whale_score)
    if liquidity is not None:
        confidence_components["liquidity_usd"] = float(liquidity)

    if lineage.lineage_mode == LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE:
        caveats.append(LINEAGE_IMPLICIT_CAVEAT)

    # whale_score_asof is research-only — never a hard gate
    research_caveat = "whale_score_asof remains research-only; not used as runtime approval rule"
    caveats.append(research_caveat)

    has_signal_support = (
        signal_score is not None
        and float(signal_score) >= REVIEW_MIN_SIGNAL_SCORE
    )
    has_whale_support = (
        whale_score is not None
        and float(whale_score) >= REVIEW_MIN_WHALE_SCORE
    )
    has_liquidity = (
        liquidity is not None
        and float(liquidity) >= REVIEW_MIN_LIQUIDITY_USD
    )

    model_consensus_available = consensus_family != "NO_MODEL_CONSENSUS_AVAILABLE"
    if model_consensus_available:
        confidence_components["model_consensus"] = 1.0
    else:
        confidence_components["model_consensus"] = 0.0
        reasons.append("no_model_consensus_available")

    if has_signal_support and has_whale_support and has_liquidity and not reasons:
        confidence = min(
            1.0,
            (
                float(signal_score or 0)
                + float(signal_confidence or 0)
                + float(whale_score or 0)
            )
            / 3.0,
        )
        reasons.append("paper_candidate_review_thresholds_met")
        reasons.append(
            "status_is_review_only_not_trade_execution"
        )
        return (
            DecisionStatusAE6.PAPER_CANDIDATE_REVIEW,
            reasons,
            caveats,
            round(confidence, 4),
            confidence_components,
        )

    if has_signal_support or signal_context.get("signal_type") in {"BUY", "WATCH", "ALERT"}:
        reasons.append("active_signal_without_full_review_thresholds")
        confidence = float(signal_score or signal_confidence or 0.3)
        return (
            DecisionStatusAE6.RESEARCH_CANDIDATE,
            reasons,
            caveats,
            round(confidence, 4),
            confidence_components,
        )

    if missingness:
        reasons.append(f"partial_identity_missing: {','.join(missingness[:5])}")

    reasons.append("insufficient_signal_and_model_consensus")
    confidence = float(signal_confidence or 0.2) if signal_confidence is not None else 0.1
    return DecisionStatusAE6.WATCH, reasons, caveats, round(confidence, 4), confidence_components


def build_decision_record(
    *,
    signal_row: dict[str, Any],
    snapshot_row: dict[str, Any] | None = None,
    raw_payload_row: dict[str, Any] | None = None,
    coin_row: dict[str, Any] | None = None,
    lineage: LineageMetadata | None = None,
    conn: sqlite3.Connection | None = None,
    max_snapshot_age_seconds: float = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    now: datetime | None = None,
) -> DecisionRecord:
    """Build a full AE6 decision record from runtime rows."""
    now = now or datetime.now(timezone.utc)

    pair_address = (
        (coin_row or {}).get("pair_address")
        or (snapshot_row or {}).get("pair_address")
        or (raw_payload_row or {}).get("pair_address")
    )
    chain = (coin_row or {}).get("chain") or (snapshot_row or {}).get("chain")
    symbol = signal_row.get("symbol") or (coin_row or {}).get("symbol")
    signal_id = signal_row.get("id")
    signal_ts = signal_row.get("timestamp")
    snapshot_id = (snapshot_row or {}).get("id")
    snapshot_ts = (snapshot_row or {}).get("timestamp")
    raw_id = (raw_payload_row or {}).get("id")
    provider = (
        (snapshot_row or {}).get("provider")
        or (raw_payload_row or {}).get("provider")
        or "dexscreener"
    )

    if lineage is None:
        window = None
        if raw_payload_row and raw_payload_row.get("timestamp") and signal_ts:
            window = f"{raw_payload_row.get('timestamp')}..{signal_ts}"
        lineage = build_lineage_metadata(
            provider=provider,
            source=signal_row.get("model_source"),
            endpoint=(raw_payload_row or {}).get("query"),
            pair_address=pair_address,
            symbol=symbol,
            snapshot_timestamp=snapshot_ts,
            signal_timestamp=signal_ts,
            raw_payload_id=raw_id,
            snapshot_id=snapshot_id,
            signal_id=signal_id,
            raw_payload_id_resolution_method=(
                LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH
                if raw_id is not None
                else LineageResolutionMethod.MISSING
            ),
            snapshot_id_resolution_method=(
                LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH
                if snapshot_id is not None
                else LineageResolutionMethod.MISSING
            ),
            signal_id_resolution_method=(
                LineageResolutionMethod.EXPLICIT_COLUMN
                if signal_id is not None
                else LineageResolutionMethod.MISSING
            ),
            raw_payload_timestamp_window=window,
        )

    identity = CandidateIdentityBlock(
        pair_address=pair_address,
        chain=chain,
        symbol=symbol,
        base_token_address=(coin_row or {}).get("token_address"),
        quote_token_address=None,
        coin_id=signal_row.get("coin_id") or (coin_row or {}).get("id"),
        candidate_id=_compute_candidate_id(
            chain=chain,
            pair_address=pair_address,
            event_timestamp=signal_ts,
            source_signal_id=signal_id,
        ),
        event_timestamp=signal_ts,
        source_signal_id=signal_id,
        source_snapshot_id=snapshot_id,
        source_raw_payload_id=raw_id,
    )

    missingness = _collect_missingness(identity)

    market_context: dict[str, Any] = {}
    if snapshot_row:
        for key in (
            "price",
            "liquidity",
            "volume_24h",
            "fdv",
            "whale_score",
            "buy_ratio",
            "txns_buys",
            "txns_sells",
            "price_change_m5",
            "price_change_h1",
            "filter_status",
        ):
            if snapshot_row.get(key) is not None:
                market_context[key] = snapshot_row[key]

    signal_context: dict[str, Any] = {
        "signal_type": signal_row.get("signal_type"),
        "score": signal_row.get("score"),
        "confidence": signal_row.get("confidence"),
        "reason": signal_row.get("reason"),
        "model_source": signal_row.get("model_source"),
    }
    features_raw = signal_row.get("features_json")
    if features_raw:
        try:
            signal_context["features"] = (
                json.loads(features_raw)
                if isinstance(features_raw, str)
                else features_raw
            )
        except (json.JSONDecodeError, TypeError):
            signal_context["features_parse_error"] = True

    model_scores = unavailable_model_scores()
    consensus = compute_consensus(model_scores)

    context_placeholders = _build_rss_context(conn, symbol)
    for field_name in (
        "helius_available",
        "solana_available",
        "wallet_intelligence_available",
        "reputation_available",
        "scam_flags_available",
    ):
        if not getattr(context_placeholders, field_name):
            context_placeholders.context_missingness.append(field_name)

    llm_context = LLMContextBlock()
    research_context = ResearchContextBlock()
    risk_context = RiskContextBlock(risk_gate_evaluated=False)

    status, reasons, caveats, confidence, confidence_components = determine_decision_status(
        identity=identity,
        lineage=lineage,
        signal_context=signal_context,
        market_context=market_context,
        consensus_family=consensus.consensus_family.value,
        missingness=missingness,
        max_snapshot_age_seconds=max_snapshot_age_seconds,
        now=now,
    )

    if consensus.consensus_caveat:
        caveats.append(consensus.consensus_caveat)

    return DecisionRecord(
        created_at_utc=_utc_now_iso(),
        mode="AUDIT",
        decision_status=status,
        decision_confidence=confidence,
        confidence_components=confidence_components,
        candidate_identity=identity,
        lineage=lineage,
        market_context=market_context,
        signal_context=signal_context,
        model_scores=model_scores,
        consensus=consensus,
        research_context=research_context,
        llm_context=llm_context,
        risk_context=risk_context,
        context_placeholders=context_placeholders,
        missingness=missingness,
        reasons=reasons,
        caveats=caveats,
        no_trade_authority=True,
    )


def fetch_recent_signal_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Load recent signals with best-effort snapshot/raw/coin joins."""
    rows = conn.execute(
        """
        SELECT s.id AS signal_id, s.timestamp AS signal_timestamp, s.coin_id,
               s.symbol, s.signal_type, s.score, s.confidence, s.reason,
               s.model_source, s.features_json,
               c.pair_address, c.chain, c.token_address, c.symbol AS coin_symbol
        FROM signals s
        LEFT JOIN coins c ON c.id = s.coin_id
        ORDER BY s.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        coin_id = row_dict.get("coin_id")
        pair_address = row_dict.get("pair_address")
        signal_ts = row_dict.get("signal_timestamp")

        snapshot_row = None
        if coin_id is not None and signal_ts:
            snap = conn.execute(
                """
                SELECT *
                FROM market_snapshots
                WHERE coin_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (coin_id, signal_ts),
            ).fetchone()
            if snap:
                snapshot_row = dict(snap)

        raw_row = None
        if pair_address and signal_ts:
            raw = conn.execute(
                """
                SELECT id, timestamp, provider, source_type, query, chain,
                       pair_address, symbol, payload_hash
                FROM raw_provider_payloads
                WHERE pair_address = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (pair_address, signal_ts),
            ).fetchone()
            if raw:
                raw_row = dict(raw)

        signal_row = {
            "id": row_dict["signal_id"],
            "timestamp": signal_ts,
            "coin_id": coin_id,
            "symbol": row_dict.get("symbol") or row_dict.get("coin_symbol"),
            "signal_type": row_dict.get("signal_type"),
            "score": row_dict.get("score"),
            "confidence": row_dict.get("confidence"),
            "reason": row_dict.get("reason"),
            "model_source": row_dict.get("model_source"),
            "features_json": row_dict.get("features_json"),
        }
        coin_row = {
            "id": coin_id,
            "pair_address": pair_address,
            "chain": row_dict.get("chain"),
            "token_address": row_dict.get("token_address"),
            "symbol": row_dict.get("coin_symbol") or row_dict.get("symbol"),
        }

        candidates.append(
            {
                "signal_row": signal_row,
                "snapshot_row": snapshot_row,
                "raw_payload_row": raw_row,
                "coin_row": coin_row,
            }
        )

    return candidates
