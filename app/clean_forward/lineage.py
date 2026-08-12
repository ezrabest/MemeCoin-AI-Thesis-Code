"""Order/position lineage reconciliation for AE15 (AE14 discrepancy aware)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.clean_forward.identity import build_instrument_identity, pair_address_for_id
from app.clean_forward.schema import (
    CleanForwardCandidate,
    CleanForwardDecisionInput,
    CleanForwardPaperExecutionLink,
    DECISION_INPUT_VERSION,
    make_clean_forward_candidate_id,
    make_clean_forward_decision_input_id,
    make_execution_link_id,
)
from app.clean_forward.serialization import stable_payload_hash

AE14_PENDING_NOTE = "AE14_POSITION_COUNTER_RECONCILIATION_PENDING"


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("row_key") or row.get("row_id") or "").strip()


def build_candidate_from_row(
    row: dict[str, Any],
    *,
    source_poll_file: str | None = None,
    source_poll_index: int | None = None,
) -> CleanForwardCandidate:
    identity = build_instrument_identity(row)
    provider = identity.provider
    pair_for_id = pair_address_for_id(identity.pair_address, chain=identity.chain)
    observed = str(
        row.get("observed_at")
        or row.get("fetched_at")
        or row.get("ingested_at")
        or row.get("last_fetched")
        or ""
    )
    payload_hash = str(row.get("provider_payload_hash") or row.get("payload_hash") or "")
    cand_id = make_clean_forward_candidate_id(
        chain=identity.chain,
        provider=provider,
        pair_address_for_id=pair_for_id,
        base_token_address=identity.base_token_address,
        quote_token_address=identity.quote_token_address,
        observed_at_or_fetched_at=observed,
        provider_payload_hash=payload_hash,
    )
    tx_buys = row.get("txns_buys_24h", row.get("txns_24h_buys"))
    tx_sells = row.get("txns_sells_24h", row.get("txns_24h_sells"))
    if tx_buys is None and isinstance(row.get("txns_24h"), dict):
        tx_buys = row["txns_24h"].get("buys")
    if tx_sells is None and isinstance(row.get("txns_24h"), dict):
        tx_sells = row["txns_24h"].get("sells")

    return CleanForwardCandidate(
        clean_forward_candidate_id=cand_id,
        source_clean_forward_row_key=_row_key(row) or f"{identity.chain}|pair|{identity.pair_address}",
        source_poll_file=source_poll_file or row.get("_source_poll_file"),
        source_poll_index=source_poll_index
        if source_poll_index is not None
        else row.get("_source_poll_index"),
        provider_payload_hash=payload_hash,
        provider_pair_url=identity.provider_pair_url,
        chain=identity.chain,
        pair_address=identity.pair_address,
        base_token_address=identity.base_token_address,
        quote_token_address=identity.quote_token_address,
        symbol_pair=str(row.get("pair") or row.get("pair_label") or "") or None,
        price_usd=_safe_float(row.get("price_usd", row.get("price"))),
        liquidity_usd=_safe_float(row.get("liquidity_usd", row.get("liquidity"))),
        volume_24h=_safe_float(row.get("volume_24h")),
        txns_buys_24h=int(tx_buys) if tx_buys is not None else None,
        txns_sells_24h=int(tx_sells) if tx_sells is not None else None,
        price_change_m5=_safe_float(row.get("price_change_m5", row.get("price_change_5m"))),
        price_change_h1=_safe_float(row.get("price_change_h1", row.get("price_change_1h"))),
        price_change_h6=_safe_float(row.get("price_change_h6", row.get("price_change_6h"))),
        price_change_h24=_safe_float(row.get("price_change_h24", row.get("price_change_24h"))),
        observed_at=row.get("observed_at"),
        fetched_at=row.get("fetched_at"),
        ingested_at=row.get("ingested_at"),
        rendered_at=row.get("rendered_at"),
        latest_provider_fetch_at=row.get("last_fetched") or row.get("fetched_at"),
        verification_status=identity.verification_status,
        freshness_status=identity.freshness_status,
        identity_status=identity.identity_status,
        clean_feed_eligible=bool(row.get("clean_feed_eligible", True)),
        paper_demo_only=identity.paper_demo_only,
        live_trading_ready=identity.live_trading_ready,
        provider=provider,
    )


def build_decision_input_for_candidate(
    candidate: CleanForwardCandidate,
    *,
    active_preset_id: str = "ae15_schema_bridge_default",
    risk_mode: str = "balanced",
    strict_mode: bool = False,
    exploration_mode: bool = False,
    gatekeeper_payload: dict[str, Any] | None = None,
    riskguard_payload: dict[str, Any] | None = None,
    max_price_age_seconds: int | None = None,
    strict_shadow_max_price_age_seconds: int | None = None,
) -> CleanForwardDecisionInput:
    snapshot_ts = str(
        candidate.observed_at or candidate.fetched_at or candidate.ingested_at or ""
    )
    decision_id = make_clean_forward_decision_input_id(
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        candidate_snapshot_timestamp=snapshot_ts,
        active_preset_id=active_preset_id,
        risk_mode=risk_mode,
        strict_mode=strict_mode,
        exploration_mode=exploration_mode,
        decision_input_version=DECISION_INPUT_VERSION,
    )
    return CleanForwardDecisionInput(
        clean_forward_decision_input_id=decision_id,
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        candidate_snapshot_timestamp=snapshot_ts,
        gatekeeper_input_payload_hash=stable_payload_hash(gatekeeper_payload)
        if gatekeeper_payload is not None
        else None,
        riskguard_input_payload_hash=stable_payload_hash(riskguard_payload)
        if riskguard_payload is not None
        else None,
        active_preset_id=active_preset_id,
        risk_mode=risk_mode,
        strict_mode=strict_mode,
        exploration_mode=exploration_mode,
        max_price_age_seconds=max_price_age_seconds,
        strict_shadow_max_price_age_seconds=strict_shadow_max_price_age_seconds,
        model_scores_available=False,
        xgb_score=None,
        tab_score=None,
        rf_score=None,
        model_score_source_status="AE15_SCHEMA_ONLY_NO_MODEL_AUTHORITY",
        consensus_tier_shadow=None,
        context_status="AE15_CONTEXT_NOT_EXECUTED",
        llm_status="AE15_LLM_NOT_CALLED",
        decision_input_version=DECISION_INPUT_VERSION,
    )


def _link_from_parts(
    *,
    candidate: CleanForwardCandidate | None,
    decision: CleanForwardDecisionInput | None,
    paper_order_id: str | None,
    paper_position_id: str | None,
    order_created_at: str | None,
    position_created_at: str | None,
    order_side: str | None,
    order_status: str | None,
    fill_status: str | None,
    position_status: str | None,
    order_notional_usd: float | None,
    fill_price_usd: float | None,
    position_quantity: float | None,
    source_provider_pair_url: str | None,
    pair_address: str | None,
    base_token_address: str | None,
    quote_token_address: str | None,
    provider_payload_hash: str | None,
    gatekeeper_decision: str | None,
    riskguard_decision: str | None,
    entry_reason: str | None,
    skip_reason: str | None,
    position_created_by: str | None,
    position_creation_reason: str | None,
    preexisting_position_detected: bool,
    reconstructed_position_detected: bool,
    duplicate_position_detected: bool,
    one_order_to_one_position_expected: bool,
    one_order_to_one_position_passed: bool | None,
    counter_consistency_status: str,
    source_clean_forward_row_key: str | None = None,
) -> CleanForwardPaperExecutionLink:
    cand_id = candidate.clean_forward_candidate_id if candidate else None
    dec_id = decision.clean_forward_decision_input_id if decision else None
    link_id = make_execution_link_id(
        clean_forward_candidate_id=cand_id or "missing_candidate",
        paper_order_id=paper_order_id,
        paper_position_id=paper_position_id,
        position_created_at=position_created_at,
    )
    return CleanForwardPaperExecutionLink(
        execution_link_id=link_id,
        clean_forward_candidate_id=cand_id,
        clean_forward_decision_input_id=dec_id,
        source_clean_forward_row_key=source_clean_forward_row_key
        or (candidate.source_clean_forward_row_key if candidate else None),
        paper_order_id=paper_order_id,
        paper_position_id=paper_position_id,
        order_created_at=order_created_at,
        position_created_at=position_created_at,
        order_side=order_side,
        order_status=order_status,
        fill_status=fill_status,
        position_status=position_status,
        order_notional_usd=order_notional_usd,
        fill_price_usd=fill_price_usd,
        position_quantity=position_quantity,
        source_provider_pair_url=source_provider_pair_url,
        pair_address=pair_address,
        base_token_address=base_token_address,
        quote_token_address=quote_token_address,
        provider_payload_hash=provider_payload_hash,
        gatekeeper_decision=gatekeeper_decision,
        riskguard_decision=riskguard_decision,
        entry_reason=entry_reason,
        skip_reason=skip_reason,
        position_created_by=position_created_by,
        position_creation_reason=position_creation_reason,
        preexisting_position_detected=preexisting_position_detected,
        reconstructed_position_detected=reconstructed_position_detected,
        duplicate_position_detected=duplicate_position_detected,
        one_order_to_one_position_expected=one_order_to_one_position_expected,
        one_order_to_one_position_passed=one_order_to_one_position_passed,
        counter_consistency_status=counter_consistency_status,
    )


def reconcile_ae14_order_position_lineage(
    ae14_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild explicit order↔position lineage from AE14 closure artifacts.

    AE14 reported paper_orders_opened=1 and paper_positions_opened=2 because:
    1) explicit PaperTrader.open_position opened Bonk/MET (order+position #1)
    2) demo_bot.run_once() opened a second position (PUMP #2) without a
       separately counted paper_orders_opened increment in the AE14 audit.

    AE15 records this explicitly and does not silently pass the mismatch.
    """
    audit = ae14_artifacts.get("audit") or {}
    selected = ae14_artifacts.get("selected_row") or {}
    pos1 = ae14_artifacts.get("paper_open_position") or {}
    demo = ae14_artifacts.get("demo_bot_run_once") or {}
    gatekeeper = ae14_artifacts.get("gatekeeper_result")
    bridge = ae14_artifacts.get("bridge_result")

    links: list[CleanForwardPaperExecutionLink] = []
    candidates: list[CleanForwardCandidate] = []
    decisions: list[CleanForwardDecisionInput] = []
    notes: list[str] = []

    if not selected or selected.get("_load_error"):
        return {
            "ok": False,
            "blocker": "AE14_SELECTED_ROW_MISSING",
            "links": [],
            "candidates": [],
            "decisions": [],
            "summary": summarize_order_position_lineage([]),
            "ae14_discrepancy_status": AE14_PENDING_NOTE,
            "notes": ["AE14 selected_clean_forward_row.json missing or unreadable"],
        }

    cand1 = build_candidate_from_row(selected, source_poll_file="ae14_selected_clean_forward_row")
    cand1.clean_feed_eligible = True
    candidates.append(cand1)
    dec1 = build_decision_input_for_candidate(
        cand1,
        gatekeeper_payload=gatekeeper if isinstance(gatekeeper, dict) else None,
        riskguard_payload=None,
        active_preset_id="ae14_closure_explicit_open",
        risk_mode="balanced",
    )
    decisions.append(dec1)

    order_id_1 = f"ae14_explicit_order:{pos1.get('id', 1)}"
    link1 = _link_from_parts(
        candidate=cand1,
        decision=dec1,
        paper_order_id=order_id_1,
        paper_position_id=str(pos1.get("id") or audit.get("opened_position_id") or "1"),
        order_created_at=pos1.get("opened_at"),
        position_created_at=pos1.get("opened_at"),
        order_side="buy",
        order_status="filled",
        fill_status="filled",
        position_status=pos1.get("status") or "OPEN",
        order_notional_usd=_safe_float(pos1.get("size_usd")),
        fill_price_usd=_safe_float(pos1.get("fill_price") or pos1.get("entry_price")),
        position_quantity=_safe_float(pos1.get("quantity")),
        source_provider_pair_url=selected.get("provider_pair_url"),
        pair_address=pos1.get("pair_address") or selected.get("pair_address"),
        base_token_address=pos1.get("base_token_address") or selected.get("base_token_address"),
        quote_token_address=pos1.get("quote_token_address") or selected.get("quote_token_address"),
        provider_payload_hash=selected.get("provider_payload_hash") or selected.get("payload_hash"),
        gatekeeper_decision="pass" if isinstance(gatekeeper, dict) else "unknown",
        riskguard_decision="not_recorded_in_ae14_explicit_path",
        entry_reason="ae14_explicit_paper_open_position",
        skip_reason=None,
        position_created_by="PaperTrader.open_position",
        position_creation_reason="AE14 explicit Clean Forward closure open (Bonk/MET)",
        preexisting_position_detected=False,
        reconstructed_position_detected=False,
        duplicate_position_detected=False,
        one_order_to_one_position_expected=True,
        one_order_to_one_position_passed=True,
        counter_consistency_status="AE15_LINKED_EXPLICIT_OPEN",
    )
    links.append(link1)

    # Position #2 from demo_bot.run_once
    opened_block = demo.get("opened") if isinstance(demo, dict) else None
    pos2 = None
    if isinstance(opened_block, dict):
        pos2 = opened_block.get("position")
    if not isinstance(pos2, dict):
        pos2 = None

    if pos2:
        # Build a synthetic row for the second position from demo payload fields.
        row2 = {
            "row_key": f"{pos2.get('chain')}|pair|{pos2.get('pair_address')}",
            "row_id": f"{pos2.get('chain')}|pair|{pos2.get('pair_address')}",
            "source_provider": "dexscreener",
            "chain": pos2.get("chain"),
            "normalized_chain_id": pos2.get("chain"),
            "pair_address": pos2.get("pair_address"),
            "provider_pair_id": pos2.get("provider_pair_id") or pos2.get("pair_address"),
            "base_token_address": pos2.get("base_token_address"),
            "quote_token_address": pos2.get("quote_token_address"),
            "pair": pos2.get("pair"),
            "price_usd": pos2.get("fill_price") or pos2.get("entry_price"),
            "liquidity_usd": pos2.get("liquidity_at_entry") or pos2.get("liquidity"),
            "observed_at": pos2.get("price_updated_at") or pos2.get("opened_at"),
            "fetched_at": pos2.get("price_updated_at") or pos2.get("opened_at"),
            "ingested_at": pos2.get("opened_at"),
            "provider_payload_hash": f"ae14_demo_bot_position_{pos2.get('id')}",
            "verification_status": "provider_pair_verified",
            "freshness_status": "fresh",
            "identity_status": "pair_and_tokens_separated",
            "shown_as_token_contract": False,
            "paper_demo_only": True,
            "live_trading_ready": False,
            "clean_feed_eligible": True,
            "provider_pair_url": None,
        }
        cand2 = build_candidate_from_row(row2, source_poll_file="ae14_demo_bot_run_once")
        candidates.append(cand2)
        dec2 = build_decision_input_for_candidate(
            cand2,
            active_preset_id="ae14_demo_bot_run_once",
            risk_mode=str(pos2.get("risk_mode") or "balanced"),
            exploration_mode=True,
            gatekeeper_payload={"source": "demo_bot_run_once"},
        )
        decisions.append(dec2)

        # AE14 did not emit a distinct paper_order_id / order counter for this open.
        order_id_2 = None
        link2 = _link_from_parts(
            candidate=cand2,
            decision=dec2,
            paper_order_id=order_id_2,
            paper_position_id=str(pos2.get("id")),
            order_created_at=None,
            position_created_at=pos2.get("opened_at"),
            order_side="buy",
            order_status=None,
            fill_status="filled",
            position_status=pos2.get("status") or "OPEN",
            order_notional_usd=_safe_float(pos2.get("size_usd")),
            fill_price_usd=_safe_float(pos2.get("fill_price") or pos2.get("entry_price")),
            position_quantity=_safe_float(pos2.get("quantity")),
            source_provider_pair_url=None,
            pair_address=pos2.get("pair_address"),
            base_token_address=pos2.get("base_token_address"),
            quote_token_address=pos2.get("quote_token_address"),
            provider_payload_hash=row2["provider_payload_hash"],
            gatekeeper_decision="pass",
            riskguard_decision="pass_or_not_fully_audited_in_ae14",
            entry_reason=pos2.get("entry_reason"),
            skip_reason=None,
            position_created_by="demo_bot.run_once",
            position_creation_reason=(
                "AE14 demo_bot.run_once opened second Clean Forward position "
                "(PUMP/MET). AE14 audit incremented paper_positions_opened but "
                "did not allocate a separate paper_orders_opened count / order id."
            ),
            preexisting_position_detected=False,
            reconstructed_position_detected=False,
            duplicate_position_detected=False,
            one_order_to_one_position_expected=True,
            one_order_to_one_position_passed=False,
            counter_consistency_status=AE14_PENDING_NOTE,
        )
        links.append(link2)
        notes.append(
            "Position #2 (PUMP) created by demo_bot.run_once without AE14 paper_order counter."
        )
    else:
        notes.append("demo_bot position #2 payload missing; cannot fully explain second position.")

    ae14_orders = int(audit.get("paper_orders_opened") or 0)
    ae14_positions = int(audit.get("paper_positions_opened") or 0)
    summary = summarize_order_position_lineage(
        [lnk.to_dict() for lnk in links],
        reported_orders_opened=ae14_orders,
        reported_positions_opened=ae14_positions,
    )
    summary["ae14_paper_orders_opened"] = ae14_orders
    summary["ae14_paper_positions_opened"] = ae14_positions
    summary["ae14_position_count_delta"] = ae14_positions - ae14_orders
    summary["ae14_discrepancy_explained"] = bool(pos2)
    summary["ae14_discrepancy_status"] = AE14_PENDING_NOTE
    summary["ae14_explanation"] = (
        "AE14 counted 1 explicit paper order (Bonk/MET PaperTrader.open_position) "
        "but 2 open positions because demo_bot.run_once opened a second CF position "
        "(PUMP/MET) without a matching AE14 paper_orders_opened increment or durable "
        "paper_order_id. Not a preexisting/reconstructed duplicate of Bonk/MET."
    )
    notes.append(AE14_PENDING_NOTE)

    # Duplicate-pair skip from demo cycle (Bonk rejected)
    skip_rows: list[dict[str, Any]] = []
    if isinstance(opened_block, dict):
        for rej in opened_block.get("rejected_attempts") or []:
            if not isinstance(rej, dict):
                continue
            skip_rows.append(
                {
                    "clean_forward_candidate_id": cand1.clean_forward_candidate_id,
                    "clean_forward_decision_input_id": dec1.clean_forward_decision_input_id,
                    "skipped_at": pos2.get("opened_at") if pos2 else None,
                    "skip_stage": "riskguard",
                    "skip_reason_code": rej.get("rejection_code") or "DUPLICATE_PAIR_ALREADY_OPEN",
                    "skip_reason_detail": "; ".join(rej.get("rejection_reasons") or []),
                    "blocked_by_riskguard": True,
                    "pair_address": rej.get("pair_address"),
                }
            )

    return {
        "ok": True,
        "blocker": None,
        "links": links,
        "candidates": candidates,
        "decisions": decisions,
        "summary": summary,
        "ae14_discrepancy_status": AE14_PENDING_NOTE,
        "notes": notes,
        "skip_rows": skip_rows,
        "bridge_ok": bool(isinstance(bridge, dict) and bridge.get("ok")),
        "selected_row_key": _row_key(selected),
    }


def summarize_order_position_lineage(
    links: list[dict[str, Any]],
    *,
    reported_orders_opened: int | None = None,
    reported_positions_opened: int | None = None,
) -> dict[str, Any]:
    order_ids = [str(x.get("paper_order_id")) for x in links if x.get("paper_order_id")]
    position_ids = [str(x.get("paper_position_id")) for x in links if x.get("paper_position_id")]

    order_to_positions: dict[str, set[str]] = defaultdict(set)
    position_to_orders: dict[str, set[str]] = defaultdict(set)
    for link in links:
        oid = link.get("paper_order_id")
        pid = link.get("paper_position_id")
        if oid and pid:
            order_to_positions[str(oid)].add(str(pid))
            position_to_orders[str(pid)].add(str(oid))

    orders_with_multiple_positions = sorted(
        oid for oid, pids in order_to_positions.items() if len(pids) > 1
    )
    positions_with_multiple_orders = sorted(
        pid for pid, oids in position_to_orders.items() if len(oids) > 1
    )
    orders_without_position = sorted(
        {
            str(x.get("paper_order_id"))
            for x in links
            if x.get("paper_order_id") and not x.get("paper_position_id")
        }
    )
    positions_without_order = sorted(
        {
            str(x.get("paper_position_id"))
            for x in links
            if x.get("paper_position_id") and not x.get("paper_order_id")
        }
    )
    duplicate_position_ids = sorted(
        pid for pid, count in Counter(position_ids).items() if count > 1
    )
    preexisting = sum(1 for x in links if x.get("preexisting_position_detected"))
    reconstructed = sum(1 for x in links if x.get("reconstructed_position_detected"))

    orders_opened = reported_orders_opened if reported_orders_opened is not None else len(set(order_ids))
    positions_opened = (
        reported_positions_opened if reported_positions_opened is not None else len(set(position_ids))
    )
    delta = positions_opened - orders_opened

    if positions_without_order or orders_with_multiple_positions or delta != 0:
        counter_status = AE14_PENDING_NOTE if delta != 0 or positions_without_order else "LINEAGE_MISMATCH"
    elif not links:
        counter_status = "NO_LINKS"
    else:
        counter_status = "CONSISTENT"

    ae14_resolved = True
    if delta != 0 or positions_without_order:
        # Explained but not resolved to 1:1 historical counters.
        ae14_resolved = False
        counter_status = AE14_PENDING_NOTE

    return {
        "orders_opened": orders_opened,
        "positions_opened": positions_opened,
        "position_count_delta": delta,
        "orders_without_position": orders_without_position,
        "positions_without_order": positions_without_order,
        "orders_with_multiple_positions": orders_with_multiple_positions,
        "positions_with_multiple_orders": positions_with_multiple_orders,
        "preexisting_positions_detected": preexisting,
        "reconstructed_positions_detected": reconstructed,
        "duplicate_position_ids": duplicate_position_ids,
        "counter_consistency_status": counter_status,
        "ae14_discrepancy_resolved": ae14_resolved,
        "ae14_discrepancy_status": (
            AE14_PENDING_NOTE
            if not ae14_resolved and (delta != 0 or positions_without_order)
            else (
                "NOT_APPLICABLE"
                if delta == 0 and not positions_without_order
                else AE14_PENDING_NOTE
            )
        ),
        "link_count": len(links),
        "unique_order_ids": sorted(set(order_ids)),
        "unique_position_ids": sorted(set(position_ids)),
    }


def detect_lineage_mismatch(
    *,
    orders_opened: int,
    positions_opened: int,
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Unit-test friendly mismatch detector."""
    summary = summarize_order_position_lineage(
        links or [],
        reported_orders_opened=orders_opened,
        reported_positions_opened=positions_opened,
    )
    mismatched = summary["position_count_delta"] != 0 or bool(summary["positions_without_order"])
    return {
        "mismatched": mismatched,
        "summary": summary,
        "requires_ae14_pending_note": mismatched,
    }
