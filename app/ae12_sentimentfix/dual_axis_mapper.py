"""Pure dual-axis mapper: semantic_signal_family x trading_opportunity_state."""

from __future__ import annotations

from typing import Any

from .semantic_taxonomy import derive_semantic_signal_family
from .trading_state import derive_trading_opportunity_state
from .types import DualAxisResult, null_safe_str


def map_dual_axis(row: dict[str, Any] | None) -> DualAxisResult:
    """
    Map an input row to dual-axis taxonomy fields.

    - Missing semantic -> UNKNOWN (never OPPORTUNISTIC)
    - OPPORTUNISTIC_SPECULATIVE -> trading_opportunity_state only (via legacy cluster)
    - Sticky cluster is legacy_cluster_label only; not authoritative semantic family
    - SOCIAL + OPPORTUNISTIC may coexist on separate axes
    """
    row = dict(row or {})
    legacy = row.get("legacy_cluster_label")
    if legacy is None or str(legacy).strip() == "":
        legacy = row.get("cluster_label")
    legacy_s = null_safe_str(legacy, "") or None
    if legacy_s and legacy_s.upper() in {"UNKNOWN", "NONE", "NULL", "N/A"}:
        # Keep literal empty legacy as None for clarity
        if legacy_s.upper() == "UNKNOWN" and not row.get("cluster_label"):
            legacy_s = None

    sem_family, sem_source, sem_conf, sem_reason = derive_semantic_signal_family(row)
    trade_state, trade_source = derive_trading_opportunity_state(
        {**row, "legacy_cluster_label": legacy_s or row.get("legacy_cluster_label")}
    )

    # Guard: opportunistic must never appear as semantic family
    if "OPPORTUNISTIC" in sem_family.upper() or "SPECULATIVE" in sem_family.upper():
        sem_family = "UNKNOWN"
        sem_source = "guard_rejected_opportunistic_as_semantic"
        sem_conf = 0.0
        sem_reason = "OPPORTUNISTIC/SPECULATIVE rejected as semantic_signal_family"

    taxonomy_status = "OK"
    if sem_family == "UNKNOWN" and trade_state == "UNKNOWN":
        taxonomy_status = "UNKNOWN_BOTH_AXES"
    elif sem_family == "UNKNOWN":
        taxonomy_status = "UNKNOWN_SEMANTIC"
    elif sem_source == "none":
        taxonomy_status = "UNKNOWN_NOT_EVALUATED"

    return DualAxisResult(
        semantic_signal_family=sem_family,
        semantic_signal_source=sem_source,
        semantic_signal_confidence=float(sem_conf),
        semantic_signal_reason=sem_reason,
        trading_opportunity_state=trade_state,
        trading_state_source=trade_source,
        legacy_cluster_label=legacy_s,
        taxonomy_status=taxonomy_status,
    )


def dual_axis_fields_for_writer(row: dict[str, Any] | None) -> dict[str, Any]:
    """Additive fields for future JSONL writers (optional keys)."""
    mapped = map_dual_axis(row)
    return {
        "semantic_signal_family": mapped["semantic_signal_family"],
        "semantic_signal_source": mapped["semantic_signal_source"],
        "semantic_signal_confidence": mapped["semantic_signal_confidence"],
        "semantic_signal_reason": mapped["semantic_signal_reason"],
        "trading_opportunity_state": mapped["trading_opportunity_state"],
        "trading_state_source": mapped["trading_state_source"],
        "legacy_cluster_label": mapped["legacy_cluster_label"],
        "taxonomy_status": mapped["taxonomy_status"],
        "taxonomy_schema_version": "AE12_SENTIMENTFIX_V1",
    }
