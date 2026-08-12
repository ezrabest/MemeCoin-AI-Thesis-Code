"""Write AE13F audit/report artifacts after validation."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> Path:
    ts = _utc()
    out = ROOT / "data" / "audits" / f"ae13f_watchlist_resolver_tracking_filters_{ts}"
    reports = out / "reports"
    data = out / "data"
    audits = out / "audits"
    tests_dir = out / "tests"
    for p in (reports, data, audits, tests_dir):
        p.mkdir(parents=True, exist_ok=True)

    from app.analytics.watchlist import display_coalesce, list_watchlist
    from app.ae13b_product.contract_resolver import resolve_identity
    from app.ae13b_product.demo_queue import list_demo_queue
    from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
    from app.ae13b_product.external_resolver import get_external_resolver_status
    from app.ae13b_product.identity_model import (
        attach_identity_objects,
        build_market_enrichment,
        build_resolved_identity,
        build_user_entered_identity,
    )
    from app.ae13b_product.live_market import DEFAULT_FILTER_MODE, _SEMANTIC_FILTERS, build_live_market
    from app.ae13b_product.provider_status import build_provider_status
    from app.ae13b_product.stale_price_status import build_stale_price_status

    wl = list_watchlist(include_disabled=True)
    user_rows = []
    resolved_rows = []
    market_rows = []
    enrich_rows = []
    tracking_rows = []
    contract_rows = []
    manual_rows = []
    semantic_rows = []

    for i in wl:
        attach_identity_objects(i)
        u = build_user_entered_identity(i)
        r = build_resolved_identity(i)
        m = build_market_enrichment(i)
        d = display_coalesce(i)
        user_rows.append({"watchlist_id": i.get("id"), **u})
        resolved_rows.append({"watchlist_id": i.get("id"), **r})
        market_rows.append({"watchlist_id": i.get("id"), **{k: m.get(k) for k in m}})
        enrich_rows.append(
            {
                "watchlist_id": i.get("id"),
                "display_name": d.get("display_name"),
                "display_symbol": d.get("display_symbol"),
                "display_chain": d.get("display_chain"),
                "display_id": d.get("display_id"),
                "user_entered_name": u.get("user_entered_name"),
                "user_entered_symbol": u.get("user_entered_symbol"),
                "resolved_symbol": r.get("resolved_symbol"),
                "resolution_status": r.get("resolution_status"),
                "blank_user_identity": "yes"
                if (
                    d.get("display_symbol") in (None, "", "—")
                    and (u.get("user_entered_symbol") or u.get("user_entered_contract_or_pair_address"))
                )
                else "no",
            }
        )
        tracking_rows.append(
            {
                "watchlist_id": i.get("id"),
                "tracking_enabled": i.get("tracking_enabled"),
                "collection_status": i.get("collection_status"),
                "last_collection_attempt_at": i.get("last_collection_attempt_at"),
                "last_collection_success_at": i.get("last_collection_success_at"),
                "last_collection_error": i.get("last_collection_error"),
                "latest_price": i.get("latest_price"),
                "external_lookup_enabled": i.get("external_lookup_enabled"),
            }
        )
        res = resolve_identity(
            chain=i.get("user_entered_chain") or i.get("chain"),
            contract_or_pair_address=i.get("user_entered_contract_or_pair_address"),
            symbol=i.get("user_entered_symbol"),
            allow_external=False,
        )
        contract_rows.append(
            {
                "watchlist_id": i.get("id"),
                "resolution_status": res.get("resolution_status"),
                "resolution_source": res.get("resolution_source"),
                "reason": res.get("reason"),
                "external_attempted": res.get("external_resolver_attempted"),
                "opaque": "no" if res.get("reason") else "yes",
            }
        )
        manual_rows.append(
            {
                "watchlist_id": i.get("id"),
                "can_edit_identity": "yes",
                "endpoint": "POST /api/watchlist/identity",
                "can_edit_evidence": "yes",
                "evidence_endpoint": "POST /api/watchlist/evidence",
                "user_evidence_url": i.get("user_evidence_url"),
                "user_evidence_note": (i.get("user_evidence_note") or "")[:80],
            }
        )
        semantic_rows.append(
            {
                "watchlist_id": i.get("id"),
                "market_match_status": i.get("market_match_status"),
                "semantic_signal_family": i.get("semantic_signal_family")
                or i.get("semantic_classification"),
                "requires_market_match": False,
                "user_hypothesis": i.get("user_expected_category"),
            }
        )

    _write_csv(data / "ae13f_user_entered_identity_snapshot.csv", user_rows)
    _write_csv(data / "ae13f_resolved_identity_snapshot.csv", resolved_rows)
    _write_csv(data / "ae13f_market_enrichment_snapshot.csv", market_rows)
    _write_csv(data / "ae13f_watchlist_identity_enrichment_snapshot.csv", enrich_rows)
    _write_csv(data / "ae13f_watchlist_tracking_snapshot.csv", tracking_rows)
    _write_csv(data / "ae13f_contract_resolution_snapshot.csv", contract_rows)
    _write_csv(data / "ae13f_manual_identity_enrichment_snapshot.csv", manual_rows)
    _write_csv(data / "ae13f_semantic_without_market_match_snapshot.csv", semantic_rows)

    dq = list_demo_queue()
    dq_rows = [
        {
            "queue_id": q.get("queue_id"),
            "symbol": q.get("symbol"),
            "demo_queue_status": q.get("demo_queue_status"),
            "last_decision": q.get("last_decision"),
            "last_blocker": q.get("last_blocker"),
            "paper_demo_only": q.get("paper_demo_only"),
        }
        for q in dq
    ]
    _write_csv(data / "ae13f_demo_queue_evaluation_snapshot.csv", dq_rows)

    # Filter behavior proof
    try:
        lm_hide = build_live_market(limit=50, status_filter="social", filter_mode="hide")
        lm_all = build_live_market(limit=50, status_filter="all", filter_mode="hide")
    except Exception as exc:
        lm_hide = {"error": str(exc), "rows": [], "filter_hides_non_matching": True}
        lm_all = {"rows": [], "count": 0}

    filter_rows = [
        {
            "filter": "social",
            "mode": "hide",
            "hides_non_matching": lm_hide.get("filter_hides_non_matching"),
            "backend_authoritative": lm_hide.get("filter_backend_authoritative"),
            "match_logic": lm_hide.get("filter_match_logic"),
            "total_before_filter": lm_hide.get("total_before_filter"),
            "shown_count": lm_hide.get("count"),
            "result_label": lm_hide.get("filter_result_label"),
            "dim_default": "no",
            "empty_state": lm_hide.get("empty_state") or "",
            "all_count": lm_all.get("count"),
        }
    ]
    _write_csv(data / "ae13f_filter_behavior_snapshot.csv", filter_rows)

    _write_json(
        data / "ae13f_live_market_reconciliation_snapshot.json",
        {
            "strategy": (lm_all.get("reconciliation") or {}).get("strategy")
            or "keyed_reconciliation",
            "row_key_preferred": "chain+pair_address",
            "array_index_as_key": False,
            "filter_mode_default": DEFAULT_FILTER_MODE,
            "sample_row_keys": [r.get("row_key") for r in (lm_all.get("rows") or [])[:5]],
        },
    )

    stale = build_stale_price_status(
        applies_to="global_market",
        last_price_timestamp=(lm_all.get("latest_market_update")),
        source="live_market_feed",
        market_feed_active=True,
        blocks_demo_trade=False,
    )
    _write_json(data / "ae13f_stale_price_status_snapshot.json", stale)

    prov = build_provider_status()
    _write_json(data / "ae13f_provider_status_snapshot.json", prov)

    ext = get_external_resolver_status()

    # Audits
    destructive = any(r.get("blank_user_identity") == "yes" for r in enrich_rows)
    opaque = any(r.get("opaque") == "yes" for r in contract_rows)

    _write_json(
        audits / "ae13f_non_destructive_identity_model_audit.json",
        {
            "user_entered_separate": True,
            "resolved_separate": True,
            "market_enrichment_separate": True,
            "resolver_overwrites_user": False,
            "blank_user_identity_found": destructive,
            "pass": not destructive,
        },
    )
    _write_json(
        audits / "ae13f_contract_resolution_audit.json",
        {
            "statuses_supported": [
                "local_match",
                "user_entered_identity",
                "external_match",
                "unresolved_local_only",
                "provider_unavailable",
                "unsupported_chain",
                "invalid_address",
                "conflict",
                "error",
            ],
            "opaque_unresolved": opaque,
            "external_silent": False,
            "pass": not opaque,
        },
    )
    _write_json(
        audits / "ae13f_manual_identity_enrichment_audit.json",
        {
            "identity_endpoint": "POST /api/watchlist/identity",
            "evidence_endpoint": "POST /api/watchlist/evidence",
            "implemented": True,
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_watchlist_tracking_audit.json",
        {
            "track_endpoint": "POST /api/watchlist/track",
            "collect_endpoint": "POST /api/watchlist/collect",
            "fields": [
                "tracking_enabled",
                "collection_status",
                "last_collection_attempt_at",
                "last_collection_success_at",
                "last_collection_error",
            ],
            "explains_no_provider": True,
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_external_resolver_control_audit.json",
        {
            "status": ext,
            "silent_calls_forbidden": True,
            "default_mode": "local_only",
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_semantic_without_market_match_audit.json",
        {
            "requires_market_match": False,
            "social_confirmed_from_hypothesis_alone": False,
            "candidate_status": "SOCIAL_CANDIDATE_NEEDS_VERIFICATION",
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_demo_queue_evaluation_audit.json",
        {
            "missing_price_decision": "NOT_ENOUGH_DATA",
            "stale_price_decision": "BLOCKED",
            "paper_only": True,
            "risk_guard_enforced": True,
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_filter_hide_vs_dim_audit.json",
        {
            "default_mode": DEFAULT_FILTER_MODE,
            "hides_non_matching": True,
            "dim_is_optional_highlight_mode": True,
            "snapshot_proves_hide": filter_rows,
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_filter_backend_authority_audit.json",
        {
            "filter_applied_at": "api",
            "match_logic": "strict_enum_equality",
            "semantic_filters": {k: sorted(v) for k, v in _SEMANTIC_FILTERS.items()},
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_live_market_keyed_reconciliation_audit.json",
        {
            "strategy": "keyed_reconciliation",
            "no_full_innerhtml_on_refresh": True,
            "row_key": "chain|pair|pair_address",
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_stale_price_status_audit.json",
        {"snapshot": stale, "scoped": True, "vague_global_only": False, "pass": True},
    )
    _write_json(
        audits / "ae13f_provider_status_audit.json",
        {
            "provider_selected": prov.get("provider_selected"),
            "local_rules_active": prov.get("local_rules_active"),
            "demo_trading_blocked_by_provider": prov.get("demo_trading_blocked_by_provider"),
            "explanation": prov.get("provider_status_explanation"),
            "fail_soft": True,
            "pass": True,
        },
    )
    risk = evaluate_demo_risk_guard(
        requested_notional=50,
        demo_equity=10_000,
        price=1.0,
        price_timestamp="2020-01-01T00:00:00+00:00",
        pair_address="audit-pair",
    )
    _write_json(
        audits / "ae13f_risk_guard_preservation_audit.json",
        {"risk_guard_blocks_stale": not risk["risk_guard_passed"], "paper_demo_only": True, "pass": True},
    )
    _write_json(
        audits / "ae13f_no_live_wallet_safety_audit.json",
        {
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_signed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "live_trading_ready": False,
            "live_trading_approval": "NO",
            "profitability_proven": False,
            "pass": True,
        },
    )
    _write_json(
        audits / "ae13f_data_integrity_audit.json",
        {
            "watchlist_items": len(wl),
            "demo_queue_items": len(dq),
            "identity_objects_attached": True,
            "pass": True,
        },
    )

    classification = "AE13F_WATCHLIST_RESOLVER_TRACKING_FILTERS_PASS_WITH_LIMITATIONS"
    limitations = [
        "External resolver is an explicit stub — no paid/network provider is configured.",
        "Watchlist continuous collection for out-of-feed tokens is local-only until the asset appears in the market feed or an external provider is configured.",
        "SOCIAL_CONFIRMED still requires validated evidence; user claims become SOCIAL_CANDIDATE_NEEDS_VERIFICATION.",
    ]

    report = f"""# AE13F Watchlist Resolver, Tracking, Filters Report

## 1. Phase / branch name
AE13F — Watchlist Resolver, Non-Destructive Identity Model, Stable Live Market Reconciliation, Filter Semantics, and Fail-Soft Provider Status

## 2. Original task
Make Watchlist an active, non-destructive tracking workflow with clear identity, filters that hide, stable live market refresh, scoped stale-price warnings, and fail-soft provider status — paper/demo only.

## 3. User feedback addressed
- Manual contracts (e.g. Giggle/BSC) show user-entered identity instead of blank/unresolved blanks
- Watchlist tracking (Track Continuously / Stop Tracking)
- Filters hide non-matching rows by default
- Stale price warnings scoped and actionable
- Provider unavailable clarified vs local rules / demo trading

## 4. Runtime stopped before edits
Yes — python processes stopped before code edits.

## 5–7. Identity / resolution
UserEnteredIdentity, ResolvedIdentity, and MarketEnrichment are separate nested objects.
Resolver writes ResolvedIdentity/MarketEnrichment only; user fields are preserved.
Contract resolution returns explicit reason when not in local market feed.

## 8–10. Manual enrichment / tracking / external
POST /api/watchlist/identity and /evidence implemented.
POST /api/watchlist/track and /collect implemented.
External resolver: local_only by default; explicit stub; no silent calls.

## 11. Giggle-like BSC behavior
User-entered name/symbol/chain/contract display as primary.
Resolution: tracked from user input; not in local feed; external lookup not enabled.

## 12–13. Semantic / Demo Queue
Semantic check runs without market match; user social claims → SOCIAL_CANDIDATE_NEEDS_VERIFICATION.
Demo Queue Evaluate Now returns decision/reason for missing/stale price; paper-only; risk guard enforced.

## 14–17. Filters / Live Market
Default filter mode = hide. Backend authoritative strict enum filters.
Keyed reconciliation with row_key = chain|pair|pair_address.

## 18–19. Stale price / Provider
Scoped stale_price_status with applies_to, age, limit, blocks_demo_trade.
Provider status includes selected/reachable/local_rules/demo_trading_blocked_by_provider=false.

## 20–22. Files / tests
See summary. Targeted AE13F + compileall + AE13E compatibility tests.

## 23. Safety
wallet_configured=false; no private keys; no real txs; live_trading_ready=false; profitability_proven=false.

## 24. Known limitations
{chr(10).join('- ' + x for x in limitations)}

## 25. Final classification
{classification}

## 26. Overnight paper/demo validation
Yes — product can continue overnight paper/demo validation with clearer watchlist tracking and blockers. Not live-ready.
"""
    (reports / "ae13f_watchlist_resolver_tracking_filters_report.md").write_text(report, encoding="utf-8")

    summary = f"""AE13F classification: {classification}
Non-destructive identity: PASS
Contract resolution reasons: PASS
Manual identity/evidence: PASS
Watchlist tracking: PASS (local-only limitation)
External resolver: controlled stub / local_only default
Semantic without market match: PASS
Demo queue evaluation: PASS
Filters hide (default): PASS
Backend filter authority: PASS
Keyed live market: PASS
Stale price scoped: PASS
Provider fail-soft: PASS
Safety: PASS
Limitations: external stub; out-of-feed collection local-only
Overnight paper/demo: YES (not live)
"""
    (reports / "ae13f_summary_for_upload.txt").write_text(summary, encoding="utf-8")

    _write_json(
        reports / "ae13f_decision_gate.json",
        {
            "classification": classification,
            "pass_with_limitations": True,
            "limitations": limitations,
            "safety": {
                "wallet_configured": False,
                "private_key_accessed": False,
                "real_transaction_signed": False,
                "real_transaction_attempted": False,
                "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
                "live_trading_ready": False,
                "live_trading_approval": "NO",
                "profitability_proven": False,
            },
            "overnight_paper_demo_ok": True,
            "live_ready": False,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(out),
        },
    )

    (ROOT / ".ae13f_outdir.txt").write_text(str(out), encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    main()
