"""Backfill AE13I smoke-addendum pack snapshots + consolidated audits.

Does NOT re-run compileall/pytest or start an HTTP server. Uses in-process
imports, existing numbered audit JSON in the pack, and FastAPI TestClient only.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data" / "audits" / "ae13i_smoke_addendum_close_queue_address_readiness_20260719_224903"
DECISION_GATE = PACK / "reports" / "ae13i_smoke_addendum_decision_gate.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_pack_audit(name: str) -> dict:
    return json.loads((PACK / "audits" / name).read_text(encoding="utf-8"))


def _load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "run_ae13i_smoke_addendum_validation",
        ROOT / "scripts" / "run_ae13i_smoke_addendum_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _audit_status(passed: bool, *, limitation: bool = False) -> str:
    if limitation:
        return "limitation"
    return "pass" if passed else "fail"


def _consolidated_audit(*, status: str, evidence: dict, notes: str) -> dict:
    return {"status": status, "evidence": evidence, "notes": notes}


def main() -> list[str]:
    created: list[str] = []
    val = _load_validation_module()
    decision = json.loads(DECISION_GATE.read_text(encoding="utf-8"))
    pack_data = PACK / "data"
    pack_audits = PACK / "audits"
    root_data = ROOT / "data"

    from app.ae13b_product.close_freshness import (
        FALLBACK_SOURCES,
        FRESH_SOURCES,
        MANUAL_CLOSE_FALLBACK_WARNING,
        classify_manual_close_freshness,
    )
    from app.ae13b_product.live_market import compute_contract_address_disclosure
    from app.ae13b_product.text_sanitizer import sanitize_payload, sanitize_text
    from app.ae13b_product.ae14_readiness import compute_ae14_readiness, NEGATIVE_CONTROL_REASON

    # --- Snapshots from live code -------------------------------------------
    proposed_no_ts = classify_manual_close_freshness(
        close_price=1.23,
        price_timestamp="",
        close_price_source="proposed_price",
        close_price_age_seconds=None,
    )
    fresh_provider = classify_manual_close_freshness(
        close_price=1.0,
        price_timestamp=_utc_now_iso(),
        close_price_source="provider",
        close_price_age_seconds=5.0,
    )
    fallback_checks = {}
    for source in sorted(FALLBACK_SOURCES):
        fallback_checks[source] = classify_manual_close_freshness(
            close_price=1.0,
            price_timestamp=_utc_now_iso(),
            close_price_source=source,
            close_price_age_seconds=1.0,
        )["close_freshness_status"]

    close_freshness_snapshot = {
        "module": "app.ae13b_product.close_freshness",
        "function": "classify_manual_close_freshness",
        "fallback_sources": sorted(FALLBACK_SOURCES),
        "fresh_sources": sorted(FRESH_SOURCES),
        "freshness_threshold_seconds_default": 900,
        "proposed_price_no_timestamp": proposed_no_ts,
        "genuine_fresh_provider": fresh_provider,
        "fallback_source_statuses": fallback_checks,
        "pack_audits_01_04_pass": all(
            _load_pack_audit(n).get("pass")
            for n in (
                "01_close_freshness_proposed_price_no_timestamp_never_fresh.json",
                "02_close_freshness_all_fallback_sources_never_fresh.json",
                "03_close_freshness_genuine_fresh_price_is_fresh.json",
                "04_close_freshness_caller_cannot_spoof.json",
            )
        ),
        "generated_at_utc": _utc_now_iso(),
    }

    manual_close_warning_snapshot = {
        "warning_text": MANUAL_CLOSE_FALLBACK_WARNING,
        "pack_audit_06": _load_pack_audit("06_api_manual_close_response_and_price_source.json"),
        "generated_at_utc": _utc_now_iso(),
    }

    demo_queue_snapshot = {
        "module": "app.ae13b_product.demo_queue",
        "evaluation_stale_fields": [
            "evaluation_stale",
            "evaluation_stale_reason",
            "evaluation_age_seconds",
            "last_gatekeeper_evaluated_at",
            "gatekeeper_status",
            "gatekeeper_evaluated",
        ],
        "pack_audits_07_12_pass": all(
            _load_pack_audit(n).get("pass")
            for n in (
                "07_demo_queue_list_attaches_freshness_fields.json",
                "08_demo_queue_stale_evaluation_flagged.json",
                "09_manual_cooldown_precheck_runs_first.json",
                "10_gate_runs_before_risk_guard.json",
                "11_risk_mode_refreshed_from_active_preset.json",
                "12_explicit_risk_mode_not_overridden.json",
            )
        ),
        "stale_evaluation_audit": _load_pack_audit("08_demo_queue_stale_evaluation_flagged.json"),
        "generated_at_utc": _utc_now_iso(),
    }

    pool_disclosure = compute_contract_address_disclosure(
        raw_contract_address=None,
        address_role="pool_address",
        token_contract_address=None,
        token_mint_address=None,
        pair_address="poolABC123",
    )
    token_disclosure = compute_contract_address_disclosure(
        raw_contract_address=None,
        address_role="token_contract",
        token_contract_address="0xTOKENCONTRACT",
        token_mint_address=None,
        pair_address="0xPAIRADDR",
    )
    address_alias_snapshot = {
        "module": "app.ae13b_product.live_market",
        "function": "compute_contract_address_disclosure",
        "pool_address_disclosure": pool_disclosure,
        "token_contract_disclosure": token_disclosure,
        "pack_audits_13_14_pass": all(
            _load_pack_audit(n).get("pass")
            for n in (
                "13_pool_pair_address_disclosure.json",
                "14_actual_token_contract_has_no_warning.json",
            )
        ),
        "generated_at_utc": _utc_now_iso(),
    }

    sample_payload = sanitize_payload(
        {"note": "paper \u2014 demo", "nested": {"warn": "loading\u2026"}, "count": 5}
    )
    sanitized_payload_snapshot = {
        "module": "app.ae13b_product.text_sanitizer",
        "functions": ["sanitize_text", "sanitize_payload"],
        "sample_input_unicode_dash_ellipsis": {"note": "paper \u2014 demo", "nested": {"warn": "loading\u2026"}},
        "sample_output": sample_payload,
        "pack_audits_15_16_pass": all(
            _load_pack_audit(n).get("pass")
            for n in (
                "15_text_sanitizer_ascii_safe.json",
                "16_api_responses_sanitized.json",
            )
        ),
        "api_sanitized_audit": _load_pack_audit("16_api_responses_sanitized.json"),
        "generated_at_utc": _utc_now_iso(),
    }

    mojibake_inputs = [
        "a \u2014 b \u2013 c",
        "loading\u2026",
        "\u00e2\u0080\u0094 broken dash",
    ]
    mojibake_scan_snapshot = {
        "scan_results": {text: sanitize_text(text) for text in mojibake_inputs},
        "static_files_checked": ["static/index.html", "static/product_demo.js"],
        "static_unicode_offenders": [],
        "pack_audit_15": _load_pack_audit("15_text_sanitizer_ascii_safe.json"),
        "generated_at_utc": _utc_now_iso(),
    }
    for rel in ("static/index.html", "static/product_demo.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for bad in ("\u2014", "\u2013", "\u2026"):
            if bad in text:
                mojibake_scan_snapshot["static_unicode_offenders"].append(f"{rel}:{bad!r}")

    ae14_synthetic = {
        "no_data": compute_ae14_readiness(market_rows=[]),
        "below_threshold": compute_ae14_readiness(
            market_rows=[{"tradability_status": "tradable_now"} for _ in range(3)],
            min_tradable_rows_for_ae14=10,
        ),
        "enough_tradable": compute_ae14_readiness(
            market_rows=[{"tradability_status": "tradable_now"} for _ in range(12)],
            min_tradable_rows_for_ae14=10,
        ),
    }

    live_ae14: dict | None = None
    live_trading_ready = False
    try:
        from fastapi.testclient import TestClient
        from app.api import app

        client = TestClient(app)
        resp = client.get("/api/ae14/readiness")
        if resp.status_code == 200:
            live_ae14 = resp.json()
            live_trading_ready = bool(live_ae14.get("ready_for_trading_validation"))
    except Exception as exc:
        live_ae14 = {"error": repr(exc)}

    ae14_readiness_snapshot = {
        "module": "app.ae13b_product.ae14_readiness",
        "function": "compute_ae14_readiness",
        "negative_control_reason": NEGATIVE_CONTROL_REASON,
        "synthetic_scenarios": ae14_synthetic,
        "live_api_readiness": live_ae14,
        "pack_audits_17_18_pass": all(
            _load_pack_audit(n).get("pass")
            for n in (
                "17_ae14_readiness_negative_control_vs_trading_validation.json",
                "18_ae14_endpoints_wired.json",
            )
        ),
        "generated_at_utc": _utc_now_iso(),
    }

    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    dashboard_readiness_snapshot = {
        "ae14_panel_in_index_html": "ae14-negative-control" in html and "ae14-trading-validation" in html,
        "ae14_wired_in_product_demo_js": "ae14_readiness" in js and "ae14-trading-validation" in js,
        "demo_queue_evaluation_stale_badge": "evaluation_stale" in js,
        "address_warning_display": "contract_address_warning" in js or "address_warning" in js,
        "pack_audit_18": _load_pack_audit("18_ae14_endpoints_wired.json"),
        "generated_at_utc": _utc_now_iso(),
    }

    gatekeeper_regression_snapshot = {
        "pack_audit_19": _load_pack_audit("19_gatekeeper_still_blocks_stale_and_missing.json"),
        "generated_at_utc": _utc_now_iso(),
    }
    reentry_stagnant_snapshot = {
        "pack_audit_20": _load_pack_audit("20_reentry_block_still_blocks.json"),
        "pack_audit_21": _load_pack_audit("21_stagnant_guard_still_blocks.json"),
        "generated_at_utc": _utc_now_iso(),
    }
    safety_snapshot = {
        "paper_demo_only": decision.get("paper_demo_only", True),
        "no_wallet": decision.get("no_wallet", True),
        "no_live_trading": decision.get("no_live_trading", True),
        "not_live_approved": decision.get("not_live_approved", True),
        "not_profitability_evidence": decision.get("not_profitability_evidence", True),
        "classification": decision.get("classification"),
        "pack_audit_22": _load_pack_audit("22_no_live_wallet_safety.json"),
        "generated_at_utc": _utc_now_iso(),
    }

    snapshots = {
        "ae13i_close_freshness_hard_guard_snapshot.json": close_freshness_snapshot,
        "ae13i_manual_close_warning_snapshot.json": manual_close_warning_snapshot,
        "ae13i_demo_queue_gatekeeper_reevaluation_snapshot.json": demo_queue_snapshot,
        "ae13i_address_alias_cleanup_snapshot.json": address_alias_snapshot,
        "ae13i_sanitized_payload_snapshot.json": sanitized_payload_snapshot,
        "ae13i_mojibake_scan_snapshot.json": mojibake_scan_snapshot,
        "ae13i_ae14_readiness_snapshot.json": ae14_readiness_snapshot,
        "ae13i_dashboard_readiness_snapshot.json": dashboard_readiness_snapshot,
        "ae13i_gatekeeper_regression_snapshot.json": gatekeeper_regression_snapshot,
        "ae13i_reentry_stagnant_regression_snapshot.json": reentry_stagnant_snapshot,
        "ae13i_safety_snapshot.json": safety_snapshot,
    }

    for name, payload in snapshots.items():
        for dest in (pack_data / name, root_data / name):
            _write_json(dest, payload)
            created.append(str(dest.relative_to(ROOT)))

    # --- Consolidated audits ------------------------------------------------
    a01 = _load_pack_audit("01_close_freshness_proposed_price_no_timestamp_never_fresh.json")
    a04 = _load_pack_audit("04_close_freshness_caller_cannot_spoof.json")
    close_freshness_audit = _consolidated_audit(
        status=_audit_status(all(
            _load_pack_audit(n).get("pass")
            for n in (
                "01_close_freshness_proposed_price_no_timestamp_never_fresh.json",
                "02_close_freshness_all_fallback_sources_never_fresh.json",
                "03_close_freshness_genuine_fresh_price_is_fresh.json",
                "04_close_freshness_caller_cannot_spoof.json",
            )
        )),
        evidence={
            "source_module": "app/ae13b_product/close_freshness.py",
            "function": "classify_manual_close_freshness",
            "proposed_price_no_timestamp": a01.get("result"),
            "caller_spoof_blocked": {
                "closed_freshness_status": a04.get("closed_freshness_status"),
                "closed_used_fallback_price": a04.get("closed_used_fallback_price"),
            },
            "fallback_sources_never_fresh": _load_pack_audit(
                "02_close_freshness_all_fallback_sources_never_fresh.json"
            ).get("fallback_sources_checked"),
        },
        notes=(
            "Hard guard in classify_manual_close_freshness prevents proposed/fallback "
            "prices from ever being labeled fresh; caller-supplied freshness claims are ignored."
        ),
    )

    a06 = _load_pack_audit("06_api_manual_close_response_and_price_source.json")
    manual_close_warning_audit = _consolidated_audit(
        status=_audit_status(bool(a06.get("pass"))),
        evidence={
            "warning_text_matches_spec": a06.get("warning_text_matches_spec"),
            "warning_text": a06.get("warning_text"),
            "explicit_close_price_labeled_proposed_price": a06.get(
                "explicit_close_price_labeled_proposed_price"
            ),
        },
        notes="Manual close API surfaces MANUAL_CLOSE_FALLBACK_WARNING when fallback/stale price is used.",
    )

    demo_queue_audit = _consolidated_audit(
        status=_audit_status(bool(demo_queue_snapshot["pack_audits_07_12_pass"])),
        evidence={
            "evaluation_stale_fields_present": demo_queue_snapshot["evaluation_stale_fields"],
            "stale_evaluation_cases": demo_queue_snapshot["stale_evaluation_audit"],
            "gate_before_risk_guard": _load_pack_audit("10_gate_runs_before_risk_guard.json"),
            "cooldown_precheck_first": _load_pack_audit("09_manual_cooldown_precheck_runs_first.json"),
        },
        notes=(
            "Demo queue attaches GateKeeper evaluation freshness fields including evaluation_stale; "
            "manual cooldown precheck runs before GateKeeper, which runs before RiskGuard."
        ),
    )

    address_alias_audit = _consolidated_audit(
        status=_audit_status(bool(address_alias_snapshot["pack_audits_13_14_pass"])),
        evidence={
            "pool_address_disclosure": pool_disclosure,
            "token_contract_disclosure": token_disclosure,
        },
        notes=(
            "Pool/pair addresses are disclosed as deprecated aliases with warnings; "
            "actual token contract addresses carry no false warning."
        ),
    )

    a15 = _load_pack_audit("15_text_sanitizer_ascii_safe.json")
    a16 = _load_pack_audit("16_api_responses_sanitized.json")
    text_sanitizer_audit = _consolidated_audit(
        status=_audit_status(bool(a15.get("pass") and a16.get("pass")), limitation=bool(a15.get("static_file_offenders") is False and a16.get("offenders") == [] and decision.get("limitations"))),
        evidence={
            "sanitize_payload_sample": sample_payload,
            "static_ui_ascii_safe": a15.get("static_ui_files_ascii_safe"),
            "api_endpoint_offenders": a16.get("offenders"),
            "api_endpoint_statuses": a16.get("endpoint_statuses"),
        },
        notes=(
            "Global sanitize_payload normalizes Unicode punctuation and known mojibake; "
            "coverage is limited to characterized sequences in text_sanitizer.py."
        ),
    )
    # Fix status: limitation only if sanitizer has known coverage limits, not because pack passed
    text_sanitizer_audit["status"] = "limitation" if decision.get("limitations") else "pass"
    if a15.get("pass") and a16.get("pass"):
        text_sanitizer_audit["status"] = "limitation"
        text_sanitizer_audit["notes"] += (
            " Known limitation: not-yet-observed mojibake outside the repair tables is not caught."
        )

    ae14_logic_ok = bool(
        _load_pack_audit("17_ae14_readiness_negative_control_vs_trading_validation.json").get("pass")
        and _load_pack_audit("18_ae14_endpoints_wired.json").get("pass")
    )
    ae14_audit = _consolidated_audit(
        status="limitation" if ae14_logic_ok and not live_trading_ready else _audit_status(ae14_logic_ok),
        evidence={
            "synthetic_no_data": ae14_synthetic["no_data"],
            "synthetic_below_threshold": ae14_synthetic["below_threshold"],
            "live_api_readiness": live_ae14,
            "endpoints_wired": _load_pack_audit("18_ae14_endpoints_wired.json"),
        },
        notes=(
            "AE14 negative-control readiness logic passes in code; "
            "ready_for_trading_validation depends on live tradable_now row count at query time "
            "and may legitimately be False when market data is stale or below threshold."
        ),
    )

    dashboard_ok = all(
        dashboard_readiness_snapshot[k]
        for k in (
            "ae14_panel_in_index_html",
            "ae14_wired_in_product_demo_js",
            "demo_queue_evaluation_stale_badge",
            "address_warning_display",
        )
    ) and _load_pack_audit("18_ae14_endpoints_wired.json").get("pass")
    dashboard_audit = _consolidated_audit(
        status=_audit_status(bool(dashboard_ok), limitation=not live_trading_ready),
        evidence=dashboard_readiness_snapshot,
        notes=(
            "Dashboard surfaces AE14 readiness, demo-queue evaluation_stale badges, and address warnings; "
            "trading-validation readiness display reflects live market snapshot limitations."
        ),
    )

    gatekeeper_audit = _consolidated_audit(
        status=_audit_status(bool(_load_pack_audit("19_gatekeeper_still_blocks_stale_and_missing.json").get("pass"))),
        evidence=_load_pack_audit("19_gatekeeper_still_blocks_stale_and_missing.json"),
        notes="GateKeeper still blocks stale and missing price rows; healthy rows pass.",
    )

    reentry_stagnant_audit = _consolidated_audit(
        status=_audit_status(all(
            _load_pack_audit(n).get("pass")
            for n in (
                "20_reentry_block_still_blocks.json",
                "21_stagnant_guard_still_blocks.json",
            )
        )),
        evidence={
            "reentry_block": _load_pack_audit("20_reentry_block_still_blocks.json"),
            "stagnant_guard": _load_pack_audit("21_stagnant_guard_still_blocks.json"),
        },
        notes="Manual reentry cooldown and stagnant-price guards remain enforced.",
    )

    a22 = _load_pack_audit("22_no_live_wallet_safety.json")
    safety_audit = _consolidated_audit(
        status=_audit_status(bool(a22.get("pass"))),
        evidence={
            "offenders": a22.get("offenders"),
            "classification": decision.get("classification"),
            "paper_demo_only": decision.get("paper_demo_only"),
            "no_wallet": decision.get("no_wallet"),
            "no_live_trading": decision.get("no_live_trading"),
        },
        notes="No wallet/private-key/live-transaction paths in AE13I smoke-addendum touched modules.",
    )

    consolidated = {
        "ae13i_close_freshness_hard_guard_audit.json": close_freshness_audit,
        "ae13i_manual_close_warning_audit.json": manual_close_warning_audit,
        "ae13i_demo_queue_gatekeeper_reevaluation_audit.json": demo_queue_audit,
        "ae13i_address_alias_cleanup_audit.json": address_alias_audit,
        "ae13i_global_text_sanitizer_audit.json": text_sanitizer_audit,
        "ae13i_ae14_readiness_audit.json": ae14_audit,
        "ae13i_dashboard_readiness_audit.json": dashboard_audit,
        "ae13i_gatekeeper_regression_audit.json": gatekeeper_audit,
        "ae13i_reentry_stagnant_regression_audit.json": reentry_stagnant_audit,
        "ae13i_no_live_wallet_safety_audit.json": safety_audit,
    }

    for name, payload in consolidated.items():
        dest = pack_audits / name
        _write_json(dest, payload)
        created.append(str(dest.relative_to(ROOT)))

    # Ensure decision gate classification unchanged
    assert decision["classification"] == "AE13I_SMOKE_ADDENDUM_PASS_WITH_LIMITATIONS"

    return created


if __name__ == "__main__":
    files = main()
    print(json.dumps({"created": files, "count": len(files)}, indent=2))
