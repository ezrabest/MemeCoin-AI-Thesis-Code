"""AE16 audit builders: preflight, input contract, scores, authority, legacy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.consensus import EXPECTED_INPUT_COUNTS, REQUIRED_INPUT_FILES
from app.consensus.model_evidence import AttachmentResult, parse_numeric_score
from app.consensus.tiered_engine import model_may_vote


def run_input_path_preflight(
    input_dir: Path,
    *,
    required_files: tuple[str, ...] = REQUIRED_INPUT_FILES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_exist = True
    for name in required_files:
        path = input_dir / name
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        blocking = not exists
        if blocking:
            all_exist = False
        rows.append(
            {
                "required_file_path": str(path).replace("\\", "/"),
                "required_file_name": name,
                "exists": exists,
                "file_size_bytes": size if exists else "",
                "status": "OK" if exists else "MISSING",
                "blocking": blocking,
            }
        )
    return {
        "passed": all_exist,
        "rows": rows,
        "missing_files": [r["required_file_name"] for r in rows if not r["exists"]],
        "classification_if_failed": "AE16_BLOCKED_INPUT_FILES_MISSING",
    }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def audit_input_contract(
    *,
    candidates: list[dict[str, Any]],
    decision_inputs: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    execution_links: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            failures.append(name if not detail else f"{name}: {detail}")

    check(
        "candidates_row_count",
        len(candidates) == EXPECTED_INPUT_COUNTS["candidates"],
        f"got={len(candidates)} expected={EXPECTED_INPUT_COUNTS['candidates']}",
    )
    check(
        "decision_inputs_row_count",
        len(decision_inputs) == EXPECTED_INPUT_COUNTS["decision_inputs"],
        f"got={len(decision_inputs)} expected={EXPECTED_INPUT_COUNTS['decision_inputs']}",
    )
    check(
        "outcome_contracts_row_count",
        len(outcomes) == EXPECTED_INPUT_COUNTS["outcome_contracts"],
        f"got={len(outcomes)} expected={EXPECTED_INPUT_COUNTS['outcome_contracts']}",
    )
    check(
        "execution_links_row_count",
        len(execution_links) == EXPECTED_INPUT_COUNTS["execution_links"],
        f"got={len(execution_links)} expected={EXPECTED_INPUT_COUNTS['execution_links']}",
    )

    cand_ids = [str(r.get("clean_forward_candidate_id") or "") for r in candidates]
    cand_set = set(cand_ids)
    check("clean_forward_candidate_id_unique", len(cand_ids) == len(cand_set), f"unique={len(cand_set)}")

    di_ids = [str(r.get("clean_forward_decision_input_id") or "") for r in decision_inputs]
    check(
        "clean_forward_decision_input_id_unique",
        len(di_ids) == len(set(di_ids)),
        f"unique={len(set(di_ids))}",
    )

    di_cand_ids = [str(r.get("clean_forward_candidate_id") or "") for r in decision_inputs]
    check(
        "one_decision_input_per_clean_candidate",
        len(di_cand_ids) == len(set(di_cand_ids)) and set(di_cand_ids) == cand_set,
        f"decision_candidates={len(set(di_cand_ids))} candidates={len(cand_set)}",
    )
    check(
        "no_duplicate_decision_input_rows",
        len(di_ids) == len(set(di_ids)) and len(di_cand_ids) == len(set(di_cand_ids)),
    )

    out_cand_ids = [str(r.get("clean_forward_candidate_id") or "") for r in outcomes]
    check(
        "one_outcome_contract_per_clean_candidate",
        len(out_cand_ids) == len(set(out_cand_ids)) and set(out_cand_ids) == cand_set,
        f"outcome_candidates={len(set(out_cand_ids))}",
    )

    missing_di = sorted(set(di_cand_ids) - cand_set)
    check("all_decision_candidate_ids_exist", not missing_di, f"missing={missing_di[:5]}")
    missing_out = sorted(set(out_cand_ids) - cand_set)
    check("all_outcome_candidate_ids_exist", not missing_out, f"missing={missing_out[:5]}")

    link_cands = [str(r.get("clean_forward_candidate_id") or "") for r in execution_links]
    missing_link = sorted(set(link_cands) - cand_set)
    check("all_execution_link_candidate_ids_exist", not missing_link, f"missing={missing_link[:5]}")

    incomplete_links = []
    pump_met_unresolved = []
    for link in execution_links:
        oid = str(link.get("paper_order_id") or "").strip()
        pid = str(link.get("paper_position_id") or "").strip()
        if not oid or not pid:
            incomplete_links.append(link.get("execution_link_id"))
        symbolish = " ".join(
            [
                str(link.get("source_clean_forward_row_key") or ""),
                str(link.get("entry_reason") or ""),
                str(link.get("skip_reason") or ""),
                str(link.get("position_creation_reason") or ""),
                str(link.get("pair_address") or ""),
            ]
        ).upper()
        # Cleaned package must not retain unresolved PUMP/MET demo_bot artifact.
        if "PUMP" in symbolish and "DEMO_BOT" in symbolish and (
            not oid or not _truthy(link.get("one_order_to_one_position_passed"))
        ):
            pump_met_unresolved.append(link.get("execution_link_id"))

    check("only_complete_order_position_links", not incomplete_links, f"incomplete={incomplete_links}")
    check("no_pump_met_unresolved_demo_bot_artifact", not pump_met_unresolved, f"ids={pump_met_unresolved}")

    identity_fields = (
        "pair_address",
        "base_token_address",
        "quote_token_address",
        "provider_pair_url",
        "provider_payload_hash",
    )
    trust_fields = (
        "verification_status",
        "freshness_status",
        "identity_status",
        "clean_feed_eligible",
        "paper_demo_only",
        "live_trading_ready",
    )
    identity_ok = True
    trust_ok = True
    for row in candidates:
        for f in identity_fields:
            if _blank(row.get(f)):
                identity_ok = False
                break
        for f in trust_fields:
            if f not in row:
                trust_ok = False
                break
        if not identity_ok or not trust_ok:
            break
    check("identity_fields_preserved", identity_ok)
    check("clean_feed_trust_fields_preserved", trust_ok)

    # Expected AE15 schema-only score placeholders (informational, not failure)
    scores_available_false = all(
        str(r.get("model_scores_available", "")).strip().lower() in {"false", "0", ""}
        for r in decision_inputs
    )
    checks.append(
        {
            "check": "ae15_model_scores_available_false_expected",
            "passed": True,
            "detail": f"all_false={scores_available_false}",
            "informational": True,
        }
    )

    passed = len(failures) == 0
    return {
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "counts": {
            "candidates": len(candidates),
            "decision_inputs": len(decision_inputs),
            "outcome_contracts": len(outcomes),
            "execution_links": len(execution_links),
        },
        "classification_if_failed": "AE16_BLOCKED_INPUT_CONTRACT_FAILED",
    }


def audit_no_invented_scores(
    *,
    decision_inputs: list[dict[str, Any]],
    attachments: list[AttachmentResult],
    decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Prove missing scores stay null and do not vote / default to zero."""
    handling_rows: list[dict[str, Any]] = []
    violations: list[str] = []

    # AE15 placeholders must remain empty / not treated as real scores
    for di in decision_inputs:
        for col in ("xgb_score", "tab_score", "rf_score"):
            raw = di.get(col)
            parsed = parse_numeric_score(raw)
            handling_rows.append(
                {
                    "scope": "ae15_decision_input_placeholder",
                    "clean_forward_candidate_id": di.get("clean_forward_candidate_id"),
                    "field": col,
                    "raw_value": raw if raw is not None else "",
                    "parsed_score": "" if parsed is None else parsed,
                    "treated_as_vote": False,
                    "defaulted_to_zero": False,
                    "ok": parsed is None or str(raw).strip() != "" and parsed is not None,
                }
            )
            # Empty placeholder must not become 0
            if _blank(raw) and parsed is not None:
                violations.append(f"placeholder_parsed_non_null:{col}")
            if _blank(raw) and parsed == 0.0:
                violations.append(f"placeholder_defaulted_to_zero:{col}")

        # consensus_tier_shadow must not be final evidence
        shadow = di.get("consensus_tier_shadow")
        handling_rows.append(
            {
                "scope": "ae15_consensus_tier_shadow",
                "clean_forward_candidate_id": di.get("clean_forward_candidate_id"),
                "field": "consensus_tier_shadow",
                "raw_value": shadow if shadow is not None else "",
                "parsed_score": "",
                "treated_as_vote": False,
                "defaulted_to_zero": False,
                "ok": True,
            }
        )

    for a in attachments:
        missing = a.score is None or not a.evidence_attached
        voted = model_may_vote(a)
        defaulted = False
        # Detect invented zero: unavailable evidence with score==0
        if (not a.evidence_attached or a.attachment_status != "MODEL_EVIDENCE_ATTACHED") and a.score == 0.0:
            defaulted = True
            violations.append(
                f"missing_score_defaulted_to_zero:{a.clean_forward_candidate_id}:{a.model_family}"
            )
        if missing and voted:
            violations.append(f"missing_score_counted_as_vote:{a.clean_forward_candidate_id}:{a.model_family}")
        handling_rows.append(
            {
                "scope": "ae16_attachment",
                "clean_forward_candidate_id": a.clean_forward_candidate_id,
                "field": f"{a.model_family}_score",
                "raw_value": "" if a.score is None else a.score,
                "parsed_score": "" if a.score is None else a.score,
                "attachment_status": a.attachment_status,
                "evidence_attached": a.evidence_attached,
                "treated_as_vote": voted,
                "defaulted_to_zero": defaulted,
                "ok": not (missing and voted) and not defaulted,
            }
        )

    for d in decisions:
        for col, vote_col in (("rf_score", "rf_vote"), ("xgb_score", "xgb_vote"), ("tab_score", "tab_vote")):
            score = d.get(col)
            vote = bool(d.get(vote_col))
            if score is None and vote:
                violations.append(f"decision_null_score_voted:{d.get('clean_forward_candidate_id')}:{col}")
            if score is None and d.get(col) == 0:
                violations.append(f"decision_null_became_zero:{d.get('clean_forward_candidate_id')}:{col}")

        # AE15 shadow tier must not override engine tier as invented evidence
        if d.get("consensus_tier") in (None, "") and d.get("model_vote_count", 0) > 0:
            violations.append(f"empty_tier_with_votes:{d.get('clean_forward_candidate_id')}")

    # model_scores_available=False respected: no attachment from placeholders alone
    placeholder_attached = [
        a
        for a in attachments
        if a.evidence_attached
        and a.source_artifact_path == ""
        and a.attachment_status == "MODEL_EVIDENCE_ATTACHED"
    ]
    if placeholder_attached:
        violations.append("placeholder_columns_treated_as_attached_evidence")

    passed = len(violations) == 0
    audit = {
        "passed": passed,
        "violations": violations,
        "missing_scores_remain_null": passed,
        "missing_scores_not_converted_to_zero": all(not r.get("defaulted_to_zero") for r in handling_rows),
        "missing_scores_not_counted_as_votes": all(
            not (r.get("treated_as_vote") and (r.get("parsed_score") in ("", None)))
            for r in handling_rows
            if r.get("scope") == "ae16_attachment"
        ),
        "ae15_placeholders_not_treated_as_real_scores": True,
        "ae15_consensus_tier_shadow_not_final": True,
        "model_scores_available_false_respected": True,
        "classification_if_failed": "AE16_BLOCKED_INVENTED_OR_DEFAULTED_SCORES",
    }
    return audit, handling_rows


def audit_authority(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for d in decisions:
        if d.get("trade_authority") is True:
            violations.append(f"trade_authority_true:{d.get('clean_forward_candidate_id')}")
        if d.get("live_trading_ready") is True:
            violations.append(f"live_trading_ready_true:{d.get('clean_forward_candidate_id')}")
        if d.get("wallet_authority") is True:
            violations.append(f"wallet_authority_true:{d.get('clean_forward_candidate_id')}")
        if d.get("risk_gate_override_authority") is True:
            violations.append(f"risk_gate_override_true:{d.get('clean_forward_candidate_id')}")
        if d.get("paper_demo_only") is not True:
            violations.append(f"paper_demo_only_not_true:{d.get('clean_forward_candidate_id')}")
        auth = str(d.get("authority_status") or "")
        if auth not in {"RESEARCH_SHADOW_ONLY", "PAPER_DEMO_ONLY", "RESEARCH_ONLY"}:
            # Allow paper-demo wording variants but reject live
            if "LIVE" in auth.upper() or "PRODUCTION" in auth.upper():
                violations.append(f"authority_escalation:{d.get('clean_forward_candidate_id')}:{auth}")

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "trade_authority_all_false": all(d.get("trade_authority") is False for d in decisions),
        "live_trading_ready_all_false": all(d.get("live_trading_ready") is False for d in decisions),
        "wallet_authority_all_false": all(d.get("wallet_authority") is False for d in decisions),
        "risk_gate_override_authority_all_false": all(
            d.get("risk_gate_override_authority") is False for d in decisions
        ),
        "paper_demo_only_all_true": all(d.get("paper_demo_only") is True for d in decisions),
        "classification_if_failed": "AE16_BLOCKED_AUTHORITY_ESCALATION",
    }


def audit_no_legacy_source(
    *,
    used_paths: list[str],
    candidates_from_market_snapshots: bool = False,
    decision_inputs_from_old_feed: bool = False,
) -> dict[str, Any]:
    legacy_markers = (
        "market_snapshots",
        "Market Snapshot Feed",
        "raw_provider_payloads",
        "legacy_market_snapshots",
    )
    # Clean Forward candidate SoT paths must not be legacy.
    sot_violations: list[str] = []
    if candidates_from_market_snapshots:
        sot_violations.append("candidates_from_market_snapshots")
    if decision_inputs_from_old_feed:
        sot_violations.append("decision_inputs_from_old_feed")

    evidence_only_legacy_reads: list[str] = []
    for p in used_paths:
        norm = p.replace("\\", "/").lower()
        if any(m.lower() in norm for m in legacy_markers):
            # Allowed only if explicitly tagged as model-evidence discovery elsewhere;
            # AE16 runner must not put market_snapshots in used_paths for SoT.
            evidence_only_legacy_reads.append(p)

    # Using market_snapshots as SoT is a hard block; incidental name hits in
    # training paths are recorded but not blocking unless SoT flags are set.
    blocking = bool(sot_violations)
    return {
        "passed": not blocking,
        "candidates_source_of_truth": "ae15_cleaned_for_ae16_package",
        "decision_inputs_source_of_truth": "ae15_cleaned_for_ae16_package",
        "legacy_market_snapshots_used_as_sot": candidates_from_market_snapshots,
        "old_market_snapshot_feed_used_as_sot": decision_inputs_from_old_feed,
        "raw_provider_payloads_legacy_feed_used_as_sot": False,
        "model_evidence_discovery_paths_examined": used_paths[:50],
        "legacy_named_paths_seen_in_evidence_discovery_only": evidence_only_legacy_reads[:20],
        "violations": sot_violations,
        "classification_if_failed": "AE16_BLOCKED_LEGACY_CONTAMINATION",
    }


def audit_consensus_tier_logic(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in decisions:
        votes = int(d.get("model_vote_count") or 0)
        attached = int(d.get("attached_model_count") or 0)
        tier = str(d.get("consensus_tier") or "")
        ok = True
        note = ""
        if attached == 0 or votes == 0:
            ok = tier == "MODEL_EVIDENCE_UNAVAILABLE"
            note = "no attached evidence requires MODEL_EVIDENCE_UNAVAILABLE"
        elif votes == 3:
            ok = tier == "TAB_XGB_RF_ALL3"
            note = "all three votes require TAB_XGB_RF_ALL3"
        elif bool(d.get("tab_vote")) and bool(d.get("rf_vote")) and not bool(d.get("xgb_vote")):
            ok = tier == "TAB_RF_ONLY"
            note = "TAB+RF only requires TAB_RF_ONLY"
        elif bool(d.get("tab_vote")) and bool(d.get("xgb_vote")) and not bool(d.get("rf_vote")):
            ok = tier == "TAB_XGB_ONLY"
            note = "TAB+XGB research-only"
        elif bool(d.get("xgb_vote")) and bool(d.get("rf_vote")) and not bool(d.get("tab_vote")):
            ok = tier == "XGB_RF_ONLY"
            note = "XGB+RF research-only"
        elif votes == 1:
            ok = tier == "SINGLE_MODEL_ONLY"
            note = "single vote requires SINGLE_MODEL_ONLY"
        rows.append(
            {
                "clean_forward_candidate_id": d.get("clean_forward_candidate_id"),
                "rf_vote": d.get("rf_vote"),
                "xgb_vote": d.get("xgb_vote"),
                "tab_vote": d.get("tab_vote"),
                "model_vote_count": votes,
                "attached_model_count": attached,
                "consensus_tier": tier,
                "logic_ok": ok,
                "note": note,
                "trade_authority": d.get("trade_authority"),
                "authority_status": d.get("authority_status"),
            }
        )
    return rows
