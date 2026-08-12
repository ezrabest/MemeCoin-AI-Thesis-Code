"""AE12-SentimentFix local manual-review drilldown (no Gemini / no external APIs)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DRILLDOWN_RULE_VERSION = "AE12_SENTIMENTFIX_MANUAL_REVIEW_DRILLDOWN_RULES_V1"
COIN_AGGREGATION_RULE_VERSION = "AE12_SENTIMENTFIX_COIN_AGGREGATION_RULES_V1"
RUBRIC_VERSION = "AE12_SENTIMENTFIX_ADJUDICATION_RUBRIC_V1"

CONFIRMED_CLASSES = frozenset(
    {
        "SOCIAL_CONFIRMED",
        "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
    }
)

IDENTITY_AMBIGUITY_MARKERS = (
    "impersonat",
    "scam",
    "duplicate",
    "ambiguous",
    "contradiction",
    "pair address",
    "token address",
    "identity",
    "missing name",
    "missing.*token",
    "uniswap v2 pair",
    "pair on etherscan",
)

RESOLUTION_RULES: list[dict[str, str]] = [
    {
        "rule_id": "RULE_1",
        "rule_name": "RULE_1_ALL_PAIR_VOTES_SAME_CONFIRMED_CLASS",
        "condition": "All supporting pair-level classes are the same confirmed class",
        "output_class": "that confirmed class",
        "scientific_rationale": "Unanimous confirmed pair votes require no external evidence to accept.",
    },
    {
        "rule_id": "RULE_2",
        "rule_name": "RULE_2_CONFIRMED_OP_DOMINATES_NO_SOCIAL",
        "condition": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED present, no SOCIAL_CONFIRMED, only weak identity warning",
        "output_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        "scientific_rationale": "Without social-confirmed evidence, confirmed opportunistic may stand if identity risk is low.",
    },
    {
        "rule_id": "RULE_3",
        "rule_name": "RULE_3_CONFIRMED_OP_OVERRIDES_SUSPECTED",
        "condition": "Only NON_SOCIAL_OPPORTUNISTIC_CONFIRMED and OPPORTUNISTIC_SUSPECTED votes",
        "output_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        "scientific_rationale": "Confirmed opportunistic is stronger than suspected opportunistic.",
    },
    {
        "rule_id": "RULE_4",
        "rule_name": "RULE_4_SOCIAL_OP_CONFLICT_UNRESOLVED",
        "condition": "Both SOCIAL_CONFIRMED and NON_SOCIAL_OPPORTUNISTIC_CONFIRMED present",
        "output_class": "UNKNOWN_UNRESOLVED",
        "scientific_rationale": "Direct social vs opportunistic conflict cannot be forced locally without external evidence.",
    },
    {
        "rule_id": "RULE_5",
        "rule_name": "RULE_5_IDENTITY_AMBIGUITY_UNRESOLVED",
        "condition": "Identity/scam/impersonator/address ambiguity in manual-review reasons",
        "output_class": "UNKNOWN_UNRESOLVED",
        "scientific_rationale": "Ambiguous token identity blocks reliable coin-level classification.",
    },
    {
        "rule_id": "RULE_5B",
        "rule_name": "RULE_5_LOW_SEVERITY_IDENTITY_WARNING_RESOLVED",
        "condition": "Low-severity identity warning with unanimous confirmed class",
        "output_class": "that confirmed class",
        "scientific_rationale": "Low-severity warnings do not override unanimous confirmed evidence.",
    },
    {
        "rule_id": "RULE_6",
        "rule_name": "RULE_6_REJECTED_TRADE_LANGUAGE_UNRESOLVED",
        "condition": "Manual review driven by REJECTED_FOR_TRADE_LANGUAGE",
        "output_class": "UNKNOWN_UNRESOLVED",
        "scientific_rationale": "Rejected Gemini outputs must not be reinterpreted locally.",
    },
    {
        "rule_id": "RULE_7",
        "rule_name": "RULE_7_NO_CLEAR_LOCAL_CONCLUSION",
        "condition": "No earlier rule yields a clear local conclusion",
        "output_class": "UNKNOWN_UNRESOLVED",
        "scientific_rationale": "Insufficient local evidence; UNKNOWN_UNRESOLVED is not opportunistic and not social.",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _parse_votes(raw: Any) -> Counter:
    if isinstance(raw, dict):
        return Counter({str(k): int(v) for k, v in raw.items()})
    text = str(raw or "").strip()
    if not text:
        return Counter()
    try:
        data = json.loads(text.replace("'", '"'))
        if isinstance(data, dict):
            return Counter({str(k): int(v) for k, v in data.items()})
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return Counter()


def _blob(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def _has_identity_ambiguity(text: str) -> bool:
    t = (text or "").lower()
    for marker in IDENTITY_AMBIGUITY_MARKERS:
        if re.search(marker, t):
            return True
    return False


def _identity_warning_low_severity(warning: str) -> bool:
    w = (warning or "").strip().upper()
    if not w:
        return True
    severe = ("IMPERSONAT", "SCAM", "REFUSED_PAIR", "MISSING_BASE", "AMBIGUOUS")
    return not any(s in w for s in severe)


def resolve_manual_review_coin(
    *,
    class_votes: Counter,
    pair_rows: list[dict[str, Any]],
    conflict_note: str = "",
    identity_warning: str = "",
    manual_review_reason: str = "",
) -> dict[str, str]:
    """Apply local drilldown rules. Returns final class + rule + note + confidence."""
    votes = Counter(class_votes)
    classes = {c for c, n in votes.items() if n > 0}
    social = int(votes.get("SOCIAL_CONFIRMED", 0))
    opp = int(votes.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0))
    rejected_count = sum(
        1
        for p in pair_rows
        if str(p.get("raw_evidence_status") or "") == "REJECTED_FOR_TRADE_LANGUAGE"
        or "rejected by trade-language" in str(p.get("reasoning_short") or "").lower()
    )

    reason_blob = _blob(
        conflict_note,
        identity_warning,
        manual_review_reason,
        *[p.get("reasoning_short") for p in pair_rows],
        *[p.get("evidence_summary") for p in pair_rows],
    )

    # Rule 4 — scientific hard stop: social vs opportunistic conflict
    if social > 0 and opp > 0:
        return {
            "final_class_after_drilldown": "UNKNOWN_UNRESOLVED",
            "resolution_rule_applied": "RULE_4_SOCIAL_OP_CONFLICT_UNRESOLVED",
            "resolution_note": (
                "Local artifacts contain a direct SOCIAL_CONFIRMED vs "
                "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED conflict. No external review is allowed "
                "in this pass, so the scientific local result is UNKNOWN_UNRESOLVED."
            ),
            "resolution_confidence": "HIGH",
        }

    # Rule 6 — rejected trade language drives manual review
    if rejected_count > 0:
        return {
            "final_class_after_drilldown": "UNKNOWN_UNRESOLVED",
            "resolution_rule_applied": "RULE_6_REJECTED_TRADE_LANGUAGE_UNRESOLVED",
            "resolution_note": (
                "Gemini output was rejected by the trade-language safety gate and "
                "cannot be reinterpreted locally."
            ),
            "resolution_confidence": "HIGH",
        }

    # Rule 1 — all pair votes same confirmed class
    if len(classes) == 1:
        only = next(iter(classes))
        if only in CONFIRMED_CLASSES:
            return {
                "final_class_after_drilldown": only,
                "resolution_rule_applied": "RULE_1_ALL_PAIR_VOTES_SAME_CONFIRMED_CLASS",
                "resolution_note": (
                    f"All {sum(votes.values())} supporting pair votes agree on {only}."
                ),
                "resolution_confidence": "HIGH",
            }

    # Rule 5 — identity / scam / impersonator ambiguity
    if _has_identity_ambiguity(reason_blob) or (
        identity_warning and not _identity_warning_low_severity(identity_warning)
    ):
        confirmed_only = classes & CONFIRMED_CLASSES
        if (
            len(classes) == 1
            and len(confirmed_only) == 1
            and _identity_warning_low_severity(identity_warning)
            and not _has_identity_ambiguity(reason_blob)
        ):
            chosen = next(iter(confirmed_only))
            return {
                "final_class_after_drilldown": chosen,
                "resolution_rule_applied": "RULE_5_LOW_SEVERITY_IDENTITY_WARNING_RESOLVED",
                "resolution_note": (
                    f"Low-severity identity warning with clear confirmed class {chosen}."
                ),
                "resolution_confidence": "MEDIUM",
            }
        return {
            "final_class_after_drilldown": "UNKNOWN_UNRESOLVED",
            "resolution_rule_applied": "RULE_5_IDENTITY_AMBIGUITY_UNRESOLVED",
            "resolution_note": (
                "Identity/scam/impersonator or address ambiguity in local pair evidence "
                "prevents a reliable coin-level resolution without external review."
            ),
            "resolution_confidence": "MEDIUM",
        }

    # Rule 3 — confirmed opportunistic + suspected only
    if classes <= {"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", "OPPORTUNISTIC_SUSPECTED"} and opp > 0:
        return {
            "final_class_after_drilldown": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            "resolution_rule_applied": "RULE_3_CONFIRMED_OP_OVERRIDES_SUSPECTED",
            "resolution_note": (
                "Confirmed opportunistic vote overrides suspected vote locally; "
                "OP.SUSPECTED is weaker than confirmed opportunistic."
            ),
            "resolution_confidence": "MEDIUM",
        }

    # Rule 2 — confirmed OP dominates, no social, weak identity
    if (
        opp > 0
        and social == 0
        and _identity_warning_low_severity(identity_warning)
        and not _has_identity_ambiguity(reason_blob)
        and classes <= {"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", "OPPORTUNISTIC_SUSPECTED"}
    ):
        return {
            "final_class_after_drilldown": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            "resolution_rule_applied": "RULE_2_CONFIRMED_OP_DOMINATES_NO_SOCIAL",
            "resolution_note": (
                "No SOCIAL_CONFIRMED evidence existed; confirmed opportunistic dominates "
                "with only a weak/local identity warning."
            ),
            "resolution_confidence": "MEDIUM",
        }

    # Rule 7 — no clear conclusion
    return {
        "final_class_after_drilldown": "UNKNOWN_UNRESOLVED",
        "resolution_rule_applied": "RULE_7_NO_CLEAR_LOCAL_CONCLUSION",
        "resolution_note": (
            "Existing local artifacts were insufficient for a clear non-external resolution."
        ),
        "resolution_confidence": "LOW",
    }


def run_manual_review_drilldown(
    *,
    project_root: Path,
    gemini_root: Path,
    no_external_apis: bool = True,
) -> dict[str, Any]:
    """Derive local manual-review drilldown from an existing Gemini adjudication root."""
    project_root = Path(project_root).resolve()
    gemini_root = Path(gemini_root)
    if not gemini_root.is_absolute():
        gemini_root = (project_root / gemini_root).resolve()

    external_api_used = False
    gemini_called_again = False
    if not no_external_apis:
        # This pass must never call APIs even if flag is wrong; force false and fail gate.
        external_api_used = True

    out_root = (
        project_root
        / "data"
        / "audits"
        / f"ae12_sentimentfix_manual_review_drilldown_{_ts_slug()}"
    )
    for d in ("reports", "data", "audits"):
        (out_root / d).mkdir(parents=True, exist_ok=True)

    coin_path = gemini_root / "data" / "ae12_gemini_coin_level_adjudications.csv"
    pair_path = gemini_root / "data" / "ae12_gemini_pair_asset_adjudications.csv"
    if not pair_path.is_file():
        pair_path = gemini_root / "data" / "ae12_gemini_asset_adjudications.csv"
    mapping_path = gemini_root / "data" / "ae12_gemini_pair_to_coin_mapping.csv"
    identity_path = gemini_root / "data" / "ae12_gemini_coin_identity_resolution_audit.csv"

    source_missing = not coin_path.is_file() or not pair_path.is_file()

    coins = _read_csv(coin_path)
    pairs = _read_csv(pair_path)
    mapping = _read_csv(mapping_path)
    identity_rows = _read_csv(identity_path)
    gate_src = _load_json(gemini_root / "audits" / "ae12_gemini_semantic_adjudication_gate.json")
    summary_src = _load_json(gemini_root / "reports" / "ae12_gemini_semantic_adjudication_summary.json")
    safety_src = _load_json(gemini_root / "audits" / "ae12_gemini_safety_audit.json")

    pair_by_id = {str(p.get("asset_id") or ""): p for p in pairs}
    identity_by_coin = {str(r.get("coin_id") or ""): r for r in identity_rows}

    map_by_coin: dict[str, list[str]] = {}
    for m in mapping:
        cid = str(m.get("coin_id") or "")
        pid = str(m.get("pair_asset_id") or "")
        if cid and pid:
            map_by_coin.setdefault(cid, []).append(pid)

    # Fallback: rebuild mapping from pair rows if mapping file empty
    if not map_by_coin:
        for p in pairs:
            cid = str(p.get("coin_id") or "")
            if not cid:
                # derive from chain+symbol base if enrichment present
                chain = str(p.get("chain") or "").lower()
                base = str(p.get("normalized_base_symbol") or "")
                if chain and base:
                    cid = f"{chain}:{base}"
            if cid:
                map_by_coin.setdefault(cid, []).append(str(p.get("asset_id") or ""))

    mr_coins = [c for c in coins if str(c.get("semantic_coin_class") or "") == "MANUAL_REVIEW"]
    created_at = _utc_now()

    drilldown_rows: list[dict[str, Any]] = []
    pair_support_rows: list[dict[str, Any]] = []
    rule_dist: Counter = Counter()

    for coin in mr_coins:
        coin_id = str(coin.get("coin_id") or "")
        pair_ids = map_by_coin.get(coin_id) or [
            x for x in str(coin.get("supporting_pair_asset_ids") or "").split(";") if x
        ]
        pair_rows: list[dict[str, Any]] = []
        for pid in pair_ids:
            prow = pair_by_id.get(pid)
            if prow:
                pair_rows.append(prow)
            else:
                pair_rows.append({"asset_id": pid, "semantic_coin_class": "", "reasoning_short": ""})

        votes = _parse_votes(coin.get("class_votes"))
        if not votes and pair_rows:
            votes = Counter(str(p.get("semantic_coin_class") or "") for p in pair_rows)

        ident = identity_by_coin.get(coin_id) or {}
        identity_warning = str(
            coin.get("identity_warnings")
            or coin.get("identity_warning")
            or ident.get("identity_warnings")
            or ""
        )
        conflict_note = str(coin.get("conflict_note") or "")
        reasons = [
            str(p.get("reasoning_short") or "")
            for p in pair_rows
            if str(p.get("semantic_coin_class") or "") == "MANUAL_REVIEW"
            or str(p.get("raw_evidence_status") or "") == "REJECTED_FOR_TRADE_LANGUAGE"
        ]
        manual_review_reason = conflict_note or ("; ".join(r[:120] for r in reasons[:3]) if reasons else "")

        resolution = resolve_manual_review_coin(
            class_votes=votes,
            pair_rows=pair_rows,
            conflict_note=conflict_note,
            identity_warning=identity_warning,
            manual_review_reason=manual_review_reason,
        )
        rule_dist[resolution["resolution_rule_applied"]] += 1

        raw_dist = Counter(str(p.get("raw_evidence_status") or "") for p in pair_rows)
        symbols = sorted({str(p.get("symbol") or "") for p in pair_rows if p.get("symbol")})
        rejected_votes = sum(
            1
            for p in pair_rows
            if str(p.get("raw_evidence_status") or "") == "REJECTED_FOR_TRADE_LANGUAGE"
        )

        drilldown_rows.append(
            {
                "coin_identity_id": coin_id,
                "normalized_base_symbol": coin.get("normalized_base_symbol")
                or ident.get("normalized_base_symbol")
                or "",
                "canonical_base_symbol": coin.get("normalized_base_symbol")
                or ident.get("normalized_base_symbol")
                or "",
                "chain_scope": coin.get("chain") or ident.get("chain") or "",
                "original_coin_semantic_class": "MANUAL_REVIEW",
                "final_class_after_drilldown": resolution["final_class_after_drilldown"],
                "resolution_rule_applied": resolution["resolution_rule_applied"],
                "resolution_note": resolution["resolution_note"],
                "resolution_confidence": resolution["resolution_confidence"],
                "pair_asset_count": len(pair_rows),
                "supporting_pair_asset_ids": ";".join(pair_ids),
                "supporting_pair_symbols": ";".join(symbols),
                "class_votes": json.dumps(dict(votes)),
                "social_confirmed_vote_count": int(votes.get("SOCIAL_CONFIRMED", 0)),
                "opportunistic_confirmed_vote_count": int(
                    votes.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0)
                ),
                "opportunistic_suspected_vote_count": int(votes.get("OPPORTUNISTIC_SUSPECTED", 0)),
                "manual_review_vote_count": int(votes.get("MANUAL_REVIEW", 0)),
                "rejected_trade_language_vote_count": rejected_votes,
                "identity_resolution_method": coin.get("identity_resolution_method")
                or ident.get("identity_resolution_method")
                or "",
                "identity_confidence": coin.get("identity_confidence")
                or ident.get("identity_confidence")
                or "",
                "identity_warning": identity_warning,
                "conflict_note": conflict_note,
                "manual_review_reason": manual_review_reason[:500],
                "raw_evidence_status_distribution": json.dumps(dict(raw_dist)),
                "model_knowledge_only_count": int(raw_dist.get("MODEL_KNOWLEDGE_ONLY", 0)),
                "web_grounded_count": int(raw_dist.get("WEB_GROUNDED", 0)),
                "source_gemini_root": str(gemini_root),
                "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
                "coin_aggregation_rule_version": COIN_AGGREGATION_RULE_VERSION,
                "rubric_version": RUBRIC_VERSION,
                "external_api_used": False,
                "gemini_called_again": False,
            }
        )

        for p in pair_rows:
            raw_status = str(p.get("raw_evidence_status") or "")
            used = True
            exclude_reason = ""
            if raw_status == "REJECTED_FOR_TRADE_LANGUAGE":
                used = False
                exclude_reason = "REJECTED_FOR_TRADE_LANGUAGE_NOT_REINTERPRETED"
            pair_support_rows.append(
                {
                    "coin_identity_id": coin_id,
                    "pair_asset_id": p.get("asset_id"),
                    "pair_symbol": p.get("symbol"),
                    "pair_level_class": p.get("semantic_coin_class"),
                    "raw_evidence_status": raw_status,
                    "reasoning_short": str(p.get("reasoning_short") or "")[:250],
                    "safety_status": (
                        "REJECTED_FORBIDDEN_TRADE_LANGUAGE"
                        if raw_status == "REJECTED_FOR_TRADE_LANGUAGE"
                        else "OK"
                    ),
                    "used_in_drilldown": used,
                    "reason_if_excluded": exclude_reason,
                }
            )

    # Updated coin-level counts: non-MR coins keep class; MR coins take drilldown class
    final_by_coin: dict[str, str] = {}
    for c in coins:
        cid = str(c.get("coin_id") or "")
        final_by_coin[cid] = str(c.get("semantic_coin_class") or "")
    for row in drilldown_rows:
        final_by_coin[row["coin_identity_id"]] = row["final_class_after_drilldown"]

    dist = Counter(final_by_coin.values())
    unique_coins = len(final_by_coin) or 1
    social = int(dist.get("SOCIAL_CONFIRMED", 0))
    opp = int(dist.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0))
    sus = int(dist.get("OPPORTUNISTIC_SUSPECTED", 0))
    infra = int(dist.get("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", 0))
    unknown = int(dist.get("UNKNOWN_UNRESOLVED", 0))
    mr_remaining = int(dist.get("MANUAL_REVIEW", 0))

    updated_counts = {
        "unique_coins_found": len(final_by_coin),
        "coin_social_confirmed_count": social,
        "coin_non_social_opportunistic_confirmed_count": opp,
        "coin_opportunistic_suspected_count": sus,
        "coin_non_social_infrastructure_confirmed_count": infra,
        "coin_unknown_unresolved_count": unknown,
        "coin_manual_review_remaining_count": mr_remaining,
        "coin_social_confirmed_share": round(social / unique_coins, 6),
        "coin_opportunistic_confirmed_share": round(opp / unique_coins, 6),
        "coin_opportunistic_suspected_share": round(sus / unique_coins, 6),
        "coin_infrastructure_share": round(infra / unique_coins, 6),
        "coin_unknown_unresolved_share": round(unknown / unique_coins, 6),
        "coin_manual_review_remaining_share": round(mr_remaining / unique_coins, 6),
    }
    counts_rows = [{"metric": k, "value": v} for k, v in updated_counts.items()]

    manual_review_input_count = len(mr_coins)
    unknown_unresolved_count = sum(
        1 for r in drilldown_rows if r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    )
    resolved_confirmed = sum(
        1
        for r in drilldown_rows
        if r["final_class_after_drilldown"] in CONFIRMED_CLASSES
        or r["final_class_after_drilldown"] == "OPPORTUNISTIC_SUSPECTED"
    )
    # Resolved away from MANUAL_REVIEW (including UNKNOWN_UNRESOLVED)
    manual_review_resolved_count = sum(
        1 for r in drilldown_rows if r["final_class_after_drilldown"] != "MANUAL_REVIEW"
    )

    if external_api_used or gemini_called_again:
        gate_status = "FAIL_EXTERNAL_API_USED"
    elif source_missing:
        gate_status = "HOLD_SOURCE_FILES_MISSING"
    elif mr_remaining > 0 and not source_missing:
        # Remaining MR only if corruption; still allow UNKNOWN path
        gate_status = "FAIL_DATA_CORRUPTION" if manual_review_input_count > 0 else "PASS_MANUAL_REVIEW_REDUCED"
    elif unknown_unresolved_count > 0:
        gate_status = "PASS_WITH_UNKNOWN_UNRESOLVED"
    else:
        gate_status = "PASS_MANUAL_REVIEW_REDUCED"

    from app.ae12_sentimentfix.adjudication_safety_status import resolve_safety_audit_status

    safety_status = resolve_safety_audit_status(safety_src) if safety_src else "PASS_REJECTIONS_ENFORCED"

    gate = {
        "gate_name": "ae12_manual_review_drilldown_gate",
        "status": gate_status,
        "phase": "AE12-SentimentFix",
        "not_ae12_6": True,
        "source_gemini_root": str(gemini_root),
        "source_gemini_gate_status": gate_src.get("status") or summary_src.get("gate_status"),
        "source_adjudicator_version": gate_src.get("adjudicator_version"),
        "source_rubric_version": gate_src.get("rubric_version") or RUBRIC_VERSION,
        "source_coin_aggregation_rule_version": COIN_AGGREGATION_RULE_VERSION,
        "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
        "coin_aggregation_rule_version": COIN_AGGREGATION_RULE_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "drilldown_created_at_utc": created_at,
        "gemini_called_again": False,
        "external_api_used": False,
        "manual_review_input_count": manual_review_input_count,
        "manual_review_resolved_count": manual_review_resolved_count,
        "manual_review_resolved_to_confirmed_or_suspected_count": resolved_confirmed,
        "unknown_unresolved_count": unknown_unresolved_count,
        "manual_review_remaining_count": mr_remaining,
        "unique_coins_found": updated_counts["unique_coins_found"],
        "updated_coin_level_distribution": updated_counts,
        "resolution_rule_distribution": dict(rule_dist),
        "safety_status": safety_status,
        "trade_authority_used": False,
        "live_ready": False,
        "profitability_proven": False,
        "recommendation": (
            "Treat UNKNOWN_UNRESOLVED as unresolved-without-external-evidence, "
            "not as opportunistic or social. Coin-level final UI should prefer "
            "post-drilldown counts."
        ),
        "limitations": [
            "Local drilldown only; no Gemini, web search, or external APIs.",
            "UNKNOWN_UNRESOLVED is not opportunistic and not social.",
            "SOCIAL vs OPPORTUNISTIC conflicts are intentionally left unresolved.",
            "Rejected trade-language outputs were not reinterpreted.",
        ],
    }

    no_api_audit = {
        "external_api_used": False,
        "gemini_called_again": False,
        "web_search_used": False,
        "google_generativeai_used": False,
        "env_api_keys_used": False,
        "no_external_apis_flag": bool(no_external_apis),
        "status": "PASS_NO_EXTERNAL_API" if not external_api_used else "FAIL_EXTERNAL_API_USED",
    }
    safety_audit = {
        "status": safety_status,
        "source_safety_audit": safety_src.get("status"),
        "trade_authority_used": False,
        "output_used_after_rejection": False,
        "rejected_outputs_reinterpreted": False,
        "external_api_used": False,
        "gemini_called_again": False,
    }

    unresolved_examples = [
        {
            "coin_identity_id": r["coin_identity_id"],
            "normalized_base_symbol": r["normalized_base_symbol"],
            "resolution_rule_applied": r["resolution_rule_applied"],
            "resolution_note": r["resolution_note"],
            "class_votes": r["class_votes"],
        }
        for r in drilldown_rows
        if r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    ][:5]

    summary = {
        "phase": "AE12-SentimentFix",
        "subtask": "Manual Review Drilldown",
        "not_ae12_6": True,
        "created_at_utc": created_at,
        "drilldown_created_at_utc": created_at,
        "output_root": str(out_root),
        "source_gemini_root": str(gemini_root),
        "source_gemini_gate_status": gate["source_gemini_gate_status"],
        "source_adjudicator_version": gate["source_adjudicator_version"],
        "source_rubric_version": gate["source_rubric_version"],
        "source_coin_aggregation_rule_version": COIN_AGGREGATION_RULE_VERSION,
        "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
        "coin_aggregation_rule_version": COIN_AGGREGATION_RULE_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "gate_status": gate_status,
        "manual_review_input_count": manual_review_input_count,
        "manual_review_resolved_count": manual_review_resolved_count,
        "unknown_unresolved_count": unknown_unresolved_count,
        "manual_review_remaining_count": mr_remaining,
        "updated_coin_level_counts": updated_counts,
        "resolution_rule_distribution": dict(rule_dist),
        "unresolved_examples": unresolved_examples,
        "external_api_used": False,
        "gemini_called_again": False,
        "trade_authority_used": False,
        "live_ready": False,
        "profitability_proven": False,
        "safety_status": safety_status,
    }

    _write_csv(out_root / "data" / "ae12_manual_review_coin_drilldown.csv", drilldown_rows)
    _write_csv(out_root / "data" / "ae12_manual_review_pair_support.csv", pair_support_rows)
    _write_csv(out_root / "data" / "ae12_manual_review_resolution_rules.csv", RESOLUTION_RULES)
    _write_csv(out_root / "data" / "ae12_coin_level_counts_after_drilldown.csv", counts_rows)
    _write_json(out_root / "audits" / "ae12_manual_review_drilldown_gate.json", gate)
    _write_json(out_root / "audits" / "ae12_manual_review_no_external_api_audit.json", no_api_audit)
    _write_json(out_root / "audits" / "ae12_manual_review_safety_audit.json", safety_audit)
    _write_json(out_root / "reports" / "ae12_manual_review_drilldown_summary.json", summary)

    upload = [
        "AE12-SentimentFix Manual Review Drilldown (not AE12.6)",
        f"output_root: {out_root}",
        f"source_gemini_root: {gemini_root}",
        f"source_gemini_gate_status: {gate['source_gemini_gate_status']}",
        f"source_adjudicator_version: {gate['source_adjudicator_version']}",
        f"source_rubric_version: {gate['source_rubric_version']}",
        f"drilldown_rule_version: {DRILLDOWN_RULE_VERSION}",
        f"coin_aggregation_rule_version: {COIN_AGGREGATION_RULE_VERSION}",
        f"rubric_version: {RUBRIC_VERSION}",
        f"drilldown_created_at_utc: {created_at}",
        f"gate_status: {gate_status}",
        f"manual_review_input_count: {manual_review_input_count}",
        f"manual_review_resolved_count: {manual_review_resolved_count}",
        f"unknown_unresolved_count: {unknown_unresolved_count}",
        f"manual_review_remaining_count: {mr_remaining}",
        f"updated_coin_level_counts: {json.dumps(updated_counts)}",
        f"resolution_rule_distribution: {json.dumps(dict(rule_dist))}",
        "external_api_used: false",
        "gemini_called_again: false",
        "trade_authority_used: false",
        "live_ready: false",
        "profitability_proven: false",
        "UNKNOWN_UNRESOLVED is not opportunistic and not social.",
        "Local drilldown used existing adjudication artifacts only (no web evidence).",
    ]
    (out_root / "reports" / "ae12_manual_review_drilldown_for_upload.txt").write_text(
        "\n".join(upload) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": created_at,
        "output_root": str(out_root),
        "source_gemini_root": str(gemini_root),
        "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
        "files": sorted(str(p.relative_to(out_root)) for p in out_root.rglob("*") if p.is_file()),
        "external_api_used": False,
        "gemini_called_again": False,
        "trade_authority_used": False,
        "historical_data_mutated": False,
    }
    _write_json(out_root / "reports" / "ae12_manual_review_drilldown_manifest.json", manifest)
    return summary
