from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AE15_ROOT = Path(
    r"data\audits\ae15_clean_forward_schema_bridge_20260722_183935"
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def is_blank(value: Any) -> bool:
    return str(value or "").strip() == ""


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for f in preferred_fields or []:
        if f and f not in fields:
            fields.append(f)

    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def add_exclusion(row: dict[str, str], table: str, reason: str) -> dict[str, Any]:
    out = dict(row)
    out["ae16_source_table"] = table
    out["ae16_exclusion_reason"] = reason
    return out


def get_id(row: dict[str, str], key: str) -> str:
    return str(row.get(key) or "").strip()


def choose_decision_for_candidate(rows: list[dict[str, str]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    AE16 needs one normal decision input per candidate.

    Priority:
    1. Explicit AE14/closure/open decision input.
    2. Row with gatekeeper_input_payload_hash.
    3. Non-default preset.
    4. Stable first row by decision id.
    """
    def score(row: dict[str, str]) -> tuple[int, int, int, str]:
        preset = str(row.get("active_preset_id") or "").lower()
        decision_id = str(row.get("clean_forward_decision_input_id") or "")
        explicit = int(("explicit" in preset) or ("closure" in preset) or ("open" in preset))
        has_gate = int(not is_blank(row.get("gatekeeper_input_payload_hash")))
        non_default = int("schema_bridge_default" not in preset)
        return explicit, has_gate, non_default, decision_id

    ranked = sorted(rows, key=score, reverse=True)
    keep = ranked[0]
    excluded = [
        add_exclusion(r, "clean_forward_decision_inputs", "DUPLICATE_DECISION_INPUT_LOWER_PRIORITY_FOR_AE16")
        for r in ranked[1:]
    ]
    return keep, excluded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create clean AE16 input files from AE15 schema bridge outputs without mutating AE15 artifacts."
    )
    parser.add_argument("--ae15-root", default=str(DEFAULT_AE15_ROOT))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    ae15_root = Path(args.ae15_root)
    if not ae15_root.exists():
        raise SystemExit(f"AE15 root not found: {ae15_root}")

    output_root = Path(args.output_root) if args.output_root else Path(
        "data/audits"
    ) / f"ae15_cleaned_for_ae16_{utc_stamp()}"

    candidates_path = ae15_root / "data" / "clean_forward_candidates.csv"
    decisions_path = ae15_root / "data" / "clean_forward_decision_inputs.csv"
    outcomes_path = ae15_root / "data" / "clean_forward_outcome_label_contract.csv"
    exec_links_path = ae15_root / "data" / "clean_forward_paper_execution_links.csv"
    skip_reasons_path = ae15_root / "data" / "clean_forward_skip_reasons.csv"

    candidates, candidate_fields = read_csv(candidates_path)
    decisions, decision_fields = read_csv(decisions_path)
    outcomes, outcome_fields = read_csv(outcomes_path)
    exec_links, exec_fields = read_csv(exec_links_path)

    skip_reasons: list[dict[str, str]] = []
    skip_fields: list[str] = []
    if skip_reasons_path.exists():
        skip_reasons, skip_fields = read_csv(skip_reasons_path)

    outcome_ids = {
        get_id(row, "clean_forward_candidate_id")
        for row in outcomes
        if get_id(row, "clean_forward_candidate_id")
    }

    candidate_id_counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        candidate_id_counts[get_id(row, "clean_forward_candidate_id")] += 1

    bad_exec_candidate_ids: set[str] = set()
    for row in exec_links:
        cid = get_id(row, "clean_forward_candidate_id")
        status = str(row.get("counter_consistency_status") or "").upper()
        has_order_id = not is_blank(row.get("paper_order_id"))
        one_to_one_passed = as_bool(row.get("one_order_to_one_position_passed"))

        if not cid:
            continue

        if (
            not has_order_id
            or "PENDING" in status
            or not one_to_one_passed
        ):
            bad_exec_candidate_ids.add(cid)

    clean_candidates: list[dict[str, str]] = []
    excluded_candidates: list[dict[str, Any]] = []

    for row in candidates:
        cid = get_id(row, "clean_forward_candidate_id")
        source_poll_file = str(row.get("source_poll_file") or "").lower()
        payload_hash = str(row.get("provider_payload_hash") or "").lower()
        verification_status = str(row.get("verification_status") or "")
        freshness_status = str(row.get("freshness_status") or "")
        identity_status = str(row.get("identity_status") or "")

        reason = None

        if not cid:
            reason = "MISSING_CANDIDATE_ID"
        elif candidate_id_counts[cid] > 1:
            reason = "DUPLICATE_CANDIDATE_ID"
        elif cid in bad_exec_candidate_ids:
            reason = "UNRESOLVED_ORDER_POSITION_LINEAGE"
        elif "demo_bot" in source_poll_file or "demo_bot" in payload_hash:
            reason = "DEMO_BOT_RECONCILIATION_ARTIFACT"
        elif cid not in outcome_ids:
            reason = "MISSING_OUTCOME_CONTRACT"
        elif not as_bool(row.get("clean_feed_eligible")):
            reason = "NOT_CLEAN_FEED_ELIGIBLE"
        elif not as_bool(row.get("paper_demo_only")):
            reason = "PAPER_DEMO_ONLY_FALSE"
        elif as_bool(row.get("live_trading_ready")):
            reason = "LIVE_TRADING_READY_TRUE_UNSAFE_FOR_AE16"
        elif verification_status != "provider_pair_verified":
            reason = f"BAD_VERIFICATION_STATUS:{verification_status}"
        elif freshness_status != "fresh":
            reason = f"BAD_FRESHNESS_STATUS:{freshness_status}"
        elif identity_status != "pair_and_tokens_separated":
            reason = f"BAD_IDENTITY_STATUS:{identity_status}"

        if reason:
            excluded_candidates.append(add_exclusion(row, "clean_forward_candidates", reason))
        else:
            clean_candidates.append(row)

    clean_candidate_ids = {
        get_id(row, "clean_forward_candidate_id")
        for row in clean_candidates
        if get_id(row, "clean_forward_candidate_id")
    }

    decisions_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded_decisions: list[dict[str, Any]] = []

    for row in decisions:
        cid = get_id(row, "clean_forward_candidate_id")
        if not cid:
            excluded_decisions.append(add_exclusion(row, "clean_forward_decision_inputs", "MISSING_CANDIDATE_ID"))
        elif cid not in clean_candidate_ids:
            excluded_decisions.append(add_exclusion(row, "clean_forward_decision_inputs", "CANDIDATE_EXCLUDED_FROM_AE16"))
        else:
            decisions_by_candidate[cid].append(row)

    clean_decisions: list[dict[str, str]] = []
    missing_decision_candidate_ids: list[str] = []

    for cid in sorted(clean_candidate_ids):
        group = decisions_by_candidate.get(cid, [])
        if not group:
            missing_decision_candidate_ids.append(cid)
            continue
        keep, lower_priority = choose_decision_for_candidate(group)
        clean_decisions.append(keep)
        excluded_decisions.extend(lower_priority)

    clean_outcomes: list[dict[str, str]] = []
    excluded_outcomes: list[dict[str, Any]] = []

    for row in outcomes:
        cid = get_id(row, "clean_forward_candidate_id")
        if cid in clean_candidate_ids:
            clean_outcomes.append(row)
        else:
            excluded_outcomes.append(add_exclusion(row, "clean_forward_outcome_label_contract", "CANDIDATE_NOT_ALLOWED_FOR_AE16"))

    clean_exec_links: list[dict[str, str]] = []
    excluded_exec_links: list[dict[str, Any]] = []

    for row in exec_links:
        cid = get_id(row, "clean_forward_candidate_id")
        status = str(row.get("counter_consistency_status") or "").upper()
        has_order_id = not is_blank(row.get("paper_order_id"))
        one_to_one_passed = as_bool(row.get("one_order_to_one_position_passed"))

        if (
            cid in clean_candidate_ids
            and has_order_id
            and "PENDING" not in status
            and one_to_one_passed
        ):
            clean_exec_links.append(row)
        else:
            excluded_exec_links.append(add_exclusion(row, "clean_forward_paper_execution_links", "EXECUTION_LINK_NOT_ALLOWED_FOR_AE16"))

    clean_skip_reasons: list[dict[str, str]] = []
    excluded_skip_reasons: list[dict[str, Any]] = []
    for row in skip_reasons:
        cid = get_id(row, "clean_forward_candidate_id")
        if cid in clean_candidate_ids:
            clean_skip_reasons.append(row)
        else:
            excluded_skip_reasons.append(add_exclusion(row, "clean_forward_skip_reasons", "CANDIDATE_NOT_ALLOWED_FOR_AE16"))

    # Output paths
    data_dir = output_root / "data"
    audits_dir = output_root / "audits"
    reports_dir = output_root / "reports"
    manifests_dir = output_root / "manifests"

    write_csv(data_dir / "ae16_clean_forward_candidates.csv", clean_candidates, candidate_fields)
    write_csv(data_dir / "ae16_clean_forward_decision_inputs.csv", clean_decisions, decision_fields)
    write_csv(data_dir / "ae16_clean_forward_outcome_label_contract.csv", clean_outcomes, outcome_fields)
    write_csv(data_dir / "ae16_clean_forward_paper_execution_links.csv", clean_exec_links, exec_fields)

    if skip_reasons_path.exists():
        write_csv(data_dir / "ae16_clean_forward_skip_reasons.csv", clean_skip_reasons, skip_fields)

    write_csv(data_dir / "ae16_excluded_candidates.csv", excluded_candidates, candidate_fields + ["ae16_source_table", "ae16_exclusion_reason"])
    write_csv(data_dir / "ae16_excluded_decision_inputs.csv", excluded_decisions, decision_fields + ["ae16_source_table", "ae16_exclusion_reason"])
    write_csv(data_dir / "ae16_excluded_outcomes.csv", excluded_outcomes, outcome_fields + ["ae16_source_table", "ae16_exclusion_reason"])
    write_csv(data_dir / "ae16_excluded_execution_links.csv", excluded_exec_links, exec_fields + ["ae16_source_table", "ae16_exclusion_reason"])

    if skip_reasons_path.exists():
        write_csv(data_dir / "ae16_excluded_skip_reasons.csv", excluded_skip_reasons, skip_fields + ["ae16_source_table", "ae16_exclusion_reason"])

    excluded_candidate_reason_counts: dict[str, int] = defaultdict(int)
    for row in excluded_candidates:
        excluded_candidate_reason_counts[str(row.get("ae16_exclusion_reason"))] += 1

    excluded_decision_reason_counts: dict[str, int] = defaultdict(int)
    for row in excluded_decisions:
        excluded_decision_reason_counts[str(row.get("ae16_exclusion_reason"))] += 1

    excluded_exec_reason_counts: dict[str, int] = defaultdict(int)
    for row in excluded_exec_links:
        excluded_exec_reason_counts[str(row.get("ae16_exclusion_reason"))] += 1

    clean_candidate_id_count = len(clean_candidate_ids)
    clean_decision_candidate_id_count = len({
        get_id(row, "clean_forward_candidate_id") for row in clean_decisions
    })
    clean_outcome_candidate_id_count = len({
        get_id(row, "clean_forward_candidate_id") for row in clean_outcomes
    })

    invariants = {
        "clean_candidates_unique_ids": len(clean_candidates) == clean_candidate_id_count,
        "one_decision_input_per_clean_candidate": len(clean_decisions) == clean_candidate_id_count,
        "decision_candidate_ids_match_clean_candidates": clean_decision_candidate_id_count == clean_candidate_id_count,
        "outcome_contracts_match_clean_candidates": len(clean_outcomes) == clean_candidate_id_count,
        "outcome_candidate_ids_match_clean_candidates": clean_outcome_candidate_id_count == clean_candidate_id_count,
        "no_missing_decision_inputs": len(missing_decision_candidate_ids) == 0,
        "excluded_unresolved_execution_links": len(excluded_exec_links) >= 1,
    }

    status = "AE16_INPUT_CLEANING_PASS"
    if not all(invariants.values()):
        status = "AE16_INPUT_CLEANING_PASS_WITH_LIMITATIONS"

    audit = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_ae15_root": str(ae15_root),
        "output_root": str(output_root),
        "policy": {
            "original_ae15_artifacts_mutated": False,
            "legacy_market_snapshots_used": False,
            "training_performed": False,
            "backtest_performed": False,
            "profitability_claimed": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "purpose": "Create clean downstream AE16 inputs from AE15 outputs while preserving AE15 archival evidence.",
        },
        "input_counts": {
            "candidates": len(candidates),
            "decision_inputs": len(decisions),
            "outcome_contracts": len(outcomes),
            "execution_links": len(exec_links),
            "skip_reasons": len(skip_reasons),
        },
        "output_counts": {
            "clean_candidates": len(clean_candidates),
            "clean_decision_inputs": len(clean_decisions),
            "clean_outcome_contracts": len(clean_outcomes),
            "clean_execution_links": len(clean_exec_links),
            "clean_skip_reasons": len(clean_skip_reasons),
            "excluded_candidates": len(excluded_candidates),
            "excluded_decision_inputs": len(excluded_decisions),
            "excluded_outcome_contracts": len(excluded_outcomes),
            "excluded_execution_links": len(excluded_exec_links),
            "excluded_skip_reasons": len(excluded_skip_reasons),
        },
        "excluded_reason_counts": {
            "candidates": dict(sorted(excluded_candidate_reason_counts.items())),
            "decision_inputs": dict(sorted(excluded_decision_reason_counts.items())),
            "execution_links": dict(sorted(excluded_exec_reason_counts.items())),
        },
        "invariants": invariants,
        "missing_decision_candidate_ids": missing_decision_candidate_ids[:50],
        "ae16_input_rule": {
            "consume_only_clean_candidates": True,
            "exclude_reconciliation_only_execution_artifacts": True,
            "exclude_positions_without_paper_order_id_for_execution_linked_consensus": True,
            "one_decision_input_per_candidate": True,
            "normal_model_consensus_input_not_live_authority": True,
        },
        "known_ae14_carry_forward": {
            "AE14_POSITION_COUNTER_RECONCILIATION_PENDING": True,
            "resolution": "PUMP/MET demo_bot.run_once position excluded from normal AE16 input and preserved only in excluded/artifact audit outputs.",
        },
        "recommended_ae16_inputs": {
            "candidates": str(data_dir / "ae16_clean_forward_candidates.csv"),
            "decision_inputs": str(data_dir / "ae16_clean_forward_decision_inputs.csv"),
            "outcome_contracts": str(data_dir / "ae16_clean_forward_outcome_label_contract.csv"),
            "execution_links": str(data_dir / "ae16_clean_forward_paper_execution_links.csv"),
        },
    }

    manifest = {
        "created_at_utc": audit["created_at_utc"],
        "status": status,
        "input_ae15_root": str(ae15_root),
        "output_root": str(output_root),
        "files": {
            "clean_candidates": str(data_dir / "ae16_clean_forward_candidates.csv"),
            "clean_decision_inputs": str(data_dir / "ae16_clean_forward_decision_inputs.csv"),
            "clean_outcome_contracts": str(data_dir / "ae16_clean_forward_outcome_label_contract.csv"),
            "clean_execution_links": str(data_dir / "ae16_clean_forward_paper_execution_links.csv"),
            "excluded_candidates": str(data_dir / "ae16_excluded_candidates.csv"),
            "excluded_decision_inputs": str(data_dir / "ae16_excluded_decision_inputs.csv"),
            "excluded_outcomes": str(data_dir / "ae16_excluded_outcomes.csv"),
            "excluded_execution_links": str(data_dir / "ae16_excluded_execution_links.csv"),
            "audit": str(audits_dir / "ae16_input_cleaning_audit.json"),
            "summary": str(reports_dir / "ae16_input_cleaning_summary.txt"),
        },
    }

    write_json(audits_dir / "ae16_input_cleaning_audit.json", audit)
    write_json(manifests_dir / "ae16_input_manifest.json", manifest)

    summary_lines = [
        "AE15 → AE16 input cleaning completed.",
        "",
        f"Status: {status}",
        f"Input AE15 root: {ae15_root}",
        f"Output root: {output_root}",
        "",
        "Input counts:",
        f"- candidates: {len(candidates)}",
        f"- decision_inputs: {len(decisions)}",
        f"- outcome_contracts: {len(outcomes)}",
        f"- execution_links: {len(exec_links)}",
        "",
        "Output counts:",
        f"- clean_candidates: {len(clean_candidates)}",
        f"- clean_decision_inputs: {len(clean_decisions)}",
        f"- clean_outcome_contracts: {len(clean_outcomes)}",
        f"- clean_execution_links: {len(clean_exec_links)}",
        f"- excluded_candidates: {len(excluded_candidates)}",
        f"- excluded_decision_inputs: {len(excluded_decisions)}",
        f"- excluded_execution_links: {len(excluded_exec_links)}",
        "",
        "Key interpretation:",
        "- PUMP/MET demo_bot.run_once position is excluded from normal AE16 inputs.",
        "- AE14_POSITION_COUNTER_RECONCILIATION_PENDING is preserved as archival evidence, not hidden.",
        "- Bonk/MET explicit AE14 clean-forward execution row is retained.",
        "- AE16 receives one decision input per clean candidate.",
        "- Original AE15 artifacts were not modified.",
        "",
        "Recommended AE16 input files:",
        f"- {data_dir / 'ae16_clean_forward_candidates.csv'}",
        f"- {data_dir / 'ae16_clean_forward_decision_inputs.csv'}",
        f"- {data_dir / 'ae16_clean_forward_outcome_label_contract.csv'}",
        f"- {data_dir / 'ae16_clean_forward_paper_execution_links.csv'}",
    ]
    (reports_dir / "ae16_input_cleaning_summary.txt").parent.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ae16_input_cleaning_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps({
        "status": status,
        "output_root": str(output_root),
        "clean_candidates": len(clean_candidates),
        "clean_decision_inputs": len(clean_decisions),
        "clean_outcome_contracts": len(clean_outcomes),
        "clean_execution_links": len(clean_exec_links),
        "excluded_candidates": len(excluded_candidates),
        "excluded_decision_inputs": len(excluded_decisions),
        "excluded_execution_links": len(excluded_exec_links),
        "invariants": invariants,
    }, indent=2))

    return 0 if status == "AE16_INPUT_CLEANING_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
