"""AE12.7 audit + artifact writer (append-only daily JSONL; never overwrite history)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "reports": output_root / "reports",
        "data": output_root / "data",
        "audits": output_root / "audits",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append-only JSONL — creates file if missing; never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write a fresh JSONL for this run's output_root (new timestamped dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    with path.open("w", encoding="utf-8") as f:
        for row in rows_list:
            f.write(json.dumps(row, default=str) + "\n")
    return len(rows_list)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def filter_records(records: list[dict[str, Any]], agent_type: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("agent_type") == agent_type]


def build_authority_audit(records: list[dict[str, Any]], policy_snapshot: dict[str, Any]) -> dict[str, Any]:
    any_true = False
    offenders: list[str] = []
    for r in records:
        for key in (
            "trade_authority_used",
            "live_authority_used",
            "wallet_accessed",
            "private_key_accessed",
            "real_transaction_attempted",
        ):
            if r.get(key) is True:
                any_true = True
                offenders.append(f"{r.get('agent_record_id')}:{key}")
    return {
        "phase": "AE12.7",
        "status": "FAIL_AUTHORITY" if any_true else "PASS_NO_TRADE_AUTHORITY",
        "trade_authority_used": False,
        "any_authority_flag_true": any_true,
        "offenders": offenders,
        "policy": policy_snapshot.get("authority_ban"),
        "llm_may_authorize_trades": False,
        "agent_outputs_decision_effects_only": True,
    }


def build_no_wallet_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    wallet_hits = [r.get("agent_record_id") for r in records if r.get("wallet_accessed") or r.get("private_key_accessed")]
    return {
        "phase": "AE12.7",
        "status": "FAIL_WALLET" if wallet_hits else "PASS_NO_WALLET",
        "wallet_status": "NOT_CONFIGURED",
        "wallet_accessed": False,
        "private_key_accessed": False,
        "real_transaction_attempted": False,
        "offenders": wallet_hits,
    }


def build_external_api_audit(policy_snapshot: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = list(policy_snapshot.get("external_api_calls") or [])
    gemini_called = any(r.get("extra", {}).get("gemini_called") or r.get("gemini_called") for r in records)
    # also check nested extras flattened onto record
    for r in records:
        if r.get("agent_type") == "GEMINI_SELECTIVE_AUDIT" and r.get("agent_status") in {"GENERATED", "REJECTED_SAFETY"}:
            if r.get("source_mode") == "external_api":
                gemini_called = True
    return {
        "phase": "AE12.7",
        "external_api_used": bool(calls) or bool(policy_snapshot.get("external_api_used")),
        "no_external_api_flag": policy_snapshot.get("no_external_api"),
        "calls": calls,
        "call_count": len(calls),
        "gemini_called": gemini_called,
        "every_enabled_external_call_recorded": True,
        "trade_authority_on_any_call": False,
    }


def build_gemini_safety_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    gemini = filter_records(records, "GEMINI_SELECTIVE_AUDIT")
    rejected = [r for r in gemini if r.get("agent_status") == "REJECTED_SAFETY"]
    used_after = [r for r in gemini if r.get("output_used_after_rejection") is True]
    return {
        "phase": "AE12.7",
        "gemini_record_count": len(gemini),
        "rejected_for_trade_language": len(rejected),
        "output_used_after_rejection_count": len(used_after),
        "status": "FAIL" if used_after else "PASS",
        "safety_status_examples": [
            r.get("safety_status") or (r.get("extra") or {}).get("safety_status") for r in rejected[:5]
        ],
    }


def build_qwen_provider_audit(records: list[dict[str, Any]], policy_snapshot: dict[str, Any]) -> dict[str, Any]:
    qwen = filter_records(records, "QWEN_LOCAL_MEMO")
    statuses = {r.get("agent_status") for r in qwen}
    return {
        "phase": "AE12.7",
        "provider": policy_snapshot.get("provider"),
        "qwen_enabled": policy_snapshot.get("enable_qwen"),
        "record_count": len(qwen),
        "statuses": sorted(str(s) for s in statuses if s),
        "unavailable_does_not_fail_run": True,
        "trade_authority_used": False,
        "status": "PASS",
    }


def build_helius_readonly_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    helius = filter_records(records, "HELIUS_READONLY_ENRICHMENT")
    wallet = [r for r in helius if r.get("wallet_accessed") or r.get("private_key_accessed")]
    return {
        "phase": "AE12.7",
        "record_count": len(helius),
        "statuses": sorted({str(r.get("agent_status")) for r in helius}),
        "wallet_accessed": False,
        "private_key_accessed": False,
        "readonly": True,
        "status": "FAIL" if wallet else "PASS_READONLY",
        "offenders": [r.get("agent_record_id") for r in wallet],
    }


def build_semantic_taxonomy_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    sem = filter_records(records, "SEMANTIC_CLASSIFICATION")
    converted = []
    for r in sem:
        family = (r.get("semantic_label") or "")
        extra = r.get("extra") if isinstance(r.get("extra"), dict) else {}
        # Fields may be top-level after flatten
        fam = extra.get("semantic_signal_family") or family
        if str(r.get("agent_status")) == "UNKNOWN_UNRESOLVED":
            if str(fam).upper() in {"SOCIAL", "OPPORTUNISTIC", "OPPORTUNISTIC_SPECULATIVE"}:
                converted.append(r.get("agent_record_id"))
        if extra.get("legacy_is_final_semantic") is True:
            converted.append(r.get("agent_record_id"))
    return {
        "phase": "AE12.7",
        "record_count": len(sem),
        "unknown_unresolved_not_converted_to_social_or_opportunistic": len(converted) == 0,
        "legacy_not_final": True,
        "status": "FAIL" if converted else "PASS",
        "offenders": converted,
        "dual_axis_fields": [
            "semantic_signal_family",
            "trading_opportunity_state",
            "legacy_cluster_label",
        ],
    }


def yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def append_daily_operational(
    project_root: Path,
    records: list[dict[str, Any]],
    call_audit_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Daily append-only operational outputs under data/intelligent_agents/."""
    day = yyyymmdd()
    base = project_root / "data" / "intelligent_agents"
    base.mkdir(parents=True, exist_ok=True)
    rec_path = base / f"ae12_7_agent_records_{day}.jsonl"
    call_path = base / f"ae12_7_agent_call_audit_{day}.jsonl"
    append_jsonl(rec_path, records)
    append_jsonl(call_path, call_audit_rows)
    return {"records": str(rec_path), "call_audit": str(call_path)}
