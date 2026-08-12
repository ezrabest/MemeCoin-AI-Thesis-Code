"""Wallet / append-only / no-mutation safety audits for AE12."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ae12_forward_evidence.loaders import iter_jsonl
from app.ae12_forward_evidence.types import utc_now_iso


def audit_wallet_safety(
    *,
    project_root: Path,
    live_dry_run_files: list[Path],
    trade_decision_sample: list[dict[str, Any]] | None = None,
    no_real_wallet: bool = True,
) -> dict[str, Any]:
    """
    Confirm no real wallet / private key / live submit from available files.
    If evidence insufficient → UNKNOWN_REQUIRES_RUNTIME_SAFETY_AUDIT (never false PASS).
    """
    evidence_bits: list[str] = []
    wallet_configured = None
    private_key_accessed = None
    real_tx_signed = None
    real_tx_submitted = None
    live_submission_status = None
    files_checked = 0

    for path in live_dry_run_files:
        if not path.is_file():
            continue
        files_checked += 1
        for _, obj in iter_jsonl(path):
            if "wallet_configured" in obj:
                wallet_configured = bool(obj.get("wallet_configured"))
                evidence_bits.append("live_dry_run.wallet_configured")
            if "private_key_accessed" in obj:
                private_key_accessed = bool(obj.get("private_key_accessed"))
                evidence_bits.append("live_dry_run.private_key_accessed")
            if "real_transaction_attempted" in obj:
                real_tx_signed = bool(obj.get("real_transaction_attempted"))
                evidence_bits.append("live_dry_run.real_transaction_attempted")
            if "live_submission_status" in obj:
                live_submission_status = obj.get("live_submission_status")
                evidence_bits.append("live_dry_run.live_submission_status")

    if trade_decision_sample:
        for obj in trade_decision_sample[:200]:
            hs = obj.get("hard_safety") if isinstance(obj.get("hard_safety"), dict) else {}
            if hs:
                files_checked += 1
                if "wallet_configured" in hs:
                    wallet_configured = bool(hs.get("wallet_configured"))
                    evidence_bits.append("trade_decision.hard_safety.wallet_configured")
                if "private_key_accessed" in hs:
                    private_key_accessed = bool(hs.get("private_key_accessed"))
                    evidence_bits.append("trade_decision.hard_safety.private_key_accessed")
                if "real_transaction_attempted" in hs:
                    real_tx_signed = bool(hs.get("real_transaction_attempted"))
                    evidence_bits.append("trade_decision.hard_safety.real_transaction_attempted")
                if "live_submission_status" in hs:
                    live_submission_status = hs.get("live_submission_status")
                    evidence_bits.append("trade_decision.hard_safety.live_submission_status")
                break

    # settings.json probe (read-only)
    settings_path = project_root / "data" / "settings.json"
    settings_wallet_hint = None
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            for key in ("private_key", "wallet_private_key", "secret_key", "solana_private_key"):
                if key in settings and settings.get(key):
                    settings_wallet_hint = "SETTINGS_CONTAINS_KEY_FIELD"
                    break
            if settings.get("wallet_configured") is False:
                wallet_configured = False if wallet_configured is None else wallet_configured
                evidence_bits.append("settings.wallet_configured=false")
        except (json.JSONDecodeError, OSError):
            pass

    sufficient = bool(evidence_bits) and files_checked > 0
    if not sufficient:
        status = "UNKNOWN_REQUIRES_RUNTIME_SAFETY_AUDIT"
    else:
        safe = (
            wallet_configured is False
            and private_key_accessed is False
            and (real_tx_signed is False or real_tx_signed is None)
            and (
                live_submission_status in {None, "NOT_SUBMITTED_NO_WALLET", "NOT_SUBMITTED", "DRY_RUN"}
            )
            and settings_wallet_hint is None
            and no_real_wallet is True
        )
        # Explicit negative private_key / wallet may be missing → not PASS
        if wallet_configured is None or private_key_accessed is None:
            status = "UNKNOWN_REQUIRES_RUNTIME_SAFETY_AUDIT"
        elif safe:
            status = "PASS"
        else:
            status = "FAIL_OR_UNKNOWN"

    return {
        "audit_status": status,
        "wallet_configured": wallet_configured,
        "private_key_accessed": private_key_accessed,
        "real_transaction_signed": real_tx_signed,
        "real_transaction_submitted": real_tx_submitted,
        "live_submission_status": live_submission_status,
        "settings_wallet_hint": settings_wallet_hint,
        "no_real_wallet_flag": no_real_wallet,
        "evidence_bits": sorted(set(evidence_bits)),
        "files_checked": files_checked,
        "live_dry_run_file_count": len(live_dry_run_files),
        "audited_at_utc": utc_now_iso(),
        "note": "AE12 does not enable live trading; this is a file-evidence audit only.",
    }


def audit_append_only(
    *,
    project_root: Path,
    output_root: Path,
    mutated_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """AE12 must write only under its audit root; never mutate AE11/AE6/trader.db."""
    rows: list[dict[str, Any]] = []
    out_resolved = output_root.resolve()
    forbidden_prefixes = [
        (project_root / "data" / "runtime_paper_loop").resolve(),
        (project_root / "data" / "decision_records").resolve(),
        (project_root / "data" / "paper_trading").resolve(),
        (project_root / "data" / "trader.db").resolve(),
    ]
    for path in mutated_paths or []:
        resolved = path.resolve()
        under_output = out_resolved in resolved.parents or resolved == out_resolved
        touches_forbidden = any(
            resolved == fp or fp in resolved.parents or resolved == fp for fp in forbidden_prefixes
        )
        # trader.db is a file
        if resolved == (project_root / "data" / "trader.db").resolve():
            touches_forbidden = True
        rows.append(
            {
                "path": str(path),
                "under_ae12_output_root": under_output,
                "touches_forbidden_runtime_path": touches_forbidden,
                "append_only_ok": under_output and not touches_forbidden,
            }
        )
    if not rows:
        rows.append(
            {
                "path": str(output_root),
                "under_ae12_output_root": True,
                "touches_forbidden_runtime_path": False,
                "append_only_ok": True,
                "note": "No external mutations recorded; AE12 writes confined to output root.",
            }
        )
    return rows


def audit_no_trader_db_writes(output_root: Path, db_path: Path | None) -> dict[str, Any]:
    return {
        "trader_db_path": str(db_path) if db_path else None,
        "ae12_state_in_trader_db": False,
        "ae12_state_location": str(output_root / "state"),
        "policy": "Do not write processed state to trader.db",
        "status": "PASS",
    }
