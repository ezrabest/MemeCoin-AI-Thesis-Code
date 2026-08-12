"""Read-only API helpers for AE12.7 intelligent agent layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.intelligent_agents.discovery import load_agent_records_from_root, load_latest_ae12_7_root


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class AE127AgentReportStore:
    """Cached reader for latest AE12.7 agent demo artifacts."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or _project_root())
        self._root: Path | None = None
        self._records: list[dict[str, Any]] | None = None
        self._ui: dict[str, Any] | None = None
        self._gate: dict[str, Any] | None = None
        self._authority: dict[str, Any] | None = None

    def refresh(self) -> dict[str, Any]:
        self._root = load_latest_ae12_7_root(self.project_root)
        self._records = load_agent_records_from_root(self._root) if self._root else []
        self._ui = self._read_json("reports/ae12_7_ui_status_summary.json")
        self._gate = self._read_json("reports/ae12_7_intelligent_agent_decision_gate.json")
        self._authority = self._read_json("audits/ae12_7_agent_authority_audit.json")
        return {
            "refreshed": True,
            "latest_root": str(self._root) if self._root else None,
            "record_count": len(self._records or []),
            "read_only": True,
        }

    def _ensure(self) -> None:
        if self._records is None:
            self.refresh()

    def _read_json(self, rel: str) -> dict[str, Any]:
        if not self._root:
            return {}
        path = self._root / rel
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def get_status(self) -> dict[str, Any]:
        self._ensure()
        ui = self._ui or {}
        gate = self._gate or {}
        return {
            "phase": "AE12.7",
            "latest_ae12_7_output_root": str(self._root) if self._root else None,
            "operating_mode": gate.get("operating_mode") or ui.get("mode"),
            "qwen_memo_status": ui.get("qwen_memo_status", "MISSING"),
            "gemini_audit_status": ui.get("gemini_audit_status", "MISSING"),
            "helius_enrichment_status": ui.get("helius_enrichment_status", "MISSING"),
            "rss_context_status": ui.get("rss_context_status", "MISSING"),
            "semantic_classification_status": ui.get("semantic_classification_status", "MISSING"),
            "gate_status": gate.get("status"),
            "trade_authority_used": False,
            "wallet_status": "NOT_CONFIGURED",
            "live_ready": False,
            "profitability_proven": False,
            "ae12_closed": False,
            "external_api_used": gate.get("external_api_used", False),
            "record_count": len(self._records or []),
            "read_only": True,
        }

    def get_recent(self, limit: int = 50) -> dict[str, Any]:
        self._ensure()
        rows = list(reversed(self._records or []))[:limit]
        return {
            "phase": "AE12.7",
            "count": len(rows),
            "rows": rows,
            "trade_authority_used": False,
            "live_ready": False,
            "read_only": True,
        }

    def get_by_candidate(self, candidate_id: str) -> dict[str, Any]:
        self._ensure()
        rows = [r for r in (self._records or []) if str(r.get("candidate_id") or "") == str(candidate_id)]
        return {
            "phase": "AE12.7",
            "candidate_id": candidate_id,
            "count": len(rows),
            "rows": rows,
            "trade_authority_used": False,
            "read_only": True,
        }

    def get_by_pair(self, pair_address: str) -> dict[str, Any]:
        self._ensure()
        rows = [r for r in (self._records or []) if str(r.get("pair_address") or "") == str(pair_address)]
        return {
            "phase": "AE12.7",
            "pair_address": pair_address,
            "count": len(rows),
            "rows": rows,
            "trade_authority_used": False,
            "read_only": True,
        }

    def get_authority_audit(self) -> dict[str, Any]:
        self._ensure()
        body = dict(self._authority or {})
        body.setdefault("trade_authority_used", False)
        body.setdefault("wallet_status", "NOT_CONFIGURED")
        body.setdefault("live_ready", False)
        body.setdefault("phase", "AE12.7")
        body["read_only"] = True
        return body

    def get_ui_summary(self) -> dict[str, Any]:
        self._ensure()
        body = dict(self._ui or {})
        if not body:
            body = {
                "phase": "AE12.7",
                "panel": "Intelligent Agent Layer",
                "qwen_memo_status": "MISSING",
                "gemini_audit_status": "MISSING",
                "helius_enrichment_status": "MISSING",
                "rss_context_status": "MISSING",
                "semantic_classification_status": "MISSING",
                "warnings_blockers": ["ae12_7_artifacts_not_found"],
                "soft_veto_recommendations": [],
                "trade_authority_used": False,
                "wallet_status": "NOT_CONFIGURED",
                "live_ready": False,
                "profitability_proven": False,
            }
        body["trade_authority_used"] = False
        body["wallet_status"] = "NOT_CONFIGURED"
        body["live_ready"] = False
        body["profitability_proven"] = False
        body["read_only"] = True
        return body


_STORE: AE127AgentReportStore | None = None


def get_ae127_agent_store() -> AE127AgentReportStore:
    global _STORE
    if _STORE is None:
        _STORE = AE127AgentReportStore()
    return _STORE
