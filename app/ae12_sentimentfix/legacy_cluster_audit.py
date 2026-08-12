"""Legacy sticky cluster_label audit + soft-expiry plan (no registry mutation)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CODE_PATHS = (
    "app/analytics/features.py",
    "app/models/predictor.py",
    "app/observability/sentiment_cluster_audit.py",
    "app/observability/candidate.py",
    "app/observability/actionability.py",
    "app/live.py",
    "app/database.py",
    "static/index.html",
)

DANGEROUS_PATTERNS = (
    (r'DEFAULT_CLUSTER\s*=\s*["\']OPPORTUNISTIC_SPECULATIVE["\']', "default_opportunistic_constant"),
    (r'\.get\(\s*["\']cluster_label["\']\s*,\s*["\']OPPORTUNISTIC_SPECULATIVE["\']\s*\)', "get_default_opportunistic"),
    (r'cluster_label:\s*str\s*=\s*["\']OPPORTUNISTIC_SPECULATIVE["\']', "dataclass_default_opportunistic"),
    (r'return\s+ClusterLabel\.OPPORTUNISTIC_SPECULATIVE', "return_opportunistic_default"),
    (r'get_persisted_cluster', "sticky_persisted_cluster"),
    (r'persist_cluster', "sticky_persist_cluster"),
    (r'Persistent cluster label', "sticky_by_design_docstring"),
)


def audit_legacy_code_paths(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in CODE_PATHS:
        path = project_root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            for pat, kind in DANGEROUS_PATTERNS:
                if re.search(pat, line):
                    rows.append(
                        {
                            "file_path": rel,
                            "line_no": i,
                            "kind": kind,
                            "line_snippet": line.strip()[:240],
                            "uses_single_axis_cluster_label": True,
                            "default_to_opportunistic": "opportunistic" in kind or "OPPORTUNISTIC" in line,
                            "sticky_persist": "sticky" in kind or "persist" in kind,
                        }
                    )
                    break
    return rows


def load_cluster_registry_rows(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "data" / "cluster_registry.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    for addr, meta in (data or {}).items():
        meta = meta or {}
        rows.append(
            {
                "contract_address": addr,
                "legacy_cluster_label": meta.get("cluster_label"),
                "symbol": meta.get("symbol"),
                "assigned_at": meta.get("assigned_at"),
                "semantic_authoritative": False,
                "soft_expiry_status": "PENDING_REEVALUATION",
                "note": "Sticky registry entry retained for audit only; not semantic_signal_family authority",
            }
        )
    return rows


def build_sticky_expiry_plan(registry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for r in registry_rows:
        plan.append(
            {
                "contract_address": r.get("contract_address"),
                "legacy_cluster_label": r.get("legacy_cluster_label"),
                "soft_expiry_status": "PENDING_REEVALUATION",
                "semantic_authoritative": False,
                "action": "Keep registry row; treat as legacy_cluster_label only; re-evaluate when dual-axis writer active",
                "do_not_delete": True,
                "do_not_rewrite_in_place": True,
            }
        )
    return plan
