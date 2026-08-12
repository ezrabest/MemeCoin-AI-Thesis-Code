"""AE12.7 candidate/opportunity discovery from existing AE11/AE12 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _latest_dir(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    dirs = sorted(
        [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _read_csv(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(dict(row))
    return rows


def _dedupe_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = str(r.get("candidate_id") or r.get("pair_address") or r.get("evidence_row_id") or id(r))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def discover_demo_candidates(
    project_root: Path,
    *,
    limit: int = 50,
    synthetic_if_empty: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load recent AE11/AE12 opportunity / missed-winner / evidence rows.
    Sparse data must not freeze the system — synthesizes minimal demo rows if needed.
    """
    audits = project_root / "data" / "audits"
    meta: dict[str, Any] = {
        "sources": [],
        "maturation_root": None,
        "synthetic_used": False,
    }

    mat = _latest_dir(audits, "ae12_forward_evidence_maturation_")
    rows: list[dict[str, Any]] = []
    if mat:
        meta["maturation_root"] = str(mat)
        for name, is_missed in (
            ("ae12_opportunity_capture_full.csv", False),
            ("ae12_candidate_evidence_rows.csv", False),
            ("ae12_missed_winners_full.csv", True),
        ):
            path = mat / "data" / name
            if not path.is_file():
                continue
            chunk = _read_csv(path, limit=limit * 3)
            for r in chunk:
                r["_source_ref"] = str(path)
                if is_missed:
                    r["is_missed_winner"] = True
                # normalize decision id
                if not r.get("source_decision_id") and r.get("decision_id"):
                    r["source_decision_id"] = r.get("decision_id")
            rows.extend(chunk)
            meta["sources"].append(str(path))

        # optional paper linkage
        paper_path = mat / "data" / "ae12_paper_trade_linkage.csv"
        if paper_path.is_file():
            paper_rows = _read_csv(paper_path, limit=limit * 2)
            by_cand = {str(p.get("candidate_id")): p for p in paper_rows if p.get("candidate_id")}
            for r in rows:
                cid = str(r.get("candidate_id") or "")
                if cid and cid in by_cand:
                    p = by_cand[cid]
                    r.setdefault("paper_order_id", p.get("paper_order_id"))
                    r.setdefault("position_id", p.get("position_id"))
            meta["sources"].append(str(paper_path))

    candidates = _dedupe_candidates(rows, limit)

    if not candidates and synthetic_if_empty:
        meta["synthetic_used"] = True
        candidates = [
            {
                "candidate_id": "demo_candidate_001",
                "source_decision_id": "demo_decision_001",
                "pair_address": "So11111111111111111111111111111111111111112",
                "symbol": "DEMO",
                "chain": "solana",
                "strict_shadow_decision": "NO_TRADE",
                "exploration_decision": "WATCH",
                "reason_not_traded": "demo_sparse_data",
                "price_freshness_status": "UNKNOWN",
                "max_return": 0.0,
                "was_traded": False,
                "semantic_signal_family": "UNKNOWN_UNRESOLVED",
                "trading_opportunity_state": "UNKNOWN",
                "legacy_cluster_label": None,
                "_source_ref": "synthetic_demo_row",
                "is_missed_winner": False,
            },
            {
                "candidate_id": "demo_candidate_002",
                "source_decision_id": "demo_decision_002",
                "pair_address": "DemoPairAddressMissingIds",
                "symbol": "MISS",
                "chain": "solana",
                "strict_shadow_decision": "NO_TRADE",
                "exploration_decision": "NO_TRADE",
                "reason_not_traded": "price_price_stale",
                "price_freshness_status": "STALE_AT_ENTRY",
                "max_return": 0.62,
                "was_traded": False,
                "audit_blockers": '["WEAK_LINEAGE","STALE_CONTEXT"]',
                "semantic_signal_family": "UNKNOWN_UNRESOLVED",
                "is_missed_winner": True,
                "missed_winner_horizons": "1h",
                "_source_ref": "synthetic_demo_row",
                # intentionally missing paper_order_id / position_id / context / llm audit
            },
        ][:limit]

    return candidates, meta


def load_latest_ae12_7_root(project_root: Path) -> Path | None:
    return _latest_dir(project_root / "data" / "audits", "ae12_7_intelligent_agent_operational_demo_")


def load_agent_records_from_root(root: Path) -> list[dict[str, Any]]:
    path = root / "data" / "ae12_7_agent_records.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
