"""Build unique-asset evidence packages with priority buckets and linkage audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .evidence_priority import (
    EVIDENCE_PRIORITY_VERSION,
    build_linkage_summary,
    classify_snippet_bucket,
    infer_linkage_method,
    is_runtime_only_snippet,
    linkage_confidence,
    marker_audit_for_text,
    select_priority_snippets,
)


def _safe(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def build_asset_id(row: dict[str, Any]) -> tuple[str, str]:
    chain = _safe(row.get("chain") or row.get("network") or "UNKNOWN").lower()
    token = _safe(
        row.get("token_address")
        or row.get("base_token_address")
        or row.get("contract_address")
        or row.get("token_contract_address")
    ).lower()
    pair = _safe(row.get("pair_address")).lower()
    symbol = _safe(row.get("symbol")).upper()
    if chain and token:
        return f"{chain}:{token}", "HIGH"
    if chain and pair:
        return f"{chain}:PAIR:{pair}", "MEDIUM"
    if chain and symbol:
        return f"{chain}:SYMBOL:{symbol}", "LOW_CONFIDENCE_IDENTITY"
    return "UNKNOWN:UNRESOLVED", "LOW_CONFIDENCE_IDENTITY"


def _hash_evidence(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_candidate_rows(ae12_root: Path, max_rows: int = 100000) -> list[dict[str, Any]]:
    path = ae12_root / "data" / "ae12_candidate_evidence_rows.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= max_rows:
                break
            rows.append(dict(row))
    return rows


def _load_sqlite_hints(project_root: Path, *, limit: int = 5000) -> dict[str, list[dict[str, Any]]]:
    db_path = project_root / "data" / "trader.db"
    out: dict[str, list[dict[str, Any]]] = {"coins": [], "sentiment_records": [], "signals": []}
    if not db_path.is_file():
        return out
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for table in ("coins", "sentiment_records", "signals"):
                try:
                    q = f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {int(limit)}"
                    out[table] = [dict(r) for r in conn.execute(q).fetchall()]
                except sqlite3.Error:
                    out[table] = []
        finally:
            conn.close()
    except Exception:
        return out
    return out


def _make_snippet(
    *,
    asset: dict[str, Any],
    text: str,
    snippet_type: str,
    source_table_or_file: str,
    source_row_id: str = "",
    source_timestamp: str = "",
    matched_on: str = "",
) -> dict[str, Any]:
    text = text[:300]
    bucket = classify_snippet_bucket(text, snippet_type=snippet_type)
    if is_runtime_only_snippet(text):
        bucket = "runtime_context"
    method = infer_linkage_method(asset=asset, source_table_or_file=source_table_or_file, matched_on=matched_on or text)
    markers = marker_audit_for_text(text)
    return {
        "text": text,
        "snippet_type": snippet_type,
        "bucket": bucket,
        "source_table_or_file": source_table_or_file,
        "source_row_id": source_row_id,
        "source_timestamp": source_timestamp,
        "linkage_method": method,
        "linkage_confidence": linkage_confidence(method),
        "matched_on": matched_on,
        "evidence_text_preview": text[:120],
        **markers,
        "used_in_classifier_evidence": False,
        "reason_if_not_used": "",
    }


def _linkage_row_from_snippet(asset: dict[str, Any], sn: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": asset.get("asset_id"),
        "chain": asset.get("chain"),
        "token_address": asset.get("token_address"),
        "contract_address": asset.get("token_address"),
        "pair_address": asset.get("pair_address"),
        "symbol": asset.get("symbol"),
        "name": asset.get("name"),
        "source_table_or_file": sn.get("source_table_or_file"),
        "source_row_id": sn.get("source_row_id"),
        "source_timestamp": sn.get("source_timestamp"),
        "snippet_type": sn.get("snippet_type"),
        "linkage_method": sn.get("linkage_method"),
        "linkage_confidence": sn.get("linkage_confidence"),
        "matched_on": sn.get("matched_on"),
        "time_delta_seconds": "",
        "evidence_text_preview": sn.get("evidence_text_preview"),
        "semantic_markers_found": json.dumps(sn.get("semantic_markers_found") or []),
        "negative_markers_found": json.dumps(sn.get("negative_markers_found") or []),
        "used_in_classifier_evidence": sn.get("used_in_classifier_evidence"),
        "reason_if_not_used": sn.get("reason_if_not_used"),
    }


def build_unique_asset_evidence_with_linkage(
    *,
    project_root: Path,
    ae12_root: Path,
    max_assets: int = 1000,
    max_snippets: int = 12,
    max_chars: int = 6000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = load_candidate_rows(ae12_root, max_rows=max_assets * 500)
    sqlite_hints = _load_sqlite_hints(project_root, limit=8000)
    by_asset: dict[str, dict[str, Any]] = {}
    all_raw_snippets: dict[str, list[dict[str, Any]]] = {}

    for i, row in enumerate(rows):
        asset_id, identity_conf = build_asset_id(row)
        if asset_id not in by_asset:
            by_asset[asset_id] = {
                "asset_id": asset_id,
                "identity_confidence": identity_conf,
                "chain": _safe(row.get("chain") or row.get("network")),
                "token_address": _safe(
                    row.get("token_address")
                    or row.get("base_token_address")
                    or row.get("contract_address")
                    or row.get("token_contract_address")
                ),
                "pair_address": _safe(row.get("pair_address")),
                "symbol": _safe(row.get("symbol")),
                "name": _safe(row.get("name")),
                "legacy_cluster_label": _safe(row.get("cluster_label")),
                "trading_opportunity_state_summary": {},
                "source_rows_used": 0,
                "source_files": set(),
            }
            all_raw_snippets[asset_id] = []
        rec = by_asset[asset_id]
        rec["source_rows_used"] += 1
        src = _safe(row.get("source_file") or "ae12_candidate_evidence_rows.csv")
        rec["source_files"].add(src)
        state = _safe(
            row.get("trading_opportunity_state")
            or row.get("exploration_decision")
            or row.get("strict_shadow_decision")
            or "UNKNOWN"
        )
        rec["trading_opportunity_state_summary"][state] = rec["trading_opportunity_state_summary"].get(state, 0) + 1

        for field, stype in (
            ("reason_for_no_trade", "candidate_reason"),
            ("reason_not_traded", "candidate_reason"),
            ("rejection_reason", "candidate_reason"),
            ("qwen_linkage_status", "llm_context"),
            ("llm_verdict", "llm_context"),
            ("llm_context", "llm_context"),
            ("narrative", "narrative"),
        ):
            txt = _safe(row.get(field))
            if txt:
                all_raw_snippets[asset_id].append(
                    _make_snippet(
                        asset=rec,
                        text=txt,
                        snippet_type=stype,
                        source_table_or_file=src,
                        source_row_id=str(i),
                        source_timestamp=_safe(row.get("timestamp") or row.get("created_at")),
                        matched_on=f"{rec.get('symbol')} {rec.get('pair_address')}",
                    )
                )
        if len(by_asset) >= max_assets and i > max_assets * 10:
            pass

    # Identity snippets (priority 1) — always add for each asset
    for rec in by_asset.values():
        asset_id = rec["asset_id"]
        if rec.get("symbol"):
            all_raw_snippets[asset_id].append(
                _make_snippet(
                    asset=rec,
                    text=f"symbol:{rec['symbol']}",
                    snippet_type="identity",
                    source_table_or_file="identity_metadata",
                    matched_on=rec.get("symbol", ""),
                )
            )
        if rec.get("name"):
            all_raw_snippets[asset_id].append(
                _make_snippet(
                    asset=rec,
                    text=f"name:{rec['name']}",
                    snippet_type="identity",
                    source_table_or_file="identity_metadata",
                    matched_on=rec.get("name", ""),
                )
            )
        if rec.get("token_address"):
            all_raw_snippets[asset_id].append(
                _make_snippet(
                    asset=rec,
                    text=f"contract:{rec['token_address']}",
                    snippet_type="identity",
                    source_table_or_file="identity_metadata",
                    matched_on=rec.get("token_address", ""),
                )
            )
        if rec.get("legacy_cluster_label"):
            all_raw_snippets[asset_id].append(
                _make_snippet(
                    asset=rec,
                    text=f"legacy_cluster:{rec['legacy_cluster_label']}",
                    snippet_type="legacy_cluster",
                    source_table_or_file="cluster_registry.json",
                    matched_on="legacy_cluster",
                )
            )

    coin_rows = sqlite_hints.get("coins", [])
    sent_rows = sqlite_hints.get("sentiment_records", [])
    for rec in by_asset.values():
        asset_id = rec["asset_id"]
        sym = rec.get("symbol", "").upper()
        pair = rec.get("pair_address", "").lower()
        for c in coin_rows:
            if sym and _safe(c.get("symbol")).upper() == sym:
                for k in ("name", "description", "tags", "notes"):
                    txt = _safe(c.get(k))
                    if txt:
                        all_raw_snippets[asset_id].append(
                            _make_snippet(
                                asset=rec,
                                text=f"coins.{k}:{txt[:220]}",
                                snippet_type="identity" if k in {"name", "description"} else "semantic",
                                source_table_or_file="coins",
                                source_row_id=str(c.get("id") or c.get("rowid") or ""),
                                source_timestamp=_safe(c.get("updated_at") or c.get("created_at")),
                                matched_on=sym,
                            )
                        )
        for s in sent_rows:
            sym_hit = sym and sym in _safe(s.get("symbol")).upper()
            pair_hit = pair and pair in _safe(s.get("token_contract_address")).lower()
            if sym_hit or pair_hit:
                txt = " ".join(
                    _safe(s.get(k)) for k in ("title", "text", "content", "summary", "source") if _safe(s.get(k))
                )
                if txt:
                    all_raw_snippets[asset_id].append(
                        _make_snippet(
                            asset=rec,
                            text=txt[:220],
                            snippet_type="sentiment",
                            source_table_or_file="sentiment_records",
                            source_row_id=str(s.get("id") or s.get("rowid") or ""),
                            source_timestamp=_safe(s.get("created_at") or s.get("timestamp")),
                            matched_on=sym or pair,
                        )
                    )

    out: list[dict[str, Any]] = []
    linkage_rows: list[dict[str, Any]] = []
    for asset_id, rec in list(by_asset.items())[:max_assets]:
        raw = all_raw_snippets.get(asset_id, [])
        selected, priority_audit = select_priority_snippets(raw, max_total=max_snippets)
        snippets = [s.get("text", "") for s in selected]
        joined = "\n".join(snippets)[:max_chars]
        markers = marker_audit_for_text(joined)
        rec["snippets"] = snippets
        rec["source_files"] = sorted(rec.get("source_files", []))
        rec["source_count"] = len(rec["source_files"])
        rec["evidence_hash"] = _hash_evidence(
            {
                "asset_id": rec["asset_id"],
                "symbol": rec.get("symbol"),
                "name": rec.get("name"),
                "snippets": snippets,
                "states": rec.get("trading_opportunity_state_summary"),
                "evidence_priority_version": EVIDENCE_PRIORITY_VERSION,
            }
        )
        rec["evidence_text"] = joined
        rec["evidence_priority_version"] = EVIDENCE_PRIORITY_VERSION
        rec.update(markers)
        rec.update(priority_audit)
        out.append(rec)
        for sn in raw:
            lr = _linkage_row_from_snippet(rec, sn)
            for sel in selected:
                if sel.get("text") == sn.get("text") and sel.get("source_row_id") == sn.get("source_row_id"):
                    lr["used_in_classifier_evidence"] = True
                    break
            linkage_rows.append(lr)

    linkage_summary = build_linkage_summary(linkage_rows)
    linkage_summary["evidence_priority_version"] = EVIDENCE_PRIORITY_VERSION
    return out, linkage_rows, linkage_summary


def build_unique_asset_evidence(
    *,
    project_root: Path,
    ae12_root: Path,
    max_assets: int = 1000,
    max_snippets: int = 12,
    max_chars: int = 6000,
) -> list[dict[str, Any]]:
    evidence, _, _ = build_unique_asset_evidence_with_linkage(
        project_root=project_root,
        ae12_root=ae12_root,
        max_assets=max_assets,
        max_snippets=max_snippets,
        max_chars=max_chars,
    )
    return evidence
