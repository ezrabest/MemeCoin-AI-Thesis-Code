"""Derive coin-level adjudications from existing Gemini pair-asset artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .adjudication_schema import ADJUDICATION_CLASSES, UI_LABELS
from .coin_identity import resolve_coin_identity

PAIR_ADJUDICATIONS = Path("data") / "ae12_gemini_asset_adjudications.csv"
PAIR_ASSET_OUT = Path("data") / "ae12_gemini_pair_asset_adjudications.csv"
COIN_LEVEL_OUT = Path("data") / "ae12_gemini_coin_level_adjudications.csv"
PAIR_TO_COIN_OUT = Path("data") / "ae12_gemini_pair_to_coin_mapping.csv"
IDENTITY_AUDIT_OUT = Path("data") / "ae12_gemini_coin_identity_resolution_audit.csv"
COIN_DIST_OUT = Path("data") / "ae12_gemini_coin_level_class_distribution.csv"
COIN_SUMMARY_OUT = Path("reports") / "ae12_gemini_coin_level_summary.json"

CONFIRMED_CLASSES = {
    "SOCIAL_CONFIRMED",
    "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
}


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
        for k in r.keys():
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


def aggregate_coin_class(pair_classes: list[str]) -> tuple[str, str, Counter]:
    """Apply coin-level aggregation rules. Returns class, conflict_note, votes."""
    votes = Counter(c for c in pair_classes if c)
    if not votes:
        return "MANUAL_REVIEW", "NO_PAIR_CLASSES", votes
    unique = set(votes)
    if len(unique) == 1:
        only = next(iter(unique))
        return only, "", votes

    social = "SOCIAL_CONFIRMED" in unique
    opp_conf = "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED" in unique
    if social and opp_conf:
        return "MANUAL_REVIEW", "CONFLICT_SOCIAL_VS_OPPORTUNISTIC_CONFIRMED", votes

    if "MANUAL_REVIEW" in unique and len(unique) > 1:
        return "MANUAL_REVIEW", "MANUAL_REVIEW_MIXED", votes

    confirmed = [c for c in unique if c in CONFIRMED_CLASSES]
    has_suspected = "OPPORTUNISTIC_SUSPECTED" in unique
    if confirmed and has_suspected and len(confirmed) == 1:
        chosen = confirmed[0]
        return chosen, "SUSPECTED_PLUS_CONFIRMED_USED_CONFIRMED", votes
    if confirmed and has_suspected and len(confirmed) > 1:
        if "SOCIAL_CONFIRMED" in confirmed and "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED" in confirmed:
            return "MANUAL_REVIEW", "CONFLICT_SOCIAL_VS_OPPORTUNISTIC_CONFIRMED", votes
        return "MANUAL_REVIEW", "MULTI_CONFIRMED_CONFLICT", votes

    if unique == {"OPPORTUNISTIC_SUSPECTED"}:
        return "OPPORTUNISTIC_SUSPECTED", "", votes

    if unique == {"MANUAL_REVIEW"}:
        return "MANUAL_REVIEW", "", votes

    return "MANUAL_REVIEW", "CLASS_CONFLICT", votes


def build_coin_level_payload(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_enriched: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    identity_audit: list[dict[str, Any]] = []
    by_coin: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in pair_rows:
        ident = resolve_coin_identity(row)
        enriched = {**row, **ident}
        pair_enriched.append(enriched)
        mapping_rows.append(
            {
                "pair_asset_id": row.get("asset_id"),
                "symbol": row.get("symbol"),
                "coin_id": ident["coin_id"],
                "normalized_base_symbol": ident["normalized_base_symbol"],
                "quote_symbol": ident["quote_symbol"],
                "chain": ident["chain"],
                "semantic_coin_class": row.get("semantic_coin_class"),
                "identity_resolution_method": ident["identity_resolution_method"],
            }
        )
        identity_audit.append(
            {
                "pair_asset_id": row.get("asset_id"),
                "symbol": row.get("symbol"),
                "coin_id": ident["coin_id"],
                "identity_resolution_method": ident["identity_resolution_method"],
                "identity_confidence": ident["identity_confidence"],
                "identity_warnings": ident["identity_warnings"],
                "base_symbol": ident["base_symbol"],
                "quote_symbol": ident["quote_symbol"],
                "normalized_base_symbol": ident["normalized_base_symbol"],
                "chain": ident["chain"],
                "token_address": ident["token_address"],
                "pair_address": ident["pair_address"],
            }
        )
        by_coin[ident["coin_id"]].append(enriched)

    coin_rows: list[dict[str, Any]] = []
    conflict_count = 0
    for coin_id, members in sorted(by_coin.items(), key=lambda kv: kv[0]):
        classes = [str(m.get("semantic_coin_class") or "") for m in members]
        coin_class, conflict_note, votes = aggregate_coin_class(classes)
        if conflict_note:
            conflict_count += 1
        methods = Counter(m.get("identity_resolution_method") for m in members)
        warnings = [m.get("identity_warnings") for m in members if m.get("identity_warnings")]
        coin_rows.append(
            {
                "coin_id": coin_id,
                "normalized_base_symbol": members[0].get("normalized_base_symbol"),
                "chain": members[0].get("chain"),
                "name": members[0].get("name") or "",
                "semantic_coin_class": coin_class,
                "ui_label": UI_LABELS.get(coin_class, coin_class),
                "supporting_pair_count": len(members),
                "supporting_pair_asset_ids": ";".join(str(m.get("asset_id") or "") for m in members),
                "class_votes": json.dumps(dict(votes)),
                "conflict_note": conflict_note,
                "identity_resolution_method": methods.most_common(1)[0][0] if methods else "",
                "identity_warning_count": len(warnings),
                "example_symbols": ";".join(sorted({str(m.get("symbol") or "") for m in members})),
            }
        )

    dist = Counter(r["semantic_coin_class"] for r in coin_rows)
    total_coins = len(coin_rows) or 1
    method_dist = Counter(r.get("identity_resolution_method") for r in identity_audit)
    identity_warning_count = sum(1 for r in identity_audit if r.get("identity_warnings"))

    pair_dist = Counter(str(r.get("semantic_coin_class") or "") for r in pair_rows)
    pair_total = len(pair_rows) or 1

    pair_asset_counts = {
        "unique_pair_assets_input": len(pair_rows),
        "pair_social_confirmed_count": int(pair_dist.get("SOCIAL_CONFIRMED", 0)),
        "pair_non_social_opportunistic_confirmed_count": int(
            pair_dist.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0)
        ),
        "pair_opportunistic_suspected_count": int(pair_dist.get("OPPORTUNISTIC_SUSPECTED", 0)),
        "pair_non_social_infrastructure_confirmed_count": int(
            pair_dist.get("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", 0)
        ),
        "pair_manual_review_count": int(pair_dist.get("MANUAL_REVIEW", 0)),
        "count_role": "audit_detail",
    }

    social = int(dist.get("SOCIAL_CONFIRMED", 0))
    opp = int(dist.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0))
    sus = int(dist.get("OPPORTUNISTIC_SUSPECTED", 0))
    infra = int(dist.get("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", 0))
    manual = int(dist.get("MANUAL_REVIEW", 0))
    coin_level_counts = {
        "unique_coins_found": len(coin_rows),
        "coin_social_confirmed_count": social,
        "coin_non_social_opportunistic_confirmed_count": opp,
        "coin_opportunistic_suspected_count": sus,
        "coin_non_social_infrastructure_confirmed_count": infra,
        "coin_manual_review_count": manual,
        "coin_social_confirmed_share": round(social / total_coins, 6),
        "coin_opportunistic_confirmed_share": round(opp / total_coins, 6),
        "coin_opportunistic_suspected_share": round(sus / total_coins, 6),
        "coin_manual_review_share": round(manual / total_coins, 6),
        "count_role": "final_ui",
    }

    distribution_rows = [
        {
            "semantic_coin_class": k,
            "ui_label": UI_LABELS.get(k, k),
            "count": int(dist.get(k, 0)),
            "share": round(int(dist.get(k, 0)) / total_coins, 6),
        }
        for k in ADJUDICATION_CLASSES
    ]

    examples_by_class: dict[str, list[dict[str, Any]]] = {k: [] for k in ADJUDICATION_CLASSES}
    for row in coin_rows:
        k = row["semantic_coin_class"]
        if k in examples_by_class and len(examples_by_class[k]) < 10:
            examples_by_class[k].append(
                {
                    "coin_id": row.get("coin_id"),
                    "symbol": (row.get("example_symbols") or "").split(";")[0],
                    "ui_label": row.get("ui_label"),
                    "semantic_coin_class": k,
                    "supporting_pair_count": row.get("supporting_pair_count"),
                    "conflict_note": row.get("conflict_note"),
                }
            )

    return {
        "pair_enriched": pair_enriched,
        "coin_rows": coin_rows,
        "mapping_rows": mapping_rows,
        "identity_audit": identity_audit,
        "distribution_rows": distribution_rows,
        "pair_asset_counts": pair_asset_counts,
        "coin_level_counts": coin_level_counts,
        "identity_resolution_method_distribution": dict(method_dist),
        "identity_warning_count": identity_warning_count,
        "conflict_count": conflict_count,
        "count_level_used_for_main_ui": "coin_level",
        "examples_by_class": examples_by_class,
        "pair_total": pair_total,
    }


def derive_coin_level_from_root(root: Path, *, write: bool = True) -> dict[str, Any]:
    """Build coin-level aggregates from an existing Gemini adjudication root."""
    root = Path(root)
    pair_path = root / PAIR_ADJUDICATIONS
    pair_rows = _read_csv(pair_path)
    payload = build_coin_level_payload(pair_rows)
    summary = {
        "pair_asset_counts": payload["pair_asset_counts"],
        "coin_level_counts": payload["coin_level_counts"],
        "identity_resolution_method_distribution": payload["identity_resolution_method_distribution"],
        "identity_warning_count": payload["identity_warning_count"],
        "conflict_count": payload["conflict_count"],
        "count_level_used_for_main_ui": "coin_level",
        "examples_by_class": payload["examples_by_class"],
        "gemini_called": False,
        "derived_from": str(pair_path),
    }
    if write:
        _write_csv(root / PAIR_ASSET_OUT, payload["pair_enriched"])
        _write_csv(root / COIN_LEVEL_OUT, payload["coin_rows"])
        _write_csv(root / PAIR_TO_COIN_OUT, payload["mapping_rows"])
        _write_csv(root / IDENTITY_AUDIT_OUT, payload["identity_audit"])
        _write_csv(root / COIN_DIST_OUT, payload["distribution_rows"])
        _write_json(root / COIN_SUMMARY_OUT, summary)
    return summary


def load_or_derive_coin_level(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {}
    root = Path(root)
    summary_path = root / COIN_SUMMARY_OUT
    if summary_path.is_file():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if (root / PAIR_ADJUDICATIONS).is_file():
        return derive_coin_level_from_root(root, write=True)
    return {}
