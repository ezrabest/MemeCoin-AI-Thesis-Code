#!/usr/bin/env python3
"""AE19 Historical Missed-Winner Review — dual-provider audit (Qwen3 + Gemini).

Audit-only. Does NOT: start AE20, train/backtest, mutate trader.db, open/close
paper trades, connect wallet, change thresholds/execution logic, or grant LLM
trade authority.

Providers run sequentially (ollama then gemini). Compact LLM input only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASE = "AE19_HISTORICAL_MISSED_WINNER_REVIEW"
AUTHORITY_STATUS = "AUDIT_ONLY_NO_TRADE_AUTHORITY"
REVIEW_TYPE = "AE19_HISTORICAL_MISSED_WINNER_REVIEW"
GPU_CONSTRAINT_NOTE = (
    "RTX 5070 12GB VRAM; providers run sequentially; compact prompt used for local Qwen3"
)
REQUIRED_PROVIDERS = ["ollama", "gemini"]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "audits"
    / "ae12_forward_evidence_maturation_20260714_235401"
    / "data"
    / "ae12_missed_winners_full.csv"
)

REQUIRED_COLUMNS = [
    "evidence_row_id",
    "candidate_id",
    "decision_id",
    "pair_address",
    "first_seen_timestamp",
    "horizon",
    "max_return",
    "threshold",
    "was_traded",
    "strict_shadow_decision",
    "exploration_decision",
    "reason_not_traded",
    "rejection_reason",
    "price_freshness_status",
    "context_missingness",
    "audit_blockers",
    "cooldown_active",
    "max_open_positions_hit",
    "duplicate_active_pair",
    "duplicate_reason",
    "no_lookahead_status",
]

LLM_ROW_FIELDS = list(REQUIRED_COLUMNS)

# Smaller row projection for LLM prompts (still covers audit-critical fields).
COMPACT_LLM_ROW_FIELDS = [
    "evidence_row_id",
    "candidate_id",
    "decision_id",
    "pair_address",
    "horizon",
    "max_return",
    "was_traded",
    "reason_not_traded",
    "rejection_reason",
    "price_freshness_status",
    "context_missingness",
    "audit_blockers",
    "cooldown_active",
    "max_open_positions_hit",
    "duplicate_active_pair",
    "no_lookahead_status",
]

# Local Qwen3:8b on 12GB cannot ingest ~40k-char prompts under default context.
# Shared package may be up to max_input_chars; Ollama prompt uses a compact derivative.
OLLAMA_MAX_PROMPT_CHARS = 8000
OLLAMA_MAX_EVIDENCE_CHARS = 5500


def ollama_review_json_schema(*, provider: str, model: str) -> dict[str, Any]:
    """Structured-output schema for Ollama format= (forces flat review object)."""
    return {
        "type": "object",
        "properties": {
            "review_type": {"type": "string"},
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "source_file": {"type": "string"},
            "authority_status": {"type": "string"},
            "execution_allowed": {"type": "boolean"},
            "paper_execution_allowed": {"type": "boolean"},
            "live_execution_allowed": {"type": "boolean"},
            "risk_override_allowed": {"type": "boolean"},
            "execution_attempted": {"type": "boolean"},
            "profitability_claimed": {"type": "boolean"},
            "summary": {"type": "string"},
            "top_findings": {"type": "array", "items": {"type": "string"}},
            "dominant_missed_winner_causes": {"type": "array", "items": {"type": "string"}},
            "deduplication_warning": {"type": "string"},
            "no_lookahead_assessment": {"type": "string"},
            "top_case_review": {"type": "object"},
            "blocker_interpretation": {"type": "object"},
            "model_specific_observations": {"type": "array", "items": {"type": "string"}},
            "recommended_research_followups": {"type": "array", "items": {"type": "string"}},
            "not_allowed_actions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "review_type",
            "provider",
            "model",
            "authority_status",
            "execution_allowed",
            "paper_execution_allowed",
            "live_execution_allowed",
            "risk_override_allowed",
            "execution_attempted",
            "profitability_claimed",
            "summary",
            "dominant_missed_winner_causes",
            "no_lookahead_assessment",
        ],
    }


def build_ollama_evidence_package(package: dict[str, Any]) -> dict[str, Any]:
    """Ultra-compact evidence for local Qwen3 — summaries + tiny case snippets only."""
    pairs = package.get("top_deduplicated_pairs_by_max_return") or []
    cands = package.get("top_deduplicated_candidates_by_max_return") or []
    raw = package.get("top_raw_rows_by_max_return") or []

    def _short_case(row: dict[str, Any]) -> dict[str, Any]:
        pair = str(row.get("pair_address") or "")
        return {
            "pair": (pair[:10] + "…" + pair[-6:]) if len(pair) > 20 else pair,
            "horizon": row.get("horizon"),
            "max_return": row.get("max_return"),
            "reason": row.get("reason_not_traded"),
            "freshness": row.get("price_freshness_status"),
            "blockers": row.get("audit_blockers"),
            "lookahead": row.get("no_lookahead_status"),
            "dup_rows": row.get("duplicate_row_count"),
        }

    return {
        "review_type": REVIEW_TYPE,
        "authority_status": AUTHORITY_STATUS,
        "source_path": package.get("source_path"),
        "gpu_constraint_note": GPU_CONSTRAINT_NOTE,
        "duplicate_heavy_warning": package.get("duplicate_heavy_warning"),
        "global_summary": package.get("global_summary"),
        "horizon_counts": package.get("horizon_counts"),
        "reason_not_traded_top": (package.get("reason_not_traded_distribution") or [])[:8],
        "audit_blockers_top": (package.get("audit_blockers_distribution") or [])[:8],
        "price_freshness_top": (package.get("price_freshness_status_distribution") or [])[:5],
        "no_lookahead_top": (package.get("no_lookahead_status_distribution") or [])[:5],
        "top_raw_short": [_short_case(r) for r in raw[:5]],
        "top_pairs_short": [_short_case(r) for r in pairs[:5]],
        "top_candidates_short": [_short_case(r) for r in cands[:5]],
    }

FALSE_AUTHORITY_FIELDS = [
    "execution_allowed",
    "paper_execution_allowed",
    "live_execution_allowed",
    "risk_override_allowed",
    "execution_attempted",
    "profitability_claimed",
]

REQUIRED_PARSED_FIELDS = [
    "review_type",
    "provider",
    "model",
    "authority_status",
    *FALSE_AUTHORITY_FIELDS,
    "summary",
    "dominant_missed_winner_causes",
    "no_lookahead_assessment",
]

STATUS_PASS = "AE19_HISTORICAL_MISSED_WINNER_REVIEW_PASS"
STATUS_PARTIAL_FAILURE = "AE19_HISTORICAL_MISSED_WINNER_REVIEW_PARTIAL_PROVIDER_FAILURE"
STATUS_PARTIAL_DEBUG = "AE19_HISTORICAL_MISSED_WINNER_REVIEW_PARTIAL_PROVIDER_DEBUG_ONLY"
STATUS_DETERMINISTIC_ONLY = "AE19_HISTORICAL_MISSED_WINNER_REVIEW_DETERMINISTIC_ONLY_NOT_AE19_FINAL"
STATUS_FAIL = "AE19_HISTORICAL_MISSED_WINNER_REVIEW_FAIL"

DUPLICATE_WARNING = (
    "WARNING: Raw top rows by max_return are duplicate-heavy and often dominated by "
    "repeated rows for the same pair_address (notably "
    "0xe7A3811098193cCf75EDEE15B4b45f2BE23D7801). Do not overcount raw top rows; "
    "prefer deduplicated pair/candidate views for case interpretation."
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, _, val = raw.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else ["value"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def sanitize_error(exc: BaseException | str) -> str:
    text = str(exc)
    text = re.sub(r"(api[_-]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", text, flags=re.I)
    text = re.sub(r"(Bearer\s+)\S+", r"\1[REDACTED]", text, flags=re.I)
    text = re.sub(r"AIza[0-9A-Za-z\-_]{10,}", "[REDACTED_KEY]", text)
    return text[:800]


# ---------------------------------------------------------------------------
# CSV load / deterministic summaries
# ---------------------------------------------------------------------------


def load_missed_winner_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]
    return rows, fieldnames


def validate_required_columns(fieldnames: list[str]) -> list[str]:
    present = set(fieldnames)
    return [c for c in REQUIRED_COLUMNS if c not in present]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_audit_blockers(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text or text in ("[]", "null", "None"):
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except json.JSONDecodeError:
        pass
    # fallback: comma-separated without JSON
    if "," in text and "[" not in text:
        return [p.strip().strip('"').strip("'") for p in text.split(",") if p.strip()]
    return [text]


def count_distribution(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        key = str(row.get(field, "") or "").strip() or "(empty)"
        counter[key] += 1
    return [{"key": k, "count": c} for k, c in counter.most_common()]


def count_blocker_distribution(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        blockers = parse_audit_blockers(row.get("audit_blockers"))
        if not blockers:
            counter["(none)"] += 1
        else:
            for b in blockers:
                counter[b] += 1
    return [{"blocker": k, "count": c} for k, c in counter.most_common()]


def top_raw_rows(rows: list[dict[str, str]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: _to_float(r.get("max_return")), reverse=True)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ranked[:top_n], start=1):
        compact = {k: row.get(k, "") for k in LLM_ROW_FIELDS}
        compact["rank"] = i
        compact["max_return"] = _to_float(row.get("max_return"))
        out.append(compact)
    return out


def project_compact_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k, "") for k in COMPACT_LLM_ROW_FIELDS}
    if "rank" in row:
        out["rank"] = row["rank"]
    if "duplicate_row_count" in row:
        out["duplicate_row_count"] = row["duplicate_row_count"]
    if "max_return" in row:
        out["max_return"] = row["max_return"]
    return out


def top_dedup_by_key(rows: list[dict[str, str]], key: str, top_n: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    dup_counts: Counter[str] = Counter()
    for row in rows:
        k = str(row.get(key, "") or "").strip() or "(empty)"
        dup_counts[k] += 1
        mr = _to_float(row.get("max_return"))
        prev = best.get(k)
        if prev is None or mr > _to_float(prev.get("max_return")):
            compact = {f: row.get(f, "") for f in LLM_ROW_FIELDS}
            compact["max_return"] = mr
            compact["dedup_key"] = k
            best[k] = compact
    ranked = sorted(best.values(), key=lambda r: _to_float(r.get("max_return")), reverse=True)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ranked[:top_n], start=1):
        row = dict(row)
        row["rank"] = i
        row["duplicate_row_count"] = int(dup_counts[str(row.get("dedup_key"))])
        out.append(row)
    return out


def build_deterministic_summary(
    rows: list[dict[str, str]],
    *,
    source_path: Path,
    top_n: int,
) -> dict[str, Any]:
    horizon_counts = {h: 0 for h in ("5m", "15m", "1h", "6h", "24h")}
    for row in rows:
        h = str(row.get("horizon", "")).strip()
        if h in horizon_counts:
            horizon_counts[h] += 1
        else:
            horizon_counts[h] = horizon_counts.get(h, 0) + 1

    reason_dist = count_distribution(rows, "reason_not_traded")
    rejection_dist = count_distribution(rows, "rejection_reason")
    freshness_dist = count_distribution(rows, "price_freshness_status")
    context_dist = count_distribution(rows, "context_missingness")
    lookahead_dist = count_distribution(rows, "no_lookahead_status")
    was_traded_dist = count_distribution(rows, "was_traded")
    blocker_dist = count_blocker_distribution(rows)

    top_raw = top_raw_rows(rows, top_n)
    top_pairs = top_dedup_by_key(rows, "pair_address", top_n)
    top_cands = top_dedup_by_key(rows, "candidate_id", top_n)

    return {
        "source_path": str(source_path.resolve()),
        "total_rows": len(rows),
        "horizon_counts": horizon_counts,
        "unique_pair_count": len({str(r.get("pair_address", "")).strip() for r in rows if str(r.get("pair_address", "")).strip()}),
        "unique_candidate_count": len({str(r.get("candidate_id", "")).strip() for r in rows if str(r.get("candidate_id", "")).strip()}),
        "unique_decision_count": len({str(r.get("decision_id", "")).strip() for r in rows if str(r.get("decision_id", "")).strip()}),
        "unique_evidence_row_count": len({str(r.get("evidence_row_id", "")).strip() for r in rows if str(r.get("evidence_row_id", "")).strip()}),
        "reason_not_traded_distribution": reason_dist,
        "rejection_reason_distribution": rejection_dist,
        "price_freshness_status_distribution": freshness_dist,
        "context_missingness_distribution": context_dist,
        "no_lookahead_status_distribution": lookahead_dist,
        "was_traded_distribution": was_traded_dist,
        "audit_blockers_distribution": blocker_dist,
        "top_raw_rows": top_raw,
        "top_pairs_dedup": top_pairs,
        "top_candidates_dedup": top_cands,
        "duplicate_heavy_warning": DUPLICATE_WARNING,
        "authority_boundary": AUTHORITY_STATUS,
    }


# ---------------------------------------------------------------------------
# Compact LLM input package
# ---------------------------------------------------------------------------


def _package_dict(
    summary: dict[str, Any],
    *,
    n_raw: int,
    n_pairs: int,
    n_cands: int,
    max_llm_rows: int,
) -> dict[str, Any]:
    # Cap list sizes used in package; also honor max_llm_rows as hard ceiling on row-like lists
    n_raw = max(0, min(n_raw, max_llm_rows))
    n_pairs = max(0, min(n_pairs, max_llm_rows))
    n_cands = max(0, min(n_cands, max_llm_rows))
    return {
        "review_type": REVIEW_TYPE,
        "authority_status": AUTHORITY_STATUS,
        "authority_boundary": (
            "AUDIT_ONLY_NO_TRADE_AUTHORITY. execution_allowed=false. "
            "paper_execution_allowed=false. live_execution_allowed=false. "
            "risk_override_allowed=false. Do not claim profitability or approve trades."
        ),
        "source_path": summary["source_path"],
        "gpu_constraint_note": GPU_CONSTRAINT_NOTE,
        "duplicate_heavy_warning": summary["duplicate_heavy_warning"],
        "global_summary": {
            "total_rows": summary["total_rows"],
            "unique_pair_count": summary["unique_pair_count"],
            "unique_candidate_count": summary["unique_candidate_count"],
            "unique_decision_count": summary["unique_decision_count"],
            "unique_evidence_row_count": summary["unique_evidence_row_count"],
        },
        "horizon_counts": summary["horizon_counts"],
        "reason_not_traded_distribution": summary["reason_not_traded_distribution"][:25],
        "rejection_reason_distribution": summary["rejection_reason_distribution"][:25],
        "price_freshness_status_distribution": summary["price_freshness_status_distribution"][:25],
        "context_missingness_distribution": summary["context_missingness_distribution"][:25],
        "no_lookahead_status_distribution": summary["no_lookahead_status_distribution"][:25],
        "audit_blockers_distribution": summary["audit_blockers_distribution"][:25],
        "was_traded_distribution": summary["was_traded_distribution"][:10],
        "top_raw_rows_by_max_return": [project_compact_row(r) for r in summary["top_raw_rows"][:n_raw]],
        "top_deduplicated_pairs_by_max_return": [
            project_compact_row(r) for r in summary["top_pairs_dedup"][:n_pairs]
        ],
        "top_deduplicated_candidates_by_max_return": [
            project_compact_row(r) for r in summary["top_candidates_dedup"][:n_cands]
        ],
        "required_output_schema_reminder": {
            "review_type": REVIEW_TYPE,
            "authority_status": AUTHORITY_STATUS,
            "execution_allowed": False,
            "paper_execution_allowed": False,
            "live_execution_allowed": False,
            "risk_override_allowed": False,
            "execution_attempted": False,
            "profitability_claimed": False,
        },
    }


def build_llm_input_package(
    summary: dict[str, Any],
    *,
    top_n: int,
    max_llm_rows: int,
    max_input_chars: int,
) -> dict[str, Any]:
    n_raw = min(top_n, max_llm_rows)
    n_pairs = min(top_n, max_llm_rows)
    n_cands = min(top_n, max_llm_rows)
    truncated = False
    truncation_steps: list[str] = []

    while True:
        package = _package_dict(
            summary,
            n_raw=n_raw,
            n_pairs=n_pairs,
            n_cands=n_cands,
            max_llm_rows=max_llm_rows,
        )
        encoded = json.dumps(package, default=str, separators=(",", ":"))
        chars = len(encoded)
        if chars <= max_input_chars:
            break
        # Deterministic shrink: reduce largest of the three lists by 1, preserving min 5
        truncated = True
        candidates = []
        if n_raw > 5:
            candidates.append(("raw", n_raw))
        if n_pairs > 5:
            candidates.append(("pairs", n_pairs))
        if n_cands > 5:
            candidates.append(("cands", n_cands))
        if not candidates:
            # Already at floor; still over limit — drop distribution tails aggressively
            for key in (
                "reason_not_traded_distribution",
                "rejection_reason_distribution",
                "price_freshness_status_distribution",
                "context_missingness_distribution",
                "audit_blockers_distribution",
                "was_traded_distribution",
                "no_lookahead_status_distribution",
            ):
                cur = package.get(key) or []
                if len(cur) > 3:
                    package[key] = cur[:3]
                    truncation_steps.append(f"{key}->3")
            encoded = json.dumps(package, default=str, separators=(",", ":"))
            chars = len(encoded)
            if chars <= max_input_chars:
                break
            for key in (
                "reason_not_traded_distribution",
                "rejection_reason_distribution",
                "price_freshness_status_distribution",
                "context_missingness_distribution",
                "audit_blockers_distribution",
                "was_traded_distribution",
                "no_lookahead_status_distribution",
            ):
                if package.get(key):
                    package[key] = (package.get(key) or [])[:1]
            truncation_steps.append("distributions_reduced_to_1")
            encoded = json.dumps(package, default=str, separators=(",", ":"))
            chars = len(encoded)
            if chars > max_input_chars:
                truncation_steps.append("min_floor_reached_still_over_limit")
            break

        # Reduce the currently largest list
        candidates.sort(key=lambda x: x[1], reverse=True)
        which = candidates[0][0]
        if which == "raw":
            n_raw -= 1
            truncation_steps.append(f"n_raw->{n_raw}")
        elif which == "pairs":
            n_pairs -= 1
            truncation_steps.append(f"n_pairs->{n_pairs}")
        else:
            n_cands -= 1
            truncation_steps.append(f"n_cands->{n_cands}")

    # Rebuild with final n_* then re-apply any distribution truncations from steps
    package = _package_dict(
        summary,
        n_raw=n_raw,
        n_pairs=n_pairs,
        n_cands=n_cands,
        max_llm_rows=max_llm_rows,
    )
    if any("->3" in s for s in truncation_steps) or "distributions_reduced_to_1" in truncation_steps:
        target = 1 if "distributions_reduced_to_1" in truncation_steps else 3
        for key in (
            "reason_not_traded_distribution",
            "rejection_reason_distribution",
            "price_freshness_status_distribution",
            "context_missingness_distribution",
            "audit_blockers_distribution",
            "was_traded_distribution",
            "no_lookahead_status_distribution",
        ):
            package[key] = (package.get(key) or [])[:target]

    encoded = json.dumps(package, default=str, separators=(",", ":"))
    # If still over after rebuild, keep shrinking row lists is not allowed below 5;
    # strip optional verbose keys from schema reminder only as last resort.
    if len(encoded) > max_input_chars:
        package.pop("required_output_schema_reminder", None)
        truncation_steps.append("dropped_schema_reminder")
        encoded = json.dumps(package, default=str, separators=(",", ":"))

    meta = {
        "input_truncated": truncated or len(encoded) > max_input_chars or bool(truncation_steps),
        "input_package_chars": len(encoded),
        "max_input_chars": max_input_chars,
        "n_raw_included": min(n_raw, len(summary["top_raw_rows"])),
        "n_pairs_included": min(n_pairs, len(summary["top_pairs_dedup"])),
        "n_candidates_included": min(n_cands, len(summary["top_candidates_dedup"])),
        "truncation_steps": truncation_steps,
        "full_csv_rows_excluded": True,
        "total_source_rows": summary["total_rows"],
    }
    package["_meta"] = meta
    return package


def render_llm_prompt(
    package: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_evidence_chars: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (prompt_text, prompt_meta). Compact prompt for constrained local models."""
    schema_keys = {
        "review_type": REVIEW_TYPE,
        "provider": provider,
        "model": model,
        "source_file": package.get("source_path"),
        "authority_status": AUTHORITY_STATUS,
        "execution_allowed": False,
        "paper_execution_allowed": False,
        "live_execution_allowed": False,
        "risk_override_allowed": False,
        "execution_attempted": False,
        "profitability_claimed": False,
        "summary": "string",
        "top_findings": ["string"],
        "dominant_missed_winner_causes": ["string"],
        "deduplication_warning": "string",
        "no_lookahead_assessment": "string",
        "top_case_review": {},
        "blocker_interpretation": {},
        "model_specific_observations": ["string"],
        "recommended_research_followups": ["string"],
        "not_allowed_actions": [
            "no trade authority",
            "no execution",
            "no live approval",
            "no risk override",
        ],
    }

    if provider == "ollama":
        payload: dict[str, Any] = build_ollama_evidence_package(package)
    else:
        base = {k: v for k, v in package.items() if k != "_meta"}
        payload = dict(base)
        for list_key in (
            "top_raw_rows_by_max_return",
            "top_deduplicated_pairs_by_max_return",
            "top_deduplicated_candidates_by_max_return",
        ):
            rows = payload.get(list_key) or []
            payload[list_key] = [project_compact_row(r) for r in rows]

    prompt_truncated = False
    truncation_steps: list[str] = []

    def _encode(p: dict[str, Any]) -> str:
        return json.dumps(p, default=str, separators=(",", ":"))

    evidence = _encode(payload)
    limit = max_evidence_chars
    if limit is not None and len(evidence) > limit:
        prompt_truncated = True
        for key in ("top_raw_short", "top_pairs_short", "top_candidates_short"):
            if key in payload and len(payload[key]) > 3:
                payload[key] = payload[key][:3]
                truncation_steps.append(f"{key}->3")
        for key in (
            "reason_not_traded_top",
            "audit_blockers_top",
            "price_freshness_top",
            "no_lookahead_top",
            "reason_not_traded_distribution",
            "audit_blockers_distribution",
        ):
            if key in payload and isinstance(payload[key], list) and len(payload[key]) > 3:
                payload[key] = payload[key][:3]
                truncation_steps.append(f"{key}->3")
        evidence = _encode(payload)
        if len(evidence) > limit:
            for key in ("top_raw_short", "top_pairs_short", "top_candidates_short"):
                if key in payload:
                    payload[key] = (payload.get(key) or [])[:2]
            truncation_steps.append("short_cases->2")
            evidence = _encode(payload)

    schema = json.dumps(schema_keys, separators=(",", ":"))
    prompt = (
        "AE19 historical missed-winner AUDIT ONLY. NO trade authority.\n"
        "Return ONE flat JSON object matching SCHEMA keys exactly.\n"
        "Do NOT echo evidence arrays. Do NOT invent UUIDs or zero addresses.\n"
        f"Set provider={provider!r} and model={model!r}.\n"
        "Fill summary, dominant_missed_winner_causes, deduplication_warning, "
        "no_lookahead_assessment, and blocker_interpretation "
        "(stale price/exploration + weak lineage/context).\n"
        "CRITICAL authority flags — all must be boolean false:\n"
        "execution_allowed, paper_execution_allowed, live_execution_allowed, "
        "risk_override_allowed, execution_attempted, profitability_claimed.\n"
        "execution_attempted means a trade/paper/live order was attempted — "
        "writing this audit does NOT make execution_attempted true; keep it false.\n"
        f"SCHEMA:{schema}\n"
        f"EVIDENCE:{evidence}\n"
    )
    meta = {
        "provider": provider,
        "prompt_chars": len(prompt),
        "evidence_chars": len(evidence),
        "prompt_truncated": prompt_truncated or (limit is not None and len(evidence) > limit),
        "max_evidence_chars": limit,
        "ollama_ultra_compact": provider == "ollama",
        "truncation_steps": truncation_steps,
    }
    return prompt, meta


# ---------------------------------------------------------------------------
# Shared JSON parser / schema validation
# ---------------------------------------------------------------------------


def _strip_bom_and_ws(text: str) -> str:
    return text.lstrip("\ufeff").strip()


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    # Full fence wrap
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Leading fence with trailing content
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_provider_json(raw_text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return (parsed_dict_or_None, parse_strategy, error)."""
    text = _strip_bom_and_ws(raw_text or "")
    if not text:
        return None, "failed", "empty_response"

    # 1) direct
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "direct_json", None
    except json.JSONDecodeError:
        pass

    # 2) strip markdown fences
    stripped = _strip_markdown_fences(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, "stripped_markdown_fence", None
    except json.JSONDecodeError:
        pass

    # 3) extract first balanced top-level object
    extracted = _extract_balanced_json_object(stripped) or _extract_balanced_json_object(text)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed, "extracted_balanced_json", None
        except json.JSONDecodeError as exc:
            return None, "failed", f"extracted_json_decode_error:{exc}"

    return None, "failed", "no_valid_json_object"


def validate_review_schema(
    parsed: dict[str, Any],
    *,
    expected_provider: str,
) -> tuple[bool, bool, list[str]]:
    """Return (schema_success, authority_success, errors)."""
    errors: list[str] = []
    for field in REQUIRED_PARSED_FIELDS:
        if field not in parsed:
            errors.append(f"missing_field:{field}")

    schema_ok = all(f in parsed for f in REQUIRED_PARSED_FIELDS)

    # Soft provider name check — normalize ollama/qwen aliases
    prov = str(parsed.get("provider") or "").strip().lower()
    expected = expected_provider.lower()
    if expected == "ollama" and prov not in ("ollama", "qwen", "ollama_qwen3_8b", "qwen3"):
        errors.append(f"provider_mismatch:{prov}")
        schema_ok = False
    elif expected == "gemini" and prov not in ("gemini", "google", "gemini_flash"):
        errors.append(f"provider_mismatch:{prov}")
        schema_ok = False

    authority_ok = True
    if str(parsed.get("authority_status") or "") != AUTHORITY_STATUS:
        authority_ok = False
        errors.append("authority_status_mismatch")
    for field in FALSE_AUTHORITY_FIELDS:
        if field not in parsed:
            authority_ok = False
            errors.append(f"authority_missing:{field}")
        elif parsed.get(field) is not False:
            authority_ok = False
            errors.append(f"authority_not_false:{field}")

    return schema_ok, authority_ok, errors


# ---------------------------------------------------------------------------
# Provider callers
# ---------------------------------------------------------------------------


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def get_ollama_timeout() -> float:
    # Prefer explicit review timeout, else OLLAMA_TIMEOUT_SECONDS, else 180s for local 8b.
    for key in ("AE19_REVIEW_OLLAMA_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS"):
        raw = os.getenv(key)
        if raw:
            try:
                return max(1.0, float(raw))
            except ValueError:
                pass
    return 180.0


def ollama_chat_endpoint() -> str:
    raw = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    normalized = raw.removesuffix("/v1")
    return f"{normalized}/api/chat"


def call_ollama_review(prompt: str, *, model: str | None = None, timeout_s: float | None = None) -> dict[str, Any]:
    """Native Ollama /api/chat — no OpenAI SDK. Never raises."""
    import urllib.error
    import urllib.request

    model_name = model or get_ollama_model()
    timeout = timeout_s if timeout_s is not None else get_ollama_timeout()
    endpoint = ollama_chat_endpoint()
    body = json.dumps(
        {
            "model": model_name,
            "stream": False,
            "think": False,
            "format": ollama_review_json_schema(provider="ollama", model=model_name),
            "options": {
                "temperature": 0,
                # Output budget only — do not raise context/GPU memory settings.
                "num_predict": 4096,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AE19 audit-only historical missed-winner reviewer. "
                        "Return ONLY valid JSON matching the schema. No trade authority. "
                        "execution_allowed=false; paper_execution_allowed=false; "
                        "live_execution_allowed=false; risk_override_allowed=false; "
                        "execution_attempted=false (audit output is not trade execution); "
                        "profitability_claimed=false."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        if data.get("error"):
            return {
                "ok": False,
                "text": "",
                "raw_json": data,
                "model": model_name,
                "endpoint": endpoint,
                "timeout_s": timeout,
                "error_type": "ollama_error_response",
                "error_message": sanitize_error(data.get("error")),
            }
        # done=false usually means context overflow / aborted generation under default ctx.
        if data.get("done") is False:
            message = data.get("message") or {}
            content = ""
            if isinstance(message, dict):
                content = str(message.get("content") or "")
            return {
                "ok": False,
                "text": content,
                "raw_json": data,
                "model": model_name,
                "endpoint": endpoint,
                "timeout_s": timeout,
                "error_type": "ollama_incomplete_done_false",
                "error_message": "Ollama returned done=false (likely context overflow; compact prompt required)",
            }
        message = data.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            thinking = message.get("thinking") or ""
            if isinstance(thinking, str) and thinking.strip():
                return {
                    "ok": False,
                    "text": "",
                    "raw_json": data,
                    "model": model_name,
                    "endpoint": endpoint,
                    "timeout_s": timeout,
                    "error_type": "ollama_empty_content_thinking_only",
                    "error_message": "thinking-only response with empty content",
                }
            return {
                "ok": False,
                "text": "",
                "raw_json": data,
                "model": model_name,
                "endpoint": endpoint,
                "timeout_s": timeout,
                "error_type": "ollama_empty_content",
                "error_message": "empty content",
            }
        return {
            "ok": True,
            "text": content,
            "raw_json": data,
            "model": model_name,
            "endpoint": endpoint,
            "timeout_s": timeout,
            "error_type": None,
            "error_message": None,
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model_name,
            "endpoint": endpoint,
            "timeout_s": timeout,
            "error_type": "ollama_unreachable",
            "error_message": sanitize_error(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model_name,
            "endpoint": endpoint,
            "timeout_s": timeout,
            "error_type": type(exc).__name__,
            "error_message": sanitize_error(exc),
        }


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def call_gemini_review(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    """Gemini audit call via google.generativeai. Never raises."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = model or get_gemini_model()
    if not api_key:
        return {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model_name,
            "error_type": "gemini_api_key_missing",
            "error_message": "GEMINI_API_KEY/GOOGLE_API_KEY missing",
        }
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        return {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model_name,
            "error_type": "google_generativeai_missing",
            "error_message": sanitize_error(exc),
        }
    try:
        genai.configure(api_key=api_key)
        system = (
            "You are an AE19 audit-only reviewer. Return JSON only. "
            "No trade authority. execution_attempted must be false "
            "(audit text is not trade execution). "
            "execution_allowed, paper_execution_allowed, live_execution_allowed, "
            "risk_override_allowed, profitability_claimed must all be false. "
            "authority_status must be AUDIT_ONLY_NO_TRADE_AUTHORITY."
        )
        gm = genai.GenerativeModel(
            model_name,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        response = gm.generate_content(prompt)
        text = ""
        if hasattr(response, "text") and response.text:
            text = str(response.text).strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", None) or []
            text = " ".join(str(getattr(p, "text", "") or "") for p in parts).strip()
        raw_json: dict[str, Any] = {"model": model_name, "text_len": len(text)}
        try:
            # Best-effort serializable snapshot without secrets
            if hasattr(response, "to_dict"):
                raw_json["response"] = response.to_dict()
        except Exception:  # noqa: BLE001
            raw_json["response_repr"] = repr(response)[:2000]
        if not text:
            return {
                "ok": False,
                "text": "",
                "raw_json": raw_json,
                "model": model_name,
                "error_type": "empty_gemini_response",
                "error_message": "empty content",
            }
        return {
            "ok": True,
            "text": text,
            "raw_json": raw_json,
            "model": model_name,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model_name,
            "error_type": type(exc).__name__,
            "error_message": sanitize_error(exc),
        }


def run_provider_review(
    *,
    provider: str,
    package: dict[str, Any],
    provider_dir: Path,
    mock_response_text: str | None = None,
    mock_used: bool = False,
    call_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    provider_dir.mkdir(parents=True, exist_ok=True)
    model = get_ollama_model() if provider == "ollama" else get_gemini_model()
    evidence_cap = OLLAMA_MAX_EVIDENCE_CHARS if provider == "ollama" else None
    prompt, prompt_meta = render_llm_prompt(
        package,
        provider=provider,
        model=model,
        max_evidence_chars=evidence_cap,
    )
    prompt_path = provider_dir / "llm_review_prompt.txt"
    raw_txt_path = provider_dir / "llm_review_raw_response.txt"
    raw_json_path = provider_dir / "llm_review_raw_response.json"
    parsed_path = provider_dir / "llm_review_parsed.json"
    gate_path = provider_dir / "provider_gate.json"

    write_text(prompt_path, prompt)
    write_json(provider_dir / "llm_review_prompt_meta.json", prompt_meta)
    started = utc_now()
    started_iso = utc_iso(started)
    t0 = time.perf_counter()

    call_result: dict[str, Any]
    if mock_response_text is not None:
        call_result = {
            "ok": True,
            "text": mock_response_text,
            "raw_json": {"mock": True, "text": mock_response_text},
            "model": model,
            "error_type": None,
            "error_message": None,
        }
        mock_used = True
    elif call_fn is not None:
        call_result = call_fn(prompt, model=model)
    elif provider == "ollama":
        call_result = call_ollama_review(prompt, model=model)
    elif provider == "gemini":
        call_result = call_gemini_review(prompt, model=model)
    else:
        call_result = {
            "ok": False,
            "text": "",
            "raw_json": None,
            "model": model,
            "error_type": "unknown_provider",
            "error_message": provider,
        }

    finished = utc_now()
    duration = time.perf_counter() - t0
    model_used = str(call_result.get("model") or model)
    raw_text = str(call_result.get("text") or "")
    write_text(raw_txt_path, raw_text)
    write_json(
        raw_json_path,
        {
            "provider": provider,
            "model": model_used,
            "ok": bool(call_result.get("ok")),
            "error_type": call_result.get("error_type"),
            "error_message_sanitized": call_result.get("error_message"),
            "raw_response_text": raw_text,
            "provider_raw": call_result.get("raw_json"),
            "prompt_meta": prompt_meta,
        },
    )

    parse_strategy = "failed"
    parse_success = False
    schema_success = False
    authority_success = False
    parsed: dict[str, Any] | None = None
    schema_errors: list[str] = []
    error_type = call_result.get("error_type")
    error_message = call_result.get("error_message")
    called = mock_response_text is not None or call_fn is not None or provider in ("ollama", "gemini")
    success = False

    if call_result.get("ok") and raw_text:
        parsed, parse_strategy, parse_err = parse_provider_json(raw_text)
        parse_success = parsed is not None
        if not parse_success:
            error_type = error_type or "parse_failed"
            error_message = error_message or parse_err
        else:
            schema_success, authority_success, schema_errors = validate_review_schema(
                parsed, expected_provider=provider
            )
            write_json(parsed_path, parsed)
            if not schema_success or not authority_success:
                error_type = error_type or "schema_or_authority_failed"
                error_message = error_message or ";".join(schema_errors)
            else:
                # Unit-test mocks still count as provider_success for folder/artifact checks;
                # final PASS for --provider both still requires mock_used=false (enforced in gate).
                success = True
    else:
        # Preserve any partial text for debugging even on call failure
        if raw_text and not (parsed_path.is_file()):
            partial, strat, _ = parse_provider_json(raw_text)
            write_json(
                parsed_path,
                {
                    "parse_success": False,
                    "error": error_message,
                    "partial_parse_strategy": strat,
                    "partial_parsed": partial,
                },
            )
        else:
            write_json(parsed_path, {"parse_success": False, "error": error_message})

    gate = {
        "provider": provider,
        "model": model_used,
        "called": bool(called),
        "success": bool(success),
        "mock_used": bool(mock_used),
        "started_at_utc": started_iso,
        "finished_at_utc": utc_iso(finished),
        "duration_seconds": round(duration, 3),
        "prompt_chars": len(prompt),
        "raw_response_chars": len(raw_text),
        "parse_strategy": parse_strategy,
        "parse_success": bool(parse_success),
        "schema_success": bool(schema_success),
        "authority_success": bool(authority_success),
        "schema_errors": schema_errors,
        "error_type": error_type,
        "error_message_sanitized": error_message,
        "prompt_meta": prompt_meta,
        "output_paths": {
            "prompt": str(prompt_path),
            "raw_response_txt": str(raw_txt_path),
            "raw_response_json": str(raw_json_path),
            "parsed": str(parsed_path),
            "gate": str(gate_path),
        },
        "execution_attempted": False,
        "paper_execution_attempted": False,
        "live_trading_attempted": False,
        "wallet_access_attempted": False,
        "risk_override_attempted": False,
        "timeout_s": call_result.get("timeout_s"),
        "endpoint": call_result.get("endpoint"),
    }
    write_json(gate_path, gate)
    return {
        "provider": provider,
        "model": model_used,
        "success": success,
        "mock_used": mock_used,
        "gate": gate,
        "parsed": parsed,
        "prompt": prompt,
        "raw_text": raw_text,
        "parse_strategy": parse_strategy,
        "prompt_meta": prompt_meta,
    }


# ---------------------------------------------------------------------------
# Reports / gates
# ---------------------------------------------------------------------------


def _dist_to_map(dist: list[dict[str, Any]], key_field: str = "key") -> dict[str, int]:
    return {str(d.get(key_field)): int(d.get("count") or 0) for d in dist}


def build_comparison(
    provider_results: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    ollama = provider_results.get("ollama") or {}
    gemini = provider_results.get("gemini") or {}
    o_parsed = ollama.get("parsed") or {}
    g_parsed = gemini.get("parsed") or {}

    o_causes = [str(x) for x in (o_parsed.get("dominant_missed_winner_causes") or [])]
    g_causes = [str(x) for x in (g_parsed.get("dominant_missed_winner_causes") or [])]
    common_causes = sorted(set(c.lower() for c in o_causes) & set(c.lower() for c in g_causes))

    def _mentions_dup(parsed: dict[str, Any]) -> bool:
        blob = " ".join(
            [
                str(parsed.get("deduplication_warning") or ""),
                str(parsed.get("summary") or ""),
                " ".join(str(x) for x in (parsed.get("top_findings") or [])),
                " ".join(str(x) for x in (parsed.get("model_specific_observations") or [])),
            ]
        ).lower()
        return any(tok in blob for tok in ("duplicat", "overcount", "dedup", "0xe7a381"))

    def _mentions_stale_or_lineage(parsed: dict[str, Any]) -> bool:
        blob = " ".join(
            [
                str(parsed.get("summary") or ""),
                " ".join(str(x) for x in (parsed.get("dominant_missed_winner_causes") or [])),
                " ".join(str(x) for x in (parsed.get("top_findings") or [])),
                json.dumps(parsed.get("blocker_interpretation") or {}, default=str),
            ]
        ).lower()
        stale = any(tok in blob for tok in ("stale", "price_fresh", "exploration"))
        lineage = any(tok in blob for tok in ("lineage", "context", "missing_context", "weak_lineage"))
        return stale and lineage

    o_profit = bool(o_parsed.get("profitability_claimed")) if o_parsed else False
    g_profit = bool(g_parsed.get("profitability_claimed")) if g_parsed else False
    o_auth_ok = bool((ollama.get("gate") or {}).get("authority_success"))
    g_auth_ok = bool((gemini.get("gate") or {}).get("authority_success"))

    agreements: list[str] = []
    differences: list[str] = []
    if ollama.get("success") and gemini.get("success"):
        if common_causes:
            agreements.append(f"Shared dominant causes (case-insensitive overlap): {common_causes}")
        if _mentions_dup(o_parsed) and _mentions_dup(g_parsed):
            agreements.append("Both providers acknowledge duplicate-heavy top rows")
        elif _mentions_dup(o_parsed) or _mentions_dup(g_parsed):
            differences.append("Only one provider clearly acknowledged duplicate-heavy tops")
        if _mentions_stale_or_lineage(o_parsed) and _mentions_stale_or_lineage(g_parsed):
            agreements.append("Both identify stale price/exploration and weak lineage/context themes")
        elif _mentions_stale_or_lineage(o_parsed) or _mentions_stale_or_lineage(g_parsed):
            differences.append("Stale-price / weak-lineage theme emphasis differs between providers")
        if (o_parsed.get("summary") or "") != (g_parsed.get("summary") or ""):
            differences.append("Summary wording differs (expected across models)")
    else:
        differences.append("One or both providers did not succeed; comparison is partial")

    det_causes = [d["key"] for d in summary.get("reason_not_traded_distribution", [])[:5]]
    ae19_04_strengthened = bool(ollama.get("success") and gemini.get("success"))

    return {
        "providers_compared": [p for p in ("ollama", "gemini") if p in provider_results],
        "agreements": agreements,
        "differences": differences,
        "common_dominant_missed_winner_causes": common_causes,
        "deterministic_top_reason_not_traded": det_causes,
        "profitability_overclaim": {
            "ollama": o_profit,
            "gemini": g_profit,
            "either_overclaims": bool(o_profit or g_profit),
        },
        "authority_boundary": {
            "ollama_authority_success": o_auth_ok,
            "gemini_authority_success": g_auth_ok,
            "either_violates": bool(
                (ollama.get("parsed") and not o_auth_ok) or (gemini.get("parsed") and not g_auth_ok)
            ),
        },
        "duplicate_heavy_identification": {
            "ollama": _mentions_dup(o_parsed) if o_parsed else False,
            "gemini": _mentions_dup(g_parsed) if g_parsed else False,
        },
        "stale_and_lineage_identification": {
            "ollama": _mentions_stale_or_lineage(o_parsed) if o_parsed else False,
            "gemini": _mentions_stale_or_lineage(g_parsed) if g_parsed else False,
        },
        "ae19_04_strengthened": ae19_04_strengthened,
        "conclusion": (
            "AE19-04 is strengthened by persisted dual-provider historical missed-winner audit artifacts. "
            "This does not prove profitability and grants no execution authority."
            if ae19_04_strengthened
            else "AE19-04 historical dual-provider evidence is incomplete; final PASS not earned."
        ),
        "ollama_summary": (o_parsed.get("summary") if o_parsed else None),
        "gemini_summary": (g_parsed.get("summary") if g_parsed else None),
        "no_model_declared_better": True,
    }


def render_comparison_md(comparison: dict[str, Any]) -> str:
    lines = [
        "# AE19 Historical Missed-Winner Provider Comparison",
        "",
        "## Agreements",
    ]
    for a in comparison.get("agreements") or ["(none)"]:
        lines.append(f"- {a}")
    lines += ["", "## Differences"]
    for d in comparison.get("differences") or ["(none)"]:
        lines.append(f"- {d}")
    lines += [
        "",
        "## Common dominant missed-winner causes",
        f"- {comparison.get('common_dominant_missed_winner_causes')}",
        "",
        "## Deterministic top reason_not_traded",
        f"- {comparison.get('deterministic_top_reason_not_traded')}",
        "",
        "## Profitability overclaim check",
        f"- {json.dumps(comparison.get('profitability_overclaim'), indent=2)}",
        "",
        "## Authority boundary",
        f"- {json.dumps(comparison.get('authority_boundary'), indent=2)}",
        "",
        "## Duplicate-heavy top rows identified",
        f"- {json.dumps(comparison.get('duplicate_heavy_identification'), indent=2)}",
        "",
        "## Stale price / weak lineage themes",
        f"- {json.dumps(comparison.get('stale_and_lineage_identification'), indent=2)}",
        "",
        "## Qwen3 / Ollama summary",
        f"- {comparison.get('ollama_summary')}",
        "",
        "## Gemini summary",
        f"- {comparison.get('gemini_summary')}",
        "",
        "## Conclusion",
        f"- AE19-04 strengthened: {comparison.get('ae19_04_strengthened')}",
        f"- {comparison.get('conclusion')}",
        "",
        "No model is declared better unless supported by outputs. "
        "No profitability claim. No trade execution recommendation.",
        "",
    ]
    return "\n".join(lines)


def resolve_providers(provider_arg: str) -> list[str]:
    if provider_arg == "none":
        return []
    if provider_arg == "both":
        return ["ollama", "gemini"]
    if provider_arg in ("ollama", "gemini"):
        return [provider_arg]
    raise ValueError(f"unsupported provider mode: {provider_arg}")


def compute_final_status(
    *,
    provider_arg: str,
    providers_requested: list[str],
    provider_success: dict[str, bool],
    deterministic_ok: bool,
) -> str:
    if not deterministic_ok:
        return STATUS_FAIL
    if provider_arg == "none" or not providers_requested:
        return STATUS_DETERMINISTIC_ONLY
    if provider_arg in ("ollama", "gemini"):
        # Single-provider debug only — never PASS
        return STATUS_PARTIAL_DEBUG
    # both
    o_ok = bool(provider_success.get("ollama"))
    g_ok = bool(provider_success.get("gemini"))
    if o_ok and g_ok:
        return STATUS_PASS
    if o_ok or g_ok:
        return STATUS_PARTIAL_FAILURE
    return STATUS_FAIL


def write_deterministic_artifacts(data_dir: Path, summary: dict[str, Any], package: dict[str, Any]) -> None:
    write_csv(
        data_dir / "missed_winner_rows_summary.csv",
        [
            {
                "total_rows": summary["total_rows"],
                "unique_pair_count": summary["unique_pair_count"],
                "unique_candidate_count": summary["unique_candidate_count"],
                "unique_decision_count": summary["unique_decision_count"],
                "unique_evidence_row_count": summary["unique_evidence_row_count"],
                "source_path": summary["source_path"],
            }
        ],
    )
    write_csv(
        data_dir / "missed_winner_reason_distribution.csv",
        [{"reason_not_traded": d["key"], "count": d["count"]} for d in summary["reason_not_traded_distribution"]],
        ["reason_not_traded", "count"],
    )
    write_csv(
        data_dir / "missed_winner_horizon_distribution.csv",
        [{"horizon": k, "count": v} for k, v in summary["horizon_counts"].items()],
        ["horizon", "count"],
    )
    write_csv(data_dir / "missed_winner_top_raw_rows.csv", summary["top_raw_rows"])
    write_csv(data_dir / "missed_winner_top_pairs_dedup.csv", summary["top_pairs_dedup"])
    write_csv(data_dir / "missed_winner_top_candidates_dedup.csv", summary["top_candidates_dedup"])
    write_csv(
        data_dir / "missed_winner_blocker_summary.csv",
        summary["audit_blockers_distribution"],
        ["blocker", "count"],
    )
    write_json(data_dir / "llm_review_input_package.json", package)


def build_main_report(
    *,
    summary: dict[str, Any],
    package: dict[str, Any],
    provider_results: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "authority_status": AUTHORITY_STATUS,
        "final_status": gate.get("final_status"),
        "source_path": summary["source_path"],
        "total_rows": summary["total_rows"],
        "horizon_counts": summary["horizon_counts"],
        "unique_pair_count": summary["unique_pair_count"],
        "unique_candidate_count": summary["unique_candidate_count"],
        "unique_decision_count": summary["unique_decision_count"],
        "unique_evidence_row_count": summary["unique_evidence_row_count"],
        "reason_not_traded_top": summary["reason_not_traded_distribution"][:10],
        "audit_blockers_top": summary["audit_blockers_distribution"][:10],
        "top_dedup_pair": (summary["top_pairs_dedup"][0] if summary["top_pairs_dedup"] else None),
        "input_package_meta": package.get("_meta"),
        "providers": {
            name: {
                "model": res.get("model"),
                "success": res.get("success"),
                "parse_strategy": res.get("parse_strategy"),
                "summary": (res.get("parsed") or {}).get("summary"),
                "dominant_missed_winner_causes": (res.get("parsed") or {}).get("dominant_missed_winner_causes"),
            }
            for name, res in provider_results.items()
        },
        "comparison_conclusion": comparison.get("conclusion"),
        "gate_flags": {
            "llm_called": gate.get("llm_called"),
            "mock_used": gate.get("mock_used"),
            "db_mutation_attempted": gate.get("db_mutation_attempted"),
            "execution_attempted": gate.get("execution_attempted"),
            "paper_execution_attempted": gate.get("paper_execution_attempted"),
            "live_trading_attempted": gate.get("live_trading_attempted"),
            "wallet_access_attempted": gate.get("wallet_access_attempted"),
            "risk_override_attempted": gate.get("risk_override_attempted"),
        },
        "ae20_started": False,
        "profitability_proven": False,
        "note": "Strengthens AE19-04 only; does not prove profitability.",
    }


def render_main_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# AE19 Historical Missed-Winner Review",
        "",
        f"- final_status: `{report.get('final_status')}`",
        f"- authority_status: `{report.get('authority_status')}`",
        f"- total_rows: {report.get('total_rows')}",
        f"- horizon_counts: {report.get('horizon_counts')}",
        f"- unique_pair_count: {report.get('unique_pair_count')}",
        f"- unique_candidate_count: {report.get('unique_candidate_count')}",
        "",
        "## Top dedup pair",
        f"- {json.dumps(report.get('top_dedup_pair'), default=str)}",
        "",
        "## Deterministic top reasons",
        f"- {report.get('reason_not_traded_top')}",
        "",
        "## Providers",
        f"- {json.dumps(report.get('providers'), indent=2, default=str)}",
        "",
        "## Gate flags",
        f"- {json.dumps(report.get('gate_flags'), indent=2)}",
        "",
        f"## Note\n{report.get('note')}",
        "",
        f"AE20 started: {report.get('ae20_started')}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_review(
    *,
    input_path: Path,
    output_root: Path | None = None,
    provider: str = "both",
    top_n: int = 20,
    max_llm_rows: int = 60,
    max_input_chars: int = 40000,
    mock_responses: dict[str, str] | None = None,
    call_overrides: dict[str, Callable[..., dict[str, Any]]] | None = None,
    sequential_call_log: list[str] | None = None,
) -> dict[str, Any]:
    mock_responses = mock_responses or {}
    call_overrides = call_overrides or {}

    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    input_size = input_path.stat().st_size
    rows, fieldnames = load_missed_winner_rows(input_path)
    missing = validate_required_columns(fieldnames)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    providers_requested = resolve_providers(provider)
    out_root = output_root or (ROOT / "data" / "audits" / f"ae19_historical_missed_winner_review_{utc_stamp()}")
    data_dir = out_root / "data"
    reports_dir = out_root / "reports"
    audits_dir = out_root / "audits"
    manifests_dir = out_root / "manifests"
    providers_dir = data_dir / "providers"
    for d in (data_dir, reports_dir, audits_dir, manifests_dir, providers_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary = build_deterministic_summary(rows, source_path=input_path, top_n=top_n)
    package = build_llm_input_package(
        summary,
        top_n=top_n,
        max_llm_rows=max_llm_rows,
        max_input_chars=max_input_chars,
    )
    write_deterministic_artifacts(data_dir, summary, package)

    provider_results: dict[str, dict[str, Any]] = {}
    # Sequential only — never concurrent
    for prov in providers_requested:
        if sequential_call_log is not None:
            sequential_call_log.append(prov)
        provider_dir = providers_dir / prov
        result = run_provider_review(
            provider=prov,
            package=package,
            provider_dir=provider_dir,
            mock_response_text=mock_responses.get(prov),
            mock_used=prov in mock_responses,
            call_fn=call_overrides.get(prov),
        )
        provider_results[prov] = result

    provider_success = {p: bool((provider_results.get(p) or {}).get("success")) for p in REQUIRED_PROVIDERS}
    for p in providers_requested:
        provider_success[p] = bool((provider_results.get(p) or {}).get("success"))

    provider_parse = {
        p: bool(((provider_results.get(p) or {}).get("gate") or {}).get("parse_success"))
        for p in providers_requested
    }
    provider_schema = {
        p: bool(((provider_results.get(p) or {}).get("gate") or {}).get("schema_success"))
        for p in providers_requested
    }
    provider_authority = {
        p: bool(((provider_results.get(p) or {}).get("gate") or {}).get("authority_success"))
        for p in providers_requested
    }

    all_required_success = all(provider_success.get(p) for p in REQUIRED_PROVIDERS)
    mock_used_any = any(bool((provider_results.get(p) or {}).get("mock_used")) for p in providers_requested)
    # Real final PASS cannot use mocks
    if mock_used_any:
        all_required_success = False

    llm_called = len(providers_requested) > 0
    final_status = compute_final_status(
        provider_arg=provider,
        providers_requested=providers_requested,
        provider_success={p: provider_success.get(p, False) for p in providers_requested},
        deterministic_ok=True,
    )
    if provider == "both" and mock_used_any and final_status == STATUS_PASS:
        final_status = STATUS_PARTIAL_FAILURE

    meta = package.get("_meta") or {}
    gate = {
        "final_status": final_status,
        "input_file_exists": True,
        "input_file_size_bytes": input_size,
        "required_columns_present": True,
        "total_rows": summary["total_rows"],
        "horizon_counts": summary["horizon_counts"],
        "unique_pair_count": summary["unique_pair_count"],
        "unique_candidate_count": summary["unique_candidate_count"],
        "unique_decision_count": summary["unique_decision_count"],
        "unique_evidence_row_count": summary["unique_evidence_row_count"],
        "no_lookahead_status_distribution": _dist_to_map(summary["no_lookahead_status_distribution"]),
        "was_traded_distribution": _dist_to_map(summary["was_traded_distribution"]),
        "deterministic_review_generated": True,
        "llm_called": llm_called,
        "required_providers": list(REQUIRED_PROVIDERS),
        "providers_requested": list(providers_requested),
        "providers_evaluated": list(providers_requested),
        "provider_success_by_name": {
            "ollama": bool(provider_success.get("ollama")),
            "gemini": bool(provider_success.get("gemini")),
        },
        "provider_parse_success_by_name": {
            "ollama": bool(provider_parse.get("ollama")),
            "gemini": bool(provider_parse.get("gemini")),
        },
        "provider_schema_success_by_name": {
            "ollama": bool(provider_schema.get("ollama")),
            "gemini": bool(provider_schema.get("gemini")),
        },
        "provider_authority_success_by_name": {
            "ollama": bool(provider_authority.get("ollama")),
            "gemini": bool(provider_authority.get("gemini")),
        },
        "all_required_providers_success": bool(all_required_success) and provider == "both" and not mock_used_any,
        "mock_used": bool(mock_used_any),
        "db_mutation_attempted": False,
        "execution_attempted": False,
        "paper_execution_attempted": False,
        "live_trading_attempted": False,
        "wallet_access_attempted": False,
        "risk_override_attempted": False,
        "authority_status": AUTHORITY_STATUS,
        "input_truncated": bool(meta.get("input_truncated")),
        "input_package_chars": int(meta.get("input_package_chars") or 0),
        "gpu_constraint_note": GPU_CONSTRAINT_NOTE,
        "ae20_started": False,
    }
    # Recompute PASS strictly
    if (
        provider == "both"
        and gate["provider_success_by_name"]["ollama"]
        and gate["provider_success_by_name"]["gemini"]
        and not gate["mock_used"]
        and gate["all_required_providers_success"]
    ):
        gate["final_status"] = STATUS_PASS
    elif provider == "both":
        if gate["provider_success_by_name"]["ollama"] or gate["provider_success_by_name"]["gemini"]:
            gate["final_status"] = STATUS_PARTIAL_FAILURE
        else:
            gate["final_status"] = STATUS_FAIL
        gate["all_required_providers_success"] = False
    else:
        gate["final_status"] = final_status
        gate["all_required_providers_success"] = False

    comparison = build_comparison(provider_results, summary)
    report = build_main_report(
        summary=summary,
        package=package,
        provider_results=provider_results,
        gate=gate,
        comparison=comparison,
    )

    write_json(audits_dir / "ae19_historical_missed_winner_review_gate.json", gate)
    write_json(reports_dir / "ae19_historical_missed_winner_review.json", report)
    write_text(reports_dir / "ae19_historical_missed_winner_review.md", render_main_report_md(report))
    write_json(reports_dir / "ae19_historical_missed_winner_provider_comparison.json", comparison)
    write_text(
        reports_dir / "ae19_historical_missed_winner_provider_comparison.md",
        render_comparison_md(comparison),
    )

    manifest = {
        "phase": PHASE,
        "created_at_utc": utc_iso(),
        "output_root": str(out_root.resolve()),
        "input_path": str(input_path.resolve()),
        "input_file_size_bytes": input_size,
        "provider_mode": provider,
        "providers_requested": providers_requested,
        "top_n": top_n,
        "max_llm_rows": max_llm_rows,
        "max_input_chars": max_input_chars,
        "input_package_meta": meta,
        "final_status": gate["final_status"],
        "artifacts": {
            "gate": str((audits_dir / "ae19_historical_missed_winner_review_gate.json").resolve()),
            "report_json": str((reports_dir / "ae19_historical_missed_winner_review.json").resolve()),
            "report_md": str((reports_dir / "ae19_historical_missed_winner_review.md").resolve()),
            "comparison_json": str(
                (reports_dir / "ae19_historical_missed_winner_provider_comparison.json").resolve()
            ),
            "comparison_md": str(
                (reports_dir / "ae19_historical_missed_winner_provider_comparison.md").resolve()
            ),
            "llm_input_package": str((data_dir / "llm_review_input_package.json").resolve()),
            "providers": {
                p: (provider_results[p].get("gate") or {}).get("output_paths")
                for p in provider_results
            },
        },
        "gpu_constraint_note": GPU_CONSTRAINT_NOTE,
        "db_mutation_attempted": False,
        "execution_attempted": False,
        "ae20_started": False,
    }
    write_json(manifests_dir / "ae19_historical_missed_winner_review_manifest.json", manifest)

    return {
        "output_root": str(out_root.resolve()),
        "gate": gate,
        "summary": summary,
        "package": package,
        "provider_results": provider_results,
        "comparison": comparison,
        "report": report,
        "manifest": manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE19 Historical Missed-Winner dual-provider review")
    p.add_argument("--provider", choices=["none", "ollama", "gemini", "both"], default="both")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--max-llm-rows", type=int, default=60)
    p.add_argument("--max-input-chars", type=int, default=40000)
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output-root", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_review(
            input_path=Path(args.input),
            output_root=Path(args.output_root) if args.output_root else None,
            provider=args.provider,
            top_n=args.top_n,
            max_llm_rows=args.max_llm_rows,
            max_input_chars=args.max_input_chars,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{PHASE}] unexpected error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    gate = result["gate"]
    print(f"output_root={result['output_root']}")
    print(f"total_rows={gate['total_rows']}")
    print(f"horizon_counts={gate['horizon_counts']}")
    print(f"required_providers={gate['required_providers']}")
    print(f"providers_evaluated={gate['providers_evaluated']}")
    print(f"provider_success_by_name.ollama={gate['provider_success_by_name'].get('ollama')}")
    print(f"provider_success_by_name.gemini={gate['provider_success_by_name'].get('gemini')}")
    print(f"all_required_providers_success={gate['all_required_providers_success']}")
    print(f"llm_called={gate['llm_called']}")
    print(f"mock_used={gate['mock_used']}")
    print(f"input_package_chars={gate['input_package_chars']}")
    print(f"input_truncated={gate['input_truncated']}")
    print(f"execution_attempted={gate['execution_attempted']}")
    print(f"paper_execution_attempted={gate['paper_execution_attempted']}")
    print(f"live_trading_attempted={gate['live_trading_attempted']}")
    print(f"wallet_access_attempted={gate['wallet_access_attempted']}")
    print(f"risk_override_attempted={gate['risk_override_attempted']}")
    print(f"authority_status={gate['authority_status']}")
    print(f"final_status={gate['final_status']}")
    return 0 if gate["final_status"] == STATUS_PASS or args.provider != "both" else 2


if __name__ == "__main__":
    raise SystemExit(main())
