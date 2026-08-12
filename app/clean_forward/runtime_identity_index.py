"""AE18 hot-path runtime canonical identity index loader.

Reads precomputed index only — no network, no audit scans, no identity repair.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "data" / "runtime"
INDEX_JSONL_PATH = RUNTIME_DIR / "canonical_market_identity_index.jsonl"
INDEX_CSV_PATH = RUNTIME_DIR / "canonical_market_identity_index.csv"

STALE_THRESHOLD_SECONDS = 86400.0  # 24h warning threshold
INDEX_MISSING_CODE = "CANONICAL_IDENTITY_INDEX_MISSING"

_INDEX_FIELDS = [
    "canonical_market_identity",
    "canonical_market_identity_type",
    "provider_pair_url_exact",
    "provider_pair_url_final_segment_exact",
    "normalized_provider_pair_url_key",
    "open_chart_url",
    "canonical_identity_status",
    "canonical_identity_source",
    "canonical_identity_case_preserved",
    "canonical_identity_missingness_reason",
    "mark_price_lookup_key",
    "mark_price_lookup_status",
    "chain",
    "dex_id",
    "provider_dex_id",
    "provider",
    "target_source",
    "symbol_pair_display",
    "symbol_pair_display_status",
    "symbol_pair_display_reason",
    "symbol_pair_address_fallback",
    "provider_resolution_status",
    "provider_probe_attempted",
    "symbol_resolution_status",
    "display_metadata_status",
    "market_data_status",
    "identity_readiness_status",
    "trade_readiness_status",
    "trade_block_reason",
    "provider_fetch_at",
    "market_data_refreshed_at",
    "last_market_update_at",
    "price_updated_at",
    "display_status",
    "market_activity_status",
    "activity_trade_readiness_status",
    "activity_trade_block_reason",
    "market_activity_blocks_demo_entry",
    "provider_txns_observed_field_count",
    "provider_txns_recent_total",
    "provider_volume_observed_field_count",
    "provider_volume_recent_total",
    "provider_price_delta_observed_field_count",
    "provider_price_delta_any_nonzero",
    "market_activity_provenance",
    "activity_uses_symbol_display",
    "activity_uses_liquidity_or_market_cap_as_activity_proxy",
    "unresolved_reason",
    "display_provenance",
    "base_token_symbol",
    "quote_token_symbol",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "provider_base_token_name",
    "provider_quote_token_name",
    "base_token_address_derived",
    "quote_token_address_derived",
    "provider_base_token_address",
    "provider_quote_token_address",
    "pair_address_derived",
    "pair_address_derivation_source",
    "pair_address_derivation_status",
    "pair_address_for_rpc",
    "rpc_address_source",
    "price_usd",
    "liquidity_usd",
    "fdv",
    "market_cap",
    "volume_m5",
    "volume_h1",
    "volume_h6",
    "volume_h24",
    "txns_m5_buys",
    "txns_m5_sells",
    "txns_h1_buys",
    "txns_h1_sells",
    "txns_h6_buys",
    "txns_h6_sells",
    "txns_h24_buys",
    "txns_h24_sells",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "pair_created_at",
    "whale_score",
    "semantic_status",
    "social_classification",
    "is_social_candidate",
    "is_social_confirmed",
    "social_source",
    "social_reason",
    "linked_sources",
    "seed_collection",
    "manual_curation_status",
    "feed_status",
    "freshness_status",
    "tradability_status",
    "verification_status",
    "identity_status",
    "last_market_update_at",
    "last_identity_rebuild_at",
    "safe_for_price_lookup",
    "safe_for_rpc_derivation",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def index_exists() -> bool:
    return INDEX_JSONL_PATH.exists() or INDEX_CSV_PATH.exists()


def _parse_iso_age_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds())
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


_BOOL_FIELDS = (
    "is_social_candidate",
    "is_social_confirmed",
    "safe_for_price_lookup",
    "safe_for_rpc_derivation",
    "canonical_identity_case_preserved",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in _BOOL_FIELDS:
            if field in row and isinstance(row[field], str):
                row[field] = row[field].strip().lower() in {"true", "1", "yes"}
    return rows


def load_runtime_identity_index() -> dict[str, Any]:
    """Load cached index. Never performs network calls or audit scans."""
    try:
        from app.runtime.ui_get_network_guard import record_runtime_index_read

        record_runtime_index_read()
    except Exception:
        pass

    start = time.perf_counter()
    source_file = ""
    rows: list[dict[str, Any]] = []

    if INDEX_JSONL_PATH.exists():
        source_file = str(INDEX_JSONL_PATH)
        rows = _read_jsonl(INDEX_JSONL_PATH)
    elif INDEX_CSV_PATH.exists():
        source_file = str(INDEX_CSV_PATH)
        rows = _read_csv(INDEX_CSV_PATH)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    if not rows and not source_file:
        return {
            "ok": False,
            "error_code": INDEX_MISSING_CODE,
            "user_message": INDEX_MISSING_CODE,
            "rebuild_instruction": (
                "Run: python scripts/rebuild_canonical_market_identity_index.py"
            ),
            "rows": [],
            "source_file": "",
            "source_file_exists": False,
            "runtime_index_rows": 0,
            "measured_load_time_ms": elapsed_ms,
            "recursive_audit_scan_used": False,
            "external_network_calls_on_load": False,
            "helius_calls_on_load": False,
            "dexscreener_calls_on_load": False,
            "pair_address_required_for_load": False,
            "stale_warning": False,
        }

    rebuild_ages = [
        _parse_iso_age_seconds(r.get("last_identity_rebuild_at")) for r in rows
    ]
    rebuild_ages_valid = [a for a in rebuild_ages if a is not None]
    stale = bool(rebuild_ages_valid) and max(rebuild_ages_valid) > STALE_THRESHOLD_SECONDS

    canonical_ids = [str(r.get("canonical_market_identity") or "") for r in rows]
    dup_count = len(canonical_ids) - len(set(x for x in canonical_ids if x))

    return {
        "ok": True,
        "rows": rows,
        "source_file": source_file,
        "source_file_exists": True,
        "runtime_index_rows": len(rows),
        "duplicate_canonical_identity_count": dup_count,
        "measured_load_time_ms": elapsed_ms,
        "recursive_audit_scan_used": False,
        "external_network_calls_on_load": False,
        "helius_calls_on_load": False,
        "dexscreener_calls_on_load": False,
        "pair_address_required_for_load": False,
        "stale_warning": stale,
        "stale_message": (
            "Canonical identity index is stale. Cached rows shown; run rebuild to refresh."
            if stale
            else ""
        ),
        "loaded_at": _utc_now(),
    }


def build_price_lookup_from_index(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Build mark-price lookup keyed by canonical_market_identity / provider_pair_url_exact."""
    out: dict[str, float] = {}
    for row in rows:
        key = str(row.get("mark_price_lookup_key") or row.get("canonical_market_identity") or "")
        if not key:
            continue
        price = row.get("price_usd")
        if price in (None, ""):
            continue
        try:
            val = float(price)
        except (TypeError, ValueError):
            continue
        if val > 0:
            out[key] = val
            url = str(row.get("provider_pair_url_exact") or "")
            if url and url != key:
                out[url] = val
    return out


def index_rows_to_market_price_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert index rows to set_market_prices() entries with canonical URL keys."""
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.append(
            {
                "canonical_market_identity": row.get("canonical_market_identity"),
                "provider_pair_url_exact": row.get("provider_pair_url_exact"),
                "mark_price_lookup_key": row.get("mark_price_lookup_key"),
                "pair_address": row.get("pair_address_derived"),
                "price_usd": row.get("price_usd"),
                "coin_id": row.get("coin_id"),
            }
        )
    return entries


def resolve_position_canonical_key(
    position: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve a position's canonical lookup key without using pair_address as primary key."""
    direct = str(
        position.get("canonical_market_identity")
        or position.get("provider_pair_url_exact")
        or position.get("provider_pair_url")
        or position.get("open_chart_url")
        or ""
    ).strip()
    if direct:
        return direct

    if not rows:
        return ""

    pair = str(position.get("pair_address") or position.get("pair_address_derived") or "").strip()
    chain = str(position.get("chain") or "").strip().lower()
    if not pair:
        return ""

    for row in rows:
        derived = str(row.get("pair_address_derived") or "").strip()
        row_chain = str(row.get("chain") or "").strip().lower()
        if derived and derived == pair and (not chain or not row_chain or row_chain == chain):
            return str(row.get("canonical_market_identity") or row.get("provider_pair_url_exact") or "")

    return ""


def _atomic_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dest))


def _looks_like_raw_address_display(text: Any) -> bool:
    """True when a display string is a raw/short token address or address pair."""
    from app.clean_forward.display_identity import UNAVAILABLE_STATUSES

    value = str(text or "").strip()
    if not value or value in UNAVAILABLE_STATUSES:
        return False
    for part in value.split("/"):
        cleaned = part.strip().replace("\u2026", "").replace("...", "")
        if cleaned.lower().startswith("0x") and len(cleaned) >= 8:
            return True
        if len(cleaned) >= 24 and cleaned.isalnum() and not cleaned.isupper():
            return True
    return False


def validate_index_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate candidate index rows before they may replace the live index."""
    identities = [str(r.get("canonical_market_identity") or "").strip() for r in rows]
    non_empty = [i for i in identities if i]
    empty_identity = len(identities) - len(non_empty)
    duplicates = len(non_empty) - len(set(non_empty))
    empty_url = sum(1 for r in rows if not str(r.get("provider_pair_url_exact") or "").strip())
    invalid_display = 0
    for r in rows:
        display = str(r.get("symbol_pair_display") or "").strip()
        if not display or display == "-" or _looks_like_raw_address_display(display):
            invalid_display += 1

    problems: list[str] = []
    if not rows:
        problems.append("no_rows")
    if empty_identity:
        problems.append(f"empty_canonical_identity:{empty_identity}")
    if duplicates:
        problems.append(f"duplicate_canonical_identity:{duplicates}")
    if empty_url:
        problems.append(f"empty_provider_pair_url:{empty_url}")
    if invalid_display:
        problems.append(f"invalid_symbol_pair_display:{invalid_display}")

    return {
        "row_count": len(rows),
        "duplicate_canonical_identity_count": duplicates,
        "empty_canonical_identity_count": empty_identity,
        "empty_provider_pair_url_count": empty_url,
        "invalid_symbol_pair_display_count": invalid_display,
        "problems": problems,
        "passed": not problems,
    }


class RuntimeIndexValidationError(RuntimeError):
    """Raised when candidate index rows fail validation; live index is preserved."""

    def __init__(self, report: dict[str, Any]):
        super().__init__("runtime_index_atomic_update_validation_failed: " + ", ".join(report["problems"]))
        self.report = report


def write_runtime_index_validated(
    rows: list[dict[str, Any]],
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Atomic, validated index update.

    Builds temp JSONL+CSV, validates them, and only then replaces the live
    files. On failure the existing runtime index is left untouched.
    """
    jsonl_path = jsonl_path or INDEX_JSONL_PATH
    csv_path = csv_path or INDEX_CSV_PATH
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "atomic_update_supported": True,
        "temp_jsonl_written": False,
        "temp_csv_written": False,
        "temp_validation_passed": False,
        "final_jsonl_replaced": False,
        "final_csv_replaced": False,
        "final_index_row_count": 0,
        "duplicate_canonical_identity_count": 0,
        "empty_canonical_identity_count": 0,
        "empty_provider_pair_url_count": 0,
        "invalid_symbol_pair_display_count": 0,
        "rollback_or_preserve_previous_on_failure_supported": True,
        "problems": [],
    }

    fieldnames = list(_INDEX_FIELDS)
    fieldnames.extend(sorted({k for r in rows for k in r} - set(fieldnames)))

    tmp_dir = jsonl_path.parent
    fd_j, tmp_j = tempfile.mkstemp(prefix=".idx_", suffix=".jsonl.tmp", dir=str(tmp_dir))
    fd_c, tmp_c = tempfile.mkstemp(prefix=".idx_", suffix=".csv.tmp", dir=str(tmp_dir))
    os.close(fd_j)
    os.close(fd_c)
    tmp_jsonl = Path(tmp_j)
    tmp_csv = Path(tmp_c)

    try:
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["temp_jsonl_written"] = True

        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        report["temp_csv_written"] = True

        # Validate what was actually written, not just the in-memory rows.
        written = _read_jsonl(tmp_jsonl)
        written_csv = _read_csv(tmp_csv)
        validation = validate_index_rows(written)
        report.update(
            {
                k: validation[k]
                for k in (
                    "duplicate_canonical_identity_count",
                    "empty_canonical_identity_count",
                    "empty_provider_pair_url_count",
                    "invalid_symbol_pair_display_count",
                    "problems",
                )
            }
        )
        if len(written) != len(rows) or len(written_csv) != len(rows):
            report["problems"] = list(report["problems"]) + [
                f"row_count_mismatch:{len(written)}/{len(written_csv)}/{len(rows)}"
            ]
            validation["passed"] = False
        if not validation["passed"]:
            raise RuntimeIndexValidationError(report | {"problems": report["problems"]})

        report["temp_validation_passed"] = True
        _atomic_replace(tmp_jsonl, jsonl_path)
        report["final_jsonl_replaced"] = True
        _atomic_replace(tmp_csv, csv_path)
        report["final_csv_replaced"] = True
        report["final_index_row_count"] = len(written)
        report["jsonl_path"] = str(jsonl_path)
        report["csv_path"] = str(csv_path)
        return report
    finally:
        for p in (tmp_jsonl, tmp_csv):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def write_runtime_index(
    rows: list[dict[str, Any]],
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
    atomic: bool = True,
) -> tuple[Path, Path]:
    """Write jsonl + csv runtime index artifacts.

    When atomic=True, writes temp files then os.replace into final paths.
    """
    jsonl_path = jsonl_path or INDEX_JSONL_PATH
    csv_path = csv_path or INDEX_CSV_PATH
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(_INDEX_FIELDS)
    extra_keys = sorted({k for r in rows for k in r} - set(fieldnames))
    fieldnames.extend(extra_keys)

    if not atomic:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return jsonl_path, csv_path

    # Atomic: temp → validate → replace
    tmp_dir = jsonl_path.parent
    fd_j, tmp_j = tempfile.mkstemp(prefix=".idx_", suffix=".jsonl.tmp", dir=str(tmp_dir))
    fd_c, tmp_c = tempfile.mkstemp(prefix=".idx_", suffix=".csv.tmp", dir=str(tmp_dir))
    os.close(fd_j)
    os.close(fd_c)
    tmp_jsonl = Path(tmp_j)
    tmp_csv = Path(tmp_c)
    try:
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                if not row.get("canonical_market_identity"):
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                if not row.get("canonical_market_identity"):
                    continue
                writer.writerow(row)
        # Validate temp is readable + non-empty when rows expected
        probe = _read_jsonl(tmp_jsonl)
        if rows and not probe:
            raise RuntimeError("runtime_index_atomic_update_validation_failed")
        _atomic_replace(tmp_jsonl, jsonl_path)
        _atomic_replace(tmp_csv, csv_path)
    except Exception:
        for p in (tmp_jsonl, tmp_csv):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise

    return jsonl_path, csv_path
