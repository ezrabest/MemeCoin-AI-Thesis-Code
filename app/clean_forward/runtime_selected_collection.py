"""Runtime Selected/Clean collection engine.

Priority: open-position mark price (0A) → active Selected/Clean (0B) →
recommended (1) → discovery/trending (2).

Exact DexScreener pair fetches only for 0A/0B. No symbol search substitute.
ae16b_* is never a pair address. Rate-limit + cooldown prevent silent drops
and infinite dead-pair loops.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.clean_forward.price_source_identity import (
    build_price_source_key,
    cell,
    is_internal_lineage_id,
    resolve_selected_target_identity,
    synthesize_dexscreener_url,
)
from app.dexscreener import BASE_URL, _HEADERS, _extract_pair

DEFAULT_SELECTED_PATH = Path(
    "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"
)
DEFAULT_PAPER_STATE = Path("data/paper_state.json")
DEFAULT_FETCH_STATE_PATH = Path("data/collection_state/selected_target_fetch_state.json")

SOURCE_QUERY_SELECTED = "selected_clean_exact_pair"
SOURCE_QUERY_OPEN = "open_position_mark_price_exact_pair"
SOURCE_QUERY_BOTH = "selected_clean_and_open_position_exact_pair"
SOURCE_TYPE_SELECTED = "selected_clean_pair_fetch"
SOURCE_TYPE_OPEN = "open_position_mark_price_fetch"

RETRY_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

DEFAULT_POLICY: dict[str, Any] = {
    "sleep_seconds_between_requests": 0.35,
    "max_concurrency": 1,
    "request_timeout_seconds": 12.0,
    "max_retries_per_target": 2,
    "exponential_backoff_base_seconds": 1.5,
    "exponential_backoff_max_seconds": 30.0,
    "retry_jitter_seconds": 0.15,
    "retry_on_http_status": sorted(RETRY_HTTP_STATUSES),
    "retry_on_timeout": True,
    "retry_on_NO_PAIRS_IN_RESPONSE": False,
    "retry_on_404": False,
}


def _runtime_shutdown_requested() -> bool:
    try:
        from app.runtime.shutdown import is_shutting_down

        return is_shutting_down()
    except Exception:
        return False


def _controlled_shutdown_fetch_result(fetch_url: str, *, started: float | None = None) -> dict[str, Any]:
    base = started if started is not None else time.perf_counter()
    return {
        "fetch_status": "CONTROLLED_SHUTDOWN_SKIP",
        "http_status": "",
        "elapsed_ms": int((time.perf_counter() - base) * 1000),
        "pair": None,
        "raw_text": "",
        "error_reason": "controlled_shutdown",
        "timeout": False,
    }


def _sleep_or_shutdown(seconds: float, sleeper: Callable[[float], None]) -> bool:
    if seconds <= 0:
        return not _runtime_shutdown_requested()
    if _runtime_shutdown_requested():
        return False
    if sleeper is not time.sleep:
        sleeper(seconds)
        return not _runtime_shutdown_requested()
    try:
        from app.runtime.shutdown import get_shutdown_event

        return not get_shutdown_event().wait(seconds)
    except Exception:
        sleeper(seconds)
        return not _runtime_shutdown_requested()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_ts(value: Any) -> datetime | None:
    text = cell(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def truthy(value: Any, default: bool = True) -> bool:
    if value is None or cell(value) == "":
        return default
    return cell(value).lower() in {"1", "true", "yes", "y", "on", "active", "enabled"}


def falsy_collection_enabled(value: Any) -> bool:
    """True when collection_enabled explicitly disables collection."""
    text = cell(value).lower()
    return text in {"0", "false", "no", "n", "off", "disabled"}


INACTIVE_MARKERS = (
    "inactive",
    "deprecated",
    "closed",
    "disabled",
    "removed",
    "rejected",
    "still_rejected",
)


def classify_price_required(resolved: dict[str, Any], raw_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decide whether a selected target is price-required for fetch obligation."""
    raw = raw_row or {}
    status = cell(resolved.get("selected_status") or resolved.get("active_status")).lower()
    inactive_reason = ""
    collection_enabled = not falsy_collection_enabled(
        raw.get("collection_enabled") if "collection_enabled" in raw else True
    )

    if falsy_collection_enabled(raw.get("collection_enabled")):
        inactive_reason = "collection_enabled_false"
        collection_enabled = False
    elif any(m in status for m in INACTIVE_MARKERS):
        inactive_reason = f"selected_status_{status or 'inactive'}"
    elif cell(resolved.get("identity_resolution_status")) != "RESOLVED":
        inactive_reason = "unresolved_identity"
    elif not cell(resolved.get("price_source_key")):
        inactive_reason = "missing_price_source_key"
    elif is_internal_lineage_id(resolved.get("display_real_pair_address")):
        inactive_reason = "ae16b_pair_identity_forbidden"

    # Acceptance statuses like PROVIDER_PAIR_RESOLVED are active.
    active = inactive_reason == ""
    price_required = bool(active and collection_enabled)
    return {
        "selected_status": resolved.get("selected_status") or "",
        "active_status": "ACTIVE" if active else "INACTIVE",
        "collection_enabled": "true" if collection_enabled else "false",
        "price_required": "true" if price_required else "false",
        "inactive_reason": inactive_reason,
    }


def load_selected_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: cell(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_open_positions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data.get("open_positions") or []
    return [p for p in positions if cell(p.get("status")).upper() == "OPEN"]


def load_fetch_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("targets"), dict):
        return {k: dict(v) for k, v in raw["targets"].items()}
    if isinstance(raw, dict):
        return {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}
    return {}


def save_fetch_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now_iso(), "targets": state}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cooldown_active(state_row: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    if not state_row:
        return False
    until = parse_ts(state_row.get("skip_until_ts"))
    if not until:
        return False
    current = now or utc_now()
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return until > current


def compute_backoff_seconds(retry_index: int, policy: dict[str, Any]) -> float:
    base = float(policy.get("exponential_backoff_base_seconds", 1.5))
    cap = float(policy.get("exponential_backoff_max_seconds", 30.0))
    jitter = float(policy.get("retry_jitter_seconds", 0.15))
    delay = min(cap, base * (2 ** max(0, retry_index)))
    delay += random.uniform(0.0, max(0.0, jitter))
    return round(delay, 3)


def apply_failure_cooldown(
    state_row: dict[str, Any],
    *,
    failure_class: str,
    http_status: Any = "",
    error_reason: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Update cooldown/dead-pair state. Never removes target from Selected/Clean."""
    current = now or utc_now()
    prev_fail = int(state_row.get("consecutive_failures") or 0)
    prev_nopairs = int(state_row.get("consecutive_no_pairs") or 0)
    prev_skip = cell(state_row.get("skip_until_ts"))

    new_fail = prev_fail
    new_nopairs = prev_nopairs
    cooldown_status = "ACTIVE_FETCH_REQUIRED"
    dead_pair_status = cell(state_row.get("dead_pair_status")) or "NONE"
    skip_until = ""
    next_action = "FETCH_NEXT_CYCLE"

    if failure_class == "SUCCESS":
        new_fail = 0
        new_nopairs = 0
        cooldown_status = "FETCH_SUCCESS"
        dead_pair_status = "NONE"
        skip_until = ""
        next_action = "CONTINUE_NORMAL"
    elif failure_class == "NO_PAIRS_IN_RESPONSE":
        new_fail = prev_fail + 1
        new_nopairs = prev_nopairs + 1
        if new_nopairs >= 3:
            skip_until = (current + timedelta(hours=6)).isoformat()
            cooldown_status = "SUSPECT_DEAD_PAIR"
            dead_pair_status = "SUSPECT_DEAD_PAIR"
            next_action = "SKIP_UNTIL_COOLDOWN_EXPIRES_NO_AUTO_REMOVE"
        elif new_nopairs == 2:
            skip_until = (current + timedelta(minutes=30)).isoformat()
            cooldown_status = "SKIPPED_COOLDOWN_ACTIVE"
            next_action = "SKIP_UNTIL_COOLDOWN_EXPIRES"
        else:
            skip_until = (current + timedelta(minutes=5)).isoformat()
            cooldown_status = "SKIPPED_COOLDOWN_ACTIVE"
            next_action = "SKIP_UNTIL_COOLDOWN_EXPIRES"
    elif failure_class in {"HTTP_404", "PROVIDER_EMPTY"}:
        new_fail = prev_fail + 1
        if str(http_status) == "404" or failure_class == "HTTP_404":
            if new_fail >= 3:
                skip_until = (current + timedelta(hours=6)).isoformat()
                cooldown_status = "SUSPECT_DEAD_PAIR"
                dead_pair_status = "SUSPECT_DEAD_PAIR"
            else:
                skip_until = (current + timedelta(minutes=30)).isoformat()
                cooldown_status = "SKIPPED_COOLDOWN_ACTIVE"
        next_action = "SKIP_UNTIL_COOLDOWN_EXPIRES_NO_AUTO_REMOVE"
    elif failure_class == "RATE_LIMIT":
        new_fail = prev_fail + 1
        backoff = compute_backoff_seconds(new_fail, DEFAULT_POLICY)
        skip_until = (current + timedelta(seconds=backoff)).isoformat()
        cooldown_status = "RATE_LIMIT_BACKOFF"
        next_action = "RETRY_AFTER_BACKOFF"
    elif failure_class in {"TIMEOUT", "HTTP_5XX", "TRANSIENT"}:
        new_fail = prev_fail + 1
        backoff = compute_backoff_seconds(new_fail, DEFAULT_POLICY)
        skip_until = (current + timedelta(seconds=backoff)).isoformat()
        cooldown_status = "TRANSIENT_FAILURE_BACKOFF"
        next_action = "RETRY_AFTER_BACKOFF"
    else:
        new_fail = prev_fail + 1
        skip_until = (current + timedelta(minutes=5)).isoformat()
        cooldown_status = "SKIPPED_COOLDOWN_ACTIVE"

    first_failure = cell(state_row.get("first_failure_at"))
    if failure_class != "SUCCESS":
        if not first_failure:
            first_failure = current.isoformat()
        last_failure = current.isoformat()
    else:
        first_failure = ""
        last_failure = ""

    audit = {
        "price_source_key": state_row.get("price_source_key") or "",
        "failure_class": failure_class,
        "previous_consecutive_failures": prev_fail,
        "new_consecutive_failures": new_fail,
        "previous_consecutive_no_pairs": prev_nopairs,
        "new_consecutive_no_pairs": new_nopairs,
        "previous_skip_until_ts": prev_skip,
        "new_skip_until_ts": skip_until,
        "dead_pair_status": dead_pair_status,
        "automatic_removal_performed": "false",
        "notes": error_reason or "cooldown_updated_no_auto_removal",
    }

    state_row.update(
        {
            "last_fetch_status": failure_class if failure_class != "SUCCESS" else "SUCCESS",
            "last_http_status": str(http_status or ""),
            "last_error_reason": error_reason,
            "consecutive_failures": new_fail,
            "consecutive_no_pairs": new_nopairs,
            "first_failure_at": first_failure,
            "last_failure_at": last_failure,
            "cooldown_status": cooldown_status,
            "skip_until_ts": skip_until,
            "dead_pair_status": dead_pair_status,
            "next_action": next_action,
            "updated_at": current.isoformat(),
        }
    )
    return audit


def build_runtime_priority_queue(
    *,
    selected_rows: list[dict[str, str]],
    open_positions: list[dict[str, Any]],
    fetch_state: dict[str, dict[str, Any]] | None = None,
    include_discovery: bool = False,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build 0A/0B/(1)/2 priority queue. Selected count is dynamic."""
    state = fetch_state or {}
    current = now or utc_now()
    queue: list[dict[str, Any]] = []
    selected_by_key: dict[str, dict[str, Any]] = {}

    resolved_selected: list[tuple[dict[str, str], dict[str, Any]]] = []
    for row in selected_rows:
        resolved = resolve_selected_target_identity(row)
        flags = classify_price_required(resolved, row)
        merged = {**resolved, **flags}
        key = cell(merged.get("price_source_key"))
        if key:
            selected_by_key[key] = merged
        resolved_selected.append((row, merged))

    # 0A open positions
    for pos in open_positions:
        chain = cell(pos.get("chain") or pos.get("provider_chain_id"))
        pair = cell(pos.get("pair_address") or pos.get("provider_pair_id"))
        if is_internal_lineage_id(pair):
            queue.append(
                {
                    "priority_rank": "0A",
                    "priority_class": "OPEN_POSITION_MARK_PRICE",
                    "price_source_key": "",
                    "provider": "dexscreener",
                    "display_chain": chain,
                    "display_real_pair_address": "",
                    "normalized_chain": chain.lower(),
                    "normalized_real_pair_address": "",
                    "provider_pair_url": "",
                    "selected_status": "NOT_SELECTED",
                    "active_status": "OPEN",
                    "collection_enabled": "true",
                    "price_required": "false",
                    "inactive_reason": "unresolved_ae16b_pair_on_open_position",
                    "open_position_status": "UNRESOLVED_IDENTITY",
                    "collection_reason": "MARK_PRICE_ONLY",
                    "eligible_for_new_trade_candidate": "false",
                    "source_reason": "open_position_unresolved_identity",
                    "expected_fetch_required": "false",
                    "identity_resolution_status": "UNRESOLVED_INTERNAL_ID_ONLY",
                    "cooldown_status": "SKIPPED_UNRESOLVED_IDENTITY",
                    "skip_until_ts": "",
                    "notes": "Open position pair identity unresolved; explicit gap",
                    "position_id": pos.get("id"),
                    "target_fetch_status_hint": "SKIPPED_UNRESOLVED_IDENTITY",
                }
            )
            continue
        if not chain or not pair:
            queue.append(
                {
                    "priority_rank": "0A",
                    "priority_class": "OPEN_POSITION_MARK_PRICE",
                    "price_source_key": "",
                    "provider": "dexscreener",
                    "display_chain": chain,
                    "display_real_pair_address": pair,
                    "normalized_chain": chain.lower(),
                    "normalized_real_pair_address": pair.lower(),
                    "provider_pair_url": "",
                    "selected_status": "NOT_SELECTED",
                    "active_status": "OPEN",
                    "collection_enabled": "true",
                    "price_required": "false",
                    "inactive_reason": "missing_chain_or_pair_on_open_position",
                    "open_position_status": "UNRESOLVED_IDENTITY",
                    "collection_reason": "MARK_PRICE_ONLY",
                    "eligible_for_new_trade_candidate": "false",
                    "source_reason": "open_position_unresolved_identity",
                    "expected_fetch_required": "false",
                    "identity_resolution_status": "UNRESOLVED_MISSING_PAIR_OR_CHAIN",
                    "cooldown_status": "SKIPPED_UNRESOLVED_IDENTITY",
                    "skip_until_ts": "",
                    "notes": "Open position missing chain/pair; explicit gap",
                    "position_id": pos.get("id"),
                    "target_fetch_status_hint": "SKIPPED_UNRESOLVED_IDENTITY",
                }
            )
            continue

        key = build_price_source_key("dexscreener", chain.lower(), pair.lower())
        in_selected = key in selected_by_key
        st = state.get(key) or {}
        on_cooldown = cooldown_active(st, now=current)
        if in_selected:
            open_status = "IN_SELECTED_CLEAN"
            collection_reason = "OPEN_POSITION_AND_SELECTED"
            eligible = "true"
            notes = "Open position overlaps Selected/Clean"
            source_reason = "open_paper_position_in_selected"
        else:
            open_status = "LEGACY_OR_OUT_OF_SELECTED_POSITION"
            collection_reason = "MARK_PRICE_ONLY"
            eligible = "false"
            notes = (
                "Open demo/paper position outside Selected/Clean; "
                "mark-price continuity only — does not promote into Selected/Clean"
            )
            source_reason = "open_paper_position_outside_selected"

        queue.append(
            {
                "priority_rank": "0A",
                "priority_class": "OPEN_POSITION_MARK_PRICE",
                "price_source_key": key,
                "provider": "dexscreener",
                "display_chain": chain,
                "display_real_pair_address": pair,
                "normalized_chain": chain.lower(),
                "normalized_real_pair_address": pair.lower(),
                "provider_pair_url": synthesize_dexscreener_url(chain, pair),
                "selected_status": "SELECTED" if in_selected else "NOT_SELECTED",
                "active_status": "OPEN",
                "collection_enabled": "true",
                "price_required": "true",
                "inactive_reason": "",
                "open_position_status": open_status,
                "collection_reason": collection_reason,
                "eligible_for_new_trade_candidate": eligible,
                "source_reason": source_reason,
                "expected_fetch_required": "false" if on_cooldown else "true",
                "identity_resolution_status": "RESOLVED",
                "cooldown_status": "SKIPPED_COOLDOWN_ACTIVE"
                if on_cooldown
                else cell(st.get("cooldown_status")) or "ACTIVE_FETCH_REQUIRED",
                "skip_until_ts": cell(st.get("skip_until_ts")),
                "notes": notes,
                "position_id": pos.get("id"),
                "target_fetch_status_hint": "SKIPPED_COOLDOWN_ACTIVE" if on_cooldown else "",
            }
        )

    open_keys = {r["price_source_key"] for r in queue if r["priority_rank"] == "0A" and r["price_source_key"]}

    # 0B selected
    for _raw, merged in resolved_selected:
        key = cell(merged.get("price_source_key"))
        st = state.get(key) or {} if key else {}
        on_cooldown = cooldown_active(st, now=current) if key else False
        price_required = merged.get("price_required") == "true"
        expected = "true" if price_required and not on_cooldown else "false"
        hint = ""
        if not price_required:
            hint = "SKIPPED_INACTIVE_TARGET"
        elif on_cooldown:
            hint = "SKIPPED_COOLDOWN_ACTIVE"
        notes = "Active Selected/Clean/Preferred target"
        if key in open_keys:
            notes = "Also covered by open-position mark-price priority 0A"
        if merged.get("inactive_reason"):
            notes = f"Not price-required: {merged.get('inactive_reason')}"

        queue.append(
            {
                "priority_rank": "0B",
                "priority_class": "SELECTED_CLEAN_ACTIVE",
                "price_source_key": key,
                "provider": merged.get("provider") or "dexscreener",
                "display_chain": merged.get("display_chain") or "",
                "display_real_pair_address": merged.get("display_real_pair_address") or "",
                "normalized_chain": merged.get("normalized_chain") or "",
                "normalized_real_pair_address": merged.get("normalized_real_pair_address") or "",
                "provider_pair_url": merged.get("provider_pair_url") or "",
                "selected_status": merged.get("selected_status") or "",
                "active_status": merged.get("active_status") or "",
                "collection_enabled": merged.get("collection_enabled") or "true",
                "price_required": merged.get("price_required") or "false",
                "inactive_reason": merged.get("inactive_reason") or "",
                "open_position_status": "ALSO_OPEN" if key in open_keys else "NONE",
                "collection_reason": "SELECTED_UNIVERSE" if price_required else "EXCLUDED_NOT_PRICE_REQUIRED",
                "eligible_for_new_trade_candidate": "true" if price_required else "false",
                "source_reason": "active_selected_clean_preferred",
                "expected_fetch_required": expected,
                "identity_resolution_status": merged.get("identity_resolution_status") or "",
                "cooldown_status": "SKIPPED_COOLDOWN_ACTIVE"
                if on_cooldown
                else (
                    "SKIPPED_INACTIVE_TARGET"
                    if not price_required
                    else cell(st.get("cooldown_status")) or "ACTIVE_FETCH_REQUIRED"
                ),
                "skip_until_ts": cell(st.get("skip_until_ts")),
                "notes": notes,
                "position_id": "",
                "target_fetch_status_hint": hint,
                "combined_target_id": merged.get("combined_target_id") or "",
                "selected_target_id": merged.get("selected_target_id") or "",
            }
        )

    if include_discovery:
        queue.append(
            {
                "priority_rank": "2",
                "priority_class": "DISCOVERY_TRENDING_BACKGROUND",
                "price_source_key": "",
                "provider": "dexscreener",
                "display_chain": "",
                "display_real_pair_address": "",
                "normalized_chain": "",
                "normalized_real_pair_address": "",
                "provider_pair_url": "",
                "selected_status": "NOT_SELECTED",
                "active_status": "N/A",
                "collection_enabled": "true",
                "price_required": "false",
                "inactive_reason": "",
                "open_position_status": "NONE",
                "collection_reason": "DISCOVERY_AFTER_SELECTED",
                "eligible_for_new_trade_candidate": "false",
                "source_reason": "discovery_trending_background",
                "expected_fetch_required": "false",
                "identity_resolution_status": "N/A",
                "cooldown_status": "N/A",
                "skip_until_ts": "",
                "notes": "Discovery only after 0A/0B/1; default smoke excludes",
                "position_id": "",
                "target_fetch_status_hint": "",
            }
        )

    rank_order = {"0A": 0, "0B": 1, "1": 2, "2": 3}
    queue.sort(key=lambda r: (rank_order.get(r["priority_rank"], 9), r.get("price_source_key") or ""))
    return queue


@dataclass
class FetchAttemptResult:
    target_attempt_id: str
    attempted_at: str
    priority_rank: str
    priority_class: str
    price_source_key: str
    provider: str
    display_chain: str
    display_real_pair_address: str
    provider_pair_url: str
    fetch_url: str
    target_fetch_status: str
    http_status: str = ""
    error_reason: str = ""
    elapsed_ms_total: int = 0
    request_attempt_count: int = 0
    raw_payload_written: str = "false"
    raw_payload_id: str = ""
    market_snapshot_written: str = "false"
    market_snapshot_id: str = ""
    source_query_written: str = ""
    source_type_written: str = ""
    selected_status: str = ""
    active_status: str = ""
    price_required: str = ""
    open_position_status: str = ""
    collection_reason: str = ""
    cooldown_status: str = ""
    skip_until_ts: str = ""
    eligible_for_new_trade_candidate: str = ""
    pair_payload: dict[str, Any] | None = None
    raw_response_text: str = ""
    retry_rows: list[dict[str, Any]] = field(default_factory=list)
    cooldown_audit: dict[str, Any] | None = None


def dex_pairs_api_url(chain: str, pair: str) -> str:
    return f"{BASE_URL}/pairs/{cell(chain)}/{cell(pair)}"


def _one_http_fetch(
    *,
    fetch_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if _runtime_shutdown_requested():
        return _controlled_shutdown_fetch_result(fetch_url, started=started)
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
            if _runtime_shutdown_requested():
                return _controlled_shutdown_fetch_result(fetch_url, started=started)
            resp = client.get(fetch_url, headers=_HEADERS)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        body = resp.text or ""
        data: Any = None
        parse_error = ""
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}:{exc}"
        pair = _extract_pair(data) if data is not None else None
        status = int(resp.status_code)
        if status == 429:
            fetch_status = "RATE_LIMIT"
        elif status == 404:
            fetch_status = "HTTP_404"
        elif status >= 500:
            fetch_status = "HTTP_5XX"
        elif status >= 400:
            fetch_status = "HTTP_ERROR"
        elif parse_error:
            fetch_status = "PARSE_ERROR"
        elif pair is None:
            fetch_status = "NO_PAIRS_IN_RESPONSE"
        else:
            fetch_status = "SUCCESS"
        return {
            "fetch_status": fetch_status,
            "http_status": status,
            "elapsed_ms": elapsed_ms,
            "pair": pair,
            "raw_text": body,
            "error_reason": parse_error
            or ("" if fetch_status == "SUCCESS" else fetch_status),
            "timeout": False,
        }
    except httpx.TimeoutException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "fetch_status": "TIMEOUT",
            "http_status": "",
            "elapsed_ms": elapsed_ms,
            "pair": None,
            "raw_text": "",
            "error_reason": f"TimeoutException:{exc}",
            "timeout": True,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "fetch_status": "HTTP_ERROR",
            "http_status": "",
            "elapsed_ms": elapsed_ms,
            "pair": None,
            "raw_text": "",
            "error_reason": f"{type(exc).__name__}:{exc}",
            "timeout": False,
        }


def source_labels_for_item(item: dict[str, Any]) -> tuple[str, str]:
    """Return (source_query, source_type) for selected/open exact fetches."""
    is_open = item.get("priority_rank") == "0A"
    is_selected = item.get("priority_rank") == "0B" or item.get("selected_status") in {
        "SELECTED",
        "PROVIDER_PAIR_RESOLVED",
    } or item.get("priority_class") == "SELECTED_CLEAN_ACTIVE"
    both = is_open and item.get("open_position_status") == "IN_SELECTED_CLEAN"
    if both or (is_open and item.get("selected_status") == "SELECTED"):
        return SOURCE_QUERY_BOTH, SOURCE_TYPE_OPEN
    if is_open:
        return SOURCE_QUERY_OPEN, SOURCE_TYPE_OPEN
    return SOURCE_QUERY_SELECTED, SOURCE_TYPE_SELECTED


def persist_exact_pair_to_db(
    pair: dict[str, Any],
    *,
    source_query: str,
    source_type: str,
    display_chain: str,
    display_pair: str,
    provider_pair_url: str,
) -> dict[str, Any]:
    """Additive write via existing DB helpers. No deletes."""
    from app import database as db
    from app.analytics.scan_persist import new_scan_id

    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    symbol = f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}"
    chain = cell(display_chain) or cell(pair.get("chainId"))
    pair_address = cell(display_pair) or cell(pair.get("pairAddress"))

    raw_id = db.insert_raw_payload(
        provider="dexscreener",
        payload=pair,
        source_type=source_type,
        query=provider_pair_url or source_query,
        chain=chain,
        pair_address=pair_address,
        symbol=symbol,
    )
    coin = db.upsert_coin(
        {
            "symbol": symbol,
            "name": base.get("name", base.get("symbol", "?")),
            "chain": chain,
            "pair_address": pair_address,
            "token_address": base.get("address"),
            "price_usd": float(pair.get("priceUsd") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "raw_ref_id": raw_id,
        }
    )
    snap_id = None
    if coin and coin.get("id"):
        txns = (pair.get("txns") or {}).get("h24") or {}
        buys = int(txns.get("buys") or 0)
        sells = int(txns.get("sells") or 0)
        price = float(pair.get("priceUsd") or 0)
        filter_status = "passed" if price > 0 else "no_price"
        drop_reason = "" if price > 0 else "missing_price_usd"
        snap_id = db.insert_market_snapshot(
            {
                "coin_id": coin["id"],
                "provider": "dexscreener",
                "chain": chain,
                "pair_address": pair_address,
                "price": price,
                "liquidity": float((pair.get("liquidity") or {}).get("usd") or 0),
                "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
                "fdv": float(pair.get("fdv") or 0) if pair.get("fdv") is not None else None,
                "txns_buys": buys,
                "txns_sells": sells,
                "txns_total": buys + sells,
                "price_change_m5": (pair.get("priceChange") or {}).get("m5"),
                "price_change_h1": (pair.get("priceChange") or {}).get("h1"),
                "price_change_h6": (pair.get("priceChange") or {}).get("h6"),
                "price_change_h24": (pair.get("priceChange") or {}).get("h24"),
                "source_query": source_query,
                "filter_status": filter_status,
                "drop_reason": drop_reason,
            }
        )
    return {
        "raw_payload_id": raw_id,
        "market_snapshot_id": snap_id,
        "coin_id": coin.get("id") if coin else None,
        "scan_id": new_scan_id(),
        "price": float(pair.get("priceUsd") or 0),
    }


def fetch_queue_item(
    item: dict[str, Any],
    *,
    policy: dict[str, Any],
    fetch_state: dict[str, dict[str, Any]],
    mode: str = "artifact-only",
    respect_cooldown: bool = True,
    sleeper: Callable[[float], None] | None = None,
    attempt_counter: list[int] | None = None,
) -> FetchAttemptResult:
    """Fetch one priority-queue item with retries/cooldown accounting."""
    sleep_fn = sleeper or time.sleep
    counter = attempt_counter if attempt_counter is not None else [0]
    counter[0] += 1
    attempt_id = f"tgt_{counter[0]:05d}"
    attempted_at = utc_now_iso()
    key = cell(item.get("price_source_key"))
    source_query, source_type = source_labels_for_item(item)

    base = FetchAttemptResult(
        target_attempt_id=attempt_id,
        attempted_at=attempted_at,
        priority_rank=cell(item.get("priority_rank")),
        priority_class=cell(item.get("priority_class")),
        price_source_key=key,
        provider=cell(item.get("provider")) or "dexscreener",
        display_chain=cell(item.get("display_chain")),
        display_real_pair_address=cell(item.get("display_real_pair_address")),
        provider_pair_url=cell(item.get("provider_pair_url")),
        fetch_url="",
        target_fetch_status="",
        selected_status=cell(item.get("selected_status")),
        active_status=cell(item.get("active_status")),
        price_required=cell(item.get("price_required")),
        open_position_status=cell(item.get("open_position_status")),
        collection_reason=cell(item.get("collection_reason")),
        eligible_for_new_trade_candidate=cell(item.get("eligible_for_new_trade_candidate")),
        source_query_written=source_query,
        source_type_written=source_type,
    )

    if _runtime_shutdown_requested():
        base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
        base.error_reason = "controlled_shutdown"
        base.cooldown_status = "CONTROLLED_SHUTDOWN"
        return base

    # Explicit exclusions
    if item.get("target_fetch_status_hint") == "SKIPPED_UNRESOLVED_IDENTITY" or (
        item.get("identity_resolution_status") or ""
    ).startswith("UNRESOLVED"):
        base.target_fetch_status = "SKIPPED_UNRESOLVED_IDENTITY"
        base.error_reason = cell(item.get("inactive_reason")) or "unresolved_identity"
        base.cooldown_status = "SKIPPED_UNRESOLVED_IDENTITY"
        return base

    if item.get("price_required") != "true" and item.get("priority_rank") == "0B":
        base.target_fetch_status = "SKIPPED_INACTIVE_TARGET"
        base.error_reason = cell(item.get("inactive_reason")) or "not_price_required"
        base.cooldown_status = "SKIPPED_INACTIVE_TARGET"
        return base

    if cell(item.get("open_position_status")).upper() == "CLOSED":
        base.target_fetch_status = "SKIPPED_CLOSED_POSITION"
        base.error_reason = "closed_position"
        return base

    st = fetch_state.setdefault(
        key,
        {
            "price_source_key": key,
            "provider": base.provider,
            "display_chain": base.display_chain,
            "display_real_pair_address": base.display_real_pair_address,
            "provider_pair_url": base.provider_pair_url,
            "consecutive_failures": 0,
            "consecutive_no_pairs": 0,
            "cooldown_status": "ACTIVE_FETCH_REQUIRED",
            "skip_until_ts": "",
            "dead_pair_status": "NONE",
        },
    )
    # Keep display fields fresh
    st["display_chain"] = base.display_chain
    st["display_real_pair_address"] = base.display_real_pair_address
    st["provider_pair_url"] = base.provider_pair_url

    if respect_cooldown and cooldown_active(st):
        base.target_fetch_status = "SKIPPED_COOLDOWN_ACTIVE"
        base.cooldown_status = "SKIPPED_COOLDOWN_ACTIVE"
        base.skip_until_ts = cell(st.get("skip_until_ts"))
        base.error_reason = "cooldown_active"
        return base

    if not base.display_chain or not base.display_real_pair_address:
        base.target_fetch_status = "SKIPPED_UNRESOLVED_IDENTITY"
        base.error_reason = "missing_chain_or_pair"
        return base

    fetch_url = dex_pairs_api_url(base.display_chain, base.display_real_pair_address)
    base.fetch_url = fetch_url
    max_retries = int(policy.get("max_retries_per_target", 2))
    timeout = float(policy.get("request_timeout_seconds", 12.0))
    inter_sleep = float(policy.get("sleep_seconds_between_requests", 0.35))

    total_elapsed = 0
    final_status = "HTTP_ERROR"
    last = None
    pair_payload = None
    raw_text = ""

    # retry_index 0 = first try; up to max_retries additional
    for retry_index in range(0, max_retries + 1):
        if retry_index > 0 or (retry_index == 0 and counter[0] > 1):
            # Pace between distinct targets on first attempt of each target handled by caller;
            # between retries always sleep backoff.
            pass
        if retry_index == 0:
            if not _sleep_or_shutdown(inter_sleep, sleep_fn):
                base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
                base.error_reason = "controlled_shutdown"
                base.cooldown_status = "CONTROLLED_SHUTDOWN"
                return base

        if _runtime_shutdown_requested():
            base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
            base.error_reason = "controlled_shutdown"
            base.cooldown_status = "CONTROLLED_SHUTDOWN"
            return base

        result = _one_http_fetch(fetch_url=fetch_url, timeout_seconds=timeout)
        total_elapsed += int(result["elapsed_ms"])
        last = result
        pair_payload = result.get("pair")
        raw_text = result.get("raw_text") or ""
        fetch_status = result["fetch_status"]
        if fetch_status == "CONTROLLED_SHUTDOWN_SKIP":
            final_status = "CONTROLLED_SHUTDOWN_SKIP"
            base.retry_rows.append(
                {
                    "attempt_id": f"{attempt_id}_r{retry_index}",
                    "retry_index": retry_index,
                    "price_source_key": key,
                    "fetch_url": fetch_url,
                    "attempted_at": utc_now_iso(),
                    "fetch_status": fetch_status,
                    "http_status": "",
                    "retry_scheduled": "false",
                    "retry_reason": "controlled_shutdown",
                    "sleep_before_next_attempt_seconds": 0,
                    "backoff_seconds": 0,
                    "elapsed_ms": result.get("elapsed_ms", 0),
                    "final_attempt_for_target": "true",
                }
            )
            break

        retry_scheduled = False
        retry_reason = ""
        backoff = 0.0
        should_retry = False
        if fetch_status == "RATE_LIMIT" or (
            str(result.get("http_status")) in {str(s) for s in policy.get("retry_on_http_status", RETRY_HTTP_STATUSES)}
            and fetch_status in {"RATE_LIMIT", "HTTP_5XX", "HTTP_ERROR"}
            and int(result.get("http_status") or 0) in RETRY_HTTP_STATUSES
        ):
            should_retry = retry_index < max_retries
            retry_reason = f"http_{result.get('http_status')}"
        elif fetch_status == "TIMEOUT" and policy.get("retry_on_timeout", True):
            should_retry = retry_index < max_retries
            retry_reason = "timeout"
        elif fetch_status == "HTTP_5XX":
            should_retry = retry_index < max_retries
            retry_reason = "http_5xx"
        elif fetch_status == "NO_PAIRS_IN_RESPONSE":
            should_retry = False  # cooldown policy handles; no immediate hammer
            retry_reason = "no_pairs_no_immediate_retry"

        if should_retry:
            backoff = compute_backoff_seconds(retry_index, policy)
            retry_scheduled = True
            if not _sleep_or_shutdown(backoff, sleep_fn):
                final_status = "CONTROLLED_SHUTDOWN_SKIP"
                retry_scheduled = False
                should_retry = False

        base.retry_rows.append(
            {
                "attempt_id": f"{attempt_id}_r{retry_index}",
                "retry_index": retry_index,
                "price_source_key": key,
                "fetch_url": fetch_url,
                "attempted_at": utc_now_iso(),
                "fetch_status": fetch_status,
                "http_status": result.get("http_status", ""),
                "retry_scheduled": "true" if retry_scheduled else "false",
                "retry_reason": retry_reason,
                "sleep_before_next_attempt_seconds": backoff if retry_scheduled else 0,
                "backoff_seconds": backoff if retry_scheduled else 0,
                "elapsed_ms": result.get("elapsed_ms", 0),
                "final_attempt_for_target": "false" if should_retry else "true",
            }
        )

        if fetch_status == "SUCCESS":
            final_status = "SUCCESS"
            break
        if fetch_status == "NO_PAIRS_IN_RESPONSE":
            final_status = "PROVIDER_EMPTY_NO_PAIRS"
            break
        if fetch_status == "HTTP_404":
            final_status = "HTTP_ERROR"
            break
        if fetch_status == "PARSE_ERROR":
            final_status = "PARSE_ERROR"
            break
        if fetch_status == "TIMEOUT":
            final_status = "TIMEOUT"
            if not should_retry:
                break
            continue
        if fetch_status == "RATE_LIMIT":
            final_status = "RATE_LIMIT_BACKOFF_EXHAUSTED" if not should_retry else "RATE_LIMIT"
            if not should_retry:
                final_status = "RATE_LIMIT_BACKOFF_EXHAUSTED"
                break
            continue
        if fetch_status == "HTTP_5XX":
            final_status = "HTTP_ERROR"
            if not should_retry:
                break
            continue
        final_status = "HTTP_ERROR"
        if not should_retry:
            break

    base.request_attempt_count = len(base.retry_rows)
    base.elapsed_ms_total = total_elapsed
    base.http_status = str((last or {}).get("http_status") or "")
    base.error_reason = cell((last or {}).get("error_reason"))
    base.raw_response_text = raw_text
    base.pair_payload = pair_payload if final_status == "SUCCESS" else None

    if final_status == "CONTROLLED_SHUTDOWN_SKIP":
        base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
        base.cooldown_status = "CONTROLLED_SHUTDOWN"
        base.error_reason = "controlled_shutdown"
        return base

    # Cooldown updates
    if final_status == "SUCCESS":
        audit = apply_failure_cooldown(st, failure_class="SUCCESS")
        base.target_fetch_status = "SUCCESS"
        base.cooldown_status = "FETCH_SUCCESS"
        base.skip_until_ts = ""
        base.cooldown_audit = audit
        if mode == "write-db" and pair_payload is not None:
            written = persist_exact_pair_to_db(
                pair_payload,
                source_query=source_query,
                source_type=source_type,
                display_chain=base.display_chain,
                display_pair=base.display_real_pair_address,
                provider_pair_url=base.provider_pair_url,
            )
            base.raw_payload_written = "true" if written.get("raw_payload_id") else "false"
            base.raw_payload_id = str(written.get("raw_payload_id") or "")
            if written.get("market_snapshot_id") and float(written.get("price") or 0) > 0:
                base.market_snapshot_written = "true"
                base.market_snapshot_id = str(written.get("market_snapshot_id"))
            elif written.get("market_snapshot_id"):
                base.market_snapshot_written = "true"
                base.market_snapshot_id = str(written.get("market_snapshot_id"))
                # price missing still recorded via filter_status
            else:
                base.market_snapshot_written = "false"
                if not written.get("price"):
                    base.error_reason = "normalization_missing_price"
        return base

    if final_status == "PROVIDER_EMPTY_NO_PAIRS":
        failure_class = "NO_PAIRS_IN_RESPONSE"
    elif final_status == "TIMEOUT":
        failure_class = "TIMEOUT"
    elif final_status == "RATE_LIMIT_BACKOFF_EXHAUSTED":
        failure_class = "RATE_LIMIT"
    elif str(base.http_status) == "404":
        failure_class = "HTTP_404"
    elif str(base.http_status).startswith("5"):
        failure_class = "HTTP_5XX"
    else:
        failure_class = "TRANSIENT"

    audit = apply_failure_cooldown(
        st,
        failure_class=failure_class,
        http_status=base.http_status,
        error_reason=base.error_reason or final_status,
    )
    base.cooldown_audit = audit
    base.target_fetch_status = final_status
    base.cooldown_status = cell(st.get("cooldown_status"))
    base.skip_until_ts = cell(st.get("skip_until_ts"))
    return base


def run_priority_fetch_cycle(
    queue: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    fetch_state: dict[str, dict[str, Any]] | None = None,
    mode: str = "artifact-only",
    respect_cooldown: bool = True,
    selected_only: bool = True,
    include_open_positions: bool = True,
    include_discovery: bool = False,
    max_targets: int | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Execute fetch attempts for eligible 0A/0B items. Default concurrency=1."""
    pol = {**DEFAULT_POLICY, **(policy or {})}
    assert int(pol.get("max_concurrency", 1)) == 1 or int(pol.get("max_concurrency", 1)) >= 1
    # Enforce sequential default for smoke safety
    if int(pol.get("max_concurrency", 1)) != 1:
        # Still sequential in this implementation; record policy value only.
        pass

    state = fetch_state if fetch_state is not None else {}
    attempts: list[FetchAttemptResult] = []
    retry_rows: list[dict[str, Any]] = []
    cooldown_audits: list[dict[str, Any]] = []
    counter = [0]
    fetched = 0

    for item in queue:
        if _runtime_shutdown_requested():
            break
        rank = item.get("priority_rank")
        if rank == "2" and not include_discovery:
            continue
        if rank == "0A" and not include_open_positions:
            continue
        if selected_only and rank not in {"0A", "0B"}:
            continue
        # Always record inactive/cooldown rows for accountability even if not network-fetching
        result = fetch_queue_item(
            item,
            policy=pol,
            fetch_state=state,
            mode=mode,
            respect_cooldown=respect_cooldown,
            sleeper=sleeper,
            attempt_counter=counter,
        )
        attempts.append(result)
        retry_rows.extend(result.retry_rows)
        if result.cooldown_audit:
            cooldown_audits.append(result.cooldown_audit)

        # Count network fetches toward max_targets
        if result.target_fetch_status not in {
            "SKIPPED_COOLDOWN_ACTIVE",
            "SKIPPED_INACTIVE_TARGET",
            "SKIPPED_CLOSED_POSITION",
            "SKIPPED_UNRESOLVED_IDENTITY",
        }:
            fetched += 1
            if max_targets is not None and fetched >= max_targets:
                # Still must account for remaining targets as NOT attempted? 
                # For bounded smoke with max_targets, remaining selected must be marked.
                # Caller handles leftover accounting when max_targets used for partial smoke.
                break

    return {
        "attempts": attempts,
        "retry_rows": retry_rows,
        "cooldown_audits": cooldown_audits,
        "fetch_state": state,
        "policy": pol,
        "network_fetch_count": fetched,
    }


def payload_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def selected_collection_enabled(environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("RUNTIME_SELECTED_COLLECTION_ENABLED", "true") or "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}
