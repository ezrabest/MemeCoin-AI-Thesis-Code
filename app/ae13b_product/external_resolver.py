"""Optional external contract/identity resolver — never called silently.

Default: local data only.
Modes: local_only | ask_before_external | external_enabled_for_watchlist
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CONFIG_PATH = DATA_DIR / "external_resolver_config.json"
CACHE_PATH = DATA_DIR / "external_resolver_cache.json"
_LOCK = threading.RLock()

MODE_LOCAL = "local_only"
MODE_ASK = "ask_before_external"
MODE_ENABLED = "external_enabled_for_watchlist"

DEFAULT_CONFIG = {
    "mode": MODE_LOCAL,
    "external_resolver_enabled": False,
    "provider_name": None,
    "rate_limit_per_minute": 6,
    "timeout_seconds": 8,
    "cache_ttl_seconds": 3600,
    "last_lookup_at": None,
    "last_lookup_status": None,
    "last_lookup_error": None,
    "paper_demo_only": True,
    "not_used_for_live_trading_authority": True,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        cfg = dict(DEFAULT_CONFIG)
        _save_config(cfg)
        return cfg
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_CONFIG)
        out = dict(DEFAULT_CONFIG)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_external_resolver_status() -> dict[str, Any]:
    with _LOCK:
        cfg = _load_config()
        enabled = bool(cfg.get("external_resolver_enabled")) and cfg.get("mode") == MODE_ENABLED
        return {
            "external_resolver_enabled": enabled,
            "mode": cfg.get("mode") or MODE_LOCAL,
            "provider_name": cfg.get("provider_name"),
            "provider_configured": bool(cfg.get("provider_name")),
            "last_lookup_at": cfg.get("last_lookup_at"),
            "last_lookup_status": cfg.get("last_lookup_status"),
            "last_lookup_error": cfg.get("last_lookup_error"),
            "rate_limit_per_minute": cfg.get("rate_limit_per_minute"),
            "timeout_seconds": cfg.get("timeout_seconds"),
            "cache_ttl_seconds": cfg.get("cache_ttl_seconds"),
            "explanation": (
                "External resolver enabled for Watchlist (explicit)."
                if enabled
                else (
                    "Ask before external lookup — user must confirm per item."
                    if cfg.get("mode") == MODE_ASK
                    else "Local data only. External resolver not enabled. "
                    "This item is tracked from user input and local data only."
                )
            ),
            "silent_calls_forbidden": True,
            "paper_demo_only": True,
            "not_used_for_live_trading_authority": True,
        }


def set_external_resolver_mode(mode: str, *, provider_name: str | None = None) -> dict[str, Any]:
    allowed = {MODE_LOCAL, MODE_ASK, MODE_ENABLED}
    if mode not in allowed:
        raise ValueError(f"mode must be one of {sorted(allowed)}")
    with _LOCK:
        cfg = _load_config()
        cfg["mode"] = mode
        cfg["external_resolver_enabled"] = mode == MODE_ENABLED
        if provider_name is not None:
            cfg["provider_name"] = provider_name or None
        cfg["updated_at"] = _utc_now()
        _save_config(cfg)
        return get_external_resolver_status()


def enable_external_lookup_for_item(item_id: str) -> dict[str, Any]:
    """Mark a single watchlist item as allowed for external lookup (still no silent call)."""
    with _LOCK:
        cfg = _load_config()
        allowed = list(cfg.get("per_item_allowlist") or [])
        if item_id not in allowed:
            allowed.append(item_id)
        cfg["per_item_allowlist"] = allowed
        cfg["updated_at"] = _utc_now()
        _save_config(cfg)
        return {
            "ok": True,
            "item_id": item_id,
            "external_lookup_allowed_for_item": True,
            "status": get_external_resolver_status(),
            "note": (
                "External lookup flagged for this item. "
                "No provider call is made until a configured provider exists and lookup is invoked."
            ),
        }


def attempt_external_lookup(
    *,
    chain: str | None,
    contract_or_pair_address: str | None,
    symbol: str | None = None,
    item_id: str | None = None,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Explicit external lookup — stubbed when no provider is configured.

    Never called from silent/background paths. Logged via config last_lookup_*.
    """
    with _LOCK:
        cfg = _load_config()
        status = get_external_resolver_status()
        mode = cfg.get("mode") or MODE_LOCAL
        allowlist = set(cfg.get("per_item_allowlist") or [])

        if mode == MODE_LOCAL:
            cfg["last_lookup_at"] = _utc_now()
            cfg["last_lookup_status"] = "blocked_local_only"
            cfg["last_lookup_error"] = "External resolver not enabled (local_only)."
            _save_config(cfg)
            return {
                "ok": False,
                "resolution_status": "provider_unavailable",
                "resolution_source": "external_provider",
                "cache_hit": False,
                "reason": status["explanation"],
                "external_resolver_attempted": False,
                "external_resolver_enabled": False,
                "provider_name": None,
            }

        if mode == MODE_ASK and not user_confirmed and (not item_id or item_id not in allowlist):
            cfg["last_lookup_at"] = _utc_now()
            cfg["last_lookup_status"] = "blocked_needs_confirmation"
            cfg["last_lookup_error"] = "Ask-before mode requires explicit confirmation."
            _save_config(cfg)
            return {
                "ok": False,
                "resolution_status": "provider_unavailable",
                "resolution_source": "external_provider",
                "cache_hit": False,
                "reason": "Ask before external lookup — confirm to proceed for this item.",
                "external_resolver_attempted": False,
                "needs_user_confirmation": True,
            }

        provider = cfg.get("provider_name")
        if not provider:
            cfg["last_lookup_at"] = _utc_now()
            cfg["last_lookup_status"] = "provider_unavailable"
            cfg["last_lookup_error"] = "No external provider configured (stub only)."
            _save_config(cfg)
            return {
                "ok": False,
                "resolution_status": "provider_unavailable",
                "resolution_source": "external_provider",
                "cache_hit": False,
                "reason": (
                    "External resolver interface is available but no provider is configured. "
                    "Item remains tracked from user input and local data only."
                ),
                "external_resolver_attempted": True,
                "external_resolver_available": False,
                "provider_name": None,
                "checked_at": _utc_now(),
            }

        # Provider configured but project intentionally does not ship paid/silent APIs.
        cfg["last_lookup_at"] = _utc_now()
        cfg["last_lookup_status"] = "stub_no_network_call"
        cfg["last_lookup_error"] = (
            f"Provider '{provider}' is named but network lookup is not implemented "
            "(no hidden external API calls)."
        )
        _save_config(cfg)
        return {
            "ok": False,
            "resolution_status": "provider_unavailable",
            "resolution_source": "external_provider",
            "cache_hit": False,
            "matched_symbol": None,
            "matched_name": None,
            "matched_chain": chain,
            "matched_contract_address": contract_or_pair_address,
            "confidence": 0.0,
            "reason": cfg["last_lookup_error"],
            "external_resolver_attempted": True,
            "external_resolver_available": False,
            "provider_name": provider,
            "checked_at": _utc_now(),
            "symbol_queried": symbol,
        }
