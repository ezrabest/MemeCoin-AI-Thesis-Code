"""Safe PATCH-style settings updates with canonical key validation."""
from __future__ import annotations

from typing import Any

from .. import database as db
from .effective_settings import CANONICAL_KEYS, SETTING_ALIASES, get_effective_settings
from .settings_normalize import (
    DECIMAL_FRACTION_PCT_KEYS,
    INT_KEYS,
    normalize_canonical_settings,
    normalize_decimal_fraction_pct,
    normalize_required_margin_after_costs_pct,
)

# Keys that must never be enabled via UI PATCH (conservative LIVE safety).
BLOCKED_PATCH_VALUES: dict[str, Any] = {
    "live_trading_enabled": True,
}

# Keys rejected on PATCH — read-only from System Configuration UI.
READ_ONLY_PATCH_KEYS = frozenset({
    "live_trading_enabled",
    "tab_standalone_trading_enabled",
    "tab_rescue_enabled",
})

# Legacy alias keys — reject; callers must use canonical keys.
LEGACY_ALIAS_KEYS = frozenset(SETTING_ALIASES.keys())

PATCHABLE_CANONICAL_KEYS = frozenset(CANONICAL_KEYS) - READ_ONLY_PATCH_KEYS


class SettingsPatchError(Exception):
    """Raised when PATCH validation fails."""

    def __init__(self, message: str, field_errors: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.field_errors = field_errors or {}


def _validate_single_key(key: str, value: Any) -> tuple[Any | None, str | None]:
    """Return (normalized_value, error_message)."""
    if key in LEGACY_ALIAS_KEYS:
        return None, f"Use canonical key instead of legacy alias '{key}'"
    if key in READ_ONLY_PATCH_KEYS:
        return None, f"'{key}' is read-only"
    if key not in PATCHABLE_CANONICAL_KEYS:
        return None, f"Unknown or non-configurable key '{key}'"
    if key in BLOCKED_PATCH_VALUES and value is BLOCKED_PATCH_VALUES[key]:
        return None, f"'{key}' cannot be enabled via settings PATCH"

    if isinstance(value, bool) or key.endswith("_enabled"):
        if not isinstance(value, bool):
            if str(value).strip().lower() in ("true", "1", "yes", "on"):
                value = True
            elif str(value).strip().lower() in ("false", "0", "no", "off"):
                value = False
            else:
                return None, "Expected boolean value"
        return value, None

    if key in DECIMAL_FRACTION_PCT_KEYS:
        norm = normalize_decimal_fraction_pct(value)
        if norm is None:
            return None, "Invalid numeric percent value"
        return norm, None

    if key == "required_margin_after_costs_pct":
        norm = normalize_required_margin_after_costs_pct(value)
        if norm is None:
            return None, "Invalid numeric value"
        return norm, None

    if key in INT_KEYS:
        try:
            if value is None or value == "":
                return None, "Required integer value"
            return int(round(float(value))), None
        except (TypeError, ValueError):
            return None, "Invalid integer value"

    if key == "min_liquidity_usd":
        try:
            fval = float(value)
            if fval < 0:
                return None, "Must be non-negative"
            return fval, None
        except (TypeError, ValueError):
            return None, "Invalid numeric value"

    if key == "tab_confidence_suffix":
        text = str(value).strip()
        if not text:
            return None, "Required text value"
        return text, None

    if key == "prompt_behavior":
        text = str(value).strip().lower()
        if text not in ("conservative", "balanced", "aggressive"):
            return None, "Must be conservative, balanced, or aggressive"
        return text, None

    if key in ("trading_mode", "mode"):
        text = str(value).strip().upper()
        if text not in ("DEMO", "LIVE"):
            return None, "Must be DEMO or LIVE"
        if text == "LIVE":
            return None, "LIVE mode cannot be set via settings PATCH"
        return text, None

    # Generic float / probability interval keys
    try:
        if value is None or value == "":
            return None, "Required numeric value"
        fval = float(value)
        return fval, None
    except (TypeError, ValueError):
        return None, "Invalid numeric value"


def patch_settings(dirty: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a dirty canonical-key payload to persisted settings.

    Returns GET /api/settings/effective-compatible dict.
    Raises SettingsPatchError with field_errors keyed by canonical name.
    """
    if not isinstance(dirty, dict):
        raise SettingsPatchError("Request body must be a JSON object")
    if not dirty:
        raise SettingsPatchError("Empty PATCH payload — send at least one changed field")

    field_errors: dict[str, str] = {}
    normalized: dict[str, Any] = {}

    for key, value in dirty.items():
        norm, err = _validate_single_key(key, value)
        if err:
            field_errors[key] = err
        else:
            normalized[key] = norm

    if field_errors:
        raise SettingsPatchError("Validation failed", field_errors=field_errors)

    # Final normalization pass (consistent with effective settings path)
    normalized = normalize_canonical_settings(normalized)

    for key, value in normalized.items():
        db.upsert_setting(key, value)

    eff = get_effective_settings()
    payload = eff.to_api_response()
    from pathlib import Path

    report_path = eff.write_audit_report()
    payload["audit_report_path"] = str(report_path)
    return payload
