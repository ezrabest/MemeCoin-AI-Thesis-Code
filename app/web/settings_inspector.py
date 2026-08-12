"""Build technical inspector rows from GET /api/settings/effective payload."""
from __future__ import annotations

from typing import Any

from app.web.percent_conversion import format_display_value, format_unit_label
from app.web.settings_field_metadata import FIELD_SPEC_BY_KEY


def _alias_for_canonical(canonical_key: str, aliases_resolved: dict[str, Any]) -> str:
    from app.observability.effective_settings import SETTING_ALIASES

    parts: list[str] = []
    for alias, target in SETTING_ALIASES.items():
        if target == canonical_key and alias in aliases_resolved:
            parts.append(f"{alias}={aliases_resolved[alias]!r}")
    return "; ".join(parts) if parts else ""


def _active_status(key: str, canonical: dict[str, Any], sources: dict[str, str]) -> str:
    val = canonical.get(key)
    source = sources.get(key, "default")
    if key == "live_trading_enabled" and not val:
        return "blocked OFF"
    if key == "economic_gate_enabled" and not val:
        return "inactive"
    if key.startswith("tab_") and key.endswith("_enabled") and not val:
        return "inactive"
    if source == "default":
        return "using default"
    if source.startswith("alias:"):
        return "configured (alias resolved)"
    if source.startswith("env:"):
        return "env override active"
    return "active"


def dependency_notes(key: str, canonical: dict[str, Any]) -> str:
    notes: list[str] = []
    mode = str(canonical.get("trading_mode") or canonical.get("mode") or "DEMO").upper()
    if key.startswith("tab_") and canonical.get("tab_confidence_boost_enabled") and not canonical.get("economic_gate_enabled"):
        notes.append("TAB configured but inactive — economic gate OFF")
    if key == "tab_confidence_boost_enabled_demo" and canonical.get("tab_confidence_boost_enabled_demo") and mode not in ("DEMO", "PAPER"):
        notes.append("TAB DEMO boost not active in current mode")
    if key == "llm_enabled_for_live" and canonical.get("llm_enabled_for_live") and not canonical.get("live_trading_enabled"):
        notes.append("LLM live flag set but LIVE trading disabled")
    return "; ".join(notes)


def build_canonical_inspector_rows(effective: dict[str, Any]) -> list[dict[str, str]]:
    canonical = effective.get("canonical") or {}
    sources = effective.get("sources") or {}
    defaults = effective.get("defaults") or {}
    aliases = effective.get("aliases_resolved") or {}

    rows: list[dict[str, str]] = []
    for key in sorted(canonical.keys()):
        internal = canonical[key]
        spec = FIELD_SPEC_BY_KEY.get(key)
        label = spec.label if spec else key.replace("_", " ").title()
        consumer = spec.consumer if spec else ""
        extra_notes = spec.notes if spec else ""
        dep = dependency_notes(key, canonical)
        notes = "; ".join(p for p in (extra_notes, dep) if p)
        rows.append({
            "ui_label": label,
            "canonical_key": key,
            "displayed_value": format_display_value(key, internal),
            "internal_value": repr(internal),
            "unit": format_unit_label(key) or type(internal).__name__,
            "source": str(sources.get(key, "default")),
            "default_value": repr(defaults.get(key, "")),
            "alias_resolved": _alias_for_canonical(key, aliases),
            "active_status": _active_status(key, canonical, sources),
            "backend_consumer": consumer,
            "notes_warnings": notes,
        })
    return rows


def _flatten_hidden(obj: Any, prefix: str, section: str) -> list[tuple[str, Any, str]]:
    items: list[tuple[str, Any, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "source":
                continue
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(_flatten_hidden(v, path, section))
            else:
                items.append((f"hidden:{section}.{path}", v, section))
    return items


def build_hidden_threshold_rows(effective: dict[str, Any]) -> list[dict[str, str]]:
    hidden = effective.get("hidden_thresholds") or {}
    rows: list[dict[str, str]] = []
    for section, block in sorted(hidden.items()):
        if not isinstance(block, dict):
            continue
        source = str(block.get("source", section))
        for path, value, _sec in _flatten_hidden(block, section, section):
            rows.append({
                "ui_label": path.replace("hidden:", "").replace(".", " / "),
                "canonical_key": path,
                "displayed_value": format_display_value("", value) if not isinstance(value, (list, dict)) else str(value),
                "internal_value": repr(value),
                "unit": type(value).__name__,
                "source": source,
                "default_value": "hard-coded",
                "alias_resolved": "",
                "active_status": "read-only threshold",
                "backend_consumer": source,
                "notes_warnings": "hidden_threshold — not directly editable",
            })
    return rows


def build_inspector_rows(effective: dict[str, Any]) -> list[dict[str, str]]:
    return build_canonical_inspector_rows(effective) + build_hidden_threshold_rows(effective)


def global_dependency_warnings(canonical: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not canonical.get("economic_gate_enabled"):
        lines.append("Economic gate is OFF — RF/economic approval will not promote candidates.")
    if not canonical.get("paper_trading_enabled"):
        mode = str(canonical.get("trading_mode") or canonical.get("mode") or "DEMO").upper()
        if mode == "DEMO":
            lines.append("Paper trading is OFF — no paper orders will be created.")
    if not canonical.get("live_trading_enabled"):
        lines.append("LIVE trading disabled.")
    if canonical.get("tab_confidence_boost_enabled") and not canonical.get("economic_gate_enabled"):
        lines.append("TAB configured but inactive because economic gate is OFF.")
    mode = str(canonical.get("trading_mode") or canonical.get("mode") or "DEMO").upper()
    if canonical.get("tab_confidence_boost_enabled_demo") and mode not in ("DEMO", "PAPER"):
        lines.append("TAB DEMO boost is not active in current mode.")
    if not canonical.get("tab_confidence_boost_enabled_live"):
        lines.append("TAB LIVE boost disabled.")
    if canonical.get("llm_enabled_for_live") and not canonical.get("live_trading_enabled"):
        lines.append("LLM LIVE setting inactive because LIVE trading is disabled.")
    return lines
