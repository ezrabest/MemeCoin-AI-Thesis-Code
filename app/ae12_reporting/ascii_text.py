"""ASCII-safe text helpers for AE12 reporting (avoid Windows mojibake)."""

from __future__ import annotations

import re
from typing import Any

# Fragments that indicate mojibake / problematic encoding in display strings.
MOJIBAKE_FRAGMENTS = (
    "â",
    "\u00e2",
    "\u0080",
    "\u0093",
    "\u0094",
)

# Characters we avoid in AE12 API/UI payload display strings.
NON_ASCII_DISPLAY_CHARS = (
    "\u2260",  # ≠
    "\u2014",  # —
    "\u2013",  # –
    "\u2026",  # …
    "\u00b7",  # ·
)


def contains_mojibake(text: str) -> bool:
    if not text:
        return False
    for frag in MOJIBAKE_FRAGMENTS:
        if frag in text:
            return True
    # Classic UTF-8-as-cp1252 mojibake for em-dash / ≠
    if "â€" in text or "â‰" in text or "â>" in text:
        return True
    return False


def to_ascii_display(text: str) -> str:
    """Replace common Unicode punctuation with ASCII for Windows-safe display."""
    if not text:
        return text
    out = (
        text.replace("\u2260", " differs from ")
        .replace("\u2014", " - ")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
        .replace("\u00b7", " | ")
    )
    # Collapse accidental double spaces from replacements
    out = re.sub(r" {2,}", " ", out)
    return out


def sanitize_ui_text(text: str | None) -> str | None:
    """Sanitize a human-facing string for ASCII-safe display (AE13G).

    Repairs common UTF-8-as-cp1252 mojibake byte sequences (e.g. em-dash
    encoded as "\u00e2\u0080\u0094") and then normalizes remaining Unicode
    dash/ellipsis punctuation to ASCII via :func:`to_ascii_display`.

    Falsy input (None, "") is returned unchanged.
    """
    if not text:
        return text
    out = str(text)
    mojibake_repairs = {
        "\u00e2\u0080\u0094": "-",  # em dash mojibake
        "\u00e2\u0080\u0093": "-",  # en dash mojibake
        "\u00e2\u0080\u00a6": "...",  # ellipsis mojibake
    }
    for bad, good in mojibake_repairs.items():
        out = out.replace(bad, good)
    return to_ascii_display(out)


def walk_strings_for_mojibake(obj: Any, *, path: str = "") -> list[str]:
    """Return list of path:snippet for any mojibake found in nested JSON-like data."""
    hits: list[str] = []
    if isinstance(obj, str):
        if contains_mojibake(obj) or any(ch in obj for ch in NON_ASCII_DISPLAY_CHARS):
            hits.append(f"{path or '<root>'}: {obj[:120]}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(walk_strings_for_mojibake(v, path=f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(walk_strings_for_mojibake(v, path=f"{path}[{i}]"))
    return hits
