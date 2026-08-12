"""AE13I Smoke Addendum (Part D) — global ASCII-safe text sanitizer.

Wraps app.ae12_reporting.ascii_text (mojibake repair + Unicode punctuation ->
ASCII normalization) so any API response payload can be sanitized in one
call before being returned to the client, regardless of which endpoint or
module produced the strings.
"""
from __future__ import annotations

from typing import Any

from app.ae12_reporting.ascii_text import sanitize_ui_text

#: Extra mojibake byte-sequences observed in this codebase's smoke tests,
#: beyond what ascii_text.sanitize_ui_text already repairs.
_EXTRA_MOJIBAKE_REPAIRS = {
    "\u00e2\u0080\u0094": "-",  # em dash mojibake (a-hat, control chars)
    "\u00e2\u0080\u0093": "-",  # en dash mojibake
    "\u00e2\u0080\u00a6": "...",  # ellipsis mojibake
    "\u00e2\u0080\u009c": '"',  # left double quote mojibake
    "\u00e2\u0080\u009d": '"',  # right double quote mojibake
    "\u00e2\u0080\u0099": "'",  # right single quote / apostrophe mojibake
    "\u00e2\u0080\u0098": "'",  # left single quote mojibake
    "\u00e2\u0080\u00a2": "*",  # bullet mojibake
}

#: Direct Unicode punctuation -> ASCII normalization applied after mojibake repair.
_UNICODE_TO_ASCII = {
    "\u2014": "-",  # em dash
    "\u2013": "-",  # en dash
    "\u2026": "...",  # ellipsis
    "\u00b7": " | ",  # middle dot
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2260": " differs from ",  # not-equal-to
}


def sanitize_text(text: str) -> str:
    """Return an ASCII-safe version of ``text``.

    Repairs known mojibake byte sequences, then normalizes remaining Unicode
    dash / ellipsis / quote / middle-dot punctuation to ASCII equivalents.
    Falsy / non-string input is returned unchanged.
    """
    if not text or not isinstance(text, str):
        return text
    out = text
    for bad, good in _EXTRA_MOJIBAKE_REPAIRS.items():
        out = out.replace(bad, good)
    # Reuse ascii_text's own mojibake repair + dash/ellipsis normalization for
    # anything not covered above (keeps a single source of truth for AE12).
    out = sanitize_ui_text(out) or out
    for bad, good in _UNICODE_TO_ASCII.items():
        out = out.replace(bad, good)
    return out


def sanitize_payload(value: Any) -> Any:
    """Recursively sanitize strings inside dict/list/tuple structures.

    Non-string leaf values (numbers, booleans, None, etc.) are returned
    unchanged. Dict keys are left untouched — only values are sanitized —
    since keys are field names, not human-facing display copy.
    """
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {k: sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(v) for v in value)
    return value
