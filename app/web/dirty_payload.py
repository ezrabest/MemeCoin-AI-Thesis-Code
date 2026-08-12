"""Build PATCH-style dirty canonical payloads."""
from __future__ import annotations

from typing import Any


def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def build_dirty_payload(
    *,
    form_values: dict[str, Any],
    loaded_canonical: dict[str, Any],
    editable_keys: set[str],
) -> dict[str, Any]:
    dirty: dict[str, Any] = {}
    for key in editable_keys:
        if key not in form_values:
            continue
        new_val = form_values[key]
        old_val = loaded_canonical.get(key)
        if not _values_equal(new_val, old_val):
            dirty[key] = new_val
    return dirty
