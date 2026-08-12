"""Shared provider pair URL lookup-key normalization (AE18 resilience).

Canonical identity remains ``provider_pair_url_exact`` (exact case preserved).
``normalized_provider_pair_url_key`` is a lookup key only — never replace
canonical identity with the normalized key.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse


class ProviderUrlKeyError(ValueError):
    """Raised when a provider URL cannot be normalized into a lookup key."""

    def __init__(self, reason: str, *, url: str = ""):
        self.reason = reason
        self.url = url
        super().__init__(reason)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_provider_pair_url_key(
    provider_pair_url_exact: str,
    *,
    require_dexscreener: bool = False,
) -> str:
    """Normalize a provider pair URL into a stable lookup key.

    Behavior:
    - trim leading/trailing whitespace
    - remove trailing slash only
    - preserve exact case of scheme, host, path, and final segment
    - do not lowercase the path or final segment
    - do not URL-decode/re-encode in a way that changes identity
    - reject empty / malformed values with an explicit reason
    - optionally reject non-DexScreener URLs (cache/override usage)
    """
    raw = _cell(provider_pair_url_exact)
    if not raw:
        raise ProviderUrlKeyError("empty_provider_pair_url", url=raw)

    if any(ch.isspace() for ch in raw):
        raise ProviderUrlKeyError("malformed_provider_url_internal_whitespace", url=raw)

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ProviderUrlKeyError("malformed_provider_url_missing_scheme_or_host", url=raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ProviderUrlKeyError("malformed_provider_url_unsupported_scheme", url=raw)

    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path or ""
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    elif path == "/":
        path = ""

    key = urlunparse((scheme, netloc, path, "", "", ""))
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        raise ProviderUrlKeyError("malformed_provider_url_missing_pair_path", url=raw)

    if require_dexscreener:
        host = netloc.lower()
        if "dexscreener.com" not in host:
            raise ProviderUrlKeyError("non_dexscreener_url_rejected", url=raw)

    return key


def try_normalize_provider_pair_url_key(
    provider_pair_url_exact: str,
    *,
    require_dexscreener: bool = False,
) -> tuple[str | None, str | None]:
    """Return ``(key, None)`` on success or ``(None, reason)`` on failure."""
    try:
        return (
            normalize_provider_pair_url_key(
                provider_pair_url_exact, require_dexscreener=require_dexscreener
            ),
            None,
        )
    except ProviderUrlKeyError as exc:
        return None, exc.reason
