"""AE18 Solana/Helius pre-flight safety — fail closed on wallet/signer env risk."""

from __future__ import annotations

import os
from typing import Any

FORBIDDEN_WALLET_ENV_VARS: tuple[str, ...] = (
    "PRIVATE_KEY",
    "WALLET_PRIVATE_KEY",
    "WALLET_SECRET",
    "WALLET_SECRET_KEY",
    "SOLANA_PRIVATE_KEY",
    "SOLANA_WALLET_PRIVATE_KEY",
    "SOLANA_KEYPAIR",
    "KEYPAIR_PATH",
    "SIGNER",
    "SIGNER_KEY",
    "SECRET_KEY",
    "MNEMONIC",
    "SEED_PHRASE",
    "JUPITER_PRIVATE_KEY",
)

PRIVATE_KEY_ENV_VARS: frozenset[str] = frozenset(
    {
        "PRIVATE_KEY",
        "WALLET_PRIVATE_KEY",
        "WALLET_SECRET",
        "WALLET_SECRET_KEY",
        "SOLANA_PRIVATE_KEY",
        "SOLANA_WALLET_PRIVATE_KEY",
        "SOLANA_KEYPAIR",
        "KEYPAIR_PATH",
        "SECRET_KEY",
        "MNEMONIC",
        "SEED_PHRASE",
        "JUPITER_PRIVATE_KEY",
    }
)

SIGNER_ENV_VARS: frozenset[str] = frozenset({"SIGNER", "SIGNER_KEY"})

WALLET_ENV_VARS: frozenset[str] = frozenset(
    {
        "WALLET_PRIVATE_KEY",
        "WALLET_SECRET",
        "WALLET_SECRET_KEY",
        "SOLANA_WALLET_PRIVATE_KEY",
    }
)


def _present_env_names(names: tuple[str, ...] | frozenset[str], env: dict[str, str] | None = None) -> list[str]:
    source = env if env is not None else os.environ
    found: list[str] = []
    for name in names:
        val = source.get(name)
        if val is not None and str(val).strip() != "":
            found.append(name)
    return sorted(found)


def run_solana_preflight_safety(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Inspect env for wallet/private-key/signer capability. Never returns values."""
    present = _present_env_names(FORBIDDEN_WALLET_ENV_VARS, env)
    wallet_present = [n for n in present if n in WALLET_ENV_VARS]
    private_present = [n for n in present if n in PRIVATE_KEY_ENV_VARS]
    signer_present = [n for n in present if n in SIGNER_ENV_VARS]
    passed = len(present) == 0
    return {
        "audit": "ae18_solana_preflight_safety_audit",
        "preflight_passed": passed,
        "wallet_env_vars_present": wallet_present,
        "private_key_env_vars_present": private_present,
        "signer_env_vars_present": signer_present,
        "forbidden_env_vars_present": present,
        "wallet_access": False,
        "private_key_access": False,
        "signer_available": False,
        "transaction_builder_available": False,
        "transaction_signing_available": False,
        "transaction_submission_available": False,
        "live_trading_enabled": False,
        "trade_authority": False,
        "fail_closed": True,
        "issues": [f"forbidden_env_present:{n}" for n in present],
    }


def resolve_rpc_config(
    *,
    allow_public_rpc: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve Helius/Solana RPC endpoint from env. Never returns API key value."""
    source = env if env is not None else os.environ
    helius_key = bool(str(source.get("HELIUS_API_KEY") or "").strip())
    helius_rpc = str(source.get("HELIUS_RPC_URL") or "").strip()
    solana_rpc = str(source.get("SOLANA_RPC_URL") or "").strip()

    provider = None
    rpc_url = ""
    helius_configured = False

    if helius_rpc:
        provider = "HELIUS_RPC"
        rpc_url = helius_rpc
        helius_configured = True
    elif helius_key:
        # Construct URL without exposing key in returned config for audits;
        # caller builds authenticated URL separately.
        provider = "HELIUS_RPC"
        rpc_url = "https://mainnet.helius-rpc.com/"
        helius_configured = True
    elif solana_rpc:
        provider = "SOLANA_RPC"
        rpc_url = solana_rpc
    elif allow_public_rpc:
        provider = "SOLANA_PUBLIC_RPC"
        rpc_url = "https://api.mainnet-beta.solana.com"

    return {
        "helius_configured": helius_configured,
        "helius_api_key_present": helius_key,
        "helius_rpc_url_present": bool(helius_rpc),
        "solana_rpc_url_present": bool(solana_rpc),
        "allow_public_rpc": allow_public_rpc,
        "rpc_provider_used": provider,
        "rpc_configured": provider is not None,
        # Never include the actual key. Include URL only when it does not embed secrets.
        "rpc_url_for_client": _safe_rpc_url(provider, helius_rpc, solana_rpc, helius_key, allow_public_rpc),
    }


def _safe_rpc_url(
    provider: str | None,
    helius_rpc: str,
    solana_rpc: str,
    helius_key: bool,
    allow_public_rpc: bool,
) -> str:
    if provider == "HELIUS_RPC":
        if helius_rpc:
            return helius_rpc
        if helius_key:
            # Caller must append api-key at request time from env; return sentinel.
            return "HELIUS_API_KEY_AUTH"
        return ""
    if provider == "SOLANA_RPC":
        return solana_rpc
    if provider == "SOLANA_PUBLIC_RPC" and allow_public_rpc:
        return "https://api.mainnet-beta.solana.com"
    return ""


def probe_rpc_auth(rpc_url: str, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    """Lightweight auth/connectivity probe. Uses getSlot (not used for AE18 context)."""
    import httpx

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": []},
                headers={"Content-Type": "application/json"},
            )
        ok = resp.status_code == 200
        body_preview = (resp.text or "")[:120]
        invalid_key = resp.status_code == 401 or "invalid api key" in body_preview.lower()
        return {
            "probe_ok": ok,
            "http_status": resp.status_code,
            "invalid_api_key": invalid_key,
            "error_preview": body_preview if not ok else "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "probe_ok": False,
            "http_status": None,
            "invalid_api_key": False,
            "error_preview": str(exc)[:120],
        }


def build_authenticated_rpc_url(config: dict[str, Any], *, env: dict[str, str] | None = None) -> str:
    """Build the actual RPC URL including Helius api-key when needed. Do not log/print."""
    source = env if env is not None else os.environ
    marker = config.get("rpc_url_for_client") or ""
    if marker == "HELIUS_API_KEY_AUTH":
        key = str(source.get("HELIUS_API_KEY") or "").strip()
        return f"https://mainnet.helius-rpc.com/?api-key={key}"
    if marker and "api-key=" not in marker.lower():
        if config.get("helius_configured") and config.get("helius_api_key_present"):
            key = str(source.get("HELIUS_API_KEY") or "").strip()
            if key and "api-key" not in marker.lower():
                sep = "&" if "?" in marker else "?"
                return f"{marker}{sep}api-key={key}"
    return marker


def maybe_fallback_to_public_rpc(
    config: dict[str, Any],
    *,
    allow_public_rpc: bool,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Probe Helius/Solana auth; optionally fall back to public RPC."""
    url = build_authenticated_rpc_url(config, env=env)
    probe = {"probe_ok": False, "skipped": True}
    if not url:
        return config, probe
    probe = probe_rpc_auth(url)
    if probe.get("probe_ok"):
        return config, probe
    if allow_public_rpc and (
        probe.get("invalid_api_key") or config.get("rpc_provider_used") == "HELIUS_RPC"
    ):
        fallback = {
            **config,
            "rpc_provider_used": "SOLANA_PUBLIC_RPC",
            "rpc_configured": True,
            "rpc_url_for_client": "https://api.mainnet-beta.solana.com",
            "helius_auth_failed": True,
            "fallback_reason": "HELIUS_AUTH_FAILED_PUBLIC_RPC_FALLBACK",
        }
        return fallback, {**probe, "fallback_applied": True}
    return config, {**probe, "fallback_applied": False}
