"""
AI-driven behavioral clustering — persistent labels assigned once at token discovery.

AE12-SentimentFix note:
LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY = True

The sticky cluster_registry / cluster_label value is LEGACY only.
It must NOT be treated as authoritative semantic_signal_family.
Use app.ae12_sentimentfix.dual_axis_mapper.map_dual_axis for semantic vs trading axes.
Missing semantic evidence must remain UNKNOWN (never default to OPPORTUNISTIC_SPECULATIVE
as a semantic family). OPPORTUNISTIC_SPECULATIVE may inform trading_opportunity_state only.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from ..models import MarketState, TokenMetadata

log = logging.getLogger("features")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CLUSTER_REGISTRY_PATH = DATA_DIR / "cluster_registry.json"

# AE12-SentimentFix: sticky cluster is legacy / audit comparison only.
LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY = True


class ClusterLabel(str, Enum):
    SOCIALLY_MOTIVATED = "SOCIALLY_MOTIVATED"
    OPPORTUNISTIC_SPECULATIVE = "OPPORTUNISTIC_SPECULATIVE"


def _load_registry() -> dict[str, dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CLUSTER_REGISTRY_PATH.exists():
        return {}
    with open(CLUSTER_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict[str, dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def get_persisted_cluster(contract_address: str) -> ClusterLabel | None:
    """Return immutable cluster label if this token was already classified."""
    entry = _load_registry().get(contract_address.strip())
    if not entry:
        return None
    raw = entry.get("cluster_label", "")
    try:
        return ClusterLabel(raw)
    except ValueError:
        return None


def persist_cluster(
    contract_address: str,
    label: ClusterLabel,
    *,
    symbol: str,
    name: str,
    reasoning: str = "",
) -> ClusterLabel:
    registry = _load_registry()
    registry[contract_address.strip()] = {
        "cluster_label": label.value,
        "symbol": symbol,
        "name": name,
        "reasoning": reasoning,
        "assigned_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    _save_registry(registry)
    return label


async def resolve_cluster_label(
    token: TokenMetadata,
    pair: dict[str, Any] | None = None,
) -> ClusterLabel:
    """
    Persistent LEGACY cluster label — evaluated once per contract_address via Gemini 2.5 Flash.

    AE12-SentimentFix: return value is legacy_cluster_label / not semantic_signal_family authority.
    Do not use this sticky label as final semantic taxonomy.
    """
    existing = get_persisted_cluster(token.contract_address)
    if existing is not None:
        return existing

    from ..models.predictor import classify_token_cluster

    base = (pair or {}).get("baseToken") or {}
    label, reasoning = await classify_token_cluster(
        symbol=token.symbol,
        name=token.name or base.get("name", token.symbol),
        network=token.network.value,
        contract_address=token.contract_address,
        description=base.get("name", ""),
    )
    persist_cluster(
        token.contract_address,
        label,
        symbol=token.symbol,
        name=token.name,
        reasoning=reasoning,
    )
    log.info("AI cluster %s → %s (%s)", token.symbol, label.value, reasoning[:80])
    return label


def legacy_cluster_as_dual_axis_seed(
    *,
    cluster_label: str | ClusterLabel | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Seed dict for dual-axis mapper; sticky cluster is legacy only."""
    raw = cluster_label.value if isinstance(cluster_label, ClusterLabel) else cluster_label
    row = {"legacy_cluster_label": raw, "cluster_label": raw}
    if extra:
        row.update(extra)
    return row


def build_feature_row(
    pair: dict[str, Any],
    state: MarketState,
    cluster_label: ClusterLabel | str,
    whale_score: float,
) -> dict[str, Any]:
    """14 live market metrics + persistent cluster for LLM inference."""
    if isinstance(cluster_label, ClusterLabel):
        cluster_str = cluster_label.value
    else:
        cluster_str = str(cluster_label)
    txns = (pair.get("txns") or {}).get("h24") or {}
    return {
        "token_contract_address": state.contract_address,
        "symbol": (pair.get("baseToken") or {}).get("symbol", "?"),
        "network": (pair.get("chainId") or "unknown").lower(),
        "price_usd": state.price_usd,
        "liquidity_usd": state.liquidity_usd,
        "volume_24h": state.volume_24h,
        "price_change_24h": state.price_change_24h,
        "price_change_1h": state.price_change_1h,
        "buy_ratio": round(state.buy_ratio, 4),
        "vol_to_liq": round(state.vol_to_liq, 4),
        "whale_score": whale_score,
        "txns_buys_24h": int(txns.get("buys") or 0),
        "txns_sells_24h": int(txns.get("sells") or 0),
        "cluster_label": cluster_str,
    }


def list_cluster_registry() -> dict[str, dict[str, Any]]:
    return _load_registry()
