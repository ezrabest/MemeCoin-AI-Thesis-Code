"""
Data models — Pydantic v2
Primary key for every token is contract_address, never symbol.
Liquidity gate: tokens below MIN_LIQUIDITY_USD are rejected at model level.
"""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MIN_LIQUIDITY_USD: float = 5_000.0

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LOG_PATH = DATA_DIR / "whale_trades_log.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class Network(str, Enum):
    SOLANA = "solana"
    ETHEREUM = "ethereum"
    BSC = "bsc"
    BASE = "base"
    ARBITRUM = "arbitrum"
    POLYGON = "polygon"
    UNKNOWN = "unknown"


class TradeType(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TokenMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_address: str
    symbol: str
    name: str
    network: Network

    @field_validator("contract_address", mode="before")
    @classmethod
    def strip_address(cls, v: str) -> str:
        return v.strip()


class MarketState(BaseModel):
    contract_address: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    price_change_24h: float = 0.0
    price_change_1h: float = 0.0
    txns_buys_24h: int = 0
    txns_sells_24h: int = 0
    timestamp: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def set_ts(cls, v: dict) -> dict:
        if not v.get("timestamp"):
            v["timestamp"] = datetime.now(timezone.utc)
        return v

    @field_validator("liquidity_usd", mode="before")
    @classmethod
    def liquidity_gate(cls, v: float) -> float:
        v = float(v)
        if v < MIN_LIQUIDITY_USD:
            raise ValueError(
                f"Liquidity ${v:,.0f} below ${MIN_LIQUIDITY_USD:,.0f} minimum — dropped"
            )
        return v

    @field_validator("price_usd", "volume_24h", mode="before")
    @classmethod
    def non_neg(cls, v: float) -> float:
        return max(0.0, float(v))

    @property
    def buy_ratio(self) -> float:
        t = self.txns_buys_24h + self.txns_sells_24h
        return self.txns_buys_24h / t if t > 0 else 0.5

    @property
    def vol_to_liq(self) -> float:
        return self.volume_24h / self.liquidity_usd if self.liquidity_usd > 0 else 0.0


class WhaleActivity(BaseModel):
    CSV_FIELDS: ClassVar[list[str]] = [
        "timestamp",
        "token_contract_address",
        "symbol",
        "network",
        "wallet_address",
        "trade_type",
        "transaction_size_usd",
        "price_usd_at_trade",
        "liquidity_usd_at_trade",
        "volume_24h_at_trade",
        "price_change_24h",
        "buy_ratio",
        "whale_score",
        "alert_type",
        "cluster_label",
    ]

    token_contract_address: str
    symbol: str
    network: Network
    wallet_address: str
    trade_type: TradeType
    transaction_size_usd: float
    price_usd_at_trade: float
    liquidity_usd_at_trade: float
    volume_24h_at_trade: float
    price_change_24h: float
    buy_ratio: float
    whale_score: float
    alert_type: str
    cluster_label: str
    timestamp: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def set_ts(cls, v: dict) -> dict:
        if not v.get("timestamp"):
            v["timestamp"] = datetime.now(timezone.utc)
        return v

    @field_validator("whale_score", "buy_ratio", mode="before")
    @classmethod
    def clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def to_csv_row(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "token_contract_address": self.token_contract_address,
            "symbol": self.symbol,
            "network": self.network.value,
            "wallet_address": self.wallet_address,
            "trade_type": self.trade_type.value,
            "transaction_size_usd": round(self.transaction_size_usd, 6),
            "price_usd_at_trade": round(self.price_usd_at_trade, 10),
            "liquidity_usd_at_trade": round(self.liquidity_usd_at_trade, 2),
            "volume_24h_at_trade": round(self.volume_24h_at_trade, 2),
            "price_change_24h": round(self.price_change_24h, 4),
            "buy_ratio": round(self.buy_ratio, 4),
            "whale_score": round(self.whale_score, 4),
            "alert_type": self.alert_type,
            "cluster_label": self.cluster_label,
        }


async def append_whale_activity(events: list[WhaleActivity]) -> None:
    if not events:
        return

    def _write() -> None:
        exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=WhaleActivity.CSV_FIELDS)
            if not exists:
                w.writeheader()
            for e in events:
                w.writerow(e.to_csv_row())

    await asyncio.get_event_loop().run_in_executor(None, _write)


class TokenRegistry:
    def __init__(self) -> None:
        self._store: dict[str, TokenMetadata] = {}

    def register(self, t: TokenMetadata) -> TokenMetadata:
        if t.contract_address not in self._store:
            self._store[t.contract_address] = t
        return self._store[t.contract_address]

    def get(self, addr: str) -> TokenMetadata | None:
        return self._store.get(addr)

    def __len__(self) -> int:
        return len(self._store)


registry = TokenRegistry()
