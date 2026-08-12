"""Smoke test for multimodal architecture."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from app.analytics.features import ClusterLabel, get_persisted_cluster, persist_cluster
from app.execution.paper import PaperTrader, compute_transaction_costs
from app.models import MarketState, WhaleActivity
from app.models.predictor import TradeDecision, get_chat_service


def main() -> None:
    assert len(WhaleActivity.CSV_FIELDS) == 15
    assert WhaleActivity.CSV_FIELDS[-1] == "cluster_label"
    print("OK: 15 CSV columns")

    state = MarketState(
        contract_address="abc",
        price_usd=1.0,
        liquidity_usd=50_000,
        volume_24h=100_000,
        price_change_24h=10,
        price_change_1h=5,
        txns_buys_24h=700,
        txns_sells_24h=300,
    )
    assert get_persisted_cluster(state.contract_address) is None
    persist_cluster(
        state.contract_address,
        ClusterLabel.SOCIALLY_MOTIVATED,
        symbol="TEST",
        name="Test Token",
    )
    cluster = get_persisted_cluster(state.contract_address)
    assert cluster == ClusterLabel.SOCIALLY_MOTIVATED
    print(f"OK: persistent cluster {cluster.value}")

    decision = TradeDecision(decision="HOLD", risk_score=40, confidence=0.5, reasoning="test")
    assert decision.decision in ("BUY", "SELL", "HOLD")
    print("OK: TradeDecision BUY/SELL/HOLD schema")

    costs = compute_transaction_costs(1000, "solana")
    assert costs.swap_fee == 15.0
    print(f"OK: costs swap={costs.swap_fee} priority={costs.priority_fee}")

    trader = PaperTrader()
    coin = {"symbol": "TEST/SOL", "chain": "solana", "price_usd": 0.01}
    pos = trader.open_position(coin, size_usd=100, cluster_label="SOCIALLY_MOTIVATED")
    closed = trader.close_position(pos["id"], cur_price=0.012)
    assert closed is not None
    print(f"OK: net_roi_pct={closed['net_roi_pct']}")

    async def chat_test() -> None:
        svc = get_chat_service()
        reply, _, tools = await svc.chat("What is the average whale score?")
        assert reply
        print(f"OK: chat tools={tools} reply={reply[:60]}...")

    asyncio.run(chat_test())
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
