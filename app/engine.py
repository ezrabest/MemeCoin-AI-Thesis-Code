"""
Whale score, signal generation, and alert detection.
Mirrors the logic from the original dexscreener.ts.
"""
from __future__ import annotations

from typing import TypedDict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _vol(p: dict) -> float:
    return float((p.get("volume") or {}).get("h24") or 0)

def _liq(p: dict) -> float:
    return float((p.get("liquidity") or {}).get("usd") or 0)

def _txns(p: dict) -> dict:
    return (p.get("txns") or {}).get("h24") or {}

def _buys(p: dict) -> int:
    return int(_txns(p).get("buys") or 0)

def _sells(p: dict) -> int:
    return int(_txns(p).get("sells") or 0)

def _pc(p: dict, window: str = "h24") -> float:
    return float((p.get("priceChange") or {}).get(window) or 0)

def _price(p: dict) -> float:
    return float(p.get("priceUsd") or 0)

def _buy_ratio(p: dict) -> float:
    b, s = _buys(p), _sells(p)
    return b / (b + s) if (b + s) > 0 else 0.5


# ── Whale score ───────────────────────────────────────────────────────────────

def compute_whale_score(pair: dict) -> float:
    vol   = _vol(pair)
    liq   = _liq(pair)
    txns  = _buys(pair) + _sells(pair)
    pc_abs = abs(_pc(pair))

    vol_score = min(vol / 1_000_000, 1.0)
    liq_score = min(liq / 500_000,   1.0)
    tx_score  = min(txns / 5_000,    1.0)
    mom_score = min(pc_abs / 50.0,   1.0)
    vtl_score = min(vol / liq, 3.0) / 3.0 if liq > 0 else 0.0

    return round(
        vol_score * 0.25
        + liq_score * 0.20
        + tx_score  * 0.20
        + mom_score * 0.15
        + vtl_score * 0.20,
        4,
    )


# ── Exposed thresholds (Phase 1 observability — values unchanged) ───────────────

SIGNAL_BUY_PROB_THRESHOLD = 0.65
SIGNAL_BUY_WHALE_THRESHOLD = 0.5
SIGNAL_BUY_LIQUIDITY_USD = 25_000
SIGNAL_WATCH_PROB_THRESHOLD = 0.55
SIGNAL_WATCH_WHALE_THRESHOLD = 0.4

WHALE_ALERT_MIN_VOLUME_24H = 5_000
WHALE_ALERT_MIN_WHALE_SCORE = 0.30


def get_signal_thresholds() -> dict[str, float]:
    """Return generate_signal gate thresholds for observability."""
    return {
        "buy_prob_threshold": SIGNAL_BUY_PROB_THRESHOLD,
        "buy_whale_score_threshold": SIGNAL_BUY_WHALE_THRESHOLD,
        "buy_liquidity_usd": float(SIGNAL_BUY_LIQUIDITY_USD),
        "watch_prob_threshold": SIGNAL_WATCH_PROB_THRESHOLD,
        "watch_whale_score_threshold": SIGNAL_WATCH_WHALE_THRESHOLD,
    }


def get_alert_thresholds() -> dict[str, float]:
    """Return detect_whale_alert gate thresholds for observability."""
    return {
        "min_volume_24h": float(WHALE_ALERT_MIN_VOLUME_24H),
        "min_whale_score": WHALE_ALERT_MIN_WHALE_SCORE,
    }


# ── Signal generation ─────────────────────────────────────────────────────────

class Signal(TypedDict):
    action: str
    probability_up: float
    expected_return: float
    downside_risk: float
    explanation: str


def generate_signal(pair: dict, whale_score: float) -> Signal:
    pc24 = _pc(pair)
    pc1h = _pc(pair, "h1")
    vol  = _vol(pair)
    liq  = _liq(pair)
    br   = _buy_ratio(pair)

    score = 0.5
    score += whale_score * 0.25
    if pc1h > 0:           score += 0.05
    if 0 < pc24 < 30:      score += 0.05
    if br > 0.6:           score += 0.10
    if br < 0.4:           score -= 0.10
    if liq > 100_000:      score += 0.05
    if vol > 100_000:      score += 0.05
    if pc24 > 50:          score -= 0.15

    prob_up  = max(0.1, min(0.95, score))
    exp_ret  = min(pc24 / 100, 0.5) if pc24 > 0 else 0.0
    downside = abs(min(pc24, 0)) / 100 + 0.02

    action = "NO_TRADE"
    if (
        prob_up >= SIGNAL_BUY_PROB_THRESHOLD
        and whale_score >= SIGNAL_BUY_WHALE_THRESHOLD
        and liq >= SIGNAL_BUY_LIQUIDITY_USD
    ):
        action = "BUY"
    elif prob_up >= SIGNAL_WATCH_PROB_THRESHOLD or whale_score >= SIGNAL_WATCH_WHALE_THRESHOLD:
        action = "WATCH"

    reasons: list[str] = []
    if whale_score > 0.6:  reasons.append(f"strong whale activity (score: {whale_score:.2f})")
    if br > 0.65:          reasons.append(f"{br*100:.0f}% buy-side pressure")
    if pc1h > 5:           reasons.append(f"+{pc1h:.1f}% in last hour")
    if liq > 100_000:      reasons.append(f"solid liquidity ${liq/1000:.0f}K")
    if pc24 > 20:          reasons.append(f"momentum +{pc24:.1f}% 24h")
    if not reasons:        reasons.append("low whale conviction — no clear signal")

    return Signal(
        action=action,
        probability_up=round(prob_up, 3),
        expected_return=round(exp_ret, 3),
        downside_risk=round(downside, 3),
        explanation="; ".join(reasons),
    )


# ── Whale alert detection ─────────────────────────────────────────────────────

class WhaleAlert(TypedDict):
    alert_type: str
    volume_usd: float
    price_impact_pct: float
    tx_count: int
    description: str


def _net_buy_txns(pair: dict) -> int:
    return _buys(pair) - _sells(pair)


def _bullish_flow(pair: dict) -> bool:
    """Buy-side or upward momentum — not 24h % alone (memecoins often retrace on 24h while pumping 1h)."""
    br = _buy_ratio(pair)
    pc24 = _pc(pair)
    pc1h = _pc(pair, "h1")
    net = _net_buy_txns(pair)
    return (
        br >= 0.58
        or pc1h > 3.0
        or (pc24 > 0 and br >= 0.50)
        or (net > 0 and br >= 0.52)
    )


def _bearish_flow(pair: dict) -> bool:
    br = _buy_ratio(pair)
    pc24 = _pc(pair)
    pc1h = _pc(pair, "h1")
    net = _net_buy_txns(pair)
    return (
        br <= 0.42
        or pc24 < -8.0
        or pc1h < -8.0
        or (net < 0 and br <= 0.45)
    )


def detect_whale_alert(pair: dict, whale_score: float) -> WhaleAlert | None:
    vol24 = _vol(pair)
    vol1h = float((pair.get("volume") or {}).get("h1") or 0)
    liq = _liq(pair)
    buys = _buys(pair)
    sells = _sells(pair)
    txns = buys + sells
    pc24 = _pc(pair)
    pc1h = _pc(pair, "h1")
    br = _buy_ratio(pair)
    vtl = vol24 / liq if liq > 0 else 0.0
    net = _net_buy_txns(pair)
    bullish = _bullish_flow(pair)
    bearish = _bearish_flow(pair)

    if vol24 < WHALE_ALERT_MIN_VOLUME_24H or whale_score < WHALE_ALERT_MIN_WHALE_SCORE:
        return None

    # ── Bullish patterns first (avoid mis-tagging as LARGE_SELL) ─────────────

    if pc24 > 80 and vol24 > 40_000 and bullish:
        return WhaleAlert(
            alert_type="PUMP_SIGNAL",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=buys,
            description=f"+{pc24:.0f}% 24h surge, ${vol24/1000:.0f}K vol, {br*100:.0f}% buy ratio.",
        )

    if pc1h > 12 and vol1h > 15_000 and (bullish or br >= 0.55):
        return WhaleAlert(
            alert_type="PUMP_SIGNAL",
            volume_usd=vol1h,
            price_impact_pct=pc1h,
            tx_count=max(1, buys),
            description=f"+{pc1h:.1f}% in 1h, ${vol1h/1000:.0f}K vol — short-term pump.",
        )

    if br >= 0.65 and vol24 > 25_000 and (net >= 0 or pc1h > 0):
        return WhaleAlert(
            alert_type="ACCUMULATION",
            volume_usd=vol24 * br,
            price_impact_pct=pc24,
            tx_count=buys,
            description=(
                f"{br*100:.0f}% buy pressure, {buys} buys vs {sells} sells "
                f"(net +{net}) — accumulation."
            ),
        )

    if vol24 > 80_000 and bullish and (pc24 > 2 or pc1h > 2 or br >= 0.60):
        return WhaleAlert(
            alert_type="LARGE_BUY",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=buys,
            description=(
                f"${vol24/1000:.0f}K volume surge, buy ratio {br*100:.0f}%, "
                f"24h {pc24:+.1f}% / 1h {pc1h:+.1f}%."
            ),
        )

    if vtl > 8 and vol24 > 40_000 and bullish:
        return WhaleAlert(
            alert_type="LARGE_BUY",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=buys,
            description=(
                f"High VTL {vtl:.1f}x with buy-side flow — "
                f"{br*100:.0f}% buys, net txns {net:+d}."
            ),
        )

    if vol24 > 50_000 and whale_score >= 0.45 and br >= 0.55 and not bearish:
        return WhaleAlert(
            alert_type="LARGE_BUY",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=buys,
            description=f"Volume surge ${vol24/1000:.0f}K with dominant buy flow ({br*100:.0f}%).",
        )

    # ── Bearish patterns ───────────────────────────────────────────────────────

    if br <= 0.38 and vol24 > 20_000 and (net < 0 or pc24 < 0):
        return WhaleAlert(
            alert_type="DISTRIBUTION",
            volume_usd=vol24 * (1 - br),
            price_impact_pct=pc24,
            tx_count=sells,
            description=f"{(1-br)*100:.0f}% sell pressure, {sells} sells — distribution.",
        )

    if vol24 > 150_000 and bearish and pc24 < -4:
        return WhaleAlert(
            alert_type="LARGE_SELL",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=sells,
            description=f"${vol24/1000:.0f}K vol, {pc24:.1f}% 24h — large sell pressure.",
        )

    if vtl > 10 and vol24 > 50_000 and bearish:
        return WhaleAlert(
            alert_type="LARGE_SELL",
            volume_usd=vol24,
            price_impact_pct=pc24,
            tx_count=sells,
            description=f"VTL {vtl:.1f}x with sell-side dominance ({br*100:.0f}% buy ratio).",
        )

    return None
