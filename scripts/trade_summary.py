"""One-off trade performance summary."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "data/paper_trades_log.csv").open(encoding="utf-8")))
sells = [r for r in rows if r["side"] == "sell"]
buys = [r for r in rows if r["side"] == "buy"]
buy_map = {r["position_id"]: r for r in buys}

wins = [s for s in sells if float(s["realized_pnl"]) > 0]
losses = [s for s in sells if float(s["realized_pnl"]) < 0]
total_pnl = sum(float(s["realized_pnl"]) for s in sells)
total_fees = sum(float(s["total_fees"]) for s in rows)
gross_pnl = sum(float(s["gross_pnl"]) for s in sells)

by_symbol = defaultdict(lambda: {"pnl": 0, "count": 0, "wins": 0, "losses": 0, "fees": 0, "gross": 0})
for s in sells:
    sym = s["symbol"]
    pnl = float(s["realized_pnl"])
    by_symbol[sym]["pnl"] += pnl
    by_symbol[sym]["count"] += 1
    by_symbol[sym]["fees"] += float(s["total_fees"])
    by_symbol[sym]["gross"] += float(s["gross_pnl"])
    if pnl > 0:
        by_symbol[sym]["wins"] += 1
    else:
        by_symbol[sym]["losses"] += 1

fee_drag = [s for s in sells if float(s["gross_pnl"]) >= 0 and float(s["realized_pnl"]) < 0]
price_loss = [s for s in sells if float(s["gross_pnl"]) < 0]

agent_sells = [s for s in sells if s["reason_code"] == "AGENT_SELL"]
manual_sells = [s for s in sells if s["reason_code"] != "AGENT_SELL"]

pre_dual = [s for s in agent_sells if s["timestamp"] < "2026-05-30T12:57:00"]
post_dual = [s for s in agent_sells if s["timestamp"] >= "2026-05-30T12:57:00"]

large_buys = [b for b in buys if float(b["notional_usd"]) > 500 and b.get("reason_code", "").startswith("AGENT")]


def stats(group):
    if not group:
        return {"n": 0, "pnl": 0, "avg_roi_pct": 0, "win_rate": 0, "avg_pnl": 0}
    pnls = [float(x["realized_pnl"]) for x in group]
    rois = [float(x["net_roi_pct"]) for x in group]
    w = sum(1 for p in pnls if p > 0)
    return {
        "n": len(group),
        "pnl": round(sum(pnls), 2),
        "avg_roi_pct": round(100 * sum(rois) / len(rois), 3),
        "win_rate": round(100 * w / len(group), 1),
        "avg_pnl": round(sum(pnls) / len(group), 2),
    }


state = json.loads((ROOT / "data/paper_state.json").read_text())
open_notional = sum(p["size_usd"] for p in state["open_positions"])

print("=== OVERVIEW ===")
print(f"Starting capital: $10,000")
print(f"Closed round-trips: {len(sells)}")
print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win rate: {100*len(wins)/len(sells):.1f}%")
print(f"Total realized PnL: ${total_pnl:,.2f}")
print(f"Total gross PnL (before fees): ${gross_pnl:,.2f}")
print(f"Total fees (buy+sell legs): ${total_fees:,.2f}")
print(f"Fee-drag losses (flat/up price, net red): {len(fee_drag)} trades, ${sum(float(s['realized_pnl']) for s in fee_drag):,.2f}")
print(f"Price-down losses: {len(price_loss)} trades, ${sum(float(s['realized_pnl']) for s in price_loss):,.2f}")

print("\n=== BY SYMBOL ===")
for sym, d in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"]):
    wr = 100 * d["wins"] / d["count"]
    print(
        f"{sym}: {d['count']} closes | PnL ${d['pnl']:,.2f} | gross ${d['gross']:,.2f} | "
        f"fees ${d['fees']:,.2f} | win rate {wr:.0f}%"
    )

print("\n=== AGENT vs MANUAL (closed) ===")
print(f"AGENT_SELL: {stats(agent_sells)}")
print(f"Other/Manual: {stats(manual_sells)}")

print("\n=== AGENT PHASES ===")
print(f"Pre dual-strategy sizing: {stats(pre_dual)}")
print(f"Post dual-strategy sizing: {stats(post_dual)}")

print("\n=== OVERSIZED EARLY AGENT BUYS (> $500) ===")
print(f"Count: {len(large_buys)}")
if large_buys:
    notionals = [float(b["notional_usd"]) for b in large_buys]
    print(f"Notional range: ${min(notionals):,.0f} – ${max(notionals):,.0f}")

print("\n=== TOP 5 WORST TRADES ===")
for s in sorted(sells, key=lambda x: float(x["realized_pnl"]))[:5]:
    roi = 100 * float(s["net_roi_pct"])
    print(
        f"#{s['position_id']} {s['symbol']} | PnL ${float(s['realized_pnl']):,.2f} | "
        f"gross ${float(s['gross_pnl']):,.2f} | ROI {roi:.2f}% | {s['timestamp'][:19]}"
    )

print("\n=== TOP 5 BEST TRADES ===")
for s in sorted(sells, key=lambda x: float(x["realized_pnl"]), reverse=True)[:5]:
    roi = 100 * float(s["net_roi_pct"])
    print(
        f"#{s['position_id']} {s['symbol']} | PnL ${float(s['realized_pnl']):,.2f} | "
        f"gross ${float(s['gross_pnl']):,.2f} | ROI {roi:.2f}%"
    )

print("\n=== CURRENT WALLET ===")
print(f"Cash: ${state['cash_usd']:,.2f}")
print(f"Open notional: ${open_notional:,.2f} across {len(state['open_positions'])} positions")
print(f"Reported total_net_pnl: ${state['total_net_pnl']:,.2f}")
print(f"Cumulative fees paid: ${state['cumulative_total_fees']:,.2f}")
est_equity = state["cash_usd"] + open_notional
print(f"Est. equity (cash + open cost basis): ${est_equity:,.2f}")
