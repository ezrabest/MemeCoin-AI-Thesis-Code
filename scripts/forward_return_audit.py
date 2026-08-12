import sqlite3, json, bisect, statistics
from collections import defaultdict, Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import generate_signal, detect_whale_alert

DB = "data/trader.db"
HORIZONS_MIN = [15, 60, 240, 1440]
BEARISH_ALERTS = {"LARGE_SELL", "DISTRIBUTION"}
BULLISH_ALERTS = {"LARGE_BUY", "ACCUMULATION", "PUMP_SIGNAL"}

ROUND_TRIP_FEE_PCT = 0.03  # 1.5% buy + 1.5% sell. Adjust if your fee model differs.

def ts_to_key(ts):
    # ISO strings with same UTC format sort lexicographically well enough here.
    return str(ts)

def safe_float(x, default=0.0):
    try:
        return float(x or default)
    except Exception:
        return default

def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {
            "n": 0,
            "win_rate_net": None,
            "avg_net": None,
            "median_net": None,
            "p10_net": None,
            "p90_net": None,
        }
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    def pct(p):
        idx = int(round((n - 1) * p))
        return vals_sorted[max(0, min(n - 1, idx))]
    return {
        "n": n,
        "win_rate_net": round(sum(1 for v in vals if v > 0) / n, 4),
        "avg_net": round(sum(vals) / n, 4),
        "median_net": round(statistics.median(vals), 4),
        "p10_net": round(pct(0.10), 4),
        "p90_net": round(pct(0.90), 4),
    }

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

print("Loading snapshots...")
cur.execute("""
SELECT 
    ms.id,
    ms.timestamp,
    ms.coin_id,
    c.symbol AS symbol,
    ms.chain,
    ms.pair_address,
    ms.price,
    ms.liquidity,
    ms.volume_24h,
    ms.txns_buys,
    ms.txns_sells,
    ms.price_change_h1,
    ms.price_change_h24,
    ms.whale_score,
    ms.buy_ratio,
    ms.filter_status
FROM market_snapshots ms
LEFT JOIN coins c ON c.id = ms.coin_id
WHERE ms.price IS NOT NULL
  AND ms.price > 0
  AND ms.pair_address IS NOT NULL
ORDER BY ms.chain, ms.pair_address, ms.timestamp
""")

rows = [dict(r) for r in cur.fetchall()]
print("rows loaded:", len(rows))

# Build price history per exact pair identity.
hist = defaultdict(list)
for r in rows:
    key = (r["chain"], r["pair_address"])
    hist[key].append((ts_to_key(r["timestamp"]), safe_float(r["price"])))

hist_ts = {}
hist_prices = {}
for key, arr in hist.items():
    hist_ts[key] = [x[0] for x in arr]
    hist_prices[key] = [x[1] for x in arr]

def future_return(row, horizon_min):
    key = (row["chain"], row["pair_address"])
    ts_list = hist_ts.get(key)
    prices = hist_prices.get(key)
    if not ts_list:
        return None

    # SQLite datetime avoids parsing timezone complexity.
    cur2 = con.execute(
        "SELECT datetime(?, '+' || ? || ' minutes') AS target_ts",
        (row["timestamp"], horizon_min),
    )
    target = cur2.fetchone()["target_ts"].replace(" ", "T")
    # timestamps contain +00:00; target does not. Prefix compare still usually ok by date/time.
    idx = bisect.bisect_left(ts_list, target)
    if idx >= len(ts_list):
        return None

    entry = safe_float(row["price"])
    exitp = prices[idx]
    if entry <= 0 or exitp <= 0:
        return None

    gross = (exitp / entry) - 1.0
    net = gross - ROUND_TRIP_FEE_PCT
    return net

def build_pair(row):
    return {
        "priceUsd": row["price"],
        "liquidity": {"usd": row["liquidity"] or 0},
        "volume": {
            "h24": row["volume_24h"] or 0,
            "h1": 0,
        },
        "txns": {
            "h24": {
                "buys": row["txns_buys"] or 0,
                "sells": row["txns_sells"] or 0,
            }
        },
        "priceChange": {
            "h1": row["price_change_h1"] or 0,
            "h24": row["price_change_h24"] or 0,
        },
    }

groups = defaultdict(lambda: defaultdict(list))
counts = Counter()
examples = defaultdict(list)

print("Evaluating decisions...")
for i, r in enumerate(rows, start=1):
    if r.get("filter_status") != "passed":
        continue

    pair = build_pair(r)
    whale_score = safe_float(r["whale_score"])
    sig = generate_signal(pair, whale_score)
    alert = detect_whale_alert(pair, whale_score)

    action = sig["action"]
    alert_type = alert["alert_type"] if alert else None

    if action == "BUY" and alert_type in BEARISH_ALERTS:
        group = "CONFLICT_engine_BUY_bearish_alert"
    elif action == "BUY" and alert_type in BULLISH_ALERTS:
        group = "ALIGNED_engine_BUY_bullish_alert"
    elif action == "BUY" and not alert_type:
        group = "ENGINE_BUY_no_alert"
    elif action != "BUY" and alert_type in BEARISH_ALERTS:
        group = "BEARISH_ALERT_no_engine_BUY"
    elif action != "BUY" and alert_type in BULLISH_ALERTS:
        group = "BULLISH_ALERT_no_engine_BUY"
    else:
        # possible demo/aggressive near miss
        prob = safe_float(sig["probability_up"])
        br = safe_float(r["buy_ratio"])
        liq = safe_float(r["liquidity"])
        if prob >= 0.75 and br >= 0.60 and liq >= 5000:
            group = "STRONG_NEARMISS_no_alert_no_buy"
        else:
            continue

    counts[group] += 1

    for h in HORIZONS_MIN:
        ret = future_return(r, h)
        groups[group][f"net_return_{h}m"].append(ret)

    if len(examples[group]) < 10:
        ex = {
            "timestamp": r["timestamp"],
            "symbol": r["symbol"],
            "coin_id": r["coin_id"],
            "pair_address": r["pair_address"],
            "engine_action": action,
            "probability_up": sig["probability_up"],
            "alert_type": alert_type,
            "liquidity": r["liquidity"],
            "volume_24h": r["volume_24h"],
            "buy_ratio": r["buy_ratio"],
            "whale_score": r["whale_score"],
            "price_change_h1": r["price_change_h1"],
            "price_change_h24": r["price_change_h24"],
        }
        for h in HORIZONS_MIN:
            ex[f"net_return_{h}m"] = future_return(r, h)
        examples[group].append(ex)

    if i % 50000 == 0:
        print("processed", i)

print("\nGROUP COUNTS:")
print(json.dumps(dict(counts), indent=2, ensure_ascii=False))

print("\nFORWARD RETURN SUMMARY, NET OF ROUND-TRIP FEES:")
summary = {}
for group, metrics in groups.items():
    summary[group] = {}
    for h in HORIZONS_MIN:
        summary[group][f"{h}m"] = summarize(metrics[f"net_return_{h}m"])
print(json.dumps(summary, indent=2, ensure_ascii=False))

print("\nEXAMPLES:")
for group, exs in examples.items():
    print("\n" + "="*100)
    print(group)
    for ex in exs:
        print(json.dumps(ex, indent=2, ensure_ascii=False))

con.close()
