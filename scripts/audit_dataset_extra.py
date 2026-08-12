import json
import sqlite3
from pathlib import Path

conn = sqlite3.connect(Path(__file__).resolve().parents[1] / "data" / "trader.db")
conn.row_factory = sqlite3.Row

passed = conn.execute(
    "SELECT * FROM market_snapshots WHERE filter_status='passed' ORDER BY id DESC LIMIT 1"
).fetchone()
print("passed_snapshot_example", json.dumps(dict(passed), indent=2))

for col in ["txns_buys", "txns_sells", "price_change_h1", "buy_ratio", "whale_score"]:
    nulls = conn.execute(
        f"SELECT COUNT(*) c FROM market_snapshots WHERE filter_status='passed' AND ({col} IS NULL)"
    ).fetchone()["c"]
    total = conn.execute(
        "SELECT COUNT(*) c FROM market_snapshots WHERE filter_status='passed'"
    ).fetchone()["c"]
    print(f"passed_null_{col}", nulls, "/", total)

scans = conn.execute("SELECT COUNT(DISTINCT scan_id) c FROM pipeline_audit").fetchone()["c"]
print("distinct_scans", scans)
row = conn.execute("SELECT MIN(timestamp) mn, MAX(timestamp) mx FROM pipeline_audit").fetchone()
print("audit_span", dict(row))
conn.close()
