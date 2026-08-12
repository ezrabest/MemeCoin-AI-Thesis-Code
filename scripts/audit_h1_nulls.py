import sqlite3
from pathlib import Path

conn = sqlite3.connect(Path(__file__).resolve().parents[1] / "data" / "trader.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
  SELECT
    CASE WHEN price_change_h1 IS NULL THEN 'null' ELSE 'set' END AS h1_status,
    source_query,
    COUNT(*) c
  FROM market_snapshots
  WHERE filter_status='passed'
  GROUP BY h1_status, source_query
""").fetchall()
for r in rows:
    print(dict(r))

first_null = conn.execute("""
  SELECT timestamp, coin_id, source_query, price_change_h1
  FROM market_snapshots
  WHERE filter_status='passed' AND price_change_h1 IS NULL
  ORDER BY id ASC LIMIT 3
""").fetchall()
print('first_null_samples', [dict(r) for r in first_null])
conn.close()
