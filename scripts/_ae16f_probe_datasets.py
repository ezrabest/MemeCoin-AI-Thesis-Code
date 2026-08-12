import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

root = Path(r"E:/Projects/Final Project/memecoin_trader")

def inspect_parquet(rel):
    p = root / rel
    pf = pq.ParquetFile(p)
    cols = pf.schema_arrow.names
    rows = pf.metadata.num_rows
    return {"path": rel.replace("\\","/"), "size_bytes": p.stat().st_size, "row_count": rows,
            "column_count": len(cols), "columns_first30": cols[:30],
            "has_target_net_profitable": "target_net_profitable" in cols,
            "target_like_columns": [c for c in cols if c == "target" or c.startswith("target_") or c.endswith("_target")]}

samples = [
    "data/training/manual_verified_datasets_clean_for_model/LIQ_5K_HIGH_ACTIVITY_x2_1h_CLEAN_MODEL_INPUT.parquet",
    "data/training/manual_verified_datasets_direct_target_v1/LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
    "data/training/manual_verified_results/phase_e3_direct_targets_v1/phase_e3_direct_target_audit_rows.parquet",
]
out = [inspect_parquet(r) for r in samples]

# enumerate representative datasets under key dirs
key_dirs = [
    "data/training/manual_verified_datasets_clean_for_model",
    "data/training/manual_verified_datasets_direct_target_v1",
    "data/training/manual_verified_results/phase_e3_direct_targets_v1",
]
catalog = []
for drel in key_dirs:
    d = root / drel
    if not d.exists():
        continue
    for p in sorted(d.glob("*.parquet")):
        rel = str(p.relative_to(root)).replace("\\","/")
        pf = pq.ParquetFile(p)
        cols = pf.schema_arrow.names
        catalog.append({
            "path": rel,
            "size_bytes": p.stat().st_size,
            "row_count": pf.metadata.num_rows,
            "has_target_net_profitable": "target_net_profitable" in cols,
            "target_like_columns": [c for c in cols if c == "target" or c.startswith("target_") or c.endswith("_target")][:10],
            "columns_first30": cols[:30],
        })

# CF overlap with historical columns (use CLEAN_MODEL_INPUT 1h)
cf_fields = [
    "price_usd","liquidity_usd","volume_24h","fdv",
    "txns_buys","txns_sells","price_change_h1","price_change_h6","price_change_h24",
    "buy_ratio","volume_to_liquidity_ratio","chain",
]
hist_cols = pq.ParquetFile(root / samples[0]).schema_arrow.names
# map CF aliases
aliases = {
    "txns_buys": ["txns_buys","txns_h24_buys","txns_buys_24h","txns_h1_buys"],
    "txns_sells": ["txns_sells","txns_h24_sells","txns_sells_24h","txns_h1_sells"],
    "price_change_h1": ["price_change_h1","price_change_1h"],
    "price_change_h6": ["price_change_h6","price_change_6h"],
    "price_change_h24": ["price_change_h24","price_change_24h"],
}
overlap = {}
for f in cf_fields:
    if f in hist_cols:
        overlap[f] = {"exact": f}
    elif f in aliases:
        found = [a for a in aliases[f] if a in hist_cols]
        overlap[f] = {"aliases_found": found} if found else {"missing": True}
    else:
        overlap[f] = {"missing": f not in hist_cols}

ae16e = pd.read_csv(root / "data/ae16e_clean_forward_rows_used.csv")
ae16e_cols = list(ae16e.columns)
ae16e_overlap = {}
for f in cf_fields:
    if f in ae16e_cols:
        ae16e_overlap[f] = ae16e_cols.index(f)
    elif f in aliases:
        found = [a for a in aliases[f] if a in ae16e_cols]
        ae16e_overlap[f] = found

print(json.dumps({
    "samples": out,
    "catalog_count": len(catalog),
    "catalog": catalog,
    "cf_vs_clean_model_input_1h": overlap,
    "cf_vs_ae16e_rows": ae16e_overlap,
    "clean_model_input_1h_all_columns_count": len(hist_cols),
}, indent=2))
