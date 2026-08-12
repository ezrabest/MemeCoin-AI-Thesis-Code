import ast, json
from pathlib import Path
root = Path(r"E:/Projects/Final Project/memecoin_trader/app/training")
files = ["direct_target_xgb_rf.py","clean_historical_rf.py","direct_target_builder.py","rf_tab_matrix.py","baseline_model.py","direct_target_tabicl.py"]
for fn in files:
    p = root / fn
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    print(fn, "=>", ", ".join(funcs))
