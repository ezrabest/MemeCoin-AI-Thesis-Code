# TabICLv2 offline evaluation setup

TabICLv2 runs in a **separate** Python environment (`.venv-tabicl`). It is not a dependency of the main MemeCoin AI Trader app.

## Environment

```powershell
python -m venv .venv-tabicl
.venv-tabicl\Scripts\pip install -r requirements-tabicl.txt
```

The v2 checkpoint `tabicl-classifier-v2-20260212.ckpt` is downloaded automatically on first run (Hugging Face cache), or pass `--model-path` to a local file.

## Run evaluation

From `.venv-tabicl`:

```powershell
.venv-tabicl\Scripts\python.exe scripts/evaluate_tabicl_v2.py
```

Memory-safe smoke run:

```powershell
.venv-tabicl\Scripts\python.exe scripts/evaluate_tabicl_v2.py --max-rows 8000 --max-features 50 --context-size 1024 --batch-size 256
```

## What it does

- Loads only `data/training/model_ready_dataset.parquet`
- Target: `target_profitable_4h` (chronological 70/15/15 split)
- Numeric/bool features only; leakage columns excluded
- Train-only median imputation + scaler
- TabICL context sampled from **train split only** (never full train, never val/test)
- Writes:
  - `data/training/models/tabicl_v2_predictions_validation.parquet`
  - `data/training/models/tabicl_v2_predictions_test.parquet`
  - `data/training/policy_backtests/tabicl_v2_report.json`

## What it does **not** do

- No live collection changes
- No demo/live mode changes
- No real trading
- No SQLite writes
- No Gemini/Ollama/LLM calls
- No RF retraining

## Main app tests

Run from `.venv`:

```powershell
python -m compileall .
python -m unittest discover -s tests
```

Unit tests for TabICL helpers run without GPU and skip TabICL inference when `tabicl` is unavailable.
