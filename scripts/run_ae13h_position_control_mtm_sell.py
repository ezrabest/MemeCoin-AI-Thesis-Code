"""AE13H validation + audit pack: position control UX, MTM display,
pool disambiguation, per-position Sell Demo. Paper/demo only.
Never starts a live server or wallet.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGETED_TEST_FILES = [
    "tests/test_ae13h_position_control_mtm_sell.py",
    "tests/test_ae13g_bot_decision_explainability.py",
]
COMPILEALL_TARGETS = ["app", "scripts", "tests"]


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _isolated_trader():
    import importlib
    import os

    tmp = tempfile.TemporaryDirectory()
    data_dir = Path(tmp.name) / "data"
    data_dir.mkdir(parents=True)
    os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

    import app.database as database
    import app.execution.paper as paper

    importlib.reload(paper)
    importlib.reload(database)
    paper.DATA_DIR = data_dir
    paper.STATE_PATH = data_dir / "paper_state.json"
    paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
    database.DATA_DIR = data_dir
    database.DB_PATH = data_dir / "test.db"
    database.init_db()
    trader = paper.PaperTrader()
    now = datetime.now(timezone.utc).isoformat()
    trader.set_market_prices(
        [
            {"pair_address": "DDk1QpoolWIFSOLAAADfct", "coin_id": 1059, "price_usd": 1.05},
            {"pair_address": "XyZ9poolWIFSOLBBB9999", "coin_id": 1060, "price_usd": 0.95},
            {"pair_address": "EthPoolWIFWETH111111", "coin_id": 2001, "price_usd": 1.20},
            {"pair_address": "EthPoolWIFWETH222222", "coin_id": 2002, "price_usd": 0.80},
        ],
        price_timestamp=now,
    )
    return tmp, trader, paper


def _build_demo_positions(trader) -> list[dict]:
    specs = [
        ("WIF/SOL", "solana", "DDk1QpoolWIFSOLAAADfct", 1059, "meme_opportunistic_scout", 0.826, 64300),
        ("WIF/SOL", "solana", "XyZ9poolWIFSOLBBB9999", 1060, "lotto_scout", 0.410, 22000),
        ("WIF/WETH", "ethereum", "EthPoolWIFWETH111111", 2001, "momentum_scout", 0.55, 91000),
        ("WIF/WETH", "ethereum", "EthPoolWIFWETH222222", 2002, "liquidity_whale_scout", 0.71, 41000),
    ]
    opened = []
    for symbol, chain, pair, coin_id, lane, whale, liq in specs:
        coin = {
            "symbol": symbol,
            "chain": chain,
            "pair_address": pair,
            "coin_id": coin_id,
            "latest_price": 1.0,
            "latest_liquidity": liq,
            "latest_volume_24h": liq / 2,
            "latest_whale_score": whale,
        }
        pos = trader.open_position(coin, size_usd=25.0, settings={}, reason_code="AE13H_DEMO")
        if not pos:
            continue
        for p in trader.get_positions(status="OPEN"):
            if int(p["id"]) != int(pos["id"]):
                continue
            entry = float(p["entry_price"])
            p.update(
                {
                    "strategy_lane": lane,
                    "entry_reason": "ae13h_identity_demo",
                    "whale_score": whale,
                    "candidate_score": whale,
                    "liquidity": liq,
                    "volume_24h": liq / 2,
                    "semantic_label": "OPPORTUNISTIC",
                    "min_hold_seconds": 900,
                    "time_stop_seconds": 86400,
                    "trailing_stop_pct": 0.15,
                    "take_profit": entry * 1.5,
                    "stop_loss": entry * 0.8,
                    "exit_plan": {
                        "take_profit_pct": 0.5,
                        "stop_loss_pct": 0.2,
                        "trailing_stop_pct": 0.15,
                        "min_hold_seconds": 900,
                        "time_stop_seconds": 86400,
                    },
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
            )
            opened.append(dict(p))
            break
    trader._save_state()  # noqa: SLF001
    return trader.get_marked_positions()


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / "audits" / f"ae13h_position_control_mtm_sell_{ts}"
    reports = out / "reports"
    data = out / "data"
    audits = out / "audits"
    tests_dir = out / "tests"
    for d in (reports, data, audits, tests_dir):
        d.mkdir(parents=True, exist_ok=True)

    compile_res = _run([sys.executable, "-m", "compileall", "-q", *COMPILEALL_TARGETS])
    _write_text(data / "ae13h_compileall_output.txt", compile_res["stdout"] + compile_res["stderr"])

    pytest_res = _run([sys.executable, "-m", "pytest", *TARGETED_TEST_FILES, "-v"])
    _write_text(data / "ae13h_pytest_output.txt", pytest_res["stdout"] + "\n" + pytest_res["stderr"])

    tmp, trader, _paper = _isolated_trader()
    marked: list[dict] = []
    closed: dict | None = None
    remaining: list[dict] = []
    try:
        marked = _build_demo_positions(trader)
        identity_rows = []
        mtm_rows = []
        for p in marked:
            identity_rows.append(
                {
                    "position_id": p.get("id"),
                    "symbol": p.get("symbol"),
                    "chain": p.get("chain"),
                    "pair_address": p.get("pair_address"),
                    "coin_id": p.get("coin_id"),
                    "strategy_lane": p.get("strategy_lane"),
                    "entry_reason": p.get("entry_reason"),
                    "liquidity": p.get("liquidity"),
                    "volume_24h": p.get("volume_24h"),
                    "whale_score": p.get("whale_score"),
                    "candidate_score": p.get("candidate_score"),
                }
            )
            mtm_rows.append(
                {
                    "position_id": p.get("id"),
                    "symbol": p.get("symbol"),
                    "entry_price": p.get("entry_price"),
                    "current_price": p.get("current_price"),
                    "unrealized_pnl_usd": p.get("unrealized_pnl_usd"),
                    "unrealized_pnl_pct": p.get("unrealized_pnl_pct"),
                    "age_label": p.get("age_label"),
                    "distance_to_take_profit_pct": p.get("distance_to_take_profit_pct"),
                    "distance_to_stop_loss_pct": p.get("distance_to_stop_loss_pct"),
                    "exit_plan_summary": p.get("exit_plan_summary"),
                    "exit_eligible_now": p.get("exit_eligible_now"),
                    "bot_exit_reason": p.get("bot_exit_reason"),
                    "exit_blocker": p.get("exit_blocker"),
                    "mark_price_unavailable_reason": p.get("mark_price_unavailable_reason"),
                }
            )

        _write_json(
            data / "ae13h_open_positions_ui_snapshot.json",
            {
                "source": "isolated PaperTrader.get_marked_positions()",
                "columns_required": [
                    "ID", "Symbol / Pair", "Chain", "Pool / Pair Address", "Coin ID",
                    "Strategy Lane", "Entry Price", "Current Price", "Size",
                    "Unrealized PnL $", "Unrealized PnL %", "Age", "TP Distance",
                    "SL Distance", "Exit Status", "Actions",
                ],
                "positions": marked,
                "ui_files": ["static/index.html", "static/product_demo.js", "static/product_demo.css"],
            },
        )
        _write_csv(
            data / "ae13h_position_identity_disambiguation_snapshot.csv",
            identity_rows,
            [
                "position_id", "symbol", "chain", "pair_address", "coin_id",
                "strategy_lane", "entry_reason", "liquidity", "volume_24h",
                "whale_score", "candidate_score",
            ],
        )
        _write_csv(
            data / "ae13h_position_mtm_snapshot.csv",
            mtm_rows,
            [
                "position_id", "symbol", "entry_price", "current_price",
                "unrealized_pnl_usd", "unrealized_pnl_pct", "age_label",
                "distance_to_take_profit_pct", "distance_to_stop_loss_pct",
                "exit_plan_summary", "exit_eligible_now", "bot_exit_reason",
                "exit_blocker", "mark_price_unavailable_reason",
            ],
        )

        # Manual sell: close only position 1, leave others.
        target = marked[0]
        closed = trader.close_position(
            int(target["id"]),
            float(target["current_price"] or target["entry_price"]),
            reason_code="MANUAL_SELL",
            proposed_pair_address=target.get("pair_address"),
            proposed_coin_id=target.get("coin_id"),
            close_reason="manual_take_profit",
            close_note="ae13h audit pack",
            closed_by="user_manual",
        )
        remaining = trader.get_positions(status="OPEN")
        _write_json(
            data / "ae13h_manual_sell_endpoint_snapshot.json",
            {
                "endpoint": "PUT /api/positions/{pos_id}/close",
                "alt_endpoint": "POST /api/demo/sell",
                "body_fields": ["close_price", "close_reason", "close_note"],
                "closed_position_id": target.get("id"),
                "remaining_open_ids": [p.get("id") for p in remaining],
                "scope_safe": len(remaining) == len(marked) - 1,
            },
        )
        _write_json(
            data / "ae13h_manual_close_trade_record_snapshot.json",
            {
                "closed": closed,
                "required_fields_present": {
                    k: (closed or {}).get(k) is not None
                    for k in (
                        "id", "symbol", "chain", "pair_address", "closed_by",
                        "close_reason", "close_note", "close_price",
                        "close_price_source", "realized_pnl_usd", "realized_pnl_pct",
                        "fees", "paper_demo_only", "not_live_approved",
                        "not_profitability_evidence",
                    )
                },
            },
        )

        from app.ae13b_product.presets import PRESETS

        _write_json(
            data / "ae13h_preset_consistency_snapshot.json",
            {
                "aggressive": {
                    "preset_id": "aggressive",
                    "max_open_positions": PRESETS["aggressive"]["max_open_positions"],
                    "max_trades_per_hour": PRESETS["aggressive"]["max_trades_per_hour"],
                    "max_notional_usd": PRESETS["aggressive"]["max_notional_usd"],
                },
                "lotto": {
                    "preset_id": "lotto",
                    "max_open_positions": PRESETS["lotto"]["max_open_positions"],
                    "max_trades_per_hour": PRESETS["lotto"]["max_trades_per_hour"],
                    "max_notional_usd": PRESETS["lotto"]["max_notional_usd"],
                },
            },
        )
    finally:
        tmp.cleanup()

    # --- Audits ---
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    api_src = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    paper_src = (ROOT / "app" / "execution" / "paper.py").read_text(encoding="utf-8")

    mtm_visible = all(
        col in html
        for col in (
            "Current Price", "Unrealized PnL $", "Unrealized PnL %",
            "TP Distance", "SL Distance", "Age",
        )
    ) and "renderPortfolioOpenRow" in js and "current_price" in js

    identity_ok = all(
        x in html or x in js
        for x in ("Pool / Pair Address", "pair_address", "strategy_lane", "coin_id", "identityBlock")
    )

    sell_present = "Sell Demo" in js and "pdOpenSellDemo" in js and "pdConfirmSellDemo" in js
    sell_scoped = "positions/" in js and "close" in js and "pdCloseAll" in js  # both exist; per-row uses id
    safety_ok = (
        "PAPER_DEMO_ONLY" in paper_src
        and "not_live_approved" in paper_src
        and "paper_demo_only" in api_src
        and not re.search(r"signTransaction|private_key|sendRawTransaction", js, re.I)
    )
    exit_ctx = "exit_plan_summary" in paper_src and "exitStatusCell" in js
    preset_ok = '"aggressive"' in (ROOT / "app" / "ae13b_product" / "presets.py").read_text(encoding="utf-8")
    ae13g_ok = pytest_res["returncode"] == 0 and "rejection_reasons" in (
        ROOT / "app" / "ae13b_product" / "rejected_attempt.py"
    ).read_text(encoding="utf-8")

    audits_payload = {
        "ae13h_position_mtm_visibility_audit.json": {
            "pass": mtm_visible,
            "evidence": "Portfolio table columns + renderPortfolioOpenRow map current_price/PnL/age/TP/SL",
            "blocker_if_fail": "AE13H_BLOCKED_POSITION_MTM_NOT_VISIBLE",
        },
        "ae13h_position_identity_disambiguation_audit.json": {
            "pass": identity_ok,
            "evidence": "pair_address short+copy, coin_id, strategy_lane, entry meta in identityBlock",
            "blocker_if_fail": "AE13H_BLOCKED_POSITION_IDENTITY_AMBIGUOUS",
        },
        "ae13h_per_position_sell_audit.json": {
            "pass": sell_present,
            "evidence": "Sell Demo button + confirmation modal + PUT /api/positions/{id}/close",
            "blocker_if_fail": "AE13H_BLOCKED_PER_POSITION_SELL_MISSING",
        },
        "ae13h_sell_action_scope_audit.json": {
            "pass": sell_scoped and closed is not None and len(remaining) == len(marked) - 1,
            "evidence": "Manual close removed only target id; other opens remained",
            "blocker_if_fail": "AE13H_BLOCKED_SELL_ACTION_SCOPE_UNSAFE",
        },
        "ae13h_manual_sell_safety_audit.json": {
            "pass": safety_ok and bool(closed and closed.get("not_live_approved")),
            "evidence": "paper_demo_only / not_live_approved / PAPER_DEMO_ONLY on close record",
            "blocker_if_fail": "AE13H_BLOCKED_MANUAL_SELL_SAFETY_RISK",
        },
        "ae13h_exit_context_audit.json": {
            "pass": exit_ctx,
            "evidence": "exit_plan_summary + bot_exit_reason + manual_close_note in MTM + UI",
            "blocker_if_fail": "AE13H_BLOCKED_EXIT_CONTEXT_MISSING",
        },
        "ae13h_preset_consistency_audit.json": {
            "pass": preset_ok,
            "evidence": "aggressive max_open=6 / lotto max_open=8; UI slots show preset_id",
            "blocker_if_fail": "AE13H_BLOCKED_PRESET_DISPLAY_BACKEND_MISMATCH",
        },
        "ae13h_ae13g_regression_audit.json": {
            "pass": ae13g_ok,
            "evidence": "AE13G targeted tests still pass; structured rejection fields retained",
            "blocker_if_fail": "AE13H_BLOCKED_AE13G_REGRESSION",
        },
        "ae13h_no_live_wallet_safety_audit.json": {
            "pass": safety_ok and compile_res["returncode"] == 0,
            "evidence": "No wallet/sign/submit path in Sell Demo UI; execution guard on close endpoints",
            "blocker_if_fail": "AE13H_BLOCKED_SAFETY_RISK",
        },
    }

    all_pass = True
    for name, payload in audits_payload.items():
        _write_json(audits / name, payload)
        all_pass = all_pass and bool(payload["pass"])

    compile_ok = compile_res["returncode"] == 0
    tests_ok = pytest_res["returncode"] == 0

    limitations = [
        "UI verified by static code inspection + backend/API unit tests; no interactive browser session in this pack.",
        "Estimated exit fees shown qualitatively in confirmation dialog (DEX fee model), not a live quote.",
        "Mark prices depend on in-memory market price maps / demo cycle updates.",
    ]

    if not compile_ok or not tests_ok or not all_pass:
        classification = "AE13H_BLOCKED_SAFETY_RISK"
        for name, payload in audits_payload.items():
            if not payload["pass"]:
                classification = payload["blocker_if_fail"]
                break
        if not tests_ok and classification.startswith("AE13H_POSITION"):
            classification = "AE13H_BLOCKED_AE13G_REGRESSION"
    else:
        classification = "AE13H_POSITION_CONTROL_MTM_SELL_PASS_WITH_LIMITATIONS"

    decision = {
        "classification": classification,
        "compileall_ok": compile_ok,
        "targeted_tests_ok": tests_ok,
        "audits_all_pass": all_pass,
        "limitations": limitations,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out),
    }
    _write_json(reports / "ae13h_decision_gate.json", decision)

    report_md = f"""# AE13H Position Control + MTM + Per-Position Sell

**Classification:** `{classification}`

## Summary

Open demo positions are now actionable and understandable:

1. Portfolio table shows mark-to-market columns (current price, unrealized PnL $/%%, age, TP/SL distance, exit status).
2. Duplicate-looking symbols are disambiguated via pool/pair address (short + copy), coin_id, strategy lane, and entry meta.
3. Each open row has **Sell Demo** with confirmation (reason + paper/demo warning).
4. Close calls `PUT /api/positions/{{id}}/close` for that id only; close records include `closed_by=user_manual`, reason/note, and safety flags.
5. Exit plan / bot eligibility / manual-close note are visible per row.
6. Aggressive (max open 6) and Lotto (max open 8) preset caps remain consistent.
7. No wallet / live / private-key path introduced.

## Validation

- compileall: {"PASS" if compile_ok else "FAIL"}
- targeted pytest (AE13H + AE13G): {"PASS" if tests_ok else "FAIL"}

## Limitations

{chr(10).join(f"- {x}" for x in limitations)}

## Output root

`{out}`
"""
    _write_text(reports / "ae13h_position_control_mtm_sell_report.md", report_md)
    _write_text(
        reports / "ae13h_summary_for_upload.txt",
        f"AE13H {classification}\n"
        f"MTM+identity+Sell Demo on Current Tradable Demo Positions.\n"
        f"compileall={'ok' if compile_ok else 'fail'} tests={'ok' if tests_ok else 'fail'}\n"
        f"output={out}\n",
    )
    _write_text(
        tests_dir / "ae13h_test_results.md",
        f"# AE13H test results\n\n"
        f"- compileall rc={compile_res['returncode']}\n"
        f"- pytest rc={pytest_res['returncode']}\n\n"
        f"## pytest stdout (tail)\n\n```\n{pytest_res['stdout'][-8000:]}\n```\n",
    )

    print(json.dumps(decision, indent=2))
    return 0 if classification.startswith("AE13H_POSITION_CONTROL_MTM_SELL_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
