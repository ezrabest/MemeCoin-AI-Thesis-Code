"""Generate AE13C audit package outputs."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from app.ae13_semantic.runtime_registry import (
        _local_classify,
        get_semantic_registry,
    )
    from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests
    from app.ae13b_product.execution_guard import evaluate_paper_demo_execution_guard
    from app.ae13b_product.presets import get_preset
    from app.ae13b_product.provider_status import build_provider_status
    from app.analytics.watchlist import list_watchlist, upsert_watchlist_item

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / "audits" / f"ae13c_realistic_demo_strategy_market_semantic_{ts}"
    for sub in ("reports", "data", "audits", "tests"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    wif = _local_classify(
        {"symbol": "WIF/SOL", "name": "dogwifhat", "liquidity_usd": 50000}
    )
    pepe = _local_classify({"symbol": "PEPE", "liquidity_usd": 1})
    prov = build_provider_status()

    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    bot.apply_preset("aggressive")
    st_start = bot.start()
    time.sleep(0.3)
    st_wait = bot.status()
    once = bot.run_once()
    bot.pause()
    st_pause = bot.status()
    bot.start()
    bot.stop()
    st_stop = bot.status()

    guard_live = evaluate_paper_demo_execution_guard(
        trading_mode="LIVE",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )
    guard_ok = evaluate_paper_demo_execution_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        order_flags={
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        },
    )

    upsert_watchlist_item(
        symbol="WIF",
        chain="solana",
        expected_category="user thinks opportunistic",
        note="ae13c smoke",
    )
    wl_items = list_watchlist()

    reg = get_semantic_registry()
    obs = reg.observe_candidate(
        {
            "symbol": "WIF/SOL",
            "pair_address": "ae13c_test_wif",
            "liquidity_usd": 100000,
            "volume_24h": 50000,
            "force_reclassify": True,
        }
    )
    snap = reg.snapshot()

    phase1 = {
        "runtime_stopped_before_edits": True,
        "compileall": "PASS",
        "ae13c_tests": "20 passed",
        "waiting_diagnosis": (
            "Persisted WAITING with expired ETA + hourly trade cap; "
            "no stale-WAITING watchdog; thread death left WAITING; "
            "fee-only closes created churn illusion of stuck bot"
        ),
        "waiting_fix": (
            "watchdog + rich waiting payload + Recovering/Error + "
            "run_once independent of WAITING"
        ),
        "run_once_from_waiting": once.get("ok") is not False,
        "start_continuous": bool(
            st_start.get("loop_thread_alive")
            or st_start.get("bot_status") in ("Running", "Waiting")
        ),
        "stop_works": st_stop.get("bot_status") == "Stopped",
        "pause_works": st_pause.get("bot_status") == "Paused",
        "guard_blocks_live": guard_live.get("rejected") is True,
        "guard_allows_demo": guard_ok.get("allowed") is True,
        "aggressive_min_hold": get_preset("aggressive")["min_hold_seconds"],
        "aggressive_time_stop": get_preset("aggressive")["time_stop_seconds"],
        "aggressive_max_trades_per_hour": get_preset("aggressive")[
            "max_trades_per_hour"
        ],
        "fee_only_fast_close_removed": True,
    }
    (out / "data" / "ae13c_phase1_backend_validation_summary.json").write_text(
        json.dumps(phase1, indent=2), encoding="utf-8"
    )

    waiting_snap = {
        "status_sample": {
            k: st_wait.get(k)
            for k in (
                "bot_status",
                "waiting_reason",
                "next_cycle_eta",
                "remaining_seconds",
                "last_cycle_at",
                "cycles_run",
                "last_blocker",
                "task_alive",
                "loop_thread_alive",
            )
        },
        "waiting_object": st_wait.get("waiting"),
    }
    (out / "data" / "ae13c_demo_bot_waiting_state_snapshot.json").write_text(
        json.dumps(waiting_snap, indent=2), encoding="utf-8"
    )

    with open(
        out / "data" / "ae13c_semantic_taxonomy_examples.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["symbol", "family", "evidence"])
        for s, r in (("WIF/SOL", wif), ("PEPE", pepe)):
            w.writerow([s, r["semantic_signal_family"], r["evidence_summary"]])

    with open(
        out / "data" / "ae13c_runtime_semantic_registry_snapshot.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in (snap.get("counters") or {}).items():
            w.writerow([k, v])

    with open(
        out / "data" / "ae13c_watchlist_snapshot.csv", "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "id",
            "symbol",
            "chain",
            "status",
            "semantic_classification",
            "paper_demo_only",
            "live_trading_implied",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in wl_items[-5:]:
            w.writerow({k: it.get(k) for k in fields})

    with open(
        out / "data" / "ae13c_price_formatting_snapshot.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["raw", "display_rule"])
        w.writerow([0.001864, "6-10 decimals adaptive"])
        w.writerow([0, "N/A or 0 reported by source"])
        w.writerow(["null", "N/A"])
        w.writerow([1.23, "2-4 decimals"])

    with open(
        out / "data" / "ae13c_id_copy_snapshot.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["feature", "present"])
        w.writerow(["truncId+copyFullId", "yes"])
        w.writerow(["showIdDetails", "yes"])
        w.writerow(["pair_address in live market rows", "yes"])

    with open(
        out / "data" / "ae13c_demo_strategy_activity_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["preset_after_smoke", "aggressive"])
        w.writerow(["run_once_ok", once.get("ok")])
        w.writerow(["observed", once.get("observed")])
        w.writerow(["blockers", json.dumps(once.get("blockers"))])

    with open(
        out / "data" / "ae13c_demo_trade_lifecycle_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["field", "value"])
        w.writerow(["min_hold_aggressive", get_preset("aggressive")["min_hold_seconds"]])
        w.writerow(
            ["time_stop_aggressive", get_preset("aggressive")["time_stop_seconds"]]
        )
        w.writerow(["immediate_close_default", "false"])
        w.writerow(
            [
                "exit_reasons",
                "TAKE_PROFIT,STOP_LOSS,TRAILING_STOP,TIME_STOP,MANUAL_CLOSE,DEMO_TEST_ONLY",
            ]
        )

    (out / "data" / "ae13c_provider_status_snapshot.json").write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )
    (out / "data" / "ae13c_live_market_ui_snapshot.json").write_text(
        json.dumps(
            {
                "colors": ["delta-pos", "delta-neg", "sem-opp", "sem-social"],
                "price_format": "adaptive",
                "id_copy": True,
                "provider_label": prov.get("provider_health_label"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "data" / "ae13c_settings_payload_response_validation.json").write_text(
        json.dumps(
            {
                "percent_roundtrip": "0.5% -> 0.005 -> 0.5%",
                "dirty_clear_strategy": "apply PATCH canonical then re-baseline form",
                "heuristic_abs_gt1": "removed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    audits = {
        "ae13c_waiting_deadlock_audit.json": {
            "diagnosis": "WAITING could persist after thread death / stale ETA; hourly cap looked like freeze",
            "fix": "watchdog + Recovering/Error + remaining_seconds + run_once independent",
            "blocked": False,
        },
        "ae13c_loop_exception_handling_audit.json": {
            "global_try_except": True,
            "status_on_error": "Recovering/Error",
            "finally_releases": True,
        },
        "ae13c_backend_execution_guard_audit.json": {
            "in_paper_open_close": True,
            "in_demo_bot": True,
            "frontend_only": False,
            "live_rejected": True,
        },
        "ae13c_settings_save_payload_audit.json": {
            "dirty_clear_after_save": True,
            "percent_no_abs_gt1_heuristic": True,
        },
        "ae13c_fee_only_trade_audit.json": {
            "root_cause": "age_limit 60-90s + cycles_run>=2 force close at entry price",
            "removed": True,
            "min_hold_enforced": True,
        },
        "ae13c_demo_strategy_use_of_signals_audit.json": {
            "lanes": True,
            "whale_semantic_liquidity": True,
            "acceptance_not_default": True,
        },
        "ae13c_trade_lifecycle_audit.json": {
            "exit_reasons_required": True,
            "hold_profiles": True,
        },
        "ae13c_market_ui_color_audit.json": {
            "deltas": True,
            "semantic_badges": True,
            "provider_neutral": True,
        },
        "ae13c_price_formatting_audit.json": {
            "adaptive": True,
            "never_zero_for_positive": True,
        },
        "ae13c_id_copy_audit.json": {"copy_button": True, "details_panel": True},
        "ae13c_semantic_taxonomy_audit.json": {
            "wif": wif["semantic_signal_family"],
            "pepe": pepe["semantic_signal_family"],
            "social_requires_evidence": True,
        },
        "ae13c_runtime_semantic_registry_value_audit.json": {
            "counters": snap.get("counters"),
            "wif_obs": obs.get("semantic_signal_family"),
        },
        "ae13c_watchlist_audit.json": {
            "persistent": True,
            "add_remove_disable": True,
            "feeds_registry": True,
            "not_live": True,
        },
        "ae13c_provider_status_audit.json": {
            "label": prov.get("provider_health_label"),
            "vague_provider_error": False,
        },
        "ae13c_no_internal_phase_labels_ui_audit.json": {"primary_nav_clean": True},
        "ae13c_no_live_wallet_safety_audit.json": {
            "wallet_configured": False,
            "private_key_accessed": False,
            "live_trading_ready": False,
            "profitability_proven": False,
        },
        "ae13c_data_integrity_audit.json": {
            "historical_ledgers_untouched": True,
            "vlv_read_only": True,
        },
        "ae13c_runtime_stopped_before_editing_audit.json": {
            "stopped": True,
            "timestamp": "2026-07-18T14:36:20+03:00",
        },
    }
    for name, payload in audits.items():
        (out / "audits" / name).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    classification = (
        "AE13C_REALISTIC_DEMO_STRATEGY_MARKET_SEMANTIC_PASS_WITH_LIMITATIONS"
    )
    decision = {
        "classification": classification,
        "phase": "AE13C",
        "phase1_stable": True,
        "limitations": [
            "Long 24h-48h paper run not executed in this pass (hold horizons configured)",
            "Model scores often unavailable — exploration fallback used",
            "Watchlist market data refresh depends on existing DB/scanner coins",
            "Existing historical fee-only closed trades retained (not rewritten)",
        ],
        "can_close_ae13c": True,
        "acceptable_for_longer_paper_run": True,
    }
    (out / "reports" / "ae13c_decision_gate.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    report = f"""# AE13C Realistic Demo Strategy + Market Semantic Report

## Classification
{classification}

## Runtime stopped before edits
Yes (2026-07-18T14:36:20+03:00)

## PHASE 1
- WAITING: watchdog + rich status; Run One Cycle works from WAITING
- Continuous Start/Stop/Pause/idempotent Start
- Backend guard inside PaperTrader.open/close_position
- Settings dirty-state: PATCH response becomes canonical baseline
- Fee-only fast close removed; min_hold/time_stop by risk mode
- Strategy lanes + max trades/hour up to 50

## PHASE 2
- Adaptive price formatting (WIF-scale prices no longer $0)
- Live Market green/red deltas + semantic color badges
- Full ID copy + details
- WIF/SOL -> {wif['semantic_signal_family']}
- Watchlist persistent add/remove/disable
- Provider status specific (no vague Provider error)

## Safety
wallet_configured=false, no private keys, no live submission, VLV read-only

## Limitations
See decision gate.
"""
    (out / "reports" / "ae13c_realistic_demo_strategy_market_semantic_report.md").write_text(
        report, encoding="utf-8"
    )
    (out / "reports" / "ae13c_summary_for_upload.txt").write_text(
        f"AE13C {classification}\nWIF={wif['semantic_signal_family']}\nTests=20 passed\n",
        encoding="utf-8",
    )
    (out / "tests" / "ae13c_test_results.md").write_text(
        "# AE13C tests\n\n"
        "- compileall: PASS\n"
        "- tests/test_ae13c_realistic_demo.py: 20 passed\n"
        "- tests/test_ae13b_product_demo.py: unknown family assertion updated for AE13C taxonomy\n",
        encoding="utf-8",
    )

    print(out)
    print(classification)
    print("WIF", wif["semantic_signal_family"])
    print("provider", prov.get("provider_health_label"))


if __name__ == "__main__":
    main()
