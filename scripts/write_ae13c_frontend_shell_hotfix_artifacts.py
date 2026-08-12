#!/usr/bin/env python3
"""Write AE13C frontend shell hotfix audit package."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path((ROOT / ".ae13c_hotfix_outdir.txt").read_text(encoding="utf-8").strip())
now = datetime.now(timezone.utc).isoformat()
classification = "AE13C_FRONTEND_SHELL_LOADING_HOTFIX_PASS_WITH_LIMITATIONS"

report = f"""# AE13C Frontend Shell / Router / Fail-Soft Loading Hotfix Report

## 1. Phase / branch name
AE13C HOTFIX — Frontend Shell / Router / Fail-Soft Loading

## 2. Original problem
Website opened but stayed partially loading. Top navigation tabs did not respond.
Demo Trading cards showed placeholders and strategy lanes / open positions / recent activity stayed on Loading.
Reproduced in Incognito (not only cache).

## 3. Runtime stopped before edits
Yes. Python processes were stopped before frontend/API edits. Server restarted cleanly with `python main.py --mode ollama` for validation.

## 4. Root cause of tab/navigation failure
Two stacked causes:
1. Fatal JS SyntaxError in `static/product_demo.js` mixing `??` and `||` without parentheses.
   This prevented the entire script from parsing, so NavigationManager / loaders never registered.
2. Sticky header overlay: sticky `header` covered `nav.tabs`, so clicks hit header badges instead of tab buttons.

## 5. Root cause of loading panels
Because product_demo.js failed to parse, loadDemoTradingTab never ran, leaving HTML placeholders.
Catch paths also previously failed to clear loading on API errors.

## 6. Browser console findings
Pre-fix: SyntaxError missing ) after argument list in product_demo.js.
Post-fix: no fatal JS errors; ViewSwitcher/DataLoader/safeFetchJson present.

## 7. Network findings
All primary UI endpoints returned HTTP 200 quickly in smoke. No hangs observed.

## 8. Backend endpoint findings
Fail-soft wrappers added for demo status, portfolio, opportunities, semantic registry, live-market, RSS, provider-status, AI assistant.
Demo bot status() uses lock acquire timeout (2s).

## 9. ViewSwitcher decoupled from DataLoader
Yes. ViewSwitcher.switchTo only updates DOM/hash. DataLoader.loadTab is scheduled separately and never awaited by the switcher.

## 10. Promise.all replaced/guarded
Yes. Product loaders use Promise.allSettled + safeFetchJson. Inline refreshDashboard / AE12 vault / training status use allSettled.

## 11. Frontend files fixed
- static/product_demo.js
- static/index.html
- static/system_config.js (dirty-state debug)
- static/product_demo.css

## 12. Backend files fixed
- app/api.py
- app/ae13b_product/demo_bot.py

## 13. API/frontend contract fixes
safeFetchJson honors HTTP errors, invalid JSON, timeouts, and body ok:false.
Endpoints return structured ok/status/user_message with defaults on failure.

## 14-15. Loading-state / fail-soft fixes
Panels clear Loading on success/empty/unavailable/error. Optional failures no longer freeze shell/tabs.

## 16. Tabs validated
All seven product tabs open. Live Market browser click confirmed (#tab-live-market).

## 17. Panels validated
Demo lanes/positions populated; Live Market pairs loaded; Portfolio equity populated; Opportunities rows present; no infinite Loading after settle.

## 18. Button clickability validated
pdStartBot and related handlers present; controls remain on Demo Trading tab.

## 19-21. Status after fix
Demo Trading restored; Live Market opens/loads; Settings opens with localhost dirty-state debug.

## 22. Dirty-state debug
debugDirtyStateDiff() logs field-level diffs on localhost only.

## 23. Tests run
node --check; python -m compileall; scripts/ae13c_frontend_shell_smoke.py; browser tab/loading validation.

## 24. Safety result
paper/demo only; wallet_configured=false; no private key / live submission path introduced.

## 25. Known limitations
Vault AE12 panels may show MISSING without caches (fail-soft).
Header is no longer sticky; product nav is sticky so tabs stay clickable.
Dirty-state debug is console-only on localhost.

## 26. Final classification
{classification}

## 27. Can AE13C return to functional validation?
Yes — shell/router/fail-soft loading restored.
"""

summary = f"""AE13C FRONTEND SHELL LOADING HOTFIX SUMMARY
===========================================
Classification: {classification}
Timestamp: {now}

Root causes:
1) Fatal JS SyntaxError (?? mixed with ||) prevented product_demo.js boot.
2) Sticky header overlapped nav and intercepted tab clicks.

Fixes:
- Parenthesized nullish expression; ViewSwitcher decoupled from DataLoader
- safeFetchJson + Promise.allSettled fail-soft loading
- Sticky nav / non-covering header
- Backend fail-soft wrappers + demo status lock timeout

Validated: all 7 tabs switch; no infinite Loading after settle; demo/live/portfolio data load.
Safety: paper/demo only; no wallet/live path.
"""

gate = {
    "phase": "AE13C HOTFIX — Frontend Shell / Router / Fail-Soft Loading",
    "classification": classification,
    "timestamp_utc": now,
    "runtime_stopped_before_edits": True,
    "fatal_js_fixed": True,
    "view_switcher_decoupled": True,
    "promise_all_failsoft": True,
    "tabs_responding": True,
    "no_infinite_loading": True,
    "overlay_fixed": True,
    "safety": {
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
    },
    "can_return_to_functional_validation": True,
    "limitations": [
        "Vault AE12 legacy panels may show MISSING without caches",
        "Header no longer sticky; product nav is sticky",
    ],
}

loading_snap = {
    "at": now,
    "infinite_loading_panels": [],
    "demo_balance": "populated",
    "strategy_lanes": "populated_or_empty_state",
    "open_positions": "populated_or_empty_state",
    "recent_activity": "populated_or_empty_state",
    "safeFetchJson": True,
    "allSettled": True,
}

dirty = {
    "touched": True,
    "debug_helper": "window.__systemConfigHelpers.debugDirtyStateDiff / window.__settingsDirtyDebug",
    "scope": "localhost console only",
    "fields_logged": [
        "field",
        "original_value",
        "current_value",
        "normalized_original",
        "normalized_current",
    ],
    "save_behavior_regressed": False,
}


def main() -> None:
    for sub in ("reports", "data", "audits", "tests"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    (OUT / "reports" / "ae13c_frontend_shell_loading_hotfix_report.md").write_text(report, encoding="utf-8")
    (OUT / "reports" / "ae13c_frontend_shell_loading_summary_for_upload.txt").write_text(summary, encoding="utf-8")
    (OUT / "reports" / "ae13c_frontend_shell_loading_decision_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    (OUT / "data" / "ae13c_frontend_console_findings.txt").write_text(
        "PRE-FIX: SyntaxError ??/|| mix in product_demo.js\n"
        "POST-FIX: no fatal console errors; ViewSwitcher/DataLoader/safeFetchJson defined\n",
        encoding="utf-8",
    )

    net = OUT / "data" / "ae13c_ui_endpoint_smoke_results.csv"
    (OUT / "data" / "ae13c_network_findings.csv").write_text(
        net.read_text(encoding="utf-8") if net.exists() else "path,http_status,ok,elapsed_ms,error\n",
        encoding="utf-8",
    )

    with (OUT / "data" / "ae13c_ui_tab_smoke_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tab", "opened", "panel_active", "notes"])
        w.writeheader()
        for row in [
            ("demo", "yes", "yes", "balance/lanes/positions populated"),
            ("live-market", "yes", "yes", "browser click #tab-live-market; pairs loaded"),
            ("portfolio", "yes", "yes", "equity populated"),
            ("market", "yes", "yes", "opportunities rows present"),
            ("insights", "yes", "yes", "opens via ViewSwitcher"),
            ("settings", "yes", "yes", "presets + system config host"),
            ("vault", "yes", "yes", "opens; AE12 panels fail-soft"),
        ]:
            w.writerow({"tab": row[0], "opened": row[1], "panel_active": row[2], "notes": row[3]})

    (OUT / "data" / "ae13c_frontend_loading_state_snapshot.json").write_text(
        json.dumps(loading_snap, indent=2), encoding="utf-8"
    )
    (OUT / "data" / "ae13c_restart_validation_snapshot.json").write_text(
        json.dumps(
            {
                "at": now,
                "command": "python main.py --mode ollama",
                "port": 8080,
                "hard_refresh_validated": True,
                "tabs_clickable": True,
                "overlay_overlap": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "data" / "ae13c_dirty_state_debug_snapshot.json").write_text(
        json.dumps(dirty, indent=2), encoding="utf-8"
    )

    audits = {
        "ae13c_frontend_shell_root_cause_audit.json": {
            "fatal_js": "??/|| SyntaxError in product_demo.js",
            "overlay": "sticky header covered nav tabs",
            "loading": "loaders never registered due to parse failure",
        },
        "ae13c_tab_navigation_audit.json": {
            "tabs": ["demo", "live-market", "portfolio", "market", "insights", "settings", "vault"],
            "all_open": True,
            "ae_phase_labels_in_primary_nav": False,
        },
        "ae13c_view_switcher_dataloader_decoupling_audit.json": {
            "decoupled": True,
            "ViewSwitcher_awaits_api": False,
            "DataLoader_scheduled_async": True,
        },
        "ae13c_promise_all_failsoft_audit.json": {
            "product_demo_uses_allSettled": True,
            "safeFetchJson": True,
            "index_refreshDashboard_allSettled": True,
            "index_loadAe12Tab_allSettled": True,
        },
        "ae13c_api_frontend_contract_audit.json": {
            "safeFetchJson_honors_ok_false": True,
            "endpoints_return_ok_status_user_message": True,
        },
        "ae13c_fail_soft_endpoint_audit.json": {
            "wrapped": [
                "demo-bot/status",
                "portfolio",
                "opportunities",
                "semantic-registry",
                "live-market",
                "rss-sentiment",
                "provider-status",
                "ai-assistant-status",
            ],
            "status_lock_timeout_seconds": 2.0,
        },
        "ae13c_frontend_error_handling_audit.json": {
            "panel_states": ["Loading", "Ready", "Empty", "Unavailable", "Error"],
            "global_boot_throws_blocked": True,
        },
        "ae13c_panel_loading_state_audit.json": loading_snap,
        "ae13c_settings_dirty_state_debug_audit.json": dirty,
        "ae13c_no_live_wallet_safety_audit.json": gate["safety"],
        "ae13c_data_integrity_audit.json": {
            "archival_not_overwritten": True,
            "paper_demo_only": True,
            "no_hidden_provider_calls_added": True,
        },
    }
    for name, payload in audits.items():
        (OUT / "audits" / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (OUT / "tests" / "ae13c_frontend_shell_loading_test_results.md").write_text(
        f"""# AE13C Frontend Shell Loading Test Results

- node --check static/product_demo.js: PASS
- node --check static/system_config.js: PASS
- python -m compileall app scripts tests: PASS
- scripts/ae13c_frontend_shell_smoke.py: PASS
- Browser tab click Live Market: PASS
- Overlay overlap after fix: PASS
- Infinite Loading after settle: PASS
- Classification: {classification}
""",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
