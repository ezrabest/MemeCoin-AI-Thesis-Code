#!/usr/bin/env python3
"""AE13C frontend shell hotfix — endpoint + static smoke checks (paper/demo only)."""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "audits" / "ae13c_frontend_shell_smoke_latest"

ENDPOINTS = [
    "/api/ae13b/demo-bot/status",
    "/api/ae13b/demo-bot/events",
    "/api/ae13b/presets",
    "/api/ae13b/portfolio",
    "/api/ae13b/opportunities?limit=10",
    "/api/ae13b/semantic-registry",
    "/api/ae13b/live-market?limit=10",
    "/api/ae13b/rss-sentiment?limit=5",
    "/api/ae13b/provider-status",
    "/api/ae13b/ai-assistant-status",
    "/api/ae13b/navigation",
    "/api/watchlist",
    "/api/settings/effective",
]


def fetch(path: str, timeout: float = 12.0) -> dict:
    url = BASE.rstrip("/") + path
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return {
                    "path": path,
                    "http_status": resp.status,
                    "ok": False,
                    "elapsed_ms": elapsed,
                    "error": "invalid_json",
                    "body_preview": body[:200],
                }
            return {
                "path": path,
                "http_status": resp.status,
                "ok": bool(data.get("ok", True)) if isinstance(data, dict) else True,
                "elapsed_ms": elapsed,
                "error": None,
                "status_field": data.get("status") if isinstance(data, dict) else None,
                "keys": list(data.keys())[:20] if isinstance(data, dict) else [],
            }
    except Exception as exc:  # noqa: BLE001
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "path": path,
            "http_status": None,
            "ok": False,
            "elapsed_ms": elapsed,
            "error": str(exc)[:300],
            "status_field": None,
            "keys": [],
        }


def check_static() -> dict:
    js_path = ROOT / "static" / "product_demo.js"
    html_path = ROOT / "static" / "index.html"
    css_path = ROOT / "static" / "product_demo.css"
    js = js_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    bad_mix = False
    for line in js.splitlines():
        stripped = line.strip()
        if "??" in stripped and "||" in stripped:
            # Broken form: a ?? b || c without grouping around RHS of ??
            if re.search(r"\?\?\s*[^(].*\|\|", stripped) and "?? (" not in stripped and "?? ((" not in stripped:
                bad_mix = True
                break

    nav_block = re.search(r'<nav class="tabs nav-product"[^>]*>([\s\S]*?)</nav>', html)
    nav_html = nav_block.group(1) if nav_block else ""
    return {
        "product_demo_js_exists": js_path.exists(),
        "has_ViewSwitcher": "ViewSwitcher" in js,
        "has_DataLoader": "DataLoader" in js,
        "has_safeFetchJson": "safeFetchJson" in js,
        "has_Promise_allSettled": "Promise.allSettled" in js,
        "unguarded_Promise_all_in_product_demo": bool(re.search(r"await\s+Promise\.all\s*\(", js)),
        "fatal_nullish_or_mix_present": bad_mix,
        "nav_sticky_clickable_css": ("z-index: 60" in html) or ("z-index: 60" in css),
        "header_not_covering_nav": "do not sticky-cover the tab bar" in html or "position: relative; z-index: 40" in html,
        "nav_tabs": re.findall(r'data-tab="([^"]+)"', nav_html),
        "ae_phase_in_primary_nav": bool(re.search(r"AE1[23]", nav_html)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "audits").mkdir(exist_ok=True)
    (OUT / "tests").mkdir(exist_ok=True)

    static = check_static()
    rows = [fetch(p) for p in ENDPOINTS]
    hang = [r for r in rows if (r.get("elapsed_ms") or 0) > 11000]
    fail = [r for r in rows if not r.get("ok") or r.get("http_status") not in (200,)]

    csv_path = OUT / "data" / "ae13c_ui_endpoint_smoke_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "http_status", "ok", "elapsed_ms", "error", "status_field"],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    snapshot = {
        "at": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "static": static,
        "endpoints": rows,
        "failures": fail,
        "hangs": hang,
        "all_endpoints_ok": len(fail) == 0 and len(hang) == 0,
        "shell_js_ok": (
            static.get("has_ViewSwitcher")
            and static.get("has_DataLoader")
            and static.get("has_safeFetchJson")
            and not static.get("fatal_nullish_or_mix_present")
            and not static.get("unguarded_Promise_all_in_product_demo")
            and not static.get("ae_phase_in_primary_nav")
        ),
    }
    (OUT / "data" / "ae13c_backend_endpoint_status_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_endpoints_ok": snapshot["all_endpoints_ok"],
                "shell_js_ok": snapshot["shell_js_ok"],
                "fail_count": len(fail),
                "static": static,
            },
            indent=2,
        )
    )
    return 0 if snapshot["all_endpoints_ok"] and snapshot["shell_js_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
