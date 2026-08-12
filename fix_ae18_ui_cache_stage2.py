from pathlib import Path
import py_compile

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

# ------------------------------------------------------------------
# app/api.py — prevent browser from caching stale index.html and log sanitizer failures
# ------------------------------------------------------------------
api_path = Path("app/api.py")
api = api_path.read_text(encoding="utf-8")
api_path.with_suffix(".py.bak_ae18_ui_cache_stage2").write_text(api, encoding="utf-8")

if 'log = logging.getLogger("api")' not in api:
    api = replace_once(
        api,
        'from typing import Any\n',
        'from typing import Any\nimport logging\n',
        "api logging import",
    )
    api = replace_once(
        api,
        'STATIC_DIR = Path(__file__).parent.parent / "static"\n',
        'STATIC_DIR = Path(__file__).parent.parent / "static"\nlog = logging.getLogger("api")\n',
        "api logger",
    )

api = replace_once(
    api,
    '''    except Exception:
        return data
''',
    '''    except Exception as exc:
        log.warning("response sanitizer failed; returning unsanitized payload: %s", exc, exc_info=True)
        return data
''',
    "_json_ok warning",
)

api = replace_once(
    api,
    '''async def dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))
''',
    '''async def dashboard():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
''',
    "dashboard no-store",
)

api_path.write_text(api, encoding="utf-8")

# ------------------------------------------------------------------
# static/index.html — force fresh product_demo.js load
# ------------------------------------------------------------------
idx_path = Path("static/index.html")
idx = idx_path.read_text(encoding="utf-8")
idx_path.with_suffix(".html.bak_ae18_ui_cache_stage2").write_text(idx, encoding="utf-8")

idx = idx.replace(
    '<script src="/static/product_demo.js"></script>',
    '<script src="/static/product_demo.js?v=ae18-ui-cache-stage2"></script>',
)

idx_path.write_text(idx, encoding="utf-8")

# ------------------------------------------------------------------
# static/product_demo.js — add cache-busting + no-store for GET JSON calls
# ------------------------------------------------------------------
js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_ui_cache_stage2").write_text(js, encoding="utf-8")

if "function cacheBustGetUrl(endpoint)" not in js:
    js = replace_once(
        js,
        '''  /**
   * safeFetchJson - never throws into global boot flow.
''',
        '''  function cacheBustGetUrl(endpoint) {
    const sep = String(endpoint || "").includes("?") ? "&" : "?";
    return String(endpoint || "") + sep + "_ts=" + Date.now();
  }

  /**
   * safeFetchJson - never throws into global boot flow.
''',
        "cacheBustGetUrl insertion",
    )

js = replace_once(
    js,
    '''      const res = await fetch(endpoint, {
        method: "GET",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        signal: ctrl ? ctrl.signal : undefined,
      });
''',
    '''      const res = await fetch(cacheBustGetUrl(endpoint), {
        method: "GET",
        cache: "no-store",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        signal: ctrl ? ctrl.signal : undefined,
      });
''',
    "safeFetchJson no-store cache busting",
)

js_path.write_text(js, encoding="utf-8")

for p in [Path("app/api.py")]:
    py_compile.compile(str(p), doraise=True)
    print(f"OK syntax: {p}")

print("AE18 stage2 UI cache/static patch applied.")
