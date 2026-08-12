from pathlib import Path
import py_compile

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

# ------------------------------------------------------------------
# static/product_demo.js
# Fix: Market Snapshot Refresh must perform POST provider refresh,
# not only GET from the cached runtime index.
# ------------------------------------------------------------------
js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_market_refresh_stage4a").write_text(js, encoding="utf-8")

# Increase refresh request limit so provider refresh covers all runtime rows.
js = js.replace(
    '''      limit: 25,
      max_rows_per_base_token: 1,
      max_rows_per_symbol: 1,
      max_verify: 40,
''',
    '''      limit: 100,
      max_rows_per_base_token: 1,
      max_rows_per_symbol: 1,
      max_verify: 80,
''',
    1,
)

if "window.refreshRuntimeIndexFromProvider" not in js:
    js = replace_once(
        js,
        '''  window.cfRefreshNow = function () {
    loadCleanForwardFeedTab({ manual: true, force: true, clearCache: false });
  };
''',
        '''  window.refreshRuntimeIndexFromProvider = async function refreshRuntimeIndexFromProvider(opts) {
    opts = opts || {};
    const body = {
      force: opts.force !== false,
      clear_cache: !!opts.clearCache,
      limit: 100,
      max_rows_per_base_token: 1,
      max_rows_per_symbol: 1,
      max_verify: 80,
      previous_rows: [],
    };
    const res = await apiJson("/api/clean-forward-feed/refresh", "POST", body, 90000);
    const d = (res && res.data) || res || {};
    if (!d || d.ok === false) {
      const err = new Error(refreshErrorMessage(d || {}));
      err.structured = true;
      throw err;
    }
    return d;
  };

  window.reloadAllRuntimeMarketSurfaces = async function reloadAllRuntimeMarketSurfaces() {
    await Promise.allSettled([
      typeof loadCleanForwardFeedTab === "function" ? loadCleanForwardFeedTab({ bootstrap: true }) : Promise.resolve(),
      typeof loadLiveMarketTab === "function" ? loadLiveMarketTab({ manual: true }) : Promise.resolve(),
      typeof loadMarketOpportunitiesTab === "function" ? loadMarketOpportunitiesTab() : Promise.resolve(),
      typeof loadPortfolioTab === "function" ? loadPortfolioTab() : Promise.resolve(),
    ]);
  };

  window.cfRefreshNow = async function () {
    try {
      await refreshRuntimeIndexFromProvider({ force: true, clearCache: false });
      await reloadAllRuntimeMarketSurfaces();
      toast("Provider refresh completed; all market surfaces reloaded.");
    } catch (e) {
      toast("Provider refresh failed - " + String((e && e.message) || e));
    }
  };
''',
        "insert shared provider refresh function before cfRefreshNow",
    )

# Replace Force Provider Refresh with global reload as well.
js = replace_once(
    js,
    '''  window.cfForceProviderRefresh = function () {
    loadCleanForwardFeedTab({ manual: true, force: true, clearCache: true });
  };
''',
    '''  window.cfForceProviderRefresh = async function () {
    try {
      await refreshRuntimeIndexFromProvider({ force: true, clearCache: true });
      await reloadAllRuntimeMarketSurfaces();
      toast("Force provider refresh completed; all market surfaces reloaded.");
    } catch (e) {
      toast("Force provider refresh failed - " + String((e && e.message) || e));
    }
  };
''',
    "replace cfForceProviderRefresh",
)

# Replace Market Snapshot refresh. This is the key bug.
js = replace_once(
    js,
    '''  window.lmRefreshNow = function () {
    loadLiveMarketTab({ manual: true });
  };
''',
    '''  window.lmRefreshNow = async function () {
    try {
      toast("Refreshing provider data...");
      await refreshRuntimeIndexFromProvider({ force: true, clearCache: false });
      await reloadAllRuntimeMarketSurfaces();
      toast("Provider refresh completed; Market Snapshot updated.");
    } catch (e) {
      toast("Provider refresh failed - " + String((e && e.message) || e));
      await loadLiveMarketTab({ manual: true });
    }
  };
''',
    "replace lmRefreshNow with provider refresh",
)

# Add cache buster to index script if not already done.
idx_path = Path("static/index.html")
idx = idx_path.read_text(encoding="utf-8")
idx_path.with_suffix(".html.bak_ae18_market_refresh_stage4a").write_text(idx, encoding="utf-8")
idx = idx.replace(
    "/static/product_demo.js?v=ae18-ui-cache-stage2",
    "/static/product_demo.js?v=ae18-market-refresh-stage4a",
)
idx = idx.replace(
    "/static/product_demo.js\"></script>",
    "/static/product_demo.js?v=ae18-market-refresh-stage4a\"></script>",
)
idx_path.write_text(idx, encoding="utf-8")

js_path.write_text(js, encoding="utf-8")

print("AE18 stage4A market refresh UI patch applied.")
print("Changed: static/product_demo.js")
print("Changed: static/index.html")
