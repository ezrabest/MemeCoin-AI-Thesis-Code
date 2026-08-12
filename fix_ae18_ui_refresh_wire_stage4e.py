from pathlib import Path
import re

js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_ui_refresh_wire_stage4e").write_text(js, encoding="utf-8")

block = r'''
  // ---------------------------------------------------------------------------
  // AE18 Stage4E: manual-only runtime refresh wiring.
  //
  // Backend/provider refresh is proven to update the canonical runtime index.
  // This block prevents the UI from staying on stale GET/bootstrap data:
  // - no automatic annoying refresh loop
  // - explicit Refresh Now / Force Provider Refresh calls POST provider refresh
  // - then reloads all runtime market surfaces from cache-busted GET endpoints
  // ---------------------------------------------------------------------------
  function ae18UiToast(message, kind) {
    try {
      if (typeof toast === "function") {
        toast(message, kind || "info");
      } else {
        console.log("[AE18 UI]", message);
      }
    } catch (_) {
      console.log("[AE18 UI]", message);
    }
  }

  async function ae18PostRuntimeProviderRefresh(opts) {
    opts = opts || {};
    const body = {
      force: true,
      clear_cache: !!opts.clearCache,
      limit: 100,
      max_rows_per_base_token: 1,
      max_rows_per_symbol: 1,
      max_verify: 80,
      previous_rows: []
    };

    const response = await fetch("/api/clean-forward-feed/refresh?_ts=" + Date.now(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "Pragma": "no-cache"
      },
      cache: "no-store",
      body: JSON.stringify(body)
    });

    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }

    if (!response.ok || !data || data.ok === false) {
      const msg = data && (data.user_message || data.detail || data.status)
        ? (data.user_message || data.detail || data.status)
        : ("HTTP " + response.status);
      throw new Error("runtime provider refresh failed: " + msg);
    }

    return data;
  }

  async function ae18ReloadRuntimeMarketViews() {
    const calls = [];

    try {
      if (typeof loadCleanForwardFeedTab === "function") {
        calls.push(loadCleanForwardFeedTab({ manual: false, force: false, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] loadCleanForwardFeedTab unavailable", err);
    }

    try {
      if (typeof loadLiveMarketTab === "function") {
        calls.push(loadLiveMarketTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] loadLiveMarketTab unavailable", err);
    }

    try {
      if (typeof loadMarketOpportunitiesTab === "function") {
        calls.push(loadMarketOpportunitiesTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] loadMarketOpportunitiesTab unavailable", err);
    }

    try {
      if (typeof loadPortfolioTab === "function") {
        calls.push(loadPortfolioTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] loadPortfolioTab unavailable", err);
    }

    if (calls.length) {
      await Promise.allSettled(calls);
    }
  }

  window.ae18ManualRuntimeRefreshAndReload = async function ae18ManualRuntimeRefreshAndReload(source, opts) {
    opts = opts || {};
    const label = source || "manual";
    ae18UiToast("Refreshing provider data and runtime market views...", "info");

    const started = Date.now();
    const refreshData = await ae18PostRuntimeProviderRefresh({
      clearCache: !!opts.clearCache
    });

    await ae18ReloadRuntimeMarketViews();

    const elapsed = ((Date.now() - started) / 1000).toFixed(1);
    const rowsRefreshed =
      refreshData &&
      refreshData.refresh &&
      refreshData.refresh.rows_refreshed !== undefined
        ? refreshData.refresh.rows_refreshed
        : "unknown";

    ae18UiToast(
      "Runtime market refresh completed from " + label + " in " + elapsed + "s; rows_refreshed=" + rowsRefreshed,
      "success"
    );

    return refreshData;
  };

  // Last assignment wins over older handlers.
  window.lmRefreshNow = function ae18LmRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Market Snapshot Refresh Now", { clearCache: false });
  };

  window.moRefreshNow = function ae18MoRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Market Opportunities Refresh Now", { clearCache: false });
  };

  window.cfRefreshNow = function ae18CfRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Clean Forward Refresh Now", { clearCache: false });
  };

  window.cfForceProviderRefresh = function ae18CfForceProviderRefresh() {
    return window.ae18ManualRuntimeRefreshAndReload("Force Provider Refresh", { clearCache: true });
  };

  // Capture old buttons/listeners that may still call stale GET-only behavior.
  document.addEventListener("click", function ae18RefreshClickCapture(event) {
    const target = event.target && event.target.closest
      ? event.target.closest("button,a,[role='button']")
      : null;
    if (!target) return;

    const id = String(target.id || "");
    const cls = String(target.className || "");
    const text = String(target.innerText || target.textContent || "").trim();

    const looksLikeRuntimeRefresh =
      id.includes("lmRefresh") ||
      id.includes("cfRefresh") ||
      id.includes("moRefresh") ||
      cls.includes("lm-refresh") ||
      cls.includes("cf-refresh") ||
      cls.includes("mo-refresh") ||
      text === "Refresh Now" ||
      text === "Force Provider Refresh" ||
      text.includes("Provider Refresh");

    if (!looksLikeRuntimeRefresh) return;

    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();

    const clearCache = text.includes("Force") || id.toLowerCase().includes("force");
    window.ae18ManualRuntimeRefreshAndReload(text || id || "captured refresh button", {
      clearCache
    }).catch(function(err) {
      console.error("[AE18 UI] manual runtime refresh failed", err);
      ae18UiToast("Runtime refresh failed: " + (err && err.message ? err.message : err), "error");
    });
  }, true);
'''

if "AE18 Stage4E: manual-only runtime refresh wiring" not in js:
    marker = "\n})();"
    pos = js.rfind(marker)
    if pos >= 0:
        js = js[:pos] + "\n" + block + "\n" + js[pos:]
    else:
        js = js + "\n" + block + "\n"

js_path.write_text(js, encoding="utf-8")

html_path = Path("static/index.html")
html = html_path.read_text(encoding="utf-8")
html_path.with_suffix(".html.bak_ae18_ui_refresh_wire_stage4e").write_text(html, encoding="utf-8")

html = re.sub(
    r'/static/product_demo\.js(?:\?v=[^"]*)?',
    '/static/product_demo.js?v=ae18-ui-refresh-wire-stage4e',
    html
)

html_path.write_text(html, encoding="utf-8")

print("AE18 stage4E UI manual refresh wiring patch applied.")
print("Changed: static/product_demo.js")
print("Changed: static/index.html")
