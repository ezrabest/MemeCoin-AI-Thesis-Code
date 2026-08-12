from pathlib import Path
import re

js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_ui_autorefresh_stage4f").write_text(js, encoding="utf-8")

block = r'''
  // ---------------------------------------------------------------------------
  // AE18 Stage4F: controlled runtime auto-refresh + hard button binding.
  //
  // Observed behavior:
  // - Backend provider refresh updates canonical runtime index correctly.
  // - Console call window.ae18ManualRuntimeRefreshAndReload(...) updates UI.
  // - Browser Ctrl+F5 / passive reload does not perform provider refresh.
  //
  // Required behavior:
  // - Manual Refresh Now buttons must execute the same POST+reload path.
  // - Auto refresh must be controlled, silent, visible-tab-only, and non-overlapping.
  // ---------------------------------------------------------------------------
  const AE18_AUTO_REFRESH_KEY = "AE18_AUTO_RUNTIME_REFRESH_ENABLED";
  const AE18_AUTO_REFRESH_SECONDS_KEY = "AE18_AUTO_RUNTIME_REFRESH_SECONDS";
  let ae18RefreshInFlight = false;
  let ae18AutoRefreshTimer = null;

  function ae18Toast(message, kind, quiet) {
    if (quiet) return;
    try {
      if (typeof toast === "function") toast(message, kind || "info");
      else console.log("[AE18 UI]", message);
    } catch (_) {
      console.log("[AE18 UI]", message);
    }
  }

  function ae18AutoRefreshEnabled() {
    const v = localStorage.getItem(AE18_AUTO_REFRESH_KEY);
    return v !== "false" && v !== "0" && v !== "off";
  }

  function ae18AutoRefreshSeconds() {
    const raw = Number(localStorage.getItem(AE18_AUTO_REFRESH_SECONDS_KEY) || "90");
    if (!Number.isFinite(raw)) return 90;
    return Math.max(30, Math.min(600, Math.floor(raw)));
  }

  async function ae18PostProviderRefreshV2(opts) {
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
      throw new Error("provider refresh failed: " + msg);
    }

    return data;
  }

  async function ae18ReloadViewsV2() {
    const calls = [];

    try {
      if (typeof loadCleanForwardFeedTab === "function") {
        calls.push(loadCleanForwardFeedTab({ manual: false, force: false, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] Clean Forward reload unavailable", err);
    }

    try {
      if (typeof loadLiveMarketTab === "function") {
        calls.push(loadLiveMarketTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] Live Market reload unavailable", err);
    }

    try {
      if (typeof loadMarketOpportunitiesTab === "function") {
        calls.push(loadMarketOpportunitiesTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] Market Opportunities reload unavailable", err);
    }

    try {
      if (typeof loadPortfolioTab === "function") {
        calls.push(loadPortfolioTab({ manual: true, force: true, clearCache: false }));
      }
    } catch (err) {
      console.warn("[AE18 UI] Portfolio reload unavailable", err);
    }

    if (calls.length) {
      await Promise.allSettled(calls);
    }
  }

  window.ae18ManualRuntimeRefreshAndReload = async function ae18ManualRuntimeRefreshAndReload(source, opts) {
    opts = opts || {};
    const quiet = !!opts.quiet;
    const sourceLabel = source || "manual";

    if (ae18RefreshInFlight) {
      ae18Toast("Runtime refresh already in flight; skipping duplicate request.", "info", quiet);
      return { ok: true, status: "skipped_duplicate_refresh" };
    }

    ae18RefreshInFlight = true;
    const started = Date.now();

    try {
      ae18Toast("Refreshing provider data and runtime views...", "info", quiet);

      const data = await ae18PostProviderRefreshV2({
        clearCache: !!opts.clearCache
      });

      await ae18ReloadViewsV2();

      const elapsed = ((Date.now() - started) / 1000).toFixed(1);
      const rowsRefreshed =
        data && data.refresh && data.refresh.rows_refreshed !== undefined
          ? data.refresh.rows_refreshed
          : "unknown";

      ae18Toast(
        "Runtime market refresh completed from " + sourceLabel + " in " + elapsed + "s; rows_refreshed=" + rowsRefreshed,
        "success",
        quiet
      );

      console.log("[AE18 UI] runtime refresh completed", {
        source: sourceLabel,
        elapsed_seconds: elapsed,
        rows_refreshed: rowsRefreshed,
        status: data && data.status,
        refreshed_at: new Date().toISOString()
      });

      return data;
    } catch (err) {
      console.error("[AE18 UI] runtime refresh failed", err);
      ae18Toast("Runtime refresh failed: " + (err && err.message ? err.message : err), "error", quiet);
      throw err;
    } finally {
      ae18RefreshInFlight = false;
    }
  };

  function ae18BindRefreshButtons() {
    const candidates = Array.from(document.querySelectorAll("button,a,[role='button']"));
    for (const el of candidates) {
      const id = String(el.id || "");
      const cls = String(el.className || "");
      const text = String(el.innerText || el.textContent || "").trim();

      const isRefresh =
        id.includes("lmRefresh") ||
        id.includes("cfRefresh") ||
        id.includes("moRefresh") ||
        id.toLowerCase().includes("refresh") ||
        cls.includes("lm-refresh") ||
        cls.includes("cf-refresh") ||
        cls.includes("mo-refresh") ||
        text === "Refresh Now" ||
        text === "Force Provider Refresh" ||
        text.includes("Refresh Now") ||
        text.includes("Provider Refresh");

      if (!isRefresh || el.dataset.ae18RefreshBound === "1") continue;

      el.dataset.ae18RefreshBound = "1";
      el.addEventListener("click", function(event) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();

        const clearCache = text.includes("Force") || id.toLowerCase().includes("force");

        window.ae18ManualRuntimeRefreshAndReload(text || id || "refresh button", {
          clearCache,
          quiet: false
        }).catch(function(err) {
          console.error("[AE18 UI] refresh button failed", err);
        });
      }, true);
    }
  }

  window.lmRefreshNow = function ae18LmRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Market Snapshot Refresh Now", {
      clearCache: false,
      quiet: false
    });
  };

  window.moRefreshNow = function ae18MoRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Market Opportunities Refresh Now", {
      clearCache: false,
      quiet: false
    });
  };

  window.cfRefreshNow = function ae18CfRefreshNow() {
    return window.ae18ManualRuntimeRefreshAndReload("Clean Forward Refresh Now", {
      clearCache: false,
      quiet: false
    });
  };

  window.cfForceProviderRefresh = function ae18CfForceProviderRefresh() {
    return window.ae18ManualRuntimeRefreshAndReload("Force Provider Refresh", {
      clearCache: true,
      quiet: false
    });
  };

  function ae18StartAutoRefresh() {
    if (ae18AutoRefreshTimer) {
      clearInterval(ae18AutoRefreshTimer);
      ae18AutoRefreshTimer = null;
    }

    const seconds = ae18AutoRefreshSeconds();

    ae18AutoRefreshTimer = setInterval(function() {
      if (!ae18AutoRefreshEnabled()) return;
      if (document.visibilityState && document.visibilityState !== "visible") return;
      if (ae18RefreshInFlight) return;

      window.ae18ManualRuntimeRefreshAndReload("silent-auto-refresh", {
        clearCache: false,
        quiet: true
      }).catch(function(err) {
        console.warn("[AE18 UI] silent auto refresh failed", err);
      });
    }, seconds * 1000);

    console.log("[AE18 UI] controlled auto-refresh enabled", {
      seconds,
      visible_tab_only: true,
      quiet: true
    });
  }

  window.ae18EnableAutoRefresh = function ae18EnableAutoRefresh(seconds) {
    localStorage.setItem(AE18_AUTO_REFRESH_KEY, "true");
    if (seconds !== undefined && seconds !== null) {
      localStorage.setItem(AE18_AUTO_REFRESH_SECONDS_KEY, String(seconds));
    }
    ae18StartAutoRefresh();
    return {
      enabled: true,
      seconds: ae18AutoRefreshSeconds()
    };
  };

  window.ae18DisableAutoRefresh = function ae18DisableAutoRefresh() {
    localStorage.setItem(AE18_AUTO_REFRESH_KEY, "false");
    if (ae18AutoRefreshTimer) {
      clearInterval(ae18AutoRefreshTimer);
      ae18AutoRefreshTimer = null;
    }
    console.log("[AE18 UI] controlled auto-refresh disabled");
    return { enabled: false };
  };

  window.ae18AutoRefreshStatus = function ae18AutoRefreshStatus() {
    return {
      enabled: ae18AutoRefreshEnabled(),
      seconds: ae18AutoRefreshSeconds(),
      in_flight: ae18RefreshInFlight,
      timer_active: !!ae18AutoRefreshTimer,
      visibility: document.visibilityState || "unknown"
    };
  };

  document.addEventListener("DOMContentLoaded", function() {
    ae18BindRefreshButtons();
    ae18StartAutoRefresh();
  });

  setTimeout(function() {
    ae18BindRefreshButtons();
    ae18StartAutoRefresh();
  }, 1000);

  const ae18Observer = new MutationObserver(function() {
    ae18BindRefreshButtons();
  });
  try {
    ae18Observer.observe(document.body, { childList: true, subtree: true });
  } catch (_) {}
'''

if "AE18 Stage4F: controlled runtime auto-refresh" not in js:
    marker = "\n})();"
    pos = js.rfind(marker)
    if pos >= 0:
        js = js[:pos] + "\n" + block + "\n" + js[pos:]
    else:
        js += "\n" + block + "\n"

js_path.write_text(js, encoding="utf-8")

html_path = Path("static/index.html")
html = html_path.read_text(encoding="utf-8")
html_path.with_suffix(".html.bak_ae18_ui_autorefresh_stage4f").write_text(html, encoding="utf-8")

html = re.sub(
    r'/static/product_demo\.js(?:\?v=[^"]*)?',
    '/static/product_demo.js?v=ae18-ui-autorefresh-stage4f',
    html
)

html_path.write_text(html, encoding="utf-8")

print("AE18 stage4F controlled UI auto-refresh patch applied.")
print("Changed: static/product_demo.js")
print("Changed: static/index.html")
