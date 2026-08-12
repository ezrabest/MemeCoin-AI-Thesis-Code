/**
 * Product ViewSwitcher + DataLoader - demo workstation first.
 * AE13C hotfix: ViewSwitcher NEVER waits on API; DataLoader is fail-soft.
 * Research/audit content stays available under the Vault tab.
 */
(function () {
  const PRODUCT_TABS = ["demo", "clean-forward", "live-market", "portfolio", "market", "insights", "settings", "vault"];

  /** ViewSwitcher: DOM tab switching only - never awaits fetch / never throws into click path. */
  const ViewSwitcher = {
    tabs: PRODUCT_TABS.slice(),
    current: "demo",
    lmFilter: "all",
    lmFilterMode: "hide",
    lmAutoRefresh: true,
    lmPinnedOnly: false,
    lmPinnedKeys: new Set(),
    lmSelectedKey: null,
    lmExpandedKey: null,
    lmLastRefreshAt: null,
    lmNextRefreshAt: null,
    lmPollMs: 8000,
    lmRowStore: new Map(),
    switchTo(name, btn) {
      try {
        const id = String(name || "demo");
        this.current = id;
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.remove("active"));
        const panel = document.getElementById("tab-" + id);
        if (panel) panel.classList.add("active");
        if (btn) btn.classList.add("active");
        else {
          const match = document.querySelector(`nav.tabs button[data-tab="${id}"]`);
          if (match) match.classList.add("active");
        }
        try {
          if (window.location && typeof window.location.hash === "string") {
            const hash = "#tab-" + id;
            if (window.location.hash !== hash) {
              history.replaceState(null, "", hash);
            }
          }
        } catch (_) { /* hash optional */ }
      } catch (err) {
        console.error("[ViewSwitcher] switchTo failed", err);
      }
    },
  };

  /** DataLoader: panel data only - never blocks navigation. */
  const DataLoader = {
    loadTab(id) {
      try {
        if (id === "demo") {
          Promise.resolve(loadDemoTradingTab()).catch((e) => console.warn("[DataLoader] demo", e));
        } else if (id === "clean-forward") {
          Promise.resolve(loadCleanForwardFeedTab()).catch((e) => console.warn("[DataLoader] clean-forward", e));
        } else if (id === "live-market") {
          Promise.resolve(loadLiveMarketTab()).catch((e) => console.warn("[DataLoader] live-market", e));
        } else if (id === "portfolio") {
          Promise.resolve(loadPortfolioTab()).catch((e) => console.warn("[DataLoader] portfolio", e));
        } else if (id === "market") {
          Promise.resolve(loadMarketOpportunitiesTab()).catch((e) => console.warn("[DataLoader] market", e));
        } else if (id === "insights") {
          Promise.resolve(loadInsightsTab()).catch((e) => console.warn("[DataLoader] insights", e));
        } else if (id === "settings") {
          try {
            const host = document.getElementById("advanced-settings-host");
            const legacy = document.getElementById("tab-settings-legacy");
            if (host && legacy && !host.dataset.mounted) {
              host.appendChild(legacy);
              legacy.classList.add("active");
              legacy.style.display = "block";
              host.dataset.mounted = "1";
            }
          } catch (e) {
            console.warn("[DataLoader] settings mount", e);
          }
          Promise.resolve(loadProductSettingsTab()).catch((e) => console.warn("[DataLoader] settings", e));
          try {
            if (typeof loadSystemConfig === "function") loadSystemConfig();
          } catch (e) {
            console.warn("[DataLoader] loadSystemConfig", e);
          }
        } else if (id === "vault") {
          try { mountVaultPanels(); } catch (e) { console.warn("[DataLoader] vault mount", e); }
          try {
            if (typeof loadAe12Tab === "function") {
              Promise.resolve(loadAe12Tab(false)).catch((e) => console.warn("[DataLoader] ae12", e));
            }
          } catch (e) { console.warn("[DataLoader] ae12 sync", e); }
          try {
            if (typeof loadAnalyticsSummary === "function") loadAnalyticsSummary();
          } catch (e) { console.warn("[DataLoader] analytics", e); }
          try {
            if (typeof loadCharts === "function") loadCharts();
          } catch (e) { console.warn("[DataLoader] charts", e); }
          try {
            if (typeof loadTrainingDatasetStatus === "function") loadTrainingDatasetStatus();
          } catch (e) { console.warn("[DataLoader] training", e); }
          Promise.resolve(loadVaultProviderFlags()).catch((e) => console.warn("[DataLoader] vault flags", e));
        }
      } catch (err) {
        console.error("[DataLoader] loadTab failed", err);
      }
    },
  };

  /**
   * Public NavigationManager: switch view first, then schedule data load.
   * Tab UI updates even if every endpoint fails.
   */
  const NavigationManager = {
    tabs: PRODUCT_TABS.slice(),
    get current() { return ViewSwitcher.current; },
    set current(v) { ViewSwitcher.current = v; },
    get lmFilter() { return ViewSwitcher.lmFilter; },
    set lmFilter(v) { ViewSwitcher.lmFilter = v; },
    switchTo(name, btn) {
      ViewSwitcher.switchTo(name, btn);
      // Fire-and-forget - never awaited by ViewSwitcher
      setTimeout(() => DataLoader.loadTab(ViewSwitcher.current), 0);
    },
  };

  window.ViewSwitcher = ViewSwitcher;
  window.DataLoader = DataLoader;
  window.NavigationManager = NavigationManager;
  window.switchTab = function (name, btn) {
    const map = {
      dashboard: "live-market",
      positions: "portfolio",
      analytics: "vault",
      ae12: "vault",
    };
    NavigationManager.switchTo(map[name] || name, btn);
  };

  function money(v, opts) {
    return formatPrice(v, opts);
  }

  /** Adaptive price formatting - never show $0 for a positive price. */
  function formatPrice(v, opts) {
    opts = opts || {};
    if (v == null || v === "") return "N/A";
    const n = Number(v);
    if (!Number.isFinite(n)) return "N/A";
    if (n === 0) return opts.zeroFromSource ? "0 reported by source" : "N/A";
    const abs = Math.abs(n);
    let text;
    if (abs >= 1) {
      text = "$" + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
    } else if (abs >= 0.01) {
      text = "$" + n.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 });
    } else if (abs >= 0.000001) {
      text = "$" + n.toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 10 });
    } else {
      text = n > 0 ? "< $0.000001" : n.toExponential(2);
    }
    return text;
  }

  function pct(v) {
    if (v == null || v === "") return "-";
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return (n >= 0 ? "+" : "") + n.toFixed(2) + "%";
  }

  function deltaClass(v) {
    const n = Number(v);
    if (!Number.isFinite(n) || n === 0) return "delta-neu";
    return n > 0 ? "delta-pos" : "delta-neg";
  }

  function pnlClass(v) {
    const n = Number(v);
    if (!Number.isFinite(n) || n === 0) return "delta-neu";
    return n > 0 ? "delta-pos" : "delta-neg";
  }

  function dash(v) {
    if (v == null || v === "") return "-";
    return esc(String(v));
  }

  /** Backend stores fractions (0.05 = +5%). */
  function pctFrac(v, digits) {
    if (v == null || v === "") return "-";
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    const d = digits == null ? 2 : digits;
    return (n >= 0 ? "+" : "") + (n * 100).toFixed(d) + "%";
  }

  // FRONTEND_PNL_PERCENT_DISPLAY_CANONICAL_V3
  // Canonical open-position DTOs from targeted_v2 expose PnL percent as percent-points:
  //   -0.321 means -0.321%, not -32.1%.
  // Older frontend helpers still expect fractions:
  //   -0.00321 -> -0.321%.
  // Prefer backend display string when present; otherwise convert targeted_v2 percent-points.
  function canonicalPnlPctFraction(p, raw) {
    const display = p && p.unrealized_pnl_pct_display != null
      ? String(p.unrealized_pnl_pct_display).trim()
      : "";
    if (display && !display.startsWith("N/A") && display.includes("%")) {
      const d = Number(display.replace("%", "").replace(",", "").trim());
      if (Number.isFinite(d)) return d / 100;
    }

    if (raw == null || raw === "") return null;
    const n = Number(raw);
    if (!Number.isFinite(n)) return null;

    if (p && String(p.ui_financial_finalizer_version || "") === "targeted_v2") {
      return n / 100;
    }

    // Legacy fallback only. Ambiguous below |1|, therefore targeted_v2/display must win first.
    return Math.abs(n) > 1 ? n / 100 : n;
  }

  function canonicalPnlPctText(p, raw, digits) {
    const display = p && p.unrealized_pnl_pct_display != null
      ? String(p.unrealized_pnl_pct_display).trim()
      : "";
    if (display && !display.startsWith("N/A") && display.includes("%")) {
      return display;
    }
    const f = canonicalPnlPctFraction(p, raw);
    return f == null ? "-" : pctFrac(f, digits == null ? 1 : digits);
  }


  function fmtPctDistance(v, kind) {
    if (v == null || v === "") return "-";
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    const absPct = Math.abs(n * 100).toFixed(1) + "%";
    if (kind === "tp") {
      if (n > 0) return absPct + " to TP";
      if (n < 0) return "past TP " + absPct;
      return "at TP";
    }
    if (kind === "sl") {
      if (n > 0) return absPct + " above SL";
      if (n < 0) return absPct + " to/past SL";
      return "at SL";
    }
    return (n >= 0 ? "+" : "-") + absPct;
  }

  function shortPool(addr) {
    const s = String(addr || "").trim();
    if (!s) return "-";
    if (s.length <= 12) return s;
    return s.slice(0, 6) + "..." + s.slice(-4);
  }

  function poolCopyCell(addr) {
    const full = String(addr || "").trim();
    if (!full) return "-";
    const safe = esc(full);
    const short = esc(shortPool(full));
    const title = "Copy pair address. This is a pool/pair address, not necessarily the token mint.";
    return `<button type="button" class="id-copy-btn mono" data-copy="${safe}" onclick="copyFullId(this)" title="${esc(title)}">${short}</button>`;
  }

  /** AE13I traffic light dot - colored circle + label, driven by mtm_traffic_light.py output. */
  function trafficLightDot(p) {
    const status = String(p.traffic_light_status || "").toLowerCase();
    const cls = status === "green" ? "tl-green" : status === "yellow" ? "tl-yellow" : "tl-red";
    const label = p.traffic_light_label || (status ? status.toUpperCase() : "Unknown");
    const reason = p.traffic_light_reason || "";
    return `<span class="tl-wrap" title="${esc(reason)}"><span class="tl-dot ${cls}"></span><span class="tl-label">${esc(label)}</span></span>`;
  }

  /** AE13I data trust line - source / address role / price age / tradability, no invented data. */
  function dataTrustLine(p) {
    const bits = [];
    const source = p.current_price_source || p.source_provider;
    if (source) bits.push("Source: " + esc(String(source)));
    const role = p.address_role;
    if (role) bits.push("Role: " + esc(String(role).replace(/_/g, " ")));
    const age = p.price_age_seconds != null ? p.price_age_seconds : p.close_price_age_seconds;
    if (age != null && Number.isFinite(Number(age))) bits.push("Age: " + Math.round(Number(age)) + "s");
    const trad = p.tradability_status;
    if (trad) bits.push(esc(String(trad)));
    if (!bits.length) return "";
    return `<div class="pf-data-trust">${bits.join(" | ")}</div>`;
  }


// BEGIN PORTFOLIO_DISPLAY_STATE_CONSISTENCY_FIX_V1
function normalizePortfolioDisplayStateV1(p) {
  if (!p || typeof p !== "object") return p;

  const num = (v) => {
    if (v === null || v === undefined) return null;
    const txt = String(v).trim();
    if (!txt || ["N/A", "NA", "NONE", "NULL", "UNAVAILABLE"].includes(txt.toUpperCase())) return null;
    const x = Number(txt);
    return Number.isFinite(x) && x > 0 ? x : null;
  };

  const px =
    num(p.current_price) ??
    num(p.current_price_numeric) ??
    num(p.current_price_usd) ??
    num(p.mark_price_usd) ??
    num(p.mark_price) ??
    num(p.latest_price) ??
    num(p.latest_price_usd) ??
    num(p.market_price_usd);

  // 1. Symbol/pair display: if API has p.symbol, never show SYMBOL_PAIR_UNAVAILABLE.
  const symbolFallback =
    (p.symbol && String(p.symbol).trim()) ||
    (p.symbol_pair_display_at_entry && String(p.symbol_pair_display_at_entry).trim()) ||
    (p.pair && String(p.pair).trim()) ||
    "";

  const symbolBad =
    !p.symbol_pair_display ||
    String(p.symbol_pair_display).toUpperCase().includes("UNAVAILABLE") ||
    String(p.symbol_pair_display).toUpperCase().includes("NOT_PRESENT");

  if (symbolFallback && symbolBad) {
    p.symbol_pair_display = symbolFallback;
    p.symbol_resolution_status = "SYMBOL_PAIR_RESOLVED_FROM_POSITION_SYMBOL";
    p.symbol_pair_display_status = "SYMBOL_PAIR_RESOLVED_FROM_POSITION_SYMBOL";
    p.symbol_pair_unavailable_reason = "";
    p.symbol_resolution_failure_reason = "";
  }

  // 2. Price/mark display-state consistency:
  // If a numeric current/mark price exists, stale PRICE_UNAVAILABLE flags are display bugs.
  if (px !== null) {
    p.current_price = px;
    p.current_price_numeric = px;
    p.current_price_usd = px;
    p.mark_price_usd = px;
    p.latest_price = px;

    p.current_price_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
    p.mark_price_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
    p.mark_price_lookup_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
    p.position_market_data_state = "MARKET_DATA_READY";
    p.financial_data_status = "READY";
    p.tradability_status = p.tradability_status || "DEMO_MARK_PRICE_AVAILABLE";

    p.mark_price_unavailable_reason = "";
    p.price_resolution_failure_reason = "";
    p.matched_market_pair_status = "MATCHED_FROM_NUMERIC_ALIAS";
    p.price_status_detail = "current mark price available from numeric alias";

    // 3. Exit status: do not let stale PRICE_NOT_AVAILABLE block a demo close display.
    const badExit =
      String(p.exit_status || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.exit_status_display || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.close_freshness_status || "").toUpperCase().includes("PRICE_NOT_AVAILABLE");

    if (badExit || !p.exit_status || !p.exit_status_display) {
      p.exit_status = "OPEN_MONITORING";
      p.exit_status_display = "Price has not reached TP/SL";
      p.close_freshness_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
      p.close_price_source = p.current_price_source || "numeric_alias_current_price";
      p.close_used_fallback_price = false;
    }
  }

  return p;
}
// END PORTFOLIO_DISPLAY_STATE_CONSISTENCY_FIX_V1


  /** AE13I/AE18 honest PnL cell - numeric current/mark aliases win over stale display strings. */
  function pnlCell(p) {
    p = normalizePortfolioDisplayStateV1(p);

    const num = (v) => {
      if (v === null || v === undefined) return null;
      const txt = String(v).trim();
      if (!txt || ["N/A", "NA", "NONE", "NULL", "UNAVAILABLE"].includes(txt.toUpperCase())) return null;
      const x = Number(txt);
      return Number.isFinite(x) ? x : null;
    };

    const posNum = (v) => {
      const x = num(v);
      return x !== null && x > 0 ? x : null;
    };

    // pctFrac expects a fraction:
    //   -0.1176 -> -11.8%
    // But some API payloads already provide percent-points:
    //   -11.7629 -> -11.8%
    // Normalize before passing into pctFrac/pnlClass.
    const pctToFraction = (v) => {
      const x = num(v);
      if (x === null) return null;
      const canonical = canonicalPnlPctFraction(p, x);
      if (canonical !== null) return canonical;
      return Math.abs(x) > 1 ? x / 100 : x;
    };

    let usd = num(p.unrealized_pnl_numeric ?? p.unrealized_pnl_usd);
    let pctv = num(p.unrealized_pnl_pct_numeric ?? p.unrealized_pnl_pct);

    const px =
      posNum(p.current_price) ??
      posNum(p.current_price_numeric) ??
      posNum(p.current_price_usd) ??
      posNum(p.mark_price_usd) ??
      posNum(p.mark_price) ??
      posNum(p.latest_price) ??
      posNum(p.latest_price_usd) ??
      posNum(p.market_price_usd);

    const qty = posNum(p.quantity);
    const entry = posNum(p.entry_price ?? p.fill_price);

    if ((usd === null || pctv === null) && px !== null && qty !== null && entry !== null && entry > 0) {
      usd = (px - entry) * qty;
      pctv = (px / entry) - 1; // fraction, not percent-points
    }

    if (usd !== null) {
      const pctFracValue = pctToFraction(pctv);
      const pctHtml = pctFracValue === null
        ? ""
        : ` <span class="${pnlClass(pctFracValue)}">${pctFrac(pctFracValue, 1)}</span>`;
      return `${money(usd)}${pctHtml}`;
    }

    if (p.unrealized_pnl_display && !String(p.unrealized_pnl_display).startsWith("N/A")) {
      const pctFracValue = pctToFraction(pctv);
      const pctHtml = pctFracValue === null
        ? ""
        : ` <span class="${pnlClass(pctFracValue)}">${pctFrac(pctFracValue, 1)}</span>`;
      return `${esc(String(p.unrealized_pnl_display))}${pctHtml}`;
    }

    if (p.position_market_data_state === "DATA_STALE" || p.financial_data_status === "STALE") {
      return `<span class="pf-pnl-unavailable" title="last_good_price is not current tradable price">N/A (STALE)</span>`;
    }

    if (p.position_market_data_state === "PRICE_UNAVAILABLE" || p.financial_data_status === "UNAVAILABLE") {
      return `<span class="pf-pnl-unavailable" title="no usable current price">N/A (UNAVAILABLE)</span>`;
    }

    const msg = p.pnl_display_message || "PnL unavailable - no fresh mark price";
    return `<span class="pf-pnl-unavailable" title="${esc(msg)}">${esc(msg)}</span>`;
  }

  /** AE13I manual reentry cooldown badge for watchlist / demo queue rows. */
  function cooldownBadge(it) {
    if (!it || !it.manual_cooldown_active) return "";
    const expiry = it.manual_cooldown_expiry || it.cooldown_expires_at_utc;
    const label = expiry ? "Re-entry blocked until " + esc(String(expiry).slice(0, 19)) : "Re-entry blocked (manual cooldown)";
    return `<div class="pd-cooldown-badge">${label}</div>`;
  }

  /** AE13I/AE18 mark price cell - numeric current/mark aliases win over stale display strings. */
  function markPriceCell(p) {
    p = normalizePortfolioDisplayStateV1(p);
    const num = (v) => {
      if (v === null || v === undefined) return null;
      const s = String(v).trim();
      if (!s || ["N/A", "NA", "NONE", "NULL", "UNAVAILABLE"].includes(s.toUpperCase())) return null;
      const x = Number(s);
      return Number.isFinite(x) && x > 0 ? x : null;
    };

    const px =
      num(p.current_price) ??
      num(p.current_price_numeric) ??
      num(p.current_price_usd) ??
      num(p.mark_price_usd) ??
      num(p.mark_price) ??
      num(p.latest_price) ??
      num(p.latest_price_usd) ??
      num(p.market_price_usd);

    if (px !== null) {
      return money(px);
    }

    if (p.position_market_data_state === "DATA_STALE") {
      const lg = p.last_good_price_display ? `<div class="meta">${esc(String(p.last_good_price_display))}</div>` : "";
      return `<span class="pf-mark-missing">N/A (STALE)${lg}</span>`;
    }

    if (p.position_market_data_state === "PRICE_UNAVAILABLE") {
      return `<span class="pf-mark-missing">N/A (UNAVAILABLE)</span>`;
    }

    if (p.current_price_display && String(p.current_price_display).startsWith("N/A")) {
      const lg = p.last_good_price_display ? `<div class="meta">${esc(String(p.last_good_price_display))}</div>` : "";
      return `<span class="pf-mark-missing">${esc(String(p.current_price_display))}${lg}</span>`;
    }

    if (p.current_price_display) {
      return esc(String(p.current_price_display));
    }

    const status = p.mark_price_lookup_status || "";
    if (status === "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED") {
      return `<span class="pf-mark-missing">LEGACY_POSITION_IDENTITY_REPAIR_NEEDED<div class="meta">Identity repair required before mark price</div></span>`;
    }

    const reason = p.mark_price_unavailable_reason || p.price_resolution_failure_reason || p.matched_market_pair_status || "PRICE_NOT_AVAILABLE";
    return `<span class="pf-mark-missing">No mark price<div class="meta">${esc(reason)}</div></span>`;
  }

  function identityBlock(p) {
    p = normalizePortfolioDisplayStateV1(p);
    const lane = p.strategy_lane || p.strategy_type || "-";
    const whale = p.whale_score != null ? Number(p.whale_score).toFixed(3) : null;
    const score = p.candidate_score != null ? Number(p.candidate_score).toFixed(3) : whale;
    const sem = p.semantic_label_human || p.semantic_label || p.cluster_label || "-";
    const liq = p.liquidity != null ? Number(p.liquidity).toLocaleString(undefined, { maximumFractionDigits: 0 }) : null;
    const vol = p.volume_24h != null ? Number(p.volume_24h).toLocaleString(undefined, { maximumFractionDigits: 0 }) : null;
    const entryBits = [];
    if (score != null) entryBits.push("whale=" + score);
    if (sem && sem !== "-") entryBits.push("sem=" + String(sem).slice(0, 28));
    if (liq != null) entryBits.push("liq=" + liq);
    if (vol != null) entryBits.push("vol24h=" + vol);
    if (p.entry_reason) entryBits.push(String(p.entry_reason).slice(0, 32));
    const token = p.token_contract_address || p.contract_address || "";
    const urlSeg = p.provider_pair_url_final_segment_exact || "";
    const canonUrl = p.canonical_market_identity || p.provider_pair_url_exact || "";
    return `<div class="pf-pos-identity">
      <div class="sym">${symbolPairCell({
        symbol_pair_display: p.symbol_pair_display || p.symbol,
        symbol_pair_available: p.symbol_pair_available,
        symbol_pair_display_reason: p.symbol_pair_display_reason,
        symbol_pair_address_fallback: p.symbol_pair_address_fallback,
      })}</div>
      <div class="meta">Lane: ${esc(lane)}</div>
      ${canonUrl ? `<div class="meta">Market URL ID: ${esc(shortPool(urlSeg || canonUrl))}</div>` : ""}
      ${token ? `<div class="meta">Token: ${esc(shortPool(token))}</div>` : ""}
      ${p.pair_address_derived || p.pair_address ? `<div class="meta">DERIVED pair: ${esc(shortPool(p.pair_address_derived || p.pair_address))}</div>` : ""}
      <div class="meta">Entry: ${esc(entryBits.join(" | ") || "-")}</div>
    </div>`;
  }

  function exitStatusCell(p) {
    p = normalizePortfolioDisplayStateV1(p);
    const plan = p.exit_plan_summary || "Exit plan not configured";
    let elig;
    if (p.bot_would_exit_now) {
      elig = `<span class="delta-pos">Bot would exit: ${esc(p.bot_exit_reason || "threshold hit")}</span>`;
    } else if (!p.exit_eligible_now) {
      elig = `<span class="delta-neg">${esc(p.exit_blocker || p.bot_exit_reason || "Not eligible")}</span>`;
    } else {
      elig = `<span class="delta-neu">${esc(p.bot_exit_reason || "Holding - TP/SL not reached")}</span>`;
    }
    const manual = `<div class="meta">${esc(p.manual_close_note || "You can manually close this demo position now.")}</div>`;
    return `<div class="pf-exit-cell"><div class="plan">${esc(plan)}</div><div class="elig">${elig}</div>${manual}</div>`;
  }

  function sellDemoButton(p) {
    const id = Number(p.id);
    if (!Number.isFinite(id)) return "-";
    return `<button type="button" class="btn btn-sm btn-sell-demo" onclick="pdOpenSellDemo(${id})">Sell Demo</button>`;
  }

  /** Cache open positions for sell confirmation (portfolio + demo tab). */
  window.__pdOpenPositionsById = window.__pdOpenPositionsById || {};

  function cacheOpenPositions(list) {
    const map = window.__pdOpenPositionsById;
    (list || []).forEach((p) => {
      if (p && p.id != null) map[String(p.id)] = p;
    });
  }

  function renderPortfolioOpenRow(p) {
    const pnlUsd = p.unrealized_pnl_usd;
    const urlSeg = p.provider_pair_url_final_segment_exact || "";
    const canonUrl = p.canonical_market_identity || p.provider_pair_url_exact || "";
    return `<tr>
      <td class="mono">${dash(p.id)}</td>
      <td>${identityBlock(p)}</td>
      <td>${trafficLightDot(p)}${dataTrustLine(p)}</td>
      <td>${dash(p.chain)}</td>
      <td>${canonUrl ? marketUrlIdCell(urlSeg || canonUrl, canonUrl) : poolCopyCell(p.pair_address_derived || p.pair_address)}</td>
      <td class="mono">${dash(p.coin_id)}</td>
      <td>${dash(p.strategy_lane || p.strategy_type)}</td>
      <td class="mono">${money(p.entry_price)}</td>
      <td class="mono">${markPriceCell(p)}</td>
      <td class="mono">${money(p.size_usd)}</td>
      <td class="mono ${pnlClass(pnlUsd)}">${pnlCell(p)}</td>
      <td class="mono">${dash(p.age_label)}</td>
      <td class="mono">${fmtPctDistance(p.distance_to_take_profit_pct, "tp")}</td>
      <td class="mono">${fmtPctDistance(p.distance_to_stop_loss_pct, "sl")}</td>
      <td>${exitStatusCell(p)}</td>
      <td>${sellDemoButton(p)}</td>
    </tr>`;
  }

  function semanticBadgeClass(family) {
    const f = String(family || "").toUpperCase();
    if (f.includes("SOCIAL_CONFIRMED")) return "sem-social";
    if (f.includes("OPPORTUNISTIC_CONFIRMED") || f.includes("NON_SOCIAL_OPPORTUNISTIC")) return "sem-opp";
    if (f.includes("SUSPECTED") || f.includes("OPPORTUNISTIC")) return "sem-suspect";
    if (f.includes("INFRASTRUCTURE")) return "sem-infra";
    if (f.includes("NEEDS_REVIEW")) return "sem-review";
    if (f.includes("UNKNOWN")) return "sem-unknown";
    return "sem-unknown";
  }

  function statusBadgeClass(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("error") || s.includes("critical") || s.includes("blocked")) return "badge-critical";
    if (s.includes("warn") || s.includes("stale") || s.includes("recover")) return "badge-warn";
    if (s.includes("demo") || s.includes("paper") || s.includes("safe")) return "badge-demo";
    if (s.includes("pass") || s.includes("fresh") || s.includes("active") || s.includes("running")) return "badge-ok";
    return "badge-neu";
  }

  function providerBadgeClass(labelOrHealth) {
    const s = String(labelOrHealth || "").toLowerCase();
    if (s === "active" || (s.includes("active") && !s.includes("inactive") && !s.includes("unavailable"))) {
      return "badge-ok";
    }
    if (
      s === "unavailable_metrics_helper" ||
      s.includes("unavailable") ||
      s.includes("metrics helper")
    ) {
      return "badge-warn";
    }
    if (s === "inactive" || s.includes("inactive") || s.includes("not configured") || s.includes("local rules") || s.includes("not needed")) {
      return "badge-neu";
    }
    if (s.includes("error") && !s.includes("not")) return "badge-warn";
    return "badge-neu";
  }

  function providerStatusLabel(prov) {
    if (!prov || typeof prov !== "object") return "Inactive";
    if (prov.provider_status_explanation) return String(prov.provider_status_explanation);
    const code = String(prov.provider_health || "").toLowerCase();
    const selected = prov.provider_selected || prov.llm_provider_selected || "";
    if (code === "active") return (selected ? selected + " " : "") + "Active - local rules active";
    if (code === "unavailable_metrics_helper") {
      return (selected || "Provider") + " selected, not reachable. Unavailable - Metrics Helper Only. Local rules still active. Demo trading not blocked by LLM.";
    }
    if (code === "inactive") return "Inactive - local rules active - no LLM assistant";
    const label = String(prov.provider_health_label || "").trim();
    if (label && label.toLowerCase() !== "provider error") return label;
    return "Inactive";
  }

  function truncId(id, n) {
    const s = String(id || "");
    if (!s) return "-";
    n = n || 10;
    if (s.length <= n + 4) return s;
    return s.slice(0, n) + "...";
  }

  function idCell(fullId, meta) {
    const id = String(fullId || "");
    if (!id) return "-";
    const addressLabel = (meta && meta.address_label) || "ID";
    const addressWarning = (meta && meta.address_warning) || "";
    const safe = esc(id);
    const short = esc(truncId(id, 10));
    const titleText = esc(addressLabel + ": " + id + (addressWarning ? " - " + addressWarning : ""));
    const payload = encodeURIComponent(JSON.stringify({
      id: id,
      chain: (meta && meta.chain) || "",
      symbol: (meta && meta.symbol) || "",
      source: (meta && meta.source) || "live_market",
      first_seen_at: (meta && meta.first_seen_at) || "",
      last_seen_at: (meta && meta.last_seen_at) || "",
      semantic_label: (meta && meta.semantic_label) || "",
      status: (meta && meta.status) || "",
      address_label: addressLabel,
      address_warning: addressWarning,
    }));
    return `<span class="id-cell" title="${titleText}">
      ${addressLabel !== "Contract address" && addressLabel !== "ID" ? `<span class="badge-pill badge-warn" style="margin-right:.25rem">${esc(addressLabel)}</span>` : ""}
      <button type="button" class="id-copy-btn mono" data-copy="${safe}" onclick="copyFullId(this)" title="Copy full ID">${short}</button>
      <button type="button" class="btn btn-sm id-detail-btn" data-meta="${payload}" onclick="showIdDetails(this)">i</button>
    </span>`;
  }

  window.copyFullId = async function copyFullId(el) {
    const text = el.getAttribute("data-copy") || el.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      if (window.toast) toast("Copied full ID");
    } catch (_) {
      prompt("Copy ID:", text);
    }
  };

  window.showIdDetails = function showIdDetails(elOrMeta) {
    let m = elOrMeta;
    if (elOrMeta && elOrMeta.getAttribute) {
      try {
        m = JSON.parse(decodeURIComponent(elOrMeta.getAttribute("data-meta") || "{}"));
      } catch (_) {
        m = {};
      }
    }
    const lines = [
      "Full ID: " + (m.id || "-"),
      "Address type: " + (m.address_label || "-"),
      "Chain: " + (m.chain || "-"),
      "Symbol / Pair: " + (m.symbol || "-"),
      "Source: " + (m.source || "-"),
      "First seen: " + (m.first_seen_at || "-"),
      "Last seen: " + (m.last_seen_at || "-"),
      "Semantic: " + (m.semantic_label || "-"),
      "Status: " + (m.status || "-"),
    ];
    if (m.address_warning) lines.push("", "Warning: " + m.address_warning);
    lines.push("", "Paper/demo research only - not live trading.");
    alert(lines.join("\n"));
  };

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cacheBustGetUrl(endpoint) {
    const sep = String(endpoint || "").includes("?") ? "&" : "?";
    return String(endpoint || "") + sep + "_ts=" + Date.now();
  }

  function marketFreshTimestamp(row) {
    row = row || {};
    return (
      row.market_data_refreshed_at ||
      row.provider_fetch_at ||
      row.price_updated_at ||
      row.last_market_update_at ||
      row.fetched_at ||
      row.last_fetched ||
      row.loaded_at ||
      row.updated_at ||
      ""
    );
  }

  function marketFreshAgeLabel(row) {
    const ts = marketFreshTimestamp(row);
    if (!ts) return "unknown freshness";
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return String(ts);
    const ageSec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
    if (ageSec < 60) return `${ageSec}s ago`;
    const ageMin = Math.floor(ageSec / 60);
    if (ageMin < 60) return `${ageMin}m ago`;
    const ageHr = Math.floor(ageMin / 60);
    return `${ageHr}h ${ageMin % 60}m ago`;
  }

  /**
   * safeFetchJson - never throws into global boot flow.
   * Returns { ok, status, data, user_message, technical_error }.
   */
  async function safeFetchJson(endpoint, fallback, timeoutMs) {
    const fb = fallback === undefined ? {} : fallback;
    const ms = timeoutMs == null ? 12000 : timeoutMs;
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    let timer = null;
    try {
      if (ctrl) timer = setTimeout(() => ctrl.abort(), ms);
      const res = await fetch(cacheBustGetUrl(endpoint), {
        method: "GET",
        cache: "no-store",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        signal: ctrl ? ctrl.signal : undefined,
      });
      const text = await res.text();
      let data = fb;
      try {
        data = text ? JSON.parse(text) : fb;
      } catch (parseErr) {
        return {
          ok: false,
          status: "error",
          data: fb,
          user_message: "Data unavailable - invalid JSON from server.",
          technical_error: String(parseErr && parseErr.message || parseErr),
        };
      }
      if (!res.ok) {
        const detail = data && (data.user_message || data.detail || data.message);
        const msg = typeof detail === "string" ? detail
          : (detail && detail.message) || ("HTTP " + res.status);
        return {
          ok: false,
          status: "unavailable",
          data: (data && typeof data === "object") ? data : fb,
          user_message: msg || "Data unavailable.",
          technical_error: "HTTP " + res.status,
        };
      }
      if (data && typeof data === "object" && data.ok === false) {
        return {
          ok: false,
          status: data.status || "unavailable",
          data: data,
          user_message: data.user_message || "Data unavailable.",
          technical_error: data.details || data.technical_error || null,
        };
      }
      const empty =
        data == null ||
        (Array.isArray(data) && data.length === 0) ||
        (typeof data === "object" && !Array.isArray(data) && Object.keys(data).length === 0);
      return {
        ok: true,
        status: empty ? "empty" : ((data && data.status) || "ready"),
        data: data,
        user_message: (data && data.user_message) || "",
        technical_error: null,
      };
    } catch (err) {
      const aborted = err && (err.name === "AbortError" || /abort/i.test(String(err)));
      return {
        ok: false,
        status: aborted ? "unavailable" : "error",
        data: fb,
        user_message: aborted
          ? "Data unavailable - request timed out."
          : ("Data unavailable - " + String(err && err.message || err)),
        technical_error: String(err && err.message || err),
      };
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  window.safeFetchJson = safeFetchJson;

  /** Mutations still throw so callers can toast; reads should use safeFetchJson. */
  async function apiJson(path, method, body, timeoutMs) {
    const opts = { method: method || "GET", headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const ms = timeoutMs == null ? 45000 : timeoutMs;
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    let timer = null;
    let timedOut = false;
    try {
      if (ctrl) {
        timer = setTimeout(() => {
          timedOut = true;
          // Always abort with an explicit reason: never "aborted without reason".
          try { ctrl.abort(new Error("NETWORK_TIMEOUT: client timeout after " + ms + "ms")); }
          catch (_) { ctrl.abort(); }
        }, ms);
        opts.signal = ctrl.signal;
      }
      const res = await fetch(path, opts);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = (data && data.detail && (data.detail.message || JSON.stringify(data.detail))) || res.statusText;
        const err = new Error(msg);
        err.refresh_error_code = "PROVIDER_HTTP_" + res.status;
        err.http_status = res.status;
        throw err;
      }
      return data;
    } catch (err) {
      if (timedOut || (err && (err.name === "AbortError" || /abort/i.test(String(err && err.message))))) {
        const e = new Error("Request timed out after " + Math.round(ms / 1000) + "s before the provider responded.");
        e.refresh_error_code = "NETWORK_TIMEOUT";
        e.recovery_instruction = "Check connectivity and retry. Force Provider Refresh can take longer on first run.";
        e.retryable = true;
        throw e;
      }
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /** Build a user-facing message from a structured refresh failure/error. */
  function refreshErrorMessage(errOrPayload) {
    const e = errOrPayload || {};
    const failure = e.refresh_failure || e;
    const code = failure.refresh_error_code || e.refresh_error_code || "UNKNOWN_PROVIDER_REFRESH_ERROR";
    const msg =
      failure.user_message ||
      e.user_message ||
      failure.refresh_error_reason ||
      (e.message ? String(e.message) : "Provider refresh failed.");
    const recovery = failure.recovery_instruction || e.recovery_instruction || "";
    let out = msg + " [" + code + "]";
    if (recovery) out += " " + recovery;
    return out;
  }

  function setText(id, v) {
    const el = document.getElementById(id);
    if (el) el.textContent = v == null || v === "" ? "-" : String(v);
  }

  function setTbodyMessage(tbodyId, colspan, message, kind) {
    const el = document.getElementById(tbodyId);
    if (!el) return;
    const cls = kind === "error" || kind === "unavailable" ? "empty panel-unavailable" : "empty";
    el.innerHTML = `<tr><td colspan="${colspan}" class="${cls}">${esc(message)}</td></tr>`;
  }

  function panelUnavailable(tbodyId, colspan, result) {
    const msg = (result && result.user_message) || "Data unavailable";
    setTbodyMessage(tbodyId, colspan, msg, "unavailable");
  }

  function setBotStatusClass(el, status) {
    if (!el) return;
    el.classList.remove(
      "product-status-running",
      "product-status-paused",
      "product-status-blocked",
      "product-status-error",
      "product-status-waiting"
    );
    const s = String(status || "").toLowerCase();
    if (s === "running") el.classList.add("product-status-running");
    else if (s === "waiting" || s === "recovering") el.classList.add("product-status-waiting");
    else if (s === "blocked" || s === "error") el.classList.add("product-status-error");
    else el.classList.add("product-status-paused");
  }

  function mountVaultPanels() {
    const legacyHost = document.getElementById("vault-legacy-host");
    const ae12Host = document.getElementById("vault-ae12-host");
    const legacyRoot = document.getElementById("legacy-engineering-root");
    const ae12Panel = document.getElementById("tab-ae12");
    if (legacyHost && legacyRoot && !legacyHost.dataset.mounted) {
      // Keep Live Market content in product tab; mount remaining engineering panels
      const dash = document.getElementById("tab-dashboard");
      if (dash) {
        // Hide stale primary legacy metric cards when shown in vault
        const legacyStats = dash.querySelector("#legacy-diag-stats");
        if (legacyStats) {
          legacyStats.insertAdjacentHTML(
            "beforebegin",
            '<div class="product-note"><strong>Historical snapshot</strong> - Legacy Social/Opportunistic cluster counters below are static research diagnostics, not current live runtime metrics.</div>'
          );
        }
      }
      legacyHost.appendChild(legacyRoot);
      legacyRoot.style.display = "block";
      legacyHost.dataset.mounted = "1";
    }
    if (ae12Host && ae12Panel && !ae12Host.dataset.mounted) {
      ae12Host.appendChild(ae12Panel);
      ae12Panel.classList.add("active");
      ae12Panel.style.display = "block";
      ae12Host.dataset.mounted = "1";
    }
  }

  async function loadVaultProviderFlags() {
    const result = await safeFetchJson("/api/ae13b/provider-status", {}, 8000);
    const el = document.getElementById("vault-provider-flags");
    if (!el) return;
    if (!result.ok) {
      el.textContent = result.user_message || "Provider status unavailable";
      return;
    }
    const p = result.data || {};
    el.textContent =
      `Runtime mode: ${p.runtime_mode || "-"} | LLM selected: ${p.llm_provider_selected || "-"}` +
      ` | ${providerStatusLabel(p)} | ${p.trade_authority || "-"}`;
  }

  window.loadDemoTradingTab = async function loadDemoTradingTab() {
    setTbodyMessage("pd-lanes-body", 7, "Loading...", "loading");
    setTbodyMessage("pd-activity-body", 5, "Loading...", "loading");
    setTbodyMessage("pd-open-body", 11, "Loading...", "loading");
    try {
      const result = await safeFetchJson("/api/ae13b/demo-bot/status", {}, 12000);
      if (!result.ok) {
        setText("pd-bot-status", "Unavailable");
        setText("pd-what", result.user_message || "Demo status unavailable");
        setText("pd-why", result.technical_error || "-");
        setText("pd-next", "Demo controls remain available - retry Refresh.");
        panelUnavailable("pd-lanes-body", 7, result);
        panelUnavailable("pd-activity-body", 5, result);
        panelUnavailable("pd-open-body", 11, result);
        return;
      }
      const st = result.data || {};
      const w = st.wallet || {};
      const explain = st.what_bot_is_doing || {};
      setText("pd-bot-status", st.bot_status || "Stopped");
      setBotStatusClass(document.getElementById("pd-bot-status"), st.bot_status);
      setText("pd-demo-mode", "Active");
      setText("pd-live-trading", "Disabled");
      setText("pd-wallet", "Not connected");
      setText("pd-balance", money(w.total_equity_usd ?? w.cash_usd));
      setText("pd-open", String(st.open_positions_count ?? 0));
      setText("pd-cycles", String(st.cycles_since_start ?? st.cycles_run ?? 0));
      setText("pd-attempts", String(st.trade_attempt_count ?? 0));
      setText("pd-trades-today", `${st.trades_opened || 0} / ${st.trades_closed || 0}`);
      setText("pd-last-cycle", st.last_cycle_at ? String(st.last_cycle_at).slice(0, 19) : "-");
      const wait = st.waiting || {};
      let etaText = st.next_cycle_eta ? String(st.next_cycle_eta).slice(0, 19) : "-";
      if (st.bot_status === "Waiting" && wait.remaining_seconds != null) {
        etaText += ` (${wait.remaining_seconds}s)`;
      }
      if (wait.thread_alive === false) etaText += " | thread dead";
      setText("pd-next-eta", etaText);
      setText("pd-realized", money(w.total_net_pnl));
      const unrealEl = document.getElementById("pd-unrealized");
      if (unrealEl) {
        unrealEl.textContent = money(w.unrealized_pnl_usd);
        unrealEl.className = "val " + pnlClass(w.unrealized_pnl_usd);
      }
      const realEl = document.getElementById("pd-realized");
      if (realEl) realEl.className = "val " + pnlClass(w.total_net_pnl);
      setText("pd-what", explain.what_is_happening || st.last_action_summary || "-");
      setText(
        "pd-why",
        explain.rejection_summary || explain.why_traded_or_not || st.last_block_reason || "-"
      );
      setText("pd-next", explain.next_action || "-");
      const ae14 = st.ae14_readiness || {};
      const negEl = document.getElementById("ae14-negative-control");
      if (negEl) {
        negEl.textContent = ae14.ready_for_negative_control ? "Ready" : "Not ready";
        negEl.className = ae14.ready_for_negative_control ? "delta-pos" : "delta-neu";
      }
      const tvEl = document.getElementById("ae14-trading-validation");
      if (tvEl) {
        tvEl.textContent = ae14.ready_for_trading_validation ? "Ready" : "Not ready";
        tvEl.className = ae14.ready_for_trading_validation ? "delta-pos" : "delta-neu";
      }
      setText("ae14-reason", ae14.reason || "-");
      setText(
        "ae14-counts",
        ae14.total_rows != null
          ? `${ae14.tradable_now_count ?? 0} fresh / ${ae14.stale_count ?? 0} stale (of ${ae14.total_rows} rows, threshold ${ae14.min_tradable_rows_for_ae14 ?? "-"})`
          : "-"
      );
      setText("ae14-next-action", ae14.recommended_next_action || "-");
      const maxOpen = explain.max_open_positions ?? st.max_open_positions ?? 0;
      const availSlots = explain.available_slots ?? 0;
      const presetLabel = st.preset_id || explain.preset_id || "-";
      const slotsBlocking = explain.max_open_blocking
        ? ` (max open positions reached - ${presetLabel} cap ${maxOpen})`
        : "";
      setText(
        "pd-slots",
        `${explain.open_positions_count ?? st.open_positions_count ?? 0}/${maxOpen} open (${presetLabel}), ${availSlots} available${slotsBlocking}`
      );
      const rejectionsEl = document.getElementById("pd-rejections");
      if (rejectionsEl) {
        const reasons = explain.top_rejection_reasons || [];
        if (reasons.length) {
          rejectionsEl.innerHTML = reasons
            .slice(0, 5)
            .map((r) => `<div>- ${esc(r.label || r.guard || "unknown")}: ${esc(r.count ?? 0)}</div>`)
            .join("");
        } else {
          rejectionsEl.textContent = explain.is_normal_no_trade_behavior
            ? "No rejections last cycle (normal scanning behavior)."
            : "-";
        }
      }
      const lanesEl = document.getElementById("pd-lanes-body");
      if (lanesEl) {
        const lanes = st.strategy_lanes || [];
        lanesEl.innerHTML = lanes.length
          ? lanes.map((ln) => `
            <tr>
              <td>${esc(ln.label || ln.id)}</td>
              <td>${ln.enabled ? "On" : "Off"}</td>
              <td class="mono">${esc(ln.candidates_seen || 0)}</td>
              <td class="mono">${esc(ln.candidates_selected || 0)}</td>
              <td class="mono">${esc(ln.trades_opened || 0)}</td>
              <td class="mono">${esc(ln.blocked_count || 0)}</td>
              <td>${esc(ln.last_reason || "-")}</td>
            </tr>`).join("")
          : '<tr><td colspan="7" class="empty">No strategy lanes</td></tr>';
      }
      const act = document.getElementById("pd-activity-body");
      if (act) {
        const rows = st.activity || [];
        act.innerHTML = rows.length
          ? rows.slice(0, 20).map((e) => `
            <tr>
              <td class="mono">${esc((e.at || "").slice(0, 19))}</td>
              <td>${esc(e.event)}</td>
              <td>${esc(e.symbol || "-")}</td>
              <td>${esc(e.summary || e.reason || e.blocker || "-")}</td>
              <td class="archive-tag">paper/demo</td>
            </tr>`).join("")
          : '<tr><td colspan="5" class="empty">No demo activity yet - start the bot or run one cycle.</td></tr>';
      }
      const openBody = document.getElementById("pd-open-body");
      if (openBody) {
        const opens = st.open_positions || [];
        cacheOpenPositions(opens);
        openBody.innerHTML = opens.length
          ?         opens.map((p) => `
            <tr>
              <td class="mono">${esc(p.id)}</td>
              <td>${identityBlock(p)}<div class="meta mono">Pool ${esc(shortPool(p.pair_address))} | coin ${esc(p.coin_id != null ? p.coin_id : "-")}</div></td>
              <td>${trafficLightDot(p)}${dataTrustLine(p)}</td>
              <td>${money(p.entry_price)}</td>
              <td>${markPriceCell(p)}</td>
              <td>${money(p.size_usd)}</td>
              <td class="${pnlClass(p.unrealized_pnl_usd)}">${pnlCell(p)}</td>
              <td>${esc(p.age_label || "-")}</td>
              <td>TP ${fmtPctDistance(p.distance_to_take_profit_pct, "tp")} / SL ${fmtPctDistance(p.distance_to_stop_loss_pct, "sl")}</td>
              <td>${exitStatusCell(p)}</td>
              <td>${sellDemoButton(p)}</td>
            </tr>`).join("")
          : '<tr><td colspan="11" class="empty">No open demo positions</td></tr>';
      }
    } catch (err) {
      console.error("[loadDemoTradingTab]", err);
      const fail = { user_message: "Data unavailable - " + String(err && err.message || err) };
      panelUnavailable("pd-lanes-body", 7, fail);
      panelUnavailable("pd-activity-body", 5, fail);
      panelUnavailable("pd-open-body", 11, fail);
    }
  };

  window.pdStartBot = async function () {
    try {
      const st = await apiJson("/api/ae13b/demo-bot/start", "POST", {});
      toast(st.last_action_summary || "Demo bot started (continuous paper loop)");
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.pdPauseBot = async function () {
    try {
      await apiJson("/api/ae13b/demo-bot/pause", "POST", {});
      toast("Demo bot paused");
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.pdStopBot = async function () {
    try {
      await apiJson("/api/ae13b/demo-bot/stop", "POST", {});
      toast("Demo bot stopped");
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.pdRunCycle = async function () {
    try {
      toast("Running one demo cycle...");
      const r = await apiJson("/api/ae13b/demo-bot/run-once", "POST", {});
      toast((r.status && r.status.last_action_summary) || "Demo cycle complete");
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.pdCloseAll = async function () {
    if (!confirm("Close all current demo positions? (paper/demo only)")) return;
    try {
      const r = await apiJson("/api/ae13b/demo-bot/close-all", "POST", {});
      toast(`Closed ${(r.closed || []).length} demo position(s)`);
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.pdResetWallet = async function () {
    if (!confirm("Reset demo wallet to $10,000? This is paper/demo only and does not affect a real wallet.")) return;
    try {
      await apiJson("/api/ae13b/demo-bot/reset-wallet", "POST", {});
      toast("Demo wallet reset to $10,000");
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };

  window.__pdSellTargetId = null;

  window.pdOpenSellDemo = function pdOpenSellDemo(posId) {
    const id = Number(posId);
    const p = (window.__pdOpenPositionsById || {})[String(id)];
    if (!p) {
      toast("Position " + id + " not found in open demo cache - refresh and retry.");
      return;
    }
    window.__pdSellTargetId = id;
    const modal = document.getElementById("pd-sell-modal");
    const summary = document.getElementById("pd-sell-summary");
    const reason = document.getElementById("pd-sell-reason");
    const note = document.getElementById("pd-sell-note");
    if (reason) reason.value = "user_exit";
    if (note) note.value = "";
    const lines = [
      "Symbol: " + (p.symbol || "-"),
      "Chain: " + (p.chain || "-"),
      "Position ID: " + (p.id != null ? p.id : "-"),
      "Pool / pair: " + shortPool(p.pair_address) + (p.pair_address ? " (" + p.pair_address + ")" : ""),
      "Coin ID: " + (p.coin_id != null ? p.coin_id : "-"),
      "Lane: " + (p.strategy_lane || p.strategy_type || "-"),
      "Entry price: " + money(p.entry_price),
      "Current price: " + (p.current_price != null ? money(p.current_price) : "No mark price"),
      "Unrealized PnL: " + (p.unrealized_pnl_usd != null ? money(p.unrealized_pnl_usd) + " (" + canonicalPnlPctText(p, p.unrealized_pnl_pct_numeric ?? p.unrealized_pnl_pct, 1) + ")" : "-"),
      "Size: " + money(p.size_usd),
      "Est. exit fees: paper DEX fee model (~1.5% + priority on Solana)",
      "",
      "This closes ONLY this demo position. Paper/demo only - not live approved.",
    ];
    if (summary) summary.textContent = lines.join("\n");
    const warnEl = document.getElementById("pd-sell-warning");
    window.__pdSellWarningShown = false;
    if (warnEl) {
      const isStaleOrFallback = p.mark_fresh === false
        || String(p.close_freshness || p.current_price_source || "").toLowerCase().includes("fallback")
        || String(p.current_price_source || "").toLowerCase().includes("entry");
      if (isStaleOrFallback) {
        warnEl.textContent = "Manual close will use last-known / fallback price. This price is not validated as fresh market data.";
        warnEl.hidden = false;
        window.__pdSellWarningShown = true;
      } else {
        warnEl.hidden = true;
        warnEl.textContent = "";
      }
    }
    if (modal) modal.hidden = false;
  };

  window.pdCancelSellDemo = function pdCancelSellDemo() {
    window.__pdSellTargetId = null;
    const modal = document.getElementById("pd-sell-modal");
    if (modal) modal.hidden = true;
  };

  window.pdConfirmSellDemo = async function pdConfirmSellDemo() {
    const id = Number(window.__pdSellTargetId);
    if (!Number.isFinite(id)) {
      toast("No position selected.");
      return;
    }
    const reasonEl = document.getElementById("pd-sell-reason");
    const noteEl = document.getElementById("pd-sell-note");
    const close_reason = (reasonEl && reasonEl.value) || "user_exit";
    const close_note = (noteEl && noteEl.value) || "";
    const btn = document.getElementById("pd-sell-confirm-btn");
    if (btn) btn.disabled = true;
    try {
      const r = await apiJson("/api/positions/" + id + "/close", "PUT", {
        close_reason: close_reason,
        close_note: close_note,
        manual_close_warning_shown: window.__pdSellWarningShown === true,
      });
      pdCancelSellDemo();
      const msg = (r && (r.message || (r.closed && ("Position " + r.closed.id + " closed manually."))))
        || ("Position " + id + " closed manually.");
      toast(r && r.warning ? msg + " " + r.warning : msg);
      delete (window.__pdOpenPositionsById || {})[String(id)];
      if (typeof loadPortfolioTab === "function") await loadPortfolioTab();
      if (typeof loadDemoTradingTab === "function") await loadDemoTradingTab();
    } catch (e) {
      toast("Sell Demo failed: " + String(e.message || e));
    } finally {
      if (btn) btn.disabled = false;
    }
  };

  // --- AE13K Clean Forward Market Feed ---
  ViewSwitcher.cfPrevHashes = ViewSwitcher.cfPrevHashes || new Map();
  ViewSwitcher.cfLastPayload = null;

  function cfFmtUsd(v) {
    if (v == null || v === "") return "-";
    const n = Number(v);
    if (!Number.isFinite(n)) return esc(String(v));
    if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return "$" + (n / 1e3).toFixed(1) + "K";
    if (n >= 1) return "$" + n.toFixed(4);
    return "$" + n.toPrecision(4);
  }

  function cfFmtPct(v) {
    if (v == null || v === "") return '<span class="cf-pct-neu">-</span>';
    const n = Number(v);
    if (!Number.isFinite(n)) return '<span class="cf-pct-neu">-</span>';
    let cls = "cf-pct-neu";
    if (n > 0) cls = "cf-pct-pos";
    else if (n < 0) cls = "cf-pct-neg";
    const sign = n > 0 ? "+" : n < 0 ? "" : "";
    return `<span class="${cls}">${sign}${n.toFixed(2)}%</span>`;
  }

  function marketUrlIdCell(finalSegment, fullUrl) {
    const seg = String(finalSegment || "").trim();
    const url = String(fullUrl || "").trim();
    if (!seg && !url) return "-";
    const short = seg ? esc(shortPool(seg)) : esc(shortPool(url));
    const copyUrl = url
      ? `<button class="btn btn-sm cf-copy-btn" type="button" onclick="cfCopyText(${JSON.stringify(url)})" title="Copy provider URL">Copy URL</button>`
      : "";
    const open = url
      ? `<a class="cf-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer" title="${esc(url)}">Open chart</a>`
      : "";
    return `<span class="mono" title="${esc(url || seg)}">${short}</span> ${open} ${copyUrl}
      <div style="font-size:.62rem;color:var(--muted);">Market URL</div>`;
  }

  function cfUrlCell(url, pairAddr) {
    const u = String(url || "");
    if (!u) return marketUrlIdCell(pairAddr, "");
    const seg = String(url).replace(/\/+$/, "").split("/").pop();
    // Primary actions: Open chart + Copy URL. Derived pair helper is details-only.
    return marketUrlIdCell(seg, u);
  }

  function cfApplyFeedPayload(d, opts) {
    opts = opts || {};
    setText("cf-badge", d.demo_mode_badge || "LIVE DISABLED / DEMO ONLY");
    if (d.error_code === "CANONICAL_IDENTITY_INDEX_MISSING") {
      setText("cf-freshness", d.error_code);
      const warn = document.getElementById("cf-warning");
      if (warn) {
        warn.textContent = (d.rebuild_instruction || "Run rebuild_canonical_market_identity_index.py") + " — " + d.error_code;
      }
      renderCleanForwardRows([]);
      return;
    }
    const loadMs = d.measured_load_time_ms != null ? " (" + d.measured_load_time_ms + "ms)" : "";
    setText("cf-freshness", (d.stale_warning ? "STALE INDEX — " : "Runtime index") + loadMs);
    setText("cf-provider", d.runtime_index_sourced ? "runtime_index" : (d.source_provider || "dexscreener"));
    const st = d.stats || {};
    setText("cf-stat-candidates", st.total_candidates_seen != null ? st.total_candidates_seen : (st.runtime_index_rows != null ? st.runtime_index_rows : st.total_candidates_seen));
    setText("cf-stat-valid", st.valid_provider_pairs != null ? st.valid_provider_pairs : st.total_rows);
    setText("cf-stat-bases", st.unique_base_tokens != null ? st.unique_base_tokens : 0);
    setText("cf-stat-pairs", st.unique_canonical_markets != null ? st.unique_canonical_markets : (st.unique_pair_addresses != null ? st.unique_pair_addresses : 0));
    setText("cf-stat-dupes", st.duplicate_pools_suppressed != null ? st.duplicate_pools_suppressed : 0);
    setText("cf-stat-invalid", st.invalid_or_unresolved_addresses != null ? st.invalid_or_unresolved_addresses : 0);
    setText("cf-stat-clean", st.clean_rows_displayed != null ? st.clean_rows_displayed : (st.total_rows != null ? st.total_rows : 0));
    const warn = document.getElementById("cf-warning");
    if (warn) {
      if (d.stale_warning) warn.textContent = d.user_message || "Canonical identity index is stale.";
      else if (d.warning) warn.textContent = d.warning;
      else warn.textContent = "";
    }
    const deferredEl = document.getElementById("cf-deferred-msg");
    if (deferredEl) {
      const deferred = Number(st.verification_deferred_count || 0);
      const rateLimited = Number(st.provider_rate_limited_count || 0);
      if (rateLimited > 0) {
        deferredEl.style.display = "";
        deferredEl.textContent = "Provider verification deferred due to rate limit. Not tradable until verified.";
      } else if (deferred > 0 && !(d.rows || []).length) {
        deferredEl.style.display = "";
        deferredEl.textContent = "Provider verification deferred. Row is not tradable until verified.";
      } else {
        deferredEl.style.display = "none";
        deferredEl.textContent = "";
      }
    }
    renderCleanForwardRows(d.rows || []);
    renderCleanForwardAlts(d.alternative_pools || []);
    cfRenderRefreshMetadata(d, opts);
    ViewSwitcher.cfLastPayload = d;
    const last = document.getElementById("cf-last-refresh");
    if (last) last.textContent = new Date().toLocaleTimeString();
  }

  function cfRenderRefreshMetadata(d, opts) {
    const meta = (d && (d.refresh || d.refresh_metadata)) || {};
    const msgEl = document.getElementById("cf-refresh-msg");
    const setMeta = function (id, val) {
      const el = document.getElementById(id);
      if (el) el.textContent = val == null || val === "" ? "-" : String(val);
    };
    setMeta("cf-meta-mode", meta.refresh_mode || (opts.bootstrap ? "tab_bootstrap_get" : "-"));
    setMeta("cf-meta-provider-fetch", meta.latest_provider_fetch_at || d.built_at_utc || "-");
    setMeta(
      "cf-meta-cache",
      meta.cache_hit_count != null
        ? meta.cache_hit_count + " / " + (meta.cache_miss_count != null ? meta.cache_miss_count : "-")
        : "-"
    );
    const ageMin = meta.cache_age_seconds_min;
    const ageMax = meta.cache_age_seconds_max;
    let ageLabel = "-";
    if (ageMin != null && ageMax != null) {
      ageLabel = ageMin === ageMax ? ageMin + "s" : ageMin + "s â€“ " + ageMax + "s";
    }
    setMeta("cf-meta-cache-age", ageLabel);
    setMeta("cf-meta-cache-ttl", meta.cache_ttl_seconds != null ? meta.cache_ttl_seconds + "s" : "-");
    setMeta("cf-meta-values-changed", meta.provider_values_changed_count != null ? meta.provider_values_changed_count : "-");
    setMeta("cf-meta-hash-changed", meta.payload_hash_changed_count != null ? meta.payload_hash_changed_count : "-");
    const defCount = Number(meta.verification_deferred_count || 0);
    const rlCount = Number(meta.provider_rate_limited_count || 0);
    setMeta("cf-meta-deferred", defCount + " / " + rlCount);
    if (!msgEl) return;
    let uiMsg = meta.ui_message || d.refresh_hint || "";
    if (!uiMsg && opts.bootstrap) {
      uiMsg = "Tab opened via GET bootstrap (may use provider verify cache). Use Refresh for explicit provider refetch.";
    }
    msgEl.textContent = uiMsg;
    msgEl.className = "cf-refresh-status-msg";
    if (/rate limit|deferred|failed/i.test(uiMsg)) msgEl.classList.add("cf-msg-deferred");
    else if (/cached provider verification|no provider refetch/i.test(uiMsg)) msgEl.classList.add("cf-msg-cache");
    else if (/values changed/i.test(uiMsg)) msgEl.classList.add("cf-msg-updated");
    else if (/no market value changes/i.test(uiMsg)) msgEl.classList.add("cf-msg-unchanged");
  }

  async function cfPostRefresh(force, clearCache) {
    const prevRows = (ViewSwitcher.cfLastPayload && ViewSwitcher.cfLastPayload.rows) || [];
    const body = {
      force: !!force,
      clear_cache: !!clearCache,
      limit: 100,
      max_rows_per_base_token: 1,
      max_rows_per_symbol: 1,
      max_verify: 80,
      previous_rows: prevRows,
    };
    const res = await apiJson("/api/clean-forward-feed/refresh", "POST", body, 60000);
    if (!res || res.ok === false) {
      const err = new Error(refreshErrorMessage(res || {}));
      const failure = (res && (res.refresh_failure || res)) || {};
      err.refresh_error_code = failure.refresh_error_code || "UNKNOWN_PROVIDER_REFRESH_ERROR";
      err.recovery_instruction = failure.recovery_instruction || "";
      err.retryable = !!failure.retryable;
      err.structured = true;
      throw err;
    }
    const d = (res && res.data) || res;
    return d;
  }

  window.marketFreshTimestamp = marketFreshTimestamp;
  window.marketFreshAgeLabel = marketFreshAgeLabel;

  window.refreshRuntimeIndexFromProvider = async function refreshRuntimeIndexFromProvider(opts) {
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

  window.cfForceProviderRefresh = async function () {
    try {
      await refreshRuntimeIndexFromProvider({ force: true, clearCache: true });
      await reloadAllRuntimeMarketSurfaces();
      toast("Force provider refresh completed; all market surfaces reloaded.");
    } catch (e) {
      toast("Force provider refresh failed - " + String((e && e.message) || e));
    }
  };

  window.loadCleanForwardFeedTab = async function loadCleanForwardFeedTab(opts) {
    opts = opts || {};
    const body = document.getElementById("cf-body");
    const hasRows = body && body.querySelector("tr[data-row-key]");
    if (!hasRows) setTbodyMessage("cf-body", 20, "Loading Clean Forward Market Feed...", "loading");
    const refreshBtn = document.getElementById("cf-refresh-btn");
    const forceBtn = document.getElementById("cf-force-refresh-btn");
    if (refreshBtn) refreshBtn.disabled = true;
    if (forceBtn) forceBtn.disabled = true;
    try {
      let d;
      if (opts.manual || opts.force || opts.clearCache) {
        d = await cfPostRefresh(!!opts.force, !!opts.clearCache);
      } else {
        const res = await safeFetchJson(
          "/api/ae13b/clean-forward-market-feed?limit=100",
          {},
          5000
        );
        d = res && res.ok ? (res.data || {}) : (res && res.data) || {};
        if (!res || !res.ok) {
          setText("cf-freshness", (res && res.user_message) || "Unavailable");
          if (!hasRows) panelUnavailable("cf-body", 20, res || { user_message: "Unavailable" });
          return;
        }
        cfApplyFeedPayload(d, { bootstrap: true });
        return;
      }
      cfApplyFeedPayload(d, { manual: true, force: opts.force, clearCache: opts.clearCache });
    } catch (e) {
      console.error("[loadCleanForwardFeedTab]", e);
      const detail = e && e.structured ? String(e.message) : refreshErrorMessage(e || {});
      const msgEl = document.getElementById("cf-refresh-msg");
      if (msgEl) {
        msgEl.textContent = detail;
        msgEl.title = (e && e.refresh_error_code) || "";
        msgEl.className = "cf-refresh-status-msg cf-msg-deferred";
      }
      if (!hasRows) panelUnavailable("cf-body", 20, { user_message: detail });
    } finally {
      if (refreshBtn) refreshBtn.disabled = false;
      if (forceBtn) forceBtn.disabled = false;
    }
  };

  /**
   * SYMBOL/PAIR display cell. Never renders a raw address pair as the primary
   * value: unresolved rows show an explicit status with the address pair kept
   * in a details line only.
   */
  function symbolPairCell(r) {
    const raw = r || {};
    const value = String(raw.symbol_pair_display || "").trim();
    const UNAVAILABLE_STATUSES = [
      "SYMBOL_PAIR_UNAVAILABLE",
      "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
      "PARTIAL_PROVIDER_SYMBOLS_MISSING",
    ];
    const unavailable =
      raw.symbol_pair_available === false ||
      value === "" ||
      UNAVAILABLE_STATUSES.indexOf(value) !== -1;
    const text = value || "SYMBOL_PAIR_UNAVAILABLE";
    if (!unavailable) return `<strong>${esc(text)}</strong>`;
    const details = [];
    if (raw.symbol_pair_known_side_symbol) details.push("known: " + String(raw.symbol_pair_known_side_symbol));
    if (raw.symbol_pair_display_reason) details.push(String(raw.symbol_pair_display_reason));
    if (raw.symbol_pair_address_fallback) details.push(String(raw.symbol_pair_address_fallback));
    const sub = details.length
      ? `<div class="meta mono" style="color:var(--muted)">${esc(details.join(" | "))}</div>`
      : "";
    return `<strong class="mono" title="${esc(text)}" style="color:var(--warn,#c90)">${esc(text)}</strong>${sub}`;
  }
  window.symbolPairCell = symbolPairCell;

  function cfShortAddr(a) {
    const s = String(a || "");
    if (s.length <= 12) return esc(s);
    return esc(s.slice(0, 6) + "â€¦" + s.slice(-4));
  }

  window.cfCopyText = function (text) {
    const t = String(text || "");
    if (!t) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(() => toast("Copied")).catch(() => toast("Copy failed"));
    } else {
      toast(t);
    }
  };

  function renderCleanForwardRows(rows) {
    const body = document.getElementById("cf-body");
    if (!body) return;
    if (!rows || !rows.length) {
      body.innerHTML = '<tr><td colspan="20" class="empty">No clean provider-verified market rows available yet.</td></tr>';
      return;
    }
    body.innerHTML = rows.map((r) => {
      const tx = r.txns_24h || {};
      const txLabel = tx.total != null
        ? String(tx.total)
        : (r.txns_24h_buys != null || r.txns_24h_sells != null)
          ? String(Number(r.txns_24h_buys || 0) + Number(r.txns_24h_sells || 0))
          : "-";
      const roleLabel = "DERIVED HELPER ID";
      const rowKey = r.data_row_key || r.canonical_market_identity || r.provider_pair_url_exact || r.row_key || "";
      const urlSeg = r.provider_pair_url_final_segment_exact || r.market_url_id || "";
      const dexLabel = r.provider_dex_id || r.dex_id || r.dex || r.dex_display || "unavailable";
      return `<tr data-row-key="${esc(rowKey)}">
        <td>${symbolPairCell(r)}</td>
        <td>${esc(r.chain || r.chain_id || r.normalized_chain_id || "-")}</td>
        <td>${esc(dexLabel)}</td>
        <td style="white-space:nowrap;">${marketUrlIdCell(urlSeg, r.provider_pair_url_exact || r.provider_pair_url || r.open_chart_url || r.dexscreener_url)}</td>
        <td class="mono" title="${esc(r.pair_address_derived || r.pair_address || "")}">${cfShortAddr(r.pair_address_derived || r.pair_address)}
          <div style="font-size:.62rem;color:var(--muted);">${esc(roleLabel)}</div>
        </td>
        <td class="mono" title="${esc(r.base_token_address || "")}">${cfShortAddr(r.base_token_address)}
          <div style="font-size:.62rem;color:var(--muted);">Base token address Â· ${esc(r.base_token_symbol || "")}</div>
        </td>
        <td class="mono" title="${esc(r.quote_token_address || "")}">${cfShortAddr(r.quote_token_address)}
          <div style="font-size:.62rem;color:var(--muted);">Quote token address Â· ${esc(r.quote_token_symbol || "")}</div>
        </td>
        <td>${cfFmtUsd(r.price_usd != null ? r.price_usd : r.price)}</td>
        <td>${cfFmtUsd(r.liquidity_usd != null ? r.liquidity_usd : r.liquidity)}</td>
        <td>${cfFmtUsd(r.volume_24h)}</td>
        <td>${esc(txLabel)}</td>
        <td>${cfFmtPct(r.price_change_m5 != null ? r.price_change_m5 : r.price_change_5m)}</td>
        <td>${cfFmtPct(r.price_change_h1 != null ? r.price_change_h1 : r.price_change_1h)}</td>
        <td>${cfFmtPct(r.price_change_h6 != null ? r.price_change_h6 : r.price_change_6h)}</td>
        <td>${cfFmtPct(r.price_change_h24 != null ? r.price_change_h24 : r.price_change_24h)}</td>
        <td style="font-size:.72rem;">${esc(String(r.last_fetched || r.fetched_at || "-").slice(0, 19))}</td>
        <td>${esc(r.freshness_status || "-")}</td>
        <td style="font-size:.72rem;">${esc(r.tradability_status || "-")}</td>
        <td style="font-size:.72rem;">${esc(r.identity_status || "-")}</td>
        <td style="font-size:.72rem;">${esc(r.verification_status || r.status || "-")}</td>
      </tr>`;
    }).join("");
  }

  function renderCleanForwardAlts(alts) {
    const body = document.getElementById("cf-alts-body");
    const countEl = document.getElementById("cf-alts-count");
    if (countEl) countEl.textContent = String((alts || []).length);
    if (!body) return;
    if (!alts || !alts.length) {
      body.innerHTML = '<tr><td colspan="9" class="empty">No alternative pools suppressed.</td></tr>';
      return;
    }
    body.innerHTML = alts.map((r) => `<tr>
      <td>${esc(r.pair || r.pair_label || "-")}</td>
      <td>${esc(r.chain_id || r.chain || "-")}</td>
      <td>${esc(r.dex_id || r.dex || "-")}</td>
      <td>${cfUrlCell(r.provider_pair_url || r.dexscreener_url, r.pair_address)}</td>
      <td class="mono" title="${esc(r.pair_address || "")}">${cfShortAddr(r.pair_address)}</td>
      <td class="mono" title="${esc(r.base_token_address || "")}">${cfShortAddr(r.base_token_address)} (${esc(r.base_token_symbol || "")})</td>
      <td>${cfFmtUsd(r.liquidity_usd)}</td>
      <td>${cfFmtUsd(r.volume_24h)}</td>
      <td style="font-size:.72rem;">${esc(r.suppressed_from_main_reason || "-")}</td>
    </tr>`).join("");
  }

  window.lmSetFilter = function (f, btn) {
    ViewSwitcher.lmFilter = f;
    document.querySelectorAll("#lm-filters .filter-chip").forEach((b) => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    // Hide mode: clear table rows immediately so non-matches are not dimmed leftovers
    if (ViewSwitcher.lmFilterMode !== "highlight") {
      const body = document.getElementById("lm-body");
      if (body) {
        body.querySelectorAll("tr[data-row-key]").forEach((tr) => tr.remove());
      }
      ViewSwitcher.lmRowStore.clear();
    }
    loadLiveMarketTab({ manual: true });
  };

  window.lmToggleFilterMode = function () {
    ViewSwitcher.lmFilterMode = ViewSwitcher.lmFilterMode === "highlight" ? "hide" : "highlight";
    const btn = document.getElementById("lm-filter-mode-btn");
    if (btn) {
      btn.textContent = ViewSwitcher.lmFilterMode === "highlight"
        ? "Mode: highlight matches"
        : "Mode: hide non-matches (default)";
    }
    if (ViewSwitcher.lmFilterMode !== "highlight") {
      const body = document.getElementById("lm-body");
      if (body) body.querySelectorAll("tr[data-row-key]").forEach((tr) => tr.remove());
      ViewSwitcher.lmRowStore.clear();
    }
    loadLiveMarketTab({ manual: true });
  };

  function lmRowKey(r) {
    if (r && r.canonical_market_identity) return String(r.canonical_market_identity);
    if (r && r.row_key) return String(r.row_key);
    const chain = String((r && r.chain) || "").toLowerCase();
    const url = (r && (r.provider_pair_url_exact || r.provider_pair_url)) || "";
    if (url) return url;
    const pair = (r && (r.pair_address || r.pair)) || "";
    const contract = (r && r.contract_address) || "";
    if (chain && pair) return chain + "|pair|" + pair;
    if (chain && contract) return chain + "|contract|" + contract;
    if (r && (r.candidate_id || r.coin_id)) return String(r.candidate_id || r.coin_id);
    return "src|" + String((r && r.source) || "live") + "|" + String((r && r.symbol) || "") + "|" + String((r && r.first_seen_at) || "");
  }

  function lmUpdateRefreshMeta() {
    const last = document.getElementById("lm-last-refresh");
    const next = document.getElementById("lm-next-refresh");
    const pausedLbl = document.getElementById("lm-refresh-paused-label");
    const pauseBtn = document.getElementById("lm-pause-btn");
    const resumeBtn = document.getElementById("lm-resume-btn");
    if (last) {
      last.textContent = ViewSwitcher.lmLastRefreshAt
        ? new Date(ViewSwitcher.lmLastRefreshAt).toLocaleTimeString()
        : "-";
    }
    if (next) {
      if (!ViewSwitcher.lmAutoRefresh) next.textContent = "paused";
      else if (ViewSwitcher.lmNextRefreshAt) next.textContent = new Date(ViewSwitcher.lmNextRefreshAt).toLocaleTimeString();
      else next.textContent = "-";
    }
    if (pausedLbl) pausedLbl.style.display = ViewSwitcher.lmAutoRefresh ? "none" : "";
    if (pauseBtn) pauseBtn.style.display = ViewSwitcher.lmAutoRefresh ? "" : "none";
    if (resumeBtn) resumeBtn.style.display = ViewSwitcher.lmAutoRefresh ? "none" : "";
  }

  window.lmPauseRefresh = function () {
    ViewSwitcher.lmAutoRefresh = false;
    lmUpdateRefreshMeta();
    toast("Market Snapshot Feed auto refresh paused - data stays visible");
  };
  window.lmResumeRefresh = function () {
    ViewSwitcher.lmAutoRefresh = true;
    ViewSwitcher.lmNextRefreshAt = Date.now() + ViewSwitcher.lmPollMs;
    lmUpdateRefreshMeta();
    toast("Market Snapshot Feed auto refresh resumed");
  };
  window.lmRefreshNow = async function () {
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
  window.lmTogglePinnedOnly = function () {
    ViewSwitcher.lmPinnedOnly = !ViewSwitcher.lmPinnedOnly;
    const btn = document.getElementById("lm-pinned-only-btn");
    if (btn) btn.textContent = ViewSwitcher.lmPinnedOnly ? "Show all rows" : "Show only pinned/followed";
    applyLiveMarketRowFilter();
  };
  window.lmTogglePin = function (key) {
    if (!key) return;
    if (ViewSwitcher.lmPinnedKeys.has(key)) ViewSwitcher.lmPinnedKeys.delete(key);
    else ViewSwitcher.lmPinnedKeys.add(key);
    const tr = document.querySelector(`#lm-body tr[data-row-key="${CSS.escape(key)}"]`);
    if (tr) {
      tr.classList.toggle("lm-row-pinned", ViewSwitcher.lmPinnedKeys.has(key));
      const pinBtn = tr.querySelector("[data-pin-btn]");
      if (pinBtn) pinBtn.textContent = ViewSwitcher.lmPinnedKeys.has(key) ? "Unpin" : "Pin";
    }
    if (ViewSwitcher.lmPinnedOnly) applyLiveMarketRowFilter();
  };
  window.lmSelectRow = function (key) {
    ViewSwitcher.lmSelectedKey = key;
    document.querySelectorAll("#lm-body tr[data-row-key]").forEach((tr) => {
      tr.classList.toggle("lm-row-selected", tr.getAttribute("data-row-key") === key);
    });
  };

  function applyLiveMarketRowFilter() {
    document.querySelectorAll("#lm-body tr[data-row-key]").forEach((tr) => {
      const key = tr.getAttribute("data-row-key");
      const show = !ViewSwitcher.lmPinnedOnly || ViewSwitcher.lmPinnedKeys.has(key);
      tr.style.display = show ? "" : "none";
    });
  }

  function lmBuildRowHtml(r) {
    const key = lmRowKey(r);
    const urlSeg = r.provider_pair_url_final_segment_exact || r.market_url_id || r.pair || "";
    const fullUrl = r.provider_pair_url_exact || r.provider_pair_url || "";
    const fullId = r.pair_address_derived || r.pair_address || urlSeg || "";
    const semFam = r.semantic_signal_family || r.semantic_label || "";
    const pinned = ViewSwitcher.lmPinnedKeys.has(key);
    const selected = ViewSwitcher.lmSelectedKey === key;
    const cls = [
      pinned ? "lm-row-pinned" : "",
      selected ? "lm-row-selected" : "",
      r._stale ? "lm-row-stale" : "",
    ].filter(Boolean).join(" ");
    return `
      <tr data-row-key="${esc(key)}" class="${cls}" onclick="lmSelectRow('${esc(key)}')">
        <td><button class="btn btn-sm" data-pin-btn onclick="event.stopPropagation();lmTogglePin('${esc(key)}')">${pinned ? "Unpin" : "Pin"}</button></td>
        <td class="mono" data-cell="time">${esc(String(r.time || "").slice(0, 19))}</td>
        <td data-cell="symbol">${symbolPairCell(r)}</td>
        <td data-cell="id">${marketUrlIdCell(urlSeg, fullUrl)}${r.pair_address_derived ? `<div class="meta mono">DERIVED: ${esc(shortPool(r.pair_address_derived))}</div>` : ""}</td>
        <td data-cell="chain">${esc(r.chain || "-")}</td>
        <td class="mono" data-cell="price" title="${esc(String(r.price ?? ""))}">${formatPrice(r.price)}</td>
        <td class="mono" data-cell="liq">${formatPrice(r.liquidity)}</td>
        <td class="mono" data-cell="vol">${formatPrice(r.volume_24h)}</td>
        <td class="mono ${deltaClass(r.price_change_5m)}" data-cell="d5">${pct(r.price_change_5m)}</td>
        <td class="mono ${deltaClass(r.price_change_1h)}" data-cell="d1">${pct(r.price_change_1h)}</td>
        <td class="mono ${deltaClass(r.price_change_6h)}" data-cell="d6">${pct(r.price_change_6h)}</td>
        <td class="mono ${deltaClass(r.price_change_24h)}" data-cell="d24">${pct(r.price_change_24h)}</td>
        <td class="mono" data-cell="buy">${r.buy_ratio != null ? Number(r.buy_ratio).toFixed(3) : "-"}</td>
        <td class="mono" data-cell="whale">${r.whale_score != null ? Number(r.whale_score).toFixed(3) : "-"}</td>
        <td data-cell="sem"><span class="sem-badge ${semanticBadgeClass(semFam)}">${esc(r.semantic_label)}</span></td>
        <td data-cell="semstat"><span class="badge-pill ${statusBadgeClass(r.semantic_status)}">${esc(r.semantic_status)}</span></td>
        <td data-cell="opp">${esc(r.opportunity_state)}</td>
        <td data-cell="status"><span class="action-pill ${statusBadgeClass(r.status)}">${esc(r.status)}</span></td>
        <td style="max-width:220px" data-cell="reason">${esc(r.reason)}</td>
        <td class="lm-actions" data-cell="actions">
          <button class="btn btn-sm" data-watch="${encodeURIComponent(JSON.stringify({
            symbol: r.symbol || "",
            pair: fullId,
            chain: r.chain || "solana",
          }))}" onclick="event.stopPropagation();addWatchFromMarket(this)">Watch</button>
          <button class="btn btn-sm" onclick="event.stopPropagation();lmAddDemoQueue(this)" data-demo="${encodeURIComponent(JSON.stringify({
            symbol: r.symbol || "",
            pair: fullId,
            chain: r.chain || "solana",
            contract_or_pair_address: fullId,
            source: "live_market",
            semantic_label: semFam,
          }))}">Demo Queue</button>
          <button class="btn btn-sm" onclick="event.stopPropagation();lmEvaluateNow(this)" data-eval="${encodeURIComponent(JSON.stringify({
            symbol: r.symbol || "",
            pair_address: fullId,
            chain: r.chain || "solana",
            row_key: key,
          }))}">Evaluate</button>
          <button class="btn btn-sm" onclick="event.stopPropagation();lmExplainRow('${esc(key)}')">Explain</button>
        </td>
      </tr>`;
  }

  function lmPatchRowCells(tr, r) {
    const set = (sel, html, asText) => {
      const el = tr.querySelector(sel);
      if (!el) return;
      if (asText) {
        if (el.textContent !== html) el.textContent = html;
      } else if (el.innerHTML !== html) {
        el.innerHTML = html;
      }
    };
    const semFam = r.semantic_signal_family || r.semantic_label || "";
    set('[data-cell="time"]', String(r.time || "").slice(0, 19), true);
    set('[data-cell="symbol"]', symbolPairCell(r), false);
    set('[data-cell="chain"]', r.chain || "-", true);
    set('[data-cell="price"]', formatPrice(r.price), true);
    set('[data-cell="liq"]', formatPrice(r.liquidity), true);
    set('[data-cell="vol"]', formatPrice(r.volume_24h), true);
    const d5 = tr.querySelector('[data-cell="d5"]');
    if (d5) { d5.className = "mono " + deltaClass(r.price_change_5m); d5.textContent = pct(r.price_change_5m); }
    const d1 = tr.querySelector('[data-cell="d1"]');
    if (d1) { d1.className = "mono " + deltaClass(r.price_change_1h); d1.textContent = pct(r.price_change_1h); }
    const d6 = tr.querySelector('[data-cell="d6"]');
    if (d6) { d6.className = "mono " + deltaClass(r.price_change_6h); d6.textContent = pct(r.price_change_6h); }
    const d24 = tr.querySelector('[data-cell="d24"]');
    if (d24) { d24.className = "mono " + deltaClass(r.price_change_24h); d24.textContent = pct(r.price_change_24h); }
    set('[data-cell="buy"]', r.buy_ratio != null ? Number(r.buy_ratio).toFixed(3) : "-", true);
    set('[data-cell="whale"]', r.whale_score != null ? Number(r.whale_score).toFixed(3) : "-", true);
    set('[data-cell="sem"]', `<span class="sem-badge ${semanticBadgeClass(semFam)}">${esc(r.semantic_label)}</span>`, false);
    set('[data-cell="semstat"]', `<span class="badge-pill ${statusBadgeClass(r.semantic_status)}">${esc(r.semantic_status)}</span>`, false);
    set('[data-cell="opp"]', r.opportunity_state || "-", true);
    set('[data-cell="status"]', `<span class="action-pill ${statusBadgeClass(r.status)}">${esc(r.status)}</span>`, false);
    set('[data-cell="reason"]', r.reason || "", true);
    tr.classList.toggle("lm-row-pinned", ViewSwitcher.lmPinnedKeys.has(lmRowKey(r)));
    tr.classList.toggle("lm-row-selected", ViewSwitcher.lmSelectedKey === lmRowKey(r));
    tr.classList.toggle("lm-row-stale", !!r._stale);
  }

  function renderLiveMarketKeyed(rows) {
    const body = document.getElementById("lm-body");
    const wrap = document.getElementById("lm-table-wrap");
    if (!body) return;
    const scrollTop = wrap ? wrap.scrollTop : 0;
    const incoming = new Map();
    (rows || []).forEach((r) => {
      const key = lmRowKey(r);
      incoming.set(key, r);
      ViewSwitcher.lmRowStore.set(key, r);
    });

    // First paint or empty â†’ full build (still keyed)
    const existing = body.querySelectorAll("tr[data-row-key]");
    if (!existing.length) {
      body.innerHTML = rows.length
        ? rows.map(lmBuildRowHtml).join("")
        : '<tr><td colspan="20" class="empty">No matching rows.</td></tr>';
      applyLiveMarketRowFilter();
      if (wrap) wrap.scrollTop = scrollTop;
      return;
    }

    // Hide mode (default): remove rows not in incoming immediately - do not dim leftovers
    const hideMode = ViewSwitcher.lmFilterMode !== "highlight";
    const seen = new Set();
    existing.forEach((tr) => {
      const key = tr.getAttribute("data-row-key");
      if (incoming.has(key)) {
        lmPatchRowCells(tr, incoming.get(key));
        tr.classList.remove("lm-row-stale");
        delete tr.dataset.staleSince;
        seen.add(key);
      } else if (hideMode) {
        tr.remove();
      } else {
        // Optional highlight mode: mark stale/dim non-matches briefly
        tr.classList.add("lm-row-stale");
        tr.dataset.staleSince = String(Date.now());
      }
    });
    rows.forEach((r) => {
      const key = lmRowKey(r);
      if (seen.has(key)) return;
      if (body.querySelector(`tr[data-row-key="${CSS.escape(key)}"]`)) return;
      body.insertAdjacentHTML("beforeend", lmBuildRowHtml(r));
    });
    if (!hideMode) {
      const now = Date.now();
      body.querySelectorAll("tr.lm-row-stale[data-row-key]").forEach((tr) => {
        const key = tr.getAttribute("data-row-key");
        if (ViewSwitcher.lmPinnedKeys.has(key)) return;
        const since = Number(tr.dataset.staleSince || now);
        if (now - since > 60000 && !incoming.has(key)) tr.remove();
      });
    }
    if (!rows.length && !body.querySelector("tr[data-row-key]")) {
      body.innerHTML = '<tr><td colspan="20" class="empty">No matching rows.</td></tr>';
    }
    applyLiveMarketRowFilter();
    if (wrap) wrap.scrollTop = scrollTop;
  }

  window.loadLiveMarketTab = async function loadLiveMarketTab(opts) {
    opts = opts || {};
    const isManual = !!opts.manual;
    if (!isManual && !ViewSwitcher.lmAutoRefresh) {
      lmUpdateRefreshMeta();
      return;
    }
    // Do NOT clear table to Loading on refresh - preserve DOM continuity
    const body = document.getElementById("lm-body");
    const hasRows = body && body.querySelector("tr[data-row-key]");
    if (!hasRows) setTbodyMessage("lm-body", 20, "Loading...", "loading");
    const rssStatus = document.getElementById("lm-rss-status");
    if (rssStatus && !hasRows) rssStatus.textContent = "Loading RSS...";
    try {
      const filt = ViewSwitcher.lmFilter || "all";
      const fmode = ViewSwitcher.lmFilterMode || "hide";
      const settled = await Promise.allSettled([
        safeFetchJson(
          "/api/ae13b/live-market?limit=50&status=" + encodeURIComponent(filt) +
            "&filter_mode=" + encodeURIComponent(fmode),
          {},
          15000
        ),
        // Cached-only sentiment: reads the local cache, never fetches RSS live on GET.
        safeFetchJson("/api/ae13b/news-sentiment-cache?limit=15", {}, 8000),
        safeFetchJson("/api/ae13b/provider-status", {}, 8000),
      ]);
      const dRes = settled[0].status === "fulfilled" ? settled[0].value : { ok: false, user_message: "Live market unavailable", data: {} };
      const rssRes = settled[1].status === "fulfilled" ? settled[1].value : { ok: false, user_message: "RSS unavailable", data: {} };
      const provRes = settled[2].status === "fulfilled" ? settled[2].value : { ok: false, data: {} };
      const d = dRes.ok ? (dRes.data || {}) : {};
      const rss = rssRes.ok ? (rssRes.data || {}) : {};
      // Fail-soft: keep provider payload even when ok=false if data present
      const prov = (provRes.data && typeof provRes.data === "object") ? provRes.data : {};

      if (!dRes.ok) {
        setText("lm-freshness", dRes.user_message || dRes.error_code || "Unavailable");
        if (!hasRows) panelUnavailable("lm-body", 20, dRes);
      } else {
        setText("lm-badge", d.demo_mode_badge || "LIVE DISABLED / DEMO ONLY");
        const staleIdx = d.stale_warning ? "STALE INDEX — " : "";
        const loadMs = d.measured_load_time_ms != null ? d.measured_load_time_ms + "ms" : "";
        setText("lm-freshness", staleIdx + (loadMs || (d.freshness && d.freshness.label) || "-"));
        setText("lm-updated", d.latest_market_update ? String(d.latest_market_update).slice(0, 19) : "-");
        setText("lm-pairs", d.live_pairs_count);
        setText("lm-passed", d.passed_filter);
        setText("lm-dropped", d.dropped_blocked);
        setText("lm-whale", d.average_whale_score != null ? Number(d.average_whale_score).toFixed(3) : "-");
        const countEl = document.getElementById("lm-filter-count");
        if (countEl) countEl.textContent = d.filter_result_label || ("Showing " + (d.count || 0));
        renderLiveMarketKeyed(d.rows || []);
      }

      const provLabel = providerStatusLabel(prov);
      setText("lm-provider", provLabel);
      const provEl = document.getElementById("lm-provider");
      if (provEl) provEl.className = providerBadgeClass(prov.provider_health || provLabel);

      setText("lm-rss-count", rss.count ?? 0);
      setText("lm-sent-avg", rss.aggregate_sentiment_score != null ? Number(rss.aggregate_sentiment_score).toFixed(3) : "-");

      await loadProductWatchlist();
      await loadDemoQueuePanel();

      const rssList = document.getElementById("lm-rss-list");
      setText("lm-rss-agg", rss.aggregate_sentiment_score != null ? Number(rss.aggregate_sentiment_score).toFixed(3) : "-");
      setText("lm-rss-updated", rss.latest_rss_update ? String(rss.latest_rss_update).slice(0, 19) : "");
      const rssCacheStatus = rss.rss_news_sentiment_status || (rssRes.ok ? "" : "NEWS_SENTIMENT_CACHE_UNAVAILABLE");
      if (rssStatus) {
        if (!rssRes.ok) {
          rssStatus.textContent =
            "NEWS_SENTIMENT_CACHE_UNAVAILABLE — " + (rssRes.user_message || "local sentiment cache unreadable");
        } else if (rssCacheStatus === "NEWS_SENTIMENT_CACHE_READY") {
          rssStatus.textContent =
            `Cached sentiment | ${rss.cached_sentiment_records_count || 0} records | ` +
            `${rss.rss_cached_items_count || 0} archived RSS payloads (local cache only)`;
        } else {
          rssStatus.textContent =
            rssCacheStatus + (rss.sentiment_cache_missing_reason ? " — " + rss.sentiment_cache_missing_reason : "");
        }
      }
      if (rssList) {
        const items = rss.items || [];
        if (!rssRes.ok) {
          rssList.innerHTML = `<div class="empty">${esc("NEWS_SENTIMENT_CACHE_UNAVAILABLE — " + (rssRes.user_message || "cache unreadable"))}</div>`;
        } else {
          rssList.innerHTML = items.length
            ? items.map((it) => {
                const s = Number(it.sentiment_score || 0);
                const cls = s > 0.05 ? "pos" : s < -0.05 ? "neg" : "neu";
                return `<div class="sentiment-item">
                <div style="font-size:.78rem">${esc(it.headline)}
                  <div style="font-size:.65rem;color:var(--muted)">${esc(it.source || "")} | ${esc(it.sentiment_label || "")}</div>
                </div>
                <span class="sentiment-score ${cls}">${s >= 0 ? "+" : ""}${s.toFixed(2)}</span>
              </div>`;
              }).join("")
            : `<div class="empty">${esc(
                (rssCacheStatus || "NEWS_SENTIMENT_CACHE_EMPTY") +
                  (rss.sentiment_cache_missing_reason ? " — " + rss.sentiment_cache_missing_reason : "")
              )}</div>`;
        }
      }
      ViewSwitcher.lmLastRefreshAt = Date.now();
      ViewSwitcher.lmNextRefreshAt = Date.now() + ViewSwitcher.lmPollMs;
      lmUpdateRefreshMeta();
    } catch (e) {
      console.error("[loadLiveMarketTab]", e);
      if (!hasRows) panelUnavailable("lm-body", 20, { user_message: "Data unavailable - " + String(e && e.message || e) });
      if (rssStatus) rssStatus.textContent = "NEWS_SENTIMENT_CACHE_UNAVAILABLE — panel load failed";
    }
  };

  window.lmSendToDemo = function () {
    toast("Use Demo Queue or Demo Trading controls (paper/demo guards apply)");
    NavigationManager.switchTo("demo");
  };

  window.lmAddDemoQueue = async function (el) {
    try {
      const payload = JSON.parse(decodeURIComponent(el.getAttribute("data-demo") || "{}"));
      await apiJson("/api/demo-queue/add", "POST", payload);
      toast("Added to Demo Trade Queue - paper only");
      await loadDemoQueuePanel();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.lmEvaluateNow = async function (el) {
    try {
      const payload = JSON.parse(decodeURIComponent(el.getAttribute("data-eval") || "{}"));
      const res = await apiJson("/api/ae13b/live-market/evaluate", "POST", payload);
      toast(
        (res.blocked ? "Blocked: " : res.selected ? "Selected: " : "Result: ") +
          (res.reason || res.next_possible_action || "done")
      );
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.lmExplainRow = function (key) {
    const r = ViewSwitcher.lmRowStore.get(key);
    if (!r) {
      toast("Row details unavailable");
      return;
    }
    toast(
      `${r.symbol || "-"} | ${r.semantic_label || "-"} | ${r.status || "-"} | ${String(r.reason || "").slice(0, 120)}`
    );
  };

  window.addWatchFromMarket = async function addWatchFromMarket(elOrObj) {
    let payload = elOrObj;
    if (elOrObj && elOrObj.getAttribute) {
      try {
        payload = JSON.parse(decodeURIComponent(elOrObj.getAttribute("data-watch") || "{}"));
      } catch (_) {
        payload = {};
      }
    }
    try {
      await apiJson("/api/watchlist/add", "POST", {
        symbol: payload.symbol || null,
        pair: payload.pair || null,
        contract_address: payload.pair || null,
        chain: payload.chain || "solana",
        expected_category: "user wants investigation",
        analyze_now: false,
      });
      toast("Added to Watchlist (paper/demo tracking only)");
      await loadProductWatchlist();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  function renderProductWatchlistItems(items) {
    const body = document.getElementById("wl-body");
    if (!body) return;
    const rows = items || [];
    body.innerHTML = rows.length
      ? rows.map((it) => {
          const displayName = it.display_name || it.user_entered_name || it.user_display_label || it.user_symbol || it.display_symbol || "Unknown";
          const displaySym = it.display_symbol || it.user_entered_symbol || it.user_symbol || it.user_entered_pair || "-";
          const chain = it.display_chain || it.user_entered_chain || it.chain || "Unknown";
          const fullId = it.full_id_copyable || it.display_id || it.user_entered_contract_or_pair_address || it.user_contract_address || it.user_pair || it.pair || "";
          const itemId = it.id || it.watchlist_id || "";
          const disabled = !!it.disabled || it.enabled === false;
          const tracking = it.tracking_enabled !== false && !disabled;
          const hyp = it.user_expected_category || it.expected_category || "-";
          const marketMatch = it.market_match_status || it.data_collection_status || "-";
          const resolution = it.resolution || {};
          const resolved = it.resolved_identity || {};
          const marketExplain = it.market_match_explanation || resolution.reason || (
            marketMatch === "waiting_for_market_match"
              ? "Tracked from user input. Not found in current local market feed. External lookup not enabled."
              : ""
          );
          const idStatus = it.identity_resolution_status || resolved.resolution_status || "user_entered_identity";
          const collection = it.collection_status || "-";
          const statuses = [
            "id:" + idStatus,
            "mkt:" + marketMatch,
            "track:" + (tracking ? "on" : "off"),
            "collect:" + collection,
            "sem:" + (it.semantic_status || "-"),
            "dq:" + (it.demo_queue_status || "not_in_queue"),
          ].join(" | ");
          const toggleBtn = disabled
            ? `<button class="btn btn-sm" data-id="${esc(itemId)}" onclick="enableWatchItem(this)">Enable</button>`
            : `<button class="btn btn-sm" data-id="${esc(itemId)}" onclick="disableWatchItem(this)">Disable</button>`;
          const trackBtn = tracking
            ? `<button class="btn btn-sm" data-id="${esc(itemId)}" data-track="0" onclick="wlTrack(this)">Stop Tracking</button>`
            : `<button class="btn btn-sm" data-id="${esc(itemId)}" data-track="1" onclick="wlTrack(this)">Track Continuously</button>`;
          const pinLbl = it.pinned ? "Unpin" : "Pin";
          const resolvedLine = resolved.resolved_symbol || resolved.resolved_name
            ? `<div class="wl-resolved">Resolved: ${esc(resolved.resolved_name || "-")} / ${esc(resolved.resolved_symbol || "-")} | ${esc(resolved.resolution_status || "")}</div>`
            : `<div class="wl-resolved">Resolution: ${esc(resolution.reason || marketExplain || "Tracked from user input.")}</div>`;
          return `<tr class="${disabled ? "wl-disabled" : ""} ${it.pinned ? "lm-row-pinned" : ""}" data-wl-id="${esc(itemId)}">
              <td><strong>${esc(displayName)}</strong>${it.pinned ? " | pinned" : ""}${resolvedLine}${cooldownBadge(it)}</td>
              <td>${esc(displaySym)}</td>
              <td>${esc(chain)}</td>
              <td>${idCell(fullId, { chain, symbol: displaySym, source: "watchlist", first_seen_at: it.first_added_at || it.added_at, last_seen_at: it.last_seen_in_market, semantic_label: it.semantic_classification, status: it.status })}</td>
              <td><span class="wl-hyp">${esc(hyp)}</span></td>
              <td><span class="sem-badge ${semanticBadgeClass(it.semantic_classification || it.semantic_signal_family)}">${esc(it.semantic_label_human || it.semantic_classification || it.semantic_signal_family || "-")}</span></td>
              <td>${esc(marketMatch)}<div class="wl-market-explain">${esc(marketExplain)}</div></td>
              <td style="font-size:.68rem;max-width:220px">${esc(statuses)}</td>
              <td style="max-width:160px;font-size:.72rem">${esc(it.evidence_summary || it.user_evidence_note || it.user_note || "-")}</td>
              <td class="wl-actions">
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlResolve(this)">Resolve</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlEditIdentity(this)">Edit Identity</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlSemantic(this)">Classify</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlDemoQueue(this)">Demo Queue</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlEvaluate(this)">Evaluate</button>
                ${trackBtn}
                <button class="btn btn-sm" data-id="${esc(itemId)}" data-pinned="${it.pinned ? "1" : "0"}" onclick="wlPin(this)">${pinLbl}</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" onclick="wlEvidence(this)">Evidence</button>
                <button class="btn btn-sm" data-id="${esc(itemId)}" data-contract="${esc(fullId)}" onclick="removeWatchItem(this)">Remove</button>
                ${toggleBtn}
              </td>
            </tr>`;
        }).join("")
      : '<tr><td colspan="10" class="empty">No watchlist items - add a symbol, pair, or contract above.</td></tr>';
  }

  window.loadProductWatchlist = async function loadProductWatchlist(prefetchedItems) {
    try {
      const note = document.getElementById("wl-note");
      if (Array.isArray(prefetchedItems)) {
        if (note) note.textContent = "Watchlist is paper/demo research tracking - not live trading.";
        renderProductWatchlistItems(prefetchedItems);
        return;
      }
      const result = await safeFetchJson("/api/watchlist", { items: [] }, 10000);
      if (!result.ok) {
        if (note) note.textContent = result.user_message || "Watchlist unavailable";
        panelUnavailable("wl-body", 10, result);
        return;
      }
      const d = result.data || {};
      if (note) note.textContent = d.note || "Watchlist is paper/demo research tracking - not live trading.";
      renderProductWatchlistItems(d.items || []);
    } catch (e) {
      console.error("[loadProductWatchlist]", e);
      panelUnavailable("wl-body", 10, { user_message: "Watchlist unavailable" });
    }
  };

  window.loadDemoQueuePanel = async function loadDemoQueuePanel() {
    const body = document.getElementById("dq-body");
    if (!body) return;
    try {
      const result = await safeFetchJson("/api/demo-queue", { items: [] }, 8000);
      if (!result.ok) {
        panelUnavailable("dq-body", 12, result);
        return;
      }
      const d = result.data || {};
      const rows = d.items || [];
      const note = document.getElementById("dq-note");
      if (note) note.textContent = (d.label || "Demo Trade Queue - paper only") + " . Manual Watchlist Scout . risk guard applies.";
      body.innerHTML = rows.length
        ? rows.map((it) => {
            const inherits = it.inherits_active_bot_preset !== false;
            const riskTitle = it.risk_profile_disclosure
              ? esc(it.risk_profile_disclosure)
              : (inherits ? "Inherits active bot preset" : "Explicit risk profile for this queue item");
            const riskCell = `<span title="${riskTitle}">${esc(it.risk_mode || "balanced")}${inherits ? " (inherited)" : ""}</span>`;
            const blockerText = it.last_blocker || "-";
            const evalBits = [];
            if (it.evaluation_stale) {
              evalBits.push('<span class="badge-pill badge-warn" title="' + esc(it.evaluation_stale_reason || "Evaluation stale - click Evaluate Now.") + '">Evaluation stale</span>');
            } else if (it.gatekeeper_evaluated) {
              evalBits.push('<span class="badge-pill badge-ok">Evaluated</span>');
            }
            if (it.gatekeeper_status) evalBits.push("gate:" + esc(it.gatekeeper_status));
            if (it.tradability_status) evalBits.push(esc(it.tradability_status));
            const evalCell = evalBits.length ? evalBits.join(" ") : "-";
            return `
            <tr class="${it.enabled === false ? "wl-disabled" : ""}">
              <td><strong>${esc(it.symbol || "-")}</strong>${cooldownBadge(it)}</td>
              <td>${esc(it.chain || "-")}</td>
              <td>${idCell(it.contract_or_pair_address || it.pair || "", { chain: it.chain, symbol: it.symbol, source: "demo_queue" })}</td>
              <td>${esc(it.strategy_lane || "Manual Watchlist Scout")}</td>
              <td>${riskCell}</td>
              <td>${esc(it.semantic_label || "-")}</td>
              <td>${esc(it.market_match_status || "-")}</td>
              <td>${esc(it.eligibility_status || it.demo_queue_status || "-")}</td>
              <td>${esc(it.last_decision || "-")}</td>
              <td style="max-width:160px;font-size:.7rem">${esc(blockerText)}</td>
              <td style="max-width:150px;font-size:.68rem">${evalCell}</td>
              <td>
                <button class="btn btn-sm" data-qid="${esc(it.queue_id)}" onclick="dqEvaluate(this)">Evaluate</button>
                <button class="btn btn-sm" data-qid="${esc(it.queue_id)}" onclick="dqRemove(this)">Remove</button>
              </td>
            </tr>`;
          }).join("")
        : '<tr><td colspan="12" class="empty">No demo queue items - use Watchlist -> Demo Queue or Live Market -> Demo Queue.</td></tr>';
    } catch (e) {
      panelUnavailable("dq-body", 12, { user_message: "Demo queue unavailable" });
    }
  };

  async function wlAction(path, el, okMsg) {
    const id = el.getAttribute("data-id");
    if (!id) { toast("Missing watchlist id"); return; }
    try {
      const res = await apiJson(path, "POST", { id });
      toast(okMsg || "Done");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
      if (path.indexOf("demo-queue") >= 0) await loadDemoQueuePanel();
      return res;
    } catch (e) {
      toast(String(e.message || e));
    }
  }
  window.wlResolve = (el) => wlAction("/api/watchlist/resolve", el, "Identity resolve complete");
  window.wlSemantic = (el) => wlAction("/api/watchlist/semantic-check", el, "Semantic check complete");
  window.wlDemoQueue = (el) => wlAction("/api/watchlist/demo-queue", el, "Added to Demo Queue - paper only");
  window.wlEvaluate = async (el) => {
    const res = await wlAction("/api/watchlist/evaluate", el, null);
    if (res) {
      const msg = (res.decision || res.last_decision || "Evaluated") + ": " +
        (res.reason || res.next_action || res.next_possible_action || "");
      toast(msg);
    }
  };
  window.wlTrack = async (el) => {
    const id = el.getAttribute("data-id");
    const enable = el.getAttribute("data-track") === "1";
    try {
      const res = await apiJson("/api/watchlist/track", "POST", { id, tracking_enabled: enable });
      toast(enable ? "Track Continuously enabled" : "Tracking stopped (item kept)");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.wlEditIdentity = async (el) => {
    const id = el.getAttribute("data-id");
    const name = prompt("Name (user-entered):", "");
    if (name === null) return;
    const symbol = prompt("Symbol (user-entered):", "") || null;
    const chain = prompt("Chain (e.g. bsc, solana):", "") || null;
    const contract = prompt("Contract / pair address:", "") || null;
    try {
      const res = await apiJson("/api/watchlist/identity", "POST", {
        id,
        name: name || null,
        symbol,
        chain,
        contract_or_pair_address: contract,
        display_label: name || null,
      });
      toast("Identity updated (user-entered preserved separately from resolver)");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.wlPin = async (el) => {
    const id = el.getAttribute("data-id");
    const pinned = el.getAttribute("data-pinned") === "1";
    try {
      const res = await apiJson("/api/watchlist/pin", "POST", { id, pinned: !pinned });
      toast(!pinned ? "Pinned" : "Unpinned");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.wlEvidence = async (el) => {
    const id = el.getAttribute("data-id");
    const url = prompt("Evidence URL (optional):", "");
    if (url === null) return;
    const note = prompt("Evidence note / claimed social mission:", "") || "";
    try {
      const res = await apiJson("/api/watchlist/evidence", "POST", {
        id,
        user_evidence_url: url,
        user_evidence_note: note,
        user_claimed_social_mission: note,
      });
      toast("Evidence saved - needs review (not auto SOCIAL_CONFIRMED)");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.dqEvaluate = async (el) => {
    const qid = el.getAttribute("data-qid");
    try {
      const res = await apiJson("/api/demo-queue/evaluate", "POST", { queue_id: qid });
      toast((res.last_decision || "Evaluated") + (res.last_blocker ? ": " + res.last_blocker : ""));
      await loadDemoQueuePanel();
    } catch (e) { toast(String(e.message || e)); }
  };
  window.dqRemove = async (el) => {
    const qid = el.getAttribute("data-qid");
    try {
      await apiJson("/api/demo-queue/remove", "POST", { queue_id: qid });
      toast("Removed from demo queue");
      await loadDemoQueuePanel();
    } catch (e) { toast(String(e.message || e)); }
  };

  window.addWatchlistItem = async function addWatchlistItem() {
    const symbol = document.getElementById("wl-symbol")?.value?.trim() || null;
    const name = document.getElementById("wl-name")?.value?.trim() || null;
    const pair = document.getElementById("wl-pair")?.value?.trim() || null;
    const chain = document.getElementById("wl-chain")?.value || "solana";
    const note = document.getElementById("wl-note-input")?.value?.trim() || null;
    const cat = document.getElementById("wl-category")?.value || "user wants investigation";
    if (!symbol && !pair && !name) {
      toast("Enter a name, symbol, or contract/pair address");
      return;
    }
    try {
      const res = await apiJson("/api/watchlist/add", "POST", {
        symbol,
        name,
        pair,
        contract_address: pair,
        chain,
        note,
        expected_category: cat,
        display_label: name || null,
        analyze_now: false,
      });
      toast("Watchlist item saved - tracking from user input");
      ["wl-symbol", "wl-name", "wl-pair", "wl-note-input"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.value = "";
      });
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.removeWatchItem = async function removeWatchItem(el) {
    const id = el.getAttribute("data-id");
    const contract = el.getAttribute("data-contract");
    if (!id && !contract) {
      toast("Cannot remove - missing item id");
      return;
    }
    try {
      const res = await apiJson("/api/watchlist/remove", "POST", {
        id: id || null,
        contract_address: contract || null,
      });
      toast("Removed from watchlist");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.disableWatchItem = async function disableWatchItem(el) {
    const id = el.getAttribute("data-id");
    const contract = el.getAttribute("data-contract");
    if (!id && !contract) {
      toast("Cannot disable - missing item id");
      return;
    }
    try {
      const res = await apiJson("/api/watchlist/disable", "POST", {
        id: id || null,
        contract_address: contract || null,
      });
      toast("Watchlist item disabled");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.enableWatchItem = async function enableWatchItem(el) {
    const id = el.getAttribute("data-id");
    if (!id) {
      toast("Cannot enable - missing item id");
      return;
    }
    try {
      const res = await apiJson("/api/watchlist/enable", "POST", { id });
      toast("Watchlist item enabled");
      if (res && Array.isArray(res.items)) await loadProductWatchlist(res.items);
      else await loadProductWatchlist();
    } catch (e) {
      toast(String(e.message || e));
    }
  };

  window.loadPortfolioTab = async function loadPortfolioTab() {
    try {
      const result = await safeFetchJson("/api/ae13b/portfolio", {}, 12000);
      if (!result.ok) {
        setText("pf-balance", "-");
        setText("pf-source", result.user_message || "Portfolio unavailable");
        panelUnavailable("pf-open-body", 16, result);
        panelUnavailable("pf-trades-body", 6, result);
        panelUnavailable("pf-archive-body", 4, result);
        return;
      }
      const d = result.data || {};
      const w = d.wallet || {};
      setText("pf-balance", money(w.total_equity_usd));
      setText("pf-cash", money(w.cash_usd));
      setText("pf-pnl", money(w.total_net_pnl));
      setText("pf-fees", money(w.cumulative_total_fees));
      setText("pf-source", (d.source_status && d.source_status.current_demo_wallet_source) || "paper/demo source of truth");
      const idxMeta = d.runtime_identity_index || {};
      if (idxMeta.error_code === "CANONICAL_IDENTITY_INDEX_MISSING") {
        setText("pf-archive-note", idxMeta.rebuild_instruction || "CANONICAL_IDENTITY_INDEX_MISSING");
      } else {
        setText("pf-archive-note", (d.source_status && d.source_status.archived_positions) || "display-only");
      }
      const openBody = document.getElementById("pf-open-body");
      const opens = d.current_open_positions || [];
      cacheOpenPositions(opens);
      if (openBody) {
        openBody.innerHTML = opens.length
          ? opens.map((p) => renderPortfolioOpenRow(p)).join("")
          : '<tr><td colspan="16" class="empty">No current tradable demo positions</td></tr>';
      }
      const tradesBody = document.getElementById("pf-trades-body");
      const trades = d.current_trades || [];
      if (tradesBody) {
        tradesBody.innerHTML = trades.length
          ? [...trades].reverse().slice(0, 40).map((t) => `
            <tr>
              <td class="mono">${esc(String(t.timestamp || "").slice(0, 19))}</td>
              <td>${esc(t.side)}</td><td>${esc(t.symbol)}</td>
              <td>${money(t.notional_usd)}</td><td>${money(t.total_fees)}</td>
              <td>${esc(t.close_reason || t.reason_code || "-")}</td>
            </tr>`).join("")
          : '<tr><td colspan="6" class="empty">No demo trades yet</td></tr>';
      }
      const arch = document.getElementById("pf-archive-body");
      const archives = d.archive_open_positions_display_only || [];
      if (arch) {
        arch.innerHTML = archives.length
          ? archives.slice(0, 15).map((p) => `
            <tr>
              <td class="mono">${esc(String(p.position_id || "").slice(0, 10))}</td>
              <td>${esc(p.symbol)}</td><td>${money(p.size_usd || p.notional_usd)}</td>
              <td class="archive-tag">archive / display-only / not tradable</td>
            </tr>`).join("")
          : '<tr><td colspan="4" class="empty">No archived display-only positions</td></tr>';
      }
    } catch (e) {
      console.error("[loadPortfolioTab]", e);
      panelUnavailable("pf-open-body", 16, { user_message: "Portfolio unavailable" });
      panelUnavailable("pf-trades-body", 6, { user_message: "Portfolio unavailable" });
      panelUnavailable("pf-archive-body", 4, { user_message: "Portfolio unavailable" });
    }
  };

  window.loadMarketOpportunitiesTab = async function loadMarketOpportunitiesTab() {
    setTbodyMessage("mkt-body", 7, "Loading...", "loading");
    try {
      const result = await safeFetchJson("/api/ae13b/opportunities?limit=40", { opportunities: [] }, 12000);
      const body = document.getElementById("mkt-body");
      if (!body) return;
      if (!result.ok) {
        panelUnavailable("mkt-body", 7, result);
        return;
      }
      const d = result.data || {};
      const rows = d.opportunities || [];
      body.innerHTML = rows.length
        ? rows.map((r) => {
            const cls =
              r.action === "Demo Buy candidate" ? "buy"
                : r.action === "Blocked" ? "blocked"
                : r.action === "Already in position" ? "held" : "watch";
            const canonical = r.canonical_market_identity || r.provider_pair_url_exact || "";
            const pairText = r.symbol_pair_display || "SYMBOL_PAIR_UNAVAILABLE";
            const pairUnavailable = r.symbol_pair_available === false;
            const pairHtml = pairUnavailable
              ? `<strong class="mono" style="color:var(--warn,#c90)">${esc(pairText)}</strong>`
              : `<strong>${esc(pairText)}</strong>`;
            const detailBits = [];
            if (r.chain) detailBits.push(esc(r.chain));
            if (r.dex_id) detailBits.push(esc(r.dex_id));
            if (pairUnavailable && r.symbol_pair_address_fallback) {
              detailBits.push(esc(r.symbol_pair_address_fallback));
            }
            // SOCIAL? = user hypothesis / unconfirmed candidate — never counted as SYSTEM VERIFIED.
            const social = r.is_social_confirmed
              ? '<span class="action-pill buy" title="SYSTEM VERIFIED social">SOCIAL</span>'
              : r.is_social_candidate
                ? '<span class="action-pill watch" title="USER HYPOTHESIS / unconfirmed candidate — not counted as confirmed social">SOCIAL?</span>'
                : (String(r.semantic_status || "").toUpperCase() === "INSUFFICIENT_EVIDENCE"
                  ? '<span class="action-pill watch" title="INSUFFICIENT EVIDENCE">INSUFF.</span>'
                  : (String(r.semantic_status || "").toUpperCase() === "CLASSIFICATION_FAILED"
                    ? '<span class="action-pill blocked" title="CLASSIFICATION FAILED">FAIL</span>'
                    : ""));
            const canBuy = r.action === "Demo Buy candidate" && canonical;
            const actionCell = canBuy
              ? `<button class="btn-mini buy" data-canonical="${esc(canonical)}"
                   onclick="mktBuyDemoCandidate(this)" title="Paper/demo only - no wallet, no live order">BUY DEMO CANDIDATE</button>`
              : `<span class="action-pill ${cls}">${esc(r.action)}</span>`;
            return `<tr data-canonical="${esc(canonical)}">
              <td>${pairHtml}<div class="mono" style="color:var(--muted)">${detailBits.join(" | ")}</div></td>
              <td class="mono">${money(r.price_usd)}</td>
              <td class="mono">${money(r.liquidity_usd)}</td>
              <td>${esc(r.semantic_label)} ${social}</td>
              <td>${actionCell}</td>
              <td style="max-width:280px">${esc(r.reason)}</td>
              <td class="mono">${esc(r.seen_count || 1)}</td>
            </tr>`;
          }).join("")
        : '<tr><td colspan="7" class="empty">No market opportunities available</td></tr>';
    } catch (e) {
      console.error("[loadMarketOpportunitiesTab]", e);
      panelUnavailable("mkt-body", 7, { user_message: "Market opportunities unavailable" });
    }
  };

  /** BUY DEMO CANDIDATE — paper/demo only, URL-first canonical identity. */
  window.mktBuyDemoCandidate = async function (el) {
    const canonical = el && el.getAttribute("data-canonical");
    if (!canonical) {
      toast("DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED - no canonical market URL");
      return;
    }
    if (el) el.disabled = true;
    try {
      const res = await apiJson("/api/ae13b/demo/buy-candidate", "POST", {
        canonical_market_identity: canonical,
      }, 30000);
      const d = (res && res.data) || res || {};
      if (d.ok) {
        toast(d.user_message || "Demo position opened (paper only)");
        await loadMarketOpportunitiesTab();
        if (typeof loadPortfolioTab === "function") await loadPortfolioTab();
      } else {
        toast((d.demo_action_status || "DEMO_ACTION_FAILED_INTERNAL_ERROR") + " - " + (d.demo_action_blocked_reason || d.user_message || "blocked"));
      }
    } catch (e) {
      toast("DEMO_ACTION_FAILED_INTERNAL_ERROR - " + String((e && e.message) || e));
    } finally {
      if (el) el.disabled = false;
    }
  };

  window.loadInsightsTab = async function loadInsightsTab() {
    try {
      const settled = await Promise.allSettled([
        safeFetchJson("/api/ae13b/semantic-registry", {}, 12000),
        safeFetchJson("/api/ae13b/provider-status", {}, 8000),
        safeFetchJson("/api/ae13b/ai-assistant-status", {}, 8000),
        safeFetchJson("/api/semantic/counts", {}, 8000),
      ]);
      const dRes = settled[0].status === "fulfilled" ? settled[0].value : { ok: false, user_message: "Semantic registry unavailable", data: {} };
      const provRes = settled[1].status === "fulfilled" ? settled[1].value : { ok: false, data: {} };
      const asstRes = settled[2].status === "fulfilled" ? settled[2].value : { ok: false, data: {} };
      const semRes = settled[3].status === "fulfilled" ? settled[3].value : { ok: false, data: {} };
      const d = dRes.ok ? (dRes.data || {}) : {};
      const prov = provRes.ok ? (provRes.data || {}) : {};
      const asst = asstRes.ok ? (asstRes.data || {}) : {};
      const sem = semRes.ok ? (semRes.data || semRes) : {};

      if (!dRes.ok) {
        setText("ins-source", dRes.user_message || "Semantic registry unavailable");
        setText("ins-social-why",
          "Social uses available social evidence from verified, legacy DB, registry and curated sources. The breakdown below shows the source of each count."
        );
        setText("ins-static-note",
          ((d.static_ae12_context && d.static_ae12_context.note) || d.static_ae12_note || "")
          + " | Static Snapshot: historical research reference."
        );
        if (d.classification_warning) {
          const warn = document.getElementById("ins-class-warn");
          if (warn) {
            warn.textContent = d.classification_warning;
            warn.style.display = "block";
          }
        }
        const body = document.getElementById("ins-registry-body");
        const records = (d.records || []).slice(0, 30);
        if (body) {
          body.innerHTML = records.length
            ? records.map((r) => `
            <tr>
              <td>${esc(r.symbol)}</td>
              <td><span class="sem-badge ${semanticBadgeClass(r.semantic_signal_family)}">${esc(r.semantic_label_human || r.semantic_signal_family)}</span></td>
              <td>${esc(r.classification_source)}</td>
              <td class="mono">${esc(r.seen_count)}</td>
              <td class="mono">${esc(String(r.last_seen_at || "").slice(0, 19))}</td>
            </tr>`).join("")
            : '<tr><td colspan="5" class="empty">Registry empty - run a demo cycle to observe candidates</td></tr>';
        }
      }
      setText("ins-gemini", (prov.gemini && prov.gemini.health_label) || d.gemini_status || "Inactive");
      setText("ins-qwen", (asst.label) || providerStatusLabel(prov) || d.qwen_status || "Inactive");
      setText("ins-rss", (prov.rss && prov.rss.health_label) || d.rss_status || "-");
      setText("ins-local", (prov.local_rules && prov.local_rules.health_label) || "Local rules active");
      const insProv = document.getElementById("ins-provider-health");
      if (insProv) {
        insProv.textContent = providerStatusLabel(prov);
        insProv.className = providerBadgeClass(prov.provider_health || providerStatusLabel(prov));
      }
      const title = document.getElementById("chat-title");
      const cap = document.getElementById("chat-capability");
      if (title && asst.label) title.textContent = asst.label;
      if (cap && asst.capability) cap.textContent = asst.capability;
    } catch (e) {
      console.error("[loadInsightsTab]", e);
      setText("ins-source", "Insights unavailable");
      panelUnavailable("ins-registry-body", 5, { user_message: "Insights unavailable" });
    }
  };

  window.loadProductSettingsTab = async function loadProductSettingsTab() {
    try {
      const [presetsRes, statusRes] = await Promise.allSettled([
        safeFetchJson("/api/ae13b/presets", { presets: [] }, 8000),
        safeFetchJson("/api/ae13b/demo-bot/status", {}, 8000),
      ]);
      const d = presetsRes.status === "fulfilled" && presetsRes.value.ok ? presetsRes.value.data : { presets: [] };
      const st = statusRes.status === "fulfilled" && statusRes.value.ok ? statusRes.value.data : {};
      const box = document.getElementById("pd-presets");
      if (!box) return;
      if (presetsRes.status !== "fulfilled" || !presetsRes.value.ok) {
        box.innerHTML = `<div class="product-note">${esc((presetsRes.value && presetsRes.value.user_message) || "Presets unavailable")}</div>`;
        return;
      }
      const current = st.preset_id || "balanced";
      box.innerHTML = (d.presets || []).map((p) => `
        <div class="preset-card ${p.id === current ? "active" : ""}" onclick="pdApplyPreset('${esc(p.id)}')">
          <h3>${esc(p.label)}</h3>
          <p>${esc(p.summary)}</p>
          <div class="mono" style="margin-top:.5rem;color:var(--muted);font-size:.7rem">
            max open ${p.max_open_positions} | ${p.max_trades_per_hour}/hr | $${p.max_notional_usd}
            | exploration ${p.exploration_enabled ? "on" : "off"}
          </div>
        </div>`).join("");
    } catch (e) {
      console.error("[loadProductSettingsTab]", e);
      const box = document.getElementById("pd-presets");
      if (box) box.innerHTML = `<div class="product-note">Settings presets unavailable</div>`;
    }
  };

  window.pdApplyPreset = async function (id) {
    try {
      await apiJson("/api/ae13b/demo-bot/preset", "POST", { preset_id: id });
      toast("Demo preset applied: " + id);
      await loadProductSettingsTab();
      await loadDemoTradingTab();
    } catch (e) { toast(String(e.message || e)); }
  };

  function attachTabHandlers() {
    document.querySelectorAll("nav.tabs.nav-product button[data-tab]").forEach((btn) => {
      if (btn.dataset.shellBound) return;
      btn.dataset.shellBound = "1";
      btn.addEventListener("click", (ev) => {
        try {
          ev.preventDefault();
          const id = btn.getAttribute("data-tab") || "demo";
          NavigationManager.switchTo(id, btn);
        } catch (err) {
          console.error("[tab click]", err);
          // Absolute fail-soft: still try to show the panel
          try {
            ViewSwitcher.switchTo(btn.getAttribute("data-tab") || "demo", btn);
          } catch (_) { /* ignore */ }
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    try {
      attachTabHandlers();
      // Switch view immediately; data load is scheduled separately and cannot block shell
      ViewSwitcher.switchTo("demo");
      setTimeout(() => DataLoader.loadTab("demo"), 0);
      setInterval(() => {
        try {
          if (ViewSwitcher.current === "demo") loadDemoTradingTab();
          if (ViewSwitcher.current === "live-market") {
            if (ViewSwitcher.lmAutoRefresh) loadLiveMarketTab({ manual: false });
            else lmUpdateRefreshMeta();
          }
        } catch (e) {
          console.warn("[poll]", e);
        }
      }, ViewSwitcher.lmPollMs || 8000);
    } catch (err) {
      console.error("[product_demo boot]", err);
      try { attachTabHandlers(); } catch (_) { /* ignore */ }
    }
  });

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


  function ae18EnsureDefaultAutoRefresh60s() {
    const enabled = localStorage.getItem(AE18_AUTO_REFRESH_KEY);
    if (enabled === null || enabled === undefined || enabled === "") {
      localStorage.setItem(AE18_AUTO_REFRESH_KEY, "true");
    }

    const seconds = localStorage.getItem(AE18_AUTO_REFRESH_SECONDS_KEY);
    if (seconds === null || seconds === undefined || seconds === "" || seconds === "90") {
      localStorage.setItem(AE18_AUTO_REFRESH_SECONDS_KEY, "60");
    }
  }

  function ae18AutoRefreshEnabled() {
    const v = localStorage.getItem(AE18_AUTO_REFRESH_KEY);
    return v !== "false" && v !== "0" && v !== "off";
  }

  function ae18AutoRefreshSeconds() {
    const raw = Number(localStorage.getItem(AE18_AUTO_REFRESH_SECONDS_KEY) || "60");
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
    ae18EnsureDefaultAutoRefresh60s();

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


})();


// BEGIN MANUAL_AI_INSIGHTS_DEAD_TAB_FIX_V2
(function () {
  "use strict";

  const setTextSafe = function (id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (value === null || value === undefined || value === "") {
      el.textContent = "-";
    } else {
      el.textContent = String(value);
    }
  };

  const numberSafe = function (value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  const fetchJsonSafe = async function (url) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  };

  const maxCount = function () {
    return Math.max.apply(null, Array.from(arguments).map(numberSafe));
  };

  window.loadInsightsTab = async function loadInsightsTab() {
    const [semRaw, summaryRaw, registryRaw, providerRaw, assistantRaw] = await Promise.all([
      fetchJsonSafe("/api/semantic/counts"),
      fetchJsonSafe("/api/analytics/summary"),
      fetchJsonSafe("/api/ae13b/semantic-registry"),
      fetchJsonSafe("/api/ae13b/provider-status"),
      fetchJsonSafe("/api/ae13b/ai-assistant-status"),
    ]);

    const sem = (summaryRaw && summaryRaw.semantic_counts) || semRaw || {};
    const cluster = (summaryRaw && summaryRaw.cluster_counts) || {};
    const regWrap = registryRaw || {};
    const reg = regWrap.data || regWrap || {};
    const ctr = reg.counters || reg.counter || reg.counts || {};
    const provider = (providerRaw && (providerRaw.data || providerRaw)) || {};
    const assistant = (assistantRaw && (assistantRaw.data || assistantRaw)) || {};

    const systemSocial = sem.system_verified_social_count ?? sem.social_confirmed_count ?? 0;
    const systemOpp = sem.system_verified_opportunistic_count ?? sem.opportunistic_confirmed_count ?? 0;

    const legacySocial = sem.legacy_db_social_count
      ?? sem.legacy_socially_motivated_count
      ?? cluster.SOCIALLY_MOTIVATED
      ?? 0;

    const legacyOpp = sem.legacy_db_opportunistic_count
      ?? sem.legacy_opportunistic_speculative_count
      ?? cluster.OPPORTUNISTIC_SPECULATIVE
      ?? 0;

    const registrySocial = sem.legacy_registry_social_count
      ?? sem.legacy_registry_socially_motivated_count
      ?? 0;

    const registryOpp = sem.legacy_registry_opportunistic_count
      ?? sem.legacy_registry_opportunistic_speculative_count
      ?? 0;

    const curatedSocial = sem.curated_social_hypothesis_count ?? 0;
    const curatedOpp = sem.curated_opportunistic_hypothesis_count ?? 0;
    const curatedUnknown = sem.curated_unknown_hypothesis_count ?? 0;

    const socialDisplay = maxCount(
      sem.social_display_count,
      systemSocial,
      legacySocial,
      registrySocial,
      curatedSocial,
      ctr.runtime_social_confirmed,
      reg.runtime_social_confirmed,
      reg.coin_social_confirmed_count
    );

    const opportunisticDisplay = maxCount(
      sem.opportunistic_display_count,
      systemOpp,
      legacyOpp,
      registryOpp,
      curatedOpp,
      ctr.runtime_opportunistic_confirmed,
      reg.runtime_opportunistic_confirmed,
      reg.coin_opportunistic_confirmed_count
    );

    const classified = maxCount(
      reg.classified,
      reg.classified_count,
      ctr.classified,
      ctr.runtime_classified,
      socialDisplay + opportunisticDisplay
    );

    const unknown = maxCount(
      reg.unknown,
      reg.unknown_count,
      ctr.runtime_unknown,
      reg.runtime_unknown,
      reg.coin_unknown_unresolved_count,
      curatedUnknown
    );

    setTextSafe("ins-source", "Semantic evidence summary");
    setTextSafe(
      "ins-social-why",
      "Social is counted from available social evidence: verified, legacy DB, registry and curated sources. The breakdown below shows each source."
    );
    setTextSafe(
      "ins-static-note",
      sem.curated_targets_file_exists
        ? "Curated target file loaded; source breakdown shown below."
        : "Curated file not loaded or disabled; showing available DB/registry/verified evidence."
    );

    setTextSafe("ins-seen", reg.runtime_rows_observed ?? ctr.runtime_rows_observed ?? ctr.rows_observed ?? reg.rows_observed ?? 0);
    setTextSafe("ins-classified", classified);
    setTextSafe("ins-unresolved", unknown);
    setTextSafe("ins-unique", reg.unique_coins ?? ctr.unique_coins ?? reg.unique_coin_count ?? 0);
    setTextSafe("ins-pairs", reg.unique_pairs ?? ctr.unique_pairs ?? reg.unique_pair_count ?? 0);
    setTextSafe("ins-static", reg.static_snapshot_count ?? reg.static_snapshot ?? 14);

    setTextSafe("ins-social", socialDisplay);
    setTextSafe("ins-opp", opportunisticDisplay);
    setTextSafe("ins-suspected", reg.runtime_suspected_opportunistic ?? ctr.runtime_suspected_opportunistic ?? reg.coin_opportunistic_suspected_count ?? 0);
    setTextSafe("ins-unknown", unknown);

    setTextSafe("ins-sem-social", systemSocial);
    setTextSafe("ins-sem-opp", systemOpp);
    setTextSafe("ins-sem-insuff", sem.system_verified_insufficient_evidence_count ?? sem.insufficient_evidence_count ?? 0);
    setTextSafe("ins-sem-failed", sem.system_verified_classification_failed_count ?? sem.classification_failed_count ?? 0);
    setTextSafe("ins-sem-legacy-social", legacySocial);
    setTextSafe("ins-sem-legacy-opp", legacyOpp);
    setTextSafe("ins-sem-registry-social", registrySocial);
    setTextSafe("ins-sem-registry-opp", registryOpp);
    setTextSafe("ins-sem-curated-social", curatedSocial);
    setTextSafe("ins-sem-curated-opp", curatedOpp);
    setTextSafe("ins-sem-curated-unknown", curatedUnknown);
    setTextSafe("ins-sem-curated-exists", sem.curated_targets_file_exists ? "yes" : "no");

    setTextSafe("ins-local", "Local rules active");
    setTextSafe("ins-gemini", (provider.gemini && provider.gemini.health_label) || provider.gemini_status || reg.gemini_status || "Inactive");
    setTextSafe("ins-qwen", assistant.label || provider.qwen_status || reg.qwen_status || "Inactive");
    setTextSafe("ins-rss", (provider.rss && provider.rss.health_label) || provider.rss_status || reg.rss_status || "-");

    const body = document.getElementById("ins-registry-body");
    const records = Array.isArray(reg.records) ? reg.records.slice(0, 30) : [];
    if (body) {
      if (records.length) {
        body.innerHTML = records.map(function (r) {
          const symbol = String(r.symbol || "");
          const label = String(r.semantic_label_human || r.semantic_signal_family || r.semantic_label || "");
          const source = String(r.classification_source || r.source || "");
          const seen = String(r.seen_count || "");
          const last = String(r.last_seen_at || "").slice(0, 19);
          return "<tr>"
            + "<td>" + symbol.replace(/[&<>]/g, "") + "</td>"
            + "<td>" + label.replace(/[&<>]/g, "") + "</td>"
            + "<td>" + source.replace(/[&<>]/g, "") + "</td>"
            + "<td class=\"mono\">" + seen.replace(/[&<>]/g, "") + "</td>"
            + "<td class=\"mono\">" + last.replace(/[&<>]/g, "") + "</td>"
            + "</tr>";
        }).join("");
      } else {
        body.innerHTML = '<tr><td colspan="5" class="empty">No registry rows available; summary counters are shown above.</td></tr>';
      }
    }
  };
})();
// END MANUAL_AI_INSIGHTS_DEAD_TAB_FIX_V2


// BEGIN MANUAL_PRICE_NA_DEMO_TRADING_FIX
(function () {
  "use strict";

  function _rowText(el) {
    const tr = el && el.closest ? el.closest("tr") : null;
    return tr ? String(tr.innerText || tr.textContent || "") : "";
  }

  function _hasPriceUnavailableText(txt) {
    const t = String(txt || "").toUpperCase();
    return (
      t.includes("PRICE_NOT_AVAILABLE") ||
      t.includes("CURRENT PRICE") && t.includes("N/A") ||
      t.includes("N/A (UNAVAILABLE)") ||
      t.includes("NOT CURRENT TRADABLE PRICE") ||
      t.includes("PRICE UNAVAILABLE")
    );
  }

  function _disableUnsafeBuyButtons() {
    const buttons = Array.from(document.querySelectorAll("button[onclick*='mktBuyDemoCandidate'], button[data-canonical]"));
    for (const b of buttons) {
      const row = _rowText(b);
      if (_hasPriceUnavailableText(row)) {
        b.disabled = true;
        b.classList.add("blocked");
        b.textContent = "PRICE N/A";
        b.title = "Demo buy blocked: current tradable price is unavailable.";
      }
    }
  }

  const _oldBuy = window.mktBuyDemoCandidate;
  window.mktBuyDemoCandidate = async function (el) {
    const row = _rowText(el);
    if (_hasPriceUnavailableText(row)) {
      if (typeof toast === "function") {
        toast("DEMO_ACTION_BLOCKED_PRICE_NOT_AVAILABLE - current tradable price is unavailable");
      } else {
        alert("DEMO_ACTION_BLOCKED_PRICE_NOT_AVAILABLE - current tradable price is unavailable");
      }
      return;
    }

    if (typeof _oldBuy === "function") {
      return await _oldBuy(el);
    }

    if (typeof toast === "function") {
      toast("DEMO_ACTION_FAILED_INTERNAL_ERROR - original buy handler unavailable");
    }
  };

  const _oldLoadMarketOpportunities = window.loadMarketOpportunitiesTab;
  if (typeof _oldLoadMarketOpportunities === "function") {
    window.loadMarketOpportunitiesTab = async function () {
      const result = await _oldLoadMarketOpportunities.apply(this, arguments);
      _disableUnsafeBuyButtons();
      return result;
    };
  }

  const _oldLoadPortfolio = window.loadPortfolioTab;
  if (typeof _oldLoadPortfolio === "function") {
    window.loadPortfolioTab = async function () {
      const result = await _oldLoadPortfolio.apply(this, arguments);
      _disableUnsafeBuyButtons();
      return result;
    };
  }

  document.addEventListener("DOMContentLoaded", function () {
    _disableUnsafeBuyButtons();
    setInterval(_disableUnsafeBuyButtons, 2000);
  });
})();
// END MANUAL_PRICE_NA_DEMO_TRADING_FIX






// FRONTEND_PORTFOLIO_RENDERER_NUMERIC_ALIAS_FIRST_V4
window.__portfolioRendererNumericAliasFirstV4 = true;


// PORTFOLIO_DISPLAY_STATE_CONSISTENCY_FIXED_V1
window.__portfolioDisplayStateConsistencyFixedV1 = true;


// BEGIN PORTFOLIO_EXIT_STATUS_PRICE_ALIAS_FIX_V1
(function () {
  "use strict";

  function num(v) {
    if (v === null || v === undefined) return null;
    const s = String(v).trim();
    if (!s || ["N/A", "NA", "NONE", "NULL", "UNAVAILABLE"].includes(s.toUpperCase())) return null;
    const x = Number(s);
    return Number.isFinite(x) && x > 0 ? x : null;
  }

  function hasCurrentMarkPrice(p) {
    return (
      num(p.current_price) ??
      num(p.current_price_numeric) ??
      num(p.current_price_usd) ??
      num(p.mark_price_usd) ??
      num(p.mark_price) ??
      num(p.latest_price) ??
      num(p.latest_price_usd) ??
      num(p.market_price_usd)
    ) !== null;
  }

  function normalizeExitStatusV1(p) {
    if (!p || typeof p !== "object") return p;
    if (!hasCurrentMarkPrice(p)) return p;

    const badExit =
      String(p.exit_status || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.exit_status_display || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.exit_status_label || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.close_freshness_status || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.price_resolution_failure_reason || "").toUpperCase().includes("PRICE_NOT_AVAILABLE") ||
      String(p.mark_price_lookup_status || "").toUpperCase().includes("PRICE_NOT_AVAILABLE");

    if (badExit || !p.exit_status || !p.exit_status_display) {
      p.exit_status = "OPEN_MONITORING";
      p.exit_status_display = "Price has not reached TP/SL";
      p.exit_status_label = "Price has not reached TP/SL";
      p.close_freshness_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
      p.close_price_source = p.current_price_source || "numeric_alias_current_price";
      p.close_used_fallback_price = false;
      p.price_resolution_failure_reason = "";
      p.mark_price_unavailable_reason = "";
      p.mark_price_lookup_status = "PRICE_OK_FROM_NUMERIC_ALIAS";
    }

    return p;
  }

  function rowForPosition(p) {
    const id = String(p.id ?? p.position_id ?? "").trim();
    const sym = String(p.symbol || p.symbol_pair_display || "").trim();

    for (const row of Array.from(document.querySelectorAll("tr"))) {
      const cells = Array.from(row.cells || []);
      if (!cells.length) continue;

      const first = String(cells[0].innerText || cells[0].textContent || "").trim();
      const text = String(row.innerText || row.textContent || "");

      if (id && first === id) return row;
      if (id && sym && text.includes(id) && text.includes(sym)) return row;
    }

    return null;
  }

  function headerIndex(row, needles, fallback) {
    const table = row.closest("table");
    const headers = table ? Array.from(table.querySelectorAll("thead th")) : [];
    for (let i = 0; i < headers.length; i++) {
      const t = String(headers[i].innerText || headers[i].textContent || "").toUpperCase();
      for (const n of needles) {
        if (t.includes(n)) return i;
      }
    }
    return fallback;
  }

  function patchExitDomFromPositions(positions) {
    if (!Array.isArray(positions)) return;

    for (const raw of positions) {
      const p = normalizeExitStatusV1(raw);
      if (!hasCurrentMarkPrice(p)) continue;

      const row = rowForPosition(p);
      if (!row || !row.cells) continue;

      const exitIdx = headerIndex(row, ["EXIT STATUS"], 14);
      const cell = row.cells[exitIdx];
      if (!cell) continue;

      const txt = String(cell.innerText || cell.textContent || "").toUpperCase();

      if (txt.includes("PRICE_NOT_AVAILABLE") || txt.includes("PRICE NOT AVAILABLE")) {
        const tp = p.take_profit_pct != null ? `TP +${p.take_profit_pct}%` : "";
        const sl = p.stop_loss_pct != null ? `SL -${p.stop_loss_pct}%` : "";
        const rule = [tp, sl].filter(Boolean).join(" | ");

        cell.innerHTML =
          `<div class="meta">${rule || "Position open"}</div>` +
          `<div>Price has not reached TP/SL</div>` +
          `<div style="color:var(--text);font-weight:700">You can manually close this demo position now.</div>`;
      }
    }
  }

  async function patchPortfolioExitStatusNowV1() {
    try {
      const res = await fetch("/api/ae13b/portfolio?exit_status_price_alias_fix=" + Date.now(), {
        cache: "no-store"
      });
      if (!res.ok) return;
      const data = await res.json();

      const positions = []
        .concat(Array.isArray(data.current_open_positions) ? data.current_open_positions : [])
        .concat(Array.isArray(data.open_positions) ? data.open_positions : []);

      for (const p of positions) normalizeExitStatusV1(p);
      patchExitDomFromPositions(positions);
    } catch (e) {}
  }

  function wrapLoader(name) {
    const old = window[name];
    if (typeof old !== "function" || old.__exitStatusPriceAliasFixV1) return;

    const wrapped = async function () {
      const out = await old.apply(this, arguments);
      setTimeout(patchPortfolioExitStatusNowV1, 50);
      setTimeout(patchPortfolioExitStatusNowV1, 300);
      setTimeout(patchPortfolioExitStatusNowV1, 1000);
      return out;
    };

    wrapped.__exitStatusPriceAliasFixV1 = true;
    window[name] = wrapped;
  }

  [
    "loadPortfolioTab",
    "loadPortfolio",
    "refreshPortfolio",
    "loadDashboard",
    "refreshAll"
  ].forEach(wrapLoader);

  document.addEventListener("DOMContentLoaded", function () {
    setTimeout(patchPortfolioExitStatusNowV1, 500);
    setTimeout(patchPortfolioExitStatusNowV1, 1500);
    setInterval(patchPortfolioExitStatusNowV1, 3000);
  });

  window.__portfolioExitStatusPriceAliasFixedV1 = true;
  window.fixPortfolioExitStatusNow = patchPortfolioExitStatusNowV1;
})();
// END PORTFOLIO_EXIT_STATUS_PRICE_ALIAS_FIX_V1



// PORTFOLIO_PNL_PCT_SCALE_TARGETED_FIXED_V2
window.__portfolioPnlPctScaleTargetedFixedV2 = true;
