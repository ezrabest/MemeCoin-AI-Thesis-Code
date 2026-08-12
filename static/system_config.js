/**
 * System Configuration - synced with GET /api/settings/effective, save via PATCH /api/settings.
 */
(function () {
  const DECIMAL_FRACTION_PCT_KEYS = new Set([
    "stop_loss_pct", "take_profit_pct", "max_position_size_pct", "max_slippage_pct",
    "baseline_slippage_pct", "round_trip_fee_pct", "required_margin_after_costs_pct",
    "max_price_drift_from_model_pct", "gas_or_priority_cost_pct", "max_daily_loss_pct",
    "max_drawdown_pct", "trailing_stop_pct",
  ]);
  const DISPLAY_AS_PERCENT_POINTS = new Set([
    "rf_probability_threshold", "tab_confidence_percentile_threshold",
  ]);

  const GROUP_TITLES = {
    gates: "Trading Gates & Actionability",
    tab: "Model & TabICL Overlay",
    costs: "Costs, Slippage & Risk",
    llm: "LLM Controls",
    safety: "Runtime Safety",
  };

  const FIELD_SPECS = [
    { key: "economic_gate_enabled", label: "Economic Gate", group: "gates", kind: "bool", consumer: "economic_gate / RF approval" },
    { key: "demo_aggressive_enabled", label: "Demo Aggressive", group: "gates", kind: "bool", consumer: "economic_gate demo path" },
    { key: "paper_trading_enabled", label: "Paper Trading", group: "gates", kind: "bool", consumer: "paper execution" },
    { key: "allow_watch_to_buy_promotion", label: "Watch→Buy Promotion", group: "gates", kind: "bool", consumer: "actionability" },
    { key: "rf_gate_enabled", label: "RF Gate", group: "gates", kind: "bool", consumer: "model_runtime_inference" },
    { key: "rf_probability_threshold", label: "RF Probability Threshold", group: "gates", kind: "number", consumer: "RF gate", step: 0.1 },
    { key: "tab_confidence_boost_enabled", label: "TAB Confidence Boost", group: "tab", kind: "bool", consumer: "TAB overlay" },
    { key: "tab_confidence_boost_enabled_demo", label: "TAB DEMO Boost", group: "tab", kind: "bool", consumer: "TAB overlay demo" },
    { key: "tab_confidence_boost_enabled_live", label: "TAB LIVE Boost", group: "tab", kind: "bool", consumer: "TAB overlay live" },
    { key: "tab_confidence_suffix", label: "TAB Confidence Suffix", group: "tab", kind: "select", consumer: "TAB model lookup", options: ["nearest_neighbors_context_4096", "nearest_neighbors_context_2048"] },
    { key: "tab_confidence_percentile_threshold", label: "TAB Percentile Threshold", group: "tab", kind: "number", consumer: "TAB overlay", step: 0.1 },
    { key: "tab_position_size_multiplier", label: "TAB Position Size Multiplier", group: "tab", kind: "number", consumer: "TAB overlay sizing", step: 0.1 },
    { key: "tab_standalone_trading_enabled", label: "TAB Standalone Trading", group: "tab", kind: "bool", consumer: "blocked - overlay only", readOnly: true },
    { key: "tab_rescue_enabled", label: "TAB Rescue", group: "tab", kind: "bool", consumer: "blocked - overlay only", readOnly: true },
    { key: "max_position_size_pct", label: "Max Position Size (%)", group: "costs", kind: "number", consumer: "economic gate / risk", step: 0.1 },
    { key: "stop_loss_pct", label: "Stop Loss (%)", group: "costs", kind: "number", consumer: "economic gate / exits", step: 0.1 },
    { key: "take_profit_pct", label: "Take Profit (%)", group: "costs", kind: "number", consumer: "economic gate / exits", step: 0.1 },
    { key: "max_slippage_pct", label: "Max Slippage (%)", group: "costs", kind: "number", consumer: "slippage / economic gate", step: 0.1 },
    { key: "baseline_slippage_pct", label: "Baseline Slippage (%)", group: "costs", kind: "number", consumer: "slippage model", step: 0.1 },
    { key: "dynamic_slippage_enabled", label: "Dynamic Slippage", group: "costs", kind: "bool", consumer: "slippage model" },
    { key: "max_price_drift_from_model_pct", label: "Max Price Drift (%)", group: "costs", kind: "number", consumer: "economic gate", step: 0.01 },
    { key: "round_trip_fee_pct", label: "Round-Trip Fee (%)", group: "costs", kind: "number", consumer: "economic gate costs", step: 0.1 },
    { key: "required_margin_after_costs_pct", label: "Required Margin After Costs (%)", group: "costs", kind: "number", consumer: "economic gate", step: 0.01 },
    { key: "min_liquidity_usd", label: "Min Liquidity (USD)", group: "costs", kind: "number", consumer: "live scan / economic gate", step: 100 },
    { key: "max_open_positions", label: "Max Open Positions", group: "costs", kind: "int", consumer: "paper / risk" },
    { key: "duplicate_pair_guard_enabled", label: "Duplicate Pair Guard", group: "costs", kind: "bool", consumer: "paper execution" },
    { key: "llm_enabled_for_demo", label: "LLM Enabled (Demo)", group: "llm", kind: "bool", consumer: "llm_gate" },
    { key: "llm_enabled_for_live", label: "LLM Enabled (Live)", group: "llm", kind: "bool", consumer: "llm_gate" },
    { key: "max_llm_calls_per_hour", label: "Max LLM Calls / Hour", group: "llm", kind: "int", consumer: "llm_gate budget" },
    { key: "max_llm_calls_per_scan", label: "Max LLM Calls / Scan", group: "llm", kind: "int", consumer: "llm_gate budget" },
    { key: "llm_cache_window_minutes", label: "LLM Cache Window (min)", group: "llm", kind: "int", consumer: "llm_gate cache" },
    { key: "live_trading_enabled", label: "LIVE Trading", group: "safety", kind: "bool", consumer: "live execution", readOnly: true },
    { key: "auto_execution_enabled", label: "Auto Execution", group: "safety", kind: "bool", consumer: "live.scan_once" },
    { key: "enforce_risk_gate", label: "Enforce Risk Gate", group: "safety", kind: "bool", consumer: "risk gate" },
    { key: "trading_mode", label: "Trading Mode", group: "safety", kind: "readonly", consumer: "live.scan_once", readOnly: true },
    { key: "mode", label: "Mode Alias", group: "safety", kind: "readonly", consumer: "alias:mode", readOnly: true },
    { key: "prompt_behavior", label: "Prompt Behavior", group: "safety", kind: "select", consumer: "LLM prompts", options: ["conservative", "balanced", "aggressive"] },
    { key: "cooldown_minutes", label: "Cooldown (minutes)", group: "safety", kind: "int", consumer: "risk" },
    { key: "max_daily_loss_pct", label: "Max Daily Loss (%)", group: "safety", kind: "number", consumer: "risk gate", step: 0.1 },
    { key: "max_drawdown_pct", label: "Max Drawdown (%)", group: "safety", kind: "number", consumer: "risk gate", step: 0.1 },
  ];

  const EDITABLE_KEYS = new Set(
    FIELD_SPECS.filter((f) => !f.readOnly && f.kind !== "readonly").map((f) => f.key)
  );

  const INSPECTOR_COLUMNS = [
    "ui_label", "canonical_key", "displayed_value", "internal_value", "unit",
    "source", "default_value", "alias_resolved", "active_status", "backend_consumer", "notes_warnings",
  ];

  const SETTING_ALIASES = {
    minLiquidity: "min_liquidity_usd",
    positionSizePct: "max_position_size_pct",
    stopLossPct: "stop_loss_pct",
    takeProfitPct: "take_profit_pct",
    mode: "trading_mode",
    tradingFee: "paper_fee_bps",
  };

  let loadedEffective = null;
  let loadedCanonical = {};
  let busy = false;

  function internalToDisplay(key, internal) {
    if (internal == null) return "";
    const val = Number(internal);
    if (Number.isNaN(val)) return String(internal);
    if (DECIMAL_FRACTION_PCT_KEYS.has(key) || DISPLAY_AS_PERCENT_POINTS.has(key)) return val * 100;
    return val;
  }

  function displayToInternal(key, display) {
    if (display === "" || display == null) return null;
    const val = Number(display);
    if (Number.isNaN(val)) return null;
    // Form always shows percent-points for these keys (internalToDisplay multiplies by 100).
    // Always divide by 100 on read - do NOT use abs(val)>1 heuristics (breaks values like 0.5%).
    if (DECIMAL_FRACTION_PCT_KEYS.has(key) || DISPLAY_AS_PERCENT_POINTS.has(key)) {
      return Math.round((val / 100) * 1e8) / 1e8;
    }
    return val;
  }

  function formatDisplayValue(key, internal) {
    if (internal == null) return "-";
    if (typeof internal === "boolean") return internal ? "ON" : "OFF";
    if (key === "min_liquidity_usd") return "$" + Number(internal).toLocaleString("en", { maximumFractionDigits: 2 });
    if (DECIMAL_FRACTION_PCT_KEYS.has(key) || DISPLAY_AS_PERCENT_POINTS.has(key)) {
      return (Number(internal) * 100).toPrecision(4) + "%";
    }
    if (typeof internal === "object") return JSON.stringify(internal);
    return String(internal);
  }

  function valuesEqual(a, b) {
    if (a == null && b == null) return true;
    if (typeof a === "boolean" || typeof b === "boolean") return Boolean(a) === Boolean(b);
    const na = typeof a === "number" ? a : (a !== "" && a != null && !Number.isNaN(Number(a)) ? Number(a) : null);
    const nb = typeof b === "number" ? b : (b !== "" && b != null && !Number.isNaN(Number(b)) ? Number(b) : null);
    if (typeof na === "number" && typeof nb === "number" && Number.isFinite(na) && Number.isFinite(nb)) {
      return Math.abs(na - nb) < 1e-8;
    }
    return a === b;
  }

  function buildDirtyPayload(formValues) {
    const dirty = {};
    for (const key of EDITABLE_KEYS) {
      if (!(key in formValues)) continue;
      if (!valuesEqual(formValues[key], loadedCanonical[key])) dirty[key] = formValues[key];
    }
    return dirty;
  }

  /** Dev-only dirty-state field diff for AE13C hotfix debugging. */
  function debugDirtyStateDiff() {
    try {
      const isDev = !!(window.location && /localhost|127\.0\.0\.1/.test(window.location.hostname));
      if (!isDev) return [];
      const current = readFormCanonical();
      const diffs = [];
      for (const key of EDITABLE_KEYS) {
        const original = loadedCanonical[key];
        const cur = current[key];
        if (!valuesEqual(cur, original)) {
          diffs.push({
            field: key,
            original_value: original,
            current_value: cur,
            normalized_original: original,
            normalized_current: cur,
          });
        }
      }
      if (diffs.length) {
        console.debug("[settings dirty-state diff]", diffs);
      }
      window.__settingsDirtyDebug = { at: new Date().toISOString(), diffs, dirty_count: diffs.length };
      return diffs;
    } catch (e) {
      console.warn("[settings dirty debug]", e);
      return [];
    }
  }

  function validateField(key, value, kind) {
    if (kind === "bool") return null;
    if (kind === "int") {
      if (value === "" || value == null) return "Required integer";
      const n = Number(value);
      if (!Number.isFinite(n) || n < 0) return "Must be non-negative integer";
      return null;
    }
    if (kind === "select" || kind === "readonly") return null;
    if (value === "" || value == null) return "Required numeric value";
    const f = Number(value);
    if (!Number.isFinite(f)) return "Invalid number";
    if (key === "min_liquidity_usd" && f < 0) return "Must be non-negative";
    if (DECIMAL_FRACTION_PCT_KEYS.has(key) && (f < 0 || f > 100)) return "Percent should be 0–100";
    if (DISPLAY_AS_PERCENT_POINTS.has(key) && (f < 0 || f > 100)) return "Should be 0–100";
    return null;
  }

  function aliasForCanonical(key, aliases) {
    const parts = [];
    for (const [alias, target] of Object.entries(SETTING_ALIASES)) {
      if (target === key && alias in aliases) parts.push(`${alias}=${JSON.stringify(aliases[alias])}`);
    }
    return parts.join("; ");
  }

  function activeStatus(key, canonical, sources) {
    const val = canonical[key];
    const source = sources[key] || "default";
    if (key === "live_trading_enabled" && !val) return "blocked OFF";
    if (key === "economic_gate_enabled" && !val) return "inactive";
    if (key.startsWith("tab_") && key.endsWith("_enabled") && !val) return "inactive";
    if (source === "default") return "using default";
    if (source.startsWith("alias:")) return "configured (alias resolved)";
    if (source.startsWith("env:")) return "env override active";
    return "active";
  }

  function buildInspectorRows(effective) {
    const canonical = effective.canonical || {};
    const sources = effective.sources || {};
    const defaults = effective.defaults || {};
    const aliases = effective.aliases_resolved || {};
    const specByKey = Object.fromEntries(FIELD_SPECS.map((f) => [f.key, f]));
    const rows = [];

    for (const key of Object.keys(canonical).sort()) {
      const internal = canonical[key];
      const spec = specByKey[key];
      rows.push({
        ui_label: spec ? spec.label : key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        canonical_key: key,
        displayed_value: formatDisplayValue(key, internal),
        internal_value: JSON.stringify(internal),
        unit: DECIMAL_FRACTION_PCT_KEYS.has(key) ? "decimal fraction percent (UI: %)" : typeof internal,
        source: String(sources[key] || "default"),
        default_value: JSON.stringify(defaults[key] ?? ""),
        alias_resolved: aliasForCanonical(key, aliases),
        active_status: activeStatus(key, canonical, sources),
        backend_consumer: spec ? spec.consumer : "",
        notes_warnings: sources[key] === "default" ? "using default" : "",
      });
    }

    const hidden = effective.hidden_thresholds || {};
    for (const [section, block] of Object.entries(hidden).sort()) {
      if (!block || typeof block !== "object") continue;
      const src = block.source || section;
      for (const [k, v] of Object.entries(block)) {
        if (k === "source" || typeof v === "object") continue;
        rows.push({
          ui_label: `${section} / ${k}`,
          canonical_key: `hidden:${section}.${k}`,
          displayed_value: String(v),
          internal_value: JSON.stringify(v),
          unit: typeof v,
          source: String(src),
          default_value: "hard-coded",
          alias_resolved: "",
          active_status: "read-only threshold",
          backend_consumer: String(src),
          notes_warnings: "hidden_threshold - not directly editable",
        });
      }
    }
    return rows;
  }

  function globalWarnings(c) {
    const lines = [];
    const mode = String(c.trading_mode || c.mode || "DEMO").toUpperCase();
    if (!c.economic_gate_enabled) lines.push("Economic gate is OFF - RF/economic approval will not promote candidates.");
    if (!c.paper_trading_enabled && mode === "DEMO") lines.push("Paper trading is OFF - no paper orders will be created.");
    if (!c.live_trading_enabled) lines.push("LIVE trading disabled.");
    if (c.tab_confidence_boost_enabled && !c.economic_gate_enabled) {
      lines.push("TAB configured but inactive because economic gate is OFF.");
    }
    if (c.tab_confidence_boost_enabled_demo && !["DEMO", "PAPER"].includes(mode)) {
      lines.push("TAB DEMO boost is not active in current mode.");
    }
    if (!c.tab_confidence_boost_enabled_live) lines.push("TAB LIVE boost disabled.");
    if (c.llm_enabled_for_live && !c.live_trading_enabled) {
      lines.push("LLM LIVE setting inactive because LIVE trading is disabled.");
    }
    if (!c.llm_enabled_for_demo && !c.llm_enabled_for_live) {
      lines.push("LLM disabled - no Qwen/Ollama/Gemini calls from runtime gate.");
    }
    return lines;
  }

  function setFieldError(key, msg) {
    const el = document.getElementById("sc-" + key);
    const err = document.getElementById("sc-err-" + key);
    if (el) el.classList.toggle("sc-invalid", Boolean(msg));
    if (err) {
      err.textContent = msg || "";
      err.style.display = msg ? "block" : "none";
    }
  }

  function clearAllFieldErrors() {
    FIELD_SPECS.forEach((f) => setFieldError(f.key, ""));
    const top = document.getElementById("sc-top-error");
    if (top) { top.textContent = ""; top.style.display = "none"; }
  }

  function readFormCanonical() {
    const out = {};
    for (const spec of FIELD_SPECS) {
      if (spec.readOnly || spec.kind === "readonly") continue;
      const el = document.getElementById("sc-" + spec.key);
      if (!el) continue;
      if (spec.kind === "bool") out[spec.key] = el.checked;
      else if (spec.kind === "select") out[spec.key] = el.value;
      else if (spec.kind === "int") out[spec.key] = parseInt(el.value, 10);
      else out[spec.key] = displayToInternal(spec.key, el.value);
    }
    return out;
  }

  function setBusy(isBusy, msg) {
    busy = isBusy;
    document.querySelectorAll(".sc-editable").forEach((el) => { el.disabled = isBusy; });
    const saveBtn = document.getElementById("sc-save-btn");
    const discardBtn = document.getElementById("sc-discard-btn");
    const status = document.getElementById("sc-saving-status");
    if (saveBtn) saveBtn.disabled = isBusy || !hasDirty();
    if (discardBtn) discardBtn.disabled = isBusy;
    if (status) status.textContent = msg || "";
  }

  function hasDirty() {
    return Object.keys(buildDirtyPayload(readFormCanonical())).length > 0;
  }

  function updateDirtyLabel() {
    const el = document.getElementById("sc-dirty-label");
    if (!el) return;
    const dirty = hasDirty();
    el.textContent = dirty ? "Unsaved changes" : "No unsaved changes";
    el.style.color = dirty ? "var(--red)" : "var(--green)";
    const saveBtn = document.getElementById("sc-save-btn");
    if (saveBtn && !busy) saveBtn.disabled = !dirty;
    if (dirty) {
      const sig = JSON.stringify(buildDirtyPayload(readFormCanonical()));
      if (window.__settingsDirtySig !== sig) {
        window.__settingsDirtySig = sig;
        debugDirtyStateDiff();
      }
    } else {
      window.__settingsDirtySig = "";
    }
  }

  function applyCanonicalToForm(canonical) {
    for (const spec of FIELD_SPECS) {
      const el = document.getElementById("sc-" + spec.key);
      if (!el) continue;
      const val = canonical[spec.key];
      if (spec.kind === "bool") el.checked = Boolean(val);
      else if (spec.kind === "readonly") el.value = val == null ? "" : String(val);
      else if (spec.kind === "select") el.value = val == null ? "" : String(val);
      else if (spec.kind === "int") el.value = val == null ? 0 : parseInt(val, 10);
      else el.value = internalToDisplay(spec.key, val);
      setFieldError(spec.key, "");
    }
  }

  function renderInspector(effective) {
    const tbody = document.getElementById("sc-inspector-body");
    if (!tbody) return;
    const rows = buildInspectorRows(effective);
    tbody.innerHTML = rows.map((r) =>
      `<tr>${INSPECTOR_COLUMNS.map((c) => `<td>${escapeHtml(r[c] || "")}</td>`).join("")}</tr>`
    ).join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderMeta(effective) {
    const hashEl = document.getElementById("sc-hash-meta");
    const warnEl = document.getElementById("sc-warnings");
    if (hashEl) {
      hashEl.textContent = `settings_hash=${effective.settings_hash || "-"} | timestamp=${effective.timestamp || "-"}`;
      if (effective.audit_report_path) hashEl.title = effective.audit_report_path;
    }
    if (warnEl) {
      const lines = globalWarnings(effective.canonical || {});
      warnEl.innerHTML = lines.map((l) => `<div>${escapeHtml(l)}</div>`).join("");
      warnEl.style.display = lines.length ? "block" : "none";
    }
  }

  async function fetchEffective() {
    const r = await fetch("/api/settings/effective");
    if (!r.ok) throw new Error(await r.text() || r.status);
    return r.json();
  }

  async function patchSettings(dirty) {
    const r = await fetch("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dirty),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      const detail = body.detail || body;
      const err = new Error(detail.message || JSON.stringify(detail));
      err.fieldErrors = detail.field_errors || {};
      throw err;
    }
    return body;
  }

  function bindChangeHandlers() {
    FIELD_SPECS.forEach((spec) => {
      const el = document.getElementById("sc-" + spec.key);
      if (!el || spec.readOnly) return;
      const handler = () => {
        if (busy) return;
        const raw = spec.kind === "bool" ? el.checked : el.value;
        const err = validateField(spec.key, raw, spec.kind);
        setFieldError(spec.key, err);
        updateDirtyLabel();
      };
      el.addEventListener("input", handler);
      el.addEventListener("change", handler);
    });
  }

  function buildFormHtml() {
    const host = document.getElementById("sc-form-sections");
    if (!host) return;
    let html = "";
    for (const groupId of Object.keys(GROUP_TITLES)) {
      html += `<div class="sc-group"><h3>${GROUP_TITLES[groupId]}</h3>`;
      if (groupId === "tab") {
        html += `<p class="sc-note">TAB is an overlay only. TAB can adjust position size after RF/economic approval. TAB cannot create trades by itself.</p>`;
      }
      html += '<div class="sc-fields">';
      for (const spec of FIELD_SPECS.filter((f) => f.group === groupId)) {
        const ro = spec.readOnly ? " disabled" : "";
        const cls = spec.readOnly ? "" : " sc-editable";
        let input = "";
        if (spec.kind === "bool") {
          input = `<label class="toggle"><input type="checkbox" id="sc-${spec.key}" class="${cls.trim()}"${ro}/> ${spec.readOnly ? "OFF (read-only)" : "Enabled"}</label>`;
        } else if (spec.kind === "select") {
          input = `<select id="sc-${spec.key}" class="${cls.trim()}"${ro}>${(spec.options || []).map((o) => `<option value="${o}">${o}</option>`).join("")}</select>`;
        } else if (spec.kind === "readonly") {
          input = `<input type="text" id="sc-${spec.key}" readonly class="sc-readonly"/>`;
        } else if (spec.kind === "int") {
          input = `<input type="number" step="1" min="0" id="sc-${spec.key}" class="${cls.trim()}"${ro}/>`;
        } else {
          input = `<input type="number" step="${spec.step || 0.1}" id="sc-${spec.key}" class="${cls.trim()}"${ro}/>`;
        }
        html += `<div class="sc-field"><label for="sc-${spec.key}">${spec.label}</label><div>${input}<div class="sc-field-error" id="sc-err-${spec.key}"></div></div></div>`;
      }
      html += "</div></div>";
    }
    host.innerHTML = html;
    bindChangeHandlers();
  }

  async function applyEffective(effective) {
    loadedEffective = effective;
    loadedCanonical = { ...(effective.canonical || {}) };
    applyCanonicalToForm(loadedCanonical);
    renderInspector(effective);
    renderMeta(effective);
    updateDirtyLabel();
  }

  window.loadSystemConfig = async function loadSystemConfig() {
    if (busy) return;
    setBusy(true, "Loading effective settings...");
    clearAllFieldErrors();
    try {
      const effective = await fetchEffective();
      await applyEffective(effective);
      const status = document.getElementById("sc-load-status");
      if (status) status.textContent = "Effective settings loaded from backend";
    } catch (e) {
      const top = document.getElementById("sc-top-error");
      if (top) {
        top.textContent = "Failed to load effective settings: " + e.message;
        top.style.display = "block";
      }
      if (loadedEffective) {
        const status = document.getElementById("sc-load-status");
        if (status) status.textContent = "Stale data - refresh failed";
      }
    } finally {
      setBusy(false, "");
    }
  };

  window.discardSystemConfig = function discardSystemConfig() {
    loadSystemConfig();
  };

  window.saveSystemConfig = async function saveSystemConfig() {
    if (busy || !hasDirty()) return;
    const form = readFormCanonical();
    let frontErrors = {};
    for (const spec of FIELD_SPECS) {
      if (spec.readOnly || spec.kind === "readonly") continue;
      const raw = spec.kind === "bool" ? form[spec.key] : document.getElementById("sc-" + spec.key)?.value;
      const err = validateField(spec.key, raw, spec.kind);
      if (err) frontErrors[spec.key] = err;
    }
    if (Object.keys(frontErrors).length) {
      Object.entries(frontErrors).forEach(([k, m]) => setFieldError(k, m));
      const top = document.getElementById("sc-top-error");
      if (top) { top.textContent = "Fix validation errors before saving."; top.style.display = "block"; }
      return;
    }

    const dirty = buildDirtyPayload(form);
    if (!Object.keys(dirty).length) return;

    setBusy(true, "Saving...");
    clearAllFieldErrors();
    const topErr = document.getElementById("sc-top-error");
    if (topErr) { topErr.textContent = ""; topErr.style.display = "none"; }
    try {
      const saved = await patchSettings(dirty);
      // Canonical saved state from backend response - clears Unsaved Changes
      const canonical = { ...(saved.canonical || {}) };
      loadedEffective = saved;
      loadedCanonical = canonical;
      applyCanonicalToForm(loadedCanonical);
      renderInspector(saved);
      renderMeta(saved);
      // Re-read form after apply so dirty compare uses same normalization path
      const after = readFormCanonical();
      loadedCanonical = { ...loadedCanonical, ...Object.fromEntries(
        EDITABLE_KEYS.filter((k) => k in after).map((k) => [k, after[k]])
      ) };
      // Prefer backend canonical for keys present in response
      for (const k of EDITABLE_KEYS) {
        if (k in canonical) loadedCanonical[k] = canonical[k];
      }
      applyCanonicalToForm(loadedCanonical);
      updateDirtyLabel();
      if (hasDirty()) {
        // Final sync: baseline = current form after backend round-trip
        loadedCanonical = { ...readFormCanonical() };
        updateDirtyLabel();
      }
      const status = document.getElementById("sc-saving-status");
      if (status) status.textContent = "Settings saved";
      if (typeof toast === "function") toast("Settings saved");
    } catch (e) {
      setBusy(false, "");
      if (e.fieldErrors) Object.entries(e.fieldErrors).forEach(([k, m]) => setFieldError(k, m));
      const top = document.getElementById("sc-top-error");
      const reason = e.message || "Save failed";
      if (top) {
        top.textContent = "Settings were not saved: " + reason;
        top.style.display = "block";
      }
      if (typeof toast === "function") toast("Settings were not saved: " + reason);
      return;
    }
    setBusy(false, "Settings saved");
  };

  document.addEventListener("DOMContentLoaded", () => {
    buildFormHtml();
  });

  // Test hooks (used by lightweight static checks if needed)
  window.__systemConfigHelpers = {
    internalToDisplay, displayToInternal, buildDirtyPayload, buildInspectorRows,
    EDITABLE_KEYS, FIELD_SPECS, DECIMAL_FRACTION_PCT_KEYS, debugDirtyStateDiff,
  };
})();
