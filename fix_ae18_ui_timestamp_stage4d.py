from pathlib import Path

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_ui_timestamp_stage4d").write_text(js, encoding="utf-8")

# Add canonical market timestamp resolver near cacheBustGetUrl.
if "function marketFreshTimestamp(row)" not in js:
    js = replace_once(
        js,
        '''  function cacheBustGetUrl(endpoint) {
    const sep = String(endpoint || "").includes("?") ? "&" : "?";
    return String(endpoint || "") + sep + "_ts=" + Date.now();
  }
''',
        '''  function cacheBustGetUrl(endpoint) {
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
''',
        "insert canonical market timestamp resolver",
    )

# Conservative broad replacements: any direct timestamp preference should now use resolver.
for old in [
    "r.last_market_update_at || r.fetched_at || r.last_fetched || r.loaded_at || r.updated_at",
    "row.last_market_update_at || row.fetched_at || row.last_fetched || row.loaded_at || row.updated_at",
    "r.fetched_at || r.last_fetched || r.loaded_at || r.updated_at",
    "row.fetched_at || row.last_fetched || row.loaded_at || row.updated_at",
]:
    js = js.replace(old, "marketFreshTimestamp(r)" if old.startswith("r.") else "marketFreshTimestamp(row)")

# If the UI has generic age/time helper calls, make sure market rows can call explicit age label.
# This is intentionally additive and safe.
if "window.marketFreshTimestamp = marketFreshTimestamp;" not in js:
    js = js.replace(
        "  window.refreshRuntimeIndexFromProvider = async function refreshRuntimeIndexFromProvider(opts) {",
        "  window.marketFreshTimestamp = marketFreshTimestamp;\n  window.marketFreshAgeLabel = marketFreshAgeLabel;\n\n  window.refreshRuntimeIndexFromProvider = async function refreshRuntimeIndexFromProvider(opts) {",
        1,
    )

js_path.write_text(js, encoding="utf-8")

idx_path = Path("static/index.html")
idx = idx_path.read_text(encoding="utf-8")
idx_path.with_suffix(".html.bak_ae18_ui_timestamp_stage4d").write_text(idx, encoding="utf-8")

idx = idx.replace(
    "/static/product_demo.js?v=ae18-market-refresh-stage4a",
    "/static/product_demo.js?v=ae18-ui-timestamp-stage4d",
)
idx = idx.replace(
    "/static/product_demo.js?v=ae18-ui-cache-stage2",
    "/static/product_demo.js?v=ae18-ui-timestamp-stage4d",
)
idx = idx.replace(
    "/static/product_demo.js\"></script>",
    "/static/product_demo.js?v=ae18-ui-timestamp-stage4d\"></script>",
)

idx_path.write_text(idx, encoding="utf-8")

print("AE18 stage4D UI timestamp resolver patch applied.")
print("Changed: static/product_demo.js")
print("Changed: static/index.html")
