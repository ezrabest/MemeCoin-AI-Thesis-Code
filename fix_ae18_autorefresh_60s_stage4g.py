from pathlib import Path
import re

js_path = Path("static/product_demo.js")
js = js_path.read_text(encoding="utf-8")
js_path.with_suffix(".js.bak_ae18_autorefresh_60s_stage4g").write_text(js, encoding="utf-8")

# Change default interval from 90 to 60 seconds.
js = js.replace(
    'const raw = Number(localStorage.getItem(AE18_AUTO_REFRESH_SECONDS_KEY) || "90");',
    'const raw = Number(localStorage.getItem(AE18_AUTO_REFRESH_SECONDS_KEY) || "60");'
)

# Add startup migration: old 90s default or missing value becomes 60s.
migration = r'''
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
'''

if "function ae18EnsureDefaultAutoRefresh60s()" not in js:
    js = js.replace(
        '  function ae18AutoRefreshEnabled() {',
        migration + '\n  function ae18AutoRefreshEnabled() {',
        1
    )

# Ensure migration runs before starting timer.
js = js.replace(
    '''  function ae18StartAutoRefresh() {
    if (ae18AutoRefreshTimer) {
''',
    '''  function ae18StartAutoRefresh() {
    ae18EnsureDefaultAutoRefresh60s();

    if (ae18AutoRefreshTimer) {
''',
    1
)

# Update script version in index.html.
html_path = Path("static/index.html")
html = html_path.read_text(encoding="utf-8")
html_path.with_suffix(".html.bak_ae18_autorefresh_60s_stage4g").write_text(html, encoding="utf-8")

html = re.sub(
    r'/static/product_demo\.js(?:\?v=[^"]*)?',
    '/static/product_demo.js?v=ae18-autorefresh-60s-stage4g',
    html
)

js_path.write_text(js, encoding="utf-8")
html_path.write_text(html, encoding="utf-8")

print("AE18 stage4G auto-refresh default 60s patch applied.")
print("Changed: static/product_demo.js")
print("Changed: static/index.html")
