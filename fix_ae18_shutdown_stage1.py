from pathlib import Path

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"FAILED: could not find block: {label}")
    return text.replace(old, new, 1)

# --------------------------------------------------------------------
# main.py — add MEMECOIN_BACKGROUND_SCANNER_ENABLED=false support
# --------------------------------------------------------------------
main_path = Path("main.py")
main = main_path.read_text(encoding="utf-8")
main_path.with_suffix(".py.bak_ae18_shutdown_stage1").write_text(main, encoding="utf-8")

main = replace_once(
    main,
    'log = logging.getLogger("main")\n\n',
    '''log = logging.getLogger("main")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

''',
    "main env bool helper",
)

main = replace_once(
    main,
    '''    task = asyncio.create_task(watcher_loop())
    get_training_scheduler().start()
    log.info("Startup complete — watcher running in background")
''',
    '''    watcher_enabled = _env_bool("MEMECOIN_BACKGROUND_SCANNER_ENABLED", True)
    task: asyncio.Task | None = None
    if watcher_enabled:
        task = asyncio.create_task(watcher_loop())
        log.info("Startup complete — watcher running in background")
    else:
        log.warning(
            "Startup complete — background watcher disabled by MEMECOIN_BACKGROUND_SCANNER_ENABLED=false"
        )
    get_training_scheduler().start()
''',
    "main startup watcher switch",
)

main = replace_once(
    main,
    '''    get_training_scheduler().stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    log.info("background scanner stopped")
    log.info("Shutdown complete")
''',
    '''    get_training_scheduler().stop()
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        log.info("background scanner stopped")
    else:
        log.info("background scanner was not started")
    log.info("Shutdown complete")
''',
    "main shutdown optional watcher",
)

main_path.write_text(main, encoding="utf-8")

# --------------------------------------------------------------------
# app/live.py — stop scan_once between every major network stage
# --------------------------------------------------------------------
live_path = Path("app/live.py")
live = live_path.read_text(encoding="utf-8")
live_path.with_suffix(".py.bak_ae18_shutdown_stage1").write_text(live, encoding="utf-8")

live = replace_once(
    live,
    '''def _persist_transparency_logs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = get_token_transparency_logs()
    with open(TRANSPARENCY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pair_row(
''',
    '''def _persist_transparency_logs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = get_token_transparency_logs()
    with open(TRANSPARENCY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _shutdown_requested() -> bool:
    try:
        from app.runtime.shutdown import is_shutting_down

        return is_shutting_down()
    except Exception:
        return False


def _pair_row(
''',
    "live shutdown helper",
)

live = replace_once(
    live,
    '''async def scan_once(local_registry: TokenRegistry) -> int:
    global _passed_tokens_log, _dropped_tokens_log, _last_scan_at

    settings = normalize_execution_settings(db.get_settings())
''',
    '''async def scan_once(local_registry: TokenRegistry) -> int:
    global _passed_tokens_log, _dropped_tokens_log, _last_scan_at

    if _shutdown_requested():
        log.info("scan_once skipped because shutdown is active")
        return 0

    settings = normalize_execution_settings(db.get_settings())
''',
    "scan_once start shutdown check",
)

live = replace_once(
    live,
    '''    except Exception as exc:
        log.error("RSS archival failed: %s", exc, exc_info=True)
        sentiment_score = 0.0

    # Priority 0A/0B exact-pair Selected/Clean + open-position mark prices BEFORE trending.
''',
    '''    except Exception as exc:
        log.error("RSS archival failed: %s", exc, exc_info=True)
        sentiment_score = 0.0

    if _shutdown_requested():
        log.info("scan_once stopped after RSS due to shutdown")
        return 0

    # Priority 0A/0B exact-pair Selected/Clean + open-position mark prices BEFORE trending.
''',
    "scan_once after RSS shutdown check",
)

live = replace_once(
    live,
    '''    except Exception as exc:
        log.error("Selected/Clean collection cycle failed: %s", exc, exc_info=True)

    pairs = await get_trending_pairs()
''',
    '''    except Exception as exc:
        log.error("Selected/Clean collection cycle failed: %s", exc, exc_info=True)

    if _shutdown_requested():
        log.info("scan_once stopped before trending due to shutdown")
        return 0

    pairs = await get_trending_pairs()

    if _shutdown_requested():
        log.info("scan_once stopped after trending fetch due to shutdown")
        return 0
''',
    "scan_once before/after trending shutdown check",
)

live = replace_once(
    live,
    '''    for pair in pairs:
        base = pair.get("baseToken") or {}
''',
    '''    for pair in pairs:
        if _shutdown_requested():
            log.info("scan_once pair loop stopped due to shutdown")
            break
        base = pair.get("baseToken") or {}
''',
    "scan_once pair loop shutdown check",
)

live = replace_once(
    live,
    '''    _last_scan_at = datetime.now(timezone.utc).isoformat()
    trader = get_paper_trader()
    trader.set_market_prices(scan_market_entries, price_timestamp=_last_scan_at)

    try:
        await _manage_open_positions(scan_market_entries, cluster_by_pair, sentiment_score, settings)
''',
    '''    if _shutdown_requested():
        log.info("scan_once stopped before position management due to shutdown")
        return len(csv_events)

    _last_scan_at = datetime.now(timezone.utc).isoformat()
    trader = get_paper_trader()
    trader.set_market_prices(scan_market_entries, price_timestamp=_last_scan_at)

    try:
        await _manage_open_positions(scan_market_entries, cluster_by_pair, sentiment_score, settings)
''',
    "scan_once before position management shutdown check",
)

live_path.write_text(live, encoding="utf-8")

# --------------------------------------------------------------------
# app/clean_forward/runtime_selected_collection.py
# — prevent sync exact-pair worker from continuing after shutdown
# --------------------------------------------------------------------
rsc_path = Path("app/clean_forward/runtime_selected_collection.py")
rsc = rsc_path.read_text(encoding="utf-8")
rsc_path.with_suffix(".py.bak_ae18_shutdown_stage1").write_text(rsc, encoding="utf-8")

rsc = replace_once(
    rsc,
    '''DEFAULT_POLICY: dict[str, Any] = {
    "sleep_seconds_between_requests": 0.35,
    "max_concurrency": 1,
    "request_timeout_seconds": 12.0,
    "max_retries_per_target": 2,
    "exponential_backoff_base_seconds": 1.5,
    "exponential_backoff_max_seconds": 30.0,
    "retry_jitter_seconds": 0.15,
    "retry_on_http_status": sorted(RETRY_HTTP_STATUSES),
    "retry_on_timeout": True,
    "retry_on_NO_PAIRS_IN_RESPONSE": False,
    "retry_on_404": False,
}


def utc_now() -> datetime:
''',
    '''DEFAULT_POLICY: dict[str, Any] = {
    "sleep_seconds_between_requests": 0.35,
    "max_concurrency": 1,
    "request_timeout_seconds": 12.0,
    "max_retries_per_target": 2,
    "exponential_backoff_base_seconds": 1.5,
    "exponential_backoff_max_seconds": 30.0,
    "retry_jitter_seconds": 0.15,
    "retry_on_http_status": sorted(RETRY_HTTP_STATUSES),
    "retry_on_timeout": True,
    "retry_on_NO_PAIRS_IN_RESPONSE": False,
    "retry_on_404": False,
}


def _runtime_shutdown_requested() -> bool:
    try:
        from app.runtime.shutdown import is_shutting_down

        return is_shutting_down()
    except Exception:
        return False


def _controlled_shutdown_fetch_result(fetch_url: str, *, started: float | None = None) -> dict[str, Any]:
    base = started if started is not None else time.perf_counter()
    return {
        "fetch_status": "CONTROLLED_SHUTDOWN_SKIP",
        "http_status": "",
        "elapsed_ms": int((time.perf_counter() - base) * 1000),
        "pair": None,
        "raw_text": "",
        "error_reason": "controlled_shutdown",
        "timeout": False,
    }


def _sleep_or_shutdown(seconds: float, sleeper: Callable[[float], None]) -> bool:
    if seconds <= 0:
        return not _runtime_shutdown_requested()
    if _runtime_shutdown_requested():
        return False
    if sleeper is not time.sleep:
        sleeper(seconds)
        return not _runtime_shutdown_requested()
    try:
        from app.runtime.shutdown import get_shutdown_event

        return not get_shutdown_event().wait(seconds)
    except Exception:
        sleeper(seconds)
        return not _runtime_shutdown_requested()


def utc_now() -> datetime:
''',
    "runtime_selected shutdown helpers",
)

rsc = replace_once(
    rsc,
    '''def _one_http_fetch(
    *,
    fetch_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
            resp = client.get(fetch_url, headers=_HEADERS)
''',
    '''def _one_http_fetch(
    *,
    fetch_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if _runtime_shutdown_requested():
        return _controlled_shutdown_fetch_result(fetch_url, started=started)
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
            if _runtime_shutdown_requested():
                return _controlled_shutdown_fetch_result(fetch_url, started=started)
            resp = client.get(fetch_url, headers=_HEADERS)
''',
    "runtime_selected one_http_fetch shutdown check",
)

rsc = replace_once(
    rsc,
    '''    # Explicit exclusions
    if item.get("target_fetch_status_hint") == "SKIPPED_UNRESOLVED_IDENTITY" or (
''',
    '''    if _runtime_shutdown_requested():
        base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
        base.error_reason = "controlled_shutdown"
        base.cooldown_status = "CONTROLLED_SHUTDOWN"
        return base

    # Explicit exclusions
    if item.get("target_fetch_status_hint") == "SKIPPED_UNRESOLVED_IDENTITY" or (
''',
    "runtime_selected fetch_queue_item early shutdown",
)

rsc = replace_once(
    rsc,
    '''        if retry_index == 0:
            sleep_fn(inter_sleep)

        result = _one_http_fetch(fetch_url=fetch_url, timeout_seconds=timeout)
''',
    '''        if retry_index == 0:
            if not _sleep_or_shutdown(inter_sleep, sleep_fn):
                base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
                base.error_reason = "controlled_shutdown"
                base.cooldown_status = "CONTROLLED_SHUTDOWN"
                return base

        if _runtime_shutdown_requested():
            base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
            base.error_reason = "controlled_shutdown"
            base.cooldown_status = "CONTROLLED_SHUTDOWN"
            return base

        result = _one_http_fetch(fetch_url=fetch_url, timeout_seconds=timeout)
''',
    "runtime_selected pre-fetch shutdown check",
)

rsc = replace_once(
    rsc,
    '''        fetch_status = result["fetch_status"]

        retry_scheduled = False
''',
    '''        fetch_status = result["fetch_status"]
        if fetch_status == "CONTROLLED_SHUTDOWN_SKIP":
            final_status = "CONTROLLED_SHUTDOWN_SKIP"
            base.retry_rows.append(
                {
                    "attempt_id": f"{attempt_id}_r{retry_index}",
                    "retry_index": retry_index,
                    "price_source_key": key,
                    "fetch_url": fetch_url,
                    "attempted_at": utc_now_iso(),
                    "fetch_status": fetch_status,
                    "http_status": "",
                    "retry_scheduled": "false",
                    "retry_reason": "controlled_shutdown",
                    "sleep_before_next_attempt_seconds": 0,
                    "backoff_seconds": 0,
                    "elapsed_ms": result.get("elapsed_ms", 0),
                    "final_attempt_for_target": "true",
                }
            )
            break

        retry_scheduled = False
''',
    "runtime_selected controlled fetch status",
)

rsc = replace_once(
    rsc,
    '''            retry_scheduled = True
            sleep_fn(backoff)

        base.retry_rows.append(
''',
    '''            retry_scheduled = True
            if not _sleep_or_shutdown(backoff, sleep_fn):
                final_status = "CONTROLLED_SHUTDOWN_SKIP"
                retry_scheduled = False
                should_retry = False

        base.retry_rows.append(
''',
    "runtime_selected shutdown-aware backoff",
)

rsc = replace_once(
    rsc,
    '''    base.pair_payload = pair_payload if final_status == "SUCCESS" else None

    # Cooldown updates
    if final_status == "SUCCESS":
''',
    '''    base.pair_payload = pair_payload if final_status == "SUCCESS" else None

    if final_status == "CONTROLLED_SHUTDOWN_SKIP":
        base.target_fetch_status = "SKIPPED_CONTROLLED_SHUTDOWN"
        base.cooldown_status = "CONTROLLED_SHUTDOWN"
        base.error_reason = "controlled_shutdown"
        return base

    # Cooldown updates
    if final_status == "SUCCESS":
''',
    "runtime_selected return controlled before cooldown",
)

rsc = replace_once(
    rsc,
    '''    for item in queue:
        rank = item.get("priority_rank")
''',
    '''    for item in queue:
        if _runtime_shutdown_requested():
            break
        rank = item.get("priority_rank")
''',
    "runtime_selected run_priority loop shutdown",
)

rsc_path.write_text(rsc, encoding="utf-8")

# --------------------------------------------------------------------
# Syntax check only
# --------------------------------------------------------------------
import py_compile

for p in [
    Path("main.py"),
    Path("app/live.py"),
    Path("app/clean_forward/runtime_selected_collection.py"),
]:
    py_compile.compile(str(p), doraise=True)
    print(f"OK syntax: {p}")

print("AE18 shutdown stage1 patch applied.")
