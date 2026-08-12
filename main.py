"""
Entry point — starts the FastAPI server and background whale watcher.
Run:  python main.py
      python main.py --mode headless|ollama|gemini
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.env_bootstrap import bootstrap_environment, log_llm_startup_config

_cli_mode = bootstrap_environment(ROOT)

import uvicorn
from app.api import app
from app import database as db
from app.live import watcher_loop
from app.llm_config import get_llm_provider, is_headless_data_collection, is_ollama_provider_active
from app.training.scheduler import get_training_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("main")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@contextlib.asynccontextmanager
async def lifespan(application):
    # ── startup ──────────────────────────────────────────────────────
    from app.runtime.shutdown import request_shutdown, reset_shutdown_for_tests

    reset_shutdown_for_tests()
    db.init_pool()
    log_llm_startup_config(_cli_mode)
    if is_headless_data_collection():
        log.info("Headless Data Collection mode active: Gemini API calls disabled.")
    elif is_ollama_provider_active():
        log.info("Ollama LLM provider active: local Qwen decisions enabled (Gemini disabled).")
    elif get_llm_provider() == "none":
        log.info("LLM_PROVIDER=none: collection running without LLM calls.")
    watcher_enabled = _env_bool("MEMECOIN_BACKGROUND_SCANNER_ENABLED", True)
    task: asyncio.Task | None = None
    if watcher_enabled:
        task = asyncio.create_task(watcher_loop())
        log.info("Startup complete — watcher running in background")
    else:
        log.warning(
            "Startup complete — background watcher disabled by MEMECOIN_BACKGROUND_SCANNER_ENABLED=false"
        )
    get_training_scheduler().start()
    yield
    # ── shutdown ─────────────────────────────────────────────────────
    request_shutdown(reason="uvicorn_lifespan_shutdown")
    log.info("shutdown requested — cancelling background refresh")
    try:
        from app.ae13b_product.demo_bot import get_demo_bot

        get_demo_bot().stop()
    except Exception as exc:
        log.warning("demo bot stop during shutdown: %s", exc)
    get_training_scheduler().stop()
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        log.info("background scanner stopped")
    else:
        log.info("background scanner was not started")
    log.info("Shutdown complete")


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log.info("Starting MemeCoin Whale Trader on port %d", port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        app_dir=str(ROOT),
    )
