"""Bounded continuous demo trading bot (paper/demo only).

Writes only through PaperTrader (canonical demo execution).
Does not write via Virtual Ledger View.
Start Demo Bot runs a managed background thread loop until Stop/Pause.

AE13C: WAITING watchdog, realistic hold horizons, strategy lanes,
exception-safe loop, Run One Cycle independent of continuous WAITING.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.ae12_reporting.ascii_text import sanitize_ui_text
from app.ae13b_product.copy import semantic_label_human
from app.ae13b_product.demo_risk_guard import (
    aggregate_rejection_counts,
    format_top_rejection_summary,
)
from app.ae13b_product.execution_guard import (
    DemoExecutionGuardError,
    assert_paper_demo_allowed,
    resolve_runtime_guard_context,
)
from app.ae13b_product.presets import (
    STRATEGY_LANES,
    clamp_trades_per_hour,
    get_preset,
)
from app.ae13_semantic.runtime_registry import get_semantic_registry

log = logging.getLogger("ae13b.demo_bot")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATE_PATH = DATA_DIR / "ae13b_demo_bot_state.json"
ACTIVITY_PATH = DATA_DIR / "ae13b_demo_bot_activity.jsonl"

# POST_STOP_LOSS_BASE_COOLDOWN_V1
def _recent_stop_loss_base_block_v1(base_symbol: str, cooldown_seconds: int = 43200) -> tuple[bool, str]:
    """Block re-entry into a base symbol after a recent paper/demo STOP_LOSS.

    Default: 12h base cooldown after STOP_LOSS.
    This is generic; it is not a PEPE-specific blacklist.
    """
    import json
    from datetime import datetime, timezone

    base = str(base_symbol or "").split("/")[0].upper().strip()
    if not base or base == "UNKNOWN":
        return False, ""

    try:
        path = ACTIVITY_PATH
        if not path.exists():
            return False, ""
        with path.open("rb") as fh:
            try:
                fh.seek(max(0, path.stat().st_size - 2_000_000))
            except Exception:
                pass
            raw = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return False, ""

    now = datetime.now(timezone.utc)

    for line in reversed(raw.splitlines()):
        if '"PAPER_SELL"' not in line or "STOP_LOSS" not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue

        if str(ev.get("event") or "") != "PAPER_SELL":
            continue
        if str(ev.get("exit_reason") or ev.get("reason") or "").upper() != "STOP_LOSS":
            continue

        sold_symbol = str(ev.get("symbol") or "")
        sold_base = sold_symbol.split("/")[0].upper().strip()
        if sold_base != base:
            continue

        ts = ev.get("at") or ev.get("closed_at") or ev.get("timestamp")
        if not ts:
            return True, f"recent_stop_loss_base:{base}"

        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (now - dt).total_seconds()
        except Exception:
            return True, f"recent_stop_loss_base:{base}"

        if age <= cooldown_seconds:
            return True, f"recent_stop_loss_base:{base}:age={int(age)}s:cooldown={cooldown_seconds}s"

    return False, ""


# STRATEGY_ARMS_FRESH_MARKET_UNIVERSE_V1
def _fresh_market_snapshot_candidate_universe_v1(limit: int = 80, max_age_seconds: int = 900, max_per_base_symbol: int = 1) -> list[dict[str, Any]]:
    """Fresh candidate universe for strict/reasoning strategy arms.

    These arms must not use stale legacy_db_coins / old watchlist rows.
    They may only evaluate pairs with a fresh market_snapshots timestamp.
    """
    import sqlite3
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    # MARKET_SNAPSHOT_BASE_DIVERSITY_STATE_V1
    base_counts: dict[str, int] = {}

    try:
        conn = sqlite3.connect(str(DATA_DIR / "trader.db"))
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                ms.id AS market_snapshot_id,
                ms.coin_id,
                ms.timestamp,
                ms.provider,
                ms.chain,
                ms.pair_address,
                ms.price,
                ms.liquidity,
                ms.volume_24h,
                ms.fdv,
                ms.whale_score,
                ms.buy_ratio,
                c.symbol,
                c.name,
                c.token_address,
                c.quote_symbol,
                c.provider_url
            FROM market_snapshots ms
            LEFT JOIN coins c ON c.id = ms.coin_id
            WHERE ms.pair_address IS NOT NULL
              AND ms.price IS NOT NULL
              AND ms.price > 0
              AND ms.liquidity IS NOT NULL
              AND ms.liquidity >= 5000
            ORDER BY ms.timestamp DESC, ms.id DESC
            LIMIT 2000
        """).fetchall()
        conn.close()
    except Exception:
        return []

    for r in rows:
        d = dict(r)
        pair = str(d.get("pair_address") or "").strip()
        ts = d.get("timestamp")
        if not pair or pair in seen or not ts:
            continue

        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (now - dt).total_seconds()
        except Exception:
            continue

        if age < 0 or age > max_age_seconds:
            continue

        symbol = d.get("symbol") or f"{d.get('chain') or 'chain'}/{pair[:6]}"
        price = float(d["price"])
        liq = float(d["liquidity"])

        # MARKET_SNAPSHOT_BASE_DIVERSITY_CAP_V1
        base_symbol = str(symbol).split("/")[0].upper().strip() if symbol else "UNKNOWN"

        # POST_STOP_LOSS_BASE_COOLDOWN_APPLIED_V1
        _loss_blocked, _loss_block_reason = _recent_stop_loss_base_block_v1(base_symbol, cooldown_seconds=43200)
        if _loss_blocked:
            continue

        if base_counts.get(base_symbol, 0) >= int(max_per_base_symbol):
            continue

        seen.add(pair)
        base_counts[base_symbol] = base_counts.get(base_symbol, 0) + 1
        out.append({
            "id": d.get("coin_id"),
            "coin_id": d.get("coin_id"),
            "symbol": symbol,
            "name": d.get("name") or symbol,
            "chain": d.get("chain"),
            "pair_address": pair,
            "token_address": d.get("token_address"),
            "quote_symbol": d.get("quote_symbol"),
            "provider": d.get("provider") or "dexscreener",
            "source_provider": d.get("provider") or "dexscreener",
            "provider_url": d.get("provider_url"),
            "latest_price": price,
            "price_usd": price,
            "price": price,
            "latest_liquidity": liq,
            "liquidity_usd": liq,
            "liquidity": liq,
            "latest_volume_24h": d.get("volume_24h"),
            "volume_24h": d.get("volume_24h"),
            "latest_fdv": d.get("fdv"),
            "latest_whale_score": d.get("whale_score") or 0,
            "whale_score": d.get("whale_score") or 0,
            "buy_ratio": d.get("buy_ratio"),
            "price_updated_at": ts,
            "liquidity_updated_at": ts,
            "last_seen_at": ts,
            "observed_at": ts,
            "market_snapshot_timestamp": ts,
            "fresh_candidate_source": "market_snapshots",
            "fresh_market_snapshot_id": d.get("market_snapshot_id"),
            "fresh_market_age_seconds": age,
        })

        if len(out) >= int(limit):
            break

    return out


# FRESHEN_DEMO_CANDIDATE_FROM_MARKET_SNAPSHOT_V1
def _freshen_demo_candidate_from_latest_snapshot_v1(coin: dict[str, Any]) -> dict[str, Any]:
    """Best-effort enrichment from freshest market_snapshots row.

    Fixes stale candidate payloads where coins.last_seen_at is old while
    market_snapshots has fresh provider-backed price/liquidity/timestamp.
    Does not weaken freshness gates; it supplies the correct timestamp.
    """
    if not isinstance(coin, dict):
        return coin

    pair = str(coin.get("pair_address") or "").strip()
    cid = coin.get("coin_id") if coin.get("coin_id") is not None else coin.get("id")

    where = []
    params = []
    if pair:
        where.append("pair_address = ?")
        params.append(pair)
    if cid is not None:
        try:
            where.append("coin_id = ?")
            params.append(int(cid))
        except Exception:
            pass

    if not where:
        return coin

    sql = f"""
        SELECT id, coin_id, timestamp, provider, chain, pair_address,
               price, liquidity, volume_24h, fdv, whale_score, buy_ratio
        FROM market_snapshots
        WHERE ({' OR '.join(where)})
          AND price IS NOT NULL
          AND price > 0
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
    """

    try:
        import sqlite3
        conn = sqlite3.connect(str(DATA_DIR / "trader.db"))
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, params).fetchone()
        conn.close()
    except Exception:
        return coin

    if not row:
        return coin

    r = dict(row)
    ts = r.get("timestamp")
    price = r.get("price")
    liquidity = r.get("liquidity")

    try:
        price_f = float(price)
    except Exception:
        return coin

    if price_f <= 0:
        return coin

    out = dict(coin)
    out["coin_id"] = out.get("coin_id") or r.get("coin_id")
    out["id"] = out.get("id") or r.get("coin_id")
    out["chain"] = out.get("chain") or r.get("chain")
    out["pair_address"] = out.get("pair_address") or r.get("pair_address")

    out["latest_price"] = price_f
    out["price_usd"] = price_f
    out["price"] = price_f

    if liquidity is not None:
        try:
            liq_f = float(liquidity)
            out["latest_liquidity"] = liq_f
            out["liquidity_usd"] = liq_f
            out["liquidity"] = liq_f
        except Exception:
            pass

    if r.get("volume_24h") is not None:
        out["latest_volume_24h"] = r.get("volume_24h")
        out["volume_24h"] = r.get("volume_24h")

    if ts:
        out["price_updated_at"] = ts
        out["liquidity_updated_at"] = ts
        out["last_seen_at"] = ts
        out["observed_at"] = ts
        out["market_snapshot_timestamp"] = ts

    out["source_provider"] = r.get("provider") or out.get("source_provider") or out.get("provider") or "dexscreener"
    out["provider"] = r.get("provider") or out.get("provider") or "dexscreener"
    out["freshened_from_market_snapshot"] = True
    out["freshened_market_snapshot_id"] = r.get("id")
    return out

_LOCK = threading.RLock()
_INSTANCE: "DemoBot | None" = None

VALID_EXIT_REASONS = frozenset(
    {
        "TAKE_PROFIT",
        "STOP_LOSS",
        "TRAILING_STOP",
        "TIME_STOP",
        "MAX_DRAWDOWN",
        "STALE_PRICE",
        "MANUAL_CLOSE",
        "RISK_LIMIT",
        "DEMO_TEST_ONLY",
        "DEMO_BOT_CLOSE_ALL",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _default_lane_stats() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lane in STRATEGY_LANES:
        out[lane["id"]] = {
            "id": lane["id"],
            "label": lane["label"],
            "enabled": bool(lane.get("enabled_default")),
            "candidates_seen": 0,
            "candidates_selected": 0,
            "trades_opened": 0,
            "blocked_count": 0,
            "last_reason": None,
        }
    return out


def _default_state() -> dict[str, Any]:
    preset = get_preset("balanced")
    return {
        "bot_status": "Stopped",  # Running | Waiting | Paused | Stopped | Blocked | Error | Recovering
        "preset_id": preset["id"],
        "max_open_positions": preset["max_open_positions"],
        "max_trades_per_hour": preset["max_trades_per_hour"],
        "max_notional_usd": preset["max_notional_usd"],
        "cooldown_seconds": preset["cooldown_seconds"],
        "exploration_enabled": preset["exploration_enabled"],
        "demo_acceptance_mode": False,
        "min_hold_seconds": preset["min_hold_seconds"],
        "time_stop_seconds": preset["time_stop_seconds"],
        "take_profit_pct": preset["take_profit_pct"],
        "stop_loss_pct": preset["stop_loss_pct"],
        "trailing_stop_pct": preset["trailing_stop_pct"],
        "expected_hold_profile": preset["expected_hold_profile"],
        # STRICT_CONSENSUS_DEMO_STATE_V1
        "strict_consensus_only": bool(preset.get("strict_consensus_only", False)),
        "strict_consensus_allowed_tiers": list(preset.get("strict_consensus_allowed_tiers") or []),
        "cycles_run": 0,
        "cycles_since_start": 0,
        "trade_attempt_count": 0,
        "trades_opened": 0,
        "trades_closed": 0,
        "last_cycle_at": None,
        "next_cycle_eta": None,
        "waiting_reason": None,
        "waiting_since": None,
        "remaining_seconds": None,
        "last_blocker": None,
        "last_trade_at": None,
        "last_block_reason": None,
        "last_selected_candidate": None,
        "last_error": None,
        "last_action_summary": "Demo bot is stopped. Press Start Demo Bot to begin continuous paper trading.",
        "activity": [],
        "hourly_trade_timestamps": [],
        "loop_active": False,
        "started_at": None,
        "updated_at_utc": _utc_now(),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
        "live_trading": False,
        "wallet_configured": False,
        "strategy_lanes": _default_lane_stats(),
        "stop_event_set": False,
        "lock_held_hint": False,
        "task_alive": False,
        "pair_cooldowns": {},  # pair -> iso timestamp until which re-entry blocked
        "locked_pairs": [],
        "last_cycle_record": None,
        "last_rejection_distribution": [],
        "last_top_rejection_summary": None,
    }


class DemoBot:
    """In-process bounded paper/demo trading controller with continuous background loop."""

    def __init__(self) -> None:
        self._state = self._load()
        # Never resume a prior Running/Waiting state after process restart without explicit Start
        if self._state.get("bot_status") in ("Running", "Waiting", "Blocked", "Recovering", "Error"):
            prev = self._state.get("bot_status")
            self._state["bot_status"] = "Stopped"
            self._state["loop_active"] = False
            self._state["next_cycle_eta"] = None
            self._state["waiting_reason"] = None
            self._state["waiting_since"] = None
            self._state["task_alive"] = False
            self._state["last_action_summary"] = (
                f"Demo bot stopped after process restart (was {prev}). "
                "Press Start Demo Bot to resume."
            )
        self._ensure_lane_stats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop_id = 0
        self._cycle_lock = threading.Lock()  # separate from state lock for run_once vs loop
        self._save()

    def _ensure_lane_stats(self) -> None:
        lanes = dict(self._state.get("strategy_lanes") or {})
        base = _default_lane_stats()
        preset = get_preset(str(self._state.get("preset_id") or "balanced"))
        enabled = set(preset.get("lanes_enabled") or [])
        for lid, row in base.items():
            if lid in lanes and isinstance(lanes[lid], dict):
                merged = {**row, **lanes[lid]}
            else:
                merged = dict(row)
            merged["enabled"] = lid in enabled if enabled else bool(row.get("enabled"))
            base[lid] = merged
        self._state["strategy_lanes"] = base

    def _load(self) -> dict[str, Any]:
        base = _default_state()
        if STATE_PATH.is_file():
            try:
                data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    base.update(data)
            except (OSError, json.JSONDecodeError):
                pass
        return base

    def _save(self) -> None:
        self._state["updated_at_utc"] = _utc_now()
        self._state["task_alive"] = self._loop_is_alive()
        self._state["stop_event_set"] = self._stop_event.is_set() if hasattr(self, "_stop_event") else False
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")

    def _append_activity(self, event: dict[str, Any]) -> None:
        # AE13G: sanitize human-facing summary text at the boundary rather than
        # touching every literal - keeps activity log/API/UI ASCII-safe on Windows.
        if event.get("summary"):
            event = {**event, "summary": sanitize_ui_text(event["summary"])}
        event = {**event, "at": _utc_now(), "paper_demo_only": True, "not_live_approved": True}
        activity = list(self._state.get("activity") or [])
        activity.insert(0, event)
        self._state["activity"] = activity[:100]
        try:
            ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ACTIVITY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            pass

    def apply_preset(self, preset_id: str) -> dict[str, Any]:
        with _LOCK:
            preset = get_preset(preset_id)
            self._state["preset_id"] = preset["id"]
            self._state["max_open_positions"] = preset["max_open_positions"]
            self._state["max_trades_per_hour"] = clamp_trades_per_hour(
                preset["max_trades_per_hour"], default=preset["max_trades_per_hour"]
            )
            self._state["max_notional_usd"] = preset["max_notional_usd"]
            self._state["cooldown_seconds"] = preset["cooldown_seconds"]
            self._state["exploration_enabled"] = preset["exploration_enabled"]
            self._state["demo_acceptance_mode"] = bool(preset.get("demo_acceptance_mode"))
            self._state["min_hold_seconds"] = preset["min_hold_seconds"]
            self._state["time_stop_seconds"] = preset["time_stop_seconds"]
            self._state["take_profit_pct"] = preset["take_profit_pct"]
            self._state["stop_loss_pct"] = preset["stop_loss_pct"]
            self._state["trailing_stop_pct"] = preset["trailing_stop_pct"]
            self._state["expected_hold_profile"] = preset["expected_hold_profile"]
            # STRICT_CONSENSUS_DEMO_APPLY_PRESET_V1
            self._state["strict_consensus_only"] = bool(preset.get("strict_consensus_only", False))
            self._state["strict_consensus_allowed_tiers"] = list(preset.get("strict_consensus_allowed_tiers") or [])
            self._ensure_lane_stats()
            self._state["last_action_summary"] = f"Preset set to {preset['label']}."
            self._save()
            return self.status()

    def set_max_trades_per_hour(self, value: int) -> dict[str, Any]:
        with _LOCK:
            self._state["max_trades_per_hour"] = clamp_trades_per_hour(value)
            self._state["last_action_summary"] = (
                f"Max paper/demo trades per hour set to {self._state['max_trades_per_hour']} "
                "(cap 50)."
            )
            self._save()
            return self.status()

    def _loop_is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _reconcile_stale_waiting(self) -> None:
        """Watchdog: recover if WAITING is stale or loop thread died."""
        status = self._state.get("bot_status")
        alive = self._loop_is_alive()
        cooldown = max(5, int(self._state.get("cooldown_seconds") or 60))
        stale_limit = cooldown * 2

        if status in ("Waiting", "Running", "Recovering") and self._state.get("loop_active") and not alive:
            self._state["bot_status"] = "Recovering"
            self._state["last_blocker"] = "Loop thread not alive while status claimed active"
            self._state["last_error"] = self._state.get("last_error") or "demo_loop_thread_dead"
            self._state["last_action_summary"] = (
                "Demo loop thread died - recovering. Press Start Demo Bot to restart continuous loop, "
                "or Run One Demo Cycle for a single cycle."
            )
            self._state["loop_active"] = False
            self._state["waiting_reason"] = "thread_dead"
            self._append_activity(
                {
                    "event": "WATCHDOG_THREAD_DEAD",
                    "previous_status": status,
                    "reason": "loop_thread_not_alive",
                }
            )
            return

        if status != "Waiting":
            return

        eta = _parse_ts(self._state.get("next_cycle_eta"))
        waiting_since = _parse_ts(self._state.get("waiting_since")) or _parse_ts(
            self._state.get("last_cycle_at")
        )
        now = datetime.now(timezone.utc)
        if waiting_since:
            waited = (now - waiting_since).total_seconds()
            if waited > stale_limit and alive:
                self._state["bot_status"] = "Recovering"
                self._state["last_blocker"] = (
                    f"WAITING exceeded 2x cycle interval ({stale_limit}s) without a cycle"
                )
                self._state["last_action_summary"] = (
                    f"Stale WAITING detected ({int(waited)}s). Loop still alive - forcing recovery to Running."
                )
                self._append_activity(
                    {
                        "event": "WATCHDOG_STALE_WAITING",
                        "waited_seconds": waited,
                        "stale_limit": stale_limit,
                    }
                )
                # Nudge loop by clearing ETA so worker can proceed next tick
                self._state["next_cycle_eta"] = now.isoformat()
                self._state["bot_status"] = "Running"
            elif waited > stale_limit and not alive:
                self._state["bot_status"] = "Error"
                self._state["last_error"] = "stale_waiting_thread_dead"
                self._state["last_blocker"] = self._state["last_error"]
                self._state["last_action_summary"] = (
                    "WAITING stale and loop thread dead. Start Demo Bot or Run One Demo Cycle."
                )
                self._state["loop_active"] = False
                self._append_activity({"event": "WATCHDOG_WAITING_DEADLOCK", "waited_seconds": waited})

        if eta and alive:
            rem = (eta - now).total_seconds()
            self._state["remaining_seconds"] = max(0, int(rem))
        else:
            self._state["remaining_seconds"] = None

    def start(self) -> dict[str, Any]:
        """Start continuous background demo loop (idempotent)."""
        with _LOCK:
            self._reconcile_stale_waiting()
            ctx = resolve_runtime_guard_context()
            if ctx["trading_mode"] not in ("DEMO", "PAPER") or ctx["live_trading_enabled"]:
                self._state["bot_status"] = "Blocked"
                self._state["loop_active"] = False
                self._state["last_block_reason"] = (
                    "Demo bot blocked: system must be in DEMO/PAPER with live trading disabled."
                )
                self._state["last_blocker"] = self._state["last_block_reason"]
                self._state["last_action_summary"] = self._state["last_block_reason"]
                self._save()
                return self.status()

            if self._loop_is_alive() and self._state.get("bot_status") in (
                "Running",
                "Waiting",
                "Paused",
                "Blocked",
                "Recovering",
                "Error",
            ):
                if self._state.get("bot_status") in ("Paused", "Recovering", "Error"):
                    self._state["bot_status"] = "Running"
                    self._state["last_error"] = None
                    self._state["waiting_reason"] = None
                    self._state["last_action_summary"] = (
                        "Demo bot resumed (no duplicate loop)."
                    )
                    self._append_activity({"event": "BOT_RESUMED", "status": "Running", "idempotent": True})
                else:
                    self._state["last_action_summary"] = (
                        "Demo bot already running - duplicate Start ignored (idempotent)."
                    )
                    self._append_activity(
                        {"event": "BOT_START_IDEMPOTENT", "status": self._state.get("bot_status")}
                    )
                self._state["loop_active"] = True
                self._save()
                return self.status()

            # Thread dead but status claimed waiting/running — restart cleanly
            self._stop_event.clear()
            self._loop_id += 1
            loop_id = self._loop_id
            self._state["bot_status"] = "Running"
            self._state["loop_active"] = True
            self._state["cycles_since_start"] = 0
            self._state["started_at"] = _utc_now()
            self._state["last_block_reason"] = None
            self._state["last_blocker"] = None
            self._state["last_error"] = None
            self._state["waiting_reason"] = None
            cooldown = int(self._state.get("cooldown_seconds") or 60)
            self._state["next_cycle_eta"] = (
                datetime.now(timezone.utc) + timedelta(seconds=1)
            ).isoformat()
            self._state["last_action_summary"] = (
                f"Demo bot started continuous paper loop (every ~{cooldown}s). "
                "It stays Running/Waiting until Stop or Pause."
            )
            self._append_activity({"event": "BOT_STARTED", "status": "Running", "loop_id": loop_id})
            self._save()

            self._thread = threading.Thread(
                target=self._loop_worker,
                name=f"ae13b-demo-bot-{loop_id}",
                args=(loop_id,),
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with _LOCK:
            self._stop_event.set()
            self._state["bot_status"] = "Stopped"
            self._state["loop_active"] = False
            self._state["next_cycle_eta"] = None
            self._state["waiting_reason"] = None
            self._state["waiting_since"] = None
            self._state["remaining_seconds"] = None
            self._state["last_action_summary"] = "Demo bot stopped. No new paper trades will be opened."
            self._append_activity({"event": "BOT_STOPPED", "status": "Stopped"})
            self._save()
            return self.status()

    def pause(self) -> dict[str, Any]:
        with _LOCK:
            self._state["bot_status"] = "Paused"
            self._state["loop_active"] = self._loop_is_alive()
            self._state["next_cycle_eta"] = None
            self._state["waiting_reason"] = "paused"
            self._state["waiting_since"] = None
            self._state["remaining_seconds"] = None
            self._state["last_action_summary"] = (
                "Demo bot paused. Continuous loop is idle until Start or Run One Demo Cycle."
            )
            self._append_activity({"event": "BOT_PAUSED", "status": "Paused"})
            self._save()
            return self.status()

    def _loop_worker(self, loop_id: int) -> None:
        log.info("Demo bot continuous loop started (loop_id=%s)", loop_id)
        try:
            while not self._stop_event.is_set():
                try:
                    with _LOCK:
                        if loop_id != self._loop_id:
                            break
                        status = self._state.get("bot_status")
                        cooldown = max(5, int(self._state.get("cooldown_seconds") or 60))

                    if status == "Paused":
                        self._stop_event.wait(1.0)
                        continue
                    if status == "Stopped":
                        break

                    try:
                        self.run_cycle(force=False, from_loop=True)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Demo bot cycle error")
                        with _LOCK:
                            self._state["bot_status"] = "Recovering"
                            self._state["last_error"] = str(exc)
                            self._state["last_blocker"] = f"cycle_exception:{exc}"
                            self._state["last_action_summary"] = f"Error in demo cycle (recovering): {exc}"
                            self._append_activity({"event": "CYCLE_ERROR", "error": str(exc)})
                            self._state["next_cycle_eta"] = (
                                datetime.now(timezone.utc) + timedelta(seconds=cooldown)
                            ).isoformat()
                            self._save()

                    with _LOCK:
                        if self._state.get("bot_status") in ("Running", "Recovering", "Error", "Blocked"):
                            if self._state.get("bot_status") != "Blocked":
                                self._state["bot_status"] = "Waiting"
                                self._state["waiting_reason"] = "cooldown_until_next_cycle"
                                self._state["waiting_since"] = _utc_now()
                                self._state["next_cycle_eta"] = (
                                    datetime.now(timezone.utc) + timedelta(seconds=cooldown)
                                ).isoformat()
                                self._state["remaining_seconds"] = cooldown
                                self._state["last_action_summary"] = (
                                    self._state.get("last_action_summary")
                                    or f"Waiting ~{cooldown}s until next demo cycle."
                                )
                                self._save()

                    deadline = time.monotonic() + cooldown
                    while time.monotonic() < deadline:
                        if self._stop_event.is_set():
                            break
                        with _LOCK:
                            if self._state.get("bot_status") in ("Stopped", "Paused"):
                                break
                            if loop_id != self._loop_id:
                                break
                            eta = _parse_ts(self._state.get("next_cycle_eta"))
                            if eta:
                                rem = (eta - datetime.now(timezone.utc)).total_seconds()
                                self._state["remaining_seconds"] = max(0, int(rem))
                        self._stop_event.wait(0.5)

                    with _LOCK:
                        if (
                            not self._stop_event.is_set()
                            and loop_id == self._loop_id
                            and self._state.get("bot_status")
                            in ("Waiting", "Running", "Error", "Blocked", "Recovering")
                        ):
                            if self._state.get("bot_status") != "Blocked":
                                self._state["bot_status"] = "Running"
                                self._state["waiting_reason"] = None
                                self._state["waiting_since"] = None
                                self._save()
                except Exception as exc:  # noqa: BLE001 — never silently exit loop
                    log.exception("Demo bot outer loop error")
                    with _LOCK:
                        self._state["bot_status"] = "Error"
                        self._state["last_error"] = str(exc)
                        self._state["last_blocker"] = f"loop_exception:{exc}"
                        self._state["last_action_summary"] = f"Loop exception (will retry): {exc}"
                        self._append_activity({"event": "LOOP_EXCEPTION", "error": str(exc)})
                        self._save()
                    self._stop_event.wait(2.0)
        finally:
            with _LOCK:
                if loop_id == self._loop_id:
                    self._state["loop_active"] = False
                    self._state["task_alive"] = False
                    if self._state.get("bot_status") not in ("Stopped", "Paused", "Blocked", "Error"):
                        self._state["bot_status"] = "Stopped"
                    self._state["next_cycle_eta"] = None
                    self._state["waiting_reason"] = None
                    self._save()
            log.info("Demo bot continuous loop exited (loop_id=%s)", loop_id)

    def status(self) -> dict[str, Any]:
        from app.execution.paper import get_paper_trader

        acquired = _LOCK.acquire(timeout=2.0)
        if not acquired:
            return {
                "ok": False,
                "status": "unavailable",
                "user_message": "Demo bot status is busy - retry shortly. Controls remain available.",
                "bot_status": "Unavailable",
                "demo_mode_active": True,
                "live_trading_disabled": True,
                "wallet_not_connected": True,
                "strategy_lanes": [],
                "activity": [],
                "open_positions": [],
                "open_positions_count": 0,
                "wallet": {},
                "what_bot_is_doing": {
                    "what_is_happening": "Status lock busy",
                    "why_traded_or_not": "Another demo cycle holds the status lock briefly.",
                    "next_action": "Retry Refresh; Start/Stop/Run Cycle still work.",
                },
            }
        try:
            self._reconcile_stale_waiting()
            self._state["task_alive"] = self._loop_is_alive()
            self._state["stop_event_set"] = self._stop_event.is_set()
            if self._state.get("bot_status") == "Waiting":
                eta = _parse_ts(self._state.get("next_cycle_eta"))
                if eta:
                    rem = (eta - datetime.now(timezone.utc)).total_seconds()
                    self._state["remaining_seconds"] = max(0, int(rem))
            snap = dict(self._state)
        finally:
            _LOCK.release()

        try:
            trader = get_paper_trader()
            wallet = trader.get_wallet_summary()
            opens = (
                trader.get_marked_positions()
                if hasattr(trader, "get_marked_positions")
                else trader.get_positions(status="OPEN")
            )
        except Exception as exc:  # noqa: BLE001
            wallet = {}
            opens = []
            snap = dict(snap) if isinstance(snap, dict) else {}
            snap["last_error"] = f"wallet_read_error: {exc}"

        return {
            **snap,
            "demo_mode_active": True,
            "live_trading_disabled": True,
            "wallet_not_connected": True,
            "continuous_loop": True,
            "loop_thread_alive": self._loop_is_alive(),
            "task_alive": self._loop_is_alive(),
            "lock_state": "RLock",
            "stop_event_state": self._stop_event.is_set(),
            "waiting": {
                "reason": snap.get("waiting_reason"),
                "next_cycle_eta": snap.get("next_cycle_eta"),
                "remaining_seconds": snap.get("remaining_seconds"),
                "last_cycle_at": snap.get("last_cycle_at"),
                "cycle_count": snap.get("cycles_run"),
                "last_blocker": snap.get("last_blocker") or snap.get("last_block_reason"),
                "thread_alive": self._loop_is_alive(),
                "task_alive": self._loop_is_alive(),
                "lock_state": "RLock",
                "stop_event_set": self._stop_event.is_set(),
                "last_error": snap.get("last_error"),
            },
            "wallet": wallet,
            "open_positions_count": len(opens),
            "open_positions": opens,
            "what_bot_is_doing": self._explain_now(wallet, opens, snap),
            "gemini_label": "AI audit/explanation only - no trade authority",
            "qwen_label": "AI Assistant - explanation only, no trade authority",
            "preset_label": get_preset(str(snap.get("preset_id") or "balanced")).get("label"),
            "strategy_lanes": list((snap.get("strategy_lanes") or {}).values()),
        }

    def events(self, limit: int = 50) -> dict[str, Any]:
        rows = list(self._state.get("activity") or [])[: max(1, min(int(limit), 200))]
        return {"count": len(rows), "events": rows, "paper_demo_only": True}

    def _explain_now(
        self, wallet: dict[str, Any], opens: list[dict[str, Any]], state: dict[str, Any]
    ) -> dict[str, Any]:
        status = state.get("bot_status", "Stopped")
        preset_id = str(state.get("preset_id") or "balanced")
        max_open_positions = int(state.get("max_open_positions") or 0)
        open_positions_count = len(opens)
        available_slots = max(0, max_open_positions - open_positions_count)
        max_open_blocking = max_open_positions > 0 and open_positions_count >= max_open_positions

        cycle_record = state.get("last_cycle_record") or {}
        candidates_seen = int(cycle_record.get("candidates_seen") or 0)
        candidates_seen_in_window = int(cycle_record.get("candidates_seen_in_window") or 0)
        candidates_selected = int(cycle_record.get("candidates_selected") or 0)
        trade_attempts = int(cycle_record.get("trade_attempts") or 0)
        trades_opened = int(cycle_record.get("trades_opened") or 0)
        if cycle_record.get("max_open_blocking"):
            max_open_blocking = True

        top_rejection_reasons = list(
            state.get("last_rejection_distribution")
            or cycle_record.get("rejection_reason_distribution")
            or []
        )
        rejection_summary = state.get("last_top_rejection_summary") or cycle_record.get(
            "top_rejection_summary"
        )

        if status == "Paused":
            why = "The demo bot is paused."
            next_action = "Press Start Demo Bot or Run One Demo Cycle."
            activity_state = "paused"
        elif status == "Stopped":
            why = "The demo bot is stopped."
            next_action = "Press Start Demo Bot for a continuous loop, or Run One Demo Cycle once."
            activity_state = "stopped"
        elif status == "Blocked":
            why = state.get("last_block_reason") or "Demo bot is blocked by a safety gate."
            next_action = "Switch to DEMO mode and keep live trading disabled."
            activity_state = "blocked"
        elif status == "Error":
            why = state.get("last_error") or "Demo bot hit an error."
            next_action = "Check last error, then Start again or Run One Demo Cycle."
            activity_state = "error"
        elif status == "Recovering":
            why = state.get("last_action_summary") or "Recovering from a loop fault."
            next_action = "Start Demo Bot to restart continuous loop, or Run One Demo Cycle."
            activity_state = "recovering"
        elif status == "Waiting":
            rem = state.get("remaining_seconds")
            why = (
                "Cycle complete - waiting for next scheduled demo cycle"
                + (f" (~{rem}s remaining)." if rem is not None else ".")
            )
            next_action = (
                f"Next cycle ETA: {state.get('next_cycle_eta') or 'soon'}. "
                "Run One Demo Cycle still works immediately."
            )
            activity_state = "waiting"
        else:
            why = state.get("last_action_summary") or "Scanning for bounded paper opportunities."
            next_action = "Continuous loop is active; waiting for candidates or next cycle."
            activity_state = "scanning"

        # AE13G: why_traded_or_not must be actionable, never just the generic
        # "open_position_rejected" placeholder. Prefer the structured top-rejection
        # summary whenever a cycle rejected candidates.
        why_traded_or_not = state.get("last_block_reason") or state.get("last_action_summary")
        if why_traded_or_not and "open_position_rejected" in str(why_traded_or_not).lower():
            why_traded_or_not = rejection_summary or why_traded_or_not
        if not why_traded_or_not:
            why_traded_or_not = "No trade attempts recorded yet this session."

        # Distinguish "normal lotto/exploration behavior" (candidates seen but
        # legitimately rejected/no match) from an actual blocker with free slots.
        is_normal_no_trade_behavior = (
            available_slots > 0
            and trades_opened == 0
            and (trade_attempts == 0 or bool(rejection_summary))
            and status not in ("Blocked", "Error")
        )

        return {
            "bot_status": status,
            "what_is_happening": sanitize_ui_text(why),
            "why_traded_or_not": sanitize_ui_text(why_traded_or_not),
            "next_action": sanitize_ui_text(next_action),
            "next_cycle_eta": state.get("next_cycle_eta"),
            "remaining_seconds": state.get("remaining_seconds"),
            "waiting_reason": state.get("waiting_reason"),
            "last_cycle_at": state.get("last_cycle_at"),
            "cycles_since_start": state.get("cycles_since_start"),
            "trade_attempt_count": state.get("trade_attempt_count"),
            "open_positions": open_positions_count,
            "cash_usd": wallet.get("cash_usd"),
            "equity_usd": wallet.get("total_equity_usd"),
            "thread_alive": self._loop_is_alive(),
            # AE13G explainability fields
            "preset_id": preset_id,
            "open_positions_count": open_positions_count,
            "max_open_positions": max_open_positions,
            "available_slots": available_slots,
            "max_open_blocking": max_open_blocking,
            "candidates_seen": candidates_seen,
            "candidates_seen_in_window": candidates_seen_in_window,
            "candidates_selected": candidates_selected,
            "trade_attempts": trade_attempts,
            "trades_opened": trades_opened,
            "top_rejection_reasons": top_rejection_reasons,
            "rejection_summary": sanitize_ui_text(rejection_summary),
            "activity_state": activity_state,
            "is_normal_no_trade_behavior": is_normal_no_trade_behavior,
        }

    def _trades_in_last_hour(self) -> int:
        now = datetime.now(timezone.utc)
        kept = []
        for ts in self._state.get("hourly_trade_timestamps") or []:
            t = _parse_ts(ts)
            if t and (now - t).total_seconds() <= 3600:
                kept.append(ts)
        self._state["hourly_trade_timestamps"] = kept
        return len(kept)

    def _order_flags(self, *, acceptance: bool = False) -> dict[str, Any]:
        flags = {
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        }
        if acceptance:
            flags.update(
                {
                    "demo_acceptance_only": True,
                    "not_strategy_evidence": True,
                }
            )
        return flags

    def run_once(self) -> dict[str, Any]:
        """Run exactly one cycle immediately — independent of continuous WAITING."""
        return self.run_cycle(force=True, from_loop=False)

    def run_cycle(self, *, force: bool = False, from_loop: bool = False) -> dict[str, Any]:
        """Run one bounded demo cycle: observe candidates, maybe buy/sell paper positions."""
        # Allow Run One Cycle even while continuous loop is Waiting (do not nest under loop cycle)
        acquired = self._cycle_lock.acquire(timeout=45.0 if force else 5.0)
        if not acquired:
            with _LOCK:
                self._state["last_blocker"] = "cycle_lock_busy"
                self._state["last_action_summary"] = (
                    "Another demo cycle is already running - try again in a moment."
                )
                self._save()
                return {
                    "ok": False,
                    "error": "cycle_lock_busy",
                    "status": self.status(),
                }
        try:
            return self._run_cycle_locked(force=force, from_loop=from_loop)
        finally:
            self._cycle_lock.release()

    def _run_cycle_locked(self, *, force: bool, from_loop: bool) -> dict[str, Any]:
        cycle_id = str(uuid.uuid4())[:12]
        started = datetime.now(timezone.utc)
        with _LOCK:
            ctx = resolve_runtime_guard_context()
            acceptance = bool(self._state.get("demo_acceptance_mode"))
            flags = self._order_flags(acceptance=acceptance)
            try:
                assert_paper_demo_allowed(
                    trading_mode=ctx["trading_mode"],
                    live_trading_enabled=ctx["live_trading_enabled"],
                    wallet_configured=False,
                    order_flags=flags,
                    demo_acceptance_mode_enabled=acceptance if acceptance else None,
                )
            except DemoExecutionGuardError as exc:
                self._state["bot_status"] = "Blocked"
                reason = exc.reasons[0] if exc.reasons else str(exc)
                self._state["last_block_reason"] = f"Blocked - {reason}"
                self._state["last_blocker"] = reason
                self._state["last_action_summary"] = self._state["last_block_reason"]
                self._append_activity({"event": "CYCLE_BLOCKED", "reasons": exc.reasons, "cycle_id": cycle_id})
                self._save()
                return {"ok": False, "status": self.status(), "error": str(exc), "reasons": exc.reasons}

            if not force and self._state.get("bot_status") in ("Paused", "Stopped"):
                self._state["last_action_summary"] = (
                    "No trade: demo bot is paused/stopped. Start the bot or run one cycle."
                )
                self._save()
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": self._state.get("bot_status"),
                    "status": self.status(),
                }

            # Snapshot config under lock, then release for I/O
            preset_id = str(self._state.get("preset_id") or "balanced")
            max_open_positions = int(self._state.get("max_open_positions") or 3)
            max_trades_per_hour = clamp_trades_per_hour(self._state.get("max_trades_per_hour"), default=12)
            max_notional_usd = float(self._state.get("max_notional_usd") or 75)
            cooldown_seconds = int(self._state.get("cooldown_seconds") or 60)
            locked_pairs = list(self._state.get("locked_pairs") or [])
            config = {
                "preset_id": preset_id,
                "max_open": max_open_positions,
                "max_per_hour": max_trades_per_hour,
                "notional": max_notional_usd,
                "exploration": bool(self._state.get("exploration_enabled")),
                "acceptance": acceptance,
                "min_hold": int(self._state.get("min_hold_seconds") or 300),
                "time_stop": int(self._state.get("time_stop_seconds") or 21600),
                "tp": float(self._state.get("take_profit_pct") or 0.18),
                "sl": float(self._state.get("stop_loss_pct") or 0.08),
                "trail": float(self._state.get("trailing_stop_pct") or 0.10),
                "hold_profile": self._state.get("expected_hold_profile"),
                # STRICT_CONSENSUS_DEMO_CONFIG_V1
                "strict_consensus_only": bool(self._state.get("strict_consensus_only")),
                "strict_consensus_allowed_tiers": list(self._state.get("strict_consensus_allowed_tiers") or []),
                "lanes": dict(self._state.get("strategy_lanes") or {}),
                "pair_cooldowns": dict(self._state.get("pair_cooldowns") or {}),
                # AE13G: bot_state passed through to PaperTrader.open_position() / the
                # risk guard so the *active* preset (e.g. lotto max_open=8) is honored
                # instead of the risk guard's hardcoded default (max_open=6).
                "bot_state": {
                    "preset_id": preset_id,
                    "risk_mode": preset_id,
                    "max_open_positions": max_open_positions,
                    "max_trades_per_hour": max_trades_per_hour,
                    "max_notional_usd": max_notional_usd,
                    "cooldown_seconds": cooldown_seconds,
                    "locked_pairs": locked_pairs,
                },
            }

        from app import database as db
        from app.execution.paper import get_paper_trader

        trader = get_paper_trader()
        try:
            if str(trader.get_wallet_summary().get("trading_mode") or "").upper() not in (
                "DEMO",
                "PAPER",
            ):
                trader.set_trading_mode("DEMO")
        except Exception:
            pass

        settings = ctx.get("settings") or db.get_settings()
        registry = get_semantic_registry()

        # AE14: for Run One Cycle / cycle?force=true / AE14 closure mode, use
        # Clean Forward Market Feed only. Never fall back to market_snapshots,
        # old watchlist, or local DB candidate universe while AE14 policy is on.
        from app.ae13b_product.ae14_candidate_source_policy import (
            AE14_CANDIDATE_SOURCE_POLICY,
            CANDIDATE_SOURCE as AE14_CANDIDATE_SOURCE,
            is_ae14_closure_mode,
            is_valid_ae14_clean_forward_row,
            requires_clean_forward_only,
        )

        clean_forward_bridge_used = False
        legacy_market_snapshots_used = False
        old_watchlist_candidates_used = False
        local_db_candidate_universe_used = False
        fresh_market_snapshot_universe_used = False
        clean_forward_rows_seen = 0
        clean_forward_candidates_selected = 0
        clean_forward_bridge_pass_count = 0
        clean_forward_bridge_block_count = 0
        gatekeeper_pass_count = 0
        gatekeeper_block_count = 0
        coins: list[dict[str, Any]] = []
        use_clean_forward_path = False
        ae14_mode = bool(force or is_ae14_closure_mode() or requires_clean_forward_only())

        if ae14_mode:
            try:
                from app.ae13b_product.clean_forward_bridge import (
                    build_clean_forward_gatekeeper_candidate,
                )
                from app.ae13b_product.clean_forward_market_feed import (
                    get_cached_clean_forward_rows,
                )

                cf_rows = get_cached_clean_forward_rows()
                clean_forward_rows_seen = len(cf_rows)
                normalized_cf: list[dict[str, Any]] = []
                for row in cf_rows:
                    if is_ae14_closure_mode() and not is_valid_ae14_clean_forward_row(row):
                        clean_forward_bridge_block_count += 1
                        continue
                    bridge = build_clean_forward_gatekeeper_candidate(row)
                    if bridge.get("ok") and isinstance(bridge.get("candidate"), dict):
                        clean_forward_bridge_pass_count += 1
                        cand = dict(bridge["candidate"])
                        # Do not invent coin_id — paper fill resolves by pair_address.
                        cand["id"] = None
                        cand["coin_id"] = None
                        cand["candidate_source"] = AE14_CANDIDATE_SOURCE
                        cand["ae14_candidate_source_policy"] = AE14_CANDIDATE_SOURCE_POLICY
                        normalized_cf.append(cand)
                    else:
                        clean_forward_bridge_block_count += 1
                if cf_rows or requires_clean_forward_only():
                    # Rows present OR AE14 closure: stay on Clean Forward path.
                    # Do not fall back to market_snapshots/DB.
                    coins = normalized_cf
                    use_clean_forward_path = True
                    clean_forward_bridge_used = True
                    legacy_market_snapshots_used = False
                    old_watchlist_candidates_used = False
                    local_db_candidate_universe_used = False
                    clean_forward_candidates_selected = len(normalized_cf)
            except Exception as exc:  # noqa: BLE001
                log.warning("AE14 clean-forward candidate load failed: %s", exc)
                if requires_clean_forward_only():
                    with _LOCK:
                        self._state["last_block_reason"] = (
                            f"AE14 Clean Forward candidate load failed ({exc})"
                        )
                        self._state["last_blocker"] = "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE"
                        self._state["last_action_summary"] = self._state["last_block_reason"]
                        self._state["cycles_run"] = int(self._state.get("cycles_run") or 0) + 1
                        self._state["last_cycle_at"] = _utc_now()
                        self._save()
                    return {
                        "ok": False,
                        "observed": 0,
                        "opened": None,
                        "closed": [],
                        "status": self.status(),
                        "error": str(exc),
                        "cycle_id": cycle_id,
                        "blocker": "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE",
                        "clean_forward_bridge_used": True,
                        "legacy_market_snapshots_used": False,
                        "old_watchlist_candidates_used": False,
                        "local_db_candidate_universe_used": False,
                        "candidate_source": AE14_CANDIDATE_SOURCE,
                        "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                    }

        if not use_clean_forward_path:
            # STRATEGY_ARMS_EARLY_FRESH_UNIVERSE_BRANCH_V2
            _strategy_arm_id_v2 = str(config.get("preset_id") or "").lower().strip()
            if _strategy_arm_id_v2 in {"strict_consensus", "reasoning_demo"}:
                coins = _fresh_market_snapshot_candidate_universe_v1(limit=80, max_age_seconds=900, max_per_base_symbol=1)
                fresh_market_snapshot_universe_used = True
                legacy_market_snapshots_used = False
                old_watchlist_candidates_used = False
                local_db_candidate_universe_used = False
                if not coins:
                    with _LOCK:
                        self._state["last_blocker"] = "no_fresh_market_snapshot_candidate"
                        self._state["last_block_reason"] = "No fresh market_snapshot candidate available for strategy arm."
            else:
                if requires_clean_forward_only():
                    with _LOCK:
                        self._state["last_block_reason"] = "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE"
                        self._state["last_blocker"] = "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE"
                        self._state["last_action_summary"] = (
                            "AE14 closure mode: Clean Forward Market Feed required; "
                            "legacy market_snapshots / watchlist / DB coins forbidden."
                        )
                        self._state["cycles_run"] = int(self._state.get("cycles_run") or 0) + 1
                        self._state["last_cycle_at"] = _utc_now()
                        self._save()
                    return {
                        "ok": False,
                        "observed": 0,
                        "opened": None,
                        "closed": [],
                        "status": self.status(),
                        "blocker": "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE",
                        "cycle_id": cycle_id,
                        "clean_forward_bridge_used": True,
                        "legacy_market_snapshots_used": False,
                        "old_watchlist_candidates_used": False,
                        "local_db_candidate_universe_used": False,
                        "candidate_source": AE14_CANDIDATE_SOURCE,
                        "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                    }
                try:
                    coins = db.get_coins(limit=40, sort_by="whale_score")
                    local_db_candidate_universe_used = True
                except Exception as exc:
                    with _LOCK:
                        self._state["bot_status"] = "Error" if from_loop else self._state.get("bot_status", "Error")
                        self._state["last_error"] = str(exc)
                        self._state["last_block_reason"] = (
                            f"No market data available yet ({exc}). Waiting for scanner / DB."
                        )
                        self._state["last_blocker"] = self._state["last_block_reason"]
                        self._state["last_action_summary"] = self._state["last_block_reason"]
                        self._state["cycles_run"] = int(self._state.get("cycles_run") or 0) + 1
                        self._state["last_cycle_at"] = _utc_now()
                        self._append_activity(
                            {"event": "CYCLE_NO_MARKET_DATA", "error": str(exc), "cycle_id": cycle_id}
                        )
                        self._save()
                        return {
                            "ok": True,
                            "observed": 0,
                            "opened": None,
                            "closed": [],
                            "status": self.status(),
                            "error": str(exc),
                            "cycle_id": cycle_id,
                            "clean_forward_bridge_used": False,
                            "legacy_market_snapshots_used": False,
                        }

            # Match watchlist into registry (legacy path only — CF path is pair-verified)
            if not use_clean_forward_path:
                try:
                    from app.analytics.watchlist import list_watchlist, refresh_watchlist_against_market

                    refresh_watchlist_against_market(coins)
                    old_watchlist_candidates_used = True
                except Exception:
                    pass

        observed = []
        for c in coins:
            cand = {
                "id": c.get("id"),
                "coin_id": c.get("id") or c.get("coin_id"),
                "symbol": c.get("symbol"),
                "name": c.get("name") or c.get("base_token_name"),
                "chain": c.get("chain"),
                "pair_address": c.get("pair_address"),
                "price_usd": c.get("latest_price") or c.get("price_usd"),
                "latest_price": c.get("latest_price") or c.get("price_usd"),
                "liquidity_usd": c.get("latest_liquidity") or c.get("liquidity_usd"),
                "latest_liquidity": c.get("latest_liquidity") or c.get("liquidity_usd"),
                "volume_24h": c.get("latest_volume_24h") or c.get("volume_24h"),
                "latest_volume_24h": c.get("latest_volume_24h") or c.get("volume_24h"),
                "whale_score": c.get("latest_whale_score"),
                "cluster_label": c.get("cluster_label"),
                "price_change_5m": c.get("price_change_5m"),
                "price_change_1h": c.get("price_change_1h"),
                "price_change_24h": c.get("price_change_24h"),
                "buy_ratio": c.get("buy_ratio"),
                "candidate_source": c.get("candidate_source"),
            }
            observed.append(registry.observe_candidate(cand))

        closes = self._maybe_close_positions(trader, settings, config)
        opens = trader.get_positions(status="OPEN")
        open_count = len(opens)
        trades_hour = 0
        with _LOCK:
            trades_hour = self._trades_in_last_hour()

        blockers: list[str] = []
        buy_result = None
        candidates_selected = 0
        candidates_seen_in_window = 0
        trade_attempts = 0
        rejected_attempts: list[dict[str, Any]] = []
        rejection_distribution: list[dict[str, Any]] = []
        top_rejection_summary: str | None = None
        max_open_blocking = False
        open_slots_available = max(0, config["max_open"] - open_count)

        if config["acceptance"]:
            blockers.append("demo_acceptance_mode_active_not_strategy")
            # Acceptance mode may still open tiny lifecycle trades with DEMO_TEST_ONLY
            buy_result = self._maybe_open_acceptance(trader, settings, coins, config)
        elif open_count >= config["max_open"]:
            blockers.append(f"max_open_positions:{config['max_open']}")
            max_open_blocking = True
        elif trades_hour >= config["max_per_hour"]:
            blockers.append(f"hourly_trade_limit:{config['max_per_hour']}")
        elif not coins:
            blockers.append(
                "no_clean_forward_candidates" if use_clean_forward_path else "no_market_coins"
            )
        else:
            buy_result = self._maybe_open_position(
                trader=trader,
                settings=settings,
                coins=coins,
                observed=observed,
                opens=opens,
                config=config,
                allow_missing_coin_id=use_clean_forward_path,
            )
            if buy_result:
                candidates_selected = int(buy_result.get("candidates_selected") or 0)
                candidates_seen_in_window = int(buy_result.get("candidates_seen_in_window") or 0)
                trade_attempts = int(buy_result.get("trade_attempts") or 0)
                rejected_attempts = list(buy_result.get("rejected_attempts") or [])
                rejection_distribution = list(buy_result.get("rejection_reason_distribution") or [])
                top_rejection_summary = buy_result.get("top_rejection_summary")
                max_open_blocking = bool(buy_result.get("max_open_blocking"))
                open_slots_available = int(
                    buy_result.get("open_slots_available")
                    if buy_result.get("open_slots_available") is not None
                    else open_slots_available
                )
                gatekeeper_pass_count = int(buy_result.get("gatekeeper_pass_count") or 0)
                gatekeeper_block_count = int(buy_result.get("gatekeeper_block_count") or 0)

        with _LOCK:
            self._state["cycles_run"] = int(self._state.get("cycles_run") or 0) + 1
            if from_loop or self._state.get("loop_active"):
                self._state["cycles_since_start"] = int(self._state.get("cycles_since_start") or 0) + 1
            self._state["last_cycle_at"] = _utc_now()
            self._state["last_error"] = None
            # AE13G: always refresh explainability state, even on a successful open,
            # so the UI/API can show what happened in prior rejected attempts too.
            self._state["last_rejection_distribution"] = rejection_distribution
            self._state["last_top_rejection_summary"] = top_rejection_summary

            if buy_result and buy_result.get("opened"):
                self._state["last_block_reason"] = None
                self._state["last_blocker"] = None
                self._state["last_action_summary"] = buy_result.get("summary")
                self._state["trades_opened"] = int(self._state.get("trades_opened") or 0) + 1
                self._state["last_trade_at"] = _utc_now()
                stamps = list(self._state.get("hourly_trade_timestamps") or [])
                stamps.append(_utc_now())
                self._state["hourly_trade_timestamps"] = stamps
            elif closes:
                self._state["last_action_summary"] = (
                    f"Closed {len(closes)} demo position(s). "
                    + (self._state.get("last_block_reason") or "Holding remaining open positions.")
                )
            elif blockers:
                reason = "No new buy: " + "; ".join(blockers)
                self._state["last_block_reason"] = reason
                self._state["last_blocker"] = blockers[0]
                self._state["last_action_summary"] = reason
            elif buy_result and buy_result.get("summary"):
                # Prefer the structured top-rejection summary when candidates were
                # rejected while open slots were still available — this is the
                # explainable fix for the "6/8 paradox" (opens blocked below the
                # active preset's max_open even though slots remained).
                if top_rejection_summary and open_slots_available > 0:
                    self._state["last_block_reason"] = top_rejection_summary
                    self._state["last_blocker"] = buy_result.get("last_block") or buy_result.get("blocker")
                    self._state["last_action_summary"] = top_rejection_summary
                else:
                    self._state["last_block_reason"] = buy_result.get("summary")
                    self._state["last_blocker"] = (
                        buy_result.get("last_block") or buy_result.get("blocker") or buy_result.get("summary")
                    )
                    self._state["last_action_summary"] = buy_result.get("summary")
            elif not self._state.get("last_block_reason"):
                self._state["last_action_summary"] = (
                    "Cycle complete - no new paper trade this round "
                    "(candidates watched; strategy lanes did not select a buy)."
                )

            completed = datetime.now(timezone.utc)
            duration_ms = int((completed - started).total_seconds() * 1000)
            paper_orders_opened = 1 if buy_result and buy_result.get("opened") else 0
            cycle_record = {
                "cycle_id": cycle_id,
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
                "duration_ms": duration_ms,
                "mode": "acceptance" if config["acceptance"] else "strategy",
                "candidates_seen": len(observed),
                "candidates_seen_in_window": candidates_seen_in_window,
                "candidates_selected": candidates_selected,
                # Cycle-local trade_attempts ONLY — never falls back to the lifetime
                # trade_attempt_count (that was part of the AE13G "6/8 paradox" bug).
                "trade_attempts": trade_attempts,
                "trades_opened": paper_orders_opened,
                "trades_closed": len(closes),
                "blockers": blockers,
                "rejected_attempts": rejected_attempts,
                "rejection_reason_distribution": rejection_distribution,
                "top_rejection_summary": top_rejection_summary,
                "max_open_blocking": max_open_blocking,
                "open_slots_available": open_slots_available,
                "max_open_positions": config["max_open"],
                "open_positions_count": open_count,
                "next_cycle_eta": self._state.get("next_cycle_eta"),
                "status_after_cycle": self._state.get("bot_status"),
                "error": None,
                "from_loop": from_loop,
                "force_once": force and not from_loop,
                # AE14 clean-forward audit
                "clean_forward_rows_seen": clean_forward_rows_seen,
                "clean_forward_candidates_selected": clean_forward_candidates_selected,
                "clean_forward_bridge_pass_count": clean_forward_bridge_pass_count,
                "clean_forward_bridge_block_count": clean_forward_bridge_block_count,
                "gatekeeper_pass_count": gatekeeper_pass_count,
                "gatekeeper_block_count": gatekeeper_block_count,
                "paper_orders_opened": paper_orders_opened,
                "paper_positions_opened": paper_orders_opened,
                "paper_positions_closed": len(closes),
                "legacy_market_snapshots_used": legacy_market_snapshots_used,
                "old_watchlist_candidates_used": old_watchlist_candidates_used,
                "local_db_candidate_universe_used": local_db_candidate_universe_used,
                "fresh_market_snapshot_universe_used": fresh_market_snapshot_universe_used,
                "clean_forward_bridge_used": clean_forward_bridge_used,
                "candidate_source": (
                    AE14_CANDIDATE_SOURCE
                    if clean_forward_bridge_used
                    else ("market_snapshots_fresh" if fresh_market_snapshot_universe_used else "legacy_db_coins")
                ),
                "ae14_candidate_source_policy": (
                    AE14_CANDIDATE_SOURCE_POLICY if clean_forward_bridge_used else None
                ),
                "live_trading_ready": False,
                "paper_demo_only": True,
                "not_profitability_evidence": True,
            }
            self._state["last_cycle_record"] = cycle_record

            if force and not from_loop:
                # Run One Cycle must not flip continuous loop status incorrectly
                pass
            elif self._state.get("bot_status") not in ("Paused", "Stopped", "Blocked"):
                if from_loop:
                    self._state["bot_status"] = "Running"

            self._append_activity(
                {
                    "event": "CYCLE_COMPLETE",
                    **cycle_record,
                    "summary": self._state.get("last_action_summary"),
                    "blocker": self._state.get("last_blocker"),
                }
            )
            self._save()
            return {
                "ok": True,
                "cycle_id": cycle_id,
                "observed": len(observed),
                "opened": buy_result,
                "closed": closes,
                "blockers": blockers,
                "status": self.status(),
                "clean_forward_bridge_used": clean_forward_bridge_used,
                "legacy_market_snapshots_used": legacy_market_snapshots_used,
                "old_watchlist_candidates_used": old_watchlist_candidates_used,
                "local_db_candidate_universe_used": local_db_candidate_universe_used,
                "fresh_market_snapshot_universe_used": fresh_market_snapshot_universe_used,
                "candidate_source": (
                    AE14_CANDIDATE_SOURCE
                    if clean_forward_bridge_used
                    else ("market_snapshots_fresh" if fresh_market_snapshot_universe_used else "legacy_db_coins")
                ),
                "ae14_candidate_source_policy": (
                    AE14_CANDIDATE_SOURCE_POLICY if clean_forward_bridge_used else None
                ),
                "clean_forward_rows_seen": clean_forward_rows_seen,
                "clean_forward_candidates_selected": clean_forward_candidates_selected,
                "clean_forward_bridge_pass_count": clean_forward_bridge_pass_count,
                "clean_forward_bridge_block_count": clean_forward_bridge_block_count,
                "gatekeeper_pass_count": gatekeeper_pass_count,
                "gatekeeper_block_count": gatekeeper_block_count,
                "paper_orders_opened": paper_orders_opened,
                "paper_positions_opened": paper_orders_opened,
                "paper_positions_closed": len(closes),
                "live_trading_ready": False,
                "paper_demo_only": True,
                "not_profitability_evidence": True,
            }

    def _lane_bump(self, lane_id: str, **fields: Any) -> None:
        lanes = self._state.setdefault("strategy_lanes", _default_lane_stats())
        row = lanes.setdefault(lane_id, {"id": lane_id, "enabled": True})
        for k, v in fields.items():
            if k.endswith("_count") or k in (
                "candidates_seen",
                "candidates_selected",
                "trades_opened",
                "blocked_count",
            ):
                row[k] = int(row.get(k) or 0) + int(v)
            else:
                row[k] = v
        lanes[lane_id] = row

    def _pick_lane(self, coin: dict[str, Any], obs: dict[str, Any] | None, config: dict[str, Any]) -> str | None:
        lanes = config.get("lanes") or {}
        family = str((obs or {}).get("semantic_signal_family") or "")
        whale = float(coin.get("latest_whale_score") or coin.get("whale_score") or 0)
        chg_1h = float(coin.get("price_change_1h") or 0)
        liq = float(coin.get("latest_liquidity") or coin.get("liquidity_usd") or 0)
        symbol = str(coin.get("symbol") or "").upper()

        # Watchlist match
        try:
            from app.analytics.watchlist import match_market_to_watchlist

            if match_market_to_watchlist(coin) and lanes.get("manual_watchlist_scout", {}).get("enabled", True):
                return "manual_watchlist_scout"
        except Exception:
            pass

        opportunistic = family in (
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            "OPPORTUNISTIC_SUSPECTED",
            "OPPORTUNISTIC_CONFIRMED",
        )
        if opportunistic and lanes.get("meme_opportunistic_scout", {}).get("enabled", True):
            if config["preset_id"] in ("lotto", "aggressive") and liq < 25_000:
                if lanes.get("lotto_scout", {}).get("enabled", False):
                    return "lotto_scout"
            return "meme_opportunistic_scout"
        if whale >= 0.55 and liq >= 20_000 and lanes.get("liquidity_whale_scout", {}).get("enabled", True):
            return "liquidity_whale_scout"
        if abs(chg_1h) >= 3.0 and lanes.get("momentum_scout", {}).get("enabled", True):
            return "momentum_scout"
        if lanes.get("rss_sentiment_watcher", {}).get("enabled", True) and ("/" in symbol):
            # Soft lane — only if exploration allows and nothing else matched strongly
            if config.get("exploration") and whale >= 0.2:
                return "rss_sentiment_watcher"
        if config.get("exploration") and lanes.get("momentum_scout", {}).get("enabled", True):
            return "momentum_scout"
        return None

    def _maybe_open_acceptance(self, trader, settings, coins, config) -> dict[str, Any] | None:
        """Explicit acceptance/test path only — marked DEMO_TEST_ONLY."""
        for coin in coins[:5]:
            # FRESHEN_DEMO_CANDIDATE_BEFORE_PRICE_GATE_V1
            coin = _freshen_demo_candidate_from_latest_snapshot_v1(coin)
            price = coin.get("latest_price")
            pair = str(coin.get("pair_address") or "")
            if not price or float(price) <= 0 or not pair:
                continue
            cid = coin.get("id")
            coin_norm = {
                "id": cid,
                "coin_id": cid,
                "symbol": coin.get("symbol"),
                "chain": coin.get("chain") or "solana",
                "pair_address": pair,
                "price_usd": float(price),
                "latest_price": float(price),
                "name": coin.get("name"),
                "latest_liquidity": coin.get("latest_liquidity"),
                "price_updated_at": coin.get("last_seen_at"),
                "liquidity_updated_at": coin.get("last_seen_at"),
                "source_provider": coin.get("provider") or "dexscreener",
            }
            trader.set_market_prices(
                [{"pair_address": pair, "coin_id": cid, "price_usd": float(price)}],
                price_timestamp=_utc_now(),
            )
            try:
                assert_paper_demo_allowed(
                    trading_mode=resolve_runtime_guard_context()["trading_mode"],
                    live_trading_enabled=False,
                    wallet_configured=False,
                    order_flags=self._order_flags(acceptance=True),
                    demo_acceptance_mode_enabled=True,
                )
            except DemoExecutionGuardError as exc:
                return {"opened": False, "reasons": exc.reasons, "blocker": "acceptance_guard"}
            # PRICE_NA_BLOCKER_INSERTED_BEFORE_OPEN_POSITION
            _price_na_coin_payload = locals().get("coin") if isinstance(locals().get("coin"), dict) else locals().get("c")
            _price_na_blocked, _price_na_reason = _manual_price_na_demo_trade_blocker(
                _price_na_coin_payload if isinstance(_price_na_coin_payload, dict) else {},
                context="demo_bot.open_position",
            )
            if _price_na_blocked:
                try:
                    self._record_activity({
                        "event": "DEMO_ACTION_BLOCKED_PRICE_NOT_AVAILABLE",
                        "reason": _price_na_reason,
                        "coin": _price_na_coin_payload,
                    })
                except Exception:
                    pass
                return None
            pos = trader.open_position(
                coin_norm,
                size_usd=min(float(config["notional"]), 25.0),
                cluster_label="DEMO_ACCEPTANCE",
                settings=settings,
                reason_code="DEMO_TEST_ONLY",
                allow_coin_price_fallback=True,
            )
            if not pos:
                continue
            self._annotate_position(
                trader,
                pos,
                lane="acceptance",
                family="UNKNOWN_INSUFFICIENT_EVIDENCE",
                entry_reason="DEMO_ACCEPTANCE_SMOKE",
                config=config,
                coin=coin,
                obs=None,
            )
            summary = (
                f"Opened acceptance test position #{pos.get('id')} {pos.get('symbol')} - "
                "DEMO_TEST_ONLY, not strategy evidence."
            )
            self._append_activity(
                {
                    "event": "PAPER_BUY",
                    "position_id": pos.get("id"),
                    "symbol": pos.get("symbol"),
                    "strategy_lane": "acceptance",
                    "exit_reason_plan": "DEMO_TEST_ONLY",
                    "summary": summary,
                }
            )
            return {"opened": True, "position": pos, "summary": summary, "acceptance": True}
        return {"opened": False, "summary": "Acceptance mode: no priced candidate for smoke trade."}

    def _annotate_position(
        self,
        trader,
        pos: dict[str, Any],
        *,
        lane: str,
        family: str,
        entry_reason: str,
        config: dict[str, Any],
        coin: dict[str, Any],
        obs: dict[str, Any] | None,
    ) -> None:
        for p in trader.get_positions(status="OPEN"):
            if int(p.get("id", -1)) != int(pos.get("id", -2)):
                continue
            entry = float(p.get("entry_price") or 0)
            tp = float(config["tp"])
            sl = float(config["sl"])
            p.update(
                {
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                    "demo_bot": True,
                    "demo_acceptance_only": bool(config.get("acceptance")),
                    "strategy_lane": lane,
                    "entry_reason": entry_reason,
                    "candidate_score": float(coin.get("latest_whale_score") or 0),
                    "model_signal_summary": "Model score unavailable - using demo exploration fallback",
                    "semantic_label": family,
                    "semantic_label_human": semantic_label_human(family),
                    "opportunity_state": (obs or {}).get("trading_opportunity_state") or "DEMO_CANDIDATE",
                    "liquidity": coin.get("latest_liquidity"),
                    "volume_24h": coin.get("latest_volume_24h"),
                    "buy_ratio": coin.get("buy_ratio"),
                    "whale_score": coin.get("latest_whale_score"),
                    "risk_mode": config["preset_id"],
                    "expected_hold_profile": config.get("hold_profile"),
                    "min_hold_seconds": config["min_hold"],
                    "time_stop_seconds": config["time_stop"],
                    "take_profit": entry * (1 + tp) if entry else p.get("take_profit"),
                    "stop_loss": entry * (1 - sl) if entry else p.get("stop_loss"),
                    "trailing_stop_pct": config["trail"],
                    "peak_price": entry,
                    "exit_plan": {
                        "take_profit_pct": tp,
                        "stop_loss_pct": sl,
                        "trailing_stop_pct": config["trail"],
                        "min_hold_seconds": config["min_hold"],
                        "time_stop_seconds": config["time_stop"],
                    },
                    "trade_authority": "PAPER_DEMO_ONLY",
                }
            )
            break
        trader._save_state()  # noqa: SLF001

    def _maybe_open_position(
        self,
        *,
        trader,
        settings: dict[str, Any],
        coins: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        opens: list[dict[str, Any]],
        config: dict[str, Any],
        allow_missing_coin_id: bool = False,
    ) -> dict[str, Any] | None:
        open_pairs = {str(p.get("pair_address") or "") for p in opens}
        open_count = len(opens)
        max_open = int(config.get("max_open") or 0)
        now = datetime.now(timezone.utc)
        pair_cd = config.get("pair_cooldowns") or {}
        bot_state = config.get("bot_state") or {}
        ranked = sorted(
            coins,
            key=lambda c: (
                float(c.get("latest_whale_score") or 0),
                float(c.get("latest_volume_24h") or c.get("volume_24h") or 0),
            ),
            reverse=True,
        )
        preset_id = config["preset_id"]
        # STRATEGY_ARMS_USE_FRESH_UNIVERSE_BEFORE_SELECTION_V1
        if str(preset_id).lower().strip() in {"strict_consensus", "reasoning_demo"}:
            coins = _fresh_market_snapshot_candidate_universe_v1(limit=80, max_age_seconds=900, max_per_base_symbol=1)
        window = 30 if preset_id in ("aggressive", "lotto") else (20 if preset_id == "balanced" else 12)
        candidates_seen_in_window = 0
        candidates_selected = 0
        trade_attempts = 0
        last_block: str | None = None
        rejected_attempts: list[dict[str, Any]] = []
        gatekeeper_pass_count = 0
        gatekeeper_block_count = 0

        def _record_rejection(
            *,
            guard: str,
            reason: str,
            code: str,
            coin_row: dict[str, Any],
            pair_addr: str,
            lane_id: str | None,
            is_attempt: bool,
        ) -> None:
            rejected_attempts.append(
                {
                    "symbol": coin_row.get("symbol"),
                    "pair_address": pair_addr,
                    "chain": coin_row.get("chain"),
                    "strategy_lane": lane_id,
                    "blocking_guards": [guard],
                    "rejection_reasons": [reason],
                    "rejection_code": code,
                    "primary_blocker": guard,
                    "is_trade_attempt": is_attempt,
                }
            )

        for coin in ranked[:window]:
            candidates_seen_in_window += 1
            pair = str(coin.get("pair_address") or "")
            if not pair:
                continue

            # AE13G: bot-side skip reasons are recorded BEFORE open_position so
            # explainability counts them even though they never became a trade_attempt.
            if pair in open_pairs:
                last_block = "duplicate_pair_guard"
                _record_rejection(
                    guard="duplicate_pair_guard",
                    reason="Blocked: duplicate pair already open",
                    code="DUPLICATE_PAIR_ALREADY_OPEN",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=None,
                    is_attempt=False,
                )
                continue
            cd_until = _parse_ts(pair_cd.get(pair))
            if cd_until and cd_until > now:
                last_block = "cooldown"
                _record_rejection(
                    guard="cooldown",
                    reason="Blocked: pair cooldown active",
                    code="PAIR_COOLDOWN_ACTIVE",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=None,
                    is_attempt=False,
                )
                continue
            # FRESHEN_DEMO_CANDIDATE_BEFORE_PRICE_GATE_V1
            coin = _freshen_demo_candidate_from_latest_snapshot_v1(coin)
            price = coin.get("latest_price")
            if not price or float(price) <= 0:
                last_block = "missing_price"
                _record_rejection(
                    guard="missing_price",
                    reason="Blocked: missing price",
                    code="MISSING_PRICE",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=None,
                    is_attempt=False,
                )
                continue
            cid = coin.get("id") if coin.get("id") is not None else coin.get("coin_id")
            if cid is None and not allow_missing_coin_id:
                continue

            obs = next(
                (
                    o
                    for o in observed
                    if str(o.get("pair_address") or "") == pair
                    or (cid is not None and o.get("coin_identity") == cid)
                ),
                None,
            )
            family = (obs or {}).get("semantic_signal_family") or "UNKNOWN_INSUFFICIENT_EVIDENCE"
            # Normalize legacy unknown
            if family in ("UNKNOWN_UNRESOLVED",):
                family = "UNKNOWN_INSUFFICIENT_EVIDENCE"

            lane = self._pick_lane(coin, obs, config)
            if not lane:
                # AE14 clean-forward: fall back to manual_watchlist_scout / momentum
                # when lane heuristics (whale/volume) do not match CF rows.
                if allow_missing_coin_id:
                    lanes = config.get("lanes") or {}
                    if lanes.get("manual_watchlist_scout", {}).get("enabled", True):
                        lane = "manual_watchlist_scout"
                    elif lanes.get("momentum_scout", {}).get("enabled", True):
                        lane = "momentum_scout"
                if not lane:
                    last_block = "no_enabled_strategy_lane"
                    with _LOCK:
                        self._lane_bump("momentum_scout", blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="no_enabled_strategy_lane",
                        reason="Blocked: no enabled strategy lane matched this candidate",
                        code="NO_ENABLED_STRATEGY_LANE",
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=None,
                        is_attempt=False,
                    )
                    continue

            with _LOCK:
                self._lane_bump(lane, candidates_seen=1)

            # STRICT_CONSENSUS_DEMO_BUY_GATE_V1
            if config.get("preset_id") == "strict_consensus" or config.get("strict_consensus_only"):
                def _pick_upper(*objs):
                    keys = (
                        "ae16_consensus_tier",
                        "consensus_tier",
                        "model_evidence_tier",
                        "evidence_tier",
                        "tier",
                        "ae16_tier",
                    )
                    for obj in objs:
                        if isinstance(obj, dict):
                            for k in keys:
                                v = obj.get(k)
                                if v is not None and str(v).strip():
                                    return str(v).strip().upper()
                    return ""

                tier = _pick_upper(coin, obs or {}, gate if isinstance(gate, dict) else {})
                allowed = set(str(x).upper() for x in (config.get("strict_consensus_allowed_tiers") or []))
                if tier not in allowed:
                    last_block = f"strict_consensus_tier_block:{tier or 'MISSING_TIER'}"
                    with _LOCK:
                        self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="strict_consensus_gate",
                        reason=f"Blocked: Strict Consensus Demo allows only {sorted(allowed)}; candidate tier={tier or 'MISSING_TIER'}",
                        code="STRICT_CONSENSUS_TIER_BLOCK",
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=lane,
                        is_attempt=False,
                    )
                    continue

            # STRATEGY_ARMS_STRICT_AND_REASONING_BUY_GATES_V1
            _arm_preset = str(config.get("preset_id") or "").lower().strip()

            def _arm_upper_from(*objs, keys=()):
                for obj in objs:
                    if isinstance(obj, dict):
                        for k in keys:
                            v = obj.get(k)
                            if v is not None and str(v).strip():
                                return str(v).strip().upper()
                return ""

            def _arm_bool_flag(flag_name):
                for obj in (_arm_coin, _arm_obs, _arm_gate):
                    if isinstance(obj, dict) and obj.get(flag_name) is True:
                        return True
                return False

            def _arm_list_values(*keys):
                vals = []
                for obj in (_arm_coin, _arm_obs, _arm_gate):
                    if not isinstance(obj, dict):
                        continue
                    for k in keys:
                        v = obj.get(k)
                        if isinstance(v, list):
                            vals.extend(str(x) for x in v if x)
                        elif isinstance(v, str) and v.strip():
                            vals.append(v.strip())
                return vals

            _arm_coin = locals().get("coin") if isinstance(locals().get("coin"), dict) else {}
            _arm_obs = locals().get("obs") if isinstance(locals().get("obs"), dict) else {}
            _arm_gate = locals().get("gate") if isinstance(locals().get("gate"), dict) else {}
            _arm_family = str(locals().get("family") or "").strip().upper()

            if _arm_preset == "strict_consensus":
                _tier = _arm_upper_from(
                    _arm_coin, _arm_obs, _arm_gate,
                    keys=(
                        "ae16_consensus_tier",
                        "consensus_tier",
                        "model_evidence_tier",
                        "evidence_tier",
                        "tier",
                        "ae16_tier",
                    ),
                )
                _allowed = {"TAB_XGB_RF_ALL3", "TAB_RF_ONLY"}
                if _tier not in _allowed:
                    last_block = f"strict_consensus_tier_block:{_tier or 'MISSING_TIER'}"
                    with _LOCK:
                        self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="strict_consensus_gate",
                        reason=f"Blocked: strict_consensus allows only {sorted(_allowed)}; candidate tier={_tier or 'MISSING_TIER'}",
                        code="STRICT_CONSENSUS_TIER_BLOCK",
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=lane,
                        is_attempt=False,
                    )
                    continue

            if _arm_preset == "reasoning_demo":
                _state = _arm_upper_from(
                    _arm_coin, _arm_obs, _arm_gate,
                    keys=(
                        "trading_opportunity_state",
                        "opportunity_state",
                        "reasoning_state",
                        "context_state",
                        "ai_review_state",
                        "semantic_state",
                    ),
                )
                _semantic = _arm_upper_from(
                    _arm_coin, _arm_obs, _arm_gate,
                    keys=(
                        "semantic_label",
                        "semantic_family",
                        "semantic_signal_family",
                        "ae8_context_status",
                        "context_status",
                        "qwen_context_status",
                        "gemini_context_status",
                    ),
                ) or _arm_family

                _reason = _arm_upper_from(
                    _arm_coin, _arm_obs, _arm_gate,
                    keys=(
                        "reasoning_decision",
                        "llm_decision",
                        "ai_review_decision",
                        "context_decision",
                        "semantic_conflict_review",
                        "qwen_decision",
                        "gemini_decision",
                    ),
                )

                _blockers = []
                for _flag in (
                    "blocked_by_ae9",
                    "stale_price",
                    "missing_context",
                    "identity_mismatch",
                    "semantic_conflict",
                    "scam_risk",
                    "reputation_risk",
                    "honeypot_risk",
                ):
                    if _arm_bool_flag(_flag):
                        _blockers.append(_flag)

                _blockers.extend(_arm_list_values(
                    "ae9_audit_blockers",
                    "audit_blockers",
                    "context_blockers",
                    "reasoning_blockers",
                    "risk_flags",
                    "source_quality_flags",
                ))

                _joined = " | ".join([_state, _semantic, _reason] + [str(x).upper() for x in _blockers])
                _hard_words = (
                    "STRICT_BLOCKED",
                    "NO_TRADE",
                    "REJECT",
                    "REJECTED",
                    "SCAM",
                    "RUG",
                    "HONEYPOT",
                    "BLACKLIST",
                    "IDENTITY_MISMATCH",
                    "SEMANTIC_CONFLICT",
                    "MISSING_CONTEXT",
                    "STALE_PRICE",
                    "UNKNOWN_INSUFFICIENT_EVIDENCE",
                )

                _has_reasoning_evidence = any(x for x in (_state, _semantic, _reason) if x and x not in {"UNKNOWN", "UNCLASSIFIED", "NONE", "NULL"})

                if (not _has_reasoning_evidence) or any(w in _joined for w in _hard_words):
                    _why = "REASONING_MISSING" if not _has_reasoning_evidence else "REASONING_RED_FLAG"
                    last_block = f"reasoning_demo_block:{_why}"
                    with _LOCK:
                        self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="reasoning_demo_gate",
                        reason=f"Blocked: Reasoning Demo requires usable clean reasoning/context evidence. state={_state or '-'} semantic={_semantic or '-'} reason={_reason or '-'} blockers={_blockers}",
                        code=_why,
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=lane,
                        is_attempt=False,
                    )
                    continue

            if family == "UNKNOWN_INSUFFICIENT_EVIDENCE" and not config["exploration"]:
                # CF AE14 validation: allow manual_watchlist_scout without exploration.
                if not (allow_missing_coin_id and lane == "manual_watchlist_scout"):
                    last_block = "unknown_without_exploration"
                    with _LOCK:
                        self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="unknown_without_exploration",
                        reason="Blocked: unknown semantic family without exploration enabled",
                        code="UNKNOWN_WITHOUT_EXPLORATION",
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=lane,
                        is_attempt=False,
                    )
                    continue
            if family == "STRICT_BLOCKED" or (obs or {}).get("trading_opportunity_state") == "STRICT_BLOCKED":
                last_block = "strict_blocked"
                with _LOCK:
                    self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                _record_rejection(
                    guard="strict_blocked",
                    reason="Blocked: strict semantic block",
                    code="STRICT_BLOCKED",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=lane,
                    is_attempt=False,
                )
                continue

            candidates_selected += 1
            with _LOCK:
                self._lane_bump(lane, candidates_selected=1)

            trade_attempts += 1
            with _LOCK:
                self._state["trade_attempt_count"] = int(self._state.get("trade_attempt_count") or 0) + 1
                self._state["last_selected_candidate"] = {
                    "symbol": coin.get("symbol"),
                    "pair_address": pair,
                    "semantic_family": family,
                    "price_usd": float(price),
                    "strategy_lane": lane,
                    "candidate_source": coin.get("candidate_source"),
                    "clean_forward_bridge_used": bool(coin.get("clean_forward_bridge_used")),
                    "legacy_market_snapshots_used": False
                    if coin.get("candidate_source") == "clean_forward_market_feed"
                    else None,
                }

            price_ts = (
                coin.get("price_updated_at")
                or coin.get("last_seen_at")
                or coin.get("observed_at")
            )
            coin_norm = {
                "id": cid,
                "coin_id": cid,
                "symbol": coin.get("symbol"),
                "chain": coin.get("chain") or "solana",
                "pair_address": pair,
                "price_usd": float(price),
                "latest_price": float(price),
                "name": coin.get("name"),
                # AE13I / AE14: carry market-row provenance through so
                # MarketDataGateKeeper's freshness/provenance checks have the
                # timestamp/provider/liquidity fields they require.
                "latest_liquidity": coin.get("latest_liquidity") or coin.get("liquidity_usd"),
                "liquidity_usd": coin.get("latest_liquidity") or coin.get("liquidity_usd"),
                "price_updated_at": price_ts,
                "liquidity_updated_at": coin.get("liquidity_updated_at") or price_ts,
                "last_seen_at": price_ts,
                "source_provider": coin.get("source_provider") or coin.get("provider") or "dexscreener",
                "base_token_address": coin.get("base_token_address"),
                "token_contract_address": coin.get("token_contract_address")
                or coin.get("base_token_address"),
                "token_mint_address": coin.get("token_mint_address"),
                "quote_token_address": coin.get("quote_token_address"),
                "address_role": coin.get("address_role") or "pair_contract",
                "paper_demo_only": True,
                "not_live_approved": True,
                "live_trading_ready": False,
                "candidate_source": coin.get("candidate_source"),
                "clean_forward_bridge_used": bool(coin.get("clean_forward_bridge_used")),
                "legacy_market_snapshots_used": bool(
                    coin.get("legacy_market_snapshots_used") or False
                ),
                "market_match_status": coin.get("market_match_status"),
                "pair": coin.get("pair"),
                "provider_pair_id": coin.get("provider_pair_id"),
            }

            # AE14: build canonical instrument identity for Clean Forward
            # candidates. Identity is mode-agnostic; policy remains paper-only.
            if allow_missing_coin_id or coin.get("candidate_source") == "clean_forward_market_feed":
                from app.ae13b_product.clean_forward_execution_instrument import (
                    build_clean_forward_execution_instrument,
                )

                built = build_clean_forward_execution_instrument(
                    coin_norm, execution_mode="paper"
                )
                if not built.get("ok") or not isinstance(built.get("instrument"), dict):
                    last_block = str(
                        built.get("block_reason") or "clean_forward_instrument_rejected"
                    )
                    with _LOCK:
                        self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                    _record_rejection(
                        guard="clean_forward_execution_instrument",
                        reason="; ".join(
                            str(r)
                            for r in (
                                built.get("block_reasons")
                                or [built.get("block_reason") or last_block]
                            )
                        ),
                        code=last_block,
                        coin_row=coin,
                        pair_addr=pair,
                        lane_id=lane,
                        is_attempt=True,
                    )
                    continue
                coin_norm.update(built["instrument"])

            trader.set_market_prices(
                [{"pair_address": pair, "coin_id": cid, "price_usd": float(price)}],
                price_timestamp=price_ts or _utc_now(),
            )
            try:
                assert_paper_demo_allowed(
                    trading_mode=resolve_runtime_guard_context()["trading_mode"],
                    live_trading_enabled=False,
                    wallet_configured=False,
                    order_flags=self._order_flags(acceptance=False),
                )
            except DemoExecutionGuardError as exc:
                reason_text = "; ".join(exc.reasons) if exc.reasons else "execution_guard_blocked"
                _record_rejection(
                    guard="execution_guard",
                    reason=reason_text,
                    code="EXECUTION_GUARD",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=lane,
                    is_attempt=True,
                )
                return self._finalize_open_result(
                    opened=False,
                    rejected_attempts=rejected_attempts,
                    candidates_seen_in_window=candidates_seen_in_window,
                    candidates_selected=candidates_selected,
                    trade_attempts=trade_attempts,
                    open_count=open_count,
                    max_open=max_open,
                    last_block="execution_guard",
                    summary="No new buy: paper/demo execution guard blocked this attempt.",
                    extra={
                        "reasons": exc.reasons,
                        "blocker": "execution_guard",
                        "gatekeeper_pass_count": gatekeeper_pass_count,
                        "gatekeeper_block_count": gatekeeper_block_count,
                    },
                )

            # AE13I: MarketDataGateKeeper runs upstream of PaperTrader.open_position()
            # so freshness/provenance/reentry/address-role blocks stop the attempt
            # before an order is ever placed.
            from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

            # AE13I fix: stagnant_price_guard now returns passed=True with
            # momentum_evidence="unknown_insufficient_delta_fields" when a
            # row has no delta fields at all, instead of hard-blocking on
            # missing data. That means it is now safe to run the guard
            # (skip_stagnant=False) even for rows from schemas that lack
            # per-row 1h/4h activity deltas -- it only blocks when it has
            # concrete evidence of a below-threshold delta.
            gate = validate_market_data_gate(coin_norm, for_open=True, skip_stagnant=False)
            if not gate.get("passed"):
                gatekeeper_block_count += 1
                last_block = (gate.get("blocking_guards") or ["freshness_gate"])[0]
                with _LOCK:
                    self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                _record_rejection(
                    guard=last_block,
                    reason=(gate.get("rejection_reasons") or ["Blocked by market data gate"])[0],
                    code=gate.get("rejection_code") or "NOT_OPENED_STALE_MARKET_DATA",
                    coin_row=coin,
                    pair_addr=pair,
                    lane_id=lane,
                    is_attempt=True,
                )
                # Surface the full gate context on the rejection record for
                # explainability (top-rejection summary, UI blocker badges).
                rejected_attempts[-1]["blocking_guards"] = list(gate.get("blocking_guards") or [last_block])
                rejected_attempts[-1]["rejection_reasons"] = list(
                    gate.get("rejection_reasons") or [rejected_attempts[-1]["rejection_reasons"][0]]
                )
                rejected_attempts[-1]["tradability_status"] = gate.get("tradability_status")
                rejected_attempts[-1]["decision"] = gate.get("decision")
                continue

            gatekeeper_pass_count += 1

            notional = float(config["notional"])
            if lane == "lotto_scout":
                notional = min(notional, 25.0)

            entry_reason = (
                f"{lane}|whale={float(coin.get('latest_whale_score') or 0):.3f}|"
                f"sem={family}|liq={float(coin.get('latest_liquidity') or coin.get('liquidity_usd') or 0):.0f}"
            )
            pos = trader.open_position(
                coin_norm,
                size_usd=min(notional, 100.0),
                cluster_label=str(family),
                settings={
                    **settings,
                    "take_profit_pct": config["tp"],
                    "stop_loss_pct": config["sl"],
                },
                reason_code="DEMO_STRATEGY_ENTRY",
                strategy_type=lane.upper(),
                allow_coin_price_fallback=True,
                bot_state=bot_state,
                pair_cooldowns=pair_cd,
                risk_mode=config["preset_id"],
                preset_id=config["preset_id"],
                gate_result=gate,
            )
            if not pos:
                last_open = trader.get_last_open_result() or {}
                blocking_guards = list(last_open.get("blocking_guards") or [])
                primary_blocker = last_open.get("primary_blocker") or (
                    blocking_guards[0] if blocking_guards else "unknown_risk_block"
                )
                rejection_reasons = list(
                    last_open.get("rejection_reasons")
                    or ([last_open.get("rejection_reason")] if last_open.get("rejection_reason") else [])
                )
                last_block = primary_blocker
                rejected_attempts.append(
                    {
                        "symbol": coin.get("symbol"),
                        "pair_address": pair,
                        "chain": coin.get("chain"),
                        "strategy_lane": lane,
                        "blocking_guards": blocking_guards or [primary_blocker],
                        "rejection_reasons": rejection_reasons or ["risk_guard_blocked"],
                        "rejection_code": last_open.get("rejection_code"),
                        "primary_blocker": primary_blocker,
                        "is_trade_attempt": True,
                    }
                )
                with _LOCK:
                    self._lane_bump(lane, blocked_count=1, last_reason=last_block)
                continue

            self._annotate_position(
                trader,
                pos,
                lane=lane,
                family=family,
                entry_reason=entry_reason,
                config=config,
                coin=coin,
                obs=obs,
            )
            with _LOCK:
                self._lane_bump(lane, trades_opened=1, last_reason="opened")
                # Avoid immediate re-entry churn on same pair
                cds = dict(self._state.get("pair_cooldowns") or {})
                cds[pair] = (now + timedelta(seconds=max(300, config["min_hold"]))).isoformat()
                self._state["pair_cooldowns"] = cds

            summary = (
                f"Opened paper position #{pos.get('id')} {pos.get('symbol')} "
                f"(${min(notional, 100):.0f}) via {lane} - {semantic_label_human(family)}. "
                f"Hold profile: {config.get('hold_profile')} "
                f"(min {config['min_hold']}s, time-stop {config['time_stop']}s). "
                "Paper/demo only; not live approved; not profitability evidence."
            )
            self._append_activity(
                {
                    "event": "PAPER_BUY",
                    "position_id": pos.get("id"),
                    "symbol": pos.get("symbol"),
                    "notional_usd": min(notional, 100.0),
                    "semantic": family,
                    "strategy_lane": lane,
                    "entry_reason": entry_reason,
                    "expected_hold_profile": config.get("hold_profile"),
                    "exit_plan": {
                        "min_hold_seconds": config["min_hold"],
                        "time_stop_seconds": config["time_stop"],
                        "take_profit_pct": config["tp"],
                        "stop_loss_pct": config["sl"],
                    },
                    "decision": {
                        "selected": True,
                        "strategy_lane": lane,
                        "candidate_id": cid,
                        "pair_id": pair,
                        "symbol": coin.get("symbol"),
                        "chain": coin.get("chain"),
                        "score_summary": f"whale={coin.get('latest_whale_score')}",
                        "semantic_summary": family,
                        "risk_summary": config["preset_id"],
                        "reason": entry_reason,
                        "expected_hold_profile": config.get("hold_profile"),
                    },
                    "summary": summary,
                }
            )
            return self._finalize_open_result(
                opened=True,
                rejected_attempts=rejected_attempts,
                candidates_seen_in_window=candidates_seen_in_window,
                candidates_selected=candidates_selected,
                trade_attempts=trade_attempts,
                open_count=open_count,
                max_open=max_open,
                last_block=None,
                summary=summary,
                extra={
                    "position": pos,
                    "semantic": family,
                    "strategy_lane": lane,
                    "gatekeeper_pass_count": gatekeeper_pass_count,
                    "gatekeeper_block_count": gatekeeper_block_count,
                    "clean_forward_bridge_used": bool(coin.get("clean_forward_bridge_used")),
                    "legacy_market_snapshots_used": bool(
                        coin.get("legacy_market_snapshots_used") or False
                    ),
                    "candidate_source": coin.get("candidate_source"),
                },
            )

        primary = last_block or "no_eligible_candidate"
        reason = (
            "No new buy: no eligible demo candidate this cycle "
            f"({primary}). "
            f"Exploration={'on' if config['exploration'] else 'off'}; preset={preset_id}."
        )
        return self._finalize_open_result(
            opened=False,
            rejected_attempts=rejected_attempts,
            candidates_seen_in_window=candidates_seen_in_window,
            candidates_selected=candidates_selected,
            trade_attempts=trade_attempts,
            open_count=open_count,
            max_open=max_open,
            last_block=primary,
            summary=reason,
            extra={
                "blocker": primary,
                "gatekeeper_pass_count": gatekeeper_pass_count,
                "gatekeeper_block_count": gatekeeper_block_count,
            },
        )

    @staticmethod
    def _finalize_open_result(
        *,
        opened: bool,
        rejected_attempts: list[dict[str, Any]],
        candidates_seen_in_window: int,
        candidates_selected: int,
        trade_attempts: int,
        open_count: int,
        max_open: int,
        last_block: str | None,
        summary: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the AE13G explainability-complete result of _maybe_open_position()."""
        distribution = aggregate_rejection_counts(rejected_attempts)
        top_summary = format_top_rejection_summary(
            rejected_attempts, candidates_selected=candidates_selected
        )
        blocking_guards = sorted(
            {g for row in rejected_attempts for g in (row.get("blocking_guards") or [])}
        )
        result: dict[str, Any] = {
            "opened": opened,
            "summary": summary,
            "last_block": last_block,
            "candidates_seen_in_window": candidates_seen_in_window,
            "candidates_selected": candidates_selected,
            "trade_attempts": trade_attempts,
            "rejected_attempts": rejected_attempts,
            "rejection_reason_distribution": distribution,
            "top_rejection_summary": top_summary,
            "blocking_guards": blocking_guards,
            "open_slots_available": max(0, max_open - open_count),
            "max_open": max_open,
            "open_count": open_count,
            "max_open_blocking": max_open > 0 and open_count >= max_open,
        }
        if extra:
            result.update(extra)
        return result

    def _maybe_close_positions(
        self, trader, settings: dict[str, Any], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Close only when explicit exit rules fire — never fee-only same-cycle churn."""
        closed: list[dict[str, Any]] = []
        opens = trader.get_positions(status="OPEN")
        now = datetime.now(timezone.utc)
        min_hold = int(config["min_hold"])
        time_stop = int(config["time_stop"])
        acceptance = bool(config.get("acceptance"))

        # Refresh market prices for open pairs so exits use real marks, not entry
        price_rows = []
        for pos in opens:
            pair = pos.get("pair_address")
            cid = pos.get("coin_id")
            # Prefer live coin price from DB if available later; use position mark if set
            mark = pos.get("mark_price") or pos.get("last_price")
            if mark and float(mark) > 0 and pair:
                price_rows.append(
                    {"pair_address": pair, "coin_id": cid, "price_usd": float(mark)}
                )
        if price_rows:
            try:
                trader.set_market_prices(price_rows, price_timestamp=_utc_now())
            except Exception:
                pass

        # Pull fresh prices from DB for open symbols
        try:
            from app import database as db

            coins = {str(c.get("pair_address") or ""): c for c in db.get_coins(limit=80)}
            fresh = []
            for pos in opens:
                pair = str(pos.get("pair_address") or "")
                c = coins.get(pair)
                if c and c.get("latest_price") and float(c["latest_price"]) > 0:
                    fresh.append(
                        {
                            "pair_address": pair,
                            "coin_id": c.get("id") or pos.get("coin_id"),
                            "price_usd": float(c["latest_price"]),
                        }
                    )
            if fresh:
                trader.set_market_prices(fresh, price_timestamp=_utc_now())
        except Exception:
            pass

        for pos in list(opens):
            opened = _parse_ts(pos.get("opened_at"))
            age_s = (now - opened).total_seconds() if opened else 0.0
            entry = float(pos.get("entry_price") or pos.get("fill_price") or 0)
            if entry <= 0:
                continue

            # Resolve current mark
            cur = None
            try:
                pair = str(pos.get("pair_address") or "")
                if pair and pair in getattr(trader, "_market_prices_by_pair", {}):
                    cur = float(trader._market_prices_by_pair[pair])  # noqa: SLF001
            except Exception:
                cur = None
            if cur is None or cur <= 0:
                cur = entry

            peak = float(pos.get("peak_price") or entry)
            if cur > peak:
                peak = cur
                # persist peak for trailing
                for p in trader.get_positions(status="OPEN"):
                    if int(p.get("id", -1)) == int(pos.get("id", -2)):
                        p["peak_price"] = peak
                        break
                try:
                    trader._save_state()  # noqa: SLF001
                except Exception:
                    pass

            pos_min_hold = int(pos.get("min_hold_seconds") or min_hold)
            pos_time_stop = int(pos.get("time_stop_seconds") or time_stop)
            tp = float(pos.get("take_profit") or entry * (1 + float(config["tp"])))
            sl = float(pos.get("stop_loss") or entry * (1 - float(config["sl"])))
            trail_pct = float(pos.get("trailing_stop_pct") or config["trail"])
            trail_level = peak * (1 - trail_pct)

            exit_reason = None
            if acceptance and age_s >= max(5, pos_min_hold):
                exit_reason = "DEMO_TEST_ONLY"
            elif age_s < pos_min_hold:
                # Respect min hold — do not close early except catastrophic SL after half min-hold
                if age_s >= pos_min_hold * 0.5 and cur <= sl * 0.95:
                    exit_reason = "STOP_LOSS"
                else:
                    continue
            elif cur >= tp:
                exit_reason = "TAKE_PROFIT"
            elif cur <= sl:
                exit_reason = "STOP_LOSS"
            elif cur <= trail_level and peak > entry * 1.02:
                exit_reason = "TRAILING_STOP"
            elif age_s >= pos_time_stop:
                exit_reason = "TIME_STOP"
            else:
                continue

            try:
                assert_paper_demo_allowed(
                    trading_mode=resolve_runtime_guard_context()["trading_mode"],
                    live_trading_enabled=False,
                    wallet_configured=False,
                    order_flags=self._order_flags(acceptance=acceptance),
                    demo_acceptance_mode_enabled=True if acceptance else None,
                )
            except DemoExecutionGuardError:
                break

            result = trader.close_position(
                int(pos["id"]),
                float(cur),
                reason_code=exit_reason if exit_reason in VALID_EXIT_REASONS else "TIME_STOP",
                close_reason="bot_auto_exit",
                close_note=f"demo_bot exit_reason={exit_reason}",
                closed_by="demo_bot",
            )
            if result:
                hold_s = age_s
                result["exit_reason"] = exit_reason
                result["holding_duration_seconds"] = hold_s
                closed.append(result)
                try:
                    from app.ae13b_product.reentry_blocks import add_system_close_block

                    add_system_close_block(result, close_reason=exit_reason, duration_seconds=300)
                except Exception:
                    log.exception("add_system_close_block failed for position #%s", pos.get("id"))
                with _LOCK:
                    self._state["trades_closed"] = int(self._state.get("trades_closed") or 0) + 1
                self._append_activity(
                    {
                        "event": "PAPER_SELL",
                        "position_id": pos.get("id"),
                        "symbol": pos.get("symbol"),
                        "exit_reason": exit_reason,
                        "holding_duration_seconds": hold_s,
                        "entry_price": entry,
                        "exit_price": cur,
                        "gross_pnl": result.get("gross_pnl"),
                        "fees": result.get("exit_fees"),
                        "net_pnl": result.get("realized_pnl"),
                        "return_pct": result.get("net_roi_pct"),
                        "summary": (
                            f"Closed paper #{pos.get('id')} {pos.get('symbol')} "
                            f"reason={exit_reason} hold={int(hold_s)}s."
                        ),
                    }
                )
        return closed

    def close_all(self) -> dict[str, Any]:
        with _LOCK:
            ctx = resolve_runtime_guard_context()
            try:
                assert_paper_demo_allowed(
                    trading_mode=ctx["trading_mode"],
                    live_trading_enabled=ctx["live_trading_enabled"],
                    wallet_configured=False,
                    order_flags=self._order_flags(acceptance=False),
                )
            except DemoExecutionGuardError as exc:
                return {"ok": False, "error": str(exc), "reasons": exc.reasons, "status": self.status()}

            from app.execution.paper import get_paper_trader

            trader = get_paper_trader()
            closed = []
            for pos in list(trader.get_positions(status="OPEN")):
                r = trader.close_position(
                    int(pos["id"]),
                    float(pos.get("entry_price") or 0) or None,
                    reason_code="MANUAL_CLOSE",
                    close_reason="user_exit",
                    close_note="close_all_demo_positions",
                    closed_by="user_manual",
                )
                if r:
                    r["exit_reason"] = "MANUAL_CLOSE"
                    closed.append(r)
                    self._state["trades_closed"] = int(self._state.get("trades_closed") or 0) + 1
            self._state["last_action_summary"] = f"Closed {len(closed)} demo position(s) (manual)."
            self._append_activity({"event": "CLOSE_ALL", "count": len(closed)})
            self._save()
            return {"ok": True, "closed": closed, "status": self.status()}

    def reset_wallet(self) -> dict[str, Any]:
        with _LOCK:
            ctx = resolve_runtime_guard_context()
            if ctx["trading_mode"] not in ("DEMO", "PAPER") or ctx["live_trading_enabled"]:
                return {
                    "ok": False,
                    "error": "Reset allowed only in DEMO/PAPER with live trading disabled.",
                    "status": self.status(),
                }
            from app.execution.paper import get_paper_trader

            wallet = get_paper_trader().reset_demo_wallet()
            self._state["last_action_summary"] = "Demo wallet reset to $10,000 paper baseline."
            self._append_activity({"event": "WALLET_RESET", "wallet": wallet})
            self._save()
            return {"ok": True, "wallet": wallet, "status": self.status()}


def get_demo_bot() -> DemoBot:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = DemoBot()
        return _INSTANCE


def reset_demo_bot_for_tests() -> None:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is not None:
            try:
                _INSTANCE.stop()
            except Exception:
                pass
        _INSTANCE = None


# BEGIN MANUAL_PRICE_NA_DEMO_TRADING_FIX
def _manual_price_na_demo_trade_blocker(coin: dict, *, context: str = "") -> tuple[bool, str]:
    """
    Product safety guard:
    A demo position must not be opened from a row that has no current tradable price.
    Entry snapshot price is not enough if current price/mark price is unavailable.

    Returns:
        (blocked, reason)
    """
    if not isinstance(coin, dict):
        return True, "PRICE_NOT_AVAILABLE: coin payload missing"

    text = " ".join(
        str(coin.get(k, ""))
        for k in (
            "price_status",
            "current_price_status",
            "current_price_label",
            "current_price_reason",
            "mark_price_status",
            "status",
            "reason",
            "skip_reason",
            "entry_price_source",
            "price_source_note",
        )
    ).upper()

    if (
        "PRICE_NOT_AVAILABLE" in text
        or "N/A" in text and "UNAVAILABLE" in text
        or "NOT CURRENT TRADABLE PRICE" in text
        or "PRICE UNAVAILABLE" in text
    ):
        return True, "PRICE_NOT_AVAILABLE: current tradable price unavailable"

    price = (
        coin.get("current_price")
        or coin.get("current_price_usd")
        or coin.get("mark_price")
        or coin.get("mark_price_usd")
        or coin.get("latest_price")
        or coin.get("price_usd")
        or coin.get("price")
    )

    try:
        price_f = float(price)
    except Exception:
        return True, "PRICE_NOT_AVAILABLE: price is missing or non-numeric"

    if price_f <= 0:
        return True, "PRICE_NOT_AVAILABLE: price is zero or negative"

    return False, ""
# END MANUAL_PRICE_NA_DEMO_TRADING_FIX

