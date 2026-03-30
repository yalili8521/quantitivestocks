"""Webhook server — receives TradingView alerts and routes to IBKR.

TradingView信号接收服务器：
- FastAPI POST /webhook 端点接收TV的JSON信号
- 解析信号类型并路由到WebhookExecutor执行
- 所有信号和执行结果都记录日志

Usage:
    python main.py gold-scalper --mode webhook --broker ibkr
    # Then set TradingView alert webhook URL to: http://<your-ngrok-url>/webhook

Signal JSON format (from Pine Script):
    {"action": "ENTRY", "direction": "SHORT", "price": 4500.5, "contracts": 6}
    {"action": "TP_HIT", "tp_level": 1, "price": 4494.5, "contracts": 1}
    {"action": "EXIT", "reason": "Hard Stop Hit", "price": 4530.5, "contracts": 6}
    {"action": "SESSION_CLOSE", "price": 4500.0}
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import uvicorn  # import early — before src/selectors shadows stdlib selectors

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.gold_scalper.alerts import ScalperAlertEngine
from src.gold_scalper.config import GoldScalperConfig, load_config
from src.gold_scalper.webhook_executor import WebhookExecutor

logger = logging.getLogger(__name__)

PT = ZoneInfo("America/Los_Angeles")

# Global executor (initialized by start_server)
_executor: Optional[WebhookExecutor] = None
_ibkr_connected: Optional[bool] = None  # None = not IBKR broker
_last_heartbeat: Optional[str] = None

app = FastAPI(title="Gold Scalper Webhook", version="1.0")


@app.post("/webhook")
async def webhook(request: Request):
    """Receive TradingView alert webhook.

    Accepts JSON body with fields:
    - action: ENTRY | TP_HIT | EXIT | SESSION_CLOSE
    - direction: LONG | SHORT (for ENTRY)
    - price: float
    - contracts: int
    - tp_level: int (for TP_HIT, 1-4)
    - reason: str (for EXIT)
    """
    global _executor

    if _executor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Executor not initialized"},
        )

    try:
        body = await request.json()
    except Exception:
        # TradingView sometimes sends plain text
        raw = await request.body()
        body_str = raw.decode("utf-8", errors="replace").strip()
        logger.warning("[WEBHOOK] Non-JSON body received: %s", body_str[:200])
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "received": body_str[:200]},
        )

    now = datetime.now(PT)
    logger.info(
        "[WEBHOOK] %s Received: %s",
        now.strftime("%H:%M:%S"),
        str(body)[:300],
    )

    # Parse and execute
    signal = _executor.parse_signal(body)
    try:
        result = _executor.execute(signal)
    except Exception as exc:
        logger.error("[WEBHOOK] Execute failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": f"Execution failed: {exc}", "ibkr_connected": _ibkr_connected},
        )

    logger.info("[WEBHOOK] Result: %s", result)
    return JSONResponse(content=result)


@app.get("/status")
async def status():
    """Health check / position status."""
    global _executor
    if _executor is None:
        return {"status": "not_initialized"}

    pos = None
    if _executor.position:
        p = _executor.position
        pos = {
            "direction": p.direction,
            "entry_price": p.entry_price,
            "remaining_contracts": p.remaining_contracts,
            "tp_hits": p.tp_hits,
            "realized_pnl": p.realized_pnl,
        }

    return {
        "status": "running",
        "position": pos,
        "daily_pnl": _executor.daily_pnl,
        "daily_trades": _executor.daily_trades,
        "daily_wl": f"{_executor.daily_wins}W/{_executor.daily_losses}L",
        "circuit_breaker": _executor._circuit_breaker_fired,
        "ibkr_connected": _ibkr_connected,
        "last_heartbeat": _last_heartbeat,
    }


@app.get("/health")
async def health():
    """Simple health check for monitoring."""
    return {"ok": True}


@app.post("/reconnect")
async def reconnect():
    """Force IBKR reconnection (call from phone if connection is stale)."""
    global _ibkr_connected, _last_heartbeat
    if _executor is None:
        return JSONResponse(status_code=503, content={"error": "Executor not initialized"})

    broker = _executor.broker
    if not hasattr(broker, "ib"):
        return {"error": "Not using IBKR broker"}

    try:
        broker._ensure_connected()
        connected = broker.ib.isConnected()
        _ibkr_connected = connected
        _last_heartbeat = datetime.now(PT).strftime("%H:%M:%S")
        if connected:
            logger.info("[RECONNECT] Manual reconnect succeeded")
            return {"status": "connected", "time": _last_heartbeat}
        else:
            logger.error("[RECONNECT] Manual reconnect failed — TWS may be down")
            return JSONResponse(
                status_code=503,
                content={"status": "disconnected", "error": "TWS not responding"},
            )
    except Exception as e:
        _ibkr_connected = False
        logger.error("[RECONNECT] Error: %s", e)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)},
        )


def _start_ibkr_heartbeat(broker, interval: int = 120) -> None:
    """Background thread that checks IBKR connection every `interval` seconds.

    If disconnected, calls _ensure_connected() to auto-reconnect before the
    next TradingView signal arrives.  Runs only when broker is IBKRGoldBroker.

    After 3 consecutive failures (~6 min), sends a Discord alert so you know
    TWS is down and needs manual restart.
    """
    global _ibkr_connected, _last_heartbeat

    # Only run for IBKR broker (has .ib attribute)
    if not hasattr(broker, "ib"):
        return

    def _heartbeat_loop():
        global _ibkr_connected, _last_heartbeat
        consecutive_failures = 0
        alerted = False

        while True:
            try:
                connected = broker.ib.isConnected()
                now_str = datetime.now(PT).strftime("%H:%M:%S")
                _last_heartbeat = now_str

                if connected:
                    _ibkr_connected = True
                    if consecutive_failures > 0:
                        logger.info(
                            "[HEARTBEAT] %s IBKR connection restored after %d failures",
                            now_str, consecutive_failures,
                        )
                        # Send recovery alert if we previously alerted
                        if alerted and _executor and _executor.alerts:
                            _executor.alerts._send(
                                "IBKR RECOVERED — connection restored at "
                                f"{now_str} PT after {consecutive_failures} failures"
                            )
                    consecutive_failures = 0
                    alerted = False
                else:
                    _ibkr_connected = False
                    consecutive_failures += 1
                    logger.warning(
                        "[HEARTBEAT] %s IBKR disconnected (attempt %d) — reconnecting...",
                        now_str, consecutive_failures,
                    )
                    try:
                        broker._ensure_connected()
                        _ibkr_connected = broker.ib.isConnected()
                    except Exception:
                        _ibkr_connected = False

                    if _ibkr_connected:
                        logger.info(
                            "[HEARTBEAT] %s IBKR reconnected successfully",
                            now_str,
                        )
                        consecutive_failures = 0
                        alerted = False
                    elif consecutive_failures >= 3 and not alerted:
                        logger.error(
                            "[HEARTBEAT] %s IBKR DOWN for %d checks — "
                            "TWS may need manual restart!",
                            now_str, consecutive_failures,
                        )
                        if _executor and _executor.alerts:
                            _executor.alerts._send(
                                "IBKR DOWN — TWS appears to have exited. "
                                f"Disconnected for {consecutive_failures * interval // 60} min "
                                f"({consecutive_failures} failed reconnects). "
                                "Webhook server is still running but CANNOT execute trades. "
                                "Please restart TWS manually."
                            )
                        alerted = True
            except Exception as e:
                _ibkr_connected = False
                consecutive_failures += 1
                logger.error("[HEARTBEAT] Error: %s", e)
            time.sleep(interval)

    t = threading.Thread(target=_heartbeat_loop, daemon=True, name="ibkr-heartbeat")
    t.start()
    logger.info("[HEARTBEAT] IBKR connection monitor started (every %ds)", interval)


def start_server(
    broker,
    config: GoldScalperConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
    webhook_url: str = "",
):
    """Start the webhook server.

    Args:
        broker: IBKRGoldBroker or PaperGoldBroker instance
        config: Gold scalper config
        host: Bind address (0.0.0.0 for external access)
        port: Port number
        webhook_url: Discord webhook URL for alerts
    """
    global _executor

    alerts = ScalperAlertEngine(webhook_url) if webhook_url else None
    _executor = WebhookExecutor(broker, config, alerts)

    # Start IBKR heartbeat if using IBKR broker
    _start_ibkr_heartbeat(broker)

    logger.info(
        "[WEBHOOK] Starting server on %s:%d — "
        "waiting for TradingView signals...",
        host, port,
    )
    logger.info(
        "[WEBHOOK] Set TradingView alert webhook URL to: "
        "http://<your-public-url>/webhook"
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,  # reduce noise
    )
