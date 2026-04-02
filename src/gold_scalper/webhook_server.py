"""Webhook server — receives TradingView alerts and routes to executors.

Multi-instrument webhook router:
- POST /webhook receives ALL TradingView alerts
- Routes by `ticker` field: BTCUSD → BTC executor, everything else → Gold executor
- GET /status returns position status for all executors

Usage:
    python main.py gold-scalper --mode webhook --broker paper
    # Then set TradingView alert webhook URL to: https://<your-cloudflare-url>/webhook

Gold signal format (Pine Script):
    {"action": "ENTRY", "direction": "SHORT", "price": 4500.5, "contracts": 6}
    {"action": "TP_HIT", "tp_level": 1, "price": 4494.5, "contracts": 1}

BTC UT Bot format:
    {"action": "buy",  "contracts": 2, "ticker": "BTCUSD", "position_size":  1, "price": 68394}
    {"action": "sell", "contracts": 2, "ticker": "BTCUSD", "position_size": -1, "price": 68394}
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import asyncio

import nest_asyncio               # allow ib_insync event loop inside uvicorn's

import uvicorn  # import early — before src/selectors shadows stdlib selectors

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.gold_scalper.alerts import ScalperAlertEngine
from src.gold_scalper.config import GoldScalperConfig, load_config
from src.gold_scalper.webhook_executor import WebhookExecutor

logger = logging.getLogger(__name__)

PT = ZoneInfo("America/Los_Angeles")

# Global executors (initialized by start_server)
_gold_executor: Optional[WebhookExecutor] = None
_btc_executor = None  # BTCWebhookExecutor, lazy import
_webull_executor = None  # WebullExecutor, lazy import
_ibkr_connected: Optional[bool] = None
_last_heartbeat: Optional[str] = None

# Tickers that route to BTC executor (Alpaca)
_BTC_TICKERS = {"BTCUSD", "BTC/USD", "BTC-USD", "XBTUSD"}

# Tickers that route to Webull executor
_WEBULL_TICKERS = set()  # populated by start_server if Webull keys present

app = FastAPI(title="Trading Webhook Server", version="2.0")


@app.post("/webhook")
async def webhook(request: Request):
    """Receive TradingView alert webhook and route by ticker.

    Routes:
    - ticker in BTC_TICKERS → BTC executor (Alpaca crypto paper)
    - everything else → Gold executor (paper/IBKR)
    """
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        body_str = raw.decode("utf-8", errors="replace").strip()
        logger.warning("[WEBHOOK] Non-JSON body received: %s", body_str[:200])
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "received": body_str[:200]},
        )

    now = datetime.now(PT)
    ticker = body.get("ticker", "").strip().upper()

    logger.info(
        "[WEBHOOK] %s Received (ticker=%s): %s",
        now.strftime("%H:%M:%S"), ticker or "none",
        str(body)[:300],
    )

    # Route by ticker
    # Priority: Webull (if configured) → BTC (Alpaca) → Gold (default)
    norm_ticker = ticker.replace("/", "").replace("-", "")
    if _webull_executor and (norm_ticker in _WEBULL_TICKERS or ticker in _WEBULL_TICKERS):
        return _handle_webull(body)
    elif ticker in _BTC_TICKERS:
        return _handle_btc(body)
    else:
        return _handle_gold(body)


def _handle_btc(body: dict) -> JSONResponse:
    """Route to BTC executor."""
    global _btc_executor

    if _btc_executor is None:
        logger.error("[WEBHOOK] BTC executor not initialized")
        return JSONResponse(
            status_code=503,
            content={"error": "BTC executor not initialized"},
        )

    try:
        result = _btc_executor.execute(body)
    except Exception as exc:
        logger.error("[WEBHOOK] BTC execute failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"BTC execution failed: {exc}"},
        )

    logger.info("[WEBHOOK] BTC result: %s", result)
    return JSONResponse(content=result)


def _handle_webull(body: dict) -> JSONResponse:
    """Route to Webull executor (BTC + MGC)."""
    global _webull_executor

    if _webull_executor is None:
        logger.error("[WEBHOOK] Webull executor not initialized")
        return JSONResponse(
            status_code=503,
            content={"error": "Webull executor not initialized"},
        )

    try:
        result = _webull_executor.execute(body)
    except Exception as exc:
        logger.error("[WEBHOOK] Webull execute failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Webull execution failed: {exc}"},
        )

    logger.info("[WEBHOOK] Webull result: %s", result)
    return JSONResponse(content=result)


def _handle_gold(body: dict) -> JSONResponse:
    """Route to Gold executor."""
    global _gold_executor

    if _gold_executor is None:
        logger.error("[WEBHOOK] Gold executor not initialized")
        return JSONResponse(
            status_code=503,
            content={"error": "Gold executor not initialized"},
        )

    signal = _gold_executor.parse_signal(body)
    try:
        result = _gold_executor.execute(signal)
    except Exception as exc:
        logger.error("[WEBHOOK] Gold execute failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Gold execution failed: {exc}"},
        )

    logger.info("[WEBHOOK] Gold result: %s", result)
    return JSONResponse(content=result)


@app.get("/status")
async def status():
    """Position status for all executors."""
    result = {"status": "running"}

    # Gold status
    if _gold_executor:
        gold_pos = None
        if _gold_executor.position:
            p = _gold_executor.position
            gold_pos = {
                "direction": p.direction,
                "entry_price": p.entry_price,
                "remaining_contracts": p.remaining_contracts,
                "tp_hits": p.tp_hits,
                "realized_pnl": p.realized_pnl,
            }
        result["gold"] = {
            "position": gold_pos,
            "daily_pnl": _gold_executor.daily_pnl,
            "daily_trades": _gold_executor.daily_trades,
            "daily_wl": f"{_gold_executor.daily_wins}W/{_gold_executor.daily_losses}L",
            "circuit_breaker": _gold_executor._circuit_breaker_fired,
        }

    # BTC status
    if _btc_executor:
        btc_pos = None
        if _btc_executor.position:
            p = _btc_executor.position
            btc_pos = {
                "direction": p.direction,
                "entry_price": p.entry_price,
                "qty": p.qty,
            }
        result["btc"] = {
            "position": btc_pos,
            "daily_pnl": _btc_executor.daily_pnl,
            "daily_trades": _btc_executor.daily_trades,
            "daily_wl": f"{_btc_executor.daily_wins}W/{_btc_executor.daily_losses}L",
            "circuit_breaker": _btc_executor._circuit_breaker_fired,
        }

    # Webull status
    if _webull_executor:
        webull_positions = {}
        for ticker, pos in _webull_executor.positions.items():
            webull_positions[ticker] = {
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "qty": pos.qty,
                "instrument_type": pos.instrument_type,
            }
        result["webull"] = {
            "positions": webull_positions if webull_positions else None,
            "daily_pnl": _webull_executor.daily_pnl,
            "daily_trades": _webull_executor.daily_trades,
            "daily_wl": f"{_webull_executor.daily_wins}W/{_webull_executor.daily_losses}L",
            "circuit_breaker": _webull_executor._circuit_breaker_fired,
            "equity": _webull_executor._equity,
        }

    result["ibkr_connected"] = _ibkr_connected
    result["last_heartbeat"] = _last_heartbeat
    return result


@app.get("/health")
async def health():
    """Simple health check for monitoring."""
    return {"ok": True}


@app.post("/reconnect")
async def reconnect():
    """Force IBKR reconnection (call from phone if connection is stale)."""
    global _ibkr_connected, _last_heartbeat
    if _gold_executor is None:
        return JSONResponse(status_code=503, content={"error": "Executor not initialized"})

    broker = _gold_executor.broker
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
    """Background thread that checks IBKR connection every `interval` seconds."""
    global _ibkr_connected, _last_heartbeat

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
                        if alerted and _gold_executor and _gold_executor.alerts:
                            _gold_executor.alerts._send(
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
                        if _gold_executor and _gold_executor.alerts:
                            _gold_executor.alerts._send(
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


def _load_env():
    """Load secrets from alpaca.env file."""
    env = {}
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env_file = os.path.join(project_root, "secrets", "alpaca.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def start_server(
    broker,
    config: GoldScalperConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
    webhook_url: str = "",
):
    """Start the webhook server with Gold + BTC executors.

    Args:
        broker: IBKRGoldBroker or PaperGoldBroker instance (for gold)
        config: Gold scalper config
        host: Bind address (0.0.0.0 for external access)
        port: Port number
        webhook_url: Discord webhook URL for alerts
    """
    global _gold_executor, _btc_executor, _webull_executor

    alerts = ScalperAlertEngine(webhook_url) if webhook_url else None

    # Initialize Gold executor
    _gold_executor = WebhookExecutor(broker, config, alerts)
    logger.info("[WEBHOOK] Gold executor initialized")

    # Initialize BTC executor (Alpaca crypto paper)
    env = _load_env()
    btc_key = env.get("ALPACA_CRYPTO_KEY", os.environ.get("ALPACA_CRYPTO_KEY", ""))
    btc_secret = env.get("ALPACA_CRYPTO_SECRET", os.environ.get("ALPACA_CRYPTO_SECRET", ""))

    if btc_key and btc_secret:
        from src.gold_scalper.btc_executor import BTCWebhookExecutor
        _btc_executor = BTCWebhookExecutor(
            api_key=btc_key,
            api_secret=btc_secret,
            alerts=alerts,
        )
        logger.info("[WEBHOOK] BTC executor initialized (Alpaca crypto paper)")
    else:
        logger.warning("[WEBHOOK] No ALPACA_CRYPTO_KEY — BTC executor disabled")

    # Webull executor disabled — MGC routes to Gold executor (default)
    # To enable later: uncomment and set live_orders=True after token approval
    # from src.gold_scalper.webull_executor import WebullExecutor
    logger.info("[WEBHOOK] MGC routes to Gold executor (Webull disabled)")

    # Start IBKR heartbeat if using IBKR broker
    _start_ibkr_heartbeat(broker)

    logger.info(
        "[WEBHOOK] Starting server on %s:%d — "
        "waiting for TradingView signals...",
        host, port,
    )
    routes = ["Gold (default)"]
    if _btc_executor:
        routes.append("BTC/Alpaca (ticker=BTCUSD)")
    # Webull disabled — MGC falls through to Gold executor
    logger.info("[WEBHOOK] Routes: %s", " + ".join(routes))
    logger.info(
        "[WEBHOOK] Set TradingView alert webhook URL to: "
        "https://<your-cloudflare-url>/webhook"
    )

    # Run uvicorn
    loop = asyncio.get_event_loop()
    nest_asyncio.apply(loop)

    uvi_config = uvicorn.Config(
        app, host=host, port=port,
        log_level="info", access_log=False,
    )
    server = uvicorn.Server(uvi_config)
    loop.run_until_complete(server.serve())
