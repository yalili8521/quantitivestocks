"""Webhook server — receives TradingView alerts and routes to executors.

Multi-instrument webhook router:
- POST /webhook receives ALL TradingView alerts
- Routes by `ticker` field: BTCUSD → BTC executor, everything else → Gold executor
- GET /status returns position status for all executors

Usage:
    python main.py gold-scalper
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
_qqq_executor = None  # QQQOptionsExecutor, lazy import

# Tickers that route to BTC executor (Alpaca)
_BTC_TICKERS = {"BTCUSD", "BTC/USD", "BTC-USD", "XBTUSD"}

# Tickers that route to QQQ options executor
_QQQ_TICKERS = {"QQQ", "QQQO"}

# Tickers that route to Webull TQQQ executor
_WEBULL_TICKERS = {"TQQQ"}

app = FastAPI(title="Trading Webhook Server", version="2.0")


@app.post("/webhook")
async def webhook(request: Request):
    """Receive TradingView alert webhook and route by ticker.

    Routes:
    - ticker in BTC_TICKERS → BTC executor (Alpaca crypto paper)
    - everything else → Gold executor (paper)
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
    # Priority: Webull (if configured) → QQQ Options → BTC (Alpaca) → Gold (default)
    norm_ticker = ticker.replace("/", "").replace("-", "")
    if _webull_executor and (norm_ticker in _WEBULL_TICKERS or ticker in _WEBULL_TICKERS):
        return _handle_webull(body)
    elif norm_ticker in _QQQ_TICKERS or ticker in _QQQ_TICKERS:
        return _handle_qqq(body)
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


def _handle_qqq(body: dict) -> JSONResponse:
    """Route to QQQ options executor."""
    global _qqq_executor

    if _qqq_executor is None:
        logger.error("[WEBHOOK] QQQ options executor not initialized")
        return JSONResponse(
            status_code=503,
            content={"error": "QQQ options executor not initialized"},
        )

    try:
        result = _qqq_executor.execute(body)
    except Exception as exc:
        logger.error("[WEBHOOK] QQQ options execute failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"QQQ options execution failed: {exc}"},
        )

    logger.info("[WEBHOOK] QQQ options result: %s", result)
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

    # QQQ options status
    if _qqq_executor:
        result["qqq_options"] = _qqq_executor.get_status()

    # Webull TQQQ status
    if _webull_executor:
        result["webull_tqqq"] = _webull_executor.get_status()

    return result


@app.get("/health")
async def health():
    """Simple health check for monitoring."""
    return {"ok": True}


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
        broker: unused (kept for interface compatibility)
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

    # Initialize QQQ options executor (uses same Alpaca account as BTC — third account)
    if btc_key and btc_secret:
        from src.gold_scalper.qqq_options_executor import QQQOptionsExecutor
        _qqq_executor = QQQOptionsExecutor(
            api_key=btc_key,
            api_secret=btc_secret,
        )
        logger.info("[WEBHOOK] QQQ options executor initialized (Alpaca paper, options level 3)")
    else:
        logger.warning("[WEBHOOK] No ALPACA_CRYPTO_KEY — QQQ options executor disabled")

    # Webull TQQQ executor
    webull_key = env.get("WEBULL_APP_KEY", os.environ.get("WEBULL_APP_KEY", ""))
    webull_secret = env.get("WEBULL_APP_SECRET", os.environ.get("WEBULL_APP_SECRET", ""))

    if webull_key and webull_secret:
        from src.gold_scalper.webull_tqqq_executor import WebullTQQQExecutor
        _webull_executor = WebullTQQQExecutor(
            app_key=webull_key,
            app_secret=webull_secret,
            live_orders=True,  # LIVE — places real orders on Webull
        )
        logger.info("[WEBHOOK] Webull TQQQ executor initialized (paper mode)")
    else:
        logger.warning("[WEBHOOK] No WEBULL_APP_KEY — Webull TQQQ executor disabled")

    logger.info(
        "[WEBHOOK] Starting server on %s:%d — "
        "waiting for TradingView signals...",
        host, port,
    )
    routes = ["Gold (default)"]
    if _btc_executor:
        routes.append("BTC/Alpaca (ticker=BTCUSD)")
    if _qqq_executor:
        routes.append("QQQ Options (ticker=QQQ)")
    if _webull_executor:
        routes.append("TQQQ/Webull (ticker=TQQQ)")
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
