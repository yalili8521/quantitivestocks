"""Vercel serverless function: GET /api/history

Returns portfolio equity history + filled orders for all account groups.
- Intraday + Swing: Alpaca paper API
- Crypto: Kraken paper state from GitHub Gist (trade log)
"""

from http.server import BaseHTTPRequestHandler
from datetime import datetime, timedelta, timezone
import json
import os

import requests as req

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

ACCOUNTS = [
    {
        "name": "Intraday",
        "group": "intraday",
        "key_env": "ALPACA_INTRADAY_KEY",
        "secret_env": "ALPACA_INTRADAY_SECRET",
    },
    {
        "name": "Swing",
        "group": "swing",
        "key_env": "ALPACA_SWING_KEY",
        "secret_env": "ALPACA_SWING_SECRET",
    },
]


def _fetch_alpaca(api_key: str, api_secret: str) -> dict:
    if not api_key or not api_secret:
        return {"portfolio": {}, "orders": [], "traded_symbols": []}

    headers = {
        "APCA-API-KEY-ID": api_key.strip(),
        "APCA-API-SECRET-KEY": api_secret.strip(),
    }

    portfolio = {}
    try:
        resp = req.get(
            f"{ALPACA_BASE}/account/portfolio/history",
            headers=headers,
            params={"period": "1W", "timeframe": "1H"},
            timeout=10,
        )
        if resp.ok:
            portfolio = resp.json()
    except Exception:
        pass

    orders = []
    try:
        resp = req.get(
            f"{ALPACA_BASE}/orders",
            headers=headers,
            params={"status": "closed", "limit": "50", "direction": "desc"},
            timeout=10,
        )
        if resp.ok:
            for o in resp.json():
                if o.get("status") == "filled":
                    orders.append({
                        "symbol":    o.get("symbol", ""),
                        "side":      o.get("side", ""),
                        "qty":       o.get("filled_qty", "0"),
                        "price":     o.get("filled_avg_price", "0"),
                        "filled_at": o.get("filled_at", ""),
                        "intent":    o.get("position_intent", ""),
                    })
    except Exception:
        pass

    traded_symbols = list({o["symbol"] for o in orders if o["symbol"]})
    try:
        pos_resp = req.get(f"{ALPACA_BASE}/positions", headers=headers, timeout=10)
        if pos_resp.ok:
            for p in pos_resp.json():
                sym = p.get("symbol", "")
                if sym and sym not in traded_symbols:
                    traded_symbols.append(sym)
    except Exception:
        pass

    return {
        "portfolio": {
            "timestamps":      portfolio.get("timestamp", []),
            "equity":          portfolio.get("equity", []),
            "profit_loss":     portfolio.get("profit_loss", []),
            "profit_loss_pct": portfolio.get("profit_loss_pct", []),
        },
        "orders":         orders,
        "traded_symbols": traded_symbols,
    }


def _fetch_kraken_history(state_filename: str = "kraken_paper_state.json") -> dict:
    """Read Kraken paper trade log from GitHub Gist."""
    gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
    if not gist_id:
        return {"portfolio": {}, "orders": []}

    try:
        resp = req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if not resp.ok:
            return {"portfolio": {}, "orders": []}

        gist_data = resp.json()

        # Read trade log if it exists
        trade_log_content = gist_data.get("files", {}).get("kraken_trade_log.json", {}).get("content", "[]")
        trade_log = json.loads(trade_log_content) if trade_log_content else []

        orders = []
        for t in trade_log:
            orders.append({
                "symbol":    t.get("symbol", ""),
                "side":      t.get("side", ""),
                "qty":       str(t.get("qty", "0")),
                "price":     str(t.get("price", "0")),
                "filled_at": t.get("time", ""),
                "intent":    t.get("intent", ""),
            })

    except Exception:
        return {"portfolio": {}, "orders": []}

    return {
        "portfolio": {},
        "orders": orders,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        accounts = []
        all_symbols: list[str] = []

        # Alpaca groups (intraday, swing)
        for cfg in ACCOUNTS:
            key    = os.environ.get(cfg["key_env"], "")
            secret = os.environ.get(cfg["secret_env"], "")
            data   = _fetch_alpaca(key, secret)
            accounts.append({
                "name":      cfg["name"],
                "group":     cfg["group"],
                "portfolio": data["portfolio"],
                "orders":    data["orders"],
            })
            for sym in data.get("traded_symbols", []):
                if sym not in all_symbols:
                    all_symbols.append(sym)

        # Crypto group — Kraken paper trade log from Gist
        crypto_data = _fetch_kraken_history("kraken_paper_state.json")
        accounts.append({
            "name":      "Crypto",
            "group":     "crypto",
            "portfolio": crypto_data["portfolio"],
            "orders":    crypto_data["orders"],
        })

        # Fetch price bars for traded symbols (use data API key)
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        data_headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

        # Dynamic start date: 30 days ago
        default_start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")
        data_base = "https://data.alpaca.markets/v2"

        bars: dict = {}
        for sym in all_symbols[:8]:
            try:
                resp = req.get(
                    f"{data_base}/stocks/{sym}/bars",
                    headers=data_headers,
                    params={
                        "timeframe": "15Min",
                        "start": default_start,
                        "limit": "1000",
                        "sort": "asc",
                        "feed": "iex",
                    },
                    timeout=10,
                )
                if resp.ok:
                    raw_bars = resp.json().get("bars", [])
                    bars[sym] = [
                        {"t": b["t"], "c": b["c"], "h": b["h"], "l": b["l"], "o": b["o"]}
                        for b in raw_bars
                    ]
            except Exception:
                pass

        body = json.dumps({"accounts": accounts, "bars": bars}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=60")
        self.end_headers()
        self.wfile.write(body)
