"""Vercel serverless function: GET /api/history

Returns portfolio equity history + filled orders for trade markers.
"""

from http.server import BaseHTTPRequestHandler
import json
import os

import requests as req

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")

        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

        # Portfolio history (1 week, hourly)
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

        # Filled orders (last 50)
        orders = []
        try:
            resp = req.get(
                f"{ALPACA_BASE}/orders",
                headers=headers,
                params={"status": "closed", "limit": "50", "direction": "desc"},
                timeout=10,
            )
            if resp.ok:
                raw = resp.json()
                for o in raw:
                    if o.get("status") == "filled":
                        orders.append({
                            "symbol": o.get("symbol", ""),
                            "side": o.get("side", ""),
                            "qty": o.get("filled_qty", "0"),
                            "price": o.get("filled_avg_price", "0"),
                            "filled_at": o.get("filled_at", ""),
                            "intent": o.get("position_intent", ""),
                        })
        except Exception:
            pass

        # Per-symbol intraday bars for traded symbols
        bars = {}
        traded_symbols = list({o["symbol"] for o in orders if o["symbol"]})
        # Also include current positions
        try:
            pos_resp = req.get(f"{ALPACA_BASE}/positions", headers=headers, timeout=10)
            if pos_resp.ok:
                for p in pos_resp.json():
                    sym = p.get("symbol", "")
                    if sym and sym not in traded_symbols:
                        traded_symbols.append(sym)
        except Exception:
            pass

        data_base = "https://data.alpaca.markets/v2"
        for sym in traded_symbols[:6]:  # limit to 6 symbols
            try:
                resp = req.get(
                    f"{data_base}/stocks/{sym}/bars",
                    headers=headers,
                    params={
                        "timeframe": "5Min",
                        "limit": "200",
                        "sort": "desc",
                        "feed": "iex",
                    },
                    timeout=10,
                )
                if resp.ok:
                    raw_bars = resp.json().get("bars", [])
                    bars[sym] = [
                        {"t": b["t"], "c": b["c"], "h": b["h"], "l": b["l"], "o": b["o"]}
                        for b in reversed(raw_bars)
                    ]
            except Exception:
                pass

        data = {
            "portfolio": {
                "timestamps": portfolio.get("timestamp", []),
                "equity": portfolio.get("equity", []),
                "profit_loss": portfolio.get("profit_loss", []),
                "profit_loss_pct": portfolio.get("profit_loss_pct", []),
            },
            "orders": orders,
            "bars": bars,
        }

        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=60")
        self.end_headers()
        self.wfile.write(body)
