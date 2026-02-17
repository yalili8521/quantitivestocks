"""Vercel serverless function: GET /api/positions

Returns live Alpaca paper-trading account and positions.
Uses raw REST API (no alpaca-py) to keep bundle small.
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

        # Fetch account
        try:
            acct_resp = req.get(f"{ALPACA_BASE}/account", headers=headers, timeout=10)
            acct = acct_resp.json() if acct_resp.ok else {}
        except Exception:
            acct = {}

        # Fetch positions
        try:
            pos_resp = req.get(f"{ALPACA_BASE}/positions", headers=headers, timeout=10)
            positions = pos_resp.json() if pos_resp.ok else []
        except Exception:
            positions = []

        data = {
            "account": {
                "equity": acct.get("equity", "0"),
                "cash": acct.get("cash", "0"),
                "buying_power": acct.get("buying_power", "0"),
            },
            "positions": [
                {
                    "symbol": p.get("symbol", ""),
                    "qty": p.get("qty", "0"),
                    "side": p.get("side", ""),
                    "avg_entry_price": p.get("avg_entry_price", "0"),
                    "current_price": p.get("current_price", "0"),
                    "unrealized_pl": p.get("unrealized_pl", "0"),
                    "unrealized_plpc": p.get("unrealized_plpc", "0"),
                    "market_value": p.get("market_value", "0"),
                }
                for p in positions
                if isinstance(p, dict)
            ],
        }

        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=30")
        self.end_headers()
        self.wfile.write(body)
