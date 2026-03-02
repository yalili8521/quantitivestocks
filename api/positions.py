"""Vercel serverless function: GET /api/positions

Returns live Alpaca paper-trading accounts and positions for all three groups.
Uses raw REST API (no alpaca-py) to keep bundle small.
"""

from http.server import BaseHTTPRequestHandler
import json
import os

import requests as req

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

ACCOUNTS = [
    {
        "name": "Options / Intraday",
        "group": "intraday",
        "key_env": "ALPACA_API_KEY",
        "secret_env": "ALPACA_API_SECRET",
    },
    {
        "name": "Swing",
        "group": "swing",
        "key_env": "ALPACA_SWING_KEY",
        "secret_env": "ALPACA_SWING_SECRET",
    },
    {
        "name": "Expansion",
        "group": "expansion",
        "key_env": "ALPACA_EXPANSION_KEY",
        "secret_env": "ALPACA_EXPANSION_SECRET",
    },
]


def _fetch(api_key: str, api_secret: str) -> dict:
    if not api_key or not api_secret:
        return {"account": {}, "positions": [], "_status": "no_key"}

    headers = {
        "APCA-API-KEY-ID": api_key.strip(),
        "APCA-API-SECRET-KEY": api_secret.strip(),
    }

    _acct_status = 0
    try:
        acct_resp = req.get(f"{ALPACA_BASE}/account", headers=headers, timeout=10)
        _acct_status = acct_resp.status_code
        acct = acct_resp.json() if acct_resp.ok else {}
    except Exception as e:
        acct = {}
        _acct_status = -1

    try:
        pos_resp = req.get(f"{ALPACA_BASE}/positions", headers=headers, timeout=10)
        positions = pos_resp.json() if pos_resp.ok else []
    except Exception:
        positions = []

    # Alpaca's `equity` field can be "0" on idle/new paper accounts.
    # Fall back to `portfolio_value`, then to `cash`, in that order.
    def _best_equity(a: dict) -> str:
        for field in ("equity", "portfolio_value", "cash"):
            val = a.get(field, "0")
            try:
                if float(val) > 0:
                    return val
            except (TypeError, ValueError):
                pass
        return "0"

    return {
        "_status": _acct_status,
        "account": {
            "equity":       _best_equity(acct),
            "cash":         acct.get("cash", "0"),
            "buying_power": acct.get("buying_power", "0"),
        },
        "positions": [
            {
                "symbol":          p.get("symbol", ""),
                "qty":             p.get("qty", "0"),
                "side":            p.get("side", ""),
                "avg_entry_price": p.get("avg_entry_price", "0"),
                "current_price":   p.get("current_price", "0"),
                "unrealized_pl":   p.get("unrealized_pl", "0"),
                "unrealized_plpc": p.get("unrealized_plpc", "0"),
                "market_value":    p.get("market_value", "0"),
            }
            for p in positions
            if isinstance(p, dict)
        ],
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        accounts = []
        for cfg in ACCOUNTS:
            key    = os.environ.get(cfg["key_env"], "")
            secret = os.environ.get(cfg["secret_env"], "")
            data   = _fetch(key, secret)
            accounts.append({
                "name":      cfg["name"],
                "group":     cfg["group"],
                "account":   data["account"],
                "positions": data["positions"],
                "_status":   data.get("_status", 0),
            })

        debug = {k: (os.environ.get(k,"")[:4] + "...") for k in
                 ["ALPACA_API_KEY","ALPACA_SWING_KEY","ALPACA_EXPANSION_KEY"]}
        body = json.dumps({"accounts": accounts, "_debug_keys": debug}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=30")
        self.end_headers()
        self.wfile.write(body)
