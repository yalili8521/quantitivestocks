"""Vercel serverless function: GET /api/positions

Returns live paper-trading accounts and positions for all groups.
- Intraday + Swing: Alpaca paper API
- Gold Scalper: paper state from GitHub Gist
- BTC: Alpaca crypto paper API
"""

from http.server import BaseHTTPRequestHandler
import json
import os

import requests as req

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

ACCOUNTS = [
    {
        "name": "ETF 5m",
        "group": "intraday",
        "key_env": "ALPACA_INTRADAY_KEY",
        "secret_env": "ALPACA_INTRADAY_SECRET",
    },
    {
        "name": "ETF Swing",
        "group": "swing",
        "key_env": "ALPACA_SWING_KEY",
        "secret_env": "ALPACA_SWING_SECRET",
    },
]


def _fetch_alpaca(api_key: str, api_secret: str) -> dict:
    if not api_key or not api_secret:
        return {"account": {}, "positions": []}

    headers = {
        "APCA-API-KEY-ID": api_key.strip(),
        "APCA-API-SECRET-KEY": api_secret.strip(),
    }

    try:
        acct_resp = req.get(f"{ALPACA_BASE}/account", headers=headers, timeout=10)
        acct = acct_resp.json() if acct_resp.ok else {}
    except Exception:
        acct = {}

    try:
        pos_resp = req.get(f"{ALPACA_BASE}/positions", headers=headers, timeout=10)
        positions = pos_resp.json() if pos_resp.ok else []
    except Exception:
        positions = []

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


def _fetch_gist_files() -> dict:
    """Fetch all files from state Gist (single API call, shared by gold + tqqq)."""
    gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
    if not gist_id:
        return {}
    try:
        resp = req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("files", {})
    except Exception:
        pass
    return {}


def _fetch_gold_scalper(gist_files: dict) -> dict:
    """Read gold scalper paper state from Gist files."""
    _empty = {"account": {"equity": "0", "cash": "0", "buying_power": "0"}, "positions": []}
    try:
        content = gist_files.get("gold_scalper_state.json", {}).get("content", "{}")
        state = json.loads(content)
    except Exception:
        return _empty

    equity = state.get("equity", 0)
    initial = state.get("initial_balance", 5000)
    position = state.get("position")
    trade_count = state.get("trade_count", 0)

    positions = []
    if position and position.get("contracts", 0) > 0:
        entry_price = position.get("entry_price", 0)
        contracts = position.get("contracts", 0)
        direction = position.get("direction", "LONG")

        # Fetch live gold price
        try:
            gold_resp = req.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if gold_resp.ok:
                chart = gold_resp.json().get("chart", {}).get("result", [{}])[0]
                current_price = chart.get("meta", {}).get("regularMarketPrice", entry_price)
            else:
                current_price = entry_price
        except Exception:
            current_price = entry_price

        pip_val = 0.10
        if direction == "LONG":
            pips = (current_price - entry_price) / pip_val
        else:
            pips = (entry_price - current_price) / pip_val
        pnl = pips * contracts  # $1 per pip per MGC contract

        positions.append({
            "symbol":          "MGC",
            "qty":             str(contracts),
            "side":            "long" if direction == "LONG" else "short",
            "avg_entry_price": str(entry_price),
            "current_price":   str(current_price),
            "unrealized_pl":   str(round(pnl, 2)),
            "unrealized_plpc": str(round(pnl / (entry_price * contracts) if entry_price * contracts > 0 else 0, 4)),
            "market_value":    str(round(current_price * contracts, 2)),
        })

    return {
        "account": {
            "equity":       str(round(equity, 2)),
            "cash":         str(round(equity, 2)),  # no margin tracking in paper mode
            "buying_power": str(round(equity * 2, 2)),
        },
        "positions": positions,
    }


def _fetch_tqqq(gist_files: dict) -> dict:
    """Read TQQQ state from Gist files (Webull live trading)."""
    _empty = {"account": {"equity": "0", "cash": "0", "buying_power": "0"}, "positions": []}
    try:
        content = gist_files.get("webull_tqqq_state.json", {}).get("content", "{}")
        state = json.loads(content)
    except Exception:
        return _empty

    equity = state.get("equity", 0)
    position = state.get("position")

    positions = []
    if position and position.get("qty", 0) > 0:
        entry_price = position.get("entry_price", 0)
        qty = position.get("qty", 0)
        direction = position.get("direction", "LONG")

        # Fetch live TQQQ price
        try:
            tqqq_resp = req.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/TQQQ?interval=1m&range=1d",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if tqqq_resp.ok:
                chart = tqqq_resp.json().get("chart", {}).get("result", [{}])[0]
                current_price = chart.get("meta", {}).get("regularMarketPrice", entry_price)
            else:
                current_price = entry_price
        except Exception:
            current_price = entry_price

        if direction == "LONG":
            pnl = qty * (current_price - entry_price)
        else:
            pnl = qty * (entry_price - current_price)

        positions.append({
            "symbol":          "TQQQ",
            "qty":             str(qty),
            "side":            "long" if direction == "LONG" else "short",
            "avg_entry_price": str(entry_price),
            "current_price":   str(current_price),
            "unrealized_pl":   str(round(pnl, 2)),
            "unrealized_plpc": str(round(pnl / (entry_price * qty) if entry_price * qty > 0 else 0, 4)),
            "market_value":    str(round(current_price * qty, 2)),
        })

    return {
        "account": {
            "equity":       str(round(equity, 2)),
            "cash":         str(round(equity, 2)),
            "buying_power": str(round(equity, 2)),
        },
        "positions": positions,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        accounts = []

        # Alpaca groups (intraday, swing)
        for cfg in ACCOUNTS:
            key    = os.environ.get(cfg["key_env"], "")
            secret = os.environ.get(cfg["secret_env"], "")
            data   = _fetch_alpaca(key, secret)
            accounts.append({
                "name":      cfg["name"],
                "group":     cfg["group"],
                "account":   data["account"],
                "positions": data["positions"],
            })

        # Single Gist fetch for gold + tqqq
        gist_files = _fetch_gist_files()

        # Gold Scalper — micro gold paper state from Gist
        gold_data = _fetch_gold_scalper(gist_files)
        accounts.append({
            "name":      "Gold MGC",
            "group":     "gold_scalper",
            "account":   gold_data["account"],
            "positions": gold_data["positions"],
        })

        # BTC — Alpaca crypto paper account
        btc_key    = os.environ.get("ALPACA_CRYPTO_KEY", "")
        btc_secret = os.environ.get("ALPACA_CRYPTO_SECRET", "")
        btc_data   = _fetch_alpaca(btc_key, btc_secret)
        accounts.append({
            "name":      "BTC",
            "group":     "btc",
            "account":   btc_data["account"],
            "positions": btc_data["positions"],
        })

        body = json.dumps({"accounts": accounts}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=30")
        self.end_headers()
        self.wfile.write(body)
