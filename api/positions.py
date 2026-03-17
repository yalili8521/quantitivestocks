"""Vercel serverless function: GET /api/positions

Returns live paper-trading accounts and positions for all four groups.
- Intraday + Swing: Alpaca paper API
- Crypto + Crypto Intraday: Kraken paper state from GitHub Gist (local paper simulation)
"""

from http.server import BaseHTTPRequestHandler
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


def _fetch_kraken_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch live prices from Kraken public API (no auth needed).

    Uses a static mapping for known Kraken pair quirks (BTC→XXBT, ETH→XETH,
    DOGE→XDG) and auto-generates the rest as {BASE}USD.
    """
    # Kraken uses non-standard names for some legacy pairs
    KRAKEN_OVERRIDES = {
        "BTC/USD": "XXBTZUSD", "ETH/USD": "XETHZUSD", "DOGE/USD": "XDGUSD",
    }

    def _to_kraken_pair(sym: str) -> str:
        if sym in KRAKEN_OVERRIDES:
            return KRAKEN_OVERRIDES[sym]
        # Generic: FIL/USD → FILUSD, OP/USD → OPUSD, etc.
        return sym.replace("/", "")

    sym_to_kraken = {s: _to_kraken_pair(s) for s in symbols}
    kraken_to_sym = {v: k for k, v in sym_to_kraken.items()}
    pairs_needed = list(sym_to_kraken.values())

    if not pairs_needed:
        return {}
    try:
        resp = req.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": ",".join(pairs_needed)},
            timeout=10,
        )
        if not resp.ok:
            return {}
        data = resp.json().get("result", {})
        prices = {}
        for pair, info in data.items():
            sym = kraken_to_sym.get(pair, pair)
            prices[sym] = float(info["c"][0])
        return prices
    except Exception:
        return {}


def _fetch_kraken_paper(state_filename: str = "kraken_paper_state.json") -> dict:
    """Read Kraken paper state from GitHub Gist + live prices from Kraken."""
    gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
    if not gist_id:
        return {"account": {}, "positions": []}

    try:
        resp = req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if not resp.ok:
            return {"account": {}, "positions": []}

        gist_data = resp.json()
        content = gist_data.get("files", {}).get(state_filename, {}).get("content", "{}")
        state = json.loads(content)
    except Exception:
        return {"account": {}, "positions": []}

    cash = state.get("cash", 0)
    initial = state.get("initial_balance", 100000)
    positions_raw = state.get("positions", {})

    # Fetch live prices for all held symbols
    live_prices = _fetch_kraken_prices(list(positions_raw.keys()))

    # Build position list with live P&L
    positions = []
    total_value = cash
    for sym, pos in positions_raw.items():
        entry_price = pos.get("entry_price", 0)
        qty = pos.get("qty", 0)
        side = pos.get("side", "LONG")
        current_price = live_prices.get(sym, entry_price)
        notional = current_price * qty

        if side == "LONG":
            pnl = (current_price - entry_price) * qty
            total_value += notional
        else:
            pnl = (entry_price - current_price) * qty
            total_value += pnl  # short P&L added to cash

        pnl_pct = pnl / (entry_price * qty) if entry_price * qty > 0 else 0

        positions.append({
            "symbol":          sym,
            "qty":             str(qty),
            "side":            "long" if side == "LONG" else "short",
            "avg_entry_price": str(entry_price),
            "current_price":   str(current_price),
            "unrealized_pl":   str(round(pnl, 2)),
            "unrealized_plpc": str(round(pnl_pct, 4)),
            "market_value":    str(round(notional, 2)),
        })

    return {
        "account": {
            "equity":       str(round(total_value, 2)),
            "cash":         str(cash),
            "buying_power": str(cash * 2),
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

        # Crypto group — Kraken paper state from Gist
        crypto_data = _fetch_kraken_paper("kraken_paper_state.json")
        accounts.append({
            "name":      "Crypto",
            "group":     "crypto",
            "account":   crypto_data["account"],
            "positions": crypto_data["positions"],
        })

        # Crypto Intraday group — separate Kraken paper state
        crypto_intraday_data = _fetch_kraken_paper("kraken_intraday_paper_state.json")
        accounts.append({
            "name":      "Crypto Intraday",
            "group":     "crypto_intraday",
            "account":   crypto_intraday_data["account"],
            "positions": crypto_intraday_data["positions"],
        })

        body = json.dumps({"accounts": accounts}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=30")
        self.end_headers()
        self.wfile.write(body)
