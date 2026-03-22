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

# No symbol filters — both groups use ETF selector which picks dynamically
# from a 42-symbol universe. Each group has its own dedicated Alpaca account.
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


def _fetch_alpaca(api_key: str, api_secret: str, allowed_symbols: set | None = None) -> dict:
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
            params={"status": "closed", "limit": "200", "direction": "desc"},
            timeout=10,
        )
        if resp.ok:
            for o in resp.json():
                if o.get("status") != "filled":
                    continue
                sym = o.get("symbol", "")
                # Skip crypto symbols on Alpaca ETF accounts
                if "/" in sym:
                    continue
                # Filter out orders for symbols not in this group
                if allowed_symbols and sym not in allowed_symbols:
                    continue
                orders.append({
                    "symbol":    sym,
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
                if allowed_symbols and sym not in allowed_symbols:
                    continue
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
        "orders":         _recent_orders(orders),
        "trades":         _recent_trades(_pair_closed_trades(orders)),
        "traded_symbols": traded_symbols,
    }


def _pair_closed_trades(orders: list) -> list:
    """Pair open→close orders per symbol into round-trip closed trades."""
    chrono = sorted(orders, key=lambda o: o.get("filled_at", ""))
    entry_prices: dict = {}   # symbol → entry price
    entry_times: dict = {}    # symbol → entry filled_at
    trades: list = []

    for o in chrono:
        sym = o.get("symbol", "")
        intent = o.get("intent", "")
        try:
            price = float(o.get("price", 0))
            qty = float(o.get("qty", 0))
        except (ValueError, TypeError):
            continue
        if not sym or not price:
            continue

        if "open" in intent:
            entry_prices[sym] = price
            entry_times[sym] = o.get("filled_at", "")
        elif "close" in intent and sym in entry_prices:
            entry = entry_prices[sym]
            is_long = o.get("side") == "sell"  # sell_to_close → was long
            if is_long:
                pnl_dollar = qty * (price - entry)
            else:
                pnl_dollar = qty * (entry - price)
            pnl_pct = ((price - entry) / entry * 100) if is_long else ((entry - price) / entry * 100)

            trades.append({
                "symbol":      sym,
                "direction":   "LONG" if is_long else "SHORT",
                "qty":         o.get("qty", "0"),
                "entry_price": str(round(entry, 6)),
                "exit_price":  o.get("price", "0"),
                "market_value": round(qty * entry, 2),
                "pnl_dollar":  round(pnl_dollar, 2),
                "pnl_pct":     round(pnl_pct, 2),
                "opened_at":   entry_times.get(sym, ""),
                "closed_at":   o.get("filled_at", ""),
            })
            del entry_prices[sym]
            entry_times.pop(sym, None)

    return trades


def _recent_orders(orders: list, days: int = 3) -> list:
    """Return only orders from the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    return [o for o in orders if (o.get("filled_at") or "") >= cutoff_iso]


def _recent_trades(trades: list, max_days: int = 3) -> list:
    """Return trades from the most recent N traded days within the last 7 calendar days."""
    if not trades:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    # Only consider trades within the last 7 calendar days
    recent = [t for t in trades if (t.get("closed_at") or "")[:10] >= cutoff]
    # Then pick the 3 most recent traded days from that window
    day_set: set = set()
    for t in recent:
        ca = t.get("closed_at", "")
        if ca:
            day_set.add(ca[:10])
    if not day_set:
        return []
    recent_days = sorted(day_set, reverse=True)[:max_days]
    return [t for t in recent if (t.get("closed_at") or "")[:10] in recent_days]


def _build_equity_curve(trade_log: list, initial_balance: float) -> dict:
    """Synthesize an equity curve from the trade log.

    Walks through trades chronologically, tracking cash and position value
    at each trade to produce timestamps + equity arrays compatible with the
    Alpaca portfolio history format (unix epoch seconds).
    """
    if not trade_log:
        return {}

    sorted_trades = sorted(trade_log, key=lambda t: t.get("time", ""))

    cash = initial_balance
    positions: dict = {}  # symbol -> signed qty (positive=long, negative=short)
    timestamps: list[int] = []
    equity: list[float] = []
    last_prices: dict = {}  # symbol -> last known price

    for t in sorted_trades:
        sym = t.get("symbol", "")
        side = t.get("side", "")
        qty = float(t.get("qty", 0))
        price = float(t.get("price", 0))
        time_str = t.get("time", "")

        if not sym or not time_str or qty == 0 or price == 0:
            continue

        last_prices[sym] = price

        # Cash flow: buy spends cash, sell receives cash (regardless of intent)
        if side == "buy":
            cash -= qty * price
            positions[sym] = positions.get(sym, 0.0) + qty
        else:  # sell
            cash += qty * price
            positions[sym] = positions.get(sym, 0.0) - qty

        # Clean up flat positions
        if sym in positions and abs(positions[sym]) < 1e-10:
            del positions[sym]

        # Equity = cash + sum(signed_qty * current_price)
        # Long (positive qty): adds market value
        # Short (negative qty): subtracts market value (cash already credited on open)
        pos_value = 0.0
        for s, signed_qty in positions.items():
            lp = last_prices.get(s, 0)
            pos_value += signed_qty * lp

        try:
            ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            timestamps.append(int(ts.timestamp()))
        except (ValueError, TypeError):
            continue
        equity.append(cash + pos_value)

    if not timestamps:
        return {}

    return {
        "timestamps": timestamps,
        "equity": equity,
        "profit_loss": [],
        "profit_loss_pct": [],
    }


def _fetch_gist_files() -> dict:
    """Fetch all files from the Kraken state Gist (single API call)."""
    gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
    if not gist_id:
        return {}
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
        if gh_token:
            headers["Authorization"] = f"token {gh_token}"
        resp = req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers,
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("files", {})
    except Exception:
        pass
    return {}


def _parse_kraken_history(
    gist_files: dict,
    trade_log_filename: str = "kraken_trade_log.json",
    state_filename: str = "kraken_paper_state.json",
) -> dict:
    """Parse Kraken paper trade log + state from pre-fetched Gist files."""
    if not gist_files:
        return {"portfolio": {}, "orders": [], "trades": []}

    try:
        trade_log_content = gist_files.get(trade_log_filename, {}).get("content", "[]")
        trade_log = json.loads(trade_log_content) if trade_log_content else []

        state_content = gist_files.get(state_filename, {}).get("content", "{}")
        state = json.loads(state_content) if state_content else {}
        initial_balance = float(state.get("initial_balance", 100000))

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

        portfolio = _build_equity_curve(trade_log, initial_balance)

    except Exception:
        return {"portfolio": {}, "orders": [], "trades": []}

    return {
        "portfolio": portfolio,
        "orders": _recent_orders(orders),
        "trades": _recent_trades(_pair_closed_trades(orders)),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        accounts = []
        all_symbols: list[str] = []

        # Alpaca groups (intraday, swing)
        for cfg in ACCOUNTS:
            key    = os.environ.get(cfg["key_env"], "")
            secret = os.environ.get(cfg["secret_env"], "")
            data   = _fetch_alpaca(key, secret, cfg.get("symbols"))
            accounts.append({
                "name":      cfg["name"],
                "group":     cfg["group"],
                "portfolio": data["portfolio"],
                "orders":    data["orders"],
                "trades":    data.get("trades", []),
            })
            for sym in data.get("traded_symbols", []):
                if sym not in all_symbols:
                    all_symbols.append(sym)

        # Single Gist fetch for all crypto data + selector rankings
        gist_files = _fetch_gist_files()

        # Crypto group — Kraken paper trade log from Gist
        crypto_data = _parse_kraken_history(
            gist_files, "kraken_trade_log.json", "kraken_paper_state.json"
        )
        accounts.append({
            "name":      "Crypto",
            "group":     "crypto",
            "portfolio": crypto_data["portfolio"],
            "orders":    crypto_data["orders"],
            "trades":    crypto_data.get("trades", []),
        })

        # Crypto Intraday group — separate Kraken paper state
        crypto_intraday_data = _parse_kraken_history(
            gist_files, "kraken_intraday_trade_log.json", "kraken_intraday_paper_state.json"
        )
        accounts.append({
            "name":      "Crypto 5m",
            "group":     "crypto_intraday",
            "portfolio": crypto_intraday_data["portfolio"],
            "orders":    crypto_intraday_data["orders"],
            "trades":    crypto_intraday_data.get("trades", []),
        })

        # Gold Scalper — paper state from same Gist
        gold_data = _parse_kraken_history(
            gist_files, "gold_scalper_trade_log.json", "gold_scalper_state.json"
        )
        accounts.append({
            "name":      "Gold MGC",
            "group":     "gold_scalper",
            "portfolio": gold_data["portfolio"],
            "orders":    gold_data["orders"],
            "trades":    gold_data.get("trades", []),
        })

        # Selector rankings from same Gist
        ranked_symbols: dict = {}
        try:
            content = gist_files.get("selector_rankings.json", {}).get("content", "{}")
            ranked_symbols = json.loads(content) if content else {}
        except Exception:
            pass

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

        body = json.dumps({"accounts": accounts, "bars": bars, "ranked_symbols": ranked_symbols}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=60")
        self.end_headers()
        self.wfile.write(body)
