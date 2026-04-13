"""Vercel serverless function: GET /api/history

Returns portfolio equity history + filled orders for all account groups.
- Intraday + Swing + BTC: Alpaca paper API
- Gold Scalper: paper state from GitHub Gist (trade log)
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
    {
        "name": "BTC",
        "group": "btc",
        "key_env": "ALPACA_CRYPTO_KEY",
        "secret_env": "ALPACA_CRYPTO_SECRET",
    },
]


def _fetch_alpaca(api_key: str, api_secret: str, allowed_symbols: set | None = None, skip_crypto: bool = True) -> dict:
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
                if skip_crypto and "/" in sym:
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


def _pair_closed_trades(orders: list, fee_pct: float = 0.0) -> list:
    """Pair open→close orders per symbol into round-trip closed trades.

    Handles position overwrites (double opens) by implicitly closing the old
    position at the new open's price.  When *fee_pct* > 0, fees on both legs
    are subtracted from P&L.
    """
    chrono = sorted(orders, key=lambda o: o.get("filled_at", ""))
    # symbol → {price, qty, side, time}
    entries: dict = {}
    trades: list = []

    def _close_trade(sym, entry, entry_qty, entry_side, entry_time,
                     exit_price, exit_qty, exit_time):
        is_long = entry_side == "buy"
        qty = min(entry_qty, exit_qty)
        if qty < 1e-10 or entry <= 0:
            return
        notional = qty * entry
        if is_long:
            raw_pnl = qty * (exit_price - entry)
        else:
            raw_pnl = qty * (entry - exit_price)
        fee = (qty * entry + qty * exit_price) * fee_pct
        pnl_dollar = raw_pnl - fee
        pnl_pct = (pnl_dollar / notional * 100) if notional else 0.0

        trades.append({
            "symbol":      sym,
            "direction":   "LONG" if is_long else "SHORT",
            "qty":         str(round(qty, 6)),
            "entry_price": str(round(entry, 6)),
            "exit_price":  str(round(exit_price, 6)),
            "market_value": round(notional, 2),
            "pnl_dollar":  round(pnl_dollar, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "opened_at":   entry_time,
            "closed_at":   exit_time,
        })

    for o in chrono:
        sym = o.get("symbol", "")
        intent = o.get("intent", "")
        side = o.get("side", "")
        try:
            price = float(o.get("price", 0))
            qty = float(o.get("qty", 0))
        except (ValueError, TypeError):
            continue
        if not sym or not price:
            continue
        filled_at = o.get("filled_at", "")

        if "open" in intent:
            # If reopening a symbol that's already open, close the old one
            if sym in entries:
                e = entries[sym]
                _close_trade(sym, e["price"], e["qty"], e["side"],
                             e["time"], price, e["qty"], filled_at)
            entries[sym] = {"price": price, "qty": qty, "side": side,
                            "time": filled_at}
        elif "close" in intent and sym in entries:
            e = entries[sym]
            _close_trade(sym, e["price"], e["qty"], e["side"],
                         e["time"], price, qty, filled_at)
            # Handle partial close
            remaining = e["qty"] - qty
            if remaining > 1e-10:
                e["qty"] = remaining
            else:
                del entries[sym]

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


def _fetch_gist_files() -> dict:
    """Fetch all files from the state Gist (single API call)."""
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


def _parse_gold_scalper_history(
    gist_files: dict,
    trade_log_filename: str = "gold_scalper_trade_log.json",
    state_filename: str = "gold_scalper_state.json",
) -> dict:
    """Parse gold scalper trade log from Gist.

    Gold scalper log format (per-exit, not paired orders):
      {"time", "symbol", "direction", "contracts", "entry", "exit", "pnl", "reason"}
    """
    if not gist_files:
        return {"portfolio": {}, "orders": [], "trades": []}

    try:
        log_content = gist_files.get(trade_log_filename, {}).get("content", "[]")
        trade_log = json.loads(log_content) if log_content else []

        state_content = gist_files.get(state_filename, {}).get("content", "{}")
        state = json.loads(state_content) if state_content else {}
        initial_balance = float(state.get("initial_balance", 5000))

        # Convert to orders format for _recent_orders
        orders = []
        for t in trade_log:
            orders.append({
                "symbol":    t.get("symbol", "MGC"),
                "side":      "sell" if t.get("direction") == "SHORT" else "buy",
                "qty":       str(t.get("contracts", 0)),
                "price":     str(t.get("exit", 0)),
                "filled_at": t.get("time", ""),
                "intent":    "",
            })

        # Convert to closed trades format directly (each TP exit = one row)
        trades = []
        for t in trade_log:
            entry = float(t.get("entry", 0))
            exit_p = float(t.get("exit", 0))
            contracts = int(t.get("contracts", 0))
            pnl = float(t.get("pnl", 0))
            direction = t.get("direction", "LONG")
            pnl_pct = (((entry - exit_p) / entry * 100) if direction == "SHORT" else ((exit_p - entry) / entry * 100)) if entry else 0.0

            trades.append({
                "symbol":       t.get("symbol", "MGC"),
                "direction":    direction,
                "qty":          str(contracts),
                "entry_price":  str(round(entry, 2)),
                "exit_price":   str(round(exit_p, 2)),
                "market_value": round(contracts * entry, 2),
                "pnl_dollar":   round(pnl, 2),
                "pnl_pct":      round(pnl_pct, 2),
                "opened_at":    "",
                "closed_at":    t.get("time", ""),
                "reason":       t.get("reason", ""),
            })

        # Build equity curve
        equity_curve = {}
        if trade_log:
            cash = initial_balance
            timestamps = []
            equity_vals = []
            for t in sorted(trade_log, key=lambda x: x.get("time", "")):
                cash += float(t.get("pnl", 0))
                try:
                    ts = datetime.fromisoformat(t["time"].replace("Z", "+00:00"))
                    timestamps.append(int(ts.timestamp()))
                except (ValueError, KeyError):
                    continue
                equity_vals.append(cash)
            if timestamps:
                equity_curve = {
                    "timestamps": timestamps,
                    "equity": equity_vals,
                    "profit_loss": [],
                    "profit_loss_pct": [],
                }

    except Exception:
        return {"portfolio": {}, "orders": [], "trades": []}

    return {
        "portfolio": equity_curve,
        "orders":    _recent_orders(orders),
        "trades":    _recent_trades(trades),
    }


def _parse_tqqq_history(gist_files: dict) -> dict:
    """Parse TQQQ trade log from Gist (Webull live trading).

    Trade log format (from webull_tqqq_executor):
      {"direction", "entry_price", "exit_price", "qty", "pnl_usd", "pnl_pct",
       "comment", "entry_time", "exit_time", "equity_after", "live"}
    """
    if not gist_files:
        return {"portfolio": {}, "orders": [], "trades": []}

    try:
        state_content = gist_files.get("webull_tqqq_state.json", {}).get("content", "{}")
        state = json.loads(state_content) if state_content else {}
        initial_equity = float(state.get("initial_equity", 0))

        log_content = gist_files.get("webull_tqqq_trade_log.json", {}).get("content", "[]")
        trade_log = json.loads(log_content) if log_content else []

        orders = []
        trades = []
        for t in trade_log:
            entry_price = float(t.get("entry_price", 0))
            exit_price = float(t.get("exit_price", 0))
            qty = int(t.get("qty", 0))
            pnl = float(t.get("pnl_usd", 0))
            direction = t.get("direction", "LONG")
            pnl_pct = float(t.get("pnl_pct", 0))

            trades.append({
                "symbol":       "TQQQ",
                "direction":    direction,
                "qty":          str(qty),
                "entry_price":  str(round(entry_price, 2)),
                "exit_price":   str(round(exit_price, 2)),
                "market_value": round(qty * entry_price, 2),
                "pnl_dollar":   round(pnl, 2),
                "pnl_pct":      round(pnl_pct, 2),
                "opened_at":    t.get("entry_time", ""),
                "closed_at":    t.get("exit_time", ""),
                "reason":       t.get("comment", ""),
            })

            orders.append({
                "symbol":    "TQQQ",
                "side":      "sell" if direction == "LONG" else "buy",
                "qty":       str(qty),
                "price":     str(exit_price),
                "filled_at": t.get("exit_time", ""),
                "intent":    "",
            })

        # Build equity curve from trade log
        equity_curve = {}
        if trade_log:
            cash = initial_equity if initial_equity > 0 else 147.22
            timestamps = []
            equity_vals = []
            for t in sorted(trade_log, key=lambda x: x.get("exit_time", "")):
                cash = float(t.get("equity_after", cash + float(t.get("pnl_usd", 0))))
                try:
                    ts = datetime.fromisoformat(t["exit_time"].replace("Z", "+00:00"))
                    timestamps.append(int(ts.timestamp()))
                except (ValueError, KeyError):
                    continue
                equity_vals.append(cash)
            if timestamps:
                equity_curve = {
                    "timestamps": timestamps,
                    "equity": equity_vals,
                    "profit_loss": [],
                    "profit_loss_pct": [],
                }

    except Exception:
        return {"portfolio": {}, "orders": [], "trades": []}

    return {
        "portfolio": equity_curve,
        "orders":    _recent_orders(orders),
        "trades":    _recent_trades(trades),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        accounts = []
        all_symbols: list[str] = []

        # Alpaca groups (intraday, swing)
        for cfg in ACCOUNTS:
            key    = os.environ.get(cfg["key_env"], "")
            secret = os.environ.get(cfg["secret_env"], "")
            is_crypto = cfg["group"] == "btc"
            data   = _fetch_alpaca(key, secret, cfg.get("symbols"), skip_crypto=not is_crypto)
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

        # Single Gist fetch for gold scalper data + selector rankings
        gist_files = _fetch_gist_files()

        # Gold Scalper — paper state from Gist (different trade log format)
        gold_data = _parse_gold_scalper_history(
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
