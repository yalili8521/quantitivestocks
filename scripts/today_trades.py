#!/usr/bin/env python3
"""Fetch today's paper trades from Alpaca and explain each one."""
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# Load env from secrets/alpaca.env
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
env_path = os.path.join(project_root, "secrets", "alpaca.env")
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx = line.find("=")
            if idx > 0:
                k, v = line[:idx].strip(), line[idx + 1 :].strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                os.environ[k] = v

api_key = os.environ.get("ALPACA_API_KEY")
api_secret = os.environ.get("ALPACA_API_SECRET")
if not api_key or not api_secret:
    print("Missing ALPACA_API_KEY or ALPACA_API_SECRET (set env or secrets/alpaca.env)")
    sys.exit(1)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
et = ZoneInfo("America/New_York")
now_et = datetime.now(et)
today_start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
today_start_utc = today_start_et.astimezone(timezone.utc)
now_utc = datetime.now(timezone.utc)
after_str = today_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
until_str = (now_utc + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

client = TradingClient(api_key, api_secret, paper=True)
req = GetOrdersRequest(
    status="closed",
    after=after_str,
    until=until_str,
    direction="asc",
    limit=500,
)
orders = list(client.get_orders(req))

filled = [o for o in orders if int(float(o.filled_qty or 0)) > 0]
if not filled:
    print(f"No filled orders today ({now_et.date()}) in paper account.")
    sys.exit(0)

def order_time(o):
    t = getattr(o, "filled_at", None) or getattr(o, "submitted_at", None)
    return t or ""

filled.sort(key=order_time)

# Infer position state before each order (qty: + long, - short, 0 flat)
position = defaultdict(int)
def label_order(o):
    side = (o.side.value if hasattr(o.side, "value") else str(o.side)).upper()
    qty = int(float(o.filled_qty or 0))
    sym = o.symbol
    before = position[sym]
    if side == "BUY":
        if before < 0:
            action = "COVER (close short)"
            position[sym] += qty
        else:
            action = "BUY (open/add long)"
            position[sym] += qty
    else:
        if before > 0:
            action = "SELL (close long)"
            position[sym] -= qty
        else:
            action = "SHORT (open/add short)"
            position[sym] -= qty
    return action

def explain(action, symbol):
    if "COVER" in action:
        return "Closing a short — triggered by trailing stop, take-profit, max loss, or signal flip (ML turned bullish)."
    if "BUY" in action and "long" in action:
        return "Opening or adding to a long — ML predicted UP with confidence above threshold and trend (+DI > -DI) agreed."
    if "SELL" in action and "close long" in action:
        return "Closing a long — triggered by trailing stop, take-profit, signal flip (ML turned bearish), or regime exit."
    if "SHORT" in action:
        return "Opening or adding to a short — ML predicted DOWN with confidence above threshold and trend (-DI > +DI) agreed."
    return "Executed by the paper trader per entry/exit rules."

print(f"\nToday's paper trades ({now_et.date()}) — {len(filled)} filled order(s)\n")
print("-" * 80)

for o in filled:
    symbol = o.symbol
    side = (o.side.value if hasattr(o.side, "value") else str(o.side)).upper()
    qty = int(float(o.filled_qty or 0))
    avg_price = float(o.filled_avg_price or 0)
    filled_at = getattr(o, "filled_at", None) or getattr(o, "submitted_at", None)
    time_str = str(filled_at)[:19] if filled_at else ""
    order_type = getattr(o, "type", None) or "market"
    action = label_order(o)
    reason = explain(action, symbol)

    print(f"  {time_str}  {symbol:14}  {side:4}  x{qty:<6}  @ ${avg_price:.2f}")
    print(f"    Action: {action}")
    print(f"    Why:    {reason}")
    print()
