#!/usr/bin/env python3
import os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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
                k, v = line[:idx].strip(), line[idx+1:].strip().strip('"')
                os.environ[k] = v

import requests

api_key    = os.environ.get("ALPACA_API_KEY")
api_secret = os.environ.get("ALPACA_API_SECRET")

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

et = ZoneInfo("America/New_York")
now_et = datetime.now(et)
today_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
after_str = today_start.strftime("%Y-%m-%dT%H:%M:%SZ")

BASE = "https://paper-api.alpaca.markets/v2"
headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}

resp = requests.get(
    f"{BASE}/orders",
    headers=headers,
    params={"status": "closed", "after": after_str, "direction": "asc", "limit": 500},
    timeout=15,
)
resp.raise_for_status()
orders = resp.json()

filled = [o for o in orders if float(o.get("filled_qty") or 0) > 0]
if not filled:
    print(f"No filled orders today ({now_et.date()}).")
    sys.exit(0)

position = defaultdict(int)

def label_and_explain(o):
    side = o["side"].upper()
    qty  = int(float(o["filled_qty"]))
    sym  = o["symbol"]
    before = position[sym]
    if side == "BUY":
        if before < 0:
            action = "COVER (close short)"
            why = ("Closed an existing short position. This is triggered by one of:\n"
                   "    • Trailing stop     — price rose too far from the short's best price\n"
                   "    • Take-profit       — short hit the profit target\n"
                   "    • Signal flip       — ML model turned bullish (UP with enough confidence)\n"
                   "    • Max loss          — loss hit the ATR-based hard floor\n"
                   "    • Regime exit       — ADX or Hurst said the trend collapsed")
        else:
            action = "BUY (open long)"
            why = ("Opened a new long position. Conditions that had to be true:\n"
                   "    • ML model predicted UP with confidence above the long threshold (0.01)\n"
                   "    • Trend filter: +DI > -DI  (trend agrees with UP)\n"
                   "    • ADX above 20 (market is trending)\n"
                   "    • Hurst exponent > 0.55 (momentum is valid)\n"
                   "    • Not in cooldown after a recent loss exit\n"
                   "    • VIX halt not active, time filter allowed entry")
        position[sym] += qty
    else:  # SELL
        if before > 0:
            action = "SELL (close long)"
            why = ("Closed an existing long position. This is triggered by one of:\n"
                   "    • Trailing stop     — price dropped too far from the long's peak\n"
                   "    • Take-profit       — long hit the profit target\n"
                   "    • Signal flip       — ML model turned bearish (DOWN with enough confidence)\n"
                   "    • Max loss          — loss hit the ATR-based hard floor\n"
                   "    • Regime exit       — ADX or Hurst said the trend collapsed")
        else:
            action = "SHORT (open short)"
            why = ("Opened a new short position. Conditions that had to be true:\n"
                   "    • ML model predicted DOWN with confidence above the short threshold (0.01)\n"
                   "    • Trend filter: -DI > +DI  (trend agrees with DOWN)\n"
                   "    • ADX above 20 (market is trending)\n"
                   "    • Hurst exponent > 0.55 (momentum is valid)\n"
                   "    • Not in cooldown after a recent loss exit\n"
                   "    • VIX halt not active, time filter allowed entry")
        position[sym] -= qty
    return action, why

print(f"\nToday's paper trades  ({now_et.date()})  —  {len(filled)} filled order(s)\n")
print("=" * 80)

for o in filled:
    sym   = o["symbol"]
    side  = o["side"].upper()
    qty   = int(float(o["filled_qty"]))
    price = float(o.get("filled_avg_price") or 0)
    t     = (o.get("filled_at") or o.get("submitted_at") or "")[:19]
    otype = o.get("type", "market")
    action, why = label_and_explain(o)

    print(f"\n  {t}  |  {sym:16}  |  {side:4}  x{qty:<6}  @  ${price:.2f}")
    print(f"  Action : {action}")
    print(f"  Why    : {why}")

print("\n" + "=" * 80)
print(f"  Total trades today: {len(filled)}")
print()
