#!/usr/bin/env python3
import os, sys, requests
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "secrets", "alpaca.env")
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            idx = line.find("=")
            if idx > 0:
                k, v = line[:idx].strip(), line[idx+1:].strip().strip('"')
                os.environ[k] = v
key = os.environ.get("ALPACA_API_KEY")
secret = os.environ.get("ALPACA_API_SECRET")
hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
r = requests.get("https://paper-api.alpaca.markets/v2/positions", headers=hdr, timeout=10)
r.raise_for_status()
positions = r.json()
if not positions:
    print("No open positions.")
    sys.exit(0)
print("\nCurrent open positions:\n")
for p in positions:
    qty = float(p["qty"])
    side = "SHORT" if qty < 0 else "LONG"
    pnl_pct = float(p["unrealized_plpc"]) * 100
    print(f"  {p['symbol']:22} {side:5}  qty={abs(qty):.0f}  entry=${float(p['avg_entry_price']):.2f}  now=${float(p['current_price']):.2f}  pnl={pnl_pct:+.2f}%")
print()
