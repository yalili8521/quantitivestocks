#!/usr/bin/env python3
import os, sys, requests

# Load all API keys
env_path = "secrets/alpaca.env"
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            idx = line.find("=")
            if idx > 0:
                k, v = line[:idx].strip(), line[idx+1:].strip().strip('"')
                os.environ[k] = v

ACCOUNTS = [
    {"name": "Intraday", "key": "ALPACA_INTRADAY_KEY", "secret": "ALPACA_INTRADAY_SECRET"},
    {"name": "Swing", "key": "ALPACA_SWING_KEY", "secret": "ALPACA_SWING_SECRET"},
    {"name": "Crypto", "key": "ALPACA_CRYPTO_KEY", "secret": "ALPACA_CRYPTO_SECRET"},
]

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

all_positions = []

for acct in ACCOUNTS:
    key = os.environ.get(acct["key"])
    secret = os.environ.get(acct["secret"])
    if not key or not secret:
        print(f"{acct['name']}: No API credentials")
        continue
    
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    try:
        r = requests.get(f"{ALPACA_BASE}/positions", headers=hdr, timeout=10)
        r.raise_for_status()
        positions = r.json()
    except Exception as e:
        print(f"{acct['name']}: Error - {e}")
        continue
    
    if not positions:
        print(f"{acct['name']}: No open positions")
        continue
    
    print(f"\n{acct['name']} open positions:")
    for p in positions:
        qty = float(p["qty"])
        side = "SHORT" if qty < 0 else "LONG"
        pnl_pct = float(p["unrealized_plpc"]) * 100
        entry = float(p["avg_entry_price"])
        current = float(p["current_price"])
        symbol = p["symbol"]
        
        print(f"  {symbol:22} {side:5}  qty={abs(qty):.0f}  entry=${entry:.2f}  now=${current:.2f}  pnl={pnl_pct:+.2f}%")
        
        all_positions.append({
            "account": acct["name"],
            "symbol": symbol,
            "side": side,
            "qty": abs(qty),
            "entry": entry,
            "current": current,
            "pnl_pct": pnl_pct
        })

if not all_positions:
    print("\nNo open positions in any account.")
    sys.exit(0)

print("\n" + "="*80)
print("EXIT PRICE CALCULATIONS")
print("="*80)

# Calculate exit prices based on different strategies
for p in all_positions:
    symbol = p["symbol"]
    entry = p["entry"]
    current = p["current"]
    side = p["side"]
    qty = p["qty"]
    
    # Calculate various exit levels
    profit_2pct = entry * 1.02 if side == "LONG" else entry * 0.98
    profit_5pct = entry * 1.05 if side == "LONG" else entry * 0.95
    break_even = entry * 1.001 if side == "LONG" else entry * 0.999  # 0.1% costs
    stop_loss = entry * 0.97 if side == "LONG" else entry * 1.03  # 3% stop
    
    print(f"\n{p['account']} - {symbol} ({side}):")
    print(f"  Entry: ${entry:.4f}, Current: ${current:.4f}")
    print(f"  +2% target:   ${profit_2pct:.4f}")
    print(f"  +5% target:   ${profit_5pct:.4f}")
    print(f"  Break-even:   ${break_even:.4f}")
    print(f"  Stop-loss:    ${stop_loss:.4f}")
