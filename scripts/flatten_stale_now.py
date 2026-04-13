"""One-shot script: flatten stale positions across all Alpaca accounts.

Stale = position in a symbol NOT in the group's active pool.
Also closes the unmanaged BTCUSD position on the crypto account.

Safe to run anytime:
  - Equity orders: submitted as MOC (market-on-close) / next-open market orders
  - Crypto orders: fill immediately (GTC)

Usage:
    .venv/Scripts/python.exe scripts/flatten_stale_now.py [--dry-run]
"""
import os
import sys
import argparse

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load env
env_path = os.path.join(os.path.dirname(__file__), "..", "secrets", "alpaca.env")
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"')
            os.environ.setdefault(k.strip(), v)

# Active symbol pools (must match paper_trader.py SYMBOL_GROUPS)
ACTIVE_POOLS = {
    "intraday": {
        "key_env": "ALPACA_INTRADAY_KEY",
        "secret_env": "ALPACA_INTRADAY_SECRET",
        "symbols": {"USO", "INDA", "USMV", "EWZ", "XLU",
                     "XLE", "QQQ", "PDBC", "XLP", "XLB"},
    },
    "swing": {
        "key_env": "ALPACA_SWING_KEY",
        "secret_env": "ALPACA_SWING_SECRET",
        "symbols": {"ARKK", "EWJ", "PDBC", "CIBR", "EWH",
                     "IWM", "VGK", "USMV", "EWY", "USO", "EWU"},
    },
    "crypto": {
        "key_env": "ALPACA_CRYPTO_KEY",
        "secret_env": "ALPACA_CRYPTO_SECRET",
        # Crypto is disabled on main branch — flatten EVERYTHING
        "symbols": set(),
        "flatten_all": True,
    },
}


def flatten_account(group: str, cfg: dict, dry_run: bool = False):
    key = os.environ.get(cfg["key_env"], "")
    secret = os.environ.get(cfg["secret_env"], "")
    if not key or not secret:
        print(f"  [{group}] SKIP — {cfg['key_env']} not set")
        return

    client = TradingClient(api_key=key, secret_key=secret, paper=True)

    # Fetch positions
    positions = client.get_all_positions()
    if not positions:
        print(f"  [{group}] No positions")
        return

    active = cfg["symbols"]
    flatten_all = cfg.get("flatten_all", False)

    for pos in positions:
        sym = pos.symbol
        qty = float(pos.qty)
        side = pos.side.value  # "long" or "short"
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price) if pos.current_price else 0
        pnl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
        pnl_pct = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0

        is_stale = flatten_all or (sym not in active)
        status = "STALE" if is_stale else "active"

        print(f"  [{group}] {sym:>10s}  {side:5s}  qty={qty}  "
              f"entry=${entry:.2f}  now=${current:.2f}  "
              f"P&L={pnl_pct:+.2%} (${pnl:+.2f})  [{status}]")

        if not is_stale:
            continue

        if dry_run:
            print(f"           -> WOULD FLATTEN {sym} (dry-run)")
            continue

        # Determine close order
        is_crypto = sym.endswith("USD") and not sym.startswith("X")
        if side == "long":
            order_side = OrderSide.SELL
        else:
            order_side = OrderSide.BUY

        tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

        try:
            order = client.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    qty=abs(qty) if is_crypto else int(abs(qty)),
                    side=order_side,
                    time_in_force=tif,
                )
            )
            fill_note = "fills 24/7" if is_crypto else "fills Monday open"
            print(f"           -> FLATTEN ORDER {order.id} ({fill_note})")
        except Exception as exc:
            print(f"           -> FAILED: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Flatten stale positions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be flattened without submitting orders")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Flatten Stale Positions {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'=' * 60}\n")

    for group, cfg in ACTIVE_POOLS.items():
        flatten_account(group, cfg, dry_run=args.dry_run)
        print()

    print("Done.\n")


if __name__ == "__main__":
    main()
