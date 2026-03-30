#!/usr/bin/env python3
"""
Unified Performance Tracker
============================
Aggregates equity, PnL, Sharpe, and drawdown across all trading accounts
(intraday Alpaca, swing Alpaca, crypto Kraken, gold scalper).

Usage:
    python src/equity_tracker.py              # print summary
    python src/equity_tracker.py --save       # save daily snapshot to CSV
    python src/equity_tracker.py --history    # print equity history
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("equity_tracker")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_CSV = os.path.join(PROJECT_ROOT, "outputs", "equity_history.csv")
PAPER_STATE_DIR = os.path.join(PROJECT_ROOT, "outputs", "paper_state")


def _get_alpaca_equity(key_env: str, secret_env: str) -> Optional[float]:
    """Get equity from an Alpaca paper account."""
    api_key = os.environ.get(key_env)
    api_secret = os.environ.get(secret_env)
    if not api_key or not api_secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, api_secret, paper=True)
        account = client.get_account()
        return float(account.equity)
    except Exception as exc:
        log.warning("Alpaca equity fetch failed (%s): %s", key_env, exc)
        return None


def _get_kraken_equity(state_file: str) -> Optional[float]:
    """Get equity from a Kraken paper state file.

    Equity = cash + sum of position notional values.
    The state file stores 'cash' and 'positions' (not 'equity').
    """
    path = os.path.join(PAPER_STATE_DIR, state_file)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        cash = float(state.get("cash", state.get("balance", state.get("equity", 0))))
        # Add position notional values (entry_price × qty for longs)
        positions = state.get("positions", {})
        position_value = 0.0
        for sym, pos in positions.items():
            qty = float(pos.get("qty", 0))
            entry_price = float(pos.get("entry_price", 0))
            side = pos.get("side", "LONG")
            # For equity tracking, use entry_price as estimate
            # (real-time pricing would require API calls)
            if side == "LONG":
                position_value += entry_price * qty
        return cash + position_value
    except Exception as exc:
        log.warning("Kraken state read failed (%s): %s", state_file, exc)
        return None


def get_all_equities() -> Dict[str, Optional[float]]:
    """Fetch equity from all accounts."""
    return {
        "intraday": _get_alpaca_equity("ALPACA_INTRADAY_KEY", "ALPACA_INTRADAY_SECRET"),
        "swing": _get_alpaca_equity("ALPACA_SWING_KEY", "ALPACA_SWING_SECRET"),
        "crypto": _get_kraken_equity("kraken_paper_state.json"),
        "gold_signal": _get_kraken_equity(os.path.join("signal", "gold_scalper_state.json")),
    }


def save_snapshot(equities: Dict[str, Optional[float]]) -> str:
    """Append today's equity snapshot to CSV."""
    os.makedirs(os.path.dirname(EQUITY_CSV), exist_ok=True)
    write_header = not os.path.exists(EQUITY_CSV)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    total = sum(v for v in equities.values() if v is not None)

    with open(EQUITY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "intraday", "swing", "crypto",
                             "gold_scalper", "total"])
        writer.writerow([
            now,
            f"{equities.get('intraday', '')}" if equities.get('intraday') is not None else "",
            f"{equities.get('swing', '')}" if equities.get('swing') is not None else "",
            f"{equities.get('crypto', '')}" if equities.get('crypto') is not None else "",
            f"{equities.get('gold_scalper', '')}" if equities.get('gold_scalper') is not None else "",
            f"{total:.2f}",
        ])
    return EQUITY_CSV


def compute_stats(csv_path: str = EQUITY_CSV) -> Dict:
    """Compute Sharpe, max drawdown, and return from equity history."""
    if not os.path.exists(csv_path):
        return {"error": "No equity history found"}

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if len(df) < 2:
        return {"error": "Need at least 2 snapshots for stats"}

    df = df.sort_values("timestamp")
    total = df["total"].astype(float)

    daily_returns = total.pct_change().dropna()
    total_return = (total.iloc[-1] / total.iloc[0]) - 1

    # Annualized Sharpe (assuming daily snapshots)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = total.cummax()
    drawdown = (total - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "total_return": f"{total_return:+.2%}",
        "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{max_dd:.2%}",
        "snapshots": len(df),
        "first_date": str(df["timestamp"].iloc[0].date()),
        "last_date": str(df["timestamp"].iloc[-1].date()),
        "current_total": f"${total.iloc[-1]:,.2f}",
    }


def print_summary():
    """Print a unified summary of all accounts."""
    equities = get_all_equities()
    total = sum(v for v in equities.values() if v is not None)

    print("\n" + "=" * 60)
    print("  UNIFIED PORTFOLIO SUMMARY")
    print("=" * 60)

    for name, equity in equities.items():
        if equity is not None:
            pct = equity / total * 100 if total > 0 else 0
            print(f"  {name:>15}:  ${equity:>12,.2f}  ({pct:5.1f}%)")
        else:
            print(f"  {name:>15}:  {'unavailable':>12}")

    print("-" * 60)
    print(f"  {'TOTAL':>15}:  ${total:>12,.2f}")
    print("=" * 60)

    # Historical stats if available
    if os.path.exists(EQUITY_CSV):
        stats = compute_stats()
        if "error" not in stats:
            print(f"\n  Historical ({stats['first_date']} to {stats['last_date']}):")
            print(f"    Total Return: {stats['total_return']}")
            print(f"    Sharpe Ratio: {stats['sharpe']}")
            print(f"    Max Drawdown: {stats['max_drawdown']}")
            print(f"    Snapshots:    {stats['snapshots']}")
    print()


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Unified portfolio performance tracker")
    parser.add_argument("--save", action="store_true", help="Save daily equity snapshot")
    parser.add_argument("--history", action="store_true", help="Print equity history")
    args = parser.parse_args()

    # Load env
    env_path = os.path.join(PROJECT_ROOT, "secrets", "alpaca.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()

    if args.save:
        equities = get_all_equities()
        path = save_snapshot(equities)
        print(f"Saved snapshot to {path}")
        print_summary()
    elif args.history:
        if os.path.exists(EQUITY_CSV):
            df = pd.read_csv(EQUITY_CSV)
            print(df.to_string(index=False))
        else:
            print("No equity history found. Run with --save first.")
    else:
        print_summary()


if __name__ == "__main__":
    main()
