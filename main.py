#!/usr/bin/env python3
"""
Quantitative Stocks — Unified CLI
====================================
Single entry point for all modules: signals, train, predict, backtest, trade.

Usage:
    python main.py signals  --provider yahoo --ml
    python main.py train    --symbol SPY --epochs 50
    python main.py predict  --symbol SPY
    python main.py backtest --symbol SPY --start 2024-01-01
    python main.py trade    --interval 5 --confidence 0.2

Environment variables:
    FRED_API_KEY      – FRED API key for VIX data
    ALPACA_API_KEY    – Alpaca API key (for alpaca/hybrid provider and paper trading)
    ALPACA_API_SECRET – Alpaca API secret
"""

import os
import sys

# Ensure project root is on sys.path so `from src.xxx import ...` works
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("""
  Quantitative Stocks — Unified CLI
  ====================================

  Usage:
    python main.py <command> [options]

  Commands:
    signals   Run the ETF sentiment signal engine
    train     Train LSTM model for a symbol
    predict   Run ML prediction for a symbol
    backtest  Walk-forward backtest with ML predictions
    trade     Start Alpaca paper trading loop

  Examples:
    python main.py signals  --provider yahoo --ml
    python main.py train    --symbol SPY --epochs 50
    python main.py train    --symbol SPY --mode intraday --interval 5min
    python main.py predict  --symbol SPY
    python main.py backtest --symbol SPY --start 2024-01-01
    python main.py backtest --symbol SPY --start 2025-01-01 --mode intraday
    python main.py trade    --confidence 0.2 --trailing-stop 0.05
    python main.py trade    --mode intraday --interval 5min

  Run `python main.py <command> --help` for command-specific options.
""")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the command from argv so each module's argparse works normally
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "signals":
        from src.signals_engine import main as signals_main
        signals_main()

    elif command == "train":
        # Inject "train" subcommand back for ml_model's argparse
        sys.argv = [sys.argv[0], "train"] + sys.argv[1:]
        from src.ml_model import main as ml_main
        ml_main()

    elif command == "predict":
        # Inject "predict" subcommand back for ml_model's argparse
        sys.argv = [sys.argv[0], "predict"] + sys.argv[1:]
        from src.ml_model import main as ml_main
        ml_main()

    elif command == "backtest":
        from src.backtester import main as backtest_main
        backtest_main()

    elif command == "trade":
        from src.paper_trader import main as trade_main
        trade_main()

    else:
        print(f"\n  Unknown command: {command!r}")
        print("  Available commands: signals, train, predict, backtest, trade")
        print("  Run `python main.py --help` for usage.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
