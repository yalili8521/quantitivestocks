#!/usr/bin/env python3
"""
Quantitative Stocks — Unified CLI
====================================
Single entry point for all modules: signals, train, predict, backtest, trade.

Usage:
    python main.py signals          --provider yahoo --ml
    python main.py train            --symbol SPY --epochs 50
    python main.py predict          --symbol SPY
    python main.py backtest         --symbol SPY --start 2024-01-01
    python main.py trade            --interval 5 --confidence 0.2
    python main.py range-backtest   --symbols SPY,QQQ --start 2024-01-01
    python main.py range-trade      --symbols SPY,QQQ --provider alpaca --group range
    python main.py report                             # generate outputs/report.html
    python main.py report --open                      # generate + open in browser

Environment variables:
    FRED_API_KEY      – FRED API key for VIX data
    ALPACA_API_KEY    – Alpaca API key (for alpaca/hybrid provider and paper trading)
    ALPACA_API_SECRET – Alpaca API secret
"""

import os
import sys

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (SRC_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("""
  Quantitative Stocks — Unified CLI
  ====================================

  Usage:
    python main.py <command> [options]

  Commands:
    signals          Run the ETF sentiment signal engine
    train            Train LSTM model for a symbol
    train-meta       Train Random Forest meta-model (requires trained primary LSTM)
    predict          Run ML prediction for a symbol
    backtest         Walk-forward backtest with ML predictions
    range-backtest   Walk-forward backtest for intraday mean reversion range strategy
    trade            Start Alpaca paper trading loop (stocks)
    range-trade      Start intraday mean reversion range paper trader (dedicated account)
    report           Generate HTML dashboard (outputs/report.html)
    spread-trade     Start options credit spread trader (activates when VIX > 25)
    train-intraday   Train LightGBM intraday momentum model (replaces LSTM for intraday group)
    train-swing      Train TFT+XGBoost swing model
    training-tables Generate CSV/HTML tables of training & backtest metrics
    check-positions Check paper account positions; recommend and run legacy handling for existing positions
    stop-paper-trader Stop the running paper trader for a group (e.g. intraday) so you can restart with different flags
    lock-status      Show whether intraday (or other group) paper trader lock is present; tells you if it's safe to start

  Examples:
    python main.py signals              --provider yahoo --ml
    python main.py train                --symbol SPY --epochs 50
    python main.py train                --symbol SPY --mode intraday --interval 5min
    python main.py train-meta           --symbol SPY
    python main.py train-meta           --symbol ALL
    python main.py predict              --symbol SPY
    python main.py backtest             --symbol SPY --start 2024-01-01
    python main.py backtest             --symbol SPY --start 2025-01-01 --mode intraday
    python main.py range-backtest       --symbols SPY,QQQ,IWM,SOXX,EWT,GLD,EEM,SLV,EWJ,EWS,XLE,INDA --start 2024-01-01
    python main.py range-backtest       --symbols SPY --start 2025-01-01 --provider alpaca
    python main.py range-trade          --symbols SPY,QQQ,IWM,SOXX,EWT,GLD,EEM,SLV,EWJ,EWS,XLE,INDA --provider alpaca --group range
    python main.py trade                --confidence 0.2 --trailing-stop 0.05
    python main.py trade                --group equities
    python main.py trade                --group commodities
    python main.py trade                --mode intraday --interval 5min
    python main.py report
    python main.py report               --open
    python main.py training-tables      # outputs/training_results_*.csv and .html
    python main.py train-intraday       --symbols SPY,QQQ,IWM,SOXX --provider alpaca
    python main.py train-swing          --symbols EWT,GLD,EEM,SLV --provider yahoo

  Run `python main.py <command> --help` for command-specific options.
""")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the command from argv so each module's argparse works normally
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "signals":
        from signals_engine import main as signals_main
        signals_main()

    elif command == "train":
        sys.argv = [sys.argv[0], "train"] + sys.argv[1:]
        from ml_model import main as ml_main
        ml_main()

    elif command == "train-meta":
        sys.argv = [sys.argv[0], "train-meta"] + sys.argv[1:]
        from ml_model import main as ml_main
        ml_main()

    elif command == "predict":
        # Inject "predict" subcommand back for ml_model's argparse
        sys.argv = [sys.argv[0], "predict"] + sys.argv[1:]
        from ml_model import main as ml_main
        ml_main()

    elif command == "backtest":
        from backtester import main as backtest_main
        backtest_main()

    elif command == "trade":
        from paper_trader import main as trade_main
        trade_main()

    elif command == "check-positions":
        from paper_trader import check_positions_main
        check_positions_main()

    elif command == "stop-paper-trader":
        from paper_trader import stop_paper_trader_main
        stop_paper_trader_main()

    elif command == "lock-status":
        from paper_trader import lock_status_main
        lock_status_main()

    elif command == "report":
        from reports import main as report_main
        report_main()

    elif command == "training-tables":
        import scripts.generate_training_tables as gen_tables
        gen_tables.main()

    elif command == "train-intraday":
        from intraday_model import main as intraday_main
        intraday_main()

    elif command == "train-swing":
        from swing_model import main as swing_main
        swing_main()

    elif command == "spread-trade":
        from spread_trader import main as spread_main
        spread_main()

    else:
        print(f"\n  Unknown command: {command!r}")
        print("  Available commands: signals, train, train-meta, train-intraday, train-swing, predict, backtest, range-backtest, trade, range-trade, report, training-tables, check-positions, stop-paper-trader, lock-status")
        print("  Run `python main.py --help` for usage.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
