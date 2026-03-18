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
    trade            Start Alpaca paper trading loop (stocks)
    report           Generate HTML dashboard (outputs/report.html)
    train-intraday   Train LightGBM intraday momentum model (replaces LSTM for intraday group)
    train-swing      Train TFT+XGBoost swing model
    train-crypto     Train swing models for crypto (BTC, ETH, SOL) → models/crypto/
    train-crypto-intraday  Train LightGBM intraday model for crypto → models/crypto_intraday/
    training-tables Generate CSV/HTML tables of training & backtest metrics
    backtest-portfolio Portfolio-level multi-symbol backtest with shared capital
    divergence-report Compare backtest vs live paper-trading results; flag divergences
    check-positions Check paper account positions; recommend and run legacy handling for existing positions
    stop-paper-trader Stop the running paper trader for a group (e.g. intraday) so you can restart with different flags
    select-symbols   Screen candidate symbols: train, backtest, classify (core/secondary/disabled), update config
    train-selector   Train cross-sectional coin selector (Layer 1 of crypto pipeline)
    rank-coins       Rank coins using trained selector and show today's top-K
    screen-universe  Layer 0: discover tradeable coins from full market (CoinGecko × Kraken)
    screen-etf-universe  Layer 0: discover tradeable ETFs from broad ETF market (yfinance)
    select-features  IC-based feature selection pipeline (--asset-class equity_swing|crypto_swing)
    batch-backtest   Batch train + OOS backtest ETF universe → promoted_symbols.json
    lock-status      Show whether intraday (or other group) paper trader lock is present; tells you if it's safe to start
    weekly-pipeline  Run weekly crypto maintenance: screen → train selector → train models → select symbols → health check

  Examples:
    python main.py signals              --provider yahoo --ml
    python main.py train                --symbol SPY --epochs 50
    python main.py train                --symbol SPY --mode intraday --interval 5min
    python main.py train-meta           --symbol SPY
    python main.py train-meta           --symbol ALL
    python main.py predict              --symbol SPY
    python main.py backtest             --symbol SPY --start 2024-01-01
    python main.py backtest             --symbol SPY --start 2025-01-01 --mode intraday
    python main.py trade                --confidence 0.01 --trailing-stop 0.05
    python main.py trade                --group swing
    python main.py trade                --group intraday
    python main.py trade                --mode intraday --interval 5min
    python main.py report
    python main.py report               --open
    python main.py training-tables      # outputs/training_results_*.csv and .html
    python main.py train-intraday       --symbols SPY,QQQ,IWM,SOXX --provider alpaca
    python main.py train-swing          --symbols EWT,GLD,EEM,SLV --provider yahoo
    python main.py train-crypto                                    # BTC, ETH, SOL → models/crypto/
    python main.py backtest-portfolio --group swing --start 2024-01-01
    python main.py backtest-portfolio --symbols SPY,QQQ,IWM --start 2024-01-01 --stress-cost-mult 2.0

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

    elif command == "train-crypto":
        # Trains swing models for all coins in screen_universe (universe.json)
        # Falls back to hardcoded list if universe.json doesn't exist
        from utils import CRYPTO_MODEL_DIR
        from universe_screener import load_universe
        _universe = load_universe(CRYPTO_MODEL_DIR)
        if _universe:
            _CRYPTO_TRAIN_SYMBOLS = ",".join(_universe)
            print(f"  Training {len(_universe)} coins from universe.json")
        else:
            _CRYPTO_TRAIN_SYMBOLS = (
                "BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD,DOT-USD,"
                "SUSHI-USD,ADA-USD,CRV-USD,AAVE-USD,RENDER-USD,"
                "NEAR-USD,LTC-USD,ARB-USD,OP-USD,FIL-USD,APT-USD,INJ-USD,"
                "WIF-USD,AR-USD,ENA-USD,LDO-USD"
            )
            print(f"  No universe.json — using hardcoded {len(_CRYPTO_TRAIN_SYMBOLS.split(','))} coins")
        sys.argv = [sys.argv[0],
                     "--symbols", _CRYPTO_TRAIN_SYMBOLS,
                     "--provider", "yahoo",
                     "--save-dir", CRYPTO_MODEL_DIR,
                     "--train-recent"] + sys.argv[1:]
        from swing_model import main as swing_main
        swing_main()

    elif command == "train-crypto-intraday":
        # Trains intraday LGB+GRU models for crypto coins
        # Auto-loads symbols from universe.json (same as train-crypto)
        from utils import CRYPTO_MODEL_DIR, CRYPTO_INTRADAY_MODEL_DIR
        from universe_screener import load_universe
        _universe = load_universe(CRYPTO_MODEL_DIR)
        if _universe:
            _CRYPTO_INTRADAY_SYMBOLS = ",".join(_universe)
            print(f"  Training intraday for {len(_universe)} coins from universe.json")
        else:
            _CRYPTO_INTRADAY_SYMBOLS = (
                "BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD,"
                "ADA-USD,CRV-USD,AAVE-USD,RENDER-USD"
            )
            print(f"  No universe.json — using hardcoded {len(_CRYPTO_INTRADAY_SYMBOLS.split(','))} coins")
        sys.argv = [sys.argv[0],
                     "--symbols", _CRYPTO_INTRADAY_SYMBOLS,
                     "--save-dir", CRYPTO_INTRADAY_MODEL_DIR,
                     "--walk-forward"] + sys.argv[1:]
        from crypto_intraday_model import main as crypto_intraday_main
        crypto_intraday_main()

    elif command == "backtest-portfolio":
        from portfolio_backtester import main as portfolio_bt_main
        portfolio_bt_main()

    elif command == "divergence-report":
        from divergence_report import main as divergence_main
        divergence_main()

    elif command == "select-symbols":
        import scripts.select_symbols as sel_sym
        sel_sym.main()

    elif command == "train-selector":
        # Train cross-sectional coin selector (Layer 1 of crypto pipeline)
        sys.argv = [sys.argv[0], "train"] + sys.argv[1:]
        from coin_selector import main as selector_main
        selector_main()

    elif command == "rank-coins":
        # Rank coins using trained selector (Layer 1 inference)
        sys.argv = [sys.argv[0], "rank"] + sys.argv[1:]
        from coin_selector import main as selector_main
        selector_main()

    elif command == "screen-universe":
        # Layer 0: discover tradeable coins from full market
        from universe_screener import main as screener_main
        screener_main()

    elif command == "model-health":
        from model_monitor import ModelMonitor
        monitor = ModelMonitor()
        print(monitor.generate_report())

    elif command == "screen-etf-universe":
        # Layer 0: discover tradeable ETFs from the broad market
        from etf_screener import main as etf_screener_main
        etf_screener_main()

    elif command == "select-features":
        # IC-based feature selection pipeline
        import scripts.select_features as sel_feat
        sel_feat.main()

    elif command == "batch-backtest":
        # Batch train + OOS backtest for ETF universe → promoted_symbols.json
        import scripts.batch_etf_backtest as batch_bt
        batch_bt.main()

    elif command == "weekly-pipeline":
        import scripts.weekly_pipeline as weekly
        weekly.main()

    elif command == "validate-risk":
        from risk_config import get_risk_config
        for group in ("intraday", "swing", "crypto"):
            risk = get_risk_config(group)
            print(f"\n  {group.upper()} risk config:")
            print(f"    position_pct:       {risk.position_pct:.0%}")
            print(f"    max_position_pct:   {risk.max_position_pct:.0%}")
            print(f"    max_sector_pct:     {risk.max_sector_pct:.0%}")
            print(f"    max_total_exposure: {risk.max_total_exposure:.0%}")
            print(f"    max_positions:      {risk.max_positions}")
            print(f"    kelly_cap:          {risk.kelly_cap:.0%}")

    else:
        print(f"\n  Unknown command: {command!r}")
        print("  Available commands: signals, train, train-meta, train-intraday, train-swing, train-crypto,")
        print("                      predict, backtest, backtest-portfolio, trade, report, training-tables,")
        print("                      divergence-report, select-symbols, check-positions, stop-paper-trader,")
        print("                      lock-status, model-health, validate-risk, train-selector, rank-coins,")
        print("                      screen-universe, screen-etf-universe, select-features, batch-backtest, weekly-pipeline")
        print("  Run `python main.py --help` for usage.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
