#!/usr/bin/env python3
"""Batch OOS backtest for all crypto coins with trained models.

Runs backtester on each coin's OOS period (first half of data, since models
are trained on second half via --train-recent), collects Sharpe ratios,
and updates oos_sharpe_registry in config/trading.json.

Usage:
    python scripts/batch_oos_backtest.py
    python scripts/batch_oos_backtest.py --start 2022-01-01 --end 2024-11-01
"""

import argparse
import glob
import json
import logging
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from utils import CRYPTO_MODEL_DIR

log = logging.getLogger("batch_oos")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

BACKTEST_DIR = os.path.join(PROJECT_ROOT, "outputs", "backtests")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "trading.json")


def get_model_universe() -> list[str]:
    """Return list of symbols with trained XGBoost swing models."""
    pattern = os.path.join(CRYPTO_MODEL_DIR, "*_xgb_swing_config.json")
    syms = []
    for p in glob.glob(pattern):
        sym = os.path.basename(p).replace("_xgb_swing_config.json", "")
        syms.append(sym)
    return sorted(syms)


def run_single_backtest(symbol: str, start: str, end: str) -> dict | None:
    """Run backtest for a single symbol, return summary dict or None on failure."""
    from backtester import Backtester, build_adapter, print_report

    try:
        adapter = build_adapter("yahoo")
        fred_key = os.environ.get("FRED_API_KEY")

        bt = Backtester(
            symbol=symbol,
            adapter=adapter,
            fred_key=fred_key,
            initial_capital=100_000,
            model_type="swing",
            model_dir=CRYPTO_MODEL_DIR,
            mode="daily",
        )
        result = bt.run(start_date=start, end_date=end)

        if result is None:
            log.warning("%s: backtest returned None", symbol)
            return None

        summary = {
            "symbol": symbol,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return_pct": round(float(result.total_return_pct), 2),
            "sharpe_ratio": round(float(result.sharpe_ratio), 3),
            "max_drawdown_pct": round(float(result.max_drawdown_pct), 2),
            "total_trades": int(result.total_trades),
            "win_rate": round(float(result.win_rate), 3),
            "profit_factor": round(float(result.profit_factor), 3) if result.profit_factor != float("inf") else None,
        }

        # Save individual summary
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        summary_path = os.path.join(BACKTEST_DIR, f"backtest_{symbol}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save equity curve
        csv_path = os.path.join(BACKTEST_DIR, f"backtest_{symbol}.csv")
        result.equity_curve.to_csv(csv_path, index=False)

        return summary

    except Exception as e:
        log.error("%s: backtest failed — %s", symbol, e)
        traceback.print_exc()
        return None


def update_oos_registry(results: dict[str, float]) -> None:
    """Update oos_sharpe_registry in config/trading.json."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    registry = config.get("oos_sharpe_registry", {})
    registry.update(results)
    # Sort by Sharpe descending
    config["oos_sharpe_registry"] = dict(
        sorted(registry.items(), key=lambda x: -x[1])
    )

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    log.info("Updated oos_sharpe_registry with %d entries (total %d)",
             len(results), len(config["oos_sharpe_registry"]))


def main():
    parser = argparse.ArgumentParser(description="Batch OOS backtest for crypto")
    parser.add_argument("--start", default="2022-01-01",
                        help="OOS start date (default: 2022-01-01)")
    parser.add_argument("--end", default="2024-11-01",
                        help="OOS end date (default: 2024-11-01, before training window)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols to backtest (default: all model_universe)")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_model_universe()

    log.info("Running OOS backtest for %d coins: %s -> %s", len(symbols), args.start, args.end)

    results = {}
    summaries = []

    for i, sym in enumerate(symbols, 1):
        log.info("=== [%d/%d] %s ===", i, len(symbols), sym)
        summary = run_single_backtest(sym, args.start, args.end)
        if summary:
            sharpe = summary["sharpe_ratio"]
            results[sym] = sharpe
            summaries.append(summary)
            log.info("%s: Sharpe=%.3f, Return=%.1f%%, Trades=%d, WR=%.0f%%",
                     sym, sharpe, summary["total_return_pct"],
                     summary["total_trades"], summary["win_rate"] * 100)
        else:
            results[sym] = -999.0  # mark as failed
            log.warning("%s: FAILED — marking as untradeable", sym)

    # Update registry
    # Only include coins that actually completed
    valid_results = {k: v for k, v in results.items() if v > -999}
    update_oos_registry(valid_results)

    # Print summary table
    print("\n" + "=" * 80)
    print("  OOS BACKTEST RESULTS (sorted by Sharpe)")
    print("=" * 80)
    print(f"  {'Symbol':<16} {'Sharpe':>8} {'Return':>10} {'Trades':>8} {'WinRate':>8} {'MaxDD':>8}")
    print("-" * 80)

    sorted_summaries = sorted(summaries, key=lambda x: -x["sharpe_ratio"])
    tradeable = 0
    for s in sorted_summaries:
        flag = " *" if s["sharpe_ratio"] > 0 else ""
        if s["sharpe_ratio"] > 0:
            tradeable += 1
        print(f"  {s['symbol']:<16} {s['sharpe_ratio']:>+8.3f} {s['total_return_pct']:>+9.1f}% "
              f"{s['total_trades']:>8} {s['win_rate']:>7.0%} {s['max_drawdown_pct']:>7.1f}%{flag}")

    print("-" * 80)
    print(f"  Total: {len(sorted_summaries)} coins | Tradeable (Sharpe > 0): {tradeable} | "
          f"Blocked: {len(sorted_summaries) - tradeable}")
    print("=" * 80)


if __name__ == "__main__":
    main()
