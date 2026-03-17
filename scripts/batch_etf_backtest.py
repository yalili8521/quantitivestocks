#!/usr/bin/env python3
"""Batch train + OOS backtest for ETF universe.

Reads etf_universe.json (from screen-etf-universe), trains swing models for
each ETF, backtests on the OOS period, and writes promoted_symbols.json for
symbols that pass quality thresholds.

Pipeline:
  1. Load ETF symbols from etf_universe.json (or --symbols override)
  2. Train XGBoost swing model per symbol (--train-recent = 75/25 split)
  3. Backtest each on OOS window
  4. Rank by Sharpe, promote symbols meeting thresholds
  5. Write promoted_symbols.json + summary table

Usage (via main.py):
    python main.py batch-backtest
    python main.py batch-backtest --symbols SPY,QQQ,GLD --skip-train
    python main.py batch-backtest --min-sharpe 0.5 --min-trades 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from utils import SWING_MODEL_DIR, BACKTEST_DIR

log = logging.getLogger("batch_etf")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Promotion thresholds
DEFAULT_MIN_SHARPE = 0.0
DEFAULT_MIN_TRADES = 3
DEFAULT_MIN_WIN_RATE = 0.40
DEFAULT_MAX_DRAWDOWN = -50.0  # max drawdown % (negative)


def train_single(symbol: str, save_dir: str) -> bool:
    """Train swing model for a single ETF. Returns True on success."""
    from signals_engine import build_adapter
    from swing_model import train_swing_model

    try:
        adapter = build_adapter("yahoo")
        fred_key = os.environ.get("FRED_API_KEY")

        model = train_swing_model(
            symbol=symbol,
            adapter=adapter,
            fred_key=fred_key,
            save_dir=save_dir,
            train_recent=True,  # 75/25 walk-forward split
        )
        return model is not None
    except Exception as e:
        log.error("%s: training failed -- %s", symbol, e)
        traceback.print_exc()
        return False


def backtest_single(symbol: str, start: str, end: str | None,
                    model_dir: str) -> dict | None:
    """Run swing backtest for a single symbol. Returns summary dict or None."""
    from backtester import Backtester
    from signals_engine import build_adapter

    try:
        adapter = build_adapter("yahoo")
        fred_key = os.environ.get("FRED_API_KEY")

        bt = Backtester(
            symbol=symbol,
            adapter=adapter,
            fred_key=fred_key,
            initial_capital=100_000,
            model_type="swing",
            model_dir=model_dir,
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
            "annualized_return_pct": round(float(result.annualized_return_pct), 2),
            "sharpe_ratio": round(float(result.sharpe_ratio), 3),
            "max_drawdown_pct": round(float(result.max_drawdown_pct), 2),
            "total_trades": int(result.total_trades),
            "win_rate": round(float(result.win_rate), 3),
            "profit_factor": (
                round(float(result.profit_factor), 3)
                if result.profit_factor != float("inf") else None
            ),
            "avg_trade_duration_days": round(float(result.avg_trade_duration_days), 1),
        }

        # Save individual summary + equity curve
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        summary_path = os.path.join(BACKTEST_DIR, f"backtest_{symbol}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(BACKTEST_DIR, f"backtest_{symbol}.csv")
        result.equity_curve.to_csv(csv_path, index=False)

        return summary

    except Exception as e:
        log.error("%s: backtest failed -- %s", symbol, e)
        traceback.print_exc()
        return None


def save_promoted_symbols(
    summaries: list[dict],
    save_dir: str,
    min_sharpe: float,
    min_trades: int,
    min_win_rate: float,
    max_drawdown: float,
) -> list[str]:
    """Filter summaries by thresholds and write promoted_symbols.json."""
    promoted = []
    rejected = []

    for s in summaries:
        reasons = []
        if s["sharpe_ratio"] < min_sharpe:
            reasons.append(f"Sharpe {s['sharpe_ratio']:.3f} < {min_sharpe}")
        if s["total_trades"] < min_trades:
            reasons.append(f"trades {s['total_trades']} < {min_trades}")
        if s["win_rate"] < min_win_rate:
            reasons.append(f"WR {s['win_rate']:.0%} < {min_win_rate:.0%}")
        if s["max_drawdown_pct"] < max_drawdown:
            reasons.append(f"DD {s['max_drawdown_pct']:.1f}% < {max_drawdown:.1f}%")

        if not reasons:
            promoted.append(s)
        else:
            rejected.append((s["symbol"], "; ".join(reasons)))

    # Sort promoted by Sharpe descending
    promoted.sort(key=lambda x: -x["sharpe_ratio"])
    promoted_symbols = [s["symbol"] for s in promoted]

    # Write promoted_symbols.json
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "promoted_symbols.json")
    payload = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "min_sharpe": min_sharpe,
            "min_trades": min_trades,
            "min_win_rate": min_win_rate,
            "max_drawdown": max_drawdown,
        },
        "count": len(promoted_symbols),
        "symbols": promoted_symbols,
        "details": promoted,
        "rejected": [{"symbol": sym, "reason": reason} for sym, reason in rejected],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log.info("Promoted %d / %d symbols -> %s", len(promoted_symbols),
             len(summaries), out_path)

    return promoted_symbols


def main():
    parser = argparse.ArgumentParser(
        description="Batch train + OOS backtest for ETF universe"
    )
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: load from etf_universe.json)")
    parser.add_argument("--start", default="2022-01-01",
                        help="OOS backtest start date (default: 2022-01-01)")
    parser.add_argument("--end", default=None,
                        help="OOS backtest end date (default: today)")
    parser.add_argument("--save-dir", default=SWING_MODEL_DIR,
                        help="Model save directory")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing models)")
    parser.add_argument("--min-sharpe", type=float, default=DEFAULT_MIN_SHARPE,
                        help=f"Min Sharpe for promotion (default: {DEFAULT_MIN_SHARPE})")
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES,
                        help=f"Min trades for promotion (default: {DEFAULT_MIN_TRADES})")
    parser.add_argument("--min-win-rate", type=float, default=DEFAULT_MIN_WIN_RATE,
                        help=f"Min win rate for promotion (default: {DEFAULT_MIN_WIN_RATE})")
    parser.add_argument("--max-drawdown", type=float, default=DEFAULT_MAX_DRAWDOWN,
                        help=f"Max drawdown %% for promotion (default: {DEFAULT_MAX_DRAWDOWN})")
    args = parser.parse_args()

    # Load symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        from etf_screener import load_etf_universe
        symbols = load_etf_universe(args.save_dir)
        if not symbols:
            log.error("No ETF universe found. Run: python main.py screen-etf-universe")
            sys.exit(1)

    log.info("Batch ETF pipeline: %d symbols", len(symbols))

    # Phase 1: Train
    if not args.skip_train:
        log.info("=== PHASE 1: Training %d swing models ===", len(symbols))
        train_results = {}
        for i, sym in enumerate(symbols, 1):
            log.info("[%d/%d] Training %s...", i, len(symbols), sym)
            ok = train_single(sym, args.save_dir)
            train_results[sym] = ok
            if not ok:
                log.warning("%s: TRAINING FAILED -- will skip backtest", sym)

        trained = [s for s, ok in train_results.items() if ok]
        failed = [s for s, ok in train_results.items() if not ok]
        log.info("Training complete: %d succeeded, %d failed", len(trained), len(failed))
        if failed:
            log.warning("Failed: %s", ", ".join(failed))
        # Only backtest successfully trained symbols
        symbols = trained
    else:
        # Filter to symbols with existing models
        existing = []
        for sym in symbols:
            model_path = os.path.join(args.save_dir, f"{sym}_xgb_swing.joblib")
            if os.path.exists(model_path):
                existing.append(sym)
            else:
                log.warning("%s: no model found at %s, skipping", sym, model_path)
        symbols = existing
        log.info("Found %d existing models (--skip-train)", len(symbols))

    if not symbols:
        log.error("No symbols to backtest!")
        sys.exit(1)

    # Phase 2: Backtest
    log.info("=== PHASE 2: Backtesting %d symbols (%s -> %s) ===",
             len(symbols), args.start, args.end or "today")

    summaries = []
    for i, sym in enumerate(symbols, 1):
        log.info("[%d/%d] Backtesting %s...", i, len(symbols), sym)
        summary = backtest_single(sym, args.start, args.end, args.save_dir)
        if summary:
            summaries.append(summary)
            log.info("  %s: Sharpe=%.3f, Return=%.1f%%, Trades=%d, WR=%.0f%%",
                     sym, summary["sharpe_ratio"], summary["total_return_pct"],
                     summary["total_trades"], summary["win_rate"] * 100)
        else:
            log.warning("  %s: BACKTEST FAILED", sym)

    if not summaries:
        log.error("All backtests failed!")
        sys.exit(1)

    # Phase 3: Promote
    log.info("=== PHASE 3: Promoting symbols ===")
    promoted = save_promoted_symbols(
        summaries, args.save_dir,
        min_sharpe=args.min_sharpe,
        min_trades=args.min_trades,
        min_win_rate=args.min_win_rate,
        max_drawdown=args.max_drawdown,
    )

    # Print summary table
    sorted_summaries = sorted(summaries, key=lambda x: -x["sharpe_ratio"])

    print(f"\n{'=' * 100}")
    print("  ETF BATCH BACKTEST RESULTS (sorted by Sharpe)")
    print(f"{'=' * 100}")
    print(f"  {'Symbol':<10} {'Sharpe':>8} {'Return':>10} {'AnnRet':>10} {'Trades':>8} "
          f"{'WinRate':>8} {'MaxDD':>8} {'PF':>8} {'Status'}")
    print(f"  {'-' * 92}")

    for s in sorted_summaries:
        status = "PROMOTED" if s["symbol"] in promoted else "rejected"
        pf_str = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "inf"
        print(f"  {s['symbol']:<10} {s['sharpe_ratio']:>+8.3f} {s['total_return_pct']:>+9.1f}% "
              f"{s['annualized_return_pct']:>+9.1f}% {s['total_trades']:>8} "
              f"{s['win_rate']:>7.0%} {s['max_drawdown_pct']:>7.1f}% {pf_str:>8}  {status}")

    print(f"  {'-' * 92}")
    print(f"  Total: {len(sorted_summaries)} symbols | "
          f"Promoted: {len(promoted)} | Rejected: {len(sorted_summaries) - len(promoted)}")
    print(f"{'=' * 100}")

    if promoted:
        print(f"\n  Promoted symbols: {', '.join(promoted)}")
        print(f"  Saved to: {os.path.join(args.save_dir, 'promoted_symbols.json')}\n")


if __name__ == "__main__":
    main()
