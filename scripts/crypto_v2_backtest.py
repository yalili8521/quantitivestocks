#!/usr/bin/env python3
"""
Crypto Strategy V2 Walk-Forward Backtest
=========================================
Backtests the v2 crypto strategy with all 7 improvements:
  1. Volatility-adjusted signal threshold  (k * ATR / price)
  2. Dynamic ensemble weighting            (rolling MAE inverse-error)
  3. Volatility-targeted position sizing   (target 15% ann vol)
  4. Correlation-aware portfolio allocation (inverse-vol weighting)
  5. Time stop                             (exit after FORWARD_DAYS)
  6. Trailing stop                         (2*ATR trigger, 1*ATR trail)
  7. Funding rate filter                   (blocked -- no historical data)

Runs a multi-asset portfolio simulation across BTC-USD, ETH-USD, SOL-USD
using trained models from models/crypto/.

Usage:
    python scripts/crypto_v2_backtest.py
    python scripts/crypto_v2_backtest.py --start 2024-06-01
    python scripts/crypto_v2_backtest.py --capital 100000 --v1  # v1 baseline comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import logging

# Suppress noisy FRED/cross-asset warnings that flood output during walk-forward
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
for _quiet in ["cross_asset_signals", "factor_signals", "market_signals",
               "swing_model", "signals_engine", "utils", "alpha_signals"]:
    logging.getLogger(_quiet).setLevel(logging.CRITICAL)

# Fetch VIX once with real FRED key before blanking it
os.environ.setdefault("FRED_API_KEY", "5e06c25a712146a59c69804dc0cdec4c")

from signals_engine import YahooFinanceAdapter, compute_atr
from swing_model import SwingPredictor, FORWARD_DAYS
from utils import CRYPTO_MODEL_DIR, _fetch_vix_for_training, COST_THRESHOLD, TARGET_RETURN

# Pre-fetch VIX data with real key
_PREFETCHED_VIX = _fetch_vix_for_training(
    os.environ.get("FRED_API_KEY"), lookback_days=800
)
print(f"  Pre-fetched {len(_PREFETCHED_VIX)} VIX rows.")

# Monkey-patch to prevent ANY FRED API calls during walk-forward:
# 1. _fetch_vix_for_training -> return cached data
import utils as _utils_module
_utils_module._fetch_vix_for_training = lambda *a, **kw: _PREFETCHED_VIX

# 2. FREDMacroFetcher.fetch_all -> return empty dict (no FRED calls)
import cross_asset_signals as _ca_module
_ca_module.FREDMacroFetcher.fetch_all = lambda self, *a, **kw: {}
_ca_module.FREDMacroFetcher.fetch_series = lambda self, *a, **kw: pd.DataFrame(columns=["date", "value"])

# 3. FREDVixFetcher.fetch -> return prefetched
from signals_engine import FREDVixFetcher
FREDVixFetcher.fetch = lambda self, *a, **kw: _PREFETCHED_VIX

# ---------------------------------------------------------------------------
# V2 strategy constants (mirror paper_trader.py)
# ---------------------------------------------------------------------------
CRYPTO_VOL_THRESHOLD_K = 0.5
CRYPTO_TARGET_VOL_ANN = 0.15
CRYPTO_MAX_HOLD_DAYS = FORWARD_DAYS   # 10
CRYPTO_TRAILING_TRIGGER_ATR = 2.0
CRYPTO_TRAILING_WIDTH_ATR = 1.0

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    entry_date: object
    entry_price: float
    direction: str         # "LONG" or "SHORT"
    size: float            # dollar amount allocated
    qty: float             # number of coins
    atr_at_entry: float
    peak_price: float = 0.0
    trailing_active: bool = False
    trailing_peak: float = 0.0
    hold_days: int = 0
    exit_date: object = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""
    er_at_entry: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class CryptoPortfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Trade] = {}   # symbol -> open trade
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[dict] = []

    def equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, trade in self.positions.items():
            p = prices.get(sym, trade.entry_price)
            if trade.direction == "LONG":
                eq += trade.qty * p
            else:
                eq += trade.qty * trade.entry_price + trade.qty * (trade.entry_price - p)
        return eq

    def record(self, date, prices: Dict[str, float]):
        self.equity_curve.append({
            "date": date,
            "equity": self.equity(prices),
        })


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    start_date: str,
    initial_capital: float = 100_000.0,
    use_v1: bool = False,
    version: str = "",
) -> dict:
    """Run walk-forward crypto backtest.

    Args:
        start_date: Backtest start (YYYY-MM-DD).
        initial_capital: Starting capital.
        use_v1: If True, run v1 baseline (legacy compat).
        version: "v1", "v1.5", or "v2". Overrides use_v1 if set.

    V1:   Fixed threshold + signal-proportional sizing + disaster stop + signal decay
    V1.5: V1 entries + time stop + trailing stop + inverse-vol allocation (hybrid)
    V2:   Vol-adjusted threshold + vol-targeted sizing + all stops + inverse-vol
    """
    if not version:
        version = "v1" if use_v1 else "v2"
    is_v1 = (version == "v1")
    has_v2_exits = (version in ("v1.5", "v2"))
    has_v2_entries = (version == "v2")
    adapter = YahooFinanceAdapter()
    vix_df = _PREFETCHED_VIX

    print(f"\n{'='*70}")
    print(f"  CRYPTO STRATEGY {version.upper()} BACKTEST")
    print(f"  Capital: ${initial_capital:,.0f} | Start: {start_date}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    if version == "v1.5":
        print(f"  Mode: V1 entries (fixed threshold) + V2 exits (time/trailing stop)")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------------------
    # 1. Fetch all daily data upfront
    # -----------------------------------------------------------------------
    all_bars: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        bars = adapter.fetch_daily(sym, 800)
        all_bars[sym] = bars
        print(f"  {sym}: {len(bars)} bars fetched")

    # BTC regime data
    btc_bars = all_bars["BTC-USD"]
    btc_close = btc_bars["close"].astype(float)
    btc_sma200 = btc_close.rolling(200).mean()
    btc_dates = pd.to_datetime(btc_bars["ts"]).dt.date

    # Build regime lookup: date -> bool
    regime_lookup: Dict = {}
    for d, c, s in zip(btc_dates, btc_close, btc_sma200):
        if not pd.isna(s):
            regime_lookup[d] = bool(c > s)

    # -----------------------------------------------------------------------
    # 2. Build per-symbol daily data aligned to common date index
    # -----------------------------------------------------------------------
    sym_data: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        bars = all_bars[sym]
        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        atr = compute_atr(high, low, close, period=14)
        rv30 = close.pct_change().rolling(30).std() * np.sqrt(365)
        sma50 = close.rolling(50).mean()
        dates = pd.to_datetime(bars["ts"]).dt.date

        df = pd.DataFrame({
            "date": dates.values,
            "close": close.values,
            "high": high.values,
            "low": low.values,
            "atr": atr.values,
            "rv30": rv30.values,
            "sma50": sma50.values,
        })
        df = df.dropna(subset=["atr", "rv30", "sma50"])
        sym_data[sym] = df.reset_index(drop=True)

    # Find common date range
    start_dt = pd.to_datetime(start_date).date()
    all_dates_sets = [set(d["date"].values) for d in sym_data.values()]
    common_dates = sorted(set.intersection(*all_dates_sets))
    common_dates = [d for d in common_dates if d >= start_dt]

    if not common_dates:
        print("  ERROR: No common dates found after start_date.")
        return {}

    print(f"\n  Backtest window: {common_dates[0]} -> {common_dates[-1]}")
    print(f"  Trading days: {len(common_dates)}")

    # -----------------------------------------------------------------------
    # 3. Load predictors
    # -----------------------------------------------------------------------
    predictors: Dict[str, SwingPredictor] = {}
    for sym in SYMBOLS:
        try:
            predictors[sym] = SwingPredictor(sym, model_dir=CRYPTO_MODEL_DIR)
            print(f"  Loaded predictor for {sym}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping {sym}")

    if not predictors:
        print("  ERROR: No predictors loaded.")
        return {}

    # -----------------------------------------------------------------------
    # 4. Walk-forward simulation
    # -----------------------------------------------------------------------
    portfolio = CryptoPortfolio(initial_capital)

    # Per-symbol date -> row lookup for fast access
    sym_date_lookup: Dict[str, Dict] = {}
    for sym in SYMBOLS:
        df = sym_data[sym]
        lookup = {}
        for _, row in df.iterrows():
            lookup[row["date"]] = row
        sym_date_lookup[sym] = lookup

    # Prediction cache: we need bars_df up to current date for each prediction
    # Pre-index bars by date for slicing
    sym_bars_date_idx: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        bars = all_bars[sym]
        bars_copy = bars.copy()
        bars_copy["_date"] = pd.to_datetime(bars_copy["ts"]).dt.date
        sym_bars_date_idx[sym] = bars_copy

    total_signals = 0
    regime_blocked = 0
    threshold_blocked = 0
    trend_blocked = 0

    for day_idx, current_date in enumerate(common_dates):
        if day_idx % 50 == 0:
            print(f"  ... day {day_idx}/{len(common_dates)} ({current_date})", flush=True)
        prices: Dict[str, float] = {}
        for sym in SYMBOLS:
            row = sym_date_lookup[sym].get(current_date)
            if row is not None:
                prices[sym] = float(row["close"])

        # --- Inverse-vol allocation (daily recalc) ---
        inv_vols = {}
        for sym in SYMBOLS:
            row = sym_date_lookup[sym].get(current_date)
            if row is not None and float(row["rv30"]) > 0:
                inv_vols[sym] = 1.0 / float(row["rv30"])
        total_inv = sum(inv_vols.values()) if inv_vols else 1.0
        weights = {s: v / total_inv for s, v in inv_vols.items()}

        # --- Check exits first ---
        closed_syms = []
        for sym, trade in list(portfolio.positions.items()):
            p = prices.get(sym, trade.entry_price)
            row = sym_date_lookup[sym].get(current_date)
            if row is None:
                continue
            atr = float(row["atr"])

            # Track peak for trailing stop
            if trade.direction == "LONG":
                if p > trade.peak_price:
                    trade.peak_price = p
                unrealized_pct = (p - trade.entry_price) / trade.entry_price
                unrealized_atr = (p - trade.entry_price) / atr if atr > 0 else 0
            else:
                if p < trade.peak_price:
                    trade.peak_price = p
                unrealized_pct = (trade.entry_price - p) / trade.entry_price
                unrealized_atr = (trade.entry_price - p) / atr if atr > 0 else 0

            trade.hold_days += 1

            exit_reason = None

            # Disaster stop: 3*ATR
            disaster_pct = 3.0 * trade.atr_at_entry / trade.entry_price if trade.atr_at_entry > 0 else 0.10
            if unrealized_pct <= -disaster_pct:
                exit_reason = "disaster_stop"

            if has_v2_exits:
                # V1.5/V2 exits:
                # Time stop
                if exit_reason is None and trade.hold_days >= CRYPTO_MAX_HOLD_DAYS:
                    exit_reason = "time_stop"

                # Trailing stop
                if exit_reason is None:
                    if not trade.trailing_active:
                        if unrealized_atr >= CRYPTO_TRAILING_TRIGGER_ATR:
                            trade.trailing_active = True
                            trade.trailing_peak = p
                    if trade.trailing_active:
                        if trade.direction == "LONG":
                            if p > trade.trailing_peak:
                                trade.trailing_peak = p
                            trail_level = trade.trailing_peak - CRYPTO_TRAILING_WIDTH_ATR * atr
                            if p <= trail_level:
                                exit_reason = "trailing_stop"
                        else:
                            if p < trade.trailing_peak:
                                trade.trailing_peak = p
                            trail_level = trade.trailing_peak + CRYPTO_TRAILING_WIDTH_ATR * atr
                            if p >= trail_level:
                                exit_reason = "trailing_stop"

            # Signal decay (both v1 and v2): get fresh prediction
            if exit_reason is None:
                bars_slice = sym_bars_date_idx[sym]
                bars_up_to = bars_slice[bars_slice["_date"] <= current_date].drop(columns=["_date"])
                if len(bars_up_to) >= 50 and sym in predictors:
                    try:
                        result = predictors[sym].predict(bars_up_to, vix_df)
                        er = result["expected_return"]
                        if trade.direction == "LONG" and er <= 0:
                            exit_reason = "signal_decay"
                        elif trade.direction == "SHORT" and er >= 0:
                            exit_reason = "signal_decay"
                    except Exception:
                        pass

            if exit_reason:
                # Close position
                trade.exit_date = current_date
                trade.exit_price = p
                if trade.direction == "LONG":
                    trade.pnl = trade.qty * (p - trade.entry_price)
                else:
                    trade.pnl = trade.qty * (trade.entry_price - p)
                trade.exit_reason = exit_reason
                portfolio.cash += trade.qty * p if trade.direction == "LONG" else (
                    trade.qty * trade.entry_price + trade.pnl)
                portfolio.closed_trades.append(trade)
                closed_syms.append(sym)

        for sym in closed_syms:
            del portfolio.positions[sym]

        # --- Check entries ---
        regime_ok = regime_lookup.get(current_date, False)

        for sym in SYMBOLS:
            if sym in portfolio.positions:
                continue  # already have position
            if sym not in predictors:
                continue

            row = sym_date_lookup[sym].get(current_date)
            if row is None:
                continue

            p = float(row["close"])
            atr = float(row["atr"])
            rv = float(row["rv30"])
            sma50 = float(row["sma50"])
            trend = "+" if p > sma50 else "-"

            # Regime gate
            if not regime_ok:
                regime_blocked += 1
                continue

            # Get prediction
            bars_slice = sym_bars_date_idx[sym]
            bars_up_to = bars_slice[bars_slice["_date"] <= current_date].drop(columns=["_date"])
            if len(bars_up_to) < 50:
                continue

            try:
                result = predictors[sym].predict(bars_up_to, vix_df)
                er = result["expected_return"]
            except Exception:
                continue

            total_signals += 1

            # Threshold: V2 uses vol-adjusted, V1/V1.5 use fixed
            if has_v2_entries:
                threshold = CRYPTO_VOL_THRESHOLD_K * atr / p
            else:
                threshold = COST_THRESHOLD  # 0.1% fixed

            if abs(er) <= threshold:
                threshold_blocked += 1
                continue

            # Trend filter
            if er > 0 and trend != "+":
                trend_blocked += 1
                continue
            if er < 0 and trend != "-":
                trend_blocked += 1
                continue

            # Direction
            direction = "LONG" if er > 0 else "SHORT"

            # Position sizing
            equity = portfolio.equity(prices)
            sym_weight = weights.get(sym, 1.0 / len(SYMBOLS))
            sym_alloc = equity * sym_weight

            if has_v2_entries:
                # V2: vol-targeted sizing
                signal_strength = min(1.0, max(0.1, abs(er) / threshold))
                vol_scale = min(2.0, CRYPTO_TARGET_VOL_ANN / rv) if rv > 0 else 1.0
                sizing = min(1.0, signal_strength * vol_scale)
                invest = sym_alloc * sizing
            else:
                # V1/V1.5: signal-proportional sizing
                signal_pct = min(1.0, max(0.1, abs(er) / TARGET_RETURN))
                invest = sym_alloc * signal_pct

            if invest > portfolio.cash:
                invest = portfolio.cash * 0.95
            if invest < 100:
                continue

            qty = invest / p
            portfolio.cash -= invest

            trade = Trade(
                symbol=sym,
                entry_date=current_date,
                entry_price=p,
                direction=direction,
                size=invest,
                qty=qty,
                atr_at_entry=atr,
                peak_price=p,
                er_at_entry=er,
            )
            portfolio.positions[sym] = trade

        portfolio.record(current_date, prices)

    # Close any remaining positions at end
    for sym, trade in list(portfolio.positions.items()):
        p = prices.get(sym, trade.entry_price)
        trade.exit_date = common_dates[-1]
        trade.exit_price = p
        if trade.direction == "LONG":
            trade.pnl = trade.qty * (p - trade.entry_price)
        else:
            trade.pnl = trade.qty * (trade.entry_price - p)
        trade.exit_reason = "end_of_backtest"
        portfolio.closed_trades.append(trade)

    # -----------------------------------------------------------------------
    # 5. Compute results
    # -----------------------------------------------------------------------
    eq_df = pd.DataFrame(portfolio.equity_curve)
    if eq_df.empty:
        print("  No equity data generated.")
        return {}

    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity / initial_capital) - 1
    n_days = len(eq_df)
    n_years = n_days / 365  # crypto uses 365

    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if total_return > -1 else -1.0

    eq_df["daily_return"] = eq_df["equity"].pct_change().fillna(0)
    daily_std = eq_df["daily_return"].std()
    sharpe = (eq_df["daily_return"].mean() / daily_std * np.sqrt(365)) if daily_std > 0 else 0.0

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["drawdown"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = eq_df["drawdown"].min()

    trades = portfolio.closed_trades
    n_trades = len(trades)
    wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
    losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    avg_win = np.mean([t.pnl / t.size for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl / t.size for t in losses]) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_hold = np.mean([t.hold_days for t in trades]) if trades else 0.0

    # Per-symbol breakdown
    sym_stats: Dict[str, dict] = {}
    for sym in SYMBOLS:
        sym_trades = [t for t in trades if t.symbol == sym]
        if not sym_trades:
            sym_stats[sym] = {"trades": 0, "pnl": 0, "win_rate": 0}
            continue
        sym_pnl = sum(t.pnl for t in sym_trades if t.pnl)
        sym_wins = sum(1 for t in sym_trades if t.pnl and t.pnl > 0)
        sym_stats[sym] = {
            "trades": len(sym_trades),
            "pnl": sym_pnl,
            "win_rate": sym_wins / len(sym_trades),
            "avg_hold": np.mean([t.hold_days for t in sym_trades]),
        }

    # Exit reason breakdown
    exit_reasons: Dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Direction breakdown
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl for t in long_trades if t.pnl) if long_trades else 0
    short_pnl = sum(t.pnl for t in short_trades if t.pnl) if short_trades else 0

    # -----------------------------------------------------------------------
    # 6. Print results
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS: CRYPTO STRATEGY {version.upper()}")
    print(f"{'='*70}")

    print(f"\n  PORTFOLIO SUMMARY:")
    print(f"    Initial Capital:    ${initial_capital:>12,.2f}")
    print(f"    Final Equity:       ${final_equity:>12,.2f}")
    print(f"    Total Return:        {total_return:>11.2%}")
    print(f"    Annualized Return:   {ann_return:>11.2%}")
    print(f"    Sharpe Ratio:        {sharpe:>11.3f}")
    print(f"    Max Drawdown:        {max_dd:>11.2%}")

    print(f"\n  TRADE STATISTICS:")
    print(f"    Total Trades:        {n_trades:>8d}")
    print(f"    Win Rate:            {win_rate:>11.1%}")
    print(f"    Avg Win:             {avg_win:>11.2%}")
    print(f"    Avg Loss:            {avg_loss:>11.2%}")
    print(f"    Profit Factor:       {profit_factor:>11.3f}")
    print(f"    Avg Hold (days):     {avg_hold:>11.1f}")

    print(f"\n  DIRECTION BREAKDOWN:")
    print(f"    Long trades:  {len(long_trades):>4d} | PnL: ${long_pnl:>+12,.2f}")
    print(f"    Short trades: {len(short_trades):>4d} | PnL: ${short_pnl:>+12,.2f}")

    print(f"\n  EXIT REASONS:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / n_trades * 100 if n_trades > 0 else 0
        print(f"    {reason:<20s}: {count:>4d} ({pct:>5.1f}%)")

    print(f"\n  PER-SYMBOL BREAKDOWN:")
    print(f"    {'Symbol':<10s} {'Trades':>7s} {'Win Rate':>9s} {'PnL':>14s} {'Avg Hold':>9s}")
    print(f"    {'-'*50}")
    for sym in SYMBOLS:
        st = sym_stats[sym]
        print(f"    {sym:<10s} {st['trades']:>7d} {st.get('win_rate', 0):>8.1%} "
              f"${st.get('pnl', 0):>+12,.2f} {st.get('avg_hold', 0):>8.1f}d")

    print(f"\n  SIGNAL FILTER BREAKDOWN:")
    print(f"    Total signals:      {total_signals:>6d}")
    print(f"    Regime blocked:     {regime_blocked:>6d}")
    print(f"    Threshold blocked:  {threshold_blocked:>6d}")
    print(f"    Trend blocked:      {trend_blocked:>6d}")

    pf_display = f"{profit_factor:.3f}" if profit_factor != float("inf") else "inf"
    print(f"\n{'='*70}")
    print(f"  {version.upper()} | Return: {total_return:+.2%} | Sharpe: {sharpe:.3f} | "
          f"MaxDD: {max_dd:.2%} | WR: {win_rate:.1%} | PF: {pf_display} | Trades: {n_trades}")
    print(f"{'='*70}\n")

    return {
        "version": version,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold": avg_hold,
        "sym_stats": sym_stats,
        "exit_reasons": exit_reasons,
        "equity_curve": eq_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Crypto V2 Strategy Backtest")
    parser.add_argument("--start", default="2024-06-01",
                        help="Backtest start date (default: 2024-06-01)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--v1", action="store_true",
                        help="Run v1 baseline (no v2 improvements)")
    parser.add_argument("--v15", action="store_true",
                        help="Run v1.5 hybrid (v1 entries + v2 exits)")
    parser.add_argument("--compare", action="store_true",
                        help="Run v1, v1.5, and v2 side-by-side")
    args = parser.parse_args()

    if args.compare:
        print("\n" + "=" * 70)
        print("  RUNNING V1 vs V1.5 vs V2 COMPARISON")
        print("=" * 70)

        r1  = run_backtest(args.start, args.capital, version="v1")
        r15 = run_backtest(args.start, args.capital, version="v1.5")
        r2  = run_backtest(args.start, args.capital, version="v2")

        results = [("V1", r1), ("V1.5", r15), ("V2", r2)]
        valid = [(n, r) for n, r in results if r]

        if len(valid) >= 2:
            print(f"\n{'='*80}")
            print(f"  V1 vs V1.5 vs V2 COMPARISON")
            print(f"{'='*80}")
            header = f"  {'Metric':<25s}"
            for name, _ in valid:
                header += f" {name:>14s}"
            print(header)
            print(f"  {'-'*len(header)}")

            metrics = [
                ("Total Return",      "total_return", "%"),
                ("Ann. Return",        "ann_return",   "%"),
                ("Sharpe Ratio",       "sharpe",       "f"),
                ("Max Drawdown",       "max_dd",       "%"),
                ("Win Rate",           "win_rate",     "%"),
                ("Profit Factor",      "profit_factor","f"),
                ("Trades",             "n_trades",     "d"),
                ("Avg Hold (days)",    "avg_hold",     "f"),
            ]
            for label, key, fmt in metrics:
                line = f"  {label:<25s}"
                for _, r in valid:
                    val = r.get(key, 0)
                    if fmt == "%":
                        line += f" {val:>13.2%}"
                    elif fmt == "d":
                        line += f" {val:>14d}"
                    else:
                        line += f" {val:>14.3f}"
                print(line)
            print(f"  {'-'*len(header)}")
            print()
    else:
        if args.v15:
            ver = "v1.5"
        elif args.v1:
            ver = "v1"
        else:
            ver = "v2"
        run_backtest(args.start, args.capital, version=ver)


if __name__ == "__main__":
    main()
