#!/usr/bin/env python3
"""
ML-Prediction-Driven Backtester
=================================
Walk-forward backtest using LSTM predictions from ml_model.py.

Usage (via main.py):
    python main.py backtest --symbol SPY --start 2024-01-01 --end 2025-12-31
    python main.py backtest --symbol QQQ --start 2023-06-01 --confidence-threshold 0.4
    python main.py backtest --symbol IWM --start 2024-01-01 --stop-loss 0.05 --capital 50000

Requires: trained model (run: python main.py train --symbol SPY first).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.signals_engine import build_adapter, FREDVixFetcher, PROJECT_ROOT
from src.ml_model import (
    FeatureEngine, DirectionLSTM, Predictor, FEATURE_COLS, SEQ_LEN,
    DEFAULT_MODEL_DIR, _fetch_vix_for_training,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtester")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")


# ===================================================================
# Trade & Portfolio tracking
# ===================================================================
@dataclass
class Trade:
    entry_date: object          # date or datetime
    entry_price: float
    direction: str              # "LONG"
    size: float                 # number of shares
    exit_date: object = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""


@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    position: Optional[Trade] = None
    closed_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


# ===================================================================
# Backtest result
# ===================================================================
@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_trade_duration_days: float
    equity_curve: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)


# ===================================================================
# Backtester
# ===================================================================
class Backtester:
    """Walk-forward backtester driven by ML predictions."""

    def __init__(
        self,
        symbol: str,
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        confidence_threshold: float = 0.2,
        stop_loss_pct: float = 0.03,
        position_pct: float = 0.95,
    ):
        self.symbol = symbol
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.position_pct = position_pct

    def run(self, start_date: str, end_date: Optional[str] = None,
            seq_len: int = SEQ_LEN) -> BacktestResult:
        """Run the walk-forward backtest.

        1. Fetch full history
        2. Build features + normalize (using saved scaler)
        3. Walk bar-by-bar: predict, apply entry/exit rules, track equity
        4. Compute performance metrics
        """
        # Fetch data
        lookback = 1000
        log.info("Fetching daily data for %s...", self.symbol)
        daily = self.adapter.fetch_daily(self.symbol, lookback)
        log.info("Got %d daily bars.", len(daily))

        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=lookback)
        log.info("Got %d VIX rows.", len(vix_df))

        # Build features using saved scaler
        engine = FeatureEngine()
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_scaler.json")
        if not os.path.exists(scaler_path):
            log.error("No scaler for %s. Train first: python main.py train --symbol %s",
                      self.symbol, self.symbol)
            sys.exit(1)
        engine.load_scaler(scaler_path)

        features = engine.build_features(daily, vix_df)
        features_norm = engine.transform(features)
        log.info("Built %d feature rows.", len(features_norm))

        # Load model
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_lstm.pt")
        model = DirectionLSTM(n_features=len(FEATURE_COLS))
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()

        # Date filtering
        daily_dates = pd.to_datetime(daily["ts"]).dt.date
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date() if end_date else datetime.now(timezone.utc).date()

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        # Walk through each bar
        feature_indices = features_norm.index.tolist()
        log.info("Walking %d bars from %s to %s...", len(feature_indices), start_dt, end_dt)

        for idx_pos in range(seq_len, len(feature_indices)):
            feat_idx = feature_indices[idx_pos]
            bar_date = daily_dates.iloc[feat_idx]

            if bar_date < start_dt:
                continue
            if bar_date > end_dt:
                break

            bar_close = float(daily["close"].iloc[feat_idx])

            # Get prediction
            window_start = idx_pos - seq_len
            window = features_norm.iloc[window_start:idx_pos].values
            x = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                prob = model(x).item()
            direction = "UP" if prob > 0.5 else "DOWN"
            confidence = abs(prob - 0.5) * 2

            # --- Exit logic ---
            if portfolio.position is not None:
                entry_price = portfolio.position.entry_price
                unrealized_pct = (bar_close - entry_price) / entry_price

                # Stop loss
                if unrealized_pct <= -self.stop_loss_pct:
                    self._close_position(portfolio, bar_date, bar_close, "stop_loss")
                # Signal flip
                elif direction == "DOWN" and confidence >= self.confidence_threshold:
                    self._close_position(portfolio, bar_date, bar_close, "signal_flip")

            # --- Entry logic ---
            if (portfolio.position is None
                    and direction == "UP"
                    and confidence >= self.confidence_threshold):
                self._open_position(portfolio, bar_date, bar_close)

            # Record daily equity
            self._record_equity(portfolio, bar_date, bar_close)

        # Close any open position at end
        if portfolio.position is not None:
            last_idx = feature_indices[-1]
            last_close = float(daily["close"].iloc[last_idx])
            last_date = daily_dates.iloc[last_idx]
            self._close_position(portfolio, last_date, last_close, "end_of_backtest")
            self._record_equity(portfolio, last_date, last_close)

        return self._compute_results(portfolio)

    def _open_position(self, portfolio: Portfolio, date, price: float) -> None:
        equity = self._current_equity(portfolio, price)
        invest = equity * self.position_pct
        shares = invest / price
        portfolio.cash -= invest
        portfolio.position = Trade(
            entry_date=date, entry_price=price,
            direction="LONG", size=shares,
        )

    def _close_position(self, portfolio: Portfolio, date, price: float,
                        reason: str) -> None:
        trade = portfolio.position
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        proceeds = trade.size * price
        trade.pnl = proceeds - (trade.size * trade.entry_price)
        portfolio.cash += proceeds
        portfolio.closed_trades.append(trade)
        portfolio.position = None

    def _current_equity(self, portfolio: Portfolio, current_price: float) -> float:
        equity = portfolio.cash
        if portfolio.position is not None:
            equity += portfolio.position.size * current_price
        return equity

    def _record_equity(self, portfolio: Portfolio, date, price: float) -> None:
        portfolio.equity_curve.append({
            "date": date,
            "equity": self._current_equity(portfolio, price),
        })

    def _compute_results(self, portfolio: Portfolio) -> BacktestResult:
        eq_df = pd.DataFrame(portfolio.equity_curve)

        if eq_df.empty:
            return BacktestResult(
                symbol=self.symbol, start_date="N/A", end_date="N/A",
                initial_capital=self.initial_capital, final_equity=self.initial_capital,
                total_return_pct=0.0, annualized_return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown_pct=0.0, total_trades=0, win_rate=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0, profit_factor=0.0,
                avg_trade_duration_days=0.0, equity_curve=eq_df,
            )

        final_equity = eq_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital) - 1

        # Daily returns for Sharpe
        eq_df["daily_return"] = eq_df["equity"].pct_change().fillna(0)
        n_days = len(eq_df)
        n_years = n_days / 252

        annualized_return = (
            (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
            if total_return > -1 else -1.0
        )

        daily_std = eq_df["daily_return"].std()
        sharpe = (eq_df["daily_return"].mean() / daily_std * np.sqrt(252)
                  if daily_std > 0 else 0.0)

        # Max drawdown
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min()

        # Trade-level stats
        trades = portfolio.closed_trades
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        avg_win = (np.mean([t.pnl / (t.size * t.entry_price) for t in wins])
                   if wins else 0.0)
        avg_loss = (np.mean([t.pnl / (t.size * t.entry_price) for t in losses])
                    if losses else 0.0)

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        def _trade_days(t: Trade) -> int:
            try:
                d1 = t.entry_date if not hasattr(t.entry_date, "date") else t.entry_date
                d2 = t.exit_date if not hasattr(t.exit_date, "date") else t.exit_date
                return (pd.Timestamp(d2) - pd.Timestamp(d1)).days
            except Exception:
                return 0

        avg_duration = (np.mean([_trade_days(t) for t in trades])
                        if trades else 0.0)

        return BacktestResult(
            symbol=self.symbol,
            start_date=str(eq_df["date"].iloc[0]),
            end_date=str(eq_df["date"].iloc[-1]),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return * 100, 2),
            annualized_return_pct=round(annualized_return * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd * 100, 2),
            total_trades=n_trades,
            win_rate=round(win_rate, 3),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
            avg_trade_duration_days=round(avg_duration, 1),
            equity_curve=eq_df[["date", "equity"]],
            trades=portfolio.closed_trades,
        )


# ===================================================================
# Report
# ===================================================================
def print_report(result: BacktestResult) -> None:
    print("\n" + "=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)
    print(f"  Symbol             : {result.symbol}")
    print(f"  Period             : {result.start_date} to {result.end_date}")
    print(f"  Initial Capital    : ${result.initial_capital:,.2f}")
    print(f"  Final Equity       : ${result.final_equity:,.2f}")
    print("-" * 60)
    print(f"  Total Return       : {result.total_return_pct:+.2f}%")
    print(f"  Annualized Return  : {result.annualized_return_pct:+.2f}%")
    print(f"  Sharpe Ratio       : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown       : {result.max_drawdown_pct:.2f}%")
    print("-" * 60)
    print(f"  Total Trades       : {result.total_trades}")
    print(f"  Win Rate           : {result.win_rate:.1%}")
    print(f"  Avg Win            : {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss           : {result.avg_loss_pct:+.2f}%")
    print(f"  Profit Factor      : {result.profit_factor:.3f}")
    print(f"  Avg Hold Duration  : {result.avg_trade_duration_days:.1f} days")
    print("=" * 60)

    # --- Trade history table ---
    if result.trades:
        print(f"\n  TRADE HISTORY ({len(result.trades)} trades)")
        print("-" * 100)
        print(f"  {'#':>3}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
              f"{'Exit $':>10}  {'Shares':>10}  {'P&L':>12}  {'Return':>8}  {'Reason'}")
        print("-" * 100)
        for i, t in enumerate(result.trades, 1):
            pnl = t.pnl or 0.0
            ret_pct = (pnl / (t.size * t.entry_price) * 100) if t.size and t.entry_price else 0.0
            print(f"  {i:>3}  {str(t.entry_date):>12}  {t.entry_price:>10,.2f}  "
                  f"{str(t.exit_date):>12}  {t.exit_price:>10,.2f}  {t.size:>10,.2f}  "
                  f"{'${:>+,.2f}'.format(pnl):>12}  {ret_pct:>+7.2f}%  {t.exit_reason}")
        print("-" * 100)

    # Save equity curve
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"backtest_{result.symbol}.csv")
    result.equity_curve.to_csv(csv_path, index=False)
    print(f"\n  Equity curve saved to {csv_path}")

    # Save trade history
    if result.trades:
        trades_csv = os.path.join(OUTPUT_DIR, f"trades_{result.symbol}.csv")
        rows = []
        for t in result.trades:
            pnl = t.pnl or 0.0
            ret_pct = (pnl / (t.size * t.entry_price)) if t.size and t.entry_price else 0.0
            rows.append({
                "entry_date": t.entry_date, "entry_price": round(t.entry_price, 2),
                "exit_date": t.exit_date, "exit_price": round(t.exit_price, 2),
                "shares": round(t.size, 2), "pnl": round(pnl, 2),
                "return_pct": round(ret_pct * 100, 2), "exit_reason": t.exit_reason,
            })
        pd.DataFrame(rows).to_csv(trades_csv, index=False)
        print(f"  Trade history saved to {trades_csv}\n")
    else:
        print()


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-driven backtester for ETF signals.",
    )
    parser.add_argument("--symbol", required=True, help="Symbol to backtest (e.g. SPY)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--confidence-threshold", type=float, default=0.2,
                        help="Min ML confidence to trade (default: 0.2)")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                        help="Stop loss percentage (default: 0.03 = 3%%)")

    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")

    bt = Backtester(
        symbol=args.symbol.upper(),
        adapter=adapter,
        fred_key=fred_key,
        initial_capital=args.capital,
        confidence_threshold=args.confidence_threshold,
        stop_loss_pct=args.stop_loss,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    print_report(result)


if __name__ == "__main__":
    main()
