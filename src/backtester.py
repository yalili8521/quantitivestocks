#!/usr/bin/env python3
"""
ML-Prediction-Driven Backtester
=================================
Walk-forward backtest using LSTM predictions from ml_model.py.
Supports LONG and SHORT positions, trailing stop, take-profit, and intraday mode.

Usage (via main.py):
    python main.py backtest --symbol SPY --start 2024-01-01
    python main.py backtest --symbol SPY --start 2024-01-01 --trailing-stop 0.05 --take-profit 0.08
    python main.py backtest --symbol SPY --start 2025-01-01 --mode intraday

Requires: trained model (run: python main.py train --symbol SPY first).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
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
    direction: str              # "LONG" or "SHORT"
    size: float                 # number of shares
    peak_price: float = 0.0     # best price since entry (highest for LONG, lowest for SHORT)
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
    price_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)


# ===================================================================
# Backtester
# ===================================================================
class Backtester:
    """Walk-forward backtester driven by ML predictions.

    Supports LONG and SHORT positions with trailing stop-loss,
    take-profit, and immediate position flipping.
    """

    def __init__(
        self,
        symbol: str,
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        confidence_threshold: float = 0.2,
        exit_confidence: float = 0.1,
        trailing_stop_pct: float = 0.05,
        take_profit_pct: float = 0.08,
        position_pct: float = 0.95,
        mode: str = "daily",
        intraday_interval: str = "5min",
    ):
        self.symbol = symbol
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.exit_confidence = exit_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.position_pct = position_pct
        self.mode = mode
        self.intraday_interval = intraday_interval

    def run(self, start_date: str, end_date: Optional[str] = None,
            seq_len: int = SEQ_LEN) -> BacktestResult:
        """Run the walk-forward backtest."""
        # Fetch data
        if self.mode == "daily":
            lookback = 1000
            log.info("Fetching daily data for %s...", self.symbol)
            bars = self.adapter.fetch_daily(self.symbol, lookback)
        else:
            start_dt_temp = pd.to_datetime(start_date).date()
            end_dt_temp = (pd.to_datetime(end_date).date() if end_date
                          else datetime.now(timezone.utc).date())
            days_needed = (end_dt_temp - start_dt_temp).days + 60  # extra for warmup
            lookback = days_needed
            log.info("Fetching %d days of %s intraday data for %s...",
                     days_needed, self.intraday_interval, self.symbol)
            bars = self.adapter.fetch_intraday(
                self.symbol, self.intraday_interval, lookback_days=days_needed)

        log.info("Got %d bars.", len(bars))

        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=max(lookback, 500))
        log.info("Got %d VIX rows.", len(vix_df))

        # Build features using saved scaler
        engine = FeatureEngine()
        suffix = "" if self.mode == "daily" else f"_{self.intraday_interval}"
        scaler_path = os.path.join(self.model_dir, f"{self.symbol}_scaler{suffix}.json")
        if not os.path.exists(scaler_path):
            log.error("No scaler for %s (%s mode). Train first: "
                      "python main.py train --symbol %s%s",
                      self.symbol, self.mode, self.symbol,
                      f" --mode {self.mode}" if self.mode != "daily" else "")
            sys.exit(1)
        engine.load_scaler(scaler_path)

        features = engine.build_features(bars, vix_df, mode=self.mode)
        features_norm = engine.transform(features)
        log.info("Built %d feature rows.", len(features_norm))

        # Load model
        weights_path = os.path.join(self.model_dir, f"{self.symbol}_lstm{suffix}.pt")
        model = DirectionLSTM(n_features=len(FEATURE_COLS))
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        model.eval()

        # Date filtering
        bar_timestamps = pd.to_datetime(bars["ts"])
        bar_dates = bar_timestamps.dt.date
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date() if end_date else datetime.now(timezone.utc).date()

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        # Walk through each bar
        feature_indices = features_norm.index.tolist()
        log.info("Walking %d bars from %s to %s...", len(feature_indices), start_dt, end_dt)

        last_bar_idx = len(feature_indices) - 1
        for idx_pos in range(seq_len, len(feature_indices)):
            feat_idx = feature_indices[idx_pos]
            bar_date = bar_dates.iloc[feat_idx]

            if bar_date < start_dt:
                continue
            if bar_date > end_dt:
                break

            is_last_bar = (idx_pos == last_bar_idx or
                           (idx_pos + 1 < len(feature_indices) and
                            bar_dates.iloc[feature_indices[idx_pos + 1]] > end_dt))

            bar_close = float(bars["close"].iloc[feat_idx])

            # Get prediction
            window_start = idx_pos - seq_len
            window = features_norm.iloc[window_start:idx_pos].values
            x = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                prob = model(x).item()
            direction = "UP" if prob > 0.5 else "DOWN"
            confidence = abs(prob - 0.5) * 2

            # --- Exit logic ---
            flip_direction = None
            if portfolio.position is not None:
                pos = portfolio.position

                # Update peak tracking
                if pos.direction == "LONG":
                    pos.peak_price = max(pos.peak_price, bar_close)
                    unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price
                    drawdown_from_peak = (pos.peak_price - bar_close) / pos.peak_price
                else:  # SHORT
                    pos.peak_price = min(pos.peak_price, bar_close)
                    unrealized_pct = (pos.entry_price - bar_close) / pos.entry_price
                    drawdown_from_peak = (bar_close - pos.peak_price) / pos.peak_price if pos.peak_price > 0 else 0

                # Take profit
                if unrealized_pct >= self.take_profit_pct:
                    self._close_position(portfolio, bar_date, bar_close, "take_profit")
                # Trailing stop
                elif drawdown_from_peak >= self.trailing_stop_pct:
                    self._close_position(portfolio, bar_date, bar_close, "trailing_stop")
                # Signal flip (lower exit confidence threshold)
                elif (pos.direction == "LONG" and direction == "DOWN"
                      and confidence >= self.exit_confidence):
                    self._close_position(portfolio, bar_date, bar_close, "signal_flip")
                    flip_direction = "SHORT"
                elif (pos.direction == "SHORT" and direction == "UP"
                      and confidence >= self.exit_confidence):
                    self._close_position(portfolio, bar_date, bar_close, "signal_flip")
                    flip_direction = "LONG"

            # --- Entry logic (includes immediate flip) ---
            # Don't open new positions on the last bar (would be force-closed at same price)
            if portfolio.position is None and not is_last_bar:
                if flip_direction is not None and confidence >= self.exit_confidence:
                    # Immediate flip — enter opposite direction
                    self._open_position(portfolio, bar_date, bar_close, flip_direction)
                elif confidence >= self.confidence_threshold:
                    if direction == "UP":
                        self._open_position(portfolio, bar_date, bar_close, "LONG")
                    elif direction == "DOWN":
                        self._open_position(portfolio, bar_date, bar_close, "SHORT")

            # Record equity
            self._record_equity(portfolio, bar_date, bar_close)

        # Close any open position at end
        if portfolio.position is not None:
            last_idx = feature_indices[-1]
            last_close = float(bars["close"].iloc[last_idx])
            last_date = bar_dates.iloc[last_idx]
            self._close_position(portfolio, last_date, last_close, "end_of_backtest")
            self._record_equity(portfolio, last_date, last_close)

        # Build price series for charts (only the backtest date range)
        price_rows = []
        for idx_pos in range(seq_len, len(feature_indices)):
            feat_idx = feature_indices[idx_pos]
            bd = bar_dates.iloc[feat_idx]
            if bd < start_dt:
                continue
            if bd > end_dt:
                break
            price_rows.append({
                "date": bd,
                "close": float(bars["close"].iloc[feat_idx]),
            })
        price_df = pd.DataFrame(price_rows)

        return self._compute_results(portfolio, price_df)

    def _open_position(self, portfolio: Portfolio, date, price: float,
                       direction: str = "LONG") -> None:
        equity = self._current_equity(portfolio, price)
        invest = equity * self.position_pct
        shares = invest / price
        portfolio.cash -= invest  # held as collateral for both LONG and SHORT
        portfolio.position = Trade(
            entry_date=date, entry_price=price,
            direction=direction, size=shares,
            peak_price=price,
        )

    def _close_position(self, portfolio: Portfolio, date, price: float,
                        reason: str) -> None:
        trade = portfolio.position
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        if trade.direction == "LONG":
            trade.pnl = trade.size * (price - trade.entry_price)
            proceeds = trade.size * price
        else:  # SHORT
            trade.pnl = trade.size * (trade.entry_price - price)
            proceeds = trade.size * trade.entry_price + trade.pnl
        portfolio.cash += proceeds
        portfolio.closed_trades.append(trade)
        portfolio.position = None

    def _current_equity(self, portfolio: Portfolio, current_price: float) -> float:
        equity = portfolio.cash
        if portfolio.position is not None:
            pos = portfolio.position
            if pos.direction == "LONG":
                equity += pos.size * current_price
            else:  # SHORT
                equity += pos.size * pos.entry_price + pos.size * (pos.entry_price - current_price)
        return equity

    def _record_equity(self, portfolio: Portfolio, date, price: float) -> None:
        portfolio.equity_curve.append({
            "date": date,
            "equity": self._current_equity(portfolio, price),
        })

    def _compute_results(self, portfolio: Portfolio,
                         price_df: pd.DataFrame = None) -> BacktestResult:
        eq_df = pd.DataFrame(portfolio.equity_curve)

        if price_df is None:
            price_df = pd.DataFrame(columns=["date", "close"])

        if eq_df.empty:
            return BacktestResult(
                symbol=self.symbol, start_date="N/A", end_date="N/A",
                initial_capital=self.initial_capital, final_equity=self.initial_capital,
                total_return_pct=0.0, annualized_return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown_pct=0.0, total_trades=0, win_rate=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0, profit_factor=0.0,
                avg_trade_duration_days=0.0, equity_curve=eq_df,
                price_series=price_df,
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
            price_series=price_df,
            drawdown_series=eq_df[["date", "drawdown"]],
            trades=portfolio.closed_trades,
        )


# ===================================================================
# Interactive Charts
# ===================================================================
def generate_charts(result: BacktestResult) -> Optional[str]:
    """Generate an interactive Plotly HTML report with 3 charts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed; skipping charts. Run: pip install plotly")
        return None

    if result.price_series.empty or result.equity_curve.empty:
        log.warning("No data to chart.")
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            f"{result.symbol} Price & Trades",
            "Portfolio Equity",
            "Drawdown",
        ),
    )

    # --- Chart 1: Price with trade markers ---
    dates = result.price_series["date"]
    closes = result.price_series["close"]

    fig.add_trace(
        go.Scatter(
            x=dates, y=closes,
            mode="lines", name="Close Price",
            line=dict(color="#636EFA", width=1.5),
        ),
        row=1, col=1,
    )

    # Trade shading
    for t in result.trades:
        pnl = t.pnl or 0.0
        color = "rgba(0,200,83,0.12)" if pnl > 0 else "rgba(255,59,48,0.12)"
        fig.add_vrect(
            x0=t.entry_date, x1=t.exit_date,
            fillcolor=color, layer="below", line_width=0,
            row=1, col=1,
        )

    # Entry markers — color by direction
    long_entries = [t for t in result.trades if t.direction == "LONG"]
    short_entries = [t for t in result.trades if t.direction == "SHORT"]

    if long_entries:
        fig.add_trace(
            go.Scatter(
                x=[t.entry_date for t in long_entries],
                y=[t.entry_price for t in long_entries],
                mode="markers", name="Buy (LONG)",
                marker=dict(symbol="triangle-up", size=12, color="#00C853",
                            line=dict(width=1, color="white")),
                text=[f"BUY {t.size:.0f} @ ${t.entry_price:,.2f}" for t in long_entries],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    if short_entries:
        fig.add_trace(
            go.Scatter(
                x=[t.entry_date for t in short_entries],
                y=[t.entry_price for t in short_entries],
                mode="markers", name="Short (SHORT)",
                marker=dict(symbol="triangle-down", size=12, color="#FF9500",
                            line=dict(width=1, color="white")),
                text=[f"SHORT {t.size:.0f} @ ${t.entry_price:,.2f}" for t in short_entries],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    # Exit markers
    exits = [t for t in result.trades if t.exit_price]
    long_exits = [t for t in exits if t.direction == "LONG"]
    short_exits = [t for t in exits if t.direction == "SHORT"]

    if long_exits:
        fig.add_trace(
            go.Scatter(
                x=[t.exit_date for t in long_exits],
                y=[t.exit_price for t in long_exits],
                mode="markers", name="Sell (LONG exit)",
                marker=dict(symbol="triangle-down", size=10, color="#FF3B30",
                            line=dict(width=1, color="white")),
                text=[f"SELL @ ${t.exit_price:,.2f}<br>P&L: ${t.pnl:+,.2f}<br>{t.exit_reason}"
                      for t in long_exits],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    if short_exits:
        fig.add_trace(
            go.Scatter(
                x=[t.exit_date for t in short_exits],
                y=[t.exit_price for t in short_exits],
                mode="markers", name="Cover (SHORT exit)",
                marker=dict(symbol="triangle-up", size=10, color="#34C759",
                            line=dict(width=1, color="white")),
                text=[f"COVER @ ${t.exit_price:,.2f}<br>P&L: ${t.pnl:+,.2f}<br>{t.exit_reason}"
                      for t in short_exits],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    # --- Chart 2: Equity curve ---
    eq_dates = result.equity_curve["date"]
    eq_values = result.equity_curve["equity"]

    fig.add_trace(
        go.Scatter(
            x=eq_dates, y=eq_values,
            mode="lines", name="Equity",
            line=dict(color="#00C853", width=2),
        ),
        row=2, col=1,
    )

    fig.add_hline(
        y=result.initial_capital, line_dash="dash",
        line_color="gray", line_width=1,
        annotation_text=f"${result.initial_capital:,.0f}",
        annotation_position="top left",
        row=2, col=1,
    )

    # Set Y-axis range to show equity changes clearly (not starting from 0)
    eq_min = float(eq_values.min())
    eq_max = float(eq_values.max())
    eq_pad = (eq_max - eq_min) * 0.1
    fig.update_yaxes(
        range=[eq_min - eq_pad, eq_max + eq_pad],
        row=2, col=1,
    )

    # --- Chart 3: Drawdown ---
    if not result.drawdown_series.empty:
        dd_dates = result.drawdown_series["date"]
        dd_values = result.drawdown_series["drawdown"] * 100

        fig.add_trace(
            go.Scatter(
                x=dd_dates, y=dd_values,
                mode="lines", name="Drawdown %",
                line=dict(color="#FF3B30", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,59,48,0.15)",
            ),
            row=3, col=1,
        )

    # --- Layout ---
    long_trades = [t for t in result.trades if t.direction == "LONG"]
    short_trades = [t for t in result.trades if t.direction == "SHORT"]
    stats_text = (
        f"Return: {result.total_return_pct:+.2f}%  |  "
        f"Sharpe: {result.sharpe_ratio:.3f}  |  "
        f"Max DD: {result.max_drawdown_pct:.2f}%  |  "
        f"Win Rate: {result.win_rate:.0%}  |  "
        f"Trades: {result.total_trades} ({len(long_trades)}L / {len(short_trades)}S)  |  "
        f"PF: {result.profit_factor:.2f}"
    )

    fig.update_layout(
        title=dict(
            text=(f"{result.symbol} Backtest: {result.start_date} to {result.end_date}"
                  f"<br><sup>{stats_text}</sup>"),
            x=0.5,
        ),
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Save and open
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chart_path = os.path.join(OUTPUT_DIR, f"backtest_{result.symbol}_chart.html")
    fig.write_html(chart_path)
    log.info("Chart saved to %s", chart_path)

    try:
        webbrowser.open(f"file:///{chart_path.replace(os.sep, '/')}")
    except Exception:
        pass

    return chart_path


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

    # Per-direction breakdown
    long_trades = [t for t in result.trades if t.direction == "LONG"]
    short_trades = [t for t in result.trades if t.direction == "SHORT"]
    if long_trades or short_trades:
        print("-" * 60)
        if long_trades:
            long_wins = sum(1 for t in long_trades if t.pnl and t.pnl > 0)
            long_wr = long_wins / len(long_trades) if long_trades else 0
            print(f"  LONG  trades: {len(long_trades):>3}   win rate: {long_wr:.0%}")
        if short_trades:
            short_wins = sum(1 for t in short_trades if t.pnl and t.pnl > 0)
            short_wr = short_wins / len(short_trades) if short_trades else 0
            print(f"  SHORT trades: {len(short_trades):>3}   win rate: {short_wr:.0%}")
    print("=" * 60)

    # --- Trade history table ---
    if result.trades:
        print(f"\n  TRADE HISTORY ({len(result.trades)} trades)")
        print("-" * 110)
        print(f"  {'#':>3}  {'Dir':>5}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
              f"{'Exit $':>10}  {'Shares':>8}  {'P&L':>12}  {'Return':>8}  {'Reason'}")
        print("-" * 110)
        for i, t in enumerate(result.trades, 1):
            pnl = t.pnl or 0.0
            ret_pct = (pnl / (t.size * t.entry_price) * 100) if t.size and t.entry_price else 0.0
            print(f"  {i:>3}  {t.direction:>5}  {str(t.entry_date):>12}  {t.entry_price:>10,.2f}  "
                  f"{str(t.exit_date):>12}  {t.exit_price:>10,.2f}  {t.size:>8,.1f}  "
                  f"{'${:>+,.2f}'.format(pnl):>12}  {ret_pct:>+7.2f}%  {t.exit_reason}")
        print("-" * 110)

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
                "direction": t.direction,
                "entry_date": t.entry_date, "entry_price": round(t.entry_price, 2),
                "exit_date": t.exit_date, "exit_price": round(t.exit_price, 2),
                "shares": round(t.size, 2), "pnl": round(pnl, 2),
                "return_pct": round(ret_pct * 100, 2), "exit_reason": t.exit_reason,
            })
        pd.DataFrame(rows).to_csv(trades_csv, index=False)
        print(f"  Trade history saved to {trades_csv}\n")
    else:
        print()

    # Generate interactive chart
    chart_path = generate_charts(result)
    if chart_path:
        print(f"  Interactive chart opened in browser: {chart_path}\n")


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
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Min ML confidence to enter (default: 0.2)")
    parser.add_argument("--exit-confidence", type=float, default=0.1,
                        help="Min ML confidence to exit/flip (default: 0.1)")
    parser.add_argument("--trailing-stop", type=float, default=0.05,
                        help="Trailing stop from peak (default: 0.05 = 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.08,
                        help="Take profit target (default: 0.08 = 8%%)")
    parser.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Backtest mode (default: daily)")
    parser.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")

    args = parser.parse_args()

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")

    bt = Backtester(
        symbol=args.symbol.upper(),
        adapter=adapter,
        fred_key=fred_key,
        initial_capital=args.capital,
        confidence_threshold=args.confidence,
        exit_confidence=args.exit_confidence,
        trailing_stop_pct=args.trailing_stop,
        take_profit_pct=args.take_profit,
        mode=args.mode,
        intraday_interval=args.interval,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    print_report(result)


if __name__ == "__main__":
    main()
