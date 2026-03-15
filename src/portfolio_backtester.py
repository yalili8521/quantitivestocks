"""
Portfolio-level backtester: simulates a single portfolio with shared capital,
exposure limits, and portfolio-level metrics across multiple symbols.

Unlike per-symbol backtesting, this runs all symbols through one cash pool
so that exposure caps, sector limits, and cash management are realistic.

Usage (via main.py):
    python main.py backtest-portfolio --symbols SPY,QQQ,IWM --start 2024-01-01
    python main.py backtest-portfolio --group swing --start 2024-01-01
    python main.py backtest-portfolio --group swing --start 2024-01-01 --stress-cost-mult 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signals_engine import build_adapter, PROJECT_ROOT, compute_atr
from utils import DEFAULT_MODEL_DIR, SWING_MODEL_DIR, BACKTEST_DIR, COST_THRESHOLD, TARGET_RETURN, _fetch_vix_for_training
from risk_config import get_risk_config, check_position_allowed, SYMBOL_SECTOR, RiskConfig
from cost_model import get_symbol_costs, simulate_fill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("portfolio_backtester")

OUTPUT_DIR = BACKTEST_DIR


@dataclass
class PortfolioTrade:
    symbol: str
    entry_date: object
    entry_price: float
    direction: str
    size: float
    atr_at_entry: float = 0.0
    exit_date: object = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""


@dataclass
class PortfolioState:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, PortfolioTrade] = field(default_factory=dict)
    closed_trades: List[PortfolioTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


@dataclass
class PortfolioResult:
    symbols: List[str]
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
    profit_factor: float
    avg_exposure: float
    max_exposure: float
    max_positions_held: int
    turnover: float
    trades_by_symbol: Dict[str, int]
    equity_curve: pd.DataFrame
    trades: List[PortfolioTrade] = field(default_factory=list)


class PortfolioBacktester:
    """Walk-forward multi-symbol backtester with shared capital and portfolio constraints."""

    def __init__(
        self,
        symbols: List[str],
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        group: str = "swing",
        model_type: str = "swing",
        stress_cost_mult: float = 1.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.symbols = symbols
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.group = group
        self.model_type = model_type
        self.stress_cost_mult = stress_cost_mult
        self.risk = risk_config or get_risk_config(group)

    def run(self, start_date: str, end_date: Optional[str] = None) -> PortfolioResult:
        """Run portfolio-level backtest across all symbols."""
        log.info("=== Portfolio Backtest: %s ===", ", ".join(self.symbols))
        log.info("Group: %s | Model: %s | Capital: $%,.0f | Stress: %.1fx",
                 self.group, self.model_type, self.initial_capital, self.stress_cost_mult)
        log.info("Risk: pos=%.0f%% max_pos=%.0f%% max_sector=%.0f%% max_exp=%.0f%%",
                 self.risk.position_pct * 100, self.risk.max_position_pct * 100,
                 self.risk.max_sector_pct * 100, self.risk.max_total_exposure * 100)

        # Load predictors for all symbols
        predictors = {}
        for sym in self.symbols:
            predictor = self._create_predictor(sym)
            if predictor is not None:
                predictors[sym] = predictor
                log.info("Loaded model for %s", sym)
            else:
                log.warning("No model for %s — skipping", sym)

        if not predictors:
            log.error("No models loaded. Nothing to backtest.")
            return self._empty_result(start_date, end_date)

        # Fetch all data
        all_bars = {}
        for sym in predictors:
            bars = self.adapter.fetch_daily(sym, 1200)
            all_bars[sym] = bars
            log.info("Fetched %d bars for %s", len(bars), sym)

        spy_bars = self.adapter.fetch_daily("SPY", 1200)
        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=1200)

        # Find common date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Build date index from SPY (most liquid)
        spy_dates = pd.to_datetime(spy_bars["ts"]).dt.date
        trade_dates = sorted(set(d for d in spy_dates if start_ts.date() <= d <= end_ts.date()))
        log.info("Trading dates: %d (from %s to %s)", len(trade_dates), trade_dates[0], trade_dates[-1])

        # Initialize portfolio
        portfolio = PortfolioState(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        # Regime filter: SPY SMA(200) + VIX
        spy_close = spy_bars["close"].astype(float)
        spy_sma200 = spy_close.rolling(200).mean()
        spy_date_map = dict(zip(spy_dates.values, spy_close.values))
        spy_sma200_map = dict(zip(spy_dates.values, spy_sma200.values))

        vix_map = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_map[d] = row["vix"]

        exposure_log = []
        max_positions_held = 0

        for day in trade_dates:
            equity = self._compute_equity(portfolio, all_bars, day)

            # Regime check
            spy_price = spy_date_map.get(day)
            spy_sma = spy_sma200_map.get(day)
            vix_val = vix_map.get(day, 20.0)
            regime_ok = True
            if spy_price and spy_sma and not np.isnan(spy_sma):
                if spy_price <= spy_sma:
                    regime_ok = False
            if self.group != "intraday" and vix_val >= 30:
                regime_ok = False

            # Check exits for all open positions
            for sym in list(portfolio.positions.keys()):
                pos = portfolio.positions[sym]
                bars = all_bars.get(sym)
                if bars is None:
                    continue
                bar_dates = pd.to_datetime(bars["ts"]).dt.date
                day_mask = bar_dates == day
                if not day_mask.any():
                    continue
                idx = day_mask.values.nonzero()[0][-1]
                current_price = float(bars["close"].iloc[idx])

                # Get prediction
                pred = self._predict(predictors.get(sym), bars, vix_df, idx)
                expected_return = pred.get("expected_return", 0.0)

                # Disaster stop
                entry_atr = pos.atr_at_entry
                entry_price = pos.entry_price
                if pos.direction == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                disaster_pct = min(
                    self.risk.disaster_stop_max_pct,
                    self.risk.disaster_stop_atr_mult * entry_atr / entry_price
                ) if entry_atr > 0 and entry_price > 0 else 0.10

                if pnl_pct <= -disaster_pct:
                    self._close_position(portfolio, sym, day, current_price, "disaster_stop")
                    continue

                # Signal decay exit
                if pos.direction == "LONG" and expected_return <= 0:
                    self._close_position(portfolio, sym, day, current_price, "signal_decay")
                    continue
                if pos.direction == "SHORT" and expected_return >= 0:
                    self._close_position(portfolio, sym, day, current_price, "signal_decay")
                    continue

            # Check entries (only if regime ok)
            if regime_ok:
                for sym in predictors:
                    if sym in portfolio.positions:
                        continue  # already in position
                    bars = all_bars.get(sym)
                    if bars is None:
                        continue
                    bar_dates = pd.to_datetime(bars["ts"]).dt.date
                    day_mask = bar_dates == day
                    if not day_mask.any():
                        continue
                    idx = day_mask.values.nonzero()[0][-1]
                    current_price = float(bars["close"].iloc[idx])

                    pred = self._predict(predictors.get(sym), bars, vix_df, idx)
                    expected_return = pred.get("expected_return", 0.0)

                    # Trend filter
                    close_series = bars["close"].astype(float)
                    sma50 = close_series.rolling(50).mean()
                    trend = 1.0 if current_price > float(sma50.iloc[idx]) else -1.0

                    enter_dir = None
                    cost_threshold = max(COST_THRESHOLD, get_symbol_costs(sym).round_trip_pct * 1.5)
                    if expected_return > cost_threshold and trend > 0:
                        enter_dir = "LONG"
                    elif expected_return < -cost_threshold and trend < 0:
                        enter_dir = "SHORT"

                    if enter_dir is None:
                        continue

                    # Position sizing
                    signal_pct = min(1.0, max(0.1, abs(expected_return) / TARGET_RETURN))
                    sizing_pct = self.risk.position_pct * signal_pct

                    invest = equity * sizing_pct

                    # Portfolio constraint check
                    pos_dict = self._positions_as_dict(portfolio, all_bars, day)
                    allowed, reason = check_position_allowed(
                        sym, invest, equity, pos_dict, self.risk
                    )
                    if not allowed:
                        continue

                    # ATR for stops
                    high = bars["high"].astype(float)
                    low = bars["low"].astype(float)
                    atr_s = compute_atr(high, low, close_series, period=14)
                    bar_atr = float(atr_s.iloc[idx]) if not np.isnan(float(atr_s.iloc[idx])) else 0.0

                    # Apply cost model
                    fill_price, filled = simulate_fill(
                        sym, current_price, enter_dir,
                        stress_mult=self.stress_cost_mult,
                    )
                    if not filled:
                        continue

                    shares = invest / fill_price
                    portfolio.cash -= invest
                    portfolio.positions[sym] = PortfolioTrade(
                        symbol=sym, entry_date=day, entry_price=fill_price,
                        direction=enter_dir, size=shares, atr_at_entry=bar_atr,
                    )

            # Record equity
            equity = self._compute_equity(portfolio, all_bars, day)
            n_positions = len(portfolio.positions)
            max_positions_held = max(max_positions_held, n_positions)
            total_invested = sum(
                pos.size * self._get_price(all_bars, pos.symbol, day)
                for pos in portfolio.positions.values()
            )
            exposure_pct = total_invested / equity if equity > 0 else 0
            exposure_log.append(exposure_pct)
            portfolio.equity_curve.append({"date": day, "equity": equity, "positions": n_positions})

        # Close remaining positions at end
        for sym in list(portfolio.positions.keys()):
            bars = all_bars.get(sym)
            if bars is not None:
                last_price = float(bars["close"].iloc[-1])
            else:
                last_price = portfolio.positions[sym].entry_price
            self._close_position(portfolio, sym, trade_dates[-1], last_price, "end_of_backtest")

        return self._compute_results(portfolio, trade_dates, exposure_log, max_positions_held)

    def _create_predictor(self, symbol: str):
        """Create predictor based on model type."""
        if self.model_type == "swing":
            try:
                from swing_model import SwingPredictor
                return SwingPredictor(symbol, model_dir=self.model_dir)
            except (FileNotFoundError, ImportError):
                pass
        try:
            from ml_model import Predictor
            return Predictor(symbol, model_dir=self.model_dir)
        except (FileNotFoundError, RuntimeError):
            return None

    def _predict(self, predictor, bars_df, vix_df, up_to_idx: int) -> dict:
        """Run prediction using data up to (and including) up_to_idx."""
        if predictor is None:
            return {"expected_return": 0.0, "direction": "FLAT"}
        try:
            # Use only bars up to current date to prevent lookahead
            bars_subset = bars_df.iloc[:up_to_idx + 1].copy()
            return predictor.predict(bars_subset, vix_df)
        except Exception:
            return {"expected_return": 0.0, "direction": "FLAT"}

    def _close_position(self, portfolio: PortfolioState, symbol: str,
                        date, price: float, reason: str) -> None:
        pos = portfolio.positions.pop(symbol)
        exit_side = "SELL" if pos.direction == "LONG" else "BUY"
        fill_price, _ = simulate_fill(symbol, price, exit_side, stress_mult=self.stress_cost_mult)
        pos.exit_date = date
        pos.exit_price = fill_price
        pos.exit_reason = reason
        if pos.direction == "LONG":
            pos.pnl = pos.size * (fill_price - pos.entry_price)
            portfolio.cash += pos.size * fill_price
        else:
            pos.pnl = pos.size * (pos.entry_price - fill_price)
            portfolio.cash += pos.size * pos.entry_price + pos.pnl
        portfolio.closed_trades.append(pos)

    def _compute_equity(self, portfolio: PortfolioState, all_bars: dict, day) -> float:
        equity = portfolio.cash
        for sym, pos in portfolio.positions.items():
            price = self._get_price(all_bars, sym, day)
            if pos.direction == "LONG":
                equity += pos.size * price
            else:
                equity += pos.size * pos.entry_price + pos.size * (pos.entry_price - price)
        return equity

    def _get_price(self, all_bars: dict, symbol: str, day) -> float:
        bars = all_bars.get(symbol)
        if bars is None:
            return 0.0
        bar_dates = pd.to_datetime(bars["ts"]).dt.date
        day_mask = bar_dates == day
        if not day_mask.any():
            # Use most recent available price
            prior = bar_dates <= day
            if prior.any():
                return float(bars["close"][prior].iloc[-1])
            return 0.0
        return float(bars["close"][day_mask].iloc[-1])

    def _positions_as_dict(self, portfolio: PortfolioState, all_bars: dict, day) -> dict:
        """Convert positions to dict format expected by check_position_allowed."""
        result = {}
        for sym, pos in portfolio.positions.items():
            price = self._get_price(all_bars, sym, day)
            result[sym] = {
                "qty": pos.size,
                "current_price": price,
                "side": pos.direction,
            }
        return result

    def _compute_results(self, portfolio: PortfolioState, trade_dates: list,
                         exposure_log: list, max_positions_held: int) -> PortfolioResult:
        eq_df = pd.DataFrame(portfolio.equity_curve)
        final_equity = eq_df["equity"].iloc[-1] if not eq_df.empty else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100

        # Annualized return
        years = len(trade_dates) / 252
        ann_return = ((final_equity / self.initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

        # Sharpe
        daily_returns = eq_df["equity"].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                  if daily_returns.std() > 0 else 0.0)

        # Max drawdown
        peak = eq_df["equity"].cummax()
        drawdown = (eq_df["equity"] - peak) / peak
        max_dd = abs(drawdown.min()) * 100

        # Trade stats
        trades = portfolio.closed_trades
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        # Trades by symbol
        trades_by_sym = {}
        for t in trades:
            trades_by_sym[t.symbol] = trades_by_sym.get(t.symbol, 0) + 1

        # Exposure stats
        avg_exposure = np.mean(exposure_log) if exposure_log else 0
        max_exposure = max(exposure_log) if exposure_log else 0

        # Turnover
        total_volume = sum(abs(t.size * t.entry_price) for t in trades)
        turnover = total_volume / (self.initial_capital * max(years, 0.01))

        result = PortfolioResult(
            symbols=self.symbols,
            start_date=str(trade_dates[0]),
            end_date=str(trade_dates[-1]),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(ann_return, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            total_trades=n_trades,
            win_rate=round(win_rate, 3),
            profit_factor=round(profit_factor, 2),
            avg_exposure=round(avg_exposure, 3),
            max_exposure=round(max_exposure, 3),
            max_positions_held=max_positions_held,
            turnover=round(turnover, 1),
            trades_by_symbol=trades_by_sym,
            equity_curve=eq_df,
            trades=trades,
        )
        return result

    def _empty_result(self, start_date, end_date) -> PortfolioResult:
        return PortfolioResult(
            symbols=self.symbols, start_date=start_date or "",
            end_date=end_date or "", initial_capital=self.initial_capital,
            final_equity=self.initial_capital, total_return_pct=0,
            annualized_return_pct=0, sharpe_ratio=0, max_drawdown_pct=0,
            total_trades=0, win_rate=0, profit_factor=0,
            avg_exposure=0, max_exposure=0, max_positions_held=0,
            turnover=0, trades_by_symbol={}, equity_curve=pd.DataFrame(),
        )


def print_portfolio_report(result: PortfolioResult) -> None:
    """Print portfolio backtest results."""
    print("\n" + "=" * 70)
    print("  PORTFOLIO BACKTEST REPORT")
    print("=" * 70)
    print(f"  Symbols:     {', '.join(result.symbols)}")
    print(f"  Period:      {result.start_date} to {result.end_date}")
    print(f"  Capital:     ${result.initial_capital:,.0f} → ${result.final_equity:,.0f}")
    print(f"  Return:      {result.total_return_pct:+.2f}% (ann: {result.annualized_return_pct:+.2f}%)")
    print(f"  Sharpe:      {result.sharpe_ratio:.3f}")
    print(f"  Max DD:      {result.max_drawdown_pct:.2f}%")
    print(f"  Trades:      {result.total_trades} (WR: {result.win_rate:.1%}, PF: {result.profit_factor:.2f})")
    print(f"  Exposure:    avg={result.avg_exposure:.1%}, max={result.max_exposure:.1%}")
    print(f"  Max pos:     {result.max_positions_held}")
    print(f"  Turnover:    {result.turnover:.1f}x/yr")
    print()
    if result.trades_by_symbol:
        print("  Trades by symbol:")
        for sym, n in sorted(result.trades_by_symbol.items(), key=lambda x: -x[1]):
            print(f"    {sym:>8}: {n}")
    print("=" * 70)

    # Save summary JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "symbols": result.symbols,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "total_return_pct": result.total_return_pct,
        "annualized_return_pct": result.annualized_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "avg_exposure": result.avg_exposure,
        "max_exposure": result.max_exposure,
        "trades_by_symbol": result.trades_by_symbol,
    }
    path = os.path.join(OUTPUT_DIR, "portfolio_backtest_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {path}")


def main() -> None:
    from paper_trader import SYMBOL_GROUPS

    parser = argparse.ArgumentParser(description="Portfolio-level multi-symbol backtester.")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (overrides --group)")
    parser.add_argument("--group", type=str, default="swing",
                        choices=list(SYMBOL_GROUPS.keys()),
                        help="Symbol group (default: swing)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--model", default="swing", choices=["lstm", "swing"])
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--stress-cost-mult", type=float, default=1.0,
                        help="Cost multiplier for stress testing (default: 1.0)")

    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = SYMBOL_GROUPS.get(args.group, [])

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")
    model_dir = args.model_dir or DEFAULT_MODEL_DIR

    bt = PortfolioBacktester(
        symbols=symbols,
        adapter=adapter,
        fred_key=fred_key,
        model_dir=model_dir,
        initial_capital=args.capital,
        group=args.group,
        model_type=args.model,
        stress_cost_mult=args.stress_cost_mult,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    print_portfolio_report(result)


if __name__ == "__main__":
    main()
