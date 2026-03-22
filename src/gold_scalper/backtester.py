"""Gold Scalper Backtester — offline replay on historical 5m bars.

回测器：
- 在历史5分钟K线上重放策略逻辑
- 模拟多时间框架偏向（从5分钟数据重采样到15m/1h/4h/日线）
- 输出交易日志CSV + 统计摘要（胜率、盈亏比、最大回撤等）
- 用于与TradingView回测结果对比验证
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from src.gold_scalper.bias_stack import BiasStack
from src.gold_scalper.config import GoldScalperConfig, load_config
from src.gold_scalper.position_manager import (
    GoldPosition, PositionManager, TradeAction,
)
from src.gold_scalper.session_filter import SessionFilter
from src.gold_scalper.signals import EntrySignalGenerator
from src.gold_scalper.sizing import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a completed backtest trade."""
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    pnl_pips: float
    exit_reason: str
    tps_hit: int
    duration_minutes: float


@dataclass
class BacktestResult:
    """Summary statistics from a backtest run."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_duration_min: float = 0.0
    win_rate: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"GOLD SCALPER BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Win Rate:        {self.win_rate:.1%} ({self.wins}W / {self.losses}L)\n"
            f"Total P&L:       ${self.total_pnl:.2f}\n"
            f"Profit Factor:   {self.profit_factor:.2f}\n"
            f"Avg Win:         ${self.avg_win:.2f}\n"
            f"Avg Loss:        ${self.avg_loss:.2f}\n"
            f"Max Drawdown:    ${self.max_drawdown:.2f}\n"
            f"Avg Duration:    {self.avg_duration_min:.0f} min\n"
            f"{'='*60}\n"
        )


class GoldScalperBacktester:
    """Replays the gold scalper strategy on historical data.

    回测流程：
    1. 获取日期范围内的5分钟K线
    2. 重采样到15m/1h/4h/日线
    3. 逐根5分钟K线遍历，模拟策略逻辑
    4. 记录每笔交易的入场/出场/盈亏
    """

    def __init__(self, config: GoldScalperConfig):
        self.config = config
        self.tz = ZoneInfo(config.timezone)
        self.bias_stack = BiasStack(config)
        self.signal_gen = EntrySignalGenerator(config)
        self.position_mgr = PositionManager(config)
        self.sizer = PositionSizer(config)
        self.session = SessionFilter(config)

    def run(
        self,
        start: str,
        end: Optional[str] = None,
        initial_equity: float = 5000.0,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD). Defaults to today.
            initial_equity: Starting equity.

        Returns:
            BacktestResult with trade log and statistics.
        """
        logger.info(f"Backtesting {self.config.symbol} from {start} to {end or 'now'}")

        # Fetch historical 5m data
        bars_5m = self._fetch_historical(start, end)
        if bars_5m.empty:
            logger.error("No historical data available")
            return BacktestResult()

        logger.info(f"Loaded {len(bars_5m)} 5-min bars")

        # Resample to higher timeframes
        bars_15m = self._resample(bars_5m, "15min")
        bars_1h = self._resample(bars_5m, "1h")
        bars_4h = self._resample(bars_5m, "4h")
        bars_1d = self._resample(bars_5m, "1D")

        logger.info(
            f"Resampled: 15m={len(bars_15m)}, 1h={len(bars_1h)}, "
            f"4h={len(bars_4h)}, 1d={len(bars_1d)} bars"
        )

        # Run simulation
        return self._simulate(
            bars_5m, bars_15m, bars_1h, bars_4h, bars_1d, initial_equity
        )

    def _simulate(
        self,
        bars_5m: pd.DataFrame,
        bars_15m: pd.DataFrame,
        bars_1h: pd.DataFrame,
        bars_4h: pd.DataFrame,
        bars_1d: pd.DataFrame,
        initial_equity: float,
    ) -> BacktestResult:
        """Walk through 5m bars and simulate the strategy.

        逐根K线遍历模拟：
        - 每根5分钟K线检查入场/出场条件
        - 使用当前K线之前的数据计算指标（避免前瞻偏差）
        """
        result = BacktestResult()
        equity = initial_equity
        cumulative_pnl = 0.0
        peak_equity = initial_equity
        position: Optional[GoldPosition] = None
        daily_pnl = 0.0
        current_day = None

        # Need at least 30 bars for indicators
        min_bars = 30

        for i in range(min_bars, len(bars_5m)):
            row = bars_5m.iloc[i]
            ts = pd.Timestamp(row["ts"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            now = ts.astimezone(self.tz)
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # Reset daily PnL
            day = now.date()
            if current_day != day:
                daily_pnl = 0.0
                current_day = day

            # Session check
            if not self.session.can_trade(now):
                # Force close if session ending
                if position and not position.is_flat:
                    should_close, reason = self.session.should_force_close(now)
                    if should_close:
                        pips = position.pips_from_entry(price, self.config.pip_value)
                        pnl = pips * position.remaining_contracts
                        equity += pnl
                        cumulative_pnl += pnl
                        daily_pnl += pnl

                        result.trades.append(BacktestTrade(
                            entry_time=position.entry_time.isoformat(),
                            exit_time=now.isoformat(),
                            direction=position.direction,
                            entry_price=position.entry_price,
                            exit_price=price,
                            contracts=position.total_contracts,
                            pnl=pnl,
                            pnl_pips=pips,
                            exit_reason=reason,
                            tps_hit=sum([
                                position.tp1_hit, position.tp2_hit,
                                position.tp3_hit, position.tp4_hit
                            ]),
                            duration_minutes=(now - position.entry_time).total_seconds() / 60,
                        ))
                        position = None
                continue

            # Get lookback windows for each TF (up to current bar, no lookahead)
            tf_bars = self._get_lookback_bars(
                i, bars_5m, bars_15m, bars_1h, bars_4h, bars_1d, ts
            )

            # Compute bias
            biases = self.bias_stack.compute(tf_bars)
            direction = self.bias_stack.is_aligned(biases)

            # ── EXIT CHECK ──
            if position and not position.is_flat:
                # Use high/low for TP/SL checks (intrabar)
                # Check if stop was hit (use low for long, high for short)
                check_price = low if position.is_long else high
                actions_sl = self.position_mgr.update(
                    position, check_price, now, direction
                )

                # Check if TP was hit (use high for long, low for short)
                check_price_tp = high if position.is_long else low
                actions_tp = self.position_mgr.update(
                    position, check_price_tp, now, direction
                )

                # Combine actions (prefer TP over SL if both on same bar)
                actions = actions_tp if actions_tp else actions_sl

                for action in actions:
                    pnl = action.pnl_pips * action.contracts
                    position.realized_pnl += pnl
                    equity += pnl
                    cumulative_pnl += pnl
                    daily_pnl += pnl

                if position.is_flat or position.remaining_contracts <= 0:
                    # Position fully closed
                    duration = (now - position.entry_time).total_seconds() / 60
                    total_pnl = position.realized_pnl
                    exit_reason = actions[-1].reason if actions else "Unknown"

                    result.trades.append(BacktestTrade(
                        entry_time=position.entry_time.isoformat(),
                        exit_time=now.isoformat(),
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=price,
                        contracts=position.total_contracts,
                        pnl=total_pnl,
                        pnl_pips=position.pips_from_entry(price, self.config.pip_value),
                        exit_reason=exit_reason,
                        tps_hit=sum([
                            position.tp1_hit, position.tp2_hit,
                            position.tp3_hit, position.tp4_hit,
                        ]),
                        duration_minutes=duration,
                    ))
                    position = None

                continue

            # ── ENTRY CHECK ──
            # Circuit breaker
            if daily_pnl <= self.config.daily_loss_limit:
                continue

            # Avoid entry near close
            avoid, _ = self.session.should_avoid_entry(now)
            if avoid:
                continue

            # Signal
            signal = self.signal_gen.evaluate(direction, bars_5m.iloc[:i+1])
            if not signal.is_valid:
                continue

            # Size (with margin enforcement)
            sizing = self.sizer.compute(
                cumulative_pnl, equity=equity, price=price,
                direction=signal.direction,
            )
            if sizing.total_contracts < 1:
                continue  # insufficient margin

            # Enter
            position = self.position_mgr.create_position(
                direction=signal.direction,
                entry_price=price,
                entry_time=now,
                total_contracts=sizing.total_contracts,
                tp_splits=sizing.tp_splits,
                runner_qty=sizing.runner_qty,
            )

        # ── COMPUTE STATISTICS ──
        # Rebuild equity curve and max drawdown from trade sequence
        eq = initial_equity
        peak_eq = initial_equity
        for t in result.trades:
            eq += t.pnl
            result.equity_curve.append(eq)
            peak_eq = max(peak_eq, eq)
            dd = peak_eq - eq
            result.max_drawdown = max(result.max_drawdown, dd)

        result.total_trades = len(result.trades)
        if result.total_trades > 0:
            wins = [t for t in result.trades if t.pnl >= 0]
            losses = [t for t in result.trades if t.pnl < 0]
            result.wins = len(wins)
            result.losses = len(losses)
            result.total_pnl = sum(t.pnl for t in result.trades)
            result.win_rate = result.wins / result.total_trades

            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
            result.profit_factor = gross_profit / gross_loss

            result.avg_win = gross_profit / len(wins) if wins else 0
            result.avg_loss = gross_loss / len(losses) if losses else 0
            result.avg_duration_min = np.mean([t.duration_minutes for t in result.trades])

        return result

    def _fetch_historical(
        self, start: str, end: Optional[str]
    ) -> pd.DataFrame:
        """Fetch historical 5m bars from Yahoo Finance.

        获取历史5分钟K线数据（通过yfinance）。
        Note: Yahoo limits 5m data to last 60 days. For ranges within
        60 days, uses period='60d'. For longer ranges, falls back to
        start/end with daily data warning.
        """
        try:
            import yfinance as yf
            from datetime import datetime as dt

            ticker = yf.Ticker(self.config.symbol)

            # Yahoo 5m data only available for last 60 days via period=
            # Try period-based fetch first (most reliable)
            start_date = dt.strptime(start, "%Y-%m-%d")
            days_ago = (dt.now() - start_date).days

            if days_ago <= 60:
                # Use period for recent data (Yahoo's preferred method for 5m)
                period = f"{min(days_ago + 1, 60)}d"
                df = ticker.history(period=period, interval="5m")
            else:
                # For older data, try start/end (may fail for 5m)
                logger.warning(
                    f"Requested {days_ago}d ago — Yahoo limits 5m to 60d. "
                    f"Fetching max available."
                )
                df = ticker.history(period="60d", interval="5m")

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            # Normalize column names
            df = df.rename(columns={
                "Datetime": "ts", "Date": "ts",
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            cols = ["ts", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]

            # Filter to requested date range if start/end specified
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                if start:
                    df = df[df["ts"] >= start]
                if end:
                    df = df[df["ts"] <= end]

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    def _resample(self, bars_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample 5m bars to higher timeframe."""
        if bars_5m.empty:
            return pd.DataFrame()

        df = bars_5m.copy()
        if "ts" in df.columns:
            df = df.set_index("ts")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled.reset_index().rename(columns={resampled.index.name or "index": "ts"})

    def _get_lookback_bars(
        self,
        idx_5m: int,
        bars_5m: pd.DataFrame,
        bars_15m: pd.DataFrame,
        bars_1h: pd.DataFrame,
        bars_4h: pd.DataFrame,
        bars_1d: pd.DataFrame,
        current_ts: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Get lookback data for each TF up to current timestamp (no lookahead).

        获取到当前时间为止的各时间框架数据（避免前瞻偏差）。
        """
        result = {}

        # 5m: use raw slice
        result["5m"] = bars_5m.iloc[:idx_5m + 1].tail(200)

        # Higher TFs: filter by timestamp
        for tf, df in [("15m", bars_15m), ("1h", bars_1h),
                       ("4h", bars_4h), ("1d", bars_1d)]:
            if df.empty:
                continue
            ts_col = df["ts"]
            if ts_col.dtype == "object":
                ts_col = pd.to_datetime(ts_col)
            mask = ts_col <= current_ts
            filtered = df[mask].tail(50)
            if not filtered.empty:
                result[tf] = filtered

        return result

    def save_trades_csv(self, result: BacktestResult, path: str) -> None:
        """Save trade log to CSV.

        保存交易日志到CSV文件。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time", "exit_time", "direction", "entry_price",
                "exit_price", "contracts", "pnl", "pnl_pips",
                "exit_reason", "tps_hit", "duration_min",
            ])
            for t in result.trades:
                writer.writerow([
                    t.entry_time, t.exit_time, t.direction,
                    f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
                    t.contracts, f"{t.pnl:.2f}", f"{t.pnl_pips:.1f}",
                    t.exit_reason, t.tps_hit, f"{t.duration_minutes:.0f}",
                ])
        logger.info(f"Trade log saved to {path}")


def main():
    """CLI entry point for gold-backtest command.

    命令行入口：
    python main.py gold-backtest --start 2025-01-01
    python main.py gold-backtest --start 2025-01-01 --end 2025-03-01
    """
    parser = argparse.ArgumentParser(
        description="Gold Scalper Backtester — TC baby V1.0"
    )
    parser.add_argument(
        "--start", type=str, required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD, defaults to today)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to gold_scalper.json"
    )
    parser.add_argument(
        "--initial-equity", type=float, default=5000.0,
        help="Starting equity"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path for trade log"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    bt = GoldScalperBacktester(config)
    result = bt.run(args.start, args.end, args.initial_equity)

    print(result)

    # Save trade log
    if args.output:
        bt.save_trades_csv(result, args.output)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        default_path = os.path.join(
            project_root, "data", "output",
            f"gold_backtest_{args.start}.csv"
        )
        bt.save_trades_csv(result, default_path)


if __name__ == "__main__":
    main()
