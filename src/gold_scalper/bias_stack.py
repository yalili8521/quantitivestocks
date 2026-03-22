"""Multi-timeframe EMA bias stack.

多时间框架偏向堆栈：
- 5个时间框架：日线(D)、4小时(4H)、1小时(1H)、15分钟(15m)、5分钟(5m)
- 每个时间框架计算EMA，价格在EMA上方=看涨，下方=看跌
- 只有全部5个时间框架同向时才给出入场信号
- 4H数据从1H数据重采样得到（大多数API不直接提供4H K线）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.gold_scalper.config import GoldScalperConfig


@dataclass
class TimeframeBias:
    """Single timeframe bias result."""
    timeframe: str      # "1d", "4h", "1h", "15m", "5m"
    close: float        # latest close price
    ema_value: float    # latest EMA value
    bias: str           # "BULL" or "BEAR"

    @property
    def is_bullish(self) -> bool:
        return self.bias == "BULL"

    @property
    def is_bearish(self) -> bool:
        return self.bias == "BEAR"

    @property
    def icon(self) -> str:
        return "🟢" if self.is_bullish else "🔴"


class BiasStack:
    """Computes and evaluates multi-TF EMA bias alignment.

    偏向堆栈计算器：
    - 不负责获取数据（数据由引擎提供，保持可测试性）
    - 只做EMA计算和方向判断
    """

    # Canonical timeframe order (highest to lowest)
    TIMEFRAMES = ("1d", "4h", "1h", "15m", "5m")

    def __init__(self, config: GoldScalperConfig):
        self.ema_periods = config.bias_ema_periods

    def compute(self, bars: Dict[str, pd.DataFrame]) -> List[TimeframeBias]:
        """Compute EMA bias for each timeframe.

        Args:
            bars: Dict mapping timeframe string to OHLCV DataFrame.
                  Each DataFrame must have at least a 'close' column.
                  E.g. {"1d": df_daily, "4h": df_4h, ...}

        Returns:
            List of TimeframeBias, one per timeframe (high to low).
        """
        results = []
        for tf in self.TIMEFRAMES:
            if tf not in bars or bars[tf] is None or bars[tf].empty:
                # If data missing, treat as neutral (BEAR to be conservative)
                results.append(TimeframeBias(
                    timeframe=tf, close=0.0, ema_value=0.0, bias="BEAR"
                ))
                continue

            df = bars[tf]
            period = self.ema_periods.get(tf, 15)
            ema = self._compute_ema(df["close"], period)

            latest_close = float(df["close"].iloc[-1])
            latest_ema = float(ema.iloc[-1])
            bias = "BULL" if latest_close > latest_ema else "BEAR"

            results.append(TimeframeBias(
                timeframe=tf,
                close=latest_close,
                ema_value=latest_ema,
                bias=bias,
            ))

        return results

    def is_aligned(self, biases: List[TimeframeBias]) -> Optional[str]:
        """Check if all timeframes are aligned in one direction.

        全部对齐判断：
        - 全部看涨(BULL) → 返回 "LONG"
        - 全部看跌(BEAR) → 返回 "SHORT"
        - 混合 → 返回 None（不交易）

        Returns:
            "LONG", "SHORT", or None.
        """
        if not biases:
            return None

        all_bull = all(b.is_bullish for b in biases)
        all_bear = all(b.is_bearish for b in biases)

        if all_bull:
            return "LONG"
        elif all_bear:
            return "SHORT"
        return None

    def bias_summary(self, biases: List[TimeframeBias]) -> str:
        """Human-readable bias summary for alerts.

        Example: "D: BULL | 4H: BULL | 1H: BULL | 15m: BULL | 5m: BULL"
        """
        labels = {
            "1d": "Daily", "4h": "4H", "1h": "1H", "15m": "15m", "5m": "5m"
        }
        parts = []
        for b in biases:
            label = labels.get(b.timeframe, b.timeframe)
            parts.append(f"{label}: {b.bias}")
        return " | ".join(parts)

    def bias_icons(self, biases: List[TimeframeBias]) -> str:
        """Emoji icon string for dashboard/alerts.

        Example: "🟢 D  🟢 4H  🟢 1H  🟢 15m  🟢 5m"
        """
        labels = {
            "1d": "D", "4h": "4H", "1h": "1H", "15m": "15m", "5m": "5m"
        }
        parts = []
        for b in biases:
            label = labels.get(b.timeframe, b.timeframe)
            parts.append(f"{b.icon} {label}")
        return "  ".join(parts)

    @staticmethod
    def _compute_ema(series: pd.Series, period: int) -> pd.Series:
        """Compute EMA matching TradingView's ta.ema() behavior.

        TradingView uses the standard EMA formula:
        EMA = close × k + EMA_prev × (1 - k), where k = 2 / (period + 1)
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H bars to 4H bars.

        从1小时数据重采样到4小时数据：
        - open: 第一根1H的开盘价
        - high: 4根1H的最高价
        - low: 4根1H的最低价
        - close: 最后一根1H的收盘价
        - volume: 4根1H的成交量之和

        Args:
            df_1h: DataFrame with 'ts', 'open', 'high', 'low', 'close', 'volume' columns.

        Returns:
            DataFrame with 4H OHLCV bars.
        """
        if df_1h.empty:
            return df_1h

        df = df_1h.copy()
        if "ts" in df.columns:
            df = df.set_index("ts")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        resampled = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        resampled = resampled.reset_index()
        if "ts" not in resampled.columns and resampled.columns[0] != "ts":
            resampled = resampled.rename(columns={resampled.columns[0]: "ts"})

        return resampled
