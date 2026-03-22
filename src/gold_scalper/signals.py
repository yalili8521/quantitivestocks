"""Entry signal generator — RSI + engulfing candle confirmation.

入场信号生成器：
- RSI(14) 在5分钟图上：> 55 做多，< 35 做空
- 吞没K线确认：上一根完成的5分钟K线必须是吞没形态
- 两个条件都满足 + 偏向对齐 = 有效入场信号
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from src.gold_scalper.config import GoldScalperConfig


@dataclass
class SignalResult:
    """Entry signal evaluation result."""
    direction: Optional[str]      # "LONG", "SHORT", or None
    bias_alignment: str           # "LONG", "SHORT", or "MIXED"
    rsi_value: float              # current RSI(14) on 5m
    engulfing: Optional[str]      # "BULLISH", "BEARISH", or None
    reason_rejected: str          # "" if valid signal, else why rejected

    @property
    def is_valid(self) -> bool:
        return self.direction is not None


class EntrySignalGenerator:
    """Generates entry signals from RSI + engulfing + bias alignment.

    信号生成逻辑：
    1. 偏向堆栈必须全部对齐（由BiasStack提供）
    2. RSI必须在阈值之上/之下
    3. 前一根K线必须是吞没形态（可选）
    """

    def __init__(self, config: GoldScalperConfig):
        self.rsi_period = config.rsi_period
        self.rsi_long_threshold = config.rsi_long_threshold
        self.rsi_short_threshold = config.rsi_short_threshold
        self.require_engulfing = config.require_engulfing

    def compute_rsi(self, closes: pd.Series) -> float:
        """Compute RSI(14) using Wilder smoothing (matches TradingView).

        Wilder平滑RSI计算：
        - 与TradingView的ta.rsi()完全一致
        - 使用指数移动平均而非简单移动平均
        """
        if len(closes) < self.rsi_period + 1:
            return 50.0  # neutral if insufficient data

        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder smoothing (same as EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return float(rsi.iloc[-1])

    def detect_engulfing(self, bars_5m: pd.DataFrame) -> Optional[str]:
        """Detect engulfing pattern on the PREVIOUS completed candle.

        吞没形态检测（使用前一根已完成的K线，避免重绘）：
        - 看涨吞没：前一根K线收盘 > 开盘，且吞没前前一根K线的实体
        - 看跌吞没：前一根K线收盘 < 开盘，且吞没前前一根K线的实体

        Uses bars[-2] and bars[-3] (the two most recent COMPLETED candles),
        because bars[-1] is the current forming candle.
        """
        if len(bars_5m) < 3:
            return None

        # Previous completed candle (bars[-2]) and the one before it (bars[-3])
        prev_open = float(bars_5m["open"].iloc[-2])
        prev_close = float(bars_5m["close"].iloc[-2])
        prev2_open = float(bars_5m["open"].iloc[-3])
        prev2_close = float(bars_5m["close"].iloc[-3])

        # Bullish engulfing: prev candle is green, engulfs the one before
        if (prev_close > prev_open and
                prev_open <= prev2_close and
                prev_close >= prev2_open):
            return "BULLISH"

        # Bearish engulfing: prev candle is red, engulfs the one before
        if (prev_close < prev_open and
                prev_open >= prev2_close and
                prev_close <= prev2_open):
            return "BEARISH"

        return None

    def evaluate(
        self,
        bias_direction: Optional[str],
        bars_5m: pd.DataFrame,
    ) -> SignalResult:
        """Full entry signal evaluation.

        完整入场信号评估：
        1. 检查偏向对齐（必须有方向）
        2. 计算RSI并检查阈值
        3. 检查吞没形态（如果启用）
        4. 全部通过 → 返回有效信号

        Args:
            bias_direction: "LONG", "SHORT", or None from BiasStack.
            bars_5m: 5-minute OHLCV DataFrame.

        Returns:
            SignalResult with direction or rejection reason.
        """
        # Default RSI
        rsi = self.compute_rsi(bars_5m["close"]) if not bars_5m.empty else 50.0
        engulfing = self.detect_engulfing(bars_5m)

        # Gate 1: Bias alignment
        if bias_direction is None:
            return SignalResult(
                direction=None,
                bias_alignment="MIXED",
                rsi_value=rsi,
                engulfing=engulfing,
                reason_rejected="Bias not aligned across all timeframes",
            )

        # Gate 2: RSI threshold
        if bias_direction == "LONG" and rsi <= self.rsi_long_threshold:
            return SignalResult(
                direction=None,
                bias_alignment=bias_direction,
                rsi_value=rsi,
                engulfing=engulfing,
                reason_rejected=f"RSI {rsi:.1f} <= {self.rsi_long_threshold} (need above for LONG)",
            )
        if bias_direction == "SHORT" and rsi >= self.rsi_short_threshold:
            return SignalResult(
                direction=None,
                bias_alignment=bias_direction,
                rsi_value=rsi,
                engulfing=engulfing,
                reason_rejected=f"RSI {rsi:.1f} >= {self.rsi_short_threshold} (need below for SHORT)",
            )

        # Gate 3: Engulfing confirmation
        if self.require_engulfing:
            if bias_direction == "LONG" and engulfing != "BULLISH":
                return SignalResult(
                    direction=None,
                    bias_alignment=bias_direction,
                    rsi_value=rsi,
                    engulfing=engulfing,
                    reason_rejected="No bullish engulfing candle",
                )
            if bias_direction == "SHORT" and engulfing != "BEARISH":
                return SignalResult(
                    direction=None,
                    bias_alignment=bias_direction,
                    rsi_value=rsi,
                    engulfing=engulfing,
                    reason_rejected="No bearish engulfing candle",
                )

        # All gates passed
        return SignalResult(
            direction=bias_direction,
            bias_alignment=bias_direction,
            rsi_value=rsi,
            engulfing=engulfing,
            reason_rejected="",
        )
