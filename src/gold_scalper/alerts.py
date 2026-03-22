"""Discord webhook alerts — matches Pine Script alert format.

Discord警报引擎：
- 入场警报（做多/做空，合约数，止盈/止损水平，偏向堆栈）
- 每个TP命中警报（新止损水平，剩余合约数）
- 跑者出场警报（偏向翻转）
- 交易总结警报（赢/输，盈亏，持仓时间）
- 时段关闭警报（每日/周五）
- 超时警报（TP1未命中）
- 熔断警报（每日亏损限制触发）

消息格式与TradingView Pine Script完全一致。
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ScalperAlertEngine:
    """Discord webhook alerts for the gold scalper.

    Wraps the existing WebhookNotifier from src/alerts.py for delivery.
    Falls back to logging if no webhook URL configured.
    """

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url
        self._notifier = None

        if webhook_url:
            try:
                from src.alerts import WebhookNotifier
                self._notifier = WebhookNotifier(webhook_url)
            except ImportError:
                logger.warning("Could not import WebhookNotifier, alerts will be logged only")

    def _send(self, content: str) -> None:
        """Send alert via webhook or log."""
        if self._notifier:
            try:
                self._notifier.send_raw(content)
            except Exception as e:
                logger.error(f"Webhook send failed: {e}")
                logger.info(f"ALERT: {content[:200]}")
        else:
            logger.info(f"ALERT: {content[:500]}")

    def on_entry(
        self,
        direction: str,
        symbol: str,
        price: float,
        contracts: int,
        scale_mult: int,
        hard_stop_pips: float,
        tp_pips: List[float],
        tp_contracts: List[int],
        runner_qty: int,
        bias_summary: str,
        rsi: float,
    ) -> None:
        """Entry alert — matches Pine Script format.

        入场警报：
        🚀 BUY NOW — LONG / SELL NOW — SHORT
        包含：价格、合约数、止盈/止损水平、偏向堆栈、RSI
        """
        if direction == "LONG":
            emoji = "🚀"
            action = "BUY NOW — LONG"
            trigger = "Bullish Engulfing"
            tp_emoji = "🟢"
        else:
            emoji = "🚀"
            action = "SELL NOW — SHORT"
            trigger = "Bearish Engulfing"
            tp_emoji = "🔴"

        tp_lines = []
        for i, (pips, qty) in enumerate(zip(tp_pips, tp_contracts), 1):
            suffix = ""
            if i == 2:
                suffix = ", move SL to break even, RISK FREE"
            tp_lines.append(
                f"{tp_emoji} **TP{i}:** +{pips:.0f} pips — close {qty} contract(s){suffix}"
            )

        msg = (
            f"**{emoji}**\n"
            f"# {action}\n\n"
            f"**Ticker:** {symbol}\n"
            f"**Entry:** ${price:.2f}\n"
            f"**Contracts:** {contracts} ({scale_mult}x scale)\n\n"
            f"**── Levels ──**\n"
            f"🛑 **Stop Loss:** {hard_stop_pips:.0f} pips\n"
            + "\n".join(tp_lines) + "\n"
            f"🔀 **Runner:** {runner_qty} contract(s) — exit on bias flip, SL, or your call\n\n"
            f"**── Bias Stack ──**\n"
            f"{bias_summary}\n"
            f"**RSI (14):** {rsi:.1f}\n"
            f"**Trigger:** {trigger}\n"
        )

        self._send(msg)

    def on_tp_hit(
        self,
        tp_level: int,
        symbol: str,
        price: float,
        tp_pips: float,
        contracts_closed: int,
        remaining: int,
        new_stop: float,
        new_stop_label: str,
    ) -> None:
        """TP hit alert.

        止盈命中警报：
        🎯 TP1 HIT — +60 PIPS
        包含：新止损水平、剩余合约数
        """
        msg = (
            f"**🎯 TP{tp_level} HIT — +{tp_pips:.0f} PIPS**\n\n"
            f"**Ticker:** {symbol}\n"
            f"**Price:** ${price:.2f}\n"
            f"**TP Target:** +{tp_pips:.0f} pips\n"
            f"**New Stop:** ${new_stop:.2f} ({new_stop_label})\n"
            f"**Remaining:** {remaining} contract(s) open\n"
        )
        if tp_level >= 2:
            msg += "\n*RISK FREE*\n"

        self._send(msg)

    def on_runner_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> None:
        """Runner exit alert.

        跑者出场警报：
        🏃 RUNNER CLOSED — BIAS FLIPPED
        """
        msg = (
            f"**🏃 RUNNER CLOSED — {reason.upper()}**\n\n"
            f"Multi-TF bias no longer aligned. Remaining contract(s) exited.\n\n"
            f"**Entry:** ${entry_price:.2f}\n"
            f"**Exit:** ${exit_price:.2f}\n"
            f"**Runner P&L:** ${pnl:.2f}\n"
        )
        self._send(msg)

    def on_trade_summary(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        total_pnl: float,
        duration_minutes: float,
        exit_reason: str,
        tp_log: List[str],
        equity: float,
        daily_wins: int,
        daily_losses: int,
        daily_trades: int,
    ) -> None:
        """Trade summary alert (win or loss).

        交易总结警报：
        ✅ TRADE CLOSED — WIN / ❌ TRADE CLOSED — LOSS
        包含：入场/出场价、盈亏、持仓时间、TP分解、账户余额
        """
        pip_move = abs(exit_price - entry_price) / 0.10  # gold pip
        time_label = (
            f"{int(duration_minutes // 60)}h{int(duration_minutes % 60)}m"
            if duration_minutes >= 60
            else f"{int(duration_minutes)}min"
        )

        if total_pnl >= 0:
            emoji = "✅"
            result = "WIN"
            pnl_str = f"+${total_pnl:.2f}"
        else:
            emoji = "❌"
            result = "LOSS"
            pnl_str = f"-${abs(total_pnl):.2f}"

        tp_breakdown = "\n".join(tp_log) if tp_log else "No TPs hit"

        msg = (
            f"**{emoji} TRADE CLOSED — {result}**\n\n"
            f"**Direction:** {direction}\n"
            f"**Entry:** ${entry_price:.2f}\n"
            f"**Exit:** ${exit_price:.2f}\n"
            f"**Move:** {pip_move:.1f} pips\n"
            f"**Duration:** {time_label}\n"
            f"**Exit Reason:** {exit_reason}\n\n"
            f"**── TP Breakdown ──**\n"
            f"{tp_breakdown}\n\n"
            f"**TOTAL P&L: {pnl_str}**\n"
            f"**Equity:** ${equity:.2f} | "
            f"**Daily W/L:** {daily_wins}W / {daily_losses}L "
            f"({daily_trades} trades)\n"
        )
        self._send(msg)

    def on_session_close(self, price: float, reason: str) -> None:
        """Session close alert.

        时段关闭警报：
        🕓 POSITIONS CLOSED — SESSION END
        """
        msg = (
            f"**🕓 POSITIONS CLOSED — {reason.upper()}**\n\n"
            f"CLOSE ALL POSITIONS\n\n"
            f"**Closing Price:** ${price:.2f}\n"
        )
        self._send(msg)

    def on_timeout(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        timeout_minutes: int,
    ) -> None:
        """Timeout alert — TP1 not hit.

        超时警报：
        ⏳ TRADE CLOSED — TIMEOUT
        """
        msg = (
            f"**⏳ TRADE CLOSED — TIMEOUT**\n\n"
            f"TP1 not hit within {timeout_minutes} minutes. Position closed.\n\n"
            f"**Entry:** ${entry_price:.2f}\n"
            f"**Exit:** ${exit_price:.2f}\n"
        )
        self._send(msg)

    def on_circuit_breaker(self, daily_pnl: float, limit: float) -> None:
        """Circuit breaker alert.

        熔断警报：
        ⛔ DAILY LOSS LIMIT HIT — NO NEW TRADES
        """
        msg = (
            f"**⛔ DAILY LOSS LIMIT HIT**\n\n"
            f"Daily P&L: ${daily_pnl:.2f} (limit: ${limit:.2f})\n"
            f"No new trades will be opened for the rest of the day.\n"
        )
        self._send(msg)
