"""Webhook executor — routes TradingView signals to IBKR.

接收TradingView webhook信号并路由到IBKR执行：
- 管理仓位状态（入场价、合约数、TP追踪）
- 应用风险控制（每日亏损限额、最大仓位、冷却期）
- 通过Discord发送警报

Signal flow: TradingView → webhook_server → WebhookExecutor → IBKRGoldBroker
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.gold_scalper.alerts import ScalperAlertEngine
from src.gold_scalper.config import GoldScalperConfig

logger = logging.getLogger(__name__)

PT = ZoneInfo("America/Los_Angeles")


@dataclass
class WebhookPosition:
    """Tracks an open position initiated by TradingView signal."""
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    total_contracts: int
    remaining_contracts: int
    tp_hits: List[int] = field(default_factory=list)  # [1, 2, ...] as TPs fire
    realized_pnl: float = 0.0


@dataclass
class WebhookSignal:
    """Parsed TradingView webhook signal."""
    action: str                 # ENTRY, TP_HIT, EXIT, SESSION_CLOSE
    direction: Optional[str]    # LONG, SHORT (for ENTRY)
    price: float
    contracts: int
    tp_level: Optional[int]     # 1-4 (for TP_HIT)
    reason: Optional[str]       # exit reason (for EXIT)
    raw: Dict                   # original JSON payload


class WebhookExecutor:
    """Routes TradingView signals to IBKR with risk controls.

    TradingView信号执行器：
    - 接收解析后的webhook信号
    - 检查风险控制（每日亏损限额、冷却期、最大仓位）
    - 通过IBKRGoldBroker下单
    - 跟踪仓位状态和每日盈亏
    """

    def __init__(
        self,
        broker,  # IBKRGoldBroker or PaperGoldBroker
        config: GoldScalperConfig,
        alerts: Optional[ScalperAlertEngine] = None,
    ):
        self.broker = broker
        self.config = config
        self.alerts = alerts

        # State
        self.position: Optional[WebhookPosition] = None
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_day: Optional[int] = None
        self._circuit_breaker_fired: bool = False
        self._cooldown_until: Optional[datetime] = None

        # State persistence — separate folder from signal-mode bot
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        state_dir = os.path.join(project_root, "outputs", "paper_state", "webhook")
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "webhook_executor_state.json")
        self._load_state()

    def parse_signal(self, payload: Dict) -> WebhookSignal:
        """Parse raw webhook JSON into a WebhookSignal.

        Handles two formats:
        1. Native: {"action": "ENTRY", "direction": "SHORT", ...}
        2. TradingView: {"action": "buy/sell", "comment": "Short Entry", ...}
           - comment determines signal type (Entry, TP1-4, Hard Stop, Session Close)
           - action (buy/sell) + comment determine direction
        """
        raw_action = payload.get("action", "").strip().lower()
        comment = payload.get("comment", "").strip()
        comment_upper = comment.upper()
        price = float(payload.get("price", 0))
        contracts = int(payload.get("contracts", 0))

        # --- Native format (action is already ENTRY/TP_HIT/EXIT/SESSION_CLOSE) ---
        if raw_action in ("entry", "tp_hit", "exit", "session_close"):
            return WebhookSignal(
                action=raw_action.upper(),
                direction=payload.get("direction", "").upper() or None,
                price=price,
                contracts=contracts,
                tp_level=int(payload.get("tp_level", 0)) or None,
                reason=payload.get("reason"),
                raw=payload,
            )

        # --- TradingView format (action is "buy" or "sell") ---
        is_buy = raw_action == "buy"

        # Determine signal type from comment
        action = "ENTRY"
        direction = None
        tp_level = None
        reason = None

        if "ENTRY" in comment_upper or "ENTER" in comment_upper:
            action = "ENTRY"
            # "Short Entry" → sell-to-open → SHORT
            # "Long Entry" → buy-to-open → LONG
            if "SHORT" in comment_upper:
                direction = "SHORT"
            elif "LONG" in comment_upper:
                direction = "LONG"
            else:
                # Fallback: buy=LONG, sell=SHORT
                direction = "LONG" if is_buy else "SHORT"

        elif "TP4" in comment_upper or "TP 4" in comment_upper:
            action = "TP_HIT"
            tp_level = 4
        elif "TP3" in comment_upper or "TP 3" in comment_upper:
            action = "TP_HIT"
            tp_level = 3
        elif "TP2" in comment_upper or "TP 2" in comment_upper:
            action = "TP_HIT"
            tp_level = 2
        elif "TP1" in comment_upper or "TP 1" in comment_upper:
            action = "TP_HIT"
            tp_level = 1
        elif "RUNNER" in comment_upper:
            action = "EXIT"
            reason = "Runner Exit"
        elif "STOP" in comment_upper:
            action = "EXIT"
            reason = "Hard Stop Hit"
        elif "SESSION" in comment_upper or "CLOSE ALL" in comment_upper:
            action = "SESSION_CLOSE"
        elif "EXIT" in comment_upper or "CLOSE" in comment_upper:
            action = "EXIT"
            reason = comment or "TradingView exit"
        else:
            # Unknown comment — log it and treat as entry if no position,
            # exit if we have a position
            logger.warning(
                "[WEBHOOK] Unknown comment '%s' with action '%s' — "
                "treating as raw action",
                comment, raw_action,
            )
            if self.position and self.position.remaining_contracts > 0:
                action = "EXIT"
                reason = comment or "Unknown signal"
            else:
                action = "ENTRY"
                direction = "LONG" if is_buy else "SHORT"

        logger.info(
            "[WEBHOOK] Parsed TV signal: action=%s dir=%s tp=%s reason=%s "
            "(raw: action=%s comment='%s')",
            action, direction, tp_level, reason, raw_action, comment,
        )

        return WebhookSignal(
            action=action,
            direction=direction,
            price=price,
            contracts=contracts,
            tp_level=tp_level,
            reason=reason,
            raw=payload,
        )

    def execute(self, signal: WebhookSignal) -> Dict:
        """Execute a parsed signal. Returns result dict.

        执行解析后的信号：
        1. 每日重置检查
        2. 根据信号类型分发到对应处理器
        3. 返回执行结果
        """
        now = datetime.now(PT)
        self._reset_daily_if_needed(now)

        result = {"status": "error", "message": "Unknown action"}

        if signal.action == "ENTRY":
            result = self._handle_entry(signal, now)
        elif signal.action == "TP_HIT":
            result = self._handle_tp_hit(signal, now)
        elif signal.action == "EXIT":
            result = self._handle_exit(signal, now)
        elif signal.action == "SESSION_CLOSE":
            result = self._handle_session_close(signal, now)
        else:
            result = {"status": "error", "message": f"Unknown action: {signal.action}"}

        self._save_state()
        return result

    def _handle_entry(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle ENTRY signal from TradingView."""

        # Risk checks
        if self._circuit_breaker_fired:
            msg = f"Circuit breaker active (daily PnL ${self.daily_pnl:.2f})"
            logger.warning("[WEBHOOK] %s — rejecting entry", msg)
            return {"status": "rejected", "message": msg}

        if self.daily_pnl <= self.config.daily_loss_limit:
            self._circuit_breaker_fired = True
            msg = f"Daily loss limit hit: ${self.daily_pnl:.2f}"
            logger.warning("[WEBHOOK] %s", msg)
            return {"status": "rejected", "message": msg}

        if self._cooldown_until and now < self._cooldown_until:
            msg = f"Cooldown until {self._cooldown_until.strftime('%H:%M:%S')}"
            logger.info("[WEBHOOK] %s — rejecting entry", msg)
            return {"status": "rejected", "message": msg}

        if self.position and self.position.remaining_contracts > 0:
            msg = "Already in position — rejecting duplicate entry"
            logger.warning("[WEBHOOK] %s", msg)
            return {"status": "rejected", "message": msg}

        # Cap contracts at max scale
        contracts = min(
            signal.contracts,
            self.config.base_contracts * self.config.max_scale_mult,
        )

        if contracts < 1:
            return {"status": "rejected", "message": "Zero contracts"}

        # Place order
        side = "BUY" if signal.direction == "LONG" else "SELL"
        order_id = self.broker.place_market_order(
            self.config.symbol, contracts, side
        )

        if not order_id:
            return {"status": "error", "message": "Order placement failed"}

        # Create position
        self.position = WebhookPosition(
            direction=signal.direction,
            entry_price=signal.price,
            entry_time=now,
            total_contracts=contracts,
            remaining_contracts=contracts,
        )

        logger.info(
            "[WEBHOOK] ENTRY %s %d ct @ $%.2f — order %s",
            signal.direction, contracts, signal.price, order_id,
        )

        # Discord alert
        if self.alerts:
            self.alerts.on_entry(
                direction=signal.direction,
                symbol=self.config.symbol,
                price=signal.price,
                contracts=contracts,
                scale_mult=contracts // self.config.base_contracts,
                hard_stop_pips=self.config.hard_stop_pips,
                tp_pips=[tp[0] for tp in self.config.tp_levels],
                tp_contracts=[tp[1] for tp in self.config.tp_levels],
                runner_qty=self.config.runner_contracts,
                bias_summary="TradingView signal",
                rsi=0,
            )

        return {
            "status": "ok",
            "message": f"{signal.direction} {contracts}ct @ ${signal.price:.2f}",
            "order_id": order_id,
        }

    def _handle_tp_hit(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle TP_HIT signal — partial close."""
        if not self.position or self.position.remaining_contracts <= 0:
            return {"status": "rejected", "message": "No position to close"}

        contracts = min(signal.contracts, self.position.remaining_contracts)
        if contracts < 1:
            return {"status": "rejected", "message": "Zero contracts to close"}

        # Close partial
        order_id = self.broker.close_partial(
            self.config.symbol, contracts,
            f"TP{signal.tp_level} (TradingView)",
        )

        if not order_id:
            return {"status": "error", "message": "Partial close failed"}

        self.position.remaining_contracts -= contracts
        if signal.tp_level:
            self.position.tp_hits.append(signal.tp_level)

        # Estimate P&L for this partial
        pip_val = self.config.pip_value
        if self.position.direction == "LONG":
            pips = (signal.price - self.position.entry_price) / pip_val
        else:
            pips = (self.position.entry_price - signal.price) / pip_val
        partial_pnl = pips * contracts
        self.position.realized_pnl += partial_pnl

        logger.info(
            "[WEBHOOK] TP%s: closed %d ct @ $%.2f — PnL $%.0f — remaining %d ct",
            signal.tp_level, contracts, signal.price, partial_pnl,
            self.position.remaining_contracts,
        )

        # Check if position fully closed
        if self.position.remaining_contracts <= 0:
            self._on_position_closed(now)

        return {
            "status": "ok",
            "message": f"TP{signal.tp_level}: {contracts}ct closed",
            "order_id": order_id,
        }

    def _handle_exit(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle EXIT signal — close all remaining."""
        if not self.position or self.position.remaining_contracts <= 0:
            return {"status": "rejected", "message": "No position to close"}

        reason = signal.reason or "TradingView exit"
        contracts = self.position.remaining_contracts

        order_id = self.broker.close_all(
            self.config.symbol, reason,
        )

        if not order_id:
            return {"status": "error", "message": "Close all failed"}

        # Estimate final P&L
        pip_val = self.config.pip_value
        if self.position.direction == "LONG":
            pips = (signal.price - self.position.entry_price) / pip_val
        else:
            pips = (self.position.entry_price - signal.price) / pip_val
        exit_pnl = pips * contracts
        self.position.realized_pnl += exit_pnl
        self.position.remaining_contracts = 0

        logger.info(
            "[WEBHOOK] EXIT: %s — %d ct @ $%.2f — PnL $%.0f",
            reason, contracts, signal.price, exit_pnl,
        )

        self._on_position_closed(now)

        return {
            "status": "ok",
            "message": f"Closed {contracts}ct — {reason}",
            "order_id": order_id,
        }

    def _handle_session_close(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle SESSION_CLOSE — close everything."""
        if not self.position or self.position.remaining_contracts <= 0:
            return {"status": "ok", "message": "No position — session close ignored"}

        signal.reason = "Session Close (TradingView)"
        return self._handle_exit(signal, now)

    def _on_position_closed(self, now: datetime) -> None:
        """Update stats after position fully closed."""
        if not self.position:
            return

        total_pnl = self.position.realized_pnl
        self.daily_pnl += total_pnl
        self.daily_trades += 1

        if total_pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        duration = (now - self.position.entry_time).total_seconds() / 60.0

        logger.info(
            "[WEBHOOK] Position closed: %s — PnL $%.2f — Duration %.0f min — "
            "Daily: $%.2f (%dW/%dL)",
            self.position.direction, total_pnl, duration,
            self.daily_pnl, self.daily_wins, self.daily_losses,
        )

        # Send trade summary alert
        if self.alerts:
            self.alerts.on_trade_summary(
                direction=self.position.direction,
                entry_price=self.position.entry_price,
                exit_price=0,  # TV provides price in the signal
                total_pnl=total_pnl,
                duration_minutes=duration,
                exit_reason=f"TPs hit: {self.position.tp_hits}",
                tp_log=[],
                equity=self.broker.get_account_equity(),
                daily_wins=self.daily_wins,
                daily_losses=self.daily_losses,
                daily_trades=self.daily_trades,
            )

        # Cooldown
        if total_pnl <= 0:
            self._cooldown_until = now + timedelta(minutes=10)
            logger.info("[WEBHOOK] Cooldown 10min after loss (until %s)",
                        self._cooldown_until.strftime("%H:%M:%S"))
        else:
            self._cooldown_until = now + timedelta(minutes=2)

        self.position = None

    def _reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily stats on new calendar day."""
        day = now.date().toordinal()
        if self._last_day != day:
            if self._last_day is not None:
                logger.info(
                    "[WEBHOOK] New day — yesterday: PnL=$%.2f, W/L=%d/%d",
                    self.daily_pnl, self.daily_wins, self.daily_losses,
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._circuit_breaker_fired = False
            self._last_day = day

    def _save_state(self) -> None:
        """Persist state to JSON."""
        state = {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "circuit_breaker": self._circuit_breaker_fired,
            "position": None,
        }
        if self.position:
            state["position"] = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "total_contracts": self.position.total_contracts,
                "remaining_contracts": self.position.remaining_contracts,
                "tp_hits": self.position.tp_hits,
                "realized_pnl": self.position.realized_pnl,
            }
        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self._state_file)
        except OSError as e:
            logger.error("[WEBHOOK] State save failed: %s", e)

    def _load_state(self) -> None:
        """Load state from JSON if exists."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.daily_trades = state.get("daily_trades", 0)
            self.daily_wins = state.get("daily_wins", 0)
            self.daily_losses = state.get("daily_losses", 0)
            self._circuit_breaker_fired = state.get("circuit_breaker", False)
            pos = state.get("position")
            if pos and pos.get("remaining_contracts", 0) > 0:
                self.position = WebhookPosition(
                    direction=pos["direction"],
                    entry_price=pos["entry_price"],
                    entry_time=datetime.now(PT),
                    total_contracts=pos["total_contracts"],
                    remaining_contracts=pos["remaining_contracts"],
                    tp_hits=pos.get("tp_hits", []),
                    realized_pnl=pos.get("realized_pnl", 0.0),
                )
                logger.info(
                    "[WEBHOOK] Restored position: %s %d ct @ $%.2f",
                    self.position.direction,
                    self.position.remaining_contracts,
                    self.position.entry_price,
                )
        except Exception as e:
            logger.warning("[WEBHOOK] State load failed: %s", e)
