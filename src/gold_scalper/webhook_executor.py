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
    """Routes TradingView signals to broker with risk controls.

    TradingView信号执行器：
    - 接收解析后的webhook信号
    - 检查风险控制（每日亏损限额、冷却期、最大仓位）
    - 通过broker下单
    - 跟踪仓位状态和每日盈亏
    - 同步状态到GitHub Gist供dashboard显示
    """

    def __init__(
        self,
        broker,  # IBKRGoldBroker or PaperGoldBroker
        config: GoldScalperConfig,
        alerts: Optional[ScalperAlertEngine] = None,
        initial_equity: float = 5000.0,
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
        self._last_entry_attempt: Optional[datetime] = None  # dedup guard
        self._entry_dedup_seconds: int = 30  # reject duplicate entries within this window

        # Equity tracking for dashboard
        self._initial_equity: float = initial_equity
        self._equity: float = initial_equity
        self._trade_log: List[Dict] = []

        # State persistence — separate folder from signal-mode bot
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        state_dir = os.path.join(project_root, "outputs", "paper_state", "webhook")
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "webhook_executor_state.json")
        self._log_file = os.path.join(state_dir, "webhook_trade_log.json")
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

        # Dedup guard — TradingView fires the same alert 2-4x at ~5s intervals
        if self._last_entry_attempt:
            elapsed = (now - self._last_entry_attempt).total_seconds()
            if elapsed < self._entry_dedup_seconds:
                msg = f"Duplicate entry rejected ({elapsed:.0f}s since last attempt)"
                logger.warning("[WEBHOOK] %s", msg)
                return {"status": "rejected", "message": msg}
        self._last_entry_attempt = now

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

        # No real broker — TV price is the fill price. Generate a local order ID.
        self._order_counter = getattr(self, "_order_counter", 0) + 1
        order_id = f"TV-{self._order_counter:06d}"

        # Create position using TradingView price as fill price
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

        # No real broker — track locally with TV price
        self._order_counter = getattr(self, "_order_counter", 0) + 1
        order_id = f"TV-{self._order_counter:06d}"

        self.position.remaining_contracts -= contracts
        if signal.tp_level:
            self.position.tp_hits.append(signal.tp_level)

        # Calculate P&L using TradingView price
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
            self._on_position_closed(now, exit_price=signal.price,
                                     exit_reason=f"TP{signal.tp_level}")

        return {
            "status": "ok",
            "message": f"TP{signal.tp_level}: {contracts}ct closed @ ${signal.price:.2f}",
            "order_id": order_id,
        }

    def _handle_exit(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle EXIT signal — close all remaining."""
        if not self.position or self.position.remaining_contracts <= 0:
            return {"status": "rejected", "message": "No position to close"}

        reason = signal.reason or "TradingView exit"
        contracts = self.position.remaining_contracts

        # No real broker — track locally with TV price
        self._order_counter = getattr(self, "_order_counter", 0) + 1
        order_id = f"TV-{self._order_counter:06d}"

        # Calculate P&L using TradingView price
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

        self._on_position_closed(now, exit_price=signal.price,
                                 exit_reason=reason)

        return {
            "status": "ok",
            "message": f"Closed {contracts}ct @ ${signal.price:.2f} — {reason}",
            "order_id": order_id,
        }

    def _handle_session_close(self, signal: WebhookSignal, now: datetime) -> Dict:
        """Handle SESSION_CLOSE — close everything."""
        if not self.position or self.position.remaining_contracts <= 0:
            return {"status": "ok", "message": "No position — session close ignored"}

        signal.reason = "Session Close (TradingView)"
        return self._handle_exit(signal, now)

    def _on_position_closed(self, now: datetime, exit_price: float = 0,
                            exit_reason: str = "") -> None:
        """Update stats after position fully closed."""
        if not self.position:
            return

        total_pnl = self.position.realized_pnl
        self.daily_pnl += total_pnl
        self.daily_trades += 1
        self._equity += total_pnl

        if total_pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        duration = (now - self.position.entry_time).total_seconds() / 60.0

        logger.info(
            "[WEBHOOK] Position closed: %s — PnL $%.2f — Duration %.0f min — "
            "Daily: $%.2f (%dW/%dL) — Equity: $%.2f",
            self.position.direction, total_pnl, duration,
            self.daily_pnl, self.daily_wins, self.daily_losses,
            self._equity,
        )

        # Append to trade log (same format as PaperGoldBroker for dashboard)
        self._trade_log.append({
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.config.symbol.replace("=F", ""),
            "direction": self.position.direction,
            "contracts": self.position.total_contracts,
            "entry": self.position.entry_price,
            "exit": exit_price or self.position.entry_price,
            "pnl": round(total_pnl, 2),
            "reason": exit_reason or f"TPs hit: {self.position.tp_hits}",
        })

        # Send trade summary alert
        if self.alerts:
            self.alerts.on_trade_summary(
                direction=self.position.direction,
                entry_price=self.position.entry_price,
                exit_price=exit_price,
                total_pnl=total_pnl,
                duration_minutes=duration,
                exit_reason=exit_reason or f"TPs hit: {self.position.tp_hits}",
                tp_log=[],
                equity=self._equity,
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
        """Persist state to JSON and sync to GitHub Gist for dashboard."""
        # Dashboard-compatible state (matches PaperGoldBroker format)
        pos_for_dashboard = None
        if self.position and self.position.remaining_contracts > 0:
            pos_for_dashboard = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "contracts": self.position.remaining_contracts,
            }

        gist_state = {
            "equity": self._equity,
            "initial_balance": self._initial_equity,
            "trade_count": len(self._trade_log),
            "position": pos_for_dashboard,
        }

        # Local state (includes executor-specific fields for restart)
        local_state = {
            **gist_state,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "circuit_breaker": self._circuit_breaker_fired,
            # Keep detailed position for executor restore
            "position_detail": None,
        }
        if self.position:
            local_state["position_detail"] = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "total_contracts": self.position.total_contracts,
                "remaining_contracts": self.position.remaining_contracts,
                "tp_hits": self.position.tp_hits,
                "realized_pnl": self.position.realized_pnl,
            }

        # Save local state
        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(local_state, f, indent=2)
            os.replace(tmp, self._state_file)
        except OSError as e:
            logger.error("[WEBHOOK] State save failed: %s", e)

        # Save local trade log
        try:
            tmp = self._log_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._trade_log, f, indent=2)
            os.replace(tmp, self._log_file)
        except OSError as e:
            logger.error("[WEBHOOK] Trade log save failed: %s", e)

        # Sync to GitHub Gist for Vercel dashboard
        self._sync_to_gist(gist_state)

    def _sync_to_gist(self, state: dict) -> None:
        """Upload state + trade log to GitHub Gist for the dashboard."""
        gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
        gh_token = os.environ.get("GITHUB_TOKEN", "")

        if not gist_id or not gh_token:
            env_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "secrets", "alpaca.env",
            )
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            v = v.strip().strip('"').strip("'")
                            if k.strip() == "KRAKEN_STATE_GIST_ID" and not gist_id:
                                gist_id = v
                            elif k.strip() == "GITHUB_TOKEN" and not gh_token:
                                gh_token = v

        if not gist_id or not gh_token:
            return

        try:
            import requests
            files = {
                "gold_scalper_state.json": {
                    "content": json.dumps(state, indent=2),
                },
            }
            if self._trade_log:
                files["gold_scalper_trade_log.json"] = {
                    "content": json.dumps(self._trade_log[-200:], indent=2),
                }
            requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers={
                    "Authorization": f"token {gh_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json={"files": files},
                timeout=10,
            )
        except Exception as exc:
            logger.debug("[WEBHOOK] Gist sync error: %s", exc)

    def _load_state(self) -> None:
        """Load state from JSON if exists."""
        # Load trade log
        if os.path.exists(self._log_file):
            try:
                with open(self._log_file) as f:
                    self._trade_log = json.load(f)
            except Exception as e:
                logger.warning("[WEBHOOK] Trade log load failed: %s", e)

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
            self._equity = state.get("equity", self._initial_equity)
            self._initial_equity = state.get("initial_balance", self._initial_equity)

            pos = state.get("position_detail") or state.get("position")
            if pos and pos.get("remaining_contracts", pos.get("contracts", 0)) > 0:
                self.position = WebhookPosition(
                    direction=pos["direction"],
                    entry_price=pos["entry_price"],
                    entry_time=datetime.now(PT),
                    total_contracts=pos.get("total_contracts", pos.get("contracts", 0)),
                    remaining_contracts=pos.get("remaining_contracts", pos.get("contracts", 0)),
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
