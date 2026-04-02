"""BTC Webhook Executor — receives UT Bot signals from TradingView, trades on Alpaca.

Signal flow: TradingView UT Bot → webhook_server → BTCWebhookExecutor → Alpaca paper

UT Bot payload format:
    {"action": "buy",  "contracts": 2, "ticker": "BTCUSD", "position_size":  1, "price": 68394}
    {"action": "sell", "contracts": 2, "ticker": "BTCUSD", "position_size": -1, "price": 68394}

position_size logic:
    1  = we should be LONG  (if flat → open long,  if short → flip to long)
   -1  = we should be SHORT (if flat → open short, if long  → flip to short)
    0  = flatten (close everything)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Alpaca crypto symbol
ALPACA_SYMBOL = "BTC/USD"


@dataclass
class BTCPosition:
    """Tracks an open BTC position."""
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    qty: float               # fractional BTC quantity
    realized_pnl: float = 0.0


class BTCWebhookExecutor:
    """Executes TradingView UT Bot signals on Alpaca crypto paper account.

    - Parses UT Bot payload (action + position_size)
    - Manages position state (one position at a time)
    - Tracks equity and P&L for dashboard
    - Syncs state to Alpaca (actual orders) — dashboard reads from Alpaca API
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        alerts=None,
        daily_loss_limit: float = -2000.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.alerts = alerts
        self.daily_loss_limit = daily_loss_limit

        # Alpaca client (lazy init)
        self._client = None

        # State
        self.position: Optional[BTCPosition] = None
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_day: Optional[int] = None
        self._circuit_breaker_fired: bool = False
        self._cooldown_until: Optional[datetime] = None
        self._last_entry_time: Optional[datetime] = None
        self._entry_dedup_seconds: int = 30

        # Equity tracking
        self._initial_equity: float = 0.0  # read from Alpaca on first call
        self._trade_log: List[Dict] = []

        # State persistence
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        state_dir = os.path.join(project_root, "outputs", "paper_state", "webhook")
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "btc_executor_state.json")
        self._log_file = os.path.join(state_dir, "btc_trade_log.json")
        self._load_state()

    @property
    def client(self):
        """Lazy-init Alpaca TradingClient."""
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                self.api_key, self.api_secret, paper=True
            )
        return self._client

    # ── Public API ──────────────────────────────────────────────

    def execute(self, payload: Dict) -> Dict:
        """Parse and execute a UT Bot webhook payload.

        Returns result dict for the webhook response.
        """
        now = datetime.now(ET)
        self._reset_daily_if_needed(now)

        action = payload.get("action", "").strip().lower()
        price = float(payload.get("price", 0))
        position_size = int(payload.get("position_size", 0))
        qty = float(payload.get("contracts", 0))
        ticker = payload.get("ticker", "BTCUSD")

        logger.info(
            "[BTC] Received: action=%s pos_size=%d qty=%.4f price=$%.2f ticker=%s",
            action, position_size, qty, price, ticker,
        )

        # Dedup guard
        if self._last_entry_time:
            elapsed = (now - self._last_entry_time).total_seconds()
            if elapsed < self._entry_dedup_seconds:
                msg = f"Duplicate rejected ({elapsed:.0f}s since last)"
                logger.warning("[BTC] %s", msg)
                return {"status": "rejected", "message": msg}

        # Circuit breaker
        if self._circuit_breaker_fired:
            msg = f"Circuit breaker active (daily PnL ${self.daily_pnl:.2f})"
            logger.warning("[BTC] %s", msg)
            return {"status": "rejected", "message": msg}

        if self.daily_pnl <= self.daily_loss_limit:
            self._circuit_breaker_fired = True
            msg = f"Daily loss limit hit: ${self.daily_pnl:.2f}"
            logger.warning("[BTC] %s", msg)
            return {"status": "rejected", "message": msg}

        # Cooldown
        if self._cooldown_until and now < self._cooldown_until:
            msg = f"Cooldown until {self._cooldown_until.strftime('%H:%M:%S')}"
            logger.info("[BTC] %s", msg)
            return {"status": "rejected", "message": msg}

        self._last_entry_time = now

        # Determine desired state from position_size (takes priority over action)
        # position_size=1 → LONG, -1 → SHORT, 0 → FLAT
        # If position_size not in payload, fall back to action (buy/sell)
        has_position_size = "position_size" in payload
        desired = None
        if has_position_size:
            if position_size > 0:
                desired = "LONG"
            elif position_size < 0:
                desired = "SHORT"
            else:
                desired = "FLAT"
        else:
            # No position_size field — use action as fallback
            if action == "buy":
                desired = "LONG"
            elif action == "sell":
                desired = "SHORT"
            else:
                desired = "FLAT"

        # Current state
        current = self.position.direction if self.position else "FLAT"

        if desired == current:
            msg = f"Already {current} — no action needed"
            logger.info("[BTC] %s", msg)
            return {"status": "ok", "message": msg}

        result = {"status": "error", "message": "Unknown"}

        # Close existing position if needed
        if current != "FLAT":
            result = self._close_position(price, now, reason=f"Flip to {desired}")

        # Open new position if not going flat
        if desired != "FLAT":
            result = self._open_position(desired, qty, price, now)

        self._save_state()
        return result

    # ── Position Management ────────────────────────────────────

    def _open_position(self, direction: str, qty: float, price: float,
                       now: datetime) -> Dict:
        """Open a new BTC position on Alpaca."""
        if qty <= 0:
            return {"status": "rejected", "message": "Zero quantity"}

        side = "buy" if direction == "LONG" else "sell"

        try:
            order = self._submit_order(side, qty)
            order_id = str(getattr(order, "id", "UNKNOWN"))
        except Exception as e:
            logger.error("[BTC] Order failed: %s", e)
            return {"status": "error", "message": f"Order failed: {e}"}

        self.position = BTCPosition(
            direction=direction,
            entry_price=price,
            entry_time=now,
            qty=qty,
        )

        logger.info(
            "[BTC] OPEN %s %.4f BTC @ $%.2f — order %s",
            direction, qty, price, order_id,
        )

        return {
            "status": "ok",
            "message": f"{direction} {qty} BTC @ ${price:.2f}",
            "order_id": order_id,
        }

    def _close_position(self, exit_price: float, now: datetime,
                        reason: str = "Signal") -> Dict:
        """Close current position, record P&L."""
        if not self.position:
            return {"status": "ok", "message": "No position to close"}

        qty = self.position.qty
        # To close: sell if LONG, buy if SHORT
        side = "sell" if self.position.direction == "LONG" else "buy"

        try:
            order = self._submit_order(side, qty)
            order_id = str(getattr(order, "id", "UNKNOWN"))
        except Exception as e:
            logger.error("[BTC] Close failed: %s", e)
            return {"status": "error", "message": f"Close failed: {e}"}

        # Calculate P&L
        if self.position.direction == "LONG":
            pnl = (exit_price - self.position.entry_price) * qty
        else:
            pnl = (self.position.entry_price - exit_price) * qty

        self.daily_pnl += pnl
        self.daily_trades += 1
        if pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        duration = (now - self.position.entry_time).total_seconds() / 60.0

        logger.info(
            "[BTC] CLOSE %s %.4f BTC @ $%.2f — PnL $%.2f — %s — "
            "Daily: $%.2f (%dW/%dL)",
            self.position.direction, qty, exit_price, pnl, reason,
            self.daily_pnl, self.daily_wins, self.daily_losses,
        )

        # Trade log
        self._trade_log.append({
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": "BTC/USD",
            "direction": self.position.direction,
            "qty": qty,
            "entry": self.position.entry_price,
            "exit": exit_price,
            "pnl": round(pnl, 2),
            "reason": reason,
        })

        # Cooldown
        if pnl < 0:
            self._cooldown_until = now + timedelta(minutes=5)
        else:
            self._cooldown_until = now + timedelta(minutes=1)

        self.position = None

        return {
            "status": "ok",
            "message": f"Closed {qty} BTC — PnL ${pnl:.2f} — {reason}",
            "order_id": order_id,
        }

    def _submit_order(self, side: str, qty: float):
        """Submit market order to Alpaca."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=ALPACA_SYMBOL,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return self.client.submit_order(req)

    # ── State Persistence ──────────────────────────────────────

    def _save_state(self) -> None:
        """Save state to local JSON file."""
        local_state = {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "circuit_breaker": self._circuit_breaker_fired,
            "position": None,
        }
        if self.position:
            local_state["position"] = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "qty": self.position.qty,
            }

        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(local_state, f, indent=2)
            os.replace(tmp, self._state_file)
        except OSError as e:
            logger.error("[BTC] State save failed: %s", e)

        # Save trade log
        try:
            tmp = self._log_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._trade_log, f, indent=2)
            os.replace(tmp, self._log_file)
        except OSError as e:
            logger.error("[BTC] Trade log save failed: %s", e)

    def _load_state(self) -> None:
        """Load state from JSON."""
        if os.path.exists(self._log_file):
            try:
                with open(self._log_file) as f:
                    self._trade_log = json.load(f)
            except Exception:
                pass

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
            if pos:
                self.position = BTCPosition(
                    direction=pos["direction"],
                    entry_price=pos["entry_price"],
                    entry_time=datetime.now(ET),
                    qty=pos["qty"],
                )
                logger.info(
                    "[BTC] Restored position: %s %.4f BTC @ $%.2f",
                    self.position.direction, self.position.qty,
                    self.position.entry_price,
                )
        except Exception as e:
            logger.warning("[BTC] State load failed: %s", e)

    def _reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily stats on new calendar day."""
        day = now.date().toordinal()
        if self._last_day != day:
            if self._last_day is not None:
                logger.info(
                    "[BTC] New day — yesterday: PnL=$%.2f, W/L=%d/%d",
                    self.daily_pnl, self.daily_wins, self.daily_losses,
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._circuit_breaker_fired = False
            self._last_day = day
