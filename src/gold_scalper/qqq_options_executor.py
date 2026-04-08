"""QQQ Options Executor — buys 0DTE ATM calls/puts based on webhook signals.

Uses 6-level TP system with discrete contracts (same as gold scalper).
Routes through the third Alpaca paper account (ALPACA_CRYPTO_KEY/SECRET).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOptionContractsRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    AssetStatus,
)

logger = logging.getLogger("qqq_options")

# ── Paths ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STATE_DIR = os.path.join(_PROJECT_ROOT, "outputs", "paper_state", "webhook")
_STATE_FILE = os.path.join(_STATE_DIR, "qqq_options_state.json")
_TRADE_LOG = os.path.join(_STATE_DIR, "qqq_options_trade_log.json")


@dataclass
class OptionsPosition:
    direction: str               # "LONG_CALL" or "LONG_PUT"
    entry_price: float           # avg premium paid per contract
    entry_time: str              # ISO format
    symbol: str                  # option symbol (e.g., QQQ260406C00588000)
    total_contracts: int
    remaining_contracts: int
    tp_hits: List[int] = field(default_factory=list)
    realized_pnl: float = 0.0
    underlying_entry: float = 0.0  # QQQ price at entry


@dataclass
class OptionsConfig:
    """Configuration for QQQ options trading."""
    # Position sizing
    max_risk_per_trade: float = 300.0    # max $ to spend on premium per trade
    target_contracts: int = 10           # ideal number of contracts to buy

    # 6-level TP: how many contracts to close at each level
    # Must sum to target_contracts
    tp1_contracts: int = 3   # close 3 at TP1
    tp2_contracts: int = 2   # close 2 at TP2
    tp3_contracts: int = 2   # close 2 at TP3
    tp4_contracts: int = 1   # close 1 at TP4
    tp5_contracts: int = 1   # close 1 at TP5
    tp6_contracts: int = 1   # close 1 at TP6 (runner)

    # TP targets (% gain on option premium)
    tp1_pct: float = 20.0    # +20% on premium
    tp2_pct: float = 40.0    # +40%
    tp3_pct: float = 60.0    # +60%
    tp4_pct: float = 80.0    # +80%
    tp5_pct: float = 120.0   # +120%
    tp6_pct: float = 200.0   # +200% (runner)

    # Stop loss (% loss on premium)
    sl_pct: float = 50.0     # -50% on premium

    # Breakeven: after TP1, move stop to entry price (0% loss)
    be_after_tp1: bool = True

    # Circuit breaker
    daily_loss_limit: float = -500.0
    cooldown_after_loss_sec: int = 600   # 10 min
    cooldown_after_win_sec: int = 120    # 2 min
    dedup_window_sec: int = 30

    # Option selection
    max_strike_distance: float = 2.0  # max $ from ATM for strike selection


class QQQOptionsExecutor:
    """Executes QQQ options trades via Alpaca API."""

    def __init__(self, api_key: str, api_secret: str,
                 config: Optional[OptionsConfig] = None):
        self._client = TradingClient(api_key, api_secret, paper=True)
        self.config = config or OptionsConfig()
        self._position: Optional[OptionsPosition] = None
        self._equity: float = 5000.0
        self._initial_equity: float = 5000.0
        self._daily_pnl: float = 0.0
        self._trade_count: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._last_entry_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        self._current_day: Optional[date] = None
        self._load_state()

    # ── Public API ──

    def execute(self, payload: Dict) -> Dict:
        """Execute a signal from the webhook."""
        now = datetime.utcnow()

        # Reset daily stats
        today = date.today()
        if self._current_day != today:
            self._daily_pnl = 0.0
            self._current_day = today

        action = str(payload.get("action", "")).lower()
        ticker = str(payload.get("ticker", "")).upper().replace("/", "").replace("-", "")
        price = float(payload.get("price", 0))
        comment = str(payload.get("comment", ""))
        position_size = payload.get("position_size")

        logger.info("[QQQ-OPT] Signal: action=%s ticker=%s price=%.2f comment=%s pos_size=%s",
                     action, ticker, price, comment, position_size)

        # Determine desired state
        if position_size is not None:
            pos_size = float(position_size)
            if pos_size > 0:
                return self._handle_entry(price, "CALL", comment, now)
            elif pos_size < 0:
                return self._handle_entry(price, "PUT", comment, now)
            elif pos_size == 0:
                return self._handle_exit(price, comment, now)
            # Partial close (0 < pos_size < 1)
            if 0 < abs(pos_size) < 1:
                return self._handle_tp(price, comment, now)

        # Fallback: parse from action
        if action == "buy":
            if self._position and "PUT" in self._position.direction:
                return self._handle_exit(price, comment, now)
            return self._handle_entry(price, "CALL", comment, now)
        elif action == "sell":
            if self._position and "CALL" in self._position.direction:
                return self._handle_exit(price, comment, now)
            return self._handle_entry(price, "PUT", comment, now)

        return {"status": "ignored", "reason": f"unknown action: {action}"}

    def get_status(self) -> Dict:
        """Return current state for dashboard."""
        return {
            "position": asdict(self._position) if self._position else None,
            "equity": self._equity,
            "daily_pnl": self._daily_pnl,
            "trade_count": self._trade_count,
            "wins": self._wins,
            "losses": self._losses,
        }

    # ── Entry ──

    def _handle_entry(self, underlying_price: float, option_type: str,
                      comment: str, now: datetime) -> Dict:
        # Guards
        if self._position is not None:
            return {"status": "rejected", "reason": "already in position"}

        if self._last_entry_time and (now - self._last_entry_time).total_seconds() < self.config.dedup_window_sec:
            return {"status": "rejected", "reason": "dedup window"}

        if self._cooldown_until and now < self._cooldown_until:
            remaining = (self._cooldown_until - now).total_seconds()
            return {"status": "rejected", "reason": f"cooldown {remaining:.0f}s remaining"}

        if self._daily_pnl <= self.config.daily_loss_limit:
            return {"status": "rejected", "reason": f"daily loss limit hit: ${self._daily_pnl:.2f}"}

        # Find the right option contract
        try:
            contract = self._find_0dte_contract(underlying_price, option_type)
        except Exception as e:
            logger.error("[QQQ-OPT] Failed to find contract: %s", e)
            return {"status": "error", "reason": str(e)}

        if not contract:
            return {"status": "error", "reason": "no suitable 0DTE contract found"}

        # Determine number of contracts
        num_contracts = self.config.target_contracts

        # Submit order
        try:
            order = self._submit_order(
                contract.symbol, OrderSide.BUY, num_contracts
            )
        except Exception as e:
            logger.error("[QQQ-OPT] Order failed: %s", e)
            return {"status": "error", "reason": str(e)}

        # Estimate premium (will be filled at market)
        est_premium = float(contract.close_price or 0) if hasattr(contract, 'close_price') else 0

        self._position = OptionsPosition(
            direction=f"LONG_{option_type}",
            entry_price=est_premium,
            entry_time=now.isoformat(),
            symbol=contract.symbol,
            total_contracts=num_contracts,
            remaining_contracts=num_contracts,
            underlying_entry=underlying_price,
        )
        self._last_entry_time = now
        self._save_state()

        logger.info("[QQQ-OPT] ENTRY: %s %d × %s @ ~$%.2f (QQQ=$%.2f)",
                     option_type, num_contracts, contract.symbol, est_premium, underlying_price)

        return {
            "status": "filled",
            "action": "entry",
            "direction": f"LONG_{option_type}",
            "symbol": contract.symbol,
            "contracts": num_contracts,
            "estimated_premium": est_premium,
            "order_id": str(order.id) if order else None,
        }

    # ── TP Hit ──

    def _handle_tp(self, price: float, comment: str, now: datetime) -> Dict:
        if not self._position:
            return {"status": "ignored", "reason": "no position"}

        # Determine TP level from comment
        tp_level = self._parse_tp_level(comment)
        if tp_level is None:
            return {"status": "ignored", "reason": "could not parse TP level"}

        contracts_to_close = self._get_tp_contracts(tp_level)
        if contracts_to_close <= 0 or contracts_to_close > self._position.remaining_contracts:
            contracts_to_close = min(contracts_to_close, self._position.remaining_contracts)

        if contracts_to_close <= 0:
            return {"status": "ignored", "reason": "no contracts to close"}

        # Submit sell order
        try:
            side = OrderSide.SELL
            order = self._submit_order(self._position.symbol, side, contracts_to_close)
        except Exception as e:
            logger.error("[QQQ-OPT] TP order failed: %s", e)
            return {"status": "error", "reason": str(e)}

        self._position.remaining_contracts -= contracts_to_close
        self._position.tp_hits.append(tp_level)

        logger.info("[QQQ-OPT] TP%d: closed %d contracts, %d remaining",
                     tp_level, contracts_to_close, self._position.remaining_contracts)

        # If fully closed
        if self._position.remaining_contracts <= 0:
            self._on_position_closed(price, f"TP{tp_level}", now)

        self._save_state()
        return {
            "status": "filled",
            "action": f"tp{tp_level}",
            "contracts_closed": contracts_to_close,
            "remaining": self._position.remaining_contracts if self._position else 0,
        }

    # ── Exit ──

    def _handle_exit(self, price: float, comment: str, now: datetime) -> Dict:
        if not self._position:
            return {"status": "ignored", "reason": "no position"}

        remaining = self._position.remaining_contracts
        if remaining <= 0:
            self._position = None
            self._save_state()
            return {"status": "ignored", "reason": "no remaining contracts"}

        # Close all remaining
        try:
            order = self._submit_order(self._position.symbol, OrderSide.SELL, remaining)
        except Exception as e:
            logger.error("[QQQ-OPT] Exit order failed: %s", e)
            return {"status": "error", "reason": str(e)}

        self._position.remaining_contracts = 0
        reason = comment if comment else "exit"
        self._on_position_closed(price, reason, now)

        return {
            "status": "filled",
            "action": "exit",
            "reason": reason,
            "contracts_closed": remaining,
        }

    # ── Option Contract Selection ──

    def _find_0dte_contract(self, underlying_price: float, option_type: str):
        """Find ATM 0DTE QQQ option contract."""
        today = date.today()

        # Round strike to nearest $1
        atm_strike = round(underlying_price)

        req = GetOptionContractsRequest(
            underlying_symbols=["QQQ"],
            status=AssetStatus.ACTIVE,
            expiration_date=today,
            type=option_type.lower(),  # "call" or "put"
            strike_price_gte=str(atm_strike - self.config.max_strike_distance),
            strike_price_lte=str(atm_strike + self.config.max_strike_distance),
        )

        response = self._client.get_option_contracts(req)
        if not response or not response.option_contracts:
            # Try next trading day if no 0DTE available
            tomorrow = today + timedelta(days=1)
            req.expiration_date = tomorrow
            response = self._client.get_option_contracts(req)

        if not response or not response.option_contracts:
            return None

        # Find closest to ATM
        contracts = response.option_contracts
        best = min(contracts, key=lambda c: abs(float(c.strike_price) - underlying_price))

        logger.info("[QQQ-OPT] Selected: %s strike=$%s exp=%s",
                     best.symbol, best.strike_price, best.expiration_date)
        return best

    # ── Order Submission ──

    def _submit_order(self, symbol: str, side: OrderSide, qty: int):
        """Submit a market order for options."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("[QQQ-OPT] Order submitted: %s %d × %s → %s",
                     side.value, qty, symbol, order.id)
        return order

    # ── Helpers ──

    def _parse_tp_level(self, comment: str) -> Optional[int]:
        """Extract TP level from comment string."""
        comment_upper = comment.upper()
        for i in range(6, 0, -1):  # check TP6 first to avoid TP1 matching TP10
            if f"TP{i}" in comment_upper:
                return i
        return None

    def _get_tp_contracts(self, tp_level: int) -> int:
        """Get number of contracts to close at given TP level."""
        mapping = {
            1: self.config.tp1_contracts,
            2: self.config.tp2_contracts,
            3: self.config.tp3_contracts,
            4: self.config.tp4_contracts,
            5: self.config.tp5_contracts,
            6: self.config.tp6_contracts,
        }
        return mapping.get(tp_level, 1)

    def _on_position_closed(self, exit_price: float, reason: str, now: datetime):
        """Handle full position close."""
        if not self._position:
            return

        # Estimate PnL (rough — actual fill prices come from Alpaca)
        # For now use underlying price movement as proxy
        direction_mult = 1 if "CALL" in self._position.direction else -1
        underlying_move = (exit_price - self._position.underlying_entry) * direction_mult
        # Very rough: option delta ~0.5 for ATM, each contract = 100 shares
        est_pnl = underlying_move * 0.5 * self._position.total_contracts * 100 / 100  # per contract lot
        # This is approximate — real PnL comes from actual fill prices

        self._daily_pnl += est_pnl
        self._equity += est_pnl
        self._trade_count += 1

        if est_pnl >= 0:
            self._wins += 1
            self._cooldown_until = now + timedelta(seconds=self.config.cooldown_after_win_sec)
        else:
            self._losses += 1
            self._cooldown_until = now + timedelta(seconds=self.config.cooldown_after_loss_sec)

        # Log trade
        trade_record = {
            "trade_num": self._trade_count,
            "direction": self._position.direction,
            "symbol": self._position.symbol,
            "entry_time": self._position.entry_time,
            "exit_time": now.isoformat(),
            "underlying_entry": self._position.underlying_entry,
            "underlying_exit": exit_price,
            "total_contracts": self._position.total_contracts,
            "tp_hits": self._position.tp_hits,
            "exit_reason": reason,
            "est_pnl": round(est_pnl, 2),
            "equity": round(self._equity, 2),
        }
        self._append_trade_log(trade_record)

        logger.info("[QQQ-OPT] CLOSED: %s reason=%s est_pnl=$%.2f equity=$%.2f",
                     self._position.direction, reason, est_pnl, self._equity)

        self._position = None
        self._save_state()

    # ── State Persistence ──

    def _save_state(self):
        os.makedirs(_STATE_DIR, exist_ok=True)
        state = {
            "equity": self._equity,
            "initial_equity": self._initial_equity,
            "daily_pnl": self._daily_pnl,
            "trade_count": self._trade_count,
            "wins": self._wins,
            "losses": self._losses,
            "position": asdict(self._position) if self._position else None,
            "last_entry_time": self._last_entry_time.isoformat() if self._last_entry_time else None,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "current_day": self._current_day.isoformat() if self._current_day else None,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=_STATE_DIR, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, _STATE_FILE)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _load_state(self):
        if not os.path.exists(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                state = json.load(f)
            self._equity = state.get("equity", 5000.0)
            self._initial_equity = state.get("initial_equity", 5000.0)
            self._daily_pnl = state.get("daily_pnl", 0.0)
            self._trade_count = state.get("trade_count", 0)
            self._wins = state.get("wins", 0)
            self._losses = state.get("losses", 0)
            if state.get("position"):
                self._position = OptionsPosition(**state["position"])
            if state.get("last_entry_time"):
                self._last_entry_time = datetime.fromisoformat(state["last_entry_time"])
            if state.get("cooldown_until"):
                self._cooldown_until = datetime.fromisoformat(state["cooldown_until"])
            if state.get("current_day"):
                self._current_day = date.fromisoformat(state["current_day"])
            logger.info("[QQQ-OPT] State loaded: equity=$%.2f trades=%d pos=%s",
                         self._equity, self._trade_count,
                         self._position.direction if self._position else "FLAT")
        except Exception as e:
            logger.warning("[QQQ-OPT] Failed to load state: %s", e)

    def _append_trade_log(self, record: Dict):
        os.makedirs(_STATE_DIR, exist_ok=True)
        trades = []
        if os.path.exists(_TRADE_LOG):
            try:
                with open(_TRADE_LOG) as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        trades.append(record)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=_STATE_DIR, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(trades, f, indent=2)
            os.replace(tmp_path, _TRADE_LOG)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
