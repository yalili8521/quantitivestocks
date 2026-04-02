"""Webull Webhook Executor — receives TradingView signals, trades on Webull paper.

Routes by instrument type:
- BTC/crypto signals  → Webull CRYPTO orders
- MGC/gold signals    → Webull FUTURES orders

Signal flow: TradingView → webhook_server → WebullExecutor → Webull paper account

Supports both UT Bot format:
    {"action": "buy",  "contracts": 2, "ticker": "BTCUSD", "position_size":  1, "price": 68394}
and Gold scalper format:
    {"action": "buy", "comment": "Long Entry", "price": 3150.5, "contracts": 6}
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Webull symbol mapping
_CRYPTO_TICKERS = {"BTCUSD", "BTC/USD", "BTC-USD", "XBTUSD"}
_FUTURES_TICKERS = {"MGC", "MGCM5", "MGCQ5", "MGCZ5", "GC", "GCM5"}

# Instrument type for Webull v3 API
INSTRUMENT_CRYPTO = "CRYPTO"
INSTRUMENT_FUTURES = "FUTURES"


@dataclass
class WebullPosition:
    """Tracks an open position on Webull."""
    ticker: str                 # original ticker from TradingView
    instrument_type: str        # "CRYPTO" or "FUTURES"
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    qty: float                  # quantity (fractional for crypto, int for futures)
    total_qty: float            # original quantity (for partial close tracking)
    tp_hits: List[int] = field(default_factory=list)
    realized_pnl: float = 0.0
    webull_order_ids: List[str] = field(default_factory=list)


class WebullExecutor:
    """Executes TradingView signals on Webull paper accounts.

    Handles both crypto (BTC) and futures (MGC) through the same
    Webull API client, routing by instrument_type.
    """

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        alerts=None,
        daily_loss_limit: float = -2000.0,
        live_orders: bool = False,
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.alerts = alerts
        self.daily_loss_limit = daily_loss_limit
        self.live_orders = live_orders  # False = local paper, True = real Webull API

        # Webull clients (lazy init, only used when live_orders=True)
        self._api_client = None
        self._trade_client = None
        self._account_ids: Dict[str, str] = {}  # instrument_type -> account_id

        # Positions (keyed by normalized ticker)
        self.positions: Dict[str, WebullPosition] = {}

        # Daily stats
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_day: Optional[int] = None
        self._circuit_breaker_fired: bool = False
        self._cooldown_until: Optional[datetime] = None
        self._last_entry_time: Optional[datetime] = None
        self._entry_dedup_seconds: int = 30

        # Equity tracking (local paper)
        self._initial_equity: float = 5000.0
        self._equity: float = 5000.0
        self._trade_log: List[Dict] = []

        # State persistence
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        state_dir = os.path.join(project_root, "outputs", "paper_state", "webhook")
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "webull_executor_state.json")
        self._log_file = os.path.join(state_dir, "webull_trade_log.json")
        self._load_state()

    # ── Webull Client ──────────────────────────────────────────

    @property
    def api_client(self):
        """Lazy-init Webull API client. Only used when live_orders=True."""
        if self._api_client is None:
            if not self.live_orders:
                raise RuntimeError("Webull API not available in local paper mode")
            from webull.core.client import ApiClient
            self._api_client = ApiClient(
                self.app_key, self.app_secret, "us"
            )
            logger.info("[WEBULL] Using production endpoint")
        return self._api_client

    @property
    def trade_client(self):
        """Lazy-init Webull Trade client."""
        if self._trade_client is None:
            from webull.trade.trade_client import TradeClient
            self._trade_client = TradeClient(self.api_client)
            # Discover account IDs
            self._discover_accounts()
        return self._trade_client

    def _discover_accounts(self) -> None:
        """Fetch account list and map account_ids by type."""
        try:
            res = self._trade_client.account_v2.get_account_list()
            data = res.json() if hasattr(res, 'json') else res
            if isinstance(data, list):
                accounts = data
            elif isinstance(data, dict):
                accounts = data.get("accounts", data.get("data", []))
            else:
                accounts = []

            for acct in accounts:
                acct_type = str(acct.get("account_type", "")).upper()
                acct_id = str(acct.get("account_id", ""))
                if acct_id:
                    # Map account types to our instrument types
                    if "CRYPTO" in acct_type:
                        self._account_ids[INSTRUMENT_CRYPTO] = acct_id
                    elif "FUTURES" in acct_type or "FUTURE" in acct_type:
                        self._account_ids[INSTRUMENT_FUTURES] = acct_id
                    elif "STOCK" in acct_type or "EQUITY" in acct_type:
                        # Stock account can also be used for some instruments
                        if INSTRUMENT_CRYPTO not in self._account_ids:
                            self._account_ids["EQUITY"] = acct_id

            logger.info(
                "[WEBULL] Discovered accounts: %s",
                {k: v[:8] + "..." for k, v in self._account_ids.items()},
            )

            if not self._account_ids:
                logger.warning(
                    "[WEBULL] No accounts found! Raw response: %s",
                    str(data)[:500],
                )
        except Exception as e:
            logger.error("[WEBULL] Account discovery failed: %s", e)

    def _get_account_id(self, instrument_type: str) -> Optional[str]:
        """Get the appropriate account_id for an instrument type."""
        # Ensure trade client is initialized
        _ = self.trade_client
        acct_id = self._account_ids.get(instrument_type)
        if not acct_id:
            logger.error(
                "[WEBULL] No account found for %s. Available: %s",
                instrument_type, list(self._account_ids.keys()),
            )
        return acct_id

    # ── Ticker Classification ─────────────────────────────────

    @staticmethod
    def classify_ticker(ticker: str) -> tuple:
        """Classify a ticker into (webull_symbol, instrument_type).

        Returns:
            (symbol, instrument_type) tuple for Webull API
        """
        ticker_upper = ticker.strip().upper().replace("/", "").replace("-", "")

        if ticker_upper in {"BTCUSD", "XBTUSD"}:
            return "BTCUSD", INSTRUMENT_CRYPTO
        elif ticker_upper in {"ETHUSD", "XETUSD"}:
            return "ETHUSD", INSTRUMENT_CRYPTO

        # Gold futures — need the active contract month
        # TradingView sends "MGC" or "MGC1!" — map to nearest month
        if ticker_upper.startswith("MGC") or ticker_upper.startswith("GC"):
            # Use the ticker as-is if it has a month code, else use generic
            if len(ticker_upper) >= 4 and ticker_upper[-1].isdigit():
                return ticker_upper, INSTRUMENT_FUTURES
            else:
                # Default to generic — Webull may need specific contract
                return "MGC", INSTRUMENT_FUTURES

        # Fallback: treat as equity
        return ticker_upper, "EQUITY"

    # ── Public API ──────────────────────────────────────────────

    def execute(self, payload: Dict) -> Dict:
        """Parse and execute a TradingView webhook payload.

        Handles both UT Bot format (position_size) and Gold scalper format
        (action + comment). Routes to Webull by instrument type.

        Returns result dict for webhook response.
        """
        now = datetime.now(ET)
        self._reset_daily_if_needed(now)

        ticker = payload.get("ticker", "").strip().upper()
        action = payload.get("action", "").strip().lower()
        price = float(payload.get("price", 0))
        contracts = float(payload.get("contracts", 0))
        position_size = payload.get("position_size")
        comment = payload.get("comment", "").strip()

        webull_symbol, instrument_type = self.classify_ticker(ticker)

        logger.info(
            "[WEBULL] Received: ticker=%s action=%s price=$%.2f qty=%.4f "
            "pos_size=%s comment='%s' → %s/%s",
            ticker, action, price, contracts,
            position_size, comment,
            webull_symbol, instrument_type,
        )

        # Dedup guard
        if self._last_entry_time:
            elapsed = (now - self._last_entry_time).total_seconds()
            if elapsed < self._entry_dedup_seconds:
                msg = f"Duplicate rejected ({elapsed:.0f}s since last)"
                logger.warning("[WEBULL] %s", msg)
                return {"status": "rejected", "message": msg}

        # Circuit breaker
        if self._circuit_breaker_fired:
            msg = f"Circuit breaker active (daily PnL ${self.daily_pnl:.2f})"
            logger.warning("[WEBULL] %s", msg)
            return {"status": "rejected", "message": msg}

        if self.daily_pnl <= self.daily_loss_limit:
            self._circuit_breaker_fired = True
            msg = f"Daily loss limit hit: ${self.daily_pnl:.2f}"
            logger.warning("[WEBULL] %s", msg)
            return {"status": "rejected", "message": msg}

        # Cooldown
        if self._cooldown_until and now < self._cooldown_until:
            msg = f"Cooldown until {self._cooldown_until.strftime('%H:%M:%S')}"
            logger.info("[WEBULL] %s", msg)
            return {"status": "rejected", "message": msg}

        self._last_entry_time = now

        # Determine desired state
        # UT Bot format: position_size field takes priority
        if position_size is not None:
            pos_size = int(position_size)
            if pos_size > 0:
                desired = "LONG"
            elif pos_size < 0:
                desired = "SHORT"
            else:
                desired = "FLAT"
        elif comment:
            # Gold scalper format — parse from comment
            return self._handle_gold_signal(
                payload, webull_symbol, instrument_type, now
            )
        else:
            # Fallback to action
            if action == "buy":
                desired = "LONG"
            elif action == "sell":
                desired = "SHORT"
            else:
                desired = "FLAT"

        # Current state for this ticker
        norm_ticker = webull_symbol
        current_pos = self.positions.get(norm_ticker)
        current = current_pos.direction if current_pos else "FLAT"

        if desired == current:
            msg = f"Already {current} on {norm_ticker} — no action needed"
            logger.info("[WEBULL] %s", msg)
            return {"status": "ok", "message": msg}

        result = {"status": "error", "message": "Unknown"}

        # Close existing position if needed
        if current != "FLAT" and current_pos:
            result = self._close_position(
                norm_ticker, price, now,
                reason=f"Flip to {desired}",
            )

        # Open new position if not going flat
        if desired != "FLAT":
            result = self._open_position(
                norm_ticker, instrument_type, desired,
                contracts, price, now,
            )

        self._save_state()
        return result

    # ── Gold Scalper Signal Handling ──────────────────────────

    def _handle_gold_signal(
        self, payload: Dict, webull_symbol: str,
        instrument_type: str, now: datetime,
    ) -> Dict:
        """Handle Gold scalper format signals (action + comment)."""
        action = payload.get("action", "").strip().lower()
        comment = payload.get("comment", "").strip().upper()
        price = float(payload.get("price", 0))
        contracts = float(payload.get("contracts", 0))
        norm_ticker = webull_symbol

        # Parse signal type from comment
        if "ENTRY" in comment or "ENTER" in comment:
            # Determine direction
            if "SHORT" in comment:
                direction = "SHORT"
            elif "LONG" in comment:
                direction = "LONG"
            else:
                direction = "LONG" if action == "buy" else "SHORT"

            # Close existing if any
            if norm_ticker in self.positions:
                self._close_position(
                    norm_ticker, price, now, reason="New entry signal"
                )

            result = self._open_position(
                norm_ticker, instrument_type, direction,
                contracts, price, now,
            )

        elif "TP" in comment:
            # Partial close on take profit
            tp_level = 0
            for i in range(1, 5):
                if f"TP{i}" in comment or f"TP {i}" in comment:
                    tp_level = i
                    break

            result = self._partial_close(
                norm_ticker, contracts, price, now,
                reason=f"TP{tp_level}",
                tp_level=tp_level,
            )

        elif "STOP" in comment:
            result = self._close_position(
                norm_ticker, price, now, reason="Hard Stop"
            )

        elif "SESSION" in comment or "CLOSE ALL" in comment:
            result = self._close_position(
                norm_ticker, price, now, reason="Session Close"
            )

        elif "EXIT" in comment or "CLOSE" in comment:
            result = self._close_position(
                norm_ticker, price, now,
                reason=payload.get("comment", "Exit"),
            )

        else:
            # Unknown — treat as entry if flat, exit if in position
            if norm_ticker in self.positions:
                result = self._close_position(
                    norm_ticker, price, now,
                    reason=payload.get("comment", "Unknown signal"),
                )
            else:
                direction = "LONG" if action == "buy" else "SHORT"
                result = self._open_position(
                    norm_ticker, instrument_type, direction,
                    contracts, price, now,
                )

        self._save_state()
        return result

    # ── Position Management ────────────────────────────────────

    def _open_position(
        self, ticker: str, instrument_type: str,
        direction: str, qty: float, price: float, now: datetime,
    ) -> Dict:
        """Open a new position on Webull."""
        if qty <= 0:
            return {"status": "rejected", "message": "Zero quantity"}

        side = "BUY" if direction == "LONG" else "SELL"

        order_id = self._submit_webull_order(
            ticker, instrument_type, side, qty, price,
        )

        self.positions[ticker] = WebullPosition(
            ticker=ticker,
            instrument_type=instrument_type,
            direction=direction,
            entry_price=price,
            entry_time=now,
            qty=qty,
            total_qty=qty,
            webull_order_ids=[order_id] if order_id else [],
        )

        logger.info(
            "[WEBULL] OPEN %s %.4f %s @ $%.2f — order %s",
            direction, qty, ticker, price, order_id,
        )

        return {
            "status": "ok",
            "message": f"{direction} {qty} {ticker} @ ${price:.2f}",
            "order_id": order_id,
        }

    def _close_position(
        self, ticker: str, exit_price: float, now: datetime,
        reason: str = "Signal",
    ) -> Dict:
        """Close full position, record P&L."""
        pos = self.positions.get(ticker)
        if not pos:
            return {"status": "ok", "message": f"No position on {ticker}"}

        qty = pos.qty
        side = "SELL" if pos.direction == "LONG" else "BUY"

        order_id = self._submit_webull_order(
            ticker, pos.instrument_type, side, qty, exit_price,
        )

        # Calculate P&L
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * qty
        else:
            pnl = (pos.entry_price - exit_price) * qty

        # For MGC futures, P&L is in pips * $1 per pip per contract
        # For crypto, it's just price diff * qty
        if pos.instrument_type == INSTRUMENT_FUTURES:
            # MGC: 1 tick = $0.10 = 0.10 point. pip_value is $1/point per ct
            pnl = pnl * 1.0  # already correct for futures point value

        total_pnl = pos.realized_pnl + pnl
        self.daily_pnl += total_pnl
        self.daily_trades += 1
        self._equity += total_pnl

        if total_pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        duration = (now - pos.entry_time).total_seconds() / 60.0

        logger.info(
            "[WEBULL] CLOSE %s %.4f %s @ $%.2f — PnL $%.2f — %s — "
            "Daily: $%.2f (%dW/%dL) — Equity: $%.2f",
            pos.direction, qty, ticker, exit_price, total_pnl, reason,
            self.daily_pnl, self.daily_wins, self.daily_losses,
            self._equity,
        )

        # Trade log
        self._trade_log.append({
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": ticker,
            "direction": pos.direction,
            "qty": qty,
            "entry": pos.entry_price,
            "exit": exit_price,
            "pnl": round(total_pnl, 2),
            "reason": reason,
            "broker": "webull",
        })

        # Cooldown
        if total_pnl < 0:
            self._cooldown_until = now + timedelta(minutes=5)
        else:
            self._cooldown_until = now + timedelta(minutes=1)

        del self.positions[ticker]

        return {
            "status": "ok",
            "message": f"Closed {qty} {ticker} — PnL ${total_pnl:.2f} — {reason}",
            "order_id": order_id,
        }

    def _partial_close(
        self, ticker: str, qty: float, price: float,
        now: datetime, reason: str = "TP", tp_level: int = 0,
    ) -> Dict:
        """Partially close a position (TP hit)."""
        pos = self.positions.get(ticker)
        if not pos or pos.qty <= 0:
            return {"status": "rejected", "message": f"No position on {ticker}"}

        close_qty = min(qty, pos.qty)
        if close_qty <= 0:
            return {"status": "rejected", "message": "Zero contracts to close"}

        side = "SELL" if pos.direction == "LONG" else "BUY"

        order_id = self._submit_webull_order(
            ticker, pos.instrument_type, side, close_qty, price,
        )

        # Calculate partial P&L
        if pos.direction == "LONG":
            pnl = (price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - price) * close_qty

        if pos.instrument_type == INSTRUMENT_FUTURES:
            pnl = pnl * 1.0

        pos.realized_pnl += pnl
        pos.qty -= close_qty
        if tp_level:
            pos.tp_hits.append(tp_level)

        logger.info(
            "[WEBULL] %s: closed %.4f %s @ $%.2f — PnL $%.2f — remaining %.4f",
            reason, close_qty, ticker, price, pnl, pos.qty,
        )

        # If fully closed, finalize
        if pos.qty <= 0:
            total_pnl = pos.realized_pnl
            self.daily_pnl += total_pnl
            self.daily_trades += 1
            self._equity += total_pnl
            if total_pnl >= 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1

            self._trade_log.append({
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": ticker,
                "direction": pos.direction,
                "qty": pos.total_qty,
                "entry": pos.entry_price,
                "exit": price,
                "pnl": round(total_pnl, 2),
                "reason": reason,
                "broker": "webull",
            })

            if total_pnl < 0:
                self._cooldown_until = now + timedelta(minutes=5)
            else:
                self._cooldown_until = now + timedelta(minutes=1)

            del self.positions[ticker]

        return {
            "status": "ok",
            "message": f"{reason}: {close_qty} {ticker} @ ${price:.2f} — PnL ${pnl:.2f}",
            "order_id": order_id,
        }

    # ── Webull Order Submission ────────────────────────────────

    def _submit_webull_order(
        self, symbol: str, instrument_type: str,
        side: str, qty: float, price: float,
    ) -> str:
        """Submit order to Webull v3 API. Returns client_order_id.

        In paper mode (live_orders=False), generates a local order ID
        and uses TradingView price as fill — no API call.
        """
        client_order_id = uuid.uuid4().hex[:12]

        if not self.live_orders:
            order_id = f"PAPER-{client_order_id}"
            logger.info(
                "[WEBULL] PAPER %s %s %.4f %s @ $%.2f — %s",
                side, symbol, qty, instrument_type, price, order_id,
            )
            return order_id

        # --- Live Webull API order ---
        account_id = self._get_account_id(instrument_type)
        if not account_id:
            raise RuntimeError(
                f"No Webull account for {instrument_type}. "
                f"Available: {list(self._account_ids.keys())}"
            )

        order_params = {
            "combo_type": "NORMAL",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "instrument_type": instrument_type,
            "market": "US",
            "order_type": "MARKET",
            "quantity": str(qty),
            "side": side,
            "time_in_force": "GTC" if instrument_type == INSTRUMENT_CRYPTO else "DAY",
            "entrust_type": "QTY",
        }

        if instrument_type != INSTRUMENT_CRYPTO:
            order_params["support_trading_session"] = "CORE"

        logger.info(
            "[WEBULL] Submitting order: %s %s %.4f %s @ MARKET (acct=%s...)",
            side, symbol, qty, instrument_type, account_id[:8],
        )

        res = self.trade_client.order_v3.place_order(
            account_id, [order_params]
        )

        try:
            resp_data = res.json() if hasattr(res, 'json') else res
            logger.info("[WEBULL] Order response: %s", str(resp_data)[:300])
        except Exception:
            logger.info("[WEBULL] Order submitted (raw): %s", str(res)[:300])

        return client_order_id

    # ── Account Queries ────────────────────────────────────────

    def get_account_balance(self, instrument_type: str = INSTRUMENT_CRYPTO) -> Dict:
        """Query account balance from Webull."""
        account_id = self._get_account_id(instrument_type)
        if not account_id:
            return {"error": f"No account for {instrument_type}"}

        try:
            res = self.trade_client.account_v2.get_account_balance(account_id)
            return res.json() if hasattr(res, 'json') else res
        except Exception as e:
            logger.error("[WEBULL] Balance query failed: %s", e)
            return {"error": str(e)}

    def get_account_positions(self, instrument_type: str = INSTRUMENT_CRYPTO) -> Dict:
        """Query open positions from Webull."""
        account_id = self._get_account_id(instrument_type)
        if not account_id:
            return {"error": f"No account for {instrument_type}"}

        try:
            res = self.trade_client.account_v2.get_account_position(account_id)
            return res.json() if hasattr(res, 'json') else res
        except Exception as e:
            logger.error("[WEBULL] Position query failed: %s", e)
            return {"error": str(e)}

    # ── State Persistence ──────────────────────────────────────

    def _save_state(self) -> None:
        """Save state to local JSON file."""
        positions_data = {}
        for ticker, pos in self.positions.items():
            positions_data[ticker] = {
                "ticker": pos.ticker,
                "instrument_type": pos.instrument_type,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "qty": pos.qty,
                "total_qty": pos.total_qty,
                "tp_hits": pos.tp_hits,
                "realized_pnl": pos.realized_pnl,
            }

        local_state = {
            "equity": self._equity,
            "initial_balance": self._initial_equity,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "circuit_breaker": self._circuit_breaker_fired,
            "positions": positions_data,
        }

        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(local_state, f, indent=2)
            os.replace(tmp, self._state_file)
        except OSError as e:
            logger.error("[WEBULL] State save failed: %s", e)

        # Save trade log
        try:
            tmp = self._log_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._trade_log, f, indent=2)
            os.replace(tmp, self._log_file)
        except OSError as e:
            logger.error("[WEBULL] Trade log save failed: %s", e)

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
            self._equity = state.get("equity", self._initial_equity)
            self._initial_equity = state.get("initial_balance", self._initial_equity)

            positions_data = state.get("positions", {})
            for ticker, pos_data in positions_data.items():
                self.positions[ticker] = WebullPosition(
                    ticker=pos_data["ticker"],
                    instrument_type=pos_data["instrument_type"],
                    direction=pos_data["direction"],
                    entry_price=pos_data["entry_price"],
                    entry_time=datetime.now(ET),
                    qty=pos_data["qty"],
                    total_qty=pos_data["total_qty"],
                    tp_hits=pos_data.get("tp_hits", []),
                    realized_pnl=pos_data.get("realized_pnl", 0.0),
                )
                logger.info(
                    "[WEBULL] Restored position: %s %s %.4f @ $%.2f",
                    pos_data["direction"], ticker,
                    pos_data["qty"], pos_data["entry_price"],
                )
        except Exception as e:
            logger.warning("[WEBULL] State load failed: %s", e)

    def _reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily stats on new calendar day."""
        day = now.date().toordinal()
        if self._last_day != day:
            if self._last_day is not None:
                logger.info(
                    "[WEBULL] New day — yesterday: PnL=$%.2f, W/L=%d/%d",
                    self.daily_pnl, self.daily_wins, self.daily_losses,
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._circuit_breaker_fired = False
            self._last_day = day
