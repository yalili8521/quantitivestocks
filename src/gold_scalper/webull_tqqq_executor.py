"""Webull TQQQ Executor — receives TradingView webhook signals, places TQQQ orders on Webull.

Signal format (from Pine Script):
    {"action": "buy",  "ticker": "TQQQ", "contracts": 1, "position_size":  1, "price": 45.50, "comment": "Long Entry"}
    {"action": "sell", "ticker": "TQQQ", "contracts": 1, "position_size":  0, "price": 46.00, "comment": "Trail Stop"}
    {"action": "sell", "ticker": "TQQQ", "contracts": 1, "position_size": -1, "price": 44.00, "comment": "Short Entry"}
    {"action": "buy",  "ticker": "TQQQ", "contracts": 1, "position_size":  0, "price": 43.50, "comment": "Trail Stop"}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from webull.core.client import ApiClient
from webull.core.common.region import Region
from webull.trade.trade_client import TradeClient

logger = logging.getLogger("webull_tqqq")

ET = ZoneInfo("America/New_York")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STATE_DIR = os.path.join(_PROJECT_ROOT, "outputs", "paper_state", "webhook")
_STATE_FILE = os.path.join(_STATE_DIR, "webull_tqqq_state.json")
_TRADE_LOG = os.path.join(_STATE_DIR, "webull_tqqq_trade_log.json")


@dataclass
class TQQQPosition:
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    entry_time: str             # ISO format
    qty: int
    realized_pnl: float = 0.0


class WebullTQQQExecutor:
    """Executes TQQQ trades on Webull via OpenAPI SDK."""

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        live_orders: bool = False,
        initial_equity: float = 5000.0,
    ):
        self._app_key = app_key
        self._app_secret = app_secret
        self._live_orders = live_orders
        self._initial_equity = initial_equity

        # Webull API client
        self._api_client: Optional[ApiClient] = None
        self._trade_client: Optional[TradeClient] = None
        self._account_id: Optional[str] = None

        # Position state
        self.position: Optional[TQQQPosition] = None
        self._equity = initial_equity

        # Daily stats
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_reset_date: Optional[str] = None

        # Risk controls
        self._circuit_breaker_fired: bool = False
        self._daily_loss_limit: float = -500.0
        self._last_signal_time: float = 0
        self._dedup_window: float = 30.0  # seconds
        self._cooldown_until: float = 0

        # State persistence
        os.makedirs(_STATE_DIR, exist_ok=True)
        self._load_state()

        # Initialize Webull connection
        if live_orders:
            self._init_webull()
            # Sync equity from real Webull balance
            bp = self._fetch_buying_power()
            if bp > 0:
                self._equity = bp
                logger.info("[WEBULL_TQQQ] Equity synced from Webull: $%.2f", bp)

        logger.info(
            "[WEBULL_TQQQ] Initialized: live=%s, equity=$%.2f, position=%s",
            live_orders, self._equity,
            self.position.direction if self.position else "FLAT",
        )

    def _init_webull(self):
        """Initialize Webull API client and discover account."""
        try:
            self._api_client = ApiClient(
                app_key=self._app_key,
                app_secret=self._app_secret,
                region_id=Region.US.value,
            )
            self._trade_client = TradeClient(self._api_client)

            # Discover account
            resp = self._trade_client.account_v2.get_account_list()
            data = resp.json() if hasattr(resp, 'json') else resp
            # Response can be a list or dict with "data" key
            if isinstance(data, list):
                accounts = data
            else:
                accounts = data.get("data", data.get("accounts", []))
            if accounts:
                self._account_id = str(accounts[0].get("account_id", accounts[0].get("accountId", "")))
                logger.info("[WEBULL_TQQQ] Found account: %s", self._account_id)
            else:
                logger.error("[WEBULL_TQQQ] No accounts found in response: %s", data)

        except Exception as e:
            logger.error("[WEBULL_TQQQ] Failed to initialize Webull: %s", e)
            logger.warning("[WEBULL_TQQQ] Will retry Webull init on first order")
            # Keep live_orders=True — retry on first trade

    def execute(self, body: dict) -> dict:
        """Process a TradingView webhook signal.

        Routes:
        - position_size > 0 = go LONG (buy to open)
        - position_size < 0 = go SHORT (sell short to open)
        - position_size == 0 = close position (sell to close or buy to cover)
        """
        now = datetime.now(ET)
        self._maybe_reset_daily(now)

        # Parse signal
        action = body.get("action", "").lower()  # buy or sell
        ticker = body.get("ticker", "TQQQ").upper()
        position_size = body.get("position_size", None)
        price = float(body.get("price", 0))
        comment = body.get("comment", "")

        if ticker != "TQQQ":
            return {"status": "skipped", "reason": f"Not TQQQ: {ticker}"}

        # Dedup
        now_ts = time.time()
        if now_ts - self._last_signal_time < self._dedup_window:
            return {"status": "skipped", "reason": "dedup"}
        self._last_signal_time = now_ts

        # Circuit breaker
        if self._circuit_breaker_fired:
            return {"status": "blocked", "reason": "circuit_breaker"}

        # Cooldown
        if now_ts < self._cooldown_until:
            return {"status": "skipped", "reason": "cooldown"}

        logger.info(
            "[WEBULL_TQQQ] Signal: action=%s pos_size=%s price=%.2f comment=%s",
            action, position_size, price, comment,
        )

        result = {}

        if position_size is not None:
            pos_size = int(position_size)

            if pos_size > 0 and self.position is None:
                # Open LONG
                result = self._open_position("LONG", price, now)

            elif pos_size < 0 and self.position is None:
                # Open SHORT
                result = self._open_position("SHORT", price, now)

            elif pos_size == 0 and self.position is not None:
                # Close position
                result = self._close_position(price, now, comment)

            elif pos_size > 0 and self.position and self.position.direction == "SHORT":
                # Flip: close short, open long
                self._close_position(price, now, "Flip to Long")
                result = self._open_position("LONG", price, now)

            elif pos_size < 0 and self.position and self.position.direction == "LONG":
                # Flip: close long, open short
                self._close_position(price, now, "Flip to Short")
                result = self._open_position("SHORT", price, now)

            else:
                result = {"status": "no_action", "reason": "already_positioned"}
        else:
            # Legacy format: use action field
            if action == "buy" and self.position is None:
                result = self._open_position("LONG", price, now)
            elif action == "sell" and self.position is None:
                result = self._open_position("SHORT", price, now)
            elif action in ("buy", "sell") and self.position is not None:
                result = self._close_position(price, now, comment)
            else:
                result = {"status": "no_action"}

        self._save_state()
        return result

    def _fetch_buying_power(self) -> float:
        """Fetch actual buying power from Webull account."""
        if not self._trade_client or not self._account_id:
            return 0.0
        try:
            resp = self._trade_client.account_v2.get_account_balance(self._account_id)
            data = resp.json() if hasattr(resp, 'json') else resp
            # Primary path: account_currency_assets[0].buying_power
            assets = data.get("account_currency_assets", [])
            if assets:
                bp = float(assets[0].get("buying_power", 0))
                if bp > 0:
                    logger.info("[WEBULL_TQQQ] Buying power from Webull: $%.2f", bp)
                    return bp
            # Fallback: top-level total_cash_balance
            for key in ("total_cash_balance", "total_net_liquidation_value",
                        "buying_power", "cash_balance"):
                val = data.get(key)
                if val is not None:
                    bp = float(val)
                    logger.info("[WEBULL_TQQQ] Buying power (fallback %s): $%.2f", key, bp)
                    return bp
            logger.warning("[WEBULL_TQQQ] Could not parse buying power from: %s", str(data)[:500])
            return 0.0
        except Exception as e:
            logger.error("[WEBULL_TQQQ] Failed to fetch buying power: %s", e)
            return 0.0

    def _calculate_qty(self, price: float) -> int:
        """Calculate position size: 80% of actual Webull buying power."""
        if price <= 0:
            return 0

        # Try to get real buying power from Webull
        if self._live_orders:
            bp = self._fetch_buying_power()
            if bp > 0:
                dollar_amount = bp * 0.80
                qty = int(dollar_amount / price)
                logger.info("[WEBULL_TQQQ] Sizing from Webull BP: $%.2f -> %d shares @ $%.2f",
                            bp, qty, price)
                if qty >= 1:
                    return qty

        # Fallback to tracked equity
        dollar_amount = self._equity * 0.80
        qty = int(dollar_amount / price)
        return max(1, qty)

    def _open_position(self, direction: str, price: float, now: datetime) -> dict:
        """Open a new position."""
        qty = self._calculate_qty(price)

        if self._live_orders:
            order_result = self._place_webull_order(
                side="BUY" if direction == "LONG" else "SELL",
                qty=qty,
                price=price,
                comment=f"{direction} Entry",
            )
            if not order_result.get("success"):
                return {"status": "error", "reason": order_result.get("error", "order_failed")}

        self.position = TQQQPosition(
            direction=direction,
            entry_price=price,
            entry_time=now.isoformat(),
            qty=qty,
        )

        logger.info(
            "[WEBULL_TQQQ] OPENED %s: %d shares @ $%.2f (equity=$%.2f)",
            direction, qty, price, self._equity,
        )

        return {
            "status": "opened",
            "direction": direction,
            "qty": qty,
            "price": price,
            "live": self._live_orders,
        }

    def _close_position(self, price: float, now: datetime, comment: str = "") -> dict:
        """Close the current position and calculate P&L."""
        if not self.position:
            return {"status": "no_position"}

        pos = self.position
        if pos.direction == "LONG":
            pnl_pct = (price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price

        pnl_usd = pos.qty * pos.entry_price * pnl_pct

        # Update equity
        self._equity += pnl_usd
        self.daily_pnl += pnl_usd
        self.daily_trades += 1

        if pnl_usd > 0:
            self.daily_wins += 1
        elif pnl_usd < 0:
            self.daily_losses += 1

        # Place close order on Webull
        if self._live_orders:
            close_side = "SELL" if pos.direction == "LONG" else "BUY"
            self._place_webull_order(
                side=close_side,
                qty=pos.qty,
                price=price,
                comment=comment or "Close",
            )

        logger.info(
            "[WEBULL_TQQQ] CLOSED %s: %d shares @ $%.2f | P&L: $%.2f (%.2f%%) | Equity: $%.2f | %s",
            pos.direction, pos.qty, price, pnl_usd, pnl_pct * 100, self._equity, comment,
        )

        # Log trade
        self._log_trade({
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "qty": pos.qty,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct * 100, 4),
            "comment": comment,
            "entry_time": pos.entry_time,
            "exit_time": now.isoformat(),
            "equity_after": round(self._equity, 2),
            "live": self._live_orders,
        })

        result = {
            "status": "closed",
            "direction": pos.direction,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "comment": comment,
            "equity": round(self._equity, 2),
            "live": self._live_orders,
        }

        self.position = None

        # Circuit breaker check
        if self.daily_pnl <= self._daily_loss_limit:
            self._circuit_breaker_fired = True
            logger.warning(
                "[WEBULL_TQQQ] CIRCUIT BREAKER: daily P&L $%.2f hit limit $%.2f",
                self.daily_pnl, self._daily_loss_limit,
            )

        # Cooldown
        if pnl_usd < 0:
            self._cooldown_until = time.time() + 300  # 5 min after loss
        else:
            self._cooldown_until = time.time() + 60   # 1 min after win

        return result

    def _place_webull_order(self, side: str, qty: int, price: float = 0, comment: str = "") -> dict:
        """Place a limit order on Webull at the TradingView price."""
        # Retry init if needed
        if not self._trade_client or not self._account_id:
            logger.info("[WEBULL_TQQQ] Retrying Webull init before order...")
            self._init_webull()
        if not self._trade_client or not self._account_id:
            logger.error("[WEBULL_TQQQ] Cannot place order: client not initialized")
            return {"success": False, "error": "not_initialized"}

        try:
            client_order_id = f"tqqq_{uuid.uuid4().hex[:16]}"

            # Limit order at TV price — uses symbol (not instrument_id)
            new_order = {
                "symbol": "TQQQ",
                "instrument_type": "STOCK",
                "side": side,
                "order_type": "LIMIT",
                "limit_price": str(round(price, 2)),
                "total_quantity": str(qty),
                "time_in_force": "DAY",
                "client_order_id": client_order_id,
                "market": "US",
                "support_trading_session": "CORE",
                "entrust_type": "QTY",
            }

            logger.info("[WEBULL_TQQQ] Placing LIMIT %s %d shares @ $%.2f", side, qty, price)

            resp = self._trade_client.order_v2.place_order(
                account_id=self._account_id,
                new_orders=[new_order],
            )

            data = resp.json() if hasattr(resp, 'json') else resp
            logger.info("[WEBULL_TQQQ] Order response: %s", str(data)[:500])

            return {"success": True, "order_id": client_order_id, "response": data}

        except Exception as e:
            logger.error("[WEBULL_TQQQ] Order failed: %s", e)
            return {"success": False, "error": str(e)}

    def get_status(self) -> dict:
        """Return current status for /status endpoint."""
        pos_info = None
        if self.position:
            pos_info = {
                "direction": self.position.direction,
                "entry_price": self.position.entry_price,
                "qty": self.position.qty,
                "entry_time": self.position.entry_time,
            }

        # Show real Webull buying power if available
        webull_bp = 0.0
        if self._live_orders:
            webull_bp = self._fetch_buying_power()

        return {
            "position": pos_info,
            "equity": round(self._equity, 2),
            "webull_buying_power": round(webull_bp, 2) if webull_bp > 0 else "unavailable",
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "daily_wl": f"{self.daily_wins}W/{self.daily_losses}L",
            "circuit_breaker": self._circuit_breaker_fired,
            "live_orders": self._live_orders,
        }

    def _maybe_reset_daily(self, now: datetime):
        """Reset daily stats at midnight ET."""
        today = now.strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            if self._last_reset_date is not None:
                logger.info(
                    "[WEBULL_TQQQ] Daily reset: prev P&L=$%.2f, %dW/%dL",
                    self.daily_pnl, self.daily_wins, self.daily_losses,
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._circuit_breaker_fired = False
            self._last_reset_date = today

    def _save_state(self):
        """Persist state to disk (atomic write) and sync to Gist."""
        state = {
            "equity": self._equity,
            "initial_equity": self._initial_equity,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "last_reset_date": self._last_reset_date,
            "circuit_breaker": self._circuit_breaker_fired,
            "position": asdict(self.position) if self.position else None,
        }
        try:
            fd, tmp = tempfile.mkstemp(dir=_STATE_DIR, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, _STATE_FILE)
        except Exception as e:
            logger.error("[WEBULL_TQQQ] Failed to save state: %s", e)

        # Sync to GitHub Gist for Vercel dashboard
        self._sync_to_gist(state)

    def _get_gist_creds(self) -> tuple:
        """Get Gist ID and GitHub token from env."""
        gist_id = os.environ.get("KRAKEN_STATE_GIST_ID", "")
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if not gist_id or not gh_token:
            env_file = os.path.join(_PROJECT_ROOT, "secrets", "alpaca.env")
            if os.path.exists(env_file):
                try:
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            k, _, v = line.partition("=")
                            k, v = k.strip(), v.strip().strip("\"'")
                            if k == "KRAKEN_STATE_GIST_ID" and not gist_id:
                                gist_id = v
                            elif k == "GITHUB_TOKEN" and not gh_token:
                                gh_token = v
                except Exception:
                    pass
        return gist_id, gh_token

    def _sync_to_gist(self, state: dict) -> None:
        """Upload state + trade log to GitHub Gist for the dashboard."""
        gist_id, gh_token = self._get_gist_creds()
        if not gist_id or not gh_token:
            return
        try:
            import requests
            # Dashboard-compatible state
            gist_state = {
                "equity": state.get("equity", 0),
                "initial_equity": state.get("initial_equity", 0),
                "position": state.get("position"),
                "daily_pnl": state.get("daily_pnl", 0),
                "daily_trades": state.get("daily_trades", 0),
                "daily_wins": state.get("daily_wins", 0),
                "daily_losses": state.get("daily_losses", 0),
                "live_orders": self._live_orders,
            }
            files = {
                "webull_tqqq_state.json": {
                    "content": json.dumps(gist_state, indent=2),
                },
            }
            # Include trade log (last 200 entries)
            trades = []
            if os.path.exists(_TRADE_LOG):
                try:
                    with open(_TRADE_LOG) as f:
                        trades = json.load(f)
                except Exception:
                    pass
            if trades:
                files["webull_tqqq_trade_log.json"] = {
                    "content": json.dumps(trades[-200:], indent=2),
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
            logger.debug("[WEBULL_TQQQ] Gist sync error: %s", exc)

    def _load_state(self):
        """Restore state from disk."""
        if not os.path.exists(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                state = json.load(f)
            self._equity = state.get("equity", self._initial_equity)
            self._initial_equity = state.get("initial_equity", self._initial_equity)
            self.daily_pnl = state.get("daily_pnl", 0)
            self.daily_trades = state.get("daily_trades", 0)
            self.daily_wins = state.get("daily_wins", 0)
            self.daily_losses = state.get("daily_losses", 0)
            self._last_reset_date = state.get("last_reset_date")
            self._circuit_breaker_fired = state.get("circuit_breaker", False)

            pos_data = state.get("position")
            if pos_data:
                self.position = TQQQPosition(**pos_data)
            logger.info("[WEBULL_TQQQ] State restored: equity=$%.2f", self._equity)
        except Exception as e:
            logger.error("[WEBULL_TQQQ] Failed to load state: %s", e)

    def _log_trade(self, trade: dict):
        """Append trade to log file."""
        trades = []
        if os.path.exists(_TRADE_LOG):
            try:
                with open(_TRADE_LOG) as f:
                    trades = json.load(f)
            except Exception:
                pass
        trades.append(trade)
        try:
            fd, tmp = tempfile.mkstemp(dir=_STATE_DIR, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(trades, f, indent=2)
            os.replace(tmp, _TRADE_LOG)
        except Exception as e:
            logger.error("[WEBULL_TQQQ] Failed to log trade: %s", e)
