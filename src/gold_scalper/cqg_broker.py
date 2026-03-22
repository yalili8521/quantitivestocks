"""CQG broker adapter for AMP Futures demo — micro gold (MGC) trading.

Connects to CQG WebAPI (wss://demoapi.cqg.com) for order execution.
Uses Yahoo Finance for price data and historical bars (simpler, more reliable).
Position tracking is local (updated from order submissions).

Usage:
    broker = CQGGoldBroker(config, username="demo95880", password="...")
    broker.place_market_order("MGC", 6, "BUY")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add gold_scalper dir to sys.path so CQG protobuf 'from WebAPI import ...' works
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from src.gold_scalper.broker_adapter import GoldBrokerAdapter, _tf_minutes
from src.gold_scalper.config import GoldScalperConfig

logger = logging.getLogger(__name__)

# CQG WebAPI constants
CQG_DEMO_HOST = "wss://demoapi.cqg.com:443"

# Order types
ORDER_TYPE_MKT = 1
ORDER_TYPE_LMT = 2
ORDER_TYPE_STP = 3

# Order sides
SIDE_BUY = 1
SIDE_SELL = 2

# Duration
DURATION_DAY = 1


def _load_cqg_env() -> tuple:
    """Load CQG credentials from secrets/alpaca.env."""
    env_file = _THIS_DIR.parent.parent / "secrets" / "alpaca.env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), val)

    username = os.environ.get("AMP_CQG_USERNAME", "")
    password = os.environ.get("AMP_CQG_PASSWORD", "")
    return username, password


class CQGGoldBroker(GoldBrokerAdapter):
    """Broker adapter for AMP Futures via CQG WebAPI demo.

    Order execution:  CQG WebAPI (real demo account on AMP)
    Price data:       Yahoo Finance (reliable, no threading needed)
    Historical bars:  Yahoo Finance
    Position tracking: Local state + CQG fill confirmations
    """

    def __init__(
        self,
        config: GoldScalperConfig,
        username: str = "",
        password: str = "",
        initial_equity: float = 50000.0,
    ):
        self.config = config
        self.symbol = config.symbol
        self._cqg_symbol = "MGC"  # CQG symbol for micro gold front month

        # CQG connection state
        self._client = None  # WebApiClient instance
        self._contract_id: Optional[int] = None
        self._account_id: Optional[int] = None
        self._base_time: Optional[str] = None
        self._msg_id = 0
        self._order_counter = 0
        self._connected = False

        # Position tracking (local)
        self._equity = initial_equity
        self._position: Optional[Dict] = None
        self._trade_log: List[dict] = []

        # State persistence
        self._state_dir = _THIS_DIR.parent.parent / "outputs" / "paper_state"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "gold_scalper_cqg_state.json"
        self._log_file = self._state_dir / "gold_scalper_cqg_trade_log.json"
        self._load_state()

        # Yahoo data adapter (lazy)
        self._yahoo_adapter = None

        # Load credentials and connect
        if not username or not password:
            username, password = _load_cqg_env()
        if not username or not password:
            raise RuntimeError(
                "AMP_CQG_USERNAME/PASSWORD not set. "
                "Add them to secrets/alpaca.env"
            )

        self._connect(username, password)

    # ── CQG Connection ─────────────────────────────────────

    def _next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _next_order_id(self) -> str:
        self._order_counter += 1
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"GS-{ts}-{self._order_counter:04d}"

    def _connect(self, username: str, password: str) -> None:
        """Connect to CQG WebAPI demo, authenticate, resolve symbol."""
        from WebAPI.webapi_2_pb2 import ClientMsg
        from WebAPI.webapi_client import WebApiClient

        logger.info("Connecting to CQG WebAPI demo (%s)...", CQG_DEMO_HOST)

        self._client = WebApiClient(need_to_log=False)
        self._client.connect(CQG_DEMO_HOST)

        # ── Step 1: Logon ──
        client_msg = ClientMsg()
        logon = client_msg.logon
        logon.user_name = username
        logon.password = password
        logon.client_app_id = "GoldScalperBot"
        logon.client_version = "1.0.0"
        logon.protocol_version_major = 2
        logon.protocol_version_minor = 230
        self._client.send_client_message(client_msg)

        resp = self._client.receive_server_message()
        from WebAPI.user_session_2_pb2 import LogonResult
        if resp.logon_result.result_code != LogonResult.ResultCode.RESULT_CODE_SUCCESS:
            raise RuntimeError(
                f"CQG logon failed: {resp.logon_result.text_message}"
            )
        self._base_time = resp.logon_result.base_time
        self._connected = True
        logger.info("CQG logon successful (base_time=%s)", self._base_time)

        # ── Step 2: Resolve MGC symbol → contract_id ──
        msg_id = self._next_msg_id()
        client_msg = ClientMsg()
        info_req = client_msg.information_requests.add()
        info_req.id = msg_id
        info_req.symbol_resolution_request.symbol = self._cqg_symbol
        self._client.send_client_message(client_msg)

        resp = self._client.receive_server_message()
        if not resp.information_reports:
            raise RuntimeError("CQG symbol resolution failed for MGC")
        metadata = resp.information_reports[0].symbol_resolution_report.contract_metadata
        self._contract_id = metadata.contract_id
        logger.info(
            "Resolved %s → contract_id=%d, title=%s",
            self._cqg_symbol, self._contract_id, metadata.title,
        )

        # ── Step 3: Get account ID ──
        msg_id = self._next_msg_id()
        client_msg = ClientMsg()
        info_req = client_msg.information_requests.add()
        info_req.id = msg_id
        info_req.accounts_request.SetInParent()
        self._client.send_client_message(client_msg)

        resp = self._client.receive_server_message()
        for report in resp.information_reports:
            if report.HasField("accounts_report"):
                for brokerage in report.accounts_report.brokerages:
                    for sales_series in brokerage.sales_series:
                        for account in sales_series.accounts:
                            self._account_id = account.account_id
                            logger.info(
                                "Account ID: %d (name=%s)",
                                account.account_id, account.name,
                            )
                            break
                        if self._account_id:
                            break
                    if self._account_id:
                        break

        if not self._account_id:
            logger.warning(
                "Could not resolve account_id — orders will fail. "
                "Check CQG account setup."
            )

        # ── Step 4: Subscribe to trade updates ──
        msg_id = self._next_msg_id()
        client_msg = ClientMsg()
        trade_sub = client_msg.trade_subscriptions.add()
        trade_sub.id = msg_id
        trade_sub.subscribe = True
        trade_sub.subscription_scopes.extend([1])  # order_status
        self._client.send_client_message(client_msg)

        # Drain trade snapshot completion
        resp = self._client.receive_server_message()
        if resp.trade_snapshot_completions:
            logger.info("Trade subscription active")
        # Drain one more message (snapshot data)
        try:
            self._client.websocket_client.settimeout(3)
            resp2 = self._client.receive_server_message()
        except Exception:
            pass
        # Reset to blocking
        self._client.websocket_client.settimeout(None)

    def disconnect(self) -> None:
        """Disconnect from CQG."""
        if self._client and self._connected:
            try:
                from WebAPI.webapi_2_pb2 import ClientMsg
                client_msg = ClientMsg()
                client_msg.logoff.text_message = "GoldScalper shutdown"
                self._client.send_client_message(client_msg)
            except Exception:
                pass
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected from CQG")

    # ── GoldBrokerAdapter interface ────────────────────────

    def get_current_price(self, symbol: str) -> float:
        """Get latest gold price via Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            logger.warning("Yahoo price fetch failed: %s", e)

        if self._position and "entry_price" in self._position:
            return self._position["entry_price"]

        raise RuntimeError("Cannot get current gold price")

    def place_market_order(
        self, symbol: str, contracts: int, side: str
    ) -> Optional[str]:
        """Place a market order on CQG demo."""
        if not self._connected or not self._account_id:
            logger.error("CQG not connected or no account_id — cannot place order")
            return None

        from WebAPI.webapi_2_pb2 import ClientMsg

        order_id = self._next_order_id()
        cqg_side = SIDE_BUY if side == "BUY" else SIDE_SELL
        msg_id = self._next_msg_id()

        client_msg = ClientMsg()
        order_req = client_msg.order_requests.add()
        order_req.request_id = msg_id
        order_req.new_order.order.account_id = self._account_id
        order_req.new_order.order.when_utc_time = 0
        order_req.new_order.order.contract_id = self._contract_id
        order_req.new_order.order.cl_order_id = order_id
        order_req.new_order.order.order_type = ORDER_TYPE_MKT
        order_req.new_order.order.duration = DURATION_DAY
        order_req.new_order.order.side = cqg_side
        order_req.new_order.order.qty.significand = contracts
        order_req.new_order.order.qty.exponent = 0
        order_req.new_order.order.is_manual = False
        self._client.send_client_message(client_msg)

        # Wait for order acknowledgment
        try:
            self._client.websocket_client.settimeout(10)
            resp = self._client.receive_server_message()
            # Check for rejection
            for reject in resp.order_request_rejects:
                logger.error(
                    "CQG order REJECTED: %s (code=%d)",
                    reject.text_message, reject.reject_code,
                )
                return None
            logger.info("CQG order acknowledged")
        except Exception as e:
            logger.warning("No ack received for order %s: %s", order_id, e)
        finally:
            self._client.websocket_client.settimeout(None)

        # Update local position tracking
        price = self.get_current_price(symbol)
        direction = "LONG" if side == "BUY" else "SHORT"

        if self._position is None:
            self._position = {
                "direction": direction,
                "entry_price": price,
                "contracts": contracts,
            }
        else:
            if self._position["direction"] == direction:
                self._position["contracts"] += contracts
            else:
                remaining = self._position["contracts"] - contracts
                if remaining <= 0:
                    self._position = None
                else:
                    self._position["contracts"] = remaining

        logger.info(
            "CQG order sent: %s %d MGC @ ~$%.2f (order_id=%s)",
            side, contracts, price, order_id,
        )
        self._save_state()
        return order_id

    def close_partial(
        self, symbol: str, contracts: int, reason: str
    ) -> Optional[str]:
        """Close N contracts at market."""
        if self._position is None:
            logger.warning("No position to close")
            return None

        price = self.get_current_price(symbol)
        exit_side = "SELL" if self._position["direction"] == "LONG" else "BUY"

        # Calculate P&L
        entry = self._position["entry_price"]
        pip_val = self.config.pip_value
        if self._position["direction"] == "LONG":
            pips = (price - entry) / pip_val
        else:
            pips = (entry - price) / pip_val
        # MGC: $1 per pip per contract
        pnl = pips * contracts

        # Place order on CQG
        order_id = self.place_market_order(symbol, contracts, exit_side)

        # Update equity
        self._equity += pnl

        # Log trade
        self._trade_log.append({
            "time": datetime.utcnow().isoformat(),
            "symbol": self._cqg_symbol,
            "direction": self._position["direction"] if self._position else "?",
            "contracts": contracts,
            "entry": entry,
            "exit": price,
            "pips": round(pips, 1),
            "pnl": round(pnl, 2),
            "reason": reason,
            "order_id": order_id or "?",
        })

        logger.info(
            "CQG close %d MGC @ $%.2f (%s) PnL: $%.2f (%+.0f pips)",
            contracts, price, reason, pnl, pips,
        )

        if self._position and self._position["contracts"] <= 0:
            self._position = None

        self._save_state()
        return order_id

    def close_all(self, symbol: str, reason: str) -> Optional[str]:
        """Close entire position."""
        if self._position is None:
            return None
        return self.close_partial(symbol, self._position["contracts"], reason)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return current position or None."""
        if self._position and self._position.get("contracts", 0) > 0:
            return dict(self._position)
        return None

    def get_account_equity(self) -> float:
        """Return tracked equity."""
        return self._equity

    def fetch_bars(
        self, symbol: str, timeframe: str, lookback: int
    ) -> pd.DataFrame:
        """Fetch OHLCV bars via Yahoo Finance (GC=F)."""
        adapter = self._get_yahoo_adapter()
        yahoo_symbol = "GC=F"

        try:
            if timeframe == "1d":
                df = adapter.fetch_daily(yahoo_symbol, lookback=lookback)
            elif timeframe == "4h":
                from src.gold_scalper.bias_stack import BiasStack
                df_1h = adapter.fetch_intraday(
                    yahoo_symbol, interval="1h",
                    lookback_days=max(5, lookback // 6)
                )
                df = BiasStack.resample_to_4h(df_1h)
            else:
                interval_map = {
                    "1h": "1h", "15m": "15m", "5m": "5min", "5min": "5min"
                }
                interval = interval_map.get(timeframe, timeframe)
                days = max(2, lookback // (390 // _tf_minutes(timeframe)))
                df = adapter.fetch_intraday(
                    yahoo_symbol, interval=interval, lookback_days=days
                )
        except Exception as e:
            logger.error("Failed to fetch %s bars: %s", timeframe, e)
            return pd.DataFrame()

        if len(df) > lookback:
            df = df.tail(lookback).reset_index(drop=True)

        return df

    def _get_yahoo_adapter(self):
        """Lazy-load Yahoo Finance adapter."""
        if self._yahoo_adapter is None:
            from src.signals_engine import YahooFinanceAdapter
            self._yahoo_adapter = YahooFinanceAdapter()
        return self._yahoo_adapter

    # ── State persistence ──────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "equity": self._equity,
            "position": self._position,
            "order_counter": self._order_counter,
            "broker": "cqg_amp_demo",
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

        with open(self._log_file, "w") as f:
            json.dump(self._trade_log[-500:], f, indent=2)

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    state = json.load(f)
                self._equity = state.get("equity", self._equity)
                self._position = state.get("position")
                self._order_counter = state.get("order_counter", 0)
                logger.info(
                    "Loaded CQG state: equity=$%.2f, position=%s",
                    self._equity, self._position,
                )
            except Exception as e:
                logger.warning("Failed to load CQG state: %s", e)

        if self._log_file.exists():
            try:
                with open(self._log_file) as f:
                    self._trade_log = json.load(f)
            except Exception:
                self._trade_log = []
