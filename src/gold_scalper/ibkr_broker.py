"""IBKR broker adapter for gold scalper — Micro Gold (MGC) via ib_insync.

Connects to TWS or IB Gateway for paper/live trading of MGC futures.
Implements the same GoldBrokerAdapter interface as PaperGoldBroker.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from src.gold_scalper.broker_adapter import GoldBrokerAdapter
from src.gold_scalper.config import GoldScalperConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBKR timeframe mapping
# ---------------------------------------------------------------------------
_TF_MAP = {
    "5m": ("5 mins", "2 D"),
    "15m": ("15 mins", "5 D"),
    "1h": ("1 hour", "20 D"),
    "4h": ("4 hours", "60 D"),
    "1d": ("1 day", "365 D"),
}


def _resolve_front_month_mgc(ib) -> "Future":
    """Resolve the front-month MGC contract by querying IBKR for all expiries.

    Returns the qualified Future contract with the nearest expiry that
    hasn't passed yet.
    """
    from ib_insync import Future

    # Ask IBKR for all MGC contracts
    generic = Future("MGC", exchange="COMEX")
    details_list = ib.reqContractDetails(generic)

    if not details_list:
        raise RuntimeError("No MGC contracts found on IBKR")

    from datetime import date

    today = date.today()
    candidates = []
    for d in details_list:
        c = d.contract
        expiry_str = c.lastTradeDateOrContractMonth  # e.g. "20260428"
        try:
            exp_date = date(int(expiry_str[:4]), int(expiry_str[4:6]), int(expiry_str[6:8]))
        except (ValueError, IndexError):
            continue
        if exp_date >= today:
            candidates.append((exp_date, c))

    if not candidates:
        raise RuntimeError("No unexpired MGC contracts found")

    # Sort by expiry, skip contracts within 30 days of expiry
    # (IBKR blocks near-expiry physically-delivered futures)
    from datetime import timedelta

    safe_candidates = [(exp, c) for exp, c in candidates if exp >= today + timedelta(days=45)]
    if not safe_candidates:
        safe_candidates = candidates  # fallback to nearest if all are near-expiry

    safe_candidates.sort(key=lambda x: x[0])
    front = safe_candidates[0][1]

    # Qualify it
    qualified = ib.qualifyContracts(front)
    if not qualified:
        raise RuntimeError(f"Could not qualify MGC {front.localSymbol}")

    return qualified[0]


class IBKRGoldBroker(GoldBrokerAdapter):
    """Interactive Brokers adapter for Micro Gold (MGC) futures."""

    def __init__(
        self,
        config: GoldScalperConfig,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        from ib_insync import IB

        self._config = config
        self._host = host
        self._port = port
        self._client_id = client_id

        # State persistence (backup — real state lives at IBKR)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        state_dir = os.path.join(project_root, "outputs", "paper_state")
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, "gold_scalper_ibkr_state.json")
        self._trade_log_file = os.path.join(
            state_dir, "gold_scalper_ibkr_trade_log.json"
        )
        self._trade_log: List[Dict] = self._load_trade_log()

        # Connect
        self.ib = IB()
        self._connect()

        # Resolve MGC front-month contract from IBKR
        self._contract = _resolve_front_month_mgc(self.ib)
        logger.info(
            "[IBKR] Connected — %s (conId=%s, mult=%s)",
            self._contract.localSymbol,
            self._contract.conId,
            self._contract.multiplier,
        )

        # Request delayed market data (type 3) if no live subscription
        self.ib.reqMarketDataType(3)

        # Local position tracking (synced from IBKR on startup)
        self._position: Optional[Dict] = None
        self._equity: float = 0.0
        self._order_counter: int = 0
        self._sync_from_ibkr()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Connect to TWS/Gateway with retry."""
        for attempt in range(3):
            try:
                if not self.ib.isConnected():
                    self.ib.connect(
                        self._host, self._port, clientId=self._client_id
                    )
                    logger.info("[IBKR] Connected to %s:%s", self._host, self._port)
                return
            except Exception as e:
                logger.warning(
                    "[IBKR] Connection attempt %d/3 failed: %s", attempt + 1, e
                )
                time.sleep(5)
        raise RuntimeError(
            f"Cannot connect to IBKR at {self._host}:{self._port} after 3 attempts"
        )

    def _ensure_connected(self) -> None:
        """Reconnect if connection dropped."""
        if not self.ib.isConnected():
            logger.warning("[IBKR] Connection lost — reconnecting...")
            self._connect()

    def _sync_from_ibkr(self) -> None:
        """Sync account equity and positions from IBKR."""
        self._ensure_connected()

        # Equity
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation" and item.currency == "USD":
                self._equity = float(item.value)
                break
        logger.info("[IBKR] Account equity: $%.2f", self._equity)

        # Positions
        positions = self.ib.positions()
        self._position = None
        for pos in positions:
            if (
                pos.contract.symbol == "MGC"
                and pos.contract.exchange == "COMEX"
            ):
                qty = pos.position
                if qty != 0:
                    self._position = {
                        "direction": "LONG" if qty > 0 else "SHORT",
                        "entry_price": pos.avgCost / float(self._contract.multiplier),
                        "contracts": abs(int(qty)),
                    }
                    logger.info(
                        "[IBKR] Loaded position: %s %d ct @ $%.2f",
                        self._position["direction"],
                        self._position["contracts"],
                        self._position["entry_price"],
                    )
                break

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------
    def get_current_price(self, symbol: str) -> float:
        """Get current MGC price from IBKR market data."""
        self._ensure_connected()

        ticker = self.ib.reqMktData(self._contract)
        self.ib.sleep(1)  # Wait for data

        # Try bid/ask midpoint first, fall back to last
        bid = ticker.bid if ticker.bid > 0 else None
        ask = ticker.ask if ticker.ask > 0 else None
        last = ticker.last if ticker.last > 0 else None

        if bid and ask:
            price = (bid + ask) / 2.0
        elif last:
            price = last
        elif ticker.close and ticker.close > 0:
            price = ticker.close
        else:
            raise RuntimeError(
                f"[IBKR] No price available for {self._contract.localSymbol}"
            )

        self.ib.cancelMktData(self._contract)
        return price

    def place_market_order(
        self, symbol: str, contracts: int, side: str
    ) -> Optional[str]:
        """Place market order on IBKR."""
        from ib_insync import MarketOrder

        self._ensure_connected()

        ib_side = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(ib_side, contracts)
        order.tif = "GTC"
        order.outsideRth = True  # Allow outside regular trading hours

        trade = self.ib.placeOrder(self._contract, order)
        logger.info(
            "[IBKR] Placed %s %d MGC — orderId=%s",
            ib_side, contracts, trade.order.orderId,
        )

        # Wait for fill (up to 30s)
        for _ in range(60):
            self.ib.sleep(0.5)
            if trade.isDone():
                break

        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            self._order_counter += 1
            order_id = f"IBKR-{self._order_counter:06d}"

            # Update local position tracking
            if side.upper() == "BUY":
                if self._position and self._position["direction"] == "SHORT":
                    # Closing short
                    self._position["contracts"] -= contracts
                    if self._position["contracts"] <= 0:
                        self._position = None
                else:
                    # Opening or adding to long
                    if self._position is None:
                        self._position = {
                            "direction": "LONG",
                            "entry_price": fill_price,
                            "contracts": contracts,
                        }
                    else:
                        self._position["contracts"] += contracts
            else:  # SELL
                if self._position and self._position["direction"] == "LONG":
                    self._position["contracts"] -= contracts
                    if self._position["contracts"] <= 0:
                        self._position = None
                else:
                    if self._position is None:
                        self._position = {
                            "direction": "SHORT",
                            "entry_price": fill_price,
                            "contracts": contracts,
                        }
                    else:
                        self._position["contracts"] += contracts

            logger.info(
                "[IBKR] FILLED %s %d MGC @ $%.2f (%s)",
                ib_side, contracts, fill_price, order_id,
            )
            self._save_state()
            return order_id

        else:
            status = trade.orderStatus.status
            logger.error(
                "[IBKR] Order NOT filled — status=%s, filled=%s/%s",
                status, trade.orderStatus.filled, contracts,
            )
            # Cancel unfilled order
            self.ib.cancelOrder(trade.order)
            return None

    def close_partial(
        self, symbol: str, contracts: int, reason: str
    ) -> Optional[str]:
        """Close N contracts and record P&L."""
        if not self._position or self._position["contracts"] <= 0:
            logger.warning("[IBKR] close_partial called but no position")
            return None

        direction = self._position["direction"]
        entry_price = self._position["entry_price"]

        # Determine close side
        close_side = "SELL" if direction == "LONG" else "BUY"

        # Get current price for P&L calculation
        try:
            current_price = self.get_current_price(symbol)
        except RuntimeError:
            current_price = entry_price  # fallback

        order_id = self.place_market_order(symbol, contracts, close_side)

        if order_id:
            # Calculate P&L
            pip_value = self._config.pip_value
            if direction == "LONG":
                pips = (current_price - entry_price) / pip_value
            else:
                pips = (entry_price - current_price) / pip_value
            pnl = pips * contracts * pip_value * float(self._contract.multiplier)

            # Update equity
            self._equity += pnl

            # Log trade
            trade_record = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": direction,
                "contracts": contracts,
                "entry": entry_price,
                "exit": current_price,
                "pnl": round(pnl, 2),
                "reason": reason,
                "order_id": order_id,
            }
            self._trade_log.append(trade_record)
            self._save_trade_log()

            logger.info(
                "[IBKR] Close %dct %s @ $%.2f (%s) PnL: $%.2f (%s)",
                contracts, symbol, current_price, direction, pnl, reason,
            )
            self._save_state()
            return order_id

        return None

    def close_all(self, symbol: str, reason: str) -> Optional[str]:
        """Close entire position."""
        if not self._position:
            return None
        return self.close_partial(symbol, self._position["contracts"], reason)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return current position or None if flat."""
        if self._position and self._position["contracts"] > 0:
            return dict(self._position)
        return None

    def get_account_equity(self) -> float:
        """Return account equity from IBKR."""
        # Refresh from IBKR periodically
        try:
            self._ensure_connected()
            summary = self.ib.accountSummary()
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    self._equity = float(item.value)
                    break
        except Exception as e:
            logger.warning("[IBKR] Equity refresh failed: %s", e)
        return self._equity

    def fetch_bars(
        self, symbol: str, timeframe: str, lookback: int
    ) -> pd.DataFrame:
        """Fetch historical bars from IBKR."""
        self._ensure_connected()

        if timeframe not in _TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        bar_size, duration = _TF_MAP[timeframe]

        # For 4h, fetch 1h and resample
        if timeframe == "4h":
            bars_1h = self.fetch_bars(symbol, "1h", lookback * 4)
            if bars_1h.empty:
                return bars_1h
            bars_1h = bars_1h.set_index("ts")
            resampled = bars_1h.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()
            resampled = resampled.reset_index()
            return resampled.tail(lookback)

        try:
            ib_bars = self.ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
        except Exception as e:
            logger.error("[IBKR] Historical data request failed: %s", e)
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        if not ib_bars:
            logger.warning("[IBKR] No bars returned for %s %s", symbol, timeframe)
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        rows = []
        for bar in ib_bars:
            rows.append({
                "ts": pd.Timestamp(bar.date),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        return df.tail(lookback)

    # ------------------------------------------------------------------
    # State persistence (backup)
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        """Save local state as backup."""
        state = {
            "equity": self._equity,
            "position": self._position,
            "order_counter": self._order_counter,
            "contract": self._contract.localSymbol,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self._state_file)

    def _load_trade_log(self) -> List[Dict]:
        """Load trade history."""
        if os.path.exists(self._trade_log_file):
            try:
                with open(self._trade_log_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_trade_log(self) -> None:
        """Save trade history."""
        tmp = self._trade_log_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._trade_log[-500:], f, indent=2)
        os.replace(tmp, self._trade_log_file)

    def disconnect(self) -> None:
        """Gracefully disconnect from IBKR."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("[IBKR] Disconnected")
