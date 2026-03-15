"""
Alpaca executor — wraps alpaca-py TradingClient with BaseExecutor interface.

Used for equity and ETF trading (intraday + swing groups).
Crypto on Alpaca is long-only; use KrakenExecutor for crypto shorts.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from execution.base import BaseExecutor

log = logging.getLogger(__name__)


def _get_session() -> str:
    """Return current market session: regular, extended, or closed."""
    from datetime import time as dt_time, datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "closed"
    t = now_et.time()
    if dt_time(4, 0) <= t < dt_time(9, 30):
        return "extended"
    if dt_time(9, 30) <= t <= dt_time(16, 0):
        return "regular"
    if dt_time(16, 0) < t <= dt_time(20, 0):
        return "extended"
    return "closed"


class AlpacaExecutor(BaseExecutor):
    """Alpaca paper/live trading executor."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )

    def get_account_summary(self) -> dict:
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

    def get_positions(self) -> Dict[str, dict]:
        positions = self.client.get_all_positions()
        result = {}
        for pos in positions:
            qty = float(pos.qty)
            side = "SHORT" if qty < 0 else "LONG"
            result[pos.symbol] = {
                "qty": abs(qty),
                "side": side,
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc),
            }
        return result

    def _submit_order(self, symbol: str, qty: float, side: OrderSide,
                      limit_price: Optional[float] = None) -> Optional[str]:
        is_crypto = "/" in symbol or (symbol.upper().endswith("USD") and len(symbol) >= 6)
        session = _get_session()
        try:
            if is_crypto:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol, qty=qty, side=side,
                        time_in_force=TimeInForce.GTC,
                    )
                )
            elif session == "extended" and limit_price is not None:
                lp = round(limit_price * (1.001 if side == OrderSide.BUY else 0.999), 2)
                order = self.client.submit_order(
                    LimitOrderRequest(
                        symbol=symbol, qty=int(qty), side=side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=lp, extended_hours=True,
                    )
                )
            else:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol, qty=int(qty), side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
            log.info("%s %s x%.6f — order %s", side.value, symbol, qty, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("Order failed %s %s x%s: %s", side.value, symbol, qty, exc)
            return None

    def buy(self, symbol: str, qty: float,
            limit_price: Optional[float] = None) -> Optional[str]:
        if qty <= 0:
            return None
        return self._submit_order(symbol, qty, OrderSide.BUY, limit_price)

    def sell(self, symbol: str, qty: float, reason: str = "",
             limit_price: Optional[float] = None) -> Optional[str]:
        if qty <= 0:
            return None
        oid = self._submit_order(symbol, qty, OrderSide.SELL, limit_price)
        if oid:
            log.info("SELL reason: %s", reason)
        return oid

    def sell_short(self, symbol: str, qty: float,
                   limit_price: Optional[float] = None) -> Optional[str]:
        if qty <= 0:
            return None
        return self._submit_order(symbol, qty, OrderSide.SELL, limit_price)

    def buy_to_cover(self, symbol: str, qty: float, reason: str = "",
                     limit_price: Optional[float] = None) -> Optional[str]:
        if qty <= 0:
            return None
        oid = self._submit_order(symbol, qty, OrderSide.BUY, limit_price)
        if oid:
            log.info("COVER reason: %s", reason)
        return oid
