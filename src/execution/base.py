"""
BaseExecutor — abstract interface for order execution.

All exchange integrations (Alpaca, Kraken, etc.) implement this interface
so paper_trader.py can swap executors without changing trading logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    def get_account_summary(self) -> dict:
        """Return {equity, cash, buying_power}."""
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, dict]:
        """Return {symbol: {qty, side, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct}}."""
        ...

    @abstractmethod
    def buy(self, symbol: str, qty: float,
            limit_price: Optional[float] = None) -> Optional[str]:
        """Open LONG position. Returns order ID or None on failure."""
        ...

    @abstractmethod
    def sell(self, symbol: str, qty: float, reason: str = "",
             limit_price: Optional[float] = None) -> Optional[str]:
        """Close LONG position. Returns order ID or None on failure."""
        ...

    @abstractmethod
    def sell_short(self, symbol: str, qty: float,
                   limit_price: Optional[float] = None) -> Optional[str]:
        """Open SHORT position. Returns order ID or None on failure."""
        ...

    @abstractmethod
    def buy_to_cover(self, symbol: str, qty: float, reason: str = "",
                     limit_price: Optional[float] = None) -> Optional[str]:
        """Close SHORT position. Returns order ID or None on failure."""
        ...
