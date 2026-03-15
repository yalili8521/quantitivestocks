"""
Execution layer — order routing to different brokers/exchanges.

All executors implement the same interface:
    get_account_summary() → {equity, cash, buying_power}
    get_positions() → {symbol: {qty, side, entry_price, ...}}
    buy(symbol, qty) → order_id
    sell(symbol, qty, reason) → order_id
    sell_short(symbol, qty) → order_id
    buy_to_cover(symbol, qty, reason) → order_id

Usage:
    from execution import get_executor
    executor = get_executor("kraken", api_key=key, api_secret=secret)
"""
from execution.base import BaseExecutor

__all__ = ["BaseExecutor", "get_executor"]


def get_executor(exchange: str, **kwargs) -> BaseExecutor:
    """Factory: create the right executor for an exchange."""
    if exchange == "kraken":
        from execution.kraken import KrakenExecutor
        return KrakenExecutor(**kwargs)
    if exchange == "alpaca":
        from execution.alpaca import AlpacaExecutor
        return AlpacaExecutor(**kwargs)
    raise ValueError(f"Unknown exchange: {exchange}")
